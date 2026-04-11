"""PersistentPipeline — orchestrates L1 (klines) → L2 (decomposed) → L3 (predictions).

L3 supports two modes:
  - Incremental (default if state file exists): compute 1 feature row from persisted
    encoder state, matching batch encoding exactly. Requires backfill_features.py to
    have initialized the state first.
  - Batch (fallback): re-encode tail(BOOTSTRAP_BARS) rows every cycle. Used when no
    incremental state is available (pre-backfill).
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from model_training.live_predict import (
    BINANCE_INTERVALS,
    CLASS_NAMES,
    TRADE_CLASSES,
    encode_live_features,
    load_production_models,
    batch_ensemble_predict,
)
from core.data_validation import (
    validate_kline_continuity,
    validate_feature_shape,
    validate_predictions_row,
)
from .csv_io import read_csv_safe, get_max_time, append_rows_atomic
from .gap_detector import BOOTSTRAP_BARS, GapDetector, TF_MINUTES
from .incremental_encoder import IncrementalEncoder
from .incremental_etl import run_incremental_etl

logger = logging.getLogger(__name__)


class PersistentPipeline:
    """Orchestrates the 3-layer persistent data pipeline.

    Layer 1: Klines (11 CSVs, one per TF)
    Layer 2: Decomposed (55 CSVs, one per TF/window combo)
    Layer 3: Predictions (1 CSV, appended each cycle)
    """

    def __init__(self, data_dir: str | Path, threshold: float = 0.75,
                 model_dir: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.klines_dir = self.data_dir / "klines"
        self.decomposed_dir = self.data_dir / "decomposed"
        self.predictions_dir = self.data_dir / "predictions"
        self.features_dir = self.data_dir / "features"
        self.threshold = threshold

        # Create directories
        for d in (self.klines_dir, self.decomposed_dir, self.predictions_dir,
                  self.features_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.features_path = self.features_dir / "features.csv"
        self.state_path = self.features_dir / "feature_state.json"

        self.gap_detector = GapDetector(self.klines_dir)

        # V3 production models -- gated by TAMAGOCHI_LOAD_V3 (default 1).
        # Contabo multi-target deploy sets TAMAGOCHI_LOAD_V3=0 so the V3
        # 5-seed bundle (~80MB) is not required on the image and never
        # touched at runtime. feature_names is always loaded from the V10
        # metadata when V3 is on; when V3 is off the multitarget predictor
        # ships its own ordered feature list (518 V10 cols).
        self.load_v3 = os.getenv("TAMAGOCHI_LOAD_V3", "1") == "1"
        if self.load_v3:
            logger.info("Loading production models (V3)...")
            self.models, self.metadata = load_production_models(model_dir)
            self.feature_names = self.metadata["feature_names"]
            logger.info(f"V3 models loaded: {len(self.models)} models, "
                        f"{len(self.feature_names)} features")
        else:
            logger.info("TAMAGOCHI_LOAD_V3=0 -- skipping V3 production models")
            self.models = None
            self.metadata = None
            # Derive the 518-col V10 feature order from the shipped
            # feature_matrix_v10.parquet (also used as the multi-target
            # cold-start backfill source). Single source of truth -- the
            # encoder, the multitarget predictor, and the parquet share
            # the exact same column order produced by encode_v10.py.
            if not self.FEATURE_MATRIX_PATH.exists():
                raise FileNotFoundError(
                    f"TAMAGOCHI_LOAD_V3=0 requires {self.FEATURE_MATRIX_PATH} "
                    f"to derive V10 feature names; not found"
                )
            _fm_cols = pd.read_parquet(
                self.FEATURE_MATRIX_PATH, columns=None
            ).columns.tolist()
            # Strip non-feature cols (time, label, ohlcv) -- match what
            # encode_live_features returns. Keeping anything that is not
            # an obvious metadata column.
            _drop = {"time", "label", "Open", "High", "Low", "Close", "Volume",
                     "Open Time", "Close Time"}
            self.feature_names = [c for c in _fm_cols if c not in _drop]
            logger.info(f"V10 feature names derived from feature matrix: "
                        f"{len(self.feature_names)} features")

        # Initialize incremental encoder (if state exists)
        self.encoder = self._init_encoder()

        # Multi-target stacking pipeline -- gated by TAMAGOCHI_MULTITARGET.
        # Loads 24 base + 8 stacking CatBoost models (~517MB) and runs the
        # full live flow each cycle, writing predictions_multitarget.csv.
        self.multitarget_enabled = os.getenv("TAMAGOCHI_MULTITARGET", "0") == "1"
        self.multitarget_predictor = None
        self.multitarget_router = None
        if self.multitarget_enabled:
            from core.multitarget_predictor import MultiTargetPredictor
            from core.multitarget_router import MultiTargetRouter
            from model_training.multitarget_config import TARGET_CONFIGS
            mt_root = Path(os.getenv(
                "MULTITARGET_ROOT",
                "model_training/results_v10/multitarget",
            ))
            mt_state = self.features_dir / "multitarget_state.json"
            mt_backfill = self.FEATURE_MATRIX_PATH if self.FEATURE_MATRIX_PATH.exists() else None
            logger.info(f"Loading multi-target models from {mt_root}...")
            self.multitarget_predictor = MultiTargetPredictor(
                models_root=mt_root,
                top_raw_features_path=mt_root / "stacking" / "top_raw_features_union.json",
                v10_feature_names=self.feature_names,
                state_path=mt_state,
                backfill_features_path=mt_backfill,
            )
            self.multitarget_router = MultiTargetRouter(TARGET_CONFIGS)
            logger.info("Multi-target predictor + router ready")

        # Binance client — lazy init
        self._client = None

    # Default feature matrix path (copied into Docker image by Dockerfile)
    FEATURE_MATRIX_PATH = Path("model_training/encoded_data/feature_matrix_v10.parquet")

    def _init_encoder(self):
        """Load incremental encoder from saved state, or auto-init from feature matrix.

        Priority:
          1. Load existing state file (fast, normal case)
          2. If missing/corrupt: auto-initialize from feature_matrix_v10.parquet
          3. If feature matrix also unavailable: fall back to batch mode
        """
        # Try loading existing state
        if self.state_path.exists():
            try:
                encoder = IncrementalEncoder.load_state(self.state_path)
                logger.info(f"Incremental encoder loaded "
                            f"(last={encoder.state['last_timestamp']})")
                return encoder
            except Exception as e:
                logger.warning(f"Failed to load encoder state: {e}")

        # Auto-initialize from feature matrix
        return self._auto_init_encoder()

    def _auto_init_encoder(self):
        """Auto-initialize encoder from feature matrix + persistent klines/decomposed."""
        fm_path = self.FEATURE_MATRIX_PATH
        if not fm_path.exists():
            logger.info("No encoder state or feature matrix — using batch fallback")
            return None

        # Check that we have klines data to initialize from
        kl_5m_path = self.klines_dir / "ml_data_5M.csv"
        if not kl_5m_path.exists():
            logger.info("No 5M klines yet — using batch fallback (will auto-init later)")
            return None

        try:
            from .state_initializer import initialize_state

            logger.info("Auto-initializing encoder from feature matrix...")
            state = initialize_state(
                str(fm_path), str(self.klines_dir), str(self.decomposed_dir))
            encoder = IncrementalEncoder(state)
            encoder.save_state(self.state_path)
            logger.info(f"Encoder auto-initialized (last={state['last_timestamp']})")
            return encoder
        except Exception as e:
            logger.warning(f"Auto-init failed: {e}. Using batch fallback.")
            return None

    def _get_client(self):
        """Lazy-init Binance client."""
        if self._client is None:
            from binance import Client
            from model_training.live_predict import _read_api_keys
            api_key, api_secret = _read_api_keys()
            self._client = Client(api_key=api_key, api_secret=api_secret, tld="com")
            self._client.ping()
            logger.info("Binance client connected")
        return self._client

    # ------------------------------------------------------------------
    # Layer 1: Klines
    # ------------------------------------------------------------------

    def update_klines(self) -> dict[str, int]:
        """Fetch new kline bars for TFs that need them, append to CSVs.

        Returns: {tf: n_new_bars} for TFs that were updated.
        """
        from binance.exceptions import BinanceAPIException, BinanceRequestException

        gaps = self.gap_detector.check_all(TIMEFRAME_ORDER)
        updates = {tf: g for tf, g in gaps.items() if g["needs_update"]}

        if not updates:
            logger.info("L1 Klines: all up-to-date")
            return {}

        client = self._get_client()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        result = {}

        for tf, gap_info in updates.items():
            interval = BINANCE_INTERVALS[tf]
            bars_needed = gap_info["bars_needed"]
            is_bootstrap = gap_info["is_bootstrap"]

            tf_min = TF_MINUTES[tf]
            lookback_ms = bars_needed * tf_min * 60 * 1000
            start_ms = now_ms - lookback_ms

            # If incremental, start from last known time (inclusive — the dedup
            # logic below removes the stale last candle and replaces it with
            # fresh data from the re-fetch, so we MUST include it here).
            if not is_bootstrap and gap_info["last_time"] is not None:
                last_t = gap_info["last_time"]
                if last_t.tzinfo is None:
                    last_t = last_t.tz_localize("UTC")
                start_ms = int(last_t.timestamp() * 1000)

            logger.info(f"  L1 {tf}: fetching ~{bars_needed} bars "
                        f"({'bootstrap' if is_bootstrap else 'incremental'})...")

            retries = 3
            backoff = 1.0
            raw = []
            next_start = start_ms

            while retries > 0:
                try:
                    batch = client.get_historical_klines(
                        symbol="BTCUSDT", interval=interval,
                        start_str=str(next_start), end_str=str(now_ms),
                        limit=1000)
                    if not batch:
                        break
                    raw.extend(batch)
                    last_open = batch[-1][0]
                    next_start = last_open + 1
                    if last_open >= now_ms or len(batch) < 1000:
                        break
                    backoff = 1.0
                except (BinanceAPIException, BinanceRequestException) as e:
                    retries -= 1
                    logger.warning(f"    API error: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2

            if not raw:
                logger.warning(f"  L1 {tf}: no data received")
                continue

            # Normalize
            cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                    "Close Time", "QAV", "NumTrades", "TBBAV", "TBQAV", "Ignore"]
            df = pd.DataFrame(raw, columns=cols)
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
            df["Close Time"] = pd.to_numeric(df["Close Time"], errors="coerce")
            for c in ("Open", "High", "Low", "Close", "Volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Drop the current incomplete candle — its Close Time hasn't passed yet.
            # Binance returns it as the last bar; storing it would permanently save
            # partial OHLCV data since dedup prevents overwriting with the closed version.
            n_before = len(df)
            df = df[df["Close Time"] <= now_ms]
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.debug(f"    {tf}: dropped {n_dropped} incomplete candle(s)")

            df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.sort_values("Open Time").reset_index(drop=True)
            df["time"] = df["Open Time"].dt.tz_localize(None)

            # Deduplicate + overwrite stale last candle
            # The last stored candle may have been incomplete (fetched before
            # candle close). Re-fetch it to ensure final OHLCV values.
            kline_path = self.klines_dir / f"ml_data_{tf}.csv"
            existing = read_csv_safe(kline_path)
            if existing is not None:
                existing["time"] = pd.to_datetime(existing["time"]).dt.tz_localize(None)
                existing_max = existing["time"].max()

                # Split new data: rows that overlap last candle + truly new rows
                df_new = df[df["time"] >= existing_max]

                if df_new.empty:
                    logger.info(f"  L1 {tf}: no new bars after dedup")
                    continue

                # Remove the stale last row(s) from existing, replace with fresh data
                existing_trimmed = existing[existing["time"] < existing_max]
                combined = pd.concat([existing_trimmed, df_new], ignore_index=True)
                combined = combined.sort_values("time").reset_index(drop=True)

                # Atomic write of the full updated file
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(self.klines_dir), suffix=".tmp")
                os.close(tmp_fd)
                try:
                    combined.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, str(kline_path))
                except Exception:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise

                n_genuinely_new = len(df_new[df_new["time"] > existing_max])
                result[tf] = len(df_new)
                logger.info(f"  L1 {tf}: +{n_genuinely_new} new bars, "
                            f"1 refreshed (last={df_new['time'].max()}, "
                            f"total={len(combined)})")
            else:
                # No existing file — write all data
                n_appended = append_rows_atomic(kline_path, df)
                result[tf] = n_appended
                logger.info(f"  L1 {tf}: +{n_appended} bars (bootstrap) "
                            f"(last={df['time'].max()})")

        # Validate kline continuity (non-blocking warnings)
        for tf in result:
            kl_path = self.klines_dir / f"ml_data_{tf}.csv"
            kl_df = read_csv_safe(kl_path)
            if kl_df is not None:
                kl_errors = validate_kline_continuity(kl_df, tf)
                for err in kl_errors:
                    logger.warning(f"  L1 validation: {err}")

        return result

    # ------------------------------------------------------------------
    # Layer 2: Decomposed
    # ------------------------------------------------------------------

    def update_decomposed(self) -> dict[tuple, pd.DataFrame]:
        """Run incremental ETL on all 55 TF/window combos.

        Returns: full decomposed dict {(tf, ws): DataFrame} for encoding.
        """
        logger.info("L2 Decomposed: running incremental ETL...")
        return run_incremental_etl(
            klines_dir=self.klines_dir,
            decomposed_dir=self.decomposed_dir,
        )

    # ------------------------------------------------------------------
    # Layer 3: Predictions
    # ------------------------------------------------------------------

    def update_predictions(self, decomposed: dict,
                           kline_updates: dict | None = None) -> pd.DataFrame | None:
        """Encode features and predict the latest 5M candle.

        Uses incremental encoder if available, otherwise falls back to batch mode.

        Args:
            decomposed: {(tf, ws): DataFrame} from L2
            kline_updates: {tf: n_new_bars} from L1 (used for incremental mode)

        Returns: prediction DataFrame row (or None if no new prediction).
        """
        if self.encoder is not None and kline_updates is not None:
            return self._update_predictions_incremental(decomposed, kline_updates)
        return self._update_predictions_batch(decomposed)

    def _update_predictions_incremental(self, decomposed: dict,
                                        kline_updates: dict) -> pd.DataFrame | None:
        """L3 incremental mode: walk through ALL missed 5M candles, predict latest.

        If the encoder is behind by >1 5M candle (e.g. container restart after
        2 hours), we process each missed candle sequentially to preserve
        EMA/rolling state continuity. Only the final row gets a prediction.
        """
        logger.info("L3 Predictions: incremental encoding...")

        # Load 5M klines (the master timeline)
        kl_5m_path = self.klines_dir / "ml_data_5M.csv"
        kl_5m = read_csv_safe(kl_5m_path)
        if kl_5m is None:
            logger.warning("  L3: missing 5M klines")
            return None
        kl_5m["time"] = pd.to_datetime(kl_5m["time"]).dt.tz_localize(None)
        kl_5m = kl_5m.sort_values("time").reset_index(drop=True)
        latest_time = kl_5m["time"].max()

        # Check for duplicate prediction
        pred_path = self.predictions_dir / "predictions.csv"
        last_pred_time = get_max_time(pred_path, time_col="time")
        if last_pred_time is not None:
            last_pred_time = pd.to_datetime(last_pred_time)
            if last_pred_time.tzinfo is not None:
                last_pred_time = last_pred_time.tz_localize(None)
            if latest_time <= last_pred_time:
                logger.info(f"  L3: no new candle (latest={latest_time}, "
                            f"last_pred={last_pred_time})")
                return None

        # Determine how many 5M candles the encoder has missed
        encoder_last = pd.to_datetime(self.encoder.state["last_timestamp"])
        if encoder_last.tzinfo is not None:
            encoder_last = encoder_last.tz_localize(None)

        new_candles = kl_5m[kl_5m["time"] > encoder_last]
        n_gap = len(new_candles)

        if n_gap == 0:
            logger.info(f"  L3: encoder up-to-date (last={encoder_last})")
            return None

        # Safety cap: if gap is too large, fall back to batch mode for this cycle.
        # 2016 candles = 7 days of 5M data. Walking through more than that would
        # be slow and error-prone. Use batch mode once, then the encoder state
        # will be stale but the prediction is still valid.
        MAX_BACKFILL_CANDLES = 2016
        if n_gap > MAX_BACKFILL_CANDLES:
            logger.warning(
                f"  L3: gap too large ({n_gap} candles, >{MAX_BACKFILL_CANDLES}). "
                f"Falling back to batch mode. Run backfill_features.py to "
                f"re-initialize incremental state.")
            return self._update_predictions_batch(decomposed)

        if n_gap > 1:
            logger.info(f"  L3: gap detected — {n_gap} candles to backfill "
                        f"({encoder_last} → {latest_time})")

        # Pre-load ALL kline and decomposed data with sorted time arrays
        # for O(log N) searchsorted lookup per candle
        kline_data = {}   # {tf: (times_array, DataFrame)}
        for tf in TIMEFRAME_ORDER:
            kl_path = self.klines_dir / f"ml_data_{tf}.csv"
            df = read_csv_safe(kl_path)
            if df is None:
                continue
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            for c in ("Open", "High", "Low", "Close", "Volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            kline_data[tf] = (df["time"].values, df)

        decomposed_data = {}  # {(tf, ws): (times_array, DataFrame)}
        for (tf, ws), df in decomposed.items():
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.sort_values("time").reset_index(drop=True)
            decomposed_data[(tf, ws)] = (df["time"].values, df)

        # Walk through each missed 5M candle sequentially.
        # CRITICAL: only pass TF data when the native-TF index CHANGES.
        # Passing the same TF row twice would corrupt EMAs, rolling buffers,
        # and lag/delta features (they'd see zero deltas). This matches the
        # batch encoder's merge_asof semantics where features are computed
        # once on native-TF data and then forward-filled to 5M.
        t0 = time.time()
        feature_row = None

        # Track previous searchsorted index per TF to detect changes
        prev_kl_idx = {tf: -1 for tf in kline_data}
        prev_dec_idx = {key: -1 for key in decomposed_data}

        for i, (_, candle_5m) in enumerate(new_candles.iterrows()):
            candle_time = candle_5m["time"]
            ct_val = np.datetime64(candle_time)

            # For each TF, find the latest row at or before this 5M candle time.
            # Only include in step dict if the index CHANGED (new native candle).
            step_klines = {}
            for tf, (times, kl_df) in kline_data.items():
                idx = np.searchsorted(times, ct_val, side="right") - 1
                if idx >= 0 and idx != prev_kl_idx[tf]:
                    step_klines[tf] = kl_df.iloc[idx]
                    prev_kl_idx[tf] = idx

            step_decomposed = {}
            for key, (times, dec_df) in decomposed_data.items():
                idx = np.searchsorted(times, ct_val, side="right") - 1
                if idx >= 0 and idx != prev_dec_idx[key]:
                    step_decomposed[key] = dec_df.iloc[idx]
                    prev_dec_idx[key] = idx

            feature_row = self.encoder.compute_row(
                step_klines, step_decomposed, candle_time)

            if n_gap > 1 and (i + 1) % 50 == 0:
                logger.info(f"    backfill progress: {i + 1}/{n_gap} candles")

        encode_time = time.time() - t0

        # Validate feature count matches expected (518 for multi-target, 508 for legacy V3 path, env-gated)
        n_feats = len(feature_row)
        if n_feats != len(self.feature_names):
            logger.error(
                f"Feature count mismatch: incremental encoder produced {n_feats}, "
                f"expected {len(self.feature_names)}. Missing features will be zero-filled.")

        # Build DataFrame for prediction (from last row only)
        feature_dict = feature_row.to_dict()
        feature_dict["time"] = latest_time
        feature_df = pd.DataFrame([feature_dict])

        # Validate feature shape
        feat_errors = validate_feature_shape(feature_df, len(self.feature_names))
        for err in feat_errors:
            logger.warning(f"  L3 validation: {err}")

        # Append to features CSV (dedup by time to avoid silent duplicate rows
        # when encoder state reset mid-cycle replays the latest row)
        append_rows_atomic(self.features_path, feature_df, dedup_col="time")

        # Save encoder state
        self.encoder.save_state(self.state_path)

        n_feats = len(feature_row)
        if n_gap > 1:
            logger.info(f"  Encoding: {n_feats} features, {n_gap} rows backfilled "
                        f"(incremental), {encode_time:.3f}s")
        else:
            logger.info(f"  Encoding: {n_feats} features, 1 row (incremental), "
                        f"{encode_time:.3f}s")

        return self._predict_and_append(feature_df, pred_path, latest_time)

    def _update_predictions_batch(self, decomposed: dict) -> pd.DataFrame | None:
        """L3 batch mode (fallback): re-encode tail(BOOTSTRAP_BARS) rows."""
        # Build klines_dict from persistent CSVs (for encoding)
        klines_dict = {}
        for tf in TIMEFRAME_ORDER:
            kl_path = self.klines_dir / f"ml_data_{tf}.csv"
            df = read_csv_safe(kl_path)
            if df is None:
                logger.warning(f"  L3: missing klines for {tf}")
                return None
            df["Open Time"] = pd.to_datetime(df["Open Time"])
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            for c in ("Open", "High", "Low", "Close", "Volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # Keep tail rows for encoding context (match BOOTSTRAP_BARS)
            klines_dict[tf] = df.tail(BOOTSTRAP_BARS).reset_index(drop=True)

        # Trim decomposed to tail for encoding
        decomposed_trimmed = {}
        for key, df in decomposed.items():
            decomposed_trimmed[key] = df.tail(BOOTSTRAP_BARS).reset_index(drop=True)

        # Encode features
        logger.info("L3 Predictions: batch encoding (fallback)...")
        t0 = time.time()
        features_df = encode_live_features(klines_dict, decomposed_trimmed)
        encode_time = time.time() - t0
        logger.info(f"  Encoding: {features_df.shape[1] - 1} features, "
                    f"{len(features_df)} rows, {encode_time:.1f}s")

        # Validate feature shape
        feat_errors = validate_feature_shape(features_df, len(self.feature_names))
        for err in feat_errors:
            logger.warning(f"  L3 validation: {err}")

        # Check for duplicate prediction
        pred_path = self.predictions_dir / "predictions.csv"
        last_pred_time = get_max_time(pred_path, time_col="time")
        latest_time = features_df["time"].iloc[-1]

        if last_pred_time is not None:
            last_pred_time = pd.to_datetime(last_pred_time)
            if last_pred_time.tzinfo is not None:
                last_pred_time = last_pred_time.tz_localize(None)
            latest_time_clean = pd.to_datetime(latest_time)
            if latest_time_clean.tzinfo is not None:
                latest_time_clean = latest_time_clean.tz_localize(None)
            if latest_time_clean <= last_pred_time:
                logger.info(f"  L3: no new candle (latest={latest_time_clean}, "
                            f"last_pred={last_pred_time})")
                return None

        last_row = features_df.iloc[[-1]]  # Keep as DataFrame
        return self._predict_and_append(last_row, pred_path, latest_time)

    def _predict_and_append(self, feature_df: pd.DataFrame,
                            pred_path: Path,
                            latest_time) -> pd.DataFrame | None:
        """Run ensemble prediction on feature row and append to predictions CSV.

        V3 path is gated by ``self.models is not None``. If the multi-target
        pipeline is enabled, it always runs in parallel (independent CSV).
        """
        # Fill missing features
        missing = set(self.feature_names) - set(feature_df.columns)
        if missing:
            logger.warning(f"  Filling {len(missing)} missing features with 0")
            for feat in missing:
                feature_df[feat] = 0.0

        pred_df = None

        # ---------------- V3 path ----------------
        if self.models is not None:
            pred_df = batch_ensemble_predict(
                self.models, feature_df, self.feature_names, threshold=self.threshold)

            # Add raw_signal (pre-threshold signal for dashboard re-filtering)
            avg_proba = np.array([
                [pred_df["prob_no_trade"].iloc[0],
                 pred_df["prob_long"].iloc[0],
                 pred_df["prob_short"].iloc[0]]
            ])
            pred_class = int(np.argmax(avg_proba[0]))
            if pred_class in TRADE_CLASSES:
                raw_signal = CLASS_NAMES[pred_class]
            else:
                raw_signal = "NO_TRADE"
            pred_df["raw_signal"] = raw_signal

            # Validate prediction row before writing
            pred_row_dict = pred_df.iloc[0].to_dict()
            pred_errors = validate_predictions_row(pred_row_dict)
            if pred_errors:
                for err in pred_errors:
                    logger.error(f"  L3 validation FAILED: {err}")
                logger.error("  Skipping append of invalid V3 prediction row")
                pred_df = None
            else:
                append_rows_atomic(pred_path, pred_df, dedup_col="time")
                signal = pred_df["signal"].iloc[0]
                conf = pred_df["confidence"].iloc[0]
                logger.info(f"  L3 V3: {signal} (conf={conf:.3f}) at {pred_df['time'].iloc[0]}")

        # ---------------- Multi-target path ----------------
        if self.multitarget_enabled and self.multitarget_predictor is not None:
            try:
                self._append_multitarget_row(feature_df, latest_time)
            except Exception as exc:
                logger.exception(f"  L3 multitarget cycle failed: {exc}")

        return pred_df

    def _append_multitarget_row(self, feature_df: pd.DataFrame, latest_time) -> None:
        """Run multi-target predictor + router and append one row to
        ``predictions_multitarget.csv``.

        Schema is a strict superset of the V3 predictions.csv -- the
        trailing ``signal``/``prob_no_trade``/``prob_long``/``prob_short``
        columns mirror the winner target's stacking probabilities so the
        Telegram formatter and dashboard can read this CSV with no code
        changes (``TELEGRAM_PREDICTIONS_CSV`` env var swap).
        """
        feature_row = feature_df.iloc[0]
        ts = pd.Timestamp(latest_time)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        pred = self.multitarget_predictor.predict(feature_row, ts)
        decision = self.multitarget_router.route(pred)

        # Persist ring buffer state every cycle (cheap; <100 ms).
        try:
            self.multitarget_predictor.save_state()
        except Exception as exc:
            logger.warning(f"  multitarget state save failed: {exc}")

        # Build the row.
        row = {
            "time": str(ts),
            "winner_target": decision.winner_target,
            "direction": decision.direction,
            "confidence": round(decision.confidence, 6),
            "sl_pct": decision.sl_pct,
            "tp_pct": decision.tp_pct,
            "max_hold_bars": decision.max_hold_bars,
            "router_decision": decision.reason,
            "firing_targets": ",".join(decision.firing_targets),
        }

        # Per-target stacking probabilities (24 cols) -- always present so
        # downstream consumers see the full state regardless of FIRE/no-fire.
        for t in self.multitarget_predictor.targets:
            tl = t.lower()
            probs = pred.stacking_probs.get(t, np.zeros(3))
            row[f"{tl}_prob_nt"] = round(float(probs[0]), 6)
            row[f"{tl}_prob_long"] = round(float(probs[1]), 6)
            row[f"{tl}_prob_short"] = round(float(probs[2]), 6)

        # V3-compatible tail: winner's probabilities + canonical signal.
        if decision.reason == "FIRE":
            wt = decision.winner_target
            wprobs = pred.stacking_probs[wt]
            row["prob_no_trade"] = round(float(wprobs[0]), 6)
            row["prob_long"] = round(float(wprobs[1]), 6)
            row["prob_short"] = round(float(wprobs[2]), 6)
            row["signal"] = decision.direction
        else:
            row["prob_no_trade"] = 1.0
            row["prob_long"] = 0.0
            row["prob_short"] = 0.0
            row["signal"] = "NO_TRADE"

        mt_path = self.predictions_dir / "predictions_multitarget.csv"
        append_rows_atomic(mt_path, pd.DataFrame([row]), dedup_col="time")
        logger.info(
            f"  L3 MT: {decision.reason} winner={decision.winner_target or '-'} "
            f"dir={decision.direction or '-'} conf={decision.confidence:.3f} "
            f"firing={row['firing_targets'] or '-'}"
        )

    # ------------------------------------------------------------------
    # Full Cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Run one full L1→L2→L3 cycle.

        Returns status dict with timing and results.
        """
        t0 = time.time()
        status = {"cycle_start": datetime.now(timezone.utc).isoformat()}

        # L1: Update klines
        t1 = time.time()
        kline_updates = self.update_klines()
        status["l1_time"] = round(time.time() - t1, 1)
        status["l1_updates"] = {tf: n for tf, n in kline_updates.items()}

        # Early exit if no new 5M candle
        if "5M" not in kline_updates and not self.gap_detector.needs_bootstrap(["5M"]):
            logger.info("No new 5M candle — skipping L2/L3")
            status["l2_time"] = 0
            status["l3_time"] = 0
            status["prediction"] = None
            status["skipped"] = True
            status["total_time"] = round(time.time() - t0, 1)
            return status

        # L2: Update decomposed
        t2 = time.time()
        decomposed = self.update_decomposed()
        status["l2_time"] = round(time.time() - t2, 1)
        status["l2_datasets"] = len(decomposed)

        # L3: Predict
        t3 = time.time()
        pred = self.update_predictions(decomposed, kline_updates=kline_updates)
        status["l3_time"] = round(time.time() - t3, 1)

        if pred is not None:
            status["prediction"] = {
                "time": str(pred["time"].iloc[0]),
                "signal": pred["signal"].iloc[0],
                "confidence": float(pred["confidence"].iloc[0]),
            }
        else:
            status["prediction"] = None

        status["skipped"] = False
        status["total_time"] = round(time.time() - t0, 1)
        return status
