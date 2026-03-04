"""PersistentPipeline — orchestrates L1 (klines) → L2 (decomposed) → L3 (predictions)."""

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
from .csv_io import read_csv_safe, get_max_time, append_rows_atomic
from .gap_detector import GapDetector, TF_MINUTES
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
        self.threshold = threshold

        # Create directories
        for d in (self.klines_dir, self.decomposed_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.gap_detector = GapDetector(self.klines_dir)

        # Load models once
        logger.info("Loading production models...")
        self.models, self.metadata = load_production_models(model_dir)
        self.feature_names = self.metadata["feature_names"]
        logger.info(f"Models loaded: {len(self.models)} models, "
                    f"{len(self.feature_names)} features")

        # Binance client — lazy init
        self._client = None

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

    def update_predictions(self, decomposed: dict) -> pd.DataFrame | None:
        """Encode features and predict the latest 5M candle.

        Loads klines from persistent CSVs, builds in-memory dicts compatible
        with encode_live_features(), predicts the last row, appends to CSV.

        Returns: prediction DataFrame row (or None if no new prediction).
        """
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
            # Keep tail 500 rows for encoding context
            klines_dict[tf] = df.tail(500).reset_index(drop=True)

        # Trim decomposed to tail for encoding
        decomposed_trimmed = {}
        for key, df in decomposed.items():
            decomposed_trimmed[key] = df.tail(500).reset_index(drop=True)

        # Encode features
        logger.info("L3 Predictions: encoding features...")
        t0 = time.time()
        features_df = encode_live_features(klines_dict, decomposed_trimmed)
        encode_time = time.time() - t0
        logger.info(f"  Encoding: {features_df.shape[1] - 1} features, "
                    f"{len(features_df)} rows, {encode_time:.1f}s")

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

        # Predict last row
        from catboost import Pool

        last_row = features_df.iloc[[-1]]  # Keep as DataFrame

        # Fill missing features
        missing = set(self.feature_names) - set(features_df.columns)
        if missing:
            logger.warning(f"  Filling {len(missing)} missing features with 0")
            for feat in missing:
                last_row[feat] = 0.0

        # Run batch predict on single row
        pred_df = batch_ensemble_predict(
            self.models, last_row, self.feature_names, threshold=self.threshold)

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

        # Append to predictions CSV
        n_appended = append_rows_atomic(pred_path, pred_df)
        signal = pred_df["signal"].iloc[0]
        conf = pred_df["confidence"].iloc[0]
        logger.info(f"  L3: {signal} (conf={conf:.3f}) at {pred_df['time'].iloc[0]}")

        return pred_df

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
        pred = self.update_predictions(decomposed)
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
