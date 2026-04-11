"""
V10 Live Prediction Pipeline — Rolling buffer approach for real-time inference.

Downloads recent kline data for all 11 timeframes, runs the full ETL + encode
pipeline in memory, and predicts using the production ensemble (3 CatBoost models).

Architecture:
  1. Download 500 bars per TF from Binance (enough for w160 warm-up + rolling features)
  2. Run iterative_regression() for all 55 TF/window combos in memory
  3. Monkey-patch encode_v3/v5/v10 loaders to use in-memory data
  4. Run V3 -> V5 -> V10 encoding pipeline (produces 518 features)
  5. Extract last row, ensemble predict with 3 production models

Data requirements (500 bars per TF):
  5M:  ~1.7 days
  1H:  ~21 days
  4H:  ~83 days
  1D:  ~500 days (~1.4yr)
  3D:  ~1500 days (~4yr)

Usage:
  python model_training/live_predict.py                         # Single prediction
  python model_training/live_predict.py --threshold 0.80        # Higher precision
  python model_training/live_predict.py --loop --interval 300   # Continuous (5min)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES, API_KEY_FILE, BOOTSTRAP_BARS as _BOOTSTRAP_BARS
from core.analysis import iterative_regression, calculate_acceleration

logger = logging.getLogger(__name__)

PRODUCTION_DIR = Path(__file__).parent / "results_v10" / "production"

# Binance interval mapping
BINANCE_INTERVALS = {
    "5M": "5m", "15M": "15m", "30M": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h",
    "6H": "6h", "8H": "8h", "12H": "12h",
    "1D": "1d", "3D": "3d",
}

# Bars to download per TF — imported from core/config.py (single source of truth)
# to ensure encoding features (cumsum, stochastic, EMA) see identical context.
BARS_PER_TF = _BOOTSTRAP_BARS

# Class names
CLASS_NAMES = {0: 'NO_TRADE', 1: 'LONG', 2: 'SHORT'}
TRADE_CLASSES = [1, 2]


# ============================================================================
# Data Download
# ============================================================================

def _read_api_keys():
    """Read Binance API keys — checks env vars first, falls back to file.

    Env var priority enables Docker deployment without mounted key files.
    """
    # Check env vars first (enables Docker / cloud deployment)
    env_key = os.environ.get("BINANCE_KEY", "") or os.environ.get("BINANCE_TESTNET_KEY", "")
    env_secret = os.environ.get("BINANCE_SECRET", "") or os.environ.get("BINANCE_TESTNET_SECRET", "")
    if env_key and env_secret:
        logger.info("Using API keys from environment variables")
        return env_key, env_secret

    # Fall back to file
    if not API_KEY_FILE.exists():
        raise FileNotFoundError(
            f"API key file not found: {API_KEY_FILE}\n"
            f"Set BINANCE_KEY/BINANCE_SECRET env vars or create file with "
            f"key on line 1 and secret on line 2.")
    with API_KEY_FILE.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("API key file must have key on line 1 and secret on line 2.")
    return lines[0], lines[1]


def download_live_klines():
    """Download recent klines for all 11 TFs from Binance.

    Returns dict: {tf_name: DataFrame with columns [Open Time, Open, High, Low, Close, Volume, time]}
    """
    from binance import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException

    api_key, api_secret = _read_api_keys()
    client = Client(api_key=api_key, api_secret=api_secret, tld="com")
    client.ping()
    logger.info("Binance connected")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    klines_dict = {}

    for tf in TIMEFRAME_ORDER:
        interval = BINANCE_INTERVALS[tf]
        logger.info(f"  Downloading {tf} ({BARS_PER_TF} bars)...")

        # Calculate lookback in ms based on TF
        tf_minutes = {
            "5M": 5, "15M": 15, "30M": 30,
            "1H": 60, "2H": 120, "4H": 240,
            "6H": 360, "8H": 480, "12H": 720,
            "1D": 1440, "3D": 4320,
        }
        lookback_ms = BARS_PER_TF * tf_minutes[tf] * 60 * 1000
        start_ms = now_ms - lookback_ms

        retries = 3
        backoff = 1.0
        raw = []
        next_start = start_ms

        while retries > 0:
            try:
                batch = client.get_historical_klines(
                    symbol="BTCUSDT", interval=interval,
                    start_str=str(next_start), end_str=str(now_ms), limit=1000,
                )
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
            raise RuntimeError(f"Failed to download {tf} klines")

        # Normalize
        cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "QAV", "NumTrades", "TBBAV", "TBQAV", "Ignore"]
        df = pd.DataFrame(raw, columns=cols)
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
        for c in ("Open", "High", "Low", "Close", "Volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop incomplete (still-forming) candle: if Open Time + candle duration > now,
        # the candle hasn't closed yet and has partial OHLCV data.
        candle_duration_ms = tf_minutes[tf] * 60 * 1000
        n_before = len(df)
        df = df[df["Open Time"].astype(np.int64) // 10**6 + candle_duration_ms <= now_ms]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"    {tf}: dropped {n_dropped} incomplete candle(s)")

        df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.sort_values("Open Time").reset_index(drop=True)

        # Add 'time' column (tz-naive, matches encoding pipeline)
        df['time'] = df['Open Time'].dt.tz_localize(None)

        klines_dict[tf] = df
        logger.info(f"    {tf}: {len(df)} bars ({df['time'].min()} -> {df['time'].max()})")

    return klines_dict


# ============================================================================
# Extended Data Download (for backfill)
# ============================================================================

def download_live_klines_extended(bars_5m=1400):
    """Download extended klines — 5M gets bars_5m bars (~4.9 days for 1400),
    other TFs keep 500 (already sufficient: 1H=21d, 1D=1.4yr).

    Returns dict: {tf_name: DataFrame with columns [Open Time, Open, High, Low, Close, Volume, time]}
    """
    from binance import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException

    api_key, api_secret = _read_api_keys()
    client = Client(api_key=api_key, api_secret=api_secret, tld="com")
    client.ping()
    logger.info("Binance connected (extended download)")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    klines_dict = {}

    tf_minutes = {
        "5M": 5, "15M": 15, "30M": 30,
        "1H": 60, "2H": 120, "4H": 240,
        "6H": 360, "8H": 480, "12H": 720,
        "1D": 1440, "3D": 4320,
    }

    for tf in TIMEFRAME_ORDER:
        interval = BINANCE_INTERVALS[tf]
        bars = bars_5m if tf == "5M" else BARS_PER_TF
        logger.info(f"  Downloading {tf} ({bars} bars)...")

        lookback_ms = bars * tf_minutes[tf] * 60 * 1000
        start_ms = now_ms - lookback_ms

        retries = 3
        backoff = 1.0
        raw = []
        next_start = start_ms

        while retries > 0:
            try:
                batch = client.get_historical_klines(
                    symbol="BTCUSDT", interval=interval,
                    start_str=str(next_start), end_str=str(now_ms), limit=1000,
                )
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
            raise RuntimeError(f"Failed to download {tf} klines")

        cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "QAV", "NumTrades", "TBBAV", "TBQAV", "Ignore"]
        df = pd.DataFrame(raw, columns=cols)
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
        for c in ("Open", "High", "Low", "Close", "Volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop incomplete (still-forming) candle
        candle_duration_ms = tf_minutes[tf] * 60 * 1000
        n_before = len(df)
        df = df[df["Open Time"].astype(np.int64) // 10**6 + candle_duration_ms <= now_ms]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"    {tf}: dropped {n_dropped} incomplete candle(s)")

        df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.sort_values("Open Time").reset_index(drop=True)
        df['time'] = df['Open Time'].dt.tz_localize(None)

        klines_dict[tf] = df
        logger.info(f"    {tf}: {len(df)} bars ({df['time'].min()} -> {df['time'].max()})")

    return klines_dict


# ============================================================================
# ETL — Run iterative_regression in memory
# ============================================================================

def run_live_etl(klines_dict):
    """Run iterative_regression for all 55 TF/window combos.

    Returns dict: {(tf, ws): DataFrame with regression columns + acceleration}
    """
    decomposed = {}

    for tf in TIMEFRAME_ORDER:
        kl = klines_dict[tf]
        df_input = kl[['Open Time', 'Close']].copy()

        for ws in WINDOW_SIZES:
            if len(df_input) < ws * 2 + 1:
                logger.warning(f"  SKIP {tf}/w{ws}: only {len(df_input)} rows (need {ws * 2 + 1})")
                continue

            result = iterative_regression(df_input, window_size=ws)
            result['acceleration'] = calculate_acceleration(result['angle'])

            # Ensure time is tz-naive datetime
            result['time'] = pd.to_datetime(result['time']).dt.tz_localize(None)
            result = result.sort_values('time').reset_index(drop=True)

            decomposed[(tf, ws)] = result
            logger.debug(f"  {tf}/w{ws}: {len(result)} rows")

    logger.info(f"  ETL complete: {len(decomposed)} decomposed datasets")
    return decomposed


# ============================================================================
# Feature Encoding — Monkey-patch approach
# ============================================================================

def encode_live_features(klines_dict, decomposed):
    """Run the full V3 -> V5 -> V10 encoding pipeline using in-memory data.

    Monkey-patches the loader functions in encode_v3, encode_v5, and encode_v10
    to read from the in-memory dictionaries instead of disk.

    Returns: DataFrame with 518 features + 'time' column.
    """
    import model_training.encode_v3 as enc3
    import model_training.encode_v5 as enc5
    import model_training.encode_v10 as enc10

    # --- Save original loaders ---
    orig_enc3_load = enc3._load_decomposed
    orig_enc5_load_klines = enc5._load_klines
    orig_enc10_load_decomposed = enc10._load_decomposed
    orig_enc10_load_klines = enc10._load_klines
    orig_enc10_load_all_windows = enc10._load_all_windows

    def _mem_load_decomposed(tf, ws):
        key = (tf, ws)
        if key not in decomposed:
            raise FileNotFoundError(f"No decomposed data for {tf}_w{ws}")
        return decomposed[key].copy()

    def _mem_load_klines(tf):
        if tf not in klines_dict:
            raise FileNotFoundError(f"No klines for {tf}")
        return klines_dict[tf].copy()

    def _mem_load_all_windows(tf):
        """Replicate enc10._load_all_windows but from memory."""
        dfs = {}
        for ws in enc10.ALL_WINDOWS:
            key = (tf, ws)
            if key in decomposed:
                df = decomposed[key]
                dfs[ws] = df[['time', 'angle', 'slope_f']].rename(
                    columns={'angle': f'angle_w{ws}', 'slope_f': f'sf_w{ws}'})

        if not dfs:
            return None, []

        merged = None
        for ws in sorted(dfs.keys()):
            if merged is None:
                merged = dfs[ws]
            else:
                merged = merged.merge(dfs[ws], on='time', how='inner')

        merged = merged.sort_values('time').reset_index(drop=True)
        return merged, sorted(dfs.keys())

    try:
        # --- Apply monkey patches ---
        enc3._load_decomposed = _mem_load_decomposed
        enc5._load_klines = _mem_load_klines
        enc10._load_decomposed = _mem_load_decomposed
        enc10._load_klines = _mem_load_klines
        enc10._load_all_windows = _mem_load_all_windows

        # --- Step 1: Build 5M base timeline ---
        key_5m_w30 = ('5M', 30)
        if key_5m_w30 not in decomposed:
            raise RuntimeError("Missing decomposed 5M/w30 data — cannot build base timeline")

        base = pd.DataFrame({'time': decomposed[key_5m_w30]['time'].copy()})
        base = base.sort_values('time').reset_index(drop=True)
        logger.info(f"  5M base: {len(base)} rows")

        # --- Step 2: V3 features (280, from decomposed w30+w120) ---
        logger.info("  Building V3 features...")
        v3_df = enc3.build_features(base)
        logger.info(f"    V3: {v3_df.shape}")

        # --- Step 3: V5 directional features (+110 from klines) ---
        logger.info("  Building V5 features...")
        v5_df = enc5.build_directional_features(v3_df)
        logger.info(f"    V5: {v5_df.shape}")

        # --- Step 4: Filter to KEEP list (203 features) ---
        keep_list = enc10._build_keep_list()
        keep_existing = [c for c in keep_list if c in v5_df.columns]
        result = v5_df[['time'] + keep_existing].copy()
        n_keep = len(keep_existing)
        n = len(result)
        base_times = pd.to_datetime(result['time'])
        logger.info(f"    KEEP: {n_keep} features, {n} rows")

        # --- Step 5: Phase B — Decomposed-based features (10 per TF) ---
        logger.info("  Phase B: Decomposed features...")
        decomp_features, slope_f_per_tf = enc10.build_decomposed_features(base_times, n)

        # --- Step 6: Phase C — Kline-based features (6 per TF + ATR) ---
        logger.info("  Phase C: Kline features...")
        kline_features = enc10.build_kline_features(base_times, n)

        # --- Step 7: Phase D — Cross-TF summaries ---
        logger.info("  Phase D: Cross-TF features...")
        cross_tf_features = enc10.build_cross_tf_features(slope_f_per_tf, n)

        # Combine Phase B-D
        all_new = {}
        all_new.update(decomp_features)
        all_new.update(kline_features)
        all_new.update(cross_tf_features)

        new_df = pd.DataFrame(all_new, index=result.index)
        result = pd.concat([result, new_df], axis=1)

        # --- Step 8: Phase E — range_position_signed ---
        logger.info("  Phase E: Range position signed...")
        rps_features = enc10.build_range_position_signed(result)
        rps_df = pd.DataFrame(rps_features, index=result.index)
        result = pd.concat([result, rps_df], axis=1)

        # --- Step 9: Phase F — Cross-Scale Convergence ---
        logger.info("  Phase F: Cross-scale features...")
        cross_scale_features = enc10.build_cross_scale_features(base_times, n)
        cs_df = pd.DataFrame(cross_scale_features, index=result.index)
        result = pd.concat([result, cs_df], axis=1)

        # --- Step 10: Phase F5 — Interaction features ---
        logger.info("  Phase F5: Interaction features...")
        interaction_features = enc10.build_interaction_features(result, n)
        int_df = pd.DataFrame(interaction_features, index=result.index)
        result = pd.concat([result, int_df], axis=1)

        # --- Step 11: Phase G — Bollinger Band extreme features ---
        logger.info("  Phase G: Bollinger Band features...")
        bb_features = enc10.build_bb_features(base_times, n)
        bb_df = pd.DataFrame(bb_features, index=result.index)
        result = pd.concat([result, bb_df], axis=1)

        n_features = len(result.columns) - 1  # minus 'time'
        logger.info(f"  Encoding complete: {n_features} features, {n} rows")

        return result

    finally:
        # --- Restore original loaders ---
        enc3._load_decomposed = orig_enc3_load
        enc5._load_klines = orig_enc5_load_klines
        enc10._load_decomposed = orig_enc10_load_decomposed
        enc10._load_klines = orig_enc10_load_klines
        enc10._load_all_windows = orig_enc10_load_all_windows


# ============================================================================
# Model Loading & Prediction
# ============================================================================

def load_production_models(model_dir=None):
    """Load the 3 production CatBoost models.

    Returns: (list of models, metadata dict)
    """
    from catboost import CatBoostClassifier

    if model_dir is None:
        model_dir = PRODUCTION_DIR

    model_dir = Path(model_dir)

    # Load metadata
    metadata_path = model_dir / "production_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Production metadata not found: {metadata_path}\n"
            f"Run: python model_training/train_v10_production.py")

    with open(metadata_path) as f:
        metadata = json.load(f)

    seeds = metadata.get('seeds', [42, 123, 777])
    feature_names = metadata.get('feature_names', [])

    # Load models
    models = []
    for seed in seeds:
        model_path = model_dir / f"production_model_s{seed}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        models.append(model)
        logger.info(f"  Loaded model s{seed}")

    logger.info(f"  {len(models)} models loaded, {len(feature_names)} features expected")
    return models, metadata


def ensemble_predict(models, feature_row, feature_names, threshold=0.75):
    """Run ensemble prediction on a single feature row.

    Args:
        models: list of CatBoost models
        feature_row: 1D array or Series of feature values (508 values)
        feature_names: list of feature column names
        threshold: minimum confidence for trade signal

    Returns: dict with signal, confidence, probabilities, etc.
    """
    from catboost import Pool

    # Ensure correct feature order
    if isinstance(feature_row, pd.Series):
        values = feature_row[feature_names].values.reshape(1, -1)
    elif isinstance(feature_row, pd.DataFrame):
        values = feature_row[feature_names].values
    else:
        values = np.array(feature_row).reshape(1, -1)

    # Average probabilities across models
    probas = []
    for model in models:
        pool = Pool(values)
        proba = model.predict_proba(pool)
        probas.append(proba[0])

    avg_proba = np.mean(probas, axis=0)

    # Determine signal
    pred_class = int(np.argmax(avg_proba))
    confidence = float(avg_proba[pred_class])

    signal = "NO_TRADE"
    if pred_class in TRADE_CLASSES and confidence >= threshold:
        signal = CLASS_NAMES[pred_class]

    # Per-model agreement
    model_signals = []
    for proba in probas:
        pc = int(np.argmax(proba))
        if pc in TRADE_CLASSES and proba[pc] >= threshold:
            model_signals.append(CLASS_NAMES[pc])
        else:
            model_signals.append("NO_TRADE")

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "probabilities": {
            "NO_TRADE": round(float(avg_proba[0]), 4),
            "LONG": round(float(avg_proba[1]), 4),
            "SHORT": round(float(avg_proba[2]), 4),
        },
        "threshold": threshold,
        "n_models": len(models),
        "model_agreement": model_signals,
        "unanimous": len(set(model_signals)) == 1,
    }


def batch_ensemble_predict(models, features_df, feature_names, threshold=0.75):
    """Run ensemble prediction on multiple rows at once (batch mode).

    Calls model.predict_proba(Pool(X_batch)) once per model on the full matrix
    (3 calls total for N rows, vs 3*N in naive approach).

    Args:
        models: list of CatBoost models
        features_df: DataFrame with feature columns + 'time'
        feature_names: list of feature column names
        threshold: minimum confidence for trade signal

    Returns: DataFrame with columns:
        time, signal, confidence, prob_no_trade, prob_long, prob_short,
        model_agreement, unanimous
    """
    from catboost import Pool

    # Extract feature matrix
    missing = set(feature_names) - set(features_df.columns)
    if missing:
        logger.warning(f"  Filling {len(missing)} missing features with 0")
        for feat in missing:
            features_df[feat] = 0.0

    X = features_df[feature_names].values
    times = features_df['time'].values

    # Batch predict — one call per model
    all_probas = []
    for i, model in enumerate(models):
        pool = Pool(X)
        proba = model.predict_proba(pool)  # shape: (N, 3)
        all_probas.append(proba)
        logger.debug(f"  Model {i}: predicted {len(proba)} rows")

    # Average probabilities across models: shape (N, 3)
    avg_proba = np.mean(all_probas, axis=0)

    # Per-row signal determination
    pred_classes = np.argmax(avg_proba, axis=1)
    confidences = np.max(avg_proba, axis=1)

    signals = []
    agreements = []
    unanimous_flags = []

    for row_idx in range(len(X)):
        pc = int(pred_classes[row_idx])
        conf = float(confidences[row_idx])

        if pc in TRADE_CLASSES and conf >= threshold:
            signal = CLASS_NAMES[pc]
        else:
            signal = "NO_TRADE"

        # Per-model agreement for this row
        model_sigs = []
        for model_proba in all_probas:
            mp = model_proba[row_idx]
            mpc = int(np.argmax(mp))
            if mpc in TRADE_CLASSES and mp[mpc] >= threshold:
                model_sigs.append(CLASS_NAMES[mpc])
            else:
                model_sigs.append("NO_TRADE")

        signals.append(signal)
        agreements.append(",".join(model_sigs))
        unanimous_flags.append(len(set(model_sigs)) == 1)

    result = pd.DataFrame({
        'time': times,
        'signal': signals,
        'confidence': np.round(confidences, 4),
        'prob_no_trade': np.round(avg_proba[:, 0], 4),
        'prob_long': np.round(avg_proba[:, 1], 4),
        'prob_short': np.round(avg_proba[:, 2], 4),
        'model_agreement': agreements,
        'unanimous': unanimous_flags,
    })

    n_trades = (result['signal'] != 'NO_TRADE').sum()
    logger.info(f"  Batch predict: {len(result)} rows, {n_trades} trade signals")
    return result


# ============================================================================
# Main Pipeline
# ============================================================================

def run_single_prediction(threshold=0.75, model_dir=None):
    """Run one complete prediction cycle.

    Returns: prediction dict with signal, confidence, timestamp, etc.
    """
    t0 = time.time()

    # Step 1: Load models
    logger.info("[1/4] Loading production models...")
    models, metadata = load_production_models(model_dir)
    feature_names = metadata['feature_names']

    # Step 2: Download live data
    logger.info("[2/4] Downloading live klines...")
    klines_dict = download_live_klines()

    # Step 3: Run ETL + Encode
    logger.info("[3/4] Running ETL + encoding pipeline...")
    t_etl = time.time()
    decomposed = run_live_etl(klines_dict)
    features_df = encode_live_features(klines_dict, decomposed)
    etl_time = time.time() - t_etl
    logger.info(f"  ETL + encoding: {etl_time:.1f}s")

    # Get last row
    last_row = features_df.iloc[-1]
    timestamp = str(last_row['time'])

    # Verify feature count
    available_features = [c for c in features_df.columns if c != 'time']
    expected_n = len(feature_names)
    actual_n = len(available_features)

    if actual_n != expected_n:
        # Check which features are missing vs extra
        missing = set(feature_names) - set(available_features)
        extra = set(available_features) - set(feature_names)
        if missing:
            logger.warning(f"  Missing {len(missing)} features: {list(missing)[:5]}...")
            # Fill missing with 0
            for feat in missing:
                features_df[feat] = 0.0
        if extra:
            logger.info(f"  {len(extra)} extra features (ignored)")

    # Step 4: Predict
    logger.info("[4/4] Running ensemble prediction...")
    prediction = ensemble_predict(models, last_row, feature_names, threshold)

    total_time = time.time() - t0
    prediction['timestamp'] = timestamp
    prediction['latency_sec'] = round(total_time, 1)

    return prediction


def main():
    parser = argparse.ArgumentParser(description="V10 Live Prediction Pipeline")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold for trade signals (default: 0.75)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory with production models (default: results_v10/production/)")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously at specified interval")
    parser.add_argument("--interval", type=int, default=300,
                        help="Prediction interval in seconds for loop mode (default: 300)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.loop:
        logger.info(f"Starting continuous prediction (interval={args.interval}s, threshold={args.threshold})")
        while True:
            try:
                prediction = run_single_prediction(
                    threshold=args.threshold, model_dir=args.model_dir)

                # Print result
                signal = prediction['signal']
                conf = prediction['confidence']
                ts = prediction['timestamp']
                latency = prediction['latency_sec']
                probs = prediction['probabilities']

                if signal == "NO_TRADE":
                    logger.info(f"[{ts}] NO_TRADE (conf={conf:.3f}) | "
                                f"L={probs['LONG']:.3f} S={probs['SHORT']:.3f} | "
                                f"{latency}s")
                else:
                    unanimous = prediction['unanimous']
                    agree_str = "UNANIMOUS" if unanimous else f"split: {prediction['model_agreement']}"
                    logger.info(f"[{ts}] >>> {signal} <<< (conf={conf:.3f}) | "
                                f"L={probs['LONG']:.3f} S={probs['SHORT']:.3f} | "
                                f"{agree_str} | {latency}s")

                # Also print JSON for programmatic consumption
                print(json.dumps(prediction, indent=2))

            except Exception as e:
                logger.error(f"Prediction failed: {e}", exc_info=True)

            logger.info(f"Sleeping {args.interval}s until next prediction...")
            time.sleep(args.interval)
    else:
        # Single prediction
        prediction = run_single_prediction(
            threshold=args.threshold, model_dir=args.model_dir)

        # Print result
        print(json.dumps(prediction, indent=2))

        signal = prediction['signal']
        conf = prediction['confidence']
        ts = prediction['timestamp']
        latency = prediction['latency_sec']

        logger.info(f"\nPrediction: {signal} (confidence={conf:.3f})")
        logger.info(f"Timestamp: {ts}")
        logger.info(f"Probabilities: {prediction['probabilities']}")
        logger.info(f"Model agreement: {prediction['model_agreement']}")
        logger.info(f"Total latency: {latency}s")


if __name__ == "__main__":
    main()
