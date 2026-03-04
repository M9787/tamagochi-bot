"""
V5 Feature Encoder — V3 (280) + 110 Directional Features = 390 total.

New directional features per TF (10 × 11 TFs = 110):
  1.  momentum_dir       = (slope_f - slope_b) × corr     (quality-weighted momentum direction)
  2.  cumsum_body_w10    = rolling sum(close - open, 10)   (candle body accumulation)
  3.  cumsum_body_w20    = rolling sum(close - open, 20)
  4.  cumsum_body_w50    = rolling sum(close - open, 50)
  5.  up_bar_ratio_w10   = count(close > open, 10) / 10   (bullish bar dominance)
  6.  up_bar_ratio_w20   = count(close > open, 20) / 20
  7.  up_bar_ratio_w50   = count(close > open, 50) / 50
  8.  stoch_pos_w10      = (close - min) / (max - min), 10 (position within range)
  9.  stoch_pos_w20      = (close - min) / (max - min), 20
  10. stoch_pos_w50      = (close - min) / (max - min), 50

Why directional: These features are ASYMMETRIC — they behave differently for
LONG vs SHORT setups, unlike V3's magnitude features (slope_f_mag, angle_slow)
which are symmetric.

Output: encoded_data/feature_matrix_v5.parquet

Usage:
    python model_training/encode_v5.py
    python model_training/encode_v5.py --force
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from core.config import TIMEFRAME_ORDER

ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
ENCODED_DIR = Path(__file__).parent / "encoded_data"

ROLLING_WINDOWS = [10, 20, 50]
NEW_FEATURES_PER_TF = 10  # 1 momentum_dir + 3 cumsum_body + 3 up_bar_ratio + 3 stoch_pos

logger = logging.getLogger(__name__)


def _load_klines(tf: str) -> pd.DataFrame:
    """Load raw OHLC kline data for a timeframe."""
    path = ACTUAL_DATA_DIR / f"ml_data_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run download_data.py first.")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['Open Time']).dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    return df


def build_directional_features(v3_features: pd.DataFrame) -> pd.DataFrame:
    """
    Add 110 directional features to existing V3 feature matrix.
    """
    base_times = pd.to_datetime(v3_features['time'])
    n = len(v3_features)
    new_features = {}

    for tf in TIMEFRAME_ORDER:
        logger.info(f"  Processing {tf}...")

        # --- Feature 1: momentum_dir (from V3 columns, already aligned) ---
        sfm = f'slope_f_mag_{tf}'
        sbs = f'slope_b_slow_{tf}'
        cs = f'corr_slow_{tf}'

        if sfm in v3_features.columns and sbs in v3_features.columns and cs in v3_features.columns:
            slope_f = v3_features[sfm].values.astype(np.float64)
            slope_b = v3_features[sbs].values.astype(np.float64)
            corr = v3_features[cs].values.astype(np.float64)
            new_features[f'momentum_dir_{tf}'] = ((slope_f - slope_b) * corr).astype(np.float32)
        else:
            logger.warning(f"  Missing V3 columns for {tf}, filling momentum_dir with 0")
            new_features[f'momentum_dir_{tf}'] = np.zeros(n, dtype=np.float32)

        # --- Features 2-10: Price-based (from raw klines) ---
        try:
            klines = _load_klines(tf)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping price features for {tf}: {e}")
            for w in ROLLING_WINDOWS:
                new_features[f'cumsum_body_{tf}_w{w}'] = np.zeros(n, dtype=np.float32)
                new_features[f'up_bar_ratio_{tf}_w{w}'] = np.zeros(n, dtype=np.float32)
                new_features[f'stoch_pos_{tf}_w{w}'] = np.zeros(n, dtype=np.float32)
            continue

        close = klines['Close'].values.astype(np.float64)
        open_ = klines['Open'].values.astype(np.float64)
        high = klines['High'].values.astype(np.float64)
        low = klines['Low'].values.astype(np.float64)

        body = close - open_  # positive = bullish, negative = bearish
        up_bar = (close > open_).astype(np.float64)

        body_series = pd.Series(body)
        up_series = pd.Series(up_bar)
        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)

        native_data = {'time': klines['time']}

        for w in ROLLING_WINDOWS:
            # cumsum_body: rolling sum of candle bodies
            native_data[f'cumsum_body_w{w}'] = body_series.rolling(
                w, min_periods=1).sum().values.astype(np.float64)

            # up_bar_ratio: fraction of bullish bars
            native_data[f'up_bar_ratio_w{w}'] = up_series.rolling(
                w, min_periods=1).mean().values.astype(np.float64)

            # stoch_pos: position within range
            roll_min = low_series.rolling(w, min_periods=1).min().values
            roll_max = high_series.rolling(w, min_periods=1).max().values
            range_val = roll_max - roll_min
            stoch = np.where(range_val > 0, (close - roll_min) / range_val, 0.5)
            native_data[f'stoch_pos_w{w}'] = stoch.astype(np.float64)

        native_df = pd.DataFrame(native_data)

        # Align to 5M base via merge_asof
        base_df = pd.DataFrame({'time': base_times})
        aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

        for w in ROLLING_WINDOWS:
            new_features[f'cumsum_body_{tf}_w{w}'] = aligned[f'cumsum_body_w{w}'].fillna(0).values.astype(np.float32)
            new_features[f'up_bar_ratio_{tf}_w{w}'] = aligned[f'up_bar_ratio_w{w}'].fillna(0.5).values.astype(np.float32)
            new_features[f'stoch_pos_{tf}_w{w}'] = aligned[f'stoch_pos_w{w}'].fillna(0.5).values.astype(np.float32)

    # Combine V3 + new directional features
    result = v3_features.copy()
    for col_name, col_data in new_features.items():
        result[col_name] = col_data

    return result


def _verify_features(df: pd.DataFrame, n_v3: int):
    """Spot-check V5 feature invariants."""
    logger.info("Running verification checks...")
    errors = 0

    expected_new = NEW_FEATURES_PER_TF * len(TIMEFRAME_ORDER)
    expected_total = n_v3 + expected_new
    actual_features = len(df.columns) - 1  # minus 'time'
    if actual_features != expected_total:
        logger.error(f"  FAIL: Expected {expected_total} features, got {actual_features}")
        errors += 1
    else:
        logger.info(f"  OK: {actual_features} features ({n_v3} V3 + {expected_new} directional)")

    # Check no NaN
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.error(f"  FAIL: NaN in columns: {nan_cols[:10]}...")
        errors += 1
    else:
        logger.info("  OK: No NaN values")

    # Check directional features have variance
    logger.info("\n  Directional feature stats:")
    for tf in ['3D', '1D', '6H', '1H', '5M']:
        col = f'momentum_dir_{tf}'
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            logger.info(f"    {col}: mean={mean:.6f}, std={std:.6f}")

        col = f'up_bar_ratio_{tf}_w20'
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            logger.info(f"    {col}: mean={mean:.4f}, std={std:.4f}")

        col = f'stoch_pos_{tf}_w20'
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            logger.info(f"    {col}: mean={mean:.4f}, std={std:.4f}")

    if errors == 0:
        logger.info(f"\n  All verification checks PASSED")
    else:
        logger.warning(f"\n  {errors} verification check(s) FAILED")
    return errors


def run_encoding(force: bool = False):
    """Run V5 encoding: load V3 features + add directional features."""
    logger.info("=" * 60)
    logger.info("V5 FEATURE ENCODER (V3 + Directional)")
    logger.info(f"  V3 base: 280 features")
    logger.info(f"  New: {NEW_FEATURES_PER_TF} per TF × {len(TIMEFRAME_ORDER)} TFs "
                f"= {NEW_FEATURES_PER_TF * len(TIMEFRAME_ORDER)} directional")
    logger.info(f"  Rolling windows: {ROLLING_WINDOWS}")
    logger.info("=" * 60)

    v5_path = ENCODED_DIR / "feature_matrix_v5.parquet"
    if v5_path.exists() and not force:
        logger.info(f"feature_matrix_v5.parquet already exists. Use --force to rebuild.")
        return

    # Load V3 feature matrix
    v3_path = ENCODED_DIR / "feature_matrix_v3.parquet"
    if not v3_path.exists():
        raise FileNotFoundError(f"Missing: {v3_path}\nRun: python model_training/encode_v3.py")

    logger.info(f"\nLoading V3 features from {v3_path}...")
    v3_features = pd.read_parquet(v3_path)
    n_v3 = len(v3_features.columns) - 1  # minus 'time'
    logger.info(f"  V3 shape: {v3_features.shape} ({n_v3} features)")

    t0 = time.time()
    df = build_directional_features(v3_features)
    elapsed = time.time() - t0

    logger.info(f"\n  Features computed in {elapsed:.1f}s")
    logger.info(f"  Shape: {df.shape}")

    _verify_features(df, n_v3)

    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving parquet...")
    df.to_parquet(v5_path, engine='pyarrow', index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"V5 ENCODING COMPLETE")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Features: {len(df.columns) - 1} ({n_v3} V3 + "
                f"{NEW_FEATURES_PER_TF * len(TIMEFRAME_ORDER)} directional)")
    logger.info(f"  Parquet: {v5_path}")
    logger.info(f"  Size: {v5_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="V5 encoder: V3 + directional features")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    run_encoding(force=args.force)
