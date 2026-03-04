"""
V3.1 Feature Encoder — 280 features (V2.4 levels + V3 deltas combined).

Keeps all important V2.4 level features (slope_f_mag, angle_slow, corr_slow,
slope_b_slow, cross_div) that carry the predictive signal. ADDS first-differenced
(Δ) features and cumsum directional anchors for stationarity and stability.
Removes only dead features: reversal_* (0/11), direction_* (0/11).

Per TF (25 features × 11 TFs = 275):
  --- V2.4 Level Features (kept for signal) ---
  1.  cross_div       = angle_w30 - angle_w120     (crossing divergence)
  2.  slope_f_mag     = slope_f_w120 continuous     (trend strength — V2 #1)
  3.  angle_slow      = angle_w120 continuous       (current state — V2 #2)
  4.  corr_slow       = corr_w120                   (signal quality separator)
  5.  slope_b_slow    = slope_b_w120                (prior regime)
  6.  cross_div_lag1  = TF-native lag1 of cross_div
  7.  cross_div_lag2  = TF-native lag2 of cross_div

  --- Already-Δ Features (shared) ---
  8.  cross_traj      = Δ(cross_div)                (divergence velocity)
  9.  accel_raw       = Δangle_w120                 (momentum direction)
  10. accel_mag       = |accel_raw|                  (impulse magnitude)

  --- Categorical (shared) ---
  11. cross_dir       = opposing-derivative crossing (+1/-1/0)
  12. cross_dir_lag1  = TF-native lag1 of cross_dir

  --- NEW Δ Features (for stability) ---
  13. d_slope_f       = Δ slope_f_w120
  14. d_slope_b       = Δ slope_b_w120
  15. d_corr          = Δ corr_w120
  16. d_accel         = Δ accel_w120 (jerk)

  --- Cumsum Directional Anchors (new) ---
  17. cs_dsf_w3       = rolling sum(d_slope_f, 3)
  18. cs_dsf_w5       = rolling sum(d_slope_f, 5)
  19. cs_dsf_w7       = rolling sum(d_slope_f, 7)
  20. cs_dsf_w11      = rolling sum(d_slope_f, 11)
  21. cs_dsf_w17      = rolling sum(d_slope_f, 17)
  22. cs_dsf_w19      = rolling sum(d_slope_f, 19)
  23. cs_dsf_w23      = rolling sum(d_slope_f, 23)

  --- TF-Native Δ Lags (new) ---
  24. d_slope_f_lag1  = TF-native lag1 of d_slope_f
  25. d_corr_lag1     = TF-native lag1 of d_corr

Cross-TF summaries (5):
  cross_long_count, cross_short_count, direction_agreement,
  trend_long_count, trend_short_count

Total: 25 × 11 + 5 = 280 features, ALL numeric.

Output: encoded_data/feature_matrix_v3.parquet

Usage:
    python model_training/encode_v3.py
    python model_training/encode_v3.py --force
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

DECOMPOSED_DIR = Path(__file__).parent / "decomposed_data"
ENCODED_DIR = Path(__file__).parent / "encoded_data"

W_FAST = 30
W_SLOW = 120

CUMSUM_WINDOWS = [3, 5, 7, 11, 17, 19, 23]

FEATURES_PER_TF = 25

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def _load_decomposed(tf: str, ws: int) -> pd.DataFrame:
    """Load one decomposed CSV."""
    path = DECOMPOSED_DIR / f"decomposed_{tf}_w{ws}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run etl.py first.")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df


def _build_5m_base() -> pd.DataFrame:
    """Build the 5M base timeline from decomposed_5M_w30.csv."""
    df = _load_decomposed("5M", W_FAST)
    base = pd.DataFrame({'time': df['time'].sort_values().reset_index(drop=True)})
    return base


# =============================================================================
# Feature Builder
# =============================================================================

def _all_feature_names(tf: str) -> list:
    """Return ordered list of all feature column names for a given TF."""
    names = [
        # V2.4 level features
        f'cross_div_{tf}',
        f'slope_f_mag_{tf}',
        f'angle_slow_{tf}',
        f'corr_slow_{tf}',
        f'slope_b_slow_{tf}',
        f'cross_div_{tf}_lag1',
        f'cross_div_{tf}_lag2',
        # Shared (already-Δ)
        f'cross_traj_{tf}',
        f'accel_raw_{tf}',
        f'accel_mag_{tf}',
        # Categorical
        f'cross_dir_{tf}',
        f'cross_dir_{tf}_lag1',
        # New Δ features
        f'd_slope_f_{tf}',
        f'd_slope_b_{tf}',
        f'd_corr_{tf}',
        f'd_accel_{tf}',
    ]
    # Cumsums
    for w in CUMSUM_WINDOWS:
        names.append(f'cs_dsf_{tf}_w{w}')
    # Δ lags
    names.extend([
        f'd_slope_f_{tf}_lag1',
        f'd_corr_{tf}_lag1',
    ])
    return names


def build_features(base: pd.DataFrame) -> pd.DataFrame:
    """
    Build 280 features (V2.4 levels + V3 deltas) aligned to 5M base.
    """
    n = len(base)
    base_times = base['time']
    features = {}

    for tf in TIMEFRAME_ORDER:
        logger.info(f"  Processing {tf}...")

        try:
            df_fast = _load_decomposed(tf, W_FAST)
            df_slow = _load_decomposed(tf, W_SLOW)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping {tf}: {e}")
            for col_name in _all_feature_names(tf):
                dtype = np.int8 if col_name.startswith(('cross_dir',)) else np.float32
                features[col_name] = np.zeros(n, dtype=dtype)
            continue

        # Merge w30 and w120 by time within this TF
        merged_native = pd.merge(
            df_fast[['time', 'angle', 'slope_f']].rename(
                columns={'angle': 'angle_fast', 'slope_f': 'slope_f_fast'}),
            df_slow[['time', 'angle', 'slope_f', 'acceleration', 'corr', 'slope_b']].rename(
                columns={'angle': 'angle_slow', 'slope_f': 'slope_f_slow',
                         'acceleration': 'accel_slow', 'corr': 'corr_slow',
                         'slope_b': 'slope_b_slow'}),
            on='time', how='inner'
        ).sort_values('time').reset_index(drop=True)

        mn = merged_native

        # === V2.4 Level Features ===
        slope_f_vals = mn['slope_f_slow'].fillna(0).values.astype(np.float64)
        slope_b_vals = mn['slope_b_slow'].fillna(0).values.astype(np.float64)
        corr_vals = mn['corr_slow'].fillna(0).values.astype(np.float64)
        accel_vals = mn['accel_slow'].fillna(0).values.astype(np.float64)
        angle_slow_native = mn['angle_slow'].fillna(0).values.astype(np.float64)

        cross_div_native = (mn['angle_fast'] - mn['angle_slow']).values.astype(np.float64)

        # cross_traj = Δ(cross_div)
        cross_traj_native = np.zeros_like(cross_div_native)
        cross_traj_native[1:] = np.diff(cross_div_native)

        accel_raw_native = accel_vals.copy()
        accel_mag_native = np.abs(accel_raw_native)

        # cross_dir — opposing-derivative crossing direction
        angle_fast_vals = mn['angle_fast'].values
        angle_slow_vals = mn['angle_slow'].values
        deriv_fast = np.zeros(len(angle_fast_vals), dtype=np.float64)
        deriv_slow = np.zeros(len(angle_slow_vals), dtype=np.float64)
        deriv_fast[1:] = np.diff(angle_fast_vals)
        deriv_slow[1:] = np.diff(angle_slow_vals)
        opposing = (deriv_fast * deriv_slow) < 0
        cross_dir_native = np.where(
            opposing & (deriv_slow > 0), 1,
            np.where(opposing & (deriv_slow < 0), -1, 0)
        ).astype(np.int8)

        # === NEW Δ Features ===
        d_slope_f = np.zeros_like(slope_f_vals)
        d_slope_f[1:] = np.diff(slope_f_vals)

        d_slope_b = np.zeros_like(slope_b_vals)
        d_slope_b[1:] = np.diff(slope_b_vals)

        d_corr = np.zeros_like(corr_vals)
        d_corr[1:] = np.diff(corr_vals)

        d_accel = np.zeros_like(accel_vals)
        d_accel[1:] = np.diff(accel_vals)

        # === Cumsum Directional Anchors ===
        d_slope_f_series = pd.Series(d_slope_f)
        cumsums = {}
        for w in CUMSUM_WINDOWS:
            cumsums[w] = d_slope_f_series.rolling(w, min_periods=1).sum().values.astype(np.float64)

        # === TF-Native Lags (all computed before merge_asof) ===
        cross_div_lag1_native = np.zeros_like(cross_div_native)
        cross_div_lag1_native[1:] = cross_div_native[:-1]
        cross_div_lag2_native = np.zeros_like(cross_div_native)
        cross_div_lag2_native[2:] = cross_div_native[:-2]

        cross_dir_lag1_native = np.zeros(len(cross_dir_native), dtype=np.int8)
        cross_dir_lag1_native[1:] = cross_dir_native[:-1]

        d_slope_f_lag1 = np.zeros_like(d_slope_f)
        d_slope_f_lag1[1:] = d_slope_f[:-1]

        d_corr_lag1 = np.zeros_like(d_corr)
        d_corr_lag1[1:] = d_corr[:-1]

        # === Build native-TF DataFrame ===
        native_data = {
            'time': mn['time'],
            # V2.4 levels
            'cross_div': cross_div_native,
            'slope_f_mag': slope_f_vals,
            'angle_slow': angle_slow_native,
            'corr_slow': corr_vals,
            'slope_b_slow': slope_b_vals,
            'cross_div_lag1': cross_div_lag1_native,
            'cross_div_lag2': cross_div_lag2_native,
            # Shared
            'cross_traj': cross_traj_native,
            'accel_raw': accel_raw_native,
            'accel_mag': accel_mag_native,
            'cross_dir': cross_dir_native,
            'cross_dir_lag1': cross_dir_lag1_native,
            # New Δ
            'd_slope_f': d_slope_f,
            'd_slope_b': d_slope_b,
            'd_corr': d_corr,
            'd_accel': d_accel,
            # Δ lags
            'd_slope_f_lag1': d_slope_f_lag1,
            'd_corr_lag1': d_corr_lag1,
        }
        for w in CUMSUM_WINDOWS:
            native_data[f'cs_dsf_w{w}'] = cumsums[w]

        native_df = pd.DataFrame(native_data)

        # Align to 5M base
        base_df = pd.DataFrame({'time': base_times})
        aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

        # Store all 25 features per TF
        # V2.4 levels
        features[f'cross_div_{tf}'] = aligned['cross_div'].fillna(0).values.astype(np.float32)
        features[f'slope_f_mag_{tf}'] = aligned['slope_f_mag'].fillna(0).values.astype(np.float32)
        features[f'angle_slow_{tf}'] = aligned['angle_slow'].fillna(0).values.astype(np.float32)
        features[f'corr_slow_{tf}'] = aligned['corr_slow'].fillna(0).values.astype(np.float32)
        features[f'slope_b_slow_{tf}'] = aligned['slope_b_slow'].fillna(0).values.astype(np.float32)
        features[f'cross_div_{tf}_lag1'] = aligned['cross_div_lag1'].fillna(0).values.astype(np.float32)
        features[f'cross_div_{tf}_lag2'] = aligned['cross_div_lag2'].fillna(0).values.astype(np.float32)
        # Shared
        features[f'cross_traj_{tf}'] = aligned['cross_traj'].fillna(0).values.astype(np.float32)
        features[f'accel_raw_{tf}'] = aligned['accel_raw'].fillna(0).values.astype(np.float32)
        features[f'accel_mag_{tf}'] = aligned['accel_mag'].fillna(0).values.astype(np.float32)
        features[f'cross_dir_{tf}'] = aligned['cross_dir'].fillna(0).values.astype(np.int8)
        features[f'cross_dir_{tf}_lag1'] = aligned['cross_dir_lag1'].fillna(0).values.astype(np.int8)
        # New Δ
        features[f'd_slope_f_{tf}'] = aligned['d_slope_f'].fillna(0).values.astype(np.float32)
        features[f'd_slope_b_{tf}'] = aligned['d_slope_b'].fillna(0).values.astype(np.float32)
        features[f'd_corr_{tf}'] = aligned['d_corr'].fillna(0).values.astype(np.float32)
        features[f'd_accel_{tf}'] = aligned['d_accel'].fillna(0).values.astype(np.float32)
        # Cumsums
        for w in CUMSUM_WINDOWS:
            features[f'cs_dsf_{tf}_w{w}'] = aligned[f'cs_dsf_w{w}'].fillna(0).values.astype(np.float32)
        # Δ lags
        features[f'd_slope_f_{tf}_lag1'] = aligned['d_slope_f_lag1'].fillna(0).values.astype(np.float32)
        features[f'd_corr_{tf}_lag1'] = aligned['d_corr_lag1'].fillna(0).values.astype(np.float32)

    # === Cross-TF Summaries ===
    cross_long_count = np.zeros(n, dtype=np.int8)
    cross_short_count = np.zeros(n, dtype=np.int8)
    for tf in TIMEFRAME_ORDER:
        col = f'cross_dir_{tf}'
        if col in features:
            cross_long_count += (features[col] == 1).astype(np.int8)
            cross_short_count += (features[col] == -1).astype(np.int8)
    features['cross_long_count'] = cross_long_count
    features['cross_short_count'] = cross_short_count
    features['direction_agreement'] = np.maximum(cross_long_count, cross_short_count).astype(np.int8)

    trend_long_count = np.zeros(n, dtype=np.int8)
    trend_short_count = np.zeros(n, dtype=np.int8)
    for tf in TIMEFRAME_ORDER:
        col = f'cs_dsf_{tf}_w7'
        if col in features:
            trend_long_count += (features[col] > 0).astype(np.int8)
            trend_short_count += (features[col] < 0).astype(np.int8)
    features['trend_long_count'] = trend_long_count
    features['trend_short_count'] = trend_short_count

    result = pd.DataFrame(features)
    result.insert(0, 'time', base_times.values)
    return result


# =============================================================================
# Verification
# =============================================================================

def _verify_features(df: pd.DataFrame):
    """Spot-check V3.1 feature invariants."""
    logger.info("Running verification checks...")
    errors = 0

    expected_features = FEATURES_PER_TF * len(TIMEFRAME_ORDER) + 5
    actual_features = len(df.columns) - 1
    if actual_features != expected_features:
        logger.error(f"  FAIL: Expected {expected_features} features, got {actual_features}")
        errors += 1
    else:
        logger.info(f"  OK: {actual_features} features ({FEATURES_PER_TF} x 11 + 5)")

    for col in df.columns:
        if col == 'time':
            continue
        if df[col].dtype == object:
            logger.error(f"  FAIL: Column {col} is object type")
            errors += 1

    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.error(f"  FAIL: NaN in columns: {nan_cols}")
        errors += 1
    else:
        logger.info("  OK: No NaN values")

    # Check V2.4 kings are present
    for tf in ['3D', '6H', '12H', '4H']:
        for feat in ['slope_f_mag', 'angle_slow', 'corr_slow', 'slope_b_slow']:
            col = f'{feat}_{tf}'
            if col not in df.columns:
                logger.error(f"  FAIL: Missing V2.4 king: {col}")
                errors += 1
            else:
                std = df[col].std()
                logger.info(f"  OK: {col} std={std:.6f}")

    # Stationarity check on Δ features
    logger.info("\n  Stationarity check (Δ features):")
    for tf in ['3D', '1D', '1H', '5M']:
        for feat in ['d_slope_f', 'd_corr']:
            col = f'{feat}_{tf}'
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                ratio = abs(mean) / std if std > 0 else 0
                logger.info(f"  {col}: mean={mean:.6f}, std={std:.6f}, ratio={ratio:.4f}")

    if errors == 0:
        logger.info(f"\n  All verification checks PASSED")
    else:
        logger.warning(f"\n  {errors} verification check(s) FAILED")
    return errors


# =============================================================================
# Main Pipeline
# =============================================================================

def run_encoding(force: bool = False):
    """Run the V3.1 combined feature encoding pipeline."""
    total_features = FEATURES_PER_TF * len(TIMEFRAME_ORDER) + 5
    logger.info("=" * 60)
    logger.info("V3.1 COMBINED FEATURE ENCODER (levels + deltas)")
    logger.info(f"  2 windows (w{W_FAST} fast, w{W_SLOW} slow) x 11 TFs")
    logger.info(f"  {FEATURES_PER_TF} per TF + 5 summaries = {total_features} features")
    logger.info(f"  V2.4 levels (signal) + V3 deltas (stability)")
    logger.info("=" * 60)

    if not DECOMPOSED_DIR.exists():
        raise FileNotFoundError(
            f"decomposed_data/ not found: {DECOMPOSED_DIR}\n"
            f"Run: python model_training/etl.py --start 2020-01-01 --end 2026-02-15"
        )

    parquet_path = ENCODED_DIR / "feature_matrix_v3.parquet"
    if parquet_path.exists() and not force:
        logger.info(f"feature_matrix_v3.parquet already exists. Use --force to rebuild.")
        return

    ENCODED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building 5M base timeline...")
    base = _build_5m_base()
    logger.info(f"  Base: {len(base)} rows, {base['time'].min()} to {base['time'].max()}")

    t0 = time.time()
    df = build_features(base)
    elapsed = time.time() - t0

    logger.info(f"  Features computed in {elapsed:.1f}s")
    logger.info(f"  Shape: {df.shape}")

    _verify_features(df)

    logger.info(f"\nSaving parquet...")
    df.to_parquet(parquet_path, engine='pyarrow', index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"V3.1 ENCODING COMPLETE")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Features: {len(df.columns) - 1} (all numeric)")
    logger.info(f"  Parquet: {parquet_path}")
    logger.info(f"  Size: {parquet_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="V3.1 combined feature encoder (280 features)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    run_encoding(force=args.force)
