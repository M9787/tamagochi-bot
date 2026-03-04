"""
V6 Feature Encoder — V5 KEEP (~203) + 20 New Directional Concepts (~192) = ~395 total.

Replaces 185 dead V5 features with 192 purposeful ones targeting DIRECTION prediction.
Three data sources never used before: p_value_f, Volume, and wick asymmetry.

New feature concepts (Aristotle's Four Causes):

I.  MATERIAL CAUSE — New Data Sources (p_value, Volume, Wicks):
    1.  trend_certainty      = -log10(clip(p_value_f_w120))     per TF
    2.  vol_ratio             = volume / ema(volume, 50)         per TF
    3.  vol_body_product      = sign(C-O) * vol_ratio            per TF
    4.  wick_asymmetry        = rolling_mean(wick_formula, 10)   per TF

II. FORMAL CAUSE — Regime Type & Signal Shape:
    5.  regime_quadrant       = sign(slope_f)*2 + sign(slope_b)  per TF
    6.  corr_regime           = corr * sign(slope_f)             per TF
    7.  corr_stability        = rolling_std(corr, 10)            per TF
    8.  slope_sign_gradient   = sign(slope_f_w30)-sign(slope_f_w120) per TF
    9.  angle_regime          = min(angle,30) * sign(slope_f)    per TF
    10. regime_change_strength= |sf-sb|*certainty*(sign differs) per TF

III.EFFICIENT CAUSE — Momentum, Force, Conviction:
    11. smoothed_momentum     = ema(slope_f-slope_b, 5)          per TF
    12. accel_signed          = acceleration * sign(slope_f)     per TF
    13. vol_impulse           = z-score(volume, 50)              per TF
    14. norm_body_accum       = rolling_sum(sign(C-O)*|C-O|/C, 20) per TF

IV. FINAL CAUSE — Trade Outcome Relevance:
    15. range_position_signed = (2*stoch_pos_w20-1)*sign(slope_f) per TF
    16. atr_normalized        = ATR(14)/close                    x3 (5M,1D,4H)
    17. cross_tf_slope_agreement = mean(sign(slope_f) for top 8 TFs) x1
    18. cross_tf_weighted_slope  = weighted_mean(slope_f, tf_weights) x1
    19. p_value_trend         = trend_certainty - lag1            per TF
    20. directional_vol_body  = ema(vol_body_product, 10)        per TF

Output: encoded_data/feature_matrix_v6.parquet

Usage:
    python model_training/encode_v6.py
    python model_training/encode_v6.py --force
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

# Directories
DECOMPOSED_DIR = Path(__file__).parent / "decomposed_data"
ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
ENCODED_DIR = Path(__file__).parent / "encoded_data"

W_FAST = 30
W_SLOW = 120

# Timeframe weights from signal_logic.py
TF_WEIGHTS = {
    "3D": 5, "1D": 4, "12H": 3.5, "8H": 3, "6H": 2.5,
    "4H": 2, "2H": 1.5, "1H": 1.2, "30M": 1.1, "15M": 1.0, "5M": 0.8
}

# Top 8 TFs for slope agreement (exclude Youngs)
TOP_8_TFS = ["3D", "1D", "12H", "8H", "6H", "4H", "2H", "1H"]

# ATR only for these TFs
ATR_TFS = ["5M", "1D", "4H"]

logger = logging.getLogger(__name__)


# =============================================================================
# KEEP List — features retained from V5
# =============================================================================

def _build_keep_list():
    """Build the list of feature columns to keep from V5 (~203 features)."""
    keep = []
    for tf in TIMEFRAME_ORDER:
        # V3 base features (13 per TF)
        keep.extend([
            f'cross_div_{tf}',
            f'slope_f_mag_{tf}',
            f'angle_slow_{tf}',
            f'corr_slow_{tf}',
            f'slope_b_slow_{tf}',
            f'cross_div_{tf}_lag1',
            f'cross_div_{tf}_lag2',
            f'cross_traj_{tf}',
            f'accel_raw_{tf}',
            f'accel_mag_{tf}',
            f'cross_dir_{tf}',
            f'cross_dir_{tf}_lag1',
            f'cs_dsf_{tf}_w23',
        ])
        # V5 directional: cumsum_body w10,w50 (2 per TF)
        for w in [10, 50]:
            keep.append(f'cumsum_body_{tf}_w{w}')
        # V5 directional: stoch_pos w10,w20,w50 (3 per TF)
        for w in [10, 20, 50]:
            keep.append(f'stoch_pos_{tf}_w{w}')
    # Specific keeps from V3 deltas (4 features)
    keep.extend([
        'd_slope_f_3D', 'd_slope_f_8H',
        'd_slope_b_1D', 'd_slope_b_3D',
    ])
    # Specific keeps from V5 directional (1 feature)
    keep.append('up_bar_ratio_3D_w50')
    return keep


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


def _load_klines(tf: str) -> pd.DataFrame:
    """Load raw OHLCV kline data for a timeframe."""
    path = ACTUAL_DATA_DIR / f"ml_data_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run download_data.py first.")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['Open Time']).dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    return df


def _compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range using EMA smoothing."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        alpha = 2.0 / (period + 1)
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


# =============================================================================
# Phase B: Decomposed-based features (regression outputs)
# =============================================================================

def build_decomposed_features(base_times, n):
    """
    Compute 10 features per TF from decomposed CSVs:
    trend_certainty, regime_quadrant, corr_regime, corr_stability,
    slope_sign_gradient, angle_regime, regime_change_strength,
    smoothed_momentum, accel_signed, p_value_trend.

    Also returns slope_f_per_tf for cross-TF summaries.
    """
    features = {}
    slope_f_per_tf = {}

    for tf in TIMEFRAME_ORDER:
        logger.info(f"  [Decomposed] {tf}...")

        try:
            df_fast = _load_decomposed(tf, W_FAST)
            df_slow = _load_decomposed(tf, W_SLOW)
        except FileNotFoundError as e:
            logger.warning(f"    Skipping {tf}: {e}")
            for feat in ['trend_certainty', 'regime_quadrant', 'corr_regime',
                         'corr_stability', 'slope_sign_gradient', 'angle_regime',
                         'regime_change_strength', 'smoothed_momentum',
                         'accel_signed', 'p_value_trend']:
                features[f'{feat}_{tf}'] = np.zeros(n, dtype=np.float32)
            continue

        # Merge w30 and w120 by time within this TF
        merged = pd.merge(
            df_fast[['time', 'slope_f']].rename(columns={'slope_f': 'slope_f_fast'}),
            df_slow[['time', 'slope_f', 'slope_b', 'p_value_f', 'corr',
                      'angle', 'acceleration']].rename(
                columns={'slope_f': 'slope_f_slow', 'slope_b': 'slope_b_slow',
                         'p_value_f': 'p_value_f_slow', 'corr': 'corr_slow',
                         'angle': 'angle_slow', 'acceleration': 'accel_slow'}),
            on='time', how='inner'
        ).sort_values('time').reset_index(drop=True)

        mn = merged
        sf_slow = mn['slope_f_slow'].fillna(0).values.astype(np.float64)
        sb_slow = mn['slope_b_slow'].fillna(0).values.astype(np.float64)
        pf_slow = mn['p_value_f_slow'].fillna(1.0).values.astype(np.float64)
        corr_vals = mn['corr_slow'].fillna(0).values.astype(np.float64)
        angle_vals = mn['angle_slow'].fillna(0).values.astype(np.float64)
        accel_vals = mn['accel_slow'].fillna(0).values.astype(np.float64)
        sf_fast = mn['slope_f_fast'].fillna(0).values.astype(np.float64)

        sign_sf = np.sign(sf_slow)

        # 1. trend_certainty = -log10(clip(p_value_f, 1e-15, 1.0))
        pf_clipped = np.clip(pf_slow, 1e-15, 1.0)
        trend_certainty = -np.log10(pf_clipped)

        # 5. regime_quadrant = sign(slope_f)*2 + sign(slope_b) → {-3,-1,+1,+3}
        regime_quadrant = np.sign(sf_slow) * 2 + np.sign(sb_slow)

        # 6. corr_regime = corr * sign(slope_f)
        corr_regime = corr_vals * sign_sf

        # 7. corr_stability = rolling_std(corr, 10) on native TF
        corr_stability = pd.Series(corr_vals).rolling(
            10, min_periods=1).std().fillna(0).values

        # 8. slope_sign_gradient = sign(slope_f_w30) - sign(slope_f_w120) → {-2, 0, +2}
        slope_sign_gradient = np.sign(sf_fast) - np.sign(sf_slow)

        # 9. angle_regime = min(angle, 30) * sign(slope_f)
        #    angle is always positive (divergence magnitude), signing adds direction
        angle_regime = np.minimum(angle_vals, 30.0) * sign_sf

        # 10. regime_change_strength = |slope_f - slope_b| * trend_certainty
        #     when sign(slope_f) != sign(slope_b), else 0
        sign_differ = np.sign(sf_slow) != np.sign(sb_slow)
        rcs = np.abs(sf_slow - sb_slow) * trend_certainty
        regime_change_strength = np.where(sign_differ, rcs, 0.0)

        # 11. smoothed_momentum = ema(slope_f - slope_b, span=5) on native TF
        smoothed_momentum = pd.Series(sf_slow - sb_slow).ewm(
            span=5, min_periods=1).mean().values

        # 12. accel_signed = acceleration * sign(slope_f)
        accel_signed = accel_vals * sign_sf

        # 19. p_value_trend = trend_certainty - trend_certainty_lag1 (TF-native)
        tc_lag1 = np.zeros_like(trend_certainty)
        tc_lag1[1:] = trend_certainty[:-1]
        p_value_trend = trend_certainty - tc_lag1

        # Build native DataFrame for merge_asof
        native_df = pd.DataFrame({
            'time': mn['time'],
            'trend_certainty': trend_certainty,
            'regime_quadrant': regime_quadrant,
            'corr_regime': corr_regime,
            'corr_stability': corr_stability,
            'slope_sign_gradient': slope_sign_gradient,
            'angle_regime': angle_regime,
            'regime_change_strength': regime_change_strength,
            'smoothed_momentum': smoothed_momentum,
            'accel_signed': accel_signed,
            'p_value_trend': p_value_trend,
            'slope_f_slow': sf_slow,
        })

        # Align to 5M base
        base_df = pd.DataFrame({'time': base_times})
        aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

        for feat in ['trend_certainty', 'regime_quadrant', 'corr_regime',
                      'corr_stability', 'slope_sign_gradient', 'angle_regime',
                      'regime_change_strength', 'smoothed_momentum',
                      'accel_signed', 'p_value_trend']:
            features[f'{feat}_{tf}'] = aligned[feat].fillna(0).values.astype(np.float32)

        # Store slope_f for cross-TF features
        slope_f_per_tf[tf] = aligned['slope_f_slow'].fillna(0).values.astype(np.float64)

    return features, slope_f_per_tf


# =============================================================================
# Phase C: Kline-based features (price/volume)
# =============================================================================

def build_kline_features(base_times, n):
    """
    Compute 6 features per TF from raw klines + ATR for 3 TFs:
    vol_ratio, vol_body_product, wick_asymmetry,
    vol_impulse, norm_body_accum, directional_vol_body.
    Plus atr_normalized for 5M, 1D, 4H.
    """
    features = {}

    for tf in TIMEFRAME_ORDER:
        logger.info(f"  [Klines] {tf}...")

        try:
            klines = _load_klines(tf)
        except FileNotFoundError as e:
            logger.warning(f"    Skipping {tf}: {e}")
            for feat in ['vol_ratio', 'vol_body_product', 'wick_asymmetry',
                         'vol_impulse', 'norm_body_accum', 'directional_vol_body']:
                features[f'{feat}_{tf}'] = np.zeros(n, dtype=np.float32)
            if tf in ATR_TFS:
                features[f'atr_normalized_{tf}'] = np.zeros(n, dtype=np.float32)
            continue

        close = klines['Close'].values.astype(np.float64)
        open_ = klines['Open'].values.astype(np.float64)
        high = klines['High'].values.astype(np.float64)
        low = klines['Low'].values.astype(np.float64)
        volume = klines['Volume'].values.astype(np.float64)

        vol_series = pd.Series(volume)
        body_sign = np.sign(close - open_)

        # 2. vol_ratio = volume / ema(volume, 50)
        vol_ema50 = vol_series.ewm(span=50, min_periods=1).mean().values
        vol_ratio = np.where(vol_ema50 > 0, volume / vol_ema50, 1.0)

        # 3. vol_body_product = sign(close - open) * vol_ratio
        vol_body_product = body_sign * vol_ratio

        # 20. directional_vol_body = ema(vol_body_product, span=10) on native TF
        directional_vol_body = pd.Series(vol_body_product).ewm(
            span=10, min_periods=1).mean().values

        # 4. wick_asymmetry = rolling_mean(((high-max(O,C))-(min(O,C)-low))/(high-low+eps), 10)
        body_top = np.maximum(close, open_)
        body_bot = np.minimum(close, open_)
        bar_range = high - low
        eps = 1e-10
        upper_wick = high - body_top
        lower_wick = body_bot - low
        wick_raw = (upper_wick - lower_wick) / (bar_range + eps)
        wick_asymmetry = pd.Series(wick_raw).rolling(
            10, min_periods=1).mean().values

        # 13. vol_impulse = (volume - mean(vol, 50)) / std(vol, 50) (z-score)
        vol_mean50 = vol_series.rolling(50, min_periods=1).mean().values
        vol_std50 = vol_series.rolling(50, min_periods=1).std().fillna(1).values
        vol_std50 = np.where(vol_std50 > 0, vol_std50, 1.0)
        vol_impulse = (volume - vol_mean50) / vol_std50

        # 14. norm_body_accum = rolling_sum(sign(C-O)*|C-O|/C, 20)
        safe_close = np.where(close > 0, close, 1.0)
        body_pct = body_sign * np.abs(close - open_) / safe_close
        norm_body_accum = pd.Series(body_pct).rolling(
            20, min_periods=1).sum().values

        # Build native DataFrame
        native_data = {
            'time': klines['time'],
            'vol_ratio': vol_ratio,
            'vol_body_product': vol_body_product,
            'directional_vol_body': directional_vol_body,
            'wick_asymmetry': wick_asymmetry,
            'vol_impulse': vol_impulse,
            'norm_body_accum': norm_body_accum,
        }

        # 16. atr_normalized = ATR(14) / close (only for 5M, 1D, 4H)
        if tf in ATR_TFS:
            atr = _compute_atr(high, low, close, period=14)
            atr_norm = np.where(close > 0, atr / close, 0.0)
            native_data['atr_normalized'] = np.nan_to_num(atr_norm, nan=0.0)

        native_df = pd.DataFrame(native_data)

        # Align to 5M base
        base_df = pd.DataFrame({'time': base_times})
        aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

        for feat in ['vol_ratio', 'vol_body_product', 'directional_vol_body',
                      'wick_asymmetry', 'vol_impulse', 'norm_body_accum']:
            features[f'{feat}_{tf}'] = aligned[feat].fillna(0).values.astype(np.float32)

        if tf in ATR_TFS:
            features[f'atr_normalized_{tf}'] = aligned['atr_normalized'].fillna(0).values.astype(np.float32)

    return features


# =============================================================================
# Phase D: Cross-TF summaries
# =============================================================================

def build_cross_tf_features(slope_f_per_tf, n):
    """
    17. cross_tf_slope_agreement = mean(sign(slope_f) for top 8 TFs)
    18. cross_tf_weighted_slope  = weighted_mean(slope_f, tf_weights)
    """
    features = {}

    # 17. cross_tf_slope_agreement
    sign_sum = np.zeros(n, dtype=np.float64)
    count = 0
    for tf in TOP_8_TFS:
        if tf in slope_f_per_tf:
            sign_sum += np.sign(slope_f_per_tf[tf])
            count += 1
    if count > 0:
        features['cross_tf_slope_agreement'] = (sign_sum / count).astype(np.float32)
    else:
        features['cross_tf_slope_agreement'] = np.zeros(n, dtype=np.float32)

    # 18. cross_tf_weighted_slope
    weighted_sum = np.zeros(n, dtype=np.float64)
    weight_total = 0.0
    for tf in TIMEFRAME_ORDER:
        if tf in slope_f_per_tf:
            w = TF_WEIGHTS.get(tf, 1.0)
            weighted_sum += slope_f_per_tf[tf] * w
            weight_total += w
    if weight_total > 0:
        features['cross_tf_weighted_slope'] = (weighted_sum / weight_total).astype(np.float32)
    else:
        features['cross_tf_weighted_slope'] = np.zeros(n, dtype=np.float32)

    return features


# =============================================================================
# Phase E: Post-alignment features (from already-aligned V5 columns)
# =============================================================================

def build_range_position_signed(result_df):
    """
    15. range_position_signed = (2 * stoch_pos_w20 - 1) * sign(slope_f_w120)
    Computed from already-aligned V5 KEEP columns.
    """
    features = {}
    for tf in TIMEFRAME_ORDER:
        stoch_col = f'stoch_pos_{tf}_w20'
        slope_col = f'slope_f_mag_{tf}'
        if stoch_col in result_df.columns and slope_col in result_df.columns:
            stoch = result_df[stoch_col].values.astype(np.float64)
            slope_f = result_df[slope_col].values.astype(np.float64)
            rps = (2 * stoch - 1) * np.sign(slope_f)
            features[f'range_position_signed_{tf}'] = rps.astype(np.float32)
        else:
            logger.warning(f"  Missing {stoch_col} or {slope_col}, filling with 0")
            features[f'range_position_signed_{tf}'] = np.zeros(
                len(result_df), dtype=np.float32)
    return features


# =============================================================================
# Verification
# =============================================================================

def _verify_features(df: pd.DataFrame, n_keep: int, n_new: int):
    """Spot-check V6 feature invariants."""
    logger.info("Running verification checks...")
    errors = 0

    actual_features = len(df.columns) - 1  # minus 'time'
    expected_total = n_keep + n_new
    if actual_features != expected_total:
        logger.error(f"  FAIL: Expected {expected_total} features, got {actual_features}")
        errors += 1
    else:
        logger.info(f"  OK: {actual_features} features ({n_keep} KEEP + {n_new} new)")

    # Check all numeric
    for col in df.columns:
        if col == 'time':
            continue
        if df[col].dtype == object:
            logger.error(f"  FAIL: Column {col} is object type")
            errors += 1

    # Check no NaN
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.error(f"  FAIL: NaN in columns: {nan_cols[:10]}...")
        errors += 1
    else:
        logger.info("  OK: No NaN values")

    # Spot-checks on new features
    logger.info("\n  New feature spot-checks:")

    # trend_certainty_3D: range [0, ~15], mean ~3-6
    col = 'trend_certainty_3D'
    if col in df.columns:
        vals = df[col]
        logger.info(f"    {col}: min={vals.min():.2f}, max={vals.max():.2f}, "
                     f"mean={vals.mean():.2f}, std={vals.std():.2f}")

    # vol_ratio_1D: range [0.1, 10+], mean ~1.0
    col = 'vol_ratio_1D'
    if col in df.columns:
        vals = df[col]
        logger.info(f"    {col}: min={vals.min():.2f}, max={vals.max():.2f}, "
                     f"mean={vals.mean():.2f}")

    # regime_quadrant_1D: values in {-3, -1, +1, +3}
    col = 'regime_quadrant_1D'
    if col in df.columns:
        uniq = sorted(df[col].unique())
        logger.info(f"    {col}: unique values = {uniq}")

    # angle_regime_3D: both positive and negative
    col = 'angle_regime_3D'
    if col in df.columns:
        n_pos = (df[col] > 0).sum()
        n_neg = (df[col] < 0).sum()
        logger.info(f"    {col}: {n_pos} positive, {n_neg} negative, "
                     f"min={df[col].min():.2f}, max={df[col].max():.2f}")

    # slope_sign_gradient: values in {-2, 0, +2}
    col = 'slope_sign_gradient_1D'
    if col in df.columns:
        uniq = sorted(df[col].unique())
        logger.info(f"    {col}: unique values = {uniq}")

    # cross_tf_slope_agreement: range [-1, +1]
    col = 'cross_tf_slope_agreement'
    if col in df.columns:
        vals = df[col]
        logger.info(f"    {col}: min={vals.min():.3f}, max={vals.max():.3f}, "
                     f"mean={vals.mean():.3f}")

    # atr_normalized_5M
    col = 'atr_normalized_5M'
    if col in df.columns:
        vals = df[col][df[col] > 0]
        if len(vals) > 0:
            logger.info(f"    {col}: min={vals.min():.6f}, max={vals.max():.6f}, "
                         f"mean={vals.mean():.6f}")

    # range_position_signed_1D: range [-1, +1]
    col = 'range_position_signed_1D'
    if col in df.columns:
        vals = df[col]
        n_pos = (vals > 0).sum()
        n_neg = (vals < 0).sum()
        logger.info(f"    {col}: {n_pos} positive, {n_neg} negative")

    # wick_asymmetry_1H
    col = 'wick_asymmetry_1H'
    if col in df.columns:
        vals = df[col]
        logger.info(f"    {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
                     f"mean={vals.mean():.4f}")

    # corr_stability_3D
    col = 'corr_stability_3D'
    if col in df.columns:
        vals = df[col]
        logger.info(f"    {col}: min={vals.min():.6f}, max={vals.max():.6f}, "
                     f"mean={vals.mean():.6f}")

    if errors == 0:
        logger.info(f"\n  All verification checks PASSED")
    else:
        logger.warning(f"\n  {errors} verification check(s) FAILED")
    return errors


# =============================================================================
# Main Pipeline
# =============================================================================

def run_encoding(force: bool = False):
    """Run V6 encoding: V5 KEEP + 20 new directional concepts."""
    logger.info("=" * 70)
    logger.info("V6 FEATURE ENCODER — 20 New Directional Concepts")
    logger.info(f"  V5 KEEP: ~203 proven features")
    logger.info(f"  NEW: 20 concepts → ~192 features (10 decomposed + 6 kline "
                f"+ ATR×3 + 2 cross-TF + 11 range_position + 11 p_value_trend "
                f"+ 11 directional_vol_body per TF)")
    logger.info("=" * 70)

    v6_path = ENCODED_DIR / "feature_matrix_v6.parquet"
    if v6_path.exists() and not force:
        logger.info(f"feature_matrix_v6.parquet already exists. Use --force to rebuild.")
        return

    # ---- Phase A: Load V5 features, filter to KEEP ----
    logger.info("\n[PHASE A] Loading V5 feature matrix and filtering to KEEP list...")
    v5_path = ENCODED_DIR / "feature_matrix_v5.parquet"
    if not v5_path.exists():
        raise FileNotFoundError(
            f"Missing: {v5_path}\nRun: python model_training/encode_v5.py first")

    v5_df = pd.read_parquet(v5_path)
    logger.info(f"  V5 shape: {v5_df.shape}")

    keep_list = _build_keep_list()
    # Filter to only columns that exist (defensive)
    keep_existing = [c for c in keep_list if c in v5_df.columns]
    missing_keep = [c for c in keep_list if c not in v5_df.columns]
    if missing_keep:
        logger.warning(f"  {len(missing_keep)} KEEP columns missing from V5: {missing_keep[:5]}...")

    result = v5_df[['time'] + keep_existing].copy()
    n_keep = len(keep_existing)
    n = len(result)
    base_times = pd.to_datetime(result['time'])
    logger.info(f"  KEEP: {n_keep} features, {n} rows")

    t0 = time.time()

    # ---- Phase B: Decomposed-based features ----
    logger.info("\n[PHASE B] Computing decomposed-based features (10 per TF)...")
    decomp_features, slope_f_per_tf = build_decomposed_features(base_times, n)
    logger.info(f"  Decomposed features: {len(decomp_features)}")

    # ---- Phase C: Kline-based features ----
    logger.info("\n[PHASE C] Computing kline-based features (6 per TF + ATR)...")
    kline_features = build_kline_features(base_times, n)
    logger.info(f"  Kline features: {len(kline_features)}")

    # ---- Phase D: Cross-TF summaries ----
    logger.info("\n[PHASE D] Computing cross-TF summaries...")
    cross_tf_features = build_cross_tf_features(slope_f_per_tf, n)
    logger.info(f"  Cross-TF features: {len(cross_tf_features)}")

    # Combine all new feature dicts, then concat once (avoids fragmentation)
    all_new = {}
    all_new.update(decomp_features)
    all_new.update(kline_features)
    all_new.update(cross_tf_features)

    new_df = pd.DataFrame(all_new, index=result.index)
    result = pd.concat([result, new_df], axis=1)

    # ---- Phase E: Post-alignment features ----
    logger.info("\n[PHASE E] Computing range_position_signed (from aligned V5 columns)...")
    rps_features = build_range_position_signed(result)
    rps_df = pd.DataFrame(rps_features, index=result.index)
    result = pd.concat([result, rps_df], axis=1)
    logger.info(f"  Range position features: {len(rps_features)}")

    n_new = len(decomp_features) + len(kline_features) + len(cross_tf_features) + len(rps_features)
    elapsed = time.time() - t0

    logger.info(f"\n  All features computed in {elapsed:.1f}s")
    logger.info(f"  Final shape: {result.shape}")

    # ---- Verification ----
    _verify_features(result, n_keep, n_new)

    # ---- Save ----
    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving parquet...")
    result.to_parquet(v6_path, engine='pyarrow', index=False)

    total_features = len(result.columns) - 1
    logger.info(f"\n{'=' * 70}")
    logger.info(f"V6 ENCODING COMPLETE")
    logger.info(f"  Rows: {len(result)}")
    logger.info(f"  Features: {total_features} ({n_keep} KEEP + {n_new} new)")
    logger.info(f"  Parquet: {v6_path}")
    logger.info(f"  Size: {v6_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="V6 encoder: V5 KEEP + 20 directional concepts")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    run_encoding(force=args.force)
