"""
V10 Feature Encoder — V6 (395) + Cross-Scale Convergence (~113) + Bollinger Band (10) = ~518 total.

Keeps ALL 395 V6 features intact (Phase A-E). Adds Phase F with cross-scale
convergence features that encode the expert's primary signal: cross-window angle
crossings across multiple timeframes.

Key insight: Discrete STATE counts (crossing_active=3, converging_count=5)
don't suffer from staircase because step-functions ARE the correct representation
for counts. When you COUNT how many of 55 oscillators are crossing simultaneously,
that count changes every few 5M bars.

Phase F — Cross-Scale Convergence:
  F1. Per-TF Cross-Window States (7 x 11 TFs = 77):
      1. xw_crosses_active   = count of 7 pairs with sign change in angle_diff
      2. xw_crosses_long     = count with LONG direction (younger falling + elder rising)
      3. xw_crosses_short    = count with SHORT direction (younger rising + elder falling)
      4. xw_converging       = count of pairs where |angle_diff| is decreasing
      5. xw_reversal_count   = count of 5 windows with active 5-point reversal
      6. xw_direction_agreement = fraction of 5 windows with same sign(slope_f)
      7. xw_cross_reversal   = 1 if any crossing AND any reversal active simultaneously

  F2. Cross-TF Composites (15):
      1-4. xtf_total_crosses/long/short/converging
      5. xtf_tfs_with_crosses
      6-9. xtf_young/adult/balzak/gran_crosses
      10. xtf_cascade_score
      11-12. xtf_direction_net/agreement
      13-14. xtf_reversal_total/confirmed
      15. xtf_convergence_momentum

  F3. Correlation Dynamics (12):
      1. corr_velocity_{tf}        x11  (diff of corr on native TF)
      2. xtf_corr_agreement        x1   (mean sign(corr) across TFs)

  F4. Temporal (5):
      hour_sin, hour_cos, dow_sin, dow_cos, is_ny_session

  F5. Volume x Convergence Interactions (4):
      convergence_volume, crossing_atr, cascade_volume, reversal_conviction

Output: encoded_data/feature_matrix_v10.parquet

Usage:
    python model_training/encode_v10.py
    python model_training/encode_v10.py --force
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

# All 5 window sizes from decomposed CSVs
ALL_WINDOWS = [30, 60, 100, 120, 160]

# 7 crossing pairs from signal_logic.py
CROSSING_PAIRS = [
    (30, 60),    # df - df1 (adjacent)
    (30, 100),   # df - df2 (skip one)
    (60, 100),   # df1 - df2 (adjacent)
    (60, 120),   # df1 - df3 (skip one)
    (100, 120),  # df2 - df3 (adjacent)
    (100, 160),  # df2 - df4 (skip one)
    (120, 160),  # df3 - df4 (adjacent)
]

# Timeframe weights from signal_logic.py
TF_WEIGHTS = {
    "3D": 5, "1D": 4, "12H": 3.5, "8H": 3, "6H": 2.5,
    "4H": 2, "2H": 1.5, "1H": 1.2, "30M": 1.1, "15M": 1.0, "5M": 0.8
}

# Top 8 TFs for slope agreement (exclude Youngs)
TOP_8_TFS = ["3D", "1D", "12H", "8H", "6H", "4H", "2H", "1H"]

# ATR only for these TFs
ATR_TFS = ["5M", "1D", "4H"]

# TF generation groups
TF_YOUNGS = ["5M", "15M", "30M"]
TF_ADULTS = ["1H", "2H", "4H"]
TF_BALZAKS = ["6H", "8H", "12H"]
TF_GRANS = ["1D", "3D"]

# F1 feature names (for fallback zero-fill)
F1_FEATURES = [
    'xw_crosses_active', 'xw_crosses_long', 'xw_crosses_short',
    'xw_converging', 'xw_reversal_count', 'xw_direction_agreement',
    'xw_cross_reversal',
]

logger = logging.getLogger(__name__)


# =============================================================================
# KEEP List — features retained from V5 (identical to V6)
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
# Phase B: Decomposed-based features (regression outputs) — UNCHANGED from V6
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

        # 5. regime_quadrant = sign(slope_f)*2 + sign(slope_b) -> {-3,-1,+1,+3}
        regime_quadrant = np.sign(sf_slow) * 2 + np.sign(sb_slow)

        # 6. corr_regime = corr * sign(slope_f)
        corr_regime = corr_vals * sign_sf

        # 7. corr_stability = rolling_std(corr, 10) on native TF
        corr_stability = pd.Series(corr_vals).rolling(
            10, min_periods=1).std().fillna(0).values

        # 8. slope_sign_gradient = sign(slope_f_w30) - sign(slope_f_w120) -> {-2, 0, +2}
        slope_sign_gradient = np.sign(sf_fast) - np.sign(sf_slow)

        # 9. angle_regime = min(angle, 30) * sign(slope_f)
        angle_regime = np.minimum(angle_vals, 30.0) * sign_sf

        # 10. regime_change_strength = |slope_f - slope_b| * trend_certainty
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
# Phase C: Kline-based features (price/volume) — UNCHANGED from V6
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
# Phase D: Cross-TF summaries — UNCHANGED from V6
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
# Phase E: Post-alignment features — UNCHANGED from V6
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
# Phase G: Bollinger Band Extreme Features
# =============================================================================

BB_TFS = ["5M", "15M", "1H", "4H", "1D"]
BB_PERIOD = 35
BB_STD = 3


def build_bb_features(base_times, n):
    """Phase G: Bollinger Band (3, 35) extreme features.

    For each TF in [5M, 15M, 1H, 4H, 1D]:
      bb_lower_pierce_{TF} = (BB_lower - Low) / Close
        Positive when candle low pierces below lower band (extreme oversold).
      bb_upper_dist_{TF} = (BB_upper - Close) / Close
        Negative when close is above upper band (extreme overbought).

    BB bands narrow in ranging markets, so even small moves register as
    extreme. These features activate precisely when regression-based
    features go quiet during signal droughts.
    """
    features = {}

    for tf in BB_TFS:
        logger.info(f"  [BB] {tf}...")

        try:
            klines = _load_klines(tf)
        except FileNotFoundError as e:
            logger.warning(f"    Skipping BB {tf}: {e}")
            features[f'bb_lower_pierce_{tf}'] = np.zeros(n, dtype=np.float32)
            features[f'bb_upper_dist_{tf}'] = np.zeros(n, dtype=np.float32)
            continue

        close = klines['Close'].values.astype(np.float64)
        low = klines['Low'].values.astype(np.float64)

        close_series = pd.Series(close)
        sma = close_series.rolling(BB_PERIOD, min_periods=1).mean().values
        std = close_series.rolling(BB_PERIOD, min_periods=1).std().fillna(0).values

        bb_lower = sma - BB_STD * std
        bb_upper = sma + BB_STD * std

        safe_close = np.where(close > 0, close, 1.0)
        lower_pierce = (bb_lower - low) / safe_close      # >0 when low pierces below band
        upper_dist = (bb_upper - close) / safe_close       # <0 when close is above band

        native_df = pd.DataFrame({
            'time': klines['time'],
            'bb_lower_pierce': np.nan_to_num(lower_pierce, nan=0.0).astype(np.float64),
            'bb_upper_dist': np.nan_to_num(upper_dist, nan=0.0).astype(np.float64),
        })

        base_df = pd.DataFrame({'time': base_times})
        aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

        features[f'bb_lower_pierce_{tf}'] = aligned['bb_lower_pierce'].fillna(0).values.astype(np.float32)
        features[f'bb_upper_dist_{tf}'] = aligned['bb_upper_dist'].fillna(0).values.astype(np.float32)

    return features


# =============================================================================
# Phase F: Cross-Scale Convergence Features (NEW in V10)
# =============================================================================

def _load_all_windows(tf):
    """Load all 5 decomposed CSVs for a TF, inner-join on time.

    Returns (merged_df, available_windows) or (None, []) if insufficient data.
    """
    dfs = {}
    for ws in ALL_WINDOWS:
        path = DECOMPOSED_DIR / f"decomposed_{tf}_w{ws}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            dfs[ws] = df[['time', 'angle', 'slope_f']].rename(
                columns={'angle': f'angle_w{ws}', 'slope_f': f'sf_w{ws}'})

    if not dfs:
        return None, []

    # Inner join all available windows on time
    merged = None
    for ws in sorted(dfs.keys()):
        if merged is None:
            merged = dfs[ws]
        else:
            merged = merged.merge(dfs[ws], on='time', how='inner')

    merged = merged.sort_values('time').reset_index(drop=True)
    return merged, sorted(dfs.keys())


def _detect_crossings(merged, available_windows):
    """Detect cross-window crossings for 7 pairs.

    Uses signal_logic.py's exact logic:
    - Crossing: prev_diff * curr_diff < 0 (sign change in angle difference)
    - LONG: younger derivative < 0 AND elder derivative > 0
    - SHORT: younger derivative > 0 AND elder derivative < 0

    Returns (crosses_active, crosses_long, crosses_short, converging) arrays.
    """
    n = len(merged)
    crosses_active = np.zeros(n, dtype=np.int32)
    crosses_long = np.zeros(n, dtype=np.int32)
    crosses_short = np.zeros(n, dtype=np.int32)
    converging = np.zeros(n, dtype=np.int32)

    for ws1, ws2 in CROSSING_PAIRS:
        if ws1 not in available_windows or ws2 not in available_windows:
            continue

        a1 = merged[f'angle_w{ws1}'].values  # younger (smaller window)
        a2 = merged[f'angle_w{ws2}'].values  # elder (larger window)
        diff = a1 - a2

        for i in range(1, n):
            # Crossing: sign change in angle difference
            if diff[i - 1] * diff[i] < 0:
                crosses_active[i] += 1
                # Direction from derivatives (exact signal_logic.py match)
                d1 = a1[i] - a1[i - 1]  # younger derivative
                d2 = a2[i] - a2[i - 1]  # elder derivative
                if d1 < 0 and d2 > 0:
                    crosses_long[i] += 1   # younger falling + elder rising = LONG
                elif d1 > 0 and d2 < 0:
                    crosses_short[i] += 1  # younger rising + elder falling = SHORT

            # Converging: |diff| decreasing (approaching crossing)
            if abs(diff[i]) < abs(diff[i - 1]):
                converging[i] += 1

    return crosses_active, crosses_long, crosses_short, converging


def _detect_reversals_per_window(angle_values, window=5):
    """5-point reversal pattern on angle series.

    Exact match to signal_logic.py detect_cycle_events:
    - BOTTOM: a>b>c>d<e (4 decreasing then upturn)
    - PEAK:   a<b<c<d>e (4 increasing then downturn)

    Returns array of 0/1 reversal indicators.
    """
    n = len(angle_values)
    reversals = np.zeros(n, dtype=np.int32)

    for i in range(window - 1, n):
        w = angle_values[i - window + 1: i + 1]
        # BOTTOM: a>b>c>d<e
        if w[0] > w[1] > w[2] > w[3] and w[3] < w[4]:
            reversals[i] = 1
        # PEAK: a<b<c<d>e
        elif w[0] < w[1] < w[2] < w[3] and w[3] > w[4]:
            reversals[i] = 1

    return reversals


def build_cross_scale_features(base_times, n):
    """Phase F: Cross-Scale Convergence Features.

    F1: Per-TF cross-window states (7 features x 11 TFs = 77)
    F2: Cross-TF composites (15)
    F3: Correlation dynamics (12)
    F4: Temporal (5)

    Returns features dict. F5 (interactions) computed separately after Phase C.
    """
    features = {}
    xw_per_tf = {}  # store per-TF aligned arrays for cross-TF composites

    # ---- F1: Per-TF Cross-Window States ----
    logger.info("  [F1] Per-TF cross-window states...")

    for tf in TIMEFRAME_ORDER:
        logger.info(f"    {tf}...")

        merged, avail_ws = _load_all_windows(tf)

        if merged is None or len(merged) < 10:
            logger.warning(f"    Skipping {tf}: insufficient data")
            for feat in F1_FEATURES:
                features[f'{feat}_{tf}'] = np.zeros(n, dtype=np.float32)
            xw_per_tf[tf] = {feat: np.zeros(n, dtype=np.float32) for feat in F1_FEATURES}
            continue

        # Cross-window crossings
        ca, cl, cs, conv = _detect_crossings(merged, avail_ws)

        # Reversals: count across all available windows
        rev_count = np.zeros(len(merged), dtype=np.int32)
        for ws in avail_ws:
            rev_count += _detect_reversals_per_window(merged[f'angle_w{ws}'].values)

        # Direction agreement: fraction of windows with same sign(slope_f)
        sf_cols = [f'sf_w{ws}' for ws in avail_ws]
        signs = np.column_stack([np.sign(merged[col].values) for col in sf_cols])
        # Majority sign
        sign_sum = np.sum(signs, axis=1)
        mode_sign = np.sign(sign_sum)
        # Handle ties (sign_sum==0): default to 0.5 agreement
        n_windows = len(avail_ws)
        agreement = np.zeros(len(merged), dtype=np.float64)
        for i in range(len(merged)):
            if mode_sign[i] == 0:
                agreement[i] = 0.5
            else:
                agreement[i] = np.mean(signs[i] == mode_sign[i])

        # Cross-reversal overlap: 1 if any crossing AND any reversal active simultaneously
        cross_rev = ((ca > 0) & (rev_count > 0)).astype(np.int32)

        # Build native DataFrame for merge_asof
        native_df = pd.DataFrame({
            'time': merged['time'],
            'xw_crosses_active': ca.astype(np.float32),
            'xw_crosses_long': cl.astype(np.float32),
            'xw_crosses_short': cs.astype(np.float32),
            'xw_converging': conv.astype(np.float32),
            'xw_reversal_count': rev_count.astype(np.float32),
            'xw_direction_agreement': agreement.astype(np.float32),
            'xw_cross_reversal': cross_rev.astype(np.float32),
        })

        # merge_asof to 5M base (backward fill, no future leakage)
        base_df = pd.DataFrame({'time': base_times})
        aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

        tf_features = {}
        for feat in F1_FEATURES:
            arr = aligned[feat].fillna(0).values.astype(np.float32)
            features[f'{feat}_{tf}'] = arr
            tf_features[feat] = arr

        xw_per_tf[tf] = tf_features

        # Log spot-check
        ca_aligned = features[f'xw_crosses_active_{tf}']
        nonzero = (ca_aligned > 0).sum()
        logger.info(f"      crosses_active: {nonzero}/{n} nonzero, "
                    f"max={ca_aligned.max():.0f}")

    # ---- F2: Cross-TF Composites ----
    logger.info("  [F2] Cross-TF composites...")

    # 1-4: Sum across all TFs
    for feat_base, feat_name in [
        ('xw_crosses_active', 'xtf_total_crosses'),
        ('xw_crosses_long', 'xtf_total_long'),
        ('xw_crosses_short', 'xtf_total_short'),
        ('xw_converging', 'xtf_total_converging'),
    ]:
        total = np.zeros(n, dtype=np.float32)
        for tf in TIMEFRAME_ORDER:
            total += xw_per_tf[tf][feat_base]
        features[feat_name] = total

    # 5: Count of TFs with at least 1 crossing
    tfs_with_crosses = np.zeros(n, dtype=np.float32)
    for tf in TIMEFRAME_ORDER:
        tfs_with_crosses += (xw_per_tf[tf]['xw_crosses_active'] >= 1).astype(np.float32)
    features['xtf_tfs_with_crosses'] = tfs_with_crosses

    # 6-9: Group crosses by TF generation
    for group_name, tf_list in [
        ('xtf_young_crosses', TF_YOUNGS),
        ('xtf_adult_crosses', TF_ADULTS),
        ('xtf_balzak_crosses', TF_BALZAKS),
        ('xtf_gran_crosses', TF_GRANS),
    ]:
        total = np.zeros(n, dtype=np.float32)
        for tf in tf_list:
            total += xw_per_tf[tf]['xw_crosses_active']
        features[group_name] = total

    # 10: Cascade score = young*1 + adult*2 + balzak*3 + gran*4
    features['xtf_cascade_score'] = (
        features['xtf_young_crosses'] * 1 +
        features['xtf_adult_crosses'] * 2 +
        features['xtf_balzak_crosses'] * 3 +
        features['xtf_gran_crosses'] * 4
    ).astype(np.float32)

    # 11: Direction net = total_long - total_short
    features['xtf_direction_net'] = (
        features['xtf_total_long'] - features['xtf_total_short']
    ).astype(np.float32)

    # 12: Direction agreement = |total_long - total_short| / (total_long + total_short + eps)
    total_dir = features['xtf_total_long'] + features['xtf_total_short'] + 1e-10
    features['xtf_direction_agreement'] = (
        np.abs(features['xtf_direction_net']) / total_dir
    ).astype(np.float32)

    # 13: Reversal total = sum(xw_reversal_count) across TFs
    rev_total = np.zeros(n, dtype=np.float32)
    for tf in TIMEFRAME_ORDER:
        rev_total += xw_per_tf[tf]['xw_reversal_count']
    features['xtf_reversal_total'] = rev_total

    # 14: Reversal confirmed = count of TFs where xw_cross_reversal == 1
    rev_confirmed = np.zeros(n, dtype=np.float32)
    for tf in TIMEFRAME_ORDER:
        rev_confirmed += (xw_per_tf[tf]['xw_cross_reversal'] >= 1).astype(np.float32)
    features['xtf_reversal_confirmed'] = rev_confirmed

    # 15: Convergence momentum = xtf_total_converging - lag1(xtf_total_converging)
    conv_total = features['xtf_total_converging']
    conv_lag1 = np.zeros(n, dtype=np.float32)
    conv_lag1[1:] = conv_total[:-1]
    features['xtf_convergence_momentum'] = (conv_total - conv_lag1).astype(np.float32)

    logger.info(f"    xtf_total_crosses: max={features['xtf_total_crosses'].max():.0f}, "
                f"mean={features['xtf_total_crosses'].mean():.2f}")
    logger.info(f"    xtf_cascade_score: max={features['xtf_cascade_score'].max():.0f}, "
                f"mean={features['xtf_cascade_score'].mean():.2f}")

    # ---- F3: Correlation Dynamics ----
    logger.info("  [F3] Correlation dynamics...")

    corr_signs = []
    for tf in TIMEFRAME_ORDER:
        logger.info(f"    corr_velocity {tf}...")
        try:
            df_slow = _load_decomposed(tf, W_SLOW)
            corr_vals = df_slow[['time', 'corr']].copy()
            corr_vals = corr_vals.sort_values('time').reset_index(drop=True)

            # diff on native TF data
            corr_arr = corr_vals['corr'].fillna(0).values.astype(np.float64)
            corr_vel = np.zeros_like(corr_arr)
            corr_vel[1:] = corr_arr[1:] - corr_arr[:-1]

            native_df = pd.DataFrame({
                'time': corr_vals['time'],
                'corr_velocity': corr_vel.astype(np.float32),
                'corr_sign': np.sign(corr_arr).astype(np.float32),
            })

            base_df = pd.DataFrame({'time': base_times})
            aligned = pd.merge_asof(base_df, native_df, on='time', direction='backward')

            features[f'corr_velocity_{tf}'] = aligned['corr_velocity'].fillna(0).values.astype(np.float32)
            corr_signs.append(aligned['corr_sign'].fillna(0).values.astype(np.float64))

        except FileNotFoundError:
            features[f'corr_velocity_{tf}'] = np.zeros(n, dtype=np.float32)
            corr_signs.append(np.zeros(n, dtype=np.float64))

    # xtf_corr_agreement = mean(sign(corr_w120)) across 11 TFs
    if corr_signs:
        corr_sign_stack = np.column_stack(corr_signs)
        features['xtf_corr_agreement'] = np.mean(corr_sign_stack, axis=1).astype(np.float32)
    else:
        features['xtf_corr_agreement'] = np.zeros(n, dtype=np.float32)

    # ---- F4: Temporal Features ----
    logger.info("  [F4] Temporal features...")
    times_dt = pd.to_datetime(base_times)
    hour = times_dt.dt.hour + times_dt.dt.minute / 60.0
    dow = times_dt.dt.dayofweek.astype(np.float64)

    features['hour_sin'] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    features['is_ny_session'] = ((times_dt.dt.hour >= 13) & (times_dt.dt.hour <= 21)).astype(np.float32)

    return features


def build_interaction_features(result_df, n):
    """Phase F5: Volume x Convergence Interactions.

    Must be called AFTER Phase C (needs vol_ratio_1D, atr_normalized_1D, vol_body_product_1D)
    and Phase F (needs xtf_total_crosses, xtf_tfs_with_crosses, xtf_cascade_score,
    xtf_reversal_confirmed).
    """
    features = {}

    # Helper to safely get column or zeros
    def _get(col):
        if col in result_df.columns:
            return result_df[col].values.astype(np.float64)
        return np.zeros(n, dtype=np.float64)

    # 1. convergence_volume = xtf_total_crosses x vol_ratio_1D
    features['convergence_volume'] = (
        _get('xtf_total_crosses') * _get('vol_ratio_1D')
    ).astype(np.float32)

    # 2. crossing_atr = xtf_tfs_with_crosses x atr_normalized_1D
    features['crossing_atr'] = (
        _get('xtf_tfs_with_crosses') * _get('atr_normalized_1D')
    ).astype(np.float32)

    # 3. cascade_volume = xtf_cascade_score x vol_body_product_1D
    features['cascade_volume'] = (
        _get('xtf_cascade_score') * _get('vol_body_product_1D')
    ).astype(np.float32)

    # 4. reversal_conviction = xtf_reversal_confirmed x vol_ratio_1D
    features['reversal_conviction'] = (
        _get('xtf_reversal_confirmed') * _get('vol_ratio_1D')
    ).astype(np.float32)

    return features


# =============================================================================
# Verification
# =============================================================================

def _verify_features(df: pd.DataFrame, n_keep: int, n_new: int):
    """Spot-check V10 feature invariants."""
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

    # ---- V6 feature spot-checks ----
    logger.info("\n  V6 feature spot-checks:")
    for col in ['trend_certainty_3D', 'vol_ratio_1D', 'regime_quadrant_1D',
                'angle_regime_3D', 'cross_tf_slope_agreement', 'atr_normalized_5M']:
        if col in df.columns:
            vals = df[col]
            logger.info(f"    {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
                         f"mean={vals.mean():.4f}")

    # ---- V10 cross-scale spot-checks ----
    logger.info("\n  V10 cross-scale spot-checks:")

    # xw_crosses_active_1D: should be 0-7, mostly 0-2
    col = 'xw_crosses_active_1D'
    if col in df.columns:
        vals = df[col]
        uniq = sorted(vals.unique())
        nonzero = (vals > 0).sum()
        logger.info(f"    {col}: unique={uniq[:8]}, nonzero={nonzero}/{len(vals)} "
                     f"({nonzero/len(vals)*100:.2f}%)")

    # xtf_total_crosses: should change frequently
    col = 'xtf_total_crosses'
    if col in df.columns:
        vals = df[col]
        changes = (vals.diff() != 0).sum()
        logger.info(f"    {col}: max={vals.max():.0f}, mean={vals.mean():.2f}, "
                     f"changes={changes}/{len(vals)} ({changes/len(vals)*100:.1f}%)")

    # xtf_cascade_score
    col = 'xtf_cascade_score'
    if col in df.columns:
        vals = df[col]
        logger.info(f"    {col}: max={vals.max():.0f}, mean={vals.mean():.2f}, "
                     f"std={vals.std():.2f}")

    # xw_reversal_count_1D: should be 0-5, mostly 0
    col = 'xw_reversal_count_1D'
    if col in df.columns:
        vals = df[col]
        nonzero = (vals > 0).sum()
        logger.info(f"    {col}: max={vals.max():.0f}, nonzero={nonzero}/{len(vals)}")

    # xtf_reversal_confirmed
    col = 'xtf_reversal_confirmed'
    if col in df.columns:
        vals = df[col]
        nonzero = (vals > 0).sum()
        logger.info(f"    {col}: max={vals.max():.0f}, nonzero={nonzero}/{len(vals)}")

    # Temporal features
    for col in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_ny_session']:
        if col in df.columns:
            vals = df[col]
            logger.info(f"    {col}: min={vals.min():.3f}, max={vals.max():.3f}, "
                         f"mean={vals.mean():.3f}")

    # Interaction features
    for col in ['convergence_volume', 'crossing_atr', 'cascade_volume', 'reversal_conviction']:
        if col in df.columns:
            vals = df[col]
            nonzero = (vals != 0).sum()
            logger.info(f"    {col}: max={vals.max():.3f}, mean={vals.mean():.4f}, "
                         f"nonzero={nonzero}/{len(vals)}")

    # Staircase check: xtf_total_crosses should NOT be constant for 288 rows
    col = 'xtf_total_crosses'
    if col in df.columns:
        vals = df[col].values
        # Check a sample of 288-row blocks (one day of 5M data)
        block_size = 288
        n_blocks = min(len(vals) // block_size, 10)
        constant_blocks = 0
        for b in range(n_blocks):
            start = b * block_size
            block = vals[start:start + block_size]
            if np.all(block == block[0]):
                constant_blocks += 1
        if constant_blocks > n_blocks * 0.5:
            logger.warning(f"    WARN: xtf_total_crosses has {constant_blocks}/{n_blocks} "
                           f"constant 288-row blocks (possible staircase)")
        else:
            logger.info(f"    OK: xtf_total_crosses is NOT staircased "
                        f"({constant_blocks}/{n_blocks} constant blocks)")

    # Corr velocity
    col = 'corr_velocity_1D'
    if col in df.columns:
        vals = df[col]
        nonzero = (vals != 0).sum()
        logger.info(f"    {col}: min={vals.min():.6f}, max={vals.max():.6f}, "
                     f"nonzero={nonzero}/{len(vals)}")

    if errors == 0:
        logger.info(f"\n  All verification checks PASSED")
    else:
        logger.warning(f"\n  {errors} verification check(s) FAILED")
    return errors


# =============================================================================
# Main Pipeline
# =============================================================================

def run_encoding(force: bool = False):
    """Run V10 encoding: V6 (395) + Cross-Scale Convergence (~113) + BB (10) = ~518."""
    logger.info("=" * 70)
    logger.info("V10 FEATURE ENCODER — Cross-Scale Convergence")
    logger.info(f"  V6 BASE: ~395 features (Phase A-E, unchanged)")
    logger.info(f"  NEW Phase F: ~113 features")
    logger.info(f"    F1: Per-TF cross-window states (7 x 11 = 77)")
    logger.info(f"    F2: Cross-TF composites (15)")
    logger.info(f"    F3: Correlation dynamics (12)")
    logger.info(f"    F4: Temporal (5)")
    logger.info(f"    F5: Volume x convergence interactions (4)")
    logger.info("=" * 70)

    v10_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    if v10_path.exists() and not force:
        logger.info(f"feature_matrix_v10.parquet already exists. Use --force to rebuild.")
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

    # Combine Phase B-D features
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

    n_v6_new = len(decomp_features) + len(kline_features) + len(cross_tf_features) + len(rps_features)
    logger.info(f"\n  V6 features computed: {n_v6_new}")

    # ---- Phase F: Cross-Scale Convergence Features (NEW) ----
    logger.info("\n[PHASE F] Computing cross-scale convergence features...")
    cross_scale_features = build_cross_scale_features(base_times, n)
    logger.info(f"  Cross-scale features (F1-F4): {len(cross_scale_features)}")

    # Add F1-F4 features to result
    cs_df = pd.DataFrame(cross_scale_features, index=result.index)
    result = pd.concat([result, cs_df], axis=1)

    # ---- Phase F5: Interaction features (needs Phase C + Phase F columns) ----
    logger.info("\n[PHASE F5] Computing volume x convergence interactions...")
    interaction_features = build_interaction_features(result, n)
    logger.info(f"  Interaction features: {len(interaction_features)}")

    int_df = pd.DataFrame(interaction_features, index=result.index)
    result = pd.concat([result, int_df], axis=1)

    # ---- Phase G: Bollinger Band Extreme Features ----
    logger.info("\n[PHASE G] Computing Bollinger Band extreme features...")
    bb_features = build_bb_features(base_times, n)
    logger.info(f"  BB features: {len(bb_features)}")

    bb_df = pd.DataFrame(bb_features, index=result.index)
    result = pd.concat([result, bb_df], axis=1)

    n_new = n_v6_new + len(cross_scale_features) + len(interaction_features) + len(bb_features)
    elapsed = time.time() - t0

    logger.info(f"\n  All features computed in {elapsed:.1f}s")
    logger.info(f"  Final shape: {result.shape}")

    # ---- Verification ----
    _verify_features(result, n_keep, n_new)

    # ---- Save ----
    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving parquet...")
    result.to_parquet(v10_path, engine='pyarrow', index=False)

    total_features = len(result.columns) - 1
    logger.info(f"\n{'=' * 70}")
    logger.info(f"V10 ENCODING COMPLETE")
    logger.info(f"  Rows: {len(result)}")
    logger.info(f"  Features: {total_features} ({n_keep} KEEP + {n_new} new)")
    logger.info(f"  V6 base features: {n_v6_new}")
    logger.info(f"  V10 new features: {len(cross_scale_features) + len(interaction_features) + len(bb_features)}")
    logger.info(f"    F1 per-TF cross-window: {sum(1 for k in cross_scale_features if k.startswith('xw_'))}")
    logger.info(f"    F2 cross-TF composites: {sum(1 for k in cross_scale_features if k.startswith('xtf_'))}")
    logger.info(f"    F3 correlation dynamics: {sum(1 for k in cross_scale_features if 'corr_velocity' in k or 'corr_agreement' in k)}")
    logger.info(f"    F4 temporal: {sum(1 for k in cross_scale_features if k in ('hour_sin','hour_cos','dow_sin','dow_cos','is_ny_session'))}")
    logger.info(f"    F5 interactions: {len(interaction_features)}")
    logger.info(f"    G  BB extremes: {len(bb_features)}")
    logger.info(f"  Parquet: {v10_path}")
    logger.info(f"  Size: {v10_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="V10 encoder: V6 + Cross-Scale Convergence")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    run_encoding(force=args.force)
