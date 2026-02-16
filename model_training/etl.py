"""
ETL Pipeline: Extract features (X) and labels (Y) for ML training.

Uses 5M as base timeline. Higher TF features are forward-filled
to the 5M grid using merge_asof (latest available value at each 5M point).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from pathlib import Path
from sklearn.mixture import GaussianMixture

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES, DATA_DIR
from core.processor import TimeframeProcessor
from core.signal_logic import SignalLogic, PricePredictor
from data.target_labeling import create_sl_tp_labels, create_atr_labels

# ML data directory (extended historical data)
ML_DATA_DIR = Path(__file__).parent / "data"

logger = logging.getLogger(__name__)

# Timeframe weights (matches signal_logic.py)
TIMEFRAME_WEIGHTS = {
    "3D": 5, "1D": 4, "12H": 3.5, "8H": 3, "6H": 2.5,
    "4H": 2, "2H": 1.5, "1H": 1.2, "30M": 1.1, "15M": 1.0, "5M": 0.8
}

# Crossing pairs (matches signal_logic.py)
CROSSING_PAIRS = [
    (30, 60), (30, 100), (60, 100), (60, 120),
    (100, 120), (100, 160), (120, 160)
]


def load_and_process(trim_months: int = 0) -> Tuple[TimeframeProcessor, SignalLogic]:
    """Load data and run signal analysis. Uses ML data dir if available."""
    processor = TimeframeProcessor()

    if ML_DATA_DIR.exists() and any(ML_DATA_DIR.glob("ml_data_*.csv")):
        logger.info(f"Using ML training data from: {ML_DATA_DIR}")
        _load_ml_data(processor)
    else:
        logger.info(f"Using default data from: {DATA_DIR}")
        processor.load_all_data()

    # Trim to last N months before processing (speeds up iterative_regression)
    if trim_months > 0:
        _trim_data(processor, trim_months)

    processor.process_all()

    logic = SignalLogic(processor=processor)
    logic.run_analysis()

    return processor, logic


def _trim_data(processor: TimeframeProcessor, months: int):
    """Trim raw data to last N months for each timeframe."""
    cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=months)
    for tf in list(processor.raw_data.keys()):
        df = processor.raw_data[tf]
        if 'Open Time' in df.columns:
            mask = df['Open Time'] >= cutoff
            before = len(df)
            processor.raw_data[tf] = df[mask].reset_index(drop=True)
            after = len(processor.raw_data[tf])
            logger.info(f"  Trimmed {tf}: {before} -> {after} rows (last {months} months)")


def _load_ml_data(processor: TimeframeProcessor):
    """Load extended ML training data into the processor."""
    for tf in TIMEFRAME_ORDER:
        ml_path = ML_DATA_DIR / f"ml_data_{tf}.csv"
        if ml_path.exists():
            df = pd.read_csv(ml_path)
            if 'index' in df.columns:
                df = df.drop(columns=['index'])
            df['Open Time'] = pd.to_datetime(df['Open Time'], utc=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna()
            processor.raw_data[tf] = df
            logger.info(f"  Loaded ML data for {tf}: {len(df)} rows")
        else:
            default_path = DATA_DIR / f"testing_data_{tf}.csv"
            if default_path.exists():
                df = pd.read_csv(default_path)
                if 'index' in df.columns:
                    df = df.drop(columns=['index'])
                df['Open Time'] = pd.to_datetime(df['Open Time'], utc=True)
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df.dropna()
                processor.raw_data[tf] = df
                logger.info(f"  Loaded default data for {tf}: {len(df)} rows")

    logger.info(f"Loaded data for {len(processor.raw_data)}/{len(TIMEFRAME_ORDER)} timeframes")


def _get_5m_base_timeline(processor: TimeframeProcessor) -> pd.DataFrame:
    """Get the 5M timeline as the base index."""
    if "5M" not in processor.results or "df" not in processor.results["5M"]:
        raise ValueError("5M data not available")

    df_5m = processor.results["5M"]["df"]
    base = pd.DataFrame({'time': pd.to_datetime(df_5m['time'])})
    base = base.sort_values('time').reset_index(drop=True)
    return base


# ============================================================================
# Feature Extraction Functions (Fixed)
# ============================================================================

def _detect_reversal_bottom(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """Detect bottom reversal: [a>b>c>d<e] pattern -> 1 where detected."""
    events = np.zeros(len(angles))
    for i in range(len(angles) - window + 1):
        w = angles[i:i + window]
        if w[0] > w[1] > w[2] > w[3] and w[3] < w[4]:
            events[i + window - 1] = 1
    return events


def _detect_reversal_peak(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """Detect peak reversal: [a<b<c<d>e] pattern -> 1 where detected."""
    events = np.zeros(len(angles))
    for i in range(len(angles) - window + 1):
        w = angles[i:i + window]
        if w[0] < w[1] < w[2] < w[3] and w[3] > w[4]:
            events[i + window - 1] = 1
    return events


def _detect_crossings_directional(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Detect angle crossover with direction.

    Returns: +1 (LONG: ws1 crosses below ws2 = ws1 was above, now below)
             -1 (SHORT: ws1 crosses above ws2 = ws1 was below, now above)
              0 (no crossing)
    Matches signal_logic.py lines 399-414.
    """
    diff = a1 - a2
    crossings = np.zeros(len(diff))
    for i in range(1, len(diff)):
        if diff[i - 1] * diff[i] < 0:
            if diff[i - 1] < 0 and diff[i] > 0:
                # ws1 was below ws2, now above = SHORT
                crossings[i] = -1
            else:
                # ws1 was above ws2, now below = LONG
                crossings[i] = 1
    return crossings


def _compute_gmm_zones(processor: TimeframeProcessor, tf: str) -> Dict[str, np.ndarray]:
    """
    Compute GMM acceleration zones for all windows of a timeframe.

    Combines acceleration from all 5 windows, fits GMM(5),
    maps each value to zone 0-4 (VERY_CLOSE -> VERY_DISTANT).
    Returns {combo: zone_array} for each window.
    """
    result = {}

    # Collect all acceleration values for this TF
    all_accel = []
    combo_accels = {}
    for i, ws in enumerate(WINDOW_SIZES):
        label = f"df{i}" if i > 0 else "df"
        if label not in processor.results.get(tf, {}):
            continue
        df = processor.results[tf][label]
        if 'acceleration' not in df.columns:
            continue
        accel = df['acceleration'].values
        combo_accels[f"{tf}_{label}"] = accel
        valid = accel[~np.isnan(accel)]
        all_accel.extend(valid.tolist())

    if len(all_accel) < 10:
        for combo, accel in combo_accels.items():
            result[combo] = np.full(len(accel), 2, dtype=np.int8)  # BASELINE
        return result

    # Fit GMM on combined acceleration
    all_arr = np.array(all_accel).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=5, covariance_type='full',
        random_state=42, n_init=3
    )
    gmm.fit(all_arr)

    # Sort cluster means by |mean| distance from zero
    means = gmm.means_.flatten()
    abs_means = np.abs(means)
    sorted_indices = np.argsort(abs_means)
    # Map: cluster_id -> zone (0=VERY_CLOSE, 4=VERY_DISTANT)
    cluster_to_zone = {}
    for rank, cluster_id in enumerate(sorted_indices):
        cluster_to_zone[cluster_id] = rank

    # Classify each combo's acceleration
    for combo, accel in combo_accels.items():
        zones = np.full(len(accel), 2, dtype=np.int8)  # default BASELINE
        valid_mask = ~np.isnan(accel)
        if valid_mask.sum() > 0:
            clusters = gmm.predict(accel[valid_mask].reshape(-1, 1))
            mapped = np.array([cluster_to_zone[c] for c in clusters], dtype=np.int8)
            zones[valid_mask] = mapped
        result[combo] = zones

    return result


def _compute_predictions(slope_f: np.ndarray, angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction features using simple linear extrapolation.

    For slope_f and angle: predict cc = 2*b - a, ccc = 2*cc - b
    Returns (pred_dir, pred_reversal):
      pred_dir: +1 if predicted slope_f > 0, -1 if < 0, 0 otherwise
      pred_reversal: 1 if predicted angle changes sign vs current
    """
    n = len(slope_f)
    pred_dir = np.zeros(n, dtype=np.int8)
    pred_reversal = np.zeros(n, dtype=np.int8)

    for i in range(2, n):
        a_s, b_s = slope_f[i - 1], slope_f[i]
        cc_s = 2 * b_s - a_s

        if cc_s > 0:
            pred_dir[i] = 1
        elif cc_s < 0:
            pred_dir[i] = -1

        a_a, b_a = angle[i - 1], angle[i]
        cc_a = 2 * b_a - a_a
        # Reversal if predicted angle sign differs from current
        if not np.isnan(b_a) and not np.isnan(cc_a):
            if (b_a > 0 and cc_a < 0) or (b_a < 0 and cc_a > 0):
                pred_reversal[i] = 1

    return pred_dir, pred_reversal


# ============================================================================
# Main Feature Builder
# ============================================================================

def build_features(processor: TimeframeProcessor, logic: SignalLogic) -> pd.DataFrame:
    """
    Build feature matrix aligned to 5M timeline.

    For each TF-window combo, extracts:
    - angle, slope_f, acceleration (raw values)
    - R_bottom, R_peak (separate reversal types)
    - A flag (acceleration quality: top 40%)
    - accel_zone (GMM zone 0-4)
    - C flags (directional crossing: +1/-1/0)
    - pred_dir, pred_reversal (prediction features)
    """
    base = _get_5m_base_timeline(processor)
    features = base.copy()

    for tf in TIMEFRAME_ORDER:
        if tf not in processor.results:
            continue

        # Pre-compute GMM zones for this TF
        gmm_zones = _compute_gmm_zones(processor, tf)

        for i, ws in enumerate(WINDOW_SIZES):
            label = f"df{i}" if i > 0 else "df"
            if label not in processor.results[tf]:
                continue

            df = processor.results[tf][label]
            combo = f"{tf}_{label}"

            tmp = pd.DataFrame({'time': pd.to_datetime(df['time'])})

            # Raw values
            for col in ['angle', 'slope_f', 'acceleration']:
                if col in df.columns:
                    tmp[f"{combo}_{col}"] = df[col].values

            # Reversal bottom flag (R_bottom)
            if 'angle' in df.columns and len(df) >= 5:
                tmp[f"{combo}_R_bottom"] = _detect_reversal_bottom(df['angle'].values)
                tmp[f"{combo}_R_peak"] = _detect_reversal_peak(df['angle'].values)

            # Acceleration quality flag (A) - top 40% by absolute value
            if 'acceleration' in df.columns:
                accel = df['acceleration'].dropna().values
                if len(accel) > 0:
                    threshold = np.percentile(np.abs(accel), 60)
                    tmp[f"{combo}_A"] = (np.abs(df['acceleration'].values) >= threshold).astype(int)

            # GMM acceleration zone (0-4)
            if combo in gmm_zones:
                tmp[f"{combo}_accel_zone"] = gmm_zones[combo]

            # Prediction features
            if 'slope_f' in df.columns and 'angle' in df.columns:
                sf = df['slope_f'].fillna(0).values
                ag = df['angle'].fillna(0).values
                pred_dir, pred_rev = _compute_predictions(sf, ag)
                tmp[f"{combo}_pred_dir"] = pred_dir
                tmp[f"{combo}_pred_reversal"] = pred_rev

            # Sort and merge_asof to 5M base (forward-fill)
            tmp = tmp.sort_values('time')
            feat_cols = [c for c in tmp.columns if c != 'time']
            if feat_cols:
                features = pd.merge_asof(
                    features.sort_values('time'),
                    tmp[['time'] + feat_cols],
                    on='time',
                    direction='backward'
                )

        # Directional crossing flags between window pairs for this TF
        for ws1, ws2 in CROSSING_PAIRS:
            idx1 = WINDOW_SIZES.index(ws1) if ws1 in WINDOW_SIZES else -1
            idx2 = WINDOW_SIZES.index(ws2) if ws2 in WINDOW_SIZES else -1
            label1 = f"df{idx1}" if idx1 > 0 else "df"
            label2 = f"df{idx2}" if idx2 > 0 else "df"
            combo1 = f"{tf}_{label1}"
            combo2 = f"{tf}_{label2}"

            a1_col = f"{combo1}_angle"
            a2_col = f"{combo2}_angle"

            if a1_col in features.columns and a2_col in features.columns:
                a1 = np.nan_to_num(features[a1_col].values, nan=0.0)
                a2 = np.nan_to_num(features[a2_col].values, nan=0.0)
                features[f"{combo1}_C_{label2}"] = _detect_crossings_directional(a1, a2)

    return features


# ============================================================================
# Feature Version Splitting
# ============================================================================

def split_versions(features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split full feature matrix into V1, V2, V3 versions."""
    time_col = features[['time']]
    feat_only = features.drop(columns=['time'])

    # V1: All features (angle, slope_f, accel, R_bottom, R_peak, A, accel_zone, C, pred)
    v1 = feat_only.copy()

    # V2: Score per combo + convergence
    v2_cols = {}
    for tf in TIMEFRAME_ORDER:
        for i, ws in enumerate(WINDOW_SIZES):
            label = f"df{i}" if i > 0 else "df"
            combo = f"{tf}_{label}"

            r_bot_col = f"{combo}_R_bottom"
            r_peak_col = f"{combo}_R_peak"
            a_col = f"{combo}_A"
            c_cols = [c for c in feat_only.columns if c.startswith(f"{combo}_C_")]

            r_vals = np.zeros(len(feat_only))
            if r_bot_col in feat_only.columns:
                r_vals = np.maximum(r_vals, feat_only[r_bot_col].fillna(0).values)
            if r_peak_col in feat_only.columns:
                r_vals = np.maximum(r_vals, feat_only[r_peak_col].fillna(0).values)

            a_vals = feat_only[a_col].fillna(0).values if a_col in feat_only.columns else np.zeros(len(feat_only))
            c_vals = feat_only[c_cols].fillna(0).abs().max(axis=1).values if c_cols else np.zeros(len(feat_only))

            v2_cols[f"{combo}_score"] = r_vals + c_vals + a_vals

            slope_col = f"{combo}_slope_f"
            if slope_col in feat_only.columns:
                slopes = feat_only[slope_col].fillna(0).values
                v2_cols[f"{combo}_dir"] = np.where(slopes > 0.01, 1, np.where(slopes < -0.01, -1, 0))

    v2 = pd.DataFrame(v2_cols)

    dir_cols = [c for c in v2.columns if c.endswith('_dir')]
    if dir_cols:
        dir_matrix = v2[dir_cols].values
        v2['convergence_long'] = (dir_matrix == 1).sum(axis=1)
        v2['convergence_short'] = (dir_matrix == -1).sum(axis=1)
        v2['convergence_ratio'] = (v2['convergence_long'] - v2['convergence_short']) / max(len(dir_cols), 1)

    score_cols = [c for c in v2.columns if c.endswith('_score')]
    if score_cols:
        v2['total_score'] = v2[score_cols].sum(axis=1)
        v2['avg_score'] = v2[score_cols].mean(axis=1)

    # V3: Binary flags only (R_bottom, R_peak, A, C)
    binary_cols = [c for c in feat_only.columns
                   if c.endswith('_R_bottom') or c.endswith('_R_peak')
                   or c.endswith('_A') or '_C_' in c]
    v3 = feat_only[binary_cols].copy() if binary_cols else pd.DataFrame()

    return {'v1': v1, 'v2': v2, 'v3': v3}


# ============================================================================
# Label Builders
# ============================================================================

def build_labels(sl_pct: float = 1.0, tp_pct: float = 2.0,
                 max_hold: int = 50) -> pd.DataFrame:
    """Generate Y labels from 5M OHLCV using fixed SL/TP simulation."""
    ml_path = ML_DATA_DIR / "ml_data_5M.csv"
    csv_path = ml_path if ml_path.exists() else DATA_DIR / "testing_data_5M.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"5M CSV not found: {csv_path}")
    logger.info(f"  Label source: {csv_path}")

    price_data = pd.read_csv(csv_path)
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])

    labels_df = create_sl_tp_labels(
        price_data, sl_pct=sl_pct, tp_pct=tp_pct,
        max_hold_periods=max_hold, price_col='Close',
        high_col='High', low_col='Low', timestamp_col='Open Time'
    )
    return labels_df


def build_atr_labels(tp_pct: float = 4.0, atr_sl_mult: float = 0.10,
                     max_hold: int = 288, atr_period: int = 14) -> pd.DataFrame:
    """Generate Y labels from 5M OHLCV using ATR-based dynamic SL."""
    ml_path = ML_DATA_DIR / "ml_data_5M.csv"
    csv_path = ml_path if ml_path.exists() else DATA_DIR / "testing_data_5M.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"5M CSV not found: {csv_path}")
    logger.info(f"  Label source: {csv_path}")

    price_data = pd.read_csv(csv_path)
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])

    labels_df = create_atr_labels(
        price_data, tp_pct=tp_pct, atr_sl_mult=atr_sl_mult,
        max_hold_periods=max_hold, atr_period=atr_period,
        price_col='Close', high_col='High', low_col='Low',
        timestamp_col='Open Time'
    )
    return labels_df


# ============================================================================
# ETL Orchestrators
# ============================================================================

def run_etl_features(trim_months: int = 13) -> Dict:
    """
    Build features only (no labels). Expensive step -- call once, reuse across configs.
    Returns versions dict + feature timestamps for alignment.

    trim_months: Trim raw data to last N months before processing.
                 13 = 1yr train + 1mo test. 0 = no trim (full dataset).
    """
    logger.info("=" * 60)
    logger.info("ETL FEATURES PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load and process (trimmed to last 13 months)
    logger.info(f"Step 1: Loading and processing (trim={trim_months} months)...")
    processor, logic = load_and_process(trim_months=trim_months)

    # Step 2: Build feature matrix
    logger.info("Step 2: Building features aligned to 5M timeline...")
    features = build_features(processor, logic)
    logger.info(f"  Full feature matrix: {features.shape}")

    # Step 3: Split into versions
    logger.info("Step 3: Splitting into feature versions...")
    versions = split_versions(features)

    for name, df in versions.items():
        logger.info(f"  {name}: {df.shape}")

    logger.info("ETL FEATURES COMPLETE")

    return {
        **versions,
        'feat_times': features['time'],
        'processor': processor,
        'logic': logic,
    }


def align_features_labels(versions: Dict, feat_times: pd.Series,
                          labels: pd.DataFrame) -> tuple:
    """Align feature versions with labels by timestamp. Returns (aligned_versions, aligned_labels)."""
    label_times = pd.to_datetime(labels['timestamp'])
    common_times = set(feat_times.values) & set(label_times.values)
    logger.info(f"  Overlapping timestamps: {len(common_times)}")

    aligned_versions = {}

    if len(common_times) == 0:
        logger.warning("  No timestamp overlap - aligning by position (right-aligned)")
        min_len = min(len(feat_times), len(labels))
        for name, df in versions.items():
            if name in ('feat_times', 'processor', 'logic'):
                continue
            aligned_versions[name] = df.tail(min_len).reset_index(drop=True)
        labels = labels.head(min_len).reset_index(drop=True)
    else:
        common_sorted = sorted(common_times)
        feat_mask = feat_times.isin(common_sorted)
        label_mask = label_times.isin(common_sorted)

        for name, df in versions.items():
            if name in ('feat_times', 'processor', 'logic'):
                continue
            aligned_versions[name] = df[feat_mask.values].reset_index(drop=True)
        labels = labels[label_mask.values].reset_index(drop=True)

    return aligned_versions, labels


def run_etl(sl_pct: float = 1.0, tp_pct: float = 2.0, max_hold: int = 50) -> Dict:
    """Full ETL: load data, build features, build labels, align and save."""
    logger.info("=" * 60)
    logger.info("ETL PIPELINE START")
    logger.info("=" * 60)

    processor, logic = load_and_process()

    logger.info("Building features aligned to 5M timeline...")
    features = build_features(processor, logic)
    logger.info(f"  Full feature matrix: {features.shape}")

    versions = split_versions(features)

    for name, df in versions.items():
        logger.info(f"  {name}: {df.shape}")

    logger.info(f"Building labels (SL={sl_pct}%, TP={tp_pct}%, hold={max_hold})...")
    labels = build_labels(sl_pct, tp_pct, max_hold)
    logger.info(f"  Labels: {labels.shape}")

    feat_times = features['time']
    aligned_versions, labels = align_features_labels(
        versions, feat_times, labels
    )

    n_rows = len(labels)
    label_dist = labels['label'].value_counts().to_dict()
    logger.info(f"Aligned to {n_rows} rows | Distribution: {label_dist}")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    for name, df in aligned_versions.items():
        df.to_csv(output_dir / f"features_{name}.csv", index=False)
    labels.to_csv(output_dir / "labels.csv", index=False)

    logger.info(f"Saved to {output_dir}")
    logger.info("ETL PIPELINE COMPLETE")

    return {
        **aligned_versions,
        'labels': labels,
        'processor': processor,
        'logic': logic
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_etl()
