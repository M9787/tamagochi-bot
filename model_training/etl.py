"""
ETL Pipeline: 5-Matrix Augmented Feature Engineering for CatBoost.

Builds 5 matrices aligned to the 5M timeline:
  M1: Reversal Binary (55 cols)
  M2: Crossing Binary (77 cols)
  M3: Acceleration GMM Zone (55 cols, categorical 0-4)
  M4: Acceleration Absolute Value (55 cols, continuous)
  M5: Direction Binary (55 cols)

Augmented: [M1|M2|M3|M4|M5] = 297 base features
With lags 1-10: 297 * 11 = 3267 total features
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.mixture import GaussianMixture

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES, DATA_DIR
from core.processor import TimeframeProcessor
from core.signal_logic import SignalLogic
from data.target_labeling import create_sl_tp_labels, create_atr_labels

ML_DATA_DIR = Path(__file__).parent / "data"

logger = logging.getLogger(__name__)

# 7 crossing pairs (matches signal_logic.py)
CROSSING_PAIRS = [
    (30, 60), (30, 100), (60, 100), (60, 120),
    (100, 120), (100, 160), (120, 160)
]


def _win_label(i: int) -> str:
    """Window index to label mapping: 0->df, 1->df1, etc."""
    return f"df{i}" if i > 0 else "df"


# ============================================================================
# Data Loading
# ============================================================================

def load_and_process(trim_months: int = 0) -> Tuple[TimeframeProcessor, SignalLogic]:
    """Load data and run signal analysis."""
    processor = TimeframeProcessor()

    if ML_DATA_DIR.exists() and any(ML_DATA_DIR.glob("ml_data_*.csv")):
        logger.info(f"Using ML training data from: {ML_DATA_DIR}")
        _load_ml_data(processor)
    else:
        logger.info(f"Using default data from: {DATA_DIR}")
        processor.load_all_data()

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
# Reversal Detection
# ============================================================================

def _detect_reversal(angles: np.ndarray) -> np.ndarray:
    """
    Detect reversal events (R flag).
    Bottom: [a>b>c>d<e] -> 1
    Peak: [a<b<c<d>e] -> 1
    """
    n = len(angles)
    events = np.zeros(n, dtype=np.int8)
    for i in range(n - 4):
        w = angles[i:i + 5]
        if w[0] > w[1] > w[2] > w[3] and w[3] < w[4]:
            events[i + 4] = 1
        elif w[0] < w[1] < w[2] < w[3] and w[3] > w[4]:
            events[i + 4] = 1
    return events


# ============================================================================
# Crossing Detection
# ============================================================================

def _detect_crossing(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Detect angle crossover between two window angle series. Returns binary."""
    diff = a1 - a2
    n = len(diff)
    crossings = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if diff[i - 1] * diff[i] < 0:
            crossings[i] = 1
    return crossings


# ============================================================================
# GMM Zone Computation
# ============================================================================

def _compute_gmm_zones_for_tf(processor: TimeframeProcessor, tf: str) -> Dict[int, np.ndarray]:
    """
    Compute GMM acceleration zones for all windows of a timeframe.
    Returns {window_size: zone_array} where zone is 0-4.
    """
    all_accel = []
    combo_data = {}

    for i, ws in enumerate(WINDOW_SIZES):
        label = _win_label(i)
        if label not in processor.results.get(tf, {}):
            continue
        df = processor.results[tf][label]
        if 'acceleration' not in df.columns:
            continue
        accel = df['acceleration'].values
        combo_data[ws] = accel
        valid = accel[~np.isnan(accel)]
        all_accel.extend(valid.tolist())

    if len(all_accel) < 10:
        return {ws: np.full(len(arr), 2, dtype=np.int8) for ws, arr in combo_data.items()}

    all_arr = np.array(all_accel).reshape(-1, 1)
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=3)
    gmm.fit(all_arr)

    means = gmm.means_.flatten()
    sorted_indices = np.argsort(np.abs(means))
    cluster_to_zone = {cluster_id: rank for rank, cluster_id in enumerate(sorted_indices)}

    result = {}
    for ws, accel in combo_data.items():
        zones = np.full(len(accel), 2, dtype=np.int8)
        valid_mask = ~np.isnan(accel)
        if valid_mask.sum() > 0:
            clusters = gmm.predict(accel[valid_mask].reshape(-1, 1))
            mapped = np.array([cluster_to_zone[c] for c in clusters], dtype=np.int8)
            zones[valid_mask] = mapped
        result[ws] = zones

    return result


# ============================================================================
# Matrix Builders (M1-M5)
# ============================================================================

def _merge_to_base(base: pd.DataFrame, tmp: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Merge a single TF-window column onto the 5M base via merge_asof."""
    tmp = tmp.sort_values('time')
    base = pd.merge_asof(
        base.sort_values('time'),
        tmp[['time', col_name]],
        on='time',
        direction='backward'
    )
    return base


def build_m1_reversal(processor: TimeframeProcessor, base: pd.DataFrame) -> pd.DataFrame:
    """M1: Reversal Binary Matrix -- 55 columns (11 TF x 5 windows)."""
    m1 = base[['time']].copy()
    col_count = 0

    for tf in TIMEFRAME_ORDER:
        if tf not in processor.results:
            continue
        for i, ws in enumerate(WINDOW_SIZES):
            label = _win_label(i)
            if label not in processor.results[tf]:
                continue
            df = processor.results[tf][label]
            col_name = f"R_{tf}_{ws}"

            if 'angle' in df.columns and len(df) >= 5:
                rev = _detect_reversal(df['angle'].values)
            else:
                rev = np.zeros(len(df), dtype=np.int8)

            tmp = pd.DataFrame({'time': pd.to_datetime(df['time']), col_name: rev})
            m1 = _merge_to_base(m1, tmp, col_name)
            col_count += 1

    m1 = m1.drop(columns=['time'])
    logger.info(f"  M1 Reversal: {col_count} columns")
    return m1


def build_m2_crossing(processor: TimeframeProcessor, base: pd.DataFrame) -> pd.DataFrame:
    """M2: Crossing Binary Matrix -- 77 columns (11 TF x 7 crossing pairs)."""
    # Collect all angle series aligned to base
    angle_on_base = {}

    for tf in TIMEFRAME_ORDER:
        if tf not in processor.results:
            continue
        for i, ws in enumerate(WINDOW_SIZES):
            label = _win_label(i)
            if label not in processor.results[tf]:
                continue
            df = processor.results[tf][label]
            if 'angle' not in df.columns:
                continue

            tmp = pd.DataFrame({
                'time': pd.to_datetime(df['time']),
                'angle': df['angle'].values
            }).sort_values('time')

            merged = pd.merge_asof(
                base[['time']].sort_values('time'),
                tmp,
                on='time',
                direction='backward'
            )
            angle_on_base[(tf, ws)] = merged['angle'].fillna(0).values

    m2 = pd.DataFrame()
    col_count = 0

    for tf in TIMEFRAME_ORDER:
        for ws1, ws2 in CROSSING_PAIRS:
            col_name = f"C_{tf}_{ws1}x{ws2}"
            key1 = (tf, ws1)
            key2 = (tf, ws2)

            if key1 in angle_on_base and key2 in angle_on_base:
                cross = _detect_crossing(angle_on_base[key1], angle_on_base[key2])
            else:
                cross = np.zeros(len(base), dtype=np.int8)

            m2[col_name] = cross
            col_count += 1

    logger.info(f"  M2 Crossing: {col_count} columns")
    return m2


def build_m3_gmm_zone(processor: TimeframeProcessor, base: pd.DataFrame) -> pd.DataFrame:
    """M3: Acceleration GMM Zone Matrix -- 55 columns (categorical 0-4)."""
    m3 = base[['time']].copy()
    col_count = 0

    for tf in TIMEFRAME_ORDER:
        if tf not in processor.results:
            continue
        gmm_zones = _compute_gmm_zones_for_tf(processor, tf)

        for i, ws in enumerate(WINDOW_SIZES):
            label = _win_label(i)
            if label not in processor.results[tf]:
                continue
            df = processor.results[tf][label]
            col_name = f"AGMM_{tf}_{ws}"

            if ws in gmm_zones:
                zones = gmm_zones[ws]
            else:
                zones = np.full(len(df), 2, dtype=np.int8)

            tmp = pd.DataFrame({'time': pd.to_datetime(df['time']), col_name: zones})
            m3 = _merge_to_base(m3, tmp, col_name)
            col_count += 1

    m3 = m3.drop(columns=['time'])
    logger.info(f"  M3 GMM Zone: {col_count} columns")
    return m3


def build_m4_accel_abs(processor: TimeframeProcessor, base: pd.DataFrame) -> pd.DataFrame:
    """M4: Acceleration Absolute Value Matrix -- 55 columns (continuous float)."""
    m4 = base[['time']].copy()
    col_count = 0

    for tf in TIMEFRAME_ORDER:
        if tf not in processor.results:
            continue
        for i, ws in enumerate(WINDOW_SIZES):
            label = _win_label(i)
            if label not in processor.results[tf]:
                continue
            df = processor.results[tf][label]
            col_name = f"AABS_{tf}_{ws}"

            if 'acceleration' in df.columns:
                abs_accel = np.abs(df['acceleration'].values)
            else:
                abs_accel = np.zeros(len(df))

            tmp = pd.DataFrame({'time': pd.to_datetime(df['time']), col_name: abs_accel})
            m4 = _merge_to_base(m4, tmp, col_name)
            col_count += 1

    m4 = m4.drop(columns=['time'])
    logger.info(f"  M4 Accel Abs: {col_count} columns")
    return m4


def build_m5_direction(processor: TimeframeProcessor, base: pd.DataFrame) -> pd.DataFrame:
    """M5: Direction Binary Matrix -- 55 columns (1 if slope_f > 0, else 0)."""
    m5 = base[['time']].copy()
    col_count = 0

    for tf in TIMEFRAME_ORDER:
        if tf not in processor.results:
            continue
        for i, ws in enumerate(WINDOW_SIZES):
            label = _win_label(i)
            if label not in processor.results[tf]:
                continue
            df = processor.results[tf][label]
            col_name = f"D_{tf}_{ws}"

            if 'slope_f' in df.columns:
                direction = (df['slope_f'].values > 0).astype(np.int8)
            else:
                direction = np.zeros(len(df), dtype=np.int8)

            tmp = pd.DataFrame({'time': pd.to_datetime(df['time']), col_name: direction})
            m5 = _merge_to_base(m5, tmp, col_name)
            col_count += 1

    m5 = m5.drop(columns=['time'])
    logger.info(f"  M5 Direction: {col_count} columns")
    return m5


# ============================================================================
# Augmentation & Lag Features
# ============================================================================

def build_augmented_matrix(processor: TimeframeProcessor) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the full augmented feature matrix: [M1|M2|M3|M4|M5] + lags 1-10.

    Returns:
        (features_df, time_series): 3267-column feature matrix and timestamp index
    """
    base = _get_5m_base_timeline(processor)
    logger.info(f"  Base 5M timeline: {len(base)} rows")

    m1 = build_m1_reversal(processor, base)
    m2 = build_m2_crossing(processor, base)
    m3 = build_m3_gmm_zone(processor, base)
    m4 = build_m4_accel_abs(processor, base)
    m5 = build_m5_direction(processor, base)

    augmented = pd.concat([m1, m2, m3, m4, m5], axis=1)
    n_base = augmented.shape[1]
    logger.info(f"  Augmented base: {augmented.shape} ({n_base} features)")

    lag_frames = [augmented]
    for lag in range(1, 11):
        lagged = augmented.shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in augmented.columns]
        lag_frames.append(lagged)

    full = pd.concat(lag_frames, axis=1)

    # Drop rows with NaN from lagging (first 10 rows)
    full = full.iloc[10:].reset_index(drop=True)
    time_series = base['time'].iloc[10:].reset_index(drop=True)

    n_total = full.shape[1]
    logger.info(f"  With lags 1-10: {full.shape} ({n_total} features)")
    logger.info(f"  Expected: {n_base} base * 11 = {n_base * 11}")

    return full, time_series


def get_categorical_feature_indices(features: pd.DataFrame) -> List[int]:
    """Get column indices for M3 GMM zone features (base + lagged) for CatBoost."""
    cat_indices = []
    for i, col in enumerate(features.columns):
        if col.startswith("AGMM_"):
            cat_indices.append(i)
    return cat_indices


def get_categorical_feature_names(features: pd.DataFrame) -> List[str]:
    """Get column names for M3 GMM zone features (base + lagged)."""
    return [col for col in features.columns if col.startswith("AGMM_")]


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
# Alignment
# ============================================================================

def align_features_labels(features: pd.DataFrame, feat_times: pd.Series,
                          labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align feature matrix with labels by timestamp."""
    label_times = pd.to_datetime(labels['timestamp'])
    common_times = set(feat_times.values) & set(label_times.values)
    logger.info(f"  Overlapping timestamps: {len(common_times)}")

    if len(common_times) == 0:
        # F002: Never use positional fallback — it silently produces garbage
        raise ValueError(
            "No timestamp overlap between features and labels. "
            f"Feature times range: {feat_times.min()} to {feat_times.max()}, "
            f"Label times range: {label_times.min()} to {label_times.max()}"
        )
    else:
        common_sorted = sorted(common_times)
        feat_mask = feat_times.isin(common_sorted)
        label_mask = label_times.isin(common_sorted)
        aligned_features = features[feat_mask.values].reset_index(drop=True)
        aligned_labels = labels[label_mask.values].reset_index(drop=True)

    return aligned_features, aligned_labels


# ============================================================================
# ETL Orchestrators
# ============================================================================

def run_etl_features(trim_months: int = 13) -> Dict:
    """
    Build 5-matrix augmented features with lags.

    Returns dict with:
        'features': DataFrame (3267 columns)
        'feat_times': Series of timestamps
        'cat_indices': list of categorical feature column indices
        'cat_names': list of categorical feature column names
        'processor': TimeframeProcessor
        'logic': SignalLogic
    """
    logger.info("=" * 60)
    logger.info("ETL FEATURES PIPELINE (5-Matrix Augmented)")
    logger.info("=" * 60)

    logger.info(f"Step 1: Loading and processing (trim={trim_months} months)...")
    processor, logic = load_and_process(trim_months=trim_months)

    logger.info("Step 2: Building augmented feature matrix [M1|M2|M3|M4|M5] + lags...")
    features, feat_times = build_augmented_matrix(processor)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    cat_indices = get_categorical_feature_indices(features)
    cat_names = get_categorical_feature_names(features)

    for col in cat_names:
        features[col] = features[col].astype(int)

    logger.info(f"  Final features: {features.shape}")
    logger.info(f"  Categorical features (M3 GMM): {len(cat_indices)}")
    logger.info("ETL FEATURES COMPLETE")

    return {
        'features': features,
        'feat_times': feat_times,
        'cat_indices': cat_indices,
        'cat_names': cat_names,
        'processor': processor,
        'logic': logic,
    }


def run_etl(sl_pct: float = 1.0, tp_pct: float = 2.0, max_hold: int = 50) -> Dict:
    """Full ETL: load data, build features, build labels, align and save."""
    logger.info("=" * 60)
    logger.info("ETL PIPELINE START (5-Matrix Augmented)")
    logger.info("=" * 60)

    feat_data = run_etl_features(trim_months=13)
    features = feat_data['features']
    feat_times = feat_data['feat_times']

    logger.info(f"Building labels (SL={sl_pct}%, TP={tp_pct}%, hold={max_hold})...")
    labels = build_labels(sl_pct, tp_pct, max_hold)
    logger.info(f"  Labels: {labels.shape}")

    aligned_features, aligned_labels = align_features_labels(
        features, feat_times, labels
    )

    n_rows = len(aligned_labels)
    label_dist = aligned_labels['label'].value_counts().to_dict()
    logger.info(f"Aligned to {n_rows} rows | Distribution: {label_dist}")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    aligned_features.to_csv(output_dir / "features_augmented.csv", index=False)
    aligned_labels.to_csv(output_dir / "labels.csv", index=False)

    logger.info(f"Saved to {output_dir}")
    logger.info("ETL PIPELINE COMPLETE")

    return {
        'features': aligned_features,
        'labels': aligned_labels,
        'cat_indices': feat_data['cat_indices'],
        'cat_names': feat_data['cat_names'],
        'processor': feat_data['processor'],
        'logic': feat_data['logic'],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_etl()
