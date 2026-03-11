"""Data validation guards -- called before any data is used or written.

Pure functions, no external dependencies beyond pandas/numpy.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# Expected gap per timeframe (used by kline continuity check)
_TF_TIMEDELTA = {
    "5M": pd.Timedelta(minutes=5),
    "15M": pd.Timedelta(minutes=15),
    "30M": pd.Timedelta(minutes=30),
    "1H": pd.Timedelta(hours=1),
    "2H": pd.Timedelta(hours=2),
    "4H": pd.Timedelta(hours=4),
    "6H": pd.Timedelta(hours=6),
    "8H": pd.Timedelta(hours=8),
    "12H": pd.Timedelta(hours=12),
    "1D": pd.Timedelta(days=1),
    "3D": pd.Timedelta(days=3),
}


def validate_predictions_row(row: dict) -> list[str]:
    """Validate a single prediction row. Returns list of error strings (empty = valid)."""
    errors = []

    # Probabilities must sum to ~1.0
    probs = [row.get("prob_no_trade", 0), row.get("prob_long", 0), row.get("prob_short", 0)]
    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) > 0.01:
        errors.append(f"Probs sum to {prob_sum:.4f}, expected ~1.0")

    # Confidence must match max(probs)
    max_prob = max(probs)
    conf = row.get("confidence", 0)
    if abs(conf - max_prob) > 0.001:
        errors.append(f"Confidence {conf} != max(probs) {max_prob}")

    # Signal must be canonical
    signal = row.get("signal")
    if signal not in ("LONG", "SHORT", "NO_TRADE"):
        errors.append(f"Invalid signal: {signal}")

    # No NaN/inf in numeric fields
    for k, v in row.items():
        if isinstance(v, float) and (v != v or abs(v) == float("inf")):
            errors.append(f"NaN/inf in {k}")

    return errors


def validate_feature_shape(features_df: pd.DataFrame, expected_count: int) -> list[str]:
    """Validate feature DataFrame shape and content."""
    errors = []
    # Subtract 'time' column from count
    non_time_cols = [c for c in features_df.columns if c != "time"]
    actual = len(non_time_cols)
    if actual != expected_count:
        errors.append(f"Feature count {actual} != expected {expected_count}")

    nan_cols = features_df[non_time_cols].columns[features_df[non_time_cols].isna().any()].tolist()
    if nan_cols:
        errors.append(f"NaN in {len(nan_cols)} columns: {nan_cols[:5]}")

    return errors


def validate_kline_continuity(df: pd.DataFrame, tf: str) -> list[str]:
    """Check for gaps in kline data."""
    errors = []
    if "time" not in df.columns and "Open Time" not in df.columns:
        return ["No time column found"]

    time_col = "time" if "time" in df.columns else "Open Time"
    times = pd.to_datetime(df[time_col]).sort_values()
    if len(times) < 2:
        return []

    expected_gap = _TF_TIMEDELTA.get(tf)
    if expected_gap is None:
        return [f"Unknown timeframe: {tf}"]

    diffs = times.diff().dropna()
    gaps = diffs[diffs > expected_gap * 1.5]
    if len(gaps) > 0:
        errors.append(f"{len(gaps)} gaps in {tf} klines (largest: {gaps.max()})")

    return errors


def validate_predictions_freshness(pred_time, max_age_sec: int | None = None) -> list[str]:
    """Check if prediction is stale."""
    if max_age_sec is None:
        from core.config import STALENESS_THRESHOLD_SEC
        max_age_sec = STALENESS_THRESHOLD_SEC

    if pred_time is None:
        return ["No prediction time provided"]

    pred_time = pd.to_datetime(pred_time)
    if pred_time.tzinfo is None:
        pred_time = pred_time.tz_localize("UTC")

    age = (datetime.now(timezone.utc) - pred_time).total_seconds()
    if age > max_age_sec:
        return [f"Prediction stale: {age:.0f}s old (max {max_age_sec}s)"]
    return []
