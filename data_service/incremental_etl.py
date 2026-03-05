"""Incremental iterative_regression — only compute new rows."""

import logging
from pathlib import Path

import pandas as pd

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from core.analysis import iterative_regression, calculate_acceleration
from .csv_io import read_csv_safe, get_max_time, append_rows_atomic

logger = logging.getLogger(__name__)


def compute_incremental_decomposition(
    klines_df: pd.DataFrame,
    existing_path: Path,
    window_size: int,
    tf: str,
) -> pd.DataFrame | None:
    """Compute decomposed rows incrementally.

    If no existing CSV: full bootstrap (run regression on all klines).
    If existing CSV: load tail context, run regression, keep only NEW rows.

    Args:
        klines_df: Full klines DataFrame for this TF (with 'time' and 'Open Time' columns)
        existing_path: Path to existing decomposed CSV (may not exist)
        window_size: Regression window size
        tf: Timeframe label (for logging)

    Returns:
        DataFrame of NEW rows to append (or None if nothing new)
    """
    existing_max = get_max_time(existing_path, time_col="time")

    if existing_max is None:
        # Full bootstrap — run regression on all klines
        df_input = klines_df[["Open Time", "Close"]].copy()
        if len(df_input) < window_size * 2 + 1:
            logger.warning(f"  SKIP {tf}/w{window_size}: only {len(df_input)} rows "
                           f"(need {window_size * 2 + 1})")
            return None

        result = iterative_regression(df_input, window_size=window_size)
        result["acceleration"] = calculate_acceleration(result["angle"])
        result["time"] = pd.to_datetime(result["time"]).dt.tz_localize(None)
        result = result.sort_values("time").reset_index(drop=True)
        logger.info(f"  {tf}/w{window_size}: BOOTSTRAP {len(result)} rows")
        return result

    # Incremental — need context for regression warm-up
    # iterative_regression needs 2*ws rows to produce 1 output row.
    # To get n_new output rows, we need 2*ws + n_new input rows.
    existing_max = pd.to_datetime(existing_max)
    if existing_max.tzinfo is not None:
        existing_max = existing_max.tz_localize(None)

    klines_df = klines_df.copy()
    klines_df["time"] = pd.to_datetime(klines_df["time"]).dt.tz_localize(None)

    # Find new klines (strictly after existing max)
    new_klines = klines_df[klines_df["time"] > existing_max]
    n_new = len(new_klines)

    if n_new == 0:
        return None

    # Context: use ALL available klines (up to full file) for regression warm-up.
    # iterative_regression builds its sliding window sequentially, so more context
    # means the window state converges closer to what a full-run would produce.
    # The minimum is 2*ws + n_new, but we use all available rows to eliminate
    # numerical divergence between incremental and full-run regression values.
    context_rows = len(klines_df)  # use all available rows
    tail_klines = klines_df.tail(context_rows).reset_index(drop=True)

    if len(tail_klines) < window_size * 2 + 1:
        logger.warning(f"  SKIP {tf}/w{window_size}: insufficient context "
                       f"({len(tail_klines)} < {window_size * 2 + 1})")
        return None

    # Run regression on context
    df_input = tail_klines[["Open Time", "Close"]].copy()
    result = iterative_regression(df_input, window_size=window_size)
    result["acceleration"] = calculate_acceleration(result["angle"])
    result["time"] = pd.to_datetime(result["time"]).dt.tz_localize(None)
    result = result.sort_values("time").reset_index(drop=True)

    # Keep only rows strictly after existing max
    new_rows = result[result["time"] > existing_max].copy()

    if new_rows.empty:
        return None

    logger.info(f"  {tf}/w{window_size}: +{len(new_rows)} new rows "
                f"(context={len(tail_klines)})")
    return new_rows


def run_incremental_etl(
    klines_dir: Path,
    decomposed_dir: Path,
    klines_dict: dict[str, pd.DataFrame] | None = None,
) -> dict[tuple, pd.DataFrame]:
    """Run incremental ETL for all 55 TF/window combos.

    If klines_dict is provided, uses in-memory data. Otherwise loads from disk.
    Appends new rows to decomposed CSVs and returns the full decomposed dict
    (loaded from disk after append) for downstream encoding.

    Returns:
        dict: {(tf, ws): DataFrame} — full decomposed data for all combos
    """
    decomposed_dir = Path(decomposed_dir)
    decomposed_dir.mkdir(parents=True, exist_ok=True)

    n_updated = 0

    for tf in TIMEFRAME_ORDER:
        # Load klines
        if klines_dict is not None and tf in klines_dict:
            kl = klines_dict[tf]
        else:
            kl_path = Path(klines_dir) / f"ml_data_{tf}.csv"
            if not kl_path.exists():
                logger.warning(f"  No klines for {tf} — skipping")
                continue
            kl = pd.read_csv(kl_path)
            kl["Open Time"] = pd.to_datetime(kl["Open Time"])
            kl["time"] = pd.to_datetime(kl["time"])
            for c in ("Open", "High", "Low", "Close", "Volume"):
                if c in kl.columns:
                    kl[c] = pd.to_numeric(kl[c], errors="coerce")

        for ws in WINDOW_SIZES:
            decomp_path = decomposed_dir / f"decomposed_{tf}_w{ws}.csv"
            new_rows = compute_incremental_decomposition(
                kl, decomp_path, ws, tf)

            if new_rows is not None:
                n_appended = append_rows_atomic(decomp_path, new_rows)
                n_updated += 1
                logger.debug(f"  Appended {n_appended} rows to {decomp_path.name}")

    logger.info(f"  Incremental ETL: {n_updated}/55 combos updated")

    # Load full decomposed data for encoding
    decomposed = {}
    for tf in TIMEFRAME_ORDER:
        for ws in WINDOW_SIZES:
            decomp_path = decomposed_dir / f"decomposed_{tf}_w{ws}.csv"
            df = read_csv_safe(decomp_path)
            if df is not None:
                df["time"] = pd.to_datetime(df["time"], format="mixed").dt.tz_localize(None)
                df = df.sort_values("time").reset_index(drop=True)
                decomposed[(tf, ws)] = df

    logger.info(f"  Loaded {len(decomposed)} decomposed datasets for encoding")
    return decomposed
