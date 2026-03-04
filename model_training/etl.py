"""
ETL Pipeline: Extract kline data → Transform via regression → 55 decomposed CSVs.

Usage:
    python model_training/etl.py --start 2020-01-01 --end 2026-02-15
    python model_training/etl.py --start 2020-01-01 --end 2026-02-15 --force
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
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from core.analysis import iterative_regression, calculate_acceleration

ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
DECOMPOSED_DIR = Path(__file__).parent / "decomposed_data"

logger = logging.getLogger(__name__)


def _load_single_tf(tf: str) -> pd.DataFrame:
    """Load one timeframe CSV from actual_data/."""
    csv_path = ACTUAL_DATA_DIR / f"ml_data_{tf}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    df['Open Time'] = pd.to_datetime(df['Open Time'], utc=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    return df


def decompose_single(tf: str, ws: int, start: str = None, end: str = None,
                      force: bool = False) -> Path:
    """
    Run iterative_regression + acceleration on one TF/window combo.
    Saves result to decomposed_data/decomposed_{tf}_w{ws}.csv.

    Returns path to output CSV.
    """
    out_path = DECOMPOSED_DIR / f"decomposed_{tf}_w{ws}.csv"

    if out_path.exists() and not force:
        logger.info(f"  SKIP {tf}/w{ws} (exists)")
        return out_path

    df_raw = _load_single_tf(tf)

    # Filter by date range
    if start:
        df_raw = df_raw[df_raw['Open Time'] >= pd.Timestamp(start, tz='UTC')]
    if end:
        df_raw = df_raw[df_raw['Open Time'] <= pd.Timestamp(end, tz='UTC')]

    if len(df_raw) < ws * 2 + 1:
        logger.warning(f"  SKIP {tf}/w{ws}: only {len(df_raw)} rows (need {ws * 2 + 1})")
        return out_path

    # Select only the columns needed for regression
    df_input = df_raw[['Open Time', 'Close']].copy()

    # Run iterative regression
    result = iterative_regression(df_input, window_size=ws)

    # Calculate acceleration from angle series
    result['acceleration'] = calculate_acceleration(result['angle'])

    # Save
    result.to_csv(out_path, index=False)
    logger.info(f"  OK {tf}/w{ws}: {len(result)} rows -> {out_path.name}")

    return out_path


def _decompose_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable on Windows spawn)."""
    tf, ws, start, end, force = args
    # Re-configure logging in subprocess
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    out_path = decompose_single(tf, ws, start=start, end=end, force=force)
    return tf, ws, out_path


def run_decomposition(start: str = None, end: str = None, force: bool = False):
    """Parallelize all 55 TF/window combos with ProcessPoolExecutor."""
    DECOMPOSED_DIR.mkdir(parents=True, exist_ok=True)

    total = len(TIMEFRAME_ORDER) * len(WINDOW_SIZES)
    logger.info(f"Decomposing {total} TF/window combos -> {DECOMPOSED_DIR}")
    if start:
        logger.info(f"  Date range: {start} to {end or 'now'}")

    # Build task list
    tasks = []
    for tf in TIMEFRAME_ORDER:
        for ws in WINDOW_SIZES:
            tasks.append((tf, ws, start, end, force))

    max_workers = min(os.cpu_count() - 1, 8) if os.cpu_count() and os.cpu_count() > 1 else 1
    logger.info(f"  Using {max_workers} parallel workers")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_decompose_worker, t): t for t in tasks}
        for future in as_completed(futures):
            tf, ws, out_path = future.result()
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                logger.info(f"  Progress: {done}/{total} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"Decomposition complete: {done}/{total} in {elapsed:.1f}s")


def run_etl(start: str = None, end: str = None, force: bool = False):
    """Main entry point for the ETL pipeline."""
    logger.info("=" * 60)
    logger.info("ETL PIPELINE: Extract + Transform")
    logger.info("=" * 60)

    if not ACTUAL_DATA_DIR.exists():
        raise FileNotFoundError(
            f"actual_data/ not found: {ACTUAL_DATA_DIR}\n"
            f"Run: python model_training/download_data.py"
        )

    csv_count = len(list(ACTUAL_DATA_DIR.glob("ml_data_*.csv")))
    logger.info(f"Source: {ACTUAL_DATA_DIR} ({csv_count} CSVs)")

    run_decomposition(start=start, end=end, force=force)

    # Summary
    decomposed_count = len(list(DECOMPOSED_DIR.glob("decomposed_*.csv")))
    logger.info(f"\nETL COMPLETE: {decomposed_count} decomposed CSVs in {DECOMPOSED_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="ETL: klines → decomposed regression CSVs")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    run_etl(start=args.start, end=args.end, force=args.force)
