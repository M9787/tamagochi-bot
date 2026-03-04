"""
Step 2.5: Precompute SL/TP labels for 5M candles → save as CSV.
Run once after download_data.py. Reused by train.py and manual exploration.

Usage:
    python model_training/build_labels.py
    python model_training/build_labels.py --sl 2.0 --tp 4.0 --max-hold 288
    python model_training/build_labels.py --force
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.target_labeling import create_sl_tp_labels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
ENCODED_DATA_DIR = Path(__file__).parent / "encoded_data"
LABELS_PATH = ENCODED_DATA_DIR / "labels_5M.csv"

# Defaults matching train.py
DEFAULT_SL = 2.0
DEFAULT_TP = 4.0
DEFAULT_MAX_HOLD = 288


def build_and_save(sl_pct: float = DEFAULT_SL, tp_pct: float = DEFAULT_TP,
                   max_hold: int = DEFAULT_MAX_HOLD, force: bool = False):
    if LABELS_PATH.exists() and not force:
        logger.info(f"Labels already exist: {LABELS_PATH}")
        logger.info("Use --force to overwrite.")
        existing = pd.read_csv(LABELS_PATH)
        dist = existing['label'].value_counts().to_dict()
        logger.info(f"  Rows: {len(existing):,} | Distribution: {dist}")
        return LABELS_PATH

    ml_path = ACTUAL_DATA_DIR / "ml_data_5M.csv"
    if not ml_path.exists():
        raise FileNotFoundError(f"5M price data not found: {ml_path}\nRun download_data.py first.")

    logger.info(f"Loading 5M price data from {ml_path}...")
    price_data = pd.read_csv(ml_path)
    if 'index' in price_data.columns:
        price_data = price_data.drop(columns=['index'])
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])

    logger.info(f"Computing SL/TP labels: SL={sl_pct}%, TP={tp_pct}%, max_hold={max_hold}")
    logger.info(f"  Price rows: {len(price_data):,}")
    logger.info(f"  Date range: {price_data['Open Time'].min()} to {price_data['Open Time'].max()}")

    t0 = time.time()
    labels_df = create_sl_tp_labels(
        price_data, sl_pct=sl_pct, tp_pct=tp_pct,
        max_hold_periods=max_hold, price_col='Close',
        high_col='High', low_col='Low', timestamp_col='Open Time'
    )
    elapsed = time.time() - t0

    ENCODED_DATA_DIR.mkdir(exist_ok=True)
    labels_df.to_csv(LABELS_PATH, index=False)

    dist = labels_df['label'].value_counts().to_dict()
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Saved: {LABELS_PATH}")
    logger.info(f"  Rows: {len(labels_df):,} | Distribution: {dist}")
    logger.info(f"  Columns: {list(labels_df.columns)}")

    return LABELS_PATH


def load_labels() -> pd.DataFrame:
    """Load precomputed labels. Raises if not found."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Precomputed labels not found: {LABELS_PATH}\n"
            f"Run: python model_training/build_labels.py"
        )
    df = pd.read_csv(LABELS_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute SL/TP labels for 5M candles")
    parser.add_argument("--sl", type=float, default=DEFAULT_SL, help=f"Stop Loss %% (default: {DEFAULT_SL})")
    parser.add_argument("--tp", type=float, default=DEFAULT_TP, help=f"Take Profit %% (default: {DEFAULT_TP})")
    parser.add_argument("--max-hold", type=int, default=DEFAULT_MAX_HOLD, help=f"Max hold periods (default: {DEFAULT_MAX_HOLD})")
    parser.add_argument("--force", action="store_true", help="Overwrite existing labels")
    args = parser.parse_args()

    build_and_save(sl_pct=args.sl, tp_pct=args.tp, max_hold=args.max_hold, force=args.force)
