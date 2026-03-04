"""Atomic CSV read/append helpers for persistent data layers."""

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read a CSV file, returning None if missing or empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None


def get_max_time(path: Path, time_col: str = "time") -> pd.Timestamp | None:
    """Get the maximum timestamp from a CSV.

    Loads the full file. For the current data volumes (<10K rows) this is fast.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        # Read header to find time column index
        df = pd.read_csv(path)
        if df.empty or time_col not in df.columns:
            return None
        max_t = pd.to_datetime(df[time_col]).max()
        if pd.isna(max_t):
            return None
        return max_t
    except Exception as e:
        logger.warning(f"Failed to get max time from {path}: {e}")
        return None


def append_rows_atomic(path: Path, new_rows: pd.DataFrame) -> int:
    """Append rows to a CSV file atomically.

    ALL writes go through temp file + os.replace() so readers never see
    a half-written file. This prevents the race condition where the bot
    reads a partially-written row from the data service.

    Returns number of rows appended.
    """
    path = Path(path)
    if new_rows is None or new_rows.empty:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing data (if any)
    existing = read_csv_safe(path)

    if existing is not None:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    # Write combined data to temp file, then atomic rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp")
    os.close(tmp_fd)
    try:
        combined.to_csv(tmp_path, index=False)
        os.replace(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return len(new_rows)


def read_tail(path: Path, n_rows: int, time_col: str = "time") -> pd.DataFrame | None:
    """Read the last n_rows from a CSV file.

    Loads the full file and returns tail. For our use case (~500 rows per TF),
    this is fast enough. For very large files, a chunked reader could be used.
    """
    df = read_csv_safe(path)
    if df is None:
        return None
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
    return df.tail(n_rows).reset_index(drop=True)
