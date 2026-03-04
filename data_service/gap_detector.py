"""TF-aware gap detection for persistent kline CSVs."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .csv_io import get_max_time

logger = logging.getLogger(__name__)

# TF duration in minutes
TF_MINUTES = {
    "5M": 5, "15M": 15, "30M": 30,
    "1H": 60, "2H": 120, "4H": 240,
    "6H": 360, "8H": 480, "12H": 720,
    "1D": 1440, "3D": 4320,
}

# Bars to download on bootstrap — 1400 gives ~4.9 days of 5M context,
# matching backfill_predictions.py for consistent regression warm-up.
BOOTSTRAP_BARS = 1400


class GapDetector:
    """Detects gaps in persistent kline CSVs and calculates bars needed."""

    def __init__(self, klines_dir: Path):
        self.klines_dir = Path(klines_dir)

    def _kline_path(self, tf: str) -> Path:
        return self.klines_dir / f"ml_data_{tf}.csv"

    def check_tf(self, tf: str) -> dict:
        """Check a single TF for gaps.

        Returns:
            {
                "needs_update": bool,
                "last_time": Timestamp or None,
                "bars_needed": int,
                "is_bootstrap": bool,
            }
        """
        path = self._kline_path(tf)
        last_time = get_max_time(path, time_col="time")

        if last_time is None:
            # CSV missing or empty — full bootstrap
            return {
                "needs_update": True,
                "last_time": None,
                "bars_needed": BOOTSTRAP_BARS,
                "is_bootstrap": True,
            }

        # Calculate how many bars are missing
        now = datetime.now(timezone.utc)
        # Make last_time tz-aware for comparison
        if last_time.tzinfo is None:
            last_time_aware = last_time.tz_localize("UTC")
        else:
            last_time_aware = last_time

        gap_minutes = (now - last_time_aware).total_seconds() / 60
        tf_min = TF_MINUTES[tf]
        bars_needed = int(gap_minutes / tf_min)

        # Only update if at least 1 full bar has elapsed
        if bars_needed < 1:
            return {
                "needs_update": False,
                "last_time": last_time,
                "bars_needed": 0,
                "is_bootstrap": False,
            }

        # Add small buffer for safety margin
        bars_needed = bars_needed + 5

        # No cap — if gap is larger than BOOTSTRAP_BARS, we still need to fill it.
        # The Binance API paginator in layers.py handles large fetches via loop.
        if bars_needed > BOOTSTRAP_BARS:
            logger.warning(f"  {tf}: large gap detected ({bars_needed} bars needed)")

        return {
            "needs_update": True,
            "last_time": last_time,
            "bars_needed": bars_needed,
            "is_bootstrap": False,
        }

    def check_all(self, timeframes: list[str]) -> dict[str, dict]:
        """Check all TFs for gaps.

        Returns: {tf: check_result_dict}
        """
        results = {}
        for tf in timeframes:
            results[tf] = self.check_tf(tf)
        return results

    def needs_bootstrap(self, timeframes: list[str]) -> bool:
        """True if any kline CSV is missing/empty (requires full bootstrap)."""
        for tf in timeframes:
            path = self._kline_path(tf)
            if not path.exists() or path.stat().st_size == 0:
                return True
        return False

    def summary(self, timeframes: list[str]) -> str:
        """Human-readable gap summary."""
        results = self.check_all(timeframes)
        lines = []
        for tf, r in results.items():
            if r["is_bootstrap"]:
                lines.append(f"  {tf}: BOOTSTRAP ({r['bars_needed']} bars)")
            elif r["needs_update"]:
                lines.append(f"  {tf}: UPDATE ({r['bars_needed']} bars, "
                             f"last={r['last_time']})")
            else:
                lines.append(f"  {tf}: UP-TO-DATE (last={r['last_time']})")
        return "\n".join(lines)
