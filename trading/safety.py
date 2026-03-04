"""Safety monitor — rolling win-rate tracking with auto-pause."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Tracks trade outcomes and pauses trading if win rate drops below break-even
    or consecutive losses exceed threshold."""

    def __init__(self, min_wr: float = 33.3, window: int = 20,
                 max_consecutive_losses: int = 5,
                 pause_cooldown_seconds: int = 3600):
        self.min_wr = min_wr
        self.window = window
        self.max_consecutive_losses = max_consecutive_losses
        self.pause_cooldown_seconds = pause_cooldown_seconds
        self.trades: list[dict] = []  # [{timestamp, win}, ...]
        self.paused = False
        self.pause_reason = ""
        self.paused_since: datetime | None = None
        self.cooldown_grace = False

    def record_trade(self, win: bool, timestamp: str | None = None):
        """Record a completed trade outcome."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        self.trades.append({"timestamp": timestamp, "win": win})
        self.cooldown_grace = False  # New trade data arrived — clear grace
        logger.info(f"Trade recorded: {'WIN' if win else 'LOSS'} "
                     f"(total: {len(self.trades)})")
        # Re-check after each trade
        self._evaluate()

    def _evaluate(self):
        """Check consecutive losses and rolling WR, update pause state."""
        # Check cooldown expiry first — auto-resume after pause_cooldown_seconds
        if self.paused and self.paused_since is not None:
            elapsed = (datetime.now(timezone.utc) - self.paused_since).total_seconds()
            if elapsed >= self.pause_cooldown_seconds:
                logger.info(
                    f"SAFETY RESUME: cooldown expired after "
                    f"{elapsed:.0f}s (limit: {self.pause_cooldown_seconds}s)")
                self.paused = False
                self.pause_reason = ""
                self.paused_since = None
                self.cooldown_grace = True  # Skip re-evaluation until new trade
                return

        # Grace period: don't re-evaluate old losses until a new trade arrives
        if self.cooldown_grace:
            return

        # Check consecutive losses first (catches rapid streaks early)
        consec_losses = 0
        for t in reversed(self.trades):
            if not t["win"]:
                consec_losses += 1
            else:
                break

        if consec_losses >= self.max_consecutive_losses:
            if not self.paused:
                self.paused_since = datetime.now(timezone.utc)
            self.paused = True
            self.pause_reason = (
                f"{consec_losses} consecutive losses "
                f"(max: {self.max_consecutive_losses})")
            logger.warning(f"SAFETY PAUSE: {self.pause_reason}")
            return

        # Check rolling WR
        if len(self.trades) < self.window:
            # Not enough data to judge
            self.paused = False
            self.pause_reason = ""
            self.paused_since = None
            return

        recent = self.trades[-self.window:]
        wins = sum(1 for t in recent if t["win"])
        wr = (wins / len(recent)) * 100

        if wr < self.min_wr:
            if not self.paused:
                self.paused_since = datetime.now(timezone.utc)
            self.paused = True
            self.pause_reason = (
                f"Rolling WR {wr:.1f}% < {self.min_wr:.1f}% "
                f"over last {self.window} trades")
            logger.warning(f"SAFETY PAUSE: {self.pause_reason}")
        else:
            self.paused = False
            self.pause_reason = ""
            self.paused_since = None

    def check(self) -> bool:
        """Return True if safe to trade, False if paused."""
        self._evaluate()
        return not self.paused

    def get_stats(self) -> dict:
        """Return current safety statistics."""
        total = len(self.trades)
        total_wins = sum(1 for t in self.trades if t["win"])

        # Rolling stats
        rolling_wr = None
        if total >= self.window:
            recent = self.trades[-self.window:]
            wins = sum(1 for t in recent if t["win"])
            rolling_wr = (wins / len(recent)) * 100

        # Consecutive losses
        consec_losses = 0
        for t in reversed(self.trades):
            if not t["win"]:
                consec_losses += 1
            else:
                break

        return {
            "total_trades": total,
            "total_wins": total_wins,
            "total_wr": (total_wins / total * 100) if total > 0 else 0.0,
            "rolling_wr": rolling_wr,
            "consecutive_losses": consec_losses,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
        }

    def to_list(self) -> dict:
        """Serialize trade history for persistence."""
        return {
            "trades": list(self.trades),
            "paused_since": self.paused_since.isoformat() if self.paused_since else None,
            "cooldown_grace": self.cooldown_grace,
        }

    def load_from_list(self, data):
        """Restore trade history from persistence.

        Accepts either a list (legacy format) or a dict with trades + paused_since.
        """
        if isinstance(data, dict):
            self.trades = data.get("trades", [])
            paused_str = data.get("paused_since")
            if paused_str:
                self.paused_since = datetime.fromisoformat(paused_str)
            else:
                self.paused_since = None
            self.cooldown_grace = data.get("cooldown_grace", False)
        else:
            # Legacy: plain list of trades
            self.trades = data
            self.paused_since = None
            self.cooldown_grace = False
        self._evaluate()
