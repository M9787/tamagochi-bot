"""Position manager — tracks position state, average entry, SL/TP, max-hold, and profit-lock."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages position state: entries, avg price, SL/TP levels, max-hold timer, profit lock.

    Position states:
        None  — no open position
        LONG  — long position open
        SHORT — short position open
    """

    def __init__(self, sl_pct: float = 2.0, tp_pct: float = 4.0,
                 max_hold_seconds: int = 86400,
                 profit_lock_trigger: float = 3.5,
                 profit_lock_floor: float = 3.0):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.max_hold_seconds = max_hold_seconds
        self.profit_lock_trigger = profit_lock_trigger
        self.profit_lock_floor = profit_lock_floor

        # Position state
        self.current_side: str | None = None
        self.entries: list[list[float]] = []  # [[price, quantity], ...]
        self.avg_entry = 0.0
        self.total_quantity = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0

        # Max-hold tracking
        self.entry_time: datetime | None = None

        # Profit lock tracking
        self.high_water_mark_pct = 0.0
        self.profit_lock_active = False

    def sync_from_exchange(self, position_info: dict):
        """Sync local state from exchange position data.

        Called on startup to recover state after restart.
        """
        amt = float(position_info.get("positionAmt", 0))
        entry = float(position_info.get("entryPrice", 0))

        if abs(amt) < 1e-8:
            # No open position
            self.reset()
            logger.info("Position sync: no open position")
            return

        self.current_side = "LONG" if amt > 0 else "SHORT"
        self.total_quantity = abs(amt)
        self.avg_entry = entry
        self.entries = [[entry, abs(amt)]]
        self.sl_price, self.tp_price = self.calculate_sl_tp(entry, self.current_side)

        # If recovering and entry_time not set, use now as conservative fallback
        if self.entry_time is None:
            self.entry_time = datetime.now(timezone.utc)

        logger.info(
            f"Position sync: {self.current_side} {self.total_quantity} BTC "
            f"@ {self.avg_entry:.2f} | SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

    def calculate_sl_tp(self, avg_entry: float, side: str) -> tuple[float, float]:
        """Calculate SL and TP prices from average entry.

        LONG:  SL = entry * (1 - sl_pct/100), TP = entry * (1 + tp_pct/100)
        SHORT: SL = entry * (1 + sl_pct/100), TP = entry * (1 - tp_pct/100)
        """
        if side == "LONG":
            sl = avg_entry * (1 - self.sl_pct / 100)
            tp = avg_entry * (1 + self.tp_pct / 100)
        else:  # SHORT
            sl = avg_entry * (1 + self.sl_pct / 100)
            tp = avg_entry * (1 - self.tp_pct / 100)
        return round(sl, 1), round(tp, 1)

    def add_entry(self, price: float, quantity: float):
        """Record a new entry (open or add-to-position).

        Recalculates average entry and SL/TP.
        """
        self.entries.append([price, quantity])
        self.total_quantity = sum(q for _, q in self.entries)
        self.avg_entry = (
            sum(p * q for p, q in self.entries) / self.total_quantity
        )
        self.sl_price, self.tp_price = self.calculate_sl_tp(
            self.avg_entry, self.current_side)

        logger.info(
            f"Entry added: {price:.2f} x {quantity:.4f} | "
            f"Avg={self.avg_entry:.2f} Total={self.total_quantity:.4f} | "
            f"SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

    def is_max_hold_expired(self) -> bool:
        """Check if position has exceeded max hold time (24h)."""
        if self.entry_time is None or self.current_side is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.entry_time).total_seconds()
        return elapsed >= self.max_hold_seconds

    def remaining_hold_seconds(self) -> float | None:
        """Seconds remaining until max_hold expiry. None if no position."""
        if self.entry_time is None or self.current_side is None:
            return None
        elapsed = (datetime.now(timezone.utc) - self.entry_time).total_seconds()
        return max(0, self.max_hold_seconds - elapsed)

    def update_profit_lock(self, mark_price: float) -> str | None:
        """Update profit tracking and check profit lock trigger.

        Returns close reason string if position should be closed, None otherwise.
        """
        if self.current_side is None:
            return None

        # Calculate unrealized PnL %
        if self.current_side == "LONG":
            pnl_pct = (mark_price - self.avg_entry) / self.avg_entry * 100
        else:  # SHORT
            pnl_pct = (self.avg_entry - mark_price) / self.avg_entry * 100

        # Update high water mark
        self.high_water_mark_pct = max(self.high_water_mark_pct, pnl_pct)

        # Check trigger
        if self.high_water_mark_pct >= self.profit_lock_trigger:
            self.profit_lock_active = True

        # Check floor
        if self.profit_lock_active and pnl_pct <= self.profit_lock_floor:
            logger.info(
                f"PROFIT LOCK triggered: peaked at {self.high_water_mark_pct:.2f}%, "
                f"now {pnl_pct:.2f}% <= {self.profit_lock_floor}% floor")
            return "PROFIT_LOCK"

        return None

    def handle_signal(self, signal: str, executor) -> tuple[str, dict | None]:
        """Process a trade signal and execute via executor.

        Returns (action, close_info) tuple:
            action:
                OPENED         — new position opened
                ADDED          — added to existing position
                CLOSED_WAITING — opposite signal, closed current, will open next cycle
                SKIPPED        — no action needed
                SL_TP_FAILED   — position closed because SL/TP couldn't be placed
                CLOSE_FAILED   — close attempt failed, will retry next cycle
            close_info:
                dict with prev_side, avg_entry, sl_price, tp_price, total_quantity,
                close_price, order_id — only set when action is CLOSED_WAITING
        """
        if signal not in ("LONG", "SHORT"):
            return "SKIPPED", None

        # Case 1: No position -> open new
        if self.current_side is None:
            return self._open_new(signal, executor), None

        # Case 2: Same direction -> add to position
        if self.current_side == signal:
            return self._add_to_position(signal, executor), None

        # Case 3: Opposite direction -> close current, wait for next signal
        return self._close_and_wait(executor)

    def _open_new(self, signal: str, executor) -> str:
        """Open a new position with SL/TP placed atomically."""
        self.current_side = signal
        order_side = "BUY" if signal == "LONG" else "SELL"

        # Pre-calculate SL/TP so executor can place them immediately after fill
        # Use mark price estimate for SL/TP (will be close enough to fill price)
        try:
            est_price = executor.get_mark_price()
        except Exception:
            est_price = 0
        if est_price > 0:
            sl, tp = self.calculate_sl_tp(est_price, signal)
        else:
            sl, tp = 0.0, 0.0

        result = executor.open_position(side=order_side,
                                        sl_price=sl, tp_price=tp)
        if result is None:
            self.reset()
            return "SKIPPED"

        self.add_entry(result["price"], result["quantity"])
        self.entry_time = datetime.now(timezone.utc)

        # Recalculate SL/TP with actual fill price (may differ slightly from estimate)
        # and update on exchange if needed
        actual_sl, actual_tp = self.sl_price, self.tp_price  # set by add_entry()
        if not executor.dry_run and (abs(actual_sl - sl) > 0.1 or abs(actual_tp - tp) > 0.1):
            logger.info(f"Adjusting SL/TP: estimated SL={sl:.1f}/TP={tp:.1f} → "
                        f"actual SL={actual_sl:.1f}/TP={actual_tp:.1f}")
            executor.update_sl_tp(side=self.current_side,
                                  sl_price=actual_sl, tp_price=actual_tp)

        if not result.get("sl_tp_ok", True):
            logger.critical(
                "SL/TP placement failed on new position — "
                "position closed by executor for safety")
            self.reset()
            return "SL_TP_FAILED"

        return "OPENED"

    def _add_to_position(self, signal: str, executor) -> str:
        """Add to existing same-direction position."""
        order_side = "BUY" if signal == "LONG" else "SELL"

        # Market order first — existing SL/TP remain intact if this fails
        result = executor.add_to_position(side=order_side)
        if result is None:
            return "SKIPPED"

        self.add_entry(result["price"], result["quantity"])

        # Update SL/TP on exchange with new avg entry
        if not executor.dry_run:
            sl_tp_ok = executor.update_sl_tp(
                side=self.current_side,
                sl_price=self.sl_price,
                tp_price=self.tp_price)
            if not sl_tp_ok:
                logger.critical(
                    "SL/TP update failed after add — "
                    "position closed by executor for safety")
                self.reset()
                return "SL_TP_FAILED"

        return "ADDED"

    def _close_and_wait(self, executor) -> tuple[str, dict | None]:
        """Close current position on opposite signal.

        Returns (action, close_info) — close_info captures pre-reset state so the
        caller can record the trade and log correct values.
        """
        logger.info(
            f"Opposite signal received — closing {self.current_side} position")

        # Capture state BEFORE reset
        close_info = {
            "prev_side": self.current_side,
            "avg_entry": self.avg_entry,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "total_quantity": self.total_quantity,
        }

        result = executor.close_position(
            local_qty=self.total_quantity, local_side=self.current_side)
        if result is not None:
            close_info["close_price"] = result.get("price", 0)
            close_info["order_id"] = result.get("order_id", "")
            self.reset()
            return "CLOSED_WAITING", close_info
        else:
            logger.error(
                "Close position FAILED — keeping local state, will retry next cycle")
            return "CLOSE_FAILED", None

    def reset(self):
        """Clear all position state."""
        self.current_side = None
        self.entries = []
        self.avg_entry = 0.0
        self.total_quantity = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.entry_time = None
        self.high_water_mark_pct = 0.0
        self.profit_lock_active = False

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "current_side": self.current_side,
            "entries": self.entries,
            "avg_entry": self.avg_entry,
            "total_quantity": self.total_quantity,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "high_water_mark_pct": self.high_water_mark_pct,
            "profit_lock_active": self.profit_lock_active,
        }

    def load_from_dict(self, data: dict):
        """Restore state from persistence."""
        self.current_side = data.get("current_side")
        self.entries = data.get("entries", [])
        self.avg_entry = data.get("avg_entry", 0.0)
        self.total_quantity = data.get("total_quantity", 0.0)
        self.sl_price = data.get("sl_price", 0.0)
        self.tp_price = data.get("tp_price", 0.0)

        entry_time_str = data.get("entry_time")
        if entry_time_str:
            self.entry_time = datetime.fromisoformat(entry_time_str)
        else:
            self.entry_time = None

        self.high_water_mark_pct = data.get("high_water_mark_pct", 0.0)
        self.profit_lock_active = data.get("profit_lock_active", False)
