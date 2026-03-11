"""Multi-trade manager — simulates independent concurrent trades for paper trading.

Each signal opens a separate $margin_per_trade trade at $leverage leverage.
Each trade has independent SL/TP/max_hold tracking. No adding to positions.
Simulated balance starts at $starting_balance with margin locking per trade.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MultiTradeManager:
    """Manages multiple concurrent simulated trades.

    Each LONG/SHORT signal opens a new independent trade:
        - Margin: $10 per trade (configurable)
        - Leverage: 20x (configurable)
        - Notional: $10 * 20 = $200 exposure per trade
        - SL: -2% of entry → lose $4 from $10 margin
        - TP: +4% of entry → gain $8 from $10 margin

    Balance starts at $1000. Each open trade locks $margin_per_trade.
    Available margin = balance - locked_margin.
    """

    def __init__(self, sl_pct: float = 2.0, tp_pct: float = 4.0,
                 max_hold_seconds: int = 86400,
                 margin_per_trade: float = 10.0, leverage: int = 20,
                 starting_balance: float = 1000.0,
                 profit_lock_trigger: float = 3.5,
                 profit_lock_floor: float = 3.0):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.max_hold_seconds = max_hold_seconds
        self.margin_per_trade = margin_per_trade
        self.leverage = leverage
        self.starting_balance = starting_balance
        self.profit_lock_trigger = profit_lock_trigger
        self.profit_lock_floor = profit_lock_floor

        # State
        self.open_trades: list[dict] = []
        self.simulated_balance: float = starting_balance
        self.cumulative_pnl_usdt: float = 0.0
        self._next_id: int = 1

    @property
    def locked_margin(self) -> float:
        """Total margin locked in open trades."""
        return sum(t["margin"] for t in self.open_trades)

    @property
    def available_margin(self) -> float:
        """Margin available for new trades."""
        return self.simulated_balance - self.locked_margin

    @property
    def total_pnl_pct(self) -> float:
        """Cumulative PnL as percentage of starting balance."""
        if self.starting_balance <= 0:
            return 0.0
        return (self.simulated_balance - self.starting_balance) / self.starting_balance * 100

    def calculate_sl_tp(self, entry_price: float, side: str) -> tuple[float, float]:
        """Calculate SL and TP prices from entry.

        LONG:  SL = entry * (1 - sl_pct/100), TP = entry * (1 + tp_pct/100)
        SHORT: SL = entry * (1 + sl_pct/100), TP = entry * (1 - tp_pct/100)
        """
        if side == "LONG":
            sl = entry_price * (1 - self.sl_pct / 100)
            tp = entry_price * (1 + self.tp_pct / 100)
        else:
            sl = entry_price * (1 + self.sl_pct / 100)
            tp = entry_price * (1 - self.tp_pct / 100)
        return round(sl, 1), round(tp, 1)

    def open_trade(self, side: str, entry_price: float) -> dict | None:
        """Open a new independent trade.

        Returns trade dict if opened, None if insufficient margin.
        """
        if side not in ("LONG", "SHORT"):
            return None

        if self.available_margin < self.margin_per_trade:
            logger.warning(
                f"Insufficient margin: available=${self.available_margin:.2f}, "
                f"required=${self.margin_per_trade:.2f}")
            return None

        notional = self.margin_per_trade * self.leverage
        quantity = notional / entry_price
        sl_price, tp_price = self.calculate_sl_tp(entry_price, side)

        trade = {
            "id": f"T{self._next_id:04d}",
            "side": side,
            "entry_price": round(entry_price, 1),
            "quantity": round(quantity, 6),
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "margin": self.margin_per_trade,
            "high_water_mark_pct": 0.0,
            "profit_lock_active": False,
        }
        self._next_id += 1
        self.open_trades.append(trade)

        logger.info(
            f"OPENED {trade['id']} {side} @ ${entry_price:,.1f} | "
            f"Qty={quantity:.6f} BTC | SL=${sl_price:,.1f} TP=${tp_price:,.1f} | "
            f"Margin=${self.margin_per_trade} | "
            f"Open trades: {len(self.open_trades)} | "
            f"Available: ${self.available_margin:.2f}")
        return trade

    def check_exits(self, current_price: float) -> list[dict]:
        """Check all open trades for exit conditions.

        Checks SL, TP, max_hold, and profit_lock for each trade.
        Returns list of close_info dicts for trades that exited.
        """
        exits = []
        remaining = []

        for trade in self.open_trades:
            reason = self._check_single_exit(trade, current_price)
            if reason:
                close_info = self._close_trade(trade, current_price, reason)
                exits.append(close_info)
            else:
                remaining.append(trade)

        self.open_trades = remaining
        return exits

    def _check_single_exit(self, trade: dict, current_price: float) -> str | None:
        """Check a single trade for exit conditions. Returns reason or None."""
        side = trade["side"]

        # SL check
        if side == "LONG" and current_price <= trade["sl_price"]:
            return "SL_TRIGGERED"
        if side == "SHORT" and current_price >= trade["sl_price"]:
            return "SL_TRIGGERED"

        # TP check
        if side == "LONG" and current_price >= trade["tp_price"]:
            return "TP_TRIGGERED"
        if side == "SHORT" and current_price <= trade["tp_price"]:
            return "TP_TRIGGERED"

        # Max hold check (24h)
        entry_time = datetime.fromisoformat(trade["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds()
        if elapsed >= self.max_hold_seconds:
            return "MAX_HOLD_24H"

        # Profit lock check
        if side == "LONG":
            pnl_pct = (current_price - trade["entry_price"]) / trade["entry_price"] * 100
        else:
            pnl_pct = (trade["entry_price"] - current_price) / trade["entry_price"] * 100

        trade["high_water_mark_pct"] = max(
            trade.get("high_water_mark_pct", 0), pnl_pct)

        if trade["high_water_mark_pct"] >= self.profit_lock_trigger:
            trade["profit_lock_active"] = True

        if trade.get("profit_lock_active") and pnl_pct <= self.profit_lock_floor:
            return "PROFIT_LOCK"

        return None

    def _close_trade(self, trade: dict, close_price: float, reason: str) -> dict:
        """Close a trade and update balance.

        Liquidation guard: loss is capped at the trade's margin (simulates
        exchange liquidation at -100% of margin / -5% at 20x leverage).
        """
        side = trade["side"]
        if side == "LONG":
            pnl_pct = (close_price - trade["entry_price"]) / trade["entry_price"] * 100
        else:
            pnl_pct = (trade["entry_price"] - close_price) / trade["entry_price"] * 100

        notional = trade["margin"] * self.leverage
        pnl_usdt = pnl_pct / 100 * notional

        # Liquidation guard: max loss is the margin (100% of margin)
        if pnl_usdt < -trade["margin"]:
            pnl_usdt = -trade["margin"]
            pnl_pct = -100.0 / self.leverage  # e.g., -5% at 20x
            reason = "LIQUIDATED"

        win = pnl_usdt > 0

        self.simulated_balance += pnl_usdt
        self.cumulative_pnl_usdt += pnl_usdt

        logger.info(
            f"CLOSED {trade['id']} {side} | Reason: {reason} | "
            f"Entry=${trade['entry_price']:,.1f} Close=${close_price:,.1f} | "
            f"PnL: {pnl_pct:+.2f}% (${pnl_usdt:+.2f}) {'WIN' if win else 'LOSS'} | "
            f"Balance: ${self.simulated_balance:.2f}")

        return {
            "trade_id": trade["id"],
            "side": side,
            "entry_price": trade["entry_price"],
            "close_price": round(close_price, 1),
            "quantity": trade["quantity"],
            "sl_price": trade["sl_price"],
            "tp_price": trade["tp_price"],
            "margin": trade["margin"],
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usdt": round(pnl_usdt, 4),
            "win": win,
            "reason": reason,
            "entry_time": trade["entry_time"],
            "balance_after": round(self.simulated_balance, 2),
        }

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Total unrealized PnL across all open trades (capped at margin per trade)."""
        total = 0.0
        for trade in self.open_trades:
            if trade["side"] == "LONG":
                pnl_pct = (current_price - trade["entry_price"]) / trade["entry_price"] * 100
            else:
                pnl_pct = (trade["entry_price"] - current_price) / trade["entry_price"] * 100
            notional = trade["margin"] * self.leverage
            pnl_usdt = pnl_pct / 100 * notional
            # Cap loss at margin (liquidation)
            if pnl_usdt < -trade["margin"]:
                pnl_usdt = -trade["margin"]
            total += pnl_usdt
        return total

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "open_trades": self.open_trades,
            "simulated_balance": round(self.simulated_balance, 4),
            "starting_balance": self.starting_balance,
            "cumulative_pnl_usdt": round(self.cumulative_pnl_usdt, 4),
            "_next_id": self._next_id,
        }

    def load_from_dict(self, data: dict):
        """Restore state from persistence."""
        self.open_trades = data.get("open_trades", [])
        self.simulated_balance = data.get("simulated_balance", self.starting_balance)
        self.starting_balance = data.get("starting_balance", 1000.0)
        self.cumulative_pnl_usdt = data.get("cumulative_pnl_usdt", 0.0)

        # Robust _next_id: parse max existing trade ID to avoid collisions
        if "_next_id" in data:
            self._next_id = data["_next_id"]
        elif self.open_trades:
            max_id = max(
                int(t["id"][1:]) for t in self.open_trades
                if t.get("id", "").startswith("T") and t["id"][1:].isdigit()
            )
            self._next_id = max_id + 1
        else:
            self._next_id = 1

        logger.info(
            f"Multi-trade state loaded: {len(self.open_trades)} open trades, "
            f"balance=${self.simulated_balance:.2f}, "
            f"cumPnL=${self.cumulative_pnl_usdt:+.2f}")
