"""HTML message templates for Telegram notifications.

Uses HTML parse mode to avoid MarkdownV2 escaping issues with prices.
"""

import io
from datetime import datetime, timezone

from core.config import TRADING_SL_PCT, TRADING_TP_PCT


def fmt_signal_alert(pred: dict) -> str:
    """Format a model signal alert (LONG or SHORT) with entry/SL/TP prices."""
    signal = pred.get("signal", "?")
    conf = pred.get("confidence", 0)
    prob_l = pred.get("prob_long", 0)
    prob_s = pred.get("prob_short", 0)
    prob_nt = pred.get("prob_no_trade", 0)
    agreement = pred.get("model_agreement", "")
    unanimous = pred.get("unanimous", False)
    time_str = pred.get("time", "?")
    btc_close = pred.get("btc_close", 0)

    emoji = "\U0001f7e2" if signal == "LONG" else "\U0001f534"  # green/red circle
    unan_str = " (unanimous)" if unanimous else ""

    lines = [
        f"{emoji} <b>SIGNAL: {signal}</b>",
        f"Confidence: <b>{conf:.3f}</b>",
        f"Prob: L={prob_l:.3f}  S={prob_s:.3f}  NT={prob_nt:.3f}",
    ]

    # Entry price + SL/TP levels (from config, not hardcoded)
    if btc_close and btc_close > 0:
        if signal == "LONG":
            sl = btc_close * (1 - TRADING_SL_PCT / 100)
            tp = btc_close * (1 + TRADING_TP_PCT / 100)
        else:
            sl = btc_close * (1 + TRADING_SL_PCT / 100)
            tp = btc_close * (1 - TRADING_TP_PCT / 100)
        lines.append("")
        lines.append(f"Entry: <b>${btc_close:,.2f}</b>")
        lines.append(f"\U0001f6d1 SL: ${sl:,.2f} (-{TRADING_SL_PCT:.0f}%)")
        lines.append(f"\U0001f3af TP: ${tp:,.2f} (+{TRADING_TP_PCT:.0f}%)")

    lines.append(f"\nModels: {agreement}{unan_str}")
    lines.append(f"Time: {time_str} UTC")

    return "\n".join(lines)


def fmt_trade_event(trade: dict) -> str:
    """Format a trade event (OPENED, CLOSED, SL/TP hit, etc.)."""
    action = trade.get("action", "?")
    signal = trade.get("signal", "?")
    side = trade.get("side", "?")
    price = trade.get("price", 0)
    conf = trade.get("confidence", 0)
    avg_entry = trade.get("avg_entry", 0)
    sl = trade.get("sl_price", 0)
    tp = trade.get("tp_price", 0)
    ts = trade.get("timestamp", "?")

    emoji_map = {
        "OPENED": "\U0001f4c8",      # chart increasing
        "ADDED": "\u2795",            # plus
        "CLOSED_WAITING": "\U0001f4c9",  # chart decreasing
        "MAX_HOLD": "\u23f0",         # alarm clock
        "MAX_HOLD_24H": "\u23f0",     # alarm clock
        "PROFIT_LOCK": "\U0001f512",  # lock
        "SL_TRIGGERED": "\U0001f6d1", # stop sign
        "TP_TRIGGERED": "\U0001f3af", # target
        "LIQUIDATED": "\U0001f4a5",   # collision
        "SL_TP_FAILED": "\u26a0\ufe0f",  # warning
        "CLOSE_FAILED": "\u274c",     # cross mark
    }
    emoji = emoji_map.get(action, "\U0001f4dd")

    lines = [f"{emoji} <b>TRADE: {action}</b>"]
    lines.append(f"Signal: {signal} | Side: {side}")
    if price:
        lines.append(f"Price: ${price:,.1f}")
    lines.append(f"Confidence: {conf:.3f}")
    if avg_entry:
        lines.append(f"Avg Entry: ${avg_entry:,.1f}")
    if sl and tp:
        lines.append(f"SL: ${sl:,.1f} | TP: ${tp:,.1f}")
    lines.append(f"Time: {ts}")

    # Add PnL for close events
    try:
        pnl_val = float(trade.get("realized_pnl_usdt", ""))
        if pnl_val == pnl_val:  # not NaN
            pnl_emoji = "\U0001f7e2" if pnl_val >= 0 else "\U0001f534"
            pct_val = float(trade.get("realized_pnl_pct", 0) or 0)
            lines.append(f"PnL: {pnl_emoji} ${pnl_val:+.2f} ({pct_val:+.2f}%)")
    except (ValueError, TypeError):
        pass
    try:
        bal_val = float(trade.get("balance_after", ""))
        if bal_val == bal_val:  # not NaN
            lines.append(f"Balance: ${bal_val:,.2f}")
    except (ValueError, TypeError):
        pass

    return "\n".join(lines)


def fmt_hourly_report(predictions_df, btc: dict | None,
                      position: dict | None, safety_stats: dict | None,
                      data_status: dict | None,
                      balance_info: dict | None = None) -> str:
    """Format the full hourly dashboard report."""
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [f"\U0001f4ca <b>Hourly Report — {now_str}</b>\n"]

    # Predictions summary
    if predictions_df is not None and not predictions_df.empty:
        total = len(predictions_df)
        if "signal" in predictions_df.columns:
            nt = (predictions_df["signal"] == "NO_TRADE").sum()
            longs = (predictions_df["signal"] == "LONG").sum()
            shorts = (predictions_df["signal"] == "SHORT").sum()
        else:
            nt, longs, shorts = total, 0, 0

        avg_conf = predictions_df["confidence"].mean() if "confidence" in predictions_df.columns else 0
        unan_count = 0
        if "unanimous" in predictions_df.columns:
            unan_col = predictions_df["unanimous"]
            if unan_col.dtype == bool:
                unan_count = unan_col.sum()
            else:
                unan_count = (unan_col.astype(str).str.lower() == "true").sum()
        unan_pct = (unan_count / total * 100) if total > 0 else 0

        lines.append("<b>Predictions (last hour):</b>")
        lines.append(f"  NT: {nt} | LONG: {longs} | SHORT: {shorts}")
        lines.append(f"  Total: {total} | Avg conf: {avg_conf:.3f} | Unanimous: {unan_pct:.0f}%")
    else:
        lines.append("<b>Predictions:</b> No data")
    lines.append("")

    # BTC price
    if btc:
        close = btc.get("close", 0)
        lines.append(f"<b>BTC:</b> ${close:,.1f}")
    else:
        lines.append("<b>BTC:</b> N/A")
    lines.append("")

    # Balance
    if balance_info and balance_info.get("account_balance") is not None:
        bal = balance_info["account_balance"]
        cum_pnl = balance_info.get("cumulative_pnl_usdt", 0)
        pnl_emoji = "\U0001f7e2" if cum_pnl >= 0 else "\U0001f534"
        # Multi-trade mode: show starting balance context
        if balance_info.get("mode") == "multi_trade":
            start_bal = balance_info.get("starting_balance", 1000.0)
            pnl_pct = (bal - start_bal) / start_bal * 100 if start_bal > 0 else 0
            lines.append(f"<b>Balance:</b> ${bal:,.2f} / ${start_bal:,.0f} | "
                         f"PnL: {pnl_emoji} ${cum_pnl:+,.2f} ({pnl_pct:+.1f}%)")
        else:
            lines.append(f"<b>Balance:</b> ${bal:,.2f} | PnL: {pnl_emoji} ${cum_pnl:+,.2f}")
        lines.append("")

    # Position / Open Trades
    # Check for multi-trade mode via balance_info (has mode field from reader)
    is_multi = balance_info and balance_info.get("mode") == "multi_trade"
    if is_multi:
        open_count = balance_info.get("open_trade_count", 0)
        locked = balance_info.get("locked_margin", 0)
        lines.append(f"<b>Open Trades:</b> {open_count} (${locked:.0f} locked)")
    else:
        lines.append("<b>Position:</b>")
        if position and position.get("current_side"):
            side = position["current_side"]
            avg_entry = position.get("avg_entry", 0)
            sl = position.get("sl_price", 0)
            tp = position.get("tp_price", 0)
            entry_time_str = position.get("entry_time")

            # PnL estimate
            pnl_str = ""
            if btc and avg_entry > 0:
                close = btc.get("close", 0)
                if side == "LONG":
                    pnl_pct = (close - avg_entry) / avg_entry * 100
                else:
                    pnl_pct = (avg_entry - close) / avg_entry * 100
                pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
                pnl_str = f" ({pnl_emoji} {pnl_pct:+.2f}%)"

            hold_str = ""
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    hold_min = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
                    hold_str = f" | Hold: {hold_min:.0f}min"
                except Exception:
                    pass

            profit_lock = position.get("profit_lock_active", False)
            lock_str = " | \U0001f512 Profit lock ACTIVE" if profit_lock else ""

            lines.append(f"  {side} @ ${avg_entry:,.1f}{pnl_str}")
            lines.append(f"  SL=${sl:,.1f} | TP=${tp:,.1f}{hold_str}{lock_str}")
        else:
            lines.append("  FLAT (no position)")
    lines.append("")

    # Safety / WR stats
    if safety_stats:
        trades_7d = safety_stats.get("7d_trades", 0)
        wr_7d = safety_stats.get("7d_wr")
        total_trades = safety_stats.get("total_trades", 0)
        total_wr = safety_stats.get("total_wr", 0)
        paused = safety_stats.get("paused", False)

        lines.append("<b>Safety:</b>")
        wr_str = f"{wr_7d:.1f}%" if wr_7d is not None else "N/A"
        lines.append(f"  7d WR: {wr_str} ({trades_7d} trades) | All-time: {total_wr:.1f}% ({total_trades} trades)")
        if paused:
            lines.append(f"  \u26a0\ufe0f PAUSED: {safety_stats.get('pause_reason', '?')}")
        lines.append("")

    # Data service
    lines.append("<b>Data Service:</b>")
    if data_status:
        state = data_status.get("state", "unknown")
        cycle = data_status.get("cycle", "?")
        state_emoji = "\u2705" if state == "running" else "\u26a0\ufe0f"
        lines.append(f"  {state_emoji} {state} | Cycle: {cycle}")

        last_cycle = data_status.get("last_cycle", {})
        if last_cycle:
            total_time = last_cycle.get("total_time", "?")
            lines.append(f"  Last cycle: {total_time}s")
    else:
        lines.append("  \u274c Status unavailable")

    return "\n".join(lines)


def fmt_position_detail(position: dict | None, btc: dict | None,
                        state: dict | None = None) -> str:
    """Format detailed position view for /position command.

    Supports both single-position and multi-trade mode.
    """
    # Multi-trade mode: show all open trades
    if state and state.get("mode") == "multi_trade":
        return _fmt_multi_trade_positions(state, btc)

    if not position or not position.get("current_side"):
        return "\U0001f4ad <b>Position: FLAT</b>\nNo open position."

    side = position["current_side"]
    avg_entry = position.get("avg_entry", 0)
    qty = position.get("total_quantity", 0)
    sl = position.get("sl_price", 0)
    tp = position.get("tp_price", 0)
    entry_time_str = position.get("entry_time")
    hwm = position.get("high_water_mark_pct", 0)
    profit_lock = position.get("profit_lock_active", False)

    emoji = "\U0001f7e2" if side == "LONG" else "\U0001f534"

    lines = [f"{emoji} <b>Position: {side}</b>\n"]
    lines.append(f"Entry: ${avg_entry:,.1f}")
    lines.append(f"Quantity: {qty:.4f} BTC")
    lines.append(f"SL: ${sl:,.1f} | TP: ${tp:,.1f}")

    if btc and avg_entry > 0:
        close = btc.get("close", 0)
        if side == "LONG":
            pnl_pct = (close - avg_entry) / avg_entry * 100
        else:
            pnl_pct = (avg_entry - close) / avg_entry * 100
        pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
        lines.append(f"Current PnL: {pnl_emoji} {pnl_pct:+.2f}%  (BTC=${close:,.1f})")

    if entry_time_str:
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            hold_min = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
            lines.append(f"Hold time: {hold_min:.0f} min")
            lines.append(f"Entry time: {entry_time_str}")
        except Exception:
            pass

    lines.append(f"High water mark: {hwm:.2f}%")
    if profit_lock:
        lines.append("\U0001f512 Profit lock: <b>ACTIVE</b>")

    return "\n".join(lines)


def _fmt_multi_trade_positions(state: dict, btc: dict | None) -> str:
    """Format multi-trade open positions."""
    open_trades = state.get("open_trades", [])
    sim_bal = state.get("simulated_balance", 0)
    start_bal = state.get("starting_balance", 1000.0)
    margin_per = state.get("margin_per_trade", 10.0)
    leverage = state.get("leverage", 20)
    close = btc.get("close", 0) if btc else 0

    if not open_trades:
        pnl_pct = (sim_bal - start_bal) / start_bal * 100 if start_bal > 0 else 0
        pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
        return (
            "\U0001f4ad <b>Positions: FLAT</b>\n"
            f"No open trades.\n\n"
            f"Balance: <b>${sim_bal:,.2f}</b> / ${start_bal:,.0f}\n"
            f"PnL: {pnl_emoji} {pnl_pct:+.2f}%\n"
            f"Mode: ${margin_per:.0f}/trade x {leverage}x"
        )

    locked = sum(t.get("margin", margin_per) for t in open_trades)
    available = sim_bal - locked
    long_count = sum(1 for t in open_trades if t.get("side") == "LONG")
    short_count = sum(1 for t in open_trades if t.get("side") == "SHORT")

    lines = [f"\U0001f4ca <b>Open Trades ({len(open_trades)})</b>"]
    lines.append(f"LONG: {long_count} | SHORT: {short_count}")
    lines.append(f"Locked: ${locked:.0f} | Available: ${available:.0f}")
    lines.append("")

    # Show each trade
    total_unreal = 0.0
    for t in open_trades:
        side = t.get("side", "?")
        entry = t.get("entry_price", 0)
        sl = t.get("sl_price", 0)
        tp = t.get("tp_price", 0)
        tid = t.get("id", "?")
        emoji = "\U0001f7e2" if side == "LONG" else "\U0001f534"

        pnl_str = ""
        if close > 0 and entry > 0:
            if side == "LONG":
                pnl_pct = (close - entry) / entry * 100
            else:
                pnl_pct = (entry - close) / entry * 100
            notional = t.get("margin", margin_per) * leverage
            pnl_usdt = pnl_pct / 100 * notional
            total_unreal += pnl_usdt
            p_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
            pnl_str = f" | {p_emoji} {pnl_pct:+.1f}% (${pnl_usdt:+.1f})"

        # Hold time
        hold_str = ""
        entry_time_str = t.get("entry_time")
        if entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                hold_min = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
                hold_str = f" | {hold_min:.0f}m"
            except Exception:
                pass

        lines.append(
            f"{emoji} {tid} {side} ${entry:,.0f} "
            f"SL=${sl:,.0f} TP=${tp:,.0f}"
            f"{hold_str}{pnl_str}")

    lines.append("")
    unreal_emoji = "\U0001f7e2" if total_unreal >= 0 else "\U0001f534"
    lines.append(f"Unrealized: {unreal_emoji} ${total_unreal:+,.2f}")

    return "\n".join(lines)


def fmt_trades_list(trades_df) -> str:
    """Format last N trades for /trades command."""
    if trades_df is None or trades_df.empty:
        return "\U0001f4dd <b>Recent Trades:</b> None"

    lines = [f"\U0001f4dd <b>Recent Trades ({len(trades_df)}):</b>\n"]

    emoji_map = {
        "OPENED": "\U0001f4c8",
        "ADDED": "\u2795",
        "CLOSED_WAITING": "\U0001f4c9",
        "MAX_HOLD": "\u23f0",
        "MAX_HOLD_24H": "\u23f0",
        "PROFIT_LOCK": "\U0001f512",
        "SL_TRIGGERED": "\U0001f6d1",
        "TP_TRIGGERED": "\U0001f3af",
        "LIQUIDATED": "\U0001f4a5",
        "SL_TP_FAILED": "\u26a0\ufe0f",
        "CLOSE_FAILED": "\u274c",
        "SKIPPED": "\u23ed\ufe0f",
    }

    for _, row in trades_df.iterrows():
        action = str(row.get("action", "?"))
        signal = str(row.get("signal", "?"))
        price = row.get("price", 0)
        conf = row.get("confidence", 0)
        ts = str(row.get("timestamp", "?"))
        emoji = emoji_map.get(action, "\U0001f4dd")

        # Shorten timestamp to HH:MM
        try:
            dt = datetime.fromisoformat(str(ts))
            ts_short = dt.strftime("%m-%d %H:%M")
        except Exception:
            ts_short = ts[:16] if len(ts) > 16 else ts

        price_str = f"${float(price):,.1f}" if price else ""
        pnl_str = ""
        try:
            pnl_val = float(row.get("realized_pnl_usdt", ""))
            if pnl_val == pnl_val:  # not NaN
                pnl_emoji = "\U0001f7e2" if pnl_val >= 0 else "\U0001f534"
                pnl_str = f" {pnl_emoji}${pnl_val:+.1f}"
        except (ValueError, TypeError):
            pass
        lines.append(f"{emoji} {ts_short} | {action} {signal} {price_str} ({float(conf):.2f}){pnl_str}")

    return "\n".join(lines)


def fmt_health(data_status: dict | None, state: dict | None) -> str:
    """Format system health check for /health command."""
    lines = ["\U0001f3e5 <b>System Health</b>\n"]

    # Data service
    lines.append("<b>Data Service:</b>")
    if data_status:
        ds_state = data_status.get("state", "unknown")
        cycle = data_status.get("cycle", "?")
        ts = data_status.get("timestamp", "?")
        state_emoji = "\u2705" if ds_state == "running" else "\u26a0\ufe0f"
        lines.append(f"  {state_emoji} State: {ds_state}")
        lines.append(f"  Cycle: {cycle}")
        lines.append(f"  Last update: {ts}")

        if ds_state == "error":
            err = data_status.get("error", "?")
            consec = data_status.get("consecutive_errors", 0)
            lines.append(f"  \u274c Error: {err}")
            lines.append(f"  Consecutive errors: {consec}")
    else:
        lines.append("  \u274c status.json not found")
    lines.append("")

    # Trading bot
    lines.append("<b>Trading Bot:</b>")
    if state:
        last_updated = state.get("last_updated", "?")
        lines.append(f"  \u2705 State file found")
        lines.append(f"  Last updated: {last_updated}")

        if state.get("mode") == "multi_trade":
            open_trades = state.get("open_trades", [])
            sim_bal = state.get("simulated_balance", 0)
            lines.append(f"  Mode: Multi-trade (paper)")
            lines.append(f"  Open trades: {len(open_trades)}")
            lines.append(f"  Balance: ${sim_bal:,.2f}")
        else:
            pos = state.get("position", {})
            side = pos.get("current_side") if pos else None
            if side:
                lines.append(f"  Position: {side} @ ${pos.get('avg_entry', 0):,.1f}")
            else:
                lines.append("  Position: FLAT")

        safety = state.get("trade_history", {})
        if isinstance(safety, dict):
            paused = safety.get("paused", False)
            if paused:
                lines.append(f"  \u26a0\ufe0f SAFETY PAUSED: {safety.get('pause_reason', '?')}")
            else:
                lines.append("  Safety: OK")
    else:
        lines.append("  \u274c state.json not found")

    return "\n".join(lines)


def fmt_status_oneliner(pred: dict | None, btc: dict | None,
                        position: dict | None,
                        state: dict | None = None) -> str:
    """Format multi-line status for /status command (HTML)."""
    lines = ["\U0001f4ca <b>Status</b>\n"]

    # Multi-trade mode
    if state and state.get("mode") == "multi_trade":
        open_trades = state.get("open_trades", [])
        sim_bal = state.get("simulated_balance", 0)
        start_bal = state.get("starting_balance", 1000.0)
        n_open = len(open_trades)
        n_long = sum(1 for t in open_trades if t.get("side") == "LONG")
        n_short = sum(1 for t in open_trades if t.get("side") == "SHORT")
        pnl_pct = (sim_bal - start_bal) / start_bal * 100 if start_bal > 0 else 0
        pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"

        btc_str = f"${btc.get('close', 0):,.1f}" if btc else "N/A"
        if n_open > 0:
            lines.append(f"Trades: <b>{n_open}</b> (L:{n_long} S:{n_short}) | BTC: {btc_str}")
        else:
            lines.append(f"Trades: <b>FLAT</b> | BTC: {btc_str}")
        lines.append(f"Balance: ${sim_bal:,.0f} | {pnl_emoji} {pnl_pct:+.1f}%")
    else:
        # Single-position mode
        pos_str = "FLAT"
        if position and position.get("current_side"):
            side = position["current_side"]
            avg_entry = position.get("avg_entry", 0)
            pnl_str = ""
            if btc and avg_entry > 0:
                close = btc.get("close", 0)
                if side == "LONG":
                    pnl_pct = (close - avg_entry) / avg_entry * 100
                else:
                    pnl_pct = (avg_entry - close) / avg_entry * 100
                pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
                pnl_str = f" {pnl_emoji}{pnl_pct:+.2f}%"
            pos_str = f"{side}{pnl_str}"

        btc_str = f"${btc.get('close', 0):,.1f}" if btc else "N/A"
        lines.append(f"Pos: <b>{pos_str}</b> | BTC: {btc_str}")

    # Probabilities — highlight highest
    if pred:
        prob_nt = pred.get("prob_no_trade", 0)
        prob_l = pred.get("prob_long", 0)
        prob_s = pred.get("prob_short", 0)

        probs = {"NT": prob_nt, "L": prob_l, "S": prob_s}
        max_key = max(probs, key=probs.get)

        prob_parts = []
        for key, val in probs.items():
            if key == max_key:
                prob_parts.append(f"<b>{key}: {val:.3f}</b>")
            else:
                prob_parts.append(f"{key}: {val:.3f}")

        signal = pred.get("signal", "NO_TRADE")
        age = pred.get("age_seconds", 0)
        age_min = age // 60
        agreement = pred.get("model_agreement", "")
        unanimous = pred.get("unanimous", False)
        unan_str = " (unanimous)" if unanimous else ""

        lines.append(f"\n{' | '.join(prob_parts)}")
        lines.append(f"Signal: <b>{signal}</b> | Models: {agreement}{unan_str}")
        lines.append(f"Age: {age_min}m")
    else:
        lines.append("\nPrediction: N/A")

    return "\n".join(lines)


def fmt_balance(account: dict | None, position: dict | None,
                btc: dict | None, state: dict | None = None) -> str:
    """Format wallet balance for /balance command.

    Supports both single-position and multi-trade mode.
    """
    if not account or account.get("account_balance") is None:
        return "\U0001f4b0 <b>Balance:</b> Not available yet\n(Updates after first trade close)"

    # Multi-trade mode
    if account.get("mode") == "multi_trade":
        return _fmt_multi_trade_balance(account, btc, state)

    bal = account["account_balance"]
    cum_pnl = account.get("cumulative_pnl_usdt", 0)

    lines = ["\U0001f4b0 <b>Account Balance</b>\n"]
    lines.append(f"Wallet: <b>${bal:,.2f}</b> USDT")

    pnl_emoji = "\U0001f7e2" if cum_pnl >= 0 else "\U0001f534"
    lines.append(f"Cumulative PnL: {pnl_emoji} <b>${cum_pnl:+,.2f}</b>")

    # Unrealized PnL estimate from position
    if position and position.get("current_side") and btc:
        side = position["current_side"]
        avg_entry = position.get("avg_entry", 0)
        qty = position.get("total_quantity", 0)
        close = btc.get("close", 0)
        if avg_entry > 0 and close > 0 and qty > 0:
            if side == "LONG":
                unreal = (close - avg_entry) * qty
            else:
                unreal = (avg_entry - close) * qty
            unreal_emoji = "\U0001f7e2" if unreal >= 0 else "\U0001f534"
            lines.append(f"Unrealized PnL: {unreal_emoji} ${unreal:+,.2f}")
            lines.append(f"Est. Total: ${bal + unreal:,.2f}")

    lines.append(f"\nLast updated: {account.get('last_updated', 'N/A')}")
    return "\n".join(lines)


def _fmt_multi_trade_balance(account: dict, btc: dict | None,
                             state: dict | None) -> str:
    """Format balance for multi-trade paper trading."""
    sim_bal = account.get("simulated_balance", 0)
    start_bal = account.get("starting_balance", 1000.0)
    cum_pnl = account.get("cumulative_pnl_usdt", 0)
    margin_per = account.get("margin_per_trade", 10.0)
    leverage = account.get("leverage", 20)
    open_count = account.get("open_trade_count", 0)
    locked = account.get("locked_margin", 0)
    available = sim_bal - locked

    pnl_pct = (sim_bal - start_bal) / start_bal * 100 if start_bal > 0 else 0

    lines = ["\U0001f4b0 <b>Paper Trading Balance</b>\n"]
    lines.append(f"Balance: <b>${sim_bal:,.2f}</b> USDT")
    lines.append(f"Starting: ${start_bal:,.0f}")

    pnl_emoji = "\U0001f7e2" if cum_pnl >= 0 else "\U0001f534"
    lines.append(f"Realized PnL: {pnl_emoji} <b>${cum_pnl:+,.2f}</b> ({pnl_pct:+.1f}%)")

    lines.append("")
    lines.append(f"Open trades: {open_count}")
    lines.append(f"Locked margin: ${locked:.0f}")
    lines.append(f"Available: ${available:.0f}")

    # Unrealized PnL from open trades
    if state and state.get("open_trades") and btc:
        close = btc.get("close", 0)
        if close > 0:
            total_unreal = 0.0
            for t in state.get("open_trades", []):
                entry = t.get("entry_price", 0)
                if entry > 0:
                    side = t.get("side", "")
                    if side == "LONG":
                        p = (close - entry) / entry * 100
                    else:
                        p = (entry - close) / entry * 100
                    notional = t.get("margin", margin_per) * leverage
                    total_unreal += p / 100 * notional
            unreal_emoji = "\U0001f7e2" if total_unreal >= 0 else "\U0001f534"
            lines.append(f"Unrealized: {unreal_emoji} ${total_unreal:+,.2f}")
            lines.append(f"Est. Total: ${sim_bal + total_unreal:,.2f}")

    lines.append(f"\nMode: ${margin_per:.0f}/trade x {leverage}x leverage")
    lines.append(f"SL: -{TRADING_SL_PCT:.0f}% ($-{margin_per * leverage * TRADING_SL_PCT / 100:.0f}) | "
                 f"TP: +{TRADING_TP_PCT:.0f}% ($+{margin_per * leverage * TRADING_TP_PCT / 100:.0f})")
    lines.append(f"\nLast updated: {account.get('last_updated', 'N/A')}")
    return "\n".join(lines)


def fmt_pnl_summary(summary: dict | None) -> str:
    """Format PnL summary for /pnl command."""
    if not summary:
        return "\U0001f4b5 <b>PnL Summary:</b> No trade data"

    lines = ["\U0001f4b5 <b>PnL Summary</b>\n"]

    for label, key in [("Today", "today"), ("7 Days", "7d"),
                       ("30 Days", "30d"), ("All Time", "all_time")]:
        s = summary.get(key, {})
        pnl = s.get("pnl_usdt", 0)
        wins = s.get("wins", 0)
        losses = s.get("losses", 0)
        count = s.get("count", 0)
        wr = s.get("wr", 0)
        emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
        lines.append(f"<b>{label}:</b> {emoji} ${pnl:+,.2f}")
        if count > 0:
            lines.append(f"  W:{wins} L:{losses} ({count} trades, {wr:.0f}% WR)")
        lines.append("")

    return "\n".join(lines)


def _get_closed_trades(trades_df, n_days: int = None):
    """Extract closed trades from DataFrame, optionally filtered to last N days.

    Returns sorted DataFrame with cumulative PnL computed, or None.
    """
    import pandas as pd

    if trades_df is None or trades_df.empty:
        return None

    close_actions = {"CLOSED_WAITING", "SL_TP_TRIGGERED", "MAX_HOLD_24H",
                     "PROFIT_LOCK", "MAX_HOLD", "SL_TRIGGERED", "TP_TRIGGERED",
                     "LIQUIDATED"}
    if "action" not in trades_df.columns:
        return None

    closes = trades_df[trades_df["action"].isin(close_actions)].copy()
    if "realized_pnl_usdt" not in closes.columns or closes.empty:
        return None

    closes["realized_pnl_usdt"] = pd.to_numeric(
        closes["realized_pnl_usdt"], errors="coerce")
    closes = closes.dropna(subset=["realized_pnl_usdt"])
    if closes.empty:
        return None

    if "timestamp" in closes.columns:
        closes["timestamp"] = pd.to_datetime(closes["timestamp"], errors="coerce")
        closes = closes.dropna(subset=["timestamp"])
        closes = closes.sort_values("timestamp")

        if n_days is not None:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
            cutoff = cutoff.replace(tzinfo=None)
            closes = closes[closes["timestamp"] >= cutoff]

    if closes.empty:
        return None

    closes["cum_pnl"] = closes["realized_pnl_usdt"].cumsum()
    closes["win"] = closes["realized_pnl_usdt"] > 0
    return closes


def generate_equity_chart(trades_df, n_days: int,
                          period_label: str) -> tuple[io.BytesIO | None, str]:
    """Generate equity curve chart as PNG BytesIO with same neon purple theme.

    Returns (chart_buf, caption_text). chart_buf is None if no data.
    """
    closes = _get_closed_trades(trades_df, n_days=n_days)
    if closes is None or closes.empty:
        return None, f"\U0001f4c8 {period_label}: No closed trades"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    times = closes["timestamp"].tolist()
    cum_pnl = closes["cum_pnl"].tolist()
    wins = closes["win"].tolist()
    pnls = closes["realized_pnl_usdt"].tolist()

    n = len(cum_pnl)
    total_pnl = cum_pnl[-1] if cum_pnl else 0
    win_count = sum(wins)
    loss_count = n - win_count
    wr = (win_count / n * 100) if n > 0 else 0

    # --- Neon purple theme (matching probability chart) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d0221")
    ax.set_facecolor("#0d0221")

    # Equity line
    ax.plot(times, cum_pnl, color="#00ffcc", linewidth=2, alpha=0.9,
            zorder=2, label="Cumulative PnL")

    # Fill area under/above zero
    ax.fill_between(times, cum_pnl, 0,
                     where=[p >= 0 for p in cum_pnl],
                     color="#00ffcc", alpha=0.1, interpolate=True)
    ax.fill_between(times, cum_pnl, 0,
                     where=[p < 0 for p in cum_pnl],
                     color="#ff00ff", alpha=0.1, interpolate=True)

    # Win/Loss markers
    win_times = [t for t, w in zip(times, wins) if w]
    win_pnls = [p for p, w in zip(cum_pnl, wins) if w]
    loss_times = [t for t, w in zip(times, wins) if not w]
    loss_pnls = [p for p, w in zip(cum_pnl, wins) if not w]

    if win_times:
        ax.scatter(win_times, win_pnls, color="#00ff88", s=50,
                   zorder=3, label=f"Win ({win_count})", edgecolors="white",
                   linewidth=0.5)
    if loss_times:
        ax.scatter(loss_times, loss_pnls, color="#ff3366", s=50,
                   zorder=3, label=f"Loss ({loss_count})", edgecolors="white",
                   linewidth=0.5, marker="X")

    # Zero line
    ax.axhline(y=0, color="#8b5cf6", linewidth=1, linestyle="--", alpha=0.5)

    # Axes styling
    ax.set_xlabel("Time (UTC)", fontsize=12, color="white")
    ax.set_ylabel("Cumulative PnL ($)", fontsize=12, color="white")
    ax.set_title(f"Equity Curve \u2014 {period_label} ({n} trades)",
                 fontsize=14, fontweight="bold", color="white")

    ax.tick_params(colors="#e0e0e0", which="both")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#e0e0e0")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Dynamic date formatting
    if n_days is not None and n_days <= 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    elif n_days is not None and n_days <= 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(
            interval=max(1, n_days // 10 if n_days else 3)))
    fig.autofmt_xdate(rotation=45)

    ax.legend(loc="upper left", fontsize=10, facecolor="#1a0a2e",
              edgecolor="#8b5cf6", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#8b5cf6")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    # Caption
    pnl_sign = "+" if total_pnl >= 0 else ""
    caption = (f"{period_label}: {pnl_sign}${total_pnl:.2f} | "
               f"WR: {wr:.0f}% ({win_count}W/{loss_count}L) | "
               f"{n} trades")

    return buf, caption


def compute_history_summary(signals_df) -> dict:
    """Compute summary stats from live signal history DataFrame.

    Only counts executed + closed trades for WR/PF stats.
    """
    import pandas as pd

    total_signals = len(signals_df)
    executed = signals_df[signals_df["executed"] == True].copy()
    executed_count = len(executed)

    # Closed = executed with a real close outcome (not "Open" or "Not Traded")
    closed = executed[~executed["outcome"].isin(["Open", "Not Traded"])].copy()
    closed_count = len(closed)

    closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    wins = int((closed["pnl_pct"] > 0).sum()) if closed_count > 0 else 0
    losses = closed_count - wins
    wr = (wins / closed_count * 100) if closed_count > 0 else 0

    # Profit factor
    if closed_count > 0:
        gains = closed["pnl_pct"].fillna(0)
        gross_wins = gains[gains > 0].sum()
        gross_losses = abs(gains[gains < 0].sum())
        pf = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")
    else:
        pf = 0

    # Total PnL in USDT
    closed["pnl_usdt"] = pd.to_numeric(closed["pnl_usdt"], errors="coerce")
    total_pnl_usdt = float(closed["pnl_usdt"].sum()) if closed_count > 0 else 0

    # Per-direction stats (executed + closed only)
    signal_col = "raw_signal" if "raw_signal" in closed.columns else "signal"
    longs = closed[closed[signal_col] == "LONG"]
    shorts = closed[closed[signal_col] == "SHORT"]
    long_count = len(longs)
    short_count = len(shorts)
    long_wins = int((longs["pnl_pct"] > 0).sum()) if long_count > 0 else 0
    short_wins = int((shorts["pnl_pct"] > 0).sum()) if short_count > 0 else 0
    long_wr = (long_wins / long_count * 100) if long_count > 0 else 0
    short_wr = (short_wins / short_count * 100) if short_count > 0 else 0

    return {
        "total_signals": total_signals, "executed": executed_count,
        "closed": closed_count,
        "wins": wins, "losses": losses,
        "wr": wr, "pf": pf, "total_pnl_usdt": total_pnl_usdt,
        "long_count": long_count, "long_wr": long_wr,
        "short_count": short_count, "short_wr": short_wr,
    }


def _format_hold_minutes(minutes) -> str:
    """Convert minutes to human-readable hold time."""
    try:
        minutes = int(float(minutes))
    except (ValueError, TypeError):
        return "?"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins:02d}m"


def fmt_history_page(signals_df, page: int, per_page: int,
                     summary: dict) -> str:
    """Format one page of signal history with summary header."""
    total_rows = len(signals_df)
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))

    start = page * per_page
    end = min(start + per_page, total_rows)
    page_df = signals_df.iloc[start:end]

    lines = [f"\U0001f4dc <b>Signal History</b> (page {page + 1}/{total_pages})\n"]

    # Summary block
    s = summary
    pf_str = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "\u221e"
    pnl_usdt = s.get("total_pnl_usdt", 0)
    pnl_sign = "+" if pnl_usdt >= 0 else ""
    lines.append("\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550 SUMMARY \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    lines.append(
        f"\u2551 \U0001f3af {s['total_signals']} signals ({s['executed']} traded)")
    lines.append(
        f"\u2551 \u2705 WR: {s['wr']:.1f}% ({s['wins']}W / {s['losses']}L)")
    lines.append(f"\u2551 \U0001f4ca PF: {pf_str}")
    lines.append(f"\u2551 \U0001f4b0 PnL: {pnl_sign}${pnl_usdt:.2f}")
    lines.append(
        f"\u2551 \U0001f4c8 LONG: {s['long_count']} ({s['long_wr']:.0f}% WR)")
    lines.append(
        f"\u2551 \U0001f4c9 SHORT: {s['short_count']} ({s['short_wr']:.0f}% WR)")
    lines.append("\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")
    lines.append("")

    signal_col = "raw_signal" if "raw_signal" in page_df.columns else "signal"

    # Outcome display mapping
    outcome_map = {
        "SL_TP_TRIGGERED": ("\U0001f3af", "SL/TP Hit"),
        "SL_TRIGGERED": ("\U0001f6d1", "SL Hit"),
        "TP_TRIGGERED": ("\U0001f3af", "TP Hit"),
        "LIQUIDATED": ("\U0001f4a5", "Liquidated"),
        "CLOSED_WAITING": ("\U0001f504", "Opposite Signal"),
        "MAX_HOLD_24H": ("\u23f0", "Max Hold"),
        "MAX_HOLD": ("\u23f0", "Max Hold"),
        "PROFIT_LOCK": ("\U0001f512", "Profit Lock"),
        "Open": ("\U0001f7e1", "Open"),
        "Not Traded": ("\u26aa", "Not Traded"),
    }

    for i, (_, row) in enumerate(page_df.iterrows()):
        rank = start + i + 1
        signal = str(row.get(signal_col, "?"))
        sig_emoji = "\U0001f7e2" if signal == "LONG" else "\U0001f534"

        try:
            ts = row["time"]
            if hasattr(ts, "strftime"):
                time_str = ts.strftime("%m-%d %H:%M")
            else:
                time_str = str(ts)[:16]
        except Exception:
            time_str = "?"

        conf = float(row.get("confidence", 0))
        executed = row.get("executed", False)
        exec_emoji = "\u2705" if executed else "\u274c"
        outcome = str(row.get("outcome", "Not Traded"))

        # Line 1: signal + time
        lines.append(
            f"#{rank}  {sig_emoji} {signal}   {time_str}")

        # Line 2: confidence + hold time (if traded) + execution status
        if executed and outcome != "Not Traded":
            hold_min = row.get("hold_minutes")
            try:
                hold_str = _format_hold_minutes(hold_min)
            except (ValueError, TypeError):
                hold_str = "?"
            lines.append(
                f"    \u26a1 {conf:.3f} \u2502 \u23f1 {hold_str} \u2502 \U0001f3e6 {exec_emoji}")
        else:
            lines.append(
                f"    \u26a1 {conf:.3f} \u2502 \U0001f3e6 {exec_emoji}")

        # Line 3: outcome + PnL
        out_emoji, out_label = outcome_map.get(outcome, ("\u2753", outcome))
        try:
            pnl = float(row.get("pnl_pct", 0) or 0)
            if outcome not in ("Not Traded", "Open") and pnl == pnl:
                lines.append(f"    {out_emoji} {out_label}  {pnl:+.2f}%")
            else:
                lines.append(f"    {out_emoji} {out_label}")
        except (ValueError, TypeError):
            lines.append(f"    {out_emoji} {out_label}")
        lines.append("")

    return "\n".join(lines)


def generate_probability_chart(predictions_df) -> io.BytesIO | None:
    """Generate probability timeline chart as PNG BytesIO.

    X = actual timestamps (hourly resampled), Y = probability (0-1),
    3 lines (LONG/SHORT/NO_TRADE). Dark neon purple theme.
    Returns None if insufficient data.
    """
    if predictions_df is None or predictions_df.empty:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    required = {"time", "prob_no_trade", "prob_long", "prob_short"}
    if not required.issubset(predictions_df.columns):
        return None

    df = predictions_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Resample to hourly averages for readability
    hourly = df[["prob_long", "prob_short", "prob_no_trade"]].resample("1h").mean().dropna()
    if hourly.empty:
        return None

    hours_span = (hourly.index[-1] - hourly.index[0]).total_seconds() / 3600
    show_markers = len(hourly) <= 72

    # --- Neon purple + cyan + pink theme ---
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d0221")
    ax.set_facecolor("#0d0221")

    marker_long = dict(marker="o", markersize=3) if show_markers else {}
    marker_short = dict(marker="s", markersize=3) if show_markers else {}

    ax.plot(hourly.index, hourly["prob_long"], color="#00ffcc",
            linewidth=2, label="LONG", alpha=0.9, **marker_long)
    ax.plot(hourly.index, hourly["prob_short"], color="#ff00ff",
            linewidth=2, label="SHORT", alpha=0.9, **marker_short)
    ax.plot(hourly.index, hourly["prob_no_trade"], color="#8b5cf6",
            linewidth=1.5, linestyle="--", alpha=0.6, label="NO_TRADE")

    # Axes styling
    ax.set_xlabel("Time (UTC)", fontsize=12, color="white")
    ax.set_ylabel("Avg Probability", fontsize=12, color="white")

    if hours_span <= 48:
        title = f"Probability Timeline ({hours_span:.0f}h)"
    else:
        title = f"Probability Timeline ({hours_span / 24:.0f}d)"
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")

    ax.set_ylim(0, 1.0)
    ax.tick_params(colors="#e0e0e0", which="both")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#e0e0e0")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Dynamic date formatting based on data span
    if hours_span <= 48:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    elif hours_span <= 168:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(hours_span / 24 / 10))))
    fig.autofmt_xdate(rotation=45)

    ax.legend(loc="upper right", fontsize=10, facecolor="#1a0a2e",
              edgecolor="#8b5cf6", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#8b5cf6")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf
