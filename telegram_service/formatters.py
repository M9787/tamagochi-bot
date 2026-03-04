"""HTML message templates for Telegram notifications.

All functions return (text, parse_mode) tuples.
Uses HTML parse mode to avoid MarkdownV2 escaping issues with prices.
"""

from datetime import datetime, timezone


def fmt_signal_alert(pred: dict) -> str:
    """Format a model signal alert (LONG or SHORT)."""
    signal = pred.get("signal", "?")
    conf = pred.get("confidence", 0)
    prob_l = pred.get("prob_long", 0)
    prob_s = pred.get("prob_short", 0)
    prob_nt = pred.get("prob_no_trade", 0)
    agreement = pred.get("model_agreement", "")
    unanimous = pred.get("unanimous", False)
    time_str = pred.get("time", "?")

    emoji = "\U0001f7e2" if signal == "LONG" else "\U0001f534"  # green/red circle
    unan_str = " (unanimous)" if unanimous else ""

    return (
        f"{emoji} <b>SIGNAL: {signal}</b>\n"
        f"Confidence: <b>{conf:.3f}</b>\n"
        f"Prob: L={prob_l:.3f}  S={prob_s:.3f}  NT={prob_nt:.3f}\n"
        f"Models: {agreement}{unan_str}\n"
        f"Time: {time_str} UTC"
    )


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
        "PROFIT_LOCK": "\U0001f512",  # lock
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

    return "\n".join(lines)


def fmt_hourly_report(predictions_df, btc: dict | None,
                      position: dict | None, safety_stats: dict | None,
                      data_status: dict | None) -> str:
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

    # Position
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


def fmt_position_detail(position: dict | None, btc: dict | None) -> str:
    """Format detailed position view for /position command."""
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
        "PROFIT_LOCK": "\U0001f512",
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
        lines.append(f"{emoji} {ts_short} | {action} {signal} {price_str} ({float(conf):.2f})")

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

        pos = state.get("position", {})
        side = pos.get("current_side")
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
                        position: dict | None) -> str:
    """Format quick one-liner for /status command."""
    parts = []

    # Position
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
            pnl_str = f" ({pnl_pct:+.2f}%)"
        parts.append(f"Pos: {side}{pnl_str}")
    else:
        parts.append("Pos: FLAT")

    # BTC price
    if btc:
        parts.append(f"BTC: ${btc.get('close', 0):,.1f}")

    # Last prediction
    if pred:
        signal = pred.get("signal", "NT")
        conf = pred.get("confidence", 0)
        age = pred.get("age_seconds", 0)
        age_min = age // 60
        parts.append(f"Last: {signal} ({conf:.2f}, {age_min}m ago)")
    else:
        parts.append("Last: N/A")

    return " | ".join(parts)
