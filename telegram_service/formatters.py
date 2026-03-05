"""HTML message templates for Telegram notifications.

Uses HTML parse mode to avoid MarkdownV2 escaping issues with prices.
"""

import io
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
        lines.append(f"<b>Balance:</b> ${bal:,.2f} | PnL: {pnl_emoji} ${cum_pnl:+,.2f}")
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
    """Format multi-line status for /status command (HTML)."""
    lines = ["\U0001f4ca <b>Status</b>\n"]

    # Position + BTC
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
                btc: dict | None) -> str:
    """Format wallet balance for /balance command."""
    if not account or account.get("account_balance") is None:
        return "\U0001f4b0 <b>Balance:</b> Not available yet\n(Updates after first trade close)"

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


def fmt_equity_curve(trades_df) -> str:
    """Format ASCII equity curve for /equity command (last 20 closed trades)."""
    if trades_df is None or trades_df.empty:
        return "\U0001f4c8 <b>Equity Curve:</b> No trade data"

    import pandas as pd

    close_actions = {"CLOSED_WAITING", "SL_TP_TRIGGERED", "MAX_HOLD_24H",
                     "PROFIT_LOCK", "MAX_HOLD"}
    if "action" not in trades_df.columns:
        return "\U0001f4c8 <b>Equity Curve:</b> No trade data"

    closes = trades_df[trades_df["action"].isin(close_actions)].copy()
    if "realized_pnl_usdt" not in closes.columns:
        return "\U0001f4c8 <b>Equity Curve:</b> No PnL data (old CSV format)"

    closes["realized_pnl_usdt"] = pd.to_numeric(
        closes["realized_pnl_usdt"], errors="coerce")
    closes = closes.dropna(subset=["realized_pnl_usdt"])

    if closes.empty:
        return "\U0001f4c8 <b>Equity Curve:</b> No PnL data"

    # Sort chronologically and take last 20
    if "timestamp" in closes.columns:
        closes = closes.sort_values("timestamp")
    closes = closes.tail(20)

    # Compute cumulative PnL
    cum_pnl = closes["realized_pnl_usdt"].cumsum().tolist()
    pnls = closes["realized_pnl_usdt"].tolist()

    if not cum_pnl:
        return "\U0001f4c8 <b>Equity Curve:</b> No data"

    max_val = max(abs(v) for v in cum_pnl) if cum_pnl else 1
    if max_val == 0:
        max_val = 1
    bar_width = 20

    n = len(cum_pnl)
    lines = [f"\U0001f4c8 <b>Equity Curve (last {n} trades)</b>\n<pre>"]

    for i, (cum, pnl) in enumerate(zip(cum_pnl, pnls)):
        bar_len = int(abs(cum) / max_val * bar_width)
        bar = "\u2588" * bar_len
        sign = "+" if pnl >= 0 else "-"
        if cum >= 0:
            line = f" {bar} ${cum:+.1f}"
        else:
            line = f"-{bar} ${cum:+.1f}"
        lines.append(f"{i+1:>2}{sign}|{line}")

    lines.append("</pre>")

    total = cum_pnl[-1] if cum_pnl else 0
    total_emoji = "\U0001f7e2" if total >= 0 else "\U0001f534"
    lines.append(f"\nNet: {total_emoji} <b>${total:+,.2f}</b>")

    return "\n".join(lines)


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
