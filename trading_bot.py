"""
V10 Trading Bot — Connects ML predictions to Binance USDT-M Futures orders.

Architecture:
    live_predict.run_single_prediction() -> signal -> trading/executor.py -> Binance Futures
    Position state managed by trading/position_manager.py
    Safety monitor in trading/safety.py

    With --data-service: reads predictions from persistent CSV (written by data_service)
    instead of running the full prediction pipeline each cycle.

Usage:
    python trading_bot.py --testnet                     # Testnet mode (default threshold=0.75)
    python trading_bot.py --live                        # Production (requires CONFIRM)
    python trading_bot.py --testnet --threshold 0.80    # Higher precision
    python trading_bot.py --testnet --amount 50         # Custom position size ($50)
    python trading_bot.py --dry-run                     # Predict only, no orders
    python trading_bot.py --dry-run --threshold 0.80    # Dry-run with custom threshold
    python trading_bot.py --testnet --data-service      # Read predictions from data service
"""

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import signal as signal_module
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force unbuffered stdout so print() output appears in logs/pipes immediately
os.environ["PYTHONUNBUFFERED"] = "1"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.structured_log import log_structured_event
from trading.executor import BinanceFuturesExecutor
from trading.position_manager import PositionManager
from trading.safety import SafetyMonitor

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent / "trading_state" / "state.json"
LOGS_DIR = Path(__file__).parent / "trading_logs"

# CSV columns for trade log
CSV_COLUMNS = [
    "timestamp", "signal", "confidence", "action", "side", "quantity",
    "price", "avg_entry", "sl_price", "tp_price", "order_id",
    "model_agreement", "unanimous", "latency_sec",
    "realized_pnl_pct", "realized_pnl_usdt", "balance_after",
]

# Graceful shutdown: Event-based so time.sleep() can be interrupted
import threading
_shutdown_event = threading.Event()
_shutdown_requested = False

# Balance and PnL tracking (updated on trade close events)
_account_balance = None
_cumulative_pnl_usdt = 0.0


# ============================================================================
# SIGTERM Handler (Docker graceful stop)
# ============================================================================

def _sigterm_handler(signum, frame):
    """Handle SIGTERM for graceful Docker shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    _shutdown_event.set()  # Wake up any Event.wait() immediately
    logger.info("SIGTERM received — shutting down after current cycle")


signal_module.signal(signal_module.SIGTERM, _sigterm_handler)


# ============================================================================
# State Persistence
# ============================================================================

def save_state(position_mgr: PositionManager, safety: SafetyMonitor):
    """Save bot state to JSON for recovery after restart (atomic write)."""
    import tempfile
    state = {
        "position": position_mgr.to_dict(),
        "trade_history": safety.to_list(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "account_balance": _account_balance,
        "cumulative_pnl_usdt": _cumulative_pnl_usdt,
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(STATE_FILE.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, str(STATE_FILE))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.debug(f"State saved to {STATE_FILE}")


def load_state(position_mgr: PositionManager, safety: SafetyMonitor) -> bool:
    """Load bot state from JSON. Returns True if state was loaded."""
    if not STATE_FILE.exists():
        return False

    try:
        state = json.loads(STATE_FILE.read_text())
        position_mgr.load_from_dict(state.get("position", {}))
        safety.load_from_list(state.get("trade_history", []))

        global _account_balance, _cumulative_pnl_usdt
        _account_balance = state.get("account_balance")
        _cumulative_pnl_usdt = state.get("cumulative_pnl_usdt", 0.0)

        logger.info(f"State loaded from {STATE_FILE} "
                     f"(updated: {state.get('last_updated', 'unknown')})")
        return True
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
        return False


# ============================================================================
# Trade Logging
# ============================================================================

def get_trade_log_path() -> Path:
    """Get today's trade log CSV path."""
    LOGS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return LOGS_DIR / f"trades_{date_str}.csv"


def log_trade(action: str, prediction: dict, position_mgr: PositionManager,
              order_result: dict | None = None, close_info: dict | None = None,
              realized_pnl_pct="", realized_pnl_usdt="", balance_after=""):
    """Append a trade entry to the CSV log.

    Args:
        close_info: Pre-reset position data from _close_and_wait. When provided,
                    avg_entry/sl_price/tp_price come from close_info instead of
                    position_mgr (which has already been reset to zeros).
        realized_pnl_pct: Percentage PnL for close events (empty for open/add).
        realized_pnl_usdt: USDT PnL for close events (empty for open/add).
        balance_after: Account balance after close (empty for open/add).
    """
    log_path = get_trade_log_path()
    write_header = not log_path.exists()

    row = {
        "timestamp": prediction.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "signal": prediction.get("signal", ""),
        "confidence": prediction.get("confidence", 0),
        "action": action,
        "side": _signal_to_side(prediction.get("signal", "")),
        "quantity": order_result.get("quantity", 0) if order_result else 0,
        "price": order_result.get("price", 0) if order_result else 0,
        "avg_entry": close_info["avg_entry"] if close_info else position_mgr.avg_entry,
        "sl_price": close_info["sl_price"] if close_info else position_mgr.sl_price,
        "tp_price": close_info["tp_price"] if close_info else position_mgr.tp_price,
        "order_id": order_result.get("order_id", "") if order_result else "",
        "model_agreement": ",".join(prediction.get("model_agreement", [])),
        "unanimous": prediction.get("unanimous", False),
        "latency_sec": prediction.get("latency_sec", 0),
        "realized_pnl_pct": realized_pnl_pct,
        "realized_pnl_usdt": realized_pnl_usdt,
        "balance_after": balance_after,
    }

    # Atomic append: read existing + append + temp file + rename
    import tempfile
    existing_rows = []
    if log_path.exists():
        try:
            with open(log_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception:
            pass  # Corrupted file — start fresh

    existing_rows.append(row)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(log_path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(existing_rows)
        os.replace(tmp_path, str(log_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.debug(f"Trade logged to {log_path}")


def _signal_to_side(signal: str) -> str:
    """Convert signal name to order side."""
    if signal == "LONG":
        return "BUY"
    elif signal == "SHORT":
        return "SELL"
    return ""


# ============================================================================
# Console Display
# ============================================================================

def print_status(prediction: dict, action: str, position_mgr: PositionManager,
                 safety: SafetyMonitor):
    """Print formatted status to console."""
    ts = prediction.get("timestamp", "?")
    signal = prediction.get("signal", "NO_TRADE")
    conf = prediction.get("confidence", 0)
    probs = prediction.get("probabilities", {})
    unanimous = prediction.get("unanimous", False)
    latency = prediction.get("latency_sec", 0)

    # Signal line
    if signal == "NO_TRADE":
        print(f"\n[{ts}] NO_TRADE (conf={conf:.3f}) | "
              f"L={probs.get('LONG', 0):.3f} S={probs.get('SHORT', 0):.3f} | "
              f"{latency}s", flush=True)
    else:
        agree_str = "UNANIMOUS" if unanimous else f"split: {prediction.get('model_agreement', [])}"
        print(f"\n[{ts}] >>> {signal} <<< (conf={conf:.3f}) | "
              f"L={probs.get('LONG', 0):.3f} S={probs.get('SHORT', 0):.3f} | "
              f"{agree_str} | {latency}s", flush=True)
        # Show entry/SL/TP when signal fires
        entry = prediction.get("entry_price", 0)
        if entry and entry > 0:
            if signal == "LONG":
                sl, tp = entry * 0.98, entry * 1.04
            else:
                sl, tp = entry * 1.02, entry * 0.96
            print(f"  Entry: ${entry:,.2f} | SL: ${sl:,.2f} (-2%) | TP: ${tp:,.2f} (+4%)",
                  flush=True)

    # Action line
    if action and action != "SKIPPED":
        print(f"  Action: {action}", flush=True)

    # Position line
    if position_mgr.current_side:
        extra = ""
        if position_mgr.profit_lock_active:
            extra = f" | ProfitLock=ACTIVE (HWM={position_mgr.high_water_mark_pct:.1f}%)"
        print(f"  Position: {position_mgr.current_side} "
              f"{position_mgr.total_quantity:.4f} BTC "
              f"@ {position_mgr.avg_entry:.2f} | "
              f"SL={position_mgr.sl_price:.2f} TP={position_mgr.tp_price:.2f}"
              f"{extra}", flush=True)
    else:
        print("  Position: FLAT", flush=True)

    # Safety line
    stats = safety.get_stats()
    if stats["total_trades"] > 0:
        wr_str = (f"{stats['7d_wr']:.1f}% 7d"
                  if stats["7d_wr"] is not None else "n/a")
        print(f"  Safety: {stats['total_trades']} trades ({stats['7d_trades']} in 7d), "
              f"WR={stats['total_wr']:.1f}% total / {wr_str} | "
              f"Consec losses: {stats['consecutive_losses']}", flush=True)
        if stats["paused"]:
            print(f"  *** PAUSED: {stats['pause_reason']} ***", flush=True)


# ============================================================================
# Exchange Sync & Force Close Helpers
# ============================================================================

def sync_exchange_state(executor: BinanceFuturesExecutor,
                        position_mgr: PositionManager,
                        safety: SafetyMonitor) -> str | None:
    """Sync local state with exchange. Detects SL/TP triggers.

    Returns action string if state changed, None otherwise.
    """
    try:
        exchange_pos = executor.get_position()
        exchange_amt = float(exchange_pos.get("positionAmt", 0))
    except Exception as e:
        logger.warning(f"Exchange sync failed: {e}")
        return None

    had_position = position_mgr.current_side is not None
    exchange_flat = abs(exchange_amt) < 1e-8

    if had_position and exchange_flat:
        # Position was closed on exchange (SL/TP triggered or manual close)
        prev_side = position_mgr.current_side
        logger.info(f"Exchange FLAT but local={prev_side} — SL/TP triggered or manual close")

        # Determine win/loss from last trade's realized PnL
        pnl = executor.get_last_trade_pnl()
        pnl_usdt = pnl if pnl is not None else 0.0
        if pnl is not None:
            win = pnl > 0
            logger.info(f"Trade PnL: {pnl:.4f} USDT ({'WIN' if win else 'LOSS'})")
        else:
            win = False
            logger.warning("Could not determine trade PnL — recording as LOSS")

        # Compute PnL percentage estimate and query balance
        close_pnl_pct = ""
        close_pnl_usdt = ""
        close_balance = ""
        if pnl is not None and position_mgr.avg_entry > 0 and position_mgr.total_quantity > 0:
            close_pnl_pct = round(
                pnl_usdt / (position_mgr.avg_entry * position_mgr.total_quantity) * 100, 4)
            close_pnl_usdt = round(pnl_usdt, 4)

        balance = executor.get_account_balance()
        if balance:
            close_balance = round(balance["total_balance"], 2)

        global _account_balance, _cumulative_pnl_usdt
        if balance:
            _account_balance = balance["total_balance"]
        _cumulative_pnl_usdt += pnl_usdt

        safety.record_trade(win=win)

        # Log the auto-close event
        log_trade(
            action="SL_TP_TRIGGERED",
            prediction={"signal": prev_side, "confidence": 0,
                        "timestamp": datetime.now(timezone.utc).isoformat()},
            position_mgr=position_mgr,
            realized_pnl_pct=close_pnl_pct,
            realized_pnl_usdt=close_pnl_usdt,
            balance_after=close_balance,
        )

        log_structured_event(logger, "SL_TP_HIT",
                             side=prev_side, pnl_usdt=pnl_usdt,
                             win=win, balance=close_balance)
        position_mgr.reset()
        return "SL_TP_TRIGGERED"

    elif not had_position and not exchange_flat:
        # Exchange has position but local doesn't — re-sync
        logger.warning("Exchange has position but local is FLAT — syncing from exchange")
        position_mgr.sync_from_exchange(exchange_pos)
        return "RESYNCED"

    elif had_position and not exchange_flat:
        # Both have position — check side agreement
        exchange_side = "LONG" if exchange_amt > 0 else "SHORT"
        if exchange_side != position_mgr.current_side:
            logger.warning(
                f"Side mismatch: local={position_mgr.current_side} "
                f"exchange={exchange_side} — resyncing")
            position_mgr.sync_from_exchange(exchange_pos)
            return "RESYNCED"

    return None


def force_close(reason: str, executor: BinanceFuturesExecutor,
                position_mgr: PositionManager,
                safety: SafetyMonitor) -> bool:
    """Force close position with reason. Records trade outcome.

    Returns True if close succeeded.
    """
    if position_mgr.current_side is None:
        return True

    logger.info(f"FORCE CLOSE: {reason} | "
                f"{position_mgr.current_side} @ {position_mgr.avg_entry:.2f}")

    # Get mark price before close for PnL estimation
    try:
        mark = executor.get_mark_price()
        if position_mgr.current_side == "LONG":
            pnl_pct = (mark - position_mgr.avg_entry) / position_mgr.avg_entry * 100
        else:
            pnl_pct = (position_mgr.avg_entry - mark) / position_mgr.avg_entry * 100
    except Exception:
        pnl_pct = 0.0

    result = executor.close_position(
        local_qty=position_mgr.total_quantity,
        local_side=position_mgr.current_side)
    if result is not None:
        win = pnl_pct > 0
        pnl_usdt = pnl_pct / 100 * position_mgr.avg_entry * position_mgr.total_quantity
        safety.record_trade(win=win)
        logger.info(f"Force close complete: ~{pnl_pct:+.2f}% "
                     f"(${pnl_usdt:+.2f}) ({'WIN' if win else 'LOSS'})")

        # Query balance and update tracking
        close_balance = ""
        balance = executor.get_account_balance()
        if balance:
            close_balance = round(balance["total_balance"], 2)

        global _account_balance, _cumulative_pnl_usdt
        if balance:
            _account_balance = balance["total_balance"]
        _cumulative_pnl_usdt += pnl_usdt

        # Log the force close
        log_trade(
            action=reason,
            prediction={"signal": position_mgr.current_side, "confidence": 0,
                        "timestamp": datetime.now(timezone.utc).isoformat()},
            position_mgr=position_mgr,
            order_result=result,
            realized_pnl_pct=round(pnl_pct, 4),
            realized_pnl_usdt=round(pnl_usdt, 4),
            balance_after=close_balance,
        )

        position_mgr.reset()
        return True
    else:
        logger.error(f"Force close FAILED for reason={reason}")
        return False


def predict_with_timeout(predict_fn, threshold: float, timeout: int = 120):
    """Run prediction with timeout. Returns prediction dict or None on timeout."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(predict_fn, threshold=threshold)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning(f"Prediction timed out after {timeout}s — skipping cycle")
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ============================================================================
# Data Service Mode — Read predictions from persistent CSV
# ============================================================================

def read_latest_prediction(data_dir: str, threshold: float) -> dict | None:
    """Read the latest prediction from the data service's persistent CSV.

    Re-applies threshold so the bot can use a different threshold than the
    data service (which writes at a low threshold to preserve all signals).

    Returns prediction dict compatible with run_single_prediction() output,
    or None if data is stale (>10 min) or unavailable.
    """
    import pandas as pd

    pred_path = Path(data_dir) / "predictions" / "predictions.csv"
    if not pred_path.exists():
        logger.warning(f"Predictions CSV not found: {pred_path}")
        return None

    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        logger.warning(f"Failed to read predictions CSV: {e}")
        return None

    if df.empty:
        logger.warning("Predictions CSV is empty")
        return None

    # Validate required columns exist (guards against corrupted/partial reads)
    required_cols = {"time", "prob_no_trade", "prob_long", "prob_short"}
    if not required_cols.issubset(df.columns):
        logger.warning(f"Predictions CSV missing columns: {required_cols - set(df.columns)}")
        return None

    # Get last row
    row = df.iloc[-1]
    pred_time = pd.to_datetime(row["time"])
    if pred_time.tzinfo is None:
        pred_time = pred_time.tz_localize("UTC")

    # Staleness check: prediction time is candle OPEN time (e.g., 21:20).
    # Normal delay chain: candle close (+5min) → data service processes (+10s)
    # → data service cycle offset (0-5min) → bot reads on its cycle (0-5min)
    # = ~10-15 min typical age. Threshold: 1200s (20min / 4 candle periods)
    # detects genuine data service outages while allowing normal timing drift.
    now = datetime.now(timezone.utc)
    age_sec = (now - pred_time).total_seconds()
    if age_sec > 1200:
        logger.warning(f"Prediction is stale ({age_sec:.0f}s old, max=1200s)")
        return None

    # Re-apply threshold
    prob_nt = float(row.get("prob_no_trade", 0))
    prob_long = float(row.get("prob_long", 0))
    prob_short = float(row.get("prob_short", 0))

    probs = [prob_nt, prob_long, prob_short]
    pred_class = int(max(range(3), key=lambda i: probs[i]))
    confidence = probs[pred_class]

    class_names = {0: "NO_TRADE", 1: "LONG", 2: "SHORT"}
    trade_classes = {1, 2}

    if pred_class in trade_classes and confidence >= threshold:
        signal = class_names[pred_class]
    else:
        signal = "NO_TRADE"

    # Parse model_agreement string back to list
    agreement_str = str(row.get("model_agreement", ""))
    model_agreement = agreement_str.split(",") if agreement_str else []

    # Get BTC close price for entry/SL/TP display
    entry_price = 0.0
    klines_path = Path(data_dir) / "klines" / "ml_data_5M.csv"
    if klines_path.exists():
        try:
            kl = pd.read_csv(klines_path)
            time_col = "time" if "time" in kl.columns else "Open Time"
            kl[time_col] = pd.to_datetime(kl[time_col])
            entry_price = float(kl.iloc[-1]["Close"])
        except Exception:
            pass

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "probabilities": {
            "NO_TRADE": round(prob_nt, 4),
            "LONG": round(prob_long, 4),
            "SHORT": round(prob_short, 4),
        },
        "threshold": threshold,
        "n_models": 3,
        "model_agreement": model_agreement,
        "unanimous": str(row.get("unanimous", "")).lower() == "true",
        "timestamp": str(row["time"]),
        "latency_sec": round(age_sec, 1),
        "entry_price": entry_price,
    }


# ============================================================================
# Main Loop
# ============================================================================

def run_bot(args):
    """Main bot loop."""
    global _shutdown_requested, _account_balance, _cumulative_pnl_usdt

    use_data_service = getattr(args, "data_service", False)
    data_dir = getattr(args, "data_dir", None)

    if not use_data_service:
        from model_training.live_predict import run_single_prediction

    # Initialize components
    executor = BinanceFuturesExecutor(
        testnet=args.testnet,
        leverage=args.leverage,
        usdt_amount=args.amount,
        dry_run=args.dry_run,
    )
    from core.config import TRADING_SL_PCT, TRADING_TP_PCT
    position_mgr = PositionManager(sl_pct=TRADING_SL_PCT, tp_pct=TRADING_TP_PCT)
    safety = SafetyMonitor(min_wr=33.3, lookback_days=7)

    # Load persisted state
    load_state(position_mgr, safety)

    # Sync position from exchange (overrides local state if exchange disagrees)
    if not args.dry_run:
        sync_exchange_state(executor, position_mgr, safety)

    # Verify SL/TP orders exist for open positions
    if position_mgr.current_side and not args.dry_run:
        logger.info("Verifying SL/TP orders on exchange...")
        sl_tp_status = executor.verify_sl_tp_orders()

        if sl_tp_status['has_sl'] is None or sl_tp_status['has_tp'] is None:
            logger.warning("SL/TP verification failed (API error) — will retry next cycle")
        elif not sl_tp_status['has_sl'] or not sl_tp_status['has_tp']:
            missing = []
            if not sl_tp_status['has_sl']:
                missing.append("SL")
            if not sl_tp_status['has_tp']:
                missing.append("TP")
            logger.critical(
                f"MISSING {'+'.join(missing)} orders for "
                f"{position_mgr.current_side} position! Re-placing...")

            sl_tp_ok = executor.update_sl_tp(
                side=position_mgr.current_side,
                sl_price=position_mgr.sl_price,
                tp_price=position_mgr.tp_price)
            if sl_tp_ok:
                logger.info("SL/TP re-placed successfully")
            else:
                logger.critical("SL/TP re-placement FAILED — emergency closing exchange position")
                executor.emergency_close()
                position_mgr.reset()
        else:
            logger.info(
                f"SL/TP verified: SL={sl_tp_status['sl_price']:.1f} "
                f"TP={sl_tp_status['tp_price']:.1f}")
        save_state(position_mgr, safety)

    mode = "DRY RUN" if args.dry_run else ("TESTNET" if args.testnet else "LIVE")
    source = "DATA SERVICE" if use_data_service else "LIVE PREDICT"
    print(f"\n{'='*60}", flush=True)
    print(f"  V10 Trading Bot — {mode} ({source})", flush=True)
    print(f"  Threshold: {args.threshold} | Amount: ${args.amount} | "
          f"Leverage: {args.leverage}x", flush=True)
    print(f"  Interval: {args.interval}s | SL: {position_mgr.sl_pct}% | "
          f"TP: {position_mgr.tp_pct}%", flush=True)
    print(f"  Risk: max_hold={position_mgr.max_hold_seconds//3600}h | "
          f"profit_lock={position_mgr.profit_lock_trigger}%/{position_mgr.profit_lock_floor}%",
          flush=True)
    print(f"  Safety: min_WR={safety.min_wr}% over {safety.lookback_days}d | "
          f"circuit_breaker=3err/10min_cap", flush=True)
    if use_data_service:
        print(f"  Data dir: {data_dir}", flush=True)
    if position_mgr.current_side:
        print(f"  Resuming {position_mgr.current_side} position: "
              f"{position_mgr.total_quantity:.4f} BTC @ {position_mgr.avg_entry:.2f}",
              flush=True)
    print(f"{'='*60}\n", flush=True)

    consecutive_errors = 0
    base_interval = args.interval

    cycle = 0
    while not _shutdown_requested:
        cycle += 1
        logger.info(f"--- Cycle {cycle} ---")

        # Circuit breaker: exponential backoff on consecutive errors
        sleep_interval = base_interval
        if consecutive_errors >= 3:
            sleep_interval = min(base_interval * (2 ** (consecutive_errors - 2)), 600)

        try:
            # Step 0: Exchange sync — detect SL/TP triggers (Bug 5 + Bug 1)
            if not args.dry_run:
                sync_result = sync_exchange_state(executor, position_mgr, safety)
                if sync_result:
                    logger.info(f"Exchange sync: {sync_result}")
                    save_state(position_mgr, safety)

                # Verify SL/TP after resync (exchange position may lack orders)
                if sync_result == "RESYNCED" and position_mgr.current_side:
                    sl_tp_status = executor.verify_sl_tp_orders()
                    if sl_tp_status['has_sl'] is None or sl_tp_status['has_tp'] is None:
                        logger.warning("SL/TP verification after RESYNC failed (API error) — will retry")
                    elif not sl_tp_status['has_sl'] or not sl_tp_status['has_tp']:
                        logger.critical("RESYNCED but SL/TP missing — re-placing")
                        sl_tp_ok = executor.update_sl_tp(
                            side=position_mgr.current_side,
                            sl_price=position_mgr.sl_price,
                            tp_price=position_mgr.tp_price)
                        if not sl_tp_ok:
                            logger.critical(
                                "SL/TP re-placement after RESYNC FAILED — emergency close")
                            position_mgr.reset()
                        save_state(position_mgr, safety)

            # Step 1: Max hold check — force close at 24h (Feature 1)
            if position_mgr.is_max_hold_expired():
                force_close("MAX_HOLD_24H", executor, position_mgr, safety)
                save_state(position_mgr, safety)

            # Step 2: Profit lock check — trailing close at 3.0% (Feature 2)
            if position_mgr.current_side and not args.dry_run:
                try:
                    mark = executor.get_mark_price()
                    lock_reason = position_mgr.update_profit_lock(mark)
                    if lock_reason:
                        force_close(lock_reason, executor, position_mgr, safety)
                        save_state(position_mgr, safety)
                except Exception as e:
                    logger.warning(f"Profit lock check failed: {e}")

            # Aggressive cycle near max hold (last 30 min → 60s interval)
            remaining = position_mgr.remaining_hold_seconds()
            if remaining is not None and 0 < remaining < 1800:
                sleep_interval = min(sleep_interval, 60)
                logger.info(f"Max hold approaching ({remaining:.0f}s left) — 60s cycle")

            # Step 2b: Periodic balance query (every 10 cycles ~50min)
            if cycle % 10 == 0 and not args.dry_run:
                try:
                    balance = executor.get_account_balance()
                    if balance:
                        _account_balance = balance["total_balance"]
                        save_state(position_mgr, safety)
                        logger.debug(f"Periodic balance update: ${_account_balance:,.2f}")
                except Exception as e:
                    logger.warning(f"Periodic balance query failed: {e}")

            # Step 3: Get prediction
            if use_data_service:
                prediction = read_latest_prediction(data_dir, args.threshold)
            else:
                prediction = predict_with_timeout(
                    run_single_prediction, args.threshold, timeout=120)
            if prediction is None:
                _shutdown_event.wait(sleep_interval)
                continue

            signal = prediction["signal"]

            # Step 4: Check safety
            if not safety.check():
                log_structured_event(logger, "SAFETY_PAUSE",
                                     reason=safety.get_stats().get("pause_reason", ""))
                print_status(prediction, "PAUSED", position_mgr, safety)
                save_state(position_mgr, safety)
                _shutdown_event.wait(sleep_interval)
                continue

            # Step 5: Act on signal
            if signal == "NO_TRADE":
                print_status(prediction, "", position_mgr, safety)
                _shutdown_event.wait(sleep_interval)
                continue

            # Execute trade action
            action, close_info = position_mgr.handle_signal(signal, executor)
            print_status(prediction, action, position_mgr, safety)

            # Record opposite-signal close as a trade outcome
            close_pnl_pct = ""
            close_pnl_usdt = ""
            close_balance = ""
            if close_info is not None:
                avg_e = close_info["avg_entry"]
                close_p = close_info.get("close_price", 0)
                if avg_e > 0 and close_p > 0:
                    if close_info["prev_side"] == "LONG":
                        pnl_pct = (close_p - avg_e) / avg_e * 100
                    else:
                        pnl_pct = (avg_e - close_p) / avg_e * 100
                    pnl_usdt = pnl_pct / 100 * avg_e * close_info.get("total_quantity", 0)
                    win = pnl_pct > 0
                    safety.record_trade(win=win)
                    logger.info(
                        f"Opposite-signal close: ~{pnl_pct:+.2f}% "
                        f"(${pnl_usdt:+.2f}) ({'WIN' if win else 'LOSS'})")

                    close_pnl_pct = round(pnl_pct, 4)
                    close_pnl_usdt = round(pnl_usdt, 4)
                    balance = executor.get_account_balance()
                    if balance:
                        close_balance = round(balance["total_balance"], 2)
                        _account_balance = balance["total_balance"]
                    _cumulative_pnl_usdt += pnl_usdt

            # Structured event logging
            if action == "OPENED":
                log_structured_event(logger, "TRADE_OPEN",
                                     signal=signal, confidence=prediction.get("confidence", 0),
                                     price=position_mgr.avg_entry, side=signal)
            if close_info is not None:
                log_structured_event(logger, "TRADE_CLOSE",
                                     prev_side=close_info.get("prev_side", ""),
                                     pnl_pct=close_pnl_pct, pnl_usdt=close_pnl_usdt,
                                     win=close_pnl_pct > 0 if close_pnl_pct else False,
                                     balance=close_balance)

            # Log trade
            order_result = None
            if action in ("OPENED", "ADDED") and position_mgr.entries:
                last_entry = position_mgr.entries[-1]
                order_result = {"price": last_entry[0], "quantity": last_entry[1],
                                "order_id": ""}
            elif close_info is not None:
                order_result = {
                    "price": close_info.get("close_price", 0),
                    "quantity": close_info.get("total_quantity", 0),
                    "order_id": close_info.get("order_id", ""),
                }
            log_trade(action, prediction, position_mgr, order_result,
                      close_info=close_info,
                      realized_pnl_pct=close_pnl_pct,
                      realized_pnl_usdt=close_pnl_usdt,
                      balance_after=close_balance)

            # Save state after every action
            save_state(position_mgr, safety)

            # Circuit breaker: reset on successful cycle
            if consecutive_errors > 0:
                logger.info(f"Circuit breaker reset (was {consecutive_errors} errors)")
            consecutive_errors = 0

        except KeyboardInterrupt:
            print("\nBot stopped by user", flush=True)
            save_state(position_mgr, safety)
            break

        except Exception as e:
            logger.error(f"Cycle {cycle} failed: {e}", exc_info=True)
            print(f"\n[ERROR] Cycle {cycle}: {e}", flush=True)
            save_state(position_mgr, safety)

            consecutive_errors += 1
            if consecutive_errors >= 3:
                logger.critical(
                    f"CIRCUIT BREAKER: {consecutive_errors} consecutive errors. "
                    f"Next interval: "
                    f"{min(base_interval * (2 ** (consecutive_errors - 2)), 600)}s")

                if consecutive_errors == 3 and not args.dry_run:
                    try:
                        executor.reconnect()
                    except Exception as re_err:
                        logger.error(f"Reconnect failed: {re_err}")

        # Wait for next cycle
        try:
            _shutdown_event.wait(sleep_interval)
        except KeyboardInterrupt:
            print("\nBot stopped by user", flush=True)
            save_state(position_mgr, safety)
            break

    # SIGTERM shutdown path
    if _shutdown_requested:
        print("\nSIGTERM shutdown — saving state...", flush=True)
        save_state(position_mgr, safety)
        print("State saved. Goodbye.", flush=True)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V10 Trading Bot — ML predictions to Binance Futures orders")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--testnet", action="store_true",
                            help="Run on Binance Futures testnet")
    mode_group.add_argument("--live", action="store_true",
                            help="Run on production Binance Futures (requires CONFIRM)")
    mode_group.add_argument("--dry-run", action="store_true",
                            help="Predict only, no orders placed")

    parser.add_argument("--threshold", type=float,
                        default=float(os.environ.get("TRADING_THRESHOLD", "0.75")),
                        help="Confidence threshold (default: $TRADING_THRESHOLD or 0.75)")
    parser.add_argument("--amount", type=float,
                        default=float(os.environ.get("TRADING_AMOUNT", "100.0")),
                        help="USDT per position (default: $TRADING_AMOUNT or 100)")
    parser.add_argument("--leverage", type=int,
                        default=int(os.environ.get("TRADING_LEVERAGE", "10")),
                        help="Leverage multiplier (default: $TRADING_LEVERAGE or 10)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between prediction cycles (default: 300)")
    parser.add_argument("--data-service", action="store_true",
                        help="Read predictions from data service CSV instead of running pipeline")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Path to data service persistent data dir (default: /data)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging (stdout + JSONL file)
    from core.structured_log import setup_logging
    setup_logging("trading_bot", log_dir="trading_logs/jsonl", debug=args.debug)

    # Production safety gate
    if args.live:
        # Warn if critical env vars are missing (using dangerous code defaults)
        env_warnings = []
        if not os.environ.get("TRADING_AMOUNT"):
            env_warnings.append(
                f"TRADING_AMOUNT not set — using default ${args.amount} "
                f"(code default is $100, intended is $10)")
        if not os.environ.get("TRADING_LEVERAGE"):
            env_warnings.append(
                f"TRADING_LEVERAGE not set — using default {args.leverage}x "
                f"(code default is 10x, intended is 20x)")
        if not os.environ.get("TRADING_THRESHOLD"):
            env_warnings.append(
                f"TRADING_THRESHOLD not set — using default {args.threshold}")
        if env_warnings:
            print("\n" + "!" * 60)
            print("  ENV VAR WARNINGS:")
            for w in env_warnings:
                print(f"  !!! {w}")
            print("!" * 60)

        print("\n" + "!" * 60)
        print("  WARNING: You are about to trade with REAL MONEY")
        print("  on Binance USDT-M Futures (BTCUSDT)")
        print(f"  Amount: ${args.amount} x {args.leverage}x leverage")
        print(f"  Threshold: {args.threshold}")
        print("!" * 60)
        confirm = input("\nType CONFIRM to proceed: ").strip()
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(1)
        args.testnet = False
    elif args.dry_run:
        args.testnet = True  # dry-run doesn't need real keys

    run_bot(args)


if __name__ == "__main__":
    main()
