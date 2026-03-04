"""Read-only data readers for all file sources.

All functions return None on file-not-found or parse error (never raise).
Paths are configurable for local testing; defaults match Docker volume layout.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default paths (Docker volume layout)
DATA_DIR = Path("/data")
STATE_DIR = Path("/app/trading_state")
LOGS_DIR = Path("/app/trading_logs")


def configure_paths(data_dir: str = None, state_dir: str = None,
                    logs_dir: str = None):
    """Override default paths (for local testing)."""
    global DATA_DIR, STATE_DIR, LOGS_DIR
    if data_dir:
        DATA_DIR = Path(data_dir)
    if state_dir:
        STATE_DIR = Path(state_dir)
    if logs_dir:
        LOGS_DIR = Path(logs_dir)


def read_data_service_status() -> dict | None:
    """Read /data/status.json written by data service healthcheck."""
    path = DATA_DIR / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"Failed to read status.json: {e}")
        return None


def read_predictions(last_n_hours: int = 1) -> pd.DataFrame | None:
    """Read predictions CSV filtered to the last N hours."""
    path = DATA_DIR / "predictions" / "predictions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "time" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["time"])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=last_n_hours)
        cutoff = cutoff.replace(tzinfo=None)  # match tz-naive CSV times
        return df[df["time"] >= cutoff].copy()
    except Exception as e:
        logger.warning(f"Failed to read predictions: {e}")
        return None


def read_latest_prediction() -> dict | None:
    """Read the last row of predictions CSV as a dict."""
    path = DATA_DIR / "predictions" / "predictions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        required = {"time", "prob_no_trade", "prob_long", "prob_short"}
        if not required.issubset(df.columns):
            return None
        row = df.iloc[-1]
        pred_time = pd.to_datetime(row["time"])
        age_sec = (datetime.now(timezone.utc) - pred_time.tz_localize("UTC")).total_seconds()
        return {
            "time": str(row["time"]),
            "prob_no_trade": float(row.get("prob_no_trade", 0)),
            "prob_long": float(row.get("prob_long", 0)),
            "prob_short": float(row.get("prob_short", 0)),
            "signal": str(row.get("signal", "NO_TRADE")),
            "confidence": float(row.get("confidence", 0)),
            "model_agreement": str(row.get("model_agreement", "")),
            "unanimous": str(row.get("unanimous", "")).lower() == "true",
            "age_seconds": round(age_sec),
        }
    except Exception as e:
        logger.warning(f"Failed to read latest prediction: {e}")
        return None


def read_trading_state() -> dict | None:
    """Read trading_state/state.json (position + trade history)."""
    path = STATE_DIR / "state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"Failed to read trading state: {e}")
        return None


def read_recent_trades(n_days: int = 7) -> pd.DataFrame | None:
    """Read and merge trade log CSVs from the last N days."""
    if not LOGS_DIR.exists():
        return None
    try:
        frames = []
        for i in range(n_days):
            date_str = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            path = LOGS_DIR / f"trades_{date_str}.csv"
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    frames.append(df)
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.sort_values("timestamp", ascending=False)
        return combined
    except Exception as e:
        logger.warning(f"Failed to read recent trades: {e}")
        return None


def read_last_n_trades(n: int = 10) -> pd.DataFrame | None:
    """Read the last N trades across all recent log files."""
    df = read_recent_trades(n_days=30)
    if df is None:
        return None
    return df.head(n)


def read_account_summary() -> dict | None:
    """Read balance + cumulative PnL from state.json."""
    state = read_trading_state()
    if not state:
        return None
    return {
        "account_balance": state.get("account_balance"),
        "cumulative_pnl_usdt": state.get("cumulative_pnl_usdt", 0.0),
        "last_updated": state.get("last_updated"),
    }


def read_trades_with_pnl(n_days: int = 30) -> pd.DataFrame | None:
    """Read trades CSV, ensure PnL columns exist (NaN if old CSV)."""
    df = read_recent_trades(n_days=n_days)
    if df is None:
        return None
    for col in ["realized_pnl_pct", "realized_pnl_usdt", "balance_after"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def compute_pnl_summary(n_days: int = 30) -> dict | None:
    """Aggregate PnL: today/7d/30d/all-time with W/L counts.

    Filters to close actions only and sums realized_pnl_usdt.
    """
    df = read_trades_with_pnl(n_days=n_days)
    if df is None:
        return None

    close_actions = {"CLOSED_WAITING", "SL_TP_TRIGGERED", "MAX_HOLD_24H",
                     "PROFIT_LOCK", "MAX_HOLD"}
    if "action" not in df.columns:
        return None

    closes = df[df["action"].isin(close_actions)].copy()
    if closes.empty:
        return None

    if "timestamp" in closes.columns:
        closes["timestamp"] = pd.to_datetime(closes["timestamp"], errors="coerce")

    closes["realized_pnl_usdt"] = pd.to_numeric(
        closes["realized_pnl_usdt"], errors="coerce")

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    def _period_stats(mask):
        sub = closes[mask]
        total_pnl = sub["realized_pnl_usdt"].sum()
        wins = int((sub["realized_pnl_usdt"] > 0).sum())
        losses = int((sub["realized_pnl_usdt"] <= 0).sum())
        count = len(sub)
        wr = (wins / count * 100) if count > 0 else 0
        return {
            "pnl_usdt": round(float(total_pnl), 2) if pd.notna(total_pnl) else 0,
            "wins": wins, "losses": losses,
            "count": count, "wr": round(float(wr), 1),
        }

    has_ts = "timestamp" in closes.columns and closes["timestamp"].notna().any()
    result = {"all_time": _period_stats(pd.Series(True, index=closes.index))}

    if has_ts:
        result["today"] = _period_stats(
            closes["timestamp"] >= now - timedelta(days=1))
        result["7d"] = _period_stats(
            closes["timestamp"] >= now - timedelta(days=7))
        result["30d"] = _period_stats(
            closes["timestamp"] >= now - timedelta(days=30))
    else:
        fallback = _period_stats(pd.Series(True, index=closes.index))
        result["today"] = result["7d"] = result["30d"] = fallback

    return result


def read_latest_btc_price() -> dict | None:
    """Read the latest BTC price from 5M klines."""
    path = DATA_DIR / "klines" / "ml_data_5M.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        row = df.iloc[-1]
        return {
            "time": str(row.get("time", "")),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
        }
    except Exception as e:
        logger.warning(f"Failed to read BTC price: {e}")
        return None
