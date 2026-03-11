#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Module for Cryptocurrency Trading Signal System

Centralized configuration for:
- Binance API settings
- Timeframes and window sizes
- Alert thresholds
- Output paths
- Bot tokens (Telegram/Discord)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directory (project root, one level up from core/)
BASE_DIR = Path(__file__).parent.parent.absolute()

# Data directory for CSV files
DATA_DIR = Path(r"C:\Users\Useer\Desktop\THE BEST PROJECT\NEW ALGORITHM CONSTRUCTION\Live Data\data")

# Output directory for processed results
OUTPUT_DIR = BASE_DIR / "output"

# API key file location (env var override, fallback to local file)
API_KEY_FILE = Path(os.environ.get("API_KEY_FILE", BASE_DIR / "api_key.txt"))

# ============================================================================
# BINANCE CONFIGURATION
# ============================================================================

SYMBOL = "BTCUSDT"
TLD = "com"  # Binance TLD (com, us, etc.)

# Timeframes with (interval_constant, lookback_days)
# These match the values in data/downloader.py
TIMEFRAMES: Dict[str, Tuple[str, int]] = {
    "3D": ("3d", 1200),
    "1D": ("1d", 700),
    "12H": ("12h", 200),
    "8H": ("8h", 180),
    "6H": ("6h", 180),
    "4H": ("4h", 100),
    "2H": ("2h", 50),
    "1H": ("1h", 30),
    "30M": ("30m", 30),
    "15M": ("15m", 15),
    "5M": ("5m", 5),
}

# Ordered list of timeframes (from longest to shortest)
TIMEFRAME_ORDER: List[str] = ["3D", "1D", "12H", "8H", "6H", "4H", "2H", "1H", "30M", "15M", "5M"]

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Window sizes for iterative regression (same for all timeframes)
WINDOW_SIZES: List[int] = [30, 60, 100, 120, 160]

# Window size labels for display
WINDOW_LABELS: Dict[int, str] = {
    30: "df (30)",
    60: "df1 (60)",
    100: "df2 (100)",
    120: "df3 (120)",
    160: "df4 (160)"
}

# ============================================================================
# TRADING CONFIGURATION (Single source of truth for all trading params)
# ============================================================================

# SL/TP percentages — used by trading bot, backfill, target labeling
TRADING_SL_PCT: float = 2.0
TRADING_TP_PCT: float = 4.0
TRADING_MAX_HOLD_CANDLES: int = 288   # 24h in 5M candles
TRADING_MAX_HOLD_SECONDS: int = 86400  # 24h in seconds

# Default confidence threshold for predictions
DEFAULT_THRESHOLD: float = 0.75

# Prediction staleness threshold — max age (seconds) before prediction is considered stale.
# Used by trading bot, telegram bot, and data validation module.
# 1200s = 20 min = 4 candle periods (candle OPEN time + processing = normal 500-700s age)
STALENESS_THRESHOLD_SEC: int = 1200

# Bootstrap bars — context window for feature encoding (must be consistent)
BOOTSTRAP_BARS: int = 1400

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

# Minimum time between duplicate alerts (seconds)
ALERT_COOLDOWN_SECONDS: int = 300  # 5 minutes

# Signals that trigger alerts (based on slope_f direction)
ALERT_SIGNALS: List[str] = [
    "BUY",
    "SELL"
]

# Signal display names
SIGNAL_NAMES: Dict[str, str] = {
    "BUY": "BUY",
    "SELL": "SELL",
    "HOLD": "HOLD"
}

# Signal colors for console output
SIGNAL_COLORS: Dict[str, str] = {
    "BUY": "green",
    "SELL": "red",
    "HOLD": "yellow",
    "HOLD (Uptrend)": "cyan",
    "HOLD (Downtrend)": "magenta"
}

# ============================================================================
# BOT CONFIGURATION
# ============================================================================

# Telegram Bot Token (get from @BotFather)
# Set via environment variable or edit directly
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Discord Bot Token (get from Discord Developer Portal)
# Set via environment variable or edit directly
DISCORD_BOT_TOKEN: str = os.environ.get("DISCORD_BOT_TOKEN", "")

# Discord channel ID for alerts (optional, can be set per-server)
DISCORD_ALERT_CHANNEL_ID: int = int(os.environ.get("DISCORD_CHANNEL_ID", "0"))

# ============================================================================
# CALENDAR CONFIGURATION (Signal Prediction Calendar)
# ============================================================================

# Hours of historical data to show in calendar
CALENDAR_HISTORY_HOURS: int = 3

# Hours of forward prediction in calendar
CALENDAR_FORWARD_HOURS: int = 4

# Time interval in minutes for calendar columns
CALENDAR_INTERVAL_MINUTES: int = 15

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Streamlit page configuration
DASHBOARD_PAGE_TITLE: str = "Crypto Trading Signals"
DASHBOARD_PAGE_ICON: str = "📈"
DASHBOARD_LAYOUT: str = "wide"

# Chart colors
CHART_COLORS: Dict[str, str] = {
    "df": "#1f77b4",   # Blue
    "df1": "#ff7f0e",  # Orange
    "df2": "#2ca02c",  # Green
    "df3": "#d62728",  # Red
    "df4": "#9467bd",  # Purple
}

# ============================================================================
# REFRESH CONFIGURATION
# ============================================================================

# Default refresh interval for continuous mode (minutes)
DEFAULT_REFRESH_INTERVAL_MINUTES: int = 5

# Minimum refresh interval (minutes)
MIN_REFRESH_INTERVAL_MINUTES: int = 1

# Maximum refresh interval (minutes)
MAX_REFRESH_INTERVAL_MINUTES: int = 60

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log file path
LOG_FILE = BASE_DIR / "trading_signals.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Console log level
CONSOLE_LOG_LEVEL = "INFO"

# File log level
FILE_LOG_LEVEL = "DEBUG"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_csv_path(timeframe: str) -> Path:
    """Get the CSV file path for a given timeframe."""
    return DATA_DIR / f"testing_data_{timeframe}.csv"


def get_output_path(filename: str) -> Path:
    """Get the output file path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def load_api_keys() -> Tuple[str, str]:
    """Load Binance API keys — env vars first, fallback to file."""
    env_key = os.environ.get("BINANCE_KEY", "") or os.environ.get("BINANCE_TESTNET_KEY", "")
    env_secret = os.environ.get("BINANCE_SECRET", "") or os.environ.get("BINANCE_TESTNET_SECRET", "")
    if env_key and env_secret:
        return env_key, env_secret

    if not API_KEY_FILE.exists():
        raise FileNotFoundError(
            f"API key file not found: {API_KEY_FILE}\n"
            f"Set BINANCE_KEY/BINANCE_SECRET env vars or create file.")

    with API_KEY_FILE.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError("API key file must have key on line 1 and secret on line 2.")

    return lines[0], lines[1]


def validate_config() -> Dict[str, bool]:
    """Validate configuration settings."""
    checks = {
        "data_dir_exists": DATA_DIR.exists(),
        "api_keys_available": bool(os.environ.get("BINANCE_KEY") or os.environ.get("BINANCE_TESTNET_KEY") or API_KEY_FILE.exists()),
        "telegram_token_set": bool(TELEGRAM_BOT_TOKEN),
        "discord_token_set": bool(DISCORD_BOT_TOKEN),
    }

    # Check for CSV files
    for tf in TIMEFRAME_ORDER:
        csv_path = get_csv_path(tf)
        checks[f"csv_{tf}_exists"] = csv_path.exists()

    return checks


if __name__ == "__main__":
    print("Configuration Module")
    print("=" * 50)

    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    print(f"\nTimeframes: {', '.join(TIMEFRAME_ORDER)}")
    print(f"Window Sizes: {WINDOW_SIZES}")

    print("\nConfiguration Validation:")
    validation = validate_config()
    for key, value in validation.items():
        status = "OK" if value else "MISSING"
        print(f"  {key}: {status}")
