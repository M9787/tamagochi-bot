"""
ML Data Downloader: Download extended historical data for model training.

Separate from the main data extraction - saves to model_training/data/
Downloads 30 days of 5M data + all TFs with extended lookbacks.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd
from binance import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from core.config import API_KEY_FILE

SYMBOL = "BTCUSDT"
TLD = "com"

# Save ML training data separately
ML_DATA_DIR = Path(__file__).parent / "actual_data"

# Extended lookbacks for ML training — 6 YEARS for all TFs
ML_INTERVALS = {
    "5M":  (Client.KLINE_INTERVAL_5MINUTE, 2190),    # 6 years = ~630,720 candles
    "15M": (Client.KLINE_INTERVAL_15MINUTE, 2190),   # 6 years = ~210,240
    "30M": (Client.KLINE_INTERVAL_30MINUTE, 2190),   # 6 years = ~105,120
    "1H":  (Client.KLINE_INTERVAL_1HOUR, 2190),      # 6 years = ~52,560
    "2H":  (Client.KLINE_INTERVAL_2HOUR, 2190),      # 6 years = ~26,280
    "4H":  (Client.KLINE_INTERVAL_4HOUR, 2190),      # 6 years = ~13,140
    "6H":  (Client.KLINE_INTERVAL_6HOUR, 2190),      # 6 years = ~8,760
    "8H":  (Client.KLINE_INTERVAL_8HOUR, 2190),      # 6 years = ~6,570
    "12H": (Client.KLINE_INTERVAL_12HOUR, 2190),     # 6 years = ~4,380
    "1D":  (Client.KLINE_INTERVAL_1DAY, 2190),       # 6 years = ~2,190
    "3D":  (Client.KLINE_INTERVAL_3DAY, 2190),       # 6 years = ~730
}


def read_api_keys(path: Path) -> Tuple[str, str]:
    """Read API keys — env vars first, fallback to file."""
    env_key = os.environ.get("BINANCE_KEY", "") or os.environ.get("BINANCE_TESTNET_KEY", "")
    env_secret = os.environ.get("BINANCE_SECRET", "") or os.environ.get("BINANCE_TESTNET_SECRET", "")
    if env_key and env_secret:
        return env_key, env_secret
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("API key file must have key on line 1 and secret on line 2.")
    return lines[0], lines[1]


def ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(client, symbol, interval, start_ms, end_ms, max_retries=6):
    all_rows = []
    next_start = start_ms
    backoff = 1.0

    while True:
        try:
            batch = client.get_historical_klines(
                symbol=symbol, interval=interval,
                start_str=str(next_start), end_str=str(end_ms), limit=1000,
            )
            if not batch:
                break

            all_rows.extend(batch)
            last_open_time = batch[-1][0]
            next_start = last_open_time + 1

            if len(all_rows) % 5000 < len(batch):
                print(f"   ...{len(all_rows)} rows")

            if last_open_time >= end_ms:
                break
            backoff = 1.0

        except (BinanceRequestException, BinanceAPIException) as e:
            if max_retries <= 0:
                raise
            print(f"  [!] API error: {e}. Retry in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 32.0)
            max_retries -= 1

    return all_rows


def normalize(raw):
    if not raw:
        return pd.DataFrame(columns=["Open Time", "Open", "High", "Low", "Close", "Volume"])

    cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].sort_values("Open Time").reset_index(drop=True)


def main():
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ML DATA DOWNLOADER")
    print(f"Output: {ML_DATA_DIR}")
    print("=" * 60)

    api_key, api_secret = read_api_keys(API_KEY_FILE)
    client = Client(api_key=api_key, api_secret=api_secret, tld=TLD)
    client.ping()
    print("[+] Binance connected\n")

    now_utc = datetime.now(timezone.utc)
    end_ts = ms(now_utc)

    for label, (interval, days_back) in ML_INTERVALS.items():
        print(f"[>] {SYMBOL} {label} — {days_back} days lookback")
        start_ts = end_ts - days_back * 86_400_000

        raw = fetch_klines(client, SYMBOL, interval, start_ts, end_ts)
        df = normalize(raw)

        fname = ML_DATA_DIR / f"ml_data_{label}.csv"
        df.reset_index().to_csv(fname, index=False)
        print(f"[+] {len(df):,} rows -> {fname}\n")

    print("=" * 60)
    print("[+] ALL DONE — ML training data ready")
    print("=" * 60)


if __name__ == "__main__":
    main()
