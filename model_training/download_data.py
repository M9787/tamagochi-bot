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

# Extended lookbacks for ML training — ALL available BTCUSDT data (~2400 days since Sept 2019)
ML_INTERVALS = {
    "5M":  (Client.KLINE_INTERVAL_5MINUTE, 2400),    # all available = ~691,200 candles
    "15M": (Client.KLINE_INTERVAL_15MINUTE, 2400),   # all available = ~230,400
    "30M": (Client.KLINE_INTERVAL_30MINUTE, 2400),   # all available = ~115,200
    "1H":  (Client.KLINE_INTERVAL_1HOUR, 2400),      # all available = ~57,600
    "2H":  (Client.KLINE_INTERVAL_2HOUR, 2400),      # all available = ~28,800
    "4H":  (Client.KLINE_INTERVAL_4HOUR, 2400),      # all available = ~14,400
    "6H":  (Client.KLINE_INTERVAL_6HOUR, 2400),      # all available = ~9,600
    "8H":  (Client.KLINE_INTERVAL_8HOUR, 2400),      # all available = ~7,200
    "12H": (Client.KLINE_INTERVAL_12HOUR, 2400),     # all available = ~4,800
    "1D":  (Client.KLINE_INTERVAL_1DAY, 2400),       # all available = ~2,400
    "3D":  (Client.KLINE_INTERVAL_3DAY, 2400),       # all available = ~800
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


def _get_existing_max_ts(label: str) -> int | None:
    """Read max timestamp from existing CSV. Returns epoch ms or None."""
    csv_path = ML_DATA_DIR / f"ml_data_{label}.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'Open Time' not in df.columns:
            return None
        ts = pd.to_datetime(df['Open Time'], utc=True).max()
        return int(ts.timestamp() * 1000)
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download ML training klines from Binance")
    parser.add_argument("--full", action="store_true",
                        help="Full re-download (default: incremental, append only new klines)")
    args = parser.parse_args()

    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)

    mode = "FULL" if args.full else "INCREMENTAL"
    print("=" * 60)
    print(f"ML DATA DOWNLOADER ({mode})")
    print(f"Output: {ML_DATA_DIR}")
    print("=" * 60)

    api_key, api_secret = read_api_keys(API_KEY_FILE)
    client = Client(api_key=api_key, api_secret=api_secret, tld=TLD)
    client.ping()
    print("[+] Binance connected\n")

    now_utc = datetime.now(timezone.utc)
    end_ts = ms(now_utc)

    for label, (interval, days_back) in ML_INTERVALS.items():
        fname = ML_DATA_DIR / f"ml_data_{label}.csv"
        full_start_ts = end_ts - days_back * 86_400_000

        if not args.full:
            existing_max = _get_existing_max_ts(label)
            if existing_max is not None:
                # Start 1ms after the last existing candle
                start_ts = existing_max + 1
                if start_ts >= end_ts:
                    print(f"[=] {SYMBOL} {label} — already up to date")
                    continue
                days_new = (end_ts - start_ts) / 86_400_000
                print(f"[>] {SYMBOL} {label} — incremental ({days_new:.1f} days new)")
            else:
                start_ts = full_start_ts
                print(f"[>] {SYMBOL} {label} — {days_back} days lookback (no existing data)")
        else:
            start_ts = full_start_ts
            print(f"[>] {SYMBOL} {label} — {days_back} days lookback")

        raw = fetch_klines(client, SYMBOL, interval, start_ts, end_ts)
        df_new = normalize(raw)

        if not args.full and fname.exists() and len(df_new) > 0:
            # Append to existing data
            df_existing = pd.read_csv(fname)
            if 'index' in df_existing.columns:
                df_existing = df_existing.drop(columns=['index'])
            df_existing['Open Time'] = pd.to_datetime(df_existing['Open Time'], utc=True)
            for c in ("Open", "High", "Low", "Close", "Volume"):
                if c in df_existing.columns:
                    df_existing[c] = pd.to_numeric(df_existing[c], errors="coerce")

            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['Open Time'], keep='last')
            df_merged = df_merged.sort_values('Open Time').reset_index(drop=True)
            df_merged.to_csv(fname, index=False)
            print(f"[+] +{len(df_new):,} new -> {len(df_merged):,} total -> {fname}\n")
        else:
            df_new.to_csv(fname, index=False)
            print(f"[+] {len(df_new):,} rows -> {fname}\n")

    print("=" * 60)
    print("[+] ALL DONE — ML training data ready")
    print("=" * 60)


if __name__ == "__main__":
    main()
