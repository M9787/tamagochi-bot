#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance Kline Downloader (multi-timeframe, robust)
- One place to set SYMBOL, OUTPUT_DIR, API_KEY_FILE
- Retries with backoff on transient API errors / rate limits
- Proper pagination for long ranges
- Clean CSVs: only ['Open Time','Close'], UTC timestamps
"""

from __future__ import annotations
import time
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import pandas as pd
from binance import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# ========= CONFIG =========
SYMBOL = "BTCUSDT"
TLD = "com"

# Where to write the CSVs
OUTPUT_DIR = Path(r"C:\Users\Useer\Desktop\THE BEST PROJECT\NEW ALGORITHM CONSTRUCTION\Live Data\data")

# API key file: first line = key, second line = secret
API_KEY_FILE = Path(r"C:\Users\Useer\Desktop\THE BEST PROJECT\NEW ALGORITHM CONSTRUCTION\Api key.txt")

# Timeframes and lookbacks (days) -> filename suffix
# Keep your original lookbacks; adjust freely.
INTERVALS = {
    "3D":  (Client.KLINE_INTERVAL_3DAY, 1200),
    "1D":  (Client.KLINE_INTERVAL_1DAY, 700),
    "12H": (Client.KLINE_INTERVAL_12HOUR, 200),
    "8H":  (Client.KLINE_INTERVAL_8HOUR, 180),
    "6H":  (Client.KLINE_INTERVAL_6HOUR, 180),
    "4H":  (Client.KLINE_INTERVAL_4HOUR, 100),
    "2H":  (Client.KLINE_INTERVAL_2HOUR, 50),
    "1H":  (Client.KLINE_INTERVAL_1HOUR, 30),
    "30M": (Client.KLINE_INTERVAL_30MINUTE, 30),
    "15M": (Client.KLINE_INTERVAL_15MINUTE, 15),
    "5M":  (Client.KLINE_INTERVAL_5MINUTE, 5),
}
# ==========================


def read_api_keys(path: Path) -> Tuple[str, str]:
    print("[*] Reading API keys...")
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("API key file must have key on line 1 and secret on line 2.")
    return lines[0], lines[1]


def init_client(api_key: str, api_secret: str) -> Client:
    print("[*] Initializing Binance client...")
    client = Client(api_key=api_key, api_secret=api_secret, tld=TLD)
    # Light ping to validate
    _ = client.ping()
    print("[+] Binance client ready.")
    return client


def ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines_with_retry(
    client: Client,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 6,
) -> List[list]:
    """
    Robust paginator w/ exponential backoff.
    """
    print(f"[>] Fetching {symbol} {interval} from {start_ms} -> {end_ms}")
    all_rows: List[list] = []
    next_start = start_ms
    backoff = 1.0

    while True:
        try:
            batch = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(next_start),
                end_str=str(end_ms),
                limit=limit,  # respected by python-binance for klines
            )
            if not batch:
                # No more data
                break

            all_rows.extend(batch)

            # Advance start to last open time + 1 ms to avoid duplicates
            last_open_time = batch[-1][0]
            next_start = last_open_time + 1

            # Progress log every ~10k rows
            if len(all_rows) % 10000 < len(batch):
                print(f"   …fetched {len(all_rows)} rows so far")

            if last_open_time >= end_ms:
                break

            # Reset backoff after a successful call
            backoff = 1.0

        except (BinanceRequestException, BinanceAPIException) as e:
            if max_retries <= 0:
                raise
            print(f"[!] API error: {e}. Retrying in {backoff:.1f}s ({max_retries} left)...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 32.0)
            max_retries -= 1

        except Exception as e:
            # Unknown error - surface it clearly
            print(f"[-] Unexpected error: {e}")
            raise

    print(f"[+] Done: {len(all_rows)} rows.")
    return all_rows


def normalize_klines(raw: List[list]) -> pd.DataFrame:
    """
    Convert raw kline rows to a tidy DataFrame.
    """
    if not raw:
        return pd.DataFrame(columns=["Open Time", "Open", "High", "Low", "Close", "Volume"])

    cols = [
        "Open Time", "Open", "High", "Low", "Close",
        "Volume", "Close Time", "Quote Asset Volume",
        "Number of Trades", "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume", "Ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)

    # Types
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep OHLCV; sort just in case
    out = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]].sort_values("Open Time").reset_index(drop=True)
    return out


def ensure_dir(path: Path) -> None:
    if not path.exists():
        print(f"[*] Creating output directory: {path}")
        path.mkdir(parents=True, exist_ok=True)


def main():
    try:
        ensure_dir(OUTPUT_DIR)
        api_key, api_secret = read_api_keys(API_KEY_FILE)
        client = init_client(api_key, api_secret)

        now_utc = datetime.now(timezone.utc)
        end_ts = ms(now_utc)

        for label, (interval, days_back) in INTERVALS.items():
            print("\n" + "=" * 70)
            print(f"[>] {SYMBOL} | {label} ({interval}) | lookback={days_back} days")

            start_ts = end_ts - days_back * 86_400_000  # days → ms

            raw = fetch_klines_with_retry(
                client=client,
                symbol=SYMBOL,
                interval=interval,
                start_ms=start_ts,
                end_ms=end_ts,
            )
            df = normalize_klines(raw)

            # Filename matches your original convention
            fname = OUTPUT_DIR / f"testing_data_{label}.csv"
            df.reset_index().to_csv(fname, index=False)
            print(f"[+] Saved {len(df):,} rows -> {fname}")

        print("\n[+] All timeframes exported successfully.")

    except FileNotFoundError as e:
        print(f"[-] Path not found: {e}")
    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"[-] Binance API error: {e}")
    except Exception as e:
        print(f"[-] Unexpected failure: {e}")
        raise


if __name__ == "__main__":
    main()
