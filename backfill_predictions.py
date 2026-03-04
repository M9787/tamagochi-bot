"""
V10 Backfill Predictions — Batch-predict 72h of 5M candles and compute actual outcomes.

Downloads extended kline data, runs the full ETL+encode pipeline, predicts on each
5M row in the lookback window, applies cooldown, and simulates actual SL/TP outcomes.

Usage:
    python backfill_predictions.py                    # 72h, threshold=0.75
    python backfill_predictions.py --hours 48         # Custom lookback
    python backfill_predictions.py --threshold 0.80   # Higher precision
    python backfill_predictions.py --cooldown 60      # Custom cooldown (candles)
"""
import argparse
import logging
import sys
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training.live_predict import (
    load_production_models,
    download_live_klines_extended,
    run_live_etl,
    encode_live_features,
    batch_ensemble_predict,
)
from data.target_labeling import _test_long_trade_fast, _test_short_trade_fast
from core.config import TRADING_SL_PCT, TRADING_TP_PCT, TRADING_MAX_HOLD_CANDLES

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent / "trading_logs"
OUTPUT_FILE = LOGS_DIR / "backfill_predictions.csv"

# SL/TP parameters — imported from core/config.py (single source of truth)
SL_PCT = TRADING_SL_PCT
TP_PCT = TRADING_TP_PCT
MAX_HOLD = TRADING_MAX_HOLD_CANDLES


def apply_cooldown(predictions_df, cooldown_candles=60):
    """Apply cooldown — skip trade signals within cooldown_candles of previous signal.

    Args:
        predictions_df: DataFrame with 'signal' and 'time' columns, sorted by time
        cooldown_candles: minimum candles between independent trades

    Returns: DataFrame with cooldown applied (some trades become NO_TRADE)
    """
    df = predictions_df.copy()
    last_trade_idx = -cooldown_candles - 1  # Allow first trade

    cooled_signals = []
    for i, row in df.iterrows():
        if row['signal'] != 'NO_TRADE':
            idx_pos = df.index.get_loc(i)
            if idx_pos - last_trade_idx >= cooldown_candles:
                cooled_signals.append(row['signal'])
                last_trade_idx = idx_pos
            else:
                cooled_signals.append('NO_TRADE')
        else:
            cooled_signals.append('NO_TRADE')

    df['signal'] = cooled_signals
    return df


def compute_actual_outcomes(predictions_df, klines_5m):
    """Compute actual SL/TP outcomes for each prediction using forward price data.

    Args:
        predictions_df: DataFrame with 'time', 'signal' columns
        klines_5m: 5M kline DataFrame with 'time', 'Open', 'High', 'Low', 'Close'

    Returns: predictions_df with added columns:
        actual_outcome, actual_gain_pct, actual_hold_periods
    """
    kl = klines_5m.sort_values('time').reset_index(drop=True)
    highs = kl['High'].values
    lows = kl['Low'].values
    closes = kl['Close'].values
    times = kl['time'].values

    # Build time → index map
    time_to_idx = {}
    for idx_pos in range(len(kl)):
        t = pd.Timestamp(times[idx_pos])
        time_to_idx[t] = idx_pos

    outcomes = []
    gains = []
    holds = []

    for _, row in predictions_df.iterrows():
        signal = row['signal']

        if signal == 'NO_TRADE':
            outcomes.append('No_Trade')
            gains.append(0.0)
            holds.append(0)
            continue

        t = pd.Timestamp(row['time'])
        idx = time_to_idx.get(t)

        if idx is None:
            # Find nearest index
            diffs = np.abs(times.astype('datetime64[ns]') - np.datetime64(t))
            idx = int(np.argmin(diffs))

        entry_price = closes[idx]

        # Check forward data availability
        remaining = len(closes) - idx - 1
        if remaining < 1:
            outcomes.append('Pending')
            gains.append(0.0)
            holds.append(0)
            continue

        # Use available data (may be less than MAX_HOLD)
        effective_max_hold = min(MAX_HOLD, remaining)

        if signal == 'LONG':
            result = _test_long_trade_fast(
                highs, lows, closes, idx, entry_price,
                SL_PCT, TP_PCT, effective_max_hold
            )
        else:  # SHORT
            result = _test_short_trade_fast(
                highs, lows, closes, idx, entry_price,
                SL_PCT, TP_PCT, effective_max_hold
            )

        # If trade didn't hit SL/TP and we had truncated data, mark as Pending
        if result['outcome'] == 'Max_Hold' and remaining < MAX_HOLD:
            outcomes.append('Pending')
            gains.append(0.0)
            holds.append(0)
            continue

        outcomes.append(result['outcome'])
        gains.append(round(result['gain_pct'], 4))
        holds.append(result['hold_periods'])

    predictions_df = predictions_df.copy()
    predictions_df['actual_outcome'] = outcomes
    predictions_df['actual_gain_pct'] = gains
    predictions_df['actual_hold_periods'] = holds
    return predictions_df


def run_backfill(hours=72, threshold=0.75, cooldown_candles=60):
    """Run the full backfill pipeline.

    Returns: DataFrame with predictions + actual outcomes, also saved to CSV.
    """
    t0 = time.time()

    # Step 1: Load models
    print(f"[1/7] Loading production models...")
    models, metadata = load_production_models()
    feature_names = metadata['feature_names']
    print(f"  {len(models)} models, {len(feature_names)} features")

    # Step 2: Download extended klines
    # 1400 bars = ~4.9 days for 5M. Scale up for longer lookbacks.
    bars_5m = max(1400, int(hours / 5 * 60) + 500)  # hours * 12 candles/hr + warm-up
    print(f"[2/7] Downloading klines (5M: {bars_5m} bars)...")
    t_dl = time.time()
    klines_dict = download_live_klines_extended(bars_5m=bars_5m)
    print(f"  Download: {time.time() - t_dl:.1f}s")

    # NOTE: Incomplete candle filtering now happens inside download_live_klines_extended()
    # (Open Time + candle duration > now check). No need for Close Time filter here.

    # Step 3: ETL
    print(f"[3/7] Running ETL...")
    t_etl = time.time()
    decomposed = run_live_etl(klines_dict)
    print(f"  ETL: {time.time() - t_etl:.1f}s")

    # Step 4: Encode features
    print(f"[4/7] Encoding features...")
    t_enc = time.time()
    features_df = encode_live_features(klines_dict, decomposed)
    print(f"  Encode: {time.time() - t_enc:.1f}s, shape={features_df.shape}")

    # Step 5: Filter to lookback window
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    features_df['time'] = pd.to_datetime(features_df['time'])
    mask = features_df['time'] >= cutoff
    features_window = features_df[mask].copy().reset_index(drop=True)
    print(f"[5/7] Filtered to last {hours}h: {len(features_window)} rows "
          f"({features_window['time'].min()} -> {features_window['time'].max()})")

    if len(features_window) == 0:
        print("  ERROR: No rows in lookback window. Check data download.")
        return None

    # Step 6: Batch predict
    print(f"[6/7] Batch predicting (threshold={threshold})...")
    t_pred = time.time()
    predictions = batch_ensemble_predict(
        models, features_window, feature_names, threshold=threshold
    )
    n_raw_trades = (predictions['signal'] != 'NO_TRADE').sum()
    print(f"  Predict: {time.time() - t_pred:.1f}s, {n_raw_trades} raw signals")

    # Compute raw_signal as pre-threshold argmax (matching data_service/layers.py).
    # This is the model's top class regardless of threshold, allowing the dashboard
    # to re-apply any threshold dynamically via its slider.
    raw_signals = []
    for _, row in predictions.iterrows():
        probs = [row['prob_no_trade'], row['prob_long'], row['prob_short']]
        pred_class = int(np.argmax(probs))
        if pred_class in (1, 2):  # LONG=1, SHORT=2
            raw_signals.append('LONG' if pred_class == 1 else 'SHORT')
        else:
            raw_signals.append('NO_TRADE')
    predictions['raw_signal'] = raw_signals

    # Step 7: Compute actual outcomes on RAW signals (before cooldown)
    # This way the CSV has outcomes for all 26 raw signals, and the dashboard
    # can re-apply any cooldown and still show correct outcomes.
    print(f"[7/7] Computing actual SL/TP outcomes...")
    t_out = time.time()
    klines_5m = klines_dict['5M'].copy()
    klines_5m['time'] = pd.to_datetime(klines_5m['time'])
    predictions = compute_actual_outcomes(predictions, klines_5m)
    print(f"  Outcomes: {time.time() - t_out:.1f}s")

    # Apply cooldown for summary stats only
    cooled = apply_cooldown(predictions, cooldown_candles)
    n_cooled_trades = (cooled['signal'] != 'NO_TRADE').sum()
    print(f"  After cooldown ({cooldown_candles} candles): {n_cooled_trades} trades")

    # Add metadata
    predictions['source'] = 'backfill'

    # Save CSV
    LOGS_DIR.mkdir(exist_ok=True)
    predictions.to_csv(OUTPUT_FILE, index=False)

    # Save 5M klines for dashboard chart
    klines_out = LOGS_DIR / "backfill_klines_5m.csv"
    klines_5m.to_csv(klines_out, index=False)
    print(f"  Klines saved: {klines_out} ({len(klines_5m)} rows)")

    # Print summary
    total_time = time.time() - t0
    trades = predictions[predictions['signal'] != 'NO_TRADE']
    resolved = trades[~trades['actual_outcome'].isin(['Pending', 'No_Trade'])]

    print(f"\n{'='*60}")
    print(f"  Backfill Complete — {total_time:.1f}s total")
    print(f"{'='*60}")
    print(f"  Period: {predictions['time'].min()} -> {predictions['time'].max()}")
    print(f"  Total 5M candles: {len(predictions)}")
    print(f"  Trade signals: {len(trades)} (after cooldown)")

    if len(trades) > 0:
        longs = (trades['signal'] == 'LONG').sum()
        shorts = (trades['signal'] == 'SHORT').sum()
        print(f"  LONG: {longs} | SHORT: {shorts}")

        outcome_dist = trades['actual_outcome'].value_counts().to_dict()
        print(f"  Outcomes: {outcome_dist}")

        if len(resolved) > 0:
            wins = (resolved['actual_outcome'] == 'TP_Hit').sum()
            wr = wins / len(resolved) * 100
            total_gain = resolved['actual_gain_pct'].sum()
            avg_gain = resolved['actual_gain_pct'].mean()
            print(f"  Resolved: {len(resolved)} trades")
            print(f"  Win Rate: {wr:.1f}% ({wins}/{len(resolved)})")
            print(f"  Total Gain: {total_gain:+.2f}%")
            print(f"  Avg Gain: {avg_gain:+.2f}%")

    print(f"\n  Saved: {OUTPUT_FILE}")
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="V10 Backfill Predictions — batch predict + actual outcomes")
    parser.add_argument("--hours", type=int, default=72,
                        help="Lookback hours (default: 72)")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold (default: 0.75)")
    parser.add_argument("--cooldown", type=int, default=60,
                        help="Cooldown candles between trades (default: 60)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    run_backfill(hours=args.hours, threshold=args.threshold,
                 cooldown_candles=args.cooldown)


if __name__ == "__main__":
    main()
