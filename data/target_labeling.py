import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_sl_tp_labels(price_data: pd.DataFrame,
                        sl_pct: float = 3.0,
                        tp_pct: float = 6.0,
                        max_hold_periods: int = 50,
                        price_col: str = 'Close',
                        high_col: str = 'High',
                        low_col: str = 'Low',
                        timestamp_col: str = 'Open Time') -> pd.DataFrame:
    """
    Create trading labels based on Stop Loss and Take Profit levels.

    For each candle, simulates both a LONG and SHORT trade forward in time.
    Labels: +1 (LONG TP hit), -1 (SHORT TP hit), 0 (No Trade).

    Args:
        price_data: DataFrame with OHLCV data
        sl_pct: Stop Loss percentage (default: 3%)
        tp_pct: Take Profit percentage (default: 6%)
        max_hold_periods: Maximum holding time (default: 50 periods)
        price_col: Close price column name
        high_col: High price column name
        low_col: Low price column name
        timestamp_col: Timestamp column name

    Returns:
        DataFrame with trading labels and outcomes
    """
    logger.info(f"Creating SL/TP Labels: SL={sl_pct}%, TP={tp_pct}%, Max Hold={max_hold_periods}")

    n = len(price_data)
    if n <= max_hold_periods:
        return pd.DataFrame()

    # Pre-extract arrays for speed (avoid repeated iloc)
    closes = price_data[price_col].values
    highs = price_data[high_col].values
    lows = price_data[low_col].values
    times = price_data[timestamp_col].values

    results = []

    for i in range(n - max_hold_periods):
        entry_price = closes[i]
        entry_time = times[i]

        # Test LONG
        long_result = _test_long_trade_fast(
            highs, lows, closes, i, entry_price, sl_pct, tp_pct, max_hold_periods
        )

        # Test SHORT
        short_result = _test_short_trade_fast(
            highs, lows, closes, i, entry_price, sl_pct, tp_pct, max_hold_periods
        )

        # Pick best label
        label, chosen = _pick_best(long_result, short_result)

        results.append({
            'timestamp': entry_time,
            'entry_price': entry_price,
            'exit_price': chosen['exit_price'],
            'label': label,
            'trade_type': {1: 'Long', -1: 'Short'}.get(label, 'No_Trade'),
            'outcome': chosen['outcome'],
            'gain_pct': chosen['gain_pct'],
            'hold_periods': chosen['hold_periods'],
            'exit_reason': chosen['exit_reason'],
            'sl_pct': sl_pct if label != 0 else 0,
            'tp_pct': tp_pct if label != 0 else 0,
            'risk_reward_ratio': tp_pct / sl_pct if label != 0 else 0,
            'entry_index': i,
        })

    labels_df = pd.DataFrame(results)

    # Pad remaining rows (insufficient future data)
    remaining = n - len(labels_df)
    if remaining > 0:
        pad = []
        for j in range(remaining):
            idx = len(labels_df) + j
            pad.append({
                'timestamp': times[idx],
                'entry_price': closes[idx],
                'exit_price': closes[idx],
                'label': 0,
                'trade_type': 'No_Trade',
                'outcome': 'Insufficient_Future_Data',
                'gain_pct': 0,
                'hold_periods': 0,
                'exit_reason': 'End_Of_Data',
                'sl_pct': 0,
                'tp_pct': 0,
                'risk_reward_ratio': 0,
                'entry_index': idx,
            })
        labels_df = pd.concat([labels_df, pd.DataFrame(pad)], ignore_index=True)

    return labels_df


def _pick_best(long_r: Dict, short_r: Dict):
    """Pick the best trade label from long/short results."""
    no_trade = {'outcome': 'No_Trade', 'gain_pct': 0, 'hold_periods': 0,
                'exit_reason': 'None', 'exit_price': 0}

    l_tp = long_r['outcome'] == 'TP_Hit'
    s_tp = short_r['outcome'] == 'TP_Hit'

    if l_tp and not s_tp:
        return 1, long_r
    if s_tp and not l_tp:
        return -1, short_r
    if l_tp and s_tp:
        # Both hit TP - choose the faster one
        if long_r['hold_periods'] <= short_r['hold_periods']:
            return 1, long_r
        return -1, short_r

    no_trade['exit_price'] = long_r.get('exit_price', 0)
    return 0, no_trade


def _test_long_trade_fast(highs, lows, closes, entry_idx, entry_price,
                          sl_pct, tp_pct, max_hold) -> Dict:
    """Test a long trade using numpy arrays for speed."""
    sl_price = entry_price * (1 - sl_pct / 100)
    tp_price = entry_price * (1 + tp_pct / 100)

    end = min(entry_idx + max_hold + 1, len(highs))
    for j in range(entry_idx + 1, end):
        h, l = highs[j], lows[j]

        # Check SL first to avoid same-bar bias
        if l <= sl_price:
            # If TP also hit in same bar, check which is more likely
            if h >= tp_price:
                # Ambiguous bar: use close to decide
                if closes[j] >= entry_price:
                    return _tp_result(entry_price, tp_price, j - entry_idx)
            return _sl_result(entry_price, sl_price, j - entry_idx, direction='long')

        if h >= tp_price:
            return _tp_result(entry_price, tp_price, j - entry_idx)

    # Max hold
    final_price = closes[entry_idx + max_hold] if entry_idx + max_hold < len(closes) else closes[-1]
    return {
        'outcome': 'Max_Hold',
        'gain_pct': (final_price - entry_price) / entry_price * 100,
        'hold_periods': max_hold,
        'exit_reason': 'Max_Hold_Reached',
        'exit_price': final_price,
    }


def _test_short_trade_fast(highs, lows, closes, entry_idx, entry_price,
                           sl_pct, tp_pct, max_hold) -> Dict:
    """Test a short trade using numpy arrays for speed."""
    sl_price = entry_price * (1 + sl_pct / 100)
    tp_price = entry_price * (1 - tp_pct / 100)

    end = min(entry_idx + max_hold + 1, len(highs))
    for j in range(entry_idx + 1, end):
        h, l = highs[j], lows[j]

        # Check SL first to avoid same-bar bias
        if h >= sl_price:
            if l <= tp_price:
                if closes[j] <= entry_price:
                    return _tp_result_short(entry_price, tp_price, j - entry_idx)
            return _sl_result(entry_price, sl_price, j - entry_idx, direction='short')

        if l <= tp_price:
            return _tp_result_short(entry_price, tp_price, j - entry_idx)

    final_price = closes[entry_idx + max_hold] if entry_idx + max_hold < len(closes) else closes[-1]
    return {
        'outcome': 'Max_Hold',
        'gain_pct': (entry_price - final_price) / entry_price * 100,
        'hold_periods': max_hold,
        'exit_reason': 'Max_Hold_Reached',
        'exit_price': final_price,
    }


def _tp_result(entry, tp_price, periods):
    return {
        'outcome': 'TP_Hit',
        'gain_pct': (tp_price - entry) / entry * 100,
        'hold_periods': periods,
        'exit_reason': 'Take_Profit',
        'exit_price': tp_price,
    }


def _tp_result_short(entry, tp_price, periods):
    return {
        'outcome': 'TP_Hit',
        'gain_pct': (entry - tp_price) / entry * 100,
        'hold_periods': periods,
        'exit_reason': 'Take_Profit',
        'exit_price': tp_price,
    }


def _sl_result(entry, sl_price, periods, direction='long'):
    if direction == 'long':
        gain = (sl_price - entry) / entry * 100
    else:
        gain = (entry - sl_price) / entry * 100
    return {
        'outcome': 'SL_Hit',
        'gain_pct': gain,
        'hold_periods': periods,
        'exit_reason': 'Stop_Loss',
        'exit_price': sl_price,
    }


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Compute Average True Range using EMA smoothing."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    # EMA smoothing
    atr = np.zeros(n)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def create_atr_labels(price_data: pd.DataFrame,
                      tp_pct: float = 4.0,
                      atr_sl_mult: float = 0.10,
                      max_hold_periods: int = 288,
                      atr_period: int = 14,
                      price_col: str = 'Close',
                      high_col: str = 'High',
                      low_col: str = 'Low',
                      timestamp_col: str = 'Open Time') -> pd.DataFrame:
    """
    Create trading labels with dynamic ATR-based SL and fixed TP.

    SL = atr_sl_mult * ATR(atr_period) at each candle (dynamic, tight).
    TP = tp_pct% fixed from entry.

    Returns DataFrame with columns: timestamp, label, entry_price, exit_price,
    sl_pct, atr_value, gain_pct, hold_periods, etc.
    """
    logger.info(f"Creating ATR Labels: TP={tp_pct}%, ATR_SL_mult={atr_sl_mult}, "
                f"ATR_period={atr_period}, Max Hold={max_hold_periods}")

    n = len(price_data)
    if n <= max_hold_periods + atr_period:
        return pd.DataFrame()

    closes = price_data[price_col].values
    highs = price_data[high_col].values
    lows = price_data[low_col].values
    times = price_data[timestamp_col].values

    atr = compute_atr(highs, lows, closes, period=atr_period)

    results = []
    for i in range(n - max_hold_periods):
        entry_price = closes[i]
        entry_time = times[i]
        atr_val = atr[i]

        if np.isnan(atr_val) or atr_val <= 0:
            results.append({
                'timestamp': entry_time, 'entry_price': entry_price,
                'exit_price': entry_price, 'label': 0,
                'trade_type': 'No_Trade', 'outcome': 'No_ATR',
                'gain_pct': 0, 'hold_periods': 0, 'exit_reason': 'No_ATR',
                'sl_pct': 0, 'tp_pct': 0, 'atr_value': 0,
                'risk_reward_ratio': 0, 'entry_index': i,
            })
            continue

        sl_amount = atr_sl_mult * atr_val
        sl_pct_actual = (sl_amount / entry_price) * 100

        long_result = _test_long_trade_fast(
            highs, lows, closes, i, entry_price, sl_pct_actual, tp_pct, max_hold_periods
        )
        short_result = _test_short_trade_fast(
            highs, lows, closes, i, entry_price, sl_pct_actual, tp_pct, max_hold_periods
        )

        label, chosen = _pick_best(long_result, short_result)

        results.append({
            'timestamp': entry_time,
            'entry_price': entry_price,
            'exit_price': chosen['exit_price'],
            'label': label,
            'trade_type': {1: 'Long', -1: 'Short'}.get(label, 'No_Trade'),
            'outcome': chosen['outcome'],
            'gain_pct': chosen['gain_pct'],
            'hold_periods': chosen['hold_periods'],
            'exit_reason': chosen['exit_reason'],
            'sl_pct': sl_pct_actual if label != 0 else 0,
            'tp_pct': tp_pct if label != 0 else 0,
            'atr_value': atr_val,
            'risk_reward_ratio': tp_pct / sl_pct_actual if label != 0 and sl_pct_actual > 0 else 0,
            'entry_index': i,
        })

    labels_df = pd.DataFrame(results)

    # Pad remaining rows
    remaining = n - len(labels_df)
    if remaining > 0:
        pad = []
        for j in range(remaining):
            idx = len(labels_df) + j
            pad.append({
                'timestamp': times[idx], 'entry_price': closes[idx],
                'exit_price': closes[idx], 'label': 0,
                'trade_type': 'No_Trade', 'outcome': 'Insufficient_Future_Data',
                'gain_pct': 0, 'hold_periods': 0, 'exit_reason': 'End_Of_Data',
                'sl_pct': 0, 'tp_pct': 0, 'atr_value': atr[idx] if idx < len(atr) else 0,
                'risk_reward_ratio': 0, 'entry_index': idx,
            })
        labels_df = pd.concat([labels_df, pd.DataFrame(pad)], ignore_index=True)

    dist = labels_df['label'].value_counts().to_dict()
    logger.info(f"ATR Labels: {len(labels_df)} rows | Distribution: {dist}")
    return labels_df


def get_labeling_summary(labels_df: pd.DataFrame) -> Dict:
    """Return labeling summary as a dict (for dashboard display)."""
    if len(labels_df) == 0:
        return {}

    trade_signals = labels_df[labels_df['label'] != 0]
    tp_hits = labels_df[labels_df['outcome'] == 'TP_Hit']
    n_trades = len(trade_signals)

    return {
        'total_periods': len(labels_df),
        'trade_signals': n_trades,
        'long_count': int((labels_df['label'] == 1).sum()),
        'short_count': int((labels_df['label'] == -1).sum()),
        'tp_hits': len(tp_hits),
        'sl_hits': int((labels_df['outcome'] == 'SL_Hit').sum()),
        'win_rate': len(tp_hits) / n_trades * 100 if n_trades > 0 else 0,
        'avg_gain': trade_signals['gain_pct'].mean() if n_trades > 0 else 0,
        'avg_hold': trade_signals['hold_periods'].mean() if n_trades > 0 else 0,
    }
