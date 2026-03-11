"""
V10 Backtest Dashboard — Visual verification of predictions vs actual SL/TP outcomes.

Displays:
- 5M candlestick chart with prediction markers colored by win/loss/pending
- SL/TP zone lines for recent trades
- Equity curve (cumulative PnL)
- Prediction detail table

Usage:
    streamlit run backtest_dashboard.py
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="V10 Backtest Dashboard", layout="wide")

# Session state init
if 'auto_update' not in st.session_state:
    st.session_state['auto_update'] = True
if 'last_backfill_time' not in st.session_state:
    st.session_state['last_backfill_time'] = None

LOGS_DIR = Path(__file__).parent / "trading_logs"
BACKFILL_CSV = LOGS_DIR / "backfill_predictions.csv"

# SL/TP parameters (must match training)
SL_PCT = 2.0
TP_PCT = 4.0

DATA_DIR = Path(os.environ.get("DATA_DIR", ""))


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data(ttl=60)
def load_backfill_data():
    """Load backfill predictions CSV."""
    if not BACKFILL_CSV.exists():
        return None
    df = pd.read_csv(BACKFILL_CSV)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(ttl=60)
def load_data_service_predictions():
    """Load predictions from data service's persistent CSV."""
    if not DATA_DIR or not (DATA_DIR / "predictions" / "predictions.csv").exists():
        return None
    pred_path = DATA_DIR / "predictions" / "predictions.csv"
    df = pd.read_csv(pred_path)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    if 'source' not in df.columns:
        df['source'] = 'data_service'
    # Compute raw_signal from probabilities for threshold re-application
    if 'raw_signal' not in df.columns and all(c in df.columns for c in ['prob_no_trade', 'prob_long', 'prob_short']):
        raw_signals = []
        for _, row in df.iterrows():
            probs = [row['prob_no_trade'], row['prob_long'], row['prob_short']]
            pred_class = int(np.argmax(probs))
            raw_signals.append('LONG' if pred_class == 1 else ('SHORT' if pred_class == 2 else 'NO_TRADE'))
        df['raw_signal'] = raw_signals
    return df


@st.cache_data(ttl=60)
def load_live_trade_logs():
    """Load live bot trade logs (trades_*.csv), normalize columns to match backfill schema."""
    if not LOGS_DIR.exists():
        return None

    csv_files = sorted(LOGS_DIR.glob("trades_*.csv"))
    if not csv_files:
        return None

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
            # Normalize column names to match backfill schema
            rename = {}
            if 'timestamp' in df.columns and 'time' not in df.columns:
                rename['timestamp'] = 'time'
            if rename:
                df = df.rename(columns=rename)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
            if 'source' not in df.columns:
                df['source'] = 'live'
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _parse_kline_csv(path):
    """Parse a kline CSV file, normalizing columns and dtypes."""
    df = pd.read_csv(path)
    if 'time' in df.columns and 'Open Time' not in df.columns:
        df = df.rename(columns={'time': 'Open Time'})
    df['Open Time'] = pd.to_datetime(df['Open Time'], utc=True).dt.tz_localize(None)
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


@st.cache_data(ttl=60)
def load_klines_5m():
    """Load 5M kline data — merge data service + backfill for maximum coverage."""
    frames = []

    # Source 1: Data service klines (live, most recent)
    if DATA_DIR:
        ds_klines = DATA_DIR / "klines" / "ml_data_5M.csv"
        if ds_klines.exists():
            frames.append(_parse_kline_csv(ds_klines))

    # Source 2: Backfill klines (wider historical coverage)
    backfill_klines = LOGS_DIR / "backfill_klines_5m.csv"
    if backfill_klines.exists():
        frames.append(_parse_kline_csv(backfill_klines))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        # Dedup by timestamp, prefer data service (first) for overlapping candles
        merged = merged.sort_values('Open Time')
        merged = merged.drop_duplicates(subset='Open Time', keep='first')
        return merged.reset_index(drop=True)

    # Fall back to static historical data
    kline_path = Path(__file__).parent / "model_training" / "actual_data" / "ml_data_5M.csv"
    if not kline_path.exists():
        return None
    return _parse_kline_csv(kline_path)


def merge_predictions(backfill_df, live_df, data_service_df=None):
    """Merge backfill, live, and data service predictions, dedup by timestamp."""
    frames = []
    if backfill_df is not None and len(backfill_df) > 0:
        frames.append(backfill_df)
    if data_service_df is not None and len(data_service_df) > 0:
        frames.append(data_service_df)
    if live_df is not None and len(live_df) > 0:
        if 'signal' in live_df.columns:
            frames.append(live_df)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged['time'] = pd.to_datetime(merged['time'], utc=True).dt.tz_localize(None)
    # Dedup: prefer data_service (real-time predictions matching bot/telegram)
    # over backfill (retroactive, uses later kline data causing signal drift)
    source_priority = {'data_service': 0, 'backfill': 1, 'live': 2}
    merged['_priority'] = merged['source'].map(source_priority).fillna(3)
    merged = merged.sort_values(['time', '_priority'], ascending=[True, True])
    merged = merged.drop_duplicates(subset='time', keep='first')
    merged = merged.drop(columns=['_priority'])
    merged = merged.sort_values('time').reset_index(drop=True)
    return merged


def enrich_outcomes(predictions_df, klines_df):
    """Compute actual SL/TP outcomes for predictions missing them."""
    if predictions_df is None or klines_df is None:
        return predictions_df

    # Add outcome columns if missing
    if 'actual_outcome' not in predictions_df.columns:
        predictions_df['actual_outcome'] = 'Pending'
        predictions_df['actual_gain_pct'] = 0.0
        predictions_df['actual_hold_periods'] = 0

    # Find rows needing enrichment (trades without outcomes)
    needs_enrichment = (
        (predictions_df['signal'] != 'NO_TRADE') &
        (predictions_df['actual_outcome'].isin(['Pending', '']) | predictions_df['actual_outcome'].isna())
    )

    if not needs_enrichment.any():
        return predictions_df

    from data.target_labeling import _test_long_trade_fast, _test_short_trade_fast

    kl = klines_df.copy()
    if 'Open Time' in kl.columns:
        kl['time'] = kl['Open Time']
        kl = kl.drop(columns=['Open Time'])
    kl['time'] = pd.to_datetime(kl['time'], utc=True).dt.tz_localize(None)
    kl = kl.sort_values('time').reset_index(drop=True)

    highs = kl['High'].values
    lows = kl['Low'].values
    closes = kl['Close'].values
    times = kl['time'].values

    time_to_idx = {pd.Timestamp(times[i]): i for i in range(len(kl))}

    for idx in predictions_df.index[needs_enrichment]:
        row = predictions_df.loc[idx]
        t = pd.Timestamp(row['time']).tz_localize(None) if hasattr(pd.Timestamp(row['time']), 'tz') and pd.Timestamp(row['time']).tz else pd.Timestamp(row['time'])

        kl_idx = time_to_idx.get(t)
        if kl_idx is None:
            diffs = np.abs(times.astype('datetime64[ns]') - np.datetime64(t))
            kl_idx = int(np.argmin(diffs))

        entry_price = closes[kl_idx]
        remaining = len(closes) - kl_idx - 1
        if remaining < 1:
            continue

        max_hold = min(288, remaining)

        if row['signal'] == 'LONG':
            result = _test_long_trade_fast(highs, lows, closes, kl_idx, entry_price, SL_PCT, TP_PCT, max_hold)
        else:
            result = _test_short_trade_fast(highs, lows, closes, kl_idx, entry_price, SL_PCT, TP_PCT, max_hold)

        if result['outcome'] == 'Max_Hold' and remaining < 288:
            continue  # Still pending

        predictions_df.loc[idx, 'actual_outcome'] = result['outcome']
        predictions_df.loc[idx, 'actual_gain_pct'] = round(result['gain_pct'], 4)
        predictions_df.loc[idx, 'actual_hold_periods'] = result['hold_periods']

    return predictions_df


# ============================================================================
# Filtering
# ============================================================================

def apply_threshold_filter(df, threshold):
    """Re-apply threshold filter to predictions.

    Works on raw_signal: restores all raw signals first, then filters
    by threshold. Actual outcomes stay intact (tied to raw signal).
    """
    if df is None:
        return None
    df = df.copy()
    # Restore raw signals if available
    if 'raw_signal' in df.columns:
        df['signal'] = df['raw_signal']
    # For rows where confidence < threshold, downgrade to NO_TRADE
    mask = (df['signal'] != 'NO_TRADE') & (df['confidence'] < threshold)
    df.loc[mask, 'signal'] = 'NO_TRADE'
    return df


def apply_cooldown_filter(df, cooldown_candles):
    """Apply cooldown spacing to trade signals.

    Operates on the already threshold-filtered 'signal' column — does NOT
    restore raw_signal (that's done by apply_threshold_filter, which must
    run first). This ensures threshold and cooldown compose correctly.
    """
    if df is None:
        return df
    df = df.copy()

    if cooldown_candles <= 0:
        return df

    last_trade_idx = -cooldown_candles - 1

    for i in range(len(df)):
        if df.iloc[i]['signal'] != 'NO_TRADE':
            if i - last_trade_idx >= cooldown_candles:
                last_trade_idx = i
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 'NO_TRADE'
    return df


# ============================================================================
# Charts
# ============================================================================

def build_candlestick_chart(klines_df, predictions_df, show_sl_tp=True, show_pending=True):
    """Build Plotly candlestick chart with prediction markers.

    Marker legend:
    - WIN (TP_Hit): filled green triangle-up (LONG) / red triangle-down (SHORT)
    - LOSS (SL_Hit): orange hollow triangle
    - MAX_HOLD: gray diamond
    - PENDING: yellow triangle
    """
    fig = go.Figure()

    # Candlestick base
    fig.add_trace(go.Candlestick(
        x=klines_df['Open Time'],
        open=klines_df['Open'],
        high=klines_df['High'],
        low=klines_df['Low'],
        close=klines_df['Close'],
        name='BTCUSDT',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ))

    if predictions_df is None or len(predictions_df) == 0:
        _style_chart(fig)
        return fig

    trades = predictions_df[predictions_df['signal'] != 'NO_TRADE'].copy()
    if len(trades) == 0:
        _style_chart(fig)
        return fig

    # Map trade times to prices from klines
    kl_time_price = klines_df.set_index('Open Time')['Close'].to_dict()
    trades['price'] = trades['time'].map(
        lambda t: _nearest_price(t, kl_time_price, klines_df))

    # --- WIN markers (TP_Hit) ---
    tp_long = trades[(trades['actual_outcome'] == 'TP_Hit') & (trades['signal'] == 'LONG')]
    if len(tp_long) > 0:
        fig.add_trace(go.Scatter(
            x=tp_long['time'], y=tp_long['price'],
            mode='markers', name='WIN LONG',
            marker=dict(symbol='triangle-up', size=14, color='#00e676',
                        line=dict(width=1, color='white')),
            hovertemplate='LONG WIN<br>%{x}<br>Price: %{y:.2f}<extra></extra>',
        ))

    tp_short = trades[(trades['actual_outcome'] == 'TP_Hit') & (trades['signal'] == 'SHORT')]
    if len(tp_short) > 0:
        fig.add_trace(go.Scatter(
            x=tp_short['time'], y=tp_short['price'],
            mode='markers', name='WIN SHORT',
            marker=dict(symbol='triangle-down', size=14, color='#ff1744',
                        line=dict(width=1, color='white')),
            hovertemplate='SHORT WIN<br>%{x}<br>Price: %{y:.2f}<extra></extra>',
        ))

    # --- LOSS markers (SL_Hit) ---
    sl_trades = trades[trades['actual_outcome'] == 'SL_Hit']
    if len(sl_trades) > 0:
        symbols = ['triangle-up-open' if s == 'LONG' else 'triangle-down-open'
                    for s in sl_trades['signal']]
        fig.add_trace(go.Scatter(
            x=sl_trades['time'], y=sl_trades['price'],
            mode='markers', name='LOSS (SL Hit)',
            marker=dict(symbol=symbols, size=14, color='#ff9100',
                        line=dict(width=2, color='#ff9100')),
            hovertemplate='SL Hit<br>%{x}<br>Price: %{y:.2f}<extra></extra>',
        ))

    # --- MAX_HOLD markers ---
    mh_trades = trades[trades['actual_outcome'] == 'Max_Hold']
    if len(mh_trades) > 0:
        fig.add_trace(go.Scatter(
            x=mh_trades['time'], y=mh_trades['price'],
            mode='markers', name='Max Hold',
            marker=dict(symbol='diamond', size=10, color='#9e9e9e',
                        line=dict(width=1, color='white')),
            hovertemplate='Max Hold<br>%{x}<br>Price: %{y:.2f}<extra></extra>',
        ))

    # --- PENDING markers ---
    if show_pending:
        pending = trades[trades['actual_outcome'] == 'Pending']
        if len(pending) > 0:
            symbols = ['triangle-up' if s == 'LONG' else 'triangle-down'
                        for s in pending['signal']]
            fig.add_trace(go.Scatter(
                x=pending['time'], y=pending['price'],
                mode='markers', name='Pending',
                marker=dict(symbol=symbols, size=12, color='#ffd600',
                            line=dict(width=1, color='white')),
                hovertemplate='Pending<br>%{x}<br>Price: %{y:.2f}<extra></extra>',
            ))

    # --- SL/TP zone lines ---
    if show_sl_tp:
        _draw_sl_tp_zones(fig, trades, klines_df, max_show=5)

    _style_chart(fig)
    return fig


def _nearest_price(t, time_price_dict, klines_df):
    """Get close price at time t, or nearest available."""
    if t in time_price_dict:
        return time_price_dict[t]
    # Find nearest
    idx = klines_df['Open Time'].searchsorted(t)
    idx = min(idx, len(klines_df) - 1)
    return klines_df.iloc[idx]['Close']


def _draw_sl_tp_zones(fig, trades_df, klines_df, max_show=5):
    """Draw dashed SL/TP lines for the most recent N trade entries."""
    resolved = trades_df[trades_df['actual_outcome'].isin(['TP_Hit', 'SL_Hit', 'Max_Hold'])]
    if len(resolved) == 0:
        resolved = trades_df  # Show for pending too if no resolved

    recent = resolved.tail(max_show)

    for _, trade in recent.iterrows():
        entry_price = trade['price']
        t_start = trade['time']
        hold = trade.get('actual_hold_periods', 0)
        if pd.isna(hold) or hold == 0:
            hold = 60  # Default display width

        # Calculate end time
        t_end = t_start + timedelta(minutes=5 * int(hold))

        if trade['signal'] == 'LONG':
            tp_price = entry_price * (1 + TP_PCT / 100)
            sl_price = entry_price * (1 - SL_PCT / 100)
        else:  # SHORT
            tp_price = entry_price * (1 - TP_PCT / 100)
            sl_price = entry_price * (1 + SL_PCT / 100)

        # TP line (green dashed)
        fig.add_trace(go.Scatter(
            x=[t_start, t_end], y=[tp_price, tp_price],
            mode='lines', line=dict(color='#00e676', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ))
        # SL line (red dashed)
        fig.add_trace(go.Scatter(
            x=[t_start, t_end], y=[sl_price, sl_price],
            mode='lines', line=dict(color='#ff1744', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ))
        # Entry line (white dotted)
        fig.add_trace(go.Scatter(
            x=[t_start, t_end], y=[entry_price, entry_price],
            mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))


def _style_chart(fig):
    """Apply dark theme styling to chart."""
    fig.update_layout(
        template='plotly_dark',
        height=600,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date',
        ),
        yaxis=dict(title='Price (USDT)'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
        ),
        hovermode='x unified',
    )


def build_equity_curve(predictions_df):
    """Build equity curve from resolved trades."""
    if predictions_df is None:
        return None

    trades = predictions_df[predictions_df['signal'] != 'NO_TRADE'].copy()
    resolved = trades[~trades['actual_outcome'].isin(['Pending', 'No_Trade'])]

    if len(resolved) == 0:
        return None

    resolved = resolved.sort_values('time').reset_index(drop=True)
    resolved['cumulative_pnl'] = ((1 + resolved['actual_gain_pct'] / 100).cumprod() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=resolved['time'],
        y=resolved['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative PnL',
        line=dict(color='#00e676', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(0, 230, 118, 0.1)',
    ))

    # Color markers by win/loss
    wins = resolved[resolved['actual_outcome'] == 'TP_Hit']
    losses = resolved[resolved['actual_outcome'] == 'SL_Hit']

    if len(wins) > 0:
        fig.add_trace(go.Scatter(
            x=wins['time'], y=wins['cumulative_pnl'],
            mode='markers', name='Win',
            marker=dict(size=8, color='#00e676', symbol='circle'),
        ))
    if len(losses) > 0:
        fig.add_trace(go.Scatter(
            x=losses['time'], y=losses['cumulative_pnl'],
            mode='markers', name='Loss',
            marker=dict(size=8, color='#ff1744', symbol='circle'),
        ))

    fig.update_layout(
        template='plotly_dark',
        height=250,
        margin=dict(l=50, r=20, t=30, b=30),
        yaxis=dict(title='Cumulative PnL (%)', zeroline=True,
                   zerolinecolor='rgba(255,255,255,0.3)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
    )
    return fig


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(predictions_df):
    """Compute summary metrics from predictions."""
    if predictions_df is None or len(predictions_df) == 0:
        return {}

    trades = predictions_df[predictions_df['signal'] != 'NO_TRADE']
    resolved = trades[~trades['actual_outcome'].isin(['Pending', 'No_Trade'])]
    pending = trades[trades['actual_outcome'] == 'Pending']

    metrics = {
        'total_candles': len(predictions_df),
        'total_trades': len(trades),
        'pending': len(pending),
        'resolved': len(resolved),
    }

    if len(resolved) > 0:
        wins = (resolved['actual_outcome'] == 'TP_Hit').sum()
        metrics['win_rate'] = wins / len(resolved) * 100
        metrics['wins'] = int(wins)
        metrics['losses'] = len(resolved) - int(wins)
        metrics['total_pnl'] = ((1 + resolved['actual_gain_pct'] / 100).prod() - 1) * 100
        metrics['avg_gain'] = resolved['actual_gain_pct'].mean()

        # LONG/SHORT breakdown
        long_trades = resolved[resolved['signal'] == 'LONG']
        short_trades = resolved[resolved['signal'] == 'SHORT']

        if len(long_trades) > 0:
            long_wins = (long_trades['actual_outcome'] == 'TP_Hit').sum()
            metrics['long_count'] = len(long_trades)
            metrics['long_wr'] = long_wins / len(long_trades) * 100
        else:
            metrics['long_count'] = 0
            metrics['long_wr'] = 0

        if len(short_trades) > 0:
            short_wins = (short_trades['actual_outcome'] == 'TP_Hit').sum()
            metrics['short_count'] = len(short_trades)
            metrics['short_wr'] = short_wins / len(short_trades) * 100
        else:
            metrics['short_count'] = 0
            metrics['short_wr'] = 0

        # Profit factor
        gross_wins = resolved[resolved['actual_gain_pct'] > 0]['actual_gain_pct'].sum()
        gross_losses = abs(resolved[resolved['actual_gain_pct'] < 0]['actual_gain_pct'].sum())
        metrics['profit_factor'] = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    else:
        metrics['win_rate'] = 0
        metrics['wins'] = 0
        metrics['losses'] = 0
        metrics['total_pnl'] = 0
        metrics['avg_gain'] = 0
        metrics['long_count'] = 0
        metrics['long_wr'] = 0
        metrics['short_count'] = 0
        metrics['short_wr'] = 0
        metrics['profit_factor'] = 0

    return metrics


def build_threshold_comparison(predictions_raw, cooldown_candles, time_cutoff, selected_threshold):
    """Build threshold comparison table and bar chart across 11 threshold levels."""
    thresholds = [round(t, 2) for t in np.arange(0.40, 0.91, 0.05)]
    rows = []

    for t in thresholds:
        filtered = apply_threshold_filter(predictions_raw, t)
        filtered = apply_cooldown_filter(filtered, cooldown_candles)
        filtered = filtered[filtered['time'] >= time_cutoff].reset_index(drop=True)
        m = compute_metrics(filtered)

        pf = m.get('profit_factor', 0)
        pf_display = round(min(pf, 99.99), 2) if pf != float('inf') else 99.99

        rows.append({
            'Threshold': f'{t:.2f}',
            'Trades': m.get('resolved', 0),
            'Win Rate %': round(m.get('win_rate', 0), 1),
            'Profit Factor': pf_display,
            'Total PnL %': round(m.get('total_pnl', 0), 1),
            'LONG': f"{m.get('long_count', 0)} ({m.get('long_wr', 0):.0f}%)",
            'SHORT': f"{m.get('short_count', 0)} ({m.get('short_wr', 0):.0f}%)",
        })

    comp_df = pd.DataFrame(rows)

    # Bar chart — PF by threshold, selected threshold highlighted green
    colors = ['#00e676' if t == f'{selected_threshold:.2f}' else '#42a5f5'
              for t in comp_df['Threshold']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df['Threshold'],
        y=comp_df['Profit Factor'],
        marker_color=colors,
        hovertemplate=(
            'Threshold: %{x}<br>'
            'PF: %{y:.2f}<br>'
            'Trades: %{customdata[0]}<br>'
            'WR: %{customdata[1]:.1f}%<br>'
            'PnL: %{customdata[2]:+.1f}%<extra></extra>'
        ),
        customdata=comp_df[['Trades', 'Win Rate %', 'Total PnL %']].values,
    ))

    # Break-even line at PF=1.0
    fig.add_hline(y=1.0, line_dash='dash', line_color='rgba(255,255,255,0.4)',
                  annotation_text='Break-even (PF=1.0)')

    fig.update_layout(
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=20, t=30, b=30),
        yaxis=dict(title='Profit Factor'),
        xaxis=dict(title='Threshold', type='category'),
        showlegend=False,
    )

    return fig, comp_df


def _show_model_status(predictions_df, full_df, threshold):
    """Show model status info bar — last signal, peak confidence, signal-free streak."""
    if predictions_df is None or len(predictions_df) == 0:
        return

    trades_in_window = predictions_df[predictions_df['signal'] != 'NO_TRADE']
    has_trades = len(trades_in_window) > 0

    parts = []

    if has_trades:
        last = trades_in_window.iloc[-1]
        parts.append(f"Last signal in view: **{last['signal']}** @ {last['time'].strftime('%Y-%m-%d %H:%M')} "
                     f"(conf {last['confidence']:.2f})")
    else:
        # Look in full dataset for last signal across all time
        last_signal_row = None
        if full_df is not None and 'raw_signal' in full_df.columns:
            all_signals = full_df[full_df['raw_signal'].isin(['LONG', 'SHORT'])]
            if len(all_signals) > 0:
                last_signal_row = all_signals.iloc[-1]

        if last_signal_row is not None:
            sig_time = pd.Timestamp(last_signal_row['time']).tz_localize(None) if pd.Timestamp(last_signal_row['time']).tzinfo else pd.Timestamp(last_signal_row['time'])
            now = predictions_df['time'].max()
            hours_ago = (now - sig_time).total_seconds() / 3600
            parts.append(f"No signals in current view | Last signal: **{last_signal_row['raw_signal']}** "
                         f"@ {sig_time.strftime('%Y-%m-%d %H:%M')} ({hours_ago:.1f}h ago)")
        else:
            parts.append("No signals in current view or full dataset")

    # Peak confidence in visible window
    if 'prob_long' in predictions_df.columns and 'prob_short' in predictions_df.columns:
        max_long = predictions_df['prob_long'].max()
        max_short = predictions_df['prob_short'].max()
        parts.append(f"Peak confidence: LONG {max_long:.1%}, SHORT {max_short:.1%} "
                     f"(threshold {threshold:.0%})")

    # Candles evaluated
    parts.append(f"{len(predictions_df)} candles evaluated")

    st.info(" | ".join(parts))


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    st.title("V10 Backtest Dashboard")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")

        time_range = st.radio(
            "Time Range",
            ["1 Day", "3 Days", "1 Week", "1 Month"],
            index=2,
            horizontal=True,
        )
        time_range_hours = {"1 Day": 24, "3 Days": 72, "1 Week": 168, "1 Month": 720}[time_range]

        threshold = st.slider("Confidence Threshold", 0.40, 0.95, 0.50, 0.05)
        cooldown = st.slider("Cooldown (candles)", 0, 120, 60, 5)
        show_sl_tp = st.checkbox("Show SL/TP Zones", value=True)
        show_pending = st.checkbox("Show Pending Trades", value=True)

        st.divider()
        st.subheader("Data Updates")
        st.checkbox("Auto-update (5 min)", value=True, key='auto_update')

        if st.button("Backfill Now", use_container_width=True):
            with st.spinner("Running backfill (~30s)..."):
                success = _run_backfill_subprocess(hours=max(time_range_hours, 720))
                st.cache_data.clear()
            if success:
                st.success("Backfill complete!")
            else:
                st.error("Backfill failed")
            st.rerun()

        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.session_state.get('last_backfill_time'):
            st.caption(f"Last updated: {st.session_state['last_backfill_time'].strftime('%H:%M:%S')}")
        else:
            st.caption("No backfill run this session")

    # --- Load data ---
    backfill_df = load_backfill_data()
    live_df = load_live_trade_logs()
    data_service_df = load_data_service_predictions()
    predictions_raw = merge_predictions(backfill_df, live_df, data_service_df)

    if predictions_raw is None or len(predictions_raw) == 0:
        st.warning("No prediction data found. Run `python backfill_predictions.py` first.")
        st.code("python backfill_predictions.py --hours 72 --threshold 0.75", language="bash")
        return

    # --- Load klines (needed for both enrichment and chart) ---
    klines_5m = load_klines_5m()

    # Enrich outcomes for predictions missing them
    predictions_raw = enrich_outcomes(predictions_raw, klines_5m)

    # Apply filters
    predictions = apply_threshold_filter(predictions_raw, threshold)
    predictions = apply_cooldown_filter(predictions, cooldown)

    # Apply time range filter
    time_cutoff = predictions['time'].max() - timedelta(hours=time_range_hours)
    predictions = predictions[predictions['time'] >= time_cutoff].reset_index(drop=True)

    if klines_5m is not None:
        # Filter klines to prediction time window (with some padding)
        pred_start = predictions['time'].min() - timedelta(hours=2)
        pred_end = predictions['time'].max() + timedelta(hours=2)
        klines_window = klines_5m[
            (klines_5m['Open Time'] >= pred_start) &
            (klines_5m['Open Time'] <= pred_end)
        ].copy().reset_index(drop=True)
    else:
        klines_window = None

    # --- Metrics Row ---
    metrics = compute_metrics(predictions)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Trades", metrics.get('total_trades', 0))
    with col2:
        wr = metrics.get('win_rate', 0)
        st.metric("Win Rate", f"{wr:.1f}%",
                   delta=f"{wr - 50:.1f}pp" if wr > 0 else None,
                   delta_color="normal")
    with col3:
        pnl = metrics.get('total_pnl', 0)
        st.metric("Total PnL", f"{pnl:+.2f}%")
    with col4:
        st.metric("LONG WR",
                   f"{metrics.get('long_wr', 0):.1f}% ({metrics.get('long_count', 0)})")
    with col5:
        st.metric("SHORT WR",
                   f"{metrics.get('short_wr', 0):.1f}% ({metrics.get('short_count', 0)})")
    with col6:
        pf = metrics.get('profit_factor', 0)
        pf_str = f"{pf:.2f}" if pf < 100 else "Inf"
        st.metric("Profit Factor", pf_str)

    # --- Model Status Bar ---
    _show_model_status(predictions, predictions_raw, threshold)

    # --- Candlestick Chart ---
    if klines_window is not None and len(klines_window) > 0:
        st.subheader("Price Chart + Predictions")
        chart = build_candlestick_chart(
            klines_window, predictions,
            show_sl_tp=show_sl_tp, show_pending=show_pending
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning("No 5M kline data found for chart. "
                    "Ensure `model_training/actual_data/ml_data_5M.csv` exists.")

    # --- Equity Curve ---
    equity_fig = build_equity_curve(predictions)
    if equity_fig is not None:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_fig, use_container_width=True)

    # --- Threshold Comparison ---
    st.subheader("Threshold Comparison")
    comp_fig, comp_df = build_threshold_comparison(
        predictions_raw, cooldown, time_cutoff, threshold)
    st.plotly_chart(comp_fig, use_container_width=True)

    def _style_comparison(row):
        styles = [''] * len(row)
        is_selected = row['Threshold'] == f'{threshold:.2f}'
        if is_selected:
            styles = ['background-color: rgba(0, 230, 118, 0.2)'] * len(row)
        wr_idx = comp_df.columns.get_loc('Win Rate %')
        wr_val = row['Win Rate %']
        if wr_val >= 50:
            color = '#00e676'
        elif wr_val >= 33.3:
            color = '#ff9100'
        elif wr_val > 0:
            color = '#ff1744'
        else:
            color = ''
        if color:
            base = styles[wr_idx]
            styles[wr_idx] = f'{base}; color: {color}' if base else f'color: {color}'
        return styles

    styled_comp = comp_df.style.apply(_style_comparison, axis=1)
    st.dataframe(styled_comp, use_container_width=True, hide_index=True)
    st.caption(f"Cooldown: {cooldown} candles | "
               f"PF = gross wins / gross losses (higher = better, 1.0 = break-even)")

    # --- Prediction Table ---
    st.subheader("Trade Signals")
    trades = predictions[predictions['signal'] != 'NO_TRADE'].copy()

    if len(trades) > 0:
        # Compute entry/SL/TP from price column or klines
        if 'price' not in trades.columns and klines_window is not None:
            kl_time_price = klines_window.set_index('Open Time')['Close'].to_dict()
            trades['price'] = trades['time'].map(
                lambda t: _nearest_price(t, kl_time_price, klines_window))
        if 'price' in trades.columns:
            trades['entry'] = trades['price'].map(lambda p: f"${p:,.2f}" if p else "")
            trades['SL (-2%)'] = trades.apply(
                lambda r: f"${r['price'] * (0.98 if r['signal'] == 'LONG' else 1.02):,.2f}"
                if r['price'] else "", axis=1)
            trades['TP (+4%)'] = trades.apply(
                lambda r: f"${r['price'] * (1.04 if r['signal'] == 'LONG' else 0.96):,.2f}"
                if r['price'] else "", axis=1)

        display_cols = ['time', 'signal', 'confidence', 'entry', 'SL (-2%)', 'TP (+4%)',
                        'prob_long', 'prob_short',
                        'model_agreement', 'unanimous', 'actual_outcome',
                        'actual_gain_pct', 'actual_hold_periods']
        display_cols = [c for c in display_cols if c in trades.columns]

        # Style the outcome column
        def color_outcome(val):
            if val == 'TP_Hit':
                return 'color: #00e676'
            elif val == 'SL_Hit':
                return 'color: #ff1744'
            elif val == 'Pending':
                return 'color: #ffd600'
            elif val == 'Max_Hold':
                return 'color: #9e9e9e'
            return ''

        styled = trades[display_cols].style.applymap(
            color_outcome, subset=['actual_outcome'] if 'actual_outcome' in display_cols else [])

        st.dataframe(styled, use_container_width=True, height=400)
    else:
        # Informative empty-state message
        n_candles = len(predictions)
        # Find nearest-miss: highest confidence non-NO_TRADE raw signal that didn't pass threshold
        nearest_msg = ""
        if 'raw_signal' in predictions.columns and 'confidence' in predictions.columns:
            raw_trades = predictions[predictions['raw_signal'].isin(['LONG', 'SHORT'])]
            if len(raw_trades) > 0:
                best = raw_trades.loc[raw_trades['confidence'].idxmax()]
                nearest_msg = (f" Nearest miss: **{best['raw_signal']}** at "
                               f"{best['time'].strftime('%m-%d %H:%M')} "
                               f"with {best['confidence']:.2f} confidence "
                               f"(threshold: {threshold:.2f}).")
            else:
                nearest_msg = " The model predicted NO_TRADE for every candle in this window."

        suggestion = ""
        if time_range_hours <= 72:
            suggestion = " Try **1 Week** or **1 Month** view to see signals outside this window."

        st.info(f"No trade signals found. {n_candles} candles evaluated — model is running "
                f"but not confident enough to trade.{nearest_msg}{suggestion}")

    # --- Footer info ---
    st.divider()
    st.caption(
        f"Data: {predictions['time'].min().strftime('%Y-%m-%d %H:%M')} → "
        f"{predictions['time'].max().strftime('%Y-%m-%d %H:%M')} | "
        f"Threshold: {threshold} | Cooldown: {cooldown} candles | "
        f"SL: {SL_PCT}% | TP: {TP_PCT}%"
    )


# ============================================================================
# Auto-refresh: backfill + reload
# ============================================================================

def _run_backfill_subprocess(hours=24):
    """Run backfill_predictions.py to get fresh predictions."""
    script = str(Path(__file__).parent / "backfill_predictions.py")
    try:
        result = subprocess.run(
            [sys.executable, script, "--hours", str(hours), "--threshold", "0.50"],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).parent),
        )
        st.session_state['last_backfill_time'] = datetime.now()
        return result.returncode == 0
    except Exception:
        return False


@st.fragment(run_every=300)
def _auto_refresh():
    """Background fragment — re-runs backfill every 5 minutes to capture new 5M candles."""
    # Skip first render — only run on periodic reruns (every 5 min)
    if not st.session_state.get('_auto_refresh_initialized', False):
        st.session_state['_auto_refresh_initialized'] = True
        return
    if st.session_state.get('auto_update', True):
        _run_backfill_subprocess(hours=720)
        st.cache_data.clear()


if __name__ == "__main__":
    _auto_refresh()
    main()
