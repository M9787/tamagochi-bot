#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit Dashboard for Cryptocurrency Trading Signal System

Enhanced Signal Board with:
1. Signal Distribution (pie/donut)
2. Probability Gauges
3. Heatmaps (timeframe × window)
4. Crossing Signals visualization
5. Reversal Signals panel
6. Acceleration Linechart (w30 per timeframe with filtering)

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import subprocess
import sys
from pathlib import Path
import time

from core.processor import TimeframeProcessor
from core.signal_logic import SignalLogic, SignalDirection, AccelerationZone, CalendarDataBuilder, ReversalType
from core.config import (
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    CHART_COLORS,
    DATA_DIR
)
from data.target_labeling import create_sl_tp_labels, get_labeling_summary

# Timeframe age-group segmentation
TF_GROUPS = {
    "Youngs": ["5M", "15M", "30M"],
    "Adults": ["1H", "2H", "4H"],
    "Balzaks": ["6H", "8H", "12H"],
    "Grans": ["1D", "3D"],
}
TF_GROUP_NAMES = list(TF_GROUPS.keys())  # ["Youngs", "Adults", "Balzaks", "Grans"]


def filter_aggregated_by_tfs(aggregated, active_tfs):
    """Create a filtered copy of AggregatedSignal with only active timeframes.

    Returns a lightweight namespace with the same attributes so render functions
    work unchanged.
    """
    from dataclasses import dataclass, field
    tfs_set = set(active_tfs)
    filtered_sigs = [s for s in aggregated.signals if s.timeframe in tfs_set]
    long_c = sum(1 for s in filtered_sigs if s.direction == SignalDirection.LONG and not s.is_crossing)
    short_c = sum(1 for s in filtered_sigs if s.direction == SignalDirection.SHORT and not s.is_crossing)
    neutral_c = sum(1 for s in filtered_sigs if s.direction == SignalDirection.NEUTRAL and not s.is_crossing)
    total = long_c + short_c + neutral_c

    # Determine dominant direction from filtered set
    if long_c >= short_c and long_c >= neutral_c:
        final_dir = SignalDirection.LONG
    elif short_c >= long_c and short_c >= neutral_c:
        final_dir = SignalDirection.SHORT
    else:
        final_dir = SignalDirection.NEUTRAL

    # Determine quality
    if total > 0:
        dom = max(long_c, short_c, neutral_c)
        pct = dom / total
        quality = "VERY_HIGH" if pct >= 0.8 else "HIGH" if pct >= 0.6 else "MEDIUM" if pct >= 0.4 else "LOW"
    else:
        quality = "LOW"

    # Return object with same interface
    class FilteredAgg:
        pass
    fa = FilteredAgg()
    fa.signals = filtered_sigs
    fa.total_signals = total
    fa.long_count = long_c
    fa.short_count = short_c
    fa.neutral_count = neutral_c
    fa.final_direction = final_dir
    fa.convergence_score = max(long_c, short_c, neutral_c)
    fa.quality = quality
    fa.timestamp = aggregated.timestamp
    return fa


# Page configuration
st.set_page_config(
    page_title="BTC Signal Board",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Initialize session state for auto-refresh
if 'last_data_refresh' not in st.session_state:
    st.session_state.last_data_refresh = None
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if 'refresh_interval_minutes' not in st.session_state:
    st.session_state.refresh_interval_minutes = 5
if 'is_refreshing' not in st.session_state:
    st.session_state.is_refreshing = False

# Auto-refresh interval in seconds
AUTO_REFRESH_SECONDS = 5 * 60  # 5 minutes


def run_data_extraction() -> bool:
    """
    Run the Binance data extraction script to fetch fresh klines.

    Returns:
        True if extraction succeeded, False otherwise
    """
    script_path = Path(__file__).parent / "data" / "downloader.py"

    if not script_path.exists():
        st.error(f"Data extraction script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(script_path.parent)
        )

        if result.returncode == 0:
            return True
        else:
            st.error(f"Data extraction failed: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        st.error("Data extraction timed out (>5 minutes)")
        return False
    except Exception as e:
        st.error(f"Data extraction error: {str(e)}")
        return False


def check_and_trigger_auto_refresh() -> bool:
    """
    Check if auto-refresh is due and trigger data extraction if needed.

    Returns:
        True if refresh was triggered, False otherwise
    """
    if not st.session_state.auto_refresh_enabled:
        return False

    now = datetime.now()
    last_refresh = st.session_state.last_data_refresh

    # First run or refresh interval exceeded
    if last_refresh is None:
        return True

    elapsed = (now - last_refresh).total_seconds()
    if elapsed >= AUTO_REFRESH_SECONDS:
        return True

    return False


def perform_full_refresh():
    """Perform full data refresh: extract data + clear cache."""
    st.session_state.is_refreshing = True

    with st.spinner("🔄 Downloading fresh data from Binance..."):
        success = run_data_extraction()

    if success:
        # Clear all caches to force reload
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.last_data_refresh = datetime.now()
        st.session_state.is_refreshing = False
        st.success("✅ Data refreshed successfully!")
        time.sleep(1)  # Brief pause to show success message
        st.rerun()
    else:
        st.session_state.is_refreshing = False


def get_theme_css(dark_mode: bool) -> str:
    """Generate CSS based on theme."""
    if dark_mode:
        return """
        <style>
            .main-signal {
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                margin: 10px 0;
            }
            .signal-long { background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); border: 3px solid #2ecc71; }
            .signal-short { background: linear-gradient(135deg, #4a1a1a 0%, #6b2d2d 100%); border: 3px solid #e74c3c; }
            .signal-neutral { background: linear-gradient(135deg, #3d3d3d 0%, #4a4a4a 100%); border: 3px solid #f39c12; }
            .metric-box {
                background-color: #1e1e1e;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 5px 0;
            }
            .stMetric { background-color: #262626; padding: 10px; border-radius: 8px; }
            div[data-testid="stMetricValue"] { font-size: 28px; }
            .reversal-card {
                background-color: #2a2a2a;
                padding: 10px;
                border-radius: 8px;
                margin: 5px 0;
                border-left: 4px solid;
            }
            .reversal-peak { border-left-color: #e74c3c; }
            .reversal-valley { border-left-color: #2ecc71; }
            .reversal-zero { border-left-color: #f39c12; }
            .price-ticker {
                background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid #0f3460;
            }
        </style>
        """
    else:
        return """
        <style>
            .main-signal {
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                margin: 10px 0;
            }
            .signal-long { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 3px solid #28a745; }
            .signal-short { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 3px solid #dc3545; }
            .signal-neutral { background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); border: 3px solid #ffc107; }
            .metric-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 5px 0;
            }
            .stMetric { background-color: #e9ecef; padding: 10px; border-radius: 8px; }
            div[data-testid="stMetricValue"] { font-size: 28px; color: #212529; }
            .reversal-card {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 8px;
                margin: 5px 0;
                border-left: 4px solid;
            }
            .reversal-peak { border-left-color: #dc3545; }
            .reversal-valley { border-left-color: #28a745; }
            .reversal-zero { border-left-color: #ffc107; }
            .price-ticker {
                background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
        </style>
        """


# Apply theme CSS
st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)


@st.cache_data(ttl=60)
def get_btc_price() -> Optional[Dict]:
    """Fetch current BTC price from Binance API."""
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": "BTCUSDT"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume'])
            }
    except Exception:
        pass
    return None


@st.fragment(run_every=60)
def render_price_ticker():
    """Render real-time BTC price ticker in header. Auto-refreshes every 60s."""
    price_data = get_btc_price()

    if price_data:
        change_color = "#2ecc71" if price_data['change_24h'] >= 0 else "#e74c3c"
        change_symbol = "+" if price_data['change_24h'] >= 0 else ""

        st.markdown(f"""
        <div class="price-ticker">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 14px; color: #888;">BTC/USDT</span>
                    <span style="font-size: 28px; font-weight: bold; margin-left: 10px;">${price_data['price']:,.2f}</span>
                    <span style="color: {change_color}; font-size: 16px; margin-left: 10px;">{change_symbol}{price_data['change_24h']:.2f}%</span>
                </div>
                <div style="text-align: right; font-size: 12px; color: #888;">
                    <div>24h High: ${price_data['high_24h']:,.2f}</div>
                    <div>24h Low: ${price_data['low_24h']:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="price-ticker">
            <span style="color: #888;">Unable to fetch price data</span>
        </div>
        """, unsafe_allow_html=True)


@st.cache_resource(ttl=300)
def get_processor() -> TimeframeProcessor:
    """Load and process data (cached)."""
    processor = TimeframeProcessor()
    processor.load_all_data()
    processor.process_all()
    return processor


@st.cache_resource(ttl=300)
def get_signal_analysis(_processor: TimeframeProcessor):
    """Run advanced signal analysis (cached).

    Args:
        _processor: Pre-loaded TimeframeProcessor to reuse (underscore prefix tells Streamlit to not hash it)
    """
    logic = SignalLogic(processor=_processor)
    aggregated = logic.run_analysis()
    return logic, aggregated


# ============================================================================
# CACHED HEATMAP MATRIX COMPUTATIONS
# These extract matrix data once from signals, avoiding repeated iteration
# ============================================================================

@st.cache_data(ttl=300)
def compute_signal_matrix(_signals_hash: str, signals_data: List[dict], active_tfs: Optional[Tuple[str, ...]] = None) -> Tuple[List[List[int]], List[List[str]]]:
    """Compute signal heatmap matrix from signals data.

    Args:
        _signals_hash: Hash for cache invalidation (underscore prefix = not hashed by Streamlit)
        signals_data: List of signal dicts with timeframe, window_size, direction, is_crossing
        active_tfs: Tuple of active timeframes (must be tuple for Streamlit caching)

    Returns:
        Tuple of (value_matrix, text_matrix)
    """
    tfs = list(active_tfs) if active_tfs else TIMEFRAME_ORDER
    matrix = []
    text_matrix = []
    for tf in tfs:
        row = []
        text_row = []
        for ws in WINDOW_SIZES:
            sig = next((s for s in signals_data
                       if s['timeframe'] == tf and s['window_size'] == ws and not s['is_crossing']), None)
            if sig:
                direction = sig['direction']
                if direction == "LONG":
                    row.append(1)
                    text_row.append("L")
                elif direction == "SHORT":
                    row.append(-1)
                    text_row.append("S")
                else:
                    row.append(0)
                    text_row.append("N")
            else:
                row.append(0)
                text_row.append("N")
        matrix.append(row)
        text_matrix.append(text_row)
    return matrix, text_matrix


@st.cache_data(ttl=300)
def compute_reversal_matrix(_signals_hash: str, signals_data: List[dict], active_tfs: Optional[Tuple[str, ...]] = None) -> Tuple[List[List[int]], List[List[str]]]:
    """Compute reversal heatmap matrix from signals data."""
    tfs = list(active_tfs) if active_tfs else TIMEFRAME_ORDER
    matrix = []
    text_matrix = []
    for tf in tfs:
        row = []
        text_row = []
        for ws in WINDOW_SIZES:
            sig = next((s for s in signals_data
                       if s['timeframe'] == tf and s['window_size'] == ws and not s['is_crossing']), None)
            if sig and sig['reversal_type'] != "NONE":
                row.append(1)
                # P = Peak (potential SHORT), B = Bottom (potential LONG)
                rev_abbr = {
                    "PEAK": "P",
                    "BOTTOM": "B"
                }.get(sig['reversal_type'], "R")
                text_row.append(rev_abbr)
            else:
                row.append(0)
                text_row.append("-")
        matrix.append(row)
        text_matrix.append(text_row)
    return matrix, text_matrix


@st.cache_data(ttl=300)
def compute_slope_matrix(_signals_hash: str, signals_data: List[dict], active_tfs: Optional[Tuple[str, ...]] = None) -> Tuple[List[List[float]], List[List[str]], float, float]:
    """Compute slope heatmap matrix from signals data."""
    tfs = list(active_tfs) if active_tfs else TIMEFRAME_ORDER
    matrix = []
    text_matrix = []
    for tf in tfs:
        row = []
        text_row = []
        for ws in WINDOW_SIZES:
            sig = next((s for s in signals_data
                       if s['timeframe'] == tf and s['window_size'] == ws and not s['is_crossing']), None)
            if sig and 'slope_f' in sig.get('details', {}):
                slope = sig['details']['slope_f']
                row.append(slope)
                text_row.append(f"{slope:.4f}")
            else:
                row.append(0)
                text_row.append("-")
        matrix.append(row)
        text_matrix.append(text_row)

    # Calculate vmin/vmax for symmetric color scaling
    flat_values = [v for row in matrix for v in row if v != 0]
    if flat_values:
        vmin = min(flat_values)
        vmax = max(flat_values)
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1

    return matrix, text_matrix, vmin, vmax


@st.cache_data(ttl=300)
def compute_acceleration_matrix(_signals_hash: str, signals_data: List[dict], active_tfs: Optional[Tuple[str, ...]] = None) -> Tuple[List[List[float]], List[List[str]], float, float]:
    """Compute acceleration heatmap matrix from signals data."""
    tfs = list(active_tfs) if active_tfs else TIMEFRAME_ORDER
    matrix = []
    text_matrix = []
    for tf in tfs:
        row = []
        text_row = []
        for ws in WINDOW_SIZES:
            sig = next((s for s in signals_data
                       if s['timeframe'] == tf and s['window_size'] == ws and not s['is_crossing']), None)
            if sig and 'acceleration' in sig.get('details', {}):
                accel = sig['details']['acceleration']
                row.append(accel)
                text_row.append(f"{accel:.3f}")
            else:
                row.append(0)
                text_row.append("-")
        matrix.append(row)
        text_matrix.append(text_row)

    # Calculate vmin/vmax for symmetric color scaling
    flat_values = [v for row in matrix for v in row if v != 0]
    if flat_values:
        vmin = min(flat_values)
        vmax = max(flat_values)
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1

    return matrix, text_matrix, vmin, vmax


def extract_signals_data(aggregated) -> Tuple[List[dict], str]:
    """Extract signal data into a cacheable format.

    Args:
        aggregated: AggregatedSignal object

    Returns:
        Tuple of (signals_data list, hash string for cache key)
    """
    signals_data = []
    for sig in aggregated.signals:
        signals_data.append({
            'timeframe': sig.timeframe,
            'window_size': sig.window_size,
            'direction': sig.direction.value,
            'is_crossing': sig.is_crossing,
            'reversal_type': sig.reversal_type.value,
            'details': sig.details
        })
    # Create a simple hash based on signal count and timestamp
    signals_hash = f"{len(signals_data)}_{aggregated.timestamp.isoformat()}"
    return signals_data, signals_hash


def render_main_signal(aggregated):
    """Render the big main signal indicator."""
    signal = aggregated.final_direction.value
    quality = aggregated.quality

    if signal == "LONG":
        color, bg_class, arrow = "#2ecc71", "signal-long", "▲"
    elif signal == "SHORT":
        color, bg_class, arrow = "#e74c3c", "signal-short", "▼"
    else:
        color, bg_class, arrow = "#f39c12", "signal-neutral", "◆"

    convergence_pct = aggregated.convergence_score / aggregated.total_signals * 100

    st.markdown(f"""
    <div class="main-signal {bg_class}">
        <h1 style="color: {color}; font-size: 64px; margin: 0; font-weight: bold;">{arrow} {signal}</h1>
        <h2 style="color: white; margin: 10px 0;">Quality: {quality}</h2>
        <p style="color: #aaa; font-size: 18px;">
            Convergence: {aggregated.convergence_score}/{aggregated.total_signals} ({convergence_pct:.0f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_gauges(aggregated):
    """Render probability gauges for each direction."""
    st.subheader("📊 Signal Probabilities")

    total = aggregated.total_signals
    long_pct = aggregated.long_count / total * 100
    short_pct = aggregated.short_count / total * 100
    neutral_pct = aggregated.neutral_count / total * 100

    # Create gauge chart
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        horizontal_spacing=0.1
    )

    # LONG gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=long_pct,
        title={'text': "LONG", 'font': {'size': 20, 'color': '#2ecc71'}},
        number={'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "#1a1a1a",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 30], 'color': '#1a2f1a'},
                {'range': [30, 60], 'color': '#1a3f1a'},
                {'range': [60, 100], 'color': '#1a4f2a'}
            ],
            'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.8, 'value': long_pct}
        }
    ), row=1, col=1)

    # NEUTRAL gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=neutral_pct,
        title={'text': "NEUTRAL", 'font': {'size': 20, 'color': '#f39c12'}},
        number={'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#f39c12"},
            'bgcolor': "#1a1a1a",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 30], 'color': '#2f2a1a'},
                {'range': [30, 60], 'color': '#3f3a1a'},
                {'range': [60, 100], 'color': '#4f4a2a'}
            ],
        }
    ), row=1, col=2)

    # SHORT gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=short_pct,
        title={'text': "SHORT", 'font': {'size': 20, 'color': '#e74c3c'}},
        number={'suffix': '%', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#e74c3c"},
            'bgcolor': "#1a1a1a",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 30], 'color': '#2f1a1a'},
                {'range': [30, 60], 'color': '#3f1a1a'},
                {'range': [60, 100], 'color': '#4f2a2a'}
            ],
        }
    ), row=1, col=3)

    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )

    st.plotly_chart(fig, width='stretch')


def render_signal_distribution(aggregated):
    """Render signal distribution donut chart."""
    st.subheader("📈 Signal Distribution")

    # Distribution data
    labels = ['LONG', 'NEUTRAL', 'SHORT']
    values = [aggregated.long_count, aggregated.neutral_count, aggregated.short_count]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=14,
        pull=[0.05 if v == max(values) else 0 for v in values]
    )])

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    # Add center text
    fig.add_annotation(
        text=f"<b>{aggregated.total_signals}</b><br>Signals",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )

    st.plotly_chart(fig, width='stretch')


def render_signal_heatmap(aggregated, signals_data: List[dict], signals_hash: str, active_tfs=None):
    """Render timeframe × window heatmap using pre-computed cached matrix."""
    st.subheader("🔥 Signal Heatmap")
    tfs = active_tfs or TIMEFRAME_ORDER
    tfs_tuple = tuple(tfs)

    # Use cached matrix computation
    matrix, text_matrix = compute_signal_matrix(signals_hash, signals_data, tfs_tuple)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"w{ws}" for ws in WINDOW_SIZES],
        y=tfs,
        colorscale=[
            [0, '#e74c3c'],      # -1 = SHORT (red)
            [0.5, '#3d3d3d'],    # 0 = NEUTRAL (gray)
            [1, '#2ecc71']       # 1 = LONG (green)
        ],
        zmin=-1,
        zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate="Timeframe: %{y}<br>Window: %{x}<br>Signal: %{text}<extra></extra>",
        showscale=False
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Window Size",
        yaxis_title="Timeframe",
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, width='stretch')


def render_reversal_heatmap(aggregated, signals_data: List[dict], signals_hash: str, active_tfs=None):
    """Render timeframe × window heatmap for angle reversals using pre-computed cached matrix."""
    st.subheader("🔄 Reversal Heatmap")
    tfs = active_tfs or TIMEFRAME_ORDER
    tfs_tuple = tuple(tfs)

    # Use cached matrix computation
    matrix, text_matrix = compute_reversal_matrix(signals_hash, signals_data, tfs_tuple)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"w{ws}" for ws in WINDOW_SIZES],
        y=tfs,
        colorscale=[
            [0, '#2d2d2d'],      # 0 = No reversal (dark)
            [1, '#9b59b6']       # 1 = Reversal (purple)
        ],
        zmin=0,
        zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hovertemplate="Timeframe: %{y}<br>Window: %{x}<br>Reversal: %{text}<extra></extra>",
        showscale=False
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Window Size",
        yaxis_title="Timeframe",
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, width='stretch')
    st.caption("P=Peak (potential SHORT), B=Bottom (potential LONG)")


def render_slope_heatmap(aggregated, signals_data: List[dict], signals_hash: str, active_tfs=None):
    """Render timeframe × window heatmap for forward slope values using pre-computed cached matrix."""
    st.subheader("📈 Forward Slope Heatmap")
    tfs = active_tfs or TIMEFRAME_ORDER
    tfs_tuple = tuple(tfs)

    # Use cached matrix computation
    matrix, text_matrix, vmin, vmax = compute_slope_matrix(signals_hash, signals_data, tfs_tuple)

    # Create heatmap with diverging colorscale
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"w{ws}" for ws in WINDOW_SIZES],
        y=tfs,
        colorscale=[
            [0, '#e74c3c'],      # Negative (red)
            [0.5, '#2d2d2d'],    # Zero (neutral)
            [1, '#2ecc71']       # Positive (green)
        ],
        zmin=vmin,
        zmax=vmax,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hovertemplate="Timeframe: %{y}<br>Window: %{x}<br>Slope: %{z:.6f}<extra></extra>",
        colorbar=dict(
            title="Slope",
            titleside="right",
            tickformat=".4f"
        )
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=80, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Window Size",
        yaxis_title="Timeframe",
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, width='stretch')
    st.caption("🟢 Positive slope (bullish) | 🔴 Negative slope (bearish)")


def render_acceleration_heatmap(aggregated, signals_data: List[dict], signals_hash: str, active_tfs=None):
    """Render timeframe × window heatmap for acceleration values using pre-computed cached matrix."""
    st.subheader("⚡ Acceleration Heatmap")
    tfs = active_tfs or TIMEFRAME_ORDER
    tfs_tuple = tuple(tfs)

    # Use cached matrix computation
    matrix, text_matrix, vmin, vmax = compute_acceleration_matrix(signals_hash, signals_data, tfs_tuple)

    # Create heatmap with diverging colorscale (orange for intensity)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"w{ws}" for ws in WINDOW_SIZES],
        y=tfs,
        colorscale=[
            [0, '#3498db'],      # Negative accel (blue - decelerating)
            [0.5, '#2d2d2d'],    # Zero (neutral)
            [1, '#f39c12']       # Positive accel (orange - accelerating)
        ],
        zmin=vmin,
        zmax=vmax,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hovertemplate="Timeframe: %{y}<br>Window: %{x}<br>Acceleration: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title="Accel",
            titleside="right",
            tickformat=".3f"
        )
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=80, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Window Size",
        yaxis_title="Timeframe",
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, width='stretch')
    st.caption("🟠 Accelerating | 🔵 Decelerating")


def render_crossing_signals_chart(aggregated):
    """Render crossing signals visualization."""
    st.subheader("✖️ Window Crossing Signals")

    crossings = [s for s in aggregated.signals if s.is_crossing]

    if not crossings:
        st.info("No crossing signals detected")
        return

    # Prepare data for visualization
    crossing_data = []
    for sig in crossings:
        crossing_data.append({
            'Timeframe': sig.timeframe,
            'Windows': f"w{sig.crossing_windows[0]}-w{sig.crossing_windows[1]}",
            'Direction': sig.direction.value,
            'Strength': sig.strength,
            'Color': '#2ecc71' if sig.direction == SignalDirection.LONG else '#e74c3c'
        })

    df = pd.DataFrame(crossing_data)

    # Create bar chart
    fig = go.Figure()

    for direction in ['LONG', 'SHORT']:
        df_dir = df[df['Direction'] == direction]
        if len(df_dir) > 0:
            fig.add_trace(go.Bar(
                x=df_dir['Timeframe'],
                y=df_dir['Strength'],
                name=direction,
                marker_color='#2ecc71' if direction == 'LONG' else '#e74c3c',
                text=df_dir['Windows'],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>%{text}<br>Strength: %{y:.2f}<extra></extra>"
            ))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        barmode='group',
        xaxis_title="Timeframe",
        yaxis_title="Strength",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, width='stretch')

    # Summary counts
    long_count = len([c for c in crossings if c.direction == SignalDirection.LONG])
    short_count = len([c for c in crossings if c.direction == SignalDirection.SHORT])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("LONG Crossings", long_count, delta_color="normal")
    with col2:
        st.metric("SHORT Crossings", short_count, delta_color="inverse")


def render_reversal_signals(logic, aggregated, active_tfs=None):
    """Render reversal signals panel (peaks, valleys, zero crossings)."""
    st.subheader("🔄 Reversal Signals")

    # Collect reversal info from signals
    reversals = []
    for sig in aggregated.signals:
        if not sig.is_crossing:
            reversal_type = None
            # Compare with ReversalType enum values (PEAK, BOTTOM, NONE)
            if sig.reversal_type == ReversalType.PEAK:
                reversal_type = "Peak"
            elif sig.reversal_type == ReversalType.BOTTOM:
                reversal_type = "Bottom"

            if reversal_type:
                reversals.append({
                    'Timeframe': sig.timeframe,
                    'Window': f"w{sig.window_size}",
                    'Type': reversal_type,
                    'Direction': sig.direction.value,
                    'Strength': sig.strength
                })

    if not reversals:
        st.info("No reversal signals detected")
        return

    df = pd.DataFrame(reversals)

    # Count by type (Peak and Bottom based on 5-point pattern detection)
    type_counts = df['Type'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peaks (potential SHORT)", type_counts.get('Peak', 0))
    with col2:
        st.metric("Bottoms (potential LONG)", type_counts.get('Bottom', 0))

    # Show top reversals
    st.markdown("**Recent Reversals:**")

    # Group by timeframe for display
    tfs = active_tfs or TIMEFRAME_ORDER
    for tf in tfs[:6]:  # Show top 6 timeframes
        tf_reversals = df[df['Timeframe'] == tf]
        if len(tf_reversals) > 0:
            types = tf_reversals['Type'].tolist()
            dirs = tf_reversals['Direction'].tolist()
            summary = ", ".join([f"{t}({d[0]})" for t, d in zip(types, dirs)])

            # Color based on dominant direction
            long_count = dirs.count('LONG')
            short_count = dirs.count('SHORT')
            if long_count > short_count:
                st.success(f"**{tf}**: {summary}")
            elif short_count > long_count:
                st.error(f"**{tf}**: {summary}")
            else:
                st.warning(f"**{tf}**: {summary}")


def render_acceleration_zones(aggregated):
    """Render acceleration zone distribution."""
    st.subheader("⚡ Acceleration Zones")

    zones = {}
    for sig in aggregated.signals:
        zone = sig.acceleration_zone.value
        zones[zone] = zones.get(zone, 0) + 1

    zone_order = ["VERY_DISTANT", "DISTANT", "BASELINE", "CLOSE", "VERY_CLOSE"]
    zone_labels = ["Very Distant", "Distant", "Baseline", "Close", "Very Close"]
    zone_colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]

    values = [zones.get(z, 0) for z in zone_order]

    fig = go.Figure(data=[go.Bar(
        x=zone_labels,
        y=values,
        marker_color=zone_colors,
        text=values,
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    )])

    fig.update_layout(
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Zone",
        yaxis_title="Count",
        showlegend=False
    )

    st.plotly_chart(fig, width='stretch')
    st.caption("💚 Distant/Very Distant = Strong signals | ❤️ Close/Very Close = Weak signals")


def render_acceleration_linechart(processor, selected_timeframes, show_price=True):
    """Render acceleration linechart for w30 across selected timeframes with price overlay."""
    st.subheader("📉 Acceleration Chart (w30) with Price Overlay")

    if not selected_timeframes:
        st.warning("Select at least one timeframe")
        return

    # Create figure with secondary y-axis for price
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = px.colors.qualitative.Set2
    price_data = None

    for i, tf in enumerate(selected_timeframes):
        if tf in processor.results:
            # Get w30 data (stored as "df" in processor.py)
            label = "df"
            if label in processor.results[tf]:
                df = processor.results[tf][label]
                if 'acceleration' in df.columns and 'time' in df.columns:
                    # Get last 100 points for cleaner chart
                    df_plot = df.tail(100)

                    fig.add_trace(
                        go.Scattergl(
                            x=df_plot['time'],
                            y=df_plot['acceleration'],
                            name=f"{tf} Accel",
                            mode='lines',
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate=f"<b>{tf}</b><br>Time: %{{x}}<br>Accel: %{{y:.4f}}<extra></extra>"
                        ),
                        secondary_y=False
                    )

                    # Store price data from the first/largest timeframe for overlay
                    if show_price and price_data is None and 'actual' in df.columns:
                        price_data = df_plot[['time', 'actual']].copy()
                        price_data['price'] = price_data['actual'] ** 2  # Convert back from sqrt

    # Add price overlay on secondary y-axis
    if show_price and price_data is not None:
        fig.add_trace(
            go.Scattergl(
                x=price_data['time'],
                y=price_data['price'],
                name="BTC Price",
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.4)', width=1),
                hovertemplate="<b>Price</b><br>Time: %{x}<br>$%{y:,.2f}<extra></extra>"
            ),
            secondary_y=True
        )

    # Add zero line for acceleration
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        height=450,
        margin=dict(l=60, r=60, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Acceleration", secondary_y=False)
    fig.update_yaxes(title_text="BTC Price ($)", secondary_y=True)

    st.plotly_chart(fig, width='stretch')


def render_angle_chart_with_confidence(processor, selected_timeframe):
    """Render angle chart with confidence bands for a single timeframe."""
    st.caption(f"Timeframe: {selected_timeframe}")

    if selected_timeframe not in processor.results:
        st.warning(f"No data for {selected_timeframe}")
        return

    fig = go.Figure()
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    for i, (label, df) in enumerate(processor.results[selected_timeframe].items()):
        if 'angle' not in df.columns or 'time' not in df.columns:
            continue

        df_plot = df.tail(100)
        ws = df.attrs.get('window_size', label)

        # Calculate confidence bands (mean +/- std)
        angle_mean = df_plot['angle'].rolling(window=20, min_periods=1).mean()
        angle_std = df_plot['angle'].rolling(window=20, min_periods=1).std()
        upper_band = angle_mean + angle_std
        lower_band = angle_mean - angle_std

        # Add confidence band (shaded area)
        fig.add_trace(go.Scatter(
            x=pd.concat([df_plot['time'], df_plot['time'][::-1]]),
            y=pd.concat([upper_band, lower_band[::-1]]),
            fill='toself',
            fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
            name=f'w{ws} band'
        ))

        # Add main angle line (WebGL for performance)
        fig.add_trace(go.Scattergl(
            x=df_plot['time'],
            y=df_plot['angle'],
            name=f"w{ws}",
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"<b>w{ws}</b><br>Time: %{{x}}<br>Angle: %{{y:.2f}}°<extra></extra>"
        ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Time",
        yaxis_title="Angle (degrees)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, width='stretch')


def render_signal_history_timeline(aggregated, active_tfs=None):
    """Render horizontal timeline showing signal changes."""
    tfs = active_tfs or TIMEFRAME_ORDER

    # Group signals by timeframe and get latest direction
    timeline_data = []
    for tf in tfs:
        tf_signals = [s for s in aggregated.signals if s.timeframe == tf and not s.is_crossing]
        if tf_signals:
            # Use the signal from window 30 (df) as the main indicator
            main_signal = next((s for s in tf_signals if s.window_size == 30), tf_signals[0])
            timeline_data.append({
                'timeframe': tf,
                'direction': main_signal.direction.value,
                'strength': main_signal.strength,
                'color': '#2ecc71' if main_signal.direction == SignalDirection.LONG else
                         '#e74c3c' if main_signal.direction == SignalDirection.SHORT else '#808080'
            })

    if not timeline_data:
        st.info("No signal history available")
        return

    # Create timeline visualization
    fig = go.Figure()

    for i, item in enumerate(timeline_data):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            marker=dict(
                size=30 + item['strength'] * 20,
                color=item['color'],
                line=dict(color='white', width=2)
            ),
            text=[item['direction'][0]],  # L, S, or N
            textposition='middle center',
            textfont=dict(color='white', size=14),
            name=item['timeframe'],
            hovertemplate=f"<b>{item['timeframe']}</b><br>{item['direction']}<br>Strength: {item['strength']:.2f}<extra></extra>"
        ))

    # Add timeframe labels below
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=False,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(timeline_data))),
            ticktext=[item['timeframe'] for item in timeline_data],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, 0.5]
        )
    )

    st.plotly_chart(fig, width='stretch')

    # Legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **LONG**")
    with col2:
        st.markdown("🔴 **SHORT**")
    with col3:
        st.markdown("⚪ **NEUTRAL**")


def render_confluence_score(aggregated, active_tfs=None):
    """Render multi-timeframe confluence score."""

    tfs = active_tfs or TIMEFRAME_ORDER
    tfs_set = set(tfs)

    # Count agreements by direction
    long_tfs = set()
    short_tfs = set()
    neutral_tfs = set()

    for sig in aggregated.signals:
        if not sig.is_crossing and sig.timeframe in tfs_set:
            if sig.direction == SignalDirection.LONG:
                long_tfs.add(sig.timeframe)
            elif sig.direction == SignalDirection.SHORT:
                short_tfs.add(sig.timeframe)
            else:
                neutral_tfs.add(sig.timeframe)

    total_tfs = len(tfs)
    dominant_direction = aggregated.final_direction.value
    dominant_count = len(long_tfs) if dominant_direction == "LONG" else len(short_tfs) if dominant_direction == "SHORT" else len(neutral_tfs)

    # Display confluence meter
    confluence_pct = dominant_count / total_tfs * 100

    col1, col2 = st.columns([2, 1])

    with col1:
        # Progress bar style confluence meter
        color = '#2ecc71' if dominant_direction == "LONG" else '#e74c3c' if dominant_direction == "SHORT" else '#f39c12'
        st.markdown(f"""
        <div style="background-color: #1e1e1e; border-radius: 10px; padding: 15px;">
            <h3 style="color: {color}; margin: 0;">{dominant_count}/{total_tfs} Timeframes Confirm {dominant_direction}</h3>
            <div style="background-color: #333; border-radius: 5px; height: 20px; margin-top: 10px;">
                <div style="background-color: {color}; width: {confluence_pct}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <p style="color: #aaa; margin-top: 5px;">{confluence_pct:.0f}% Agreement</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("LONG TFs", len(long_tfs))
        st.metric("SHORT TFs", len(short_tfs))
        st.metric("NEUTRAL TFs", len(neutral_tfs))


def render_timeframe_matrix(aggregated, active_tfs=None):
    """Render compact timeframe signal matrix."""
    st.subheader("📋 Signal Matrix")
    tfs = active_tfs or TIMEFRAME_ORDER

    matrix_data = []

    for tf in tfs:
        tf_signals = [s for s in aggregated.signals if s.timeframe == tf and not s.is_crossing]

        row = {"TF": tf}
        for ws in WINDOW_SIZES:
            sig = next((s for s in tf_signals if s.window_size == ws), None)
            if sig:
                if sig.direction == SignalDirection.LONG:
                    row[f"w{ws}"] = "🟢"
                elif sig.direction == SignalDirection.SHORT:
                    row[f"w{ws}"] = "🔴"
                else:
                    row[f"w{ws}"] = "⚪"
            else:
                row[f"w{ws}"] = "-"

        # Add crossing
        crossing = next((s for s in aggregated.signals if s.timeframe == tf and s.is_crossing), None)
        if crossing:
            row["X"] = "🟢" if crossing.direction == SignalDirection.LONG else "🔴"
        else:
            row["X"] = "-"

        matrix_data.append(row)

    df = pd.DataFrame(matrix_data)
    st.dataframe(df, width='stretch', hide_index=True, height=420)


def render_decision_summary(aggregated):
    """Render final decision summary with recommendation."""
    st.subheader("🎯 Decision Summary")

    signal = aggregated.final_direction.value
    quality = aggregated.quality

    # High timeframe analysis
    high_tf = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    for sig in aggregated.signals:
        if sig.timeframe in ["3D", "1D", "12H"]:
            high_tf[sig.direction.value] = high_tf.get(sig.direction.value, 0) + 1

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Analysis:**")
        st.write(f"• Final Signal: **{signal}**")
        st.write(f"• Quality: **{quality}**")
        st.write(f"• Convergence: **{aggregated.convergence_score}/{aggregated.total_signals}**")
        st.write(f"• High TF (3D/1D/12H): L:{high_tf['LONG']} S:{high_tf['SHORT']} N:{high_tf['NEUTRAL']}")

    with col2:
        st.markdown("**Recommendation:**")
        if quality in ["VERY_HIGH", "HIGH"] and signal != "NEUTRAL":
            if signal == "LONG":
                st.success("✅ **LONG** - High confidence entry")
            else:
                st.error("✅ **SHORT** - High confidence entry")
        elif quality == "MEDIUM":
            st.warning("⚠️ Moderate confidence - proceed with caution")
        else:
            st.info("ℹ️ Low confidence - wait for better setup")


def render_calendar_table(aggregated, logic):
    """
    Render the signal prediction calendar table.

    Calendar shows:
    - Rows: timeframe-window combinations (55 rows)
    - Columns: Last 3 hours + 4 hours forward in 15min intervals
    - Cells: Color-coded signals with convergence indicators
    """
    st.subheader("📅 Signal Prediction Calendar")

    # Build calendar using processor results for historical data
    builder = CalendarDataBuilder(history_hours=3, forward_hours=4, interval_minutes=15)
    # Pass processor results for historical data lookup
    processor_results = logic.processor.results if logic.processor else None
    calendar_df = builder.build_calendar_df(aggregated.signals, processor_results=processor_results)

    # Get convergence summary
    conv_summary = builder.get_convergence_summary(calendar_df)

    # Display convergence summary at top using Plotly bar chart
    st.markdown("**Convergence Summary (by time):**")

    # Determine which columns are predictions (after now)
    now_str = datetime.utcnow().strftime("%H:%M")
    now_idx = 0
    for i, row in conv_summary.iterrows():
        if row['Time'] >= now_str:
            now_idx = i
            break

    # Build data for stacked bar chart
    times = conv_summary['Time'].tolist()
    long_counts = conv_summary['LONG'].tolist()
    short_counts = [-x for x in conv_summary['SHORT'].tolist()]  # Negative for below axis

    # Create figure with stacked bars
    fig_conv = go.Figure()

    # Add LONG bars (positive, above axis)
    fig_conv.add_trace(go.Bar(
        x=times,
        y=long_counts,
        name='LONG',
        marker_color='#2ecc71',
        hovertemplate='%{x}<br>LONG: %{y}<extra></extra>'
    ))

    # Add SHORT bars (negative, below axis)
    fig_conv.add_trace(go.Bar(
        x=times,
        y=short_counts,
        name='SHORT',
        marker_color='#e74c3c',
        hovertemplate='%{x}<br>SHORT: %{customdata}<extra></extra>',
        customdata=conv_summary['SHORT'].tolist()
    ))

    # Add vertical line for "now" using a shape (works with categorical x-axis)
    if now_idx > 0 and now_idx < len(times):
        fig_conv.add_shape(
            type="line",
            x0=times[now_idx],
            x1=times[now_idx],
            y0=-max(conv_summary['SHORT'].tolist()) if conv_summary['SHORT'].tolist() else 0,
            y1=max(conv_summary['LONG'].tolist()) if conv_summary['LONG'].tolist() else 0,
            line=dict(color="#f39c12", width=2, dash="dash")
        )
        fig_conv.add_annotation(
            x=times[now_idx],
            y=max(conv_summary['LONG'].tolist()) if conv_summary['LONG'].tolist() else 0,
            text="NOW",
            showarrow=False,
            font=dict(color="#f39c12", size=10),
            yshift=10
        )

    fig_conv.update_layout(
        height=180,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'size': 10},
        barmode='relative',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title="Count", zeroline=True, zerolinecolor='white', zerolinewidth=1)
    )

    st.plotly_chart(fig_conv, width='stretch')
    st.caption("🟢 LONG (above) | 🔴 SHORT (below) | 🟡 Dashed line = NOW (predictions after)")

    # Aggregated Score Summary Table (above calendar)
    st.markdown("**📊 Aggregated Signal Score by Time:**")

    score_summary = builder.get_aggregated_score_table(calendar_df)

    # Display columns to show (include Is Forecast for styling logic)
    display_cols = ['Time', 'Reversals', 'Crossings', 'Accel Quality', 'Total Score', 'LONG', 'SHORT', 'Direction', 'Strength %']
    score_display = score_summary[display_cols].copy()

    # Get forecast flags for row styling
    is_forecast_flags = score_summary['Is Forecast'].tolist()

    # Find the NOW row index (first forecast row)
    now_row_idx = next((i for i, f in enumerate(is_forecast_flags) if f), len(is_forecast_flags))

    # Pre-compute max values for styling
    max_total_score = score_display['Total Score'].max() if len(score_display) > 0 else 1
    max_total_score = max(max_total_score, 1)
    max_reversals = score_display['Reversals'].max() if score_display['Reversals'].max() > 0 else 1
    max_crossings = score_display['Crossings'].max() if score_display['Crossings'].max() > 0 else 1
    max_accel = score_display['Accel Quality'].max() if score_display['Accel Quality'].max() > 0 else 1
    max_direction = max(score_display['LONG'].max(), score_display['SHORT'].max(), 1)

    # Background colors for past vs future
    PAST_BG = 'rgba(30, 40, 50, 0.4)'      # Dark blue-gray for historical
    FUTURE_BG = 'rgba(60, 40, 70, 0.4)'    # Dark purple for predictions
    NOW_BORDER = 'border-top: 3px solid #f39c12;'  # Orange border for NOW row

    def style_score_row(row):
        """Apply heatmap styling to score summary table row with past/future differentiation."""
        styles = []
        row_idx = row.name
        is_future = row_idx >= now_row_idx
        is_now_row = row_idx == now_row_idx

        # Base background for past vs future
        base_bg = FUTURE_BG if is_future else PAST_BG
        border_style = NOW_BORDER if is_now_row else ''

        # Iterate only over the actual columns in score_display to avoid mismatch
        for col in display_cols:
            if col not in row.index:
                styles.append(f'background-color: {base_bg}; {border_style}')
                continue

            if col == 'Time':
                # Highlight NOW time column
                if is_now_row:
                    styles.append(f'background-color: rgba(243, 156, 18, 0.7); color: white; font-weight: bold; {border_style}')
                else:
                    styles.append(f'background-color: {base_bg}; color: {"#bb99ff" if is_future else "#88aacc"}; {border_style}')
            elif col == 'Direction':
                if row[col] == 'LONG':
                    styles.append(f'background-color: rgba(46, 204, 113, 0.6); color: white; font-weight: bold; {border_style}')
                elif row[col] == 'SHORT':
                    styles.append(f'background-color: rgba(231, 76, 60, 0.6); color: white; font-weight: bold; {border_style}')
                else:
                    styles.append(f'background-color: {base_bg}; color: #aaa; {border_style}')
            elif col == 'Total Score':
                intensity = min(row[col] / max_total_score, 1.0) * 0.8
                styles.append(f'background-color: rgba(155, 89, 182, {intensity}); color: white; {border_style}')
            elif col == 'Reversals':
                intensity = min(row[col] / max_reversals, 1.0) * 0.6
                styles.append(f'background-color: rgba(52, 152, 219, {intensity}); color: white; {border_style}')
            elif col == 'Crossings':
                intensity = min(row[col] / max_crossings, 1.0) * 0.6
                styles.append(f'background-color: rgba(52, 152, 219, {intensity}); color: white; {border_style}')
            elif col == 'Accel Quality':
                intensity = min(row[col] / max_accel, 1.0) * 0.6
                styles.append(f'background-color: rgba(52, 152, 219, {intensity}); color: white; {border_style}')
            elif col == 'LONG':
                intensity = min(row[col] / max_direction, 1.0) * 0.5
                styles.append(f'background-color: rgba(46, 204, 113, {intensity}); color: white; {border_style}')
            elif col == 'SHORT':
                intensity = min(row[col] / max_direction, 1.0) * 0.5
                styles.append(f'background-color: rgba(231, 76, 60, {intensity}); color: white; {border_style}')
            else:
                styles.append(f'background-color: {base_bg}; {border_style}')

        return styles

    styled_score = score_display.style.apply(style_score_row, axis=1)

    st.dataframe(
        styled_score,
        height=250,
        width='stretch',
        hide_index=True
    )

    st.caption("⬛ Dark = Historical | 🟪 Purple tint = Forecast | 🟧 Orange = NOW | 🟣 Score intensity | 🔵 Components | 🟢/🔴 Direction")

    st.markdown("---")

    # Display calendar with scrolling
    st.markdown("**Full Calendar (scroll to see all timeframe-window combinations):**")

    # Identify NOW column (first column that contains prediction markers '~')
    # Time columns are everything except "TF-Window"
    calendar_time_cols = [c for c in calendar_df.columns if c != "TF-Window"]

    # Find the NOW column index by checking for prediction markers
    now_col_name = None
    future_cols = set()
    for col in calendar_time_cols:
        has_prediction = any('~' in str(v) for v in calendar_df[col] if isinstance(v, str))
        if has_prediction:
            future_cols.add(col)
            if now_col_name is None:
                now_col_name = col

    # Also add summary columns as non-time columns
    summary_cols = {'ΣR', 'ΣC', 'ΣA'}

    # Get time columns
    time_cols = [c for c in calendar_df.columns if c != "TF-Window"]

    # Add ROW totals (R, C, A per TF-Window row)
    row_r_totals = []
    row_c_totals = []
    row_a_totals = []

    for idx, row in calendar_df.iterrows():
        r_count = 0
        c_count = 0
        a_count = 0
        for col in time_cols:
            val = row[col]
            if isinstance(val, str) and ':' in val:
                parts = val.replace('~', '').split(':')
                if len(parts) >= 3:
                    flags = parts[2]
                    if 'R' in flags:
                        r_count += 1
                    if 'C' in flags:
                        c_count += 1
                    if 'A' in flags:
                        a_count += 1
        row_r_totals.append(str(r_count))
        row_c_totals.append(str(c_count))
        row_a_totals.append(str(a_count))

    # Add row total columns to calendar
    calendar_with_row_totals = calendar_df.copy()
    calendar_with_row_totals['ΣR'] = row_r_totals
    calendar_with_row_totals['ΣC'] = row_c_totals
    calendar_with_row_totals['ΣA'] = row_a_totals

    # Add COLUMN totals (footer rows)
    r_col_totals = {"TF-Window": "TOTAL R"}
    c_col_totals = {"TF-Window": "TOTAL C"}
    a_col_totals = {"TF-Window": "TOTAL A"}

    # Calculate column totals
    for col in time_cols:
        r_count = 0
        c_count = 0
        a_count = 0
        for val in calendar_df[col]:
            if isinstance(val, str) and ':' in val:
                parts = val.replace('~', '').split(':')
                if len(parts) >= 3:
                    flags = parts[2]
                    if 'R' in flags:
                        r_count += 1
                    if 'C' in flags:
                        c_count += 1
                    if 'A' in flags:
                        a_count += 1
        r_col_totals[col] = str(r_count)
        c_col_totals[col] = str(c_count)
        a_col_totals[col] = str(a_count)

    # Calculate grand totals for row total columns
    grand_r = sum(int(x) for x in row_r_totals)
    grand_c = sum(int(x) for x in row_c_totals)
    grand_a = sum(int(x) for x in row_a_totals)

    r_col_totals['ΣR'] = str(grand_r)
    r_col_totals['ΣC'] = ''
    r_col_totals['ΣA'] = ''
    c_col_totals['ΣR'] = ''
    c_col_totals['ΣC'] = str(grand_c)
    c_col_totals['ΣA'] = ''
    a_col_totals['ΣR'] = ''
    a_col_totals['ΣC'] = ''
    a_col_totals['ΣA'] = str(grand_a)

    # Create footer DataFrame and append
    footer_df = pd.DataFrame([r_col_totals, c_col_totals, a_col_totals])
    calendar_with_footer = pd.concat([calendar_with_row_totals, footer_df], ignore_index=True)

    # Background colors for past vs future columns
    PAST_COL_BG = 'rgba(25, 35, 45, 0.6)'       # Dark blue-gray for historical
    FUTURE_COL_BG = 'rgba(50, 30, 60, 0.6)'     # Dark purple for predictions
    NOW_COL_BORDER = 'border-left: 3px solid #f39c12;'  # Orange border for NOW column

    def style_calendar_df(df):
        """
        Apply column-aware styling to calendar DataFrame.
        Returns a DataFrame of CSS styles matching the input shape.
        """
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        for col in df.columns:
            is_future_col = col in future_cols
            is_now_col = col == now_col_name
            is_summary_col = col in summary_cols
            is_tf_col = col == 'TF-Window'

            for row_idx in df.index:
                val = df.loc[row_idx, col]

                # Column border for NOW
                col_border = NOW_COL_BORDER if is_now_col else ''

                # Check if this is a footer row (TOTAL rows)
                is_footer = row_idx >= len(calendar_with_row_totals)

                if is_footer:
                    # Footer row styling
                    if isinstance(val, str):
                        if val.startswith("TOTAL"):
                            styles.loc[row_idx, col] = f'background-color: rgba(52, 73, 94, 0.9); color: #f1c40f; font-weight: bold; {col_border}'
                        elif val.isdigit() and int(val) > 0:
                            count = int(val)
                            intensity = min(count / 20, 1.0) * 0.8
                            # Future footer cells get purple tint
                            if is_future_col:
                                styles.loc[row_idx, col] = f'background-color: rgba(180, 130, 200, {intensity}); color: black; font-weight: bold; {col_border}'
                            else:
                                styles.loc[row_idx, col] = f'background-color: rgba(241, 196, 15, {intensity}); color: black; font-weight: bold; {col_border}'
                        else:
                            base_bg = FUTURE_COL_BG if is_future_col else 'rgba(52, 73, 94, 0.7)'
                            styles.loc[row_idx, col] = f'background-color: {base_bg}; color: #7f8c8d; {col_border}'
                    else:
                        styles.loc[row_idx, col] = f'background-color: rgba(52, 73, 94, 0.7); color: #7f8c8d; {col_border}'
                    continue

                # TF-Window column (row labels)
                if is_tf_col:
                    styles.loc[row_idx, col] = 'background-color: rgba(40, 50, 60, 0.8); color: #aabbcc; font-weight: bold;'
                    continue

                # Summary columns (ΣR, ΣC, ΣA)
                if is_summary_col:
                    if isinstance(val, str) and val.isdigit() and int(val) > 0:
                        count = int(val)
                        intensity = min(count / 10, 1.0) * 0.7
                        styles.loc[row_idx, col] = f'background-color: rgba(241, 196, 15, {intensity}); color: black; font-weight: bold;'
                    else:
                        styles.loc[row_idx, col] = 'background-color: rgba(40, 50, 60, 0.5); color: #7f8c8d;'
                    continue

                # Regular signal cells - apply past/future background
                base_bg = FUTURE_COL_BG if is_future_col else PAST_COL_BG

                # Empty or missing data
                if not isinstance(val, str) or val in ['-', '?']:
                    if val == '?':
                        styles.loc[row_idx, col] = f'background-color: rgba(243, 156, 18, 0.2); color: #f39c12; {col_border}'
                    else:
                        styles.loc[row_idx, col] = f'background-color: {base_bg}; color: #555; {col_border}'
                    continue

                # Parse signal value
                is_prediction = '~' in val
                val_clean = val.replace('~', '')

                parts = val_clean.split(':')
                direction = parts[0] if len(parts) >= 1 else 'N'
                try:
                    score = int(parts[1]) if len(parts) >= 2 else 0
                except ValueError:
                    score = 0

                # Calculate color intensity based on score
                base_intensity = 0.25 + (score * 0.22)
                intensity = min(base_intensity, 0.92)

                # Apply direction color with past/future tint
                if direction == 'L':
                    if is_future_col:
                        # Future LONG: green with purple tint
                        styles.loc[row_idx, col] = f'background-color: rgba(46, 180, 120, {intensity}); color: white; {col_border}'
                    else:
                        # Past LONG: pure green
                        styles.loc[row_idx, col] = f'background-color: rgba(46, 204, 113, {intensity}); color: white; {col_border}'
                elif direction == 'S':
                    if is_future_col:
                        # Future SHORT: red with purple tint
                        styles.loc[row_idx, col] = f'background-color: rgba(200, 76, 100, {intensity}); color: white; {col_border}'
                    else:
                        # Past SHORT: pure red
                        styles.loc[row_idx, col] = f'background-color: rgba(231, 76, 60, {intensity}); color: white; {col_border}'
                else:
                    # Neutral
                    styles.loc[row_idx, col] = f'background-color: {base_bg}; color: #aaa; {col_border}'

        return styles

    # Apply styling
    styled_df = calendar_with_footer.style.apply(style_calendar_df, axis=None)

    # Rename NOW column header to highlight it
    if now_col_name and now_col_name in calendar_with_footer.columns:
        new_columns = {now_col_name: f"▶{now_col_name}◀"}
        styled_df = styled_df.set_table_styles([
            {'selector': f'th:contains("{now_col_name}")',
             'props': [('background-color', '#f39c12'), ('color', 'black'), ('font-weight', 'bold')]}
        ], overwrite=False)

    st.dataframe(
        styled_df,
        height=450,
        width='stretch',
        hide_index=True
    )

    # Legend
    st.markdown("""
    **Cell Format: `Direction:Score:Flags`**

    **Direction:** `L` = LONG, `S` = SHORT, `N` = NEUTRAL

    **Score (0-3):** +1 reversal, +1 crossing, +1 accel quality

    **Visual:**
    - ⬛ **Dark blue-gray** = Historical data
    - 🟪 **Purple tint** = Future predictions
    - 🟧 **Orange border** = NOW column divider
    - 🟢 Green = LONG, 🔴 Red = SHORT
    - Darker = higher score
    """)


# ============================================================================
# VALIDATION / DEBUG VIEWS
# ============================================================================

def render_unified_signal_validation(processor, logic, selected_timeframe, selected_windows):
    """
    Unified Signal Validation chart.

    One or more angle lines colored segment-by-segment by GMM acceleration zone.
    Reversals marked with vertical dashed lines + triangles per window.
    Crossings shown only between selected windows that form valid pairs
    (adjacent or neighbor-of-neighbor).
    Last 200 data points.

    Args:
        selected_windows: List of window sizes (e.g. [30], [30, 60], [60, 100, 120])
    """
    # Valid crossing pairs: adjacent + neighbor-of-neighbor (deduplicated)
    VALID_CROSSING_PAIRS = {
        (30, 60), (60, 100), (100, 120), (120, 160),   # adjacent
        (30, 100), (60, 120), (100, 160),               # skip-one
    }

    ws_to_label = {30: 'df', 60: 'df1', 100: 'df2', 120: 'df3', 160: 'df4'}
    ws_names = [f"{ws_to_label[ws]}(w{ws})" for ws in selected_windows]
    st.subheader(f"Signal Validation - {selected_timeframe} / {', '.join(ws_names)}")

    if selected_timeframe not in processor.results:
        st.warning(f"No data for {selected_timeframe}")
        return

    # ── Fit GMM on this timeframe's combined acceleration ──
    all_accel = []
    for lbl, ws_df in processor.results[selected_timeframe].items():
        if 'acceleration' in ws_df.columns:
            all_accel.extend(ws_df['acceleration'].dropna().values)
    gmm_key = f"unified_{selected_timeframe}"
    gmm = logic.fit_gmm_acceleration(np.array(all_accel), gmm_key)

    zone_style = {
        AccelerationZone.VERY_DISTANT: ("#e74c3c", "Very Distant"),
        AccelerationZone.DISTANT:      ("#f39c12", "Distant"),
        AccelerationZone.BASELINE:     ("#f1c40f", "Baseline"),
        AccelerationZone.CLOSE:        ("#87ceeb", "Close"),
        AccelerationZone.VERY_CLOSE:   ("#95a5a6", "Very Close"),
    }

    fig = go.Figure()
    legend_added = set()

    # Global y-range across all selected windows (for vertical lines)
    global_y_min = float('inf')
    global_y_max = float('-inf')

    # Track all reversals and per-window data for summary
    all_bottom_t, all_peak_t = [], []

    # ── Draw each selected window ──
    for ws in selected_windows:
        ws_idx = WINDOW_SIZES.index(ws)
        ws_label = f"df{ws_idx}" if ws_idx > 0 else "df"
        ws_display = f"{ws_label}(w{ws})"

        if ws_label not in processor.results[selected_timeframe]:
            continue

        df = processor.results[selected_timeframe][ws_label]
        if 'angle' not in df.columns or 'time' not in df.columns:
            continue

        df_plot = df.tail(200).copy().reset_index(drop=True)
        angles = df_plot['angle'].values
        times = df_plot['time'].values
        n = len(angles)

        # Update global y-range
        global_y_min = min(global_y_min, float(np.nanmin(angles)))
        global_y_max = max(global_y_max, float(np.nanmax(angles)))

        # Classify every point by GMM zone
        accel_col = df_plot['acceleration'].values if 'acceleration' in df_plot.columns else np.full(n, np.nan)
        point_zones = []
        for val in accel_col:
            if np.isnan(val) or gmm is None:
                point_zones.append(AccelerationZone.BASELINE)
            else:
                point_zones.append(logic.classify_acceleration(val, gmm))

        # Draw angle line as coloured segments
        i = 0
        while i < n - 1:
            zone = point_zones[i]
            color, zone_name = zone_style.get(zone, ("#808080", "Unknown"))
            j = i + 1
            while j < n and point_zones[j] == zone:
                j += 1
            end = min(j + 1, n)

            # Legend: show zone name only once across all windows
            show_legend = zone_name not in legend_added
            legend_added.add(zone_name)

            fig.add_trace(go.Scattergl(
                x=times[i:end],
                y=angles[i:end],
                mode='lines',
                line=dict(color=color, width=3),
                name=zone_name,
                legendgroup=zone_name,
                showlegend=show_legend,
                hovertemplate=f"<b>{ws_display} | {zone_name}</b><br>Time: %{{x}}<br>Angle: %{{y:.2f}}<extra></extra>"
            ))
            i = j

        # ── Reversal markers for this window ──
        events = logic.detect_cycle_events(angles, window=5)

        bottom_t, bottom_a = [], []
        peak_t, peak_a = [], []
        for idx in range(n):
            if events[idx] == 1:
                rev = logic.get_reversal_type_at_index(events, idx, angles)
                if rev == ReversalType.BOTTOM:
                    bottom_t.append(pd.to_datetime(times[idx]))
                    bottom_a.append(angles[idx])
                elif rev == ReversalType.PEAK:
                    peak_t.append(pd.to_datetime(times[idx]))
                    peak_a.append(angles[idx])

        all_bottom_t.extend(bottom_t)
        all_peak_t.extend(peak_t)

        # Triangle markers
        if bottom_t:
            fig.add_trace(go.Scatter(
                x=bottom_t, y=bottom_a,
                mode='markers',
                name=f'{ws_display} Bottom ({len(bottom_t)})',
                marker=dict(color='#2ecc71', size=16, symbol='triangle-up',
                            line=dict(color='white', width=2)),
                hovertemplate=f"<b>{ws_display} BOTTOM (LONG)</b><br>Time: %{{x}}<br>Angle: %{{y:.2f}}<extra></extra>"
            ))
        if peak_t:
            fig.add_trace(go.Scatter(
                x=peak_t, y=peak_a,
                mode='markers',
                name=f'{ws_display} Peak ({len(peak_t)})',
                marker=dict(color='#e74c3c', size=16, symbol='triangle-down',
                            line=dict(color='white', width=2)),
                hovertemplate=f"<b>{ws_display} PEAK (SHORT)</b><br>Time: %{{x}}<br>Angle: %{{y:.2f}}<extra></extra>"
            ))

    # Compute vertical line y-range
    y_pad = (global_y_max - global_y_min) * 0.05
    vline_lo = global_y_min - y_pad
    vline_hi = global_y_max + y_pad

    # Draw reversal vertical lines (deduplicate times)
    drawn_reversal_times = set()
    for t in all_bottom_t:
        t_key = str(t)
        if t_key not in drawn_reversal_times:
            drawn_reversal_times.add(t_key)
            fig.add_trace(go.Scatter(
                x=[t, t], y=[vline_lo, vline_hi],
                mode='lines', line=dict(color='#2ecc71', width=2, dash='dash'),
                showlegend=False, hoverinfo='skip'
            ))
    for t in all_peak_t:
        t_key = str(t)
        if t_key not in drawn_reversal_times:
            drawn_reversal_times.add(t_key)
            fig.add_trace(go.Scatter(
                x=[t, t], y=[vline_lo, vline_hi],
                mode='lines', line=dict(color='#e74c3c', width=2, dash='dash'),
                showlegend=False, hoverinfo='skip'
            ))

    # ── Crossing markers (only if 2+ windows selected forming valid pairs) ──
    relevant_crossings = []
    if len(selected_windows) >= 2:
        # Find which selected window pairs are valid crossing pairs
        valid_selected_pairs = []
        for i_w in range(len(selected_windows)):
            for j_w in range(i_w + 1, len(selected_windows)):
                pair = tuple(sorted((selected_windows[i_w], selected_windows[j_w])))
                if pair in VALID_CROSSING_PAIRS:
                    valid_selected_pairs.append(pair)

        if valid_selected_pairs:
            # Collect angle data for crossing detection
            angles_dict = {}
            times_series = None
            for ws in selected_windows:
                ws_idx = WINDOW_SIZES.index(ws)
                ws_label = f"df{ws_idx}" if ws_idx > 0 else "df"
                if ws_label not in processor.results[selected_timeframe]:
                    continue
                ws_df = processor.results[selected_timeframe][ws_label]
                if 'angle' not in ws_df.columns:
                    continue
                ws_plot = ws_df.tail(200)
                angles_dict[ws] = ws_plot['angle']
                if times_series is None or len(ws_plot) < len(times_series):
                    times_series = ws_plot['time']

            if times_series is not None:
                crossings = logic.detect_angle_crossings(angles_dict, times_series)
                # Keep only crossings between valid selected pairs
                relevant_crossings = [
                    c for c in crossings
                    if tuple(sorted(c['windows'])) in valid_selected_pairs
                ]

                cross_long_t, cross_long_a = [], []
                cross_short_t, cross_short_a = [], []
                cross_long_labels, cross_short_labels = [], []

                # Use first selected window's angle for y-position of crossing markers
                first_ws = selected_windows[0]
                first_ws_idx = WINDOW_SIZES.index(first_ws)
                first_label = f"df{first_ws_idx}" if first_ws_idx > 0 else "df"
                first_df = processor.results[selected_timeframe].get(first_label)
                first_times = first_df['time'].tail(200).values if first_df is not None else None
                first_angles = first_df['angle'].tail(200).values if first_df is not None else None

                for c in relevant_crossings:
                    cross_time = pd.to_datetime(c['time'])
                    # Find y-value on the first window's line
                    y_val = 0
                    if first_times is not None and first_angles is not None:
                        time_diffs = np.abs(pd.to_datetime(first_times) - cross_time)
                        nearest = np.argmin(time_diffs)
                        y_val = first_angles[nearest]

                    ws1, ws2 = c['windows']
                    lbl1 = ws_to_label.get(ws1, f'w{ws1}')
                    lbl2 = ws_to_label.get(ws2, f'w{ws2}')
                    pair_label = f"{lbl1}x{lbl2}"

                    if c['direction'] == SignalDirection.LONG:
                        cross_long_t.append(cross_time)
                        cross_long_a.append(y_val)
                        cross_long_labels.append(pair_label)
                    else:
                        cross_short_t.append(cross_time)
                        cross_short_a.append(y_val)
                        cross_short_labels.append(pair_label)

                # Vertical dotted lines for crossings
                for t in cross_long_t:
                    fig.add_trace(go.Scatter(
                        x=[t, t], y=[vline_lo, vline_hi],
                        mode='lines', line=dict(color='#00ff88', width=2, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))
                for t in cross_short_t:
                    fig.add_trace(go.Scatter(
                        x=[t, t], y=[vline_lo, vline_hi],
                        mode='lines', line=dict(color='#ff4466', width=2, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))

                # Diamond markers
                if cross_long_t:
                    fig.add_trace(go.Scatter(
                        x=cross_long_t, y=cross_long_a,
                        mode='markers+text',
                        name=f'Crossing LONG ({len(cross_long_t)})',
                        marker=dict(color='#00ff88', size=14, symbol='diamond',
                                    line=dict(color='white', width=1.5)),
                        text=cross_long_labels,
                        textposition='bottom center',
                        textfont=dict(color='#00ff88', size=10, family='Arial Black'),
                        hovertemplate="<b>Crossing LONG</b><br>%{text}<br>Time: %{x}<extra></extra>"
                    ))
                if cross_short_t:
                    fig.add_trace(go.Scatter(
                        x=cross_short_t, y=cross_short_a,
                        mode='markers+text',
                        name=f'Crossing SHORT ({len(cross_short_t)})',
                        marker=dict(color='#ff4466', size=14, symbol='diamond',
                                    line=dict(color='white', width=1.5)),
                        text=cross_short_labels,
                        textposition='bottom center',
                        textfont=dict(color='#ff4466', size=10, family='Arial Black'),
                        hovertemplate="<b>Crossing SHORT</b><br>%{text}<br>Time: %{x}<extra></extra>"
                    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

    fig.update_layout(
        height=600,
        margin=dict(l=60, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Time",
        yaxis_title="Angle (degrees)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    n_crossings = len(relevant_crossings)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bottoms (LONG)", len(all_bottom_t))
    with col2:
        st.metric("Peaks (SHORT)", len(all_peak_t))
    with col3:
        st.metric("Crossings", n_crossings)

    # Show which crossing pairs are active
    if len(selected_windows) >= 2:
        valid_selected = []
        for i_w in range(len(selected_windows)):
            for j_w in range(i_w + 1, len(selected_windows)):
                pair = tuple(sorted((selected_windows[i_w], selected_windows[j_w])))
                lbl1 = ws_to_label.get(pair[0], '?')
                lbl2 = ws_to_label.get(pair[1], '?')
                if pair in VALID_CROSSING_PAIRS:
                    valid_selected.append(f"{lbl1} x {lbl2}")
        if valid_selected:
            st.caption(f"Active crossing pairs: **{', '.join(valid_selected)}**")
        else:
            st.caption("No valid crossing pairs in selection (need adjacent or neighbor-of-neighbor)")
    elif len(selected_windows) == 1:
        st.caption("Select 2+ windows to see crossing signals")

    st.caption(
        "**Line color** = GMM acceleration zone | "
        "**Green dashed + triangle** = BOTTOM (LONG) | "
        "**Red dashed + triangle** = PEAK (SHORT) | "
        "**Green dotted + diamond** = Crossing LONG | "
        "**Red dotted + diamond** = Crossing SHORT"
    )


def render_prediction_graph(processor, logic, selected_timeframe, selected_windows):
    """
    Prediction graph with rolling backtest and forward prediction.

    Layout (6 rows):
      Rows 1-3: Actual vs rolling linear extrapolation overlay + future cc/ccc
      Rows 4-6: Difference (actual - predicted) per metric

    The rolling prediction computes next = v[i-1] + (v[i-1] - v[i-2]) for each
    historical point i>=2, so we can see how well extrapolation tracks reality.
    """
    from core.signal_logic import PricePredictor
    import numpy as np

    VALID_CROSSING_PAIRS = {
        (30, 60), (60, 100), (100, 120), (120, 160),
        (30, 100), (60, 120), (100, 160),
    }
    ws_to_label = {30: 'df', 60: 'df1', 100: 'df2', 120: 'df3', 160: 'df4'}
    ws_line_colors = {30: '#2ecc71', 60: '#3498db', 100: '#87ceeb', 120: '#9b59b6', 160: '#f39c12'}

    ws_names = [f"{ws_to_label[ws]}(w{ws})" for ws in selected_windows]
    st.subheader(f"Prediction (cc / ccc) - {selected_timeframe} / {', '.join(ws_names)}")

    if selected_timeframe not in processor.results:
        st.warning(f"No data for {selected_timeframe}")
        return

    metrics = ['angle', 'slope_f', 'acceleration']
    metric_titles = ['Angle', 'Slope_f', 'Acceleration',
                     'Angle diff (actual-pred)', 'Slope_f diff', 'Accel diff']
    HISTORY = 50  # rolling backtest uses more points for meaningful error view

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=metric_titles,
        row_heights=[0.2, 0.15, 0.15, 0.17, 0.17, 0.16]
    )

    predicted_angles = {}  # ws -> {current, cc, ccc}

    for ws in selected_windows:
        ws_idx = WINDOW_SIZES.index(ws)
        ws_label_str = f"df{ws_idx}" if ws_idx > 0 else "df"
        ws_display = f"{ws_to_label[ws]}(w{ws})"
        color = ws_line_colors.get(ws, '#ffffff')

        if ws_label_str not in processor.results[selected_timeframe]:
            continue

        df_data = processor.results[selected_timeframe][ws_label_str]

        for metric_i, metric in enumerate(metrics):
            row_actual = metric_i + 1   # rows 1,2,3  — actual + rolling pred overlay
            row_diff = metric_i + 4     # rows 4,5,6  — difference

            if metric not in df_data.columns or 'time' not in df_data.columns:
                continue

            full_series = df_data[metric].dropna()
            full_times = df_data['time']
            if len(full_series) < 4:
                continue

            all_vals = full_series.values
            all_times = full_times.values

            # ── Rolling linear extrapolation over last HISTORY points ──
            # For each i >= 2: rolling_pred[i] = v[i-1] + (v[i-1] - v[i-2])
            start_idx = max(2, len(all_vals) - HISTORY)
            rolling_times = []
            rolling_pred = []
            rolling_actual = []
            rolling_diff = []

            for i in range(start_idx, len(all_vals)):
                pred_val = PricePredictor.predict_next(all_vals[i-2], all_vals[i-1])
                actual_val = all_vals[i]
                rolling_times.append(all_times[i])
                rolling_pred.append(pred_val)
                rolling_actual.append(actual_val)
                rolling_diff.append(actual_val - pred_val)

            rolling_times = np.array(rolling_times)

            # ── Forward prediction: cc, ccc ──
            a, b = all_vals[-2], all_vals[-1]
            cc = PricePredictor.predict_next(a, b)
            ccc = PricePredictor.predict_next(b, cc)

            if metric == 'angle':
                predicted_angles[ws] = {'current': b, 'cc': cc, 'ccc': ccc}

            last_t = pd.to_datetime(all_times[-1])
            prev_t = pd.to_datetime(all_times[-2])
            dt = last_t - prev_t
            pred_times = [last_t, last_t + dt, last_t + dt * 2]
            pred_vals = [b, cc, ccc]

            # ── Row 1-3: Actual line (solid) ──
            fig.add_trace(go.Scattergl(
                x=rolling_times,
                y=rolling_actual,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'{ws_display} actual',
                legendgroup=ws_display,
                showlegend=(metric_i == 0),
                hovertemplate=f"<b>{ws_display}</b><br>%{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>"
            ), row=row_actual, col=1)

            # ── Row 1-3: Rolling prediction overlay (dotted, lighter) ──
            pred_color = color.replace(')', ',0.6)').replace('rgb', 'rgba') if 'rgb' in color else color
            fig.add_trace(go.Scattergl(
                x=rolling_times,
                y=rolling_pred,
                mode='lines',
                line=dict(color=color, width=1.5, dash='dot'),
                name=f'{ws_display} predicted',
                legendgroup=f'{ws_display}_pred',
                showlegend=(metric_i == 0),
                opacity=0.7,
                hovertemplate=f"<b>{ws_display} predicted</b><br>%{{x}}<br>pred: %{{y:.4f}}<extra></extra>"
            ), row=row_actual, col=1)

            # ── Row 1-3: Forward cc/ccc dashed extension ──
            fig.add_trace(go.Scatter(
                x=pred_times,
                y=pred_vals,
                mode='lines+markers',
                line=dict(color=color, width=3, dash='dash'),
                marker=dict(size=10, symbol='circle', line=dict(color='white', width=2)),
                legendgroup=ws_display,
                showlegend=False,
                hovertemplate=f"<b>{ws_display} PREDICTED</b><br>%{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>"
            ), row=row_actual, col=1)

            # ── Row 1-3: cc/ccc annotations (offset to avoid overlap) ──
            fig.add_annotation(
                x=pred_times[1], y=cc,
                text=f"cc={cc:.3f}",
                showarrow=True, arrowhead=2, arrowcolor=color,
                ax=0, ay=-30,
                font=dict(color=color, size=10),
                bgcolor='rgba(0,0,0,0.6)',
                row=row_actual, col=1
            )
            fig.add_annotation(
                x=pred_times[2], y=ccc,
                text=f"ccc={ccc:.3f}",
                showarrow=True, arrowhead=2, arrowcolor=color,
                ax=0, ay=30,
                font=dict(color=color, size=10),
                bgcolor='rgba(0,0,0,0.6)',
                row=row_actual, col=1
            )

            # ── Row 4-6: Difference (actual - predicted) as area chart ──
            diff_arr = np.array(rolling_diff)
            pos_diff = np.where(diff_arr >= 0, diff_arr, 0)
            neg_diff = np.where(diff_arr < 0, diff_arr, 0)

            # Positive difference (green fill)
            fig.add_trace(go.Scatter(
                x=rolling_times, y=pos_diff,
                mode='lines',
                line=dict(color='rgba(46,204,113,0.8)', width=0),
                fill='tozeroy',
                fillcolor='rgba(46,204,113,0.3)',
                name=f'{ws_display} +diff',
                legendgroup=ws_display,
                showlegend=False,
                hovertemplate=f"<b>{ws_display} diff</b><br>%{{x}}<br>+%{{y:.4f}}<extra></extra>"
            ), row=row_diff, col=1)

            # Negative difference (red fill)
            fig.add_trace(go.Scatter(
                x=rolling_times, y=neg_diff,
                mode='lines',
                line=dict(color='rgba(231,76,60,0.8)', width=0),
                fill='tozeroy',
                fillcolor='rgba(231,76,60,0.3)',
                name=f'{ws_display} -diff',
                legendgroup=ws_display,
                showlegend=False,
                hovertemplate=f"<b>{ws_display} diff</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>"
            ), row=row_diff, col=1)

            # Difference line on top
            fig.add_trace(go.Scattergl(
                x=rolling_times, y=diff_arr,
                mode='lines',
                line=dict(color=color, width=1.5),
                legendgroup=ws_display,
                showlegend=False,
                hovertemplate=f"<b>{ws_display} error</b><br>%{{x}}<br>diff: %{{y:.4f}}<extra></extra>"
            ), row=row_diff, col=1)

            # MAE annotation on difference subplot
            mae = float(np.mean(np.abs(diff_arr)))
            fig.add_annotation(
                x=rolling_times[-1], y=0,
                text=f"MAE={mae:.3f}",
                showarrow=False,
                font=dict(color='white', size=9),
                bgcolor='rgba(0,0,0,0.6)',
                xanchor='right',
                row=row_diff, col=1
            )

            # ── Row 1-3: Predicted reversals (angle only) ──
            if metric == 'angle' and len(all_vals) >= 4:
                extended = list(all_vals[-4:]) + [cc, ccc]
                for check_idx in [4, 5]:
                    if check_idx < len(extended):
                        w = extended[check_idx - 4:check_idx + 1]
                        if len(w) == 5:
                            pred_time = pred_times[check_idx - 3] if check_idx - 3 < len(pred_times) else pred_times[-1]
                            pred_angle = w[4]
                            if w[0] > w[1] > w[2] > w[3] and w[3] < w[4]:
                                fig.add_trace(go.Scatter(
                                    x=[pred_time], y=[pred_angle],
                                    mode='markers',
                                    marker=dict(color='#2ecc71', size=18, symbol='triangle-up',
                                                line=dict(color='white', width=2)),
                                    legendgroup=ws_display, showlegend=False,
                                    hovertemplate=f"<b>PREDICTED BOTTOM (LONG)</b><br>{ws_display}<extra></extra>"
                                ), row=1, col=1)
                            elif w[0] < w[1] < w[2] < w[3] and w[3] > w[4]:
                                fig.add_trace(go.Scatter(
                                    x=[pred_time], y=[pred_angle],
                                    mode='markers',
                                    marker=dict(color='#e74c3c', size=18, symbol='triangle-down',
                                                line=dict(color='white', width=2)),
                                    legendgroup=ws_display, showlegend=False,
                                    hovertemplate=f"<b>PREDICTED PEAK (SHORT)</b><br>{ws_display}<extra></extra>"
                                ), row=1, col=1)

    # ── Predicted crossings between selected windows ──
    if len(selected_windows) >= 2 and predicted_angles:
        for i_w in range(len(selected_windows)):
            for j_w in range(i_w + 1, len(selected_windows)):
                ws1, ws2 = selected_windows[i_w], selected_windows[j_w]
                pair = tuple(sorted((ws1, ws2)))
                if pair not in VALID_CROSSING_PAIRS:
                    continue
                if ws1 not in predicted_angles or ws2 not in predicted_angles:
                    continue

                p1, p2 = predicted_angles[ws1], predicted_angles[ws2]
                curr_diff = p1['current'] - p2['current']
                cc_diff = p1['cc'] - p2['cc']
                ccc_diff = p1['ccc'] - p2['ccc']
                lbl1, lbl2 = ws_to_label[ws1], ws_to_label[ws2]

                first_ws = selected_windows[0]
                first_ws_idx = WINDOW_SIZES.index(first_ws)
                first_label = f"df{first_ws_idx}" if first_ws_idx > 0 else "df"
                first_df = processor.results[selected_timeframe].get(first_label)
                if first_df is None:
                    continue
                last_t = pd.to_datetime(first_df['time'].values[-1])
                prev_t = pd.to_datetime(first_df['time'].values[-2])
                dt = last_t - prev_t

                for step_diff, step_label, step_mult, step_vals in [
                    (curr_diff * cc_diff < 0, 'cc', 1, (p1['cc'], p2['cc'])),
                    (cc_diff * ccc_diff < 0, 'ccc', 2, (p1['ccc'], p2['ccc'])),
                ]:
                    if step_diff:
                        ref_diff = cc_diff if step_label == 'cc' else ccc_diff
                        cross_dir = "LONG" if ref_diff > 0 else "SHORT"
                        cross_color = '#00ff88' if cross_dir == "LONG" else '#ff4466'
                        fig.add_annotation(
                            x=last_t + dt * step_mult, y=sum(step_vals) / 2,
                            text=f"X {lbl1}x{lbl2} {cross_dir}",
                            showarrow=True, arrowhead=2, arrowcolor=cross_color,
                            font=dict(color=cross_color, size=11, family='Arial Black'),
                            bgcolor='rgba(0,0,0,0.7)',
                            row=1, col=1
                        )

    # Zero lines on all rows
    for row in range(1, 7):
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=row, col=1)

    # "NOW" vertical divider on actual rows (1-3)
    for ws in selected_windows:
        ws_idx = WINDOW_SIZES.index(ws)
        ws_label_str = f"df{ws_idx}" if ws_idx > 0 else "df"
        if ws_label_str in processor.results.get(selected_timeframe, {}):
            _now_df = processor.results[selected_timeframe][ws_label_str]
            _now_t = pd.to_datetime(_now_df['time'].values[-1])
            for row_i in range(1, 4):
                metric = metrics[row_i - 1]
                if metric in _now_df.columns:
                    _col_data = _now_df[metric].dropna().tail(HISTORY)
                    if len(_col_data) > 0:
                        y_lo, y_hi = float(_col_data.min()), float(_col_data.max())
                        y_margin = (y_hi - y_lo) * 0.1 if y_hi != y_lo else 1.0
                        fig.add_trace(go.Scatter(
                            x=[_now_t, _now_t], y=[y_lo - y_margin, y_hi + y_margin],
                            mode='lines',
                            line=dict(color='#ffff00', width=2, dash='dot'),
                            showlegend=False, hoverinfo='skip'
                        ), row=row_i, col=1)
            fig.add_annotation(
                x=_now_t, y=1.02, text="NOW", showarrow=False,
                font=dict(color='#ffff00', size=11, family='Arial Black'),
                xref='x', yref='y domain', row=1, col=1
            )
            break

    fig.update_layout(
        height=1100,
        margin=dict(l=60, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Angle", row=1, col=1)
    fig.update_yaxes(title_text="Slope_f", row=2, col=1)
    fig.update_yaxes(title_text="Accel", row=3, col=1)
    fig.update_yaxes(title_text="Angle err", row=4, col=1)
    fig.update_yaxes(title_text="Slope err", row=5, col=1)
    fig.update_yaxes(title_text="Accel err", row=6, col=1)
    fig.update_xaxes(title_text="Time", row=6, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Solid** = actual | **Dotted** = rolling linear extrapolation | "
        "**Dashed + dots** = forward cc/ccc | "
        "**Green/red fill** = prediction error (actual - predicted) | "
        "**MAE** = mean absolute error | "
        "Formula: `next = b + (b - a)` — simple linear extrapolation from last two values"
    )


def _load_ohlcv(timeframe: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a timeframe. Returns None if High/Low columns missing."""
    csv_path = DATA_DIR / f"testing_data_{timeframe}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    # Need OHLCV columns
    needed = ['Open Time', 'Open', 'High', 'Low', 'Close']
    if not all(c in df.columns for c in needed):
        return None
    df['Open Time'] = pd.to_datetime(df['Open Time'], utc=True)
    for c in ('Open', 'High', 'Low', 'Close'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Open Time', 'Open', 'High', 'Low', 'Close'])
    return df


def render_target_labeling_chart(selected_timeframe: str, sl_pct: float, tp_pct: float,
                                  max_hold: int, n_candles: int = 100):
    """
    Render candlestick chart with SL/TP target labeling arrows.

    Shows OHLC candles with:
    - Green up-arrows for LONG entry points (TP hit as long)
    - Red down-arrows for SHORT entry points (TP hit as short)
    - Dashed SL/TP lines radiating from each entry arrow
    - Summary metrics (win rate, avg hold, etc.)
    """
    st.subheader(f"Target Labeling — {selected_timeframe}")

    ohlcv = _load_ohlcv(selected_timeframe)
    if ohlcv is None:
        st.warning(
            f"No OHLCV data for {selected_timeframe}. "
            "Re-download data (Full Refresh) to get Open/High/Low/Close columns."
        )
        return

    if len(ohlcv) < max_hold + 10:
        st.warning(f"Insufficient data for {selected_timeframe}: {len(ohlcv)} rows (need > {max_hold})")
        return

    # Run target labeling on full dataset
    labels_df = create_sl_tp_labels(ohlcv, sl_pct=sl_pct, tp_pct=tp_pct, max_hold_periods=max_hold)

    if len(labels_df) == 0:
        st.warning("No labels generated.")
        return

    # Take last N candles for display
    display_df = ohlcv.tail(n_candles).copy().reset_index(drop=True)
    start_idx = len(ohlcv) - n_candles
    display_labels = labels_df.iloc[start_idx:start_idx + n_candles].copy().reset_index(drop=True)

    # Build candlestick figure
    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=display_df['Open Time'],
        open=display_df['Open'],
        high=display_df['High'],
        low=display_df['Low'],
        close=display_df['Close'],
        name='BTCUSDT',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ))

    # Separate LONG and SHORT entries
    long_mask = display_labels['label'] == 1
    short_mask = display_labels['label'] == -1

    long_entries = display_df[long_mask.values] if long_mask.any() else pd.DataFrame()
    short_entries = display_df[short_mask.values] if short_mask.any() else pd.DataFrame()
    long_labels = display_labels[long_mask] if long_mask.any() else pd.DataFrame()
    short_labels = display_labels[short_mask] if short_mask.any() else pd.DataFrame()

    # LONG arrows (green triangles below candles)
    if len(long_entries) > 0:
        fig.add_trace(go.Scatter(
            x=long_entries['Open Time'],
            y=long_entries['Low'] * 0.999,
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#00e676',
                        line=dict(width=1, color='white')),
            name=f'LONG Entry ({len(long_entries)})',
            hovertemplate=(
                'LONG Entry<br>'
                'Time: %{x}<br>'
                'Price: %{customdata[0]:.2f}<br>'
                'Gain: %{customdata[1]:.2f}%<br>'
                'Hold: %{customdata[2]} bars<br>'
                'Exit: %{customdata[3]}<extra></extra>'
            ),
            customdata=np.column_stack([
                long_labels['entry_price'].values,
                long_labels['gain_pct'].values,
                long_labels['hold_periods'].values,
                long_labels['exit_reason'].values,
            ]) if len(long_labels) > 0 else None,
        ))

    # SHORT arrows (red triangles above candles)
    if len(short_entries) > 0:
        fig.add_trace(go.Scatter(
            x=short_entries['Open Time'],
            y=short_entries['High'] * 1.001,
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='#ff1744',
                        line=dict(width=1, color='white')),
            name=f'SHORT Entry ({len(short_entries)})',
            hovertemplate=(
                'SHORT Entry<br>'
                'Time: %{x}<br>'
                'Price: %{customdata[0]:.2f}<br>'
                'Gain: %{customdata[1]:.2f}%<br>'
                'Hold: %{customdata[2]} bars<br>'
                'Exit: %{customdata[3]}<extra></extra>'
            ),
            customdata=np.column_stack([
                short_labels['entry_price'].values,
                short_labels['gain_pct'].values,
                short_labels['hold_periods'].values,
                short_labels['exit_reason'].values,
            ]) if len(short_labels) > 0 else None,
        ))

    # Draw SL/TP zones for recent entries (last 5 of each type to avoid clutter)
    _draw_sl_tp_zones(fig, display_df, display_labels, long_mask, short_mask,
                      sl_pct, tp_pct, max_show=5)

    fig.update_layout(
        title=f"BTCUSDT {selected_timeframe} — SL {sl_pct}% / TP {tp_pct}% (last {n_candles} candles)",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics below chart
    summary = get_labeling_summary(labels_df)
    if summary:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Win Rate", f"{summary['win_rate']:.1f}%")
        c2.metric("LONG Entries", summary['long_count'])
        c3.metric("SHORT Entries", summary['short_count'])
        c4.metric("Avg Gain", f"{summary['avg_gain']:.2f}%")
        c5.metric("Avg Hold", f"{summary['avg_hold']:.1f} bars")

        # Trade outcome breakdown
        st.caption(
            f"Total: {summary['trade_signals']} trades / {summary['total_periods']} periods | "
            f"TP Hits: {summary['tp_hits']} | SL Hits: {summary['sl_hits']} | "
            f"R/R: {tp_pct/sl_pct:.1f}:1"
        )


def _draw_sl_tp_zones(fig, display_df, display_labels, long_mask, short_mask,
                      sl_pct, tp_pct, max_show=5):
    """Draw dashed SL/TP lines for the most recent N entries."""
    times = display_df['Open Time'].values

    # Recent LONG entries
    long_idxs = display_labels.index[long_mask].tolist()
    for idx in long_idxs[-max_show:]:
        if idx >= len(display_df):
            continue
        entry_price = display_labels.loc[idx, 'entry_price']
        hold = int(display_labels.loc[idx, 'hold_periods'])
        t_start = times[idx]
        t_end_idx = min(idx + max(hold, 1), len(times) - 1)
        t_end = times[t_end_idx]

        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

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

    # Recent SHORT entries
    short_idxs = display_labels.index[short_mask].tolist()
    for idx in short_idxs[-max_show:]:
        if idx >= len(display_df):
            continue
        entry_price = display_labels.loc[idx, 'entry_price']
        hold = int(display_labels.loc[idx, 'hold_periods'])
        t_start = times[idx]
        t_end_idx = min(idx + max(hold, 1), len(times) - 1)
        t_end = times[t_end_idx]

        tp_price = entry_price * (1 - tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)

        # TP line (red dashed, profit for short = price going down)
        fig.add_trace(go.Scatter(
            x=[t_start, t_end], y=[tp_price, tp_price],
            mode='lines', line=dict(color='#ff1744', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ))
        # SL line (orange dashed)
        fig.add_trace(go.Scatter(
            x=[t_start, t_end], y=[sl_price, sl_price],
            mode='lines', line=dict(color='#ff9100', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ))


@st.fragment(run_every=30)
def _render_auto_refresh_timer():
    """
    Background timer that checks if auto-refresh is due.
    Runs every 30 seconds to monitor the 5-minute refresh interval.
    """
    if st.session_state.auto_refresh_enabled and not st.session_state.is_refreshing:
        if st.session_state.last_data_refresh:
            elapsed = (datetime.now() - st.session_state.last_data_refresh).total_seconds()
            if elapsed >= AUTO_REFRESH_SECONDS:
                # Trigger full page rerun to perform refresh
                st.rerun()


def main():
    """Main dashboard entry point."""

    # Check for auto-refresh on page load (before rendering sidebar)
    if not st.session_state.is_refreshing and check_and_trigger_auto_refresh():
        perform_full_refresh()
        return  # Will rerun after refresh

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ Controls")

        # Auto-refresh section
        st.subheader("🔄 Data Refresh")

        # Auto-refresh toggle
        auto_refresh = st.toggle(
            "Auto-refresh (5 min)",
            value=st.session_state.auto_refresh_enabled,
            help="Automatically download fresh data from Binance every 5 minutes"
        )
        if auto_refresh != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh

        # Show last refresh time and countdown
        if st.session_state.last_data_refresh:
            last_refresh = st.session_state.last_data_refresh
            elapsed = (datetime.now() - last_refresh).total_seconds()
            remaining = max(0, AUTO_REFRESH_SECONDS - elapsed)
            mins, secs = divmod(int(remaining), 60)

            st.caption(f"📅 Last refresh: {last_refresh.strftime('%H:%M:%S')}")
            if st.session_state.auto_refresh_enabled:
                st.caption(f"⏱️ Next refresh in: {mins}m {secs}s")

                # Progress bar for countdown
                progress = 1 - (remaining / AUTO_REFRESH_SECONDS)
                st.progress(progress, text=None)
        else:
            st.caption("📅 No data refresh yet")

        # Manual refresh buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Full Refresh", help="Download fresh data from Binance"):
                perform_full_refresh()
        with col2:
            if st.button("♻️ Cache Only", help="Clear cache without re-downloading"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

        st.markdown("---")

        # Theme toggle
        st.subheader("🎨 Theme")
        theme_toggle = st.toggle(
            "Dark Mode",
            value=st.session_state.dark_mode,
            help="Toggle between dark and light mode"
        )
        if theme_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = theme_toggle
            st.rerun()

        st.markdown("---")
        st.subheader("Timeframe Groups")

        # Group multiselect — page-level filter
        selected_groups = st.multiselect(
            "Select Groups",
            TF_GROUP_NAMES,
            default=TF_GROUP_NAMES,
            help="Filter all dashboard views by timeframe group",
            key="tf_groups"
        )
        # Compute active timeframes preserving TIMEFRAME_ORDER ordering
        _active_set = set()
        for g in selected_groups:
            _active_set.update(TF_GROUPS[g])
        active_tfs = [tf for tf in TIMEFRAME_ORDER if tf in _active_set]
        if not active_tfs:
            active_tfs = TIMEFRAME_ORDER  # fallback: show all if nothing selected

        # Show which TFs are active
        st.caption(f"Active: {', '.join(active_tfs)} ({len(active_tfs)} TFs)")

        st.markdown("---")
        st.subheader("Filters")

        # Timeframe filter for acceleration chart (scoped to active groups)
        _accel_defaults = [tf for tf in ["3D", "1D", "4H", "1H"] if tf in active_tfs]
        selected_tfs = st.multiselect(
            "Acceleration Chart Timeframes",
            active_tfs,
            default=_accel_defaults or active_tfs[:4],
            help="Select timeframes to show in acceleration chart"
        )

        # Timeframe for angle chart (scoped to active groups)
        angle_chart_tf = st.selectbox(
            "Angle Chart Timeframe",
            active_tfs,
            index=0,
            help="Select timeframe for angle chart with confidence bands"
        )

        st.markdown("---")
        st.subheader("View Options")

        show_heatmap = st.checkbox("Show Signal Heatmap", value=True)
        show_reversal_heatmap = st.checkbox("Show Reversal Heatmap", value=True)
        show_slope_heatmap = st.checkbox("Show Slope Heatmap", value=True)
        show_accel_heatmap = st.checkbox("Show Acceleration Heatmap", value=True)
        show_gauges = st.checkbox("Show Probability Gauges", value=True)
        show_reversals = st.checkbox("Show Reversal Signals", value=True)
        show_confluence = st.checkbox("Show Confluence Score", value=True)
        show_timeline = st.checkbox("Show Signal Timeline", value=True)
        show_angle_chart = st.checkbox("Show Angle Chart", value=True)
        show_price_overlay = st.checkbox("Show Price Overlay", value=True)
        show_calendar = st.checkbox("Show Prediction Calendar", value=True)

        st.markdown("---")
        st.subheader("Validation / Debug Views")

        show_unified_validation = st.checkbox("Unified Signal Validation", value=False)
        show_prediction_graph = st.checkbox("Prediction (cc / ccc)", value=False)
        show_target_labeling = st.checkbox("Target Labeling (SL/TP)", value=False)

        # Selectors for unified validation view (scoped to active groups)
        if show_unified_validation:
            val_tf_select = st.selectbox("Validation Timeframe", active_tfs, index=0, key="val_tf")
            _ws_options = [f"df (w{WINDOW_SIZES[0]})"] + [f"df{i} (w{WINDOW_SIZES[i]})" for i in range(1, len(WINDOW_SIZES))]
            _ws_selected = st.multiselect("Validation Windows", _ws_options, default=[_ws_options[0]], key="val_ws")
            val_ws_select = [WINDOW_SIZES[_ws_options.index(lbl)] for lbl in _ws_selected] if _ws_selected else [WINDOW_SIZES[0]]
        else:
            val_tf_select, val_ws_select = active_tfs[0], [WINDOW_SIZES[0]]

        # Selectors for target labeling view
        if show_target_labeling:
            tl_tf_select = st.selectbox("Candle Timeframe", active_tfs, index=0, key="tl_tf")
            tl_sl = st.slider("Stop Loss %", 0.5, 10.0, 3.0, 0.5, key="tl_sl")
            tl_tp = st.slider("Take Profit %", 1.0, 20.0, 6.0, 0.5, key="tl_tp")
            tl_hold = st.slider("Max Hold (bars)", 10, 100, 50, 5, key="tl_hold")
        else:
            tl_tf_select, tl_sl, tl_tp, tl_hold = active_tfs[0], 3.0, 6.0, 50

        st.markdown("---")
        st.caption(f"📁 Data: {DATA_DIR}")
        st.caption(f"🕐 Page loaded: {datetime.now().strftime('%H:%M:%S')}")

    # Load data
    with st.spinner("Loading data..."):
        processor = get_processor()
        logic, aggregated = get_signal_analysis(processor)

    # Filter aggregated signals by active timeframe groups
    filtered_agg = filter_aggregated_by_tfs(aggregated, active_tfs)

    # Extract signals data for cached heatmap computations (filtered)
    signals_data, signals_hash = extract_signals_data(filtered_agg)

    # Header with price ticker
    col_title, col_price = st.columns([2, 1])
    with col_title:
        st.title("📊 BTC/USDT Signal Board")
    with col_price:
        render_price_ticker()

    # Row 1: Main Signal + Distribution + Metrics
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        render_main_signal(filtered_agg)

    with col2:
        render_signal_distribution(filtered_agg)

    with col3:
        st.markdown("### 📊 Metrics")
        st.metric("Total Signals", filtered_agg.total_signals)
        strong = len([s for s in filtered_agg.signals if s.strength >= 0.5])
        st.metric("Strong Signals", strong)
        crossings = len([s for s in filtered_agg.signals if s.is_crossing])
        st.metric("Crossings", crossings)

    st.markdown("---")

    # Row 2: Probability Gauges
    if show_gauges:
        render_probability_gauges(filtered_agg)
        st.markdown("---")

    # Row 3: Signal Heatmap + Crossing Signals
    if show_heatmap:
        col1, col2 = st.columns([2, 1])
        with col1:
            render_signal_heatmap(filtered_agg, signals_data, signals_hash, active_tfs)
        with col2:
            render_crossing_signals_chart(filtered_agg)
        st.markdown("---")

    # Row 3.5: Additional Heatmaps (Reversal, Slope, Acceleration)
    # Using cached matrix computations for performance
    _atf = active_tfs  # capture for lambdas
    heatmaps_to_show = []
    if show_reversal_heatmap:
        heatmaps_to_show.append(("reversal", lambda agg, t=_atf: render_reversal_heatmap(agg, signals_data, signals_hash, t)))
    if show_slope_heatmap:
        heatmaps_to_show.append(("slope", lambda agg, t=_atf: render_slope_heatmap(agg, signals_data, signals_hash, t)))
    if show_accel_heatmap:
        heatmaps_to_show.append(("accel", lambda agg, t=_atf: render_acceleration_heatmap(agg, signals_data, signals_hash, t)))

    if heatmaps_to_show:
        if len(heatmaps_to_show) == 1:
            heatmaps_to_show[0][1](filtered_agg)
        elif len(heatmaps_to_show) == 2:
            col1, col2 = st.columns(2)
            with col1:
                heatmaps_to_show[0][1](filtered_agg)
            with col2:
                heatmaps_to_show[1][1](filtered_agg)
        else:  # 3 heatmaps
            col1, col2, col3 = st.columns(3)
            with col1:
                heatmaps_to_show[0][1](filtered_agg)
            with col2:
                heatmaps_to_show[1][1](filtered_agg)
            with col3:
                heatmaps_to_show[2][1](filtered_agg)
        st.markdown("---")

    # Row 4: Reversals + Acceleration Zones
    if show_reversals:
        col1, col2 = st.columns(2)
        with col1:
            render_reversal_signals(logic, filtered_agg, active_tfs)
        with col2:
            render_acceleration_zones(filtered_agg)
        st.markdown("---")

    # Row 5: Acceleration Line Chart with optional price overlay
    render_acceleration_linechart(processor, selected_tfs, show_price=show_price_overlay)

    st.markdown("---")

    # Row 6: Angle Chart with Confidence Bands (lazy loaded)
    if show_angle_chart:
        with st.expander("📐 Angle Chart with Confidence Bands", expanded=False):
            render_angle_chart_with_confidence(processor, angle_chart_tf)

    # Row 7: Signal Timeline (lazy loaded)
    if show_timeline:
        with st.expander("📅 Signal History Timeline", expanded=False):
            render_signal_history_timeline(filtered_agg, active_tfs)

    # Row 8: Confluence Score (lazy loaded)
    if show_confluence:
        with st.expander("🎯 Confluence Score", expanded=False):
            render_confluence_score(filtered_agg, active_tfs)

    # Row 8.5: Signal Prediction Calendar
    if show_calendar:
        st.markdown("---")
        with st.expander("📅 Signal Prediction Calendar (3h history + 4h forecast)", expanded=True):
            render_calendar_table(filtered_agg, logic)

    st.markdown("---")

    # Row 9: Matrix + Decision
    col1, col2 = st.columns([2, 1])
    with col1:
        render_timeframe_matrix(filtered_agg, active_tfs)
    with col2:
        render_decision_summary(filtered_agg)

    # Validation / Debug Views
    if show_unified_validation or show_prediction_graph or show_target_labeling:
        st.markdown("---")
        st.header("Validation Views")

        if show_unified_validation:
            with st.expander("Unified Signal Validation", expanded=True):
                render_unified_signal_validation(processor, logic, val_tf_select, val_ws_select)

        if show_prediction_graph:
            with st.expander("Prediction (cc / ccc)", expanded=True):
                render_prediction_graph(processor, logic, val_tf_select, val_ws_select)

        if show_target_labeling:
            with st.expander("Target Labeling (SL/TP Candle Chart)", expanded=True):
                render_target_labeling_chart(tl_tf_select, tl_sl, tl_tp, tl_hold)

    # Footer
    st.markdown("---")
    st.caption(f"BTC/USDT Signal Analysis | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Auto-refresh timer fragment (runs every 30 seconds to check if refresh needed)
    _render_auto_refresh_timer()


if __name__ == "__main__":
    main()
