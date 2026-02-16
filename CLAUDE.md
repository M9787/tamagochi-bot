# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cryptocurrency trading signal analysis system for BTCUSDT that:
1. Downloads kline data from Binance across **11 timeframes**
2. Performs rolling linear regression with **5 window sizes** (30, 60, 100, 120, 160)
3. Generates **55 signal sources** (11 TF x 5 windows) using reversal detection, angle crossings, and GMM acceleration filtering
4. Displays results in a Streamlit dashboard with a 55-row calendar table, heatmaps, and prediction views
5. **ML Pipeline**: XGBoost model trained on R/C/A heatmap features (X) with SL/TP trade labels (Y)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download fresh data from Binance
python data/downloader.py

# Launch Streamlit dashboard (primary interface)
python main.py --mode dashboard

# Run single analysis (console + CSV output)
python main.py --mode manual

# Continuous monitoring (refreshes every N minutes)
python main.py --mode continuous --interval 5

# Validate configuration and data availability
python main.py --validate

# Run with specific timeframes/windows
python main.py -t 1H,4H,1D -w 30,60,100

# Suppress CSV/file export
python main.py --mode manual --no-export

# Start Telegram bot
python main.py --mode telegram

# Start Discord bot
python main.py --mode discord

# ML Model Training (separate pipeline)
python model_training/train.py
```

## Architecture

```
main.py                        # CLI entry point, mode dispatcher
dashboard.py                   # Streamlit dashboard (21 render functions, ~3000 lines)
test_session.py                # Test session runner
│
├── core/                      # Signal processing engine
│   ├── config.py              # Centralized configuration (paths, TFs, windows, thresholds)
│   ├── analysis.py            # Core: iterative_regression, calculate_acceleration
│   ├── processor.py           # TimeframeProcessor: loads CSVs, runs analysis on all 55 TF/window combos
│   └── signal_logic.py        # Signal system: SignalLogic, PricePredictor, CalendarDataBuilder
│
├── data/                      # Data acquisition & labeling
│   ├── downloader.py          # Binance data downloader (writes to DATA_DIR)
│   └── target_labeling.py     # SL/TP target labeling encoder (+1/-1/0)
│
├── notifications/             # Alert & bot layer
│   ├── alerts.py              # AlertEngine: signal detection, deduplication, multi-channel dispatch
│   ├── console_alerts.py      # Colored console output, CSV logging, continuous mode
│   ├── telegram_bot.py        # Telegram bot integration
│   └── discord_bot.py         # Discord bot integration
│
├── model_training/            # ML Pipeline
│   ├── etl.py                 # Feature extraction (X) + label generation (Y)
│   ├── train.py               # XGBoost training (V1/V2/V3)
│   ├── visualize.py           # Results visualization (equity curve, confusion matrix, etc.)
│   ├── download_data.py       # ML-specific extended data downloader
│   ├── data/                  # ML training data CSVs
│   └── results/               # Saved models, predictions, plots
│
├── tests/                     # Unit tests
├── archive/                   # Legacy files (Power BI, old docs)
├── output/                    # Analysis output
└── logs/                      # Log files
```

### Data Flow (Current)

```
Binance API → CSVs → TimeframeProcessor.load_all_data()
  → iterative_regression() → calculate_acceleration()
  → SignalLogic.run_analysis()
    → detect_cycle_events()      (reversal: 5-point pattern)
    → detect_angle_crossings()   (crossing: angle crossover between windows)
    → fit_gmm_acceleration()     (accel quality: GMM clustering)
    → aggregate_signals()        (convergence scoring with TF weights)
  → CalendarDataBuilder.build_calendar_df()
    → PricePredictor.predict_next()  (2-step linear extrapolation)
  → Dashboard (Streamlit)
```

### ML Data Flow

```
processor_results (55 TF-window combos)
  → ETL: extract R/C/A flags + angle/slope_f/accel per timestamp
    → X: 3 feature versions (flat ~330 cols, score ~60 cols, binary ~165 cols)
  → SL/TP labeling on price data
    → Y: +1 (LONG TP hit), -1 (SHORT TP hit), 0 (No Trade)
  → XGBoost training → backtest → visualization
```

## Key Configuration (config.py)

- `DATA_DIR` - Where CSVs are read from (must match data extraction script output)
- `OUTPUT_DIR` - Where analysis results are written
- `TIMEFRAME_ORDER` - 11 timeframes: `3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M`
- `WINDOW_SIZES` - `[30, 60, 100, 120, 160]` mapping to `df, df1, df2, df3, df4`
- `CALENDAR_HISTORY_HOURS` (3), `CALENDAR_FORWARD_HOURS` (4), `CALENDAR_INTERVAL_MINUTES` (15)
- `CHART_COLORS` - Per-window colors for Plotly charts
- Bot tokens via environment: `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN`

## Signal System (signal_logic.py) - PRIMARY

### Classes

| Class | Purpose |
|-------|---------|
| `SignalLogic` | Main engine: reversal detection, GMM clustering, crossing detection, aggregation |
| `PricePredictor` | Linear extrapolation for 2-step forward prediction |
| `CalendarDataBuilder` | Builds the 55-row x time-column calendar grid with R/C/A scoring |
| `AggregatedSignal` | Dataclass holding final direction, convergence score, counts, quality |
| `Signal` | Individual signal with direction, strength, flags, metadata |

### Signal Scoring (0-3 points per cell)

Each of the 55 signal sources is scored with three independent detectors:

| Flag | Detector | Score | Logic |
|------|----------|-------|-------|
| **R** (Reversal) | 5-point pattern | +1 | Bottom: `[a>b>c>d<e]` → LONG. Peak: `[a<b<c<d>e]` → SHORT |
| **C** (Crossing) | Angle crossover | +1 | Angle lines from two windows cross (7 defined pairs) |
| **A** (Accel) | GMM quality | +1 | Acceleration in top 40% by GMM zone classification |

**Direction**: `slope_f > 0` → LONG (L), `slope_f < 0` → SHORT (S)

### Crossing Pairs

7 pairs between window sizes: `(30,60), (30,100), (60,100), (60,120), (100,120), (100,160), (120,160)`

### Timeframe Weights

`3D=5, 1D=4, 12H=3.5, 8H=3, 6H=2.5, 4H=2, 2H=1.5, 1H=1.2, 30M=1.1, 15M=1.0, 5M=0.8`

### Timeframe Groups

- **Youngs**: 5M, 15M, 30M — shape Adults/Balzaks, use for scalping
- **Adults**: 1H, 2H, 4H — core trading signals
- **Balzaks**: 6H, 8H, 12H — confirmation layer
- **Grans**: 1D, 3D — regime/trend direction

**Key insight**: Youngs reverse first → Adults follow → Balzaks confirm (5-10 min cascade)

### GMM Acceleration Zones

5 clusters via Gaussian Mixture Model: `VERY_DISTANT, DISTANT, BASELINE, CLOSE, VERY_CLOSE`

### Calendar Cell Format

```
Direction:Score:Flags[~]

L:3:RCA   = LONG, 3pts, Reversal+Crossing+Accel (historical)
S:2:RC~   = SHORT, 2pts, Reversal+Crossing (PREDICTED)
N:0:      = NEUTRAL, 0pts
```

- Historical cells: actual data lookup from processor results
- Predicted cells (future): linear extrapolation, marked with `~` suffix

### Prediction Formula (PricePredictor.predict_next)

```python
# Simple linear extrapolation from last two values
next = b + (b - a)    # where a = previous, b = current

# Two-step prediction
cc = b + (b - a)           # first prediction
ccc = cc + (cc - b)        # second prediction (= b + 2*(b-a))
```

Applied to angle, slope_f, and acceleration independently.

## Target Labeling (target_labeling.py)

### SL/TP Trade Outcome Encoder

For each candle, simulates LONG and SHORT trades forward:
- **+1**: LONG Take Profit hit first
- **-1**: SHORT Take Profit hit first
- **0**: Neither TP hit (No Trade / Max Hold)

```python
create_sl_tp_labels(price_data, sl_pct=3.0, tp_pct=6.0, max_hold_periods=50)
```

- Defaults: SL=3%, TP=6% (adjustable per run; train.py uses SL=2%, TP=6%, hold=288)
- Checks SL before TP on same bar (avoids bias)
- ATR-based SL/TP as default (manual override available)
- Validated visually on candlestick chart with entry arrows

## ML Pipeline (model_training/)

### Architecture: "Dr. Strange" Heatmap Reader

XGBoost reads 55-cell heatmap snapshots simultaneously — learning which convergence patterns of R/C/A across all TF-window combinations predict profitable trades.

### Feature Versions (X) — NEEDS REFACTOR

| Version | Description | Columns | Status |
|---------|-------------|---------|--------|
| **V1 Flat** | angle, slope_f, accel, R, C, A for all 55 combos | ~330 | Needs rework |
| **V2 Score** | Score (0-3) per 55 combos + convergence counts | ~60 | Needs rework |
| **V3 Binary** | R, C, A binary flags for all 55 combos | ~165 | Needs rework |

### Labels (Y)

SL/TP encoded: `+1` (LONG), `-1` (SHORT), `0` (No Trade)

Alternative: `build_atr_labels()` in `etl.py` — ATR-based dynamic SL with fixed TP

### Optimization Targets

- **Win Rate** — % of trades hitting TP
- **Cumulative Profit** — P&L curve with fixed TP
- **Sharpe Ratio** — risk-adjusted return
- **Max Drawdown** — worst peak-to-trough

### Retraining

Manual trigger only. No auto-retraining for now.

## Core Algorithm (analysis.py)

### Iterative Regression

`iterative_regression(df, window_size)` - For each data point, maintains a sliding window of `2 * window_size` prices:
- **Backward window:** older N points
- **Forward window:** newer N points
- Uses `sqrt(price)` for calculations
- Performs `scipy.stats.linregress` on each half

**Outputs per row:** `slope_b`, `slope_f`, `intercept_b/f`, `p_value_b/f`, `corr`, `spearman`, `angle`, `actual`, `time`

### Acceleration

`calculate_acceleration(angle_series)` - First difference of angle values

## Dashboard (dashboard.py)

### Layout Overview

The Streamlit dashboard (port 8501) displays:

1. **Main Signal Panel** - Overall direction (LONG/SHORT/NEUTRAL), quality, convergence score
2. **Probability Gauges** - LONG/SHORT probability based on signal counts
3. **Heatmaps** (4 types) - Signal strength, reversal events, slope, acceleration across TF x window grid
4. **Crossing Signals** - Bar chart of crossing events by timeframe
5. **Reversal Signals** - Panel showing reversal events by timeframe
6. **Acceleration Zones** - GMM zone distribution + line chart with optional price overlay
7. **Angle Chart** - Angle values with confidence bands
8. **Signal Timeline** - Historical signal progression
9. **Confluence Score** - Convergence metrics across timeframes
10. **Calendar Table** - 55-row calendar with R/C/A scoring, color-coded by direction+score
11. **Timeframe Matrix** - Summary matrix + decision panel
12. **Price Ticker** - Live BTC price display
13. **Decision Summary** - Trading decision panel
14. **Signal Distribution** - Signal distribution chart

### Validation / Debug Views (sidebar toggles)

- **Unified Signal Validation** - Angle lines colored by GMM zone + reversal markers + crossing overlays
- **Prediction Graph** - Rolling linear extrapolation backtest overlay + difference charts with MAE
- **Target Labeling Chart** - Candlestick with SL/TP entry arrows (green=LONG, red=SHORT)

### Timeframe Group Filter

Page-level multi-select filter affecting all dashboard sections:
- **Youngs**: 5M, 15M, 30M
- **Adults**: 1H, 2H, 4H
- **Balzaks**: 6H, 8H, 12H
- **Grans**: 1D, 3D

### Auto-Refresh

Dashboard supports auto-refresh (default: 5 min) with progress bar countdown. Manual "Full Refresh" re-downloads data from Binance.

## Reference Documentation

- **User Story**: `.claude/user story/user_story_signals.md` - Full specification with examples
- **Crossing Examples**: `.claude/long example - angle cross*.png`, `.claude/Short Example - angle cross*.png`
- **Project Checkpoint** (2026-02-05): `.claude/PROJECT_CHECKPOINT.md`
- **Quick Reference**: `.claude/QUICK_REFERENCE.md`
- **Architecture Diagrams**: `.claude/ARCHITECTURE.md`
- **Component Docs**: `.claude/PROJECT_COMPONENTS.md`
