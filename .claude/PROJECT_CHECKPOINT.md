# Tamagochi Project Checkpoint
## Cryptocurrency Trading Signal Analysis System

**Date:** 2026-02-05
**Version:** 1.0
**Status:** Working Streamlit Dashboard

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Concepts & Terminology](#2-core-concepts--terminology)
3. [Data Pipeline](#3-data-pipeline)
4. [Signal Generation Strategy](#4-signal-generation-strategy)
5. [Scoring System](#5-scoring-system)
6. [Prediction Methodology](#6-prediction-methodology)
7. [Calendar Table](#7-calendar-table)
8. [Dashboard Visuals](#8-dashboard-visuals)
9. [File Structure](#9-file-structure)
10. [Code Examples](#10-code-examples)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Project Overview

### Purpose
Analyze BTCUSDT price data across multiple timeframes to generate trading signals (LONG/SHORT) using:
- Rolling linear regression
- 5-point reversal pattern detection
- Acceleration quality filtering (GMM zones)
- Angle crossing detection between windows
- Convergence scoring across 55 signal sources

### Key Numbers
- **11 Timeframes:** 3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M
- **5 Window Sizes:** 30 (df), 60 (df1), 100 (df2), 120 (df3), 160 (df4)
- **55 Signal Sources:** 11 TF × 5 windows
- **Max Score per Cell:** 3 points

---

## 2. Core Concepts & Terminology

### 2.1 Timeframes
| Label | Interval | Lookback Days | Weight |
|-------|----------|---------------|--------|
| 3D | 3 days | 1200 | Highest |
| 1D | 1 day | 700 | Very High |
| 12H | 12 hours | 200 | High |
| 8H | 8 hours | 180 | High |
| 6H | 6 hours | 180 | Medium-High |
| 4H | 4 hours | 100 | Medium |
| 2H | 2 hours | 50 | Medium |
| 1H | 1 hour | 30 | Low-Medium |
| 30M | 30 minutes | 30 | Low |
| 15M | 15 minutes | 15 | Low |
| 5M | 5 minutes | 5 | Lowest |

**Rule:** Higher timeframe = higher impact weight

### 2.2 Window Sizes (Rolling Regression)
| Label | Size | Description |
|-------|------|-------------|
| df | 30 | Fastest, most reactive |
| df1 | 60 | Fast |
| df2 | 100 | Medium |
| df3 | 120 | Slow |
| df4 | 160 | Slowest, smoothest |

### 2.3 Regression Metrics
For each data point, we split a window of `2 × window_size` prices into:
- **Backward window:** Older N points
- **Forward window:** Newer N points

Metrics calculated:
| Metric | Description |
|--------|-------------|
| `slope_b` | Backward slope (past trend) |
| `slope_f` | Forward slope (current/future trend) |
| `angle` | Angle between backward and forward slopes |
| `spearman` | Spearman correlation |
| `acceleration` | Rate of change of angle |

### 2.4 Direction Types
| Direction | Condition | Meaning |
|-----------|-----------|---------|
| LONG (L) | slope_f > 0 | Price trending up |
| SHORT (S) | slope_f < 0 | Price trending down |
| NEUTRAL (N) | slope_f ≈ 0 | No clear trend |

### 2.5 Reversal Types (5-Point Pattern)
| Type | Pattern | Meaning |
|------|---------|---------|
| BOTTOM | `[a>b>c>d<e]` | Decreasing then reversal UP → potential LONG |
| PEAK | `[a<b<c<d>e]` | Increasing then reversal DOWN → potential SHORT |
| NONE | Neither pattern | No reversal detected |

**Example:**
```
BOTTOM pattern: angles = [45, 30, 15, 5, 20]
  - 45 > 30 > 15 > 5 (decreasing)
  - 5 < 20 (reversal up)
  - Result: BOTTOM detected → LONG signal component
```

### 2.6 Acceleration Zones (GMM Clustering)
Acceleration values are clustered into 5 zones:
| Zone | Quality | Signal Impact |
|------|---------|---------------|
| VERY_DISTANT | High | Strong signal boost |
| DISTANT | High | Good signal boost |
| BASELINE | Neutral | No boost |
| CLOSE | Low | Weak/ignore |
| VERY_CLOSE | Low | Very weak/ignore |

**Implementation:** Top 40% by absolute value = distant/very_distant (quality signals)

### 2.7 Angle Crossings
Like Moving Average crossovers, but for angles between adjacent windows:
| Crossing | Meaning |
|----------|---------|
| df up, df1 down | SHORT (smaller window above larger) |
| df down, df1 up | LONG (smaller window below larger) |

---

## 3. Data Pipeline

### 3.1 Data Flow
```
Binance API
    ↓
data_downloader.py (downloads klines)
    ↓
CSV files (testing_data_{TF}.csv)
    ↓
TimeframeProcessor.load_all_data()
    ↓
iterative_regression() → slope_b, slope_f, angle, acceleration
    ↓
SignalLogic.run_analysis()
    ↓
CalendarDataBuilder.build_calendar_df()
    ↓
Dashboard visualization
```

### 3.2 CSV Format
Each timeframe CSV contains:
```
open_time, open, high, low, close, volume, close_time, ...
```

### 3.3 Iterative Regression Algorithm
```python
def iterative_regression(df, window_size):
    """
    For each point i:
    1. backward_window = prices[i - window_size : i]
    2. forward_window = prices[i : i + window_size]
    3. Apply sqrt() transformation to prices
    4. Linear regression on each window
    5. Calculate slope_b, slope_f, angle between them
    """
```

---

## 4. Signal Generation Strategy

### 4.1 Signal Formula
```
signal = reversal_event + GMM_quality + direction(slope_f)
```

### 4.2 Signal Components
| Component | Points | Condition |
|-----------|--------|-----------|
| Reversal | +1 | 5-point pattern detected (PEAK or BOTTOM) |
| Crossing | +1 | Angle crossing with adjacent window |
| Acceleration Quality | +1 | Acceleration in distant/very_distant zone |

### 4.3 Quality Filter
- **High quality signals:** distant/very_distant zone + reversal + clear direction
- **Low quality / False positives:** other GMM zones

### 4.4 Signal Detection Flow
```python
def detect_signal(data_row, all_accels, adjacent_angles):
    # 1. Get direction from slope_f
    direction = "L" if slope_f > 0 else "S" if slope_f < 0 else "N"

    # 2. Check for reversal (5-point pattern)
    has_reversal = detect_5_point_reversal(angles[-5:])

    # 3. Check for angle crossing
    has_crossing = check_angle_crossing(angle, adjacent_angle)

    # 4. Check acceleration quality
    accel_quality = check_gmm_zone(acceleration, all_accels)

    # 5. Compute score
    score = sum([has_reversal, has_crossing, accel_quality])

    return f"{direction}:{score}:{flags}"
```

---

## 5. Scoring System

### 5.1 Cell Format
```
Direction:Score:Flags

Examples:
- L:3:RCA  = LONG, score 3, has Reversal + Crossing + Accel
- S:2:RC   = SHORT, score 2, has Reversal + Crossing
- L:1:A    = LONG, score 1, only Accel Quality
- N:0:     = NEUTRAL, score 0, no components
```

### 5.2 Flags
| Flag | Meaning |
|------|---------|
| R | Reversal detected (5-point pattern) |
| C | Crossing detected (angle crossover) |
| A | Acceleration quality (distant/very_distant) |

### 5.3 Score Interpretation
| Score | Strength | Confidence |
|-------|----------|------------|
| 3 | Maximum | Very high - all factors aligned |
| 2 | Strong | High - two factors aligned |
| 1 | Weak | Low - single factor only |
| 0 | None | No signal |

### 5.4 Aggregation
Per time slot (across 55 TF-window combinations):
- **Total Score:** Sum of all individual scores
- **Reversals:** Count of cells with R flag
- **Crossings:** Count of cells with C flag
- **Accel Quality:** Count of cells with A flag
- **LONG/SHORT:** Count of L vs S directions
- **Strength %:** Dominant direction percentage

---

## 6. Prediction Methodology

### 6.1 The child() Formula
```python
def child(a, b, c):
    """
    Predict next value from three historical values.

    Args:
        a: Previous of previous value (oldest)
        b: Previous value (middle)
        c: Current value (newest)

    Returns:
        Predicted next value
    """
    angle = c - b                      # Current momentum
    curve = abs(a - 2*b + c)           # Curvature (acceleration)
    ratio = min(abs(c) / (abs(b) + 1), 1.5)  # Relative strength

    return c + angle * ratio / (1 + curve/2)
```

### 6.2 Two-Step Prediction
```python
# Given values: [..., a, b, c]
cc = child(a, b, c)        # Step 1: predict next
ccc = child(b, c, cc)      # Step 2: predict next-next
```

### 6.3 What Gets Predicted
| Metric | Purpose |
|--------|---------|
| Angle | Reversal detection in future |
| Slope_f | Direction prediction |
| Acceleration | Quality filtering for predictions |

### 6.4 Prediction Markers
- Historical data: `L:2:RC`
- Predicted data: `L:2:RC~` (tilde suffix)

---

## 7. Calendar Table

### 7.1 Structure
```
Main Grid:
  Rows: 55 (11 TF × 5 windows)
    - 3D-df, 3D-df1, 3D-df2, 3D-df3, 3D-df4
    - 1D-df, 1D-df1, ...
    - ...
    - 5M-df, 5M-df1, 5M-df2, 5M-df3, 5M-df4

  Columns: Time slots (15-minute intervals)
    - Last 3 hours (historical)
    - Next 4 hours (predicted)

Row Totals (right side):
  - ΣR: Sum of R flags across all time columns for this TF-window
  - ΣC: Sum of C flags across all time columns for this TF-window
  - ΣA: Sum of A flags across all time columns for this TF-window

Footer Rows (bottom):
  - TOTAL R: Sum of R flags across all 55 TF-windows per time column
  - TOTAL C: Sum of C flags across all 55 TF-windows per time column
  - TOTAL A: Sum of A flags across all 55 TF-windows per time column

Grand Totals (bottom-right intersection):
  - ΣR column / TOTAL R row: Total R count across entire calendar
  - ΣC column / TOTAL C row: Total C count across entire calendar
  - ΣA column / TOTAL A row: Total A count across entire calendar
```

### 7.2 Cell Values
| Value | Meaning |
|-------|---------|
| `L:2:RC` | LONG, score 2, Reversal+Crossing (historical) |
| `S:1:A~` | SHORT, score 1, Accel Quality (predicted) |
| `L:3:RCA` | LONG, score 3, all flags: Reversal+Crossing+Accel |
| `N:0:` | NEUTRAL, score 0, no components |
| `-` | No data available for this cell |
| `?` | Cannot compute prediction |

**Cell Format:** `Direction:Score:Flags[~]`
- **Direction:** L (LONG), S (SHORT), N (NEUTRAL) - from slope_f
- **Score:** 0-3 points (sum of flag contributions)
- **Flags:** R (Reversal), C (Crossing), A (Accel Quality)
- **~:** Suffix indicates predicted (future) value

### 7.3 Color Coding
| Color | Meaning | Intensity Formula |
|-------|---------|-------------------|
| Green | LONG direction | `0.2 + (score * 0.25)` → 0.2, 0.45, 0.7, 0.95 |
| Red | SHORT direction | `0.2 + (score * 0.25)` → 0.2, 0.45, 0.7, 0.95 |
| Gray | NEUTRAL | Low opacity (intensity * 0.5) |
| Dashed border | Prediction (future) | 2px dashed white with 0.5 opacity |
| Yellow | Totals (footer/row totals) | Intensity based on count (0-0.8) |
| Dark background | TOTAL labels | rgba(52, 73, 94, 0.9) |

**Footer/Total Styling:**
- TOTAL R/C/A labels: Dark background (#34495e at 90%) with yellow text (#f1c40f)
- Total count cells: Yellow intensity based on value (higher count = brighter yellow)
- Zero counts: Muted gray color (#7f8c8d)

### 7.4 Building Process
```python
builder = CalendarDataBuilder(
    history_hours=3,    # 3 hours back
    forward_hours=4,    # 4 hours forward
    interval_minutes=15 # 15-min granularity
)

calendar_df = builder.build_calendar_df(
    signals,           # Signal list from analysis
    processor_results  # Regression results for historical lookup
)
```

### 7.5 Row and Column Totals Calculation

**Row Totals (per TF-window):**
```python
# For each row in calendar:
for col in time_columns:
    val = row[col]
    if ':' in val:
        flags = val.replace('~', '').split(':')[2]  # Get flags part
        if 'R' in flags: r_count += 1
        if 'C' in flags: c_count += 1
        if 'A' in flags: a_count += 1
# Add as columns: ΣR, ΣC, ΣA
```

**Column Totals (footer rows):**
```python
# For each time column:
for val in calendar_df[col]:
    if ':' in val:
        flags = val.replace('~', '').split(':')[2]
        if 'R' in flags: r_count += 1
        if 'C' in flags: c_count += 1
        if 'A' in flags: a_count += 1
# Add as rows: TOTAL R, TOTAL C, TOTAL A
```

**Grand Totals:**
```python
grand_r = sum(row_r_totals)  # Total R across all cells
grand_c = sum(row_c_totals)  # Total C across all cells
grand_a = sum(row_a_totals)  # Total A across all cells
```

---

## 8. Dashboard Visuals

### 8.1 Layout Overview
```
┌─────────────────────────────────────────────────────────────┐
│ SIDEBAR                                                       │
│ ┌─────────────────────┐                                      │
│ │ [Refresh Data]      │ ← Clears cache (TTL=300s)           │
│ │ Theme Toggle        │                                      │
│ │ Filters             │                                      │
│ │ View Options        │                                      │
│ └─────────────────────┘                                      │
├─────────────────────────────────────────────────────────────┤
│  Header: BTC/USDT Signal Board                + Price Ticker │
├─────────────────────────────────────────────────────────────┤
│  Analysis Summary                                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ LONG    │ │ SHORT   │ │ SIGNALS │ │ STATUS  │           │
│  │   12    │ │    8    │ │   66    │ │   OK    │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Convergence Summary (Plotly Bar Chart)                     │
│  [====LONG====]     ← positive bars                         │
│  ───────────────────┼─────────────────── NOW line           │
│  [===SHORT===]      ← negative bars                         │
├─────────────────────────────────────────────────────────────┤
│  Aggregated Signal Score by Time (Heatmap Table)            │
│  Time | Reversals | Crossings | Accel | Total | L | S | Dir │
│  00:00│    5      │     3     │   2   │  10   │ 8 │ 3 │LONG │
│  00:15│    3      │     2     │   1   │   6   │ 4 │ 5 │SHORT│
│  ...                                                         │
├─────────────────────────────────────────────────────────────┤
│  Full Calendar (55 rows × N columns + totals)               │
│  TF-Window │ 00:00 │ 00:15 │ ... │ 03:00~ │ ΣR │ ΣC │ ΣA  │
│  3D-df     │ L:2:RC│ L:1:R │ ... │ L:1:R~ │  5 │  3 │  2  │
│  3D-df1    │ L:1:C │ L:1:A │ ... │ S:1:C~ │  2 │  4 │  1  │
│  ...                                                         │
│  5M-df4    │ S:1:A │ N:0:  │ ... │ L:0:~  │  1 │  0 │  3  │
│  ─────────────────────────────────────────────────────────  │
│  TOTAL R   │   8   │   5   │ ... │   4    │ 245│    │     │
│  TOTAL C   │   3   │   4   │ ... │   2    │    │ 156│     │
│  TOTAL A   │   6   │   3   │ ... │   5    │    │    │ 198 │
├─────────────────────────────────────────────────────────────┤
│  Reversal Signals Panel                                      │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ PEAK (SHORT)     │  │ BOTTOM (LONG)    │                │
│  │ - 4H-df at 12:30 │  │ - 1H-df at 13:00 │                │
│  │ - 2H-df1 at 12:45│  │ - 30M-df at 13:15│                │
│  └──────────────────┘  └──────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  Heatmaps (Reversal / Slope / Acceleration)                 │
│  [Visual matrices showing patterns across TF-windows]       │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Visual Components

#### 8.2.1 Convergence Bar Chart
- **X-axis:** Time slots (15-min intervals)
- **Y-axis:** Count
- **Green bars (up):** LONG count
- **Red bars (down):** SHORT count
- **Yellow dashed line:** NOW marker

#### 8.2.2 Aggregated Score Table
| Column | Color | Description |
|--------|-------|-------------|
| Time | None | Time slot label |
| Reversals | Blue (intensity) | Count of R flags |
| Crossings | Blue (intensity) | Count of C flags |
| Accel Quality | Blue (intensity) | Count of A flags |
| Total Score | Purple (intensity) | Sum of all scores |
| LONG | Green (intensity) | Count of L directions |
| SHORT | Red (intensity) | Count of S directions |
| Direction | Green/Red | Dominant direction |
| Strength % | Yellow-Green | Percentage of dominant |

#### 8.2.3 Calendar Table
- **Row index:** TF-Window (e.g., "15M-df")
- **Columns:** Time slots + row totals (ΣR, ΣC, ΣA)
- **Footer rows:** TOTAL R, TOTAL C, TOTAL A per time column
- **Cell color:** Green (LONG) / Red (SHORT) with intensity by score
- **Dashed border:** Predictions (future columns)
- **Total cells:** Yellow intensity based on count value
- **TOTAL labels:** Dark background with yellow text

#### 8.2.4 Sidebar Controls
| Control | Function |
|---------|----------|
| Refresh Data button | Clears `st.cache_resource` and `st.cache_data`, then reruns |
| Theme Toggle | Switch between dark/light mode |
| Acceleration Chart Timeframes | Multi-select for chart filtering |
| Angle Chart Timeframe | Single select for detailed view |
| View Options | Checkboxes for showing/hiding components |

#### 8.2.5 Caching
| Cache Type | TTL | Purpose |
|------------|-----|---------|
| `@st.cache_resource(ttl=300)` | 5 min | TimeframeProcessor (data loading) |
| `@st.cache_resource(ttl=300)` | 5 min | Signal analysis results |
| `@st.cache_data(ttl=300)` | 5 min | Heatmap matrix computations |
| `@st.cache_data(ttl=60)` | 1 min | BTC price ticker |

**Refresh behavior:**
- Click "Refresh Data" in sidebar clears all caches
- Forces fresh data load from CSVs
- Re-runs all signal analysis

---

## 9. File Structure

```
Tamagochi/
├── config.py              # Centralized configuration
├── processor.py           # TimeframeProcessor: loads CSVs, runs regression
├── analysis.py            # Core algorithms: iterative_regression, acceleration
├── signal_logic.py        # SignalLogic, PricePredictor, CalendarDataBuilder
├── dashboard.py           # Streamlit UI
├── main.py               # CLI entry point
├── alerts.py             # AlertEngine for notifications
├── target_labeling.py  # Majority vote aggregation
├── console_alerts.py     # Console output formatting
├── telegram_bot.py       # Telegram integration
├── discord_bot.py        # Discord integration
├── data_downloader.py  # Binance data downloader
├── requirements.txt      # Dependencies
├── CLAUDE.md            # Project instructions
└── .claude/
    ├── PROJECT_CHECKPOINT.md  # This file - comprehensive documentation
    ├── QUICK_REFERENCE.md     # Quick lookup card
    └── ARCHITECTURE.md        # System architecture and data flow
```

---

## 10. Code Examples

### 10.1 Running the Dashboard
```bash
# Install dependencies
pip install -r requirements.txt

# Download fresh data
python "data_downloader.py"

# Launch dashboard
python -m streamlit run dashboard.py --server.port 8501
```

### 10.2 Signal Detection
```python
from signal_logic import SignalLogic

# Initialize and run analysis
logic = SignalLogic()
aggregated = logic.run_analysis()

# Get signals
for signal in aggregated.signals:
    print(f"{signal.timeframe}-{signal.window}: {signal.direction} "
          f"reversal={signal.reversal_type} accel_zone={signal.accel_zone}")
```

### 10.3 Building Calendar
```python
from signal_logic import CalendarDataBuilder

builder = CalendarDataBuilder(
    history_hours=3,
    forward_hours=4,
    interval_minutes=15
)

calendar_df = builder.build_calendar_df(
    aggregated.signals,
    processor_results=logic.processor.results
)

# Get aggregated scores
score_table = builder.get_aggregated_score_table(calendar_df)
```

### 10.4 Prediction
```python
from signal_logic import PricePredictor

# Predict next angle
angles = [45.2, 42.1, 38.5]  # Last 3 angles
a, b, c = angles[-3], angles[-2], angles[-1]

pred_cc = PricePredictor.child(a, b, c)   # Next angle
pred_ccc = PricePredictor.child(b, c, pred_cc)  # Next-next angle

print(f"Predicted angles: {pred_cc:.2f}, {pred_ccc:.2f}")
```

---

## 11. Configuration Reference

### 11.1 Key Settings (config.py)
```python
# Timeframes
TIMEFRAME_ORDER = ["3D", "1D", "12H", "8H", "6H", "4H", "2H", "1H", "30M", "15M", "5M"]

# Window sizes
WINDOW_SIZES = [30, 60, 100, 120, 160]

# Calendar settings
CALENDAR_HISTORY_HOURS = 3
CALENDAR_FORWARD_HOURS = 4
CALENDAR_INTERVAL_MINUTES = 15

# Paths
DATA_DIR = Path(r"...\Live Data\data")
OUTPUT_DIR = BASE_DIR / "output"
```

### 11.2 Quality Thresholds
```python
# Acceleration quality: top 40% = distant/very_distant
ACCEL_QUALITY_PERCENTILE = 60  # Values above 60th percentile

# Alert cooldown
ALERT_COOLDOWN_SECONDS = 300  # 5 minutes between duplicate alerts
```

---

## Quick Reference Card

### Signal Formula
```
signal = reversal_event + GMM_quality + direction(slope_f)
```

### Cell Format
```
Direction:Score:Flags[~]
L:2:RC~  = LONG, 2 points, Reversal+Crossing, Predicted
```

### 5-Point Patterns
```
BOTTOM: [a>b>c>d<e] → LONG potential
PEAK:   [a<b<c<d>e] → SHORT potential
```

### child() Formula
```python
child(a,b,c) = c + (c-b) * min(|c|/(|b|+1), 1.5) / (1 + |a-2b+c|/2)
```

### Dashboard URL
```
http://localhost:8501
```

---

*Generated by Claude Code - 2026-02-05*
