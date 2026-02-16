# Project Components Reference

Consolidated from 7 layered component documents.

---

# 01 - Input Layer

## Data Source
Binance REST API via `python-binance` client (`get_historical_klines`).

## CSV Structure
```
Open Time, Close
2025-01-01 00:00:00+00:00, 42150.50
```
- **Open Time**: UTC timestamp (ISO 8601)
- **Close**: Closing price (float)

## Timeframes & Lookbacks
| Label | Interval | Lookback (days) |
|-------|----------|-----------------|
| 3D    | 3d       | 1200            |
| 1D    | 1d       | 700             |
| 12H   | 12h      | 200             |
| 8H    | 8h       | 180             |
| 6H    | 6h       | 180             |
| 4H    | 4h       | 100             |
| 2H    | 2h       | 50              |
| 1H    | 1h       | 30              |
| 30M   | 30m      | 30              |
| 15M   | 15m      | 15              |
| 5M    | 5m       | 5               |

## File Naming
```
testing_data_{TF}.csv
```

## Data Flow
```
Binance API → fetch_klines_with_retry() → normalize_klines() → DATA_DIR/testing_data_{TF}.csv
```

## Key Files
- `data_downloader.py`: Download script
- `config.py`: `DATA_DIR`, `TIMEFRAMES` dict

---

# 02 - Regression Layer

## Core Function
```python
iterative_regression(df, window_size, cut_index=None)
```

## Sliding Window
- Total window: `2 x window_size`
- Split: backward (older N) | forward (newer N)

## Price Transformation
```python
price_column = df['Close'].map(np.sqrt)
```

## Linear Regression
```python
slope_b, intercept_b, r_b, p_value_b, _ = linregress(step, backward)
slope_f, intercept_f, r_f, p_value_f, _ = linregress(step, forward)
```

## Angle Between Lines
```python
def angle_between_lines(slope1, slope2):
    theta1 = math.atan(slope1)
    theta2 = math.atan(slope2)
    return math.degrees(abs(theta1 - theta2))
```

## Output Columns
| Column      | Description                        |
|-------------|------------------------------------|
| count       | Row counter                        |
| slope_b     | Backward window slope              |
| slope_f     | Forward window slope               |
| intercept_b | Backward intercept                 |
| intercept_f | Forward intercept                  |
| p_value_b   | Backward p-value                   |
| p_value_f   | Forward p-value                    |
| corr        | Pearson correlation (b-f)          |
| spearman    | Spearman correlation (b-f)         |
| angle       | Angle between regression lines     |
| actual      | Current sqrt(price)                |
| time        | Timestamp                          |

## Key Files
- `analysis.py`: `iterative_regression()`, `angle_between_lines()`

---

# 03 - Feature Extraction Layer

## Acceleration Formula
```python
def calculate_acceleration(angles):
    return angles.diff()
```

## Window Sizes
| Label | Window Size |
|-------|-------------|
| df    | 30          |
| df1   | 60          |
| df2   | 100         |
| df3   | 120         |
| df4   | 160         |

## Signal Matrix
**55 signal sources** = 11 timeframes x 5 window sizes

## Minimum Data Requirements
| Window | Min Rows |
|--------|----------|
| 30     | 61       |
| 60     | 121      |
| 100    | 201      |
| 120    | 241      |
| 160    | 321      |

## Key Files
- `analysis.py`: `calculate_acceleration()`
- `processor.py`: `TimeframeProcessor.process_timeframe()`

---

# 04 - Signal Generation Layer

## Reversal Detection (5-Point Pattern)
```python
# Bottom: [a>b>c>d<e] -> LONG reversal
# Peak: [a<b<c<d>e] -> SHORT reversal
```

## Direction from Forward Slope
```python
slope_f > 0 -> LONG
slope_f < 0 -> SHORT
else -> NEUTRAL
```

## GMM Acceleration Zones (5 Clusters)
| Zone         | Quality    |
|--------------|------------|
| VERY_DISTANT | Strong     |
| DISTANT      | Good       |
| BASELINE     | Neutral    |
| CLOSE        | Weak       |
| VERY_CLOSE   | Very weak  |

High quality: `distant` or `very_distant` zone (top 40%).

## Crossing Pairs
```python
CROSSING_PAIRS = [
    (30, 60), (30, 100), (60, 100),
    (60, 120), (100, 120), (100, 160), (120, 160)
]
```

## Key Files
- `signal_logic.py`: `SignalLogic`, `detect_cycle_events()`

---

# 05 - Aggregation Layer

## Timeframe Weights
| TF   | Weight |
|------|--------|
| 3D   | 5.0    |
| 1D   | 4.0    |
| 12H  | 3.5    |
| 8H   | 3.0    |
| 6H   | 2.5    |
| 4H   | 2.0    |
| 2H   | 1.5    |
| 1H   | 1.2    |
| 30M  | 1.1    |
| 15M  | 1.0    |
| 5M   | 0.8    |

## Weighted Score Calculation
```python
for signal in signals:
    weight = TIMEFRAME_WEIGHTS[signal.timeframe]
    weighted_strength = signal.strength * weight
    if signal.direction == LONG:
        long_score += weighted_strength
    elif signal.direction == SHORT:
        short_score += weighted_strength
```

## Quality Thresholds
| Ratio  | Quality   |
|--------|-----------|
| >= 70% | VERY_HIGH |
| >= 50% | HIGH      |
| >= 30% | MEDIUM    |
| < 30%  | LOW       |

## Key Files
- `signal_logic.py`: `SignalLogic.aggregate_signals()`

---

# 06 - Prediction & Visualization Layer

## Child Prediction Formula
```python
def child(a, b, c):
    angle = c - b
    curve = abs(a - 2*b + c)
    ratio = min(abs(c) / (abs(b) + 1), 1.5)
    return c + angle * ratio / (1 + curve/2)
```

## Two-Step Prediction
```python
cc = child(a, b, c)      # Step 1
ccc = child(b, c, cc)    # Step 2
```
Applied to: `angle`, `slope_f`, `acceleration`

## Calendar Cell Format
```
Direction:Score:Flags[~]
```
| Example   | Meaning                         |
|-----------|---------------------------------|
| L:3:RCA   | LONG, 3pts, R+C+A flags         |
| S:2:RC~   | SHORT, 2pts, R+C, PREDICTED     |
| N:0:      | NEUTRAL, 0pts                   |

## R/C/A Scoring System
| Flag | Condition              | Points |
|------|------------------------|--------|
| R    | Reversal detected      | +1     |
| C    | Crossing detected      | +1     |
| A    | Accel quality (top 40%)| +1     |

## Calendar Dimensions
- **Rows**: 55 (11 TF x 5 windows)
- **Columns**: 3h history + 4h prediction (15min intervals)

## Key Files
- `signal_logic.py`: `PricePredictor.child()`, `CalendarDataBuilder`
- `dashboard.py`: Streamlit UI (port 8501)
- `config.py`: `CALENDAR_HISTORY_HOURS`, `CALENDAR_FORWARD_HOURS`

---

# 07 - Streamlit Dashboard Visuals

## Main Components
1. **Price Ticker** - `render_price_ticker()` - Auto-refresh every 60s
2. **Main Signal Indicator** - `render_main_signal()` - LONG/SHORT/NEUTRAL with quality
3. **Probability Gauges** - `render_probability_gauges()` - LONG/SHORT/NEUTRAL percentages
4. **Signal Distribution Donut** - `render_signal_distribution()`
5. **Signal Heatmap** (TF x Window) - `render_signal_heatmap()`
6. **Reversal Heatmap** - `render_reversal_heatmap()`
7. **Slope Heatmap** - `render_slope_heatmap()`
8. **Acceleration Heatmap** - `render_acceleration_heatmap()`
9. **Crossing Signals Bar Chart** - `render_crossing_signals_chart()`
10. **Reversal Signals Panel** - `render_reversal_signals()`
11. **Acceleration Zones** - `render_acceleration_zones()`
12. **Acceleration Line Chart** - `render_acceleration_linechart()` - w30 per TF
13. **Angle Chart with Confidence Bands** - `render_angle_chart_with_confidence()`
14. **Signal History Timeline** - `render_signal_history_timeline()`
15. **Confluence Score** - `render_confluence_score()`
16. **Prediction Calendar** - `render_calendar_table()` - 55 rows x N time columns
17. **Signal Matrix** - `render_timeframe_matrix()`
18. **Decision Summary** - `render_decision_summary()`

## Sidebar Controls
| Control | Type | Default |
|---------|------|---------|
| Auto-refresh (5 min) | toggle | True |
| Full Refresh | button | - |
| Dark Mode | toggle | True |
| Accel Chart TFs | multiselect | 3D,1D,4H,1H |
| Angle Chart TF | selectbox | 3D |
| View checkboxes | checkbox | True |

## Caching Strategy
| Decorator | TTL | Purpose |
|-----------|-----|---------|
| `@st.cache_data` | 60s | BTC price API |
| `@st.cache_resource` | 300s | Processor, SignalLogic |
| `@st.cache_data` | 300s | Heatmap matrices |
| `@st.fragment(run_every=30)` | - | Auto-refresh timer |

## Calendar Styling
| Area | Historical | Predictions |
|------|-----------|-------------|
| Background | Dark blue-gray | Dark purple |
| LONG cells | Pure green | Green with purple tint |
| SHORT cells | Pure red | Red with purple tint |
| NOW column | Orange border | - |

---

*Consolidated from 01-07 component documents*
