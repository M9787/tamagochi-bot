# System Architecture

## Data Flow Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BINANCE API                                    │
│                         (BTCUSDT klines)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              data_downloader.py                               │
│  - Downloads klines for 11 timeframes                                   │
│  - Saves to CSV: testing_data_{TF}.csv                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CSV FILES (DATA_DIR)                                 │
│  testing_data_3D.csv, testing_data_1D.csv, ..., testing_data_5M.csv   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  TimeframeProcessor (processor.py)                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ load_all_data()                                                  │   │
│  │  - Reads all 11 CSVs                                            │   │
│  │  - Returns Dict[timeframe, DataFrame]                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ process_all()                                                    │   │
│  │  - For each TF × window_size:                                   │   │
│  │    - Run iterative_regression()                                 │   │
│  │    - Calculate acceleration                                     │   │
│  │  - Returns results[tf][ws_label] = DataFrame                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  iterative_regression (analysis.py)                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Input: DataFrame with 'close' prices, window_size               │   │
│  │                                                                  │   │
│  │ For each row i:                                                 │   │
│  │   backward_window = prices[i-N : i]                             │   │
│  │   forward_window = prices[i : i+N]                              │   │
│  │   sqrt_transform()                                              │   │
│  │   linregress(backward) → slope_b, intercept_b                   │   │
│  │   linregress(forward) → slope_f, intercept_f                    │   │
│  │   angle = atan(slope_f - slope_b)                               │   │
│  │                                                                  │   │
│  │ Output columns:                                                 │   │
│  │   slope_b, slope_f, intercept_b, intercept_f,                   │   │
│  │   p_value_b, p_value_f, corr, spearman, angle, actual, time    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SignalLogic (signal_logic.py)                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ run_analysis()                                                   │   │
│  │  - Uses TimeframeProcessor.results                              │   │
│  │  - For each TF × window:                                        │   │
│  │    - detect_cycle_events() → reversal detection                 │   │
│  │    - segment_acceleration() → GMM clustering                    │   │
│  │    - get_direction_from_slope() → LONG/SHORT                    │   │
│  │  - Returns AggregatedResult(signals=[...])                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ detect_cycle_events()                                           │   │
│  │  - 5-point pattern detection                                    │   │
│  │  - BOTTOM: [a>b>c>d<e] → 1                                      │   │
│  │  - PEAK: [a<b<c<d>e] → 1                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ segment_acceleration()                                          │   │
│  │  - GMM(n_components=5) on acceleration values                   │   │
│  │  - Assign zone: VERY_DISTANT, DISTANT, BASELINE, CLOSE, VERY_CLOSE │
│  │  - distant/very_distant = quality signals                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               CalendarDataBuilder (signal_logic.py)                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ build_calendar_df()                                             │   │
│  │  - Creates 55 rows (11 TF × 5 windows)                          │   │
│  │  - Columns: 3h history + 4h forward (15min intervals)           │   │
│  │                                                                  │   │
│  │  For each cell:                                                 │   │
│  │    if historical:                                               │   │
│  │      - Look up actual data from processor_results               │   │
│  │      - Check reversal, crossing, accel_quality                  │   │
│  │      - _compute_cell_score() → "L:2:RC"                         │   │
│  │    if future:                                                   │   │
│  │      - Use child() formula for predictions                      │   │
│  │      - _compute_cell_score() + "~" → "L:2:RC~"                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ _compute_cell_score(has_reversal, has_crossing, accel_quality)  │   │
│  │  - score = R(+1) + C(+1) + A(+1)                                │   │
│  │  - flags = "R" if reversal + "C" if crossing + "A" if accel    │   │
│  │  - Returns: "Direction:Score:Flags" e.g., "L:2:RC"             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ get_aggregated_score_table()                                    │   │
│  │  - Per time column: count R, C, A flags                        │   │
│  │  - Sum total scores, count L vs S                              │   │
│  │  - Calculate dominant direction and strength %                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               PricePredictor (signal_logic.py)                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ child(a, b, c) → predicted_value                                │   │
│  │                                                                  │   │
│  │   angle = c - b                    # momentum                   │   │
│  │   curve = abs(a - 2*b + c)         # curvature                  │   │
│  │   ratio = min(|c|/(|b|+1), 1.5)    # relative strength         │   │
│  │   return c + angle * ratio / (1 + curve/2)                      │   │
│  │                                                                  │   │
│  │ Two-step prediction:                                            │   │
│  │   cc = child(a, b, c)              # step 1                     │   │
│  │   ccc = child(b, c, cc)            # step 2                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dashboard (dashboard.py)                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Caching Layer (TTL=300s / 5 minutes)                            │   │
│  │  - @st.cache_resource: TimeframeProcessor, SignalLogic          │   │
│  │  - @st.cache_data: Heatmap matrices, convergence data           │   │
│  │  - Refresh Data button: clears all caches                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ render_calendar_table()                                         │   │
│  │  - Convergence bar chart (Plotly)                               │   │
│  │  - Aggregated score table (styled DataFrame)                    │   │
│  │  - Full calendar with row/column totals:                        │   │
│  │    • Row totals: ΣR, ΣC, ΣA columns (right side)               │   │
│  │    • Footer rows: TOTAL R, TOTAL C, TOTAL A per time column    │   │
│  │    • Grand totals: bottom-right intersection                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ style_cell_with_footer(val)                                     │   │
│  │  - Parse "L:2:RC" → direction, score, flags                     │   │
│  │  - Green for LONG, Red for SHORT                                │   │
│  │  - Intensity based on score (0.2 + score*0.25)                  │   │
│  │  - Dashed border for predictions (~)                            │   │
│  │  - Yellow intensity for totals (count-based)                    │   │
│  │  - Dark background for TOTAL labels                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Other components:                                               │   │
│  │  - render_analysis_summary() → metrics cards                    │   │
│  │  - render_reversal_signals() → PEAK/BOTTOM panels              │   │
│  │  - render_heatmaps() → reversal/slope/accel matrices           │   │
│  │  - render_decision_summary() → final recommendation            │   │
│  │  - Sidebar: Refresh button, theme toggle, view options         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Browser View   │
                           │  localhost:8501 │
                           └─────────────────┘
```

## Class Reference

### SignalLogic
```python
class SignalLogic:
    processor: TimeframeProcessor
    results: Dict[str, Dict[str, DataFrame]]

    def run_analysis() -> AggregatedResult
    def detect_cycle_events(angles) -> int
    def segment_acceleration(values) -> Dict[int, AccelerationZone]
    def get_report(aggregated) -> str
```

### CalendarDataBuilder
```python
class CalendarDataBuilder:
    history_hours: int = 3
    forward_hours: int = 4
    interval_minutes: int = 15

    def build_calendar_df(signals, processor_results) -> DataFrame
    def get_convergence_summary(calendar_df) -> DataFrame
    def get_aggregated_score_table(calendar_df) -> DataFrame
    def count_convergence(calendar_df, time_col) -> Dict
    def _compute_cell_score(has_reversal, has_crossing, accel_quality, direction) -> str
    def _check_acceleration_quality(accel_value, all_accels) -> bool
    def _detect_historical_crossing(processor_results, tf, ws, idx) -> bool
    def _predict_angle_crossings(processor_results, tf) -> Dict
```

### Dashboard Row/Column Totals (in render_calendar_table)
```python
# Row totals calculation (per TF-window row):
for row in calendar_df.iterrows():
    r_count, c_count, a_count = 0, 0, 0
    for col in time_cols:
        val = row[col]
        if ':' in val:
            flags = val.replace('~', '').split(':')[2]
            if 'R' in flags: r_count += 1
            if 'C' in flags: c_count += 1
            if 'A' in flags: a_count += 1
    # Append: ΣR, ΣC, ΣA columns

# Column totals calculation (footer rows):
for col in time_cols:
    r_count, c_count, a_count = 0, 0, 0
    for val in calendar_df[col]:
        if ':' in val:
            flags = val.replace('~', '').split(':')[2]
            if 'R' in flags: r_count += 1
            if 'C' in flags: c_count += 1
            if 'A' in flags: a_count += 1
    # Create: TOTAL R, TOTAL C, TOTAL A rows

# Grand totals (bottom-right intersection):
grand_r = sum(row_r_totals)
grand_c = sum(row_c_totals)
grand_a = sum(row_a_totals)
```

### PricePredictor
```python
class PricePredictor:
    @staticmethod
    def child(a, b, c) -> float

    @classmethod
    def predict_forward(series, steps=2) -> List[float]

    @classmethod
    def predict_angle(df, steps=2) -> Tuple[float, float]

    @classmethod
    def predict_slope(df, steps=2) -> Tuple[float, float]

    @classmethod
    def predict_acceleration(df, steps=2) -> Tuple[float, float]
```

### Enums
```python
class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class AccelerationZone(Enum):
    VERY_DISTANT = "VERY_DISTANT"  # Quality
    DISTANT = "DISTANT"            # Quality
    BASELINE = "BASELINE"
    CLOSE = "CLOSE"
    VERY_CLOSE = "VERY_CLOSE"

class ReversalType(Enum):
    PEAK = "PEAK"      # [a<b<c<d>e] → SHORT
    BOTTOM = "BOTTOM"  # [a>b>c>d<e] → LONG
    NONE = "NONE"
```

### Data Classes
```python
@dataclass
class SignalResult:
    timeframe: str
    window_size: int
    window_label: str
    direction: SignalDirection
    reversal_type: ReversalType
    accel_zone: AccelerationZone
    score: int
    timestamp: datetime
    price: float
    slope_f: float
    angle: float
    acceleration: float

@dataclass
class AggregatedResult:
    signals: List[SignalResult]
    long_count: int
    short_count: int
    neutral_count: int
    dominant_direction: SignalDirection
    confidence: float
    timestamp: datetime
```

### Styling Functions (dashboard.py)
```python
def style_cell_with_footer(val):
    """
    Style cells in calendar table including totals.

    Returns CSS style string for:
    - Signal cells: Green/Red based on direction, intensity by score
    - TOTAL labels: Dark background (rgba(52,73,94,0.9)), yellow text (#f1c40f)
    - Total count cells: Yellow intensity based on count (0-0.8)
    - Predictions: 2px dashed white border
    """
    if val.startswith("TOTAL"):
        return 'background-color: rgba(52, 73, 94, 0.9); color: #f1c40f; font-weight: bold;'

    if val.isdigit():
        count = int(val)
        intensity = min(count / 20, 1.0) * 0.8
        return f'background-color: rgba(241, 196, 15, {intensity}); color: black; font-weight: bold;'

    # Signal cell styling...
    direction = parts[0]  # L, S, N
    score = int(parts[1])
    intensity = 0.2 + (score * 0.25)  # 0.2, 0.45, 0.7, 0.95

    if direction == 'L':
        return f'background-color: rgba(46, 204, 113, {intensity}); color: white;'
    elif direction == 'S':
        return f'background-color: rgba(231, 76, 60, {intensity}); color: white;'
```
