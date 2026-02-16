# Quick Reference Card

## Signal Formula
```
signal = reversal_event(+1) + crossing_event(+1) + accel_quality(+1)
```

## Cell Format
```
Direction:Score:Flags[~]

L:3:RCA  = LONG,  3 pts, all components (Reversal+Crossing+Accel)
S:2:RC   = SHORT, 2 pts, Reversal+Crossing
L:1:A    = LONG,  1 pt,  Accel Quality only
L:0:     = LONG,  0 pts, no components (just direction)
N:0:     = NEUTRAL, 0 pts, no clear trend
S:1:R~   = SHORT, 1 pt, Reversal, PREDICTED (~ = future)
L:2:CA~  = LONG,  2 pts, Crossing+Accel, PREDICTED
```

## Flags
| Flag | Meaning | Detection |
|------|---------|-----------|
| R | Reversal | 5-point pattern |
| C | Crossing | Angle crossover between windows |
| A | Accel Quality | Top 40% acceleration |

## 5-Point Reversal Patterns
```
BOTTOM → LONG potential
angles: [50, 40, 30, 20, 35]
         ↓   ↓   ↓   ↓   ↑
         decreasing → reversal UP

PEAK → SHORT potential
angles: [20, 30, 40, 50, 35]
         ↑   ↑   ↑   ↑   ↓
         increasing → reversal DOWN
```

## child() Prediction Formula
```python
def child(a, b, c):
    angle = c - b
    curve = abs(a - 2*b + c)
    ratio = min(abs(c) / (abs(b) + 1), 1.5)
    return c + angle * ratio / (1 + curve/2)

# Usage
pred_1 = child(a, b, c)      # Next value
pred_2 = child(b, c, pred_1) # Value after next
```

## Grid Size
```
11 Timeframes: 3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M
× 5 Windows:   df(30), df1(60), df2(100), df3(120), df4(160)
= 55 signal sources
```

## Calendar Structure
```
         |←── 3h historical ──→|←── 4h predicted ──→| Row Totals
TF-Window| 00:00 | 00:15 | NOW | 03:15~ | 03:30~ |  ΣR | ΣC | ΣA |
─────────┼───────┼───────┼─────┼────────┼────────┼─────┼────┼────┼
3D-df    | L:2:RC| L:1:R |  ▓  | S:1:C~ | S:2:RC~|   5 |  3 |  2 |
3D-df1   | L:1:A | S:0:  |  ▓  | L:1:R~ | L:1:A~ |   2 |  1 |  4 |
...      | ...   | ...   |  ▓  | ...    | ...    | ... | ...|... |
5M-df4   | S:1:C | L:2:RA|  ▓  | L:0:~  | L:1:R~ |   1 |  2 |  3 |
─────────┼───────┼───────┼─────┼────────┼────────┼─────┼────┼────┼
TOTAL R  |   8   |   5   |     |   4    |   6    | 245 |    |    |
TOTAL C  |   3   |   4   |     |   2    |   3    |     | 156|    |
TOTAL A  |   6   |   3   |     |   5    |   4    |     |    | 198|
```

**Row Totals:** ΣR, ΣC, ΣA = sum of flags across all time columns for each row
**Footer Rows:** TOTAL R/C/A = sum of flags across all 55 rows per time column
**Grand Totals:** Bottom-right corner = total count across entire calendar

## Color Coding
```
GREEN intensity  = LONG + score (0.2, 0.45, 0.7, 0.95)
RED intensity    = SHORT + score (0.2, 0.45, 0.7, 0.95)
GRAY             = NEUTRAL (half intensity)
DASHED border    = Prediction (future cells)
YELLOW intensity = Totals (footer rows, row totals)
DARK background  = TOTAL labels (rgba(52,73,94,0.9))
```

## Sidebar Controls
```
[Refresh Data] → Clears cache (TTL=300s) and reloads
Theme Toggle   → Dark/Light mode
View Options   → Show/hide dashboard components
```

## Aggregation Summary
```
Per time slot (across 55 rows):
- Reversals:    count of R flags
- Crossings:    count of C flags
- Accel Quality: count of A flags
- Total Score:  sum of all scores
- Direction:    LONG if L > S, else SHORT
- Strength %:   max(L,S) / (L+S) × 100
```

## Commands
```bash
# Download data
python data_downloader.py

# Run dashboard
python -m streamlit run dashboard.py --server.port 8501
```

## Key Files
```
signal_logic.py  → SignalLogic, PricePredictor, CalendarDataBuilder
analysis.py      → iterative_regression, calculate_acceleration
dashboard.py     → Streamlit UI
config.py        → TIMEFRAME_ORDER, WINDOW_SIZES, CALENDAR_*
```
