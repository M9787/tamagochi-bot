# Data Exploration Findings — Complete Feature Discovery

> Manual analysis of raw decomposed CSVs across 8 market regimes.
> Purpose: persist signal knowledge for future Claude sessions & guide encode.py improvements.
> Date: 2026-02-19

## 8 Market Regimes Analyzed
1. **Feb 8, 2021** — CHOP: BTC ~38.8K, drops 38K → recovers (no direction)
2. **May 19, 2021** — CRASH: BTC 43K → 38.6K in hours (-10%)
3. **Nov 8-9, 2022** — FTX crash: BTC 18.5K grinding down
4. **Oct 23, 2023** — PUMP START: BTC 30K → 30.9K → 35K+
5. **Mar 11, 2024** — ATH PUMP: BTC 68.5K → 71.3K (+4%)
6. **Mar 12-13, 2020** — COVID V-REVERSAL: BTC 6K → 4.8K → bounced to 5.5K
7. **Jun 15, 2023** — RANGE: BTC stuck at 24.95K (50-cent moves in an hour)
8. **Jun 7, 2024** — ATH INDECISION: BTC 70.8K hovering near ATH, no breakout

---

## DISCOVERED PATTERNS (13 total)

### Pattern 1: slope_b × slope_f Sign = Regime Type
| slope_b | slope_f | = | Example |
|---------|---------|---|---------|
| + | + (larger) | Trend ACCELERATING | Oct 23 pump |
| + | - | REGIME CHANGE up→down | May 19 crash |
| - | + | REGIME CHANGE down→up | COVID V-reversal bottom |
| ~0 | ~0 | CHOP / NO TRADE | Jun 15 range |

### Pattern 2: corr = Convergence Hypothesis as Float
| corr | = | Example |
|------|---|---------|
| > +0.7 | CONFIRMED TREND (both halves agree) | Oct 23 pump: +0.88 |
| +0.3 to +0.7 | Mild trend / chop | Feb 8 chop: +0.54 |
| < -0.5 | REGIME CHANGE (halves fight) | May 19 crash: -0.76 |

**THE key separator**: Real moves have |corr| > 0.6. Chop has corr in +0.3 to +0.6.

### Pattern 3: angle = Divergence Magnitude
- Range: angle < 1.0 (slopes parallel)
- Chop: angle 0.5-2.5 (mild divergence)
- Pump start: angle 2-8 (growing divergence)
- Crash: angle 10-30+ (extreme divergence)
- V-reversal: angle peaks at 29 then DECLINES (peak = maximum stress)

### Pattern 4: acceleration = Impulse Strength
- Range: |accel| < 0.3
- Normal trend: |accel| = 0.3-1.0
- Strong impulse: |accel| = 1.0-2.0
- Extreme impulse: |accel| > 3.0 (May 19 crash: 4.78, COVID bottom: 1.72)

### Pattern 5: Cross-Window Lead/Lag (SAME TF)
At Oct 23 00:00 before pump:
| Window | slope_f | angle |
|--------|---------|-------|
| w30 | +0.025 | 1.93 |
| w60 | +0.005 | 0.16 |
| w100 | +0.001 | 0.05 |
| w160 | -0.0003 | 0.03 |

w30 detects 12-60x earlier. The GRADIENT across windows = signal freshness.
**When w30 has OPPOSITE SIGN from w100** = regime transition moment.

### Pattern 6: Cross-TF Support/Conflict
**Strong (ALL TFs agree)** = Oct 23: 5M=+0.175, 1H=+0.195, 4H=+0.210 → pump continues
**Weak (TFs CONFLICT)** = Feb 8: 5M=+0.052, 1H=-0.145 → chop, no follow-through
**Crash confirmed** = May 19: 5M=-0.180, 1H=-0.548, 4H=-0.794 → all negative

### Pattern 7: p_value_f = Trend Reliability
- Strong trend: p_value_f = 1e-13 (extremely reliable)
- Fading: p_value_f = 0.30 (not significant)
- Transition: p_value_f drops from 0.3 → 1e-08 as new trend establishes

### Pattern 8: V-Reversal Bottom Signature
COVID bottom (Mar 13, 03:20 5M w30):
- slope_b = -0.347 (prior crash), slope_f = +0.091 (RECOVERY!)
- angle = 24.4 (extreme divergence = maximum stress)
- corr flipping from + to - (old regime breaking)
- w30 slope_f POSITIVE while w100 slope_f still NEGATIVE (-0.131)

### Pattern 9: corr Zero-Crossing = Regime Completion
FTX crash Nov 8 (5M w30):
- 20:00: corr = -0.704 (backward UP, forward DOWN = divergence)
- 20:40: corr = -0.004 (CROSSING ZERO!)
- 21:00: corr = +0.269 (both halves now DOWN = new regime confirmed)
The speed of this crossing tells how violent the regime change is.

### Pattern 10: Ranging Market Fingerprint
Jun 15 2023 (5M w30):
- |slope_f| = 0.006-0.012, |slope_b| = 0.003-0.011
- angle = 0.02-0.52
- corr = +0.33 to +0.66
ALL values SMALL and STABLE. No impulse, no divergence.

### Pattern 11: ATH Indecision Pattern
Jun 7 2024 (5M w30):
- slope_f: +0.025-0.042 (small positive)
- corr: OSCILLATING -0.30 → +0.69 → +0.41 (unstable!)
- p_value_f: swinging 0.02 ↔ 5e-10
Different from chop (corr stable) and real moves (corr extreme). Corr VOLATILITY = indecision.

### Pattern 12: Daily Macro Context
Jul 2023 (1D w30):
- slope_f = +0.608 (strong daily uptrend)
- slope_b = -0.023 (prior month was down)
- angle = 32.6 (massive macro divergence = regime change)
Higher TF provides direction context for lower TF trades.

### Pattern 13: Volume Dimension (from kline data)
- Crashes: 2000-4000 BTC/5min
- Ranges: 30-140 BTC/5min
- Volume is 20-50x higher during real moves (independent conviction signal)
Currently NOT in decomposed data — would need separate encoding from kline CSVs.

---

## WHAT SEPARATES CHOP vs REAL MOVES (The Key Table)

| Feature | RANGE (skip) | CHOP (noise) | PUMP (long) | CRASH (short) | V-REVERSAL |
|---------|-------------|-------------|-------------|---------------|------------|
| slope_f mag | < 0.015 | 0.01-0.05 | > 0.05 | > 0.10 (neg) | flips sign |
| angle | < 1.0 | 0.5-2.5 | 2-8+ | 10-30+ | peaks then drops |
| accel | < 0.3 | 0.2-0.5 | 0.7-2.0 | 1.0-5.0 | large then reverses |
| corr | +0.3 to +0.7 | +0.3 to +0.7 | > +0.7 or < -0.5 | < -0.6 | flips sign |
| p_value_f | varies | 0.001-0.02 | < 1e-06 | < 1e-10 | drops rapidly |
| slope_b×slope_f | same sign, tiny | same sign, small | same sign, growing | OPPOSITE signs | OPPOSITE signs |

---

## WHAT THE CURRENT R/C/A ENCODING LOSES

The current `encode.py` compresses each signal source into a cell string like `"L:3:RCA"` — this **discards**:

1. **corr** (convergence float) — completely absent, yet THE key separator between chop and real moves
2. **slope_f magnitude** — only SIGN is kept (+1/-1), magnitude lost (0.01 vs 0.18 = same encoding)
3. **slope_b** — entirely absent (prior regime indicator)
4. **angle magnitude** — compressed to binary R flag (reversal yes/no), raw range 0-30+ lost
5. **acceleration magnitude** — compressed to binary A flag (GMM zone), raw range 0-5+ lost
6. **p_value_f** — entirely absent (trend statistical reliability)

The R/C/A encoding is good for dashboard visualization but throws away the continuous information that CatBoost needs to distinguish regime types.

---

## REDUNDANT / SKIP COLUMNS

### spearman vs corr: REDUNDANT
Checked across all regimes. They track within 0.1 of each other. Including both adds noise, not signal. **Skip spearman.**

### intercept_b, intercept_f: DERIVED
These are just the Y-intercepts of the regression — fully determined by slopes + positions. No independent information. **Skip.**

### p_value_b: LOW VALUE
Backward trend reliability is less actionable than forward. **Skip for now.**

---

## RAW FEATURE RECOMMENDATION (Tier-based)

### Tier 1 — Must include (unique, strong signal from 8 regimes):
1. **slope_f** — direction + magnitude (55 combos)
2. **slope_b** — prior regime indicator (55 combos)
3. **angle** — divergence magnitude (55 combos)
4. **acceleration** — impulse strength (55 combos)
5. **corr** — convergence measure (55 combos)

### Tier 2 — Should include (meaningful additional signal):
6. **log10(p_value_f)** — trend reliability (55 combos)

### Tier 3 — Consider later:
7. **Volume** — conviction measure (needs kline data, 11 TFs)
8. **corr stability** — rolling std of corr (needs computation)

### Feature counts:
- Tier 1 only: 55 × 5 = 275 features
- Tier 1 + 2: 55 × 6 = 330 features
- With lag(1..5): 330 × 6 = 1,980 features (still 1:318 ratio with 630K rows)
- Without lags: 330 features (1:1909 ratio — very healthy)

### Recommendation for encode.py:
Replace or augment the 330 categorical cell features with 330 raw float features (6 floats × 55 combos). Can keep the 24 numeric summaries. Total with lags: ~1,980 + 24 = ~2,004 features. Feature-to-sample ratio remains healthy at 1:314.
