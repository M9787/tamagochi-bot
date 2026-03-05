# V1→V4 Full Experiment Retrospective — Quant Analysis

**Date**: 2026-02-20
**Dataset**: 630,266 5M candles (2020-01-01 to 2026-02-15)
**Label Distribution**: NO_TRADE=72.7% (458,081), SHORT=13.7% (86,413), LONG=13.6% (85,772)
**Trade Parameters**: SL=2%, TP=4%, break-even precision=33.3%

---

## Experiment Inventory: 11 Experiments, ~130 Sub-Runs

| # | Experiment | Script | Features | Architecture | Status |
|---|-----------|--------|----------|-------------|--------|
| 1 | V1 Binary | train.py | 354 (330 cat + 24 num) | Binary Logloss, drops NO_TRADE | FAILED |
| 2 | V1 Random Search | random_search.py | 47-200 selected | 3-class, random weights/features | FAILED |
| 3 | V1 Raw Decomposed | raw_decomposed_search.py | 269-377 of 1122 | 3-class, raw regression outputs | FAILED |
| 4 | V2.1 Main | train_v2.py | 135 numeric | 3-class, single split + permutation | PARTIAL |
| 5 | V2.1 Validation | train_v2.py (seeds+rolling) | 135 | 5 seeds + 4 rolling windows | PARTIAL |
| 6 | **V3 Ensemble** | train_v3.py | 280 (levels + deltas) | 3-class, 10-seed ensemble, 90d test | **BEST** |
| 7 | V4 Baseline | train_v4.py | 174 | 2-stage gate+direction | FAILED |
| 8 | V4 Hyperparam | tune_gate/dir_v4.py | 174 | 40+40 config grid search | INSIGHTS |
| 9 | V4 Option A | train_v4_walkforward.py | 201 (174+27 dir) | 2-stage, 5-fold walk-forward | FAILED |
| 10 | V4 Option B | train_v4_3class.py | 174 | 3-class, 5-fold walk-forward | FAILED |
| 11 | V4 Option C | train_v4_asymmetric.py | 174 | Dual LONG+SHORT gate, 5-fold WF | FAILED |

---

## Evolution Results — Key Metrics Per Phase

### Phase 1: V1 — Wrong Problem Formulation

| Metric | V1 Binary | V1 Random Search (best of 100) | V1 Raw Decomposed (best of 10) |
|--------|-----------|-------------------------------|-------------------------------|
| **Precision** | 42.5% (biased) | 50.0% | 28.0% |
| **Win Rate** | 42.5% | 50.0% | 28.0% |
| **Profit** | +1560% (fake) | +$2.00 | -$11.20 |
| **Max DD** | -650% | -$24 | N/A |
| **AUC** | **0.423** (anti-predictive) | 0.552-0.706 | 0.739-0.810 |
| **Trades** | 2,826 (all test rows) | 19-20 | 350 |

**Root causes**:
- **C1**: Dropped 72.7% NO_TRADE rows (458k of 630k) — model never learns WHEN to trade
- **C2**: Within-TF convergence = noise (50.1% accuracy on 11,719 events)
- **C3**: Cross-TF cascades = no directional edge (50-51.5% WR, p>0.33)
- **C4**: AUC=0.423 means flipping predictions improves results
- PCA on categoricals = anti-pattern (compression destroys CTR encoding)

**V1 Random Search detail** (100 Latin Hypercube weight configs):
- Only 4/100 runs profitable, barely (+$0.20-$2.00)
- Best: Run #13, 50% WR on 20 trades, AUC=0.552
- Mean PnL: -$418, mean WR: 8.59%

**V1 Raw Decomposed detail** (10 weight configs, raw regression features):
- 0/10 runs profitable (100% loss rate)
- Best: Run #7, 28% WR on 350 trades, AUC=0.801, still lost $11.20
- Critical insight: 92% accuracy + 80% AUC + 28% WR = still unprofitable → accuracy/AUC ≠ directional predictability

---

### Phase 2: V2.1 — First Real Signal (Fragile)

**Architecture**: 135 numeric features (11 TFs × 12 per TF + 3 summaries), 3-class CatBoost, single 30d test split

| Metric | V2.1 Single Split @0.55 | V2.1 Single @0.60 | V2.1 Seed-Avg @0.60 | V2.1 Rolling Win-B @0.62 |
|--------|------------------------|-------------------|---------------------|-------------------------|
| **Precision** | 8.1% | 6.2% | 41.4% ± 17.2% | 50.2% |
| **Win Rate** | 8.1% | 6.2% | 41.4% | 50.2% |
| **Profit** | -1672% | -1060% | +111.6% | +230% |
| **Trades** | 1,103 | 650 | 264.6 avg | 227 |
| **SHORT Preds** | 0 | 0 | N/A | N/A |

**Permutation test**: Shuffled model at threshold ≥0.46 makes 0 trades (real model makes 2,237). Real signal confirmed.

**Seed study** (5 seeds × 8 thresholds):

| Threshold | Mean Precision | Std | Min | Max | Mean Trades | Mean Profit |
|-----------|---------------|-----|-----|-----|-------------|-------------|
| 0.55 | 36.1% | ±15.1% | 22.9% | 59.9% | 658.8 | -86.4% |
| 0.57 | 40.5% | ±17.1% | 24.5% | 68.6% | 501.4 | +66.4% |
| 0.60 | 41.4% | ±17.2% | 24.7% | 70.1% | 264.6 | +111.6% |
| 0.62 | 26.1% | ±27.6% | 2.6% | 61.5% | 140.6 | -11.2% |

**Rolling window** (4 × 30d periods):

| Window | @0.55 Trades | @0.55 Precision | @0.60 Trades | @0.60 Profit |
|--------|-------------|-----------------|-------------|-------------|
| Oct 16 - Nov 15 | 0 | 0% | 0 | $0 |
| Nov 16 - Dec 15 | 877 | 16.3% | 342 | -318% |
| Dec 16 - Jan 15 | 0 | 0% | 0 | $0 |
| **Jan 16 - Feb 15** | **945** | **22.9%** | **453** | **+228%** |

**Key finding**: Results are **seed-dependent** (±17% std) and **time-period dependent** (2 of 4 rolling windows produce zero trades). Only Jan 2026 window profitable.

---

### Phase 3: V3 — Best Results (LONG-Only)

**Architecture**: 280 features (V2.1 levels + first-differenced deltas + cumsum directional anchors), 10-seed ensemble, 90d test (Nov 17, 2025 - Feb 15, 2026)

| Metric | V3 @0.38 | V3 @0.40 | **V3 @0.42 (BEST)** | V3 @0.44+ |
|--------|----------|----------|---------------------|-----------|
| **Precision** | 15.7% | 27.8% | **63.1%** | 0% (0 trades) |
| **Win Rate** | 15.7% | 27.8% | **63.1%** | 0% |
| **Profit** | -2390% | -314% | **+252%** | $0 |
| **Max DD** | -3604% | -2390% | **-102%** | $0 |
| **AUC (macro)** | 0.688 | 0.688 | **0.688** | 0.688 |
| **Trades** | 2,263 | 946 | **141** (ALL LONG) | 0 |
| **Profit Factor** | 0.613 | 0.373 | **3.423** | 0 |
| **Sharpe** | -0.233 | -0.483 | **0.617** | 0 |
| **LONG Preds** | N/A | 856 | **141** | 0 |
| **SHORT Preds** | N/A | 90 | **0** | 0 |

**Per-class AUC**: NO_TRADE=0.720, LONG=0.608, SHORT=0.737

**V3 @0.42 Confusion Matrix** (25,920 test rows):
```
              Predicted
Actual    NO_TRADE  LONG  SHORT
NO_TRADE   20,213    45     0
LONG        1,938    89     0
SHORT       3,628     7     0
```

**V3 @0.42 equity curve**: Monotonically rising from $0 to $252 with only one -$2 dip (at trade ~28). Max drawdown just -$102 (-1.4 trades worth).

**V3 is the champion.** But critical caveats:
- ALL 141 trades are LONG — zero SHORT predictions at profitable threshold
- Threshold window 0.40→0.44 is razor-thin (0.42 is the only profitable tick)
- 90-day test is more credible than V2.1's 30-day
- Ensemble shows binary consensus: all 10 seeds agree at 0.0 (std=1.2%), all go silent at 0.42+ (std=0%)

**Top 10 V3 Features**:

| Rank | Feature | Importance | Type |
|------|---------|-----------|------|
| 1 | slope_f_mag_3D | 7.59 | Level (KING) |
| 2 | angle_slow_3D | 6.50 | Level |
| 3 | cs_dsf_1D_w23 | 6.39 | Cumsum (NEW in V3) |
| 4 | cs_dsf_3D_w23 | 5.69 | Cumsum (NEW in V3) |
| 5 | angle_slow_6H | 5.50 | Level |
| 6 | accel_mag_1H | 4.72 | Level |
| 7 | corr_slow_3D | 4.12 | Level |
| 8 | d_slope_b_1D | 4.05 | Delta (NEW in V3) |
| 9 | slope_f_mag_1D | 4.02 | Level |
| 10 | slope_b_slow_3D | 3.71 | Level |

**Dead features**: 98 of 280 (35%) — most small-window cumsums and Young timeframe deltas contribute nothing.

---

### Phase 4: V4 — Architecture Experiments (All Failed)

**Base architecture**: 2-stage Gate (TRADE/NO_TRADE binary) + Direction (LONG/SHORT binary on TRADE-only rows)

#### V4 Baseline (single split, 30d test)

| Metric | V4 Baseline @g0.50/d0.50 | V4 @g0.55/d0.65 |
|--------|--------------------------|-----------------|
| **Gate AUC** | 0.710 | 0.710 |
| **Direction AUC** | 0.538 | 0.538 |
| **Combined AUC** | 0.726 macro | 0.726 |
| **Trades** | 3,530 | 114 |
| **Trade Precision** | 25.4% | 28.9% |
| **LONG Precision** | 17.8% | N/A |
| **SHORT Precision** | 60.3% | N/A |
| **Profit** | -1,684% | +86% |
| **Max DD** | -2,454% | N/A |

**Per-class AUC**: NO_TRADE=0.710, LONG=0.870, SHORT=0.598

**Permutation test** (at g0.55/d0.70):
| Model | Trades | Precision | Profit |
|-------|--------|-----------|--------|
| Real | 156 | 23.1% | +86% |
| Shuffled gate + real dir | 89 | 8.0% | -292% |
| Real gate + shuffled dir | 0 | 0% | $0 |
| Both shuffled | 0 | 0% | $0 |

Gate signal is real. Direction signal collapses when gate is shuffled.

#### V4 Hyperparam Tuning (40+40 configs)

**Gate tuning winner**: depth=5, lr=0.05, l2=26.33, leaf=20, sub=0.76 → Test AUC=0.707

**Direction tuning winner**: depth=5, lr=0.05, l2=26.33, leaf=10, sub=0.76, Balanced → Test AUC=0.824

**Direction tuning OUTLIER** (config #3 of 40):
- depth=6, lr=0.08, l2=2.75, leaf=10, sub=0.96, custom 1.5× weights
- Dir AUC: 0.635 (LOWER than winner)
- SHORT Precision @0.55: **96.2%** (only outlier to predict SHORT)
- Combined profit: **+1,476%** on 2,010 trades
- **Single-split only** — walk-forward proves it doesn't generalize

#### V4 Options A/B/C (5-fold walk-forward each)

| Metric | V4-A WalkFwd | V4-B 3Class WF | V4-C Dual-Gate |
|--------|--------------|----------------|----------------|
| **Best Avg Precision** | 3.1% | 18.1% | 2.0% |
| **Best Avg Profit** | -608% | -1,076% | -119% avg |
| **Gate AUC range** | 0.60-0.83 | 0.531 macro | L:0.47-0.78 / S:0.41-0.71 |
| **Dir AUC range** | **0.19-0.96** (!!!) | SHORT=0.403 | N/A |
| **Avg Trades** | 115 | 1,180 | 85 |
| **Permutation** | Pass (gate) | Mixed | **FAIL** (shuffled better 3/5) |

**V4-A Direction Health Report** (per-fold Dir AUC on TRADE-only):

| Fold | Period | Dir AUC | SHORT@0.55 | LONG@0.55 | Verdict |
|------|--------|---------|-----------|----------|---------|
| 0 | Jan-Feb 2026 | **0.187** | 24 | 547 | Anti-predictive |
| 1 | Dec-Jan 2026 | **0.957** | 100 | 109 | Overfitting |
| 2 | Nov-Dec 2025 | 0.614 | 295 | 1,519 | Marginal |
| 3 | Oct-Nov 2025 | 0.564 | 98 | 347 | Marginal |
| 4 | Sep-Oct 2025 | 0.535 | 0 | 0 | Random |

**V4-C Permutation FAILURE detail** (fold-level):

| Fold | Real Precision | Shuffled Precision | Real Profit | Shuffled Profit | Winner |
|------|---------------|-------------------|-------------|-----------------|--------|
| 0 | 7.2% | **19.7%** | -13,536% | -7,044% | **Shuffled** |
| 1 | 5.9% | 1.5% | -13,562% | -16,494% | Real |
| 2 | 10.8% | **11.4%** | -11,496% | -11,376% | **Shuffled** |
| 3 | 6.5% | **9.9%** | -13,890% | -12,144% | **Shuffled** |
| 4 | 5.2% | 6.2% | -14,592% | -14,052% | **Shuffled** |

Shuffled model beats real in 3 of 5 folds. Dual-gate architecture learns no signal.

---

## 5 Global Issues

### 1. Direction Signal Absent from Features

Across ALL 11 experiments: SHORT predictions collapse to zero at useful thresholds.

**Evidence**:
- V3 @0.42: 141 LONG, **0 SHORT** → 63.1% precision
- V2.1 @0.55: 1,103 LONG, **0 SHORT** → 8.1% precision
- V4 Dir best: 25.6% LONG precision, **0% SHORT** @0.55
- V4-B 3class @0.70: 1,180 trades, only 21 SHORT predictions

Features measure magnitude (how much) but not asymmetric direction (which way). The model cannot distinguish LONG from SHORT at operational thresholds.

### 2. Extreme Time-Period Sensitivity

**V2.1 rolling**: 2/4 windows produce zero trades. Jan 2026 = +228% profit, Nov 2025 = -318%.

**V4-A walk-forward**: Dir AUC swings **0.187→0.957** across 30-day folds. The model fits to market regime, not generalizable signal.

**V3**: 90d test credible but only covers Nov 2025 - Feb 2026 (mostly bullish Jan 2026).

### 3. Razor-Thin Profitable Threshold Windows

| Model | Loss Threshold | Profit Threshold | Zero Threshold | Window Width |
|-------|---------------|-----------------|----------------|-------------|
| V3 | 0.40 (-314%) | **0.42 (+252%)** | 0.44 (0 trades) | **0.02** |
| V2.1 | 0.55 (-86%) | 0.57 (+66%) | 0.62 (-11%) | 0.05 |
| V4 | g0.50/d0.50 (-1684%) | g0.55/d0.70 (+86%) | g0.60+ (0) | razor thin |

One threshold tick = catastrophic difference. Not operationally viable without adaptive mechanism.

### 4. Seed Sensitivity

**V2.1 @0.60**: 41.4% ± 17.2% precision across 5 seeds (range: 24.7% - 70.1%)

**V3 @0.0**: 22.6% ± 1.2% precision across 10 seeds (low variance at baseline)

Decision boundary is on a knife edge for V2.1. V3 ensemble reduces this but at cost of binary threshold behavior.

### 5. Permutation Tests Pass But Profit Fails

| Model | Permutation Result | Trading Result |
|-------|-------------------|---------------|
| V2.1 | Shuffled=0 trades at ≥0.46 (PASS) | Only profitable @0.55+ |
| V3 | Shuffled=0 trades at all thresholds (PASS) | Only profitable @0.42 exact |
| V4 Gate | Shuffled gate = -292% vs real +86% (PASS) | Direction still fails |
| V4-C | **Shuffled beats real 3/5 folds (FAIL)** | All loss |

Every real model (except V4-C) beats shuffled — proving gate/structure detection is real. But direction conversion fails. The model knows WHEN something happens but not WHICH WAY.

---

## What to KEEP

| Asset | Why | Evidence | Location |
|-------|-----|---------|----------|
| 3-class formulation | Correct problem framing | All V2+ outperform V1 binary | All V2+ scripts |
| Gate detection capability | AUC 0.60-0.83 consistently | Permutation tests pass | train_v4.py gate model |
| 135 numeric core features | Proven signal foundation | 90/135 used (67%) | encode_v2.py |
| First-differenced deltas | V3's AUC boost to 0.688 | cs_dsf_w23 = top-3 importance | encode_v3.py |
| Walk-forward validation | Exposes overfitting single splits hide | V4-A: 0.19-0.96 AUC swing | V4 options A/B/C |
| Permutation testing | Essential sanity check | Catches V4-C failure | All V2+ scripts |
| SL/TP labeling (2%/4%) | Consistent, validated across all experiments | build_labels.py |
| Top features | slope_f_mag_3D, angle_slow_{6H,12H,4H,3D}, cross_traj_1D, cs_dsf_*_w23 | V2.1+V3 importance |
| 10-seed ensemble | Reduces seed sensitivity | V3 std=1.2% vs V2.1 std=17.2% | train_v3.py |
| 90d test window | More credible than 30d | V3 catches more regimes | train_v3.py |

## What to MODIFY/DROP

| Item | Why | Evidence | Action |
|------|-----|---------|--------|
| Direction prediction | Features are symmetric, can't distinguish L/S | 0 SHORT preds at all profitable thresholds | Need asymmetric features |
| Single temporal split | Hides catastrophic instability | V4-A: Dir AUC 0.19-0.96 | Always walk-forward |
| Binary reversal (0/1) | Dead after 5M alignment (<0.5% density) | 0/11 importance in V2.1+V3 | Drop or make continuous |
| direction_* sign features | Redundant with slope_f_mag | 0/11 importance in V2.1 | Drop |
| Fixed threshold | Too fragile, one tick = disaster | V3: 0.40=-314%, 0.42=+252%, 0.44=$0 | Adaptive threshold or ensemble |
| 30-day test windows | Too short, regime-dependent | V2.1: 2/4 windows zero trades | Use 90d+ like V3 |
| Within-TF convergence | Autocorrelated (50.1% hit rate) | C2 in V1 audit | Drop |
| 2-stage gate+direction | Direction stage adds noise | V4-A/B/C all failed walk-forward | Use 3-class directly |
| Dual-gate architecture | No signal learned (shuffled better) | V4-C permutation FAIL 3/5 | Drop |
| Small cumsum windows | Dead features (w3,w5,w7,w11,w17,w19) | 0 importance in V3 | Keep only w23 |
| Young TF deltas | Dead features after alignment | 0 importance in V3 | Drop |

---

## Feature Evolution Timeline

| Version | Total | Numeric | Categorical | Key Additions | Dead at Test |
|---------|-------|---------|-------------|---------------|-------------|
| V1 | 354 | 24 | 330 | Cell strings + lags | N/A (whole approach failed) |
| V2.1 | 135 | 135 | 0 | cross_div, cross_traj, cross_dir, TF-native lags | 22 (16%) |
| V3 | 280 | 280 | 0 | Δ features, cumsum anchors (w3-w23), corr_slow, slope_b_slow | 98 (35%) |
| V4 | 174 | 174 | 0 | V3 selective subset (only w23 cumsums) | ~40 (23%) |

**Surviving "king" features across all versions**:
1. `slope_f_mag_3D` — 3D trend strength (V2.1: 6.19, V3: 7.59 — consistently #1)
2. `angle_slow_{6H,12H,4H}` — current angle state at Balzak/Adult TFs
3. `cross_traj_1D` — 1D divergence velocity
4. `cs_dsf_{1D,3D}_w23` — cumsum directional anchors (V3 only, top-5)
5. `corr_slow_3D` — 3D correlation (resurfaces in V3)

---

## Scorecard

| Capability | Status | Best Evidence | Best Number |
|-----------|--------|---------------|-------------|
| Detect WHEN to trade | **SOLVED** (AUC 0.6-0.8) | Gate model, V2.1-V4 | Gate AUC=0.710 |
| Predict LONG | **PARTIAL** (63% on 141 trades) | V3 @0.42, narrow window | 63.1% precision |
| Predict SHORT | **UNSOLVED** | 0 SHORT preds at useful thresholds | 0% at all versions |
| Stable across time | **UNSOLVED** | Walk-forward: 0.19-0.96 AUC swing | 5-fold variance |
| Stable across seeds | **PARTIAL** (V3 good, V2.1 bad) | V3: ±1.2% std, V2.1: ±17.2% | 10-seed ensemble |
| Break-even (33.3%) sustained | **PARTIAL** | V3 only, LONG-only, 1 threshold | 63.1% > 33.3% |
| Profitable trading | **PARTIAL** | V3 @0.42 | +252%, PF=3.42 |
| Operational readiness | **NOT READY** | Razor-thin threshold, no SHORT, regime-dependent | N/A |

---

## Appendix A: Full Threshold Tables

### V2.1 Test Set (8,640 rows, 30d)

| Threshold | Trades | Trade Precision | LONG Preds | SHORT Preds | Profit | WR |
|-----------|--------|-----------------|-----------|------------|--------|-----|
| 0.0 | 4,172 | 14.2% | N/A | N/A | -4780% | 14.2% |
| 0.42 | 3,101 | 13.1% | N/A | N/A | -3760% | 13.1% |
| 0.46 | 2,237 | 13.1% | N/A | N/A | -2722% | 13.1% |
| 0.50 | 1,740 | 10.5% | N/A | N/A | -2382% | 10.5% |
| 0.55 | 1,103 | 8.1% | 1,103 | 0 | -1672% | 8.1% |
| 0.60 | 650 | 6.2% | 650 | 0 | -1060% | 6.2% |

### V3 Test Set (25,920 rows, 90d)

| Threshold | Trades | Trade Precision | LONG Preds | SHORT Preds | Profit | Max DD | PF | Sharpe |
|-----------|--------|-----------------|-----------|------------|--------|--------|-----|--------|
| 0.0 | 5,417 | 23.5% | 3,503 | 1,914 | -3208% | -4104% | 0.613 | -0.233 |
| 0.35 | 4,559 | 23.5% | N/A | N/A | -2698% | N/A | N/A | N/A |
| 0.38 | 2,263 | 15.7% | N/A | N/A | -2390% | -3604% | 0.613 | -0.233 |
| 0.40 | 946 | 27.8% | 856 | 90 | -314% | -1040% | 0.770 | -0.123 |
| **0.42** | **141** | **63.1%** | **141** | **0** | **+252%** | **-102%** | **3.423** | **0.617** |
| 0.44 | 0 | 0% | 0 | 0 | $0 | $0 | 0 | 0 |

### V4 Baseline (8,640 rows, 30d) — Selected Pairs

| Gate | Dir | Trades | LONG | SHORT | Precision | Profit |
|------|-----|--------|------|-------|-----------|--------|
| 0.25 | 0.50 | 8,640 | 6,499 | 2,141 | 10.9% | -11,604% |
| 0.50 | 0.50 | 3,530 | 2,898 | 632 | 25.4% | -1,684% |
| 0.50 | 0.65 | 159 | 159 | 0 | 26.4% | +162% |
| 0.55 | 0.65 | 114 | 114 | 0 | 28.9% | +86% |
| 0.55 | 0.70 | 156 | N/A | N/A | 23.1% | +86% |

### V4-B 3-Class Walk-Forward (43,200 rows, 5×30d folds aggregated)

| Threshold | Trades | Precision | Profit | Max DD | PF | Sharpe |
|-----------|--------|-----------|--------|--------|-----|--------|
| 0.35 | 9,052 | 8.7% | -13,406% | -13,406% | 0.189 | -0.878 |
| 0.50 | 3,028 | 10.1% | -4,226% | -4,226% | 0.224 | -0.773 |
| 0.60 | 1,817 | 14.0% | -2,110% | -2,110% | N/A | N/A |
| 0.70 | 1,180 | 18.1% | -1,076% | -1,306% | N/A | N/A |

---

## Appendix B: Architecture Comparison

### V1: Binary (DEPRECATED)
```
5M candles → encode.py (330 cat + 24 num = 354 features)
  → DROP NO_TRADE (72.7%)
  → CatBoost Logloss (LONG=1 / SHORT=0)
  → Single temporal split
```

### V2.1: 3-Class Single Split
```
5M candles → encode_v2.py (135 numeric features)
  → KEEP ALL ROWS (3-class: NO_TRADE/LONG/SHORT)
  → CatBoost MultiClass + Balanced weights
  → 3-way split (train/val/test) + permutation test
```

### V3: 3-Class 10-Seed Ensemble
```
5M candles → encode_v3.py (280 numeric: levels + deltas + cumsums)
  → KEEP ALL ROWS (3-class)
  → 10× CatBoost MultiClass + Balanced weights
  → Average probabilities across seeds
  → 3-way split (5.5yr train / 3mo val / 3mo test) + permutation
```

### V4: Two-Stage Gate + Direction
```
5M candles → encode_v4.py (174 numeric features)
  → Stage 1: Gate (TRADE/NO_TRADE, all rows)
  → Stage 2: Direction (LONG/SHORT, TRADE-only rows)
  → 2D threshold grid (gate × dir)
  → Various validation: single split, walk-forward, dual-gate
```

---

## Appendix C: Hypothesis Testing Summary

| Hypothesis | Tested In | Result | p-value | Verdict |
|-----------|-----------|--------|---------|---------|
| Within-TF convergence predicts direction | V1 (analysis_convergence) | 50.1% accuracy | N/A | **REJECTED** |
| Cross-TF cascades predict direction | V1 (analysis_cascade) | 50-51.5% WR | >0.33 | **REJECTED** |
| Dropping NO_TRADE improves LONG/SHORT | V1 Binary | AUC=0.42 | N/A | **REJECTED** |
| 3-class with all rows works | V2.1+ | AUC=0.688 | N/A | **CONFIRMED** |
| Gate model detects trade timing | V2.1-V4 | AUC=0.60-0.83 | Permutation pass | **CONFIRMED** |
| Features contain directional signal | V4-A walk-forward | AUC 0.19-0.96 | High variance | **INCONCLUSIVE** |
| 2-stage beats 3-class | V4 baseline vs V4-B | 25.4% vs 18.1% | N/A | **CONFIRMED** (single split) |
| 2-stage generalizes across time | V4-A walk-forward | All folds loss | N/A | **REJECTED** |
| Dual independent gates work | V4-C | Shuffled > real 3/5 folds | N/A | **REJECTED** |
| Ensemble reduces seed variance | V3 | std=1.2% vs V2.1 std=17.2% | N/A | **CONFIRMED** |
| Longer test window more credible | V3 (90d) vs V2.1 (30d) | V3 more stable | N/A | **CONFIRMED** |
| Custom class weights help SHORT | V4 dir tuning outlier | 96.2% SHORT @0.55 | Single split only | **INCONCLUSIVE** |

---

*Generated 2026-02-20 from verified result JSON files in model_training/results_v2/, results_v3/, results_v4/*
