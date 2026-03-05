# V10 Cross-Scale Convergence — Experiment Results (2026-02-24)

## Overview

V10 = 508 features (395 V6 base + 113 new Phase F cross-scale convergence features).
Goal: Encode the expert's primary signal (cross-window angle crossings) that died in V2-V9.

### New Phase F Features (113 total)

| Phase | Features | Count | Purpose |
|-------|----------|-------|---------|
| F1 | `xw_crosses_active_{tf}`, `xw_crosses_long_{tf}`, etc. | 77 (7 x 11 TFs) | Per-TF cross-window crossing states (counts, not continuous) |
| F2 | `xtf_total_crosses`, `xtf_cascade_score`, etc. | 15 | Cross-TF composites (sum/weight across TFs) |
| F3 | `corr_velocity_{tf}`, `xtf_corr_agreement` | 12 | Correlation rate of change + cross-TF agreement |
| F4 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_ny_session` | 5 | Temporal cyclical features |
| F5 | `convergence_volume`, `crossing_atr`, `cascade_volume`, `reversal_conviction` | 4 | Volume x convergence interactions |

### Key Design: Discrete Counts Survive Staircase

Prior versions (V2-V9) used continuous angle values that became staircased on merge_asof to 5M.
V10 uses INTEGER COUNTS of crossings/reversals. Counts are STATE information — the staircase IS correct.
CatBoost splits on "crosses_active >= 2" which is meaningful.

### Hyperparameters (V7 d8 params)

```python
iterations=5000, depth=8, learning_rate=0.02, l2_leaf_reg=15,
min_data_in_leaf=500, loss_function='MultiClass',
eval_metric='TotalF1:average=Macro;use_weights=false',
early_stopping_rounds=500, random_strength=1, border_count=254,
subsample=0.7, bootstrap_type='Bernoulli', task_type='GPU',
class_weights=[0.5, 2.0, 2.0]
```

---

## Experiment 1: Long OOS (Single Split, 3 Seeds)

**Test period**: Sep 2025 - Jan 2026 (120 days)
**Scripts**: `encode_v10.py`, `train_v10_long_oos.py`
**Output**: `results_v10/long_oos/`

### Per-Seed Results

| Seed | AUC | @0.70 Profit | @0.70 Prec | @0.75 Profit | @0.75 Prec | @0.80 Profit | @0.80 Prec |
|------|-----|-------------|-----------|-------------|-----------|-------------|-----------|
| 42 | 0.884 | +330% | 61.4% | +336% | 66.6% | +324% | 72.6% |
| 123 | 0.877 | +352% | 60.5% | +298% | 67.3% | +216% | 75.6% |
| 777 | 0.885 | +366% | 61.2% | +386% | 65.2% | +288% | 70.1% |
| **Mean** | **0.882** | **+349%** | **61.0%** | **+340%** | **66.4%** | **+276%** | **72.8%** |

### Top Features (Long OOS, avg across 3 seeds)

| # | Feature | Importance | V10? |
|---|---------|-----------|------|
| 1 | vol_body_product_1D | 7.82 | No (V6) |
| 2 | atr_normalized_1D | 7.34 | No (V6) |
| 3 | vol_body_product_3D | 6.50 | No (V6) |
| 4 | stoch_pos_1D_w10 | 4.79 | No (V5) |
| 5 | vol_body_product_12H | 3.16 | No (V6) |
| 6 | vol_ratio_1D | 2.92 | No (V6) |
| 7 | **hour_sin** | **2.58** | **YES** |
| 8 | vol_body_product_8H | 2.55 | No (V6) |
| 9 | stoch_pos_3D_w10 | 2.02 | No (V5) |
| 10 | vol_ratio_3D | 1.79 | No (V6) |
| 12 | **dow_sin** | **1.47** | **YES** |
| 19 | **dow_cos** | **0.97** | **YES** |

**V10 in top-20**: 3/20 (hour_sin, dow_sin, dow_cos — all temporal F4)
**Crossing features F1/F2**: ~0.56% total importance — effectively DEAD

### Conclusion (Long OOS)

- V10 achieves +349% @0.70 (vs V8's +353%) — comparable
- @0.80 hits 72.8% precision — first time above 70% target
- Only temporal features (F4) contribute meaningfully from V10
- Crossing counts (F1/F2), correlation dynamics (F3), interactions (F5) all dead

---

## Experiment 2: Young+Adult Only (Drop Balzak/Gran TF Features)

**Hypothesis**: If 1D/3D dominate via volume features, maybe dropping them forces model to use faster TFs + crossing features.
**Scripts**: `train_v10_young_adult.py`
**Output**: `results_v10/young_adult_oos/`
**Features kept**: 5M, 15M, 30M, 1H, 2H, 4H + global (dropped `_3D`, `_1D`, `_12H`, `_8H`, `_6H`)

### Key Results

| Metric | Full V10 (508 feat) | Young+Adult |
|--------|---------------------|-------------|
| Features | 508 | ~210 |
| @0.70 mean profit | +349% | Much lower |
| Top feature | vol_body_product_1D (7.82) | cascade_volume (11.14) |

### Top Features (Young+Adult)

| # | Feature | Importance | V10? |
|---|---------|-----------|------|
| 1 | **cascade_volume** | **11.14** | **YES (F5)** |
| 2 | atr_normalized_4H | 7.25 | No (V6) |
| 3 | **dow_sin** | **3.28** | **YES (F4)** |
| 4 | cs_dsf_4H_w23 | 2.94 | No (V5) |
| 5 | angle_slow_4H | 2.52 | No (V5) |
| 6 | **hour_sin** | **2.36** | **YES (F4)** |
| 7 | vol_body_product_4H | 2.13 | No (V6) |
| 8 | cross_tf_weighted_slope | 2.07 | No (V6) |
| 19 | **xtf_corr_agreement** | **1.35** | **YES (F3)** |

### Conclusions (Young+Adult)

1. **1D/3D features ARE the model** — removing them collapses profit dramatically
2. **cascade_volume (#1 at 11.14)** — V10 interaction feature becomes king when 1D/3D removed
3. **4H is strongest fast TF** — atr_normalized_4H, angle_slow_4H, vol_body_product_4H all top-10
4. **V10 features shine when not competing with 1D/3D**: 4/20 top features are V10 (vs 3/20 in full)
5. **Crossing counts still dead** even without competition — the expert signal doesn't encode into ML

---

## Experiment 3: Walk-Forward Validation (AUTHORITATIVE)

**Architecture**: 4-fold expanding window, 3 seeds/fold = 12 models
**Scripts**: `train_v10_walkforward.py`
**Output**: `results_v10/walkforward/`
**Same WF structure as V6/V7**: embargo (7d), cooldown (60 candles), val-threshold

### Per-Fold Results (avg across 3 seeds)

| Fold | Period | AUC | @0.70 Profit | @0.70 Prec | @0.75 Profit | @0.75 Prec | @0.80 Profit | @0.80 Prec |
|------|--------|-----|-------------|-----------|-------------|-----------|-------------|-----------|
| 0 | Feb-May '25 | 0.874 | +135% | 71.4% | +131% | 78.1% | +102% | 84.9% |
| 1 | May-Aug '25 | 0.879 | +14% | 38.2% | +14% | 42.4% | +7% | 48.0% |
| 2 | Aug-Nov '25 | 0.850 | +18% | 46.6% | +29% | 60.4% | +23% | 75.0% |
| 3 | Nov '25-Jan '26 | 0.904 | +87% | 65.9% | +97% | 69.8% | +85% | 74.4% |
| **Total** | | **0.877** | **+253%** | **55.5%** | **+271%** | **62.7%** | **+217%** | **70.6%** |

### Cross-Version Comparison (Walk-Forward @0.70)

| Model | AUC | Total Profit | Avg Precision | Folds Profitable |
|-------|-----|-------------|---------------|-----------------|
| V6 d2 | 0.870 | +246% | 51.0% | 4/4 |
| V7 d8 | 0.871 | +227% | ~51% | 4/4 |
| **V10 d8** | **0.877** | **+253%** | **55.5%** | **4/4** |

### V10 Precision Progression (Walk-Forward)

| Threshold | Total Profit | Avg Precision | Folds Profitable | Notes |
|-----------|-------------|---------------|-----------------|-------|
| @0.65 | +259% | 50.9% | 4/4 | |
| @0.70 | +253% | 55.5% | 4/4 | Beats V6 (+246%) |
| @0.75 | **+271%** | **62.7%** | **4/4** | **BEST PROFIT** |
| @0.80 | +217% | **70.6%** | **4/4** | **HITS 70% TARGET** |

### Walk-Forward Pass/Fail Checks (ALL 6 PASS)

| # | Check | Result |
|---|-------|--------|
| 1 | AUC > 0.70 all folds | PASS (min=0.835) |
| 2 | @0.70: 4/4 folds profitable | PASS (+253% total) |
| 3 | @0.75: 4/4 folds profitable | PASS (+271% total) |
| 4 | @0.80: 4/4 folds profitable | PASS (+217% total) |
| 5 | @0.80 avg precision >= 70% | PASS (70.6%) |
| 6 | Seed stability (AUC CV < 5%) | PASS (2.4%) |

### Walk-Forward Feature Stability (12 runs)

- 14 features appear in ALL 12 runs' top-20
- `hour_sin` [V10 F4] in top-20 of 12/12 runs — only V10 feature with perfect stability
- `dow_sin`, `dow_cos` [V10 F4] in ~10/12 runs
- Crossing features (F1/F2) never appear in any fold's top-20

### Fold 2 Improvement

V6 Fold 2 was the only losing fold (-6% honest, +36% @0.70).
V10 Fold 2 @0.80: **+23% profit, 75% precision** — temporal features help in mild chop regime.

---

## Key Findings

### What V10 Proved

1. **Temporal features are real signal** — `hour_sin` (trading session timing) is the only V10 feature with consistent importance across all 12 walk-forward runs. Market microstructure (session times) matters.

2. **70% precision achieved** — @0.80 threshold delivers 70.6% avg precision across walk-forward, meeting the expert manual WR target for the first time.

3. **Crossing counts are permanently dead** — Even with correct discrete encoding (counts 0-7, staircase-safe), cross-window crossing features (F1/F2) contribute <1% importance. The expert's primary visual signal does not survive ML encoding.

4. **Volume-direction composites remain king** — `vol_body_product_1D` is #1 in all experiments. The model trades on volume conviction, not angle convergence.

5. **Profit-precision tradeoff is smooth** — @0.70(+253%, 55.5%) → @0.75(+271%, 62.7%) → @0.80(+217%, 70.6%). No cliff edges.

### Recommended Operating Points

| Use Case | Threshold | Profit | Precision | Trades |
|----------|-----------|--------|-----------|--------|
| Maximum profit | @0.75 | +271% (WF) | 62.7% | ~55/fold |
| Maximum precision | @0.80 | +217% (WF) | 70.6% | ~30/fold |
| Balanced | @0.70 | +253% (WF) | 55.5% | ~70/fold |

### V10 vs Prior Versions Summary

| Version | Features | AUC (WF) | @0.70 Profit | @0.70 Prec | @0.80 Prec | Status |
|---------|----------|----------|-------------|-----------|-----------|--------|
| V6 | 395 | 0.870 | +246% | 51.0% | N/A | Previous best |
| V7 | 395 | 0.871 | +227% | ~51% | N/A | Seed validation |
| **V10** | **508** | **0.877** | **+253%** | **55.5%** | **70.6%** | **CURRENT BEST** |

---

## Files

| File | Purpose |
|------|---------|
| `model_training/encode_v10.py` | 508-feature encoder (V6 base + Phase F) |
| `model_training/train_v10_long_oos.py` | Single-split 3-seed training |
| `model_training/train_v10_young_adult.py` | Young+Adult ablation (no 1D/3D features) |
| `model_training/train_v10_walkforward.py` | 4-fold walk-forward with 3 seeds (AUTHORITATIVE) |
| `model_training/results_v10/long_oos/` | Single-split results (3 models, equity PNGs, trade CSVs) |
| `model_training/results_v10/young_adult_oos/` | Y+A ablation results |
| `model_training/results_v10/walkforward/` | Walk-forward results (12 models, summary CSV, equity PNGs) |
