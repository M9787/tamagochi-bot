# ML Pipeline V10 -- Production Model

## Overview

- **508 features** = 395 V6 base + 113 Phase F (cross-scale convergence + temporal)
- **3-class** (NO_TRADE/LONG/SHORT), CatBoost GPU, 3-seed ensemble (seeds 42/123/777)
- **First model to achieve 70% precision target** (@0.80 = 70.6% avg walk-forward)
- Labels: SL/TP trade outcome simulation on 5M candles (SL=2%, TP=4%, max_hold=288 candles/24h)

## CatBoost Hyperparameters (V7 d8)

```python
iterations=5000, depth=8, learning_rate=0.02, l2_leaf_reg=15,
min_data_in_leaf=500, loss_function='MultiClass',
eval_metric='TotalF1:average=Macro;use_weights=false',
early_stopping_rounds=600, random_strength=1, border_count=254,
subsample=0.7, bootstrap_type='Bernoulli', task_type='GPU',
class_weights=[0.5, 2.0, 2.0]
```

## Walk-Forward Architecture

4-fold expanding window with embargo (7d), cooldown (60 candles/5hr), val-threshold selection:

| Fold | Train End | Val End | Embargo End | Test End | Regime |
|------|-----------|---------|-------------|----------|--------|
| 0 | 2025-01-01 | 2025-02-01 | 2025-02-08 | 2025-05-01 | Trending |
| 1 | 2025-04-01 | 2025-05-01 | 2025-05-08 | 2025-08-01 | Heavy chop |
| 2 | 2025-07-01 | 2025-08-01 | 2025-08-08 | 2025-11-01 | Mild chop |
| 3 | 2025-10-01 | 2025-11-01 | 2025-11-08 | 2026-01-15 | Mixed |

## V10 Walk-Forward Results (4 folds x 3 seeds = 12 runs)

| Threshold | Total Profit | Avg Precision | Folds Profitable |
|-----------|-------------|---------------|------------------|
| @0.70 | **+253%** | 55.5% | 4/4 |
| @0.75 | **+271%** | 62.7% | 4/4 |
| @0.80 | **+217%** | **70.6%** | 4/4 |

AUC: 0.877 +/- 0.021. ALL 6 pass/fail checks PASS.

## SL/TP Sweep Winner: C1 (SL=0.5%, TP=1.0%)

5x V10 profit: +1,381% @0.75, 78.5% precision @0.80. Tighter targets give 2x more trade labels. Full results: `model_training/SLTP_SWEEP_RESULTS.md`

## Production Models

- Location: `model_training/results_v10/production/`
- Files: `production_model_s{42,123,777}.cbm` + `production_metadata.json`
- Features: 508 (V10), default threshold: 0.75
- Live pipeline (`live_predict.py`): download klines -> ETL -> encode 508 features -> ensemble predict

## Experiment History

| Version | Features | AUC | Walk-Forward | Best Result | Status |
|---------|----------|-----|-------------|-------------|--------|
| V1 | 354 | 0.42 | N/A | Anti-predictive | DEPRECATED |
| V2.1 | 135 | ~0.70 | N/A | +510% @0.60 | Baseline |
| V6 | 395 | 0.870 | PASS (4/4) | +246% @0.70 | V10 base |
| V7 | 395 | 0.871 | PASS (6/6) | +227% @0.70 | Multi-seed |
| **V10** | **508** | **0.877** | **PASS (12/12)** | **+271% @0.75** | **PRODUCTION** |
| V12 | 561 | ~0.877 | Tied V10 | No improvement | DEPRECATED |
| **C1** | **508** | **0.866** | **PASS (12/12)** | **+1,381% @0.75** | **NEW BEST** |

## Key Lessons (V1-V10)

1. Dropping NO_TRADE rows = model never learns WHEN to trade (V1 fatal flaw)
2. Walk-forward mandatory -- single-split inflates 100x (V6: 13,716% -> 130%)
3. Cooldown (60 candles/5hr) reveals true trade count (97-99% reduction)
4. Volume-direction composites dominate all regression features (vol_body_product_1D = #1)
5. TF-native lags: MUST shift on native data BEFORE merge_asof
6. Temporal features (hour_sin) = real signal; crossing counts die in ML
7. Val threshold selection noisy on small val sets -- use fixed thresholds
8. SL/TP label definition = #1 hyperparameter (tight 0.5/1.0 -> 5x profit vs wide 2/4)

Full analysis: `model_training/EXPERIMENT_RETROSPECTIVE.md`

## Detailed Result Documents

| Document | Path |
|----------|------|
| V10 Walk-Forward Results | `model_training/V10_EXPERIMENT_RESULTS.md` |
| V10 2yr OOS Audit | `model_training/V10_2YR_OOS_AUDIT.md` |
| SL/TP Sweep Results | `model_training/SLTP_SWEEP_RESULTS.md` |
| V6 Quant Assessment | `model_training/V6_QUANT_ASSESSMENT.md` |
| V7 Experiment Results | `model_training/V7_EXPERIMENT_RESULTS.md` |
| V7 Cross-Experiment | `model_training/V7_CROSS_EXPERIMENT_ANALYSIS.md` |
| Experiment Retrospective | `model_training/EXPERIMENT_RETROSPECTIVE.md` |
| Experiment History Archive | `model_training/EXPERIMENT_HISTORY.md` |
