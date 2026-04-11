# ML Pipeline V10 -- Production Model

## Overview

- **518 features** = 395 V6 base + 113 Phase F (cross-scale convergence + temporal) + 10 Phase G (Bollinger Band extremes)
- **3-class** (NO_TRADE/LONG/SHORT), CatBoost GPU, 3-seed ensemble (seeds 42/123/777)
- **First model to achieve 70% precision target** (@0.80 = 70.6% avg walk-forward)
- Labels: SL/TP trade outcome simulation on 5M candles (SL=2%, TP=4%, max_hold=288 candles/24h)
- **Data**: Aug 17, 2017 -> present (~8yr). 904K rows for 5M. Backfilled via `download_backfill.py`

## CatBoost Hyperparameters

### Current Best: Optuna Winner + d12 (walk-forward validated)

```python
depth=12, iterations=3000, learning_rate=0.052, l2_leaf_reg=1.5,
min_data_in_leaf=1550, loss_function='MultiClass',
eval_metric='TotalF1:average=Macro;use_weights=false',
early_stopping_rounds=600, random_strength=0.58, border_count=68,
subsample=0.58, bootstrap_type='Bernoulli', task_type='GPU',
class_weights=[0.38, 2.37, 2.37]
```

### Previous: V7 d8 baseline (used in production + SL/TP sweep)

```python
depth=8, iterations=5000, learning_rate=0.02, l2_leaf_reg=15,
min_data_in_leaf=500, random_strength=1, border_count=254,
subsample=0.7, class_weights=[0.5, 2.0, 2.0]
```

## Optuna Hyperparameter Search (2026-03-30)

100-trial TPE search on 2 folds (0+3), 3 seeds, profit@0.75 objective. SQLite: `results_v10/optuna_search/optuna_study.db`.

**Search space**: depth[4-10], iterations[1000-8000], lr[0.005-0.1], l2[1-50], leaf[50-2000], subsample[0.5-0.95], border[32-254], rs[0.1-10], cw_nt[0.2-0.8], cw_trade[1-4].

**Key findings**:
- depth has strongest correlation with profit (r=+0.509); d=9 dominates top 15
- Low l2_leaf_reg wins (top=2.4 vs bottom=7.4); high min_data_in_leaf (1500+) = more robust
- Trial 34 (d9, lr=0.052, l2=1.5, leaf=1550) = +288% total profit
- Trial 94 (d9, lr=0.025, l2=1.1, leaf=1300) = PF 7.70, 79.2% precision (fewer trades, highest quality)

## Depth Frontier Experiments (2026-03-31)

Tested Optuna winner params at increasing depths:

| Config | @0.75 Total | @0.80 Total | @0.80 Prec | @0.80 PF | AUC |
|--------|-------------|-------------|------------|----------|-----|
| V7 d8 (baseline) | +271% | +217% | 70.6% | ~4.0 | 0.877 |
| **Optuna+d12** | **+277%** | **+256%** | **~80%** | **~6.0** | **0.872** |
| Optuna+d14 | +209% | +150% | ~90% | ~6.6 | 0.858 |

**d12 = best balance** of profit and precision. d14 too selective (ultra-high precision but leaves money on table).

d14 params tested: depth=14, lr=0.025, l2=3.0, leaf=2000, rs=5.0 (rest same as d12).

## Walk-Forward Architecture

4-fold expanding window with embargo (7d), cooldown (60 candles/5hr), val-threshold selection:

| Fold | Train End | Val End | Embargo End | Test End | Regime |
|------|-----------|---------|-------------|----------|--------|
| 0 | 2025-01-01 | 2025-02-01 | 2025-02-08 | 2025-05-01 | Trending |
| 1 | 2025-04-01 | 2025-05-01 | 2025-05-08 | 2025-08-01 | Heavy chop |
| 2 | 2025-07-01 | 2025-08-01 | 2025-08-08 | 2025-11-01 | Mild chop |
| 3 | 2025-10-01 | 2025-11-01 | 2025-11-08 | 2026-01-15 | Mixed |

## V10 Walk-Forward Results (4 folds x 3 seeds = 12 runs)

### Optuna+d12 (current best, 2026-03-31)

| Threshold | Total Profit | Avg Precision | Folds Profitable |
|-----------|-------------|---------------|------------------|
| @0.70 | **+231%** | ~74% | 4/4 |
| @0.75 | **+277%** | ~80% | 4/4 |
| @0.80 | **+256%** | **~80%** | 4/4 |

AUC: 0.872 +/- 0.019. ALL 6 pass/fail checks PASS.

### V7 d8 baseline (original V10)

| Threshold | Total Profit | Avg Precision | Folds Profitable |
|-----------|-------------|---------------|------------------|
| @0.70 | +253% | 55.5% | 4/4 |
| @0.75 | +271% | 62.7% | 4/4 |
| @0.80 | +217% | 70.6% | 4/4 |

AUC: 0.877 +/- 0.021. ALL 6 pass/fail checks PASS.

## SL/TP Sweep Winner: C1 (SL=0.5%, TP=1.0%)

5x V10 profit: +1,381% @0.75, 78.5% precision @0.80. Tighter targets give 2x more trade labels. Full results: `model_training/SLTP_SWEEP_RESULTS.md`

## Production Models

- Location: `model_training/results_v10/production/`
- Files: `production_model_s{42,123,777}.cbm` + `production_metadata.json`
- Features: 518 (V10), default threshold: 0.75
- Live pipeline (`live_predict.py`): download klines -> ETL -> encode 518 features -> ensemble predict

## Experiment History

| Version | Features | AUC | Walk-Forward | Best Result | Status |
|---------|----------|-----|-------------|-------------|--------|
| V1 | 354 | 0.42 | N/A | Anti-predictive | DEPRECATED |
| V2.1 | 135 | ~0.70 | N/A | +510% @0.60 | Baseline |
| V6 | 395 | 0.870 | PASS (4/4) | +246% @0.70 | V10 base |
| V7 | 395 | 0.871 | PASS (6/6) | +227% @0.70 | Multi-seed |
| V10 (V7 d8) | 508 | 0.877 | PASS (12/12) | +271% @0.75 | Baseline |
| **V10 (Optuna d12)** | **508** | **0.872** | **PASS (12/12)** | **+277% @0.75, 80% prec** | **BEST @2/4** |
| V10 (Optuna d14) | 508 | 0.858 | PASS (12/12) | +209% @0.75, 90% prec | Too selective |
| **V10 (d14 + BB)** | **518** | **0.856** | **PASS (12/12)** | **+232% @0.75, +250% @0.70** | **+BB features** |
| V12 | 561 | ~0.877 | Tied V10 | No improvement | DEPRECATED |
| **C1** | **508** | **0.866** | **PASS (12/12)** | **+1,381% @0.75** | **NEW BEST (tight SL/TP)** |

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

## Stacking Meta-Model V2 (2026-03-31)

5 base models with diverse labels → OOS probabilities → meta-model with 131 interaction features.

### Architecture

| Stage | What | Period |
|-------|------|--------|
| Base models train | V10/C1/C2/H12/H48 on 508 features | 2020-02-17 → 2024-07-07 |
| Meta-features gen | OOS probs + interactions (131 feat) | 2024-07-07 → 2026-03-31 |
| Meta-model train | CatBoost d5 on meta-features | 2024-07-07 → 2025-12-31 |
| Meta-model test | Held-out evaluation | 2026-01-01 → 2026-03-31 |

### Base Models

| Model | SL% | TP% | max_hold | Perspective |
|-------|-----|-----|----------|-------------|
| V10 | 2.0 | 4.0 | 288 (24h) | Standard |
| C1 | 0.5 | 1.0 | 72 (6h) | Micro-scalping |
| C2 | 0.5 | 1.5 | 72 (6h) | Tight |
| H12 | 2.0 | 4.0 | 144 (12h) | Half-day horizon |
| H48 | 2.0 | 4.0 | 576 (48h) | 2-day horizon |

### Meta-Model Params (Optuna-optimized)

```python
depth=5, iterations=3000, learning_rate=0.0427, l2_leaf_reg=0.635,
min_data_in_leaf=1700, class_weights=[0.89, 1.32, 1.32],
random_strength=1.43, border_count=50, subsample=0.52
```

Key: shallower trees (d5 not d8), near-equal class weights, high leaf min — very different from base model params.

### Feature Categories (131 total)

- 15 raw probs (5 models × 3 classes)
- 50 lagged probs (5 models × 2 directions × 5 windows: 1h/6h/12h/24h/48h)
- 20 temporal momentum (prob change over 6h/24h per model per direction)
- 10 rolling stability (rolling std of probs over 6h/24h)
- 8 voting/confirmation (majority vote, V10-uncertain-but-confirmed)
- 4 cross-horizon convergence (horizon flow, H12-H48 agreement)
- 24 other interactions (entropy, spreads, consensus, conf ratios, etc.)

### Results (held-out test Jan-Mar 2026)

AUC: 0.8506

| Threshold | Trades | WR | Profit | PF |
|-----------|--------|-----|--------|-----|
| @0.55 | 103 | 61.2% | +172% | 3.15 |
| @0.65 | 67 | 70.1% | +148% | 4.70 |
| @0.70 | 56 | 75.0% | +140% | 6.00 |
| @0.75 | 44 | 81.8% | +128% | 9.00 |
| @0.80 | 32 | 78.1% | +86% | 7.14 |

Best asymmetric: LONG@0.50 / SHORT@0.42 → 120 trades, WR=58.3%, +180%, PF=2.80

### Key Findings

1. C1/C2 (tight SL/TP) contribute ~2% importance — horizon diversity (H12/H48) is what matters
2. lag288 (prob 24h ago) = 2nd most important signal — momentum of conviction
3. Entropy features contribute ~8% — model uncertainty is tradeable
4. SHORT needs lower threshold than LONG (model naturally more conservative on SHORT)

Scripts: `train_stacking_v2.py`, `optuna_meta_search.py`
Results: `results_v10/stacking_v2/`, `results_v10/optuna_meta_search/`

## Stacking Meta-Model V3 (2026-04-05)

Replaces V2. Drops C1/C2 (2% combined importance), adds ASYM (1%/6%/288) and TREND (3%/3%/576) base models. Dual meta-targets: PRIMARY (2%/4%) for volatile weeks, TIGHT (1%/2%) for ranging markets.

| Base Model | SL% | TP% | max_hold | Purpose |
|------------|------|------|----------|---------|
| V10 | 2.0 | 4.0 | 288 | Anchor |
| H12 | 2.0 | 4.0 | 144 | Half-day horizon |
| H48 | 2.0 | 4.0 | 576 | 2-day horizon |
| ASYM | 1.0 | 6.0 | 288 | Asymmetric runner |
| TREND | 3.0 | 3.0 | 576 | Symmetric trend |

Signal router: PRIMARY fires → trade 2%/4%. Only TIGHT fires → trade 1%/2%. Neither → NO_TRADE.

V3 results @0.70: 114 trades (42.8/mo), WR=75.4%, PF=6.39, Profit=+194%. Max gap reduced from 13.2d to 8.3d.

Scripts: `train_stacking_v3.py`
Results: `results_v10/stacking_v3/` (primary/, tight/, combined/)

### V3 Walk-Forward Validation (2026-04-06)

4-fold walk-forward with nested training: 5 base models -> OOS meta-features -> meta-models -> router.

**Dual-meta baseline** (3-class meta-model): +299% total, PASS 4/4. TIGHT stream fills summer gap.

### Meta-Labeling (Lopez de Prado) — CURRENT BEST (2026-04-07)

Replaces 3-class meta-model with binary "profitable?" filter. Direction from base consensus (avg_long vs avg_short, threshold=0.30). Meta-model only filters false positives. **PASS 4/4, +321% total.**

```
Stage 1: 5 base models predict 3-class probabilities
Stage 2: Base consensus determines direction (avg_long vs avg_short >= 0.30)
Stage 3: Binary CatBoost predicts "profitable?" (Logloss, depth=4)
Stage 4: If consensus != NO_TRADE AND profitable_prob >= 0.70 → trade
```

| Fold | Trades | WR | Profit | PF |
|------|--------|----|--------|----|
| 0 | 132 (71P+61T) | 62.1% | +166% | 3.08 |
| 1 (summer) | 58 (11P+47T) | 58.6% | +42% | 2.31 |
| 2 | 100 (39P+61T) | 50.0% | +59% | 1.81 |
| 3 | 41 (7P+34T) | 73.2% | +54% | 4.86 |
| **Total** | **331** | **~61%** | **+321%** | — |

**Why it wins:** Binary filter simpler than 3-class. Fixes Fold 2 (+18%→+59%). AUCs 0.83-0.93 vs 0.79-0.88. Never combine with calibration (kills disagreement signal).

META_LABEL_PARAMS: `depth=4, lr=0.0427, l2=1.0, leaf=1700, sub=0.52, rs=1.43`

### Grid Search: Meta-Label Hyperparameters (2026-04-08)

3072 combos (6 params, wide steps) on Fold 0 inner CV. 3.6h GPU. Results: `results_v10/grid_search_meta_label/`

**Key findings:**
1. **3 of 6 params irrelevant** — leaf, subsample, random_strength have ZERO effect
2. **Only depth, lr, l2 matter** — 52 unique prediction signatures across 696 valid combos
3. **75% of combos produce zero trades** — meta-model defaults to "not profitable"
4. **depth x lr is master interaction** — deep+fast = more trades/lower WR, shallow+slow = fewer/higher WR
5. **Current d=4 on a cliff edge** — max trade volume but sensitive to perturbation
6. **d=9 + l2=30 sweet spot** for profit (deep trees + heavy regularization)

| Config | Fold 0 Profit | Trades | WR | PF |
|--------|--------------|--------|-----|-----|
| Current (d=4, lr=0.0427, l2=1.0) | +151% | 132 | 59.8% | 2.82 |
| Candidate A: d=9, lr=0.10, l2=30 | +148% | 67 | 70.1% | 4.70 |
| Candidate B: d=7, lr=0.10, l2=8 | +140% | 86 | 60.5% | 3.06 |

**Next:** 4-fold walkforward of A and B to validate.

Scripts: `train_stacking_v3_walkforward.py`, `grid_search_meta_label.py`, `validate_v3_audit.py`
Results: `results_v10/stacking_v3_walkforward/`, `results_v10/grid_search_meta_label/`

## Phase G: Bollinger Band Extreme Features (2026-04-05)

10 new features added to all 3 encoding paths (batch, incremental, live_predict). BB(3, 35) for 5M, 15M, 1H, 4H, 1D:
- `bb_lower_pierce_{TF}` = (BB_lower - Low) / Close — positive when low pierces below band
- `bb_upper_dist_{TF}` = (BB_upper - Close) / Close — negative when close above band

Activates in ranging markets when regression features go quiet. Total features: 508 → 518.

Walk-forward validation (4-fold, 3 seeds, 518 features): AUC=0.856±0.032, +250% @0.70 (4/4 folds profitable). Compared to 508-feature baseline (AUC=0.858, +247% @0.70): essentially unchanged at low thresholds, **+23% improvement at @0.75, +26% at @0.80**. BB features rank: `bb_lower_pierce_1H` #18/518 (0.85%), total BB group ~3.3%. Short TFs (1H, 5M, 15M) strongest.

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
| Optuna Search Results | `model_training/results_v10/optuna_search/` |
