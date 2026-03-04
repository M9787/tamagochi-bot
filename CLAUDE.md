# CLAUDE.md

## Project Overview

Cryptocurrency trading signal analysis system for BTCUSDT that:
1. Downloads kline data from Binance across **11 timeframes** (6 years of history)
2. Performs rolling linear regression with **5 window sizes** (30, 60, 100, 120, 160) — 55 signal sources
3. Displays results in a **Streamlit dashboard** for visual feature engineering confirmation
4. **ML Pipeline V10** (current): V6 base (395 features) + 113 cross-scale convergence features = 508 total → CatBoost GPU 3-class → walk-forward validation

## MANDATORY: Model Evaluation Checklist

**EVERY training script MUST include these metrics. No exceptions.**

| Metric | Why |
|--------|-----|
| **ROC AUC** (per-class + macro OVR) | Does model rank predictions correctly? |
| **ROC Curve + PR Curve PNGs** | Operating point and precision/recall tradeoff |
| **Per-class Precision/Recall/F1** | Are LONG and SHORT balanced? |
| **Confusion Matrix PNG** | Where are errors going? |
| **Trade Precision** (at threshold) | Actual trading accuracy |
| **Profit Factor, Max Drawdown, Sharpe** | Risk-adjusted performance |
| **Equity Curve PNG** | Smooth or spiky? |
| **Trade Log CSV** | Per-trade audit trail with timestamps |

> **No model training without full performance evaluation. EVER. If you can't see ROC/AUC, the model doesn't exist.**

## Core Algorithm (analysis.py)

`iterative_regression(df, window_size)` is the **atom** of the entire system. For each data point, maintains a sliding window of `2 * window_size` prices using `sqrt(price)`, performs `scipy.stats.linregress` on each half.

| Output | What it captures |
|--------|-----------------|
| `slope_b` | Backward regression slope (historical trend) |
| `slope_f` | Forward regression slope (current direction vector) |
| `angle` | Angle between backward and forward slopes (trend divergence) |
| `acceleration` | First difference of angle (rate of change of divergence) |

**Outputs per row:** `slope_b`, `slope_f`, `intercept_b/f`, `p_value_b/f`, `corr`, `spearman`, `angle`, `actual`, `time`

## Signal Theory (Expert-Validated)

### Signal Component Hierarchy

| Component | Role | Priority |
|-----------|------|----------|
| **Crossing (C)** | PREDICTS | Primary trigger — angle convergence/cross/divergence event |
| **Reversal (R)** | CONFIRMS | Secondary — 5-point angle pattern validates crossing |
| **Direction (slope_f)** | ALIGNS | Context — LONG or SHORT territory |
| **Acceleration (A)** | FILTERS | Quality gate — reject CLOSE/VERY_CLOSE GMM zones |

### Crossing Physics (Ball & Gravity Model)

Angle behaves like a **ball trajectory with momentum**. Crossings are LEADING indicators — each is an impulse for a future event, not about the present:
- **Young ball falling + Elder ball rising** at cross = **LONG** (momentum transfer)
- **Elder ball falling + Young ball rising** at cross = **SHORT** (momentum dying)
- The DIVERGENCE after crossing is the signal, not the crossing point itself
- Crossing is a CONTINUOUS trajectory (converge → cross → diverge), not binary
- Higher timeframes = longer delay + bigger percentage move. Lags must be TF-native (shift BEFORE merge_asof)

### Acceleration Filter (GMM)

5 clusters via Gaussian Mixture Model: `VERY_DISTANT, DISTANT, BASELINE, CLOSE, VERY_CLOSE`
- **REJECT** signals in `CLOSE` and `VERY_CLOSE` — most false signals live there
- Quality signals come from `VERY_DISTANT, DISTANT, BASELINE` only

### Trading Philosophy

- **Precision >> Recall**: better to miss trades than generate false ones
- **Asymmetric loss**: predicting NO_TRADE when signal exists = safe; predicting TRADE when none = LOSS
- Target: **60%+ precision** with 4% TP / 2% SL (break-even = 33.3%). Expert manual WR: 70%+
- Only **~10% of market moments** have real tradeable signals. Labels: NO_TRADE=72.7%, SHORT=13.7%, LONG=13.6%

## Architecture

```
main.py                        # CLI entry point
dashboard.py                   # Streamlit dashboard (feature engineering microscope)
backtest_dashboard.py          # Streamlit dashboard (predictions vs actuals, candlestick + markers)
backfill_predictions.py        # Batch-predict N hours of 5M candles + compute SL/TP outcomes
trading_bot.py                 # V10 trading bot (dry-run / testnet / live Binance Futures)
core/
  config.py                    # Centralized config (paths, TFs, windows, thresholds)
  analysis.py                  # iterative_regression, calculate_acceleration
  processor.py                 # TimeframeProcessor: 55 TF/window combos
  signal_logic.py              # SignalLogic, PricePredictor, CalendarDataBuilder
data/
  downloader.py                # Binance data downloader
  target_labeling.py           # SL/TP target labeling encoder (_test_long/short_trade_fast)
trading/
  executor.py                  # BinanceFuturesExecutor (testnet/live order execution)
  position_manager.py          # PositionManager (SL/TP tracking, entry averaging)
  safety.py                    # SafetyMonitor (7-day aggregated WR pause)
telegram_service/
  bot.py                       # TelegramMonitorBot (commands + push notifications)
  formatters.py                # HTML message templates
  readers.py                   # Read-only data readers (predictions, state, trades)
  subscribers.py               # Persistent subscriber store (atomic JSON)
  service.py                   # Entry point and CLI args
  Dockerfile                   # Minimal container image
  requirements.txt             # python-telegram-bot + pandas
model_training/
  live_predict.py              # Live prediction pipeline (single + batch + extended download)
  train_v10_production.py      # Train production models (3 seeds) → results_v10/production/
  download_data.py             # Step 1: 6yr data → actual_data/
  etl.py                       # Step 2: klines → regression → decomposed_data/ (55 CSVs)
  encode_v6.py                 # Step 3: 395 features → feature_matrix_v6.parquet
  encode_v10.py                # Step 3b: 508 features → feature_matrix_v10.parquet (V6 + Phase F)
  train_v6.py                  # Step 4: tournament (11 iters) → results_v6/
  train_v6_walkforward.py      # Step 5: 4-fold walk-forward → results_v6/walkforward/
  train_v7_walkforward.py      # V7: depth=8 multi-seed (3 seeds x 4 folds) → results_v7/walkforward/
  train_v10_long_oos.py        # V10: single-split 3-seed → results_v10/long_oos/
  train_v10_walkforward.py     # V10: 4-fold WF 3 seeds → results_v10/walkforward/ (AUTHORITATIVE)
  train_v10_sltp_screen.py     # SL/TP sweep: 10-config screening → results_v10/sltp_screen/
  train_v10_sltp_walkforward.py # SL/TP sweep: top 3 full WF → results_v10/sltp_winners/
  train_v10_young_adult.py     # V10: Young+Adult ablation → results_v10/young_adult_oos/
  build_labels.py              # SL/TP label generation (shared)
  visualize.py                 # Results visualization
  encode_v2.py / train_v2.py   # V2.1 baseline (kept for reference)
  encode.py / train.py         # V1 deprecated
  actual_data/                 # 11 raw kline CSVs (~630K rows for 5M)
  decomposed_data/             # 55 regression CSVs (13 cols each)
  encoded_data/                # feature_matrix_v{2,3,5,6,10}.parquet + labels_5M.csv
  results_v6/                  # Tournament results
  results_v6/walkforward/      # Walk-forward results
  results_v7/walkforward/      # V7 multi-seed results (12 models, audit docs)
  results_v10/                 # V10 results (long_oos, young_adult_oos, walkforward)
  results_v10/production/      # 3 production CatBoost models (.cbm) + metadata JSON
trading_logs/                  # Runtime outputs
  backfill_predictions.csv     # Backfill output (raw_signal + actual outcomes for all rows)
  backfill_klines_5m.csv       # 5M klines from last backfill (used by backtest dashboard)
  trades_YYYY-MM-DD.csv        # Live bot trade logs (one per day)
```

### Key Configuration (config.py)

- `TIMEFRAME_ORDER`: `3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M`
- `WINDOW_SIZES`: `[30, 60, 100, 120, 160]` mapping to `df, df1, df2, df3, df4`
- 7 crossing pairs: `(30,60), (30,100), (60,100), (60,120), (100,120), (100,160), (120,160)`
- TF weights: `3D=5, 1D=4, 12H=3.5, 8H=3, 6H=2.5, 4H=2, 2H=1.5, 1H=1.2, 30M=1.1, 15M=1.0, 5M=0.8`
- TF groups: **Youngs** (5M,15M,30M) → **Adults** (1H,2H,4H) → **Balzaks** (6H,8H,12H) → **Grans** (1D,3D)

## Commands

```bash
pip install -r requirements.txt
python data/downloader.py                              # Live data for dashboard
python main.py --mode dashboard                        # Launch Streamlit dashboard

# ML Pipeline V6 (5-step sequence)
python model_training/download_data.py                 # 1. Download 6yr data → actual_data/
python model_training/etl.py --start 2020-01-01 --end 2026-02-15 [--force]  # 2. ETL → decomposed_data/
python model_training/encode_v6.py [--force]           # 3. Encode → feature_matrix_v6.parquet
python model_training/train_v6.py                      # 4. Tournament → results_v6/
python model_training/train_v6_walkforward.py          # 5. Walk-forward → results_v6/walkforward/

# V7 Multi-Seed Validation (depth=8, 3 seeds x 4 folds)
python model_training/train_v7_walkforward.py          # → results_v7/walkforward/ (resumable via checkpoint)

# V10 Cross-Scale Convergence (508 features, CURRENT BEST)
python model_training/encode_v10.py [--force]          # 3b. Encode → feature_matrix_v10.parquet (508 features)
python model_training/train_v10_long_oos.py            # Single-split 3-seed → results_v10/long_oos/
python model_training/train_v10_walkforward.py         # 4-fold WF 3 seeds → results_v10/walkforward/ (AUTHORITATIVE)

# Live Prediction & Trading
python model_training/live_predict.py                         # Single prediction (threshold=0.75)
python model_training/live_predict.py --threshold 0.80        # Higher precision
python model_training/live_predict.py --loop --interval 300   # Continuous (5min)

# Backfill & Backtest Dashboard
python backfill_predictions.py                                # 72h backfill, threshold=0.50 recommended
python backfill_predictions.py --hours 168 --threshold 0.50   # 1 week, capture all signals
python backfill_predictions.py --threshold 0.80 --cooldown 60 # Higher precision, custom cooldown
streamlit run backtest_dashboard.py                           # Predictions vs actuals dashboard

# Trading Bot
python trading_bot.py --dry-run                               # Predict only, no orders
python trading_bot.py --dry-run --threshold 0.80              # Dry-run with custom threshold
python trading_bot.py --testnet                               # Binance Futures testnet
python trading_bot.py --live                                  # Production (requires CONFIRM)

# Legacy (reference only)
python model_training/encode_v2.py && python model_training/train_v2.py  # V2.1 baseline
```

## ML Pipeline V6 — Current Best (Walk-Forward Validated 2026-02-21)

### Overview

- **395 features** = 203 V5 KEEP + 192 new (20 directional concepts)
- Three new data sources: **p_value_f** (trend certainty), **Volume** (conviction), **wicks** (pressure)
- **3-class** (NO_TRADE/LONG/SHORT), CatBoost GPU
- Labels: SL/TP trade outcome simulation on 5M candles (SL=2%, TP=4%, max_hold=288 candles/24h)

### CatBoost Hyperparameters

```python
iterations=6000, depth=2, learning_rate=0.02, l2_leaf_reg=15,
min_data_in_leaf=500, loss_function='MultiClass',
eval_metric='TotalF1:average=Macro;use_weights=false',
early_stopping_rounds=600, random_strength=1, border_count=254,
subsample=0.7, bootstrap_type='Bernoulli', task_type='GPU',
class_weights=[0.5, 2.0, 2.0]
```

### Data Flow

```
actual_data/ (6yr klines, 11 TFs)
  → etl.py: iterative_regression() + calculate_acceleration()
  → decomposed_data/ (55 CSVs, 13 cols each — w30+w120 used)

encode_v6.py 5-phase pipeline:
  Phase A: Load V5 parquet → filter to 203 KEEP features
  Phase B: Load decomposed CSVs → 10 new per TF (p_value, regime, momentum)
  Phase C: Load raw klines → 6 new per TF + 3 ATR (volume, wicks)
  Phase D: Cross-TF summaries (slope_agreement, weighted_slope)
  Phase E: range_position_signed from aligned columns
  → encoded_data/feature_matrix_v6.parquet (630K rows x 395 features)
```

### 20 New Feature Concepts (V6)

| # | Feature | Formula | Count |
|---|---------|---------|-------|
| 1 | `trend_certainty` | -log10(clip(p_value_f_w120)) | x11 |
| 2 | `vol_ratio` | volume / ema(volume, 50) | x11 |
| 3 | `vol_body_product` | sign(C-O) * vol_ratio | x11 |
| 4 | `wick_asymmetry` | rolling_mean(wick_formula, 10) | x11 |
| 5 | `regime_quadrant` | sign(slope_f)*2 + sign(slope_b) | x11 |
| 6 | `corr_regime` | corr * sign(slope_f) | x11 |
| 7 | `corr_stability` | rolling_std(corr, 10) | x11 |
| 8 | `slope_sign_gradient` | sign(slope_f_w30) - sign(slope_f_w120) | x11 |
| 9 | `angle_regime` | min(angle,30) * sign(slope_f) | x11 |
| 10 | `regime_change_strength` | \|sf-sb\|*certainty*(sign differs) | x11 |
| 11 | `smoothed_momentum` | ema(slope_f-slope_b, 5) | x11 |
| 12 | `accel_signed` | acceleration * sign(slope_f) | x11 |
| 13 | `vol_impulse` | z-score(volume, 50) | x11 |
| 14 | `norm_body_accum` | rolling_sum(sign(C-O)*\|C-O\|/C, 20) | x11 |
| 15 | `range_position_signed` | (2*stoch_pos_w20-1)*sign(slope_f) | x11 |
| 16 | `atr_normalized` | ATR(14)/close | x3 |
| 17 | `cross_tf_slope_agreement` | mean(sign(slope_f) for top 8 TFs) | x1 |
| 18 | `cross_tf_weighted_slope` | weighted_mean(slope_f, tf_weights) | x1 |
| 19 | `p_value_trend` | trend_certainty - lag1 (TF-native) | x11 |
| 20 | `directional_vol_body` | ema(vol_body_product, 10) | x11 |

### Walk-Forward Architecture (Authoritative Validation)

4-fold expanding window with audit fixes:

| Fix | Description |
|-----|-------------|
| **Embargo (7d)** | Gap between val and test — SL/TP labels look up to 288 candles forward |
| **Cooldown (60 candles)** | 5hr gap between trades — reveals independent trades (97-99% reduction) |
| **Val-threshold** | Threshold selected on val set profit, applied to test |
| **Honest pass/fail** | Uses val-selected threshold, not best-on-test |

| Fold | Train End | Val End | Embargo End | Test End | Regime |
|------|-----------|---------|-------------|----------|--------|
| 0 | 2025-01-01 | 2025-02-01 | 2025-02-08 | 2025-05-01 | Trending |
| 1 | 2025-04-01 | 2025-05-01 | 2025-05-08 | 2025-08-01 | Heavy chop |
| 2 | 2025-07-01 | 2025-08-01 | 2025-08-08 | 2025-11-01 | Mild chop |
| 3 | 2025-10-01 | 2025-11-01 | 2025-11-08 | 2026-01-15 | Mixed |

### Walk-Forward Results

**OVERALL: PASS** — AUC 0.870 +/- 0.025, @0.70 = 4/4 folds profitable

#### Honest Results (val-selected thresholds, after cooldown)

| Fold | Period | AUC | Threshold | Trades | Profit | Shorts? |
|------|--------|-----|-----------|--------|--------|---------|
| 0 | Feb-May '25 | 0.863 | @0.50 | 160 | **+64%** | YES |
| 1 | May-Aug '25 | 0.884 | @0.70 | 34 | **+10%** | YES |
| 2 | Aug-Nov '25 | 0.833 | @0.65 | 63 | **-6%** | YES |
| 3 | Nov '25-Jan '26 | 0.901 | @0.60 | 104 | **+62%** | YES |
| **Total** | | **0.870** | | **361** | **+130%** | **4/4** |

#### Fixed @0.70 (all folds profitable)

| Fold | Precision | Profit | PF | Sharpe |
|------|-----------|--------|-----|--------|
| 0 | 66.3% | +130% | 2.76 | 0.499 |
| 1 | 29.9% | +10% | 1.24 | 0.101 |
| 2 | 51.7% | +36% | 1.82 | 0.286 |
| 3 | 56.0% | +70% | 1.95 | 0.320 |
| **Avg** | **51.0%** | **+61.5%** | **1.94** | **0.301** |

Snooped profit: +250% (bias = +120%). Precision is on raw predictions; after cooldown ~2% become actual trades.

### Top Features (Walk-Forward, 4 folds)

| # | Feature | Importance | Stability | Source |
|---|---------|-----------|-----------|--------|
| 1 | vol_body_product_1D | 12.70 | 4/4 | V6 NEW |
| 2 | vol_body_product_3D | 8.94 | 4/4 | V6 NEW |
| 3 | stoch_pos_1D_w10 | 8.79 | 4/4 | V5 KEEP |
| 4 | atr_normalized_1D | 7.58 | 4/4 | V6 NEW |
| 5 | vol_body_product_8H | 4.21 | 4/4 | V6 NEW |
| 6 | vol_body_product_12H | 3.93 | 4/4 | V6 NEW |
| 7 | stoch_pos_3D_w10 | 3.07 | 4/4 | V5 KEEP |
| 8 | vol_ratio_1D | 2.40 | 4/4 | V6 NEW |

16/20 top features in ALL 4 folds. V6 new features hold 6 of top 8. Features used per fold: 99-266 of 395.

### Audit (all PASS)

Test never seen in training, temporal splits with no overlap, labels forward-looking, features backward-only (merge_asof), val-only early stopping, conservative trading sim, no data leakage. Full audit: `model_training/V6_QUANT_ASSESSMENT.md`

### Known Limitations

- Precision @0.70 avg=51% (below 60% target, above 33.3% break-even)
- Fold 2 loses at val-selected threshold (-6%); edge varies ~5x across regimes
- Small trade counts after cooldown (34-160 per fold) limit statistical confidence
- Seed sensitivity not validated (single seed=42)
- Needs: regime filter, position sizing (Kelly on Sharpe=0.30), paper trade 30d, multi-seed

## Signal System (signal_logic.py)

| Class | Purpose |
|-------|---------|
| `SignalLogic` | Reversal detection, GMM clustering, crossing detection, aggregation |
| `PricePredictor` | 2-step linear extrapolation for forward prediction |
| `CalendarDataBuilder` | 55-row x time-column calendar grid with R/C/A scoring |

**Signal scoring** (0-3 points per cell): **R** (+1, 5-point reversal pattern), **C** (+1, angle crossover), **A** (+1, GMM quality filter). Direction: `slope_f > 0` = LONG, `< 0` = SHORT.

**Cascade theory**: Youngs reverse first → Adults follow (median 15min) → Balzaks confirm. 3.1% of Young reversals cascade. Full Y→A→B = 36 in 6 years. Coverage: 7.7% of 5M candles.

## Dashboard (dashboard.py)

**Purpose**: Feature engineering microscope — visually confirms that signal logic produces meaningful patterns before trusting ML. Secondary: live trading signal display. Workflow: tweak signal logic → visually confirm on dashboard → run ML pipeline.

**Key views**: Signal panel, probability gauges, heatmaps (4 types), crossing/reversal signals, acceleration zones, angle chart, calendar table (55-row), timeframe matrix, equity chart. Auto-refresh (5 min).

## ML Pipeline V10 — Current Best (Walk-Forward Validated 2026-02-24)

### Overview

- **508 features** = 395 V6 base + 113 new Phase F (cross-scale convergence)
- Encodes expert's cross-window crossing signal as discrete counts (staircase-safe)
- Uses V7 d8 hyperparameters (depth=8, iterations=5000)
- **First model to achieve 70% precision target** (@0.80 = 70.6% avg walk-forward)

### V10 New Feature Phases (Phase F, 113 features)

| Phase | Features | Count | Description |
|-------|----------|-------|-------------|
| F1 | `xw_crosses_active_{tf}`, `xw_crosses_long_{tf}`, etc. | 77 | Per-TF cross-window crossing counts (0-7) for 7 window pairs |
| F2 | `xtf_total_crosses`, `xtf_cascade_score`, etc. | 15 | Cross-TF composites (sums, groups, cascade weighting) |
| F3 | `corr_velocity_{tf}`, `xtf_corr_agreement` | 12 | Correlation dynamics (rate of change) |
| F4 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_ny_session` | 5 | Temporal cyclical features |
| F5 | `convergence_volume`, `cascade_volume`, etc. | 4 | Volume x convergence interactions |

### Walk-Forward Results (4 folds x 3 seeds = 12 runs)

| Threshold | F0 Profit | F1 Profit | F2 Profit | F3 Profit | Total | Avg Precision | Folds Prof |
|-----------|-----------|-----------|-----------|-----------|-------|---------------|------------|
| @0.70 | +135% | +14% | +18% | +87% | **+253%** | 55.5% | 4/4 |
| @0.75 | +131% | +14% | +29% | +97% | **+271%** | 62.7% | 4/4 |
| @0.80 | +102% | +7% | +23% | +85% | **+217%** | **70.6%** | 4/4 |

AUC: 0.877 +/- 0.021 across 12 runs. ALL 6 pass/fail checks PASS.

### Key Findings

1. **Only temporal features (F4) contribute** — `hour_sin` in top-20 of all 12/12 runs; crossing counts (F1/F2) dead at <1% importance
2. **70% precision achieved** — @0.80 meets expert manual WR target for first time
3. **V10 @0.75 = best validated profit** — +271% beats V6's +246% at 62.7% precision
4. **Fold 2 fixed** — V6's losing fold (-6%) now +23% at @0.80

Full results: `model_training/V10_EXPERIMENT_RESULTS.md`

## SL/TP Target Sweep — C1 Winner (Walk-Forward Validated 2026-02-26)

### Overview

- **10 SL/TP configs screened** (Stage 1: 1 fold, 1 seed), top 3 advanced to full walk-forward
- **C1 (SL=0.5%, TP=1.0%)** = clear winner, 5x V10 baseline profit
- Same V10 508 features, V7 d8 CatBoost params throughout
- Key insight: tighter SL/TP gives model 2x more trade labels (49% NO_TRADE vs 73%), dramatically improving learning

### Walk-Forward Results: C1 (4 folds x 3 seeds = 12 runs)

| Threshold | F0 Profit | F1 Profit | F2 Profit | F3 Profit | Total | Avg Precision | Folds Prof |
|-----------|-----------|-----------|-----------|-----------|-------|---------------|------------|
| @0.70 | +423% | +314% | +287% | +346% | **+1,369%** | 71.1% | 4/4 |
| @0.75 | +433% | +315% | +293% | +340% | **+1,381%** | 74.7% | 4/4 |
| @0.80 | +422% | +300% | +286% | +326% | **+1,333%** | **78.5%** | 4/4 |

AUC: 0.866 +/- 0.015 across 12 runs. ALL 6 pass/fail checks PASS.

### Key Findings

1. **5x profit improvement** — C1 +1,381% vs V10 +271% @0.75, largest gain in project history
2. **Label density drives it** — 49% NO_TRADE (C1) vs 73% (V10) = 2x more trade examples for the model
3. **Precision improved** — 78.5% @0.80 (C1) vs 70.6% (V10), tighter targets don't sacrifice accuracy
4. **Feature shift** — stochastic position + ATR dominate instead of vol_body_product_1D; model learns short-horizon positioning
5. **Fold 1 (chop) fixed** — +315% (C1) vs +14% (V10) @0.75; tight targets excel in choppy markets

Full results: `model_training/SLTP_SWEEP_RESULTS.md`

## Experiment History (V1-V10, SL/TP Sweep)

| Version | Features | AUC | Walk-Forward | Best Result | SHORT? | Status |
|---------|----------|-----|-------------|-------------|--------|--------|
| V1 | 354 | 0.42 | N/A | Anti-predictive | N/A | DEPRECATED |
| V2.1 | 135 | ~0.70 | N/A | +510% @0.60 (no cooldown) | 0 | Baseline |
| V3 | 280 | 0.688 | N/A | +252% @0.42 (ALL LONG) | 0 | LONG-only |
| V4 | 280 | 0.710 | **FAIL** | Dir AUC 0.187→0.957 | 0 | FAILED |
| V5 | 390 | 0.77 | N/A | 27% precision | 0 | Superseded |
| V6 | 395 | 0.870 | PASS (4/4) | +130% honest, +246% @0.70 | All folds | V10 base |
| V6-HP | 395 | 0.833 | Single | +64% (Config#30) | YES | HP Search |
| V7 | 395 | 0.871 | PASS (6/6) | +173% honest, +227% @0.70 | All folds | Multi-seed validation |
| **V10** | **508** | **0.877** | **PASS (12/12)** | **+271% @0.75, 70.6% prec @0.80** | **All folds** | **V10 base** |
| **SLTP-C1** | **508** | **0.866** | **PASS (12/12)** | **+1,381% @0.75, 78.5% prec @0.80** | **All folds** | **NEW BEST** |

### V7 Results (2026-02-23) — Multi-Seed Validation of d8

- depth=8, 3 seeds (42/123/777), 4 folds = 12 runs
- AUC: 0.871 +/- 0.017 (min=0.849, max=0.898)
- Seed totals: [+166%, +178%, +174%], **CV=2.9%** — seed sensitivity RESOLVED
- @0.70: F0=+121, F1=+15, F2=+8, F3=+82 = +227% total (4/4 profitable)
- **Does NOT beat V6 at fixed @0.70** (V6=+246% vs V7=+227%)
- **Beats V6 at 5/6 thresholds** (@0.42-@0.65), wider profitable range (6 vs 3 thresholds)
- **Recommended operating point: V7 d8 @0.65** (+235%, seed-validated, 2x threshold margin)
- Val threshold selection noisy (only 1/4 folds agree across seeds)
- LONG WR systematically worse than SHORT (33% vs 55% in F0)
- Full results: `model_training/V7_EXPERIMENT_RESULTS.md`

### 5 Global Issues — Final Status (2026-02-23)

1. SHORT=0: **SOLVED** (shorts in all 4 WF folds, V6 and V7)
2. Time-period sensitivity: **IMPROVED** (@0.70 all positive, edge varies 5x)
3. Threshold windows: **SOLVED** (@0.70 = 4/4 folds in both V6 and V7)
4. Seed sensitivity: **SOLVED** (V7: CV=2.9% across 3 seeds)
5. Permutation pass/profit fail: **SOLVED** (profit works at @0.70)

### Key Lessons (V1-V10)

- Dropping NO_TRADE rows = model never learns WHEN to trade (V1 fatal flaw → V2+ 3-class)
- Binary reversal (0/1) dead after 5M alignment (<0.5% density) → removed V5+
- TF-native lags: MUST shift on native data BEFORE merge_asof (V2.1 FIX)
- Within-TF convergence = autocorrelated noise (50.1% WR) → only w30+w120 used
- Walk-forward mandatory — single-split can inflate 100x (V6: 13,716% → 130%)
- Cooldown (60 candles/5hr) reveals true independent trade count (97-99% reduction)
- Volume-direction composites dominate all regression-based features
- AUC eval_metric not supported on CatBoost GPU — use MultiClass loss
- Deeper trees (d8) spread feature importance but don't improve @0.70 profit vs d2 (V7 finding)
- Val threshold selection is noisy on small val sets (~8.9K rows, 94% NO_TRADE) — use fixed thresholds
- LONG signals weaker than SHORT across all folds (V7 audit: 33% vs 55% WR)
- Cross-window crossing counts (discrete states) still die in ML — expert visual signal ≠ ML feature (V10 finding)
- Temporal features (hour_sin, dow_sin) are real signal — market microstructure timing matters (V10 finding)
- Higher thresholds (@0.75-@0.80) unlock precision without cliff edges — profit degrades smoothly (V10 finding)
- SL/TP label definition is the #1 hyperparameter — tighter targets (0.5%/1.0%) give 5x more profit than wide (2%/4%) by providing 2x more trade labels for the model to learn from (SL/TP sweep finding)

Full analysis: `model_training/EXPERIMENT_RETROSPECTIVE.md`

### Post-Deployment Audit (2026-03-02)

3 bugs found and fixed during first testnet deployment:

| Bug | Impact | Fix |
|-----|--------|-----|
| **Thread pool leak** in `predict_with_timeout()` | Pool not cleaned up on non-timeout exceptions (ConnectionError, KeyError) → resource leak over hours | `try/except/finally` with `pool.shutdown(wait=False, cancel_futures=True)` in `finally` block |
| **Infinite re-pause loop** in `SafetyMonitor` | After consecutive-loss cooldown expires, same losing trades immediately re-trigger pause → bot permanently paused | `cooldown_grace` flag: skip re-evaluation until next `record_trade()` call |
| **Stale `trading_state.json` references** | `deploy/gce-setup.sh` and `.dockerignore` referenced old flat file path instead of `trading_state/` directory | Updated `touch trading_state.json` → `mkdir -p trading_state`, `.dockerignore` entry → `trading_state/` |

Additional cleanup: removed dead dry-run code block in `executor.py:close_position()` (unreachable after early return), removed `streamlit` from `requirements-docker.txt` (~150MB bloat, not needed in headless bot), added version pins to `requirements-docker.txt` matching `requirements.txt`.

## Legacy Pipelines (Reference Only)

**V2.1** (Audit-verified 2026-02-19): 135 numeric features, 2 windows (w30+w120) x 11 TFs, 3-class CatBoost with permutation test. Best: +510% @0.60 (366 trades, no cooldown). Key features: slope_f_mag_3D, angle_slow_6H. Dead: all reversal_*, all direction_*. Details: `encode_v2.py`, `train_v2.py`

**V1** (Deprecated): 354 categorical features, binary classification, dropped NO_TRADE rows. AUC=0.42 (anti-predictive). Full audit: `model_training/V1_model_story.md`

## Live Prediction Pipeline (live_predict.py)

In-memory pipeline: downloads live klines → runs ETL → encodes 508 features → ensemble predicts with 3 production CatBoost models (seeds 42/123/777). No disk I/O during inference.

### Key Functions

| Function | Purpose |
|----------|---------|
| `download_live_klines()` | 500 bars per TF (5M=1.7d, 1D=1.4yr, 3D=4yr) |
| `download_live_klines_extended(bars_5m)` | Configurable 5M bars (1400=4.9d, 2516=8.7d), other TFs=500 |
| `run_live_etl(klines_dict)` | iterative_regression() for all 55 TF/window combos in memory |
| `encode_live_features(klines_dict, decomposed)` | Monkey-patches V3→V5→V10 encoders to use in-memory data |
| `load_production_models()` | Loads 3 .cbm models from `results_v10/production/` |
| `ensemble_predict(models, row, features, threshold)` | Single-row: avg probabilities across 3 models |
| `batch_ensemble_predict(models, df, features, threshold)` | Batch: 3 calls total for N rows (not 3*N) |
| `run_single_prediction(threshold)` | Full pipeline: download → ETL → encode → predict last row |

### Production Models

- Location: `model_training/results_v10/production/`
- Files: `production_model_s42.cbm`, `production_model_s123.cbm`, `production_model_s777.cbm`, `production_metadata.json`
- Created by: `python model_training/train_v10_production.py`
- Features: 508 (V10), threshold: 0.75 default

## Trading Bot (trading_bot.py)

Connects ML predictions to Binance USDT-M Futures. Runs `live_predict.run_single_prediction()` every 5min, manages position state, enforces safety limits.

### Key Implementation Details

**`predict_with_timeout()`** — Wraps prediction in a `ThreadPoolExecutor` with 120s timeout. Uses `try/except/finally` pattern: the `finally` block calls `pool.shutdown(wait=False, cancel_futures=True)` to prevent thread pool leaks on any exception (timeout, ConnectionError, KeyError, etc.), not just timeout.

**`SafetyMonitor.cooldown_grace`** — Prevents infinite re-pause loop when cooldown expires. Problem: after a consecutive-loss pause expires, `check()` would immediately re-evaluate the same losing trades and re-trigger the pause. Fix: when cooldown expires, sets `cooldown_grace=True` which skips consecutive-loss and rolling-WR re-evaluation until a new trade arrives via `record_trade()` (which clears the grace flag). Persisted in `to_list()`/`load_from_list()` for restart survival.

### Components

| Module | Class | Purpose |
|--------|-------|---------|
| `trading/executor.py` | `BinanceFuturesExecutor` | Order execution (testnet/live), leverage, position sizing |
| `trading/position_manager.py` | `PositionManager` | SL/TP tracking (SL=2%, TP=4%), entry averaging, exchange sync |
| `trading/safety.py` | `SafetyMonitor` | 7-day aggregated WR (min 33.3%), cooldown_grace (prevents re-pause loop) |

### State & Logging

- State persistence: `trading_state/state.json` (position + trade history, survives restart)
- Trade logs: `trading_logs/trades_YYYY-MM-DD.csv` (one file per day)
- Columns: timestamp, signal, confidence, action, side, quantity, price, avg_entry, sl_price, tp_price, order_id, model_agreement, unanimous, latency_sec

### Modes

| Mode | Flag | Description |
|------|------|-------------|
| Dry Run | `--dry-run` | Predict only, no orders, no exchange keys needed |
| Testnet | `--testnet` | Real orders on Binance Futures testnet |
| Live | `--live` | Production trading, requires typing CONFIRM |

### API Key Configuration

The trading bot authenticates via **Windows environment variables** loaded by `BinanceFuturesExecutor._init_client()` in `trading/executor.py`. Keys are set system-wide using `setx` (persist across reboots/sessions).

| Environment Variable | Mode | Description |
|---------------------|------|-------------|
| `BINANCE_TESTNET_KEY` | Testnet | Binance Futures Testnet API key |
| `BINANCE_TESTNET_SECRET` | Testnet | Binance Futures Testnet secret key |
| `BINANCE_KEY` | Live | Binance Futures production API key |
| `BINANCE_SECRET` | Live | Binance Futures production secret key |
| `TELEGRAM_BOT_TOKEN` | All | Telegram bot token from @BotFather |

**Testnet keys active** (set 2026-03-02, verified connected):
- Source: [testnet.binancefuture.com](https://testnet.binancefuture.com)
- Testnet wallet: 5,000 USDT (default allocation)
- Endpoint: `https://testnet.binancefuture.com`

**How keys are loaded** (`trading/executor.py:40-51`):
```python
# Testnet mode
api_key = os.environ.get("BINANCE_TESTNET_KEY", "")
api_secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
# Production mode
api_key = os.environ.get("BINANCE_KEY", "")
api_secret = os.environ.get("BINANCE_SECRET", "")
```

**Setting keys** (run in CMD or PowerShell, requires terminal restart):
```bash
setx BINANCE_TESTNET_KEY "your-testnet-api-key"
setx BINANCE_TESTNET_SECRET "your-testnet-secret-key"
```

**Security notes:**
- Keys are stored as user-level Windows environment variables (not in code or `.env` files)
- Never commit API keys to git — `.gitignore` should exclude `.env` if one is ever created
- Testnet keys carry no financial risk (fake funds)
- Production keys (`BINANCE_KEY`/`BINANCE_SECRET`) should only be set when ready for live trading
- Dry-run mode (`--dry-run`) skips key loading entirely — no credentials needed

## Telegram Monitoring Service (telegram_service/)

### Overview

Read-only monitoring bot for Telegram. Polls shared volumes for changes and pushes alerts to subscribers. No write access to trading data.

- **Bot**: [@tamagochi_trading_bot](https://t.me/tamagochi_trading_bot)
- **Token env var**: `TELEGRAM_BOT_TOKEN` (in `.env`, loaded via `docker-compose.yml` `env_file`)
- **Container**: `tamagochi-telegram` (python:3.11-slim, ~50MB)
- **Dependencies**: `python-telegram-bot>=20.0`, `pandas>=1.5.0`

### Commands (8 total)

| Command | Description |
|---------|-------------|
| `/start` | Subscribe to push notifications |
| `/stop` | Unsubscribe |
| `/status` | Quick one-liner (position, BTC price, last signal) |
| `/stats` | Full hourly dashboard report |
| `/position` | Detailed position with PnL estimate |
| `/trades` | Last 10 trades |
| `/health` | System health check (data service + bot status) |
| `/help` | List all commands |

### Push Notifications (automatic)

| Job | Interval | What |
|-----|----------|------|
| `poll_changes_job` | 60s | Detects new LONG/SHORT signals + new trade events, broadcasts to subscribers |
| `hourly_report_job` | 3600s | Full dashboard: predictions, BTC price, position, safety stats, data service health |

### Data Sources (all read-only)

| Source | Path | What |
|--------|------|------|
| Predictions | `/data/predictions/predictions.csv` | Latest signal, confidence, model agreement |
| Data service health | `/data/status.json` | State, cycle time, last update |
| BTC price | `/data/klines/ml_data_5M.csv` | Last 5M OHLCV candle |
| Trading state | `/app/trading_state/state.json` | Position, trade history, safety status |
| Trade logs | `/app/trading_logs/trades_*.csv` | Daily trade log CSVs |

### Subscriber Persistence

- File: `/app/telegram_data/subscribers.json` (mounted to `./telegram_data/` on host)
- Atomic writes (temp file + `os.replace()`)
- Survives container restarts

### Key Files

| File | Purpose |
|------|---------|
| `telegram_service/bot.py` | `TelegramMonitorBot` class — commands, polling jobs, broadcasting |
| `telegram_service/formatters.py` | HTML message templates for all notification types |
| `telegram_service/readers.py` | Read-only data readers with `configure_paths()` |
| `telegram_service/subscribers.py` | `SubscriberStore` — atomic JSON persistence for chat IDs |
| `telegram_service/service.py` | Entry point, CLI args, path configuration |
| `telegram_service/Dockerfile` | Minimal container (python-telegram-bot + pandas) |
| `telegram_service/requirements.txt` | Container-specific dependencies |

## Backfill & Backtest Dashboard

### Backfill Script (backfill_predictions.py)

One-shot batch predictor: downloads extended klines, runs full ETL+encode pipeline, predicts every 5M candle in the lookback window, simulates SL/TP outcomes.

**Flow (~25s for 1 week):**
1. Load 3 production models
2. `download_live_klines_extended(bars_5m)` — scale bars to cover lookback + warm-up
3. `run_live_etl()` → 55 decomposed datasets
4. `encode_live_features()` → all rows x 508 features
5. Filter to last N hours
6. `batch_ensemble_predict()` at low threshold (0.50) to capture all potential signals
7. Compute actual SL/TP outcomes via `_test_long_trade_fast()`/`_test_short_trade_fast()`
8. Save `raw_signal` column (pre-cooldown) so dashboard can re-apply cooldown dynamically
9. Predictions in last 288 candles (24h) → "Pending" (insufficient forward data)

**Output:** `trading_logs/backfill_predictions.csv` + `trading_logs/backfill_klines_5m.csv`

**CSV schema:** `time, signal, confidence, prob_no_trade, prob_long, prob_short, model_agreement, unanimous, raw_signal, actual_outcome, actual_gain_pct, actual_hold_periods, source`

**actual_outcome values:** `TP_Hit`, `SL_Hit`, `Max_Hold`, `Pending`, `No_Trade`

**Important:** Run with `--threshold 0.50` to capture all signals. The dashboard threshold slider filters dynamically — if you bake in threshold=0.75 at backfill time, signals below 0.75 are lost.

### Backtest Dashboard (backtest_dashboard.py)

Streamlit dashboard for visual verification of V10 predictions against actual price action.

**Layout:**
- **Metrics row**: Trades, Win Rate, Total PnL, LONG WR, SHORT WR, Profit Factor
- **Candlestick chart**: 5M BTCUSDT with color-coded prediction markers
  - Green triangle-up = WIN LONG (TP_Hit)
  - Red triangle-down = WIN SHORT (TP_Hit)
  - Orange hollow triangle = LOSS (SL_Hit)
  - Gray diamond = Max_Hold
  - Yellow triangle = Pending
  - Dashed SL/TP zone lines for last 5 trades
- **Equity curve**: Cumulative PnL for resolved trades
- **Trade table**: Scrollable with color-coded outcomes

**Sidebar controls** (all dynamic, no re-backfill needed):
- Confidence threshold slider (0.50–0.95, default 0.70)
- Cooldown slider (0–120 candles, default 60)
- Show SL/TP zones toggle
- Show pending trades toggle
- Refresh button

**Data sources** (merged + deduped by timestamp):
- `trading_logs/backfill_predictions.csv` (backfill, has raw_signal + actual outcomes)
- `trading_logs/trades_*.csv` (live bot logs, if any)
- Klines: `trading_logs/backfill_klines_5m.csv` (preferred) or `model_training/actual_data/ml_data_5M.csv` (fallback)

**Key design decisions:**
- `raw_signal` column preserves pre-cooldown/pre-threshold signals — dashboard filters dynamically
- Threshold filter restores raw signals first, then applies new threshold
- Cooldown filter restores raw signals first, then applies new cooldown
- Klines are tz-naive throughout (UTC stripped on load) to avoid datetime comparison errors

## Deployment & Operations (Docker → Binance Testnet)

### Architecture

Three Docker services orchestrated by `docker-compose.yml`, deployed to a GCE c3-standard-4 instance:

| Service | Container | Role |
|---------|-----------|------|
| `tamagochi-data` | `data_service/Dockerfile` | Persistent 3-layer pipeline: L1 (klines) → L2 (regression) → L3 (predictions). Writes to shared volume every 5 min. |
| `tamagochi-bot` | `Dockerfile` (root) | Reads predictions from shared volume. Places orders on Binance Futures. Manages SL/TP, position state, safety. |
| `tamagochi-telegram` | `telegram_service/Dockerfile` | Read-only monitoring bot. Push notifications (signals, trades, hourly reports) + 8 interactive commands. |

Shared via Docker named volume `persistent_data` (data service writes, bot + telegram read).

### Current Testnet Config

```
--testnet --data-service --data-dir /data --threshold 0.70 --amount 10 --leverage 20
```

V10 model (SL=2%/TP=4%), 3-seed ensemble (`results_v10/production/`).

### Risk Engine (Implemented 2026-03-04)

Three-layer protection system:

| Layer | Features |
|-------|----------|
| **Network** | 10s API timeout, 3x retry (0/2/5s backoff), client reconnection, circuit breaker (3 errors → exponential backoff, 10min cap) |
| **Position** | SL/TP verification on startup + resync, emergency close (30 attempts blocking loop), partial fill handling, zero-fill guard, max hold aggressive cycle (60s in last 30min) |
| **Safety** | 7-day aggregated WR monitor (33.3% min, requires 3+ trades), cooldown grace (prevents re-pause loop), profit lock trailing stop (3.5% trigger / 3.0% floor) |

Full specification: `model_training/RISK_ENGINE.md`

### Data Service Pipeline

| Layer | What | Storage |
|-------|------|---------|
| L1 | Fetch live klines from Binance (11 TFs) | `/data/klines/*.csv` |
| L2 | Incremental `iterative_regression()` (55 combos) | `/data/decomposed/*.csv` |
| L3 | Encode 508 features → 3 CatBoost ensemble | `/data/predictions/predictions.csv` |

Gap detection ensures incremental updates. Atomic CSV writes prevent corruption. Health check via `/data/status.json`.

### Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service orchestration (3 services + shared volume) |
| `Dockerfile` | Trading bot image (slim: code + production models only) |
| `data_service/Dockerfile` | Data service image |
| `data_service/service.py` | Main loop, SIGTERM handler, cycle orchestration |
| `data_service/layers.py` | 3-layer pipeline (L1/L2/L3) |
| `data_service/csv_io.py` | Atomic CSV read/append |
| `data_service/incremental_etl.py` | Incremental regression (gap-aware) |
| `data_service/gap_detector.py` | TF-aware kline gap detection |
| `telegram_service/bot.py` | Telegram bot: commands, push notifications, polling jobs |
| `telegram_service/formatters.py` | HTML message templates for all notification types |
| `telegram_service/readers.py` | Read-only data readers (predictions, state, trades, health) |
| `telegram_service/subscribers.py` | Persistent subscriber store (atomic JSON writes) |
| `telegram_service/service.py` | Entry point and CLI argument parser |
| `telegram_service/Dockerfile` | Telegram bot image (python-telegram-bot + pandas only) |
| `.env` | API keys (Binance testnet/production + Telegram bot token) |
| `requirements-docker.txt` | Minimal runtime deps (no training packages) |
| `deploy/gce-setup.sh` | GCE instance bootstrap (Docker install, .env template) |

### Quick Commands

```bash
docker compose up -d --build          # Build + start all 3 services
docker compose logs -f                # Follow all logs
docker compose logs --tail=50 tamagochi-bot      # Bot logs
docker compose logs --tail=50 tamagochi-telegram  # Telegram logs
docker compose stop                   # Graceful stop (SIGTERM)
docker compose restart tamagochi-bot  # Restart bot only
docker compose down                   # Stop + remove containers
```

### Post-Deployment Audit (2026-03-02)

3 bugs found and fixed during first testnet deployment — see main Trading Bot section above.

Full operations guide: `model_training/OPS_RUNBOOK.md`

### Docker E2E Test (2026-03-04) — ALL 8 CHECKS PASS

Full end-to-end test of Docker stack on local machine connected to Binance testnet.

**Bugs fixed during testing (2 in `trading_bot.py`):**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **Staleness threshold too tight** | Prediction time = candle OPEN (not close). Normal age 500-700s due to candle duration + processing + cycle offset. Old 600s limit rejected nearly all predictions. | 600s → 1200s (4 candle periods). Catches real outages, allows normal timing drift. |
| **SIGTERM during sleep** | `time.sleep(300)` not interrupted by signal handlers. Bot slept through 60s Docker grace period → SIGKILL without state save. | Replaced all `time.sleep(interval)` with `threading.Event.wait(interval)` + `_shutdown_event.set()` in SIGTERM handler. Shutdown now <1s. |

**Verification results:**

| Check | Status |
|-------|--------|
| `docker compose build` (both images) | PASS |
| `docker compose up -d` (both services start) | PASS |
| Data service L1→L2→L3 cycle (508 features, ~10s) | PASS |
| Bot reads prediction from shared volume (~500s age) | PASS |
| Bot banner shows risk engine params | PASS |
| Graceful shutdown saves state (<1s) | PASS |
| Restart recovers state (timestamp + position) | PASS |
| Data service failure → bot detects stale, no crash, recovers | PASS |

### Production Readiness Audit (2026-03-04) — Known Issues for Live Trading

Full audit across DevOps, Data Engineering, and Quant domains. Issues ranked by real-money impact:

| # | Severity | Domain | Issue | Testnet Impact | Live Impact |
|---|----------|--------|-------|---------------|-------------|
| 1 | **HIGH** | Trading | Naked position window (2-5s between open and SL/TP placement) | None | Critical — flash crash + 20x = wipeout |
| 2 | **HIGH** | Data | Non-atomic CSV append (race between data service write and bot read) | Skip cycle | Wrong signal |
| 3 | **HIGH** | Data | Incomplete candle dedup (stale OHLCV persists after restart) | None | Feature drift over days |
| 4 | **MEDIUM** | Trading | CLOSE_FAILED no escalation (opposite-signal close retries forever) | Annoying | Stuck in losing position |
| 5 | **MEDIUM** | Trading | RESYNC resets entry_time (max-hold timer restarts) | None | Position overstays 24h |
| 6 | **MEDIUM** | Data | No NaN guard on predictions | Unlikely | Wrong trade decisions |
| 7 | **MEDIUM** | Trading | No realized_pnl in trade CSV log | None | Audit trail gap |
| 8 | **LOW** | Trading | Stale prediction → duplicate ADD to position | Size drift | Over-scaling |

**Decision: Skip fixes for testnet deployment. Fix HIGHs before switching to real money.**

### GCE Deployment Status (2026-03-04) — IN PROGRESS

- **VM created**: GCE Debian 12, 4 vCPU, 16GB RAM, Singapore (`asia-southeast1`)
- **Docker installed**: Docker CE + Compose plugin on the VM
- **Pending**: SSH reconnect → copy code → create .env → `docker compose up -d --build`
- **Account**: martinproject.varzhapetyan@gmail.com, $300 free credits
- **Setup script**: `deploy/gce-setup.sh` (Docker install + directory structure)

**Next session steps (in order):**
1. SSH into GCE VM (click SSH in console.cloud.google.com)
2. Test `docker --version` works
3. Clone/copy code to `/opt/tamagochi`
4. Create `.env` with testnet API keys
5. `docker compose up -d --build`
6. Verify first L1→L2→L3 cycle + bot reads prediction
7. Monitor 24-48h of testnet operation

## Code Constraints

- Fallback zero-fill column names MUST match normal-path names exactly
- Label alignment: explicit `sort_values` required (`isin` preserves original order)
- CatBoost GPU: AUC eval_metric not supported — use `MultiClass` loss + `TotalF1` eval
- Labels: SL=2%, TP=4%, max_hold=288 (5M candles = 24h)

## Reference Documentation

- **V7 Cross-Experiment Analysis**: `model_training/V7_CROSS_EXPERIMENT_ANALYSIS.md` (V7 vs V6 robustness comparison — V7 d8 wins 5/6 thresholds)
- **V7 Experiment Results**: `model_training/V7_EXPERIMENT_RESULTS.md` (multi-seed audit, 2026-02-23)
- **V7 HP Search Results**: `model_training/HYPERPARAM_V7_RESULTS.md` (depth exploration, partial)
- **V6 Quant Assessment**: `model_training/V6_QUANT_ASSESSMENT.md`
- **V1 Audit**: `model_training/V1_model_story.md`
- **Experiment Retrospective**: `model_training/EXPERIMENT_RETROSPECTIVE.md`
- **Manual Signal Findings**: `.claude/claude_manual_signal_finding.md`
- **User Story**: `.claude/user story/user_story_signals.md`
- **App Screenshots**: `.claude/app-screen1png.png`, `.claude/app-screen2.png`
- **Architecture/Components**: `.claude/ARCHITECTURE.md`, `.claude/PROJECT_COMPONENTS.md`
- **V10 Experiment Results**: `model_training/V10_EXPERIMENT_RESULTS.md` (cross-scale convergence, walk-forward 70% precision, 2026-02-24)
- **V10 2yr OOS Audit**: `model_training/V10_2YR_OOS_AUDIT.md` (forensic audit, 40/41 PASS, GO LIVE recommendation, 2026-02-25)
- **SL/TP Sweep Results**: `model_training/SLTP_SWEEP_RESULTS.md` (10-config sweep, C1 winner +1,381% @0.75, 2026-02-26)
- **Experiment History Archive**: `model_training/EXPERIMENT_HISTORY.md`
- **Operations Runbook**: `model_training/OPS_RUNBOOK.md` (deployment, monitoring, incident response, troubleshooting, rollback — full ops guide, 2026-03-04)
- **Risk Engine Spec**: `model_training/RISK_ENGINE.md` (API retry, emergency close, circuit breaker, SL/TP verification, partial fills — technical reference, 2026-03-04)
- **GCE Update History**: `deploy/UPDATE_HISTORY.md` (chronological log of all GCE deployment updates, config changes, and deployment steps)
