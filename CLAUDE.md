# CLAUDE.md

## Project Overview

Cryptocurrency trading signal analysis system for BTCUSDT that:
1. Downloads kline data from Binance across **11 timeframes** (6 years of history)
2. Performs rolling linear regression with **5 window sizes** (30, 60, 100, 120, 160) — 55 signal sources
3. Displays results in a **Streamlit dashboard** for visual feature engineering confirmation
4. **ML Pipeline V10** (current): V6 base (395 features) + 113 cross-scale convergence features = 508 total → CatBoost GPU 3-class → walk-forward validation
5. **Live trading** on Binance Futures via Docker (3 containers) on GCE, monitored via Telegram bot

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

> **No model training without full performance evaluation. EVER.**

## Core Algorithm (analysis.py)

`iterative_regression(df, window_size)` is the **atom** of the entire system. Sliding window of `2 * window_size` prices using `sqrt(price)`, `scipy.stats.linregress` on each half.

| Output | What it captures |
|--------|-----------------|
| `slope_b` | Backward regression slope (historical trend) |
| `slope_f` | Forward regression slope (current direction vector) |
| `angle` | Angle between backward and forward slopes (trend divergence) |
| `acceleration` | First difference of angle (rate of change of divergence) |

## Signal Theory (Expert-Validated)

| Component | Role | Priority |
|-----------|------|----------|
| **Crossing (C)** | PREDICTS | Primary trigger — angle convergence/cross/divergence event |
| **Reversal (R)** | CONFIRMS | Secondary — 5-point angle pattern validates crossing |
| **Direction (slope_f)** | ALIGNS | Context — LONG or SHORT territory |
| **Acceleration (A)** | FILTERS | Quality gate — reject CLOSE/VERY_CLOSE GMM zones |

**Crossing Physics**: Angle = ball trajectory with momentum. Young ball falling + Elder rising at cross = LONG. Elder falling + Young rising = SHORT. DIVERGENCE after crossing is the signal. Higher TFs = longer delay + bigger move. Lags must be TF-native (shift BEFORE merge_asof).

**Trading Philosophy**: Precision >> Recall. Target **60%+ precision** with 4% TP / 2% SL (break-even = 33.3%). Only ~10% of market moments are tradeable. Labels: NO_TRADE=72.7%, SHORT=13.7%, LONG=13.6%.

## Architecture

```
main.py                        # CLI entry point
dashboard.py                   # Streamlit dashboard (feature engineering microscope)
backtest_dashboard.py          # Streamlit backtest dashboard (predictions vs actuals)
backfill_predictions.py        # Batch-predict N hours of 5M candles + SL/TP outcomes
trading_bot.py                 # V10 trading bot (dry-run / testnet / live)
core/
  config.py                    # Centralized config (paths, TFs, windows, thresholds)
  analysis.py                  # iterative_regression, calculate_acceleration
  processor.py                 # TimeframeProcessor: 55 TF/window combos
  signal_logic.py              # SignalLogic, PricePredictor, CalendarDataBuilder
data/
  downloader.py                # Binance data downloader
  target_labeling.py           # SL/TP target labeling encoder
trading/
  executor.py                  # BinanceFuturesExecutor (order execution + balance query)
  position_manager.py          # PositionManager (SL/TP tracking, entry averaging)
  safety.py                    # SafetyMonitor (7-day aggregated WR pause)
telegram_service/
  bot.py                       # TelegramMonitorBot (11 commands + push notifications)
  formatters.py                # HTML message templates (status, PnL, equity, etc.)
  readers.py                   # Read-only data readers (predictions, state, trades, PnL)
  subscribers.py               # Persistent subscriber store (atomic JSON)
  service.py                   # Entry point and CLI args
  Dockerfile                   # Minimal container image
data_service/
  service.py                   # Main loop, SIGTERM handler, cycle orchestration
  layers.py                    # 3-layer pipeline (L1/L2/L3)
  csv_io.py                    # Atomic CSV read/append
  incremental_etl.py           # Incremental regression (gap-aware)
  gap_detector.py              # TF-aware kline gap detection
  Dockerfile                   # Data service image
model_training/
  live_predict.py              # Live prediction pipeline (single + batch)
  train_v10_production.py      # Train production models (3 seeds)
  download_data.py             # Step 1: 6yr data → actual_data/
  etl.py                       # Step 2: klines → regression → decomposed_data/
  encode_v6.py                 # Step 3: 395 features → feature_matrix_v6.parquet
  encode_v10.py                # Step 3b: 508 features → feature_matrix_v10.parquet
  build_labels.py              # SL/TP label generation (shared)
  results_v10/production/      # 3 production CatBoost models (.cbm) + metadata
trading_logs/                  # Runtime: backfill CSVs + daily trade logs
trading_state/                 # Runtime: state.json (position + trade history + balance)
deploy/                        # GCE setup script + update history
```

### Key Configuration (config.py)

- `TIMEFRAME_ORDER`: `3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M`
- `WINDOW_SIZES`: `[30, 60, 100, 120, 160]` mapping to `df, df1, df2, df3, df4`
- 7 crossing pairs: `(30,60), (30,100), (60,100), (60,120), (100,120), (100,160), (120,160)`
- TF groups: **Youngs** (5M,15M,30M) → **Adults** (1H,2H,4H) → **Balzaks** (6H,8H,12H) → **Grans** (1D,3D)

## Commands

```bash
# ML Pipeline (V10)
python model_training/download_data.py                 # 1. Download 6yr data → actual_data/
python model_training/etl.py --start 2020-01-01 --end 2026-02-15 [--force]  # 2. ETL
python model_training/encode_v10.py [--force]          # 3. Encode → 508 features
python model_training/train_v10_walkforward.py         # 4. Walk-forward validation
python model_training/train_v10_production.py          # 5. Train production models (3 seeds)

# Live Prediction & Trading Bot
python model_training/live_predict.py --threshold 0.75        # Single prediction
python model_training/live_predict.py --loop --interval 300   # Continuous (5min)
python trading_bot.py --dry-run                               # Predict only, no orders
python trading_bot.py --testnet                               # Binance Futures testnet
python trading_bot.py --live                                  # Production (requires CONFIRM)

# Backfill & Backtest Dashboard
python backfill_predictions.py --hours 168 --threshold 0.50   # 1 week backfill
python -m streamlit run backtest_dashboard.py                  # Backtest dashboard

# Feature Engineering Dashboard
python data/downloader.py                              # Live data for dashboard
python main.py --mode dashboard                        # Streamlit dashboard
```

## ML Pipeline V10 — Production Model

### Overview

- **508 features** = 395 V6 base + 113 Phase F (cross-scale convergence + temporal)
- **3-class** (NO_TRADE/LONG/SHORT), CatBoost GPU, 3-seed ensemble (seeds 42/123/777)
- **First model to achieve 70% precision target** (@0.80 = 70.6% avg walk-forward)
- Labels: SL/TP trade outcome simulation on 5M candles (SL=2%, TP=4%, max_hold=288 candles/24h)

### CatBoost Hyperparameters (V7 d8)

```python
iterations=5000, depth=8, learning_rate=0.02, l2_leaf_reg=15,
min_data_in_leaf=500, loss_function='MultiClass',
eval_metric='TotalF1:average=Macro;use_weights=false',
early_stopping_rounds=600, random_strength=1, border_count=254,
subsample=0.7, bootstrap_type='Bernoulli', task_type='GPU',
class_weights=[0.5, 2.0, 2.0]
```

### Walk-Forward Architecture

4-fold expanding window with embargo (7d), cooldown (60 candles/5hr), val-threshold selection:

| Fold | Train End | Val End | Embargo End | Test End | Regime |
|------|-----------|---------|-------------|----------|--------|
| 0 | 2025-01-01 | 2025-02-01 | 2025-02-08 | 2025-05-01 | Trending |
| 1 | 2025-04-01 | 2025-05-01 | 2025-05-08 | 2025-08-01 | Heavy chop |
| 2 | 2025-07-01 | 2025-08-01 | 2025-08-08 | 2025-11-01 | Mild chop |
| 3 | 2025-10-01 | 2025-11-01 | 2025-11-08 | 2026-01-15 | Mixed |

### V10 Walk-Forward Results (4 folds x 3 seeds = 12 runs)

| Threshold | Total Profit | Avg Precision | Folds Profitable |
|-----------|-------------|---------------|------------------|
| @0.70 | **+253%** | 55.5% | 4/4 |
| @0.75 | **+271%** | 62.7% | 4/4 |
| @0.80 | **+217%** | **70.6%** | 4/4 |

AUC: 0.877 +/- 0.021. ALL 6 pass/fail checks PASS. Full results: `model_training/V10_EXPERIMENT_RESULTS.md`

### SL/TP Sweep Winner: C1 (SL=0.5%, TP=1.0%)

5x V10 profit: +1,381% @0.75, 78.5% precision @0.80. Tighter targets give 2x more trade labels → better learning. Full results: `model_training/SLTP_SWEEP_RESULTS.md`

### Production Models

- Location: `model_training/results_v10/production/`
- Files: `production_model_s{42,123,777}.cbm` + `production_metadata.json`
- Features: 508 (V10), default threshold: 0.75
- Live pipeline (`live_predict.py`): download klines → ETL → encode 508 features → ensemble predict

## Trading Bot (trading_bot.py)

Runs `live_predict.run_single_prediction()` every 5min, manages positions on Binance USDT-M Futures.

### Components

| Module | Class | Purpose |
|--------|-------|---------|
| `trading/executor.py` | `BinanceFuturesExecutor` | Order execution, leverage, position sizing, balance query |
| `trading/position_manager.py` | `PositionManager` | SL/TP tracking (SL=2%, TP=4%), entry averaging, exchange sync |
| `trading/safety.py` | `SafetyMonitor` | 7-day aggregated WR (min 33.3%), cooldown_grace |

### State & Logging

- **State**: `trading_state/state.json` — position, trade history, account_balance, cumulative_pnl_usdt
- **Trade logs**: `trading_logs/trades_YYYY-MM-DD.csv` (one file per day)
- **CSV columns** (17): timestamp, signal, confidence, action, side, quantity, price, avg_entry, sl_price, tp_price, order_id, model_agreement, unanimous, latency_sec, realized_pnl_pct, realized_pnl_usdt, balance_after

### PnL Tracking

Realized PnL computed on all 3 close paths (opposite-signal, SL/TP triggered, force close). Balance queried from Binance API after each close. Cumulative PnL and account balance persisted in state.json.

### Risk Engine

| Layer | Features |
|-------|----------|
| **Network** | 10s timeout, 3x retry, circuit breaker (3 errors → exponential backoff, 10min cap) |
| **Position** | SL/TP placed atomically after open, verification on startup + resync, emergency close (30 attempts) |
| **Safety** | 7-day WR monitor (33.3% min), cooldown grace, profit lock trailing stop (3.5%/3.0%) |

Full spec: `model_training/RISK_ENGINE.md`

### Key Implementation Details

- **`predict_with_timeout()`**: ThreadPoolExecutor with 120s timeout, `pool.shutdown()` in `finally` block prevents leaks
- **`SafetyMonitor.cooldown_grace`**: Prevents infinite re-pause after cooldown expires. `record_trade()` clears grace flag
- **SIGTERM handling**: `threading.Event.wait()` replaces `time.sleep()` for <1s shutdown
- **Staleness threshold**: 1200s (prediction time = candle OPEN, normal age 500-700s)

### API Keys

| Variable | Mode | Description |
|----------|------|-------------|
| `BINANCE_TESTNET_KEY` / `_SECRET` | Testnet | Binance Futures testnet |
| `BINANCE_KEY` / `_SECRET` | Live | Production trading |
| `TELEGRAM_BOT_TOKEN` | All | Telegram bot from @BotFather |

Local: Windows env vars via `setx`. GCE: `.env` file at `/opt/tamagochi/.env`, loaded by `docker-compose.yml`.

## Telegram Monitoring Bot (telegram_service/)

Read-only monitoring bot: [@tamagochi_trading_bot](https://t.me/tamagochi_trading_bot)

### Commands (11 total)

| Command | Description |
|---------|-------------|
| `/start` | Subscribe to push notifications |
| `/stop` | Unsubscribe |
| `/status` | Quick one-liner (position, BTC price, last signal) |
| `/stats` | Full hourly dashboard report |
| `/position` | Detailed position with PnL estimate |
| `/balance` | Account balance and cumulative PnL |
| `/pnl` | PnL summary (today / 7d / 30d / all-time) |
| `/equity` | ASCII equity curve (last 20 trades) |
| `/trades` | Last 10 trades with PnL |
| `/health` | System health check |
| `/help` | List all commands |

### Push Notifications

| Job | Interval | What |
|-----|----------|------|
| `poll_changes_job` | 60s | New LONG/SHORT signals + trade events (with PnL) |
| `hourly_report_job` | 3600s | Full dashboard: predictions, BTC, position, balance, safety, health |

### Data Sources (all read-only via shared Docker volumes)

| Source | Path |
|--------|------|
| Predictions | `/data/predictions/predictions.csv` |
| BTC price | `/data/klines/ml_data_5M.csv` |
| Data service health | `/data/status.json` |
| Trading state + balance | `/app/trading_state/state.json` |
| Trade logs | `/app/trading_logs/trades_*.csv` |

Subscribers persisted at `/app/telegram_data/subscribers.json` (atomic writes).

## Backfill & Backtest Dashboard

**Backfill** (`backfill_predictions.py`): Downloads extended klines, runs ETL+encode, predicts every 5M candle, simulates SL/TP outcomes. Run with `--threshold 0.50` to capture all signals (dashboard filters dynamically).

**Backtest Dashboard** (`backtest_dashboard.py`): Streamlit app with candlestick chart, color-coded prediction markers, equity curve, trade table. Sidebar: threshold slider, cooldown slider, SL/TP zone toggle.

**Output files**: `trading_logs/backfill_predictions.csv`, `trading_logs/backfill_klines_5m.csv`

## Deployment & Operations

### GCE Instance (RUNNING)

- **VM**: instance-20260303-232149, c3-standard-4, asia-southeast1-b
- **Project**: project-4d1ee130-e5dc-4495-a47
- **Account**: martinproject.varzhapetyan@gmail.com
- **Code**: `/opt/tamagochi` (cloned from GitHub)
- **Config**: V10, threshold=0.70, $10, 20x leverage, SL=2%/TP=4%

### Docker Architecture

3 services in `docker-compose.yml`, shared via Docker named volume `persistent_data`:

| Service | Container | Role |
|---------|-----------|------|
| `tamagochi-data` | `data_service/Dockerfile` | L1 (klines) → L2 (regression) → L3 (predictions), every 5 min |
| `tamagochi-bot` | `Dockerfile` (root) | Reads predictions, places orders, manages SL/TP + safety |
| `tamagochi-telegram` | `telegram_service/Dockerfile` | 11 commands + push notifications (read-only) |

### GitHub Repository

- **URL**: https://github.com/M9787/tamagochi-bot.git (private, master branch)
- **Workflow**: Local edit → push to GitHub → GCE pull + rebuild

### Deployment Commands (gcloud CLI)

```bash
# SSH into GCE VM
gcloud compute ssh instance-20260303-232149 --zone=asia-southeast1-b --project=project-4d1ee130-e5dc-4495-a47

# Deploy latest code (run on GCE or via gcloud ssh --command)
cd /opt/tamagochi && git pull origin master && docker compose up -d --build

# Remote deploy from local machine (one-liner)
gcloud compute ssh instance-20260303-232149 --zone=asia-southeast1-b --project=project-4d1ee130-e5dc-4495-a47 --command="cd /opt/tamagochi && git pull origin master && docker compose up -d --build"

# View logs
gcloud compute ssh instance-20260303-232149 --zone=asia-southeast1-b --project=project-4d1ee130-e5dc-4495-a47 --command="cd /opt/tamagochi && docker compose logs --tail=50 tamagochi-bot"
```

### Docker Commands (run on GCE)

```bash
docker compose up -d --build          # Build + start all 3 services
docker compose logs -f                # Follow all logs
docker compose logs --tail=50 tamagochi-bot      # Bot logs
docker compose logs --tail=50 tamagochi-telegram  # Telegram logs
docker compose stop                   # Graceful stop (SIGTERM)
docker compose restart tamagochi-bot  # Restart bot only
docker compose down                   # Stop + remove containers
```

### Git Commands

```bash
git add <files> && git commit -m "message" && git push origin master  # Push changes
git pull origin master                                                 # Pull on GCE
```

### Environment Variables (.env on GCE)

```
BINANCE_TESTNET_KEY=...
BINANCE_TESTNET_SECRET=...
BINANCE_KEY=...
BINANCE_SECRET=...
TELEGRAM_BOT_TOKEN=...
```

Located at `/opt/tamagochi/.env`, loaded via `docker-compose.yml` `env_file`. Never commit to git.

### Production Readiness — Known Issues

All 3 HIGH issues FIXED (atomic SL/TP, atomic CSV, candle dedup). Remaining MEDIUMs:
- CLOSE_FAILED no escalation (opposite-signal close retries forever)
- RESYNC resets entry_time (max-hold timer restarts)

Operations guide: `model_training/OPS_RUNBOOK.md`

## Experiment History

| Version | Features | AUC | Walk-Forward | Best Result | Status |
|---------|----------|-----|-------------|-------------|--------|
| V1 | 354 | 0.42 | N/A | Anti-predictive | DEPRECATED |
| V2.1 | 135 | ~0.70 | N/A | +510% @0.60 | Baseline |
| V6 | 395 | 0.870 | PASS (4/4) | +246% @0.70 | V10 base |
| V7 | 395 | 0.871 | PASS (6/6) | +227% @0.70 | Multi-seed validation |
| **V10** | **508** | **0.877** | **PASS (12/12)** | **+271% @0.75** | **PRODUCTION** |
| **SLTP-C1** | **508** | **0.866** | **PASS (12/12)** | **+1,381% @0.75** | **NEW BEST** |

### Key Lessons (V1-V10)

1. Dropping NO_TRADE rows = model never learns WHEN to trade (V1 fatal flaw)
2. Walk-forward mandatory — single-split inflates 100x (V6: 13,716% → 130%)
3. Cooldown (60 candles/5hr) reveals true trade count (97-99% reduction)
4. Volume-direction composites dominate all regression features (vol_body_product_1D = #1)
5. TF-native lags: MUST shift on native data BEFORE merge_asof
6. Temporal features (hour_sin) = real signal; crossing counts die in ML
7. Val threshold selection noisy on small val sets — use fixed thresholds
8. SL/TP label definition = #1 hyperparameter (tight 0.5/1.0 → 5x profit vs wide 2/4)

Full analysis: `model_training/EXPERIMENT_RETROSPECTIVE.md`

## Code Constraints

- Fallback zero-fill column names MUST match normal-path names exactly
- Label alignment: explicit `sort_values` required (`isin` preserves original order)
- CatBoost GPU: AUC eval_metric not supported — use `MultiClass` loss + `TotalF1` eval
- Labels: SL=2%, TP=4%, max_hold=288 (5M candles = 24h)

## Reference Documentation

| Document | Path |
|----------|------|
| V10 Walk-Forward Results | `model_training/V10_EXPERIMENT_RESULTS.md` |
| V10 2yr OOS Audit | `model_training/V10_2YR_OOS_AUDIT.md` |
| SL/TP Sweep Results | `model_training/SLTP_SWEEP_RESULTS.md` |
| V6 Quant Assessment | `model_training/V6_QUANT_ASSESSMENT.md` |
| V7 Experiment Results | `model_training/V7_EXPERIMENT_RESULTS.md` |
| V7 Cross-Experiment Analysis | `model_training/V7_CROSS_EXPERIMENT_ANALYSIS.md` |
| Experiment Retrospective | `model_training/EXPERIMENT_RETROSPECTIVE.md` |
| Experiment History Archive | `model_training/EXPERIMENT_HISTORY.md` |
| Risk Engine Spec | `model_training/RISK_ENGINE.md` |
| Operations Runbook | `model_training/OPS_RUNBOOK.md` |
| GCE Update History | `deploy/UPDATE_HISTORY.md` |
| Signal Findings | `.claude/claude_manual_signal_finding.md` |
| V1 Audit | `model_training/V1_model_story.md` |
