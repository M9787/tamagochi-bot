# Architecture

## File Tree

```
main.py                        # CLI entry point
dashboard.py                   # Streamlit dashboard (feature engineering microscope)
backtest_dashboard.py          # Streamlit backtest dashboard (predictions vs actuals)
backfill_predictions.py        # Batch-predict N hours of 5M candles + SL/TP outcomes
trading_bot.py                 # V10 trading bot (dry-run / testnet / live)
core/
  config.py                    # Centralized config (paths, TFs, windows, thresholds, staleness)
  analysis.py                  # iterative_regression, calculate_acceleration
  data_validation.py           # Data governance guards (prob sum, feature shape, kline gaps, freshness)
  processor.py                 # TimeframeProcessor: 55 TF/window combos
  signal_logic.py              # SignalLogic, PricePredictor, CalendarDataBuilder
  structured_log.py            # JSONL logging: JsonFormatter + RotatingFileHandler + structured events
data/
  downloader.py                # Binance data downloader
  target_labeling.py           # SL/TP target labeling encoder
trading/
  executor.py                  # BinanceFuturesExecutor (order execution + balance query)
  position_manager.py          # PositionManager (single-position: SL/TP, entry averaging)
  multi_trade_manager.py       # MultiTradeManager (multi-trade: concurrent $10/20x paper trades)
  safety.py                    # SafetyMonitor (7-day aggregated WR pause)
telegram_service/
  bot.py                       # TelegramMonitorBot (13 commands + push notifications)
  formatters.py                # HTML message templates (status, PnL, equity, etc.)
  readers.py                   # Read-only data readers (predictions, state, trades, PnL)
  subscribers.py               # Persistent subscriber store (atomic JSON)
  service.py                   # Entry point and CLI args
  Dockerfile                   # Minimal container image
data_service/
  service.py                   # Main loop, SIGTERM handler, cycle orchestration
  layers.py                    # 3-layer pipeline (L1/L2/L3), dual-mode L3 (incremental/batch)
  csv_io.py                    # Atomic CSV read/append
  incremental_etl.py           # Incremental regression (gap-aware)
  incremental_encoder.py       # Stateful V10 feature encoder (1 row per cycle from state)
  state_initializer.py         # Extract encoder state from feature matrix + raw data
  gap_detector.py              # TF-aware kline gap detection
  Dockerfile                   # Data service image
backfill_features.py           # One-time gap backfill (feature matrix -> today)
model_training/
  live_predict.py              # Live prediction pipeline (single + batch)
  train_v10_production.py      # Train production models (3 seeds)
  download_data.py             # Step 1: 6yr data -> actual_data/
  etl.py                       # Step 2: klines -> regression -> decomposed_data/
  encode_v6.py                 # Step 3: 395 features -> feature_matrix_v6.parquet
  encode_v10.py                # Step 3b: 508 features -> feature_matrix_v10.parquet
  build_labels.py              # SL/TP label generation (shared)
  results_v10/production/      # 3 production CatBoost models (.cbm) + metadata
dashboard/
  Dockerfile                   # Streamlit dashboard container (port 8501)
requirements-dashboard.txt     # Dashboard Python dependencies
trading_logs/                  # Runtime: backfill CSVs + daily trade logs
trading_state/                 # Runtime: state.json (position + trade history + balance)
logs/bot/                      # Runtime: JSONL structured logs (trading_bot.jsonl)
deploy/                        # GCE setup script + update history
```

## Key Configuration (config.py -- single source of truth)

- `TIMEFRAME_ORDER`: `3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M`
- `WINDOW_SIZES`: `[30, 60, 100, 120, 160]` mapping to `df, df1, df2, df3, df4`
- `TRADING_SL_PCT=2.0`, `TRADING_TP_PCT=4.0`, `TRADING_MAX_HOLD_CANDLES=288`
- `DEFAULT_THRESHOLD=0.75`, `BOOTSTRAP_BARS=1400`, `STALENESS_THRESHOLD_SEC=1200`
- 7 crossing pairs: `(30,60), (30,100), (60,100), (60,120), (100,120), (100,160), (120,160)`
- TF groups: **Youngs** (5M,15M,30M) -> **Adults** (1H,2H,4H) -> **Balzaks** (6H,8H,12H) -> **Grans** (1D,3D)
