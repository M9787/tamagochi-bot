# Trading Bot (trading_bot.py)

Runs `live_predict.run_single_prediction()` every 5min, manages positions on Binance USDT-M Futures.

## Components

| Module | Class | Purpose |
|--------|-------|---------|
| `trading/executor.py` | `BinanceFuturesExecutor` | Order execution, leverage, position sizing, balance query |
| `trading/position_manager.py` | `PositionManager` | SL/TP tracking (SL=2%, TP=4%), entry averaging, exchange sync |
| `trading/safety.py` | `SafetyMonitor` | 7-day aggregated WR (min 33.3%), cooldown_grace |

## State & Logging

- **State**: `trading_state/state.json` -- position, trade history, account_balance, cumulative_pnl_usdt
- **Trade logs**: `trading_logs/trades_YYYY-MM-DD.csv` (one file per day)
- **CSV columns** (17): timestamp, signal, confidence, action, side, quantity, price, avg_entry, sl_price, tp_price, order_id, model_agreement, unanimous, latency_sec, realized_pnl_pct, realized_pnl_usdt, balance_after

## PnL Tracking

Realized PnL computed on all 3 close paths (opposite-signal, SL/TP triggered, force close). Balance queried from Binance API after each close. Cumulative PnL and account balance persisted in state.json.

## Risk Engine

| Layer | Features |
|-------|----------|
| **Network** | 10s timeout, 3x retry, circuit breaker (3 errors -> exponential backoff, 10min cap) |
| **Position** | SL/TP placed atomically after open, verification on startup + resync, emergency close (30 attempts) |
| **Safety** | 7-day WR monitor (33.3% min), cooldown grace, profit lock trailing stop (3.5%/3.0%) |

Full spec: `model_training/RISK_ENGINE.md`

## Key Implementation Details

- **`predict_with_timeout()`**: ThreadPoolExecutor with 120s timeout, `pool.shutdown()` in `finally` block prevents leaks
- **`SafetyMonitor.cooldown_grace`**: Prevents infinite re-pause after cooldown expires. `record_trade()` clears grace flag
- **SIGTERM handling**: `threading.Event.wait()` replaces `time.sleep()` for <1s shutdown
- **Staleness threshold**: 1200s (prediction time = candle OPEN, normal age 500-700s)

## API Keys

| Variable | Mode | Description |
|----------|------|-------------|
| `BINANCE_TESTNET_KEY` / `_SECRET` | Testnet | Binance Futures testnet |
| `BINANCE_KEY` / `_SECRET` | Live | Production trading |
| `TELEGRAM_BOT_TOKEN` | All | Telegram bot from @BotFather |

Local: Windows env vars via `setx`. GCE: `.env` file at `/opt/tamagochi/.env`, loaded by `docker-compose.yml`.

## Production Readiness

All 3 HIGH issues FIXED (atomic SL/TP, atomic CSV, candle dedup). Remaining MEDIUMs:
- CLOSE_FAILED no escalation (opposite-signal close retries forever)
- RESYNC resets entry_time (max-hold timer restarts)

Operations guide: `model_training/OPS_RUNBOOK.md`
