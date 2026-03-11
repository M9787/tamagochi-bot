# Trading Bot (trading_bot.py)

Runs `live_predict.run_single_prediction()` every 5min, manages positions on Binance USDT-M Futures.

## Components

| Module | Class | Purpose |
|--------|-------|---------|
| `trading/executor.py` | `BinanceFuturesExecutor` | Order execution, leverage, position sizing, balance query |
| `trading/position_manager.py` | `PositionManager` | SL/TP tracking, entry averaging, exchange sync (live/testnet) |
| `trading/multi_trade_manager.py` | `MultiTradeManager` | Independent concurrent paper trades (dry-run only) |
| `trading/safety.py` | `SafetyMonitor` | 7-day aggregated WR (min 33.3%), cooldown_grace |

## Modes

| Mode | Manager | Trades | Description |
|------|---------|--------|-------------|
| `--dry-run` | `MultiTradeManager` | Multiple concurrent | Paper trading: $1000 simulated balance, $10/20x per signal, independent SL/TP per trade |
| `--testnet` | `PositionManager` | Single position | Binance testnet with real order execution |
| `--live` | `PositionManager` | Single position | Production trading on Binance Futures |

## Multi-Trade Mode (dry-run)

Each LONG/SHORT signal opens a **separate independent trade**:
- Margin: $10 per trade, Leverage: 20x → $200 notional exposure per trade
- SL: -2% of entry → -$4 max loss per trade; TP: +4% → +$8 max win per trade
- Max hold: 24h (86400s); Profit lock: trigger 3.5% / floor 3.0%
- **Liquidation guard**: Loss capped at margin ($10). If price gaps beyond -5%, trade is LIQUIDATED, not negative margin
- Margin locking: available_margin = simulated_balance - sum(open_trade.margin). New trades rejected if insufficient
- Close actions: `SL_TRIGGERED`, `TP_TRIGGERED`, `MAX_HOLD_24H`, `PROFIT_LOCK`, `LIQUIDATED`
- Trade IDs: sequential `T0001`, `T0002`, etc. (robust recovery from state on restart)
- CLI: `--starting-balance 1000` (default), `--amount 10` (margin per trade), `--leverage 20`

## State & Logging

- **State**: `trading_state/state.json` -- **3 rotated backups** (`.json.1/.json.2/.json.3`) for crash recovery
- **Multi-trade state format**: `mode: "multi_trade"`, `multi_trade: {open_trades, simulated_balance, ...}`, plus top-level `account_balance` / `cumulative_pnl_usdt` for telegram compatibility
- **Single-position state format**: `position`, `trade_history`, `account_balance`, `cumulative_pnl_usdt`
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
- **Staleness threshold**: `STALENESS_THRESHOLD_SEC` from config (1200s). Validated via `core.data_validation.validate_predictions_freshness()`
- **Prediction row validation**: `validate_predictions_row()` rejects NaN/inf probs, prob sum != 1.0, invalid signal values
- **Entry/SL/TP display**: Console shows entry price + SL (-2%) + TP (+4%) in BTC when signal fires

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
