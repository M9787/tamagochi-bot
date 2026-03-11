# Telegram Monitoring Bot (telegram_service/)

Read-only monitoring bot: [@tamagochi_trading_bot](https://t.me/tamagochi_trading_bot)

## Commands (13 total)

| Command | Description |
|---------|-------------|
| `/start` | Subscribe to push notifications |
| `/stop` | Unsubscribe |
| `/status` | Quick one-liner (open trades/balance in multi-trade, position in single) |
| `/stats` | Full hourly dashboard report (adapts to multi-trade mode) |
| `/position` | All open trades with individual PnL (multi-trade) or single position detail |
| `/balance` | Simulated balance, margin info, SL/TP $ amounts (multi-trade) or account balance |
| `/pnl` | PnL summary (today / 7d / 30d / all-time) |
| `/equity` | ASCII equity curve (last 20 trades) |
| `/trades` | Last 10 trades with PnL |
| `/history` | Paginated signal history with summary stats (reads live `predictions.csv` + trade logs) |
| `/threshold` | View or set per-subscriber signal threshold |
| `/health` | System health check |
| `/help` | List all commands |

## Push Notifications

| Job | Interval | What |
|-----|----------|------|
| `poll_changes_job` | 60s | New LONG/SHORT signals (with entry price, SL -2%, TP +4%) + trade events (with PnL). **Staleness guard**: skips signal alerts when prediction age > `STALENESS_THRESHOLD_SEC` (1200s), but always broadcasts trade events |
| `hourly_report_job` | 3600s | Full dashboard: predictions, BTC, position, balance, safety, health |

## Data Sources (all read-only via shared Docker volumes)

| Source | Path |
|--------|------|
| Predictions | `/data/predictions/predictions.csv` |
| BTC price | `/data/klines/ml_data_5M.csv` |
| Data service health | `/data/status.json` |
| Trading state + balance | `/app/trading_state/state.json` |
| Trade logs | `/app/trading_logs/trades_*.csv` |

Subscribers persisted at `/app/telegram_data/subscribers.json` (atomic writes).

## Multi-Trade Mode Support

When `state.json` has `mode: "multi_trade"`, all formatting functions adapt:
- `/status`: Shows open trade count + simulated balance instead of position side
- `/position`: Lists all open trades with individual entry/SL/TP/unrealized PnL
- `/balance`: Shows simulated balance, starting balance, available margin, margin per trade, SL/TP in $ terms
- `/stats` (hourly report): Shows open trade count and locked margin
- `/health`: Displays multi-trade mode indicator, open count, balance
- Signal alerts: Entry price + SL/TP calculated from config (`TRADING_SL_PCT`/`TRADING_TP_PCT`)
- Close action emojis: `SL_TRIGGERED` (🔴), `TP_TRIGGERED` (🟢), `LIQUIDATED` (💀), `MAX_HOLD_24H` (⏰)
