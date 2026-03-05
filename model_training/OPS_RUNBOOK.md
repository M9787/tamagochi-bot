# V10 Trading Bot — Operations Runbook

> Authoritative guide for deploying, operating, and troubleshooting the V10 BTCUSDT trading bot on Binance Futures testnet/production.

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [First Deployment](#3-first-deployment)
4. [Configuration Reference](#4-configuration-reference)
5. [Monitoring & Health Checks](#5-monitoring--health-checks)
6. [Risk Engine](#6-risk-engine)
7. [Incident Response](#7-incident-response)
8. [Troubleshooting](#8-troubleshooting)
9. [Maintenance](#9-maintenance)
10. [Rollback Procedures](#10-rollback-procedures)

---

## 1. Architecture Overview

### Docker Services

```
┌───────────────────────────────────────────────────────┐
│  docker-compose.yml                                   │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │  tamagochi-data (data_service.service)          │  │
│  │  - Fetches live klines from Binance (L1)        │  │
│  │  - Runs incremental regression (L2)             │  │
│  │  - Encodes 508 features + ensemble predict (L3) │  │
│  │  - Writes predictions.csv + status.json         │  │
│  │  - Cycle: every 300s (5 min)                    │  │
│  └──────────────┬──────────────────────────────────┘  │
│                 │ persistent_data volume (rw)          │
│  ┌──────────────▼──────────────────────────────────┐  │
│  │  tamagochi-bot (trading_bot.py)                 │  │
│  │  - Reads predictions from shared volume (ro)    │  │
│  │  - Places orders on Binance Futures             │  │
│  │  - Manages SL/TP, position state, safety        │  │
│  │  - Risk engine: retry, circuit breaker, etc.    │  │
│  │  - Writes: trading_logs/, trading_state/        │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

### Data Flow

1. **L1 (Klines)**: Binance REST API → `/data/klines/*.csv` (11 TFs, incremental append)
2. **L2 (Decomposed)**: `iterative_regression()` → `/data/decomposed/*.csv` (55 TF/window combos)
3. **L3 (Predictions)**: 508 features → 3 CatBoost models → `/data/predictions/predictions.csv`
4. **Bot**: reads `predictions.csv` → applies threshold → places orders → logs to `trading_logs/`

### Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service orchestration (data + bot) |
| `Dockerfile` | Trading bot container image |
| `data_service/Dockerfile` | Data service container image |
| `.env` | API keys (Binance testnet/production) |
| `requirements-docker.txt` | Minimal runtime dependencies |
| `deploy/gce-setup.sh` | GCE instance bootstrap script |

---

## 2. Prerequisites

### Infrastructure

| Component | Spec | Notes |
|-----------|------|-------|
| **VM** | GCE e2-small (2 vCPU, 2GB RAM) | ~$15/mo, sufficient for both containers |
| **Docker** | Docker Engine + Compose plugin | `deploy/gce-setup.sh` installs both |
| **Disk** | 20GB standard | Klines grow ~100MB/year |
| **Network** | Outbound HTTPS to Binance API | No inbound ports needed |

### API Keys

| Variable | Mode | Source |
|----------|------|--------|
| `BINANCE_TESTNET_KEY` | Testnet | [testnet.binancefuture.com](https://testnet.binancefuture.com) |
| `BINANCE_TESTNET_SECRET` | Testnet | Same portal |
| `BINANCE_KEY` | Production | [binance.com](https://www.binance.com) API management |
| `BINANCE_SECRET` | Production | Same portal |

### Production Models

Located at `model_training/results_v10/production/`:
- `production_model_s42.cbm`
- `production_model_s123.cbm`
- `production_model_s777.cbm`
- `production_metadata.json`

Created by: `python model_training/train_v10_production.py`

---

## 3. First Deployment

### 3.1 GCE Instance Setup

```bash
# SSH into fresh GCE instance
chmod +x deploy/gce-setup.sh
./deploy/gce-setup.sh
# Follow printed instructions
```

### 3.2 Code + Keys

```bash
cd /opt/tamagochi
git clone <repo-url> .

# Create .env with testnet keys
cat > .env << 'EOF'
BINANCE_TESTNET_KEY=your-testnet-api-key
BINANCE_TESTNET_SECRET=your-testnet-secret-key
EOF
chmod 600 .env
```

### 3.3 Build & Launch

```bash
# Build both images + start
docker compose up -d --build

# Watch startup (data service needs ~2min for first bootstrap)
docker compose logs -f

# Verify health
docker compose ps
# tamagochi-data should show "healthy" after start_period (120s)
# tamagochi-bot starts after data service is healthy
```

### 3.4 Verify First Cycle

```bash
# Data service: check predictions are flowing
docker compose exec tamagochi-data cat /data/status.json
# Expect: {"state": "running", "cycle": 1, ...}

# Bot: check it's reading predictions
docker compose logs tamagochi-bot --tail=20
# Expect: "--- Cycle 1 ---" + prediction output

# Check SL/TP verification on startup
docker compose logs tamagochi-bot | grep "SL/TP"
# Expect: "SL/TP verified" or "Verifying SL/TP orders"
```

### 3.5 Post-Deploy Checklist

- [ ] `docker compose ps` shows both services running
- [ ] Data service status.json shows `"state": "running"`
- [ ] Bot logs show prediction cycles (5-min intervals)
- [ ] `trading_state/state.json` exists (created after first cycle)
- [ ] `trading_logs/` has today's CSV (created on first trade)
- [ ] Testnet balance visible on [testnet.binancefuture.com](https://testnet.binancefuture.com)

---

## 4. Configuration Reference

### Bot Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--testnet` | required* | Run on Binance Futures testnet |
| `--live` | required* | Run on production (requires typing CONFIRM) |
| `--dry-run` | required* | Predict only, no orders |
| `--threshold` | 0.75 | Min confidence to trigger trade (0.50-0.95) |
| `--amount` | 100.0 | USDT per position |
| `--leverage` | 10 | Leverage multiplier (1-125) |
| `--interval` | 300 | Seconds between cycles |
| `--data-service` | false | Read from data service CSV (vs live pipeline) |
| `--data-dir` | /data | Path to shared data volume |
| `--debug` | false | Enable debug-level logging |

*One of `--testnet`, `--live`, `--dry-run` is required.

### Current Testnet Config (docker-compose.yml)

```
--testnet --data-service --data-dir /data --threshold 0.70 --amount 10 --leverage 20
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| threshold | 0.70 | All 4 WF folds profitable at @0.70 |
| amount | $10 | Small size for testnet validation |
| leverage | 20x | Higher leverage to test margin behavior |
| SL/TP | 2%/4% | V10 baseline targets |

### Data Service Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | ./persistent_data | Root of persistent storage |
| `--interval` | 300 | Seconds between prediction cycles |
| `--threshold` | 0.75 | Prediction threshold (bot can override) |
| `--model-dir` | (auto) | Custom model directory |
| `--debug` | false | Debug logging |

### Risk Parameters (Hardcoded)

| Parameter | Value | Location |
|-----------|-------|----------|
| SL | 2.0% | `position_manager.py:18` |
| TP | 4.0% | `position_manager.py:18` |
| Max hold | 86400s (24h) | `position_manager.py:18` |
| Profit lock trigger | 3.5% | `position_manager.py:18` |
| Profit lock floor | 3.0% | `position_manager.py:18` |
| Min rolling WR | 33.3% | `trading_bot.py:450` |
| Rolling window | 20 trades | `trading_bot.py:450` |
| Max consecutive losses | 5 | `trading_bot.py:450` |
| Pause cooldown | 3600s (1h) | `safety.py:15` |
| API timeout | 10s | `executor.py:56` |
| API retry attempts | 3 | `executor.py:92` |
| API retry backoff | 0s, 2s, 5s | `executor.py:98` |
| Emergency close attempts | 30 | `executor.py:421` |
| Emergency close interval | 10s | `executor.py:421` |
| Circuit breaker trigger | 3 errors | `trading_bot.py:521` |
| Circuit breaker max backoff | 600s (10min) | `trading_bot.py:522` |
| Stop grace period | 60s | `docker-compose.yml:22` |

---

## 5. Monitoring & Health Checks

### Docker Native

```bash
# Service status
docker compose ps

# Live logs (both services)
docker compose logs -f

# Bot only (last 50 lines)
docker compose logs --tail=50 tamagochi-bot

# Data service only
docker compose logs --tail=50 tamagochi-data
```

### Data Service Health

The data service writes `/data/status.json` every cycle:

```json
{
  "state": "running",
  "cycle": 42,
  "last_cycle": {
    "duration_sec": 12.3,
    "l1_bars_fetched": {"5M": 1, "15M": 0, ...},
    "l2_rows_added": 55,
    "l3_prediction": {"signal": "NO_TRADE", "confidence": 0.82}
  },
  "consecutive_errors": 0
}
```

**States**: `starting` → `running` (normal) | `error` (recoverable) | `stopped` (SIGTERM)

Docker health check polls this file every 30s. If state is not `running` or `starting` for 3 consecutive checks, Docker marks the container as unhealthy.

### Bot Health Indicators

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `--- Cycle N ---` | Normal 5-min heartbeat | None |
| `CIRCUIT BREAKER: N consecutive errors` | API failures stacking up | Check Binance status |
| `EMERGENCY CLOSE attempt` | Naked position detected | Monitor until resolved |
| `SAFETY PAUSE` | WR below threshold or loss streak | Auto-resumes after cooldown |
| `SL/TP verified` | Startup check passed | None |
| `MISSING SL+TP orders` | Orders disappeared from exchange | Auto re-places |
| `RESYNCED` | Local/exchange state mismatch | Auto-corrected |

### Key Log Files

| Path | Content |
|------|---------|
| `trading_logs/trades_YYYY-MM-DD.csv` | Per-trade audit trail |
| `trading_state/state.json` | Position + safety state (survives restart) |

---

## 6. Risk Engine

Full details: [`RISK_ENGINE.md`](RISK_ENGINE.md)

### Summary of Protections

| Layer | Feature | Behavior |
|-------|---------|----------|
| **Network** | API timeout (10s) | Prevents indefinite blocking |
| **Network** | API retry (3x, backoff 0/2/5s) | Handles transient failures |
| **Network** | Client reconnection | Recreates Binance client on persistent errors |
| **Network** | Circuit breaker (3 errors → backoff) | Exponential backoff up to 10min cap |
| **Position** | SL/TP verification (startup + resync) | Re-places missing orders |
| **Position** | Emergency close (30 attempts) | Blocks until naked position resolved |
| **Position** | Partial fill handling | Reads executedQty, logs partial fills |
| **Position** | Zero fill guard | Rejects orders that filled 0 quantity |
| **Position** | Max hold timer + aggressive cycle | 60s checks in last 30 minutes |
| **Safety** | Rolling WR monitor (33.3% min) | Auto-pauses below break-even |
| **Safety** | Consecutive loss limit (5) | Auto-pauses on loss streaks |
| **Safety** | Cooldown grace | Prevents re-pause loop after recovery |
| **Safety** | Profit lock (3.5% trigger / 3.0% floor) | Trailing stop on profitable positions |
| **Docker** | SIGTERM handler | Saves state, interrupts emergency close |
| **Docker** | 60s stop grace period | Time for cleanup before SIGKILL |

---

## 7. Incident Response

### 7.1 Bot Not Placing Trades

**Symptoms**: Cycles running but no OPENED/ADDED actions

1. Check if signal is NO_TRADE at current threshold:
   ```bash
   docker compose logs --tail=20 tamagochi-bot | grep "conf="
   ```
2. Check if safety is PAUSED:
   ```bash
   docker compose logs tamagochi-bot | grep "PAUSED"
   ```
3. Check if data is stale (>10min old):
   ```bash
   docker compose logs tamagochi-bot | grep "stale"
   ```
4. Check data service health:
   ```bash
   docker compose exec tamagochi-data cat /data/status.json
   ```

### 7.2 Naked Position (No SL/TP)

**Symptoms**: `EMERGENCY CLOSE attempt` in logs

This is handled automatically. The emergency close loop:
- Retries up to 30 times with 10s intervals
- Checks if position already closed by SL/TP between attempts
- Respects SIGTERM for graceful Docker stop
- Logs `MANUAL INTERVENTION REQUIRED` if all 30 attempts fail

**Manual intervention** (if automatic fails):
```bash
# Check position on exchange
docker compose exec tamagochi-bot python -c "
from trading.executor import BinanceFuturesExecutor
e = BinanceFuturesExecutor(testnet=True)
print(e.get_position())
print(e.get_open_orders())
"

# Or close manually on testnet.binancefuture.com
```

### 7.3 Circuit Breaker Active

**Symptoms**: `CIRCUIT BREAKER: N consecutive errors` in logs

1. Check Binance API status: [binance.com/en/api-status](https://www.binance.com/en/api-status)
2. Auto-reconnect triggers at exactly 3 errors
3. Backoff formula: `min(300 * 2^(errors-2), 600)` seconds
4. Resets automatically on first successful cycle

**If stuck for >30min**:
```bash
docker compose restart tamagochi-bot
```

### 7.4 Data Service Down

**Symptoms**: Bot logs show `Predictions CSV not found` or `stale`

```bash
# Check data service status
docker compose ps tamagochi-data
docker compose logs --tail=30 tamagochi-data

# If crashed, it auto-restarts (unless-stopped policy)
# Force rebuild if needed:
docker compose up -d --build tamagochi-data
```

Bot behavior: continues cycling, skips prediction, does NOT close existing positions. Existing SL/TP orders remain on exchange.

### 7.5 State File Corruption

```bash
# Stop bot gracefully
docker compose stop tamagochi-bot

# Backup corrupted state
cp trading_state/state.json trading_state/state.json.bak

# Delete state (bot will resync from exchange on restart)
rm trading_state/state.json

# Restart
docker compose up -d tamagochi-bot
```

On restart without state file, the bot:
1. Syncs position from exchange (picks up any open position)
2. Recalculates SL/TP from exchange entry price
3. Verifies SL/TP orders exist, re-places if missing
4. Starts fresh safety history (no pause state)

---

## 8. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Bot exits immediately | Missing API keys in .env | Check `docker compose logs tamagochi-bot` for RuntimeError |
| `Set leverage failed` | Already set (normal) | Warning, not error — safe to ignore |
| `Margin type already ISOLATED` | Already set (normal) | Info message — safe to ignore |
| Prediction always NO_TRADE | Threshold too high | Lower `--threshold` (try 0.65-0.70) |
| `Prediction is stale` | Data service behind | Check data service logs, may need restart |
| `No position to close` | SL/TP already triggered | Normal — exchange closed position |
| `Cancel orders failed` | No orders to cancel | Warning, not error — safe to ignore |

### Checking Exchange State

```bash
# Enter bot container
docker compose exec tamagochi-bot bash

# Python shell inside container
python -c "
from trading.executor import BinanceFuturesExecutor
e = BinanceFuturesExecutor(testnet=True)
print('Position:', e.get_position())
print('Mark:', e.get_mark_price())
print('Orders:', e.get_open_orders())
"
```

### Docker Compose Commands

```bash
docker compose up -d --build     # Build + start (background)
docker compose stop              # Graceful stop (SIGTERM)
docker compose restart           # Stop + start
docker compose down              # Stop + remove containers
docker compose logs -f           # Follow all logs
docker compose logs --tail=100   # Last 100 lines
docker compose ps                # Service status
docker compose exec <svc> bash   # Shell into container
```

---

## 9. Maintenance

### Log Rotation

Trade logs accumulate at `trading_logs/trades_YYYY-MM-DD.csv` (one per day). Each file is small (<100KB). Archive monthly:

```bash
# Archive old logs (>30 days)
find trading_logs/ -name "trades_*.csv" -mtime +30 -exec gzip {} \;
```

### Data Volume Growth

| Data | Growth Rate | Action |
|------|-------------|--------|
| Klines (11 CSVs) | ~100MB/year | Trim to 2 years if needed |
| Decomposed (55 CSVs) | ~500MB/year | Trim with klines |
| Predictions CSV | ~5MB/year | Keep all (audit trail) |
| Trade logs | ~1MB/year | Archive monthly |
| State JSON | <1KB | No action needed |

### Updating Production Models

When new models are trained:

```bash
# 1. Train new production models locally
python model_training/train_v10_production.py

# 2. Copy to server
scp model_training/results_v10/production/*.cbm user@server:/opt/tamagochi/model_training/results_v10/production/
scp model_training/results_v10/production/production_metadata.json user@server:/opt/tamagochi/model_training/results_v10/production/

# 3. Rebuild + restart both services (models are baked into Docker images)
docker compose up -d --build
```

### Updating Bot Config

Edit `docker-compose.yml` command line, then:
```bash
docker compose up -d tamagochi-bot  # Recreates bot only
```

---

## 10. Rollback Procedures

### Roll Back Code

```bash
# Find previous working commit
git log --oneline -10

# Check out previous version
git checkout <commit-hash>

# Rebuild
docker compose up -d --build
```

### Roll Back to Manual Mode

```bash
# Stop automated trading
docker compose stop tamagochi-bot

# Position state is preserved in trading_state/state.json
# SL/TP orders remain on exchange
# Manage position manually on testnet.binancefuture.com
```

### Emergency: Close All Positions

```bash
# Via bot container
docker compose exec tamagochi-bot python -c "
from trading.executor import BinanceFuturesExecutor
e = BinanceFuturesExecutor(testnet=True)
e.cancel_all_orders()
e.close_position()
print('All positions and orders cleared')
"

# Then stop bot
docker compose stop tamagochi-bot
```

### Full Teardown

```bash
docker compose down              # Stop + remove containers
docker volume rm tamagochi_persistent_data  # Remove shared data
rm -rf trading_state/ trading_logs/         # Remove local state
```

---

## Appendix: Future Multi-Bot Setup

When deploying multiple SL/TP configs (C1, C2, C3) simultaneously, extend `docker-compose.yml`:

```yaml
# Each bot gets its own state/logs but shares data service
tamagochi-bot-c1:
  build: .
  command: ["python", "-u", "trading_bot.py", "--testnet", "--data-service",
            "--data-dir", "/data", "--threshold", "0.70", "--amount", "10"]
  volumes:
    - ./trading_logs_c1:/app/trading_logs
    - ./trading_state_c1:/app/trading_state
    - persistent_data:/data:ro

tamagochi-bot-v10:
  build: .
  command: ["python", "-u", "trading_bot.py", "--testnet", "--data-service",
            "--data-dir", "/data", "--threshold", "0.70", "--amount", "10"]
  volumes:
    - ./trading_logs_v10:/app/trading_logs
    - ./trading_state_v10:/app/trading_state
    - persistent_data:/data:ro
```

**Requirements for multi-bot**:
- Each bot needs its own `trading_state/` and `trading_logs/` directories
- Separate production models per SL/TP config (not yet trained for C1/C2/C3)
- Single data service serves all bots (predictions are pre-threshold)
- Binance testnet allows multiple positions — but **only one per symbol** on Futures
- For true multi-config testing: use sub-accounts or sequential testing
