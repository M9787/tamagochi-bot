# GCE Deployment Update History

## VM Info
- **Instance**: instance-20260303-232149, c3-standard-4, asia-southeast1-b
- **IP**: 35.197.139.45 (ephemeral — changes on stop/start)
- **Project**: project-4d1ee130-e5dc-4495-a47
- **Repo**: https://github.com/M9787/tamagochi-bot.git (private, master branch)
- **Path**: `/opt/tamagochi`

---

## Update #2 — 2026-03-04 (Safety Rule Simplification)

**Commit**: TBD (safety rule update)

### Changes
| File | Change |
|------|--------|
| `trading/safety.py` | Full rewrite: removed consecutive-loss pause + rolling-window WR; replaced with 7-day aggregated WR |
| `trading_bot.py` | Updated SafetyMonitor instantiation + banner text |

### Safety Rules — Before vs After

| Rule | Before | After |
|------|--------|-------|
| Consecutive loss pause | 5 SL hits → 1h pause | **REMOVED** |
| Rolling WR check | Last 20 trades < 33.3% → pause | **REMOVED** |
| 7-day aggregated WR | N/A | < 33.3% over last 7 days → pause (min 3 trades) |
| Cooldown after pause | 1h | 1h (unchanged) |
| Grace flag | Yes | Yes (unchanged) |

### Rationale
- Consecutive-loss pause was too aggressive for testnet evaluation — 5 SL hits in a row would stop the bot before gathering enough data
- Rolling 20-trade window too short to be statistically meaningful
- 7-day window gives a more stable and fair evaluation of model performance

### Deployment Steps
```bash
cd /opt/tamagochi
sudo git pull origin master
sudo docker compose up -d --build
sudo docker compose logs --tail=10 tamagochi-bot  # verify banner shows new safety params
```

---

## Update #1 — 2026-03-04 (Initial Deployment + 3 HIGH Fixes)

**Commit**: 2a9531c

### Changes
| File | Change |
|------|--------|
| `trading/executor.py` | `open_position()` accepts sl_price/tp_price, places SL/TP atomically after fill |
| `trading/position_manager.py` | `_open_new()` pre-calculates SL/TP from mark price, passes to executor |
| `data_service/csv_io.py` | `append_rows_atomic()` reads existing + concat + temp + rename for ALL writes |
| `data_service/layers.py` | Candle dedup uses `>=` to overwrite stale last candle with fresh data |
| `trading_bot.py` | Column validation guard in `read_latest_prediction()` |

### 3 HIGH Severity Fixes
1. **Naked position window**: SL/TP now placed immediately after market order fill (was 2-5s gap)
2. **Non-atomic CSV**: All writes use temp file + `os.replace()` (was direct append)
3. **Incomplete candle dedup**: Last stored candle overwritten with fresh closed data (was `>`, now `>=`)

### Config
```
--testnet --data-service --data-dir /data --threshold 0.70 --amount 10 --leverage 20
V10 model, SL=2%/TP=4%, 3-seed ensemble
```

### Verification
- Both containers UP + HEALTHY
- First L1->L2->L3 cycle: 16.6s, 508 features
- Bot connected to Binance Futures testnet, 20x leverage
- All 8 E2E checks PASS
