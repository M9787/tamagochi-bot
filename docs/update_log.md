# CLAUDE.md Update Log

## 2026-03-12 (Session 2)

**Changes detected**: Multi-trade paper trading system implemented. New file `trading/multi_trade_manager.py`. `trading_bot.py` rewritten for dual-mode (multi-trade dry-run vs single-position live/testnet). `telegram_service/formatters.py` updated with multi-trade display for all commands. `telegram_service/readers.py` extended for multi-trade state. New close actions: SL_TRIGGERED, TP_TRIGGERED, LIQUIDATED, MAX_HOLD_24H. CLI defaults changed ($10/20x/$1000). Hardcoded SL/TP replaced with config imports.
**Files updated**: CLAUDE.md (multi-trade constraint), docs/architecture.md (file tree), docs/trading-bot.md (modes, multi-trade section, state formats), docs/telegram-bot.md (command descriptions, multi-trade support section), MEMORY.md (trading mode, CLI defaults, close actions)
**Summary**: Full multi-trade paper trading implementation. Each signal opens independent $10/20x trade with separate SL/TP tracking. $1000 simulated balance with margin locking. All 13 telegram commands adapted for multi-trade display. 6 bugs found and fixed during audit (ID collisions, CSV zeros, missing liquidation guard, uncapped unrealized PnL, missing close actions, config params).

## 2026-03-12

**Changes detected**: 3-layer data governance framework implemented (resilience/validation/alerting). New file `core/data_validation.py`. 6 bugs found by 4-agent audit and fixed: broken ThreadPoolExecutor timeout, telegram staleness skipping trade events, missing bot prediction row validation, dashboard hardcoded SL/TP/max_hold, csv_io return value. New shared constant `STALENESS_THRESHOLD_SEC=1200` in config.py.
**Files updated**: CLAUDE.md (5 new code constraints, staleness in Key Config), docs/architecture.md (file tree + config), docs/trading-bot.md (state backups, validation), docs/telegram-bot.md (staleness guard), MEMORY.md (governance TODO)
**Summary**: Comprehensive data governance after 19h silent hang. Cycle timeout prevents hangs, validation rejects corrupt predictions before trading, staleness guard suppresses stale signal alerts while preserving trade event broadcasts. All 6 audit-identified bugs verified fixed by re-audit.

## 2026-03-11 (Session 2)

**Changes detected**: Bot Dockerfile missing backfill_predictions.py. Dashboard/Telegram signal divergence caused by merge priority (backfill retroactive klines vs real-time data service). Dashboard kline loading was single-source instead of merged. 30-day backfill run (48.7% WR @0.50). GCE disk cleanup (97%→86%).
**Files updated**: CLAUDE.md (new constraint: dashboard merge priority), Dockerfile (COPY backfill_predictions.py), backtest_dashboard.py (merge priority + kline merging), MEMORY.md (disk, backfill stats, insights)
**Summary**: Fixed signal alignment between Telegram and Dashboard by flipping merge priority to data_service > backfill. Root cause: backfill downloads klines retroactively so higher-TF forming candles have different Close prices than real-time. Added backfill to bot Dockerfile. Merged kline sources for full 30-day chart coverage.

## 2026-03-11

**Changes detected**: Testnet→Mainnet dry-run migration. docker-compose.yml `--testnet`→`--dry-run`. GCE .env updated with mainnet keys. Dashboard tz-aware datetime ValueError fixed. Backfill CSV kline MergeError fixed (string dtype→datetime). Data governance: old testnet trade logs archived, predictions cleared and regenerated fresh from mainnet, orphan Docker volume removed.
**Files updated**: CLAUDE.md (2 new code constraints), docs/deployment.md (mode, .env key order, bot role), MEMORY.md (disk, insights, TODOs)
**Summary**: Full migration from testnet to mainnet dry-run mode. Fixed 2 pandas 3.0 datetime bugs (dashboard + backfill). Added datetime normalization and CSV kline parsing constraints to CLAUDE.md. All data frames verified as mainnet-sourced.

## 2026-03-05 (Session 3)

**Changes detected**: Entry/SL/TP price display added to telegram push, bot console, and dashboard table. GCE rebuild with Docker prune (91% -> 75% disk).
**Files updated**: docs/telegram-bot.md, docs/trading-bot.md, MEMORY.md
**Summary**: Documented entry price + SL/TP display feature across all signal outputs. Updated disk usage in memory after Docker image prune.

## 2026-03-05 (Session 2)

**Changes detected**: System consolidation — structured JSONL logging, dashboard containerization, data alignment, GCE deploy
**Files updated**: CLAUDE.md (commands section), docs/architecture.md (file tree), docs/deployment.md (4 services + logging)
**Summary**: Added core/structured_log.py, dashboard/Dockerfile, requirements-dashboard.txt. Docker compose now runs 4 containers. All services write JSONL logs with rotation. Dashboard reads from data service volume. Backfill reuses data service klines.

## 2026-03-05

**Changes detected**: Major restructure — CLAUDE.md slimmed from 511 to 87 lines
**Files updated**: CLAUDE.md, MEMORY.md, docs/architecture.md, docs/ml-pipeline.md, docs/trading-bot.md, docs/telegram-bot.md, docs/deployment.md, docs/pipeline-alignment.md
**Summary**: Redistributed all secondary information from CLAUDE.md into 6 docs/ reference files. CLAUDE.md now contains only core identity, algorithm, signal theory, constraints, commands, and reference table. Context savings: 83% (34KB -> 6KB loaded per session).
