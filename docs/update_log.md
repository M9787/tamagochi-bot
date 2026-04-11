# CLAUDE.md Update Log

## 2026-04-11 (pm)

**Changes detected**: Phase H.1 executed on Contabo VPS (Docker 29.4.0 + Compose v5.1.2 installed, log rotation, project dir). Parity test PASS 21.75s pre-H.1. Zero local code changes. Local-env workaround documented: Windows Git Bash has no `sshpass`, replaced by `C:\tmp\contabo_ssh.py` (paramiko, reads `CONTABO_PASS` from env only). `contabo-master` subagent type not spawnable in current CLI build -- main session runs the protocol inline.
**Files updated**: `docs/deployment-contabo.md` (H.0/H.1 marked DONE, Windows deploy note added), `docs/update_log.md`.
**Summary**: Contabo VPS is Docker-ready for H.2 (`.env` push) -- waiting on `BINANCE_KEY`/`BINANCE_SECRET`/`TELEGRAM_BOT_TOKEN` source decision from user. CLAUDE.md unchanged (193 lines -- above the skill's 100-line aspiration but no new core constraints warranted edits this session).

## 2026-04-11

**Changes detected**: Contabo multi-target live deploy stack (feat/multitarget-live commit 15ffab5): new `core/multitarget_{feature_builder,predictor,router}.py`, `docker-compose.contabo.yml`, `deploy/contabo_bootstrap.sh`, `tests/test_multitarget_parity.py` (PASS 29.92s), Dockerfile build args `COPY_V3_MODELS`/`COPY_MT_MODELS`, env gates `TAMAGOCHI_LOAD_V3`/`TAMAGOCHI_MULTITARGET`/`TELEGRAM_PREDICTIONS_CSV`, additive edits in `data_service/layers.py`, `trading/multi_trade_manager.py` (`lock_mode`), `trading_bot.py` (`--multitarget`), `telegram_service/readers.py`. Self-improvement loop: 3 hooks (`correction_detector`, `failure_logger`, `stop_reflector`), 3 skills (`reflect`, `learn`, `consolidate`), `.claude/LEARNINGS.md` auto-imported. 11-test empirical validation suite executed, all pass, zero drift.
**Files updated**: `docs/deployment-contabo.md` (new), `docs/self-improvement-loop.md` (new), `CLAUDE.md` (2 reference rows added), `docs/update_log.md`.
**Summary**: Documented Contabo parallel multi-target deployment architecture (code built, deploy pending Phase H) and the empirically validated self-improvement loop. CLAUDE.md grew 184->186 lines.

## 2026-04-08

**Changes detected**: Grid search for binary meta-label hyperparameters (3072 combos, 3.6h). Meta-labeling (Lopez de Prado) validated as best model (+306%, PASS 4/4). Pipeline audit (17/19 PASS). New files: `grid_search_meta_label.py`, `validate_v3_audit.py`, `generate_experiment_report.py`. Results: `results_v10/grid_search_meta_label/grid_results.csv` + `best_params.json`.
**Files updated**: CLAUDE.md (commands +2, reference table updated), docs/ml-pipeline.md (meta-labeling section, grid search findings, candidates A/B)
**Summary**: Meta-labeling is the confirmed best model (+306%). Grid search found only 3/6 params matter (depth, lr, l2). Current d=4 sits on cliff edge for trade volume. Candidates A (d=9,lr=0.10,l2=30) and B (d=7,lr=0.10,l2=8) pending 4-fold validation.

## 2026-04-06

**Changes detected**: V3 walk-forward validation script created and run. Results: PASS 4/4, summer drought solved (+54% vs +20%), +299% total vs V10 +250%.
**Files updated**: CLAUDE.md (commands, reference table), docs/ml-pipeline.md (V3 walkforward results section)
**Summary**: Created train_stacking_v3_walkforward.py and validated V3 across all seasons. TIGHT stream fills summer gap (64/77 trades). ASYM/TREND class weight fix identified as next multiplier.


## 2026-04-06

**Changes detected**: Walk-forward validation completed with 518 features (Phase G BB). Results: AUC=0.856±0.032, +250% @0.70, +232% @0.75 (4/4 folds). BB feature importance: bb_lower_pierce_1H #18/518.
**Files updated**: docs/ml-pipeline.md (walkforward results, experiment table row), docs/update_log.md
**Summary**: 518-feature model validated. BB features improve higher thresholds (+23% at 0.75, +26% at 0.80). Production-ready.

## 2026-04-05

**Changes detected**: Phase G Bollinger Band features (10 new, 508->518) added to all 3 encoding paths (encode_v10.py, incremental_encoder.py, live_predict.py). Stacking V3 created (ASYM/TREND base models, dual meta-targets PRIMARY+TIGHT). n=1 parity bug found and fixed in incremental encoder.
**Files updated**: CLAUDE.md (518 features, V3 command, V3 reference), docs/ml-pipeline.md (518 count, Phase G section, Stacking V3 section), docs/pipeline-alignment.md (508->518 in 3 places), docs/update_log.md
**Summary**: Full Phase G propagation across pipeline + stacking V3 dual-target architecture. All 3 encoding paths verified for math parity (n=1 and n=35 tests pass). Backfill logic confirmed safe (batch encoder with 1400+ bars warm-up).

## 2026-03-31 (session 2)

**Changes detected**: Stacking meta-model V2 pipeline built and optimized. New files: `train_stacking.py` (V1), `train_stacking_v2.py` (V2, 5 base models + 131 interaction features), `optuna_meta_search.py` (100-trial meta-model param search), `compare_models.py` (prod d8 vs d12 comparison). GitNexus MCP hooks enforced via `~/.claude/settings.json`. Asymmetric threshold search added (separate LONG/SHORT thresholds).
**Files updated**: CLAUDE.md (stacking commands + reference), docs/ml-pipeline.md (full stacking V2 section), docs/update_log.md
**Summary**: Stacking meta-model with 5 diverse base models (V10/C1/C2/H12/H48). Optuna found d5 meta-model params beat d8 baseline. 131 features including temporal momentum, voting/confirmation, cross-horizon convergence. @0.75: WR=81.8%, PF=9.00. Asymmetric thresholds show SHORT needs lower threshold than LONG.

## 2026-03-31

**Changes detected**: Optuna 100-trial hyperparameter search completed (Trial 34 winner). Walk-forward validated Optuna+d12 (+277% @0.75, 80% prec) and d14 (+209%, 90% prec). Historical data backfilled to Aug 2017 (904K 5M rows). New files: `download_backfill.py`, `optuna_hyperparam_search.py`. `download_data.py` extended to 2400 days. `train_v10_walkforward.py` params updated through d12→d14 experiments.
**Files updated**: CLAUDE.md (commands: backfill + optuna), docs/ml-pipeline.md (hyperparams, Optuna results, depth experiments, data coverage, experiment history), docs/update_log.md
**Summary**: Hyperparameter optimization session. Optuna+d12 is new best config for 2%/4% SL/TP (precision +10% over V7 d8 baseline). d14 too selective. Data coverage extended from ~6yr to ~8yr via backfill.

## 2026-03-30

**Changes detected**: GitNexus MCP installed and configured (`npx gitnexus analyze` + `npx gitnexus setup`). Auto-generated 102-line block appended to CLAUDE.md, plus AGENTS.md, `.claude/skills/gitnexus/` (6 skills), PreToolUse/PostToolUse hooks, `.gitnexus` added to .gitignore.
**Files updated**: CLAUDE.md (moved GitNexus block to docs/gitnexus.md, added reference row — back to 112 lines), docs/gitnexus.md (new — full GitNexus tool reference)
**Summary**: GitNexus MCP setup for code intelligence (746 nodes, 2195 edges, 61 flows). Moved verbose GitNexus instructions from CLAUDE.md to docs/gitnexus.md to keep CLAUDE.md lean.

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
