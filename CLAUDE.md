# CLAUDE.md

@.claude/LEARNINGS.md

> **Before proposing any approach, check `.claude/LEARNINGS.md` for rules that override defaults.**

## Project

Crypto trading signal system for BTCUSDT: Binance klines (11 TFs, ~8yr) -> rolling regression (5 windows) -> CatBoost 3-class ML (518 features, V10) -> live trading on Binance Futures via Docker on GCE. + Multi-target system (8 SL/TP horizons: scalp to position, 32 models).

## Core Algorithm (analysis.py)

`iterative_regression(df, window_size)` -- sliding window of `2*window_size` prices using `sqrt(price)`, `linregress` on each half -> **slope_b** (historical), **slope_f** (current), **angle** (divergence), **acceleration** (rate of change).

## Signal Theory

- **Crossing** = primary trigger (angle convergence/cross/divergence between young+elder windows)
- **Reversal** = secondary confirmation (5-point angle pattern)
- **Direction** (slope_f) = context (LONG/SHORT territory)
- **Acceleration** = quality gate (reject CLOSE/VERY_CLOSE GMM zones)
- **Philosophy**: Precision >> Recall. Target 60%+ precision. SL=2%, TP=4% (break-even=33.3%). Only ~10% of moments tradeable. Labels (approximate): NO_TRADE~73%, SHORT~14%, LONG~14%.

## Multi-Target System (8 SL/TP Horizons)

8 base models (different SL/TP labels) + 8 stacking models = 8 independent signals from scalping to position trading. All 8/8 PASS on OOS test (Jul 2025 - Mar 2026). AUCs 0.855-0.875.

| Target | SL/TP | max_hold | cd | Category | Best Thresh | Profit | @0.80 Prec | @0.80 Profit | @0.80 Trades |
|--------|-------|----------|-----|----------|-------------|--------|------------|--------------|--------------|
| T1 | 0.5/1.0 | 72 | 15 | Scalp | @0.65 | +1121% | 81.6% | +992% | 1532 |
| T2 | 0.5/1.5 | 108 | 15 | Scalp | @0.50 | +1169% | 81.5% | +831% | 779 |
| T3 | 0.5/2.0 | 144 | 15 | Scalp | @0.42 | +1143% | 79.1% | +776% | 564 |
| T4 | 1.0/2.0 | 144 | 30 | Intraday | @0.75 | +668% | 80.2% | +661% | 560 |
| T5 | 1.0/3.0 | 216 | 30 | Intraday | @0.60 | +630% | 65.7% | +553% | 463 |
| T6 | 1.0/4.0 | 288 | 30 | Swing | @0.60 | +490% | 54.9% | +389% | 231 |
| T7 | 2.0/4.0 | 288 | 60 | Swing | @0.75 | +252% | 64.6% | +236% | 146 |
| T8 | 2.0/6.0 | 432 | 60 | Position | @0.65 | +154% | 71.1% | +150% | 49 |

**Inference pipeline**: encode 518 V10 features -> run 24 base models (8 targets x 3 seeds), average probs per target -> build interaction features (~150) + top-50 raw features -> run 8 stacking models -> 8 independent signals.

**Deployment models (32 files, ~517MB total; full `results_v10/multitarget/` dir ≈760MB including label cache and OOS parquets)**:
- Base: `results_v10/multitarget/base_models/base_model_T{1-8}_s{42,123,777}.cbm` (24 files, 511MB)
- Stacking: `results_v10/multitarget/stacking/stacking_model_T{1-8}.cbm` (8 files, 5.4MB)

**Config**: `model_training/multitarget_config.py` (targets, params, timeline, paths)

## MANDATORY: Model Evaluation Checklist

**Every training script MUST include**: ROC AUC (per-class + macro), ROC/PR curve PNGs, per-class P/R/F1, confusion matrix PNG, trade precision at threshold, profit factor + max drawdown + Sharpe, equity curve PNG, trade log CSV. **No exceptions.**

## Key Config (config.py)

- TFs: `3D,1D,12H,8H,6H,4H,2H,1H,30M,15M,5M` | Windows: `[30,60,100,120,160]`
- SL=2%, TP=4%, max_hold=288, threshold=0.75, bootstrap=1400, staleness=1200s
- 7 crossing pairs | TF groups: Youngs(5M,15M,30M) -> Adults(1H,2H,4H) -> Balzaks(6H,8H,12H) -> Grans(1D,3D)

## Signal Derivation (CANONICAL -- must be identical everywhere)

```python
probs = [prob_no_trade, prob_long, prob_short]
pred_class = argmax(probs)             # 0=NO_TRADE, 1=LONG, 2=SHORT
if pred_class in (1, 2) and probs[pred_class] >= threshold:
    signal = "LONG" if pred_class == 1 else "SHORT"
else:
    signal = "NO_TRADE"
```

Used in: `batch_ensemble_predict()`, `trading_bot.read_latest_prediction()`, `telegram_service/bot.py`.

## Code Constraints (CRITICAL -- prevents bugs)

- Fallback zero-fill column names MUST match normal-path names exactly
- Label alignment: explicit `sort_values` required (`isin` preserves original order)
- CatBoost GPU: AUC eval_metric not supported -- use `MultiClass` loss + `TotalF1` eval
- Labels: SL=2%, TP=4%, max_hold=288 (5M candles = 24h)
- **Incremental encoder features MUST match batch encoder** -- any encode change must mirror in `incremental_encoder.py`
- **Gap backfill must track TF index changes** -- only feed TF data when searchsorted index changes
- **State backup rotation** -- `feature_state.json` keeps 3 backups for corruption recovery; `trading_state/state.json` also keeps 3 rotated backups
- **Data validation** -- `core/data_validation.py` guards: prob sum ~1.0, canonical signal, no NaN/inf, feature count, kline continuity, freshness
- **Cycle timeout** -- `data_service/service.py` wraps `run_cycle()` in 300s timeout + watchdog thread (prevents silent hangs)
- **Predictions dedup** -- `append_rows_atomic()` supports `dedup_col` param; predictions CSV deduped by `time`
- **Staleness guard** -- `STALENESS_THRESHOLD_SEC=1200` in config; used by trading bot (skip cycle), telegram (skip signal alert, trade events still flow), data_validation (freshness check)
- TF-native lags: MUST shift BEFORE merge_asof
- **Datetime normalization**: All `pd.to_datetime()` on user-facing data MUST use `utc=True` + `.dt.tz_localize(None)` -- CSV sources mix tz-aware/naive timestamps
- **CSV kline loading**: When loading data service klines from CSV (vs API), MUST parse `time`/`Open Time` as datetime and numerics as float -- `pd.read_csv` returns strings with pandas 3.0 StringDtype
- **Dashboard merge priority**: `data_service > backfill > live` for overlapping timestamps -- backfill re-downloads klines retroactively (higher TF candle closes differ from real-time), causing signal drift vs data service/bot/telegram
- **Multi-trade mode** -- dry-run uses `MultiTradeManager`: each signal opens separate $10/20x trade, no adding to positions, independent SL/TP/max_hold per trade, $1000 simulated balance with margin locking, liquidation guard caps loss at margin
- **Download incremental mode** -- `download_data.py` defaults to incremental: reads existing CSV max timestamp, fetches only new klines, appends with dedup (keep='last'). Use `--full` to force complete re-download (overwrites ml_data CSVs, losing backfill history before the 2400-day window)
- **ETL determinism** -- `etl.py` feeds ALL available klines to `iterative_regression()` regardless of `--start`/`--end`. Date filters apply to OUTPUT only (after regression), ensuring features are always initialized from the same data point
- **ETL incremental** -- `etl.py --incremental` runs full regression but only appends rows newer than existing decomposed data
- **Multi-target deployment needs all 24 base models (3 seeds x 8 targets)** -- stacking trained on 3-seed averaged probabilities; using fewer seeds changes probability distribution
- **Stacking features depend on ALL 8 base model outputs** -- cannot run one target's stacking without all 8 base probabilities
- **Multi-target timeline boundary**: base models trained on data < 2024-01-01, stacking on OOS 2024-01-02 to 2026-03-03
- **Multi-target signal derivation**: same canonical signal logic per target, each with own threshold (see Multi-Target table)

## GitNexus Usage (MANDATORY -- enforced by hooks)

**BEFORE using Grep, Glob, or Read on code files, ALWAYS call a GitNexus MCP tool first.** A PreToolUse prompt hook will block raw searches that skip GitNexus.

- **Exploring code**: `gitnexus_query({query: "topic"})` THEN Read specific files
- **Understanding a symbol**: `gitnexus_context({name: "functionName"})` for callers, callees, flows
- **Before editing**: `gitnexus_impact({target: "symbol"})` to check blast radius
- **Before committing**: `gitnexus_detect_changes({scope: "staged"})` to verify scope
- **Renaming**: `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})`
- **Custom graph queries**: `gitnexus_cypher({query: "MATCH ..."})`

Exceptions: Reading config files, non-code files (docs, CSV, JSON), or files the user explicitly provides a path for.

## Contabo VPS Ops (MANDATORY)

**ALL operations on the Contabo VPS MUST go through the `contabo-master` agent.** Spawn it via the Agent tool (`subagent_type: "contabo-master"`) for any SSH, Docker, rsync, systemd, firewall, or diagnostic work on the VPS (root, Singapore -- host in git-ignored `.claude/contabo.local.conf`). Never run raw `ssh root@...` or `sshpass` from the main session -- the agent enforces the credential protocol.

**Password handling**: prompt the user in chat at session start (or on first SSH op), pass through as `CONTABO_PASS` env var in the agent's Bash session only via `sshpass -e`. NEVER write the password to any file in the project (`.env`, `.claude/`, plan files, committed configs, logs, trading_logs, or anywhere under the repo). `unset CONTABO_PASS` when done. IP + user live in `.claude/contabo.local.conf` (git-ignored; template committed as `.claude/contabo.local.conf.example`).

## Commands (Quick Reference)

```bash
# ML Pipeline
python model_training/download_data.py                          # 1. Download data (incremental, appends new klines)
python model_training/download_data.py --full                   # 1. Full re-download (2400d, overwrites existing)
python model_training/download_backfill.py                      # 1b. Backfill Aug 2017 -> Sep 2019
python model_training/etl.py --force                            # 2. ETL full rebuild (regression on ALL data)
python model_training/etl.py --incremental                      # 2. ETL append new rows only
python model_training/encode_v10.py                             # 3. Encode 518 features
python model_training/train_v10_walkforward.py                  # 4. Walk-forward validation
python model_training/train_v10_production.py                   # 5. Train production models
python model_training/train_stacking_v3.py                      # 6. Stacking V3 (5 base + dual meta)
python model_training/train_stacking_v3_walkforward.py          # 7. V3 walk-forward (4-fold validation)
python model_training/optuna_hyperparam_search.py               # Optuna TPE search (100 trials, SQLite)
python model_training/optuna_meta_search.py                     # Optuna for meta-model params
python model_training/grid_search_meta_label.py                 # Grid search: binary meta-label params (3072 combos)
python model_training/compare_ml_candidates.py                  # Compare walk-forward results across candidates
python model_training/validate_v3_audit.py                      # V3 pipeline integrity audit (19 checks)
python model_training/train_multitarget_base.py                  # 8. Multi-target base (8 targets x 3 seeds, walk-forward + final)
python model_training/train_multitarget_base.py --targets T1 T3  # Specific targets only
python model_training/train_multitarget_stacking.py              # 9. Multi-target stacking (8 models)

# Live Trading
python trading_bot.py --dry-run | --testnet | --live

# Backfill & Dashboard
python backfill_predictions.py --hours 168 --threshold 0.50
python -m streamlit run backtest_dashboard.py
python main.py --mode dashboard

# Docker (4 containers: data + bot + telegram + dashboard)
docker compose up -d --build
docker compose logs --tail=20
# Query structured logs
grep '"TRADE_OPEN"' logs/bot/trading_bot.jsonl | jq .
```

## Agents (Opus, `.claude/agents/`)

| Agent | Specialty | Use for |
|-------|-----------|---------|
| `github-master` | Git, GitHub CLI, code review | Commits, PRs, branch ops, review |
| `gcloud-master` | GCE, gcloud CLI, remote ops | Deploy, SSH, instance management |
| `docker-master` | Docker, Compose, containers | Dockerfile, builds, compose, security |
| `linux-master` | DevOps, networking, sysadmin | Infra, firewall, auth, databases, logs |
| `contabo-master` | Contabo VPS, Ubuntu sysadmin, SSH ops, Docker on VPS | ALL Contabo VPS operations (multi-target stack) |
| `data-scientist` | ML audit, statistics, validation | Model validation, data exploration, pattern analysis |

## Reference Docs

| Topic | Path |
|-------|------|
| **Architecture & file tree** | `docs/architecture.md` |
| **ML Pipeline V10** (hyperparams, walk-forward, experiments) | `docs/ml-pipeline.md` |
| **Trading Bot** (risk engine, state, PnL) | `docs/trading-bot.md` |
| **Telegram Bot** (commands, push, data sources) | `docs/telegram-bot.md` |
| **Deployment** (GCE, Docker, env vars) | `docs/deployment.md` |
| **Contabo Multi-Target Deploy** (feat/multitarget-live, 32 models, router, parity test) | `docs/deployment-contabo.md` |
| **Pipeline Alignment** (3 encoding paths, invariants) | `docs/pipeline-alignment.md` |
| Signal Findings | `.claude/claude_manual_signal_finding.md` |
| Risk Engine Spec | `model_training/RISK_ENGINE.md` |
| Ops Runbook | `model_training/OPS_RUNBOOK.md` |
| V10 Results | `model_training/V10_EXPERIMENT_RESULTS.md` |
| SL/TP Sweep | `model_training/SLTP_SWEEP_RESULTS.md` |
| Experiment Retrospective | `model_training/EXPERIMENT_RETROSPECTIVE.md` |
| **Stacking V3 + Meta-Labeling** (best model: +321%, PASS 4/4) | `docs/ml-pipeline.md` (Stacking section) |
| **Grid Search Meta-Label** (3072 combos, 6 findings, candidates A/B/C) | `model_training/results_v10/grid_search_meta_label/` |
| **V3 Walk-Forward Results** (4-fold, meta-labeling winner) | `model_training/results_v10/stacking_v3_walkforward/` |
| **Stacking Next Steps** (meta-labeling +321%, experiments, priorities) | `model_training/NEXT_STEPS.md` |
| **Stacking Research** (meta-learner simplification, calibration, meta-labeling -- path for future model tuning) | `docs/stacking-research.md` |
| **Planned Features Phase H** (regime detection: BB width, ADX, ATR ratio, volume, Hurst) | `docs/to_add_features.md` |
| **GitNexus** (MCP tools, impact analysis, graph queries) | `docs/gitnexus.md` |
| Update Log | `docs/update_log.md` |
| V10 2yr OOS Audit | `model_training/V10_2YR_OOS_AUDIT.md` |
| **Multi-Target Config** (8 targets, params, timeline) | `model_training/multitarget_config.py` |
| **Multi-Target Base Results** (8/8 PASS, walk-forward) | `model_training/results_v10/multitarget/multitarget_base_results.json` |
| **Multi-Target Stacking Results** (8/8 PASS, OOS) | `model_training/results_v10/multitarget/stacking/multitarget_stacking_summary.json` |
| **Learnings Loop** (Reflect-Abstract-Rule entries, auto-loaded) | `.claude/LEARNINGS.md` |
| **Self-Improvement Loop** (hooks, skills, reflect/learn/consolidate flow, validation) | `docs/self-improvement-loop.md` |
