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

**Deployment models (32 files, ~516MB total; full `results_v10/multitarget/` dir ≈760MB including label cache and OOS parquets)**:
- Base: `results_v10/multitarget/base_models/base_model_T{1-8}_s{42,123,777}.cbm` (24 files, 511MB)
- Stacking: `results_v10/multitarget/stacking/stacking_model_T{1-8}.cbm` (8 files, 4.9MB)

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
- **Predictions/features dedup** -- `append_rows_atomic()` supports `dedup_col` param; predictions, features, and multitarget CSVs all deduped by `time` to prevent silent double-writes on encoder-state reset or gap-backfill replay
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

## Contabo VPS Ops (MANDATORY)

**ALL operations on the Contabo VPS MUST go through the `contabo-master` agent.** Spawn it via the Agent tool (`subagent_type: "contabo-master"`) for any SSH, Docker, rsync, systemd, firewall, or diagnostic work on the VPS (root, Singapore -- host in git-ignored `.claude/contabo.local.conf`). Never run raw `ssh root@...` or `sshpass` from the main session -- the agent enforces the credential protocol.

**Password handling**: prompt the user in chat at session start (or on first SSH op), pass through as `CONTABO_PASS` env var in the agent's Bash session only via `sshpass -e`. NEVER write the password to any file in the project (`.env`, `.claude/`, plan files, committed configs, logs, trading_logs, or anywhere under the repo). `unset CONTABO_PASS` when done. IP + user live in `.claude/contabo.local.conf` (git-ignored; template committed as `.claude/contabo.local.conf.example`).

## Commands (Quick Reference)

```bash
# ML Pipeline
python model_training/download_data.py                          # 1. Download data (incremental, appends new klines)
python model_training/download_data.py --full                   # 1. Full re-download (2400d, overwrites existing)
python model_training/etl.py --force                            # 2. ETL full rebuild (regression on ALL data)
python model_training/etl.py --incremental                      # 2. ETL append new rows only
python model_training/encode_v10.py                             # 3. Encode 518 features
python model_training/train_v10_walkforward.py                  # 4. Walk-forward validation (single-target research)
python model_training/train_v10_production.py                   # 5. Train single-target production model
python model_training/train_multitarget_base.py                 # 6. Multi-target base (8 targets x 3 seeds)
python model_training/train_multitarget_base.py --targets T1 T3 # Specific targets only
python model_training/train_multitarget_stacking.py             # 7. Multi-target stacking (8 models, PRODUCTION on Contabo)
# Note: stacking_v3, optuna_*, grid_search_*, compare_*, validate_v3_audit are local research scripts (gitignored)

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
| Update Log | `docs/update_log.md` |
| V10 2yr OOS Audit | `model_training/V10_2YR_OOS_AUDIT.md` |
| **Multi-Target Config** (8 targets, params, timeline) | `model_training/multitarget_config.py` |
| **Multi-Target Base Results** (8/8 PASS, walk-forward) | `model_training/results_v10/multitarget/multitarget_base_results.json` |
| **Multi-Target Stacking Results** (8/8 PASS, OOS) | `model_training/results_v10/multitarget/stacking/multitarget_stacking_summary.json` |
| **Self-Improvement Loop** (hooks, skills, reflect/learn/consolidate flow, validation) | `docs/self-improvement-loop.md` |

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **tamagochi-bot** (840 symbols, 2451 relationships, 68 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/tamagochi-bot/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/tamagochi-bot/context` | Codebase overview, check index freshness |
| `gitnexus://repo/tamagochi-bot/clusters` | All functional areas |
| `gitnexus://repo/tamagochi-bot/processes` | All execution flows |
| `gitnexus://repo/tamagochi-bot/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
