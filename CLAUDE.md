# CLAUDE.md

## Project

Crypto trading signal system for BTCUSDT: Binance klines (11 TFs, 6yr) -> rolling regression (5 windows) -> CatBoost 3-class ML (508 features, V10) -> live trading on Binance Futures via Docker on GCE.

## Core Algorithm (analysis.py)

`iterative_regression(df, window_size)` -- sliding window of `2*window_size` prices using `sqrt(price)`, `linregress` on each half -> **slope_b** (historical), **slope_f** (current), **angle** (divergence), **acceleration** (rate of change).

## Signal Theory

- **Crossing** = primary trigger (angle convergence/cross/divergence between young+elder windows)
- **Reversal** = secondary confirmation (5-point angle pattern)
- **Direction** (slope_f) = context (LONG/SHORT territory)
- **Acceleration** = quality gate (reject CLOSE/VERY_CLOSE GMM zones)
- **Philosophy**: Precision >> Recall. Target 60%+ precision. SL=2%, TP=4% (break-even=33.3%). Only ~10% of moments tradeable. Labels: NO_TRADE=72.7%, SHORT=13.7%, LONG=13.6%.

## MANDATORY: Model Evaluation Checklist

**Every training script MUST include**: ROC AUC (per-class + macro), ROC/PR curve PNGs, per-class P/R/F1, confusion matrix PNG, trade precision at threshold, profit factor + max drawdown + Sharpe, equity curve PNG, trade log CSV. **No exceptions.**

## Key Config (config.py)

- TFs: `3D,1D,12H,8H,6H,4H,2H,1H,30M,15M,5M` | Windows: `[30,60,100,120,160]`
- SL=2%, TP=4%, max_hold=288, threshold=0.75, bootstrap=1400
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
- **State backup rotation** -- `feature_state.json` keeps 3 backups for corruption recovery
- TF-native lags: MUST shift BEFORE merge_asof

## Commands (Quick Reference)

```bash
# ML Pipeline
python model_training/download_data.py                          # 1. Download 6yr data
python model_training/etl.py --start 2020-01-01 --end 2026-02-15  # 2. ETL
python model_training/encode_v10.py                             # 3. Encode 508 features
python model_training/train_v10_walkforward.py                  # 4. Walk-forward validation
python model_training/train_v10_production.py                   # 5. Train production models

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
| `data-scientist` | ML audit, statistics, validation | Model validation, data exploration, pattern analysis |

## Reference Docs

| Topic | Path |
|-------|------|
| **Architecture & file tree** | `docs/architecture.md` |
| **ML Pipeline V10** (hyperparams, walk-forward, experiments) | `docs/ml-pipeline.md` |
| **Trading Bot** (risk engine, state, PnL) | `docs/trading-bot.md` |
| **Telegram Bot** (commands, push, data sources) | `docs/telegram-bot.md` |
| **Deployment** (GCE, Docker, env vars) | `docs/deployment.md` |
| **Pipeline Alignment** (3 encoding paths, invariants) | `docs/pipeline-alignment.md` |
| Signal Findings | `.claude/claude_manual_signal_finding.md` |
| Risk Engine Spec | `model_training/RISK_ENGINE.md` |
| Ops Runbook | `model_training/OPS_RUNBOOK.md` |
| V10 Results | `model_training/V10_EXPERIMENT_RESULTS.md` |
| SL/TP Sweep | `model_training/SLTP_SWEEP_RESULTS.md` |
| Experiment Retrospective | `model_training/EXPERIMENT_RETROSPECTIVE.md` |
