# Contabo Multi-Target Live Deployment

Parallel deployment of the **multi-target** CatBoost stack (24 base + 8 stacking models, 32 total CBM files, ~517MB) on a fresh Contabo Ubuntu VPS. Runs alongside the GCE V3 single-target stack without touching `master`. Branch: `feat/multitarget-live` (commit `15ffab5`).

## Architecture

4-service stack, near-verbatim copy of `docker-compose.yml` with a minimal diff for multi-target flags and a separate named volume `multitarget_data`.

```
Binance klines (11 TFs)
  -> data_service.IncrementalEncoder (reused, unchanged) -- 518 V10 features
  -> MultiTargetPredictor
       |-> 24 base CatBoost models (base_model_T{1-8}_s{42,123,777}.cbm) -- avg across 3 seeds
       |   (push per-target probs into 289-row ring buffer: 288 history + 1 current)
       |-> build_stacking_meta_features() -- 258 meta features (shared with training)
       |-> 8 stacking CatBoost models (stacking_model_T{1-8}.cbm)
  -> MultiTargetRouter
       |-> uniform 0.80 threshold per target
       |-> require >=2 firing targets sharing direction
       |-> winner = max(firing, key=confidence) -- SL/TP/max_hold from winning target's config
  -> /data/predictions/predictions_multitarget.csv (V3-compatible superset schema)
  -> trading_bot --multitarget
       |-> MultiTradeManager(lock_mode=True) -- blocks concurrent opens
       |-> per-trade SL/TP/max_hold from router decision
```

**Critical**: the full pipeline runs all 32 models every cycle. The 8 stacking models consume 258 meta-features built from 24 base-model outputs -- they cannot run in isolation. Shipping stacking-only would be a deploy mistake.

## New modules (feat/multitarget-live)

| File | Purpose |
|---|---|
| `core/multitarget_feature_builder.py` | Extracted 258-feature stacking meta builder. `EXPECTED_META_FEATURE_COUNT = 258`. Training imports via `build_interaction_features` alias shim. |
| `core/multitarget_predictor.py` | Loads 24 base + 8 stacking, runs full flow per cycle. 289-row ring buffer per target, atomic state persistence (3 backups), eager cold-start from `feature_matrix_v10.parquet`. |
| `core/multitarget_router.py` | `UNIFORM_THRESHOLD = 0.80`, 5 reasons (FIRE / WARMING / NO_FIRING / NOT_ENOUGH_TARGETS / NO_AGREEMENT). No cooldown. |
| `docker-compose.contabo.yml` | Minimal diff from GCE compose: `COPY_V3_MODELS=0` / `COPY_MT_MODELS=1`, `TAMAGOCHI_MULTITARGET=1` / `TAMAGOCHI_LOAD_V3=0`, `TELEGRAM_PREDICTIONS_CSV=/data/predictions/predictions_multitarget.csv`, dashboard bound to `127.0.0.1:8501`, `./bootstrap:/app/model_training/encoded_data:ro`. |
| `deploy/contabo_bootstrap.sh` | Rsync repo + 32 CBMs + parquet slice, then `docker compose up -d --build`. `set -euo pipefail`. Requires pre-created `/opt/tamagochi-multitarget/.env`. |
| `deploy/contabo.rsync-exclude` | Skip `.git`, `label_cache/`, `oos_probabilities/`, local state dirs, `.env`. |
| `tests/test_multitarget_parity.py` | Bit-level parity contract: `build_stacking_meta_features` output == training-code forward pass at ATOL=1e-6. **Deploy blocker if red.** Last run: PASS (29.92s). |

## Touched files (additive only)

- `data_service/layers.py` -- `TAMAGOCHI_LOAD_V3` gate (line 76), `TAMAGOCHI_MULTITARGET` gate (line 115), MT branch in `_predict_and_append` (line 651), new `_append_multitarget_row` writer (line 659). V3-compatible CSV superset means telegram formatters need zero changes.
- `trading/multi_trade_manager.py` -- `lock_mode` kwarg (line 34), blocks concurrent opens when set (line 108), per-trade SL/TP/max_hold overrides. State reload never restores `lock_mode` from disk (constructor arg always wins).
- `trading_bot.py` -- `read_latest_multitarget` (line 622), `use_multitarget` branch (line 748, skips V3 `PositionManager` path), `--multitarget` CLI flag (line 1235).
- `telegram_service/readers.py` -- `_predictions_path()` reads `TELEGRAM_PREDICTIONS_CSV` env var; hardcoded paths replaced.
- `Dockerfile` + `data_service/Dockerfile` -- `ARG COPY_V3_MODELS=1 / COPY_MT_MODELS=0` build args.
- `model_training/train_multitarget_stacking.py` -- deleted local builder (~145 lines), replaced with `from core.multitarget_feature_builder import build_stacking_meta_features as build_interaction_features`. **Naming gotcha**: `build_interaction_features` already exists in `encode_v10.py` as an unrelated helper -- the extracted function is renamed to avoid collision.

## Environment flags

| Flag | Default (GCE) | Contabo |
|---|---|---|
| `TAMAGOCHI_LOAD_V3` | `1` | `0` |
| `TAMAGOCHI_MULTITARGET` | `0` | `1` |
| `MULTITARGET_ROOT` | unset | `/app/model_training/results_v10/multitarget` |
| `TELEGRAM_PREDICTIONS_CSV` | unset (defaults to `/data/predictions/predictions.csv`) | `/data/predictions/predictions_multitarget.csv` |

## Locked decisions

| Item | Choice |
|---|---|
| Mode | `--dry-run` paper trading |
| Balance / sizing | $1000 start, $10 margin/trade, 20x leverage |
| Routing | Highest-prob winner, min agreement = 2, all firing targets share direction |
| Threshold | Uniform 0.80 for all 8 targets (no per-target sweep) |
| Concurrency | Lock mode: one open trade at a time total |
| SL/TP source | Winning target's `TARGET_CONFIGS[T]` |
| Cooldown | **None, ever** -- fire on next FIRE signal after close |
| Dashboard | 4-service parity, bound `127.0.0.1:8501`, access via `ssh -L 8501:localhost:8501 contabo` |

## Deploy flow (Phase H)

1. **H.0** -- Collect VPS IP, SSH user, SSH key path, Binance keys, Ubuntu version. **DONE**
2. **H.1** -- Install Docker on fresh VPS via one-shot SSH heredoc (`apt` + docker-ce + compose plugin). **DONE 2026-04-11**: Docker 29.4.0 + Compose v5.1.2 on Ubuntu 24.04.4 LTS (kernel 6.8, 191G free, 12G RAM, 6 vCPU), `/etc/docker/daemon.json` log rotation (`json-file` 10m x 3), `/opt/tamagochi-multitarget` created root:root, `hello-world` smoke passed.
3. **H.2** -- Create `/opt/tamagochi-multitarget/.env` via SSH heredoc (never echoed to chat). **PENDING .env values**
4. **H.3** -- Rsync repo + 32 CBMs + parquet, then `docker compose up -d --build` (~8-12 min including 517MB models).
5. **H.4** -- Phase F verification (see below).
6. **H.5** -- 24h soak watchdog. Merge `feat/multitarget-live` to master only after soak passes.

**Windows/Git Bash deploy note**: `sshpass` is not available on the Windows host this project develops on. The credential protocol is still honored via a temp helper at `C:\tmp\contabo_ssh.py` (paramiko, reads `CONTABO_HOST`/`CONTABO_USER`/`CONTABO_PASS` from env, command from stdin, nothing written to the repo). Usage: `echo '<cmd>' | python /c/tmp/contabo_ssh.py`. `unset CONTABO_PASS` when done. The `contabo-master` agent subagent type is not registered in the current CLI build -- the main session executes the same protocol inline instead.

## Verification checklist (run after bootstrap)

1. `docker ps --format "table {{.Names}}\t{{.Status}}"` -- 4 services Up + healthy.
2. `docker exec tamagochi-data tail -n 1 /data/predictions/predictions_multitarget.csv` -- fresh row within 10 min.
3. `docker logs tamagochi-data | grep "Loaded model" | wc -l` -- expect **32**.
4. `docker exec tamagochi-data jq '.buffers.T1 | length' /data/multitarget_state.json` -- expect **289**.
5. `docker logs tamagochi-data | grep "UNIFORM_THRESHOLD"` -- 0.80.
6. Lock mode: `jq '.multi_trade.open_trades | length'` never exceeds 1.
7. Telegram `/start` in new channel -> welcome; `/status` -> live multitarget prediction.
8. Dashboard via `ssh -L 8501:localhost:8501 contabo` -> http://localhost:8501.
9. Parity test (pre-deploy, blocking): `pytest tests/test_multitarget_parity.py`.
10. 24h soak: no unhandled exceptions, predictions CSV growing 1 row/5min.

## Rollback

- **GCE**: unchanged. Master untouched. No-op rollback.
- **Contabo**: `ssh contabo "cd /opt/tamagochi-multitarget && docker compose -f docker-compose.contabo.yml down -v"` -- drops containers and `multitarget_data` volume.
- **Branch**: `feat/multitarget-live` can be abandoned without merging.

## Known risks

- Stacking feature parity (mitigated by `tests/test_multitarget_parity.py` -- deploy blocker).
- Cold-start walltime 60-120s on 4 vCPU (288 x 24 base calls, logged every 32 rows).
- Binance mainnet key on Contabo: `--dry-run` CLI flag enforced in compose command; no code path reads `BINANCE_KEY` outside `BinanceFuturesExecutor`.
- Dashboard on 127.0.0.1 only; SSH tunnel required. No public port 8501.
- Lock mode state drift: constructor always overwrites loaded value.
