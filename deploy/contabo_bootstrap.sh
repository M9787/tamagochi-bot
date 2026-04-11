#!/usr/bin/env bash
# deploy/contabo_bootstrap.sh -- one-shot deploy of the multi-target
# stack to a fresh Contabo Ubuntu VPS.
#
# Usage:
#   deploy/contabo_bootstrap.sh <HOST> <USER> <SSH_KEY>
#
# Example:
#   deploy/contabo_bootstrap.sh 123.45.67.89 root ~/.ssh/contabo_ed25519
#
# Pre-conditions on the VPS (must be set up by the user once):
#   * Docker + Docker Compose plugin installed
#   * /opt/tamagochi-multitarget/ exists and is owned by $CONTABO_USER
#   * /opt/tamagochi-multitarget/.env populated with the env vars listed
#     in plan Phase E.3 (BINANCE_KEY, TELEGRAM_BOT_TOKEN, etc.)
#
# What this script does:
#   1. Verifies .env exists on the host
#   2. rsyncs the repo (minus heavy artifacts and secrets)
#   3. rsyncs the multi-target model directory (~517M of CatBoost files)
#   4. rsyncs the feature_matrix_v10.parquet bootstrap slice
#   5. docker compose up -d --build
#   6. Prints health-check commands

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 HOST USER SSH_KEY" >&2
    exit 1
fi

CONTABO_HOST="$1"
CONTABO_USER="$2"
SSH_KEY="$3"
REMOTE="/opt/tamagochi-multitarget"

if [[ ! -f "$SSH_KEY" ]]; then
    echo "ERROR: SSH key not found: $SSH_KEY" >&2
    exit 1
fi

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new $CONTABO_USER@$CONTABO_HOST"
RSYNC_SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

echo
echo "=========================================================="
echo "  Contabo multi-target deploy"
echo "  Host: $CONTABO_USER@$CONTABO_HOST"
echo "  Remote: $REMOTE"
echo "=========================================================="
echo

# 0. Pre-flight: .env on the host
echo "[0/5] Verifying $REMOTE/.env exists on host..."
if ! $SSH "test -f $REMOTE/.env"; then
    echo "ERROR: $REMOTE/.env missing on host." >&2
    echo "Create it first with the variables listed in plan Phase E.3." >&2
    exit 1
fi
echo "      OK"
echo

# 1. Repo (minus heavy artifacts, secrets, training results)
echo "[1/5] rsync repo to $REMOTE/ ..."
rsync -avz --delete --progress \
      -e "$RSYNC_SSH" \
      --exclude-from=deploy/contabo.rsync-exclude \
      ./ "$CONTABO_USER@$CONTABO_HOST:$REMOTE/"
echo

# 2. Multi-target model directory (517 MB of CatBoost files + JSONs).
# Excluded from the main rsync above, sent here so we can show progress
# on a single dedicated transfer.
echo "[2/5] rsync multi-target models (~517M)..."
$SSH "mkdir -p $REMOTE/model_training/results_v10/multitarget"
rsync -avz --progress \
      -e "$RSYNC_SSH" \
      --exclude='label_cache/' \
      --exclude='oos_probabilities/' \
      --exclude='oos_probabilities_all.parquet' \
      --exclude='feature_importance_T*.csv' \
      model_training/results_v10/multitarget/base_models \
      model_training/results_v10/multitarget/stacking \
      model_training/results_v10/multitarget/multitarget_base_results.json \
      "$CONTABO_USER@$CONTABO_HOST:$REMOTE/model_training/results_v10/multitarget/"
echo

# 3. feature_matrix_v10.parquet bootstrap slice. Mounted into the
# data_service container as a read-only bind via the compose file
# (./bootstrap -> /app/model_training/encoded_data:ro). Used by the
# multi-target predictor for cold-start backfill (288 base inferences).
echo "[3/5] rsync feature_matrix_v10.parquet bootstrap slice..."
$SSH "mkdir -p $REMOTE/bootstrap"
rsync -avz --progress -e "$RSYNC_SSH" \
      model_training/encoded_data/feature_matrix_v10.parquet \
      "$CONTABO_USER@$CONTABO_HOST:$REMOTE/bootstrap/"
echo

# 4. docker compose up -d --build
echo "[4/5] docker compose up -d --build (remote)..."
$SSH "cd $REMOTE && docker compose -f docker-compose.contabo.yml up -d --build"
echo

# 5. Post-deploy health echoes
echo "[5/5] Verify health:"
echo
echo "  $SSH 'docker ps --format \"table {{.Names}}\t{{.Status}}\"'"
echo "  $SSH 'docker logs tamagochi-data --tail 80'"
echo "  $SSH 'docker exec tamagochi-data ls -la /data/predictions'"
echo
echo "Dashboard tunnel (run on dev box):"
echo "  ssh -i $SSH_KEY -L 8501:localhost:8501 $CONTABO_USER@$CONTABO_HOST"
echo "  then open http://localhost:8501"
echo
echo "Done."
