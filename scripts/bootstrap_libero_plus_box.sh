#!/usr/bin/env bash
# Turnkey LIBERO-Plus run on a fresh GPU box.
#
# Chains the whole line that was assembled by hand on 93.91.156.102:
#   1. python deps for the openpi repo (uv sync)
#   2. the isolated LIBERO-Plus evaluation environment (py3.8 venv + 6 GB assets)
#   3. the 16 GB dataset, mirrored straight from the Hugging Face CDN
#   4. normalisation statistics, the 30k pi0.5 fine-tune, and the 32-shard sweep
#
# Every stage is idempotent, so re-running after an interruption resumes instead
# of starting over. Set HF_TOKEN for authenticated pulls if the CDN throttles.
#
# Usage:
#   REPO_DIR=/workspace/ts_JEPA_simpenv \
#   DATASET_ROOT=/workspace/libero_plus_lerobot \
#   bash scripts/bootstrap_libero_plus_box.sh
set -uo pipefail

ROOT="${REPO_DIR:-/workspace/ts_JEPA_simpenv}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/libero_plus_lerobot}"
BRANCH="${BRANCH:-feat/libero-plus-ft}"
EXP_NAME="${EXP_NAME:-libero_plus_30k}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-64}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "1/4 installing python dependencies"
cd "$ROOT"
uv sync --frozen || { log "uv sync failed"; exit 1; }

log "2/4 installing the LIBERO-Plus evaluation environment"
bash scripts/setup_libero_plus.sh install || { log "eval env install failed"; exit 1; }

log "3/4 mirroring the dataset into $DATASET_ROOT"
mkdir -p "$DATASET_ROOT"
.venv/bin/python -u scripts/fetch_libero_plus_dataset.py --root "$DATASET_ROOT" --workers 16 \
  || { log "dataset mirror failed"; exit 1; }

log "pointing HF_LEROBOT_HOME/local/libero_plus_lerobot at the mirror"
mkdir -p /workspace/lerobot_home/local
rm -f /workspace/lerobot_home/local/libero_plus_lerobot
ln -sfn "$DATASET_ROOT" /workspace/lerobot_home/local/libero_plus_lerobot

log "4/4 norm stats -> ${STEPS}-step fine-tune -> sweep"
HF_LEROBOT_HOME=/workspace/lerobot_home HF_HOME=/workspace/.hf_home \
VIDEOS_TOTAL=28694 EXP_NAME="$EXP_NAME" STEPS="$STEPS" BATCH="$BATCH" \
DATASET_ROOT="$DATASET_ROOT" \
  nohup bash scripts/auto_libero_plus_pipeline.sh >/workspace/auto_pipeline_bootstrap.log 2>&1 &

log "training chain launched; the evaluation chain follows automatically:"
log "  RUN_ID=pi05-plus-30k nohup bash scripts/auto_libero_plus_eval.sh >/workspace/auto_eval.log 2>&1 &"
