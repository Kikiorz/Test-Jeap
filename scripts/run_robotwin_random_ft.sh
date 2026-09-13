#!/usr/bin/env bash
# Fine-tune the released pi0.5 base on the 20-task RoboTwin 2.0 randomized set.
#
# Prerequisites:
#   bash scripts/build_robotwin_random20.sh         (dataset)
#   python scripts/compute_norm_stats.py --config-name=pi05_robotwin_random20_ft
#
# Env knobs: STEPS, BATCH, FSDP, NUM_WORKERS, CUDA_VISIBLE_DEVICES, EXP, LOG,
#            CKPT_DIR, SAVE_INTERVAL, KEEP_PERIOD, ROOT.
set -euo pipefail

ROOT="${ROOT:-/workspace/robotwin_ws}"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-128}"
FSDP="${FSDP:-4}"
NUM_WORKERS="${NUM_WORKERS:-48}"
EXP="${EXP:-robotwin_random20_ft}"
LOG="${LOG:-/workspace/robotwin_random_ft.log}"
CKPT_DIR="${CKPT_DIR:-/workspace/artifacts/checkpoints_robotwin_random}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
# A checkpoint is 31 GB (12 GB params + 19 GB optimizer state) and the box has
# ~35 GB spare, so nothing is retained beyond the newest one.
KEEP_PERIOD="${KEEP_PERIOD:-100000}"
# RESUME=1 continues from the newest checkpoint in CKPT_DIR instead of wiping it
# (used after moving the run to another box).
RESUME="${RESUME:-0}"
HF_HOME="${HF_HOME:-/workspace/.hf_home}"
NCCL_PRELOAD="${NCCL_PRELOAD:-/usr/lib/x86_64-linux-gnu/libnccl.so.2}"
# The interpreter can live outside the repo: on a box whose scratch filesystem is
# mounted noexec (vast.ai's /dev/shm), the venv has to sit on the root disk while
# the data and checkpoints stay on the fast scratch mount.
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ "$RESUME" == "1" ]]; then
  RESUME_FLAG=(--resume)
else
  RESUME_FLAG=(--overwrite)
fi
if [[ -f "$NCCL_PRELOAD" ]]; then
  PRELOAD_ENV=(LD_PRELOAD="$NCCL_PRELOAD")
else
  PRELOAD_ENV=()
fi

cd "$ROOT"
mkdir -p "$CKPT_DIR"

echo "[$(date -u +%H:%M:%S)] pi05_robotwin_random20_ft exp=${EXP} steps=${STEPS} batch=${BATCH} fsdp=${FSDP}" | tee -a "$LOG"

PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"${PRELOAD_ENV[@]}" \
HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "$PYTHON" -u scripts/train.py pi05_robotwin_random20_ft \
    --exp-name="$EXP" \
    --checkpoint-base-dir="$CKPT_DIR" \
    --no-wandb-enabled \
    --num-workers="$NUM_WORKERS" \
    --batch-size="$BATCH" \
    --fsdp-devices="$FSDP" \
    --num-train-steps="$STEPS" \
    --save-interval="$SAVE_INTERVAL" \
    --keep-period="$KEEP_PERIOD" \
    "${RESUME_FLAG[@]}" \
    >>"$LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] training finished" | tee -a "$LOG"
