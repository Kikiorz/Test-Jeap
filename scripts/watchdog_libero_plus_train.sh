#!/usr/bin/env bash
# Keep the shared-box LIBERO-Plus fine-tune alive without supervision.
#
# The run shares four GPUs with another training job, so it only has ~17 GB of
# headroom per device and the host overlay is ~100% full. Both of those have
# already killed it once (RESOURCE_EXHAUSTED, ENOSPC). Everything heavy lives on
# /dev/shm, which makes a restart cheap - it re-reads the base checkpoint and
# the precomputed norm stats and continues from step 0.
#
# Usage: nohup bash scripts/watchdog_libero_plus_train.sh > /workspace/train_watchdog.log 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
EXP_NAME="${EXP_NAME:-libero_plus_30k}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-32}"
CKPT_BASE="${CKPT_BASE:-/dev/shm/ckpt_libero}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
LOG=/workspace/train_libero_plus.log

cd "$ROOT"
restarts=0

launch() {
  printf '[%s] launching %s-step fine-tune (restart %s/%s)\n' \
    "$(date -u +%H:%M:%S)" "$STEPS" "$restarts" "$MAX_RESTARTS"
  PYTHONPATH="$ROOT/src" HF_HOME=/workspace/.hf_home HF_LEROBOT_HOME=/workspace/lerobot_home \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OPENPI_SAVE_OPT_STATE=0 \
    nohup .venv/bin/python -u scripts/train.py pi05_libero_plus \
      --exp-name="$EXP_NAME" \
      --checkpoint-base-dir="$CKPT_BASE" \
      --overwrite --no-wandb-enabled \
      --num-train-steps="$STEPS" --batch-size="$BATCH" --fsdp-devices=4 \
      --save-interval=5000 --keep-period=30000 \
      >"$LOG" 2>&1 &
}

while :; do
  if [ -d "$CKPT_BASE/pi05_libero_plus/$EXP_NAME/$((STEPS - 1))" ]; then
    printf '[%s] final checkpoint present, watchdog done\n' "$(date -u +%H:%M:%S)"
    exit 0
  fi
  if ! pgrep -f "scripts/train.py pi05_libero_plus" >/dev/null; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      printf '[%s] training died %s times, giving up\n' "$(date -u +%H:%M:%S)" "$restarts"
      exit 1
    fi
    printf '[%s] training is not running, restarting\n' "$(date -u +%H:%M:%S)"
    restarts=$((restarts + 1))
    sleep 60
    launch
  fi
  sleep 300
done
