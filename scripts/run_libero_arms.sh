#!/usr/bin/env bash
# P2 on the rebuilt box: two arms, identical budget/data/seed, differing only by
# the new pieces (Con2 refinement + whole-prefix VLM context).
#   A = Con1 livecross only
#   B = complete algorithm (Con1 + Con2 + VLM context tokens)
set -euo pipefail

# XLA's GEMM autotuner spawns ptxas; the image ships a 1024 fd soft limit.
ulimit -n 65535 2>/dev/null || true

ROOT="${ROOT:-/dev/shm/ts_jepa}"
REPO="${REPO:-$ROOT/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
STEPS="${STEPS:-4500}"
BATCH="${BATCH:-64}"
LOG_ROOT="$ROOT/logs"
mkdir -p "$LOG_ROOT" "$ROOT/checkpoints"

train() {
  local config="$1" exp="$2"
  printf '[%s] training %s (%s)\n' "$(date -u +%H:%M:%S)" "$exp" "$config"
  PYTHONPATH="$REPO/src" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  HF_HUB_OFFLINE=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    "$PY" -u scripts/train.py "$config" \
      --exp-name="$exp" \
      --checkpoint-base-dir="$ROOT/checkpoints" \
      --no-wandb-enabled --num-workers=0 --overwrite \
      --num-train-steps="$STEPS" --keep-period=500 \
      --batch-size="$BATCH" --con1-lr-multiplier=2 \
      >>"$LOG_ROOT/${exp}.log" 2>&1
}

cd "$REPO"
train pi05_libero_con1_adapter_livecross_40k libero_a_con1
train pi05_libero_con1con2_ctx_40k      libero_b_full
printf '[%s] both arms done\n' "$(date -u +%H:%M:%S)"
