#!/usr/bin/env bash
# P2: two arms, identical budget/data/seed, differing only by the Con2
# refinement module. Arm A is Con1 livecross; arm B adds Con2.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
STEPS="${STEPS:-5000}"

train() {
  local config="$1" exp="$2"
  printf '[%s] training %s (%s)\n' "$(date -u +%H:%M:%S)" "$exp" "$config"
  PYTHONPATH="$ROOT/src" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    "$ROOT/.venv/bin/python" -u scripts/train.py "$config" \
      --exp-name="$exp" \
      --checkpoint-base-dir=/workspace/artifacts/checkpoints \
      --no-wandb-enabled --num-workers=0 --overwrite \
      --num-train-steps="$STEPS" --keep-period=1000 \
      --batch-size=64 --con1-lr-multiplier=2 \
      >> "/workspace/artifacts/con2/${exp}.log" 2>&1
}

train pi05_libero_con1_adapter_livecross_40k arm_a_con1_5000
train pi05_libero_con1con2_livecross_40k  arm_b_con1con2_5000
printf '[%s] both arms done\n' "$(date -u +%H:%M:%S)"
