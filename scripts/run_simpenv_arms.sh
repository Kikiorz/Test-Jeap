#!/usr/bin/env bash
# SimpENV P2: two arms on the pi0.5 base with Bridge data, differing only by
# Con2. Arm A = Con1, arm B = Con1 + Con2. 8k steps each, FSDP over four GPUs.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
cd "$ROOT"
STEPS="${STEPS:-6000}"
WARM_STEPS="${WARM_STEPS:-2000}"
BATCH="${BATCH:-32}"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/lerobot_home
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

train() {
  local config="$1" exp="$2"
  printf '[%s] training %s (%s)\n' "$(date -u +%H:%M:%S)" "$exp" "$config"
  .venv/bin/python -u scripts/train.py "$config" \
    --exp-name="$exp" \
    --checkpoint-base-dir=/workspace/checkpoints \
    --overwrite --no-wandb-enabled \
    --num-train-steps="${WARM_STEPS_OVERRIDE:-$STEPS}" --batch-size="$BATCH" --fsdp-devices=4
}

# Phase 0: warm up the Con1 delta head on its own (everything else frozen).
WARM_STEPS_OVERRIDE="$WARM_STEPS" train pi05_bridge_con1_warm simpenv_head_warm
train pi05_bridge_con1     simpenv_armA_con1
train pi05_bridge_con1con2 simpenv_armB_con1con2
printf '[%s] both arms done\n' "$(date -u +%H:%M:%S)"
