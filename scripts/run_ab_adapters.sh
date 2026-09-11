#!/usr/bin/env bash
# Train the two latent-space variants of the Con1 action adapter back to back on
# the *same* 4-GPU topology, so their checkpoints restore under the same mesh
# (a 2-GPU checkpoint cannot be restored by the 4-device probe). Recipe is the
# one that produced the -1.95% held-out flow gain: frozen 16 layers,
# batch 64, con1-lr-multiplier 2, 1000 steps.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
STEPS="${STEPS:-1000}"

train() {
  local config="$1" exp="$2"
  printf '[%s] training %s\n' "$(date -u +%H:%M:%S)" "$exp"
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
      --num-train-steps="$STEPS" --keep-period=1 \
      --batch-size=64 --con1-lr-multiplier=2
}

train pi05_libero_con1_action_adapter_40k     con1_action_adapter_ab_jepa_1k
train pi05_libero_con1_action_adapter_vlm_40k con1_action_adapter_ab_vlm_1k
printf '[%s] done\n' "$(date -u +%H:%M:%S)"
