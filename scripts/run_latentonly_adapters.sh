#!/usr/bin/env bash
# Train the latent-only Con1 variants: the latent-free action adapter branch is
# removed, leaving the cross-attention over the predicted delta as the only
# added capacity, with a small non-zero output init so the key/value/query
# projections receive gradient from step 0. Same recipe as the adapter A/B.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
STEPS="${STEPS:-1500}"

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

train pi05_libero_con1_latentonly_jepa_40k con1_latentonly_jepa_v2
train pi05_libero_con1_latentonly_vlm_40k  con1_latentonly_vlm_v2
printf '[%s] done\n' "$(date -u +%H:%M:%S)"
