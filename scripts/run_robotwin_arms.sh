#!/usr/bin/env bash
# RoboTwin ablation: two arms, identical budget/data/seed, differing only by the
# new modules.
#   A = Con1 livecross (gated cross-attention + latent-free adapter)
#   B = A + Con2 refinement + whole-prefix VLM context
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"
STEPS="${STEPS:-4500}"
LOG="${LOG:-/workspace/robotwin_arms.log}"

train() {
  local config="$1" exp="$2"
  echo "[$(date -u +%H:%M:%S)] training $exp ($config)" | tee -a "$LOG"
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
      >>"$LOG" 2>&1
  echo "[$(date -u +%H:%M:%S)] finished $exp" | tee -a "$LOG"
}

train pi05_robotwin_con1_livecross_20k robotwin_a_con1
train pi05_robotwin_con1con2_ctx_20k  robotwin_b_full
echo "[$(date -u +%H:%M:%S)] both arms done" | tee -a "$LOG"
