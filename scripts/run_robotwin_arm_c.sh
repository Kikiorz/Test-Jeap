#!/usr/bin/env bash
# Arm C of the RoboTwin ablation: arm A's livecross coupling plus action
# conditioning of the anchored-delta head.
#
# Why this arm exists: at 4,500 steps arm A already moved the action flow loss
# by -11.8% while `con1_delta_nmse` sat at 0.88 (arm B, with the whole VLM
# prefix, reached 0.78). A head that explains ~15% of the future-latent
# variance leaves Con2 almost nothing to refine, so the lever that can make
# Con2 matter is a delta head with a real signal. Feeding the demonstrated
# action chunk in makes the target "what this chunk causes" instead of "what
# happens next on average". This is the only change relative to arm A.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"
# 12k keeps every arm inside the disk budget while staying well past the point
# where the head-only run had already converged. Arm A and B carry a 12k
# checkpoint, so the final probe compares every arm at that step.
STEPS="${STEPS:-12000}"
KEEP_PERIOD="${KEEP_PERIOD:-3000}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
LOG="${LOG:-/workspace/robotwin_arm_c.log}"

free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  echo "[$(date -u +%H:%M:%S)] refusing to start arm C: only ${free_gb}G free (< ${MIN_FREE_GB}G)" | tee -a "$LOG"
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] arm C starting: ${free_gb}G free, target $STEPS steps, keep every $KEEP_PERIOD" | tee -a "$LOG"

PYTHONPATH="$ROOT/src" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "$ROOT/.venv/bin/python" -u scripts/train.py pi05_robotwin_con1_actcond_20k \
    --exp-name=robotwin_c_actcond \
    --checkpoint-base-dir=/workspace/artifacts/checkpoints \
    --no-wandb-enabled --num-workers=12 \
    --num-train-steps="$STEPS" --keep-period="$KEEP_PERIOD" \
    --batch-size=64 --con1-lr-multiplier=2 \
    >>"$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] arm C finished" | tee -a "$LOG"
