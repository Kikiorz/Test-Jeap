#!/usr/bin/env bash
# Arm D: arm A's livecross coupling with the anchored-delta head trained on its
# own schedule (5x the model LR, i.e. 1e-4).
#
# Why: `scripts/train_robotwin_head_cpu.py` trains a head with the same
# architecture, on the same cache, alone, and reaches a held-out delta NMSE of
# 0.737 in 3,000 steps at lr 1e-4. The jointly trained head sits at 0.82 after
# ~8,000 GPU steps and is still falling, i.e. it is undertrained because it
# shares the model's effective 2e-5. Every arm so far carries that handicap, so
# the Con1 action gain measured so far is a lower bound. Exactly one variable
# changes relative to arm A.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"
STEPS="${STEPS:-12000}"
KEEP_PERIOD="${KEEP_PERIOD:-3000}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"
LOG="${LOG:-/workspace/robotwin_arm_d.log}"

free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  echo "[$(date -u +%H:%M:%S)] refusing to start arm D: only ${free_gb}G free (< ${MIN_FREE_GB}G)" | tee -a "$LOG"
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] arm D starting: ${free_gb}G free, target $STEPS steps, keep every $KEEP_PERIOD" | tee -a "$LOG"

PYTHONPATH="$ROOT/src" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "$ROOT/.venv/bin/python" -u scripts/train.py pi05_robotwin_con1_headlr_20k \
    --exp-name=robotwin_d_headlr \
    --checkpoint-base-dir=/workspace/artifacts/checkpoints \
    --no-wandb-enabled --num-workers=12 \
    --num-train-steps="$STEPS" --keep-period="$KEEP_PERIOD" \
    --batch-size=64 --con1-lr-multiplier=2 \
    >>"$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] arm D finished" | tee -a "$LOG"
