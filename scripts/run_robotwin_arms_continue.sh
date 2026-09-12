#!/usr/bin/env bash
# Continue both RoboTwin arms from their existing checkpoints. The 4,500-step
# A/B left Con2 and the whole-prefix context at +0.10% (t = 0.16) -- i.e. inert
# -- and the RoboTwin Con1 head starts from random init (unlike the LIBERO head,
# which was pre-trained for 20k steps), so the natural next test is whether those
# branches matter once the head and the action expert have actually converged.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"
STEPS="${STEPS:-15000}"
LOG="${LOG:-/workspace/robotwin_arms_continue.log}"

continue_train() {
  local config="$1" exp="$2"
  echo "[$(date -u +%H:%M:%S)] continuing $exp to $STEPS steps" | tee -a "$LOG"
  PYTHONPATH="$ROOT/src" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    "$ROOT/.venv/bin/python" -u scripts/train.py "$config" \
      --exp-name="$exp" \
      --checkpoint-base-dir=/workspace/artifacts/checkpoints \
      --no-wandb-enabled --num-workers=12 --resume \
      --num-train-steps="$STEPS" --keep-period=1000 \
      --batch-size=64 --con1-lr-multiplier=2 \
      >>"$LOG" 2>&1
  echo "[$(date -u +%H:%M:%S)] finished $exp" | tee -a "$LOG"
}

continue_train pi05_robotwin_con1_livecross_20k robotwin_a_con1
continue_train pi05_robotwin_con1con2_ctx_20k  robotwin_b_full
echo "[$(date -u +%H:%M:%S)] both arms continued" | tee -a "$LOG"
