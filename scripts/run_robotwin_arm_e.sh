#!/usr/bin/env bash
# Arm E: the two levers that each beat base on their own, combined.
#
# At n=200, arm B (whole-prefix context) is -2.45% against base and arm D (head
# LR x5) is -2.29%, while arm A itself is -1.01%. B and D act through independent
# mechanisms - what the head reads, versus how fast it learns - and this is the
# only untested pairing that points at a Con1 variant which measurably improves
# the policy.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"
STEPS="${STEPS:-9001}"
KEEP_PERIOD="${KEEP_PERIOD:-3000}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"
LOG="${LOG:-/workspace/robotwin_arm_e.log}"

free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  echo "[$(date -u +%H:%M:%S)] refusing to start arm E: only ${free_gb}G free (< ${MIN_FREE_GB}G)" | tee -a "$LOG"
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] arm E starting: ${free_gb}G free, $STEPS steps, keep every $KEEP_PERIOD" | tee -a "$LOG"

PYTHONPATH="$ROOT/src" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "$ROOT/.venv/bin/python" -u scripts/train.py pi05_robotwin_con1_ctxheadlr_20k \
    --exp-name=robotwin_e_ctxheadlr \
    --checkpoint-base-dir=/workspace/artifacts/checkpoints \
    --no-wandb-enabled --num-workers=12 \
    --num-train-steps="$STEPS" --keep-period="$KEEP_PERIOD" \
    --batch-size=64 --con1-lr-multiplier=2 \
    >>"$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] arm E finished" | tee -a "$LOG"

# Probe it against the stored rows: same 200 batches, same seed, step 9000.
for _ in $(seq 1 20); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  (( busy == 0 )) && break
  echo "[$(date -u +%H:%M:%S)] $busy GPU(s) still busy; waiting" | tee -a "$LOG"
  sleep 30
done

echo "[$(date -u +%H:%M:%S)] probing arm E at step 9000" | tee -a "$LOG"
PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
  "$ROOT/.venv/bin/python" -u scripts/probe_con1_budget_and_conditioning.py \
    --config pi05_robotwin_con1_ctxheadlr_20k --exp-name robotwin_e_ctxheadlr \
    --checkpoint-step 9000 --batch-size 8 --batches 200 --budgets 0.05 \
    --out /workspace/artifacts/con2/hp/robotwin_ab_e.json >>"$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] arm E probed" | tee -a "$LOG"
