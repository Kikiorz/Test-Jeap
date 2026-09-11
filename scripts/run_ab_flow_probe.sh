#!/usr/bin/env bash
# Paired held-out flow-loss probe for the latent-space A/B: the JEPA-latent
# adapter and the VLM-latent adapter, both restored at the same step count and
# scored with the identical protocol (24 batches x 16 samples, paired against
# the alpha=0 / exact-base reference inside each batch).
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
STEP="${STEP:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCHES="${BATCHES:-24}"
BUDGETS="${BUDGETS:-0.05 0.0}"

run() {
  local config="$1" exp="$2" out="$3"
  PYTHONPATH="$ROOT/src" \
  CUDA_VISIBLE_DEVICES="${AB_GPUS:-2,3}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1 \
    "$ROOT/.venv/bin/python" -u scripts/probe_con1_budget_and_conditioning.py \
      --config "$config" --exp-name "$exp" --checkpoint-step "$STEP" \
      --batch-size "$BATCH_SIZE" --batches "$BATCHES" \
      --budgets $BUDGETS --out "$out"
}

run pi05_libero_con1_action_adapter_40k con1_action_adapter_ab_jepa_1k \
    /workspace/artifacts/con2/ab_flow_jepa.json
run pi05_libero_con1_action_adapter_vlm_40k con1_action_adapter_ab_vlm_1k \
    /workspace/artifacts/con2/ab_flow_vlm.json
