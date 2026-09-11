#!/usr/bin/env bash
# Paired held-out flow probe for the latent-only Con1 variants, using the same
# protocol (24 batches x 8 samples, same seed) as the base and adapter probes so
# all four numbers are directly comparable.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
STEP="${STEP:-1499}"

run() {
  local config="$1" exp="$2" out="$3"
  printf '[%s] probing %s\n' "$(date -u +%H:%M:%S)" "$exp"
  PYTHONPATH="$ROOT/src" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1 \
    "$ROOT/.venv/bin/python" -u scripts/probe_con1_budget_and_conditioning.py \
      --config "$config" --exp-name "$exp" --checkpoint-step "$STEP" \
      --batch-size 8 --batches 24 --budgets 0.05 --out "$out"
}

run pi05_libero_con1_latentonly_jepa_40k con1_latentonly_jepa_v2 \
    /workspace/artifacts/con2/ab_flow_latentonly_jepa.json
run pi05_libero_con1_latentonly_vlm_40k con1_latentonly_vlm_v2 \
    /workspace/artifacts/con2/ab_flow_latentonly_vlm.json
