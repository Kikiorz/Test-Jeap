#!/usr/bin/env bash
# The same four-way comparison with a wider batch budget, so the 1-2% effects
# are resolved at more than two standard errors.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
BATCHES="${BATCHES:-72}"

probe() {
  local config="$1" exp="$2" step="$3" out="$4"; shift 4
  # Restoring the 2B-parameter policy and then building the probe's model graph
  # sits right at the 32 GB ceiling, so the first attempt occasionally dies with
  # a host-side OOM during graph construction. Retry rather than lose the run.
  for attempt in 1 2 3 4; do
    printf '[%s] %s (attempt %d)\n' "$(date -u +%H:%M:%S)" "$out" "$attempt"
    if PYTHONPATH="$ROOT/src" \
       CUDA_VISIBLE_DEVICES=0,1,2,3 \
       XLA_PYTHON_CLIENT_PREALLOCATE=false \
       LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
       NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1 \
         "$ROOT/.venv/bin/python" -u scripts/probe_con1_budget_and_conditioning.py \
           --config "$config" --exp-name "$exp" --checkpoint-step "$step" \
           --batch-size 8 --batches "$BATCHES" --budgets 0.05 --out "$out" "$@"; then
      return 0
    fi
    sleep 20
  done
  return 1
}

JEPA_CFG=pi05_libero_con1_action_adapter_40k
VLM_CFG=pi05_libero_con1_action_adapter_vlm_40k
LO_JEPA=pi05_libero_con1_latentonly_jepa_40k
LO_VLM=pi05_libero_con1_latentonly_vlm_40k

probe "$JEPA_CFG" con1_action_adapter_ab_jepa_1k 999 \
    /workspace/artifacts/con2/wide_base.json --zero-correction
probe "$JEPA_CFG" con1_action_adapter_ab_jepa_1k 999 \
    /workspace/artifacts/con2/wide_adapter_jepa.json
probe "$LO_JEPA" con1_latentonly_jepa_v2 1499 \
    /workspace/artifacts/con2/wide_latentonly_jepa.json
probe "$LO_VLM" con1_latentonly_vlm_v2 1499 \
    /workspace/artifacts/con2/wide_latentonly_vlm.json
