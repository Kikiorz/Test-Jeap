#!/usr/bin/env bash
# Paired held-out flow probes for the two arms plus the exact base reference.
# The reference zeroes the Con1 correction at its output projection, so the
# comparison is against the policy the two arms started from.
set -euo pipefail
ulimit -n 65535 2>/dev/null || true

WORK="${WORK:-/workspace/ts_jepa}"
REPO="${REPO:-$WORK/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
STEP="${STEP:-4499}"
BATCHES="${BATCHES:-72}"

probe() {
  local config="$1" exp="$2" out="$3"; shift 3
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$out"
  PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 \
    "$PY" -u "$REPO/scripts/probe_con1_budget_and_conditioning.py" \
      --config "$config" --exp-name "$exp" --checkpoint-step "$STEP" \
      --batch-size 8 --batches "$BATCHES" --budgets 0.05 --out "$out" "$@"
}

cd "$REPO"
probe pi05_libero_con1_adapter_livecross_40k libero_a_con1 \
      "$WORK/artifacts/probe_base.json" --zero-correction
probe pi05_libero_con1_adapter_livecross_40k libero_a_con1 \
      "$WORK/artifacts/probe_arm_a.json"
probe pi05_libero_con1con2_ctx_40k libero_b_full \
      "$WORK/artifacts/probe_arm_b.json"
printf '[%s] probes done\n' "$(date -u +%H:%M:%S)"
