#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT/eval_manifests/libero_plus_l45_200}"
RESULTS_PARENT="${RESULTS_PARENT:-/workspace/artifacts/evals/l45_200}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-jepawam60k-seed42-l45-200-v2}"
METHOD_RUN_ID="${METHOD_RUN_ID:-actr-step10000-seed42-l45-200-v2}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999}"
METHOD_CHECKPOINT="${METHOD_CHECKPOINT:-$ROOT/checkpoints/pi05_libero_actr_stage2/actr_stage2_seed42/10000}"
REPORT_PATH="${REPORT_PATH:-$RESULTS_PARENT/actr-step10000-vs-jepawam60k-seed42-l45-200-v2.json}"

mkdir -p "$RESULTS_PARENT"

echo "Running sealed L4/L5-200 baseline"
MANIFEST_ROOT="$MANIFEST_ROOT" \
RESULTS_ROOT="$RESULTS_PARENT/$BASELINE_RUN_ID" \
    "$ROOT/scripts/run_layout100_checkpoint_eval.sh" \
    "$BASELINE_RUN_ID" pi05_libero_vjepa_aux "$BASELINE_CHECKPOINT"

echo "Running sealed L4/L5-200 method"
MANIFEST_ROOT="$MANIFEST_ROOT" \
RESULTS_ROOT="$RESULTS_PARENT/$METHOD_RUN_ID" \
    "$ROOT/scripts/run_layout100_checkpoint_eval.sh" \
    "$METHOD_RUN_ID" pi05_libero_actr_stage2 "$METHOD_CHECKPOINT"

"$ROOT/.venv/bin/python" "$ROOT/scripts/compare_layout100_results.py" \
    "$RESULTS_PARENT/$BASELINE_RUN_ID" \
    "$RESULTS_PARENT/$METHOD_RUN_ID" \
    --output "$REPORT_PATH"

echo "L4/L5-200 experiment complete: $REPORT_PATH"
