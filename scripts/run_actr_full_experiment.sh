#!/usr/bin/env bash
set -euo pipefail

# Complete the sealed Con1 experiment after the supervised training pipeline.
# This wrapper is deliberately resumable: the rollout workers journal every
# episode and run_layout100_checkpoint_eval.sh starts them with RESUME=1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TRAIN_PROGRAM="${TRAIN_PROGRAM:-actr-train-pipeline}"
POLL_SECONDS="${POLL_SECONDS:-45}"
RUN_TAG="${RUN_TAG:-seed42-layout100}"

BASELINE_CONFIG="${BASELINE_CONFIG:-pi05_libero_vjepa_aux}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999}"
METHOD_CONFIG="${METHOD_CONFIG:-pi05_libero_actr_stage2}"
METHOD_CHECKPOINT="${METHOD_CHECKPOINT:-$ROOT/checkpoints/pi05_libero_actr_stage2/actr_stage2_seed42/19999}"

RESULTS_PARENT="${RESULTS_PARENT:-/workspace/artifacts/evals/layout100}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-jepawam60k-${RUN_TAG}}"
METHOD_RUN_ID="${METHOD_RUN_ID:-actr-${RUN_TAG}}"
BASELINE_RESULTS="$RESULTS_PARENT/$BASELINE_RUN_ID"
METHOD_RESULTS="$RESULTS_PARENT/$METHOD_RUN_ID"
REPORT_PATH="${REPORT_PATH:-$RESULTS_PARENT/actr-vs-jepawam-${RUN_TAG}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "OpenPI environment not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -d "$BASELINE_CHECKPOINT/params" ]]; then
    echo "Baseline checkpoint is incomplete: $BASELINE_CHECKPOINT" >&2
    exit 2
fi

echo "Waiting for completed Stage-2 checkpoint: $METHOD_CHECKPOINT"
while [[ ! -d "$METHOD_CHECKPOINT/params" ]]; do
    status="$(supervisorctl status "$TRAIN_PROGRAM" 2>&1 || true)"
    echo "$(date -Is) $status"
    if [[ "$status" != *"RUNNING"* ]]; then
        echo "Training stopped before the Stage-2 checkpoint became available." >&2
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

# A committed final checkpoint can appear just before the parent training
# script returns. Wait until the training process releases the GPUs so the
# four policy servers cannot race the final async checkpoint write.
while true; do
    status="$(supervisorctl status "$TRAIN_PROGRAM" 2>&1 || true)"
    echo "$(date -Is) $status"
    if [[ "$status" != *"RUNNING"* ]]; then
        break
    fi
    sleep "$POLL_SECONDS"
done

echo "Running sealed Layout100 baseline: $BASELINE_RUN_ID"
"$ROOT/scripts/run_layout100_checkpoint_eval.sh" \
    "$BASELINE_RUN_ID" "$BASELINE_CONFIG" "$BASELINE_CHECKPOINT"

echo "Running sealed Layout100 Con1 method: $METHOD_RUN_ID"
"$ROOT/scripts/run_layout100_checkpoint_eval.sh" \
    "$METHOD_RUN_ID" "$METHOD_CONFIG" "$METHOD_CHECKPOINT"

echo "Computing paired Layout100 statistics: $REPORT_PATH"
"$PYTHON_BIN" "$ROOT/scripts/compare_layout100_results.py" \
    "$BASELINE_RESULTS" "$METHOD_RESULTS" \
    --output "$REPORT_PATH"

echo "ACTR full experiment complete: $REPORT_PATH"
