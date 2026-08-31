#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
STAGE1_EXP="${STAGE1_EXP:-actr_stage1_seed42}"
STAGE2_EXP="${STAGE2_EXP:-actr_stage2_seed42}"

"$PYTHON_BIN" scripts/train.py pi05_libero_actr_stage1 --exp-name "$STAGE1_EXP"

stage1_params="$ROOT/checkpoints/pi05_libero_actr_stage1/$STAGE1_EXP/7999/params"
if [[ ! -d "$stage1_params" ]]; then
    echo "Stage-1 final params not found: $stage1_params" >&2
    exit 1
fi
stage1_link="$ROOT/data/weights/pi05_jepawam_actr_stage1/params"
mkdir -p "$(dirname "$stage1_link")"
if [[ -L "$stage1_link" ]]; then
    if [[ "$(readlink -f "$stage1_link")" != "$(readlink -f "$stage1_params")" ]]; then
        echo "Existing stage-1 link points elsewhere: $stage1_link" >&2
        exit 1
    fi
elif [[ -e "$stage1_link" ]]; then
    echo "Refusing to replace non-symlink stage-1 path: $stage1_link" >&2
    exit 1
else
    ln -s "$stage1_params" "$stage1_link"
fi

# Stage 1 must establish the mechanism before Stage 2 is allowed to optimize
# actions.  Matching actions should predict their realized transition target
# more accurately than a batch-marginal-preserving action derangement.
CUDA_VISIBLE_DEVICES="${ACTR_DIAGNOSTIC_GPU:-0}" \
    "$PYTHON_BIN" scripts/evaluate_actr_action_sensitivity.py \
    "$stage1_params" \
    --require-gate \
    --output /workspace/artifacts/logs/actr-stage1-action-sensitivity.json

"$PYTHON_BIN" scripts/train.py pi05_libero_actr_stage2 --exp-name "$STAGE2_EXP"

stage2_params="$ROOT/checkpoints/pi05_libero_actr_stage2/$STAGE2_EXP/19999/params"
if [[ ! -d "$stage2_params" ]]; then
    echo "Stage-2 final params not found: $stage2_params" >&2
    exit 1
fi
echo "ACTR pipeline complete: $stage2_params"
