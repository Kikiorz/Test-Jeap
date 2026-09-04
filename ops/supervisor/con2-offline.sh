#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con
export PYTHONPATH=/workspace/ts_JEPA_con/src
export HF_HUB_OFFLINE=1

con1_root=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl
con1_step="$(find "${con1_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -1)"
test -n "${con1_step}"
source_params="${con1_root}/${con1_step}/params"
test -d "${source_params}"

exec /workspace/openpi_jepawam/.venv/bin/python scripts/train_con1_con2.py con2-offline \
  --source-params "${source_params}" \
  --exp-name h10_rank4 \
  --checkpoint-base-dir /workspace/artifacts/checkpoints \
  --resume
