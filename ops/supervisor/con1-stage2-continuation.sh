#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con
export PYTHONPATH=/workspace/ts_JEPA_con/src
export HF_HUB_OFFLINE=1

source_params=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/10000/params
test -d "${source_params}"

exec /workspace/openpi_jepawam/.venv/bin/python scripts/train_con1_con2.py con1-full \
  --source-params "${source_params}" \
  --exp-name h10_t16_d128_nojl_cont20k \
  --checkpoint-base-dir /workspace/artifacts/checkpoints \
  --num-train-steps 10000 \
  --overwrite
