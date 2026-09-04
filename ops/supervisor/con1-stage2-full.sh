#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con
export PYTHONPATH=/workspace/ts_JEPA_con/src
export HF_HUB_OFFLINE=1

warmup_root=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1_warmup/h10_t16_d128_nojl
warmup_step="$(find "${warmup_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -1)"
test -n "${warmup_step}"
source_params="${warmup_root}/${warmup_step}/params"
test -d "${source_params}"

exec /workspace/openpi_jepawam/.venv/bin/python scripts/train_con1_con2.py con1-full \
  --source-params "${source_params}" \
  --exp-name h10_t16_d128_nojl \
  --checkpoint-base-dir /workspace/artifacts/checkpoints \
  --num-train-steps 20000 \
  --resume
