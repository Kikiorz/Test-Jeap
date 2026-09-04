#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con
export PYTHONPATH=/workspace/ts_JEPA_con/src
export HF_HUB_OFFLINE=1

stage1_summary=/workspace/artifacts/con1/stage1_h10_t16_d128_nojl/summary.json
stage1_manifest=/workspace/artifacts/con1/stage1_h10_t16_d128_nojl/change_targets_raw/manifest.json
base_params=/workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999/params

test -f "${stage1_summary}"
test -f "${stage1_manifest}"

/workspace/openpi_jepawam/.venv/bin/python scripts/check_con1_step0_policy_deviation.py \
  --base-params "${base_params}" \
  --batch-size 4 \
  --num-steps 10 \
  --data-config-name pi05_libero_vjepa_con1_warmup \
  --output /workspace/artifacts/con1/step0_policy_deviation.json

exec /workspace/openpi_jepawam/.venv/bin/python scripts/train_con1_con2.py con1-warmup \
  --source-params "${base_params}" \
  --exp-name h10_t16_d128_nojl \
  --checkpoint-base-dir /workspace/artifacts/checkpoints \
  --resume
