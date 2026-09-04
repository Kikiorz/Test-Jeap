#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con
export PYTHONPATH=/workspace/ts_JEPA_con/src
exec /workspace/openpi_jepawam/.venv/bin/python scripts/train_vjepa_change_teacher.py \
  --dataset-root /workspace/artifacts/datasets/lerobot_libero \
  --displacement-target-root /workspace/artifacts/vjepa_targets/libero_vjepa2_1_vitg_384_offset10_displacement1408 \
  --norm-stats /workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999/assets/physical-intelligence/libero/norm_stats.json \
  --output-dir /workspace/artifacts/con1/stage1_h10_t16_d128_nojl \
  --future-offset 10 \
  --batch-size 128 \
  --steps 50000 \
  --eval-every 1000 \
  --patience 5 \
  --change-token-dim 128 \
  --resume
