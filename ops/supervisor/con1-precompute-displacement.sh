#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con
exec /workspace/MIKASA-Robo/.venv/bin/python scripts/precompute_vjepa_displacement_targets.py \
  --dataset-root /workspace/artifacts/datasets/lerobot_libero \
  --hf-port /workspace/artifacts/models/vjepa21_hf_port \
  --output-root /workspace/artifacts/vjepa_targets/libero_vjepa2_1_vitg_384_offset10_displacement1408 \
  --future-offset 10 \
  --batch-size 16 \
  --device cuda:0 \
  --worker-rank "${WORKER_RANK}" \
  --world-size 4
