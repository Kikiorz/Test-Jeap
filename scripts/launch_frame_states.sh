#!/usr/bin/env bash
# Stage 1 of the Con1 cache rebuild: per-frame frozen V-JEPA 2.1 states.
# Episodes are sharded by `episode_index % world_size`, so every worker sees an
# interleaved, task-balanced slice.
set -euo pipefail

ROOT="${ROOT:-/dev/shm/ts_jepa}"
REPO="${REPO:-$ROOT/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
DATASET="${DATASET:-$ROOT/data/libero_lerobot}"
CHECKPOINT="${CHECKPOINT:-$ROOT/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SRC="${VJEPA_SRC:-$ROOT/vjepa2_src}"
OUTPUT="${OUTPUT:-$ROOT/con1/frame_states_v1}"
WORLD="${WORLD:-4}"
BATCH="${BATCH:-8}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs}"
mkdir -p "$LOG_ROOT" "$OUTPUT"

cd "$REPO"
for ((rank = 0; rank < WORLD; rank++)); do
  PYTHONPATH="$REPO/src" OMP_NUM_THREADS=4 \
    nohup "$PY" -u scripts/cache_con1_frame_states.py \
      --dataset-root "$DATASET" \
      --checkpoint "$CHECKPOINT" \
      --vjepa-source-root "$VJEPA_SRC" \
      --output-root "$OUTPUT" \
      --image-keys image wrist_image \
      --device "cuda:$rank" \
      --batch-size "$BATCH" \
      --worker-rank "$rank" \
      --world-size "$WORLD" \
      >"$LOG_ROOT/frame_states_rank$rank.log" 2>&1 &
done
printf 'launched %s workers -> %s\n' "$WORLD" "$OUTPUT"
