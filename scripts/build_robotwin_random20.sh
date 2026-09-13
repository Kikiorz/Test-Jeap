#!/usr/bin/env bash
# Build the 20-task RoboTwin 2.0 *randomized* LeRobot v2.1 dataset.
#
# The official release has no randomized split at all: the randomized
# demonstrations only exist per task as
# `dataset/<task>/aloha-agilex_randomized_500.zip`. The archives are 2.6-5.1 GB
# each and the box only has ~50 GB free, so this script downloads exactly one
# archive, converts it (appending to the same dataset), then deletes it.
#
# Usage:  bash scripts/build_robotwin_random20.sh
# Env:    EPISODES (default 200), OUT, CACHE, WORKERS, PYTHON, HF_TOKEN

set -uo pipefail

TASKS=(
  adjust_bottle
  beat_block_hammer
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_mic
  lift_pot
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_cans_plasticbox
  place_empty_cup
  place_object_basket
  place_shoe
  press_stapler
  shake_bottle_horizontally
  shake_bottle
  stack_bowls_three
  stack_bowls_two
)

EPISODES=${EPISODES:-200}
OUT=${OUT:-/workspace/data/robotwin_random20_inline}
CACHE=${CACHE:-/workspace/robotwin/code/data/download_cache/dataset}
PYTHON=${PYTHON:-/workspace/robotwin_ws/.venv/bin/python}
WORKERS=${WORKERS:-48}
TOKEN=${HF_TOKEN:-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}
BASE_URL=https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/dataset

mkdir -p "${CACHE}"
echo "[build] tasks=${#TASKS[@]} episodes/task=${EPISODES} out=${OUT}"

for task in "${TASKS[@]}"; do
  archive="aloha-agilex_randomized_500.zip"
  local_zip="${CACHE}/${task}/${archive}"

  if [[ ! -f "${local_zip}" ]]; then
    free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    if (( free_gb < 8 )); then
      echo "[build] ABORT: only ${free_gb}G free before ${task}"; exit 1
    fi
    mkdir -p "${CACHE}/${task}"
    echo "[build] downloading ${task}/${archive} (${free_gb}G free)"
    if ! curl -fL --retry 5 --retry-delay 10 -H "Authorization: Bearer ${TOKEN}" \
        -o "${local_zip}.part" "${BASE_URL}/${task}/${archive}"; then
      echo "[build] FAILED download ${task}; keeping the partial file for a retry"
      mv -f "${local_zip}.part" "${local_zip}.partial" 2>/dev/null
      continue
    fi
    mv "${local_zip}.part" "${local_zip}"
  else
    echo "[build] reusing cached archive for ${task}"
  fi

  echo "[build] converting ${task}"
  if "${PYTHON}" /workspace/robotwin_ws/scripts/build_robotwin_random_dataset.py \
      --zip "${task}=${local_zip}" \
      --output "${OUT}" \
      --episodes-per-task "${EPISODES}" \
      --workers "${WORKERS}" \
      --append; then
    rm -f "${local_zip}"
    echo "[build] ${task} done, archive removed"
  else
    echo "[build] FAILED converting ${task}; archive kept for inspection"
  fi
  df -BG --output=avail / | tail -1 | tr -dc '0-9' | xargs -I{} echo "[build] free now {}G"
done

echo "[build] ALL DONE"
