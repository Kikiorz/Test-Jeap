#!/usr/bin/env bash
# Extract every episode video to 256x256 JPEG frames with ffmpeg (hardware
# accelerated), 24 files in parallel. The python per-frame decoder tops out at
# ~4 cores; ffmpeg saturates the box and turns this into a minutes-long job.
set -euo pipefail

DATASET="${DATASET:-/workspace/robotwin2/RoboTwin_v21}"
OUT="${OUT:-/workspace/robotwin2/frames_tmp}"
JOBS="${JOBS:-24}"

mkdir -p "$OUT"
for camera in observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist; do
  find "$DATASET/videos" -path "*${camera}*" -name "episode_*.mp4" -print0 |
    xargs -0 -P "$JOBS" -I{} bash -c '
      file="{}"
      camera="'"$camera"'"
      episode="$(basename "$file" .mp4)"
      target="'"$OUT"'/${camera}/${episode}"
      mkdir -p "$target"
      ffmpeg -nostdin -loglevel error -y -fps_mode passthrough -i "$file" -vf scale=256:256 -q:v 3 "$target/%05d.jpg"
    '
  echo "done $camera"
done
echo "frames extracted to $OUT"
du -sh "$OUT"
