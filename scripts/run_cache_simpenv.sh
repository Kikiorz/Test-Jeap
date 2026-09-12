#!/usr/bin/env bash
# Shard the pi0.5 VLM latent cache across the four GPUs, then merge the shard
# manifests into the single manifest.json the training loader expects.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
cd "$ROOT"

export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/lerobot_home
export HF_HUB_OFFLINE=1

OUT="${OUT:-/workspace/data/bridge_vlm_latents}"
SHARDS="${SHARDS:-4}"
mkdir -p "$OUT/episodes"

pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="$shard" nohup .venv/bin/python scripts/cache_simpenv_vlm_latents.py \
    --output-root "$OUT" --shard "$shard" --shards "$SHARDS" \
    >"/workspace/cache_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

.venv/bin/python - "$OUT" "$SHARDS" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); shards = int(sys.argv[2])
entries, latent_dim, r_shape = [], None, None
for shard in range(shards):
    manifest = json.loads((out / f"manifest.shard{shard}.json").read_text())
    entries.extend(manifest["episodes"])
    latent_dim = manifest["latent_dim"]; r_shape = manifest["r_shape"]
entries.sort(key=lambda e: e["id"])
if len({e["id"] for e in entries}) != len(entries):
    raise SystemExit("duplicate episode ids across shards")
manifest = {
    "schema": "con1-anchored-features-v1",
    "anchor_source": "current_only_frozen_teacher",
    "latent_space": "pi05_vlm_pooled_prefix",
    "latent_dim": latent_dim,
    "r_shape": r_shape,
    "chunks_size": 1000,
    "complete": True,
    "episodes": entries,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("MERGED", len(entries), "episodes, latent_dim", latent_dim)
PY
