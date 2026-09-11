#!/usr/bin/env python3
"""Derive a VLM-latent feature cache from the existing anchored cache.

The anchored cache already stores, for every frame of every episode, the PI0.5
VLM predictive tokens ``r`` (64 x 2048). The Con1 head predicts the future in
the *latent* space, which the cache stores as ``z`` (2816-dim pooled V-JEPA
2.1). This script writes a sibling cache whose ``z`` is instead the mean over
the 64 VLM tokens, so the entire existing pipeline (dataset, head, training
recipe) can be pointed at the VLM latent by changing only ``--latent-dim``.

``r`` is symlinked, not copied: the head's input tokens are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/workspace/artifacts/con1/anchored_40k_features_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/workspace/artifacts/con1/anchored_40k_features_vlm_pooled"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    source = args.source
    output = args.output
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())

    done = 0
    for index, entry in enumerate(manifest["episodes"]):
        if index % args.shards != args.shard:
            continue
        r_src = source / entry["r"]
        z_src = source / entry["z"]
        r_dst = output / entry["r"]
        z_dst = output / f"episodes/{entry['id']:06d}_z.npy"
        if not r_dst.exists():
            os.symlink(r_src, r_dst)
        if z_dst.exists():
            done += 1
            continue
        r = np.load(r_src, mmap_mode="r")
        pooled = np.asarray(r, dtype=np.float32).mean(1).astype(np.float16)
        z_old = np.load(z_src, mmap_mode="r")
        if pooled.shape != (len(z_old), r.shape[-1]):
            raise ValueError(f"Unexpected pooled shape {pooled.shape} for episode {entry['id']}")
        np.save(z_dst, pooled)
        done += 1
        if done % 100 == 0:
            print(json.dumps({"processed": done, "last_id": entry["id"]}), flush=True)

    print(json.dumps({"shard": args.shard, "processed": done}), flush=True)
    if args.shards != 1:
        return

    # Rewrite the manifest for the new target space.
    contract_bytes = (source / "contract.json").read_bytes()
    (output / "contract.json").write_bytes(contract_bytes)
    manifest["latent_dim"] = 2048
    for entry in manifest["episodes"]:
        z_path = output / f"episodes/{entry['id']:06d}_z.npy"
        entry["z"] = f"episodes/{entry['id']:06d}_z.npy"
        entry["z_sha256"] = hashlib.sha256(z_path.read_bytes()).hexdigest()
        entry["source_z_sha256"] = entry["z_sha256"]
    manifest["contract_sha256"] = hashlib.sha256(contract_bytes).hexdigest()
    manifest["latent_space"] = "pi05_vlm_predictive_token_mean"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("WROTE", output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
