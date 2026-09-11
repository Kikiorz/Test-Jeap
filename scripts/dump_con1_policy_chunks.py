#!/usr/bin/env python3
"""Dump the Con1 policy's own action chunks on chosen (episode, frame) pairs.

Used by the WM action-judgment probe: the question is whether a V-JEPA 2-AC
gradient step taken at the *policy's* chunk moves it towards the demonstrated
chunk. That needs (a) the policy's chunks and (b) the demonstrated chunks on the
same frames, which this writes as one npz.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.training import config as _configs
from probe_vjepa_ac_libero import load_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero_con1_action_adapter_40k")
    parser.add_argument("--checkpoint", required=True, help="trained policy directory")
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--anchored", type=Path,
                        default=Path("/workspace/artifacts/con1/anchored_40k_features_v1"))
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = _configs.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    model = policy._model

    manifest = json.loads((args.anchored / "manifest.json").read_text())
    lengths = {e["id"]: e["length"] for e in manifest["episodes"]}
    z_paths = {e["id"]: args.anchored / e["z"] for e in manifest["episodes"]}
    prompts = {row["episode_index"]: row["tasks"][0]
               for row in map(json.loads, (args.dataset / "meta" / "episodes.jsonl").read_text().splitlines())}

    rng = np.random.default_rng(args.seed)
    records = {"episode": [], "frame": [], "policy_chunk": [], "demo_chunk": []}
    for episode in args.episodes:
        frames, state, action = load_episode(args.dataset, episode, "agentview")
        wrist = load_episode(args.dataset, episode, "wrist")[0]
        z = np.load(z_paths[episode], allow_pickle=False)
        length = lengths[episode]
        # keep frames whose 10-step demo chunk and WM context are fully inside
        low, high = 20, length - args.horizon - 1
        if high <= low:
            continue
        index = np.sort(rng.choice(np.arange(low, high), size=min(args.frames_per_episode, high - low), replace=False))
        prompt = prompts[episode]
        for frame in index:
            frame = int(frame)
            inputs = policy._input_transform({
                "observation/image": frames[frame],
                "observation/wrist_image": wrist[frame],
                "observation/state": state[frame].astype(np.float32),
                "prompt": prompt,
            })
            observation = model_lib.Observation.from_dict(
                jax.tree.map(lambda x: jnp.asarray(x)[None], inputs))
            observation = dataclasses.replace(
                observation, con1_current_latent=jnp.asarray(z[frame], dtype=jnp.float32)[None],
                vjepa_target=None)
            chunk = np.asarray(model.sample_actions(jax.random.key(args.seed), observation))[0]
            records["episode"].append(episode)
            records["frame"].append(frame)
            records["policy_chunk"].append(chunk.astype(np.float32))
            records["demo_chunk"].append(action[frame:frame + args.horizon].astype(np.float32))
        print(json.dumps({"episode": episode, "samples": len(records["frame"])}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{k: np.asarray(v) for k, v in records.items()})
    print("WROTE", args.out, "samples", len(records["frame"]))


if __name__ == "__main__":
    main()
