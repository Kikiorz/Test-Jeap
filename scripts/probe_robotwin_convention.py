#!/usr/bin/env python3
"""Which state/action convention was the RoboTwin JEPA-WAM base trained on?

Statistics alone cannot separate the candidates (the official release stores
``action[t] == state[t+1]``), so this asks the model itself: the flow-matching
target ``u_t = noise - action`` only matches the network's own velocity when the
action chunk is expressed the way the checkpoint expects. Everything except the
candidate transform is held fixed (same frames, same prompts, same noise), so
the ranking is a direct read-out of the base's convention.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as configs


def load_data(path: str, episodes: list[int], frames_per_episode: int):
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
    state = np.asarray(table["observation.state"].to_pylist(), np.float32)
    action = np.asarray(table["action"].to_pylist(), np.float32)
    index = np.asarray(table["episode_index"].to_pylist())
    rows = []
    rng = np.random.default_rng(0)
    for episode in episodes:
        where = np.flatnonzero(index == episode)
        if len(where) < 60:
            continue
        picks = rng.choice(where[:-55], size=min(frames_per_episode, len(where) - 55), replace=False)
        for start in picks:
            rows.append((int(episode), int(start), state[where[0]:where[-1] + 1], action[where[0]:where[-1] + 1]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_robotwin_con1con2_ctx_20k")
    parser.add_argument("--checkpoint", default=(
        "/workspace/artifacts/models/jepa_wam_pi05_robotwin/checkpoints/openpi/"
        "pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_b128_fsdp4_gpu0123_seed42/19999"))
    parser.add_argument("--data", default="/workspace/robotwin2/RoboTwin_lerobot_v30/data/chunk-000/file-000.parquet")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--frames-per-episode", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = configs.get_config(args.config)
    # Evaluate the *base* policy: the Con1/Con2 modules must not contribute.
    config = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, use_con1=False, use_con2=False),
        num_workers=0, wandb_enabled=False, exp_name="convention_probe", resume=True,
        batch_size=1)
    policy = policy_config.create_trained_policy(config, args.checkpoint)

    rows = load_data(args.data, args.episodes, args.frames_per_episode)
    horizon = args.horizon
    batches = []
    for episode, start, state, action in rows:
        if start + horizon + 1 >= len(state):
            continue
        batches.append({
            "state": state[start],
            "abs": action[start:start + horizon],
            "delta_now": action[start:start + horizon] - state[start],
            "delta_step": action[start:start + horizon] - state[start:start + horizon],
            "center_ep": action[start:start + horizon] - state.mean(0),
        })
    print(json.dumps({"samples": len(batches), "horizon": horizon}), flush=True)

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    rng = jax.random.key(0)
    report = {}
    for name in ("abs", "delta_now", "delta_step", "center_ep"):
        losses = []
        for item in batches:
            observation = policy._input_transform({
                "observation/state": item["state"],
                "observation/image": image,
                "observation/wrist_image": image,
                "observation/wrist_image_right": image,
                "prompt": "pick up the object",
                "actions": item[name].astype(np.float32),
            })
            observation = _model.Observation.from_dict(observation)
            observation = dataclasses.replace(
                observation,
                images=jax.tree.map(lambda x: jnp.asarray(x)[None], observation.images),
                state=jnp.asarray(observation.state)[None],
                actions=jnp.asarray(observation.actions)[None],
                tokenized_prompt=jnp.asarray(observation.tokenized_prompt)[None],
                tokenized_prompt_mask=jnp.asarray(observation.tokenized_prompt_mask)[None],
            )
            loss = policy._model.compute_loss(rng, observation, observation.actions, train=True)
            flow = loss[0] if isinstance(loss, tuple) else loss
            losses.append(float(jnp.mean(flow)))
        report[name] = {"mean_flow": float(np.mean(losses)), "n": len(losses)}
        print(json.dumps({name: report[name]}), flush=True)

    best = min(report, key=lambda k: report[k]["mean_flow"])
    report["best"] = best
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    with open(args.out, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
