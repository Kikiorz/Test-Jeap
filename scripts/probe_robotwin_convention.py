#!/usr/bin/env python3
"""Identify the state/action convention the RoboTwin JEPA-WAM base was trained on.

The released RoboTwin dataset stores absolute joint targets (``action[t] ==
state[t+1]``), while the base checkpoint's action statistics have ~zero mean and
a different spread, so the release is not in the base's action space. The base's
own flow-matching loss is the decisive test: with the frames, prompts, flow time
and noise held fixed, only the correct convention makes the target
``u_t = noise - action`` line up with the network's velocity.

Every candidate is pushed through the *same* transform chain the training job
uses, so normalisation and padding are identical to training; nothing is fitted
or approximated by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.models import model as _model
from openpi.training import config as configs, data_loader, sharding

# Dims mirrored between the release and the base (shoulder/elbow of both arms)
# and the two gripper dims, which the base keeps in [-1, 1].
FLIP_DIMS = (1, 2, 8, 9)
GRIPPER_DIMS = (6, 13)


def transform_state(state: np.ndarray, flip: bool, gripper: bool) -> np.ndarray:
    out = np.array(state, dtype=np.float32, copy=True)
    if flip:
        out[..., list(FLIP_DIMS)] *= -1.0
    if gripper:
        out[..., list(GRIPPER_DIMS)] = 2.0 * out[..., list(GRIPPER_DIMS)] - 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_robotwin_con1con2_ctx_20k")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = configs.get_config(args.config)
    data = dataclasses.replace(
        base.data, con1_latent_root=None,
        base_config=dataclasses.replace(base.data.base_config, con1_latent_root=None))
    model_config = dataclasses.replace(
        base.model, use_con1=False, use_con2=False, use_vjepa_aux=False)
    config = dataclasses.replace(
        base, data=data, model=model_config, batch_size=1, num_workers=0,
        wandb_enabled=False, exp_name="convention_probe", resume=False)

    mesh = sharding.make_mesh(config.fsdp_devices)
    _, rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, rng, mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)

    data_config = data.create(config.assets_dirs, model_config)
    raw_dataset = data_loader.create_torch_dataset(
        data_config, model_config.action_horizon, model_config)
    dataset = data_loader.transform_dataset(raw_dataset, data_config)

    horizon = model_config.action_horizon
    raw_samples = [raw_dataset[i] for i in range(args.samples)]
    print(json.dumps({"samples": len(raw_samples), "horizon": horizon}), flush=True)

    def variant_actions(name, actions, states):
        if name == "abs":
            return actions
        if name == "delta_now":
            return actions - states[:1]
        if name == "delta_step":
            return actions - states
        if name == "delta_end":
            return actions - states[-1:]
        if name == "delta_mean":
            return actions - states.mean(axis=0, keepdims=True)
        raise ValueError(name)

    action_variants = ["abs", "delta_now", "delta_step", "delta_end", "delta_mean"]
    state_variants = {"raw": (False, False), "flip": (True, False), "flip+grip": (True, True)}

    report = {}
    for state_name, (flip, gripper) in state_variants.items():
        for action_name in action_variants:
            losses = []
            for sample in raw_samples:
                raw_state = np.asarray(sample["observation.state"], np.float32)
                raw_actions = np.asarray(sample["actions"], np.float32)[:horizon]
                # action[t] == state[t+1] in this release, so the chunk's states
                # are the current state followed by the chunk's own targets.
                chunk_states = np.concatenate([raw_state[None], raw_actions[:-1]], axis=0)
                variant_state = transform_state(raw_state, flip, gripper)
                variant_actions = variant_actions(action_name, raw_actions, chunk_states)
                patched = dict(sample)
                patched["observation.state"] = variant_state
                patched["actions"] = variant_actions
                out = dataset._transform(patched)
                item = {}
                for key, value in out.items():
                    if key in ("actions", "actions_is_pad"):
                        continue
                    if isinstance(value, dict):
                        item[key] = {name: np.asarray(entry)[None] for name, entry in value.items()}
                    else:
                        item[key] = np.asarray(value)[None]
                observation = _model.Observation.from_dict(item)
                actions = jnp.asarray(np.asarray(out["actions"], np.float32)[None])
                result = model.compute_loss(jax.random.key(0), observation, actions, train=True)
                flow = result[0] if isinstance(result, tuple) else result
                losses.append(float(jnp.mean(flow)))
            key = f"{state_name}|{action_name}"
            report[key] = float(np.mean(losses))
            print(json.dumps({key: report[key]}), flush=True)

    ranked = sorted((value, key) for key, value in report.items())
    report["_ranking"] = [f"{key}: {value:.4f}" for value, key in ranked]
    report["_best"] = ranked[0][1]
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    with open(args.out, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
