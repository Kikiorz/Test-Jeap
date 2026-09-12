#!/usr/bin/env python3
"""Identify the state/action convention the RoboTwin JEPA-WAM base was trained on.

The released RoboTwin release stores absolute joint targets (``action[t] ==
state[t+1]``), while the base checkpoint's action statistics have ~zero mean and
a different spread, so the release is not in the base's action space. The base's
own flow-matching loss is the decisive test: with the frames, prompts, noise and
flow time held fixed, only the correct convention makes the target ``u_t =
noise - action`` match the network's velocity.
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

# Dims whose sign is mirrored between the release and the base (shoulder/elbow
# of both arms), and the two gripper dims, which the base keeps in [-1, 1].
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

    raw = [raw_dataset[i] for i in range(args.samples)]
    transformed = [dataset[i] for i in range(args.samples)]
    print(json.dumps({"raw_keys": sorted(raw[0].keys()),
                      "transformed_keys": sorted(transformed[0].keys())}), flush=True)

    horizon = model_config.action_horizon
    action_key = "actions" if "actions" in raw[0] else "action"
    raw_actions = np.stack([np.asarray(r[action_key], np.float32)[:horizon] for r in raw])
    raw_states = np.stack([np.asarray(r["observation.state"], np.float32) for r in raw])
    norm_actions = np.stack([np.asarray(t["actions"], np.float32)[:horizon] for t in transformed])
    norm_state = np.stack([np.asarray(t["state"], np.float32) for t in transformed])
    physical = raw_actions.shape[-1]
    print(json.dumps({"raw_actions": list(raw_actions.shape),
                      "norm_actions": list(norm_actions.shape),
                      "physical_dims": physical}), flush=True)

    # The pipeline normalises with the checkpoint's statistics; recover that
    # affine map per dimension from the raw/normalised pair so any candidate
    # convention can be normalised the same way.
    def fit(raw_values, norm_values):
        x = raw_values.reshape(-1, physical)
        y = norm_values.reshape(-1, physical)
        keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[keep], y[keep]
        if len(x) < 2:
            raise ValueError("not enough finite samples to fit the normalisation map")
        scale = np.ones(physical, np.float32)
        shift = np.zeros(physical, np.float32)
        for d in range(physical):
            if np.ptp(x[:, d]) < 1e-9:
                continue
            design = np.stack([x[:, d], np.ones(len(x))], axis=1)
            solution, *_ = np.linalg.lstsq(design, y[:, d], rcond=None)
            scale[d], shift[d] = solution
        residual = float(np.abs((x * scale + shift) - y).max())
        return scale, shift, residual

    action_scale, action_shift, action_residual = fit(
        raw_actions, norm_actions[..., :physical] if norm_actions.shape[-1] >= physical else norm_actions)
    state_scale, state_shift, state_residual = fit(
        raw_states, norm_state[..., :physical] if norm_state.shape[-1] >= physical else norm_state)
    print(json.dumps({"action_fit_residual": action_residual, "state_fit_residual": state_residual}),
          flush=True)

    # Build the batched observation once; only the state and the action target
    # change between candidates.
    def to_observation(sample):
        item = {key: np.asarray(value)[None] for key, value in sample.items() if key != "actions"}
        return _model.Observation.from_dict(item)

    action_variants = {
        "abs": lambda a, s: a,
        "delta_now": lambda a, s: a - s[:, :1, :],
        "delta_step": lambda a, s: a - s[:, :horizon, :],
        "delta_end": lambda a, s: a - s[:, -1:, :],
        "delta_mean": lambda a, s: a - s.mean(axis=1, keepdims=True),
    }
    state_variants = {
        "raw": (False, False),
        "flip": (True, False),
        "flip+grip": (True, True),
    }

    report = {}
    for state_name, (flip, gripper) in state_variants.items():
        for action_name, fn in action_variants.items():
            losses = []
            for index, sample in enumerate(transformed):
                observation = to_observation(sample)
                variant_state = transform_state(raw_states[index], flip, gripper)
                normalised_state = variant_state * state_scale + state_shift
                padded_state = np.zeros(observation.state.shape[-1], np.float32)
                padded_state[:physical] = normalised_state
                candidate = fn(raw_actions[index][None], raw_states[index][None])[0]
                normalised = candidate * action_scale + action_shift
                padded = np.zeros(observation.actions.shape[-1], np.float32)
                padded[:horizon, :physical] = normalised
                observation = dataclasses.replace(
                    observation,
                    state=jnp.asarray(padded_state)[None],
                )
                out = model.compute_loss(jax.random.key(0), observation,
                                         jnp.asarray(padded)[None], train=True)
                flow = out[0] if isinstance(out, tuple) else out
                losses.append(float(jnp.mean(flow)))
            report[f"{state_name}|{action_name}"] = float(np.mean(losses))
            print(json.dumps({f"{state_name}|{action_name}": report[f"{state_name}|{action_name}"]}),
                  flush=True)

    best = min(report, key=report.get)
    report["_best"] = best
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    with open(args.out, "w") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
