#!/usr/bin/env python3
"""Test JEPA transition-energy guidance on cached pi0.5 flow velocities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import transition_energy
import run_transition_inverse_gate as inverse_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--nochange", type=Path, required=True)
    parser.add_argument("--energy-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-time", type=float, default=0.5)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    return parser.parse_args()


def _pool(value: np.ndarray) -> jax.Array:
    return transition_energy.normalize(jnp.asarray(inverse_gate._pool_24_to_8(value)))


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    predicted = _pool(np.asarray(np.load(args.predicted, mmap_mode="r")))
    nochange = _pool(np.asarray(np.load(args.nochange, mmap_mode="r")))
    _, validation_indices = inverse_gate._episode_split(
        samples["episode_indices"], samples["task_indices"]
    )
    validation_set = set(validation_indices.tolist())
    selected_numpy = np.asarray(
        [
            index
            for index, sample_index in enumerate(np.asarray(cache["sample_indices"]))
            if int(sample_index) in validation_set and float(cache["time"][index]) <= args.max_time
        ],
        dtype=np.int32,
    )
    if not len(selected_numpy):
        raise ValueError("No cached validation flow tuples satisfy max-time")
    selected = jnp.asarray(selected_numpy)
    sample_indices = np.asarray(cache["sample_indices"])[selected_numpy]
    action_tau = jnp.asarray(cache["action_tau"])[selected, :, :7]
    base_velocity = jnp.asarray(cache["base_velocity"])[selected, :, :7]
    target_velocity = jnp.asarray(cache["target_velocity"])[selected, :, :7]

    model = transition_energy.TransitionActionEnergy(
        horizon=action_tau.shape[1],
        action_dim=action_tau.shape[2],
        transition_dim=predicted.shape[-1],
        width=args.width,
        num_heads=args.num_heads,
    )
    template = model.init(jax.random.key(0), predicted[:2], action_tau[:2])["params"]
    params = serialization.from_bytes(template, args.energy_params.read_bytes())

    tasks = np.asarray(samples["task_indices"])
    episodes = np.asarray(samples["episode_indices"])
    shuffled_sample = np.empty_like(sample_indices)
    for position, sample_index in enumerate(sample_indices):
        candidates = np.flatnonzero(
            (tasks == tasks[sample_index]) & (episodes != episodes[sample_index])
        )
        shuffled_sample[position] = int(candidates[0]) if len(candidates) else sample_index

    @jax.jit
    def gradient(transition, action):
        def energy(value):
            transition_embedding, action_embedding, _ = model.apply(
                {"params": params}, transition, value
            )
            return -jnp.sum(transition_embedding * action_embedding)

        return jax.grad(energy)(action)

    base_per_tuple = jnp.mean(jnp.square(base_velocity - target_velocity), axis=(1, 2))
    result = {
        "config": {
            "tuple_count": int(len(selected_numpy)),
            "max_time": args.max_time,
            "guidance_scales": args.guidance_scales,
        },
        "base_flow_mse": float(jnp.mean(base_per_tuple)),
        "transitions": {},
    }
    transition_inputs = {
        "predicted": predicted[jnp.asarray(sample_indices)],
        "nochange": nochange[jnp.asarray(sample_indices)],
        "within_task_shuffled_predicted": predicted[jnp.asarray(shuffled_sample)],
    }
    for name, transition in transition_inputs.items():
        energy_gradient = gradient(transition, action_tau)
        scale_results = {}
        for scale in args.guidance_scales:
            guided_velocity = base_velocity + scale * energy_gradient
            per_tuple = jnp.mean(jnp.square(guided_velocity - target_velocity), axis=(1, 2))
            delta = per_tuple - base_per_tuple
            scale_results[str(scale)] = {
                "flow_mse": float(jnp.mean(per_tuple)),
                "mean_flow_mse_delta": float(jnp.mean(delta)),
                "median_flow_mse_delta": float(jnp.median(delta)),
                "fraction_improved": float(jnp.mean(delta < 0)),
                "guidance_rms": float(scale * jnp.sqrt(jnp.mean(jnp.square(energy_gradient)))),
            }
        result["transitions"][name] = {"scales": scale_results}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
