#!/usr/bin/env python3
"""Check whether transition-energy gradients locally improve action proposals."""

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
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--nochange", type=Path, required=True)
    parser.add_argument("--energy-params", type=Path, required=True)
    parser.add_argument(
        "--proposals",
        type=Path,
        help="Optional rerank selections.npz; uses its first frozen-pi0.5 candidate instead of synthetic noise.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--proposal-noise", type=float, default=0.1)
    parser.add_argument("--step-sizes", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--seed", type=int, default=149)
    return parser.parse_args()


def _pool(value: np.ndarray) -> jax.Array:
    return transition_energy.normalize(jnp.asarray(inverse_gate._pool_24_to_8(value)))


def _actions(cache: np.lib.npyio.NpzFile, count: int) -> jax.Array:
    first: dict[int, int] = {}
    for tuple_index, sample_index in enumerate(np.asarray(cache["sample_indices"])):
        first.setdefault(int(sample_index), tuple_index)
    return jnp.asarray(cache["actions"][[first[index] for index in range(count)], :, :7])


def _bootstrap_interval(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    observed = _pool(np.asarray(np.load(args.observed, mmap_mode="r")))
    predicted = _pool(np.asarray(np.load(args.predicted, mmap_mode="r")))
    nochange = _pool(np.asarray(np.load(args.nochange, mmap_mode="r")))
    actions = _actions(cache, len(observed))
    _, default_validation_indices = inverse_gate._episode_split(
        samples["episode_indices"], samples["task_indices"]
    )

    model = transition_energy.TransitionActionEnergy(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        transition_dim=observed.shape[-1],
        width=args.width,
        num_heads=args.num_heads,
    )
    template = model.init(jax.random.key(args.seed), observed[:2], actions[:2])["params"]
    params = serialization.from_bytes(template, args.energy_params.read_bytes())

    if args.proposals is None:
        validation_indices = default_validation_indices
        validation = jnp.asarray(validation_indices)
        rng = np.random.default_rng(args.seed)
        expert = actions[validation]
        proposal = expert + jnp.asarray(
            rng.normal(0.0, args.proposal_noise, size=expert.shape), dtype=expert.dtype
        )
        proposal_source = "expert_plus_gaussian_noise"
    else:
        proposal_data = np.load(args.proposals, allow_pickle=False)
        validation_indices = np.asarray(proposal_data["validation_indices"])
        if not np.array_equal(validation_indices, default_validation_indices):
            raise ValueError("Proposal validation indices do not match the episode split")
        validation = jnp.asarray(validation_indices)
        expert = jnp.asarray(proposal_data["experts"])
        proposal = jnp.asarray(proposal_data["candidates"][:, 0])
        proposal_source = "frozen_pi05_first_candidate"
    tasks = np.asarray(samples["task_indices"])[validation_indices]
    shuffled_indices = validation_indices.copy()
    for task in np.unique(tasks):
        positions = np.flatnonzero(tasks == task)
        shuffled_indices[positions] = np.roll(validation_indices[positions], 1)

    @jax.jit
    def energy_gradient(transition, action):
        def objective(value):
            transition_embedding, action_embedding, _ = model.apply(
                {"params": params}, transition, value
            )
            return -jnp.sum(transition_embedding * action_embedding)

        return jax.value_and_grad(objective)(action)

    base_per_sample = jnp.mean(jnp.square(proposal - expert), axis=(1, 2))
    result = {
        "config": {
            "count": int(len(validation_indices)),
            "proposal_noise": args.proposal_noise,
            "proposal_source": proposal_source,
            "step_sizes": args.step_sizes,
            "seed": args.seed,
        },
        "base_mse": float(jnp.mean(base_per_sample)),
        "transitions": {},
    }
    transition_inputs = {
        "observed": observed[validation],
        "predicted": predicted[validation],
        "nochange": nochange[validation],
        "within_task_shuffled_predicted": predicted[jnp.asarray(shuffled_indices)],
    }
    for transition_offset, (name, transition) in enumerate(transition_inputs.items()):
        energy, gradient = energy_gradient(transition, proposal)
        per_step = {}
        for step_size in args.step_sizes:
            corrected = proposal - step_size * gradient
            corrected_per_sample = jnp.mean(jnp.square(corrected - expert), axis=(1, 2))
            delta = corrected_per_sample - base_per_sample
            delta_array = np.asarray(delta)
            per_step[str(step_size)] = {
                "mse": float(jnp.mean(corrected_per_sample)),
                "mean_mse_delta": float(jnp.mean(delta)),
                "median_mse_delta": float(jnp.median(delta)),
                "fraction_improved": float(jnp.mean(delta < 0)),
                "bootstrap_95ci_mean_delta": _bootstrap_interval(
                    delta_array,
                    args.seed + transition_offset * len(args.step_sizes) + len(per_step),
                ),
                "mean_action_step_rms": float(
                    jnp.mean(jnp.sqrt(jnp.mean(jnp.square(step_size * gradient), axis=(1, 2))))
                ),
            }
        result["transitions"][name] = {
            "energy": float(energy),
            "gradient_rms": float(jnp.sqrt(jnp.mean(jnp.square(gradient)))),
            "steps": per_step,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
