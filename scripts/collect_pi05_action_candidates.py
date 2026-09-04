#!/usr/bin/env python3
"""Collect frozen pi0.5 action samples for policy-aware energy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
import run_transition_inverse_gate as inverse_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--batch-samples", type=int, default=2)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=163)
    return parser.parse_args()


def _raw_sample(samples: Any, index: int) -> dict[str, Any]:
    return {
        "observation/state": samples["states"][index],
        "observation/image": samples["current_base"][index],
        "observation/wrist_image": samples["current_wrist"][index],
        "prompt": str(samples["prompts"][index]),
        "actions": samples["action_chunks"][index],
    }


def _stack(values: list[Any]) -> Any:
    return jax.tree.map(lambda *items: np.stack(items), *values)


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    train, validation = inverse_gate._episode_split(
        samples["episode_indices"], samples["task_indices"]
    )
    if args.split == "train":
        selected = train
    elif args.split == "validation":
        selected = validation
    else:
        selected = np.arange(len(samples["episode_indices"]))

    policy = policy_config.create_trained_policy(
        config_lib.get_config(args.config), args.policy_checkpoint
    )
    transformed = [policy._input_transform(_raw_sample(samples, int(index))) for index in selected]
    action_dim = int(samples["action_chunks"].shape[-1])
    experts = np.stack(
        [np.asarray(item["actions"], dtype=np.float32)[..., :action_dim] for item in transformed]
    )
    observations = [{key: value for key, value in item.items() if key != "actions"} for item in transformed]

    sample_fn = nnx_utils.module_jit(policy._model.sample_actions, static_argnames=("num_steps",))
    rng = np.random.default_rng(args.seed)
    candidate_batches: list[np.ndarray] = []
    for start in range(0, len(selected), args.batch_samples):
        stop = min(start + args.batch_samples, len(selected))
        repeated = [
            observations[index]
            for index in range(start, stop)
            for _ in range(args.num_candidates)
        ]
        observation = model_lib.Observation.from_dict(
            jax.tree.map(lambda value: jnp.asarray(value), _stack(repeated))
        )
        noise = rng.standard_normal(
            ((stop - start) * args.num_candidates, 10, 32), dtype=np.float32
        )
        candidates = sample_fn(
            jax.random.key(0), observation, noise=jnp.asarray(noise), num_steps=args.solver_steps
        )
        candidate_batches.append(
            np.asarray(candidates)[..., :action_dim].reshape(
                stop - start, args.num_candidates, 10, action_dim
            )
        )
        print(json.dumps({"sampled": stop, "total": len(selected)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_indices=selected,
        candidates=np.concatenate(candidate_batches),
        experts=experts,
    )


if __name__ == "__main__":
    main()
