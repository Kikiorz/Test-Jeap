#!/usr/bin/env python3
"""Cache frozen pi0.5 velocities for a small transition-guidance experiment."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws-per-sample", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=59)
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
    if args.draws_per_sample < 1 or args.batch_size < 1:
        raise ValueError("draws-per-sample and batch-size must be positive")
    samples = np.load(args.samples, allow_pickle=False)
    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    transformed = [policy._input_transform(_raw_sample(samples, index)) for index in range(len(samples["states"]))]
    actions = np.stack([np.asarray(item["actions"], dtype=np.float32) for item in transformed])
    states = np.stack([np.asarray(item["state"], dtype=np.float32) for item in transformed])
    observation_items = [
        {key: value for key, value in item.items() if key != "actions"} for item in transformed
    ]

    rng = np.random.default_rng(args.seed)
    total = len(actions) * args.draws_per_sample
    sample_indices = np.repeat(np.arange(len(actions)), args.draws_per_sample)
    noise = rng.standard_normal((total, *actions.shape[1:]), dtype=np.float32)
    time = (rng.beta(1.5, 1.0, size=total) * 0.999 + 0.001).astype(np.float32)
    selected_actions = actions[sample_indices]
    action_tau = time[:, None, None] * noise + (1.0 - time[:, None, None]) * selected_actions
    target_velocity = noise - selected_actions

    velocity_fn = nnx_utils.module_jit(policy._model.predict_action_velocity)
    base_velocity: list[np.ndarray] = []
    for start in range(0, total, args.batch_size):
        stop = min(start + args.batch_size, total)
        indices = sample_indices[start:stop]
        observation_dict = _stack([observation_items[int(index)] for index in indices])
        observation = model_lib.Observation.from_dict(
            jax.tree.map(lambda value: jnp.asarray(value), observation_dict)
        )
        velocity = velocity_fn(
            observation,
            jnp.asarray(action_tau[start:stop]),
            jnp.asarray(time[start:stop]),
        )
        base_velocity.append(np.asarray(velocity, dtype=np.float32))
        print(json.dumps({"cached": stop, "total": total}), flush=True)

    output = {
        "sample_indices": sample_indices.astype(np.int32),
        "episode_indices": samples["episode_indices"][sample_indices],
        "task_indices": samples["task_indices"][sample_indices],
        "states": states[sample_indices],
        "actions": selected_actions,
        "action_tau": action_tau.astype(np.float32),
        "target_velocity": target_velocity.astype(np.float32),
        "base_velocity": np.concatenate(base_velocity, axis=0),
        "time": time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": len(actions),
                "draws_per_sample": args.draws_per_sample,
                "tuple_count": total,
                "base_flow_mse": float(
                    np.mean(np.square(output["base_velocity"] - output["target_velocity"]))
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
