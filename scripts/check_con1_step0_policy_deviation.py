#!/usr/bin/env python3
"""Measure how much the zero-step Con1 initialization perturbs JEPA-WAM.

The comparison uses the released checkpoint, identical current observations,
identical action noise, and identical Euler schedules.  Con1 additionally
samples its Change source because that variable does not exist in the base
policy.  This script is a model-level diagnostic, not a rollout evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import weight_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--change-kv-init-scale", type=float, default=1e-3)
    parser.add_argument(
        "--data-config-name",
        help="Use one deterministic real-data batch from this training config instead of synthetic inputs",
    )
    parser.add_argument("--max-endpoint-relative-rms", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_config(*, con1: bool) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        use_vjepa_aux=True,
        vjepa_num_queries=64,
        vjepa_query_grid_size=8,
        vjepa_target_grid_size=24,
        vjepa_target_dim=1408,
        vjepa_action_attends_queries=False,
        use_action_change_mmdit=con1,
        change_num_tokens=16,
        change_token_dim=128,
        change_joint_start_layer=12,
        change_loss_weight=0.3,
        change_train_action_late=False,
    )


def make_observation(batch_size: int, key: jax.Array) -> model_lib.Observation:
    keys = jax.random.split(key, 3)
    images = {
        "base_0_rgb": jax.random.uniform(keys[0], (batch_size, 224, 224, 3), minval=-1.0, maxval=1.0),
        "left_wrist_0_rgb": jax.random.uniform(keys[1], (batch_size, 224, 224, 3), minval=-1.0, maxval=1.0),
        "right_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
    }
    prompt = jax.random.randint(keys[2], (batch_size, 32), 0, 1024, dtype=jnp.int32)
    return model_lib.Observation(
        images=images,
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.zeros((batch_size,), dtype=jnp.bool_),
        },
        state=jnp.zeros((batch_size, 32), dtype=jnp.float32),
        tokenized_prompt=prompt,
        tokenized_prompt_mask=jnp.ones_like(prompt, dtype=jnp.bool_),
    )


def load_model(config: pi0_config.Pi0Config, loader, key: jax.Array):
    model = config.create(key)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loader.load(state.to_pure_dict()))
    return nnx.merge(graphdef, state)


def error_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    difference_rms = float(np.sqrt(np.mean(np.square(difference))))
    return {
        "reference_rms": reference_rms,
        "absolute_mae": float(np.mean(np.abs(difference))),
        "absolute_rms": difference_rms,
        "relative_rms": difference_rms / max(reference_rms, 1e-12),
        "max_abs": float(np.max(np.abs(difference))),
    }


def main() -> None:
    args = parse_args()
    root_key = jax.random.key(args.seed)
    observation_key, noise_key, change_key, model_key = jax.random.split(root_key, 4)
    if args.data_config_name:
        loader_config = dataclasses.replace(
            training_config.get_config(args.data_config_name),
            batch_size=args.batch_size,
            num_workers=0,
        )
        loader = data_loader.create_data_loader(loader_config, shuffle=False, num_batches=1)
        observation, _ = next(iter(loader))
        observation_source = f"real:{args.data_config_name}:first_batch"
    else:
        observation = make_observation(args.batch_size, observation_key)
        observation_source = "synthetic_random"
    action_noise = jax.random.normal(noise_key, (args.batch_size, 10, 32))
    change_noise = jax.random.normal(change_key, (args.batch_size, 16, 128))
    times = (0.9, 0.5, 0.1)

    baseline = load_model(
        make_config(con1=False),
        weight_loaders.ActionChangeCheckpointWeightLoader(str(args.base_params)),
        model_key,
    )
    baseline_actions = np.asarray(
        baseline.sample_actions(root_key, observation, num_steps=args.num_steps, noise=action_noise)
    )
    baseline_velocities = {
        str(time): np.asarray(
            baseline.predict_action_velocity(
                observation,
                action_noise,
                jnp.full((args.batch_size,), time, dtype=jnp.float32),
            )
        )
        for time in times
    }
    del baseline
    gc.collect()
    jax.clear_caches()

    con1 = load_model(
        make_config(con1=True),
        weight_loaders.ActionChangeCheckpointWeightLoader(
            str(args.base_params), change_kv_init_scale=args.change_kv_init_scale
        ),
        jax.random.fold_in(model_key, 1),
    )
    con1_actions = np.asarray(
        con1.sample_actions(root_key, observation, num_steps=args.num_steps, noise=action_noise)
    )
    con1_velocities = {}
    for time in times:
        value, _ = con1.predict_action_change_velocity(
            observation,
            action_noise,
            change_noise,
            jnp.full((args.batch_size,), time, dtype=jnp.float32),
            jnp.full((args.batch_size,), time, dtype=jnp.float32),
        )
        con1_velocities[str(time)] = np.asarray(value)

    report = {
        "base_params": str(args.base_params),
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "observation_source": observation_source,
        "change_kv_init_scale": args.change_kv_init_scale,
        "action_endpoint": error_summary(baseline_actions[..., :7], con1_actions[..., :7]),
        "action_velocity": {
            time: error_summary(baseline_velocities[time][..., :7], con1_velocities[time][..., :7])
            for time in baseline_velocities
        },
    }
    payload = json.dumps(report, indent=2)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if report["action_endpoint"]["relative_rms"] > args.max_endpoint_relative_rms:
        raise RuntimeError(
            "Con1 initialization perturbs the released policy too strongly: "
            f"relative endpoint RMS={report['action_endpoint']['relative_rms']:.6f} > "
            f"{args.max_endpoint_relative_rms:.6f}"
        )


if __name__ == "__main__":
    main()
