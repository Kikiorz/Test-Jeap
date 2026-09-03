#!/usr/bin/env python3
"""Train the minimal CoFlow core on frozen real JEPA/action tuples.

The experiment compares a coupled transition/action flow against an otherwise
matched model that can only use the fixed JEPA transition prior.  It is a fast
scientific gate before changing the large Pi0 transformer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import coflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--observed-transition", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--transition-weight", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _pool_24_to_8(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[1] != 24 * 24:
        raise ValueError(f"Expected [N,576,D] transition array, got {value.shape}")
    value = value.reshape(value.shape[0], 8, 3, 8, 3, value.shape[-1])
    return value.mean(axis=(2, 4)).reshape(value.shape[0], 64, value.shape[-1])


def _episode_split(episode_indices: np.ndarray, task_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for task in np.unique(task_indices):
        episodes = np.unique(episode_indices[task_indices == task])
        if len(episodes) < 2:
            raise ValueError(f"Task {task} has fewer than two sampled episodes")
        train_episodes.update(episodes[::2].tolist())
        validation_episodes.update(episodes[1::2].tolist())
    train = np.flatnonzero(np.isin(episode_indices, list(train_episodes)))
    validation = np.flatnonzero(np.isin(episode_indices, list(validation_episodes)))
    return train, validation


def _normalized_actions(policy: Any, samples: Any) -> np.ndarray:
    transformed_actions: list[np.ndarray] = []
    for index in range(len(samples["states"])):
        raw = {
            "observation/state": samples["states"][index],
            "observation/image": samples["current_base"][index],
            "observation/wrist_image": samples["current_wrist"][index],
            "prompt": str(samples["prompts"][index]),
            "actions": samples["action_chunks"][index],
        }
        transformed = policy._input_transform(raw)
        transformed_actions.append(np.asarray(transformed["actions"], dtype=np.float32))
    return np.stack(transformed_actions)


def _train_mode(
    mode: coflow.CoFlowMode,
    actions: jax.Array,
    prior: jax.Array,
    observed: jax.Array,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    model = coflow.CoFlowCore(
        action_dim=actions.shape[-1],
        transition_dim=prior.shape[-1],
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        mode=mode,
    )
    dummy_time = jnp.full((1,), 0.5)
    dummy_transition, _ = coflow.transition_interpolant(prior[:1], observed[:1], dummy_time)
    params = model.init(
        jax.random.key(args.seed), actions[:1], dummy_transition, prior[:1], dummy_time
    )["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    def losses(parameters, batch_actions, batch_prior, batch_observed, noise, time):
        expanded_time = time[:, None, None]
        action_tau = expanded_time * noise + (1.0 - expanded_time) * batch_actions
        action_velocity = noise - batch_actions
        transition_tau, transition_velocity = coflow.transition_interpolant(
            batch_prior, batch_observed, time
        )
        predicted_action, predicted_transition = model.apply(
            {"params": parameters}, action_tau, transition_tau, batch_prior, time
        )
        action_loss = jnp.mean(jnp.square(predicted_action - action_velocity))
        transition_loss = jnp.mean(jnp.square(predicted_transition - transition_velocity))
        return action_loss + args.transition_weight * transition_loss, (action_loss, transition_loss)

    @jax.jit
    def train_step(parameters, state, batch_actions, batch_prior, batch_observed, key):
        noise_key, time_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, batch_actions.shape)
        time = jax.random.beta(time_key, 1.5, 1.0, (batch_actions.shape[0],)) * 0.999 + 0.001
        (loss, parts), gradients = jax.value_and_grad(losses, has_aux=True)(
            parameters, batch_actions, batch_prior, batch_observed, noise, time
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, parts, optax.global_norm(gradients)

    @jax.jit
    def heldout_loss(parameters, batch_actions, batch_prior, batch_observed, key):
        noise_key, time_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, batch_actions.shape)
        time = jax.random.beta(time_key, 1.5, 1.0, (batch_actions.shape[0],)) * 0.999 + 0.001
        return losses(parameters, batch_actions, batch_prior, batch_observed, noise, time)[1]

    @jax.jit
    def integrate(parameters, batch_prior, noise):
        action = noise
        transition = batch_prior
        dt = -1.0 / args.solver_steps

        def body(step, carry):
            action_value, transition_value = carry
            time_value = jnp.full((action_value.shape[0],), 1.0 + step * dt)
            action_velocity, transition_velocity = model.apply(
                {"params": parameters}, action_value, transition_value, batch_prior, time_value
            )
            return action_value + dt * action_velocity, transition_value + dt * transition_velocity

        return jax.lax.fori_loop(0, args.solver_steps, body, (action, transition))

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    key = jax.random.key(args.seed + 1)
    for step in range(args.steps):
        selected = rng.choice(train_indices, args.batch_size, replace=len(train_indices) < args.batch_size)
        key, step_key = jax.random.split(key)
        params, optimizer_state, loss, parts, gradient_norm = train_step(
            params,
            optimizer_state,
            actions[selected],
            prior[selected],
            observed[selected],
            step_key,
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {
                "step": step + 1,
                "total_loss": float(loss),
                "action_loss": float(parts[0]),
                "transition_loss": float(parts[1]),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps({"mode": mode, **record}), flush=True)

    validation = jnp.asarray(validation_indices)
    eval_key = jax.random.key(args.seed + 100)
    action_loss, transition_loss = heldout_loss(
        params, actions[validation], prior[validation], observed[validation], eval_key
    )
    noise = jax.random.normal(jax.random.key(args.seed + 200), actions[validation].shape)
    generated_action, generated_transition = integrate(params, prior[validation], noise)
    action_mse = jnp.mean(jnp.square(generated_action - actions[validation]))
    matched_transition = jnp.mean(
        1.0 - jnp.sum(coflow.normalize_transition(generated_transition) * observed[validation], axis=-1)
    )
    shuffled_indices = jnp.roll(validation, 1)
    shuffled_target_distance = jnp.mean(
        1.0 - jnp.sum(coflow.normalize_transition(generated_transition) * observed[shuffled_indices], axis=-1)
    )
    shuffled_prior = prior[jnp.roll(validation, 1)]
    shuffled_action, _ = integrate(params, shuffled_prior, noise)
    shuffled_prior_action_mse = jnp.mean(jnp.square(shuffled_action - actions[validation]))

    second_noise = jax.random.normal(jax.random.key(args.seed + 201), actions[validation].shape)
    _, second_transition = integrate(params, prior[validation], second_noise)
    action_to_transition_sensitivity = jnp.mean(
        jnp.abs(generated_transition - second_transition)
    )
    metrics = {
        "mode": mode,
        "heldout_action_flow_loss": float(action_loss),
        "heldout_transition_flow_loss": float(transition_loss),
        "integrated_action_mse": float(action_mse),
        "shuffled_prior_action_mse": float(shuffled_prior_action_mse),
        "prior_shuffle_delta": float(shuffled_prior_action_mse - action_mse),
        "generated_to_matched_transition": float(matched_transition),
        "generated_to_shuffled_transition": float(shuffled_target_distance),
        "matched_transition_margin": float(shuffled_target_distance - matched_transition),
        "action_to_transition_sensitivity": float(action_to_transition_sensitivity),
        "history": history,
    }
    return metrics, params


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.solver_steps < 1:
        raise ValueError("steps, batch-size, and solver-steps must be positive")

    from openpi.policies import policy_config
    from openpi.training import config as config_lib

    samples = np.load(args.samples, allow_pickle=False)
    if "action_chunks" not in samples:
        raise ValueError("Sample archive lacks action_chunks; regenerate it with the current audit script")
    predicted = _pool_24_to_8(np.asarray(np.load(args.prediction, mmap_mode="r"), dtype=np.float32))
    observed = _pool_24_to_8(
        np.asarray(np.load(args.observed_transition, mmap_mode="r"), dtype=np.float32)
    )
    predicted = np.asarray(coflow.normalize_transition(jnp.asarray(predicted)), dtype=np.float32)
    observed = np.asarray(coflow.normalize_transition(jnp.asarray(observed)), dtype=np.float32)
    if not len(samples["states"]) == len(predicted) == len(observed):
        raise ValueError("Sample and transition counts differ")

    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    actions = _normalized_actions(policy, samples)
    if actions.shape[1] != 10:
        raise ValueError(f"Expected a 10-step action chunk, got {actions.shape}")
    train_indices, validation_indices = _episode_split(
        samples["episode_indices"], samples["task_indices"]
    )
    action_array = jnp.asarray(actions)
    prior_array = jnp.asarray(predicted)
    observed_array = jnp.asarray(observed)

    mode_metrics: dict[str, Any] = {}
    parameter_sets = {}
    for mode in ("fixed", "coflow"):
        metrics, parameters = _train_mode(
            mode,
            action_array,
            prior_array,
            observed_array,
            train_indices,
            validation_indices,
            args,
        )
        mode_metrics[mode] = metrics
        parameter_sets[mode] = parameters

    result = {
        "sample_count": int(len(actions)),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "transition_weight": args.transition_weight,
        "width": args.width,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "solver_steps": args.solver_steps,
        "modes": mode_metrics,
        "coflow_minus_fixed": {
            "heldout_action_flow_loss": mode_metrics["coflow"]["heldout_action_flow_loss"]
            - mode_metrics["fixed"]["heldout_action_flow_loss"],
            "integrated_action_mse": mode_metrics["coflow"]["integrated_action_mse"]
            - mode_metrics["fixed"]["integrated_action_mse"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, parameters in parameter_sets.items():
        with (args.output_dir / f"{mode}_params.msgpack").open("wb") as handle:
            handle.write(serialization.to_bytes(parameters))
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
