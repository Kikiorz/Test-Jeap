#!/usr/bin/env python3
"""First-order meta-align one JEPA TTT step with a later pi0.5 action.

The inner update uses only the H10 JEPA prediction loss.  The outer update is
the original pi0.5 active-action flow loss on a later state from the same
episode.  Only the shared low-rank image-token adapter is optimized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flax import serialization
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import pi0
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
import run_jepa_ttt_gradient_gate as gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--nochange-teacher", type=Path, required=True)
    parser.add_argument("--alignment-head", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--inner-learning-rate", type=float, default=0.1)
    parser.add_argument("--outer-learning-rate", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-validation-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=113)
    return parser.parse_args()


def _split_pairs(samples: Any) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    episodes = np.asarray(samples["episode_indices"])
    tasks = np.asarray(samples["task_indices"])
    frames = np.asarray(samples["frame_indices"])
    train_pairs: list[tuple[int, int]] = []
    validation_pairs: list[tuple[int, int]] = []
    for task in np.unique(tasks):
        task_episodes = np.unique(episodes[tasks == task])
        for split, selected_episodes in (
            (train_pairs, task_episodes[::2]),
            (validation_pairs, task_episodes[1::2]),
        ):
            for episode in selected_episodes:
                indices = np.flatnonzero(episodes == episode)
                indices = indices[np.argsort(frames[indices])]
                split.extend(zip(indices[:-1].tolist(), indices[1:].tolist(), strict=True))
    return train_pairs, validation_pairs


def _normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _copy_state(state: Any) -> Any:
    return jax.tree.map(lambda value: jnp.array(value), state)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.max_validation_pairs < 1:
        raise ValueError("steps and max-validation-pairs must be positive")

    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    teacher = np.asarray(np.load(args.teacher, mmap_mode="r"))
    nochange = np.asarray(np.load(args.nochange_teacher, mmap_mode="r"))
    train_pairs, validation_pairs = _split_pairs(samples)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_pairs)
    rng.shuffle(validation_pairs)
    validation_pairs = validation_pairs[: args.max_validation_pairs]

    first_tuple_for_sample: dict[int, int] = {}
    for tuple_index, sample_index in enumerate(np.asarray(cache["sample_indices"])):
        first_tuple_for_sample.setdefault(int(sample_index), tuple_index)

    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    model = policy._model
    gate._load_h10_alignment_head(model, args.alignment_head)
    model.use_jepa_ttt_adapter = True
    model.jepa_ttt_adapter = pi0.JepaTTTAdapter(2048, args.adapter_rank, rngs=nnx.Rngs(args.seed))
    model.eval()

    graphdef, state = nnx.split(model)
    adapter_state, frozen_state = state.split(nnx_utils.PathRegex("jepa_ttt_adapter/.*"), ...)
    initial_adapter_state = _copy_state(adapter_state)
    inner_optimizer = optax.sgd(args.inner_learning_rate)
    inner_optimizer_state = inner_optimizer.init(adapter_state)
    outer_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.outer_learning_rate))
    outer_optimizer_state = outer_optimizer.init(adapter_state)

    def support_loss(parameters, observation, target):
        value = nnx.merge(graphdef, frozen_state, parameters)
        prediction = value.predict_vjepa_from_observation(observation)
        return jnp.mean(1.0 - jnp.sum(_normalize(prediction) * _normalize(target), axis=-1))

    def query_loss(parameters, observation, action_tau, time, target_velocity):
        value = nnx.merge(graphdef, frozen_state, parameters)
        velocity = value.predict_action_velocity(observation, action_tau, time)
        error = velocity[..., : args.active_action_dim] - target_velocity[..., : args.active_action_dim]
        return jnp.mean(jnp.square(error))

    def inner_update(parameters, gradients, scale):
        updates, _ = inner_optimizer.update(gradients, inner_optimizer_state, parameters)
        updates = jax.tree.map(lambda value: scale * value, updates)
        return optax.apply_updates(parameters, updates)

    @jax.jit
    def meta_step(
        parameters,
        optimizer_state,
        support_observation,
        support_target,
        query_observation,
        query_action_tau,
        query_time,
        query_target_velocity,
        inner_scale,
    ):
        def meta_objective(value):
            support_value, support_gradients = jax.value_and_grad(support_loss)(
                value, support_observation, support_target
            )
            # First-order MAML: deployment still uses the exact JEPA gradient,
            # while the outer graph avoids a full second-order 2B-model Hessian.
            support_gradients = jax.tree.map(jax.lax.stop_gradient, support_gradients)
            adapted = inner_update(value, support_gradients, inner_scale)
            query_value = query_loss(
                adapted, query_observation, query_action_tau, query_time, query_target_velocity
            )
            return query_value, (support_value, optax.global_norm(support_gradients))

        (loss, (support_value, inner_grad_norm)), gradients = jax.value_and_grad(
            meta_objective, has_aux=True
        )(parameters)
        updates, optimizer_state = outer_optimizer.update(gradients, optimizer_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, optimizer_state, loss, support_value, inner_grad_norm, optax.global_norm(gradients)

    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        support_index, query_index = train_pairs[step % len(train_pairs)]
        support_observation = gate._observation(policy, samples, support_index)
        query_observation = gate._observation(policy, samples, query_index)
        query_tuple = first_tuple_for_sample[query_index]
        adapter_state, outer_optimizer_state, loss, support_value, inner_grad_norm, outer_grad_norm = meta_step(
            adapter_state,
            outer_optimizer_state,
            support_observation,
            jnp.asarray(teacher[support_index : support_index + 1]),
            query_observation,
            jnp.asarray(cache["action_tau"][query_tuple : query_tuple + 1]),
            jnp.asarray(cache["time"][query_tuple : query_tuple + 1]),
            jnp.asarray(cache["target_velocity"][query_tuple : query_tuple + 1]),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        record = {
            "step": step + 1,
            "support_index": support_index,
            "query_index": query_index,
            "support_jepa_loss": float(support_value),
            "post_update_query_action_loss": float(loss),
            "inner_gradient_norm": float(inner_grad_norm),
            "outer_gradient_norm": float(outer_grad_norm),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    tasks = np.asarray(samples["task_indices"])
    episodes = np.asarray(samples["episode_indices"])

    def evaluate(parameters, name: str) -> dict[str, Any]:
        records = []
        for support_index, query_index in validation_pairs:
            query_tuple = first_tuple_for_sample[query_index]
            support_observation = gate._observation(policy, samples, support_index)
            query_observation = gate._observation(policy, samples, query_index)
            action_tau = jnp.asarray(cache["action_tau"][query_tuple : query_tuple + 1])
            time = jnp.asarray(cache["time"][query_tuple : query_tuple + 1])
            target = jnp.asarray(cache["target_velocity"][query_tuple : query_tuple + 1])
            candidates = np.flatnonzero(
                (tasks == tasks[support_index]) & (episodes != episodes[support_index])
            )
            wrong_index = int(candidates[0]) if len(candidates) else support_index
            item: dict[str, Any] = {
                "support_index": support_index,
                "query_index": query_index,
                "task_index": int(tasks[support_index]),
            }
            for target_name, support_target in (
                ("correct", teacher[support_index : support_index + 1]),
                ("nochange", nochange[support_index : support_index + 1]),
                ("within_task_shuffled", teacher[wrong_index : wrong_index + 1]),
            ):
                _, _, before, jepa_before, _, _ = meta_step(
                    parameters,
                    outer_optimizer_state,
                    support_observation,
                    jnp.asarray(support_target),
                    query_observation,
                    action_tau,
                    time,
                    target,
                    jnp.asarray(0.0, dtype=jnp.float32),
                )
                _, _, after, _, _, _ = meta_step(
                    parameters,
                    outer_optimizer_state,
                    support_observation,
                    jnp.asarray(support_target),
                    query_observation,
                    action_tau,
                    time,
                    target,
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
                item[target_name] = {
                    "query_before": float(before),
                    "query_after": float(after),
                    "query_delta": float(after - before),
                    "support_jepa_before": float(jepa_before),
                }
            records.append(item)

        summary = {}
        for target_name in ("correct", "nochange", "within_task_shuffled"):
            deltas = np.asarray([record[target_name]["query_delta"] for record in records])
            summary[target_name] = {
                "mean_query_delta": float(np.mean(deltas)),
                "median_query_delta": float(np.median(deltas)),
                "fraction_query_improved": float(np.mean(deltas < 0)),
                "mean_query_before": float(np.mean([r[target_name]["query_before"] for r in records])),
                "mean_query_after": float(np.mean([r[target_name]["query_after"] for r in records])),
            }
        return {"name": name, "summary": summary, "records": records}

    result = {
        "config": {
            "steps": args.steps,
            "adapter_rank": args.adapter_rank,
            "inner_learning_rate": args.inner_learning_rate,
            "outer_learning_rate": args.outer_learning_rate,
            "validation_pairs": len(validation_pairs),
            "first_order": True,
        },
        "history": history,
        "untrained": evaluate(initial_adapter_state, "untrained_zero_adapter"),
        "meta_aligned": evaluate(adapter_state, "meta_aligned_adapter"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_metrics = args.output_dir / ".metrics.json.tmp"
    with temporary_metrics.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_metrics.replace(args.output_dir / "metrics.json")
    with (args.output_dir / "adapter_params.msgpack").open("wb") as handle:
        handle.write(serialization.to_bytes(adapter_state.to_pure_dict()))
    print(json.dumps({key: value["summary"] for key, value in result.items() if key in ("untrained", "meta_aligned")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
