#!/usr/bin/env python3
"""Measure whether one tracker-free JEPA update improves a later pi0.5 action.

This is a diagnostic, not the final meta-training loop.  A zero-initialized
low-rank image-token adapter is attached to a released JEPA-WAM policy.  For an
episode-ordered support/query pair, the support update uses only the H10 V-JEPA
prediction loss.  The query metric is the frozen policy's active-action flow
loss.  Correct, no-change and within-task-shuffled future targets are compared
under identical update norms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--nochange-teacher", type=Path, required=True)
    parser.add_argument("--alignment-head", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--inner-learning-rates", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--seed", type=int, default=101)
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


def _observation(policy: Any, samples: Any, index: int) -> model_lib.Observation:
    transformed = policy._input_transform(_raw_sample(samples, index))
    transformed = {key: value for key, value in transformed.items() if key != "actions"}
    batched = _stack([transformed])
    return model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, batched))


def _validation_pairs(samples: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    episodes = np.asarray(samples["episode_indices"])
    tasks = np.asarray(samples["task_indices"])
    frames = np.asarray(samples["frame_indices"])
    validation_episodes: list[int] = []
    for task in np.unique(tasks):
        task_episodes = np.unique(episodes[tasks == task])
        validation_episodes.extend(task_episodes[1::2].tolist())
    for episode in validation_episodes:
        indices = np.flatnonzero(episodes == episode)
        indices = indices[np.argsort(frames[indices])]
        pairs.extend(zip(indices[:-1].tolist(), indices[1:].tolist(), strict=True))
    return pairs


def _load_h10_alignment_head(model: pi0.Pi0, path: Path) -> None:
    values = np.load(path)
    assignments = (
        (model.vjepa_alignment_norm.scale, values["norm_scale"]),
        (model.vjepa_alignment_norm.bias, values["norm_bias"]),
        (model.vjepa_alignment_in.kernel, values["in_kernel"]),
        (model.vjepa_alignment_in.bias, values["in_bias"]),
        (model.vjepa_alignment_out.kernel, values["out_kernel"]),
        (model.vjepa_alignment_out.bias, values["out_bias"]),
    )
    for variable, value in assignments:
        variable.value = jnp.asarray(value, dtype=variable.value.dtype)


def _normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _tree_difference_norm(left: Any, right: Any) -> jax.Array:
    squares = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.sum(jnp.square(x - y)), left, right))
    return jnp.sqrt(jnp.sum(jnp.stack(squares)))


def main() -> None:
    args = parse_args()
    if args.max_pairs < 1 or args.adapter_rank < 1:
        raise ValueError("max-pairs and adapter-rank must be positive")

    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    teacher = np.asarray(np.load(args.teacher, mmap_mode="r"))
    nochange = np.asarray(np.load(args.nochange_teacher, mmap_mode="r"))
    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    model = policy._model
    _load_h10_alignment_head(model, args.alignment_head)
    model.use_jepa_ttt_adapter = True
    model.jepa_ttt_adapter = pi0.JepaTTTAdapter(2048, args.adapter_rank, rngs=nnx.Rngs(args.seed))
    model.eval()

    graphdef, state = nnx.split(model)
    adapter_filter = nnx_utils.PathRegex("jepa_ttt_adapter/.*")
    adapter_state, frozen_state = state.split(adapter_filter, ...)
    optimizer = optax.sgd(1.0)
    optimizer_state = optimizer.init(adapter_state)

    def support_loss(parameters, observation, target):
        value = nnx.merge(graphdef, frozen_state, parameters)
        prediction = value.predict_vjepa_from_observation(observation)
        return jnp.mean(1.0 - jnp.sum(_normalize(prediction) * _normalize(target), axis=-1))

    support_value_and_grad = jax.jit(jax.value_and_grad(support_loss))

    @jax.jit
    def query_loss(parameters, observation, action_tau, time, target_velocity):
        value = nnx.merge(graphdef, frozen_state, parameters)
        velocity = value.predict_action_velocity(observation, action_tau, time)
        error = velocity[..., : args.active_action_dim] - target_velocity[..., : args.active_action_dim]
        return jnp.mean(jnp.square(error))

    @jax.jit
    def update(parameters, gradients, learning_rate):
        updates, _ = optimizer.update(gradients, optimizer_state, parameters)
        return optax.apply_updates(
            parameters, jax.tree.map(lambda value: learning_rate * value, updates)
        )

    first_tuple_for_sample: dict[int, int] = {}
    for tuple_index, sample_index in enumerate(np.asarray(cache["sample_indices"])):
        first_tuple_for_sample.setdefault(int(sample_index), tuple_index)

    pairs = _validation_pairs(samples)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    validation_tasks = np.asarray(samples["task_indices"])
    sample_episodes = np.asarray(samples["episode_indices"])
    shuffled_target_index: dict[int, int] = {}
    for support_index, _ in pairs:
        candidates = np.flatnonzero(
            (validation_tasks == validation_tasks[support_index])
            & (sample_episodes != sample_episodes[support_index])
        ).tolist()
        if not candidates:
            candidates = np.flatnonzero(
                (validation_tasks == validation_tasks[support_index])
                & (np.arange(len(validation_tasks)) != support_index)
            ).tolist()
        shuffled_target_index[support_index] = candidates[0] if candidates else support_index

    records: list[dict[str, Any]] = []
    for pair_number, (support_index, query_index) in enumerate(pairs):
        support_observation = _observation(policy, samples, support_index)
        query_observation = _observation(policy, samples, query_index)
        query_tuple = first_tuple_for_sample[query_index]
        query_action_tau = jnp.asarray(cache["action_tau"][query_tuple : query_tuple + 1])
        query_time = jnp.asarray(cache["time"][query_tuple : query_tuple + 1])
        query_target = jnp.asarray(cache["target_velocity"][query_tuple : query_tuple + 1])
        correct_target = jnp.asarray(teacher[support_index : support_index + 1])
        nochange_target = jnp.asarray(nochange[support_index : support_index + 1])
        shuffled_index = shuffled_target_index[support_index]
        shuffled_target = jnp.asarray(teacher[shuffled_index : shuffled_index + 1])

        base_query_loss = float(
            query_loss(adapter_state, query_observation, query_action_tau, query_time, query_target)
        )
        record: dict[str, Any] = {
            "pair": pair_number,
            "support_index": support_index,
            "query_index": query_index,
            "task_index": int(validation_tasks[support_index]),
            "base_query_action_loss": base_query_loss,
            "updates": {},
        }
        for target_name, target in (
            ("correct", correct_target),
            ("nochange", nochange_target),
            ("within_task_shuffled", shuffled_target),
        ):
            before_jepa, gradients = support_value_and_grad(adapter_state, support_observation, target)
            grad_norm = float(optax.global_norm(gradients))
            per_rate = {}
            for learning_rate in args.inner_learning_rates:
                adapted = update(adapter_state, gradients, jnp.asarray(learning_rate, dtype=jnp.float32))
                after_jepa = float(support_loss(adapted, support_observation, target))
                after_query = float(
                    query_loss(adapted, query_observation, query_action_tau, query_time, query_target)
                )
                per_rate[str(learning_rate)] = {
                    "support_jepa_before": float(before_jepa),
                    "support_jepa_after": after_jepa,
                    "query_action_after": after_query,
                    "query_action_delta": after_query - base_query_loss,
                    "adapter_update_norm": float(_tree_difference_norm(adapted, adapter_state)),
                }
            record["updates"][target_name] = {"gradient_norm": grad_norm, "rates": per_rate}
        records.append(record)
        print(json.dumps(record), flush=True)

    summary: dict[str, Any] = {
        "pair_count": len(records),
        "adapter_rank": args.adapter_rank,
        "active_action_dim": args.active_action_dim,
        "rates": {},
    }
    for learning_rate in args.inner_learning_rates:
        key = str(learning_rate)
        rate_result: dict[str, Any] = {}
        for target_name in ("correct", "nochange", "within_task_shuffled"):
            deltas = np.asarray(
                [record["updates"][target_name]["rates"][key]["query_action_delta"] for record in records]
            )
            jepa_deltas = np.asarray(
                [
                    record["updates"][target_name]["rates"][key]["support_jepa_after"]
                    - record["updates"][target_name]["rates"][key]["support_jepa_before"]
                    for record in records
                ]
            )
            rate_result[target_name] = {
                "mean_query_action_delta": float(np.mean(deltas)),
                "median_query_action_delta": float(np.median(deltas)),
                "fraction_query_improved": float(np.mean(deltas < 0)),
                "mean_support_jepa_delta": float(np.mean(jepa_deltas)),
            }
        summary["rates"][key] = rate_result

    result = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
