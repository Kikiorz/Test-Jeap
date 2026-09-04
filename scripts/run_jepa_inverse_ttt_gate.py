#!/usr/bin/env python3
"""Test a tracker-free JEPA + executed-action online update.

The support loss uses only signals available after a robot executes an action:
the current-only JEPA-WAM prediction, the realized V-JEPA transition, and the
already-known executed action.  A frozen inverse readout supplies an
action-decodability loss.  The metric is the frozen pi0.5 action-flow loss on a
later state from the same held-out episode.

This is a mechanism gate, not a training or rollout script.
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

from openpi.models import pi0
from openpi.models import transition_inverse
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
import run_jepa_ttt_gradient_gate as gate
import run_transition_inverse_gate as inverse_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--nochange-teacher", type=Path, required=True)
    parser.add_argument("--alignment-head", type=Path, required=True)
    parser.add_argument("--inverse-params", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--inverse-width", type=int, default=128)
    parser.add_argument("--inverse-heads", type=int, default=4)
    parser.add_argument("--inverse-weight", type=float, default=1.0)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--query-draws", type=int, default=4)
    parser.add_argument("--inner-learning-rates", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--seed", type=int, default=127)
    return parser.parse_args()


def _pool_24_to_8(value: jax.Array) -> jax.Array:
    if value.ndim != 3 or value.shape[1] != 24 * 24:
        raise ValueError(f"Expected [B,576,D], got {value.shape}")
    value = value.reshape(value.shape[0], 8, 3, 8, 3, value.shape[-1])
    return value.mean(axis=(2, 4)).reshape(value.shape[0], 64, value.shape[-1])


def _copy_state(state: Any) -> Any:
    return jax.tree.map(lambda value: jnp.array(value), state)


def main() -> None:
    args = parse_args()
    if args.max_pairs < 1 or args.query_draws < 1 or args.inverse_weight < 0:
        raise ValueError("max-pairs/query-draws must be positive and inverse-weight nonnegative")

    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    teacher = np.asarray(np.load(args.teacher, mmap_mode="r"))
    nochange = np.asarray(np.load(args.nochange_teacher, mmap_mode="r"))
    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    states_numpy, actions_numpy = inverse_gate._policy_arrays(policy, samples)
    states = jnp.asarray(states_numpy)
    actions = jnp.asarray(actions_numpy)

    inverse_model = transition_inverse.TransitionInverseDecoder(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        width=args.inverse_width,
        num_heads=args.inverse_heads,
        mode="transition",
    )
    inverse_template = inverse_model.init(
        jax.random.key(args.seed),
        jnp.zeros((1, 64, teacher.shape[-1]), dtype=jnp.float32),
        states[:1],
    )["params"]
    inverse_params = serialization.from_bytes(inverse_template, args.inverse_params.read_bytes())

    model = policy._model
    gate._load_h10_alignment_head(model, args.alignment_head)
    model.use_jepa_ttt_adapter = True
    model.jepa_ttt_adapter = pi0.JepaTTTAdapter(2048, args.adapter_rank, rngs=nnx.Rngs(args.seed))
    model.eval()
    graphdef, state = nnx.split(model)
    adapter_state, frozen_state = state.split(nnx_utils.PathRegex("jepa_ttt_adapter/.*"), ...)
    adapter_state = _copy_state(adapter_state)

    def support_loss(parameters, observation, transition_target, state_value, action_target):
        value = nnx.merge(graphdef, frozen_state, parameters)
        prediction = value.predict_vjepa_from_observation(observation)
        prediction_normalized = gate._normalize(prediction)
        target_normalized = gate._normalize(transition_target)
        jepa_loss = jnp.mean(1.0 - jnp.sum(prediction_normalized * target_normalized, axis=-1))
        inverse_prediction = inverse_model.apply(
            {"params": inverse_params}, gate._normalize(_pool_24_to_8(prediction)), state_value
        )
        inverse_loss = jnp.mean(jnp.square(inverse_prediction - action_target))
        return jepa_loss + args.inverse_weight * inverse_loss, (jepa_loss, inverse_loss)

    support_value_and_grad = jax.jit(jax.value_and_grad(support_loss, has_aux=True))

    @jax.jit
    def query_loss(parameters, observation, action_tau, time, target_velocity):
        value = nnx.merge(graphdef, frozen_state, parameters)
        velocity = value.predict_action_velocity(observation, action_tau, time)
        error = velocity[..., : args.active_action_dim] - target_velocity[..., : args.active_action_dim]
        return jnp.mean(jnp.square(error))

    @jax.jit
    def update(parameters, gradients, learning_rate):
        return jax.tree.map(lambda parameter, gradient: parameter - learning_rate * gradient, parameters, gradients)

    tuples_for_sample: dict[int, list[int]] = {}
    for tuple_index, sample_index in enumerate(np.asarray(cache["sample_indices"])):
        tuples_for_sample.setdefault(int(sample_index), []).append(tuple_index)

    pairs = gate._validation_pairs(samples)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    task_indices = np.asarray(samples["task_indices"])
    episode_indices = np.asarray(samples["episode_indices"])

    records: list[dict[str, Any]] = []
    for pair_number, (support_index, query_index) in enumerate(pairs):
        candidates = np.flatnonzero(
            (task_indices == task_indices[support_index])
            & (episode_indices != episode_indices[support_index])
        )
        wrong_index = int(candidates[0]) if len(candidates) else support_index
        support_observation = gate._observation(policy, samples, support_index)
        query_observation = gate._observation(policy, samples, query_index)
        query_tuples = tuples_for_sample[query_index][: args.query_draws]

        def mean_query_loss(parameters) -> float:
            values = []
            for query_tuple in query_tuples:
                values.append(
                    float(
                        query_loss(
                            parameters,
                            query_observation,
                            jnp.asarray(cache["action_tau"][query_tuple : query_tuple + 1]),
                            jnp.asarray(cache["time"][query_tuple : query_tuple + 1]),
                            jnp.asarray(cache["target_velocity"][query_tuple : query_tuple + 1]),
                        )
                    )
                )
            return float(np.mean(values))

        base_query = mean_query_loss(adapter_state)
        item: dict[str, Any] = {
            "pair": pair_number,
            "support_index": int(support_index),
            "query_index": int(query_index),
            "task_index": int(task_indices[support_index]),
            "base_query_action_loss": base_query,
            "updates": {},
        }
        variants = (
            ("correct", teacher[support_index : support_index + 1], actions[support_index : support_index + 1]),
            ("nochange", nochange[support_index : support_index + 1], actions[support_index : support_index + 1]),
            (
                "within_task_shuffled_transition",
                teacher[wrong_index : wrong_index + 1],
                actions[support_index : support_index + 1],
            ),
            (
                "within_task_shuffled_action",
                teacher[support_index : support_index + 1],
                actions[wrong_index : wrong_index + 1],
            ),
        )
        for name, transition_target, action_target in variants:
            (support_total, (jepa_value, inverse_value)), gradients = support_value_and_grad(
                adapter_state,
                support_observation,
                jnp.asarray(transition_target),
                states[support_index : support_index + 1],
                action_target,
            )
            per_rate = {}
            for learning_rate in args.inner_learning_rates:
                adapted = update(adapter_state, gradients, jnp.asarray(learning_rate, dtype=jnp.float32))
                after_query = mean_query_loss(adapted)
                per_rate[str(learning_rate)] = {
                    "query_action_after": after_query,
                    "query_action_delta": after_query - base_query,
                    "adapter_update_norm": float(gate._tree_difference_norm(adapted, adapter_state)),
                }
            item["updates"][name] = {
                "support_total": float(support_total),
                "support_jepa": float(jepa_value),
                "support_inverse": float(inverse_value),
                "gradient_norm": float(jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree.map(lambda value: jnp.sum(jnp.square(value)), gradients),
                    initializer=jnp.asarray(0.0),
                ) ** 0.5),
                "rates": per_rate,
            }
        records.append(item)
        print(json.dumps(item), flush=True)

    variant_names = (
        "correct",
        "nochange",
        "within_task_shuffled_transition",
        "within_task_shuffled_action",
    )
    rate_summary: dict[str, Any] = {}
    for learning_rate in args.inner_learning_rates:
        key = str(learning_rate)
        rate_summary[key] = {}
        for name in variant_names:
            deltas = np.asarray([record["updates"][name]["rates"][key]["query_action_delta"] for record in records])
            rate_summary[key][name] = {
                "mean_query_delta": float(np.mean(deltas)),
                "median_query_delta": float(np.median(deltas)),
                "fraction_query_improved": float(np.mean(deltas < 0)),
            }
        correct = np.asarray(
            [record["updates"]["correct"]["rates"][key]["query_action_delta"] for record in records]
        )
        controls = {
            name: np.asarray([record["updates"][name]["rates"][key]["query_action_delta"] for record in records])
            for name in variant_names[1:]
        }
        rate_summary[key]["specificity"] = {
            f"correct_better_than_{name}": int(np.sum(correct < values))
            for name, values in controls.items()
        }
        rate_summary[key]["specificity"]["correct_better_than_all_controls"] = int(
            np.sum(np.logical_and.reduce([correct < values for values in controls.values()]))
        )

    result = {
        "config": {
            "pairs": len(records),
            "query_draws": args.query_draws,
            "inverse_weight": args.inverse_weight,
            "inner_learning_rates": args.inner_learning_rates,
            "seed": args.seed,
        },
        "summary": rate_summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
