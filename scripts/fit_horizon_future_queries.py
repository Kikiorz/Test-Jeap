#!/usr/bin/env python3
"""Small gate for relearning JEPA-WAM future queries at the action horizon.

The released Pi0.5 checkpoint learned its future-query representation with a
longer target offset.  This script keeps the VLM and Action Expert frozen and
updates only the 64 future-query tokens plus the existing alignment head on an
episode-disjoint H-step sample archive.  A frozen inverse decoder trained on
realized transitions supplies a training-only action-decodability constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flax import nnx
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as model_lib
from openpi.models import transition_inverse
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--inverse-params", type=Path, required=True)
    parser.add_argument("--alignment-head", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--inverse-weight", type=float, default=0.1)
    parser.add_argument("--inverse-width", type=int, default=128)
    parser.add_argument("--inverse-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=263)
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _pool_24_to_8(value: jax.Array) -> jax.Array:
    if value.ndim != 3 or value.shape[1] != 24 * 24:
        raise ValueError(f"Expected [B,576,D], got {value.shape}")
    value = value.reshape(value.shape[0], 8, 3, 8, 3, value.shape[-1])
    return value.mean(axis=(2, 4)).reshape(value.shape[0], 64, value.shape[-1])


def _episode_split(episode_indices: np.ndarray, task_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for task in np.unique(task_indices):
        episodes = np.unique(episode_indices[task_indices == task])
        if len(episodes) < 2:
            raise ValueError(f"Task {task} has fewer than two episodes")
        train_episodes.update(episodes[::2].tolist())
        validation_episodes.update(episodes[1::2].tolist())
    return (
        np.flatnonzero(np.isin(episode_indices, list(train_episodes))),
        np.flatnonzero(np.isin(episode_indices, list(validation_episodes))),
    )


def _load_alignment_head(model: Any, path: Path) -> None:
    values = np.load(path, allow_pickle=False)
    assignments = (
        (model.vjepa_alignment_norm.scale, values["norm_scale"]),
        (model.vjepa_alignment_norm.bias, values["norm_bias"]),
        (model.vjepa_alignment_in.kernel, values["in_kernel"]),
        (model.vjepa_alignment_in.bias, values["in_bias"]),
        (model.vjepa_alignment_out.kernel, values["out_kernel"]),
        (model.vjepa_alignment_out.bias, values["out_bias"]),
    )
    for variable, value in assignments:
        if tuple(variable.value.shape) != tuple(value.shape):
            raise ValueError(f"Alignment shape mismatch: {variable.value.shape} != {value.shape}")
        variable.value = jnp.asarray(value, dtype=variable.value.dtype)


def _export_query_head(model: Any, path: Path) -> None:
    np.savez(
        path,
        query_tokens=np.asarray(model.vjepa_query_tokens.value, dtype=np.float32),
        norm_scale=np.asarray(model.vjepa_alignment_norm.scale.value, dtype=np.float32),
        norm_bias=np.asarray(model.vjepa_alignment_norm.bias.value, dtype=np.float32),
        in_kernel=np.asarray(model.vjepa_alignment_in.kernel.value, dtype=np.float32),
        in_bias=np.asarray(model.vjepa_alignment_in.bias.value, dtype=np.float32),
        out_kernel=np.asarray(model.vjepa_alignment_out.kernel.value, dtype=np.float32),
        out_bias=np.asarray(model.vjepa_alignment_out.bias.value, dtype=np.float32),
    )


def _raw_sample(samples: Any, index: int) -> dict[str, Any]:
    return {
        "observation/state": samples["states"][index],
        "observation/image": samples["current_base"][index],
        "observation/wrist_image": samples["current_wrist"][index],
        "prompt": str(samples["prompts"][index]),
        "actions": samples["action_chunks"][index],
    }


def _transform_samples(policy: Any, samples: Any) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    observations: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    executable_width = int(samples["action_chunks"].shape[-1])
    for index in range(len(samples["states"])):
        transformed = policy._input_transform(_raw_sample(samples, index))
        states.append(np.asarray(transformed["state"], dtype=np.float32))
        actions.append(np.asarray(transformed.pop("actions"), dtype=np.float32)[..., :executable_width])
        observations.append(transformed)
    return observations, np.stack(states), np.stack(actions)


def _batch_observation(observations: list[dict[str, Any]], indices: np.ndarray) -> model_lib.Observation:
    values = [observations[int(index)] for index in indices]
    batch = jax.tree.map(lambda *items: np.stack(items), *values)
    return model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, batch))


def _cosine_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-6)
    return np.mean(1.0 - np.sum(first * second, axis=-1), axis=-1)


def _prediction_metrics(
    prediction: np.ndarray,
    matched: np.ndarray,
    nochange: np.ndarray,
    mismatched: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    pred_matched = _cosine_distance(prediction[indices], matched[indices])
    pred_nochange = _cosine_distance(prediction[indices], nochange[indices])
    pred_mismatch = _cosine_distance(prediction[indices], mismatched[indices])
    persistence = _cosine_distance(nochange[indices], matched[indices])
    return {
        "count": int(len(indices)),
        "prediction_to_matched": float(pred_matched.mean()),
        "persistence_to_matched": float(persistence.mean()),
        "relative_improvement_over_persistence": float(
            np.mean((persistence - pred_matched) / np.maximum(persistence, 1e-6))
        ),
        "matched_vs_nochange_margin": float((pred_nochange - pred_matched).mean()),
        "matched_vs_mismatched_margin": float((pred_mismatch - pred_matched).mean()),
        "matched_preference_rate": float(np.mean(pred_matched < pred_mismatch)),
    }


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.inverse_weight < 0:
        raise ValueError("steps, batch-size and learning-rate must be positive; inverse-weight must be nonnegative")

    samples = np.load(args.samples, allow_pickle=False)
    matched = np.asarray(np.load(args.teacher_dir / "teacher_matched.npy", mmap_mode="r"), dtype=np.float32)
    nochange = np.asarray(np.load(args.teacher_dir / "teacher_nochange.npy", mmap_mode="r"), dtype=np.float32)
    mismatched = np.asarray(np.load(args.teacher_dir / "teacher_mismatched.npy", mmap_mode="r"), dtype=np.float32)
    if not (len(samples["states"]) == len(matched) == len(nochange) == len(mismatched)):
        raise ValueError("Sample and teacher counts differ")
    train_indices, validation_indices = _episode_split(
        np.asarray(samples["episode_indices"]), np.asarray(samples["task_indices"])
    )

    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    model = policy._model
    if args.alignment_head is not None:
        _load_alignment_head(model, args.alignment_head)
    model.eval()
    observations, states, actions = _transform_samples(policy, samples)

    inverse_model = transition_inverse.TransitionInverseDecoder(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        width=args.inverse_width,
        num_heads=args.inverse_heads,
        mode="transition",
    )
    inverse_template = inverse_model.init(
        jax.random.key(args.seed),
        _pool_24_to_8(jnp.asarray(matched[:1])),
        jnp.asarray(states[:1]),
    )["params"]
    inverse_params = serialization.from_bytes(inverse_template, args.inverse_params.read_bytes())

    graphdef, full_state = nnx.split(model)
    trainable_filter = nnx.Any(
        nnx_utils.PathRegex("vjepa_query_tokens"),
        nnx_utils.PathRegex("vjepa_alignment_(norm|in|out)/.*"),
    )
    trainable_state, frozen_state = full_state.split(trainable_filter, ...)
    trainable_paths = ["/".join(str(part) for part in path) for path, _ in trainable_state.flat_state()]
    if "vjepa_query_tokens" not in trainable_paths or len(trainable_paths) != 7:
        raise RuntimeError(f"Unexpected trainable parameter set: {trainable_paths}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(trainable_state)

    def objective(parameters, observation, target, state, action):
        value = nnx.merge(graphdef, frozen_state, parameters)
        prediction = value.predict_vjepa_from_observation(observation)
        normalized_prediction = _normalize(prediction)
        normalized_target = _normalize(target)
        jepa_loss = jnp.mean(1.0 - jnp.sum(normalized_prediction * normalized_target, axis=-1))
        predicted_action = inverse_model.apply(
            {"params": inverse_params}, _normalize(_pool_24_to_8(prediction)), state
        )
        inverse_loss = jnp.mean(jnp.square(predicted_action - action))
        return jepa_loss + args.inverse_weight * inverse_loss, (jepa_loss, inverse_loss)

    @jax.jit
    def train_step(parameters, optimizer_value, observation, target, state, action):
        (loss, components), gradients = jax.value_and_grad(objective, has_aux=True)(
            parameters, observation, target, state, action
        )
        updates, optimizer_value = optimizer.update(gradients, optimizer_value, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, optimizer_value, loss, components, optax.global_norm(gradients)

    @jax.jit
    def predict(parameters, observation):
        value = nnx.merge(graphdef, frozen_state, parameters)
        return value.predict_vjepa_from_observation(observation)

    @jax.jit
    def inverse_error(transition, state, action):
        estimate = inverse_model.apply(
            {"params": inverse_params}, _normalize(_pool_24_to_8(transition)), state
        )
        return jnp.mean(jnp.square(estimate - action))

    def predict_indices(parameters, indices: np.ndarray) -> np.ndarray:
        outputs = []
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            outputs.append(np.asarray(predict(parameters, _batch_observation(observations, selected)), dtype=np.float16))
        return np.concatenate(outputs, axis=0)

    all_indices = np.arange(len(samples["states"]), dtype=np.int32)
    baseline_prediction = predict_indices(trainable_state, all_indices)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        selected = rng.choice(
            train_indices, size=args.batch_size, replace=len(train_indices) < args.batch_size
        ).astype(np.int32)
        trainable_state, optimizer_state, loss, components, gradient_norm = train_step(
            trainable_state,
            optimizer_state,
            _batch_observation(observations, selected),
            jnp.asarray(matched[selected]),
            jnp.asarray(states[selected]),
            jnp.asarray(actions[selected]),
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {
                "step": step + 1,
                "loss": float(loss),
                "jepa_loss": float(components[0]),
                "inverse_loss": float(components[1]),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    adapted_prediction = predict_indices(trainable_state, all_indices)
    validation = jnp.asarray(validation_indices)
    result = {
        "config": {
            "policy_config": args.config,
            "policy_checkpoint": str(args.policy_checkpoint.resolve()),
            "alignment_head": str(args.alignment_head.resolve()) if args.alignment_head else None,
            "inverse_params": str(args.inverse_params.resolve()),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "inverse_weight": args.inverse_weight,
            "trainable_paths": trainable_paths,
            "train_episode_count": int(len(np.unique(samples["episode_indices"][train_indices]))),
            "validation_episode_count": int(len(np.unique(samples["episode_indices"][validation_indices]))),
        },
        "baseline_train": _prediction_metrics(
            baseline_prediction, matched, nochange, mismatched, train_indices
        ),
        "baseline_validation": _prediction_metrics(
            baseline_prediction, matched, nochange, mismatched, validation_indices
        ),
        "adapted_train": _prediction_metrics(
            adapted_prediction, matched, nochange, mismatched, train_indices
        ),
        "adapted_validation": _prediction_metrics(
            adapted_prediction, matched, nochange, mismatched, validation_indices
        ),
        "inverse_validation": {
            "realized_mse": float(
                inverse_error(
                    jnp.asarray(matched[validation_indices]),
                    jnp.asarray(states[validation_indices]),
                    jnp.asarray(actions[validation_indices]),
                )
            ),
            "prediction_before_mse": float(
                inverse_error(
                    jnp.asarray(baseline_prediction[validation_indices]),
                    jnp.asarray(states[validation_indices]),
                    jnp.asarray(actions[validation_indices]),
                )
            ),
            "prediction_after_mse": float(
                inverse_error(
                    jnp.asarray(adapted_prediction[validation_indices]),
                    jnp.asarray(states[validation_indices]),
                    jnp.asarray(actions[validation_indices]),
                )
            ),
        },
        "history": history,
    }

    trained_model = nnx.merge(graphdef, frozen_state, trainable_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _export_query_head(trained_model, args.output_dir / "query_head_after.npz")
    np.save(args.output_dir / "prediction_before.npy", baseline_prediction, allow_pickle=False)
    np.save(args.output_dir / "prediction_after.npy", adapted_prediction, allow_pickle=False)
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
