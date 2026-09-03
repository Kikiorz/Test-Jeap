#!/usr/bin/env python3
"""Small, episode-held-out test for adapting JEPA-WAM to a new future horizon.

This is a Phase-0 falsification utility, not the final training pipeline.  It
caches frozen future-query states from a released policy checkpoint and asks
whether a lightweight alignment-head update can predict V-JEPA targets at the
action horizon.  The split is by episode and every reported control uses the
same frozen sample archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--train-components",
        choices=("out", "head"),
        default="out",
        help="Train only the final alignment projection or the complete alignment head.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _make_episode_split(episode_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put alternating episodes into train/validation without frame leakage."""
    unique = np.unique(episode_indices)
    if len(unique) < 2:
        raise ValueError("At least two episodes are required")
    train_episodes = set(unique[::2].tolist())
    validation_episodes = set(unique[1::2].tolist())
    if not validation_episodes:
        validation_episodes.add(unique[-1].item())
        train_episodes.discard(unique[-1].item())
    train = np.flatnonzero(np.isin(episode_indices, list(train_episodes)))
    validation = np.flatnonzero(np.isin(episode_indices, list(validation_episodes)))
    if not len(train) or not len(validation):
        raise ValueError("Episode split produced an empty partition")
    return train, validation


def _cache_query_states(policy: Any, samples: Any, batch_size: int) -> np.ndarray:
    from flax import nnx

    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    @nnx.jit
    def forward(model, observation):
        observation = model_lib.preprocess_observation(
            None,
            observation,
            train=False,
            geometric_augmentation=False,
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=mask, positions=positions
        )
        return prefix_output[:, -model.vjepa_num_queries :]

    result: list[np.ndarray] = []
    count = len(samples["states"])
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        transformed = []
        for index in range(start, end):
            raw = {
                "observation/state": samples["states"][index],
                "observation/image": samples["current_base"][index],
                "observation/wrist_image": samples["current_wrist"][index],
                "prompt": str(samples["prompts"][index]),
            }
            transformed.append(policy._input_transform(raw))
        batch = jax.tree.map(lambda *values: np.stack(values), *transformed)
        observation = model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
        result.append(np.asarray(forward(policy._model, observation), dtype=np.float32))
        print(f"query-cache: {end}/{count}", flush=True)
    return np.concatenate(result, axis=0)


def _extract_head(model: Any) -> dict[str, jax.Array]:
    return {
        "norm_scale": jnp.asarray(model.vjepa_alignment_norm.scale.value, dtype=jnp.float32),
        "norm_bias": jnp.asarray(model.vjepa_alignment_norm.bias.value, dtype=jnp.float32),
        "in_kernel": jnp.asarray(model.vjepa_alignment_in.kernel.value, dtype=jnp.float32),
        "in_bias": jnp.asarray(model.vjepa_alignment_in.bias.value, dtype=jnp.float32),
        "out_kernel": jnp.asarray(model.vjepa_alignment_out.kernel.value, dtype=jnp.float32),
        "out_bias": jnp.asarray(model.vjepa_alignment_out.bias.value, dtype=jnp.float32),
    }


def _predict(params: dict[str, jax.Array], query: jax.Array) -> jax.Array:
    mean = jnp.mean(query, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(query - mean), axis=-1, keepdims=True)
    hidden = (query - mean) * jax.lax.rsqrt(variance + 1e-6)
    hidden = hidden * params["norm_scale"] + params["norm_bias"]
    hidden = jax.nn.gelu(hidden @ params["in_kernel"] + params["in_bias"])
    value = hidden @ params["out_kernel"] + params["out_bias"]
    value = value.reshape(value.shape[0], 8, 8, value.shape[-1])
    value = jax.image.resize(value, (value.shape[0], 24, 24, value.shape[-1]), method="linear")
    value = value.reshape(value.shape[0], 24 * 24, value.shape[-1])
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _cosine_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-6)
    return np.mean(1.0 - np.sum(first * second, axis=-1), axis=-1)


def _metrics(
    prediction: np.ndarray,
    matched: np.ndarray,
    nochange: np.ndarray,
    mismatched: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
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
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("steps, batch-size and learning-rate must be positive")

    from openpi.policies import policy_config
    from openpi.training import config as config_lib

    samples = np.load(args.samples, allow_pickle=False)
    matched = np.load(args.teacher_dir / "teacher_matched.npy", mmap_mode="r")
    nochange = np.load(args.teacher_dir / "teacher_nochange.npy", mmap_mode="r")
    mismatched = np.load(args.teacher_dir / "teacher_mismatched.npy", mmap_mode="r")
    if not (len(samples["states"]) == len(matched) == len(nochange) == len(mismatched)):
        raise ValueError("Sample and teacher target counts differ")
    train_indices, validation_indices = _make_episode_split(samples["episode_indices"])

    policy = policy_config.create_trained_policy(
        config_lib.get_config(args.config), args.policy_checkpoint
    )
    query = _cache_query_states(policy, samples, args.batch_size)
    params = _extract_head(policy._model)
    frozen_names = {"norm_scale", "norm_bias", "in_kernel", "in_bias"} if args.train_components == "out" else set()
    labels = {name: ("train" if name not in frozen_names else "freeze") for name in params}
    transform = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "train": optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
                "freeze": optax.set_to_zero(),
            },
            labels,
        ),
    )
    optimizer_state = transform.init(params)

    query_device = jnp.asarray(query)
    target_device = jnp.asarray(np.asarray(matched, dtype=np.float32))

    def loss_fn(head, batch_query, batch_target):
        prediction = _predict(head, batch_query)
        target = batch_target / jnp.maximum(jnp.linalg.norm(batch_target, axis=-1, keepdims=True), 1e-6)
        return jnp.mean(1.0 - jnp.sum(prediction * target, axis=-1))

    @jax.jit
    def train_step(head, state, batch_query, batch_target):
        loss, gradients = jax.value_and_grad(loss_fn)(head, batch_query, batch_target)
        updates, state = transform.update(gradients, state, head)
        return optax.apply_updates(head, updates), state, loss, optax.global_norm(updates)

    baseline_prediction = np.asarray(_predict(params, query_device), dtype=np.float16)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        batch_indices = rng.choice(train_indices, size=args.batch_size, replace=len(train_indices) < args.batch_size)
        params, optimizer_state, loss, gradient_norm = train_step(
            params,
            optimizer_state,
            query_device[batch_indices],
            target_device[batch_indices],
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {
                "step": step + 1,
                "train_batch_loss": float(loss),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    adapted_prediction = np.asarray(_predict(params, query_device), dtype=np.float16)
    result = {
        "config": args.config,
        "policy_checkpoint": str(args.policy_checkpoint.resolve()),
        "samples": str(args.samples.resolve()),
        "teacher_dir": str(args.teacher_dir.resolve()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_components": args.train_components,
        "train_episode_count": int(len(np.unique(samples["episode_indices"][train_indices]))),
        "validation_episode_count": int(len(np.unique(samples["episode_indices"][validation_indices]))),
        "baseline_train": _metrics(baseline_prediction, matched, nochange, mismatched, train_indices),
        "baseline_validation": _metrics(
            baseline_prediction, matched, nochange, mismatched, validation_indices
        ),
        "adapted_train": _metrics(adapted_prediction, matched, nochange, mismatched, train_indices),
        "adapted_validation": _metrics(
            adapted_prediction, matched, nochange, mismatched, validation_indices
        ),
        "history": history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "query_states.npy", query.astype(np.float16), allow_pickle=False)
    np.save(args.output_dir / "prediction_before.npy", baseline_prediction, allow_pickle=False)
    np.save(args.output_dir / "prediction_after.npy", adapted_prediction, allow_pickle=False)
    np.savez(
        args.output_dir / "alignment_head_after.npz",
        **{name: np.asarray(value, dtype=np.float32) for name, value in params.items()},
    )
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
