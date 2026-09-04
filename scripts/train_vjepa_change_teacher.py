#!/usr/bin/env python3
"""Train and export the Con1 Stage-1 V-JEPA change teacher.

The inverse decoder only receives learned change tokens. Current images,
language, state, and noisy actions are deliberately absent from this trainer.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import shutil
from typing import Any

from flax import jax_utils
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.models import vjepa_change_tokenizer as tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--displacement-target-root", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-offset", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--num-change-tokens", type=int, default=16)
    parser.add_argument("--change-token-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--resampler-depth", type=int, default=3)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-width", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--max-episodes", type=int, help="Optional smoke-run limit; full training leaves this unset")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _target_path(root: Path, episode: int) -> Path:
    return root / "targets" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.npy"


def _episode_records(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if paths:
        table = pa.concat_tables([pq.read_table(path) for path in paths])
        return sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
    path = root / "meta" / "episodes.jsonl"
    if path.is_file():
        with path.open() as handle:
            return sorted((json.loads(line) for line in handle), key=lambda row: int(row["episode_index"]))
    raise FileNotFoundError("Dataset has neither meta/episodes/*.parquet nor meta/episodes.jsonl")


def _packed_episode_path(root: Path, record: dict[str, Any]) -> Path:
    return (
        root
        / "data"
        / f"chunk-{int(record['data/chunk_index']):03d}"
        / f"file-{int(record['data/file_index']):03d}.parquet"
    )


def _validate_target_contract(root: Path, future_offset: int) -> tuple[int, int]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("kind") != "vjepa2_nochange_referenced_displacement":
        raise ValueError(f"Expected a JL-free V-JEPA2 displacement cache, got {manifest.get('kind')!r}")
    if manifest.get("feature_reduction") != "none":
        raise ValueError(f"Stage 1 must not use a fixed channel projection: {manifest.get('feature_reduction')!r}")
    if int(manifest["future_offset"]) != future_offset:
        raise ValueError(f"Displacement target offset is {manifest['future_offset']}, expected {future_offset}")
    if manifest.get("tail_policy") != "exclude_t_plus_h_outside_episode":
        raise ValueError(f"Unexpected displacement tail policy: {manifest.get('tail_policy')!r}")
    shape = tuple(int(value) for value in manifest["target_shape"])
    if len(shape) != 2:
        raise ValueError(f"Expected token target shape, got {shape}")
    if shape[-1] != 1408:
        raise ValueError(f"Expected all 1408 V-JEPA2 channels, got {shape}")
    return shape


def _fixed_list_to_numpy(column: Any, width: int) -> np.ndarray:
    column = column.combine_chunks()
    if hasattr(column, "values"):
        values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
        return values.reshape(len(column), width)
    return np.asarray(column.to_pylist(), dtype=np.float32).reshape(len(column), width)


class EpisodeDataset:
    """In-memory action index with memory-mapped per-episode displacement."""

    def __init__(
        self,
        dataset_root: Path,
        displacement_root: Path,
        horizon: int,
        max_episodes: int | None = None,
    ):
        info = _read_json(dataset_root / "meta" / "info.json")
        records = _episode_records(dataset_root)
        self.actions: dict[int, np.ndarray] = {}
        self.lengths: dict[int, int] = {}
        sample_episodes: list[np.ndarray] = []
        sample_frames: list[np.ndarray] = []
        splits: list[np.ndarray] = []
        episode_count = int(info["total_episodes"])
        if len(records) != episode_count:
            raise ValueError(f"Episode metadata has {len(records)} rows, expected {episode_count}")
        if max_episodes is not None:
            episode_count = min(episode_count, max_episodes)
        for episode in range(episode_count):
            record = records[episode]
            if int(record["episode_index"]) != episode:
                raise ValueError(f"Episode metadata is non-contiguous at row {episode}: {record}")
            displacement_path = _target_path(displacement_root, episode)
            if not displacement_path.exists():
                raise FileNotFoundError(f"Missing displacement for episode {episode}: {displacement_path}")
            packed_path = _packed_episode_path(dataset_root, record)
            schema_names = pq.read_schema(packed_path).names
            action_key = "actions" if "actions" in schema_names else "action"
            table = pq.read_table(
                packed_path,
                columns=[action_key, "episode_index", "frame_index"],
                filters=[("episode_index", "=", episode)],
            )
            if table.num_rows != int(record["length"]):
                raise ValueError(
                    f"Episode {episode} has {table.num_rows} packed rows, metadata says {record['length']}"
                )
            order = np.argsort(np.asarray(table["frame_index"].to_numpy()), kind="stable")
            if not np.array_equal(np.asarray(table["frame_index"].to_numpy())[order], np.arange(table.num_rows)):
                raise ValueError(f"Episode {episode} has invalid frame_index values")
            actions = _fixed_list_to_numpy(table[action_key], 7)
            actions = actions[order]
            target_shape = np.load(displacement_path, mmap_mode="r", allow_pickle=False).shape
            valid = len(actions) - horizon
            if target_shape[0] != valid:
                raise ValueError(
                    f"Episode {episode} mismatch: actions={actions.shape}, displacement={target_shape}, horizon={horizon}"
                )
            if valid <= 0:
                continue
            self.actions[episode] = actions
            self.lengths[episode] = len(actions)
            sample_episodes.append(np.full(valid, episode, dtype=np.int32))
            sample_frames.append(np.arange(valid, dtype=np.int32))
            split = 1 if episode % 10 == 0 else (2 if episode % 10 == 1 else 0)
            splits.append(np.full(valid, split, dtype=np.uint8))
        self.sample_episode = np.concatenate(sample_episodes)
        self.sample_frame = np.concatenate(sample_frames)
        self.split = np.concatenate(splits)
        self.displacement_root = displacement_root
        self.horizon = horizon
        self._cache: dict[int, np.ndarray] = {}

    def indices(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split).astype(np.int32)

    def _features(self, episode: int) -> np.ndarray:
        if episode not in self._cache:
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[episode] = np.load(
                _target_path(self.displacement_root, episode), mmap_mode="r", allow_pickle=False
            )
        return self._cache[episode]

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        episodes = self.sample_episode[indices]
        frames = self.sample_frame[indices]
        example = self._features(int(episodes[0]))
        displacement = np.empty((len(indices), *example.shape[1:]), dtype=np.float32)
        actions = np.empty((len(indices), self.horizon, 7), dtype=np.float32)
        for episode in np.unique(episodes):
            output_rows = np.flatnonzero(episodes == episode)
            episode_frames = frames[output_rows]
            value = self._features(int(episode))
            displacement[output_rows] = value[episode_frames]
            action = self.actions[int(episode)]
            for output_row, frame in zip(output_rows, episode_frames, strict=True):
                actions[output_row] = action[frame : frame + self.horizon]
        return displacement, actions


def _load_action_stats(path: Path) -> dict[str, np.ndarray]:
    contents = _read_json(path)
    value = contents.get("norm_stats", contents)["actions"]
    return {key: np.asarray(value[key], dtype=np.float32) for key in ("q01", "q99")}


def _normalize_actions(actions: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (actions - stats["q01"]) / (stats["q99"] - stats["q01"] + 1e-6) * 2.0 - 1.0


def _huber_per_example(prediction: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
    error = jnp.abs(prediction - target)
    value = jnp.where(error <= delta, 0.5 * jnp.square(error), delta * (error - 0.5 * delta))
    return jnp.mean(value, axis=(1, 2))


def _tree_first(replica: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value[0]), replica)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    values = sorted(output_dir.glob("checkpoints/step_*/state.msgpack"))
    return values[-1] if values else None


def main() -> None:
    args = parse_args()
    target_shape = _validate_target_contract(args.displacement_target_root, args.future_offset)
    devices = jax.local_devices()
    if args.batch_size % len(devices):
        raise ValueError(f"Batch {args.batch_size} must be divisible by {len(devices)} devices")
    dataset = EpisodeDataset(
        args.dataset_root,
        args.displacement_target_root,
        args.future_offset,
        max_episodes=args.max_episodes,
    )
    train_indices = dataset.indices(0)
    validation_all = dataset.indices(1)
    test_indices = dataset.indices(2)
    if min(len(train_indices), len(validation_all), len(test_indices)) == 0:
        raise ValueError("Episode-disjoint split produced an empty partition")
    validation_indices = validation_all[: min(args.eval_size, len(validation_all))]
    action_stats = _load_action_stats(args.norm_stats)

    model = tokenizer.Stage1Teacher(
        num_tokens=args.num_change_tokens,
        token_dim=args.change_token_dim,
        width=args.width,
        resampler_depth=args.resampler_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        ffn_width=args.ffn_width,
        horizon=args.future_offset,
        action_dim=7,
    )
    params = model.init(jax.random.key(args.seed), jnp.zeros((1, *target_shape), dtype=jnp.float32))["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.05,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.01))
    optimizer_state = optimizer.init(params)
    start_step = 0
    best_validation = float("inf")
    stale_evaluations = 0
    if args.resume and (checkpoint := _latest_checkpoint(args.output_dir)) is not None:
        restored = serialization.from_bytes(
            {"params": params, "optimizer_state": optimizer_state}, checkpoint.read_bytes()
        )
        params = restored["params"]
        optimizer_state = restored["optimizer_state"]
        metadata = _read_json(checkpoint.with_name("metadata.json"))
        start_step = int(metadata["step"])
        best_validation = float(metadata["best_validation_huber"])
        stale_evaluations = int(metadata["stale_evaluations"])

    params = jax_utils.replicate(params, devices=devices)
    optimizer_state = jax_utils.replicate(optimizer_state, devices=devices)

    @functools.partial(jax.pmap, axis_name="devices")
    def train_step(parameters, state, displacement, actions):
        def loss_fn(value):
            _, prediction = model.apply({"params": value}, displacement)
            return jnp.mean(_huber_per_example(prediction, actions))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        gradients = jax.lax.pmean(gradients, "devices")
        loss = jax.lax.pmean(loss, "devices")
        updates, state = optimizer.update(gradients, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, state, loss, optax.global_norm(gradients)

    @jax.jit
    def evaluate(parameters, displacement, actions):
        _, prediction = model.apply({"params": parameters}, displacement)
        return _huber_per_example(prediction, actions), prediction

    def validation_metrics(parameters: Any) -> dict[str, float]:
        losses: list[np.ndarray] = []
        absolute_errors: list[np.ndarray] = []
        for start in range(0, len(validation_indices), args.batch_size):
            selected = validation_indices[start : start + args.batch_size]
            displacement, actions = dataset.batch(selected)
            actions = _normalize_actions(actions, action_stats)
            loss, prediction = evaluate(parameters, displacement, actions)
            losses.append(np.asarray(loss))
            absolute_errors.append(np.abs(np.asarray(prediction) - actions))
        return {
            "validation_huber": float(np.concatenate(losses).mean()),
            "validation_normalized_mae": float(np.concatenate(absolute_errors).mean()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "stage1_start",
                "devices": [str(device) for device in devices],
                "target_shape": target_shape,
                "change_shape": [args.num_change_tokens, args.change_token_dim],
                "train_samples": len(train_indices),
                "validation_samples": len(validation_all),
                "test_samples": len(test_indices),
                "start_step": start_step,
            }
        ),
        flush=True,
    )
    last_step = start_step
    # A resumed run may already have reached the pre-registered early-stop
    # condition and only need to finish exporting the selected teacher.  Do
    # not perform an extra 1k optimization interval in that case.
    training_steps = range(start_step, args.steps) if stale_evaluations < args.patience else range(0)
    for step in training_steps:
        sample_rng = np.random.default_rng(args.seed + step)
        selected = sample_rng.choice(train_indices, args.batch_size, replace=False)
        displacement, actions = dataset.batch(selected)
        actions = _normalize_actions(actions, action_stats)
        local_batch = args.batch_size // len(devices)
        reshape = lambda value: value.reshape(len(devices), local_batch, *value.shape[1:])
        params, optimizer_state, loss, gradient_norm = train_step(
            params, optimizer_state, reshape(displacement), reshape(actions)
        )
        last_step = step + 1
        if last_step == 1 or last_step % 100 == 0:
            record = {
                "event": "stage1_train",
                "step": last_step,
                "train_huber": float(np.asarray(loss)[0]),
                "gradient_norm": float(np.asarray(gradient_norm).mean()),
                "learning_rate": float(schedule(last_step)),
            }
            if not np.isfinite([record["train_huber"], record["gradient_norm"], record["learning_rate"]]).all():
                raise FloatingPointError(f"Non-finite training metrics: {record}")
            print(json.dumps(record), flush=True)
        if last_step % args.eval_every == 0 or last_step == args.steps:
            host_params = _tree_first(params)
            metrics = validation_metrics(host_params)
            improved = metrics["validation_huber"] < best_validation
            if improved:
                best_validation = metrics["validation_huber"]
                stale_evaluations = 0
                (args.output_dir / "best_params.msgpack").write_bytes(serialization.to_bytes(host_params))
            else:
                stale_evaluations += 1
            checkpoint_dir = args.output_dir / "checkpoints" / f"step_{last_step:06d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state = {"params": host_params, "optimizer_state": _tree_first(optimizer_state)}
            (checkpoint_dir / "state.msgpack").write_bytes(serialization.to_bytes(state))
            metadata = {
                "step": last_step,
                "best_validation_huber": best_validation,
                "stale_evaluations": stale_evaluations,
                **metrics,
            }
            (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            # The optimizer state is much larger than the deployable tokenizer.
            # Keep one resumable state plus the separately maintained best
            # params, otherwise 1k-step snapshots can exhaust the remote disk.
            for old_checkpoint in args.output_dir.glob("checkpoints/step_*"):
                if old_checkpoint != checkpoint_dir:
                    shutil.rmtree(old_checkpoint)
            print(json.dumps({"event": "stage1_eval", "improved": improved, **metadata}), flush=True)
            if stale_evaluations >= args.patience:
                print(json.dumps({"event": "early_stop", "step": last_step}), flush=True)
                break

    best_template = _tree_first(params)
    best_params_path = args.output_dir / "best_params.msgpack"
    if best_params_path.exists():
        best_params = serialization.from_bytes(best_template, best_params_path.read_bytes())
    else:
        best_params = best_template
        best_params_path.write_bytes(serialization.to_bytes(best_params))

    if not args.skip_export:

        @functools.partial(jax.pmap, axis_name="devices")
        def encode(parameters, displacement):
            return model.apply({"params": parameters}, displacement, method=model.encode)

        export_root = args.output_dir / "change_targets_raw"
        export_params = jax_utils.replicate(best_params, devices=devices)
        channel_sum = np.zeros(args.change_token_dim, dtype=np.float64)
        channel_square_sum = np.zeros(args.change_token_dim, dtype=np.float64)
        channel_count = 0
        for episode in sorted(dataset.actions):
            episode_indices = np.flatnonzero(dataset.sample_episode == episode).astype(np.int32)
            path = _target_path(export_root, episode)
            expected_shape = (len(episode_indices), args.num_change_tokens, args.change_token_dim)
            value: np.ndarray | None = None
            if path.is_file():
                try:
                    existing = np.load(path, mmap_mode="r", allow_pickle=False)
                    if existing.shape == expected_shape and existing.dtype == np.float16:
                        value = np.asarray(existing, dtype=np.float32)
                except (OSError, ValueError):
                    value = None
            if value is None:
                outputs: list[np.ndarray] = []
                for start in range(0, len(episode_indices), args.batch_size):
                    selected = episode_indices[start : start + args.batch_size]
                    displacement, _ = dataset.batch(selected)
                    real_batch = len(displacement)
                    # Keep one static per-device shape so JAX compiles the
                    # export encoder once rather than once per episode tail.
                    padded_batch = args.batch_size
                    if padded_batch != real_batch:
                        displacement = np.pad(
                            displacement,
                            ((0, padded_batch - real_batch), (0, 0), (0, 0)),
                            mode="constant",
                        )
                    local_batch = padded_batch // len(devices)
                    sharded = displacement.reshape(len(devices), local_batch, *displacement.shape[1:])
                    encoded = np.asarray(encode(export_params, sharded), dtype=np.float32)
                    outputs.append(encoded.reshape(padded_batch, *encoded.shape[2:])[:real_batch])
                value = np.concatenate(outputs)
                if value.shape != expected_shape:
                    raise ValueError(f"Episode {episode} encoded shape {value.shape}, expected {expected_shape}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = path.with_suffix(".tmp.npy")
                np.save(temporary_path, value.astype(np.float16), allow_pickle=False)
                temporary_path.replace(path)
            if episode % 10 not in (0, 1):
                channel_sum += value.sum(axis=(0, 1), dtype=np.float64)
                channel_square_sum += np.square(value, dtype=np.float64).sum(axis=(0, 1))
                channel_count += value.shape[0] * value.shape[1]
            if episode % 100 == 0:
                print(json.dumps({"event": "stage1_export", "episode": episode}), flush=True)
        mean = channel_sum / channel_count
        variance = np.maximum(channel_square_sum / channel_count - np.square(mean), 1e-12)
        stats = {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": channel_count,
            "token_shape": [args.num_change_tokens, args.change_token_dim],
            "future_offset": args.future_offset,
        }
        (args.output_dir / "change_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        manifest = {
            "format_version": 1,
            "kind": "con1_stage1_change_endpoint",
            "target_shape": [args.num_change_tokens, args.change_token_dim],
            "target_dtype": "float16",
            "future_offset": args.future_offset,
            "dataset_total_episodes": len(dataset.actions),
            "dataset_total_frames": int(sum(dataset.lengths.values())),
            "valid_sample_count": int(len(dataset.sample_episode)),
            "chunks_size": 1000,
            "normalization": "per_channel_train_split_mean_std",
            "normalization_mean": stats["mean"],
            "normalization_std": stats["std"],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {
        "event": "stage1_complete",
        "last_step": last_step,
        "best_validation_huber": best_validation,
        "best_params": str(best_params_path),
        "exported": not args.skip_export,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
