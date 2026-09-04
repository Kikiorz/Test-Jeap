#!/usr/bin/env python3
"""Audit the frozen JEPA predictor in the openpi_jepawam Pi0.5 checkpoint.

This is a read-only, three-stage diagnostic:

1. ``sample`` extracts reproducible current/future observations from expert
   LIBERO demonstrations without modifying the dataset.
2. ``teacher`` runs the exact frozen V-JEPA 2.1 ViT-G target encoder used by
   openpi_jepawam on matched, no-change, and same-task mismatched pairs.
3. ``predict`` runs the already-trained Pi0.5 JEPA query/prediction branch and
   compares it with all three targets.

No parameters are updated.  In particular, a low matched loss is not accepted
as evidence of dynamics prediction unless it also beats the no-change target
and prefers the matched future over a same-task mismatched future.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np


DEFAULT_TASKS = (0, 10, 20, 30)
TARGET_PATCHES = 24 * 24
TARGET_DIM = 1408


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--dataset-root", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--task-indices", nargs="*", type=int, default=list(DEFAULT_TASKS))
    sample.add_argument("--episodes-per-task", type=int, default=2)
    sample.add_argument("--frames-per-episode", type=int, default=4)
    sample.add_argument("--future-offset", type=int, default=31)
    sample.add_argument("--seed", type=int, default=17)

    teacher = subparsers.add_parser("teacher")
    teacher.add_argument("--samples", type=Path, required=True)
    teacher.add_argument("--vjepa-source-root", type=Path)
    teacher.add_argument("--checkpoint", type=Path)
    teacher.add_argument(
        "--hf-port",
        type=Path,
        help="Numerically validated HF port of the same V-JEPA 2.1 target_encoder.",
    )
    teacher.add_argument("--output-dir", type=Path, required=True)
    teacher.add_argument("--device", default="cuda:0")
    teacher.add_argument("--batch-size", type=int, default=1)
    teacher.add_argument("--min-free-gib", type=float, default=16.0)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--samples", type=Path, required=True)
    predict.add_argument("--teacher-dir", type=Path, required=True)
    predict.add_argument("--policy-checkpoint", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--config", default="pi05_libero_vjepa_aux")
    predict.add_argument("--batch-size", type=int, default=4)
    predict.add_argument(
        "--prediction-file",
        type=Path,
        help="Reuse an already generated frozen predictor output instead of running Pi0.5 again.",
    )
    predict.add_argument(
        "--alignment-head",
        type=Path,
        help="Optional horizon-adapted JEPA alignment-head parameters (.npz).",
    )
    predict.add_argument("--bootstrap-replicates", type=int, default=5000)
    predict.add_argument("--seed", type=int, default=17)

    return parser.parse_args()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_all_parquets(paths: list[Path], columns: list[str] | None = None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(path, columns=columns) for path in paths]
    if not tables:
        raise FileNotFoundError("No parquet files matched")
    return pa.concat_tables(tables)


def _decode_requested_video_frames(path: Path, frame_indices: set[int]) -> dict[int, np.ndarray]:
    import av

    if not frame_indices:
        return {}
    requested = sorted(frame_indices)
    wanted = set(requested)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in wanted:
                decoded[frame_index] = frame.to_ndarray(format="rgb24")
            if frame_index >= requested[-1]:
                break
    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(f"Video {path} did not yield requested frames {missing[:8]}")
    return decoded


def _decode_requests(requests: list[tuple[Path, int]]) -> list[np.ndarray]:
    by_path: dict[Path, set[int]] = defaultdict(set)
    for path, frame_index in requests:
        by_path[path].add(frame_index)
    decoded = {
        path: _decode_requested_video_frames(path, indices)
        for path, indices in by_path.items()
    }
    return [decoded[path][frame_index] for path, frame_index in requests]


def _video_path(root: Path, key: str, chunk: int, file_index: int) -> Path:
    return root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def _build_wrong_future_indices(
    tasks: np.ndarray, episodes: np.ndarray, progress: np.ndarray
) -> np.ndarray:
    wrong = np.empty(len(tasks), dtype=np.int64)
    for index in range(len(tasks)):
        candidates = np.flatnonzero((tasks == tasks[index]) & (episodes != episodes[index]))
        if not len(candidates):
            raise ValueError(
                f"Task {tasks[index]} needs at least two selected episodes for the mismatched-future control"
            )
        wrong[index] = candidates[np.argmin(np.abs(progress[candidates] - progress[index]))]
    return wrong


def sample_expert_pairs(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    root = args.dataset_root
    with (root / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    fps = float(info["fps"])

    episode_paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_table = _read_all_parquets(episode_paths)
    episode_rows = {int(row["episode_index"]): row for row in episode_table.to_pylist()}

    data_paths = sorted((root / "data").rglob("*.parquet"))
    episode_tasks: dict[int, int] = {}
    data_cache: dict[tuple[int, int], Any] = {}
    for path in data_paths:
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        pairs = set(zip(table["episode_index"].to_pylist(), table["task_index"].to_pylist(), strict=True))
        for episode, task in pairs:
            previous = episode_tasks.setdefault(int(episode), int(task))
            if previous != int(task):
                raise RuntimeError(f"Episode {episode} has multiple task indices")

    task_table = pq.read_table(root / "meta" / "tasks.parquet")
    prompts = {
        int(task): str(prompt)
        for task, prompt in zip(
            task_table["task_index"].to_pylist(),
            task_table["__index_level_0__"].to_pylist(),
            strict=True,
        )
    }

    by_task: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_tasks.items():
        by_task[task].append(episode)
    rng = random.Random(args.seed)
    selected: list[int] = []
    for task in args.task_indices:
        candidates = sorted(by_task.get(task, []))
        if len(candidates) < args.episodes_per_task:
            raise ValueError(f"Task {task} has {len(candidates)} episodes")
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[: args.episodes_per_task]))

    records: list[dict[str, Any]] = []
    current_base_requests: list[tuple[Path, int]] = []
    future_base_requests: list[tuple[Path, int]] = []
    current_wrist_requests: list[tuple[Path, int]] = []
    states: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []

    for episode_index in selected:
        row = episode_rows[episode_index]
        task_index = episode_tasks[episode_index]
        length = int(row["length"])
        last_start = length - args.future_offset - 1
        if last_start < 0:
            raise ValueError(f"Episode {episode_index} is shorter than future offset")
        positions = np.linspace(0, last_start, args.frames_per_episode + 2, dtype=np.int64)[1:-1]

        data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if data_key not in data_cache:
            data_cache[data_key] = pq.read_table(
                root / "data" / f"chunk-{data_key[0]:03d}" / f"file-{data_key[1]:03d}.parquet",
                columns=["observation.state", "action", "episode_index", "frame_index"],
            )
        data_table = data_cache[data_key]
        episode_mask = np.asarray(data_table["episode_index"].to_numpy()) == episode_index
        episode_data = data_table.filter(episode_mask)
        frame_values = np.asarray(episode_data["frame_index"].to_numpy())
        if not np.array_equal(frame_values, np.arange(length)):
            raise RuntimeError(f"Episode {episode_index} frame indices are not contiguous")

        base_chunk = int(row["videos/observation.images.image/chunk_index"])
        base_file = int(row["videos/observation.images.image/file_index"])
        wrist_chunk = int(row["videos/observation.images.image2/chunk_index"])
        wrist_file = int(row["videos/observation.images.image2/file_index"])
        base_start = round(float(row["videos/observation.images.image/from_timestamp"]) * fps)
        wrist_start = round(float(row["videos/observation.images.image2/from_timestamp"]) * fps)
        base_path = _video_path(root, "observation.images.image", base_chunk, base_file)
        wrist_path = _video_path(root, "observation.images.image2", wrist_chunk, wrist_file)

        state_values = np.asarray(episode_data["observation.state"].to_pylist(), dtype=np.float32)
        action_values = np.asarray(episode_data["action"].to_pylist(), dtype=np.float32)
        for local_index in positions.tolist():
            future_index = local_index + args.future_offset
            current_base_requests.append((base_path, base_start + local_index))
            future_base_requests.append((base_path, base_start + future_index))
            current_wrist_requests.append((wrist_path, wrist_start + local_index))
            states.append(state_values[local_index])
            action_chunks.append(action_values[local_index:future_index])
            records.append(
                {
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "frame_index": local_index,
                    "future_frame_index": future_index,
                    "progress": local_index / max(length - 1, 1),
                    "prompt": prompts[task_index],
                }
            )

    current_base = _decode_requests(current_base_requests)
    future_base = _decode_requests(future_base_requests)
    current_wrist = _decode_requests(current_wrist_requests)

    task_values = np.asarray([record["task_index"] for record in records], dtype=np.int64)
    episode_values = np.asarray([record["episode_index"] for record in records], dtype=np.int64)
    progress_values = np.asarray([record["progress"] for record in records], dtype=np.float32)
    wrong_indices = _build_wrong_future_indices(task_values, episode_values, progress_values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        current_base=np.stack(current_base).astype(np.uint8),
        future_base=np.stack(future_base).astype(np.uint8),
        current_wrist=np.stack(current_wrist).astype(np.uint8),
        states=np.stack(states).astype(np.float32),
        action_chunks=np.stack(action_chunks).astype(np.float32),
        task_indices=task_values,
        episode_indices=episode_values,
        frame_indices=np.asarray([record["frame_index"] for record in records], dtype=np.int64),
        future_frame_indices=np.asarray([record["future_frame_index"] for record in records], dtype=np.int64),
        progress=progress_values,
        prompts=np.asarray([record["prompt"] for record in records], dtype=np.str_),
        wrong_future_indices=wrong_indices,
        future_offset=np.asarray(args.future_offset, dtype=np.int64),
    )
    manifest = {
        "dataset_root": str(root.resolve()),
        "sample_count": len(records),
        "task_indices": list(args.task_indices),
        "episodes_per_task": args.episodes_per_task,
        "frames_per_episode": args.frames_per_episode,
        "future_offset": args.future_offset,
        "seed": args.seed,
        "records": records,
    }
    _json_dump(args.output.with_suffix(".json"), manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


def _preprocess_teacher_images(images: np.ndarray) -> np.ndarray:
    from PIL import Image
    from precompute_vjepa_pair_targets import preprocess_image

    return np.stack([preprocess_image(Image.fromarray(image)) for image in images])


def compute_teacher_targets(args: argparse.Namespace) -> None:
    import torch

    samples = np.load(args.samples, allow_pickle=False)
    current = samples["current_base"]
    future = samples["future_base"]
    wrong_future = future[samples["wrong_future_indices"]]
    count = len(current)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_port is not None:
        from transformers import AutoModel

        device = torch.device(args.device)
        model = AutoModel.from_pretrained(
            args.hf_port,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.eval().to(device)
        teacher_source = {
            "kind": "hf_bit_exact_port",
            "path": str(args.hf_port.resolve()),
            "upstream_state_dict_key": "target_encoder",
        }
    else:
        if args.checkpoint is None or args.vjepa_source_root is None:
            raise ValueError("Official teacher requires --checkpoint and --vjepa-source-root")
        from precompute_vjepa_pair_targets import load_target_encoder

        model, device = load_target_encoder(args)
        teacher_source = {
            "kind": "official_meta_checkpoint",
            "checkpoint": str(args.checkpoint.resolve()),
            "vjepa_source_root": str(args.vjepa_source_root.resolve()),
        }
    kinds = {
        "matched": (current, future),
        "nochange": (current, current),
        "mismatched": (current, wrong_future),
    }
    with torch.inference_mode():
        for kind, (first_images, second_images) in kinds.items():
            output_path = args.output_dir / f"teacher_{kind}.npy"
            targets = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float16,
                shape=(count, TARGET_PATCHES, TARGET_DIM),
            )
            for start in range(0, count, args.batch_size):
                end = min(start + args.batch_size, count)
                first = _preprocess_teacher_images(first_images[start:end])
                second = _preprocess_teacher_images(second_images[start:end])
                pair = np.stack((first, second), axis=2)
                tensor = torch.from_numpy(pair).to(device=device, dtype=torch.bfloat16)
                if args.hf_port is not None:
                    output = model(pixel_values_videos=tensor, skip_predictor=True).last_hidden_state
                else:
                    output = model(tensor)
                if isinstance(output, list):
                    output = output[-1]
                expected = (end - start, TARGET_PATCHES, TARGET_DIM)
                if tuple(output.shape) != expected:
                    raise ValueError(f"Unexpected V-JEPA output {tuple(output.shape)}, expected {expected}")
                targets[start:end] = output.float().cpu().numpy().astype(np.float16)
                print(f"teacher {kind}: {end}/{count}", flush=True)
                del tensor, output
            targets.flush()
            del targets
    _json_dump(
        args.output_dir / "teacher_manifest.json",
        {
            "samples": str(args.samples.resolve()),
            "sample_count": count,
            "teacher_source": teacher_source,
            "target_shape": [count, TARGET_PATCHES, TARGET_DIM],
            "target_dtype": "float16",
            "controls": ["matched", "nochange", "same-task mismatched future"],
        },
    )


def _unit_patch(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _cosine_distance_by_sample(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = _unit_patch(first)
    second = _unit_patch(second)
    return np.mean(1.0 - np.sum(first * second, axis=-1), axis=-1)


def _flat_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first.astype(np.float32).reshape(len(first), -1)
    second = second.astype(np.float32).reshape(len(second), -1)
    numerator = np.sum(first * second, axis=-1)
    denominator = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), 1e-8)
    return numerator / denominator


def _bootstrap_mean(
    values: np.ndarray,
    replicates: int,
    seed: int,
    clusters: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    result = {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }
    if clusters is not None:
        clusters = np.asarray(clusters)
        unique_clusters = np.unique(clusters)
        cluster_means = np.asarray([values[clusters == cluster].mean() for cluster in unique_clusters])
        cluster_indices = rng.integers(
            0, len(cluster_means), size=(replicates, len(cluster_means))
        )
        cluster_bootstrap = cluster_means[cluster_indices].mean(axis=1)
        result.update(
            {
                "episode_cluster_count": int(len(unique_clusters)),
                "episode_cluster_ci95_low": float(np.percentile(cluster_bootstrap, 2.5)),
                "episode_cluster_ci95_high": float(np.percentile(cluster_bootstrap, 97.5)),
            }
        )
    return result


def _predict_jepa_latents(policy: Any, samples: Any, batch_size: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp
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
        query_output = prefix_output[:, -model.vjepa_num_queries :]
        return model.predict_vjepa_target(query_output)

    predictions: list[np.ndarray] = []
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
        predictions.append(np.asarray(forward(policy._model, observation), dtype=np.float16))
        print(f"predictor: {end}/{count}", flush=True)
    return np.concatenate(predictions, axis=0)


def _load_alignment_head(model: Any, path: Path) -> None:
    """Load the six alignment-head arrays produced by fit_horizon_alignment_head.py."""
    import jax.numpy as jnp

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
            raise ValueError(
                f"Alignment-head shape mismatch: model={variable.value.shape}, file={value.shape}"
            )
        variable.value = jnp.asarray(value, dtype=variable.value.dtype)


def _change_map(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = _unit_patch(first)
    second = _unit_patch(second)
    return (1.0 - np.sum(first * second, axis=-1)).reshape(len(first), 24, 24)


def _map_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first.reshape(len(first), -1).astype(np.float32)
    second = second.reshape(len(second), -1).astype(np.float32)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.maximum(np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1), 1e-8)
    return np.sum(first * second, axis=1) / denominator


def _make_change_figure(
    path: Path,
    samples: Any,
    true_maps: np.ndarray,
    predicted_maps: np.ndarray,
    matched_loss: np.ndarray,
    nochange_loss: np.ndarray,
    max_rows: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    rows = min(max_rows, len(matched_loss))
    order = np.argsort(matched_loss - nochange_loss)
    chosen = np.concatenate([order[: rows // 2], order[-(rows - rows // 2) :]])
    figure, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
    for row, index in enumerate(chosen.tolist()):
        axes[row, 0].imshow(samples["current_base"][index])
        axes[row, 0].set_title(f"current ep{samples['episode_indices'][index]} f{samples['frame_indices'][index]}")
        axes[row, 1].imshow(samples["future_base"][index])
        axes[row, 1].set_title(f"real future +{int(samples['future_offset'])}")
        axes[row, 2].imshow(true_maps[index], cmap="magma")
        axes[row, 2].set_title("teacher true-change map")
        axes[row, 3].imshow(predicted_maps[index], cmap="magma")
        axes[row, 3].set_title(
            f"predicted-change map\nmatched={matched_loss[index]:.4f}, nochange={nochange_loss[index]:.4f}"
        )
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def predict_and_score(args: argparse.Namespace) -> None:
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    samples = np.load(args.samples, allow_pickle=False)
    matched = np.load(args.teacher_dir / "teacher_matched.npy", mmap_mode="r")
    nochange = np.load(args.teacher_dir / "teacher_nochange.npy", mmap_mode="r")
    mismatched = np.load(args.teacher_dir / "teacher_mismatched.npy", mmap_mode="r")
    train_config = training_config.get_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prediction_file is not None:
        prediction = np.load(args.prediction_file, allow_pickle=False)
    else:
        policy = policy_config.create_trained_policy(train_config, args.policy_checkpoint)
        if args.alignment_head is not None:
            _load_alignment_head(policy._model, args.alignment_head)
        prediction = _predict_jepa_latents(policy, samples, args.batch_size)
    np.save(args.output_dir / "predicted_targets.npy", prediction)

    pred_matched = _cosine_distance_by_sample(prediction, matched)
    pred_nochange = _cosine_distance_by_sample(prediction, nochange)
    pred_mismatched = _cosine_distance_by_sample(prediction, mismatched)
    true_change = _cosine_distance_by_sample(nochange, matched)
    mismatch_change = _cosine_distance_by_sample(mismatched, matched)

    unit_prediction = _unit_patch(prediction)
    unit_matched = _unit_patch(matched)
    unit_nochange = _unit_patch(nochange)
    true_delta = unit_matched - unit_nochange
    predicted_delta = unit_prediction - unit_nochange
    residual_cosine = _flat_cosine(predicted_delta, true_delta)
    residual_norm_ratio = np.linalg.norm(predicted_delta.reshape(len(predicted_delta), -1), axis=1) / np.maximum(
        np.linalg.norm(true_delta.reshape(len(true_delta), -1), axis=1), 1e-8
    )
    true_maps = _change_map(nochange, matched)
    predicted_maps = _change_map(nochange, prediction)
    map_correlation = _map_correlation(predicted_maps, true_maps)

    matched_vs_nochange_margin = pred_nochange - pred_matched
    persistence_improvement = (true_change - pred_matched) / np.maximum(true_change, 1e-8)
    specificity_margin = pred_mismatched - pred_matched
    clusters = samples["episode_indices"]

    def summarize(values: np.ndarray, seed_offset: int) -> dict[str, float]:
        return _bootstrap_mean(
            values,
            args.bootstrap_replicates,
            args.seed + seed_offset,
            clusters=clusters,
        )

    result = {
        "protocol": {
            "model": "openpi_jepawam Pi0.5",
            "config": args.config,
            "checkpoint": str(args.policy_checkpoint.resolve()),
            "alignment_head": (
                str(args.alignment_head.resolve()) if args.alignment_head is not None else None
            ),
            "frozen": True,
            "sample_count": int(len(prediction)),
            "future_offset": int(samples["future_offset"]),
            "task_indices": sorted(np.unique(samples["task_indices"]).astype(int).tolist()),
            "important_limitation": "Source expert-data audit; not a held-out or OOD evaluation.",
        },
        "metrics": {
            "prediction_to_matched_cosine_distance": summarize(pred_matched, 0),
            "nochange_persistence_to_matched_cosine_distance": summarize(true_change, 1),
            "relative_improvement_over_nochange_persistence": summarize(persistence_improvement, 2),
            "prediction_to_nochange_target_cosine_distance": summarize(pred_nochange, 3),
            "matched_vs_nochange_margin": summarize(matched_vs_nochange_margin, 4),
            "matched_vs_nochange_preference_rate": float(np.mean(pred_matched < pred_nochange)),
            "prediction_to_mismatched_cosine_distance": summarize(pred_mismatched, 5),
            "matched_specificity_margin": summarize(specificity_margin, 6),
            "matched_preference_rate": float(np.mean(pred_matched < pred_mismatched)),
            "teacher_true_change_magnitude": summarize(true_change, 7),
            "teacher_mismatch_separation": summarize(mismatch_change, 8),
            "change_residual_cosine": summarize(residual_cosine, 9),
            "change_norm_ratio": summarize(residual_norm_ratio, 10),
            "spatial_change_map_correlation": summarize(map_correlation, 11),
        },
        "interpretation_gates": {
            "beats_nochange_persistence": bool(np.mean(persistence_improvement) > 0.0),
            "closer_to_matched_than_nochange": bool(np.mean(matched_vs_nochange_margin) > 0.0),
            "prefers_matched_future": bool(np.percentile(specificity_margin, 2.5) > 0.0),
            "positive_change_direction": bool(np.percentile(residual_cosine, 2.5) > 0.0),
            "positive_spatial_change_alignment": bool(np.percentile(map_correlation, 2.5) > 0.0),
        },
        "per_sample": [
            {
                "task_index": int(samples["task_indices"][index]),
                "episode_index": int(samples["episode_indices"][index]),
                "frame_index": int(samples["frame_indices"][index]),
                "matched_loss": float(pred_matched[index]),
                "nochange_loss": float(pred_nochange[index]),
                "mismatched_loss": float(pred_mismatched[index]),
                "relative_improvement_over_nochange_persistence": float(
                    persistence_improvement[index]
                ),
                "matched_vs_nochange_margin": float(matched_vs_nochange_margin[index]),
                "specificity_margin": float(specificity_margin[index]),
                "change_residual_cosine": float(residual_cosine[index]),
                "spatial_change_map_correlation": float(map_correlation[index]),
            }
            for index in range(len(prediction))
        ],
    }
    _json_dump(args.output_dir / "metrics.json", result)
    _make_change_figure(
        args.output_dir / "change_maps.png",
        samples,
        true_maps,
        predicted_maps,
        pred_matched,
        pred_nochange,
    )
    print(json.dumps({"metrics": result["metrics"], "gates": result["interpretation_gates"]}, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "sample":
        sample_expert_pairs(args)
    elif args.command == "teacher":
        compute_teacher_targets(args)
    elif args.command == "predict":
        predict_and_score(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
