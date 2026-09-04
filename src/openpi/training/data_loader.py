from collections import OrderedDict
from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import datasets as hf_datasets
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.video_utils import decode_video_frames
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class PackedLiberoDataset(Dataset):
    """Read LIBERO converted to packed LeRobot parquet/video files.

    Some exported LIBERO copies contain modern episode metadata pointing into
    ``file-xxx.parquet`` and timestamp ranges of packed videos while retaining
    a v2.0 ``info.json``.  Older LeRobot releases try to find one file per
    episode and then download the dataset again.  This adapter follows the
    authoritative per-episode file/timestamp metadata without modifying the
    dataset or changing sample semantics.
    """

    _CAMERAS = {
        "image": "observation.images.image",
        "wrist_image": "observation.images.image2",
    }

    def __init__(self, metadata: lerobot_dataset.LeRobotDatasetMetadata, action_horizon: int):
        self.root = Path(metadata.root)
        self.action_horizon = action_horizon
        episode_paths = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not episode_paths:
            raise FileNotFoundError(f"Packed episode metadata not found under {self.root}")
        table = pa.concat_tables([pq.read_table(path) for path in episode_paths])
        self.episodes = sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
        if [int(row["episode_index"]) for row in self.episodes] != list(range(len(self.episodes))):
            raise ValueError("Packed LIBERO episode indices are not contiguous")
        self.episode_from = np.asarray([row["dataset_from_index"] for row in self.episodes], dtype=np.int64)
        self.episode_to = np.asarray([row["dataset_to_index"] for row in self.episodes], dtype=np.int64)
        if not np.array_equal(self.episode_from[1:], self.episode_to[:-1]):
            raise ValueError("Packed LIBERO episode ranges are not contiguous")

        data_files = sorted((self.root / "data").rglob("file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"Packed data files not found under {self.root}")
        self.rows = hf_datasets.load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in data_files]},
            split="train",
        )
        if len(self.rows) != int(self.episode_to[-1]):
            raise ValueError(f"Packed rows={len(self.rows)}, episode metadata ends at {self.episode_to[-1]}")
        if int(self.rows[0]["index"]) != 0 or int(self.rows[-1]["index"]) != len(self.rows) - 1:
            raise ValueError("Packed parquet files are not in global sample-index order")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = int(index)
        row = self.rows[index]
        episode = int(np.searchsorted(self.episode_to, index, side="right"))
        record = self.episodes[episode]
        if int(row["episode_index"]) != episode:
            raise ValueError(f"Row {index} belongs to episode {row['episode_index']}, expected {episode}")
        end = int(self.episode_to[episode])
        action_indices = np.minimum(np.arange(index, index + self.action_horizon), end - 1)
        action_column = "actions" if "actions" in self.rows.column_names else "action"
        actions = np.asarray(self.rows.select(action_indices.tolist())[action_column], dtype=np.float32)

        timestamp = float(row["timestamp"])
        output = {
            "state": np.asarray(row["observation.state"], dtype=np.float32),
            "actions": actions,
            "timestamp": np.float32(timestamp),
            "frame_index": np.int64(row["frame_index"]),
            "episode_index": np.int64(episode),
            "index": np.int64(index),
            "task_index": np.int64(row["task_index"]),
        }
        for output_key, video_key in self._CAMERAS.items():
            metadata_key = f"videos/{video_key}"
            chunk = int(record[f"{metadata_key}/chunk_index"])
            file_index = int(record[f"{metadata_key}/file_index"])
            start_time = float(record[f"{metadata_key}/from_timestamp"])
            video_path = self.root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            frame = decode_video_frames(
                video_path,
                [start_time + timestamp],
                tolerance_s=1e-4,
                backend="pyav",
            )
            output[output_key] = frame.squeeze(0)
        return output


def _uses_packed_episode_storage(root: str | Path) -> bool:
    root = Path(root)
    return (root / "meta" / "episodes").is_dir() and any((root / "data").rglob("file-*.parquet"))


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class VJepaTargetDataset(Dataset):
    """Adds one precomputed V-JEPA target row to each LeRobot sample."""

    def __init__(
        self,
        dataset: Dataset,
        target_root: str | Path,
        *,
        expected_shape: tuple[int, int],
        expected_future_offset: int | None = None,
        expected_image_key: str | None = None,
        expected_num_frames: int | None = None,
        mmap_cache_size: int = 16,
    ):
        if mmap_cache_size < 1:
            raise ValueError("V-JEPA mmap cache size must be positive")
        self._dataset = dataset
        self._target_root = Path(target_root)
        self._expected_shape = expected_shape
        self._mmap_cache_size = mmap_cache_size
        self._cache: OrderedDict[int, np.memmap] = OrderedDict()

        manifest_path = self._target_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"V-JEPA target manifest not found: {manifest_path}")
        with manifest_path.open() as f:
            manifest = json.load(f)
        if tuple(manifest.get("target_shape", ())) != expected_shape:
            raise ValueError(
                f"V-JEPA target shape mismatch: expected {expected_shape}, manifest has {manifest.get('target_shape')}"
            )
        if manifest.get("target_dtype") != "float16":
            raise ValueError(f"Expected float16 V-JEPA targets, got {manifest.get('target_dtype')}")
        if expected_future_offset is not None and manifest.get("future_offset") != expected_future_offset:
            raise ValueError(
                "V-JEPA future offset mismatch: "
                f"expected {expected_future_offset}, manifest has {manifest.get('future_offset')}"
            )
        if expected_image_key is not None and manifest.get("image_key") != expected_image_key:
            raise ValueError(
                f"V-JEPA image key mismatch: expected {expected_image_key!r}, manifest has {manifest.get('image_key')!r}"
            )
        if expected_num_frames is not None and manifest.get("dataset_total_frames") != expected_num_frames:
            raise ValueError(
                "V-JEPA frame count mismatch: "
                f"expected {expected_num_frames}, manifest has {manifest.get('dataset_total_frames')}"
            )
        self._chunks_size = int(manifest.get("chunks_size", 1000))
        if self._chunks_size < 1:
            raise ValueError("V-JEPA target manifest chunks_size must be positive")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = dict(self._dataset[index])
        episode_index = int(np.asarray(sample["episode_index"]).item())
        frame_index = int(np.asarray(sample["frame_index"]).item())
        target = self._get_episode(episode_index)
        if not 0 <= frame_index < target.shape[0]:
            raise IndexError(
                f"Frame {frame_index} is outside V-JEPA target episode {episode_index} with {target.shape[0]} rows"
            )
        # The copy decouples the row from an mmap that may be evicted before collation.
        sample["vjepa_target"] = np.array(target[frame_index], copy=True)
        return sample

    def _get_episode(self, episode_index: int) -> np.memmap:
        if episode_index in self._cache:
            target = self._cache.pop(episode_index)
            self._cache[episode_index] = target
            return target

        path = (
            self._target_root
            / "targets"
            / f"chunk-{episode_index // self._chunks_size:03d}"
            / f"episode_{episode_index:06d}.npy"
        )
        target = np.load(path, mmap_mode="r", allow_pickle=False)
        if target.ndim != 3 or tuple(target.shape[1:]) != self._expected_shape or target.dtype != np.float16:
            self._close_mmap(target)
            raise ValueError(f"Invalid V-JEPA target array: {path}, shape={target.shape}, dtype={target.dtype}")
        self._cache[episode_index] = target
        while len(self._cache) > self._mmap_cache_size:
            _, evicted = self._cache.popitem(last=False)
            self._close_mmap(evicted)
        return target

    @staticmethod
    def _close_mmap(value: np.ndarray) -> None:
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class ChangeTargetDataset(Dataset):
    """Filters invalid episode tails and attaches normalized Stage-1 change endpoints."""

    def __init__(
        self,
        dataset: Dataset,
        target_root: str | Path,
        *,
        expected_shape: tuple[int, int],
        expected_future_offset: int,
        expected_num_frames: int | None = None,
        mmap_cache_size: int = 16,
    ):
        if mmap_cache_size < 1:
            raise ValueError("Change-target mmap cache size must be positive")
        self._dataset = dataset
        self._target_root = Path(target_root)
        self._expected_shape = expected_shape
        self._mmap_cache_size = mmap_cache_size
        self._cache: OrderedDict[int, np.memmap] = OrderedDict()
        manifest_path = self._target_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Change-target manifest not found: {manifest_path}")
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if tuple(manifest.get("target_shape", ())) != expected_shape:
            raise ValueError(
                f"Change-target shape mismatch: expected {expected_shape}, got {manifest.get('target_shape')}"
            )
        if manifest.get("target_dtype") != "float16":
            raise ValueError(f"Expected float16 change targets, got {manifest.get('target_dtype')}")
        if int(manifest.get("future_offset", -1)) != expected_future_offset:
            raise ValueError(
                f"Change-target horizon mismatch: expected {expected_future_offset}, "
                f"got {manifest.get('future_offset')}"
            )
        if expected_num_frames is not None and int(manifest.get("dataset_total_frames", -1)) != expected_num_frames:
            raise ValueError(
                f"Change-target frame count mismatch: expected {expected_num_frames}, "
                f"got {manifest.get('dataset_total_frames')}"
            )
        self._chunks_size = int(manifest.get("chunks_size", 1000))
        self._mean = np.asarray(manifest["normalization_mean"], dtype=np.float32)
        self._std = np.asarray(manifest["normalization_std"], dtype=np.float32)
        if self._mean.shape != (expected_shape[-1],) or self._std.shape != self._mean.shape:
            raise ValueError(f"Invalid change normalization shapes: {self._mean.shape}, {self._std.shape}")

        valid_indices: list[np.ndarray] = []
        raw_cursor = 0
        episode_count = int(manifest["dataset_total_episodes"])
        for episode in range(episode_count):
            target = np.load(self._path(episode), mmap_mode="r", allow_pickle=False)
            if target.ndim != 3 or tuple(target.shape[1:]) != expected_shape or target.dtype != np.float16:
                VJepaTargetDataset._close_mmap(target)
                raise ValueError(f"Invalid change-target episode {episode}: shape={target.shape}, dtype={target.dtype}")
            valid_length = target.shape[0]
            VJepaTargetDataset._close_mmap(target)
            valid_indices.append(np.arange(raw_cursor, raw_cursor + valid_length, dtype=np.int64))
            raw_cursor += valid_length + expected_future_offset
        if raw_cursor != len(dataset):
            raise ValueError(f"Change-target episode lengths imply {raw_cursor} rows, dataset has {len(dataset)}")
        self._valid_indices = np.concatenate(valid_indices)
        if len(self._valid_indices) != int(manifest["valid_sample_count"]):
            raise ValueError("Change-target valid sample count disagrees with manifest")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = dict(self._dataset[int(self._valid_indices[int(index)])])
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        target = self._get_episode(episode)
        if not 0 <= frame < target.shape[0]:
            raise IndexError(f"Frame {frame} is outside valid change targets for episode {episode}")
        value = np.asarray(target[frame], dtype=np.float32)
        sample["change_target"] = ((value - self._mean) / (self._std + 1e-6)).astype(np.float32)
        return sample

    def _path(self, episode: int) -> Path:
        return (
            self._target_root
            / "targets"
            / f"chunk-{episode // self._chunks_size:03d}"
            / f"episode_{episode:06d}.npy"
        )

    def _get_episode(self, episode: int) -> np.memmap:
        if episode in self._cache:
            target = self._cache.pop(episode)
            self._cache[episode] = target
            return target
        target = np.load(self._path(episode), mmap_mode="r", allow_pickle=False)
        self._cache[episode] = target
        while len(self._cache) > self._mmap_cache_size:
            _, evicted = self._cache.popitem(last=False)
            VJepaTargetDataset._close_mmap(evicted)
        return target

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class PointFlowTargetDataset(Dataset):
    """Adds cached self-supervised semantic transport to every LeRobot frame.

    Each episode file stores ``[frames, points, horizon, 3]`` float16 rows;
    the last dimension is normalized ``x, y, confidence``.  Initial query
    coordinates are shared in the manifest so the dataset does not duplicate
    them for every frame.
    """

    def __init__(
        self,
        dataset: Dataset,
        target_root: str | Path,
        *,
        expected_num_points: int,
        expected_horizon: int,
        expected_image_key: str | None = None,
        expected_num_frames: int | None = None,
        mmap_cache_size: int = 16,
    ):
        if mmap_cache_size < 1:
            raise ValueError("Point-flow mmap cache size must be positive")
        self._dataset = dataset
        self._target_root = Path(target_root)
        self._expected_shape = (expected_num_points, expected_horizon, 3)
        self._mmap_cache_size = mmap_cache_size
        self._cache: OrderedDict[int, np.memmap] = OrderedDict()

        manifest_path = self._target_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Point-flow target manifest not found: {manifest_path}")
        with manifest_path.open() as file:
            manifest = json.load(file)
        if tuple(manifest.get("target_shape", ())) != self._expected_shape:
            raise ValueError(
                f"Point-flow target shape mismatch: expected {self._expected_shape}, "
                f"manifest has {manifest.get('target_shape')}"
            )
        if manifest.get("target_dtype") != "float16":
            raise ValueError(f"Expected float16 point-flow targets, got {manifest.get('target_dtype')}")
        if expected_image_key is not None and manifest.get("image_key") != expected_image_key:
            raise ValueError(
                f"Point-flow image key mismatch: expected {expected_image_key!r}, "
                f"manifest has {manifest.get('image_key')!r}"
            )
        if expected_num_frames is not None and manifest.get("dataset_total_frames") != expected_num_frames:
            raise ValueError(
                "Point-flow frame count mismatch: "
                f"expected {expected_num_frames}, manifest has {manifest.get('dataset_total_frames')}"
            )
        query_points = np.asarray(manifest.get("query_points"), dtype=np.float32)
        if query_points.shape != (expected_num_points, 2):
            raise ValueError(
                f"Point-flow query shape mismatch: expected {(expected_num_points, 2)}, got {query_points.shape}"
            )
        if not np.isfinite(query_points).all() or np.any((query_points < 0) | (query_points > 1)):
            raise ValueError("Point-flow query coordinates must be finite and normalized to [0, 1]")
        self._query_points = query_points
        self._chunks_size = int(manifest.get("chunks_size", 1000))
        if self._chunks_size < 1:
            raise ValueError("Point-flow target manifest chunks_size must be positive")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = dict(self._dataset[index])
        episode_index = int(np.asarray(sample["episode_index"]).item())
        frame_index = int(np.asarray(sample["frame_index"]).item())
        episode = self._get_episode(episode_index)
        if not 0 <= frame_index < episode.shape[0]:
            raise IndexError(
                f"Frame {frame_index} is outside point-flow episode {episode_index} with {episode.shape[0]} rows"
            )
        row = np.asarray(episode[frame_index], dtype=np.float32)
        sample["point_flow_queries"] = self._query_points.copy()
        sample["point_flow_target"] = row[..., :2].copy()
        sample["point_flow_visibility"] = np.clip(row[..., 2], 0.0, 1.0).copy()
        return sample

    def _get_episode(self, episode_index: int) -> np.memmap:
        if episode_index in self._cache:
            target = self._cache.pop(episode_index)
            self._cache[episode_index] = target
            return target
        path = (
            self._target_root
            / "targets"
            / f"chunk-{episode_index // self._chunks_size:03d}"
            / f"episode_{episode_index:06d}.npy"
        )
        target = np.load(path, mmap_mode="r", allow_pickle=False)
        if target.ndim != 4 or tuple(target.shape[1:]) != self._expected_shape or target.dtype != np.float16:
            VJepaTargetDataset._close_mmap(target)
            raise ValueError(f"Invalid point-flow target array: {path}, shape={target.shape}, dtype={target.dtype}")
        self._cache[episode_index] = target
        while len(self._cache) > self._mmap_cache_size:
            _, evicted = self._cache.popitem(last=False)
            VJepaTargetDataset._close_mmap(evicted)
        return target

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if jnp.issubdtype(spec.dtype, jnp.floating):
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0).astype(spec.dtype)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    if _uses_packed_episode_storage(dataset_meta.root):
        logging.info("Using packed LIBERO adapter for %s at %s", repo_id, dataset_meta.root)
        dataset = PackedLiberoDataset(dataset_meta, action_horizon)
    else:
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )

    use_vjepa_aux = bool(getattr(model_config, "use_vjepa_aux", False))
    use_action_change = bool(getattr(model_config, "use_action_change_mmdit", False))
    if use_vjepa_aux and not use_action_change and data_config.vjepa_target_root is None:
        raise ValueError("V-JEPA auxiliary training requires data.vjepa_target_root")
    if data_config.vjepa_target_root is not None:
        if not use_vjepa_aux:
            raise ValueError("V-JEPA targets were configured for a model with use_vjepa_aux=False")
        if data_config.vjepa_future_offset is None or data_config.vjepa_image_key is None:
            raise ValueError("V-JEPA targets require an expected future offset and image key")
        dataset = VJepaTargetDataset(
            dataset,
            data_config.vjepa_target_root,
            expected_shape=(model_config.vjepa_target_grid_size**2, model_config.vjepa_target_dim),
            expected_future_offset=data_config.vjepa_future_offset,
            expected_image_key=data_config.vjepa_image_key,
            expected_num_frames=len(dataset),
            mmap_cache_size=data_config.vjepa_mmap_cache_size,
        )

    if use_action_change and data_config.change_target_root is None:
        raise ValueError("Action–Change MMDiT training requires data.change_target_root")
    if data_config.change_target_root is not None:
        if not use_action_change:
            raise ValueError("Change targets were configured for a model with use_action_change_mmdit=False")
        if data_config.change_future_offset is None:
            raise ValueError("Change targets require an expected future offset")
        dataset = ChangeTargetDataset(
            dataset,
            data_config.change_target_root,
            expected_shape=(model_config.change_num_tokens, model_config.change_token_dim),
            expected_future_offset=data_config.change_future_offset,
            expected_num_frames=len(dataset),
            mmap_cache_size=data_config.change_mmap_cache_size,
        )

    use_point_flow = bool(getattr(model_config, "use_point_flow", False))
    if use_point_flow and data_config.point_flow_target_root is None:
        raise ValueError("Point-flow training requires data.point_flow_target_root")
    if data_config.point_flow_target_root is not None:
        if not use_point_flow:
            raise ValueError("Point-flow targets were configured for a model with use_point_flow=False")
        if data_config.point_flow_horizon is None or data_config.point_flow_image_key is None:
            raise ValueError("Point-flow targets require an expected horizon and image key")
        dataset = PointFlowTargetDataset(
            dataset,
            data_config.point_flow_target_root,
            expected_num_points=model_config.point_flow_num_points,
            expected_horizon=data_config.point_flow_horizon,
            expected_image_key=data_config.point_flow_image_key,
            expected_num_frames=len(dataset),
            mmap_cache_size=data_config.point_flow_mmap_cache_size,
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
