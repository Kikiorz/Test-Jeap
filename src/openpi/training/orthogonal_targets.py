"""Current-anchored targets and episode splits for the user's Con1 paper."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np


def orthogonal_trajectory(states, frame: int, horizon: int):
    """Return [H,D] labels and [H] validity, using j=1..H, never j=0.

    Out-of-episode positions are zero placeholders with mask=False. They are
    not repeated endpoint supervision. Each valid future uses the SAME u_t.
    """
    if horizon < 1 or states.ndim != 2 or not 0 <= frame < len(states):
        raise ValueError("Invalid state array, current frame, or horizon")
    count = min(horizon, len(states) - frame - 1)
    value = np.asarray(states[frame:frame + count + 1], dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1e-8):
        raise ValueError("Frame states must be finite, nonzero vectors")
    unit = value / norm
    anchor, future = unit[0], unit[1:]
    target = np.zeros((horizon, states.shape[-1]), dtype=np.float32)
    target[:count] = future - np.sum(future * anchor, axis=-1, keepdims=True) * anchor
    mask = np.arange(horizon) < count
    return target, mask


def direct_trajectory(states, frame: int, horizon: int):
    """Raw z[t+j] - z[t], j=1..H: no normalization or projection.

    Cast cached states to fp32 BEFORE subtraction. Every valid horizon has
    the same current anchor; episode-tail placeholders are never supervised.
    Unlike unit-sphere targets, finite zero state vectors are valid here.
    """
    if horizon < 1 or states.ndim != 2 or not 0 <= frame < len(states):
        raise ValueError("Invalid state array, current frame, or horizon")
    count = min(horizon, len(states) - frame - 1)
    value = np.asarray(states[frame:frame + count + 1], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Frame states must be finite")
    target = np.zeros((horizon, states.shape[-1]), dtype=np.float32)
    target[:count] = value[1:] - value[0]
    return target, np.arange(horizon) < count


def episode_split_indices(task_ids, episode_ids, *, split, fraction=0.1, seed=42):
    """Deterministic task-stratified episode holdout, not a leaking frame split."""
    task_ids = np.asarray(task_ids, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if task_ids.shape != episode_ids.shape or task_ids.ndim != 1:
        raise ValueError("Task and episode columns must be equal-length vectors")
    if split not in ("all", "train", "validation") or not 0 < fraction < 1:
        raise ValueError("Invalid episode split or validation fraction")
    if split == "all":
        return np.arange(len(task_ids), dtype=np.int64)
    # Reject task-changing episodes instead of assigning one episode to both sets.
    episode_order = np.argsort(episode_ids, kind="stable")
    adjacent_same = np.diff(episode_ids[episode_order]) == 0
    if np.any(adjacent_same & (np.diff(task_ids[episode_order]) != 0)):
        raise ValueError("An episode contains multiple tasks")
    heldout = []
    for task in np.unique(task_ids):
        episodes = np.unique(episode_ids[task_ids == task])
        if len(episodes) < 2:
            raise ValueError(f"Task {task} needs at least two episodes for a holdout")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(task)]))
        count = min(len(episodes) - 1, max(1, round(len(episodes) * fraction)))
        heldout.extend(rng.permutation(episodes)[:count].tolist())
    is_validation = np.isin(episode_ids, heldout)
    return np.flatnonzero(is_validation if split == "validation" else ~is_validation)


class OrthogonalTransitionDataset:
    """Attach labels from independent frame states; retain the historical class name."""

    def __init__(self, dataset, root, *, horizon, expected_dim, expected_num_frames,
                 source_root=None, mmap_cache_size=16, target_mode="orthogonal"):
        self._dataset = dataset
        self._root = Path(root)
        self._horizon = horizon
        self._dim = expected_dim
        self._cache_size = mmap_cache_size
        if target_mode not in ("orthogonal", "direct"):
            raise ValueError(f"Unknown transition target mode: {target_mode}")
        self._target_mode = target_mode
        if horizon < 1 or expected_dim < 1 or mmap_cache_size < 1:
            raise ValueError("Invalid transition dataset shape or cache size")
        with (self._root / "manifest.json").open() as handle:
            manifest = json.load(handle)
        if manifest.get("kind") != "con1_independent_vjepa_frame_states":
            raise ValueError("Paper Con1 requires independent frame states, not old future-pair targets")
        if manifest.get("state_dim") != expected_dim or manifest.get("state_dtype") != "float16":
            raise ValueError("Frame-state dimensions or dtype disagree with model")
        if manifest.get("dataset_total_frames") != expected_num_frames:
            raise ValueError("Frame-state cache and complete source dataset have different frame counts")
        if source_root is not None:
            digest = hashlib.sha256()
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
                digest.update(name.encode())
                digest.update((Path(source_root) / "meta" / name).read_bytes())
            if manifest.get("dataset_metadata_sha256") != digest.hexdigest():
                raise ValueError("Frame-state cache and dataset metadata identities disagree")
        if manifest.get("temporal_input") != "[o_k,o_k]; never [o_t,o_future]":
            raise ValueError("Frame-state cache does not guarantee current-only teacher inputs")
        self._chunks_size = int(manifest["chunks_size"])
        if self._chunks_size < 1:
            raise ValueError("Invalid cache chunks_size")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = dict(self._dataset[int(index)])
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        if episode not in self._cache:
            path = (self._root / "states" / f"chunk-{episode // self._chunks_size:03d}"
                    / f"episode_{episode:06d}.npy")
            if not path.with_suffix(".json").is_file():
                raise FileNotFoundError(f"Teacher episode is not committed: {path}")
            states = np.load(path, mmap_mode="r", allow_pickle=False)
            if states.ndim != 2 or states.shape[-1] != self._dim or states.dtype != np.float16:
                raise ValueError(f"Invalid independent state cache: {path}")
            self._cache[episode] = states
        states = self._cache.pop(episode)
        self._cache[episode] = states
        while len(self._cache) > self._cache_size:
            _, old = self._cache.popitem(last=False)
            old._mmap.close()
        trajectory = direct_trajectory if self._target_mode == "direct" else orthogonal_trajectory
        sample["transition_target"], sample["transition_valid"] = trajectory(
            states, frame, self._horizon)
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state
