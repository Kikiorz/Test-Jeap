"""Episode-local feature cache. No future frame is exposed as a head input."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def anchored_example(r, states, frame, horizon):
    if horizon < 1 or r.ndim != 3 or states.ndim != 2 or len(r) != len(states) or not 0 <= frame < len(states):
        raise ValueError("Invalid episode arrays, frame or horizon")
    anchor = np.array(states[frame], dtype=np.float32)
    count = min(horizon, len(states) - frame - 1)
    future = np.broadcast_to(anchor, (horizon, len(anchor))).copy()
    future[:count] = np.asarray(states[frame + 1:frame + 1 + count], dtype=np.float32)
    return {"r_tokens": np.array(r[frame], dtype=np.float32), "anchor": anchor,
            "future_target": future, "valid": np.arange(horizon) < count}


def split_episodes(episodes, *, fraction=.1, seed=42):
    if not 0 < fraction < 1 or len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Invalid validation fraction or duplicate episode IDs")
    tasks = {e["task_id"] for e in episodes}
    if len(episodes) / max(len(tasks), 1) < 2:
        # Task-level holdout is impossible when a dataset has roughly one episode
        # per task: the RoboTwin release carries ~2,400 instruction strings for
        # ~2,500 episodes, so holding out "a task" would hold out everything.
        # Fall back to a seeded episode-level holdout, which is still disjoint.
        rng = np.random.default_rng(np.random.SeedSequence([seed, "episode-level"]))
        ids = np.asarray(sorted(e["id"] for e in episodes))
        heldout = set(rng.permutation(ids)[:max(1, round(len(ids) * fraction))].tolist())
        return ([e for e in episodes if e["id"] not in heldout],
                [e for e in episodes if e["id"] in heldout])
    heldout = set()
    for task in sorted({e["task_id"] for e in episodes}):
        ids = sorted(e["id"] for e in episodes if e["task_id"] == task)
        if len(ids) < 2:
            raise ValueError("Each task needs >=2 episodes for a non-leaking holdout")
        rng = np.random.default_rng(np.random.SeedSequence([seed, task]))
        heldout.update(rng.permutation(ids)[:min(len(ids) - 1, max(1, round(len(ids) * fraction)))].tolist())
    return ([e for e in episodes if e["id"] not in heldout], [e for e in episodes if e["id"] in heldout])


class FeatureDataset:
    def __init__(self, root, *, horizon, split, seed=42, fraction=.1):
        self.root = Path(root)
        raw = (self.root / "manifest.json").read_bytes()
        self.identity = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("schema") != "con1-anchored-features-v1" or not self.manifest.get("complete"):
            raise ValueError("Requires a completed anchored feature cache, not legacy targets")
        if self.manifest.get("anchor_source") != "current_only_frozen_teacher":
            raise ValueError("Anchor must come from current observation only")
        if split not in ("train", "validation"):
            raise ValueError("Use explicit train or validation split")
        groups = split_episodes(self.manifest["episodes"], fraction=fraction, seed=seed)
        self.episodes = groups[split == "validation"]
        self.horizon = horizon
        self.index = [(e, t) for e in self.episodes for t in range(e["length"] - 1)]
        if not self.index:
            raise ValueError("No supervised transitions")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, frame = self.index[index]
        if episode["id"] not in self._cache:
            r = np.load(self.root / episode["r"], mmap_mode="r", allow_pickle=False)
            z = np.load(self.root / episode["z"], mmap_mode="r", allow_pickle=False)
            if r.shape != (episode["length"], *self.manifest["r_shape"]) or z.shape != (episode["length"], self.manifest["latent_dim"]):
                raise ValueError("Cache shapes disagree with manifest")
            self._cache[episode["id"]] = (r, z)
        arrays = self._cache.pop(episode["id"])
        self._cache[episode["id"]] = arrays
        while len(self._cache) > 256:
            self._cache.popitem(last=False)
        return anchored_example(*arrays, frame, self.horizon)


def batch(dataset, indices):
    examples = [dataset[int(i)] for i in indices]
    result = {k:np.stack([e[k] for e in examples]) for k in examples[0]}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Non-finite cache input/target")
    return result
