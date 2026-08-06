import dataclasses
import json

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _ListDataset:
    def __init__(self, items):
        self._items = items

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


def test_vjepa_target_dataset(tmp_path):
    target_root = tmp_path / "targets_root"
    episode_dir = target_root / "targets" / "chunk-000"
    episode_dir.mkdir(parents=True)
    (target_root / "manifest.json").write_text(
        json.dumps(
            {
                "target_shape": [6, 4],
                "target_dtype": "float16",
                "future_offset": 50,
                "image_key": "observation.images.cam_high",
            }
        )
    )
    episode = np.arange(3 * 6 * 4, dtype=np.float16).reshape(3, 6, 4)
    np.save(episode_dir / "episode_000007.npy", episode, allow_pickle=False)
    dataset = _data_loader.VJepaTargetDataset(
        _ListDataset(
            [
                {"episode_index": np.int64(7), "frame_index": np.int64(0)},
                {"episode_index": np.int64(7), "frame_index": np.int64(2)},
            ]
        ),
        target_root,
        expected_shape=(6, 4),
        expected_future_offset=50,
        expected_image_key="observation.images.cam_high",
        mmap_cache_size=1,
    )

    np.testing.assert_array_equal(dataset[0]["vjepa_target"], episode[0])
    np.testing.assert_array_equal(dataset[1]["vjepa_target"], episode[2])
    assert dataset[0]["vjepa_target"].dtype == np.float16

    with pytest.raises(ValueError, match="future offset mismatch"):
        _data_loader.VJepaTargetDataset(
            _ListDataset([]),
            target_root,
            expected_shape=(6, 4),
            expected_future_offset=31,
        )


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
