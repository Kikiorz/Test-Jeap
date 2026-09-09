import json

import numpy as np
import pytest

from openpi.training.orthogonal_targets import (
    OrthogonalTransitionDataset,
    direct_trajectory,
    episode_split_indices,
    orthogonal_trajectory,
)


def test_paper_formula_common_anchor_and_tail_mask():
    states = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
    target, valid = orthogonal_trajectory(states, 0, 4)
    np.testing.assert_allclose(target[:2], [[0, 1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]])
    np.testing.assert_array_equal(valid, [True, True, False, False])
    np.testing.assert_array_equal(target[2:], 0)
    # The second target is relative to state 0, not state 1.
    adjacent, _ = orthogonal_trajectory(states, 1, 1)
    assert not np.allclose(target[1], adjacent[0])
    end, mask = orthogonal_trajectory(states, 2, 4)
    assert not mask.any()
    assert not end.any()


def test_scale_invariance_orthogonality_and_identical_states():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(12, 24)).astype(np.float32)
    original, _ = orthogonal_trajectory(states, 0, 10)
    scaled, _ = orthogonal_trajectory(states * np.arange(1, 13)[:, None], 0, 10)
    np.testing.assert_allclose(original, scaled, atol=1e-7)
    np.testing.assert_allclose(original @ states[0], 0, atol=5e-7)
    no_change, _ = orthogonal_trajectory(np.repeat(states[:1], 12, axis=0), 0, 10)
    np.testing.assert_allclose(no_change, 0, atol=1e-7)


def test_invalid_states_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        orthogonal_trajectory(np.zeros((3, 4)), 0, 2)


def test_direct_delta_is_raw_common_anchor_with_masked_tail():
    states = np.array([[1, 2, 0], [3, 6, 4], [7, 9, 2]], dtype=np.float16)
    target, mask = direct_trajectory(states, 0, 4)
    np.testing.assert_array_equal(target, [[2, 4, 4], [6, 7, 2], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(mask, [True, True, False, False])
    scaled, _ = direct_trajectory(states * 2, 0, 4)
    np.testing.assert_array_equal(scaled, target * 2)  # Not normalized.
    shifted, _ = direct_trajectory(states + 8, 0, 4)
    np.testing.assert_array_equal(shifted, target)
    end, valid = direct_trajectory(states, 2, 4)
    assert not end.any() and not valid.any()
    assert target.dtype == np.float32


def test_direct_delta_promotes_before_subtract_and_allows_zero_states():
    states = np.array([[60000, 0], [-60000, 0]], dtype=np.float16)
    target, _ = direct_trajectory(states, 0, 1)
    np.testing.assert_array_equal(target, [[-120000., 0.]])
    zero, _ = direct_trajectory(np.zeros((3, 4)), 0, 2)
    assert not zero.any()
    with pytest.raises(ValueError, match="finite"):
        direct_trajectory(np.array([[1., np.nan], [0., 1.]]), 0, 1)


def test_episode_split_no_overlap_all_tasks_and_reproducibility():
    episodes = np.repeat(np.arange(40), 3)
    tasks = episodes // 10
    train = episode_split_indices(tasks, episodes, split="train")
    val = episode_split_indices(tasks, episodes, split="validation")
    assert len(train) + len(val) == len(tasks)
    assert not set(episodes[train]) & set(episodes[val])
    assert set(tasks[train]) == set(tasks[val]) == {0, 1, 2, 3}
    np.testing.assert_array_equal(val, episode_split_indices(tasks, episodes, split="validation"))
    bad_tasks = tasks.copy()
    bad_tasks[1] = 100
    with pytest.raises(ValueError, match="multiple tasks"):
        episode_split_indices(bad_tasks, episodes, split="validation")


def test_cache_wrapper_uses_source_frame_and_checks_contract(tmp_path):
    manifest = {"kind": "con1_independent_vjepa_frame_states", "state_dim": 3, "state_dtype": "float16",
                "dataset_total_frames": 3, "chunks_size": 1000,
                "temporal_input": "[o_k,o_k]; never [o_t,o_future]"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    folder = tmp_path / "states/chunk-000"
    folder.mkdir(parents=True)
    np.save(folder / "episode_000007.npy", np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.float16))
    (folder / "episode_000007.json").write_text("{}")
    source = [{"episode_index": 7, "frame_index": 1, "task_index": 9}]
    data = OrthogonalTransitionDataset(source, tmp_path, horizon=3, expected_dim=3, expected_num_frames=3)
    row = data[0]
    assert row["task_index"] == 9
    np.testing.assert_array_equal(row["transition_valid"], [True, False, False])
    assert row["transition_target"].shape == (3, 3)
    direct = OrthogonalTransitionDataset(source, tmp_path, horizon=3, expected_dim=3,
                                         expected_num_frames=3, target_mode="direct")[0]
    np.testing.assert_array_equal(direct["transition_target"], [[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(direct["transition_valid"], row["transition_valid"])
    with pytest.raises(ValueError, match="target mode"):
        OrthogonalTransitionDataset(source, tmp_path, horizon=3, expected_dim=3,
                                    expected_num_frames=3, target_mode="typo")
    with pytest.raises(ValueError, match="different frame counts"):
        OrthogonalTransitionDataset(source, tmp_path, horizon=3, expected_dim=3, expected_num_frames=4)
