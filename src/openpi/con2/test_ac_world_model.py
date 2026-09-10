"""CPU-only tests for the Con2 world-model wrapper.

These run without the 11.76 GB checkpoint and without a GPU: the pure mappings
are checked directly and the rollout/ranking plumbing is checked against a stub
predictor that follows the released signature.
"""

import numpy as np
import pytest

from openpi.con2 import ac_world_model as wm


def test_map_libero_state_keeps_the_gripper_channel_alive():
    """Mirrored finger joints must not cancel; the channel must actually vary."""
    opening = np.array([0.0, 0.01, 0.02, 0.039], dtype=np.float32)
    state = np.zeros((len(opening), wm.LIBERO_STATE_DIM), dtype=np.float32)
    state[:, 0:3] = np.arange(3, dtype=np.float32)
    state[:, 6] = opening
    state[:, 7] = -opening  # robosuite reports the two fingers mirrored
    mapped = wm.map_libero_state(state)
    assert mapped.shape == (len(opening), wm.STATE_DIM)
    expected = 1.0 - opening / wm.LIBERO_GRIPPER_TRAVEL
    np.testing.assert_allclose(mapped[:, 6], expected, rtol=1e-5)
    assert mapped[:, 6].std() > 0.1, "gripper channel collapsed to a constant"


def test_map_libero_state_rejects_wrong_shape():
    with pytest.raises(ValueError):
        wm.map_libero_state(np.zeros((4, 7), dtype=np.float32))


def test_map_libero_action_variants():
    action = np.ones((2, wm.ACTION_DIM), dtype=np.float32)
    np.testing.assert_allclose(wm.map_libero_action(action, "raw"), action)
    robosuite = wm.map_libero_action(action, "robosuite")
    np.testing.assert_allclose(robosuite[0, :3], 0.05)
    np.testing.assert_allclose(robosuite[0, 3:6], 0.5)
    calibrated = wm.map_libero_action(action, "calibrated")
    np.testing.assert_allclose(calibrated, np.tile(wm.LIBERO_ACTION_CALIBRATION, (2, 1)))
    # On the pose dims each variant is a further contraction: the dataset action
    # is a setpoint that the controller ramps over several control steps (the
    # gripper dim is passed through untouched, so it is excluded here).
    assert (np.abs(calibrated[:, :6]).max() < np.abs(robosuite[:, :6]).max()
            < np.abs(action[:, :6]).max())
    with pytest.raises(ValueError):
        wm.map_libero_action(action, "nope")


class StubPredictor:
    """Mimics the released signature: (context[B, T*N, D], actions[B, T, 7], states[B, T, 7])."""

    def __init__(self, dim=wm.FEATURE_DIM, tokens=wm.TOKENS_PER_FRAME):
        self.dim = dim
        self.tokens = tokens
        self.calls = []

    def __call__(self, context, actions, states):
        import torch

        self.calls.append((tuple(context.shape), tuple(actions.shape), tuple(states.shape)))
        # Predict a constant block per batch element: the first action component.
        # Deliberately independent of the context so ranking tests are exact.
        value = actions[:, -1:, :1]  # [B, 1, 1]
        return value.expand(-1, self.tokens, self.dim).contiguous()


def _context(frames=3, tokens=wm.TOKENS_PER_FRAME, dim=wm.FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(frames, tokens, dim)).astype(np.float32)


def test_predict_uses_only_the_last_action_and_state():
    stub = StubPredictor()
    model = wm.ACWorldModel(predictor=stub, device="cpu", normalize_reps=False)
    context = _context()
    actions = np.zeros((3, wm.ACTION_DIM), dtype=np.float32)
    states = np.zeros((3, wm.STATE_DIM), dtype=np.float32)
    actions[-1, 0] = 0.5
    prediction = model.predict(context, actions, states)
    assert tuple(prediction.shape) == (1, wm.TOKENS_PER_FRAME, wm.FEATURE_DIM)
    assert stub.calls == [((1, 3 * wm.TOKENS_PER_FRAME, wm.FEATURE_DIM), (1, 3, wm.ACTION_DIM), (1, 3, wm.STATE_DIM))]


def test_rollout_grows_context_and_repeats_the_last_action():
    stub = StubPredictor()
    model = wm.ACWorldModel(predictor=stub, device="cpu", normalize_reps=False)
    context = _context(frames=2)
    actions = np.ones((2, wm.ACTION_DIM), dtype=np.float32)
    states = np.ones((2, wm.STATE_DIM), dtype=np.float32)
    prediction = model.rollout(context, actions, states, steps=3)
    assert tuple(prediction.shape) == (1, wm.TOKENS_PER_FRAME, wm.FEATURE_DIM)
    assert len(stub.calls) == 3
    assert [call[1][1] for call in stub.calls] == [2, 3, 4], "context must grow one frame per step"


def test_predict_and_rollout_accept_tensors_and_sliced_views():
    """Regression: rollout feeds torch tensors back in; a numpy-only path crashes."""
    import torch

    stub = StubPredictor()
    model = wm.ACWorldModel(predictor=stub, device="cpu", normalize_reps=False)
    context = torch.from_numpy(_context(frames=2))
    actions = torch.zeros(2, wm.ACTION_DIM)
    states = torch.zeros(2, wm.STATE_DIM)
    prediction = model.predict(context, actions, states)
    assert tuple(prediction.shape) == (1, wm.TOKENS_PER_FRAME, wm.FEATURE_DIM)

    # A reversed view (negative strides) must not be handed to torch.as_tensor.
    reversed_actions = actions.flip(0).numpy()[::-1]
    assert model.predict(context, reversed_actions, states).shape[0] == 1


def test_score_actions_ranks_the_executed_action_first():
    """With a predictor that copies the true action through, ranking must be exact."""
    model = wm.ACWorldModel(predictor=StubPredictor(), device="cpu", normalize_reps=False)
    context = _context(frames=2)
    true_action = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actions = np.stack([true_action, true_action])
    states = np.zeros((2, wm.STATE_DIM), dtype=np.float32)
    target = np.full((wm.TOKENS_PER_FRAME, wm.FEATURE_DIM), 0.3, dtype=np.float32)
    other = np.array([-0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    energies, ranks = model.score_actions(context, actions, states, [true_action, other, np.zeros_like(true_action)],
                                          target, steps=1)
    assert ranks[0] == 0, f"executed action should win, got ranks {ranks} energies {energies}"
    assert energies[0] == 0.0


def test_ranking_metrics_match_their_definitions():
    assert wm.top1_rate([0, 1, 0, 2]) == 0.5
    assert wm.mean_rank_percentile([0, 1, 2], candidates=3) == pytest.approx(0.5)
    assert wm.mean_rank_percentile([0, 0, 0], candidates=3) == 0.0


def test_action_candidates_keeps_the_executed_action_and_count():
    rng = np.random.default_rng(0)
    pool = np.tile(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), (5, 1))
    true_action = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    candidates = wm.action_candidates(true_action, pool, count=8, rng=rng, scale=0.1)
    assert len(candidates) == 8
    np.testing.assert_allclose(candidates[0], true_action)
    assert not np.allclose(candidates[1], true_action), "the zero candidate must differ"
