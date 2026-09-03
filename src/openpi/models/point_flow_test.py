import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import point_flow


def _planner() -> point_flow.PointFlowPlanner:
    return point_flow.PointFlowPlanner(
        transition_width=16,
        query_grid_size=2,
        horizon=3,
        hidden_width=16,
        num_layers=1,
        num_heads=4,
        ffn_width=32,
        rngs=nnx.Rngs(0),
    )


def test_semantic_transport_recovers_feature_permutation():
    current = jnp.eye(4, dtype=jnp.float32)[None]
    # The feature initially at index 0 appears at future index 2, index 1 at
    # index 3, and vice versa.
    future = current[:, jnp.asarray([2, 3, 0, 1])]

    endpoints, confidence = point_flow.semantic_transport_flow(
        current,
        future,
        grid_size=2,
        temperature=0.01,
        spatial_sigma=10.0,
    )

    coordinates = point_flow._grid_coordinates(2)  # noqa: SLF001
    expected = coordinates[jnp.asarray([2, 3, 0, 1])]
    np.testing.assert_allclose(endpoints[0], expected, atol=1e-4)
    assert np.all(np.asarray(confidence) > 0.99)


def test_semantic_transport_preserves_static_spatial_grid():
    features = jnp.eye(4, dtype=jnp.float32)[None]
    endpoints, _ = point_flow.semantic_transport_flow(features, features, grid_size=2)
    np.testing.assert_allclose(endpoints[0], point_flow._grid_coordinates(2), atol=1e-4)  # noqa: SLF001


def test_planner_starts_as_visible_persistence_predictor():
    planner = _planner()
    transition = jax.random.normal(jax.random.key(1), (2, 4, 16))
    points = jnp.asarray(
        [
            [[0.2, 0.3], [0.7, 0.8]],
            [[0.4, 0.6], [0.9, 0.1]],
        ],
        dtype=jnp.float32,
    )

    tracks, visibility_logits = planner(transition, points)

    assert tracks.shape == (2, 2, 3, 2)
    assert visibility_logits.shape == (2, 2, 3)
    expected_tracks = jnp.broadcast_to(points[:, :, None, :], tracks.shape)
    np.testing.assert_allclose(tracks, expected_tracks, atol=2e-5)
    np.testing.assert_allclose(visibility_logits, 4.0)
    assert np.isfinite(np.asarray(tracks)).all()


def test_planner_rejects_non_spatial_query_count():
    planner = _planner()
    transition = jnp.zeros((1, 5, 16), dtype=jnp.float32)
    points = jnp.zeros((1, 2, 2), dtype=jnp.float32)

    try:
        planner(transition, points)
    except ValueError as error:
        assert "Expected 4 JEPA queries" in str(error)
    else:
        raise AssertionError("A non-square transition-query count must be rejected")


def test_conditioner_is_identity_at_initialization_and_uses_tracks_when_enabled():
    conditioner = point_flow.PointFlowConditioner(
        action_width=16,
        plan_width=16,
        num_heads=4,
        rngs=nnx.Rngs(2),
    )
    action_hidden = jax.random.normal(jax.random.key(3), (1, 4, 16))
    points = jnp.asarray([[[0.2, 0.3], [0.7, 0.8]]], dtype=jnp.float32)
    stationary = jnp.broadcast_to(points[:, :, None, :], (1, 2, 3, 2))
    moving = stationary.at[:, :, :, 0].add(jnp.asarray([0.02, 0.05, 0.08])[None, None, :])
    visibility = jnp.ones((1, 2, 3), dtype=jnp.float32)

    identity = conditioner(action_hidden, points, stationary, visibility)
    np.testing.assert_allclose(identity, action_hidden, atol=1e-6)

    conditioner.cross_attention.output.kernel.value = jnp.full_like(
        conditioner.cross_attention.output.kernel.value, 0.01
    )
    stationary_output = conditioner(action_hidden, points, stationary, visibility)
    moving_output = conditioner(action_hidden, points, moving, visibility)
    assert not np.allclose(stationary_output, moving_output)


def test_point_flow_loss_is_low_for_correct_visible_tracks():
    target_tracks = jnp.asarray(
        [[[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], [[0.8, 0.7], [0.7, 0.6], [0.6, 0.5]]]],
        dtype=jnp.float32,
    )
    visible = jnp.ones((1, 2, 3), dtype=jnp.float32)
    visibility_logits = jnp.full((1, 2, 3), 8.0, dtype=jnp.float32)

    loss, metrics = point_flow.point_flow_loss(target_tracks, visibility_logits, target_tracks, visible)

    assert float(loss[0]) < 1e-3
    np.testing.assert_allclose(metrics["track_loss"], 0.0)
    np.testing.assert_allclose(metrics["ade"], 0.0)
    np.testing.assert_allclose(metrics["fde"], 0.0)
    np.testing.assert_allclose(metrics["visibility_accuracy"], 1.0)
