import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import predictive_routing as pr


def test_delta_temporal_shape_and_routing_gradient():
    head, plastic = pr.init_delta_head(jax.random.key(1), input_dim=12, horizon=4, delta_dim=6, width=8)
    future = jax.random.normal(jax.random.key(2), (2, 5, 12))
    delta = pr.delta_head(head, plastic, future)
    assert delta.shape == (2, 4, 6)
    route = pr.init_routing(jax.random.key(3), action_dim=8, delta_dim=6, horizon=4)
    action = jax.random.normal(jax.random.key(4), (2, 4, 8))
    output, p = pr.route_action(route, action, delta)
    np.testing.assert_array_equal(action, output)
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
    route["gate"] = jnp.array(0.2)
    gradient = jax.grad(lambda z: pr.route_action(route, action, z)[0].sum())(delta)
    assert float(jnp.linalg.norm(gradient)) > 0
    weights = pr.supervision_weights(p, delta, gradient)
    assert float(weights.min()) >= 0.5 / 4
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
    derivative = jax.grad(lambda z: pr.supervision_weights(p, z, gradient).sum())(delta)
    np.testing.assert_array_equal(derivative, jnp.zeros_like(delta))


def test_meta_gradient_learns_through_prediction_update():
    theta = {"down": jnp.ones((3, 2)), "up": jnp.ones((2, 3))}
    transform = pr.init_gradient_adapter(jax.random.key(0), theta, rank=2)
    gradient = jax.tree.map(lambda x: x * 2, theta)
    identity = pr.transform_gradient(transform, gradient)
    for a, b in zip(jax.tree.leaves(identity), jax.tree.leaves(gradient), strict=True):
        np.testing.assert_array_equal(a, b)
    support = lambda p: sum((x ** 2).sum() for x in jax.tree.leaves(p))
    query = lambda p: sum(((x - 2) ** 2).sum() for x in jax.tree.leaves(p))
    grad = jax.grad(lambda t: pr.meta_objective(t, theta, support, query, learning_rate=0.1))(transform)
    assert float(jnp.linalg.norm(grad["down"]["u"])) > 0
    assert float(jnp.linalg.norm(grad["up"]["u"])) > 0


def test_targets_detached_and_unexecuted_masked():
    delta = jnp.ones((1, 4, 3))
    target = jnp.zeros_like(delta)
    mask = jnp.array([[1, 1, 0, 0]])
    changed = target.at[:, 2:].set(1000)
    assert pr.prediction_loss(delta, target, mask=mask) == pr.prediction_loss(delta, changed, mask=mask)
    np.testing.assert_array_equal(jax.grad(lambda t: pr.prediction_loss(delta, t))(target), jnp.zeros_like(target))

