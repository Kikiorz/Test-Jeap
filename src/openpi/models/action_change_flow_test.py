import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import action_change_flow


def _inputs():
    keys = jax.random.split(jax.random.key(1), 6)
    action_target = jax.random.normal(keys[0], (2, 10, 7))
    action_noise = jax.random.normal(keys[1], action_target.shape)
    change_target = jax.random.normal(keys[2], (2, 16, 16))
    change_noise = jax.random.normal(keys[3], change_target.shape)
    current_hidden = jax.random.normal(keys[4], (2, 64, 48))
    state = jax.random.normal(keys[5], (2, 8))
    time = jnp.asarray([1.0, 0.25])
    action_tau, _ = action_change_flow.rectified_interpolant(action_target, action_noise, time)
    change_tau, _ = action_change_flow.rectified_interpolant(change_target, change_noise, time)
    return action_tau, change_tau, current_hidden, state, time


def test_rectified_interpolant_has_correct_endpoints():
    target = jax.random.normal(jax.random.key(2), (2, 4, 3))
    noise = jax.random.normal(jax.random.key(3), target.shape)
    at_source, velocity = action_change_flow.rectified_interpolant(target, noise, jnp.ones((2,)))
    at_target, _ = action_change_flow.rectified_interpolant(target, noise, jnp.zeros((2,)))
    np.testing.assert_allclose(at_source, noise, atol=1e-6)
    np.testing.assert_allclose(at_target, target, atol=1e-6)
    np.testing.assert_allclose(velocity, noise - target, atol=1e-6)


def test_joint_flow_shapes_and_gradients():
    inputs = _inputs()
    model = action_change_flow.ActionChangeCoFlow(width=64, depth=2, num_heads=4)
    variables = model.init(jax.random.key(4), *inputs)
    action_velocity, change_velocity = model.apply(variables, *inputs)
    assert action_velocity.shape == inputs[0].shape
    assert change_velocity.shape == inputs[1].shape

    def loss(params):
        outputs = model.apply({"params": params}, *inputs)
        return sum(jnp.mean(jnp.square(output)) for output in outputs)

    leaves = jax.tree.leaves(jax.grad(loss)(variables["params"]))
    assert leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)


def test_independent_mode_has_no_cross_stream_effect():
    inputs = _inputs()
    model = action_change_flow.ActionChangeCoFlow(width=64, depth=1, num_heads=4, mode="independent")
    variables = model.init(jax.random.key(5), *inputs)
    first_action, first_change = model.apply(variables, *inputs)
    changed_action, _ = model.apply(variables, inputs[0], inputs[1] * -3.0, *inputs[2:])
    _, changed_change = model.apply(variables, inputs[0] * -3.0, inputs[1], *inputs[2:])
    np.testing.assert_allclose(first_action, changed_action, atol=1e-6)
    np.testing.assert_allclose(first_change, changed_change, atol=1e-6)


def test_joint_mode_is_bidirectionally_coupled():
    inputs = _inputs()
    model = action_change_flow.ActionChangeCoFlow(width=64, depth=1, num_heads=4, mode="joint")
    variables = model.init(jax.random.key(6), *inputs)
    first_action, first_change = model.apply(variables, *inputs)
    changed_action, _ = model.apply(variables, inputs[0], inputs[1] * -3.0, *inputs[2:])
    _, changed_change = model.apply(variables, inputs[0] * -3.0, inputs[1], *inputs[2:])
    assert not np.allclose(first_action, changed_action)
    assert not np.allclose(first_change, changed_change)
