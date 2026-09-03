import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import coflow


def _inputs():
    key = jax.random.key(7)
    action_key, prior_key, observed_key = jax.random.split(key, 3)
    action = jax.random.normal(action_key, (2, 10, 7))
    prior = jax.random.normal(prior_key, (2, 64, 32))
    observed = jax.random.normal(observed_key, (2, 64, 32))
    time = jnp.asarray([1.0, 0.25])
    transition, velocity = coflow.transition_interpolant(prior, observed, time)
    return action, prior, observed, transition, velocity, time


def test_transition_interpolant_endpoints_and_norms():
    _, prior, observed, _, _, _ = _inputs()
    source, velocity = coflow.transition_interpolant(prior, observed, jnp.ones((2,)))
    target, _ = coflow.transition_interpolant(prior, observed, jnp.zeros((2,)))
    np.testing.assert_allclose(source, coflow.normalize_transition(prior), atol=1e-6)
    np.testing.assert_allclose(target, coflow.normalize_transition(observed), atol=1e-6)
    np.testing.assert_allclose(source - target, velocity, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(source, axis=-1), 1.0, atol=1e-5)


def test_coflow_core_shapes_are_joint():
    action, prior, _, transition, _, time = _inputs()
    model = coflow.CoFlowCore(action_dim=7, transition_dim=32, width=64, depth=2, num_heads=4)
    variables = model.init(jax.random.key(3), action, transition, prior, time)
    action_velocity, transition_velocity = model.apply(variables, action, transition, prior, time)
    assert action_velocity.shape == action.shape
    assert transition_velocity.shape == transition.shape
    assert np.isfinite(np.asarray(action_velocity)).all()
    assert np.isfinite(np.asarray(transition_velocity)).all()


def test_transition_velocity_head_starts_as_identity_transport():
    action, prior, _, transition, _, time = _inputs()
    model = coflow.CoFlowCore(action_dim=7, transition_dim=32, width=64, depth=2, num_heads=4)
    variables = model.init(jax.random.key(33), action, transition, prior, time)
    _, transition_velocity = model.apply(variables, action, transition, prior, time)
    np.testing.assert_allclose(transition_velocity, 0.0, atol=0.0)


def test_pure_noise_action_cannot_change_transition_stream():
    _, prior, _, transition, _, _ = _inputs()
    action = jax.random.normal(jax.random.key(41), (2, 10, 32))
    time = jnp.ones((2,))
    block = coflow.CoFlowBlock(width=32, num_heads=4, mode="coflow")
    transition_hidden = transition
    prior_hidden = prior
    variables = block.init(jax.random.key(4), action, transition_hidden, prior_hidden, time)
    _, first = block.apply(variables, action, transition_hidden, prior_hidden, time)
    _, second = block.apply(variables, action * -3.0, transition_hidden, prior_hidden, time)
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_formed_action_changes_transition_but_has_stopped_gradient():
    _, prior, _, transition, _, _ = _inputs()
    action = jax.random.normal(jax.random.key(51), (2, 10, 32))
    time = jnp.full((2,), 0.25)
    block = coflow.CoFlowBlock(width=32, num_heads=4, mode="coflow")
    variables = block.init(jax.random.key(5), action, transition, prior, time)
    _, first = block.apply(variables, action, transition, prior, time)
    _, second = block.apply(variables, action * -3.0, transition, prior, time)
    assert not np.allclose(first, second)

    def transition_sum(action_value):
        return block.apply(variables, action_value, transition, prior, time)[1].sum()

    gradient = jax.grad(transition_sum)(action)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-7)


def test_fixed_condition_does_not_leak_transition_interpolant_to_action():
    _, prior, _, transition, _, time = _inputs()
    action = jax.random.normal(jax.random.key(61), (2, 10, 32))
    block = coflow.CoFlowBlock(width=32, num_heads=4, mode="fixed")
    variables = block.init(jax.random.key(6), action, transition, prior, time)
    first, _ = block.apply(variables, action, transition, prior, time)
    second, _ = block.apply(variables, action, transition * -4.0, prior, time)
    np.testing.assert_allclose(first, second, atol=1e-6)
