import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import action_change_bottleneck as bottleneck


def test_latent_displacement_is_zero_for_identical_pairs():
    value = jax.random.normal(jax.random.key(1), (2, 12, 32))
    displacement = bottleneck.latent_displacement(value, value)
    np.testing.assert_allclose(displacement, 0.0, atol=0.0)


def test_displacement_norm_matches_cosine_distance():
    first = jax.random.normal(jax.random.key(2), (2, 12, 32))
    second = jax.random.normal(jax.random.key(3), (2, 12, 32))
    first_norm = bottleneck.l2_normalize_tokens(first)
    second_norm = bottleneck.l2_normalize_tokens(second)
    displacement = bottleneck.latent_displacement(first, second)
    squared_norm = jnp.sum(jnp.square(displacement), axis=-1)
    expected = 2.0 * (1.0 - jnp.sum(first_norm * second_norm, axis=-1))
    np.testing.assert_allclose(squared_norm, expected, atol=2e-6)


def test_phase_a_shapes_and_gradients():
    displacement = jax.random.normal(jax.random.key(4), (2, 24, 64))
    target = jax.random.normal(jax.random.key(5), (2, 10, 7))
    model = bottleneck.PhaseAModel(width=32, depth=2, num_heads=4)
    variables = model.init(jax.random.key(6), displacement)
    change, action = model.apply(variables, displacement)
    assert change.shape == (2, 16, 16)
    assert action.shape == target.shape

    def loss(params):
        _, prediction = model.apply({"params": params}, displacement)
        return jnp.mean(jnp.square(prediction - target))

    gradients = jax.grad(loss)(variables["params"])
    leaves = jax.tree.leaves(gradients)
    assert leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)
