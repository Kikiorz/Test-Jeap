import jax
import jax.numpy as jnp
import pytest

from openpi.models import action_consequence


def test_action_conditioned_consequence_shapes_and_gradients():
    model = action_consequence.ActionConditionedConsequence(
        horizon=10, action_dim=7, transition_dim=32, width=16, num_heads=4
    )
    current = jnp.ones((3, 64, 32))
    action = jnp.ones((3, 10, 7))
    target = jnp.ones((3, 64, 32))
    params = model.init(jax.random.key(0), current, action, target)["params"]

    consequence, encoded_target, scale = model.apply(
        {"params": params}, current, action, target
    )
    assert consequence.shape == encoded_target.shape == (3, 16)
    assert scale.shape == ()
    assert jnp.allclose(jnp.linalg.norm(consequence, axis=-1), 1.0)

    gradients = jax.grad(
        lambda value: -model.apply({"params": value}, current, action, target)[0].sum()
    )(params)
    leaves = jax.tree.leaves(gradients)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in leaves)


def test_action_conditioned_consequence_rejects_bad_shapes():
    model = action_consequence.ActionConditionedConsequence(
        horizon=10, action_dim=7, transition_dim=32, width=16, num_heads=4
    )
    with pytest.raises(ValueError, match="current"):
        model.init(
            jax.random.key(0),
            jnp.ones((2, 63, 32)),
            jnp.ones((2, 10, 7)),
            jnp.ones((2, 63, 32)),
        )
