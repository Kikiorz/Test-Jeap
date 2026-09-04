import jax
import jax.numpy as jnp

from openpi.models import transition_energy


def test_transition_action_energy_shapes_and_normalization():
    model = transition_energy.TransitionActionEnergy(
        horizon=10, action_dim=7, transition_dim=1408, width=64, num_heads=4
    )
    transition = jnp.ones((3, 64, 1408))
    action = jnp.ones((3, 10, 7))
    params = model.init(jax.random.key(0), transition, action)
    transition_embedding, action_embedding, scale = model.apply(params, transition, action)

    assert transition_embedding.shape == (3, 64)
    assert action_embedding.shape == (3, 64)
    assert scale.shape == ()
    assert jnp.allclose(jnp.linalg.norm(transition_embedding, axis=-1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(action_embedding, axis=-1), 1.0, atol=1e-5)

