import jax
import jax.numpy as jnp
import numpy as np

from openpi.con1.modules import ActionDeltaCrossAttention, AnchoredDeltaHead, anchored_loss


def test_no_q0_and_zero_residual():
    model = ActionDeltaCrossAttention(12)
    variables = model.init(jax.random.key(0), jnp.ones((2, 4, 12)), jnp.ones((2, 4, 8)))
    out = model.apply(variables, jnp.ones((2, 4, 12)), jnp.ones((2, 4, 8)))
    np.testing.assert_array_equal(out["hidden"], 1.)
    assert "query_tokens" not in str(variables)
    assert out["attention"].shape == (2, 4, 4)


def test_anchor_delta_shapes_and_loss():
    model = AnchoredDeltaHead(horizon=3, latent_dim=6, width=8)
    r = jnp.ones((2, 5, 7)); z = jnp.ones((2, 6))
    variables = model.init(jax.random.key(1), r, z)
    out = model.apply(variables, r, z)
    assert out["delta"].shape == (2, 3, 6)
    valid = jnp.array([[True, True, False], [True, False, False]])
    target = jnp.concatenate([z[:, None], z[:, None], z[:, None]], axis=1)
    loss, metrics = anchored_loss(out["delta"], z, target, valid)
    assert bool(jnp.isfinite(loss)); assert int(metrics["valid_count"]) == 3


def test_two_terms_are_not_claimed_independent():
    z = jnp.zeros((1, 2)); target = jnp.ones((1, 1, 2)); d = jnp.zeros_like(target)
    valid = jnp.ones((1, 1), dtype=bool)
    one, _ = anchored_loss(d, z, target, valid, delta_weight=0.)
    two, _ = anchored_loss(d, z, target, valid, delta_weight=1.)
    np.testing.assert_allclose(two, 2 * one)
