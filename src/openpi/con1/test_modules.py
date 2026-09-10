import jax
import jax.numpy as jnp
import numpy as np

from openpi.con1.modules import ActionDeltaCrossAttention, AnchoredDeltaHead, anchored_loss
from openpi.con1.modules import control_weighted_delta_loss


def test_no_q0_and_zero_residual():
    model = ActionDeltaCrossAttention(12)
    variables = model.init(jax.random.key(0), jnp.ones((2, 4, 12)), jnp.ones((2, 4, 8)))
    out = model.apply(variables, jnp.ones((2, 4, 12)), jnp.ones((2, 4, 8)))
    np.testing.assert_array_equal(out["hidden"], 1.)
    assert "query_tokens" not in str(variables)
    assert out["attention"].shape == (2, 4, 4)


def test_action_adapter_is_a_no_op_at_init_but_gets_gradient():
    plain = ActionDeltaCrossAttention(12, width=8)
    adapted = ActionDeltaCrossAttention(12, width=8, use_action_adapter=True)
    h = jax.random.normal(jax.random.key(41), (2, 4, 12))
    d = jax.random.normal(jax.random.key(42), (2, 4, 8))
    plain_vars = plain.init(jax.random.key(43), h, d)
    adapted_vars = adapted.init(jax.random.key(43), h, d)
    np.testing.assert_allclose(
        np.asarray(adapted.apply(adapted_vars, h, d)["hidden"]),
        np.asarray(plain.apply(plain_vars, h, d)["hidden"]), rtol=0, atol=1e-6)

    def loss(vars):
        return jnp.mean(jnp.square(adapted.apply(vars, h, d)["hidden"]))

    grads = jax.grad(loss)(adapted_vars)
    import flax.traverse_util as traverse_util
    flat = traverse_util.flatten_dict(grads, sep="/")
    assert float(jnp.abs(flat["params/adapter_out/kernel"]).max()) > 0.0


def test_residual_budget_caps_the_forward_perturbation():
    """With a huge adapter the correction must still respect the relative cap."""
    model = ActionDeltaCrossAttention(12, width=8, use_action_adapter=True,
                                      adapter_scale=1e3, residual_budget=0.05)
    h = jax.random.normal(jax.random.key(51), (2, 4, 12))
    d = jax.random.normal(jax.random.key(52), (2, 4, 8))
    variables = model.init(jax.random.key(53), h, d)
    out = model.apply(variables, h, d)
    corr = np.asarray(out["correction"], np.float64)
    hid = np.asarray(h, np.float64)
    corr_rms = np.sqrt((corr**2).mean(-1))
    base_rms = np.sqrt((hid**2).mean(-1))
    assert np.all(corr_rms <= 0.05 * base_rms + 1e-6), (corr_rms, base_rms)
    # Unbounded, the same setup must violate the cap, proving the test bites.
    loose = ActionDeltaCrossAttention(12, width=8, use_action_adapter=True,
                                      adapter_scale=1e3, residual_budget=0.0)
    loose_out = loose.apply(loose.init(jax.random.key(53), h, d), h, d)
    loose_rms = np.sqrt((np.asarray(loose_out["correction"], np.float64) ** 2).mean(-1))
    assert np.any(loose_rms > 0.05 * base_rms + 1e-6)


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


def test_action_conditioning_shapes_and_strict_causality():
    horizon, latent_dim, width, action_dim = 4, 6, 8, 3
    model = AnchoredDeltaHead(horizon=horizon, latent_dim=latent_dim, width=width,
                              action_dim=action_dim, use_action_conditioning=True)
    r = jax.random.normal(jax.random.key(3), (2, 5, 7))
    z = jax.random.normal(jax.random.key(4), (2, latent_dim))
    actions = jax.random.normal(jax.random.key(5), (2, horizon, action_dim))
    variables = model.init(jax.random.key(6), r, z, actions)
    # The value projection is zero-initialised by design, so activate it before
    # testing causality; otherwise every horizon would be trivially unchanged.
    import flax.traverse_util as traverse_util
    flat = dict(traverse_util.flatten_dict(variables, sep="/"))
    flat["params/action_value/kernel"] = jax.random.normal(
        jax.random.key(9), flat["params/action_value/kernel"].shape)
    variables = traverse_util.unflatten_dict(flat, sep="/")
    out = model.apply(variables, r, z, actions)
    assert out["delta"].shape == (2, horizon, latent_dim)
    assert np.all(np.isfinite(np.asarray(out["delta"])))
    # Horizon j must not see actions taken after step j, so perturbing the
    # action at index k may only change horizons >= k.
    for k in range(horizon):
        perturbed = actions.at[:, k, :].add(5.0)
        after = model.apply(variables, r, z, perturbed)["delta"]
        diff = np.abs(np.asarray(after - out["delta"])).max(axis=(0, 2))
        assert np.all(diff[:k] == 0.0), (k, diff)
        assert diff[k] > 0.0, (k, diff)


def test_action_conditioning_requires_action_chunk():
    model = AnchoredDeltaHead(horizon=3, latent_dim=6, width=8,
                              action_dim=3, use_action_conditioning=True)
    r = jnp.ones((1, 5, 7)); z = jnp.ones((1, 6))
    variables = model.init(jax.random.key(7), r, z, jnp.ones((1, 3, 3)))
    try:
        model.apply(variables, r, z)
    except ValueError as exc:
        assert "action chunk" in str(exc)
    else:
        raise AssertionError("missing action chunk must raise")


def test_head_with_flags_off_keeps_the_legacy_parameter_tree():
    import flax.traverse_util as traverse_util

    head = AnchoredDeltaHead(horizon=3, latent_dim=6, width=8)
    variables = head.init(jax.random.key(8), jnp.ones((2, 5, 7)), jnp.ones((2, 6)))
    names = set(traverse_util.flatten_dict(variables, sep="/"))
    assert not any("direct_readout" in k or "pool_norm" in k for k in names), names
    enabled = AnchoredDeltaHead(horizon=3, latent_dim=6, width=8, action_dim=3,
                                use_action_conditioning=True, use_direct_readout=True)
    variables = enabled.init(jax.random.key(9), jnp.ones((2, 5, 7)), jnp.ones((2, 6)),
                             jnp.ones((2, 3, 3)))
    names = set(traverse_util.flatten_dict(variables, sep="/"))
    assert any("direct_readout" in k for k in names)
    assert any("action_in" in k for k in names)


def test_action_branch_is_a_no_op_at_initialisation():
    """With the value projection zeroed, enabling conditioning must not change
    the head's output until it has been trained."""
    horizon, latent_dim, width, action_dim = 3, 6, 8, 3
    plain = AnchoredDeltaHead(horizon=horizon, latent_dim=latent_dim, width=width)
    cond = AnchoredDeltaHead(horizon=horizon, latent_dim=latent_dim, width=width,
                             action_dim=action_dim, use_action_conditioning=True)
    r = jax.random.normal(jax.random.key(11), (2, 5, 7))
    z = jax.random.normal(jax.random.key(12), (2, latent_dim))
    actions = jax.random.normal(jax.random.key(13), (2, horizon, action_dim))
    plain_vars = plain.init(jax.random.key(14), r, z)
    cond_vars = cond.init(jax.random.key(14), r, z, actions)
    reference = plain.apply(plain_vars, r, z)["delta"]
    forward = cond.apply(cond_vars, r, z, actions)["delta"]
    # Shared branches have the same parameters, and the action branch adds zero,
    # so the two outputs must agree exactly.
    np.testing.assert_allclose(np.asarray(forward), np.asarray(reference), rtol=0, atol=1e-6)


def test_action_gradient_reaches_the_conditioning_parameters():
    horizon, latent_dim, width, action_dim = 3, 6, 8, 3
    head = AnchoredDeltaHead(horizon=horizon, latent_dim=latent_dim, width=width,
                             action_dim=action_dim, use_action_conditioning=True)
    r = jax.random.normal(jax.random.key(21), (2, 5, 7))
    z = jax.random.normal(jax.random.key(22), (2, latent_dim))
    actions = jax.random.normal(jax.random.key(23), (2, horizon, action_dim))
    variables = head.init(jax.random.key(24), r, z, actions)

    def loss(vars):
        return jnp.mean(jnp.square(head.apply(vars, r, z, actions)["delta"]))

    grads = jax.grad(loss)(variables)
    import flax.traverse_util as traverse_util
    flat = traverse_util.flatten_dict(grads, sep="/")
    value_grad = flat["params/action_value/kernel"]
    assert float(jnp.abs(value_grad).max()) > 0.0


def test_vlm_context_is_a_no_op_at_initialisation_but_still_gets_gradient():
    horizon, latent_dim, width, ctx_dim = 3, 6, 8, 11
    plain = AnchoredDeltaHead(horizon=horizon, latent_dim=latent_dim, width=width)
    ctx_head = AnchoredDeltaHead(horizon=horizon, latent_dim=latent_dim, width=width,
                                 vlm_context_dim=ctx_dim, use_vlm_context=True)
    r = jax.random.normal(jax.random.key(31), (2, 5, 7))
    z = jax.random.normal(jax.random.key(32), (2, latent_dim))
    ctx = jax.random.normal(jax.random.key(33), (2, ctx_dim))
    plain_vars = plain.init(jax.random.key(34), r, z)
    ctx_vars = ctx_head.init(jax.random.key(34), r, z, None, ctx)
    reference = plain.apply(plain_vars, r, z)["delta"]
    forward = ctx_head.apply(ctx_vars, r, z, None, ctx)["delta"]
    np.testing.assert_allclose(np.asarray(forward), np.asarray(reference), rtol=0, atol=1e-6)

    def loss(vars):
        return jnp.mean(jnp.square(ctx_head.apply(vars, r, z, None, ctx)["delta"]))

    grads = jax.grad(loss)(ctx_vars)
    import flax.traverse_util as traverse_util
    flat = traverse_util.flatten_dict(grads, sep="/")
    assert float(jnp.abs(flat["params/vlm_context_out/kernel"]).max()) > 0.0


def test_two_terms_are_not_claimed_independent():
    z = jnp.zeros((1, 2)); target = jnp.ones((1, 1, 2)); d = jnp.zeros_like(target)
    valid = jnp.ones((1, 1), dtype=bool)
    one, _ = anchored_loss(d, z, target, valid, delta_weight=0.)
    two, _ = anchored_loss(d, z, target, valid, delta_weight=1.)
    np.testing.assert_allclose(two, 2 * one)


def test_reverse_flow_gradient_reaches_head_after_zero_init_warmup():
    head = AnchoredDeltaHead(horizon=3, latent_dim=6, width=8)
    cross = ActionDeltaCrossAttention(action_width=12, width=8)
    r = jax.random.normal(jax.random.key(10), (2, 5, 7))
    z = jax.random.normal(jax.random.key(11), (2, 6))
    h = jax.random.normal(jax.random.key(12), (2, 3, 12))
    hp = head.init(jax.random.key(13), r, z)["params"]
    delta = head.apply({"params": hp}, r, z)["delta"]
    cp = cross.init(jax.random.key(14), h, delta)["params"]

    def flow(hp, cp):
        d = head.apply({"params": hp}, r, z)["delta"]
        hidden = cross.apply({"params": cp}, h, d)["hidden"]
        return jnp.square(hidden[..., :7] - .3).mean()

    # Exact zero initialization initially blocks flow->head, as documented.
    first = jax.grad(flow, argnums=0)(hp, cp)
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(first))
    # A first output-projection update unblocks the path; no latent loss used.
    gradient = jax.grad(flow, argnums=1)(hp, cp)
    cp = jax.tree.map(lambda p, g: p - .1 * g, cp, gradient)
    second = jax.grad(flow, argnums=0)(hp, cp)
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(second))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(second))


def test_sgr_mask_detach_and_zero_sensitivity_fallback():
    delta = jnp.ones((1, 3, 2))
    z = jnp.zeros((1, 2))
    target = jnp.array([[[2., 2.], [4., 4.], [jnp.nan, jnp.nan]]])
    mask = jnp.array([[True, True, False]])
    attention = jnp.array([[[.9, .09, .01], [.3, .3, .4]]])
    zeros = jnp.zeros_like(delta)
    fn = lambda d, a, s, b: control_weighted_delta_loss(d, z, target, mask, a, s, beta=b)[0]
    uniform = fn(delta, attention, zeros, 0.)
    np.testing.assert_allclose(uniform, 5.)
    np.testing.assert_allclose(fn(delta, attention, zeros, .5), uniform)
    grad = jax.grad(fn)(delta, attention, zeros, .5)
    np.testing.assert_array_equal(grad[:, 2], 0.)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(jax.grad(fn, 1)(delta, attention, jnp.ones_like(delta), .5), 0.)
    np.testing.assert_array_equal(jax.grad(fn, 2)(delta, attention, jnp.ones_like(delta), .5), 0.)
