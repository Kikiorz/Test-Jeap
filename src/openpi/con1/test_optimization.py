import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.con1.optimization import flow_weight, mask_action_updates, scale_group_updates, stage_values


def test_stage_boundaries():
    fn = jax.jit(stage_values)
    for step, expected in [(0, (1, 0)), (1999, (1, 0)), (2000, (2, 0)),
                           (6999, (2, 0)), (7000, (3, 0)), (11999, (3, .5))]:
        stage, beta = fn(step)
        assert int(stage) == expected[0]
        np.testing.assert_allclose(beta, expected[1])


def test_flow_weight_is_high_early_and_decays_to_floor():
    fn = jax.jit(lambda step: flow_weight(step, initial=2., final=1., warmup=2000, decay_steps=15000))
    np.testing.assert_allclose(fn(0), 2.)
    np.testing.assert_allclose(fn(1999), 2.)
    assert fn(9500) > 1.4
    np.testing.assert_allclose(fn(17000), 1., rtol=1e-6)
    assert fn(3000) < fn(1999)


def test_fusion_lr_multiplier_does_not_change_other_groups():
    updates = {
        "con1_cross_attention": {"out": jnp.ones(2), "alpha_logit": jnp.array(1.)},
        "con1_delta_head": {"kernel": jnp.ones(2)},
        "action_out_proj": {"kernel": jnp.ones(2)},
    }
    for step in (1000, 2500):
        normal = scale_group_updates(updates, step)
        raised = jax.jit(lambda u: scale_group_updates(u, step, fusion_multiplier=3.))(updates)
        np.testing.assert_allclose(raised["con1_cross_attention"]["out"],
                                   3 * normal["con1_cross_attention"]["out"])
        np.testing.assert_array_equal(raised["con1_cross_attention"]["alpha_logit"],
                                      normal["con1_cross_attention"]["alpha_logit"])
        for name in ("con1_delta_head", "action_out_proj"):
            np.testing.assert_array_equal(raised[name]["kernel"], normal[name]["kernel"])


def test_adam_state_and_freeze_are_exact_under_jit():
    model = nnx.Dict(
        PaliGemma=nnx.Dict(llm=nnx.Dict(layers=nnx.Dict(mlp_1=nnx.Dict(w=nnx.Param(jnp.ones((18, 2))))))),
        action_out_proj=nnx.Dict(kernel=nnx.Param(jnp.ones((2, 7)))),
        con1_delta_head=nnx.Dict(kernel=nnx.Param(jnp.ones((2, 2)))),
        con1_cross_attention=nnx.Dict(out=nnx.Param(jnp.ones((2, 2))), alpha_logit=nnx.Param(jnp.array(0.))))
    params = nnx.state(model)
    tx = optax.adamw(1e-5, weight_decay=.1)
    opt_state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)

    @jax.jit
    def update(params, state, step):
        g = mask_action_updates(grads, freeze_all=step < 2000)
        u, state = tx.update(g, state, params)
        u = scale_group_updates(u, step)
        u = mask_action_updates(u, freeze_all=step < 2000)
        return optax.apply_updates(params, u), state

    warmed, state = update(params, opt_state, 0)
    np.testing.assert_array_equal(warmed.PaliGemma.llm.layers.mlp_1.w.value, 1.)
    np.testing.assert_array_equal(warmed.action_out_proj.kernel.value, 1.)
    assert not np.array_equal(warmed.con1_delta_head.kernel.value, params.con1_delta_head.kernel.value)
    joint, _ = update(warmed, state, 2000)
    np.testing.assert_array_equal(joint.PaliGemma.llm.layers.mlp_1.w.value[:14], 1.)
    assert np.all(np.asarray(joint.PaliGemma.llm.layers.mlp_1.w.value[14:]) != 1.)
    assert np.all(np.asarray(joint.action_out_proj.kernel.value) != 1.)


def test_config_selects_only_intended_names():
    from openpi.training.config import get_config
    config = get_config("pi05_libero_con1_three_stage_40k")
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    paths = list(nnx.state(model, config.trainable_filter).flat_state())
    assert any(path[0] == "action_out_proj" for path in paths)
    assert any(path[0] == "con1_delta_head" for path in paths)
    assert any(path[0] == "con1_cross_attention" for path in paths)
    for path in paths:
        if path[0] == "PaliGemma":
            assert "layers" in path and any(str(x).endswith("_1") for x in path)
        else:
            assert path[0] in ("action_out_proj", "con1_delta_head", "con1_cross_attention")
