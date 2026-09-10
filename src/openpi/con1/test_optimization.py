import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.con1.optimization import mask_action_updates, scale_group_updates, stage_values


def test_stage_boundaries():
    fn = jax.jit(stage_values)
    for step, expected in [(0, (1, 0)), (1999, (1, 0)), (2000, (2, 0)),
                           (6999, (2, 0)), (7000, (3, 0)), (11999, (3, .5))]:
        stage, beta = fn(step)
        assert int(stage) == expected[0]
        np.testing.assert_allclose(beta, expected[1])


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
