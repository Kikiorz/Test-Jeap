import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
import numpy as np

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_jepa_ttt_adapter_is_zero_initialized_and_receives_gradient():
    adapter = _pi0.JepaTTTAdapter(32, 4, rngs=nnx.Rngs(0))
    tokens = jax.random.normal(jax.random.key(1), (2, 5, 32))
    output = adapter(tokens)
    assert jnp.array_equal(output, tokens)

    graphdef, params = nnx.split(adapter, nnx.Param)

    def loss_fn(value):
        module = nnx.merge(graphdef, value)
        return jnp.sum(module(tokens))

    gradients = jax.grad(loss_fn)(params)
    assert jnp.linalg.norm(gradients["up"]["kernel"].value) > 0


def test_jepa_ttt_adapter_gate_bypasses_and_clips_residual():
    adapter = _pi0.JepaTTTAdapter(32, 4, rngs=nnx.Rngs(0))
    tokens = jax.random.normal(jax.random.key(1), (2, 5, 32))
    baseline = adapter(tokens)
    # Make a nonzero residual to test the module-level gate.
    adapter.up.kernel.value = jnp.ones_like(adapter.up.kernel.value) * 0.01
    adapter.gate.value = jnp.asarray(0.5)
    half = adapter(tokens)
    adapter.gate.value = jnp.asarray(2.0)
    clipped = adapter(tokens)
    np.testing.assert_array_equal(baseline, tokens)
    np.testing.assert_allclose(half - tokens, 0.5 * (clipped - tokens), rtol=1e-5, atol=1e-5)


def test_jepa_ttt_adapter_requires_jepa_branch():
    with pytest.raises(ValueError, match="requires the JEPA-WAM"):
        _pi0_config.Pi0Config(pi05=True, use_jepa_ttt_adapter=True)


def test_rapr_router_continuous_residual_is_linear_and_receives_gradient():
    router = _pi0.ActionPredictiveRouter(16, 8, 3, 12, rngs=nnx.Rngs(0))
    hidden = jax.random.normal(jax.random.key(1), (2, 3, 16))
    delta = jax.random.normal(jax.random.key(2), (2, 3, 8))
    router.out.kernel.value = jnp.ones_like(router.out.kernel.value) * 0.01

    bypassed, _ = router(hidden, delta, gate_override=0.0)
    enabled, routes = router(hidden, delta, gate_override=1.0)
    half, _ = router(hidden, delta, gate_override=0.5)
    np.testing.assert_array_equal(bypassed, hidden)
    assert not np.array_equal(np.asarray(enabled), np.asarray(hidden))
    np.testing.assert_allclose(half - hidden, 0.5 * (enabled - hidden), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(routes.sum(-1), 1.0, atol=1e-6)
    gradient = jax.grad(lambda value: router(hidden, value, gate_override=1.0)[0].sum())(delta)
    assert float(jnp.linalg.norm(gradient)) > 0
    alpha_gradient = jax.grad(
        lambda alpha: router(hidden, delta, gate_override=alpha)[0].sum()
    )(jnp.asarray(1.0))
    assert float(jnp.abs(alpha_gradient)) > 0


def test_rapr_router_zero_function_initialization_preserves_base_exactly():
    router = _pi0.ActionPredictiveRouter(16, 8, 3, 12, rngs=nnx.Rngs(0))
    hidden = jax.random.normal(jax.random.key(1), (2, 3, 16))
    delta = jax.random.normal(jax.random.key(2), (2, 3, 8))
    routed, _ = router(hidden, delta)
    np.testing.assert_array_equal(routed, hidden)
    alpha = jax.nn.sigmoid(router.alpha_logit.value)
    assert 0.0 < float(alpha) < 1.0


def test_rapr_requires_pretrained_jepa_pi05_and_freezes_base():
    with pytest.raises(ValueError, match="RAPR requires"):
        _pi0_config.Pi0Config(use_rapr=True)
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=2,
        vjepa_target_dim=8,
        use_rapr=True,
        rapr_delta_dim=8,
        rapr_width=8,
    )
    frozen = _get_frozen_state(config)
    paths = ["/".join(path) if isinstance(path, tuple) else str(path) for path in frozen]
    assert paths
    assert all("rapr_delta_head" not in path and "rapr_router" not in path for path in paths)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    assert not isinstance(model.rapr_router.alpha_logit, nnx.Param)
    assert any("PaliGemma" in path for path in paths)
