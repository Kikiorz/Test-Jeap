import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

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


def test_jepa_ttt_adapter_requires_jepa_branch():
    with pytest.raises(ValueError, match="requires the JEPA-WAM"):
        _pi0_config.Pi0Config(pi05=True, use_jepa_ttt_adapter=True)
