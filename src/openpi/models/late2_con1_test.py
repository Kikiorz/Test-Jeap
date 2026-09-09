"""Late-two integration: original parameter tree, exact anchor, both retrievals."""

import dataclasses

import flax.nnx as nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models.orthogonal_con1 import ActionConditionedQFormer, control_weights
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config


class Harness(nnx.Module):
    _late2_context = Pi0._late2_context
    _route_late2_velocity = Pi0._route_late2_velocity

    def __init__(self):
        config = dataclasses.replace(gemma.get_config("dummy"), width=16, mlp_dim=32, depth=3)
        llm = bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="bfloat16",
                                        adarms=True, capture_action_states=True))
        llm.lazy_init(rngs=nnx.Rngs(4), method="init", use_adarms=[False, True])
        self.PaliGemma = nnx.Dict(llm=llm)
        self.action_horizon = 2
        self.rapr_router = ActionConditionedQFormer(16, 8, 2, 8, learnable_alpha=True, rngs=nnx.Rngs(5))
        self.action_out_proj = nnx.Linear(16, 7, rngs=nnx.Rngs(6), param_dtype=jnp.bfloat16)

    def context(self):
        prefix = jax.random.normal(jax.random.key(7), (1, 3, 16))
        suffix = jax.random.normal(jax.random.key(8), (1, 2, 16))
        mask = jnp.ones((1, 5, 5), bool).at[:, :3, 3:].set(False)
        positions = jnp.arange(5)[None]
        condition = jnp.ones((1, 16))
        (out, cache, states) = self.PaliGemma.llm(
            [prefix, suffix], positions, mask, [None, condition], return_action_states=True)
        context = self._late2_context(states[-2], out[1], self.action_out_proj(out[1]),
                                     positions[:, 3:], mask[:, 3:],
                                     tuple(value[-1, :, :3] for value in cache), condition)
        return context, (prefix, suffix, mask, positions, condition), out, cache


@pytest.fixture
def harness():
    return Harness()


def test_capture_does_not_change_baseline_outputs_cache_or_parameters(harness):
    _, inputs, outputs, cache = harness.context()
    config = dataclasses.replace(gemma.get_config("dummy"), width=16, mlp_dim=32, depth=3)
    original = bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="bfloat16", adarms=True))
    original.lazy_init(rngs=nnx.Rngs(4), method="init", use_adarms=[False, True])
    captured_params, original_params = nnx.state(harness.PaliGemma.llm), nnx.state(original)
    assert captured_params.flat_state().keys() == original_params.flat_state().keys()
    for path, variable in original_params.flat_state().items():
        np.testing.assert_array_equal(captured_params.flat_state()[path].value, variable.value)
    prefix, suffix, mask, positions, condition = inputs
    expected, expected_cache = original([prefix, suffix], positions, mask, [None, condition])
    for actual, baseline in zip(jax.tree.leaves((outputs, cache)), jax.tree.leaves((expected, expected_cache))):
        np.testing.assert_array_equal(actual, baseline)


def test_zero_readout_is_exact_baseline_with_two_nonzero_alphas(harness):
    context, *_ = harness.context()
    delta = jax.random.normal(jax.random.key(10), (1, 2, 8))
    value, routes, base, correction, details = harness._route_late2_velocity(context, delta)
    np.testing.assert_array_equal(value, base)
    np.testing.assert_array_equal(correction, 0)
    np.testing.assert_array_equal(details["first_correction"], 0)
    np.testing.assert_array_equal(details["last_correction"], 0)
    assert routes.shape == (1, 2, 2)
    graph, state = nnx.split(harness)
    compiled = jax.jit(lambda params, ctx, d: nnx.merge(graph, params)._route_late2_velocity(ctx, d))
    compiled_value, _, _, compiled_correction, _ = compiled(state, context, delta)
    np.testing.assert_array_equal(compiled_value, base)
    np.testing.assert_array_equal(compiled_correction, 0)


def test_both_retrievals_affect_action_and_last_a_supplies_r(harness):
    context, *_ = harness.context()
    delta = jax.random.normal(jax.random.key(10), (1, 2, 8))
    harness.rapr_router.out.kernel.value = jax.random.normal(
        jax.random.key(11), harness.rapr_router.out.kernel.value.shape) * .01
    value, routes, base, correction, details = harness._route_late2_velocity(context, delta)
    assert float(jnp.linalg.norm(details["first_correction"])) > 0
    assert float(jnp.linalg.norm(details["last_correction"])) > 0
    np.testing.assert_array_equal(routes, harness.rapr_router.residual(details["action_hidden"], delta)[1])
    assert not np.allclose(routes, details["layer17_routes"])
    np.testing.assert_allclose(correction, details["first_correction"] + details["last_correction"])
    gradient = jax.grad(lambda d: jnp.square(harness._route_late2_velocity(context, d)[0]).mean())(delta)
    assert float(jnp.linalg.norm(gradient)) > 0
    weights = control_weights(routes, delta, gradient)
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
    np.testing.assert_array_equal(harness._route_late2_velocity(context, delta, gate_override=0.)[0], base)
    half = harness._route_late2_velocity(context, delta, gate_override=.5)[0]
    assert float(jnp.linalg.norm(half - base)) > 0
    assert float(jnp.linalg.norm(value - half)) > 0


def test_invalid_placement_cannot_silently_use_final_denoising():
    with pytest.raises(ValueError, match="final_expert_layer"):
        Pi0Config(pi05=True, use_vjepa_aux=True, use_rapr=True, rapr_paper_orthogonal=True,
                  rapr_late_layer_count=2, rapr_control_stage="final_denoise")


@pytest.mark.parametrize("joint", [False, True])
def test_full_loss_gradients_trace_without_any_extra_denoising(monkeypatch, joint):
    def forbidden(*args, **kwargs):
        raise AssertionError("Late-two training must not run a denoising trajectory for r")
    monkeypatch.setattr(Pi0, "_final_denoise_routes", forbidden)
    config = Pi0Config(
        pi05=True, use_vjepa_aux=True, use_rapr=True, rapr_paper_orthogonal=True,
        rapr_late_layer_count=2, rapr_control_stage="final_expert_layer",
        rapr_train_action_expert=joint, rapr_learnable_alpha=joint,
        paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2,
        vjepa_num_queries=4, vjepa_query_grid_size=2, vjepa_target_grid_size=2,
        vjepa_target_dim=8, rapr_delta_dim=6, rapr_width=8,
    )
    def traced():
        model = config.create(jax.random.key(0))
        obs, actions = config.fake_obs(1), config.fake_act(1)
        def objective(candidate):
            flow, _, _, metrics = candidate.compute_all_loss_components(
                jax.random.key(1), obs, actions, reference_velocity=jnp.zeros_like(actions))
            return flow.mean() + .1 * metrics["rapr_prediction_loss"].mean(), metrics
        trainable = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))
        return nnx.value_and_grad(objective, argnums=nnx.DiffState(0, trainable), has_aux=True)(model)
    (_, metrics), gradients = nnx.eval_shape(traced)
    assert metrics["rapr_layer17_flow7_gain"].shape == (1,)
    assert metrics["rapr_layer18_flow7_gain"].shape == (1,)
    paths = ["/".join(map(str, path)) for path in gradients.flat_state()]
    assert any("rapr_delta_head" in path for path in paths)
    assert any("rapr_router" in path for path in paths)
    assert any("action_out_proj" in path for path in paths) == joint
    assert any("alpha_logit" in path for path in paths) == joint
