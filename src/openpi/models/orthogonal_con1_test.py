import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
import pytest

from openpi.models.orthogonal_con1 import ActionConditionedQFormer, OrthogonalDeltaHead, control_weights, prediction_metrics


def test_direct_r_times_s_and_zero_sensitivity_fallback():
    routes = jnp.array([[[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]]])
    delta = jnp.ones((1, 3, 2))
    gradient = jnp.array([[[0.5, 0.5], [1., 1.], [2., 2.]]])
    expected = 0.5 / 3 + 0.5 * np.array([[0.2, 0.4, 2.4]]) / 3
    np.testing.assert_allclose(control_weights(routes, delta, gradient), expected, atol=1e-7)
    # Absolute gradient scale cancels, even well below 1e-12.
    np.testing.assert_allclose(control_weights(routes, delta, gradient * 1e-20), expected, atol=1e-7)
    np.testing.assert_allclose(control_weights(routes, delta, gradient * 0), 1 / 3, atol=1e-7)
    grad = jax.grad(lambda z: control_weights(routes, z, gradient).sum())(delta)
    np.testing.assert_array_equal(grad, 0)


def test_control_weights_mask_and_squared_euclidean_risk():
    delta = jnp.ones((1, 3, 2))
    valid = jnp.array([[True, False, True]])
    weights = control_weights(jnp.ones((1, 3, 3)) / 3, delta, delta, valid=valid)
    np.testing.assert_allclose(weights, [[0.5, 0, 0.5]])
    target = jnp.zeros_like(delta).at[:, 1].set(1000)
    metrics = prediction_metrics(delta, target, weights, valid)
    np.testing.assert_allclose(metrics["rapr_prediction_loss"], 2)
    np.testing.assert_allclose(metrics["rapr_delta_mse"], 1)
    zeros = control_weights(jnp.ones((1, 3, 3)) / 3, delta, delta, valid=valid & False)
    np.testing.assert_array_equal(zeros, 0)


def test_qformer_zero_function_continuous_scale_and_gradients():
    router = ActionConditionedQFormer(12, 8, 4, 16, learnable_alpha=True, rngs=nnx.Rngs(0))
    hidden = jax.random.normal(jax.random.key(1), (2, 4, 12))
    delta = jax.random.normal(jax.random.key(2), (2, 4, 8))
    output, routes = router(hidden, delta)
    np.testing.assert_array_equal(output, hidden)
    np.testing.assert_allclose(routes.sum(-1), 1, atol=1e-6)
    router.out.kernel.value = jax.random.normal(jax.random.key(3), router.out.kernel.value.shape) * 0.1
    full, _ = router(hidden, delta, gate_override=1.)
    half, _ = router(hidden, delta, gate_override=0.5)
    np.testing.assert_allclose(half - hidden, 0.5 * (full - hidden), atol=2e-7)
    grad_delta = jax.grad(lambda d: jnp.square(router(hidden, d)[0]).sum())(delta)
    assert float(jnp.linalg.norm(grad_delta)) > 0
    gradients = nnx.grad(lambda r: jnp.square(r(hidden, delta)[0]).sum())(router)
    assert float(jnp.abs(gradients.alpha_logit.value)) > 0
    _, other_routes = router(hidden[:, ::-1], delta)
    assert not np.allclose(routes, other_routes)


def test_current_only_predictor_outputs_full_teacher_dimension():
    head = OrthogonalDeltaHead(12, 4, 28, 16, rngs=nnx.Rngs(1))
    result = head(jnp.ones((2, 6, 12)))
    assert result.shape == (2, 4, 28)
    assert bool(jnp.isfinite(result).all())


def test_module_diagnostics_do_not_mutate_router_or_normal_forward():
    router = ActionConditionedQFormer(12, 8, 4, 16, learnable_alpha=True, rngs=nnx.Rngs(0))
    hidden = jax.random.normal(jax.random.key(1), (2, 4, 12))
    delta = jax.random.normal(jax.random.key(2), (2, 4, 8))
    router.out.kernel.value = jax.random.normal(jax.random.key(3), router.out.kernel.value.shape) * .1
    before = np.asarray(router(hidden, delta)[0])
    metrics = router.diagnostic_retrieval(hidden, delta)
    np.testing.assert_array_equal(router(hidden, delta)[0], before)
    assert metrics["uniform_residual"].shape == hidden.shape
    assert float(jnp.min(metrics["condition_kl"])) >= -1e-6
    router.gamma_out.kernel.value = jnp.zeros_like(router.gamma_out.kernel.value)
    router.gamma_out.bias.value = jnp.zeros_like(router.gamma_out.bias.value)
    metrics = router.diagnostic_retrieval(hidden, delta)
    actual, _ = router.residual(hidden, delta)
    np.testing.assert_allclose(metrics["unconditioned_residual"], actual, atol=1e-7)
    np.testing.assert_allclose(metrics["condition_kl"], 0, atol=1e-7)


def test_paper_uses_alpha_not_legacy_deployment_gate():
    from types import SimpleNamespace
    from openpi.models.pi0 import Pi0

    model = SimpleNamespace(rapr_paper_orthogonal=True, rapr_runtime_gate=nnx.Variable(jnp.array(0.)))
    assert Pi0._rapr_residual_scale(model) == 1.0
    assert Pi0._rapr_residual_scale(model, .25) == .25
    model.rapr_paper_orthogonal = False
    assert Pi0._rapr_residual_scale(model) == 0.0


def test_prediction_diagnostics_handle_perfect_prediction_and_invalid_tail():
    target = jnp.array([[[1., 0.], [0., 1.], [99., 99.]]])
    valid = jnp.array([[True, True, False]])
    weights = jnp.array([[.5, .5, 0.]])
    metrics = prediction_metrics(target, target, weights, valid)
    np.testing.assert_allclose(metrics["rapr_delta_nmse"], 0)
    np.testing.assert_allclose(metrics["rapr_delta_cosine"], 1)
    np.testing.assert_allclose(metrics["rapr_prediction_target_norm_ratio"], 1)
    np.testing.assert_allclose(metrics["rapr_h3_zero_mse"], 0)
    np.testing.assert_allclose(metrics["rapr_h3_valid"], 0)
    assert all(np.isfinite(value).all() for value in prediction_metrics(target * 0, target * 0, weights * 0, valid & False).values())


def test_minimal_prediction_diagnostics_preserve_loss_and_gradients():
    delta = jax.random.normal(jax.random.key(40), (3, 4, 8))
    target = jax.random.normal(jax.random.key(41), delta.shape)
    valid = jnp.array([[True, True, True, True], [True, False, True, False], [False] * 4])
    weights = control_weights(jnp.ones((3, 4, 4)) / 4, delta, delta, valid=valid)
    full = prediction_metrics(delta, target, weights, valid)
    minimal = prediction_metrics(delta, target, weights, valid, full_diagnostics=False)
    assert set(minimal) == {"rapr_prediction_loss", "rapr_delta_nmse", "rapr_q_uniform_l1"}
    for key, value in minimal.items():
        np.testing.assert_array_equal(value, full[key])
    gradients = [jax.grad(lambda d: prediction_metrics(
        d, target, weights, valid, full_diagnostics=mode)["rapr_prediction_loss"].sum())(delta)
        for mode in (True, False)]
    np.testing.assert_array_equal(*gradients)


def test_paper_stage_freeze_boundaries_and_observation_spec():
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, use_vjepa_aux=True, use_rapr=True, rapr_paper_orthogonal=True,
                       paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2,
                       vjepa_num_queries=4, vjepa_query_grid_size=2, vjepa_target_grid_size=2,
                       vjepa_target_dim=8, rapr_delta_dim=6, rapr_width=8)
    stage1 = nnx.eval_shape(config.create, jax.random.key(0))
    trained1 = nnx.state(stage1, nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter())))
    paths1 = ["/".join(map(str, path)) for path in trained1.flat_state()]
    assert paths1 and all("rapr_delta_head" in path or "rapr_router" in path for path in paths1)
    assert not isinstance(stage1.rapr_router.alpha_logit, nnx.Param)
    config2 = dataclasses.replace(config, rapr_train_action_expert=True, rapr_learnable_alpha=True)
    stage2 = nnx.eval_shape(config2.create, jax.random.key(0))
    trained2 = nnx.state(stage2, nnx.All(nnx.Param, nnx.Not(config2.get_freeze_filter())))
    paths2 = ["/".join(map(str, path)) for path in trained2.flat_state()]
    assert any("llm" in path and "_1" in path for path in paths2)
    assert any("action_out_proj" in path for path in paths2)
    assert any("alpha_logit" in path for path in paths2)
    assert not any("PaliGemma/img" in path or "vjepa_" in path for path in paths2)
    assert all("_1" in path for path in paths2 if "llm" in path)
    obs, _ = config.inputs_spec(batch_size=2)
    assert obs.transition_valid.shape == (2, 2)
    assert obs.task_index.shape == (2,)


def test_missing_paper_targets_cannot_fall_back_to_repeated_old_features():
    from openpi.models.pi0 import Pi0
    from types import SimpleNamespace

    model = SimpleNamespace(rapr_paper_orthogonal=True)
    obs = SimpleNamespace(transition_target=None, vjepa_target=jnp.ones((1, 4, 8)))
    with pytest.raises(ValueError, match="no legacy fallback"):
        Pi0._rapr_transition_target(model, obs)


@pytest.mark.parametrize("late_layers", [0, 2])
def test_full_model_paper_loss_shape_trace_includes_masked_metrics(late_layers):
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, use_vjepa_aux=True, use_rapr=True, rapr_paper_orthogonal=True,
                       rapr_control_stage="final_expert_layer", paligemma_variant="dummy",
                       action_expert_variant="dummy", action_horizon=2, vjepa_num_queries=4,
                       vjepa_query_grid_size=2, vjepa_target_grid_size=2, vjepa_target_dim=8,
                       rapr_delta_dim=6, rapr_width=8, rapr_late_layer_count=late_layers)

    def forward():
        model = config.create(jax.random.key(0))
        observation = dataclasses.replace(config.fake_obs(2),
                                          transition_valid=jnp.array([[True, True], [True, False]]),
                                          task_index=jnp.array([10, 20]))
        return model.compute_all_loss_components(jax.random.key(1), observation, config.fake_act(2))

    flow, aux, _, metrics = nnx.eval_shape(forward)
    assert flow.shape == (2, 2) and aux.shape == (2,)
    assert metrics["rapr_delta_nmse"].shape == (2,)
    assert metrics["libero_goal_flow_numerator"].shape == (2,)


def test_final_denoise_routes_use_actual_last_step_and_are_detached():
    from openpi.models.pi0 import Pi0
    from types import SimpleNamespace

    def llm(inputs, **kwargs):
        prefix, suffix = inputs
        return (prefix, suffix), {}

    def router(hidden, delta, **kwargs):
        score = hidden[..., 0]
        routes = jax.nn.softmax(jnp.stack([score, -score], axis=-1), axis=-1)
        return hidden, routes

    fake = SimpleNamespace(
        PaliGemma=SimpleNamespace(llm=llm), rapr_control_num_steps=10,
        action_horizon=2, vjepa_action_attends_queries=True,
        embed_prefix=lambda obs: (jnp.ones((1, 2, 2)), jnp.ones((1, 2), bool), jnp.zeros((2,), bool)),
        embed_suffix=lambda obs, x, t: (x, jnp.ones((1, 2), bool), jnp.zeros((2,), bool), None),
        rapr_router=router, action_out_proj=jnp.ones_like,
        _route_action_velocity=lambda hidden, delta, **kwargs: (
            jnp.ones_like(hidden), router(hidden, delta)[1], jnp.ones_like(hidden), jnp.zeros_like(hidden)),
    )
    noise = jnp.ones((1, 2, 2))
    delta = jnp.ones((1, 2, 3))
    routes = Pi0._final_denoise_routes(fake, None, noise, delta)
    expected = jax.nn.softmax(jnp.array([0.1, -0.1]))
    np.testing.assert_allclose(routes[0, 0], expected, atol=1e-6)
    derivative = jax.grad(lambda n: Pi0._final_denoise_routes(fake, None, n, delta).sum())(noise)
    np.testing.assert_array_equal(derivative, 0)


def test_full_model_final_denoise_shape_trace_with_fixed_reference():
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, use_vjepa_aux=True, use_rapr=True, rapr_paper_orthogonal=True,
                       rapr_control_stage="final_denoise", rapr_control_num_steps=2,
                       paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2,
                       vjepa_num_queries=4, vjepa_query_grid_size=2, vjepa_target_grid_size=2,
                       vjepa_target_dim=8, rapr_delta_dim=6, rapr_width=8)

    def forward():
        model = config.create(jax.random.key(0))
        reference_config = dataclasses.replace(config, use_rapr=False, rapr_paper_orthogonal=False)
        reference = reference_config.create(jax.random.key(0))
        obs, actions = config.fake_obs(1), config.fake_act(1)
        velocity = reference.reference_training_velocity(jax.random.key(1), obs, actions, train=False)
        return model.compute_all_loss_components(jax.random.key(1), obs, actions, reference_velocity=velocity)

    _, _, _, metrics = nnx.eval_shape(forward)
    assert metrics["rapr_reference_velocity_rms"].shape == (1,)


def test_small_residual_is_not_lost_in_bfloat16_hidden_state():
    from openpi.models.pi0 import Pi0
    from types import SimpleNamespace

    projection = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    projection.kernel.value = jnp.eye(2, dtype=jnp.bfloat16)
    projection.bias.value = jnp.zeros(2, dtype=jnp.bfloat16)
    hidden = jnp.ones((1, 2, 2), dtype=jnp.bfloat16)
    correction = jnp.full((1, 2, 2), 1e-4, dtype=jnp.float32)
    fake = SimpleNamespace(rapr_paper_orthogonal=True, action_out_proj=projection,
                           rapr_router=SimpleNamespace(residual=lambda *a, **k: (correction, None)))
    velocity, _, base, residual = Pi0._route_action_velocity(fake, hidden, None)
    np.testing.assert_array_equal(hidden + correction.astype(jnp.bfloat16), hidden)
    np.testing.assert_allclose(velocity - base, 1e-4, rtol=2e-4)
    np.testing.assert_array_equal(residual, correction)
