import dataclasses

from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp

from openpi.models import pi0_config
from openpi.training import weight_loaders


def _dummy_config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        **kwargs,
    )


def test_disabled_model_has_no_vjepa_parameters():
    model = _dummy_config().create(jax.random.key(0))
    params = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    assert not any("vjepa_" in path for path in params)


def test_vjepa_aux_forward_and_inference_prefix():
    config = _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=3,
        vjepa_target_dim=16,
    )
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(1), config.fake_act(1)

    flow_loss, aux_loss = model.compute_loss_components(jax.random.key(1), observation, actions)
    assert flow_loss.shape == (1, config.action_horizon)
    assert aux_loss is not None
    assert aux_loss.shape == (1,)
    assert jnp.all(jnp.isfinite(flow_loss))
    assert jnp.all(jnp.isfinite(aux_loss))

    inference_observation = dataclasses.replace(observation, vjepa_target=None)
    _, prefix_mask, prefix_ar_mask = model.embed_prefix(inference_observation)
    assert jnp.all(prefix_mask[:, -config.vjepa_num_queries :])
    assert prefix_ar_mask[-config.vjepa_num_queries :].tolist() == [True, False, False, False]


def test_compact_vjepa_supervision_uses_native_query_grid():
    config = _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=6,
        vjepa_target_dim=16,
        vjepa_compact_target_dim=8,
    )
    model = config.create(jax.random.key(0))
    query = jax.random.normal(jax.random.key(1), (2, 4, 64))
    prediction = model.predict_vjepa_supervision(query)

    assert config.vjepa_supervision_shape == (4, 8)
    assert config.inputs_spec(batch_size=2)[0].vjepa_target.shape == (2, 4, 8)
    assert prediction.shape == (2, 4, 8)
    assert jnp.allclose(jnp.linalg.norm(prediction, axis=-1), 1.0, atol=1e-5)


def test_masked_queries_do_not_change_flow_prediction():
    base_config = _dummy_config()
    aux_config = _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=3,
        vjepa_target_dim=16,
        vjepa_action_attends_queries=False,
    )
    base_model = base_config.create(jax.random.key(0))
    aux_model = aux_config.create(jax.random.key(0))
    base_observation, actions = base_config.fake_obs(1), base_config.fake_act(1)
    aux_observation = dataclasses.replace(
        base_observation,
        vjepa_target=jnp.ones((1, 9, 16), dtype=jnp.float16),
    )

    base_flow, _ = base_model.compute_loss_components(jax.random.key(1), base_observation, actions)
    aux_flow, _ = aux_model.compute_loss_components(jax.random.key(1), aux_observation, actions)
    assert jnp.allclose(base_flow, aux_flow, atol=2e-3, rtol=2e-3)


def _actr_config(*, stage: int = 2) -> pi0_config.Pi0Config:
    return _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=3,
        vjepa_target_dim=16,
        use_actr=True,
        actr_stage=stage,
        actr_injection_layer=2,
        actr_interaction_dim=16,
        actr_num_heads=4,
        actr_ffn_dim=32,
    )


def test_actr_checkpoint_load_restores_serialized_none_bias_leaves():
    config = _actr_config(stage=1)
    model = config.create(jax.random.key(0))
    params = nnx.state(model, nnx.Param).to_pure_dict()
    flat = flax.traverse_util.flatten_dict(params)
    assert any(value is None for value in flat.values())

    # Orbax omits optional None leaves such as the bias of a bias-free NNX
    # Linear.  Loading must restore those structural leaves without inventing
    # any numerical parameter.
    serialized = flax.traverse_util.unflatten_dict(
        {key: value for key, value in flat.items() if value is not None}
    )
    restored = config.load(serialized)

    assert restored.actr is not None
    assert restored.actr.action_k_ar.bias.value is None
    assert restored.actr.transition_q_ar.bias.value is None


def test_actr_zero_gate_is_exact_warm_start():
    base_config = _dummy_config(
        use_vjepa_aux=True,
        vjepa_num_queries=4,
        vjepa_query_grid_size=2,
        vjepa_target_grid_size=3,
        vjepa_target_dim=16,
    )
    actr_config = _actr_config()
    base_model = base_config.create(jax.random.key(0))
    actr_model = actr_config.create(jax.random.key(0))
    # The ACTR model stores the same scanned blocks as exact early/late axis
    # slices.  Load the base parameters through the same lossless migration
    # used by the released checkpoint warm-start.
    split_base = weight_loaders._split_scanned_layers(  # noqa: SLF001
        nnx.state(base_model).to_pure_dict(), actr_config.actr_injection_layer
    )
    merged = weight_loaders._merge_params(  # noqa: SLF001
        split_base, nnx.state(actr_model).to_pure_dict(), missing_regex=".*actr.*"
    )
    graphdef, state = nnx.split(actr_model)
    state.replace_by_pure_dict(merged)
    actr_model = nnx.merge(graphdef, state)
    observation, actions = base_config.fake_obs(1), base_config.fake_act(1)

    base_flow, base_aux = base_model.compute_loss_components(jax.random.key(1), observation, actions)
    actr_flow, actr_aux = actr_model.compute_loss_components(jax.random.key(1), observation, actions)

    assert jnp.array_equal(base_flow, actr_flow)
    assert base_aux is not None and actr_aux is not None
    # Prefix-only cached attention and joint prefix/suffix attention use
    # different matrix shapes, so bfloat16 reduction order can perturb the
    # diagnostic JEPA cosine very slightly even though the prefix cannot read
    # suffix tokens.  The policy-output equivalence below remains exact.
    assert jnp.allclose(base_aux, actr_aux, atol=1e-3, rtol=1e-3)

    inference_observation = dataclasses.replace(observation, vjepa_target=None)
    noise = jax.random.normal(jax.random.key(2), (1, base_config.action_horizon, base_config.action_dim))
    base_actions = base_model.sample_actions(jax.random.key(3), inference_observation, num_steps=2, noise=noise)
    actr_actions = actr_model.sample_actions(jax.random.key(3), inference_observation, num_steps=2, noise=noise)
    assert jnp.array_equal(base_actions, actr_actions)


def test_actr_is_ordered_and_stage_one_does_not_modify_action():
    model = _actr_config(stage=1).create(jax.random.key(0))
    transition = jax.random.normal(jax.random.key(1), (2, 4, 64))
    action = jax.random.normal(jax.random.key(2), (2, 2, 64))
    time = jnp.full((2,), 0.25)
    model.actr.gate_ar.value = jnp.asarray(1.0)

    refined_transition, refined_action = model.actr(transition, action, time)
    shuffled_transition, _ = model.actr(transition, action[::-1], time)

    assert not jnp.allclose(refined_transition, transition)
    assert not jnp.allclose(refined_transition, shuffled_transition)
    assert jnp.array_equal(refined_action, action)


def test_actr_stage_two_feeds_refined_transition_back_to_action():
    model = _actr_config(stage=2).create(jax.random.key(0))
    transition = jax.random.normal(jax.random.key(1), (2, 4, 64))
    action = jax.random.normal(jax.random.key(2), (2, 2, 64))
    time = jnp.full((2,), 0.25)
    model.actr.gate_ar.value = jnp.asarray(1.0)
    model.actr.gate_ra.value = jnp.asarray(1.0)

    refined_transition, refined_action = model.actr(transition, action, time)

    assert not jnp.allclose(refined_transition, transition)
    assert not jnp.allclose(refined_action, action)


def test_actr_freezes_every_released_checkpoint_parameter():
    config = _actr_config(stage=2)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    trainable = nnx.state(model, nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))).flat_state()

    assert trainable
    assert all("actr" in str(path) for path in trainable)
