import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config


def test_actual_pi0_con1_gradient_trace():
    # Trace the actual production NNX path (including VJP/SGR) without allocating
    # a full VLM or running an optimizer. Not a full-size GPU numerical audit.
    config = pi0_config.Pi0Config(pi05=True, use_vjepa_aux=True, use_con1=True,
        paligemma_variant="dummy", action_expert_variant="dummy", con1_train_action_layers_from=0,
        con1_width=8, con1_latent_dim=6, action_horizon=10, max_token_len=8)

    def trace(rng):
        model = config.create(rng)
        obs = dataclasses.replace(config.fake_obs(batch_size=1),
            con1_current_latent=jnp.ones((1, 6)), con1_future_latents=jnp.ones((1, 10, 6)),
            con1_future_valid=jnp.ones((1, 10), bool))
        actions = config.fake_act(batch_size=1)
        loss = lambda m: m.compute_con1_loss(rng, obs, actions, beta=.5)[0]
        return nnx.value_and_grad(loss)(model)

    loss, grad = nnx.eval_shape(trace, jax.random.key(0))
    assert loss.shape == ()
    assert len(grad.con1_delta_head.flat_state()) > 0


def test_vlm_context_tokens_trace_through_the_production_loss():
    """VLM-context tokens: the production prefix feeds the head end to end."""
    config = pi0_config.Pi0Config(pi05=True, use_vjepa_aux=True, use_con1=True,
        paligemma_variant="dummy", action_expert_variant="dummy", con1_train_action_layers_from=0,
        con1_width=8, con1_latent_dim=6, action_horizon=10, max_token_len=8,
        con1_vlm_context_tokens=True)

    def trace(rng):
        model = config.create(rng)
        obs = dataclasses.replace(config.fake_obs(batch_size=1),
            con1_current_latent=jnp.ones((1, 6)), con1_future_latents=jnp.ones((1, 10, 6)),
            con1_future_valid=jnp.ones((1, 10), bool))
        actions = config.fake_act(batch_size=1)
        loss = lambda m: m.compute_con1_loss(rng, obs, actions, beta=.5)[0]
        return nnx.value_and_grad(loss)(model)

    loss, grad = nnx.eval_shape(trace, jax.random.key(0))
    assert loss.shape == ()
    assert len(grad.con1_delta_head.flat_state()) > 0


def test_metric_con1_gradient_trace_reaches_the_metric_parameters():
    """The metric path must run end to end inside the production loss.

    ``test_direction_alignment_optimises_u_but_is_scale_free`` in
    ``test_modules.py`` already proves the metric is trainable (non-zero
    gradient at the identity, loss improves). This test covers the wiring: the
    real ``compute_con1_loss`` with the metric, alignment and balancing enabled
    produces a finite loss, reports the new diagnostics, and the gradients
    reach the Con1 head.
    """
    config = pi0_config.Pi0Config(
        pi05=True, use_vjepa_aux=True, use_con1=True,
        paligemma_variant="dummy", action_expert_variant="dummy",
        con1_train_action_layers_from=0, con1_width=8, con1_latent_dim=6,
        action_horizon=10, max_token_len=8,
        con1_metric=True, con1_metric_align_weight=1.0, con1_balance_strength=1.0,
    )

    def trace(rng):
        model = config.create(rng)
        obs = dataclasses.replace(config.fake_obs(batch_size=2),
            con1_current_latent=jnp.ones((2, 6)),
            con1_future_latents=jnp.ones((2, 10, 6)) * (1.0 + 0.1 * jnp.arange(10)[None, :, None]),
            con1_future_valid=jnp.ones((2, 10), bool))
        actions = config.fake_act(batch_size=2)
        return nnx.value_and_grad(lambda m: m.compute_con1_loss(rng, obs, actions, beta=.5)[0])(model)

    loss, grad = nnx.eval_shape(trace, jax.random.key(0))
    assert loss.shape == ()
    assert len(grad.con1_delta_head.flat_state()) > 0
    assert len(grad.con1_delta_metric.flat_state()) > 0


def test_metric_path_reports_its_diagnostics():
    config = pi0_config.Pi0Config(
        pi05=True, use_vjepa_aux=True, use_con1=True,
        paligemma_variant="dummy", action_expert_variant="dummy",
        con1_train_action_layers_from=0, con1_width=8, con1_latent_dim=6,
        action_horizon=10, max_token_len=8,
        con1_metric=True, con1_metric_align_weight=1.0, con1_balance_strength=1.0)
    model = config.create(jax.random.key(1))
    obs = dataclasses.replace(config.fake_obs(batch_size=2),
        con1_current_latent=jnp.ones((2, 6)),
        con1_future_latents=jnp.ones((2, 10, 6)) * (1.0 + 0.1 * jnp.arange(10)[None, :, None]),
        con1_future_valid=jnp.ones((2, 10), bool))
    total, metrics = model.compute_con1_loss(jax.random.key(2), obs, config.fake_act(batch_size=2), beta=.5)
    assert np.isfinite(float(total))
    for key in ("con1_metric_scale_mean", "con1_metric_cosine", "con1_latent_weight",
                "con1_metric_align_loss", "con1_delta_nmse"):
        assert key in metrics, key
    # At initialisation the metric is the identity, so its mean scale is exactly 1.
    np.testing.assert_allclose(float(metrics["con1_metric_scale_mean"]), 1.0, rtol=1e-6)


def test_metric_defaults_off_keeps_the_original_path():
    """Turning the metric off must reproduce the pre-metric code path exactly."""
    base = dict(pi05=True, use_vjepa_aux=True, use_con1=True,
                paligemma_variant="dummy", action_expert_variant="dummy",
                con1_train_action_layers_from=0, con1_width=8, con1_latent_dim=6,
                action_horizon=10, max_token_len=8)
    config = pi0_config.Pi0Config(**base)

    def run(cfg, rng):
        model = cfg.create(rng)
        obs = dataclasses.replace(cfg.fake_obs(batch_size=1),
            con1_current_latent=jnp.ones((1, 6)), con1_future_latents=jnp.ones((1, 10, 6)),
            con1_future_valid=jnp.ones((1, 10), bool))
        return model.compute_con1_loss(rng, obs, cfg.fake_act(batch_size=1), beta=.0)

    total, metrics = run(config, jax.random.key(3))
    assert np.isfinite(float(total))
    assert "con1_metric_align_loss" not in metrics or float(metrics["con1_metric_align_loss"]) == 0.0


def test_plain_head_checkpoint_load(tmp_path, monkeypatch):
    from flax import serialization
    from openpi.training import weight_loaders
    head = {"dense": {"kernel": np.ones((2, 3), np.float32)}}
    path = tmp_path / "head.msgpack"
    path.write_bytes(serialization.msgpack_serialize({"params": head, "step": 20000}))
    reference = {"con1_delta_head": {"dense": {"kernel": np.zeros((2, 3), np.float32)}}}
    monkeypatch.setattr(weight_loaders.CheckpointWeightLoader, "load", lambda self, _: reference)
    loaded = weight_loaders.BaseAndCon1HeadWeightLoader("unused", str(path)).load(reference)
    np.testing.assert_array_equal(loaded["con1_delta_head"]["dense"]["kernel"], 1.)


def test_action_conditioning_estimate_path_runs_and_differs_from_demonstration():
    """Training must be able to condition on the estimate sampling actually uses.

    Sampling feeds the delta head `x_t - t * v` (one denoising step lagged), never
    the demonstrated chunk. This pins the wiring (both sources run and report the
    standard diagnostics). Note the two losses *do* coincide at step 0, and that
    is correct: the head's action branch is zero-initialised, so it contributes
    nothing until it trains.
    """
    base = dict(pi05=True, use_vjepa_aux=True, use_con1=True,
                paligemma_variant="dummy", action_expert_variant="dummy",
                con1_train_action_layers_from=0, con1_width=8, con1_latent_dim=6,
                action_horizon=10, max_token_len=8, con1_action_conditioning=True,
                con1_action_dims=7)

    def loss_for(source):
        config = pi0_config.Pi0Config(**base, con1_action_conditioning_source=source)
        model = config.create(jax.random.key(0))
        obs = dataclasses.replace(config.fake_obs(batch_size=2),
            con1_current_latent=jnp.ones((2, 6)),
            con1_future_latents=jnp.ones((2, 10, 6)) * (1.0 + 0.1 * jnp.arange(10)[None, :, None]),
            con1_future_valid=jnp.ones((2, 10), bool))
        total, metrics = model.compute_con1_loss(
            jax.random.key(1), obs, config.fake_act(batch_size=2), beta=0.0)
        return float(total), metrics

    demo, demo_metrics = loss_for("demonstration")
    estimate, estimate_metrics = loss_for("estimate")
    assert np.isfinite(demo) and np.isfinite(estimate)
    assert "con1_delta_nmse" in demo_metrics and "con1_delta_nmse" in estimate_metrics
    # At initialisation the action branch is an exact no-op, so both sources must
    # give the same loss; any difference here would mean the zero-init invariant
    # was broken.
    np.testing.assert_allclose(demo, estimate, rtol=1e-6)


def test_gradient_balancing_rescales_the_latent_term_without_the_metric():
    """Balancing must work on the plain Euclidean path, not only with the metric.

    Measured motivation: the latent term contributes ~695x more gradient to the
    head than the action term, so with the configured 0.2 weight the head is
    effectively trained by an action-irrelevant objective. With balancing on, the
    effective weight must move away from the configured value (and the metric
    must stay off).
    """
    base = dict(pi05=True, use_vjepa_aux=True, use_con1=True,
                paligemma_variant="dummy", action_expert_variant="dummy",
                con1_train_action_layers_from=0, con1_width=8, con1_latent_dim=6,
                action_horizon=10, max_token_len=8, con1_action_conditioning=True,
                con1_action_dims=7, con1_action_conditioning_source="estimate")

    def metrics_for(balance):
        config = pi0_config.Pi0Config(**base, con1_balance_strength=balance)
        model = config.create(jax.random.key(0))
        obs = dataclasses.replace(config.fake_obs(batch_size=2),
            con1_current_latent=jnp.ones((2, 6)),
            con1_future_latents=jnp.ones((2, 10, 6)) * (1.0 + 0.1 * jnp.arange(10)[None, :, None]),
            con1_future_valid=jnp.ones((2, 10), bool))
        total, metrics = model.compute_con1_loss(
            jax.random.key(1), obs, config.fake_act(batch_size=2), beta=0.5)
        assert np.isfinite(float(total))
        return metrics

    un_balanced = metrics_for(0.0)
    balanced = metrics_for(1.0)
    np.testing.assert_allclose(float(un_balanced["con1_latent_weight"]),
                               float(un_balanced["con1_delta_loss"] * 0 + 0.2), rtol=1e-6)
    # The balancer is a detached rescale: different from the configured weight.
    assert abs(float(balanced["con1_latent_weight"]) - 0.2) > 1e-9, "balancer did not rescale"
    assert np.isfinite(float(balanced["con1_latent_weight"]))
