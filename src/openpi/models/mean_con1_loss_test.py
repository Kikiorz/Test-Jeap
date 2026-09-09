import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.orthogonal_con1 import control_weights, prediction_metrics, prediction_output_gradient
from openpi.training.action_freeze import validate_start
from openpi.training.config import get_config, paper_con1_config, paper_con1_mean_from5k_config


@pytest.mark.parametrize("full", [False, True])
def test_feature_mean_scales_loss_and_gradient_only(full):
    delta = jax.random.normal(jax.random.key(1), (3, 4, 8))
    target = jax.random.normal(jax.random.key(2), delta.shape)
    valid = jnp.array([[True] * 4, [True, False, True, False], [False] * 4])
    weights = control_weights(jnp.ones((3, 4, 4)) / 4, delta, delta, valid=valid)
    def metrics(value, reduction):
        return prediction_metrics(value, target, weights, valid,
                                  full_diagnostics=full, feature_reduction=reduction)
    summed, averaged = metrics(delta, "sum"), metrics(delta, "mean")
    np.testing.assert_allclose(averaged["rapr_prediction_loss"], summed["rapr_prediction_loss"] / 8)
    for key in summed:
        if key != "rapr_prediction_loss":
            np.testing.assert_array_equal(averaged[key], summed[key])
    loss = lambda value, reduction: .1 * metrics(value, reduction)["rapr_prediction_loss"].mean()
    gs, gm = [jax.grad(loss)(delta, r) for r in ("sum", "mean")]
    np.testing.assert_allclose(gm, gs / 8, rtol=2e-6, atol=1e-9)
    np.testing.assert_array_equal(gm[~valid], 0)
    for reduction, automatic in (("sum", gs), ("mean", gm)):
        expected = prediction_output_gradient(delta, target, weights, loss_weight=.1,
                                              feature_reduction=reduction)
        np.testing.assert_allclose(automatic, expected, rtol=2e-6, atol=1e-9)
    assert all(np.isfinite(np.asarray(value)).all() for value in averaged.values())


def test_mean_keeps_q_sum_not_an_extra_horizon_average():
    delta = jnp.ones((1, 4, 2816))
    valid = jnp.array([[True, True, False, False]])
    weights = jnp.array([[.25, .75, 0., 0.]])
    out = prediction_metrics(delta, delta * 0, weights, valid, feature_reduction="mean")
    np.testing.assert_allclose(out["rapr_prediction_loss"], 1)
    with pytest.raises(ValueError, match="sum or mean"):
        prediction_metrics(delta, delta * 0, weights, valid, feature_reduction="invalid")


def test_mean_branch_is_separate_complete_5k_warm_start():
    new = paper_con1_mean_from5k_config()
    old = paper_con1_config(joint=True, late2=True, direct_delta=True, no_training_reference=True)
    start = paper_con1_config(joint=False, late2=True, direct_delta=True)
    assert new.name != old.name and new.checkpoint_dir != old.checkpoint_dir
    assert get_config(new.name) == new
    assert new.model == dataclasses.replace(old.model, rapr_prediction_reduction="mean")
    assert old.model.rapr_prediction_reduction == "sum"
    assert new.weight_loader.require_complete
    assert new.weight_loader.params_path == str(start.checkpoint_dir / "4999/params")
    assert new.rapr_action_freeze_anchor == new.weight_loader.params_path
    assert new.rapr_action_freeze_from_start and new.rapr_action_train_last_n == 2
    assert new.training_step_offset == 5000 and new.num_train_steps == 15000
    assert new.rapr_reference_params is None and new.model.rapr_nonregression_weight == 0
    for key in ("lr_schedule", "optimizer", "data", "batch_size", "fsdp_devices", "keep_steps"):
        assert getattr(new, key) == getattr(old, key)
    validate_start(new, resuming=False, completed_updates=0)
    validate_start(new, resuming=True, completed_updates=1)
    broken = dataclasses.replace(new, weight_loader=dataclasses.replace(new.weight_loader, require_complete=False))
    with pytest.raises(ValueError, match="complete freeze-anchor"):
        validate_start(broken, resuming=False, completed_updates=0)
    legacy = paper_con1_config(joint=True, late2=True, direct_delta=True,
                              no_training_reference=True, train_action_last2=True)
    with pytest.raises(ValueError, match="8k-or-later"):
        validate_start(legacy, resuming=False, completed_updates=0)
    validate_start(legacy, resuming=True, completed_updates=3001)


def test_full_model_mean_changes_no_action_or_q_forward(monkeypatch):
    import flax.nnx as nnx
    from openpi.models import gemma, pi0
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.minimal_con1_diagnostics_test import TinyVision

    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 512)
    monkeypatch.setattr(pi0._siglip, "Module", lambda *, num_classes, **kw: TinyVision(num_classes))
    cfg=Pi0Config(pi05=True,use_vjepa_aux=True,use_rapr=True,rapr_paper_orthogonal=True,
                  rapr_late_layer_count=2,rapr_control_stage="final_expert_layer",
                  rapr_train_action_expert=True,rapr_learnable_alpha=True,rapr_nonregression_weight=0,
                  paligemma_variant="dummy",action_expert_variant="dummy",action_horizon=2,max_token_len=2,
                  vjepa_num_queries=4,vjepa_query_grid_size=2,vjepa_target_grid_size=2,
                  vjepa_target_dim=8,rapr_delta_dim=6,rapr_width=8)
    model=cfg.create(jax.random.key(0))
    model.rapr_router.out.kernel.value=jax.random.normal(jax.random.key(5),model.rapr_router.out.kernel.value.shape)*.01
    obs=dataclasses.replace(cfg.fake_obs(2),transition_target=jax.random.normal(jax.random.key(7),(2,2,6)),
                            transition_valid=jnp.array([[True,False],[True,True]]))
    actions=cfg.fake_act(2);rng=jax.random.key(9)
    def evaluate(m):
        graph,state=nnx.split(m)
        return jax.jit(lambda s: nnx.merge(graph,s).compute_all_loss_components(rng,obs,actions))(state)
    fs,js,_,summed=evaluate(model)
    model.rapr_prediction_reduction="mean"
    fm,jm,_,mean=evaluate(model)
    np.testing.assert_array_equal(fs,fm)
    np.testing.assert_array_equal(js,jm)
    for name in ('rapr_q_uniform_l1','rapr_delta_nmse','rapr_delta_action_gradient_rms'):
        np.testing.assert_array_equal(summed[name],mean[name])
    for name in ('rapr_prediction_loss','rapr_delta_prediction_gradient_rms'):
        np.testing.assert_allclose(mean[name],summed[name]/6,rtol=2e-6,atol=1e-10)
    assert np.asarray(mean['rapr_delta_action_gradient_rms']).max()>0
