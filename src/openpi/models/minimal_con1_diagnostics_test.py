"""Three-metric telemetry must not change the paper's loss or AdamW update.

Use the real Pi0/layer-routing/trainer with a tiny vision front end and dummy
experts. These are numerical wiring tests, not real-model throughput evidence.
"""

import dataclasses
import functools
import importlib.util
from pathlib import Path

import flax.linen as linen
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import gemma, pi0
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils
from openpi.training.config import TrainConfig
from openpi.training.utils import FrozenReference, TrainState


class TinyVision(linen.Module):
    num_classes: int

    @linen.compact
    def __call__(self, image, *, train=False):
        return linen.Dense(self.num_classes)(image.mean((1, 2)))[:, None, :], {}


@pytest.mark.parametrize("joint,nonregression_weight,last_n,reduction", [
    (False, 1.0, 0, "sum"), (True, 1.0, 0, "sum"), (True, 0.0, 0, "sum"),
    (True, 0.0, 2, "sum"), (True, 0.0, 2, "mean")])
def test_minimal_diagnostics_preserve_full_trainer_update(monkeypatch, joint, nonregression_weight, last_n, reduction):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 512)
    monkeypatch.setattr(pi0._siglip, "Module", lambda *, num_classes, **kw: TinyVision(num_classes))
    config = Pi0Config(
        pi05=True, use_vjepa_aux=True, use_rapr=True, rapr_paper_orthogonal=True,
        rapr_late_layer_count=2, rapr_control_stage="final_expert_layer",
        rapr_train_action_expert=joint, rapr_learnable_alpha=joint,
        rapr_nonregression_weight=nonregression_weight,
        rapr_prediction_reduction=reduction,
        paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2, max_token_len=2,
        vjepa_num_queries=4, vjepa_query_grid_size=2, vjepa_target_grid_size=2,
        vjepa_target_dim=8, rapr_delta_dim=6, rapr_width=8,
    )
    model = config.create(jax.random.key(0))
    # Nonzero readout is essential: zero initialization alone could conceal a
    # severed sensitivity/non-regression path.
    model.rapr_router.out.kernel.value = jax.random.normal(
        jax.random.key(5), model.rapr_router.out.kernel.value.shape) * .01
    model.train()
    params = nnx_utils.state_map(nnx.state(model), config.get_freeze_filter(),
                                lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    model = nnx.merge(nnx.graphdef(model), params)
    cfg = TrainConfig(name="minimal_diagnostics_test", model=config,
                      freeze_filter=config.get_freeze_filter(), ema_decay=None,
                      rapr_action_train_last_n=last_n)
    tx = optax.adamw(1e-4)
    state = TrainState(step=jnp.asarray(700), params=params, model_def=nnx.graphdef(model),
                       opt_state=tx.init(params.filter(cfg.trainable_filter)), tx=tx, ema_decay=None)
    reference = None
    if joint and nonregression_weight > 0:
        # A distinct fixed graph exercises the original-reference branch; the
        # real-checkpoint baseline-purity audit is performed separately.
        base = nnx.clone(model)
        base.use_rapr = False
        base.rapr_late_layer_count = 0
        base.eval()
        reference = FrozenReference(params=nnx.state(base), model_def=nnx.graphdef(base))
    observation, actions = config.fake_obs(1), config.fake_act(1)
    observation = dataclasses.replace(
        observation, transition_target=jax.random.normal(jax.random.key(10), (1, 2, 6)),
        transition_valid=jnp.array([[True, False]]))
    path = Path(__file__).resolve().parents[3] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("minimal_diagnostics_train_test", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    if nonregression_weight == 0:
        def forbidden_reference(*args, **kwargs):
            raise AssertionError("Disabled reference must never execute a model forward")
        monkeypatch.setattr(pi0.Pi0, "reference_training_velocity", forbidden_reference)
        assert trainer.init_fixed_reference(cfg, None) == (None, None)
    key = jax.random.key(9)
    full_state, full_info = jax.jit(functools.partial(trainer.train_step, cfg))(
        key, state, (observation, actions), reference)

    def forbidden(*args, **kwargs):
        raise AssertionError("Minimal training must skip auxiliary retrieval and extra denoising")

    monkeypatch.setattr(pi0._orthogonal_con1.ActionConditionedQFormer, "diagnostic_retrieval", forbidden)
    monkeypatch.setattr(pi0.Pi0, "_final_denoise_routes", forbidden)
    minimal_cfg = dataclasses.replace(cfg, rapr_minimal_diagnostics=True)
    minimal_state, minimal_info = jax.jit(functools.partial(trainer.train_step, minimal_cfg))(
        key, state, (observation, actions), reference)
    for actual, expected in zip(jax.tree.leaves(minimal_state), jax.tree.leaves(full_state), strict=True):
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32),
                                   rtol=1e-6, atol=1e-7)
    for name in ("loss", "flow_loss", "rapr_prediction_loss", "rapr_nonregression_weight",
                 "rapr_route_gate", "rapr_delta_nmse", "rapr_q_uniform_l1", "rapr_residual_flow7_gain"):
        np.testing.assert_allclose(minimal_info[name], full_info[name], rtol=1e-6, atol=1e-7)
    if nonregression_weight > 0:
        np.testing.assert_allclose(minimal_info["rapr_nonregression_loss"], full_info["rapr_nonregression_loss"],
                                   rtol=1e-6, atol=1e-7)
    else:
        for info in (minimal_info, full_info):
            assert "rapr_nonregression_loss" not in info
            assert "weighted_rapr_nonregression_loss" not in info
            np.testing.assert_allclose(info["loss"], info["flow_loss"] + info["weighted_vjepa_loss"]
                                       + info["weighted_rapr_prediction_loss"], rtol=1e-6, atol=1e-7)
        assert int(minimal_state.step) == 701
    extras = {name for name in minimal_info if name.startswith("rapr_")} - {
        "rapr_prediction_loss", "rapr_nonregression_loss", "rapr_nonregression_weight", "rapr_route_gate"}
    assert extras == {"rapr_delta_nmse", "rapr_q_uniform_l1", "rapr_residual_flow7_gain"}
    assert "param_norm" not in minimal_info
    assert not any(name.endswith(("update_norm", "relative_update")) for name in minimal_info)
    for path, before in state.params.filter(config.get_freeze_filter()).flat_state().items():
        np.testing.assert_array_equal(minimal_state.params.flat_state()[path].value, before.value)
    if last_n:
        from openpi.training.action_freeze import ACTION, SCANNED_ACTION
        late_changes = 0
        for path, before in state.params.flat_state().items():
            if not ACTION(path, before):
                continue
            after = minimal_state.params.flat_state()[path].value
            if SCANNED_ACTION(path, before):
                np.testing.assert_array_equal(after[:-last_n], before.value[:-last_n])
                late_changes += np.count_nonzero(np.asarray(after[-last_n:]) != np.asarray(before.value[-last_n:]))
            else:
                np.testing.assert_array_equal(after, before.value)
        assert late_changes > 0
