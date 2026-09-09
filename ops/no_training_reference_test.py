"""Opt-out is training-only: preserve architecture, optimizer and evaluation."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from openpi.training.config import paper_con1_config
from run_paper_con1_experiment import Experiment


def test_opt_out_changes_only_training_reference_and_penalty():
    before = paper_con1_config(joint=True, late2=True, direct_delta=True)
    after = paper_con1_config(joint=True, late2=True, direct_delta=True, no_training_reference=True)
    assert after.model == dataclasses.replace(before.model, rapr_nonregression_weight=0.0)
    assert after.rapr_reference_params is None
    for key in ("lr_schedule", "optimizer", "data", "batch_size", "fsdp_devices", "num_train_steps",
                "training_step_offset", "keep_steps", "name", "exp_name"):
        assert getattr(after, key) == getattr(before, key)
    assert after.policy_metadata["training_nonregression_weight"] == 0
    assert before.rapr_reference_params is not None
    with pytest.raises(ValueError, match="joint-stage"):
        paper_con1_config(joint=False, no_training_reference=True)


def test_reference_loading_is_skipped_not_replaced(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("no_reference_train_test", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled reference loader was called")

    off = paper_con1_config(joint=True, late2=True, direct_delta=True, no_training_reference=True)
    monkeypatch.setattr(trainer._weight_loaders, "CheckpointWeightLoader", forbidden)
    assert not trainer.needs_fixed_training_reference(off)
    assert trainer.init_fixed_reference(off, None) == (None, None)
    enabled = dataclasses.replace(off, model=dataclasses.replace(off.model, rapr_nonregression_weight=1.0))
    assert trainer.needs_fixed_training_reference(enabled)
    with pytest.raises(ValueError, match="fixed original checkpoint"):
        trainer.init_fixed_reference(enabled, None)


def test_orchestrator_disables_reference_only_for_joint_training(tmp_path):
    experiment = Experiment(tmp_path, late2=True, direct_delta=True, no_training_reference=True)
    assert "--no-training-reference" not in experiment.variant_args()
    assert "--no-training-reference" not in experiment.training_args(1)
    assert "--no-training-reference" in experiment.training_args(2)
    assert experiment.config(joint=False).model.rapr_nonregression_weight == 1
    assert experiment.config(joint=True).model.rapr_nonregression_weight == 0
    # Validation/rollout retain their existing arguments and fixed baselines.
    assert experiment.variant_args() == ["--late2", "--direct-delta"]


def test_last_two_branch_propagates_training_scope_not_evaluation_change(tmp_path):
    experiment = Experiment(tmp_path, late2=True, direct_delta=True, no_training_reference=True,
                            train_action_last2=True)
    assert "--train-action-last2" not in experiment.variant_args()
    assert "--train-action-last2" not in experiment.training_args(1)
    assert "--train-action-last2" in experiment.training_args(2)
    assert experiment.config(joint=False).rapr_action_train_last_n == 0
    assert experiment.config(joint=True).rapr_action_train_last_n == 2
    assert experiment.config(joint=True).rapr_reference_params is None
