"""Requires the repo's training environment, but runs without model computation."""

import dataclasses

import pytest

from openpi.training.config import PAPER_CON1_BASE_PARAMS, paper_con1_config
import run_paper_con1_experiment as experiment_module


def test_direct_restart_keeps_architecture_and_uses_new_checkpoints():
    for joint in (False, True):
        old = paper_con1_config(joint=joint, late2=True)
        new = paper_con1_config(joint=joint, late2=True, direct_delta=True)
        assert dataclasses.asdict(old.model) == dataclasses.asdict(new.model)
        assert dataclasses.replace(new.data, transition_target_mode="orthogonal") == old.data
        assert new.checkpoint_dir != old.checkpoint_dir
        assert new.policy_metadata["transition_target_mode"] == "direct"
        if joint:
            start = paper_con1_config(joint=False, late2=True, direct_delta=True)
            assert new.weight_loader.params_path == str(start.checkpoint_dir / "4999/params")
        else:
            assert new.weight_loader.params_path == PAPER_CON1_BASE_PARAMS
    with pytest.raises(ValueError, match="last-two"):
        paper_con1_config(joint=False, direct_delta=True)


def test_direct_orchestration_has_no_early_or_duplicate_final_evaluation(tmp_path, monkeypatch):
    experiment = experiment_module.Experiment(tmp_path, late2=True, direct_delta=True)
    calls = []
    monkeypatch.setattr(experiment, "prerequisites", lambda *args: None)
    monkeypatch.setattr(experiment, "wait_cache", lambda: None)
    monkeypatch.setattr(experiment, "event", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_module, "latest_checkpoint", lambda folder: -1)
    monkeypatch.setattr(experiment_module, "paired_results", lambda *args: {"complete": True})
    monkeypatch.setattr(experiment, "run", lambda name, command: calls.append(("train", name, command)))
    monkeypatch.setattr(experiment, "validation", lambda name, stage, checkpoint: calls.append(("validate", name)))

    def evaluate(name, config, checkpoint, panel):
        calls.append(("evaluate", name, panel))
        folder = tmp_path / "evaluations" / name
        folder.mkdir(parents=True)
        return folder

    monkeypatch.setattr(experiment, "evaluation", evaluate)
    experiment.execute(tmp_path / "smoke.json", skip_four_gpu_smoke=True)
    train = [item for item in calls if item[0] == "train"]
    assert [item[1] for item in train] == ["train_stage1_5000", "train_stage2_5000", "train_stage2_10000", "train_stage2_15000"]
    assert all("--direct-delta" in item[2] and "--late2" in item[2] for item in train)
    assert [item[1] for item in calls if item[0] == "validate"] == ["stage1_5000", "stage2_5000", "stage2_10000", "stage2_15000"]
    assert calls[0][0] == "train"
    assert [item[1] for item in calls if item[0] == "evaluate"] == [
        "baseline_monitor", "stage1_5000", "stage2_5000", "stage2_10000", "baseline_final", "candidate_final"]
