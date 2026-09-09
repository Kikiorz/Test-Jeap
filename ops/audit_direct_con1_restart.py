#!/usr/bin/env python3
"""CPU-only raw-target/config audit; no policy evaluation or optimizer updates."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("This audit must not initialize GPUs")
    if args.output.exists():
        raise FileExistsError(args.output)
    from openpi.training import config as configs, data_loader
    from paper_con1_protocol import DIRECT_DELTA_MILESTONES
    from run_paper_con1_eval import atomic_json
    from validate_paper_con1 import column

    stages = [configs.paper_con1_config(joint=joint, late2=True, direct_delta=True) for joint in (False, True)]
    old = [configs.paper_con1_config(joint=joint, late2=True) for joint in (False, True)]
    for new, previous in zip(stages, old, strict=True):
        assert dataclasses.asdict(new.model) == dataclasses.asdict(previous.model)
        assert new.checkpoint_dir != previous.checkpoint_dir
        assert new.data.transition_target_mode == "direct"
        assert dataclasses.replace(new.data, transition_target_mode="orthogonal") == previous.data
        assert new.lr_schedule == previous.lr_schedule and new.optimizer == previous.optimizer
    assert stages[0].weight_loader.params_path == configs.PAPER_CON1_BASE_PARAMS
    assert stages[1].weight_loader.params_path == str(stages[0].checkpoint_dir / "4999/params")
    assert stages[1].training_step_offset == 5000
    assert [stages[stage-1].training_step_offset + step
            for stage, points in DIRECT_DELTA_MILESTONES.items() for step in points] == [5000, 10000, 15000, 20000]

    config = stages[0]
    data = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    tasks = column(dataset, "task_index")
    manifest = json.loads((Path(data.transition_state_root) / "manifest.json").read_text())
    checked, energy, identities = [], [], []
    for task in range(40):
        indices = np.flatnonzero(tasks == task)
        assert len(indices) > 0
        for index in indices[np.linspace(0, len(indices)-1, 4, dtype=int)]:
            sample = dataset[int(index)]
            episode = int(np.asarray(sample["episode_index"]))
            frame = int(np.asarray(sample["frame_index"]))
            cache = (Path(data.transition_state_root) / "states"
                     / f"chunk-{episode // manifest['chunks_size']:03d}" / f"episode_{episode:06d}.npy")
            states = np.load(cache, mmap_mode="r", allow_pickle=False)
            count = min(config.model.action_horizon, len(states)-frame-1)
            expected = np.zeros((config.model.action_horizon, config.model.rapr_delta_dim), np.float32)
            expected[:count] = (np.asarray(states[frame+1:frame+count+1], np.float32)
                                - np.asarray(states[frame], np.float32))
            np.testing.assert_array_equal(sample["transition_target"], expected)
            np.testing.assert_array_equal(sample["transition_valid"], np.arange(config.model.action_horizon) < count)
            assert np.isfinite(expected).all()
            energy.extend(np.square(expected[:count]).sum(-1).tolist())
            identities.append({"task": task, "episode": episode, "frame": frame, "valid_horizons": count})
        checked.append(task)
    assert energy
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {
        "passed": True, "target_mode": "direct", "formula": "z[t+j]-z[t], j=1..H, common current anchor",
        "normalization": "none", "projection": "none", "action_reconstruction_added": False,
        "model_architecture_unchanged": True, "checked_tasks": checked, "checked_rows": identities,
        "training_frames": len(dataset), "phase_updates": [c.num_train_steps for c in stages],
        "global_evaluation_updates": [5000, 10000, 15000, 20000],
        "stage1_initial_params": stages[0].weight_loader.params_path,
        "stage2_initial_params": stages[1].weight_loader.params_path,
        "sample_target_energy_mean": float(np.mean(energy)), "sample_target_energy_max": float(np.max(energy)),
        "prediction_loss_weight": config.model.rapr_loss_weight,
        "note": "Real-data label/config audit only. No model forward, training step or closed-loop evaluation was run.",
    })
    print(json.dumps({"passed": True, "output": str(args.output), "checked_tasks": len(checked),
                      "target_energy_mean": float(np.mean(energy))}), flush=True)


if __name__ == "__main__":
    main()
