#!/usr/bin/env python3
"""Only the requested 5k->20k mean-feature branch; never retrain stage 1."""

import argparse
import dataclasses
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from audit_con1_frame_cache import audit
from openpi.training.config import PAPER_CON1_BASE_PARAMS, paper_con1_mean_from5k_config
from paper_con1_protocol import paired_results
from run_paper_con1_eval import atomic_json
from run_paper_con1_experiment import Experiment, latest_checkpoint, read_json


class MeanFrom5kExperiment(Experiment):
    def __init__(self, root):
        super().__init__(root, late2=True, direct_delta=True, no_training_reference=True)

    def config(self, *, joint):
        if not joint:
            raise ValueError("This branch starts from a completed 5k checkpoint, not stage 1")
        return paper_con1_mean_from5k_config()

    def variant_args(self):
        return ["--mean-from5k"]

    def execute_branch(self, *, input_report, test_report):
        inputs, tests = read_json(input_report), read_json(test_report)
        if not inputs or not inputs.get("passed") or not tests or not tests.get("passed"):
            raise RuntimeError("Input-integrity and mean-loss tests must pass before training")
        config = self.config(joint=True)
        anchor = Path(config.weight_loader.params_path)
        if not (anchor / "_METADATA").is_file() or not (anchor.parent / "_CHECKPOINT_METADATA").is_file():
            raise RuntimeError("The committed, complete 5k checkpoint is missing")
        panels = read_json(self.root / "panels.json")
        if not panels or sum(len(rows) for rows in panels["monitor"].values()) != 84:
            raise RuntimeError("Copy the original fixed evaluation panels before starting")
        cache = audit(Path("/workspace/artifacts/datasets/lerobot_libero"),
                      Path(config.data.transition_state_root))
        if not cache["complete"]:
            raise RuntimeError("The complete four-suite teacher frame cache is required")
        atomic_json(self.root / "cache_audit.json", cache)
        repo = Path(__file__).resolve().parents[1]
        files = ("ops/run_mean_con1_experiment.py", "ops/train_paper_con1.py", "scripts/train.py",
                 "src/openpi/models/orthogonal_con1.py", "src/openpi/models/pi0.py",
                 "src/openpi/models/pi0_config.py", "src/openpi/training/config.py",
                 "src/openpi/training/action_freeze.py", "ops/validate_paper_con1.py")
        provenance = {"unix_time": time.time(), "config": dataclasses.asdict(config),
                      "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files},
                      "initial_global_updates": 5000, "additional_updates": 15000,
                      "milestones": [10000, 15000, 20000], "input_report": str(input_report),
                      "test_report": str(test_report),
                      "comparison_caveat": "Old run trained all Action blocks from 5k to 8k and initially used non-regression; this run freezes the first 16 and disables non-regression from 5k."}
        atomic_json(self.root / f"provenance_{time.time_ns()}.json",
                    json.loads(json.dumps(provenance, default=repr)))
        self.event("mean_branch_preflight_passed", initial_checkpoint=str(anchor))
        # Three real updates belong to this run (global 5001..5003), not a
        # reset/discarded comparison. No rollout before the 10k milestone.
        audit_report=self.root / "runtime/first_updates_audit.json"
        if not audit_report.exists():
            if latest_checkpoint(config.checkpoint_dir) < 2:
                self.run("train_first_three", [self.python, "ops/train_paper_con1.py", "--stage", "2",
                                               "--until-step", "3", "--mean-from5k"])
            self.run("audit_first_three", ["env", "JAX_PLATFORMS=cpu", "CUDA_VISIBLE_DEVICES=",
                                            self.python, "ops/audit_mean_first_updates.py", "--checkpoint",
                                            str(config.checkpoint_dir / "2"), "--output", str(audit_report)])
        if not read_json(audit_report).get("passed"):
            raise RuntimeError("Real first-update parameter freeze audit did not pass")
        for completed in (5000, 10000, 15000):
            name = f"stage2_{completed}"
            if latest_checkpoint(config.checkpoint_dir) < completed - 1:
                self.run(f"train_{name}", [self.python, "ops/train_paper_con1.py", "--stage", "2",
                                         "--until-step", str(completed), "--mean-from5k"])
            checkpoint = config.checkpoint_dir / str(completed - 1)
            self.validation(name, 2, checkpoint)
            if completed == 15000:
                continue
            baseline = self.evaluation("baseline_monitor", "pi05_libero_paper_reference",
                                       Path(PAPER_CON1_BASE_PARAMS).parent, "monitor")
            candidate = self.evaluation(name, config.name, checkpoint, "monitor")
            pair = paired_results(sorted((baseline / "journals").glob("*.jsonl")),
                                  sorted((candidate / "journals").glob("*.jsonl")))
            if not pair["complete"]:
                raise RuntimeError("Milestone evaluation is incomplete")
            atomic_json(candidate / "paired_vs_original.json", pair)
            self.event("milestone_evaluated", global_completed_updates=5000 + completed, paired=pair)
        baseline = self.evaluation("baseline_final", "pi05_libero_paper_reference",
                                   Path(PAPER_CON1_BASE_PARAMS).parent, "final")
        candidate = self.evaluation("candidate_final", config.name, config.checkpoint_dir / "14999", "final")
        pair = paired_results(sorted((baseline / "journals").glob("*.jsonl")),
                              sorted((candidate / "journals").glob("*.jsonl")))
        if not pair["complete"]:
            raise RuntimeError("Final evaluation is incomplete")
        atomic_json(self.root / "final_paired.json", pair)
        self.event("experiment_finished", global_completed_updates=20000, paired=pair,
                   note="Completion is not a claim of improvement or non-regression")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    args = parser.parse_args()
    experiment = MeanFrom5kExperiment(args.root)
    with (args.root / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.execute_branch(input_report=args.input_report, test_report=args.test_report)
        except Exception as error:
            experiment.event("stopped_on_error", error=repr(error))
            raise


if __name__ == "__main__":
    main()
