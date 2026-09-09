#!/usr/bin/env python3
"""Resumable 5k+15k experiment, with offline and real closed-loop checkpoints."""

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from audit_con1_frame_cache import audit
from openpi.training.config import PAPER_CON1_BASE_PARAMS, paper_con1_config
from paper_con1_protocol import DIRECT_DELTA_MILESTONES, MILESTONES, SUITES, choose_panels, paired_results
from run_paper_con1_eval import PLUS, REPO, atomic_json


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def latest_checkpoint(folder):
    candidates = [int(path.name) for path in folder.iterdir()
                  if path.name.isdigit() and (path / "params/_METADATA").is_file()
                  and (path / "train_state/_METADATA").is_file()] if folder.exists() else []
    return max(candidates, default=-1)


class Experiment:
    def __init__(self, root, *, late2=False, direct_delta=False, no_training_reference=False,
                 train_action_last2=False):
        self.root = root
        self.late2 = late2
        self.direct_delta = direct_delta
        self.no_training_reference = no_training_reference
        self.train_action_last2 = train_action_last2
        if direct_delta and not late2:
            raise ValueError("Direct-delta restart must retain last-two-layer retrieval")
        self.root.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        self.python = str(REPO / ".venv/bin/python")
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", PYTHONPATH=str(REPO / "src"),
                        HF_HOME="/workspace/.hf_home", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4",
                        XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.90")

    def config(self, *, joint):
        return paper_con1_config(joint=joint, late2=self.late2, direct_delta=self.direct_delta,
                                 no_training_reference=self.no_training_reference and joint,
                                 train_action_last2=self.train_action_last2 and joint)

    def variant_args(self):
        return (["--late2"] if self.late2 else []) + (["--direct-delta"] if self.direct_delta else [])

    def training_args(self, stage):
        return self.variant_args() + (["--no-training-reference"]
                                      if self.no_training_reference and stage == 2 else []) + (
            ["--train-action-last2"] if self.train_action_last2 and stage == 2 else [])

    def event(self, event, **details):
        record = {"event": event, "unix_time": time.time(), **details}
        atomic_json(self.root / "status.json", record)
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def run(self, name, command):
        self.event("starting", job=name, command=command)
        log = self.root / "logs" / f"{name}.log"
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=handle, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    self.event("running", job=name, pid=process.pid, log=str(log))
            if code:
                self.event("failed", job=name, returncode=code, log=str(log))
                raise subprocess.CalledProcessError(code, command)
        self.event("finished", job=name, log=str(log))

    def evaluation(self, name, config, checkpoint, panel):
        output = self.root / "evaluations" / name
        completed = read_json(output / "summary.json")
        if not completed or not completed.get("complete"):
            self.run(name, [self.python, "ops/run_paper_con1_eval.py", "--config", config,
                           "--checkpoint", str(checkpoint), "--panels", str(self.root / "panels.json"),
                           "--panel", panel, "--output", str(output)])
        report = read_json(output / "summary.json")
        if not report or not report["complete"]:
            raise RuntimeError(f"Incomplete real evaluation: {name}")
        return output

    def validation(self, name, stage, checkpoint):
        output = self.root / "validation" / f"{name}.json"
        if not output.exists():
            self.run(f"validate_{name}", [self.python, "ops/validate_paper_con1.py", "--stage", str(stage),
                                        "--checkpoint", str(checkpoint), "--output", str(output)]
                     + self.variant_args())
        if not read_json(output).get("complete"):
            raise RuntimeError(f"Incomplete offline validation: {name}")
        return output

    def prerequisites(self, smoke_report, device_equivalence_report=None, transition_report=None):
        report = read_json(smoke_report)
        if not report or not report.get("passed") or report.get("initial_max_chunk_error") != 0:
            raise RuntimeError(f"Real-model initial/gradient/stage-switch smoke has not passed: {smoke_report}")
        if self.late2 and report.get("retrieval_layers") != "last_two_shared":
            raise RuntimeError("Late-two training cannot reuse output-only equivalence evidence")
        if self.late2:
            device_report = read_json(device_equivalence_report) if device_equivalence_report else None
            if (not device_report or not device_report.get("equivalence_passed")
                or device_report.get("backend") != "gpu"
                or device_report.get("retrieval_layers") != "last_two_shared"
                or device_report.get("initial_max_velocity_error") != 0
                or device_report.get("initial_max_chunk_error") != 0):
                raise RuntimeError("Late-two needs its own actual GPU zero-initialization equivalence report")
        if self.direct_delta:
            transition = read_json(transition_report) if transition_report else None
            if (not transition or not transition.get("passed") or transition.get("target_mode") != "direct"
                or not transition.get("model_architecture_unchanged")
                or transition.get("checked_tasks") != list(range(40))):
                raise RuntimeError("Direct restart requires a real-data/config direct-target audit")
            self.event("direct_target_audit_passed", report=str(transition_report),
                       note="Only labels change. Existing initial-equivalence reports cover the unchanged architecture, not direct-delta training efficacy.")
        self.event("prerequisites_passed", smoke_report=str(smoke_report))
        classification = json.loads((PLUS / "libero/libero/benchmark/task_classification.json").read_text())
        panels = choose_panels(classification)
        previous = read_json(self.root / "panels.json")
        if previous is not None and previous != panels:
            raise ValueError("Refusing to change previously fixed evaluation panels")
        atomic_json(self.root / "panels.json", panels)
        sources = ["scripts/train.py", "scripts/serve_policy.py", "src/openpi/models/pi0.py",
                   "src/openpi/models/pi0_config.py", "src/openpi/models/gemma.py", "src/openpi/models/model.py",
                   "src/openpi/models/orthogonal_con1.py", "src/openpi/training/config.py",
                   "src/openpi/training/data_loader.py", "src/openpi/training/orthogonal_targets.py",
                   "src/openpi/training/seekable_sampler.py", "src/openpi/training/weight_loaders.py",
                   "src/openpi/training/paper_metrics.py", "src/openpi/policies/policy.py",
                   "src/openpi/policies/policy_config.py", "examples/libero/main.py",
                   "ops/validate_paper_con1.py", "ops/run_paper_con1_eval.py", "ops/run_paper_con1_experiment.py"]
        sources.append("src/openpi/training/action_freeze.py")
        provenance = {"unix_time": time.time(), "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in sources},
            "transition_target_mode": "direct" if self.direct_delta else "orthogonal",
            "milestones": DIRECT_DELTA_MILESTONES if self.direct_delta else MILESTONES,
            "stage_configs": [dataclasses.asdict(self.config(joint=joint)) for joint in (False, True)]}
        # Preserve each launch's exact source/config record, including a
        # repaired resume, without pretending the dirty tree is its git HEAD.
        provenance = json.loads(json.dumps(provenance, default=repr))
        atomic_json(self.root / f"provenance_{time.time_ns()}.json", provenance)

    def wait_cache(self):
        while True:
            report = audit(Path("/workspace/artifacts/datasets/lerobot_libero"),
                           Path("/workspace/artifacts/con1/orthogonal_all4_frame_states_v1"))
            atomic_json(self.root / "cache_audit.json", report)
            if report["errors"]:
                raise RuntimeError(f"Teacher cache has errors: {report['errors'][:3]}")
            if report["complete"]:
                # Allow the successful writers to exit; never overlap their
                # GPU buffers with a full-batch training compile.
                query = subprocess.run(["supervisorctl", "status", "con1_orthogonal_features:*"],
                                       text=True, capture_output=True)
                statuses = query.stdout
                if any(f"con1_orthogonal_features_0{i}" not in statuses for i in range(4)):
                    raise RuntimeError(f"Cannot verify cache worker status: {query.stdout} {query.stderr}")
                if "RUNNING" not in statuses and "STARTING" not in statuses:
                    self.event("cache_complete", frames=report["complete_frames"])
                    return
            else:
                result = subprocess.run(["supervisorctl", "status", "con1_orthogonal_features:*"],
                                        text=True, capture_output=True)
                if "RUNNING" not in result.stdout and "STARTING" not in result.stdout:
                    raise RuntimeError("Teacher cache incomplete and no extraction process is running")
            self.event("waiting_for_live_cache_workers", frames=report["complete_frames"], total=report["total_frames"])
            time.sleep(30)

    def four_gpu_smoke(self):
        if self.late2:
            raise ValueError("Late-two uses its own verified preflight, not historical wiring smoke")
        root = self.root / "four_gpu_smoke"
        for stage, steps in ((1, 3), (1, 5), (2, 3)):
            config = paper_con1_config(joint=stage == 2, late2=self.late2)
            folder = root / config.name / "four_gpu_wiring"
            if latest_checkpoint(folder) < steps - 1:
                self.run(f"four_gpu_stage{stage}_to{steps}", [self.python, "ops/smoke_paper_con1_four_gpu.py",
                         "--stage", str(stage), "--until-step", str(steps), "--root", str(root)])
            latest = read_json(folder / "latest_metrics.json")
            if not latest or latest["completed_updates"] < steps:
                raise RuntimeError("Four-GPU trainer did not record the requested completed updates")
            if stage == 1 and (abs(latest["rapr_route_gate"] - .05) > 1e-7 or latest["rapr_action_update_norm"] != 0):
                raise RuntimeError("Stage 1 fixed-alpha/frozen-action invariant failed")
            if stage == 2 and latest["rapr_action_update_norm"] <= 0:
                raise RuntimeError("Joint action expert received no parameter update")
        config = paper_con1_config(joint=True)
        self.validation("four_gpu_wiring", 2, root / config.name / "four_gpu_wiring/2")
        atomic_json(root / "report.json", {"passed": True, "batch_size": 128, "gpus": 4,
                    "phases": ["stage1_fresh_3", "stage1_restore_to5", "stage2_fresh_optimizer_3"],
                    "note": "Wiring/performance smoke only; formal training starts fresh from original checkpoint."})

    def execute(self, smoke_report, *, baseline_only=False, skip_four_gpu_smoke=False,
                device_equivalence_report=None, transition_report=None):
        self.prerequisites(smoke_report, device_equivalence_report, transition_report)
        # Direct restart does not run any rollout before the user's first 5k
        # boundary. A copied reference panel is checked by evaluation()/pairing
        # at that boundary; it is never silently treated as a candidate result.
        baseline = None
        if not self.direct_delta or baseline_only:
            baseline = self.evaluation("baseline_monitor", "pi05_libero_paper_reference",
                                       Path(PAPER_CON1_BASE_PARAMS).parent, "monitor")
        if baseline_only:
            self.event("baseline_preparation_complete", formal_training_started=False)
            return
        self.wait_cache()
        if skip_four_gpu_smoke:
            self.event("extended_smoke_skipped_by_user", reason="User requested immediate full training",
                       note="Passed real-model preflight retained; extended restore/joint smoke is not claimed passed.")
        else:
            self.four_gpu_smoke()
        milestones = DIRECT_DELTA_MILESTONES if self.direct_delta else MILESTONES
        for stage in (1, 2):
            config = self.config(joint=stage == 2)
            if milestones[stage][-1] != config.num_train_steps:
                raise ValueError("Evaluation milestones and training phase length disagree")
            for completed in milestones[stage]:
                name = f"stage{stage}_{completed}"
                if latest_checkpoint(config.checkpoint_dir) < completed-1:
                    self.run(f"train_{name}", [self.python, "ops/train_paper_con1.py", "--stage", str(stage),
                                              "--until-step", str(completed)]
                             + self.training_args(stage))
                checkpoint = config.checkpoint_dir / str(completed-1)
                self.validation(name, stage, checkpoint)
                if self.direct_delta and stage == 2 and completed == config.num_train_steps:
                    # One final evaluation at 20k on the independent final
                    # panel below, not an extra 20k monitor-panel rollout.
                    continue
                if baseline is None:
                    baseline = self.evaluation("baseline_monitor", "pi05_libero_paper_reference",
                                               Path(PAPER_CON1_BASE_PARAMS).parent, "monitor")
                candidate = self.evaluation(name, config.name, checkpoint, "monitor")
                pair = paired_results(sorted((baseline / "journals").glob("*.jsonl")),
                                      sorted((candidate / "journals").glob("*.jsonl")))
                atomic_json(candidate / "paired_vs_original.json", pair)
                if not pair["complete"]:
                    raise RuntimeError(f"Incomplete paired result: {name}")
                self.event("milestone_evaluated", stage=stage, completed_updates=completed,
                           global_completed_updates=config.training_step_offset+completed, paired=pair)
        final_base = self.evaluation("baseline_final", "pi05_libero_paper_reference",
                                     Path(PAPER_CON1_BASE_PARAMS).parent, "final")
        final_config = self.config(joint=True)
        final_checkpoint = final_config.checkpoint_dir / str(final_config.num_train_steps - 1)
        final_candidate = self.evaluation("candidate_final", final_config.name, final_checkpoint, "final")
        result = paired_results(sorted((final_base / "journals").glob("*.jsonl")),
                                sorted((final_candidate / "journals").glob("*.jsonl")))
        atomic_json(self.root / "final_paired.json", result)
        if not result["complete"]:
            raise RuntimeError("Final paired benchmark incomplete")
        self.event("experiment_finished", checkpoint=str(final_checkpoint),
                   global_completed_updates=final_config.training_step_offset+final_config.num_train_steps,
                   final_paired=result, note="Completion is not a claim of improvement or non-regression.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--late2", action="store_true")
    parser.add_argument("--direct-delta", action="store_true")
    parser.add_argument("--train-action-last2", action="store_true")
    parser.add_argument("--no-training-reference", action="store_true",
                        help="Disable joint-stage training reference only; fixed baseline evaluations remain enabled.")
    parser.add_argument("--transition-report", type=Path)
    parser.add_argument("--device-equivalence-report", type=Path)
    parser.add_argument("--skip-four-gpu-smoke", action="store_true",
                        help="Explicit user-directed bypass of extended smoke; keeps real-model preflight/cache checks.")
    args = parser.parse_args()
    experiment = Experiment(args.root, late2=args.late2, direct_delta=args.direct_delta,
                            no_training_reference=args.no_training_reference,
                            train_action_last2=args.train_action_last2)
    with (args.root / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.execute(args.smoke_report, baseline_only=args.baseline_only,
                               skip_four_gpu_smoke=args.skip_four_gpu_smoke,
                               device_equivalence_report=args.device_equivalence_report,
                               transition_report=args.transition_report)
        except Exception as error:
            experiment.event("stopped_on_error", error=repr(error))
            raise


if __name__ == "__main__":
    main()
