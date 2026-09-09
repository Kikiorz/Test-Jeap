#!/usr/bin/env python3
"""Pure JEPA-WAM only: full Plus, 4 policy servers, 16 environments per GPU."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_con1_full_plus import build_manifest
from run_paper_con1_eval import PLUS, REPO, atomic_json

ROOT = Path('/workspace/artifacts/research_reports/con/pure_jepawam_full_plus_4x16_20260909')
CHECKPOINT = Path('/workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999')
CONFIG = 'pi05_libero_paper_reference'


def audit_metadata(metadata):
    tree = metadata['tree_metadata']
    forbidden = [name for name in tree if any(word in name.lower()
                 for word in ('rapr', 'change_', 'router', 'adapter', 'lora', 'point_flow'))]
    if len(tree) != 58 or forbidden:
        raise ValueError(f'Not the expected pure 58-leaf checkpoint: {len(tree)=}, {forbidden=}')
    return dict(parameter_leaves=len(tree), forbidden_parameters=forbidden)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), model='pure_jepa_wam_pi05_60k', checkpoint=str(CHECKPOINT),
                      training_enabled=False, slots_per_gpu=[16]*4, expected_episodes=10030)
        def event(state, **extra):
            status.update(state=state, unix_time=time.time(), **extra)
            atomic_json(ROOT/'status.json', status)
            print(json.dumps(status), flush=True)
        try:
            event('preflight')
            train = subprocess.run(['supervisorctl', 'status', 'con1_mean_experiment'],
                                   capture_output=True, text=True)
            if 'STOPPED' not in train.stdout:
                raise RuntimeError('Con1 training must remain explicitly STOPPED: '+train.stdout)
            from openpi.training.config import get_config
            config = get_config(CONFIG)
            if (config.model.use_rapr or config.model.use_point_flow
                    or config.model.use_action_change_mmdit or config.model.use_jepa_ttt_adapter):
                raise RuntimeError('Baseline configuration unexpectedly enables an added method')
            if Path(config.weight_loader.params_path).resolve() != (CHECKPOINT/'params').resolve():
                raise RuntimeError('Reference configuration points to a different checkpoint')
            if not (CHECKPOINT/'_CHECKPOINT_METADATA').is_file():
                raise FileNotFoundError('Uncommitted baseline checkpoint')
            metadata = (CHECKPOINT/'params/_METADATA').read_bytes()
            purity = audit_metadata(json.loads(metadata))
            norm = CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json'
            if not norm.is_file():
                raise FileNotFoundError(norm)
            purity.update(passed=True, checkpoint=str(CHECKPOINT), config=CONFIG,
                          metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                          norm_file_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
                          hub_repo='CokeAnd1ce/JEPA_WAM',
                          hub_revision='ca10ccbc191d8f56b4346487913e043b2722b6d2',
                          use_rapr=False, use_point_flow=False, use_action_change_mmdit=False,
                          use_jepa_ttt_adapter=False)
            atomic_json(ROOT/'purity_audit.json', purity)
            manifest = build_manifest(json.loads((PLUS/'libero/libero/benchmark/task_classification.json').read_text()))
            panels = ROOT/'panels.json'
            if panels.exists() and json.loads(panels.read_text()) != manifest:
                raise RuntimeError('Existing task and seed contract differs')
            atomic_json(panels, manifest)
            concurrency = ROOT/'concurrency.json'
            atomic_json(concurrency, dict(slots_per_gpu=[16]*4, trial=None))
            command = [sys.executable, '-u', str(REPO/'ops/run_con1_dynamic_eval.py'),
                       '--config', CONFIG, '--checkpoint', str(CHECKPOINT), '--panels', str(panels),
                       '--panel', 'final', '--output', str(ROOT/'baseline'), '--port-base', '8900',
                       '--concurrency', str(concurrency)]
            with (ROOT/'evaluation.log').open('a') as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    event('running', child_pid=child.pid)
                    progress = ROOT/'baseline/progress.json'
                    if progress.exists():
                        try:
                            value = json.loads(progress.read_text())
                        except json.JSONDecodeError:
                            value = None
                        if value is not None:
                            with (ROOT/'throughput_history.jsonl').open('a') as history:
                                history.write(json.dumps(value)+'\n')
                    try:
                        result = child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            if result:
                raise RuntimeError(f'Evaluation exited {result}; journals retained')
            summary = json.loads((ROOT/'baseline/summary.json').read_text())
            if not summary['complete'] or summary['completed_episodes'] != 10030:
                raise RuntimeError('Incomplete full Plus evaluation')
            event('complete', child_pid=None, successes=summary['successes'])
        except Exception as error:
            event('error', error=repr(error))
            raise


if __name__ == '__main__':
    main()
