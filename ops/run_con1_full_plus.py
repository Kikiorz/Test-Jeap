#!/usr/bin/env python3
"""Evaluate the existing 10k and 15k checkpoints, sequentially on four GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from paper_con1_protocol import CATEGORIES, SUITES, paired_results
from run_paper_con1_eval import PLUS, REPO, atomic_json

ROOT = Path('/workspace/artifacts/research_reports/con/con1_full_plus_10k_15k_20260909')
CHECKPOINT_ROOT = Path('/workspace/artifacts/checkpoints/pi05_libero_paper_con1_direct_delta_stage2/all4_direct_delta_late2_joint_action_last2_from8k')
EXPECTED = {'libero_10': 2519, 'libero_goal': 2591, 'libero_object': 2518, 'libero_spatial': 2402}


def build_manifest(classification):
    if set(classification) != set(SUITES):
        raise ValueError('Unexpected suite set')
    tasks = {}
    for suite in SUITES:
        rows = classification[suite]
        if len(rows) != EXPECTED[suite] or [r['id'] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f'Incomplete/duplicate/reordered IDs: {suite}')
        if len({r['name'] for r in rows}) != len(rows) or {r['category'] for r in rows} != set(CATEGORIES):
            raise ValueError(f'Unexpected task names/categories: {suite}')
        tasks[suite] = [dict(task_id=r['id'] - 1, name=r['name'], category=r['category'],
                             difficulty_level=r['difficulty_level']) for r in rows]
    return dict(schema_version=1, classification_sha256=hashlib.sha256(
        json.dumps(classification, sort_keys=True).encode()).hexdigest(),
        final=tasks, final_episode_seed=743, expected_episodes=10030,
        note='All 10030 Plus tasks; one trial each; no standard LIBERO. Includes monitor tasks; not a disjoint holdout.')


def prepare(root):
    manifest = build_manifest(json.loads((PLUS / 'libero/libero/benchmark/task_classification.json').read_text()))
    for step in ('4999', '9999'):
        for item in ('params', '_CHECKPOINT_METADATA'):
            if not (CHECKPOINT_ROOT / step / item).exists():
                raise FileNotFoundError(CHECKPOINT_ROOT / step / item)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'panels.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to replace an existing task/seed contract')
    atomic_json(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--concurrency', type=Path)
    args = parser.parse_args()
    manifest = prepare(args.root)
    if args.prepare_only:
        print(json.dumps({'expected': manifest['expected_episodes'], 'suites': EXPECTED}))
        return
    lock = (args.root / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(pid=os.getpid(), total_per_model=10030, training_enabled=False,
                  scheduler='dynamic_batches_32' if args.dynamic else 'fixed_suite',
                  priority_categories=['Camera Viewpoints', 'Robot Initial States', 'Sensor Noise', 'Objects Layout'])
    def event(state, **fields):
        status.update(state=state, unix_time=time.time(), **fields)
        atomic_json(args.root / 'status.json', status)
        print(json.dumps(status), flush=True)
    try:
        for label, step in [('10k', '4999'), ('15k', '9999')]:
            output = args.root / label
            summary = output / 'summary.json'
            if summary.exists():
                prior = json.loads(summary.read_text())
                if prior.get('complete') and prior.get('completed_episodes') == 10030:
                    event('model_complete', model=label, resumed_complete=True)
                    continue
            runner = 'run_con1_dynamic_eval.py' if args.dynamic else 'run_paper_con1_eval.py'
            command = [str(REPO / '.venv/bin/python'), '-u', str(REPO / 'ops' / runner),
                       '--config', 'pi05_libero_paper_con1_direct_delta_stage2',
                       '--checkpoint', str(CHECKPOINT_ROOT / step), '--panels', str(args.root / 'panels.json'),
                       '--panel', 'final', '--output', str(output), '--port-base', '8800']
            if args.concurrency:
                if not args.dynamic:
                    raise ValueError('Concurrency requires dynamic scheduling')
                command += ['--concurrency', str(args.concurrency)]
            event('starting', model=label, checkpoint=str(CHECKPOINT_ROOT / step))
            with (args.root / f'{label}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    event('running', model=label, child_pid=child.pid)
                    try:
                        result = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            if result:
                raise RuntimeError(f'{label} evaluation exited {result}; journals retained for resume')
            report = json.loads(summary.read_text())
            if not report['complete'] or report['completed_episodes'] != 10030:
                raise RuntimeError(f'{label} is incomplete')
            event('model_complete', model=label)
        journal_dir = 'canonical' if args.dynamic else 'journals'
        if args.dynamic:
            from con1_dynamic_queue import collect, export_canonical
            for label in ('10k', '15k'):
                records, headers = collect(args.root / label, manifest)
                export_canonical(args.root / label, records, headers)
        paired = paired_results(sorted((args.root / '10k' / journal_dir).glob('*.jsonl')),
                                sorted((args.root / '15k' / journal_dir).glob('*.jsonl')))
        paired.update(reference_model='10k', candidate_model='15k')
        atomic_json(args.root / 'paired_15k_vs_10k.json', paired)
        event('complete', model='both', child_pid=None)
    except Exception as error:
        event('error', error=repr(error))
        raise


if __name__ == '__main__':
    main()
