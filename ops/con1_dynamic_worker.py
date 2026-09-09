#!/usr/bin/env python3
"""Persistent LIBERO CPU/render worker; policy stays resident on its GPU."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

from con1_dynamic_queue import claim, finish, effective_limits, has_pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--concurrency', type=Path)
    parser.add_argument('--warmup-marker', type=Path)
    options = parser.parse_args()
    while ((options.warmup_marker and options.slot > 0 and not options.warmup_marker.exists())
           or (options.concurrency and options.slot >= effective_limits(options.concurrency)[options.gpu])):
        if not has_pending(options.queue):
            return
        time.sleep(2)
    # Import the unchanged rollout implementation from the evaluation venv.
    sys.path.insert(0, '/workspace/ts_JEPA_con/examples/libero')
    import main as rollout
    import numpy as np
    class ReadyJournal(rollout._EpisodeJournal):
        def append(self, record):
            result = super().append(record)
            if options.warmup_marker and record.get('status') in ('success', 'failure'):
                options.warmup_marker.touch(exist_ok=True)
            return result
    logging.basicConfig(level=logging.INFO)
    manifest = json.loads(options.panels.read_text())
    cached = {}
    while True:
        if options.concurrency and options.slot >= effective_limits(options.concurrency)[options.gpu]:
            if not has_pending(options.queue):
                break
            time.sleep(2)
            continue
        batch = claim(options.queue, options.gpu, options.slot)
        if batch is None:
            break
        batch_id, suite, category, ids = batch
        print(json.dumps(dict(event='batch_start', gpu=options.gpu, slot=options.slot, batch=batch_id,
                              suite=suite, category=category, tasks=ids)), flush=True)
        if suite not in cached:
            args = rollout.Args(host='127.0.0.1', port=options.port, run_id=options.output.name,
                task_suite_name=suite, benchmark_mode='plus',
                benchmark_revision='4976dc30028e805ff8094b55501d532c48fec182',
                classification_path='/workspace/artifacts/benchmarks/LIBERO-plus/libero/libero/benchmark/task_classification.json',
                task_ids=[r['task_id'] for r in manifest['final'][suite]],
                num_trials_per_task=1, seed=manifest['final_episode_seed'], replan_steps=5,
                save_video=False, retry_errors=True)
            np.random.seed(args.seed)
            rollout._validate_eval_args(args, 'plus')
            bench = rollout.benchmark.get_benchmark_dict()[suite]()
            rollout._validate_benchmark_protocol(args, 'plus', bench.n_tasks)
            infos = rollout._build_task_infos(args, 'plus', bench, bench.n_tasks)
            max_steps = rollout._get_max_steps(suite)
            header = rollout._make_run_header(args, 'plus', infos, bench.n_tasks, max_steps)
            cached[suite] = args, bench, infos, max_steps, header
        args, bench, infos, max_steps, header = cached[suite]
        suffix = f'_slot{options.slot}_multi' if options.concurrency else '_dynamic'
        path = options.output / 'journals' / f'plus_{suite}_gpu{options.gpu}{suffix}.jsonl'
        # Headers describe the unchanged full-suite protocol; execution is a disjoint queue subset.
        with ReadyJournal(str(path), run_header=header, resume=True, retry_errors=True) as journal:
            rollout._evaluate_tasks(args, 'plus', bench, infos, ids, bench.n_tasks,
                                    max_steps, path, journal)
            for task_id in ids:
                row = journal.records.get((suite, task_id, 0))
                if not row or row['status'] == 'error':
                    raise RuntimeError(f'Batch {batch_id}: missing/error episode {suite}/{task_id}; stop and retain journals')
        finish(options.queue, batch_id)
        print(json.dumps(dict(event='batch_done', gpu=options.gpu, batch=batch_id)), flush=True)


if __name__ == '__main__':
    main()
