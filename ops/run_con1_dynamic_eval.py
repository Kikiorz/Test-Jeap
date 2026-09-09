#!/usr/bin/env python3
"""Four persistent policy servers + work-stealing, persistent rollout workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from con1_dynamic_queue import collect, create_queue, export_canonical, make_batches, queue_status, totals, effective_limits, PRIORITY, OTHER
from run_paper_con1_eval import PLUS, PLUS_REVISION, REPO, STANDARD_REVISION, atomic_json, wait_ready


def policy_command(config, checkpoint, port):
    command = [str(REPO/'.venv/bin/python'), '-u', str(REPO/'scripts/serve_policy.py'),
               '--env', 'LIBERO', '--host', '127.0.0.1', '--port', str(port)]
    if config != 'pi05_libero_paper_reference':
        command += ['--rapr-runtime-gate', '1.0']
    return command + ['policy:checkpoint', '--policy.config', config, '--policy.dir', str(checkpoint)]


def category_suffix(category):
    return '' if category is None else '_'+category.lower().replace(' ', '_')


def selected_records(records, category):
    return records if category is None else {k:r for k,r in records.items() if r['category']==category}


def selected_batches(manifest, records, category):
    batches = make_batches(manifest, records, size=4 if category else 32)
    return batches if category is None else [b for b in batches if b[1]==category]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--panel', choices=['final'], default='final')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port-base', type=int, default=8800)
    parser.add_argument('--concurrency', type=Path)
    parser.add_argument('--category', choices=PRIORITY+OTHER)
    args = parser.parse_args()
    manifest = json.loads(args.panels.read_text())
    actual = subprocess.check_output(['git', '-C', str(PLUS), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PLUS_REVISION:
        raise ValueError('Plus benchmark changed')
    classification = json.loads((PLUS/'libero/libero/benchmark/task_classification.json').read_text())
    if hashlib.sha256(json.dumps(classification, sort_keys=True).encode()).hexdigest() != manifest['classification_sha256']:
        raise ValueError('Classification changed')
    if not (args.checkpoint/'params').is_dir():
        raise FileNotFoundError(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    for sub in ['journals', 'logs', 'runtime']:
        (args.output/sub).mkdir(exist_ok=True)
    contract = dict(config=args.config, checkpoint=str(args.checkpoint.resolve()), panel=args.panel,
                    panels=manifest, plus_revision=PLUS_REVISION, standard_revision=STANDARD_REVISION)
    contract_path = args.output/'manifest.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Existing model/benchmark contract differs')
    atomic_json(contract_path, contract)
    records, headers = collect(args.output, manifest)
    batches = selected_batches(manifest, records, args.category)
    expected = sum(args.category is None or r['category']==args.category
                   for rows in manifest['final'].values() for r in rows)
    suffix = category_suffix(args.category)
    queue = args.output/'runtime'/f'queue_{time.time_ns()}.sqlite'
    create_queue(queue, batches)
    processes, servers, workers, handles = [], [], [], []
    last_limits = None
    def publish():
        nonlocal last_limits
        rows, hdr = collect(args.output, manifest)
        rows = selected_records(rows, args.category)
        groups = totals(rows)
        limits = effective_limits(args.concurrency) if args.concurrency else [1]*4
        if limits != last_limits:
            with (args.output/'runtime/scheduling_events.jsonl').open('a') as audit:
                audit.write(json.dumps(dict(unix_time=time.time(),queue=str(queue),
                    slots_per_gpu=limits,completed=len(rows),kind='concurrency_change'))+'\n')
            last_limits = limits
        value = dict(unix_time=time.time(), groups=groups, completed=len(rows),
                     successes=sum(r['status']=='success' for r in rows.values()),
                     scheduler='dynamic_batches_4' if args.category else 'dynamic_batches_32', workers=queue_status(queue),
                     slots_per_gpu=limits, category=args.category, expected_episodes=expected)
        atomic_json(args.output/f'progress{suffix}.json', value)
        return rows, hdr, value
    publish()
    try:
        if batches:
            for gpu in range(4):
                port = args.port_base + gpu
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind(('127.0.0.1', port))
                log = (args.output/'logs'/f'dynamic_server_gpu{gpu}.log').open('a')
                handles.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO/'src'),
                    HF_HOME='/workspace/.hf_home', HF_HUB_OFFLINE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                    XLA_PYTHON_CLIENT_MEM_FRACTION='0.8', OMP_NUM_THREADS='4')
                server = subprocess.Popen(policy_command(args.config, args.checkpoint, port),
                    cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                servers.append(server); processes.append(server)
            for gpu, server in enumerate(servers):
                wait_ready(server,args.port_base+gpu)
            for gpu in range(4):
                runtime = args.output/'runtime'/f'dynamic_gpu{gpu}'
                runtime.mkdir(exist_ok=True)
                benchmark_root = PLUS/'libero/libero'
                (runtime/'config.yaml').write_text('\n'.join(f'{k}: {v}' for k,v in dict(
                    benchmark_root=benchmark_root,bddl_files=benchmark_root/'bddl_files',
                    init_states=benchmark_root/'init_files',datasets=PLUS/'libero/datasets',
                    assets=benchmark_root/'assets').items())+'\n')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                    MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
                    LIBERO_CONFIG_PATH=str(runtime), PYTHONPATH=f'{PLUS}:{REPO}/packages/openpi-client/src')
                for slot in range(16 if args.concurrency else 1):
                    log = (args.output/'logs'/f'dynamic_worker_gpu{gpu}_slot{slot}.log').open('a')
                    handles.append(log)
                    command = [str(REPO/'examples/libero/.venv-plus/bin/python'), '-u',
                        str(REPO/'ops/con1_dynamic_worker.py'), '--queue',str(queue),'--output',str(args.output),
                        '--panels',str(args.panels),'--gpu',str(gpu),'--port',str(args.port_base+gpu), '--slot',str(slot)]
                    if args.concurrency:
                        command += ['--concurrency',str(args.concurrency)]
                        command += ['--warmup-marker',str(args.output/'runtime'/f'{queue.stem}_gpu{gpu}.ready')]
                    worker = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                    workers.append(worker); processes.append(worker)
            while True:
                for worker in workers:
                    if worker.poll() not in (None,0):
                        raise RuntimeError(f'Worker {worker.pid} exited {worker.returncode}')
                if any(server.poll() is not None for server in servers):
                    raise RuntimeError('Policy server exited unexpectedly')
                publish()
                if all(worker.poll()==0 for worker in workers):
                    break
                time.sleep(10)
        records, headers, live = publish()
        errors = sum(r['status']=='error' for r in records.values())
        complete = len(records)==expected and errors==0
        report = dict(expected_episodes=expected, completed_episodes=len(records),
                      successes=live['successes'], errors=errors, groups=live['groups'], complete=complete,
                      category=args.category, scope='category' if args.category else 'full_plus')
        atomic_json(args.output/f'summary{suffix}.json',report)
        if not complete:
            raise RuntimeError('Evaluation incomplete; journals retained')
        if args.category is None:
            export_canonical(args.output,records,headers)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
