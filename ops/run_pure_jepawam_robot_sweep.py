#!/usr/bin/env python3
"""Pinned official 50k -> 40k -> 30k Robot-only evaluation; reuse 60k."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from con1_dynamic_queue import collect
from run_con1_dynamic_eval import selected_records
from run_paper_con1_eval import REPO, atomic_json
from run_pure_jepawam_full_plus import audit_metadata, CHECKPOINT as REFERENCE, ROOT as PREVIOUS

ROOT = Path('/workspace/artifacts/research_reports/con/pure_jepawam_robot_checkpoints_20260909')
MODELS = Path('/workspace/artifacts/models/jepa_wam_pi05_robot_sweep')
HUB = 'CokeAnd1ce/JEPA_WAM'
REVISION = 'ca10ccbc191d8f56b4346487913e043b2722b6d2'
CATEGORY = 'Robot Initial States'
CONFIG = 'pi05_libero_paper_reference'
ORDER = ('50k', '40k', '30k')
RUN = 'checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128'
PREFIXES = {'50k': RUN+'_continue60k_exact/50000',
            '40k': RUN+'_continue60k_exact/40000', '30k': RUN+'/29999'}


def checkpoint(label):
    return MODELS / PREFIXES[label]


def inference_file(path, prefix):
    if not path.startswith(prefix+'/'):
        return False
    relative = path[len(prefix)+1:]
    return relative == '_CHECKPOINT_METADATA' or relative.startswith(('params/', 'assets/'))


def pinned_files(label):
    url = f'https://huggingface.co/api/models/{HUB}/tree/{REVISION}/{PREFIXES[label]}?recursive=true&limit=1000'
    files = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            files.extend(row for row in json.load(response)
                         if row['type']=='file' and inference_file(row['path'], PREFIXES[label]))
            links = response.headers.get('Link', '')
        url = next((p.split('>')[0].strip()[1:] for p in links.split(',') if 'rel="next"' in p), None)
    if not files or not any(r['path'].endswith('/params/_METADATA') for r in files):
        raise RuntimeError('Missing official inference files')
    return files


def verify_file(path, row):
    if not path.is_file() or path.stat().st_size != row['size']:
        raise ValueError('Missing/wrong-size official file: '+str(path))
    lfs = row.get('lfs')
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f'blob {row["size"]}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    expected = lfs['oid'] if lfs else row['oid']
    if digest.hexdigest() != expected:
        raise ValueError('Official checksum mismatch: '+str(path))


def audit_checkpoint(label):
    path = checkpoint(label)
    if not (path/'_CHECKPOINT_METADATA').is_file():
        raise ValueError('Checkpoint marker missing')
    metadata = (path/'params/_METADATA').read_bytes()
    audit = audit_metadata(json.loads(metadata))
    reference = json.loads((REFERENCE/'params/_METADATA').read_text())
    if json.loads(metadata)['tree_metadata'] != reference['tree_metadata']:
        raise ValueError('Parameter structure differs from pure reference')
    norm = Path('assets/physical-intelligence/libero/norm_stats.json')
    if (path/norm).read_bytes() != (REFERENCE/norm).read_bytes():
        raise ValueError('Normalization differs; requires explicit protocol review')
    return dict(audit, checkpoint=str(path), config=CONFIG, hub_repo=HUB, hub_revision=REVISION,
                metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                norm_sha256=hashlib.sha256((path/norm).read_bytes()).hexdigest())


def event(mode, state, **fields):
    value = dict(mode=mode, state=state, pid=os.getpid(), unix_time=time.time(),
                 training_enabled=False, category=CATEGORY, order=ORDER, **fields)
    atomic_json(ROOT/f'{mode}_status.json', value)
    print(json.dumps(value), flush=True)


def download():
    MODELS.mkdir(parents=True, exist_ok=True)
    for label in ORDER:
        ready = ROOT/f'{label}_ready.json'
        if ready.exists():
            audit_checkpoint(label)
            continue
        files = pinned_files(label)
        needed = sum(row['size'] for row in files if not (MODELS/row['path']).exists())
        if shutil.disk_usage(MODELS).free < needed + 10*1024**3:
            raise RuntimeError('Insufficient storage headroom')
        atomic_json(ROOT/f'{label}_official_files.json', files)
        event('download', 'downloading', model=label, bytes=sum(r['size'] for r in files))
        command = ['/venv/main/bin/hf', 'download', HUB, *[r['path'] for r in files],
                   '--revision', REVISION, '--local-dir', str(MODELS), '--max-workers', '4']
        env = dict(os.environ, HF_HUB_OFFLINE='0', HF_HUB_DISABLE_PROGRESS_BARS='1')
        subprocess.run(command, env=env, check=True)
        event('download', 'verifying', model=label)
        for row in files:
            verify_file(MODELS/row['path'], row)
        atomic_json(ready, dict(audit_checkpoint(label), verified_files=len(files),
                              verified_bytes=sum(r['size'] for r in files), all_hashes_verified=True))
        event('download', 'model_ready', model=label)
    event('download', 'complete')


def summarize(rows):
    def group(items):
        counts = Counter(r['status'] for r in items)
        return dict(total=len(items), successes=counts['success'], failures=counts['failure'],
                    errors=counts['error'], success_rate=counts['success']/len(items) if items else None)
    return dict(group(list(rows.values())),
                by_level={str(level):group([r for r in rows.values() if r['difficulty_level']==level])
                          for level in range(1,6)},
                by_suite={s:group([r for r in rows.values() if r['task_suite_name']==s])
                          for s in sorted({r['task_suite_name'] for r in rows.values()})})


def paired(rows, reference):
    if rows.keys() != reference.keys():
        raise ValueError('Paired task coverage differs')
    for key, row in rows.items():
        for field in ('seed','episode_seed','max_steps','task_name','category','difficulty_level'):
            if row[field] != reference[key][field]:
                raise ValueError('Paired episode contract differs: '+field)
    return dict(wins=sum(r['status']=='success' and reference[k]['status']=='failure' for k,r in rows.items()),
                losses=sum(r['status']=='failure' and reference[k]['status']=='success' for k,r in rows.items()))


def evaluation_command(label):
    return [str(REPO/'.venv/bin/python'), '-u', str(REPO/'ops/run_con1_dynamic_eval.py'),
            '--config', CONFIG, '--checkpoint', str(checkpoint(label)), '--panels', str(ROOT/'panels.json'),
            '--panel', 'final', '--output', str(ROOT/label), '--port-base', '8900',
            '--concurrency', str(ROOT/'concurrency.json'), '--category', CATEGORY]


def evaluate():
    for service in ('con1_mean_experiment', 'pure_jepawam_full_plus_4x16'):
        result = subprocess.run(['supervisorctl','status',service], capture_output=True, text=True)
        if 'STOPPED' not in result.stdout:
            raise RuntimeError(service+' must remain STOPPED: '+result.stdout)
    from openpi.training.config import get_config
    model = get_config(CONFIG).model
    if any(getattr(model, flag) for flag in ('use_rapr','use_point_flow','use_action_change_mmdit','use_jepa_ttt_adapter')):
        raise ValueError('Added method enabled in pure model config')
    manifest = json.loads((PREVIOUS/'panels.json').read_text())
    reference, _ = collect(PREVIOUS/'baseline', manifest)
    reference = selected_records(reference, CATEGORY)
    expected = {(s,r['task_id'],0) for s,rr in manifest['final'].items() for r in rr if r['category']==CATEGORY}
    if set(reference) != expected or len(expected)!=1550 or any(r['status']=='error' for r in reference.values()):
        raise ValueError('60k Robot reference is not complete')
    if (ROOT/'panels.json').exists() and json.loads((ROOT/'panels.json').read_text())!=manifest:
        raise ValueError('Existing task/seed manifest differs')
    atomic_json(ROOT/'panels.json', manifest)
    atomic_json(ROOT/'concurrency.json', dict(slots_per_gpu=[16]*4, trial=None))
    comparisons = {'60k':dict(summarize(reference), reused=True, source=str(PREVIOUS/'baseline'))}
    atomic_json(ROOT/'comparison.json', comparisons)
    for label in ORDER:
        while not (ROOT/f'{label}_ready.json').exists():
            status_file = ROOT/'download_status.json'
            if status_file.exists() and json.loads(status_file.read_text())['state']=='error':
                raise RuntimeError('Download failed; inspect download_status.json')
            event('evaluation', 'waiting_for_download', model=label, expected_episodes=1550)
            time.sleep(10)
        audit_checkpoint(label)
        with (ROOT/f'{label}_evaluation.log').open('a') as log:
            child = subprocess.Popen(evaluation_command(label), cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            while True:
                event('evaluation', 'running', model=label, checkpoint=str(checkpoint(label)),
                      child_pid=child.pid, expected_episodes=1550, slots_per_gpu=[16]*4)
                try:
                    result = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if result:
            raise RuntimeError(f'{label} evaluator exited {result}; journals retained')
        report = json.loads((ROOT/label/'summary_robot_initial_states.json').read_text())
        if not report['complete'] or report['completed_episodes']!=1550 or report['errors']:
            raise ValueError('Incomplete evaluation; stopping sequence')
        rows, _ = collect(ROOT/label, manifest)
        if len(rows)!=1550 or any(r['category']!=CATEGORY for r in rows.values()):
            raise ValueError('Unexpected evaluation scope')
        comparisons[label] = dict(summarize(rows), versus_60k=paired(rows,reference), checkpoint=str(checkpoint(label)))
        atomic_json(ROOT/'comparison.json', comparisons)
        event('evaluation', 'model_complete', model=label, result=comparisons[label])
    event('evaluation', 'complete', models=list(comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('download','evaluation'))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/f'{args.mode}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            (download if args.mode=='download' else evaluate)()
        except Exception as error:
            event(args.mode, 'error', error=repr(error))
            raise


if __name__=='__main__':
    main()
