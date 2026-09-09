#!/usr/bin/env python3
"""After pinned downloads, regenerate both caches on four GPUs; never train."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path('/workspace/ts_JEPA_con')
ROOT=Path('/workspace/artifacts/research_reports/con/paper_con1_mean_from5k_20260909')
RUNTIME=ROOT/'runtime'
TEACHER=Path('/workspace/vjepa2/vjepa2_1_vitg_384.pt')
TEACHER_SHA='b417628f1618c8bd52c0f419800b802f65794288c6ef2ff85c1341a1ae587cba'


def record(name,value):
    destination=RUNTIME/(name+'.json')
    temporary=destination.with_suffix(f'.json.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(destination)


def event(name,**details):
    value={'event':name,'unix_time':time.time(),**details}
    record('teacher_rebuild_status',value);print(json.dumps(value),flush=True)


def wait_marker(name,service):
    path=RUNTIME/(name+'.json')
    while not path.is_file():
        status=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True).stdout
        if not any(word in status for word in ('RUNNING','STARTING')):
            raise RuntimeError('Required download ended without success: '+service)
        event('waiting',required=str(path));time.sleep(30)
    if not json.loads(path.read_text()).get('passed'):raise RuntimeError('Input marker did not pass')


def main():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    while not TEACHER.is_file():
        status=subprocess.run(['supervisorctl','status','con1_mean_teacher_download'],capture_output=True,text=True).stdout
        if any(word in status for word in ('RUNNING','STARTING')):
            event('waiting_for_teacher_download');time.sleep(30);continue
        download=TEACHER.with_suffix('.pt.download')
        if not download.is_file() or download.stat().st_size!=16878318788:raise RuntimeError('Teacher download incomplete')
        with download.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=TEACHER_SHA:raise RuntimeError('Teacher differs from the original cache encoder')
        download.rename(TEACHER)
    with TEACHER.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=TEACHER_SHA:raise RuntimeError('Wrong teacher checkpoint')
    record('teacher_weight_verified',{'passed':True,'sha256':digest,'path':str(TEACHER)})
    wait_marker('hf_libero_complete','con1_mean_public_downloads')
    if not json.loads((RUNTIME/'github_mean_code_verified.json').read_text()).get('passed'):
        raise RuntimeError('Published code must be verified before cache rebuild')
    wait_marker('teacher_preflight_complete','con1_mean_preparation')
    source=subprocess.check_output(['git','-C','/workspace/vjepa2','rev-parse','HEAD'],text=True).strip()
    if source!='204698b45b3712590f06245fbfba32d3be539812':raise RuntimeError('Wrong encoder source revision')
    # Preserve old partial copies; only this invocation's caches may be resumed.
    started=RUNTIME/'teacher_rebuild_started.json'
    if not started.exists():
        moved=[]
        for directory in (Path('/workspace/artifacts/con1/orthogonal_all4_frame_states_v1'),
                          Path('/workspace/artifacts/vjepa_targets/libero_vjepa2_1_vitg_384_offset10_base_wrist')):
            if directory.exists():
                backup=directory.with_name(directory.name+'.partial-direct-20260909')
                if backup.exists():raise RuntimeError('Refusing to overwrite earlier partial cache backup')
                directory.rename(backup);moved.append(str(backup))
        record('teacher_rebuild_started',{'unix_time':time.time(),'teacher_sha256':digest,'preserved_partial':moved})
    def worker(rank):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(REPO/'src'))
        with (RUNTIME/f'teacher_rebuild_gpu{rank}.log').open('a') as log:
            subprocess.run([str(REPO/'.venv/bin/python'),'-u','ops/rebuild_mean_teacher_worker.py','--rank',str(rank)],
                           cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        event('worker_finished',rank=rank)
    event('four_gpu_cache_rebuild_started')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(worker,range(4)):pass
    env=dict(os.environ,PYTHONPATH=str(REPO/'src')+':'+str(REPO/'ops'),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
             HF_HOME='/workspace/.hf_home',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for script,name in [('ops/audit_mean_teacher_inputs.py','rebuilt_teacher_target_audit'),
                        ('ops/audit_direct_con1_restart.py','rebuilt_direct_target_audit')]:
        report=RUNTIME/(name+'.json')
        if report.exists():
            if json.loads(report.read_text()).get('passed'):continue
            raise RuntimeError('Earlier target audit failed: '+str(report))
        with (RUNTIME/(name+'.log')).open('a') as log:
            subprocess.run([str(REPO/'.venv/bin/python'),script,'--output',str(RUNTIME/(name+'.json'))],
                           cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    record('teacher_rebuild_complete',{'passed':True,'unix_time':time.time(),'teacher_sha256':digest,
        'source_revision':source,'note':'Both caches regenerated on the new host. 5k model availability is a separate prerequisite.'})
    event('cache_rebuild_complete')


if __name__=='__main__':
    try:main()
    except Exception as error:event('cache_rebuild_failed',error=repr(error));raise
