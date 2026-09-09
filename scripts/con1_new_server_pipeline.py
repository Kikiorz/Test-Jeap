"""Resumable public-source download -> four-GPU cache -> head-only 5k."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ART = Path('/workspace/artifacts')
STATE = ART/'con1/independent_frame_states_v1'
CACHE = ART/'con1/anchored_40k_features_v1'
DATA = ART/'datasets/lerobot_libero'
BASE = ART/'models/jepa_wam_pi05_robot_sweep'
PREFIX = 'checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000'
VJEPA = Path('/workspace/vjepa2')
TEACHER = VJEPA/'vjepa2_1_vitg_384.pt'
OUTPUT = ART/'checkpoints/con1_anchored_head_40k_5k_four'
RUNTIME = ART/'con1/new_server_pipeline'
HF = ['uv','tool','run','--from','huggingface-hub==1.30.0','hf']
ENV = dict(os.environ, HF_HOME='/workspace/.hf_home', HF_HUB_OFFLINE='0',
    HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DOWNLOAD_TIMEOUT='120',
    XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2',
    PYTHONPATH=f'{REPO}/src:{REPO}/packages/openpi-client/src:{REPO}/scripts')
MULTI = dict(ENV, CUDA_VISIBLE_DEVICES='0,1,2,3',
    LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libnccl.so.2',
    NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')


def record(stage, **values):
    data = dict(stage=stage, unix_time=time.time(), **values)
    tmp = RUNTIME/'status.tmp'
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(RUNTIME/'status.json')
    print(json.dumps(data), flush=True)


def run(name, command, env=None, cwd=REPO, timeout=None):
    print(json.dumps({'event':'start','job':name,'command':list(map(str, command))}), flush=True)
    with (RUNTIME/f'{name}.log').open('a') as log:
        subprocess.run(list(map(str,command)), env=env or ENV, cwd=cwd, stdout=log,
            stderr=subprocess.STDOUT, check=True, timeout=timeout)
    print(json.dumps({'event':'complete','job':name}), flush=True)


def teacher():
    revision = '204698b45b3712590f06245fbfba32d3be539812'
    if not (VJEPA/'.git').exists():
        run('teacher_clone',['git','clone','https://github.com/facebookresearch/vjepa2.git',VJEPA])
    dirty = subprocess.check_output(['git','-C',str(VJEPA),'status','--porcelain','--untracked-files=no'],text=True)
    if dirty: raise RuntimeError('Teacher source has tracked edits; refusing checkout')
    run('teacher_pin',['git','-C',VJEPA,'checkout','--detach',revision])
    if not TEACHER.exists():
        temp = TEACHER.with_suffix('.pt.download')
        run('teacher_download',['curl','--fail','--silent','--show-error','--location','--retry','5',
            '--continue-at','-','--output',temp,'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt'])
        temp.replace(TEACHER)
    with TEACHER.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    if digest != 'b417628f1618c8bd52c0f419800b802f65794288c6ef2ff85c1341a1ae587cba':
        raise RuntimeError('V-JEPA teacher checksum mismatch')


def dataset():
    revision='a4336d589d589045d1c56423ffdf3b88a0e19b1f'
    run('dataset_download',HF+['download','physical-intelligence/libero','--type','dataset',
        '--revision',revision,'--local-dir',DATA,'--max-workers','8'])
    run('dataset_verify',HF+['cache','verify','physical-intelligence/libero','--type','dataset',
        '--revision',revision,'--local-dir',DATA,'--fail-on-missing-files'])
    info=json.loads((DATA/'meta/info.json').read_text())
    if info['total_episodes']!=1693 or info['total_frames']!=273465:
        raise RuntimeError('Unexpected dataset size')


def baseline():
    revision='ca10ccbc191d8f56b4346487913e043b2722b6d2'
    run('baseline_download',HF+['download','CokeAnd1ce/JEPA_WAM','--revision',revision,
        '--local-dir',BASE,'--include',PREFIX+'/*','--exclude',PREFIX+'/train_state/*',
        '--max-workers','8'])
    run('baseline_verify',HF+['cache','verify','CokeAnd1ce/JEPA_WAM',
        '--revision',revision,'--local-dir',BASE])
    for name in ('params/_METADATA','assets/physical-intelligence/libero/norm_stats.json'):
        if not (BASE/PREFIX/name).is_file():raise RuntimeError('Missing checkpoint component '+name)


def main():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record('preflight')
        run('four_gpu_preflight',[sys.executable,'-u','scripts/con1_four_gpu_preflight.py'],
            env=MULTI,timeout=180)
        run('torch_preflight',[sys.executable,'-c',
            'import torch; print(torch.__version__, torch.version.cuda); '
            'assert torch.cuda.device_count()==4; '
            '[(torch.ones((64,64),device=f"cuda:{i}",dtype=torch.bfloat16) @ '
            'torch.ones((64,64),device=f"cuda:{i}",dtype=torch.bfloat16)).cpu() for i in range(4)]'])
        record('downloading_public_inputs')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for future in concurrent.futures.as_completed([pool.submit(fn) for fn in (teacher,dataset,baseline)]):
                future.result()
        record('caching_teacher_states')
        cmd=[sys.executable,'-u','scripts/cache_con1_frame_states.py','--dataset-root',DATA,
            '--checkpoint',TEACHER,'--vjepa-source-root',VJEPA,'--output-root',STATE,
            '--batch-size','8','--world-size','4']
        # Commit the common manifest once before workers read it concurrently.
        run('state_manifest',cmd+['--worker-rank','0','--max-episodes','0'])
        children=[]; logs=[]
        try:
            for rank in range(4):
                log=(RUNTIME/f'states_{rank}.log').open('a');logs.append(log)
                children.append(subprocess.Popen(list(map(str,cmd+['--worker-rank',rank])),
                    cwd=REPO,env=dict(ENV,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT))
            while not all(p.poll()==0 for p in children):
                if any(p.poll() not in (None,0) for p in children):
                    raise RuntimeError('Teacher worker failed; see states_N.log')
                time.sleep(10)
        finally:
            for child in children:
                if child.poll() is None:child.terminate()
            for child in children:
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            for log in logs:log.close()
        record('caching_official_40k_R')
        run('feature_cache',[sys.executable,'-u','scripts/cache_con1_features.py',
            '--dataset',DATA,'--states',STATE,'--checkpoint',BASE/PREFIX,'--output',CACHE,
            '--gpus','0,1,2,3','--batch-size','8'])
        manifest=json.loads((CACHE/'manifest.json').read_text())
        if not manifest.get('complete') or len(manifest['episodes'])!=1693:
            raise RuntimeError('Feature cache is not complete')
        record('training_head_only',target_steps=5000,base_parameters_updated=False)
        command=[sys.executable,'-u','-m','openpi.con1.train_head','--cache',CACHE,'--output',OUTPUT,
            '--steps','5000','--batch-size','128','--horizon','10','--latent-dim','2816','--width','512']
        if (OUTPUT/'run.json').exists():command.append('--resume')
        run('training',command,env=MULTI)
        result=json.loads((OUTPUT/'status.json').read_text())
        if result['state']!='complete' or result['step']<5000:raise RuntimeError('Training incomplete')
        record('complete',output=str(OUTPUT),training_result=result,
            evaluation_scope='held-out delta prediction; no policy evaluation')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        record('error',error=repr(exc))
        raise
