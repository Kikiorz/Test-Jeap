#!/usr/bin/env python3
"""Fail-closed new-host preparation; successful checks start the managed trainer."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path('/workspace/artifacts/research_reports/con/paper_con1_mean_from5k_20260909')
REPO = Path('/workspace/ts_JEPA_con')
PYTHON = str(REPO/'.venv/bin/python')
ANCHOR = Path('/workspace/artifacts/checkpoints/pi05_libero_paper_con1_direct_delta_stage1/all4_direct_delta_late2_fixed005_5k/4999')


def record(name, value):
    path = ROOT/'runtime'/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def event(name, **details):
    value={'event':name,'unix_time':time.time(),**details}
    record('preparation_status.json',value)
    print(json.dumps(value),flush=True)


def run(name, command, *, cpu=True):
    env=dict(os.environ,PYTHONPATH=str(REPO/'src')+':'+str(REPO/'ops'),
             HF_HOME='/workspace/.hf_home',HF_HUB_OFFLINE='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if cpu:
        env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    else:
        env.pop('JAX_PLATFORMS',None)
        env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    event('checking',check=name)
    with (ROOT/'runtime'/f'{name}.log').open('a') as log:
        subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def wait_for(path, service):
    while not path.exists():
        query=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True)
        if not any(state in query.stdout for state in ('RUNNING','STARTING')):
            raise RuntimeError('Preparation service stopped without its success marker: '+str(path))
        event('waiting_for_public_inputs',required=str(path),service=service)
        time.sleep(30)
    if not json.loads(path.read_text()).get('passed'):
        raise RuntimeError('Input marker did not pass: '+str(path))


def main():
    (ROOT/'runtime').mkdir(parents=True,exist_ok=True)
    wait_for(ROOT/'runtime/github_mean_code_verified.json','con1_mean_preparation')
    tests=[
        'src/openpi/models/mean_con1_loss_test.py',
        'src/openpi/models/orthogonal_con1_test.py',
        'src/openpi/models/minimal_con1_diagnostics_test.py',
        'src/openpi/training/action_freeze_test.py',
        'ops/mean_con1_experiment_test.py',
        'ops/mean_teacher_rebuild_test.py',
    ]
    xml=ROOT/'runtime/loss_tests.xml'
    run('loss_tests',[PYTHON,'-m','pytest','-q',*tests,'--junitxml='+str(xml)])
    suites=ET.parse(xml).getroot()
    cases=list(suites.iter('testcase'))
    if not cases or list(suites.iter('failure')) or list(suites.iter('error')):
        raise RuntimeError('Mean-loss test report is not a clean pass')
    record('loss_tests.json',{'passed':True,'tests':len(cases),'xml':str(xml),'unix_time':time.time(),
                             'scope':'CPU loss/gradient/freeze/trainer and orchestration unit tests, not policy efficacy'})
    run('four_gpu_runtime',[PYTHON,'-c',
        'import jax,jax.numpy as jnp,numpy as np; '
        'devices=jax.devices(); assert len(devices)==4 and all(d.platform=="gpu" for d in devices); '
        '[np.testing.assert_allclose(np.asarray(jax.device_put(jnp.ones((8,8)),d)@jax.device_put(jnp.eye(8),d)),1) for d in devices]; '
        'print(devices); import torch; assert torch.cuda.device_count()==4; '
        '[torch.ones(2,device=f"cuda:{i}").sum().item() for i in range(4)]; print("JAX and Torch CUDA smoke passed")'],cpu=False)
    link=Path('/workspace/.hf_home/lerobot/physical-intelligence/libero')
    link.parent.mkdir(parents=True,exist_ok=True)
    target=Path('/workspace/artifacts/datasets/lerobot_libero')
    if link.is_symlink():
        if link.resolve()!=target:raise RuntimeError('Unexpected dataset cache symlink')
    elif link.exists():raise RuntimeError('Dataset cache path already exists and is not the expected symlink')
    else:link.symlink_to(target)
    record('teacher_preflight_complete.json',{'passed':True,'tests':len(cases),'unix_time':time.time()})
    # The HF model and official datasets use pinned revisions, not direct copies.
    wait_for(ROOT/'runtime/public_sources_complete.json','con1_mean_public_downloads')
    wait_for(ROOT/'runtime/hf_5k_complete.json','con1_mean_5k_download')
    wait_for(ROOT/'runtime/teacher_rebuild_complete.json','con1_mean_teacher_rebuild')
    event('checking',check='full_5k_checkpoint_hashes')
    manifest=json.loads((REPO/'ops/mean_5k_manifest.json').read_text())
    expected=manifest['sha256'];actual={}
    for name in expected:
        with (ANCHOR/name).open('rb') as handle:
            actual[name]=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=expected:raise RuntimeError('5k checkpoint checksum mismatch')
    record('checkpoint_5k_sha256.json',{'passed':True,'files':actual,'repo_id':manifest['repo_id'],
                                       'revision':manifest['revision']})
    # Reproduce the original immutable panels from the pinned public benchmark.
    run('fixed_panels',[PYTHON,'-c',
        'import json; from pathlib import Path; from paper_con1_protocol import choose_panels; '
        'from run_paper_con1_eval import atomic_json; '
        'p=Path('+repr(str(ROOT/'panels.json'))+'); '
        'c=json.loads(Path("/workspace/artifacts/benchmarks/LIBERO-plus/libero/libero/benchmark/task_classification.json").read_text()); '
        'v=choose_panels(c); assert not p.exists() or json.loads(p.read_text())==v; atomic_json(p,v)'])
    run('direct_targets',[PYTHON,'ops/audit_direct_con1_restart.py','--output',str(ROOT/'runtime/direct_target_audit.json')])
    run('teacher_targets',[PYTHON,'ops/audit_mean_teacher_inputs.py','--output',str(ROOT/'runtime/teacher_target_audit.json')])
    run('egl_runtime',[str(REPO/'examples/libero/.venv-plus/bin/python'),'-c',
        'import os; os.environ["MUJOCO_GL"]="egl"; os.environ["PYOPENGL_PLATFORM"]="egl"; '
        'import mujoco; m=mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>"); '
        'd=mujoco.MjData(m); mujoco.mj_forward(m,d); r=mujoco.Renderer(m,64,64); r.update_scene(d); '
        'assert r.render().shape==(64,64,3); r.close(); print("EGL renderer passed")'],cpu=False)
    record('input_audit.json',{'passed':True,'unix_time':time.time(),'checkpoint':str(ANCHOR),
                              'checkpoint_sha256_report':str(ROOT/'runtime/checkpoint_5k_sha256.json'),
                              'teacher_target_report':str(ROOT/'runtime/teacher_target_audit.json'),
                              'direct_target_report':str(ROOT/'runtime/direct_target_audit.json'),
                              'gpu_runtime':'4 GPUs, JAX and Torch primitives passed',
                              'render_runtime':'EGL passed',
                              'input_inventory':'Pinned GitHub/HF sources; 33 checkpoint SHA-256 hashes; newly generated teacher caches',
                              'code_report':str(ROOT/'runtime/github_mean_code_verified.json'),
                              'cache_report':str(ROOT/'runtime/teacher_rebuild_complete.json')})
    event('ready_to_train')
    subprocess.run(['supervisorctl','start','con1_mean_experiment'],check=True)
    event('training_orchestrator_started')


if __name__=='__main__':
    try:main()
    except Exception as error:
        event('preparation_failed',error=repr(error))
        raise
