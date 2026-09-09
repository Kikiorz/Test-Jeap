#!/usr/bin/env python3
"""Wait for seed7 60k Robot completion, then evaluate only official 30k Robot."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from con1_dynamic_queue import collect
from run_paper_con1_eval import REPO, atomic_json
from run_pure_jepawam_robot_sweep import audit_checkpoint, checkpoint, paired, summarize, ROOT as DOWNLOAD_ROOT

ROOT = Path('/workspace/artifacts/research_reports/con/pure_jepawam_30k_robot_seed7_20260909')
PREVIOUS = Path('/workspace/artifacts/research_reports/con/pure_jepawam_60k_robot_seed7_20260909')
CATEGORY = 'Robot Initial States'
CONFIG = 'pi05_libero_paper_reference'


def service_state(name):
    result = subprocess.run(['supervisorctl','status',name],capture_output=True,text=True)
    fields = result.stdout.split()
    if len(fields)<2 or fields[0]!=name:
        raise RuntimeError('Unable to resolve service '+name)
    return fields[1]


def predecessor_complete(status, summary, service):
    if status.get('state')=='error' or service in ('FATAL','BACKOFF','STOPPED','UNKNOWN'):
        raise RuntimeError('60k predecessor failed or was paused; do not start 30k')
    if service!='EXITED':
        return False
    if (status.get('state')!='complete' or status.get('evaluation_seed')!=7
            or status.get('category')!=CATEGORY or not summary.get('complete')
            or summary.get('completed_episodes')!=1550 or summary.get('errors')!=0):
        raise RuntimeError('60k exited without a complete seed7 Robot result')
    return True


def command():
    return [str(REPO/'.venv/bin/python'),'-u',str(REPO/'ops/run_con1_dynamic_eval.py'),
            '--config',CONFIG,'--checkpoint',str(checkpoint('30k')),
            '--panels',str(ROOT/'panels.json'),'--panel','final','--output',str(ROOT/'baseline'),
            '--port-base','8900','--concurrency',str(ROOT/'concurrency.json'),'--category',CATEGORY]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status = dict(pid=os.getpid(),model='pure_jepa_wam_pi05_30k',checkpoint=str(checkpoint('30k')),
                      evaluation_seed=7,category=CATEGORY,expected_episodes=1550,
                      slots_per_gpu=[16]*4,training_enabled=False)
        def event(state,**fields):
            status.update(state=state,unix_time=time.time(),**fields)
            atomic_json(ROOT/'status.json',status)
            print(json.dumps(status),flush=True)
        try:
            for name in ('con1_mean_experiment','pure_jepawam_robot_sweep','pure_jepawam_full_plus_4x16'):
                if service_state(name)!='STOPPED':
                    raise RuntimeError('Unexpected active task: '+name)
            ready=json.loads((DOWNLOAD_ROOT/'30k_ready.json').read_text())
            if not ready.get('all_hashes_verified') or ready['checkpoint']!=str(checkpoint('30k')):
                raise RuntimeError('30k download not verified')
            audit=audit_checkpoint('30k')
            if any(ready.get(k)!=audit[k] for k in ('hub_revision','metadata_sha256','norm_sha256')):
                raise RuntimeError('30k differs from verified download')
            atomic_json(ROOT/'purity_audit.json',dict(audit,download_audit=str(DOWNLOAD_ROOT/'30k_ready.json')))
            while True:
                previous_status=json.loads((PREVIOUS/'status.json').read_text())
                summary_path=PREVIOUS/'baseline/summary_robot_initial_states.json'
                previous_summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
                if predecessor_complete(previous_status,previous_summary,service_state('pure_jepawam_60k_robot_seed7')):
                    break
                event('waiting_for_60k',predecessor=str(PREVIOUS))
                time.sleep(10)
            manifest=json.loads((PREVIOUS/'panels.json').read_text())
            if manifest['final_episode_seed']!=7:
                raise RuntimeError('Wrong seed in predecessor manifest')
            reference,_=collect(PREVIOUS/'baseline',manifest)
            expected={(s,r['task_id'],0) for s,rr in manifest['final'].items() for r in rr if r['category']==CATEGORY}
            if set(reference)!=expected or len(reference)!=1550 or any(r['status']=='error' for r in reference.values()):
                raise RuntimeError('Incomplete reference journals')
            panels=ROOT/'panels.json'
            if panels.exists() and json.loads(panels.read_text())!=manifest:
                raise RuntimeError('Refusing to change an existing task/seed contract')
            atomic_json(panels,manifest)
            atomic_json(ROOT/'concurrency.json',dict(slots_per_gpu=[16]*4,trial=None))
            from openpi.training.config import get_config
            model=get_config(CONFIG).model
            if any(getattr(model,k) for k in ('use_rapr','use_point_flow','use_action_change_mmdit','use_jepa_ttt_adapter')):
                raise RuntimeError('Pure model unexpectedly enables Con1')
            with (ROOT/'evaluation.log').open('a') as log:
                child=subprocess.Popen(command(),cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                while True:
                    event('running',child_pid=child.pid)
                    try:
                        result=child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            if result:
                raise RuntimeError(f'30k evaluator exited {result}; journals retained')
            report=json.loads((ROOT/'baseline/summary_robot_initial_states.json').read_text())
            if not report['complete'] or report['completed_episodes']!=1550 or report['errors']:
                raise RuntimeError('Incomplete 30k Robot evaluation')
            rows,_=collect(ROOT/'baseline',manifest)
            comparison=dict(seed=7,category=CATEGORY,reference_60k=summarize(reference),
                            candidate_30k=summarize(rows),paired_30k_vs_60k=paired(rows,reference))
            atomic_json(ROOT/'comparison_30k_vs_60k_seed7.json',comparison)
            event('complete',child_pid=None,successes=report['successes'])
        except Exception as error:
            event('error',error=repr(error))
            raise


if __name__=='__main__':
    main()
