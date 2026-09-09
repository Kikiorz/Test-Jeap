import dataclasses
import json
from pathlib import Path

import run_mean_con1_experiment as module
from openpi.training.config import paper_con1_mean_from5k_config


def test_only_joint_training_and_three_milestones(tmp_path, monkeypatch):
    root=tmp_path/'report';root.mkdir()
    anchor=tmp_path/'stage1/4999/params';anchor.mkdir(parents=True)
    (anchor/'_METADATA').write_text('{}')
    (anchor.parent/'_CHECKPOINT_METADATA').write_text('{}')
    for name in ('inputs.json','tests.json'):
        (root/name).write_text(json.dumps({'passed':True}))
    (root/'panels.json').write_text(json.dumps({'monitor':{'suite':[{}]*84}}))
    (root/'runtime').mkdir()
    (root/'runtime/first_updates_audit.json').write_text(json.dumps({'passed':True}))
    base=paper_con1_mean_from5k_config()
    config=dataclasses.replace(base, checkpoint_base_dir=str(tmp_path/'checkpoints'),
                               weight_loader=dataclasses.replace(base.weight_loader,params_path=str(anchor)),
                               rapr_action_freeze_anchor=str(anchor))
    experiment=module.MeanFrom5kExperiment(root)
    monkeypatch.setattr(experiment,'config',lambda **kw: config)
    monkeypatch.setattr(experiment,'event',lambda *a,**kw:None)
    monkeypatch.setattr(module,'audit',lambda *a:{'complete':True})
    monkeypatch.setattr(module,'latest_checkpoint',lambda *a:-1)
    monkeypatch.setattr(module,'paired_results',lambda *a:{'complete':True})
    calls=[]
    monkeypatch.setattr(experiment,'run',lambda name,args:calls.append(('train',name,args)))
    monkeypatch.setattr(experiment,'validation',lambda name,stage,ckpt:calls.append(('validation',name,stage)))
    def evaluate(name,*args):
        calls.append(('evaluate',name,args[-1]))
        folder=root/'evaluations'/name;folder.mkdir(parents=True,exist_ok=True)
        return folder
    monkeypatch.setattr(experiment,'evaluation',evaluate)
    experiment.execute_branch(input_report=root/'inputs.json',test_report=root/'tests.json')
    trains=[c for c in calls if c[0]=='train']
    assert [c[1] for c in trains]==['train_stage2_5000','train_stage2_10000','train_stage2_15000']
    assert all('--mean-from5k' in c[2] for c in trains)
    assert [c[1] for c in calls if c[0]=='validation']==['stage2_5000','stage2_10000','stage2_15000']
    assert [c[1] for c in calls if c[0]=='evaluate']==[
        'baseline_monitor','stage2_5000','baseline_monitor','stage2_10000','baseline_final','candidate_final']
    assert (root/'final_paired.json').exists()
