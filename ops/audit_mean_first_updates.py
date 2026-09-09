#!/usr/bin/env python3
"""CPU-only parameter check after the first three real mean-branch updates."""
import argparse
import json
import os
from pathlib import Path
import re

import flax.traverse_util
import numpy as np

from openpi.models.model import restore_params
from openpi.training.config import paper_con1_mean_from5k_config
from run_paper_con1_eval import atomic_json


def main():
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('Use the CPU for this audit')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=paper_con1_mean_from5k_config()
    before=flax.traverse_util.flatten_dict(restore_params(config.weight_loader.params_path,restore_type=np.ndarray),sep='/')
    after=flax.traverse_util.flatten_dict(restore_params(args.checkpoint/'params',restore_type=np.ndarray),sep='/')
    if before.keys()!=after.keys():raise RuntimeError('Checkpoint parameter leaves changed')
    action=re.compile(r'.*(llm.*_1|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj|action_time_mlp_in|action_time_mlp_out).*')
    violations=[];late_changed=0;con1_changed=0;dtype_changes={}
    for name,old in before.items():
        new=after[name]
        if old is None or new is None:
            if old is not None or new is not None:violations.append(name)
            continue
        if old.shape!=new.shape or not np.isfinite(new).all():
            violations.append(name);continue
        if old.dtype!=new.dtype:dtype_changes[name]=[str(old.dtype),str(new.dtype)]
        if name.startswith(('rapr_delta_head/','rapr_router/')):
            con1_changed+=int(np.count_nonzero(old!=new))
        elif action.fullmatch(name):
            if '/layers/' in name:
                if old.shape[0]!=18:raise RuntimeError('Unexpected Action depth')
                if not np.array_equal(old[:16],new[:16]):violations.append(name)
                late_changed+=int(np.count_nonzero(old[16:]!=new[16:]))
            elif not np.array_equal(old,new):violations.append(name)
        elif not np.array_equal(old,new):violations.append(name)
    result={'passed':not violations and late_changed>0 and con1_changed>0,
            'checkpoint':str(args.checkpoint),'initial_checkpoint':config.weight_loader.params_path,
            'frozen_parameter_violations':violations,'late_action_changed_elements':late_changed,
            'con1_changed_elements':con1_changed,'dtype_changes':dtype_changes,
            'note':'Values of frozen weights must match 5k exactly. Lossless bf16->fp32 promotion at the new phase is recorded, not counted as training.'}
    atomic_json(args.output,result)
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('First-update freeze audit failed')


if __name__=='__main__':main()
