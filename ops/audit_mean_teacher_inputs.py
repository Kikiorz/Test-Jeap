#!/usr/bin/env python3
"""Read-only shape/finite-row audit of every episode's unchanged teacher target."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_paper_con1_eval import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path('/workspace/artifacts/vjepa_targets/libero_vjepa2_1_vitg_384_offset10_base_wrist')
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['target_shape']==[128,1408] and manifest['future_offset']==10
    episodes=[json.loads(line) for line in Path('/workspace/artifacts/datasets/lerobot_libero/meta/episodes.jsonl').read_text().splitlines()]
    checked=0
    for row in episodes:
        index=int(row['episode_index']);length=int(row['length'])
        prefix=root/'targets'/f'chunk-{index//1000:03d}'/f'episode_{index:06d}'
        info=json.loads(prefix.with_suffix('.json').read_text())
        assert info['episode_index']==index and info['episode_length']==length
        assert info['future_offset']==10
        values=np.load(prefix.with_suffix('.npy'),mmap_mode='r',allow_pickle=False)
        assert values.shape==(length,128,1408) and values.dtype==np.float16
        assert np.isfinite(values[[0,length//2,length-1]]).all()
        checked+=1
    assert checked==manifest['dataset_total_episodes']==1693
    atomic_json(args.output,{'passed':True,'checked_episodes':checked,'shape':[128,1408],
                             'future_offset':10,'scope':'Every episode header/identity; first/middle/last rows finite. Not a full-array finite scan.'})


if __name__=='__main__':main()
