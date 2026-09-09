#!/usr/bin/env python3
"""Rebuild the pinned two-view 8x8 JEPA auxiliary cache without changing labels.

Pooling matches the earlier cache generator: mean on the BF16 teacher output,
then cast to FP16. This is NOT the independent-frame Con1 delta target.
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

import precompute_vjepa_pair_targets as pair
from precompute_vjepa_displacement_targets import ensure_manifest


def contract(args, info, episodes):
    value=pair.run_contract(args, info, episodes)
    keys=['image','wrist_image']
    if any(info['features'][key]['dtype']!='image' for key in keys):
        raise ValueError('This pinned regeneration recipe expects embedded images')
    value.update(image_keys=keys,target_shape=[128,1408],target_grid=[8,8])
    return value


def pool_features(output):
    if output.ndim!=3 or tuple(output.shape[1:])!=(576,1408):
        raise ValueError(f'Unexpected dense teacher output: {output.shape}')
    return output.reshape(output.shape[0],8,3,8,3,1408).mean(dim=(2,4)).reshape(output.shape[0],64,1408)


def process_episode(args, info, episode, model, device):
    import torch
    index,length=int(episode['episode_index']),int(episode['length'])
    path=pair.target_path(args.output_root,index)
    metadata=pair.metadata_path(args.output_root,index)
    shape=(length,128,1408)
    if path.exists():
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.shape!=shape or values.dtype!=np.float16:raise ValueError('Invalid existing pair cache: '+str(path))
        if metadata.exists():return
        raise ValueError('Pair cache lacks committed sidecar: '+str(path))
    table=pq.read_table(pair.input_path(args.dataset_root,index),
                        columns=['image','wrist_image','frame_index','episode_index','index'])
    if table.num_rows!=length or not np.array_equal(table['frame_index'].to_numpy(),np.arange(length)):
        raise ValueError('Noncontiguous episode frames')
    if not np.all(table['episode_index'].to_numpy()==index):raise ValueError('Cross-episode data')
    if not np.array_equal(table['index'].to_numpy(),np.arange(episode['global_start'],episode['global_start']+length)):
        raise ValueError('Incorrect global data cursor')
    images={key:[pair.decode_image(row,args.dataset_root) for row in table[key].to_pylist()]
            for key in ('image','wrist_image')}
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{path.name}.rebuild-{os.getpid()}')
    values=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=shape)
    started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,length,args.batch_size):
            end=min(start+args.batch_size,length)
            future=[min(t+args.future_offset,length-1) for t in range(start,end)]
            for view,key in enumerate(('image','wrist_image')):
                current=np.stack([pair.preprocess_image(images[key][t]) for t in range(start,end)])
                after=np.stack([pair.preprocess_image(images[key][t]) for t in future])
                video=torch.from_numpy(np.stack((current,after),axis=2)).to(device,dtype=torch.bfloat16)
                output=model(video)
                if isinstance(output,list):output=output[-1]
                pooled=pool_features(output)
                if not torch.isfinite(pooled).all():raise ValueError('Non-finite pair teacher targets')
                values[start:end,view*64:(view+1)*64]=pooled.to(device='cpu',dtype=torch.float16).numpy()
    values.flush();del values
    temporary.replace(path)
    value=pair.episode_metadata(contract(args,info,[episode]),episode,path,min(args.future_offset,length))
    value['target_shape']=list(shape)
    pair.write_json_atomic(metadata,value)
    print(f'pair complete episode={index} frames={length} seconds={time.monotonic()-started:.2f}',flush=True)


def main():
    args=pair.parse_args()
    if args.future_offset!=10:raise ValueError('This experiment requires offset 10')
    info=pair.read_json(args.dataset_root/'meta/info.json')
    episodes=pair.read_episodes(args.dataset_root/'meta/episodes.jsonl')
    ensure_manifest(args.output_root,contract(args,info,episodes))
    model,device=pair.load_target_encoder(args)
    assigned=[row for row in episodes if row['episode_index']%args.world_size==args.worker_rank]
    if args.max_episodes is not None:assigned=assigned[:args.max_episodes]
    for episode in assigned:process_episode(args,info,episode,model,device)


if __name__=='__main__':main()
