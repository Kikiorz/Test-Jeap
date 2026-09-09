#!/usr/bin/env python3
"""One GPU, one frozen encoder, both distinct teacher caches; no training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import precompute_con1_frame_states as frames
import precompute_vjepa_multiview_targets as views
import precompute_vjepa_pair_targets as pair
from precompute_vjepa_displacement_targets import ensure_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--max-episodes',type=int)
    cli=parser.parse_args()
    if cli.rank not in range(4):raise ValueError('Requires rank 0..3')
    common=dict(dataset_root=Path('/workspace/artifacts/datasets/lerobot_libero'),
        checkpoint=Path('/workspace/vjepa2/vjepa2_1_vitg_384.pt'),vjepa_source_root=Path('/workspace/vjepa2'),
        image_key='image',image_keys=['image','wrist_image'],future_offset=10,
        batch_size=8,device='cuda:0',min_free_gib=20,worker_rank=cli.rank,world_size=4)
    frame_args=argparse.Namespace(**common,output_root=Path('/workspace/artifacts/con1/orthogonal_all4_frame_states_v1'))
    pair_args=argparse.Namespace(**common,output_root=Path('/workspace/artifacts/vjepa_targets/libero_vjepa2_1_vitg_384_offset10_base_wrist'))
    info=pair.read_json(frame_args.dataset_root/'meta/info.json')
    episodes=pair.read_episodes(frame_args.dataset_root/'meta/episodes.jsonl')
    if len(episodes)!=1693 or sum(row['length'] for row in episodes)!=273465:raise ValueError('Wrong complete dataset')
    ensure_manifest(frame_args.output_root,frames.make_contract(frame_args,info,episodes))
    ensure_manifest(pair_args.output_root,views.contract(pair_args,info,episodes))
    model,device=pair.load_target_encoder(frame_args)
    assigned=[row for row in episodes if int(row['episode_index'])%4==cli.rank]
    if cli.max_episodes is not None:assigned=assigned[:cli.max_episodes]
    for row in assigned:
        frames.process_episode(frame_args,info,row,model,device)
        views.process_episode(pair_args,info,row,model,device)
    print(json.dumps({'event':'worker_done','rank':cli.rank,'episodes':len(assigned)}),flush=True)


if __name__=='__main__':main()
