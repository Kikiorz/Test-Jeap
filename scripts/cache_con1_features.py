#!/usr/bin/env python3
"""Current-only official JEPA-WAM R + verified independent teacher z cache.

Supervisor should run the coordinator. Workers shard whole episodes, never
frames. Commits are atomic per episode; incomplete workers cannot publish a
complete manifest. No optimizer or training is invoked by this program.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def atomic_npy(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--worker', type=int)
    p.add_argument('--base-config', default='pi05_libero_vjepa_aux')
    p.add_argument('--num-queries', type=int, default=64)
    p.add_argument('--latent-dim', type=int, default=2816)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--image-keys', nargs='+', default=['image', 'wrist_image'])
    return p.parse_args()


def episodes(args):
    return [json.loads(line) for line in (args.dataset/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]


def source_identity(root):
    h = hashlib.sha256()
    for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl'):
        h.update(name.encode())
        h.update((root/'meta'/name).read_bytes())
    return h.hexdigest()


def paths(args, eid):
    root = args.output/'episodes'
    return root/f'{eid:06d}_r.npy', root/f'{eid:06d}_z.npy', root/f'{eid:06d}.json'


def validate_sources(args):
    rows = episodes(args)
    m = json.loads((args.states/'manifest.json').read_text())
    if (m['kind'] != 'con1_independent_vjepa_frame_states' or m['state_dim'] != args.latent_dim
            or m['channel_projection'] != 'none'
            or m['temporal_input'] != '[o_k,o_k]; never [o_t,o_future]'
            or m['dataset_metadata_sha256'] != source_identity(args.dataset)
            or m['dataset_total_episodes'] != len(rows)
            or m['dataset_total_frames'] != sum(e['length'] for e in rows)):
        raise ValueError('Teacher state/dataset contract mismatch')
    # Old manifest describes its OLD downstream orthogonal loader, not raw
    # stored states. These arrays must be non-normalized raw Phi(o_k).
    for e in rows:
        eid = e['episode_index']
        p = args.states/'states'/f'chunk-{eid//m["chunks_size"]:03d}'/f'episode_{eid:06d}.npy'
        record = json.loads(p.with_suffix('.json').read_text())
        z = np.load(p, mmap_mode='r', allow_pickle=False)
        if (record['event'] != 'complete' or record['episode'] != eid
                or record['frames'] != e['length'] or z.shape != (e['length'],args.latent_dim)
                or z.dtype != np.float16 or not np.isfinite(z).all()):
            raise ValueError(f'Uncommitted/invalid raw teacher states: {eid}')
    return rows, m


def worker(args):
    import jax
    import jax.numpy as jnp
    import pyarrow.parquet as pq
    from precompute_vjepa_pair_targets import decode_image
    from openpi.models import model as model_lib
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config

    contract = json.loads((args.output/'contract.json').read_text())
    rows = episodes(args)
    tasks = {e['task_index']:e['task'] for e in map(json.loads,(args.dataset/'meta/tasks.jsonl').read_text().splitlines())}
    base = config.get_config(args.base_config)
    base = dataclasses.replace(base, model=dataclasses.replace(base.model, vjepa_target_grid_size=8))
    policy = policy_config.create_trained_policy(base, args.checkpoint)
    extract = nnx_utils.module_jit(policy._model.extract_predictive_tokens)
    state_manifest = json.loads((args.states/'manifest.json').read_text())
    count = frames = 0
    started = time.time()
    status_path = args.output/f'worker_{args.worker}.json'
    atomic_json(status_path, dict(state='loaded', pid=os.getpid(), completed=0, frames=0, unix_time=time.time()))
    for e in rows[args.worker::len(args.gpus.split(','))]:
        eid, length = e['episode_index'], e['length']
        rp,zp,commit = paths(args,eid)
        if commit.exists():
            saved = json.loads(commit.read_text())
            if saved['contract_sha256'] != digest(args.output/'contract.json') or saved['r_sha256'] != digest(rp) or saved['z_sha256'] != digest(zp):
                raise ValueError(f'Existing episode changed: {eid}')
            count += 1; frames += length
            continue
        chunk = eid // int(json.loads((args.dataset/'meta/info.json').read_text()).get('chunks_size',1000))
        source = args.dataset/'data'/f'chunk-{chunk:03d}'/f'episode_{eid:06d}.parquet'
        schema_names = pq.read_schema(source).names
        video_layout = any(key not in schema_names for key in args.image_keys)
        if video_layout:
            # Frames live in one mp4 per episode (LeRobot v2.1 video layout), so
            # only the tabular columns come from the parquet.
            table = pq.read_table(source, columns=['state','frame_index','episode_index','task_index'])
            from precompute_vjepa_pair_targets import decode_video
            decoded = {}
            for key in args.image_keys:
                video_path = (args.dataset/'videos'/f'chunk-{chunk:03d}'/key/f'episode_{eid:06d}.mp4')
                decoded[key] = decode_video(video_path, length)
        else:
            table = pq.read_table(source, columns=[*args.image_keys,'state','frame_index','episode_index','task_index'])
        samples = table.to_pylist()
        if len(samples) != length or any(row['episode_index'] != eid or row['frame_index'] != i for i,row in enumerate(samples)):
            raise ValueError('Parquet frame ordering/episode mismatch')
        task_ids = {row['task_index'] for row in samples}
        if len(task_ids) != 1:
            raise ValueError('Task changes within episode')
        result = []
        for offset in range(0, length, args.batch_size):
            batch = samples[offset:offset+args.batch_size]
            transformed = [policy._input_transform({
                **{f'observation/{key}':(np.asarray(decoded[key][offset+index])
                                         if video_layout else np.asarray(decode_image(row[key],args.dataset)))
                   for key in args.image_keys},
                'observation/state':np.asarray(row['state'],dtype=np.float32),
                'prompt':tasks[row['task_index']],
            }) for index, row in enumerate(batch)]
            valid = len(transformed)
            transformed += [transformed[-1]] * (args.batch_size-valid)
            values = jax.tree.map(lambda *xs:jnp.asarray(np.stack(xs)), *transformed)
            observation = model_lib.Observation.from_dict(values)
            # Explicitly strip optional targets: prefix inference is current-only.
            observation = dataclasses.replace(observation, vjepa_target=None)
            prediction = np.asarray(extract(observation))[:valid]
            if prediction.shape[1:] != (args.num_queries,2048) or not np.isfinite(prediction).all():
                raise ValueError(f'Bad R shape/values: {prediction.shape}')
            result.append(prediction.astype(np.float16))
            atomic_json(status_path, dict(state='extracting', pid=os.getpid(), completed=count,
                frames=frames, current_episode=eid, current_frame=offset+valid, unix_time=time.time(),
                seconds=time.time()-started))
        r = np.concatenate(result)
        source_z = args.states/'states'/f'chunk-{eid//state_manifest["chunks_size"]:03d}'/f'episode_{eid:06d}.npy'
        z = np.load(source_z, allow_pickle=False)
        if z.shape != (length,args.latent_dim) or not np.isfinite(z).all():
            raise ValueError('Teacher source changed')
        atomic_npy(rp,r)
        atomic_npy(zp,z)
        record = dict(id=eid,length=length,task_id=next(iter(task_ids)),
            r=str(rp.relative_to(args.output)), z=str(zp.relative_to(args.output)),
            r_sha256=digest(rp),z_sha256=digest(zp), source_z_sha256=digest(source_z),
            source_parquet_sha256=digest(source),contract_sha256=digest(args.output/'contract.json'))
        if record['z_sha256'] != record['source_z_sha256']:
            raise ValueError('Raw teacher latent copy differs')
        atomic_json(commit,record)
        count += 1; frames += length
        print(json.dumps(dict(event='episode_complete',worker=args.worker,**record)),flush=True)
    atomic_json(status_path,dict(state='complete',pid=os.getpid(),completed=count,frames=frames,unix_time=time.time()))


def coordinator(args):
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        (args.output/'episodes').mkdir(exist_ok=True)
        (args.output/'logs').mkdir(exist_ok=True)
        children=[]; handles=[]
        status=args.output/'status.json'
        try:
            atomic_json(status,dict(state='validating_sources',pid=os.getpid(),unix_time=time.time()))
            rows, teacher = validate_sources(args)
            params_meta=json.loads((args.checkpoint/'params/_METADATA').read_text())
            leaves=params_meta['tree_metadata']
            if len(leaves)!=58 or any(any(bad in key.lower() for bad in ('rapr','con1','change_','adapter')) for key in leaves):
                raise ValueError('Not a pure 58-leaf JEPA-WAM checkpoint')
            checkpoint_hashes={str(p.relative_to(args.checkpoint)):digest(p)
                for p in sorted(args.checkpoint.rglob('*')) if p.is_file() and '.cache' not in p.parts}
            # Pin the source file tree, including the one read-only R accessor.
            code_hashes={str(p.relative_to(REPO)):digest(p) for p in sorted((REPO/'src/openpi').rglob('*.py'))}
            contract=dict(schema='con1-anchored-feature-contract-v1', checkpoint=str(args.checkpoint),
                checkpoint_sha256=checkpoint_hashes, teacher_manifest=teacher,
                teacher_manifest_sha256=digest(args.states/'manifest.json'),
                dataset_metadata_sha256=source_identity(args.dataset),code_sha256=code_hashes,
                r_definition=f'official frozen JEPA-WAM prefix last{args.num_queries}, current images/prompt; no action suffix',
                r_shape=[args.num_queries,2048],r_dtype='float16',latent_dim=args.latent_dim,
                anchor_source='current_only_frozen_teacher',horizon=args.horizon,
                batch_size=args.batch_size,gpus=args.gpus,training_enabled=False,
                teacher_sha256=digest(Path(teacher['checkpoint'])))
            existing=args.output/'contract.json'
            if existing.exists() and json.loads(existing.read_text())!=contract:
                raise ValueError('Refusing to mix cache contracts')
            atomic_json(existing,contract)
            required=sum(e['length'] for e in rows)*(args.num_queries*2048+args.latent_dim)*2
            if shutil.disk_usage(args.output).free < required+10*1024**3:
                raise ValueError('Insufficient free disk for full uncompressed cache')
            atomic_json(args.output/'manifest.json',dict(schema='con1-anchored-features-v1',complete=False,
                anchor_source='current_only_frozen_teacher',r_shape=[args.num_queries,2048],
                latent_dim=args.latent_dim,episodes=[]))
            for rank,gpu in enumerate(args.gpus.split(',')):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONPATH=f'{REPO}/src:{REPO}/packages/openpi-client/src',
                    HF_HUB_OFFLINE='1',HF_HOME='/workspace/.hf_home')
                log=(args.output/'logs'/f'worker_{rank}.log').open('a');handles.append(log)
                child=subprocess.Popen([sys.executable,'-u',__file__,*sys.argv[1:],'--worker',str(rank)],
                    env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
            while True:
                if any(c.poll() not in (None,0) for c in children):
                    raise RuntimeError('Feature worker failed; inspect worker logs')
                records=[json.loads(p.read_text()) for p in (args.output/'episodes').glob('*.json')]
                atomic_json(status,dict(state='extracting',pid=os.getpid(),worker_pids=[c.pid for c in children],
                    completed=len(records),frames=sum(r['length'] for r in records),expected=len(rows),
                    unix_time=time.time(),training_enabled=False))
                if all(c.poll()==0 for c in children):break
                time.sleep(10)
            if {r['id'] for r in records}!={e['episode_index'] for e in rows}:
                raise ValueError('Incomplete episode coverage')
            for r in records:
                if digest(args.output/r['r'])!=r['r_sha256'] or digest(args.output/r['z'])!=r['z_sha256']:
                    raise ValueError('Final cache checksum mismatch')
            atomic_json(args.output/'manifest.json',dict(schema='con1-anchored-features-v1',complete=True,
                anchor_source='current_only_frozen_teacher',r_shape=[args.num_queries,2048],
                latent_dim=args.latent_dim,horizon=args.horizon,
                contract_sha256=digest(existing),episodes=sorted(records,key=lambda e:e['id'])))
            atomic_json(status,dict(state='complete',completed=len(records),frames=sum(r['length'] for r in records),
                unix_time=time.time(),training_enabled=False))
        except Exception as exc:
            atomic_json(status,dict(state='error',error=repr(exc),unix_time=time.time(),training_enabled=False))
            raise
        finally:
            for c in children:
                if c.poll() is None:c.terminate()
            for c in children:
                try:c.wait(timeout=20)
                except subprocess.TimeoutExpired:c.kill();c.wait()
            for f in handles:f.close()


if __name__=='__main__':
    args=parse()
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    worker(args) if args.worker is not None else coordinator(args)
