"""Four-GPU head-only training; official 40k and teacher are offline inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.con1.data import FeatureDataset, batch
from openpi.con1.modules import AnchoredDeltaHead, anchored_loss


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def write_json(path,data):
    atomic(path,(json.dumps(data,indent=2,sort_keys=True)+'\n').encode())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--horizon',type=int,default=10,choices=[10])
    p.add_argument('--latent-dim',type=int,default=2816)
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--learning-rate',type=float,default=5e-5)
    p.add_argument('--delta-weight',type=float,default=1.)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1 or args.learning_rate<=0:raise ValueError('Invalid training settings')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'train.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.output/'run.json').exists() and not args.resume:raise ValueError('Existing run: use --resume')
    started=time.time()
    def status(state,**extra):
        write_json(args.output/'status.json',dict(state=state,unix_time=time.time(),
            elapsed_seconds=time.time()-started,base_parameters_updated=False,**extra))
    status('initializing')
    try:
        train=FeatureDataset(args.cache,horizon=10,split='train',seed=args.seed)
        valid=FeatureDataset(args.cache,horizon=10,split='validation',seed=args.seed)
        provenance=json.loads((args.cache/'contract.json').read_text())
        if not provenance['checkpoint'].endswith('/40000'):raise ValueError('Official 40k cache required')
        if hashlib.sha256((args.cache/'contract.json').read_bytes()).hexdigest()!=train.manifest['contract_sha256']:
            raise ValueError('Cache identity changed')
        devices=jax.local_devices();nd=len(devices)
        if args.batch_size%nd:raise ValueError('Global batch must divide device count')
        model=AnchoredDeltaHead(10,args.latent_dim,args.width)
        params=model.init(jax.random.key(args.seed),jnp.zeros((1,*train.manifest['r_shape'])),
                          jnp.zeros((1,args.latent_dim)))['params']
        lr=optax.linear_schedule(0.,args.learning_rate,100)
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(lr,weight_decay=1e-4))
        opt_state=tx.init(params)
        contract=dict(schema='con1-anchored-head-v2',cache_identity=train.identity,
            checkpoint=provenance['checkpoint'],seed=args.seed,horizon=10,width=args.width,
            latent_dim=args.latent_dim,delta_weight=args.delta_weight,learning_rate=args.learning_rate,
            batch_size=args.batch_size,devices=nd,train_episodes=[e['id'] for e in train.episodes],
            validation_episodes=[e['id'] for e in valid.episodes],train_frames=len(train),validation_frames=len(valid),
            parameter_count=sum(x.size for x in jax.tree.leaves(params)),
            optimized_modules=['AnchoredDeltaHead'],base_parameters_updated=False,
            reduction='mean_feature_then_mean_valid_positions',attention_heads=4,
            architecture='anchor_MLP_chunk_expand_cross_attention_FFN_shared_delta_projection')
        step0=0
        if args.resume:
            if json.loads((args.output/'run.json').read_text())!=contract:raise ValueError('Resume contract mismatch')
            latest=json.loads((args.output/'latest.json').read_text())
            restored=serialization.from_bytes({'params':params,'opt_state':opt_state,'step':0},
                (args.output/latest['file']).read_bytes())
            params,opt_state,step0=restored['params'],restored['opt_state'],int(restored['step'])
        else:write_json(args.output/'run.json',contract)
        def loss_fn(params,values):
            out=model.apply({'params':params},values['r_tokens'],values['anchor'])
            return anchored_loss(out['delta'],values['anchor'],values['future_target'],values['valid'],
                                 delta_weight=args.delta_weight)
        def update(params,opt_state,values):
            (_,metrics),grad=jax.value_and_grad(loss_fn,has_aux=True)(params,values)
            count=metrics['valid_count'].astype(jnp.float32)
            total=jax.lax.psum(count,'devices')
            # Exact global masked mean, even for different tail masks per GPU.
            grad=jax.tree.map(lambda x:jax.lax.psum(x*count,'devices')/jnp.maximum(total,1),grad)
            reduced={k:jax.lax.psum(v*count,'devices')/jnp.maximum(total,1)
                     for k,v in metrics.items() if k not in ('valid_count','delta_nmse')}
            reduced['valid_count']=total
            reduced['delta_nmse']=reduced['delta_mse']/jnp.maximum(reduced['copy_current_mse'],1e-12)
            reduced['grad_norm']=optax.global_norm(grad)
            updates,opt_state=tx.update(grad,opt_state,params)
            return optax.apply_updates(params,updates),opt_state,reduced
        update=jax.pmap(update,axis_name='devices')
        evaluate=jax.jit(lambda p,b:loss_fn(p,b)[1])
        params=jax.device_put_replicated(params,devices)
        opt_state=jax.device_put_replicated(opt_state,devices)
        unrep=lambda tree:jax.tree.map(lambda x:np.asarray(x[0]),tree)
        def save(step):
            filename=f'checkpoint_{step:06d}.msgpack'
            atomic(args.output/filename,serialization.to_bytes(dict(params=unrep(params),opt_state=unrep(opt_state),step=step)))
            write_json(args.output/'latest.json',dict(step=step,file=filename))
        val_indices=[];offset=0
        for e in valid.episodes:
            n=e['length']-1
            val_indices.extend((offset+np.linspace(0,n-1,min(16,n),dtype=int)).tolist());offset+=n
        def validation(step,full=False):
            indices=np.arange(len(valid)) if full else np.array(val_indices)
            weighted={};count=0;weights=unrep(params)
            for off in range(0,len(indices),args.batch_size):
                ids=indices[off:off+args.batch_size];b=batch(valid,ids)
                padding=args.batch_size-len(ids)
                if padding:
                    b={k:np.concatenate([v,np.repeat(v[-1:],padding,axis=0)]) for k,v in b.items()}
                    b['valid'][-padding:]=False
                metrics=evaluate(weights,{k:jnp.asarray(v) for k,v in b.items()})
                n=float(metrics['valid_count']);count+=n
                for k in ('delta_mse','copy_current_mse','loss'):weighted[k]=weighted.get(k,0)+float(metrics[k])*n
            record={k:v/max(count,1) for k,v in weighted.items()}
            record.update(step=step,scope='full_holdout' if full else 'fixed_monitor',frames=len(indices),
                          delta_nmse=record['delta_mse']/max(record['copy_current_mse'],1e-12))
            write_json(args.output/f'validation_{step:06d}{"_full" if full else ""}.json',record)
            print(json.dumps({'validation':record}),flush=True)
            return record
        save(step0);initial=validation(step0)
        epoch_order=None;epoch_id=-1
        for step in range(step0,args.steps):
            ids=[]
            for position in range(step*args.batch_size,(step+1)*args.batch_size):
                ep=position//len(train)
                if ep!=epoch_id:
                    epoch_order=np.random.default_rng(np.random.SeedSequence([args.seed,ep])).permutation(len(train))
                    epoch_id=ep
                ids.append(epoch_order[position%len(train)])
            before=time.time();b=batch(train,ids)
            b={k:jnp.asarray(v.reshape(nd,args.batch_size//nd,*v.shape[1:])) for k,v in b.items()}
            params,opt_state,metrics=update(params,opt_state,b)
            record={k:float(v[0]) for k,v in metrics.items()}
            if not all(np.isfinite(v) for v in record.values()):raise FloatingPointError('Nonfinite metric')
            if step==step0 or (step+1)%10==0:
                record.update(step=step+1,step_seconds=time.time()-before,unix_time=time.time())
                with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                status('training',step=step+1,target_steps=args.steps,metrics=record)
            if step==step0 or (step+1)%1000==0 or step+1==args.steps:
                save(step+1)
                if step!=step0:validation(step+1)
        final=validation(args.steps,full=True)
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(unrep(params))):raise FloatingPointError('Nonfinite parameters')
        status('complete',step=args.steps,validation=final,initial_monitor=initial,
               interpretation='Prediction fit only, not proof of policy or TTT benefit')
    except Exception as exc:status('error',error=repr(exc));raise


if __name__=='__main__':main()
