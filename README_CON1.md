# Clean Con1 (current-anchored latent delta)

This checkout is based on the author's JEPA-WAM π0.5 code at commit
`cdf8413f59af3e59fea8e899f90c3d6bc8f9125f`, with one isolated Con1 design.
It intentionally contains no historical RAPR, binary gate, orthogonal-delta,
mean-loss, Con2/Con3, or evaluation-queue code.

## Definition

The frozen teacher produces the current latent `z_t = Phi(o_t)` and future
labels `z*_{t+j}`.  A head reads current-only predictive tokens `R_t` and the
current anchor, then emits a delta:

```
delta_hat[1:H] = DeltaHead(R_t, z_t)
z_hat[t+j] = z_t + delta_hat[j]
```

The head uses action-free fixed sinusoidal horizon slots; it has no learnable
`Q0`.  Action-conditioned retrieval uses only `Q = Wq LayerNorm(H_A)` and a
zero-initialized continuous residual with `alpha=0.05`.

The reconstruction and delta losses are algebraically identical when the
anchor is the teacher's `z_t*`: `L_recon + lambda L_delta` is merely
`(1+lambda)L_delta`.  The implementation documents this instead of claiming
two independent signals.  `mean` reduction is the default to avoid a hidden
2816x loss-scale change; `sum` is available explicitly in the module.

## Head-only training

### Generate the offline feature cache first

`scripts/cache_con1_features.py` uses the official pure **40k (`40000`)**
checkpoint in the provided launcher, not an old Con1 checkpoint. It extracts
the last 64 prefix tokens on current observations, with no action suffix or
future labels. The input transform and frozen model are from the pinned author
checkout. A read-only accessor was added; the policy sampling code is unchanged.

Existing independent V-JEPA `[o_k,o_k]` raw frame states can be reused after
checking dataset identity, dimensions, finite values and episode commit records.
The legacy state-cache manifest describes an orthogonal downstream loader; the
stored arrays are raw latents, copied without normalization or projection.
The large future-pair auxiliary cache is NOT used as R or current anchor.

```bash
PYTHONPATH=src:packages/openpi-client/src python scripts/cache_con1_features.py \
  --dataset /path/to/lerobot_libero --states /path/to/independent_frame_states \
  --checkpoint /path/to/official/checkpoint/40000 \
  --output /path/to/anchored_40k_features_v1 --gpus 0,1,2,3 --batch-size 8
```

The coordinator records checkpoint/source/teacher SHA-256 hashes, runs one worker
per GPU on disjoint episodes, and commits `[T,64,2048]` R and `[T,2816]` z files.
The final manifest is marked complete only after all episodes and hashes pass.
No training starts automatically. `scripts/con1-feature-cache.conf` is an example
Supervisor configuration; replace its paths/python environment for a new server.
The Python environment's directory name does not select the checkpoint.

Prepare a completed cache with manifest schema `con1-anchored-features-v1`:

```
R_t       # current-observation predictive tokens only, [T,N,E]
z_t       # current-observation frozen teacher latent, [T,D]
z_{t+j}*  # labels only; never supplied as head input
```

Then train only the new head (the base model has no optimizer parameters):

```bash
PYTHONPATH=src python -m openpi.con1.train_head \
  --cache /path/to/cache --output runs/head_5k \
  --steps 5000 --latent-dim 2816 --batch-size 128
```

The cache enforces episode-local tail masks and a deterministic episode-level
train/validation split.  `manifest.json` records the cache identity and
`base_parameters_updated: false`.  This phase is a diagnostic of whether the
latent interface is learnable; it is not yet a closed-loop policy result.

## Migration and compatibility

This head checkpoint is intentionally **not compatible** with old Con1/RAPR
checkpoints.  The original π0.5 checkpoint remains the only base checkpoint;
future integration must load the head explicitly and verify this schema.
