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
