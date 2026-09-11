# Con1 accuracy: learned latent metric (Con2's F) + gradient balancing

Goal: raise **Con1's action accuracy** without losing the "the policy predicts
the true future latent" property. Two measurements motivate the change and two
invariants constrain it.

## Why (measured, 192 held-out samples)

| quantity | value |
|---|---:|
| cos(latent-MSE gradient, action-flow gradient) on `delta_z` | **-0.006** |
| flow reduction at equal `delta_z` step: action direction | **+0.97%** |
| flow reduction at equal `delta_z` step: latent-MSE direction | **-0.03%** |
| latent-vs-action gradient magnitude on the head (training normalisation) | **695x** |

So the current latent term is (a) pointed orthogonally to what improves the
action and (b) ~700x louder than the action term, which with the configured
0.2-vs-1.0 weights leaves the head trained almost entirely by an
action-irrelevant objective. Re-weighting cannot fix either problem: a weight in
`[0, 1]` cannot close a 700x gap and cannot rotate a direction. That is the gap
this change fills.

## What was added (all opt-in, all identity at step 0)

| piece | what it does | invariant |
|---|---|---|
| `DeltaMetric` | learnable diagonal metric `m = 1 + 0.5 tanh(theta)` on the residual | `theta = 0` -> exactly the Euclidean loss; `m >= 0.5` so the metric stays positive definite and its only zero is still `delta = delta*` |
| `metric_delta_loss` | the same masked, horizon-weighted objective under that metric | identical value to the current loss at initialisation |
| `direction_alignment_loss` | trains `theta` so `grad F` points along the action direction (`g_action` VJP) | scale-free (cosine), so it cannot be gamed by shrinking magnitudes |
| `balanced_latent_weight` | detached factor that makes the latent and flow gradient magnitudes comparable | `strength = 0` reproduces the configured weights exactly |

The target never changes: `delta_z*` is still the teacher's true future latent
delta, so the *solution set* is unchanged and Z keeps its semantics. Only the
weighting of residual directions moves.

Parameterisation traps found while implementing this (both would have silently
frozen the metric at the Euclidean identity):

1. `M = I + U U^T` is **quadratic** in `U` -> `dF/dU = 0` at `U = 0`.
2. `A = p q^T` is **bilinear** in `(p, q)` -> an exact saddle at the origin.

The shipped form is linear in `theta`, which is what makes it trainable; a
test asserts a non-zero gradient there.

## How to run

```bash
# 4 GPUs, stage 1 of the three-stage schedule (2k steps)
supervisorctl start con1_metric_stage1
# or directly
python -u scripts/train.py pi05_libero_con1_metric_40k \
  --exp-name=con1_metric_stage1_2k --num-train-steps=2000 --batch-size=64 \
  --con1-lr-multiplier=2 --no-wandb-enabled --num-workers=0
```

`pi05_libero_con1_metric_40k` = the deployed adapter config plus
`con1_metric=True`, `con1_metric_align_weight=0.1`, `con1_balance_strength=1.0`.

## Acceptance criteria (all four, or the change is rejected)

| criterion | how | requirement |
|---|---|---|
| semantic retention | held-out Euclidean latent NMSE | no worse than the adapter run by more than 2% |
| action accuracy | held-out paired flow loss (same protocol as the -1.95% measurement) | strictly better than -1.95% |
| direction actually learned | logged `con1_metric_cosine` | rises from ~0 (init) towards >0.3 |
| closed loop | LIBERO-Plus **L5 Camera Viewpoints (243) + L5 Robot Initial States (69)**, paired against the current adapter and the official baseline | no regression, and a positive paired margin if any |

The first two are cheap and run every 1k steps; the closed loop is the expensive
gate and is only run once the offline criteria pass.

## Status

* Code + config + 31 unit/integration tests: done (`feat/con1-clean-anchored-delta`).
* Defaults off, so every existing run and checkpoint is unchanged.
* Stage-1 training and the L5 comparison: pending a free GPU window (the L5
  reference evaluation of the *current* adapter is running on the same machine).
