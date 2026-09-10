# Con1: reciprocal anchored latent transition

This branch contains only the current Con1 design. It does not define the old
binary gate, RAPR router, or previous multi-adapter experiments.

## Model

The official JEPA-WAM PI0.5 40k checkpoint supplies frozen predictive tokens
and the current V-JEPA state: `R_t` is `[64,2048]` and `z_t` is `[2816]`.
The anchored head predicts ten latent displacements:

```
delta_hat[1:10] = D(R_t, z_t)
z_hat[t+j] = z_t + delta_hat[j]
```

There is no learned `Q0`. Before action block 14, one Cross-Attention module
uses `Q=Wq LayerNorm(H_A)` and `K,V=Wk,Wv(delta_hat)`:

```
H_A' = H_A + alpha * CrossAttention(Q, K, V)
```

`alpha` is continuous and initialized to 0.05. The output projection is
zero-initialized, preserving the base action function at step zero. This is a
residual initialization, not a binary deployment gate.

The first 14 action blocks and the VLM are frozen. In the joint phase, only
blocks 14--17, `action_out_proj`, Con1, and the delta head are trainable. The
V-JEPA teacher, `R_t` producer, and `z_t` anchor are always stopped-gradient.

## Loss and reverse constraint

The latent target is the episode-local difference `z*_{t+j} - z*_{t}`. The
joint loss is `L = L_flow + 0.2 L_delta + 1e-3 L_residual`. `L_flow` is
computed after Cross-Attention on the physical seven action dimensions and valid
episode-tail positions. Its gradient therefore flows through the adapter into
`delta_hat` and the transition head: this is the reverse action-to-latent
constraint. Stage 3 adds detached action-sensitivity weighting across horizons;
it does not introduce second-order gradients.

## Three-stage schedule

The run has exactly 12,000 updates, starting from the official 40k base plus
the separately validated 20k latent-head checkpoint.

| Stage | Updates | Trainable additions | SGR |
|---|---:|---|---|
| 1. residual warm-up | 0--1,999 | Con1 and delta head; action expert/output frozen | off |
| 2. reciprocal joint | 2,000--6,999 | Con1, delta head, blocks 14--17, output projection | off |
| 3. sensitivity joint | 7,000--11,999 | same as stage 2 | beta ramps 0 to 0.5 |

No extra ten-step sampler is run in training; each update uses one random flow
time and one principal forward. Validation uses a fixed episode holdout.

## Metrics and provenance

Retain `con1_delta_loss`, `con1_delta_nmse`, `con1_alpha`, residual energy,
`flow_loss`, total loss, and gradient norm. Checkpoints use the normal OpenPI
manager. A loss improvement is not a policy-success guarantee; final claims
require paired LIBERO-Plus rollouts against the pure 40k base.

The cache must identify the official 40k checkpoint, pinned V-JEPA2.1 teacher,
and pinned LIBERO data. Future labels enter only the loss. The head-only trainer
is retained only to reproduce the initialization checkpoint, not as the Con1
experiment itself.
