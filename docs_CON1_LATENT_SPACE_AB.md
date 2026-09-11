# Does the Con1 latent space matter? (JEPA pooled vs VLM tokens)

Motivation: every coupling attempt so far used the pooled V-JEPA 2.1 latent and
measured as a no-op. The candidate fix was to route the action-facing residual
through the PI0.5 VLM predictive tokens instead, which the OOD probing showed
are both action-informative and perturbation-stable, and whose future is far
more learnable. This document reports the direct A/B.

## 1. Delta-target comparison (existing cache, 1800 train / 360 test frames)

`scripts/compare_latent_targets.py`. Held-out R^2 of the 10-step action chunk;
`prev` is the previous action alone, which is the bar every latent has to clear.

| target | dim | prev only | + true delta | + predicted delta | predictability (cos) |
|---|---|---|---|---|---|
| JEPA pooled (incumbent) | 2816 | 0.731 | 0.474 | 0.704 | +0.19 |
| VLM pooled | 2048 | 0.731 | 0.548 | 0.716 | +0.15 |
| VLM tokens (unpooled) | 131072 | 0.731 | 0.254 | 0.511 | +0.04 |

At horizon 1 the pattern is identical (prev 0.950; true-delta arms 0.838 /
0.860 / 0.143). **No delta target adds information over the previous action**,
so the "predict the future latent" objective does not carry the signal Con2
needs, whichever latent is used.

## 2. Future-latent learnability (identical head, identical budget)

`scripts/launch_head_ab.sh`, anchored delta head, width 512, batch 256,
lr 1e-5. Metric is held-out delta MSE divided by the copy-current baseline
(1.0 = "predict no change"; lower is better).

| steps | VLM latent (2048) | JEPA latent (2816) |
|---|---|---|
| 0 | 1.005 | 1.000 |
| 1k | **0.652** | 0.836 |
| 3k | **0.533** | 0.753 |
| 4k | ~0.50 | 0.732 |
| 9k | **0.454** | — |

The VLM latent is clearly the better *prediction target*. That is the strongest
argument the VLM space had; the next section shows it does not transfer.

## 3. Action-side A/B (the decisive measurement)

Both adapters were retrained from scratch with the recipe that produced the
-1.95% flow gain (frozen 16 layers, batch 64, `con1-lr-multiplier 2`, 1000
steps), on the **same 4-GPU topology** and the same seed, changing only the
latent root, the latent dimension and the (already trained) delta head. Scored
with `probe_con1_budget_and_conditioning.py`, 24 paired batches x 8 samples.

| adapter latent | flow loss (mean +- stderr) | correction RMS |
|---|---|---|
| JEPA pooled | 0.021001 +- 0.003962 | 0.3454 |
| VLM pooled | 0.020996 +- 0.003962 | 0.3453 |

Paired per-batch difference (JEPA - VLM): **+5.4e-6 +- 4.5e-6**, i.e. 0.026% of
the flow at 1.2 sigma. Swapping the latent space — with a strictly better
trained delta head on a strictly more predictable target — is invisible in the
action objective.

## 4. Why: the adapter's usable capacity is latent-free

`src/openpi/con1/modules.py`, `ActionDeltaCrossAttention.__call__`:

```python
correction = jax.nn.sigmoid(logit) * residual        # latent-conditioned path
if self.use_action_adapter:
    normed = nn.LayerNorm(name="adapter_norm")(action_hidden)
    hidden = nn.Dense(self.width, name="adapter_in")(normed)
    hidden = nn.gelu(hidden)
    adapter = nn.Dense(self.action_width, name="adapter_out",
                       kernel_init=nn.initializers.zeros_init())(hidden)
    correction = correction + self.adapter_scale * adapter   # latent-free path
```

The correction that actually moves the flow is the branch fed only by
`action_hidden`; the code comment already says it "adds capacity that does not
depend on the latent prediction at all". Two consequences, both observed:

* the correction RMS is identical to three significant digits across the two
  latent spaces (0.3454 vs 0.3453), so the latent-conditioned term is inert in
  both;
* the measured `cos(dMSE/dDelta, dflow/dDelta) = -0.006` and the ~700x gradient
  imbalance follow from the same structure: the latent only enters through a
  zero-initialised attention output that the optimizer has no reason to grow,
  because the latent-free branch already explains the available gain.

## Conclusion and next step

The latent space is not the bottleneck: JEPA pooled, VLM pooled and VLM tokens
all fail to add action information, and swapping between them changes the
action objective by 0.03%. What has to change is the *architecture of the
coupling*, not the target:

1. remove (or explicitly gate) the latent-free `adapter` branch so the only
   trainable capacity left is the latent-conditioned cross-attention, and
2. supervise that path directly on the action objective rather than only
   through the delta MSE, e.g. require the gated correction to reduce the flow
   loss given the delta, and initialise `out` non-zero so the key/value
   projections receive gradient from step 0.

Until one of those is in place, any latent-space choice will keep measuring as
a no-op, which is exactly what all six experiments to date show.
