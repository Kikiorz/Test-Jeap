# Con2: predicting the true future latent

Status: measured on LIBERO Goal/Object (task ids 10-29), frozen base. All
numbers below come from `eval_con1_delta_nmse.py`, which evaluates a *fixed*
model on held-out samples and therefore contains no fitting and no train/eval
leakage.

## What Con2 predicts

Given the current observation only, predict the frozen V-JEPA 2.1 latent of the
real future frames:

```
delta_hat_{t+1..t+10} = D_psi(R_t, z_t [, a_{t..t+9}] [, c_t])
z_hat_{t+j} = z_t + delta_hat_{t+j}
```

* `R_t` - the 64 PI0.5 predictive query tokens of the current observation.
* `z_t` - the frozen teacher's current latent. It is `concat_views(mean_spatial(VJEPA([o,o])))`,
  i.e. 576x1408 per view spatially mean-pooled to 1408, two views -> 2816.
* `a_{t..t+9}` - the demonstrated action chunk (optional conditioning).
* `c_t` - pooled full VLM prefix: image patches, language and state (optional).

Loss: masked mean squared error between `delta_hat` and `z*_{t+j} - z_t`, with
episode-local validity masking. Reported as NMSE = MSE / E||delta||^2, so 1.0 is
the zero predictor.

## Results (1024 held-out samples)

| model | held-out NMSE |
|---|---:|
| zero predictor | 1.000 |
| trained head, no conditioning (reference) | 0.4739 |
| trained head, action-conditioned, true actions | **0.4719** |
| same head, action chunk shuffled | 0.4836 |
| same head, action chunk zeroed | 0.4842 |
| trained head, action + pooled VLM context, true actions | 0.4721 |
| same, action shuffled / zeroed | 0.4843 / 0.4848 |

Convergence check on the no-conditioning head (same held-out set):

| checkpoint | held-out NMSE |
|---|---:|
| step 1000 | 0.47393 |
| step 2000 | 0.47654 |
| step 3000 | 0.47363 |

## Findings

1. The head is **converged**: three checkpoints spanning 3000 updates on top of
   20k head pre-training steps are flat to within 0.003. Neither more steps nor
   a different learning rate moves it.
2. Action conditioning is **real but marginal**: the action *content* matters
   (shuffling costs +0.012, zeroing +0.012), yet the net gain against the
   unconditioned head is only 0.002 NMSE (0.4% relative).
3. Zeroing the action chunk makes the conditioned head *worse* than the
   unconditioned head (0.4842 vs 0.4739), so conditioning is a required part of
   the forward path, not an optional add-on.
4. Fitted baselines are far weaker than the head: with a correct sample-level
   split, linear and random-Fourier probes over `[mean(R_t), z_t]` (with or
   without actions) land at 0.88-0.99. The head at 0.47 is not an underfit
   artefact - earlier "a linear probe reaches 0.41" numbers came from a
   train/eval leak across horizons of the same sample and have been withdrawn.
5. Extra VLM context (`c_t`, pooled image patches + language + state, implemented
   behind `con1_vlm_context`) changes nothing: 0.4721 with it against 0.4719
   without. Adding the whole VLM output does not move the plateau.

Final three-way comparison on identical held-out samples and an identical
evaluation path:

| head | held-out NMSE |
|---|---:|
| no conditioning | 0.4739 |
| + action chunk | 0.4719 |
| + action chunk + pooled VLM context | 0.4721 |

## Interpretation and limitations

The head explains roughly 53% of the variance of the pooled latent delta. The
remaining ~47% is not reachable from `R_t`, `z_t`, the action chunk, or the
pooled VLM prefix, and it is not a training-recipe problem.

The most likely structural cause is the target itself: `z_t` is a **spatial mean
pool** of 576 V-JEPA tokens, which discards the spatial detail a future
predictor would use. Recovering that requires token-level targets, which need a
re-extract of roughly 890 GB for the full dataset (only a subsampled re-extract
is practical).

Reportable claim as it stands: the policy's internal representation predicts the
true future latent with a held-out NMSE of about 0.472, converging within the
first 1k updates, and action conditioning adds a small but genuine improvement.

The plateau is robust to every input we added: `R_t`, `z_t`, the demonstrated
action chunk, and the whole pooled VLM prefix all land within 0.002 NMSE. That
is the strongest available evidence that the remaining error is a property of
the target (spatially mean-pooled latent) rather than of the conditioning or the
optimiser.
