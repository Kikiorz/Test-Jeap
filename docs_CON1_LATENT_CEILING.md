# RoboTwin: how predictable is the future-latent delta?

## The question

The Con1 head is trained against `con1_delta_nmse = mse / mse(predicting zero)`,
i.e. `1.0` means "learned nothing" and `0.0` means "perfect". On RoboTwin it sits
at **0.86** (arm A) / **0.78** (arm B, whole-prefix context), so it explains only
14-22% of the delta variance.

Con2 refines that delta (`delta_used = delta_pred + F(delta_pred, z_t)`), so the
whole Con2 story depends on whether that delta carries predictable signal at all.
Two readings were possible:

* the delta is noise, and Con2's premise has to change;
* the delta is predictable and the *head* is the bottleneck.

## The measurement

`scripts/probe_robotwin_latent_ceiling.py` fits ridge regression on the cached
features (same cache the training reads) and reports the same NMSE. Features:
the current latent `z_t` (4224), the pooled 64 predictive tokens `r_t` (2048), or
both. 15,000 train samples / 5,000 held-out-episode samples, CPU only.

## Result

Mean NMSE over the head's 16 horizons (the exact quantity `con1_delta_nmse`
reports), at the best ridge strength `lambda = 0.3 * n`:

| predictor | mean NMSE h=1..16 | h=1 | h=8 | h=16 |
|---|---|---|---|---|
| predict zero | 1.000 | 1.000 | 1.000 | 1.000 |
| `r_t` only (linear) | 0.893 | 0.993 | 0.895 | 0.803 |
| `z_t` only (linear) | **0.659** | 0.887 | 0.645 | 0.530 |
| `z_t + r_t` (linear) | 0.666 | 0.899 | 0.653 | 0.531 |
| **Con1 head (arm A, trained)** | **0.86** | - | - | - |

## Conclusions

1. **The delta is predictable; the head is not learning it.** A *linear* map from
   the current latent alone reaches 0.659, while the trained head - attention
   over 64x2048 tokens plus a residual MLP - only reaches 0.86. The head is
   strictly worse than a linear baseline on a subset of its own inputs.
2. **The predictive tokens `r_t` carry almost no delta information.** `r_t` alone
   sits near the predict-zero line, and adding it to `z_t` slightly *hurts*
   (0.666 vs 0.659). This independently reproduces the earlier pooled-token
   probe and says Con1's benefit to the action is not coming from `r_t`.
3. **Longer horizons are easier, not harder.** NMSE falls monotonically from 0.89
   at h=1 to 0.53 at h=16: per-frame feature noise dominates short deltas while
   the systematic drift accumulates, so the signal-to-noise ratio improves with
   the horizon.

## What this implies

Con2's premise survives: there is real signal to refine. The bottleneck is head
capacity/optimisation. `Pi0Config.con1_direct_readout` already exists for exactly
this diagnosis - its comment records the same finding on LIBERO ("the attention
path alone was measured to underfit badly (0.483 held-out NMSE against a 0.414
linear floor)") - and the head's raw token mean is deliberately exposed so the
readout can be warm-started from a closed-form solution.

## Reproduce

```bash
.venv/bin/python scripts/probe_robotwin_latent_ceiling.py \
  --horizons 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --lambdas 0.1 0.3 1.0 \
  --out /workspace/artifacts/con2/robotwin_latent_ceiling_allh.json
```

Raw output: `/workspace/artifacts/con2/robotwin_latent_ceiling_allh.json`.
