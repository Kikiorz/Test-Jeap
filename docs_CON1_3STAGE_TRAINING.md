# Con1 three-stage training record

Status: implementation review only. Do not launch training until the gradient
audit below passes.

## Frozen and trainable state

Frozen throughout: the official JEPA-WAM PI0.5 VLM/image encoder, V-JEPA2.1
teacher and current anchor, and action blocks 0--13. Stage 1 trains only the
delta head and one residual Cross-Attention adapter; action expert and output
projection updates are masked. Stages 2 and 3 additionally train blocks 14--17
and `action_out_proj`.

## Boundaries

The schedule is encoded in `Pi0Config`: stage 1 is 2,000 updates with beta 0;
stage 2 is 5,000 updates with beta 0; stage 3 is 5,000 updates with beta
linearly ramping from 0 to 0.5. Total is 12,000 updates. A resume must keep the
same schedule and must not reuse the previous old reciprocal directory.

## Objective

For one random flow-matching time, the model performs one principal forward.
The post-injection action flow error supplies a VJP through Cross-Attention into
`delta_hat` and the transition head. The latent loss uses only valid future
frames within the same episode; padded tail entries are excluded before
arithmetic.

## Required preflight

1. Run module tests for zero-residual equivalence and nonzero action gradient.
2. Instantiate the official 40k model and verify the head namespace and shapes.
3. Run one four-GPU update; require finite losses/gradients and nonzero Con1
   gradients.
4. Verify stage 1 leaves action/output parameters unchanged and the first-14
   gradient mask remains zero; verify updates begin after the 2k boundary.
5. Only then create a new Supervisor experiment directory and launch 12k.

The old `pi05_libero_con1_reciprocal_40k/full_reciprocal_12k` output is not a valid resume
point for this implementation review.

The first attempted launch was discarded: passing a non-contiguous episode list
directly to LeRobot renumbered its action index table. The current loader reads
the complete table and applies an original-index `Subset`, preserving episode
boundaries and the cache's episode IDs.

## Authorized continuation: 12k to 17k

After the initial 12,000 updates, continue for 5,000 more using `--resume
--num-train-steps=17000`. Restore both model and Adam state from checkpoint
`11999` (directory name is the zero-based loop index; saved optimizer step is
12,000). Keep the existing experiment directory; do not use `--overwrite`.

This extends stage 3: beta stays at 0.5; there is no new warm-up or reinitialization.
The LR after the first 100 updates remains 1e-5 for the latent head, 5e-6 for
the fusion projections, and 1e-6 for alpha, action blocks 14--17 and action_out_proj.
Blocks 0--13, the VLM and the teacher remain frozen. Global batch stays 4 across
four GPUs. `--keep-period=1` preserves the 12k checkpoint `11999` as well as the
new checkpoints despite its non-multiple-of-1000 directory name.

The optimizer resumes exactly; the current data loader does not checkpoint its
iterator position and restarts its deterministic shuffled order. Training-loss
decline alone does not establish held-out improvement or lack of convergence.

The later batch-4 continuation is retained only as an aborted diagnostic. The
replacement experiment is `con1_three_stage_b128_17k`, initialized fresh from
the official 40k checkpoint plus the 20k head, with global batch 128.

## Bounded fusion learning-rate probe (2026-09-10)

**Superseded before launch:** the user subsequently requested direct training
with increased fusion LR instead of this 300-update probe. The probe is an
unused diagnostic utility, not a completed experiment or validation result.

The batch-128 run is paused at the user's request. Its checkpoint `1000`
contains 1001 completed updates and remains untouched. Do not restart its
Supervisor command with `--overwrite` to resume: that would erase the run.

`scripts/probe_con1_fusion_lr.py` restores this checkpoint including Adam state
into a separate experiment `con1_b128_fusion3e5_probe300_from1k`. Only fusion
projection updates are multiplied by 3 (LR 3e-5); head stays 1e-5, alpha-logit
stays 1e-6, and the action expert stays frozen. Global batch is still 128.
The 300 additional updates stay within stage 1; the probe saves and exits,
without automatically continuing the 17k schedule.

Before training and every 100 updates, evaluate the exact same 512 held-out
examples and fixed flow noise/times. Validation uses the existing episode split
(seed 42), shuffled selection seed 20260910, and noise seed 701. These are
flow/delta diagnostics, not rollout success rates. The data iterator restarts
on restore, as it does in the existing trainer. A before/after improvement
alone does not isolate learning-rate causality without a matched 1e-5 control.

## Active request: direct higher-LR continuation to 17k

Experiment: `con1_b128_fusion3e5_17k_from1k`. Copy the complete original
checkpoint `1000` into this new directory, then use `--resume` (never
`--overwrite`). Restore model, Adam moments, and update counter (1001);
continue to 17000 total updates, not 17000 additional updates. Unsaved updates
after the 1k checkpoint are not recovered. The original experiment is stopped
and preserved. Global batch remains 128 across four GPUs.

Set `con1_cross_attention_lr_multiplier=3`: fusion projections use 3e-5 in
stage 1 and 1.5e-5 in stages 2/3. Head remains 1e-5; alpha-logit remains 1e-6;
action blocks 14--17 and output remain frozen until update 2000, then use
1e-6. Stage 2 starts at 2000 and stage 3 at 7000; beta reaches 0.5 at 12000
and remains there through 17000. No additional warmup or batch change.
Tests for stage masks, unchanged non-fusion update scaling, and model gradient
integration passed (11 tests) before launch. This run has training metrics;
the canceled probe's fixed validation is not automatically run by train.py.

## Active request: high-then-decay flow weighting

The user asked that the action/flow objective be weighted higher early and
cosinely decay toward a still-high floor. `Pi0Config` now carries
`con1_flow_weight_initial=2.0`, `con1_flow_weight_final=1.0`, and
`con1_flow_weight_decay_steps=15000`. During stage 1 (updates < 2000) the flow
weight is pinned at 2.0; from 2000 through 17000 it decays from 2.0 to 1.0, so
it stays above 1.5 for roughly the first half of the decay and never falls below
the previous unit weight. The raw `flow_loss` metric stays unweighted for
comparability; `weighted_flow_loss` and `flow_weight` are logged separately.

This supersedes the previous `con1_b128_fusion3e5_17k_from1k` run, which was
stopped before stage 2 and does not include the flow-weight schedule. The
replacement experiment is `con1_b128_fusion3e5_floww_17k_from1k`, restored from
the same pristine `1000` checkpoint with the 3x fusion LR and the new flow
weighting active from update 1000 onward. The old runs are preserved, not
deleted.

## Active request: 10x learning rates

The user judged the updates too small and asked for every learning rate to be
raised by one order of magnitude. `TrainConfig.con1_lr_multiplier` now wraps the
base schedule in `_optimizer.ScaledSchedule`, so the multiplier scales every
trainable group at once. With `--con1-lr-multiplier=10` and the retained
`--con1-cross-attention-lr-multiplier=3`, the effective rates become:

| Group | Before | 10x |
|---|---:|---:|
| delta head | 1e-5 | 1e-4 |
| cross-attention (stage 1 / 2-3) | 3e-5 / 1.5e-5 | 3e-4 / 1.5e-4 |
| alpha-logit | 1e-6 | 1e-5 |
| action blocks 14-17 + output | 1e-6 | 1e-5 |

Experiment `con1_b128_lr10x_floww_17k_from1k` restores the same pristine `1000`
checkpoint with the 3x fusion multiplier, the high-then-decay flow weight, and
the 10x uniform LR. This is an aggressive step: restored Adam moments were
trained at 1e-5, so the first updates are roughly 10x larger. The trainer aborts
on nonfinite metrics and the supervisor does not auto-restart, so a blow-up stops
the run instead of corrupting it. The prior `floww` run is stopped and kept.

### Outcome of the 10x attempt (stopped)

`con1_b128_lr10x_floww_17k_from1k` was stopped after 91 updates (last logged
step 1091) because the residual branch grew without bound instead of settling:
`con1_residual_energy` rose 1041 -> 1091 as 5.7e-5, 1.6e-4, 4.9e-4, 1.3e-3,
2.3e-3, 4.9e-3 (roughly doubling every 10 updates) while `con1_delta_loss`
drifted up from 0.0184 to 0.0189. All values stayed finite, so the trainer's
nonfinite guard never fired; the decision to stop was manual. With
`con1_residual_weight=1e-3` the residual penalty at 5e-3 contributes only ~5e-6,
so nothing bounded the correction once the fusion LR reached 3e-4.

Conclusion: one order of magnitude on every group is too aggressive for the
current residual parametrization. A smaller uniform multiplier (e.g. 3x, which
keeps fusion at 9e-5 in stage 1) is the sensible next attempt. The 1k source
checkpoint is untouched.

## Current request: fresh 17k, uniform 2x LR, batch 64

The user asked to ignore every earlier Con1 run, apply a uniform 2x learning
rate, and train a fresh 17k experiment at global batch 64. This is a clean start
from update 0 using the named config's own weight loader (official 40k JEPA-WAM
params plus the 20k anchored delta head); no checkpoint is resumed and no prior
directory is reused. The 3x cross-attention special case is dropped, so
`con1_cross_attention_lr_multiplier` stays at its 1.0 default.

Effective rates with `--con1-lr-multiplier=2` over the 1e-5 base schedule:

| Group | 2x value |
|---|---:|
| delta head | 2e-5 |
| cross-attention (stage 1 / 2-3) | 2e-5 / 1e-5 |
| alpha-logit | 2e-6 |
| action blocks 14-17 + output | 2e-6 |

The high-then-decay flow weight is retained: 2.0 through update 2000, then a
cosine decay to 1.0 at 17000. Stage boundaries are unchanged (2000 / 7000 / beta
ramp to 0.5 at 12000). Batch 64 halves per-step cost, so wall-clock should be
roughly half the batch-128 runs if the step is compute bound.

Experiment: `con1_b64_lr2x_floww_17k`.

## Action-conditioned Con2 predictor (2026-09-10)

### Correction: earlier probe NMSE numbers were leak-inflated

The probe scripts (`probe_con1_feature_sufficiency.py`,
`probe_con1_vlm_conditioning.py`, `probe_con1_action_conditioning.py`) shuffled
(sample, horizon) *pairs* and then split those pairs into train/val/eval. The
same sample therefore appeared in training under some horizons and in eval under
others, and because a sample's deltas across horizons are highly correlated, the
fit could interpolate instead of generalise. Every NMSE those scripts reported
for a *fitted* estimator is therefore invalid, including the "0.414 linear
floor" that motivated both the extra readout and the ridge warm start.

`fit_con1_readout.py` uses the correct protocol (split by sample) with the same
features `[mean(R), z_t]` and finds:

| features | best lambda | held-out NMSE |
|---|---:|---:|
| raw, 4864-d | 1e4 | 0.871 |
| random projection to 1024-d | 1e4 | 0.884 |

So a linear map from these features reaches only ~0.87, far worse than the
trained delta head at 0.474 (held out, sample split). Consequences:

* The delta head is **not** underfit relative to a linear baseline; the
  "a plain ridge beats the head by 0.07" claim is withdrawn.
* The closed-form ridge warm start is not viable (0.87 > 0.474) and must not be
  used.
* The action-conditioning gain (+0.022) and the VLM-extra-input gain (+0.009)
  were measured under the same leak and need re-measurement with a sample-level
  split before any claim is made.
* NMSE measured for a *fixed* model is unaffected by this bug, so the head's
  0.4739 held-out number and the earlier alpha-scan stay valid.

### Corrected probe numbers (sample-level split)

`probe_con1_action_conditioning.py` re-run with the fixed split, 1536 samples,
lambda chosen on a held-out validation split:

| probe | eval NMSE |
|---|---:|
| trained delta head (fixed model, unaffected by the bug) | 0.4908 |
| rff(z_t + R_t), no action | 0.9362 |
| rff(z_t + R_t + causal actions) | 0.8842 |
| rff(z_t + R_t + full action chunk) | 0.9197 |
| linear z_t + R_t + causal actions | 0.9803 |

Two conclusions, both opposite to the leak-inflated run:

1. The trained head beats every fitted probe by a wide margin (0.49 against a
   best of 0.88). The head is **not** underfit, and the "extra readout" and
   "ridge warm start" work was aimed at a problem that does not exist. The ridge
   warm start is abandoned.
2. Action conditioning is still a real effect and is *larger* than the leaked
   estimate suggested: 0.9362 -> 0.8842, i.e. 0.052 NMSE (5.6% relative). Causal
   conditioning also beats the non-causal full chunk (0.8842 vs 0.9197), so the
   causal mask is doing work.

Note lambda sits at the top of the grid for these probes, so the fitted numbers
are conservative; this does not change the ordering or the fact that the head
remains far better than any probe.

Measured first with the kernel-ridge probe `scripts/probe_con1_action_conditioning.py`
(1024 samples, 60/20/20 split, matched pipeline):

| probe | eval NMSE |
|---|---:|
| existing delta head | 0.4825 |
| z_t only | 0.4410 |
| z_t + R_t (control, no action) | 0.4361 |
| causal action prefix alone | 0.9500 |
| z_t + R_t + causal actions | 0.4143 |

Two readings. Actions alone are nearly as useless as the zero predictor
(0.95 vs 1.0), because the same action moves the latent differently in different
states, so the gain only appears through state-action interaction terms. And the
matched control puts the action-conditioning gain at 0.4361 -> 0.4143, i.e.
about 0.022 NMSE. The earlier verbal expectation of reaching 0.2-0.3 is not
supported by this measurement.

Implementation in `src/openpi/con1/modules.py`:

* `AnchoredDeltaHead` gains `use_action_conditioning` / `action_dim`. Actions are
  embedded as horizon-indexed tokens and attended with a strict causal mask, so
  horizon j only reads actions a_{t..t+j-1}. `test_action_conditioning_shapes_and_strict_causality`
  checks that perturbing action k changes horizon >= k only.
* A per-horizon `direct_readout` from `[mean(R), z_t]` was added because the
  attention path alone underfit (0.4825 against a 0.414 linear floor); this path
  can express that floor immediately while the nonlinear branches refine it.
* `train.py`-side: ground-truth actions condition the head during training. At
  sampling time `_con1_prefix` runs once and the delta is re-derived per
  denoising step from the clean-action estimate `a_hat = x_t - time * velocity`
  (one-step lagged), which is the intended policy/world-model coupling.
* `BaseAndCon1HeadWeightLoader` now allows head parameters that are absent from
  an older head checkpoint (the new branches keep their fresh init) while still
  rejecting unknown keys and shape mismatches.

Config `pi05_libero_con1_action_cond_40k` runs the new head. A 2000-step stage-1
smoke run measures whether the new head actually lowers `con1_delta_nmse`
against the 0.48 baseline.

## Diagnostic 1: deterministic alpha scan (2026-09-10)

`scripts/probe_con1_residual_causal.py` reuses the trainer's own restore path,
fixes one batch of 64 samples plus its flow noise and time, and varies only the
gate alpha (and optionally the residual output scale). Paired differences are
causal for the frozen parameters, not sampling noise.

Checkpoint 3000 (stage 2, action expert already unfrozen), `alpha_0` flow =
0.019281:

| alpha | correction RMS | flow | delta vs alpha=0 |
|---:|---:|---:|---:|
| 0.05 (learned) | 0.030 | 0.019312 | +3.1e-5 |
| 0.25 | 0.150 | 0.019540 | +2.6e-4 |
| 0.5 | 0.299 | 0.020136 | +8.6e-4 |
| 1.0 | 0.599 | 0.022508 | +3.2e-3 |
| learned alpha, residual x10 | 0.300 | 0.020142 | +8.6e-4 |

Checkpoint 1000 (stage 1, base frozen), `alpha_0` flow = 0.019803:

| alpha | correction RMS | flow | delta vs alpha=0 |
|---:|---:|---:|---:|
| 0.05 (learned) | 0.0028 | 0.019800 | -3.0e-6 |
| 0.25 | 0.0140 | 0.019790 | -1.3e-5 |
| 0.5 | 0.0280 | 0.019783 | -2.0e-5 |
| 1.0 | 0.0561 | 0.019815 | +1.2e-5 |
| learned alpha, residual x10 | 0.0281 | 0.019780 | -2.4e-5 |

Reading: the branch has real authority (at alpha=1 it moves the action chunk by
L2 0.053 against a velocity RMS of 1.02). In the frozen-base regime a roughly
10x stronger residual reaches an optimum near correction RMS 0.028 and improves
flow by about 2e-5 (0.1%). After stage-2 training the residual has grown to RMS
0.599, ten times past that optimum, and every gate value now makes flow worse,
with the degradation scaling like RMS squared. So the residual direction holds a
small usable signal, but its magnitude is unbounded and stage 2 overshoots it.

## Diagnostic 2: feature sufficiency (2026-09-10)

`scripts/probe_con1_feature_sufficiency.py` collects R_t (PI0.5 query tokens),
z_t, and cached future latents, then fits exact kernel-ridge probes in the full
2816-d output space on a held-out split. 1536 samples, 60/20/20 split, lambda
selected on the validation split.

| probe | eval NMSE |
|---|---:|
| zero predictor | 1.000 |
| existing delta head | 0.480 |
| linear from z_t only | 0.417 |
| linear from pooled R_t only | 0.424 |
| linear from both | 0.414 |
| random-Fourier nonlinear from both | 0.530 |

Reading: the trained head is underfit, since a plain ridge from the same inputs
beats it by 0.066 NMSE. But the ceiling is about 0.41, so roughly 41% of the
delta variance is not recoverable from these inputs at all. Pooled R_t alone
(0.424) is slightly worse than z_t alone (0.417), and adding R_t to z_t buys only
0.003 NMSE, so the expensive predictive tokens carry almost no delta information
beyond the current latent. The pooled representation also handicaps R_t, so the
"head is underfit" finding is robust while the R_t ceiling is an upper bound on
how good a token-level head could be.

Combined with Diagnostic 1, improving the head from 0.48 to 0.41 NMSE is
unlikely to change the flow outcome much, because the residual's usable effect
at its optimum is already only about 0.1%.

### Stage 2 action-expert rate raised to 1e-5

The user judged 2e-6 for the unfrozen action blocks too slow and asked for
1e-5. `TrainConfig.con1_action_lr_multiplier` now controls that group separately
from alpha (which stays at its 0.1 factor). Setting
`--con1-action-lr-multiplier=0.5` over the 2e-5 base gives exactly 1e-5 for
action blocks 14-17 and `action_out_proj`; the delta head stays 2e-5,
cross-attention stays 2e-5 / 1e-5, and alpha stays 2e-6.

The running experiment is resumed in place from its own checkpoint `1000` with
`--resume` (never `--overwrite`) so the schedule, Adam state, and experiment
directory continue. Only the stage-2 action rate changes; stage 1 behavior is
identical because those parameters are masked until update 2000. The ~41
unsaved updates past checkpoint 1000 are not recovered.
