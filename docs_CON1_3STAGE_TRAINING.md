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
