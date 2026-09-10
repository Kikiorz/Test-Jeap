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
