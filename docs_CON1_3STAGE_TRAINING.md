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
