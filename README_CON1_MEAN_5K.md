# Direct-delta Con1: mean-loss continuation from 5k

This branch snapshots the current Con1 implementation and focused tests, not
training data, private model checkpoints, unpublished paper text or credentials.
Historical experimental code inherited from the base commit is not the definition
of this run; use `pi05_libero_paper_con1_direct_delta_mean_stage2`.

The run loads the complete direct-delta stage-1 checkpoint `4999`, creates the
joint-stage optimizer, and performs another 15,000 updates. It freezes Action
blocks 0–15 and all non-block Action projections/norms; blocks 16–17, Con1, and
the continuous residual coefficient are trainable. Freezing masks gradients and
final AdamW updates, including momentum/weight decay. This is not stage-1
optimizer restoration and is not an exact continuation of the older 8k branch.

The target is raw `z[t+j] - z[t]`, computed in float32 from the unchanged frozen
teacher frame states. Prediction loss averages feature dimension D=2816 and
uses the existing normalized, detached temporal weights q; lambda stays 0.1.
There is no additional horizon division, whitening, or action reconstruction.
Training does not run an independent original-baseline forward/non-regression
penalty. These choices do not guarantee non-regression in closed-loop success.

The fixed cumulative evaluation milestones are 10k, 15k and 20k. Training resumes
after the first two evaluations. The first three actual joint updates are retained
and audited against the 5k frozen-weight anchor before continuing.

## Runtime and entry points

Use the project environment, `PYTHONPATH=src`, and the Blackwell runtime pins in
`ops/requirements-con-blackwell*.txt`. Checkpoint and data locations are specified
by the configuration; supply verified inputs before launching training.

```sh
PYTHONPATH=src .venv/bin/python ops/train_paper_con1.py --stage 2 --until-step 5000 --mean-from5k
```

The two teacher caches are distinct: independent `[o_k,o_k]` frame states for
Con1, and `[o_t,o_min(t+10,last)]` base/wrist 8x8 pooled token targets for the
frozen JEPA auxiliary loss. `ops/rebuild_mean_teacher_worker.py --rank N` rebuilds
both with one frozen encoder per GPU; launch ranks 0–3 with separate visible GPUs.
The loader, not the cached frame-state diagnostic metadata, selects direct deltas.

Focused loss/freeze tests:

```sh
PYTHONPATH=src:ops JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  src/openpi/models/mean_con1_loss_test.py src/openpi/models/orthogonal_con1_test.py \
  src/openpi/models/minimal_con1_diagnostics_test.py src/openpi/training/action_freeze_test.py \
  ops/mean_con1_experiment_test.py ops/mean_teacher_rebuild_test.py
```

Passing implementation tests is not evidence of improved policy performance.
