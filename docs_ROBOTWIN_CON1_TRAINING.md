# Con1 on RoboTwin 2.0 — training plan (robotwin branch)

Status: plan for review. No RoboTwin training is launched yet. This branch is
created from `feat/con1-clean-anchored-delta` (the clean Con1 implementation)
and will receive the RoboTwin-specific data/config adaptation below.

## 1. Goal

Run the existing three-stage Con1 (anchored latent-delta head + residual
cross-attention adapter) on RoboTwin 2.0, starting from the author's
JEPA-WAM-pretrained pi0.5 `19999` checkpoint, and train only on the Clean
demonstrations of the 20 tasks used in the paper.

## 2. Base model

- Hub repo: `CokeAnd1ce/JEPA_WAM`
- Revision: `ca10ccbc191d8f56b4346487913e043b2722b6d2`
- Checkpoint:

  `checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_b128_fsdp4_gpu0123_seed42/19999`

- Download scope: `params` (~12.5 GB) + `assets` + `_CHECKPOINT_METADATA`.
  Exclude the 33 GB `train_state`.
- Structure: 58 base parameter leaves, same namespace as the clean JEPA-WAM
  pi0.5 model (`PaliGemma`, `action_in_proj`, `action_out_proj`,
  `time_mlp_in/out`, `vjepa_alignment_*`, `vjepa_query_tokens`).
- Observed from `params/_METADATA`:
  - `vjepa_query_tokens` shape `[16, 2048]` -> 16 V-JEPA query tokens.
  - `vjepa_alignment_out` -> 1408-dimensional V-JEPA 2.1 target.
  - FSDP-4 sharding, global batch 128, seed 42.
  - `norm_stats` asset id is `local/robotwin_clean_20`.

There is also a non-`vjepa_aux` plain pi0.5 `19999`
(`pi05_robotwin_clean20_b128_fsdp4_gpu0123_seed42/19999`). Con1 builds on the
JEPA-WAM variant, so the `vjepa_aux` checkpoint above is the base; the plain
one can be kept as an optional ablation reference.

## 3. Clean-20 tasks (from the JEPA-WAM paper, Appendix B.2 / D)

RoboTwin 2.0 uses 20 manipulation tasks in two groups of 10. The policy is
trained only on Clean demonstrations and evaluated on both Clean and Random.
The 20 tasks are:

1. Adjust Bottle
2. Beat Block Hammer
3. Click Alarmclock
4. Click Bell
5. Dump Bin Bigbin
6. Grab Roller
7. Handover Mic
8. Lift Pot
9. Place Bread Basket
10. Place Bread Skillet
11. Place Burger Fries
12. Place Cans Plasticbox
13. Place Empty Cup
14. Place Object Basket
15. Place Shoe
16. Press Stapler
17. Shake Bottle Horizontally
18. Shake Bottle
19. Stack Bowls Three
20. Stack Bowls Two

## 4. RoboTwin configuration (to adapt)

The clean branch currently has only the LIBERO Con1 config
(`pi05_libero_con1_three_stage_40k`). We add a RoboTwin equivalent. Paper
states:

- bimanual, 14-dimensional actions;
- action horizon 50;
- three cameras: one external plus two wrist views;
- RoboTwin flow matching uses x-prediction (predict clean trajectory), unlike
  LIBERO velocity prediction.

The pi0.5 checkpoint's action head is smaller than the generic `gemma_300m`
default, and the V-JEPA branch uses 16 query tokens rather than the LIBERO
default 64. Before writing the final `Pi0Config`, a shape-match step must pin:

- exact `action_dim` / `action_horizon` that reproduce the checkpoint param
  shapes (`action_in_proj.kernel`, `action_out_proj.kernel`, `time_mlp_*`);
- `vjepa_num_queries = 16`, `vjepa_query_grid_size = 4`,
  `vjepa_target_grid_size`, `vjepa_target_dim = 1408`;
- the `vjepa_future_offset` used to produce the offline future targets
  ("delta50" in the run name is the most likely value and must be verified
  against the checkpoint/teacher cache);
- the exact camera keys and prompt format.

The shape-match is performed by instantiating candidate `Pi0Config`s and
comparing their `nnx` parameter tree against the downloaded
`params/_METADATA`; the weight loader will reject any mismatch, so the
correct values are verifiable before the first update.

## 5. Data

- Public source: `lerobot/robotwin_unified` (LeRobot v3.0).
- Robot: dual-arm ALOHA, `observation.state` and `action` shape `[14]`.
- Extract only Clean demonstrations for the 20 tasks above. The checkpoint's
  `norm_stats` asset id is `local/robotwin_clean_20`, so the local converted
  dataset directory should be named consistently (e.g.
  `/workspace/artifacts/datasets/robotwin_clean_20`) and its `norm_stats.json`
  must be the one restored from the 19999 checkpoint assets.

Open item: the exact task/episode filter for "Clean demonstrations" (which
episodes in `lerobot/robotwin_unified` belong to the Clean setting of the 20
tasks) needs to be pinned from the dataset metadata / task split before
conversion.

## 6. V-JEPA teacher and Con1 latent cache

The Con1 objective needs, for every training frame:

- frozen current R tokens from the 19999 base (the last 16 V-JEPA query tokens);
- independent V-JEPA 2.1 teacher future latents.

Plan mirrors the LIBERO pipeline:

1. Pin and download the V-JEPA 2.1 teacher weights (same teacher as LIBERO,
   `vjepa2_1_vitg_384.pt`), unless a different teacher revision is required.
2. Precompute teacher frame states over the RoboTwin Clean-20 subset on 4 GPUs.
3. Precompute the frozen 19999 R cache.
4. Build the anchored latent cache (`con1_latent_root`) with the same manifest
   format as `anchored_40k_features_v1`.

The `vjepa_future_offset` and any RoboTwin-specific image key / resize must be
verified before this long precompute is run.

## 7. Con1 three-stage schedule (unchanged logic, RoboTwin data)

Reuse the existing schedule in `src/openpi/con1/optimization.py`:

- Stage 1 (0-2k): train only `con1_delta_head` + `con1_cross_attention`; action
  expert and output projection frozen.
- Stage 2 (2k-7k): additionally train action blocks 14-17 and `action_out_proj`.
- Stage 3 (7k+): add the action-sensitivity coupling, `sgr_beta` ramping to 0.5.

Weight loader: base params from the RoboTwin 19999 checkpoint, Con1 head either
zero-initialized or from the existing LIBERO-trained delta head (to be decided;
recommend zero-init first for a clean RoboTwin start). Base parameter leaves
remain frozen; only `con1_*`, upper action blocks, and `action_out_proj` are
trainable.

## 8. Launch plan

1. `robotwin` branch: add RoboTwin data config + Con1 config + weight loader.
2. Pin the model config by checkpoint shape-match.
3. Download/prepare the RoboTwin Clean-20 dataset and V-JEPA cache.
4. One four-GPU smoke update; require finite loss/gradients and exact
   zero-residual equivalence at step 0.
5. Launch the full run under supervisor, save checkpoints every 1k, and log the
   same metrics as the LIBERO Con1 run.

## 9. Risks / decisions needed

- Confirm `vjepa_aux` (not plain pi0.5) 19999 is the intended base.
- Confirm `action_dim`/`action_horizon`/`vjepa_future_offset` and the exact
  Clean-20 episode filter; these are the main remaining unknowns.
- Confirm whether the Con1 delta head starts zero-initialized or is transferred
  from the LIBERO-trained head.
