# LIBERO-Plus full sweep: Con1 + Con2 candidate

Model under test: **JEPA-WAM PI0.5 40k base + Con1 + Con2 + whole-prefix VLM
context**, fine-tuned 4500 steps from the released 40k checkpoint (16 layers
frozen, batch 64, lr 1e-5 x2 on the Con1/Con2 modules, stage 1 boundary at step
2000 so the action expert's last four blocks also train).
Inference is pure feed-forward: the Con2 refinement runs
`Δ̃ = Δ̂ + F(Δ̂, z_t)` with the trained `F`, and `z_t` is recomputed online from
the current camera frames. **No test-time gradient adaptation (TTT) is used.**

## 1. Four suites, complete coverage

Every task of LIBERO-Plus `libero_10`, `libero_spatial`, `libero_object` and
`libero_goal` was evaluated: 1 trial per task, seed 7, replan 5,
**10030 / 10030 tasks** (no coverage gap).

| suite | evaluated | total | success | category-macro |
|---|---|---|---|---|
| libero_10 | 2519 | 2519 | 80.2% | 82.1% |
| libero_spatial | 2402 | 2402 | 92.1% | 92.6% |
| libero_object | 2518 | 2518 | 87.8% | 88.3% |
| libero_goal | 2591 | 2591 | 80.6% | 82.2% |
| **pooled** | **10030** | **10030** | **85.1%** | — |

## 2. Perturbation category x suite

| suite | Background | Camera | Language | Light | Layout | Robot | Noise |
|---|---|---|---|---|---|---|---|
| libero_10 | 94.8% | **47.7%** | 91.6% | 93.4% | 92.3% | 79.4% | 75.7% |
| libero_spatial | 99.2% | 75.0% | 96.2% | 98.6% | 98.2% | 88.3% | 92.6% |
| libero_object | 92.3% | 76.8% | 83.3% | 98.3% | 91.1% | 80.4% | 95.7% |
| libero_goal | 95.7% | 64.2% | 72.7% | 97.5% | 71.8% | 82.2% | 91.6% |
| **pooled** | **95.5%** | **65.5%** | **85.8%** | **97.0%** | **87.7%** | **82.4%** | **88.4%** |

Pooled sample sizes: Background 1076, Camera 1599, Language 1537, Light 1142,
Layout 1525, Robot 1550, Noise 1601.

## 3. Difficulty x suite

| suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| libero_10 | 94.7% | 97.1% | 90.9% | 82.0% | **53.5%** |
| libero_spatial | 97.1% | 92.1% | 89.8% | 93.1% | 85.3% |
| libero_object | 93.9% | 94.9% | 94.3% | 86.5% | 73.4% |
| libero_goal | 96.0% | 95.1% | 86.9% | 74.4% | **52.2%** |
| **pooled** | **95.5%** | **94.6%** | **90.5%** | **83.9%** | **61.9%** |

Pooled sample sizes: L1 1644, L2 2202, L3 2094, L4 1886, L5 2083, plus 121 tasks
that LIBERO-Plus leaves without a difficulty label (95.9%, listed separately).

## 4. Paired significance on libero_10 (vs the official 40k base policy)

The baseline side was stopped part-way, so this is a **paired comparison over the
overlap** (1061 episodes, same tasks and seeds):

| group | n | candidate | baseline | delta | p |
|---|---|---|---|---|---|
| Background Textures | 289 | 94.1% | 93.4% | +0.7pp | 0.83 |
| Camera Viewpoints | 370 | 49.5% | 45.9% | +3.5pp | 0.073 |
| Robot Initial States | 393 | 79.4% | 77.4% | +2.0pp | 0.31 |
| **L5 (hardest)** | **380** | **47.4%** | **41.8%** | **+5.5pp** | **0.0086** |
| **overall** | **1061** | **73.0%** | **70.9%** | **+2.2pp** | **0.040** |

## 5. Mechanism check: held-out flow loss

Paired over 72 batches x 8 samples, reference = the same checkpoint with the Con1
correction zeroed at its output projection (exact base policy):

| variant | flow | delta vs base | t |
|---|---|---|---|
| base (correction = 0) | 0.016516 | — | — |
| Arm A (Con1 only) | 0.016334 | −1.10% | 1.5 |
| **Arm B (Con1 + Con2 + VLM context, the deployed model)** | 0.016327 | −1.14% | 1.6 |

## 6. Comparison with the PACE reference table

| category | PACE reference | this run | delta |
|---|---|---|---|
| Camera Viewpoints | 75.2 | 65.5 | **−9.7** |
| Robot Initial States | 82.1 | 82.4 | +0.3 |
| Language Instructions | 86.4 | 85.8 | −0.6 |
| Light Conditions | 96.9 | 97.0 | +0.1 |
| Background Textures | 95.1 | 95.5 | +0.4 |
| Sensor Noise | 87.5 | 88.4 | +0.9 |
| Objects Layout | 88.7 | 87.7 | −1.0 |
| **overall** | **87.4** | **85.1** | **−2.3** |

The single large gap is **Camera Viewpoints (−9.7pp)**, and it is concentrated in
`libero_10` (47.7%) while `libero_object`/`libero_spatial` reach 75-77%. That is
the most actionable direction for the latent-OOD story.

## 7. Caveats

1. One trial per task: 10030 episodes give roughly +-1pp resolution overall, and
   a single category of ~1500 tasks about +-2.5pp. No multi-seed replication yet.
2. The paired table covers 1061 of 2519 `libero_10` tasks because the baseline
   side was stopped early; the four-suite absolute table has full coverage.
3. Infrastructure failures (policy-server timeouts) are excluded from every
   number and their tasks were re-run; the journals also carry the failed
   attempts, so any re-analysis must apply the same filter.
4. Policy servers silently stop answering `/healthz` after ~6-9 h under load;
   long sweeps should rotate them. Two batches of affected episodes were
   re-run here.

## 8. Artifacts

| item | path |
|---|---|
| candidate checkpoint | `checkpoints/pi05_libero_con1con2_ctx_40k/libero_b_full/4499` |
| Con1-only arm | `checkpoints/pi05_libero_con1_adapter_livecross_40k/libero_a_con1/4499` |
| Con1 delta head | `checkpoints/con1_anchored_head_40k/checkpoint_012000.msgpack` |
| latent cache | `con1/anchored_40k_features_v1` (1693 episodes, 69 GB) |
| journals | `ts_JEPA_libero/data/libero-eval/con1con2_plus_full/` |
| summary JSON | `artifacts/four_suite_summary.json` |
| sweep scripts | `run_libero10_fill.sh`, `run_remaining_suites.sh`, `summarize_all_suites.py` |
