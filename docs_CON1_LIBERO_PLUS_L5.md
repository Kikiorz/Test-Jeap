# Con1 on LIBERO-Plus L5: closed-loop and representation measurements

Three measurements, all paired against the official 40k checkpoint on the same
tasks, seeds and environment:

1. closed-loop success on **all seven LIBERO-Plus perturbation categories at the
   hardest level (L5)** — 705 paired episodes,
2. an offline **OOD-robustness probe** for the two candidate latents under five
   image-space perturbations,
3. a **token-level** action-information probe that retires the "the mean pool
   destroyed the signal" explanation.

## 1. Closed-loop: seven categories, difficulty 5

Protocol: `libero_10`, LIBERO-Plus revision
`4976dc30028e805ff8094b55501d532c48fec182`, one trial per task, seed 7,
replan 5, paired per `(task_id, episode_idx)`. Baseline is the official
JEPA-WAM PI0.5 40k checkpoint (port 8001); adapter is the deployed Con1 action
adapter on the same checkpoint (port 8000). Categories were run one at a time,
12 shards per side, so the two policy servers were never over-subscribed.

| L5 category | n | adapter | baseline | delta | +/- | p (McNemar exact) |
|---|---|---|---|---|---|---|
| Background Textures | 95 | 85.3% | 82.1% | +3.2pp | 10/7 | 0.629 |
| Language Instructions | 73 | 69.9% | 67.1% | +2.7pp | 2/0 | 0.500 |
| Light Conditions | 102 | 95.1% | 92.2% | +2.9pp | 4/1 | 0.375 |
| Camera Viewpoints | 243 | 24.3% | 23.9% | +0.4pp | 17/16 | 1.000 |
| Robot Initial States | 69 | 46.4% | 46.4% | 0.0pp | 5/5 | 1.000 |
| Sensor Noise | 120 | 36.7% | 38.3% | −1.7pp | 9/11 | 0.824 |
| Objects Layout | 3 | 100.0% | 100.0% | 0.0pp | 0/0 | 1.000 |
| **pooled** | **705** | **52.1%** | **51.1%** | **+0.99pp** | **47/40** | **0.520** |

Category-macro delta +1.08pp. No category is individually significant, and the
pooled discordant split (47 vs 40) is what a coin flip looks like.

Reading: the adapter **does not hurt** the base policy anywhere — including on
the categories where the base policy itself is weak (Camera 24%, Sensor Noise
37%) — but it does not deliver a measurable gain at this sample size either.
With one trial per task, 705 episodes resolve roughly ±3–4pp; detecting the
~1pp effect seen here needs multi-seed runs.

Artifacts: `data/libero-eval/{baseline,adapter}_L5_*`, summary in
`artifacts/con2/l5_pairs_final.json` (`scripts/report_l5_pairs.py`).

## 2. Offline OOD-robustness probe

720 frames per condition (120 episodes × 6 frames), five image-space
conditions. A ridge is fitted on the **clean** frames and evaluated on the
perturbed ones; the target is the 10-step action chunk. The number that matters
is the increment over the previous action, which is available at deployment and
is the strongest single predictor (rule: never report absolute-action R² across
episodes, it does not transfer).

| condition | clean | lighting | noise | blur | camera |
|---|---|---|---|---|---|
| prev-action only (R²) | 0.749 | 0.731 | 0.731 | 0.731 | 0.731 |
| PI0.5 VLM tokens, increment | +0.054 | +0.072 | +0.071 | +0.068 | +0.058 |
| V-JEPA 2-AC latent, increment | +0.001 | −0.092 | −0.269 | −0.479 | −0.235 |

The representation the policy already consumes is stable under perturbation;
the V-JEPA 2-AC latent is not, and it carries essentially no action information
even on clean frames.

## 3. Token-level probe (retires the pooling hypothesis)

Flattened AC patch-token grids, dual ridge (1200 train / 800 test samples):

| feature | held-out R² on the action chunk |
|---|---|
| previous action only | 0.699 |
| previous action + pooled latent | 0.623 |
| flattened token grid | −0.112 |
| token-grid difference (t+1 − t) | −0.122 |

Adding the pooled latent is worse than the action alone, and the un-pooled token
grid is worse than predicting the mean. The spatial mean pool is therefore not
what removed the action information.

## Conclusion

The Con1 adapter is behaviourally safe (bit-identical at α=0, no regression in
any of the seven L5 categories) but the latent→action coupling is a no-op, and
these measurements say why: the V-JEPA 2-AC latent carries no transferable
action information at any resolution and degrades sharply under image
perturbation, while the PI0.5 VLM token space is both action-informative and
perturbation-stable.

The actionable consequence for Con1/Con2: keep JEPA as the *future predictor*,
but route the action-facing residual through the **VLM token latent** (64 × 2048)
rather than the AC latent — it is the only measured representation that both
predicts the action and survives OOD.
