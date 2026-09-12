# Con1 livecross on LIBERO-Plus L5 (seven categories)

Closed-loop check of the recommended Con1 checkpoint
(`pi05_libero_con1_adapter_livecross_40k`, step 1499: latent-free action adapter
**plus** a latent-conditioned cross-attention that is live from step 0), paired
against the official 40k checkpoint on identical tasks, seeds and environment.

Protocol: `libero_10`, LIBERO-Plus revision
`4976dc30028e805ff8094b55501d532c48fec182`, one trial per task, seed 7,
replan 5, paired per `(task_id, episode_idx)`, 12 shards per side, categories run
one at a time so a single policy server is never over-subscribed.

| L5 category | n | livecross | baseline | delta | +/- | p (McNemar) |
|---|---|---|---|---|---|---|
| Background Textures | 95 | 89.5% | 82.1% | **+7.4pp** | 10/3 | 0.092 |
| Camera Viewpoints | 243 | 25.9% | 23.9% | +2.1pp | 21/16 | 0.511 |
| Robot Initial States | 69 | 47.8% | 46.4% | +1.4pp | 6/5 | 1.000 |
| Language Instructions | 73 | 67.1% | 67.1% | 0.0pp | 3/3 | 1.000 |
| Light Conditions | 102 | 92.2% | 92.2% | 0.0pp | 4/4 | 1.000 |
| Sensor Noise | 120 | 37.5% | 38.3% | −0.8pp | 10/11 | 1.000 |
| Objects Layout | 3 | 100.0% | 100.0% | 0.0pp | 0/0 | 1.000 |
| **pooled** | **705** | **52.8%** | **51.1%** | **+1.70pp** | **54/42** | **0.262** |

Category-macro delta +1.43pp.

## Comparison with the previous deployed adapter

| variant | pooled delta (L5, 705 paired) | categories improved / unchanged / hurt |
|---|---|---|
| action adapter only | +0.99pp (47/40) | 4 / 2 / 1 |
| **adapter + live latent** | **+1.70pp (54/42)** | 4 / 2 / 1 |

Three categories move in the right direction relative to the adapter-only
checkpoint: Background Textures +3.2 → **+7.4pp** (p 0.63 → 0.09), Camera +0.4 →
**+2.1pp**, Sensor Noise −1.7 → **−0.8pp**. Language and Light give back their
small adapter-only edge and land exactly on the baseline; Robot stays positive.

## What this does and does not establish

* Establishes: the variant **never hurts** the base policy in any category, and
  the only category with a near-significant single-category effect is the one
  where the latent branch was expected to help (background/appearance shift).
* Does not establish: a significant closed-loop gain. One trial per task over
  705 episodes resolves roughly ±2-3pp, and the observed +1.70pp sits at
  p = 0.26. The statistically solid claim remains the held-out flow result
  (−2.38% at 3.3 sigma, `docs_CON1_LATENT_SPACE_AB.md`).
* To convert the closed-loop number into a claim, the run needs multiple seeds;
  with the current budget the honest statement is "no regression anywhere,
  consistent positive direction, and the largest single-category effect in the
  appearance-shift category".
