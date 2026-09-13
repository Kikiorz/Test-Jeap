# Claims, evidence, and the wording that survives

Every number produced in the 2026-09-12/13 RoboTwin session, sorted by whether it
can actually be claimed. "corrected" means Holm-Bonferroni over the nine
pre-registered comparisons at alpha = 0.05 (`scripts/robustness_report.py`).

## Supported

| claim | evidence | wording that is safe |
|---|---|---|
| **The correction reads the latent** | shuffling the predicted delta across the batch (same shape, scale, wrong sample; no parameter change) costs **+3.59%, t = +3.55**, n = 200 | "Con1's correction depends on the predicted latent: replacing it with a permuted one degrades the action loss by 3.6%." |
| **The delta path alone is insufficient** | keeping only the delta path (adapter removed) costs **+4.97%, p = 0.0051** against arm A, the only mechanism comparison surviving Holm; the reverse (adapter only) is +1.27%, p = 0.154 | "Removing the latent-free adapter while retaining the latent path significantly degrades the objective." Note this is an asymmetry in *significance*: the two diagnostic rows compared directly with each other differ by -3.53% at **t = -2.21**, which is not resolved, so "the adapter matters more" is only directional |
| **No variant beats another** | arm D vs C is -0.19% (t = -0.21) and arm B vs C is -0.36% (t = -0.46) at n = 200 | "The three modified arms are mutually indistinguishable at this budget." |
| **The latent head can be made better than linear** | arm D's head reaches **0.646** NMSE against a measured linear ridge ceiling of **0.659** | "The delta head can be trained past the linear predictor fitted on the same features." |
| **Better latent prediction does not transfer** | arm B +0.107 NMSE and arm D +0.198 NMSE against arm A, both clearing the pre-registered head rule; neither clears the action rule (B vs A p = 0.101, D vs A p = 0.290) | "Substantial improvements in latent-prediction accuracy did not produce a significant action improvement, across two independently modified arms." |
| **Relaxing the residual cap hurts** | budget 0.05 -> 0.20 makes the correction 2.8x larger and the flow **+57.5%**, paired **t = +7.41** at **n = 200** (0.002839 -> 0.004472, same 200 batches) | "The correction is at its useful magnitude; increasing the residual budget degrades the objective." |

## Refuted or unsupported

| claim | why it fails |
|---|---|
| "Con2 improves the action" | `b vs a` = -1.46%, **p = 0.101**, does not survive correction. Across three training points the paired t is +0.16, -1.47, +0.49 |
| "Training the head properly is the lever" | arm D fails the pre-registered rule it was built for: D vs A is **-1.29%, t = -1.06**, bar was t >= 3 |
| "Con1 gives an 11.8% gain" | that is one checkpoint. Arm A vs base: **-11.77% (t=-7.0) at 4.5k, -1.01% (t=-0.94) at 9k, -2.91% (t=-3.14) at 15k**. Not monotone; the 9k point is not significant |
| "Our model beats the released policy by 98%" | the `released base` row scores 0.195 because it is evaluated on the full 2,500-episode release while it was trained only on the 20-task Clean subset. That row is out of distribution, not a result |
| "The latent carries no action information" (as a claim about *our* pipeline) | it does: the shuffle control above contradicts the stronger reading. The correct version is narrower - *accuracy* of the latent does not transfer |
| "The residual budget is throttling the correction" | the budget sweep says the opposite; at 0.2 the flow is 60% worse |

## Still open

| question | what it needs |
|---|---|
| Does any of this hold closed-loop? | the RoboTwin simulator. Everything here is flow loss on the training distribution |
| Does it hold on the paper's 20 Clean tasks? | the release carries no task identity (raw parquet columns are state/action/timestamp/frame/episode/index/task_index), so a Clean-20 filter needs an external mapping |
| Is the head rule met because of the head, or because of co-adaptation? | the offline-head graft is *worse* when inserted (+1.18%, ns), so the fusion path is adapted to the jointly trained head; separating the two needs a fresh arm with a warm-started head |

## The one-sentence version

The correction is causal and reads the latent, but improving the latent's
accuracy does not reach the action - measured four independent ways, and it is
why Con2, whose premise is that accuracy transfers, measures inert at every
training point tested.
