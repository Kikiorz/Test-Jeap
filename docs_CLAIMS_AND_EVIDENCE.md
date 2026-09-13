# Claims, evidence, and the wording that survives

## Cross-platform correction: on LIBERO the flow proxy under-powers the method

The LIBERO-Plus sweep (`docs_LIBERO_PLUS_RESULTS.md`, same session, other lane)
finished while this document was being written, and it changes how the RoboTwin
numbers below should be read.

On LIBERO, *closed-loop* says the method works:

| group | n | candidate | base | delta | p |
|---|---:|---:|---:|---:|---:|
| L5 (hardest) | 380 | 47.4% | 41.8% | **+5.5pp** | **0.0086** |
| overall | 1061 | 73.0% | 70.9% | **+2.2pp** | **0.040** |

The same model's *flow* check does not:

| variant | flow | delta vs base | t |
|---|---:|---:|---:|
| Arm A (Con1 only) | 0.016334 | -1.10% | 1.5 |
| Arm B (Con1 + Con2 + ctx, deployed) | 0.016327 | -1.14% | 1.6 |

**The identical model is a significant +2.2pp closed-loop gain and a
non-significant -1.1% flow gain.** So flow loss under-powers this method - on a
platform where the answer is known.

**Consequence for everything below**: every RoboTwin result in this document is a
*flow* measurement, and they are therefore **lower bounds on resolution**. The
"no difference" outcomes (all five arms within 0.7pp at 3k; B/C/D/E vs A not
significant) are real for the flow metric, but they do not rule out
closed-loop differences of the size LIBERO shows. The honest status of the
RoboTwin comparison is "not resolved by the metric we have", and the only way to
settle it is the simulator (see `docs_NEXT_STEPS.md`, option C, whose case this
result strengthens).

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
| **Accuracy is not the mediator, measured within the experiment** | per-batch correlation between the correction's benefit and that batch's latent NMSE over 200 batches: **pearson r = -0.066 (p = 0.36)**, spearman -0.064; median split +5.3e-5 vs +5.2e-6, t = +0.77, p = 0.44 | "The correction's benefit is not significantly related to how accurately that sample's latent was predicted." (A directional hint that the better-predicted half benefits ~10x more exists but is not resolvable at this budget.) |

**Accuracy is not the mediator, on any of the three axes we can measure.** The
within-experiment correlation is zero; across arms the worst latent (C, 0.827)
gets a better benefit than A (0.774); and across arm A's own training the biggest
benefit belongs to its worst latent (0.874 at 4.5k, -11.8%). This is the single
most consistent result of the session and it is what rules Con2 out: its premise
is that accuracy transfers, and accuracy does not track the benefit anywhere.
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

## RETRACTED: the "threshold pattern" does not survive two consistency checks

## The finding that dominates everything else: the benefit decays with training

Arm A against its own silence control. Every point below is n = 200, same seed,
same protocol, and the base row is the *same checkpoint* with the correction
silenced, so the comparison is paired within checkpoint:

| A checkpoint | benefit vs base | paired t |
|---|---:|---:|
| **3,000** | **-14.29%** | **-19.02** |
| 4,500 (n=48, earlier protocol) | -11.77% | -7.0 |
| **6,000** | **-3.75%** | **-4.67** |
| 9,000 | -1.01% | -0.94 |
| **12,000** | **-0.87%** | **-1.05** |
| 14,999 | -2.83% | -3.6 |

One arm, one measurement protocol, a **16x range** - from an extremely strong
effect (t = -19) to nothing - driven purely by which checkpoint is scored.

This reframes the whole session:

1. **The correction behaves as a warm-up accelerant.** It is worth 14% while the
   policy is still poorly trained and almost nothing once the base path has
   converged; there is simply less left for a residual to fix.
2. **Every arm comparison in this document was made at step 9,000, which is
   where the effect is near its minimum (-1.0%).** That is why B, C, D and E all
   look indistinguishable: at a point where the whole effect is 1%, no
   modification of it can be resolved.
3. **Any future variant comparison should be scored early (3k-6k)**, where the
   effect is 4-14% and the signal-to-noise is an order of magnitude better.

### ...and the variants are genuinely identical there

Scoring every arm at 3,000 steps - where the effect is 14% rather than 1% - and
at 6,000:

| arm | vs base @3k | t | vs base @6k | t |
|---|---:|---:|---:|---:|
| A (unmodified Con1) | -14.29% | -19.0 | -3.75% | -4.7 |
| B (whole-prefix ctx) | -14.06% | -17.8 | -4.13% | -5.4 |
| C (action-conditioned head) | -14.37% | -19.9 | -3.79% | -4.1 |
| D (head LR x5) | -13.77% | -17.4 | -1.60% | -1.5 |
| E (ctx + head LR) | -13.71% | -18.2 | -3.19% | -3.6 |

At 3,000 steps all five arms sit within 0.7 percentage points of each other at
t ~ 18. This is not a failure to resolve a difference - it is a difference-free
result measured with twenty times the signal-to-noise. **No tested modification
to Con1 changes what the correction does.**

Read together with the three-axis result below, the picture is: the correction's
benefit is large early, decays as the base converges, and is not explained by the
latent head's accuracy at any point.

An earlier draft of this document argued that the head only has to reach ~0.81
NMSE and that further accuracy is worthless, based on a five-arm cross-section.
That draft was wrong twice over, and both errors are worth recording.

**Error 1 - the NMSE values were single log lines.** `con1_delta_nmse` in
`metrics.jsonl` is one batch's value, logged every 10 steps, and it is noisy to
about +-0.03: the same arm reads 0.8192 at step 9031 and 0.8443 at step 9041. A
single line is not "the arm's NMSE". Using the median of the last 300 logged
values instead:

| arm | NMSE median (last 300) | NMSE near 9000 | vs base |
|---|---:|---:|---:|
| A | 0.7739 | 0.8175 | -1.01% |
| B | 0.6680 | 0.7087 | -2.44% |
| **C** | **0.8271** | 0.8056 | **-2.09%** |
| D | 0.6677 | 0.6459 | -2.27% |
| E | 0.6380 | 0.6157 | -2.27% |

**Arm C has the *worst* latent of any arm** (0.827, worse than A's 0.774) **and a
better flow than A** (-2.09% vs -1.01%). The variable that was supposed to
explain the pattern inverts it.

**Error 2 - arm A's own trajectory contradicts the pattern.** Its NMSE falls
monotonically across training while its benefit does not:

| A checkpoint | NMSE | A vs base |
|---|---:|---:|
| 4.5k | 0.8737 | **-11.77%** |
| 9k | 0.8192 | -1.01% |
| 15k | 0.7604 | -2.91% |

The largest benefit belongs to the *worst* latent. Whatever drives the
correction's benefit, it is not the head's accuracy - and that now holds on all
three axes available: across arms, across training steps, and within the
experiment (r = -0.07 between per-batch benefit and per-batch NMSE).

## Arm E: the two head levers are not additive

Arm E combines the two levers that each beat base on their own. The files are
distinct (different config, exp name, md5 and correction RMS) and the paired
comparison is exact:

| comparison (n = 200) | relative | t | p | Holm threshold | verdict |
|---|---:|---:|---:|---:|---|
| e vs base | -2.29% | -2.55 | 0.0116 | 0.0125 | **survives** |
| e vs a | -1.29% | -1.14 | 0.257 | 0.0167 | not significant |
| e vs b | +0.17% | +0.20 | 0.845 | 0.0250 | not significant |
| **e vs d** | **0.00%** | **0.00** | **0.999** | 0.0500 | **identical** |

Adding the whole-prefix context to the head-LR arm changes nothing. The levers
saturate.

### This corrects an over-statement of mine

The full four-arm picture:

```
A (unmodified Con1)        vs base  -1.01%   p = 0.347   not significant
B / D / E (head changed)   vs base  -2.3 to -2.5%   p < 0.012   all significant
```

So head quality **does** matter: every variant that changes the head clears the
silence control, and the unmodified Con1 does not, worth about 1.3% - saturating
at -2.3%.

**But the statistical trap has to be stated alongside it**: the pairwise
comparisons of B, D and E against A are *not* significant (p = 0.10 / 0.29 / 0.26).
"A is not significant and B/D/E are" is **not** the same as "B/D/E beat A", and
must not be written as if it were. The strongest defensible sentence is:

> Three variants that change the delta head each beat the silence control at
> p < 0.012 (-2.3% to -2.5%) while the unmodified Con1 does not (-1.0%, p = 0.35),
> and the two levers do not combine (E and D are identical to within 0.00%).

## Outlier robustness of the paired statistics

`scripts/robustness_trim.py` recomputes each comparison after dropping the 5% of
batches with the highest baseline flow - the hardest episodes, a bias-free trim.

| comparison | full t | after trimming the hardest 5% |
|---|---:|---:|
| a_only_residual vs a | +2.84 | **+5.04** |
| d vs base | -2.66 | -2.95 |
| b vs base | -2.99 | -2.49 |
| a vs base | -0.94 | -0.38 |
| a_grafted vs a | +1.33 | +2.24 |

Two readings that matter:

* The **one mechanism statement that survives Holm correction gets stronger**
  under trimming (+2.84 -> +5.04), so it is not carried by a few pathological
  episodes - the hard batches were masking it.
* **Arm A's null is robust too** (-0.38 on the easier 95%), so it is not an
  artefact of hard episodes either.

The script also reports a trim by |paired difference|; that column is a
winner's-curse filter that inflates |t| by construction and must not be quoted as
evidence of robustness. The docstring and the printed output both say so.

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
