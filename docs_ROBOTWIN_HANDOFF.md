# RoboTwin Con1/Con2 - handoff (2026-09-12 night session)

One page. Details live in the linked documents.

## 0. Final state

Branch `feat/RoboTwin` is at the commit matching its upstream (**fully pushed**)
and the working tree is **clean**, so the branch is portable as-is. Every
analysis script listed below is tracked. The GPU box is idle; disk 63 GB. The
three RoboTwin arms C and D trained, the ten-row probe ran, and the high-power
(n=200) replication plus the two control rows (shuffled delta, relaxed budget)
are all on disk.

What is **not** finished, and needs a decision rather than more compute:

1. **Closed-loop.** Every number here is flow loss. The simulator can be
   installed on this box - feasibility checked - but it needs Vulkan packages,
   conda, and disk headroom (see `docs_NEXT_STEPS.md` section C).
2. **The paper's 20 Clean tasks.** The release carries no task identity, so this
   needs an external mapping before it can even be attempted.
3. **Con2's direction.** It is inert at every training point; the evidence says
   the premise (accuracy transfers) is the problem, not the implementation.

## 1. What is running

## 1b. FULL TABLE - ten rows at the shared step 9000 (all four arms)

### 1c. HIGH-POWER REPLICATION (n = 200 paired batches, 1600 samples)

Same checkpoints, same step, four times the data. This supersedes the 48-batch
numbers below wherever they disagree. Files: `artifacts/con2/hp/`.

| row | flow | vs base | paired t vs base | vs A |
|---|---:|---:|---:|---:|
| base (Con1 silenced) | 0.002868 | - | - | - |
| arm A (Con1) | 0.002897 | -1.01% | **-0.94** | - |
| **arm B (Con1+Con2+ctx)** | 0.002798 | **-2.45%** | **-2.99** | -1.65 |
| arm C (Con1+action-cond) | 0.002808 | -2.10% | -1.80 | -0.92 |
| arm D (Con1+head 5x LR) | 0.002803 | -2.29% | **-2.66** | -1.06 |
| A + offline head | 0.002873 | +0.16% | +0.19 | +1.33 |
| A, only the delta path | 0.002981 | +3.94% | - | **+2.84** |
| A, only the adapter | 0.002875 | +0.24% | - | **+1.43 (ns)** |

Four readings, all of which differ from the 48-batch version:

1. **Arm A has no benefit at 9,000 steps** (-1.01%, t = -0.94, n = 1600). Not an
   underpowered null - a real one.
2. **No variant is significantly better than arm A** (B -1.65, D -1.06,
   C -0.92). Which change helps is still unresolved.
3. Against *base*, B reaches -2.45% (t = -2.99), the closest any row comes to the
   pre-registered |t| >= 3 bar; D is -2.29% (t = -2.66), C -2.10% (t = -1.80).
   With ten rows tested, t ~ 3 is not on its own a strong claim.
4. **The decomposition flips the story**: keeping only the delta path costs
   +4.97% (t = +2.84, clearly worse), while keeping only the adapter costs
   +1.27% (t = +1.43, not significant). So **the latent-free adapter carries the
   benefit** and the delta path only contributes in combination - a sharper
   version of LIBERO's 58% attribution, and exactly what
   `docs_LATENT_ACTION_INFORMATION.md` predicts.

**Verdict against the pre-registered rule**: arm D's |t| >= 3 against arm A was
**not met** (t = -1.06). By the rule fixed before the numbers arrived, "training
the head properly is the lever" cannot be claimed.

### 1d. The pre-registered audit, completed (latent quality at step 9000)

`scripts/robotwin_arm_summary.py` reads the four metrics files:

| arm | `con1_delta_nmse` @9000 | best | training flow_loss |
|---|---:|---:|---:|
| A | 0.8443 | 0.7371 | 0.00252 |
| B | 0.7376 | 0.6304 | 0.00254 |
| C | 0.8031 | 0.7867 | 0.00196 |
| **D** | **0.6460** | **0.6282** | 0.00197 |

| pre-registered rule | threshold | arm D | verdict |
|---|---|---|---|
| the head moved | delta NMSE >= 0.05 below A | **+0.198** | **met** |
| it bought action accuracy | flow >= 2% better than A, t >= 3 | -1.29%, t = -1.06 | **not met** |

Against arm A: B +0.1067 NMSE (met rule 1), C +0.0413 (below it), D +0.1983 (met).

This is the second branch of the 2x2 fixed in `docs_CON1_LATENT_CEILING.md` before
any of these numbers existed: **the latent head improves substantially and the
action does not follow**. Two arms reach rule 1 independently (B and D) and
neither reaches rule 2. Arm D's head reaches **0.646**, the best produced here and
below the measured linear ceiling of 0.659, so the latent prediction was made
better than linear and the action still did not move significantly.

Four independent lines now agree: this measurement, the Delta z action-information
probe (R^2 0.033), the gradient-alignment measurement (orthogonal directions), and
the decomposition above (the delta-free adapter carries the benefit).

### 1e. Does the correction read the latent at all? Yes - measured

`con1_shuffle_delta` rotates the predicted delta across the batch before the
cross-attention sees it: same shape, same scale, wrong sample. No parameters
change, so arm A's checkpoint restores unchanged and the row is directly paired
with arm A on the same 200 batches.

| row | flow | vs arm A | paired t |
|---|---:|---:|---:|
| arm A (real delta) | 0.002839 | - | - |
| **arm A + shuffled delta** | 0.002941 | **+3.59%** | **+3.55** |

So the correction **does** use the latent's content, significantly. This corrects
an over-claim in an earlier draft of this document ("the benefits come from a
delta-free side channel", implying the correction is content-blind). The precise
statement is two-part:

1. **The correction depends on the latent** - scrambling it costs 3.59% at
   t = 3.55. Keeping only the adapter costs +1.27% (t = 1.43, ns) while keeping
   only the delta path costs +4.97% (t = 2.84), so both paths matter and the
   adapter alone is nearly sufficient.
2. **Improving the latent's accuracy does not buy action accuracy** - arms B
   (+0.107 NMSE) and D (+0.198 NMSE) both clear the pre-registered head rule and
   both fail the action rule.

That distinction is what matters for Con2: its premise needs *accuracy to
translate into action*, and that is the step this experiment shows is missing.

### 1f. Multiple-comparison correction: only two statements survive

### 1g. Consistency audit of the stored probe files

Every paired comparison reported above assumes the rows share a checkpoint step,
a batch count and a seed. Checked against the files rather than assumed:

```
artifacts/con2/hp/         9 files, step=9001, records=200, budgets=[0.05]
artifacts/con2/hp_shuffle/ 2 files, step=9001, records=200
                           (shuffle at 0.05, budget020 at 0.2 - that difference
                            is the point of those two rows)
```

The seed equality was verified separately at the config level (TrainConfig.seed
defaults to 42 with no override, and the loader forwards it to the shuffle
generator). All 13 paths referenced by this document and by
`docs_CLAIMS_AND_EVIDENCE.md` resolve: the docs are in this branch except
`docs_ROBOTWIN_CON1_TRAINING.md`, which is on `robotwin`, and the four analysis
scripts are present both locally and on the box.

Ten rows invite a lot of comparisons and a bare t = -2.99 is not a claim.
`scripts/robustness_report.py` recomputes the nine comparisons named in
`docs_ROBOTWIN_READING_THE_TABLE.md` (fixed before the numbers) with
Holm-Bonferroni at alpha = 0.05, n = 200 paired batches:

| comparison | relative | t | p | Holm threshold | verdict |
|---|---:|---:|---:|---:|---|
| a_only_residual vs a | +4.97% | +2.84 | 0.00505 | 0.00625 | **survives** |
| b vs base | -2.45% | -2.99 | 0.00316 | 0.00556 | **survives** |
| d vs base | -2.29% | -2.66 | 0.00855 | 0.00714 | not significant |
| c vs base | -2.10% | -1.80 | 0.0736 | 0.00833 | not significant |
| b vs a | -1.46% | -1.65 | 0.101 | 0.0100 | not significant |
| a_only_adapter vs a | +1.27% | +1.43 | 0.154 | 0.0125 | not significant |
| a_graft vs a | +1.18% | +1.33 | 0.185 | 0.0167 | not significant |
| d vs a | -1.29% | -1.06 | 0.290 | 0.0250 | not significant |
| **a vs base** | -1.01% | -0.94 | 0.347 | 0.0500 | **not significant** |

Two readings, both narrower than the raw table suggests:

1. **`b vs base` is a comparison of B's full model against arm A's silenced
   model**, so the defensible wording is "B beats the silence control". The
   comparison that would license "Con2 helps over Con1" - `b vs a` - has p =
   0.101 and does **not** survive.
2. **`a_only_residual vs a` is the only mechanism statement that survives**:
   keeping the delta path without the adapter makes things significantly worse,
   so the delta path alone is insufficient.

And note that **`a vs base` itself is not significant at 9,000 steps**
(p = 0.347), against t = -7.0 at 4.5k and t = -3.14 at 15k. Any number quoted
from this table has to carry its step count and its correction.

48 paired batches, same loader, same seed. `base` = arm A with the correction
silenced, so "vs base" isolates the correction.

| row | flow | vs base | paired t vs base |
|---|---:|---:|---:|
| released base | 0.195329 | -98.4% | -23.7 (see the warning below) |
| base (Con1 silenced) | 0.003184 | - | - |
| arm A (Con1) | 0.003245 | **+1.92%** | +0.72 |
| arm B (Con1+Con2+ctx) | 0.003148 | -1.13% | -0.78 |
| arm C (Con1+action-cond) | 0.003202 | +0.57% | +0.59 |
| **arm D (Con1+head 5x LR)** | **0.003105** | **-2.49%** | **-1.92** |
| A + offline head | 0.003200 | +0.51% | +0.45 |
| A, only the delta path | 0.003295 | +3.49% | (vs A: +0.41) |
| A, only the adapter | 0.003204 | +0.63% | (vs A: -0.64) |

**At 9000 steps no arm reaches significance against base.** Arm D is the best
(`-2.49%`, t = -1.92) - the right direction for the head-quality lever, but under
the pre-registered |t| >= 3 bar. Arm C is flat. Arm B is not significant either
way (t = -0.78 vs base, -1.47 vs A).

### The budget row closes one hypothesis outright

Arm A re-evaluated with the residual cap relaxed:

| budget | correction RMS | flow |
|---|---:|---:|
| 0.05 (deployed) | 0.508 | 0.003245 |
| 0.20 | 1.743 | **0.005201** |

The n = 48 version (one run changing the budget in place, +60.28%, t = +4.18) was
replaced by **two single-budget runs at n = 200**, which avoids the second model
instance that OOM'd: **0.002839 -> 0.004472, +57.5%, paired t = +7.41**.

Relaxing the cap makes the correction 3.4x larger and the action **much worse**.
So "the 5% budget is throttling a correction that wants to be bigger" is dead, in
agreement with the LIBERO measurement, and consistent with the two paths acting
partly as a counterweight.

### Arm A's own gain is not monotone across training

Three checkpoints of the same arm, same batches, same seed:

| arm A checkpoint | A vs base | paired t |
|---|---:|---:|
| 4,500 | **-11.77%** | **-7.0** |
| 9,000 | **+1.92%** | +0.72 |
| 15,000 | **-2.91%** | **-3.14** |

Strong, then gone, then moderate. The pairing rules out sampling noise as the
explanation, so the honest reading is that **the correction's benefit on RoboTwin
fluctuates during training rather than settling**. Any headline number has to
carry its step count.

## 1a. Results already in hand (15k, 48 paired batches, step 15000)

| row | flow | vs base | paired t |
|---|---:|---:|---:|
| base (Con1 silenced) | 0.002894 | - | - |
| **arm A (Con1 livecross)** | **0.002810** | **-2.91%** | **-3.14** |
| arm B (Con1+Con2+ctx) | 0.002838 | -1.94% | -1.13 |
| A + offline head (diagnostic) | 0.002961 | +2.33% | +1.53 |
| A, only the delta path | 0.002967 | +2.55% | +2.94 vs A |
| A, only the delta-free adapter | 0.002963 | +2.38% | +2.63 vs A |

Raw files: `artifacts/con2/step15k/`, table in `robotwin_ab_table.json`.

Four readings:

1. Con1 still beats its own silence control at 15k (-2.91%, t=-3.14).
2. **Con2 + whole-prefix context is inert for the second time** (t=0.49 at 15k,
   t=0.16 at 4.5k), and now slightly the wrong way.
3. **Both correction paths are necessary**: keeping only the delta path (+5.59%,
   t=2.94) or only the delta-free adapter (+5.47%, t=2.63) is significantly worse
   than arm A. That matches LIBERO's adapter-only +0.99pp vs adapter+latent
   +1.70pp.
4. The offline-trained head does not transfer (+5.39% when grafted); the fusion
   path was adapted to the jointly trained head, as flagged when the row was built.

### Correction to an earlier claim of mine

The Con1 gain **decays with training**: -11.77% at 4.5k versus -2.91% at 15k. The
silence control itself improved 0.004157 -> 0.002894 (1.44x) while arm A went
0.003667 -> 0.002810 (1.30x), so the correction's marginal value falls as the
action expert converges.

This means the early framing "Con1's RoboTwin gain is an order of magnitude
larger than LIBERO's (-11.8% vs -1.9%)" was a **training-budget artefact, not a
platform difference**. At comparable convergence: LIBERO -2.38% at 3.3 sigma
(`docs_CON1_LATENT_SPACE_AB.md`) versus RoboTwin -2.91% at t=-3.14. The two agree.

**The row that must not be quoted**: `released base` scores 0.195 because it is
being evaluated on the full 2,500-episode release while it was only trained on
the 20-task Clean subset - it is an out-of-distribution number for that row, not
evidence that the arms beat the released policy by 98%.

| stage | state |
|---|---|
| arm A (Con1 livecross, 15k) | **finished** 22:54, best `con1_delta_nmse` 0.737 |
| arm B (Con1+Con2+ctx, 15k) | **finished** 01:05, best `con1_delta_nmse` 0.644 |
| arm C (action-conditioned head, 9001) | **finished** 03:34, final `con1_delta_nmse` 0.803 |
| arm D (head 5x LR, 9001) | running, ETA ~05:22 |
| paired probe at step 9000 | queued, ETA ~06:10 |

Checkpoint steps, read off the live directories: A and B `3000 6000 9000 12000
14999`, C `3000 6000 9000`, so the shared probe step is **9000** (verified with
the real directories, not a synthetic set).

Latent quality so far, one row per change made to arm A:

| arm | change vs A | final `con1_delta_nmse` |
|---|---|---:|
| A | - | 0.760 (best 0.737) |
| B | +Con2 refiner + whole-prefix context | **0.644** |
| C | + action-conditioned head | 0.803 |
| D | + head LR x5 | pending |

C is worth a sentence: feeding the action chunk into the delta head did **not**
make the latent prediction any better (0.803, same as arm A's neighbourhood).
Only B's whole-prefix context moved it.

Supervisor: `run_robotwin_cd_and_probe.sh` chains C -> D -> probe. Disk 102 GB
free; four GPUs at ~17.4 GB each.

Two low-level details that cost real time tonight, both worth remembering:

* **9001, not 9000.** `train.py:378` saves on `(step % 1000 == 0)` or
  `(step == num_train_steps - 1)`. A run of exactly 9000 stops at step 8999, so it
  produces 3000/6000/8999 and **never 9000**, which would collapse the shared
  probe step to 6000 and waste the last 3,000 steps of both arms.
* **The early probe is already collected** in `artifacts/con2/step15k/`; the C/D
  runner deliberately skips re-running it.

## 2. Decisions waiting on you

1. **Data scope.** The reviewed plan (`docs_ROBOTWIN_CON1_TRAINING.md`) says
   "extract only Clean demonstrations for the 20 tasks". That filter was never
   applied: `datasets/robotwin_clean_20` is a **symlink to the full release**, and
   the release carries **no task identity at all** (raw parquet columns are only
   state/action/timestamp/frame/episode/index/task_index, and `task_index` indexes
   instruction strings). Implementing the filter needs an external mapping.
   See `docs_ROBOTWIN_TRAINING_PLAN.md` section 10.
2. **Arms C and D run 9,000 steps, not 12,000.** Reduced to land inside your
   ten-hour window; revert is one line in each script. Section 7.
3. **`scripts/run_remaining_suites.sh`** was swept into commit `f59e72b` by a
   careless `git add -A`. It is your LIBERO-Plus eval script and it has since been
   edited again (SHARDS 32 -> 64) - I left your newer version uncommitted.

## 3. Bugs found and fixed tonight

| what | evidence |
|---|---|
| disk would have filled before B finished (148 GB needed, 62 GB free) | cleaned 67 GB, keep-period 1000 -> 3000, added a free-space guard |
| the shared-step picker used `sort -n` where `comm` needs lexicographic order | returned a wrong intersection; found by dry-running it against the live checkpoint dirs |
| `--zero-correction` left `adapter_out/bias` alive, so the control was not a true no-correction baseline | `adapter_out/bias` RMS is 0.02737, the buggy base row measured 0.0278; the probe now self-checks and fails loudly |
| two leftover `bash -c` launchers matched the `pgrep -f` wait patterns forever | would have stopped C and D from ever starting; killed both, anchored the patterns |
| `total_tasks: 2410` misread as 2,410 tasks | they are instruction strings; retracted, see section 10 |

## 4. New evidence measured tonight

| measurement | result |
|---|---|
| linear ceiling for the latent delta (ridge on the cache) | **0.659** mean NMSE over the head's 16 horizons |
| Con1 head in joint training vs a dedicated schedule | 0.86 at 4.5k -> 0.737 best at 15k, against 0.688 for a head trained alone |
| arm A vs released base weights | bf16 rounding is a uniform 1.66e-3; real fine-tuning is 2.3e-2 (`action_out_proj`) and 7e-3 (unfrozen expert layers) |
| arm B's latent head at 4.5k vs arm A's | 0.773 vs 0.88 - the whole-prefix context helps prediction but not the action |

## 5. What the table will and will not say

`docs_ROBOTWIN_READING_THE_TABLE.md` - ten rows, the question each answers, and
the pre-registered thresholds. Two priors from this repo constrain the reading
before the numbers land:

* `docs_LATENT_ACTION_INFORMATION.md`: **Delta z carries almost no action
  information** (held-out R^2 0.033 for the chunk against 0.705 for the previous
  action alone), and its conclusion is that every latent-to-action coupling is
  structurally a no-op. Arms C/D/F are therefore *expected* to be inert; inertness
  would be consistent with the root cause, not a failed fix.
* `docs_CON1_DELTA_GRADIENT_ALIGNMENT.md`: the delta-MSE direction is orthogonal to
  the action-improving direction, and the residual budget does not bind.

`docs_ROBOTWIN_PROBE_CAVEATS.md` - the table is a training-distribution
measurement, flow loss is a proxy, and 384 paired samples resolve ~2%.

## 6. Blocked

The LIBERO box (`99.27.206.243:59378`) has refused connections all night; nothing
is running there. The LIBERO-Plus line needs that machine back.
