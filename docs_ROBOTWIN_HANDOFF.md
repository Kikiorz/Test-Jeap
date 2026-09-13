# RoboTwin Con1/Con2 - handoff (2026-09-12 night session)

One page. Details live in the linked documents.

## 1. What is running

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
| arm B (Con1+Con2+ctx, 15k) | running, ~step 6.0k, ETA 01:00-01:30 |
| arm C (action-conditioned head, 9001) | running, ETA ~03:35 |
| arm D (head 5x LR, 9001) | queued, ETA ~05:25 |
| 10-row paired probe | queued, ETA ~06:05 |

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
