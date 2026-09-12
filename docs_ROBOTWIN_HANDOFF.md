# RoboTwin Con1/Con2 - handoff (2026-09-12 night session)

One page. Details live in the linked documents.

## 1. What is running

| stage | state |
|---|---|
| arm A (Con1 livecross, 15k) | **finished** 22:54, best `con1_delta_nmse` 0.737 |
| arm B (Con1+Con2+ctx, 15k) | running, ~step 6.0k, ETA 01:00-01:30 |
| arm C (action-conditioned head, 9k) | queued, ~02:50 |
| arm D (head 5x LR, 9k) | queued, ~04:40 |
| 10-row paired probe | queued, ~05:35 |

Supervisor: `run_robotwin_post.sh` (PID 42017) chains C -> D -> probe. Disk
122 GB free; four GPUs at ~17.2 GB each.

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
