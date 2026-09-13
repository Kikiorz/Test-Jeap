# Next steps: four candidate directions, with their evidence and cost

Written after the RoboTwin Con1/Con2 validation. Every claim below is backed by a
measurement in `docs_CLAIMS_AND_EVIDENCE.md`; the point here is to decide what to
do next rather than to report what was done.

## Where the evidence has left us

* The correction is causal and reads the latent (scrambling the delta costs 3.59%,
  t = 3.55).
* Its benefit is small and unstable across training (-11.8% / -1.0% / -2.9% at
  4.5k / 9k / 15k against its own silence control).
* Improving the latent head does not move the action, measured four ways
  (between arms, at the extreme where the head beats the linear ceiling, within
  the experiment, and from the mechanism side).
* Con2, whose premise is that accuracy translates, is inert at every training
  point tested.
* Everything above is flow loss on the training distribution. **No closed-loop
  number exists.**

## C - Measure closed-loop before doing anything else (recommended first)

**Hypothesis under test**: the offline metric we have been optimising is either a
reasonable proxy for success rate, or it is not, and everything above is noise in
a metric nobody scores.

**Why first**: four of the five supported claims are about *flow loss*. If flow
loss does not track success rate on this benchmark, then a large part of the
session's conclusions are about the wrong quantity, and every candidate below
inherits that problem. Conversely if it does track, the small effects we measured
become interpretable.

**Cost**: install the RoboTwin simulator and run the 20-task evaluation for the
handful of checkpoints already on disk (A, B, C, D). The checkpoints and the
policy server path already exist; the simulator does not.

### Feasibility check on this box (2026-09-13)

Verified rather than assumed:

| requirement | state |
|---|---|
| `NVIDIA_DRIVER_CAPABILITIES` includes `graphics` | **yes** - the container has `all`, and the RoboTwin docs call this essential (omitting it segfaults on missing Vulkan) |
| NVIDIA driver >= 520 | **yes**, 580.95.05 |
| RTX GPU for ray tracing | **yes**, 4x RTX 5090 |
| network to github / huggingface / pypi | **yes**, all 200 |
| root inside the container | **yes** |
| Vulkan driver stack | **missing** - `vulkaninfo` exists but there is no `/usr/share/vulkan/icd.d`; needs `apt install libvulkan1 mesa-vulkan-drivers vulkan-tools` |
| conda | **missing** - the official eval launcher takes `--eval-env-conda-env`, so a Miniconda install is the path of least resistance |
| disk | **63 GB free, and this is the real risk** - the asset bundle's size is not documented up front |

Install path: Vulkan packages, Miniconda, `git clone --recurse-submodules`
RoboTwin, `scripts/_install.sh` (compiles CuRobo, mplib, pytorch3d - the docs say
20 minutes but the builds usually take longer), then `scripts/_download_assets.sh`.

**Disk remedy if needed**: `artifacts/con1/robotwin_clean20_19999_features_v1` is
139 GB and is only needed to *train* Con1 - closed-loop evaluation of the existing
checkpoints does not read it. Deleting it would take free space to ~200 GB, but it
is a deliberate, one-way decision and it means re-caching before any retrain.

**What would change our mind**: if arm A/B/C/D are indistinguishable in success
rate but differ in flow (or vice versa), the flow-loss line has to be rebuilt
around whatever the simulator says.

## A - Ranking-based adaptation (the design the evidence actually points at)

**Hypothesis**: the world model is a usable *scorer* even though it is a bad
*gradient source* - it ranks the executed chunk first 49%/67% of the time against
8.3% chance, while its gradient moves the action away from the demonstration in
84% of cases (`docs_CON1_WM_ACTION_JUDGMENT.md`).

**Why it is attractive**: it is the only mechanism in the repository whose
*positive* half has already been measured on the target platform, and it transmits
to the action by construction (the adapter is moved toward better-scoring
candidates) rather than hoping a latent improvement propagates.

**Cost**: no base retraining - the world model exists and the ranking half is
validated. Needs a candidate sampler at deploy time and an offline protocol for
the adaptation half.

**What would change our mind**: if the ranking score does not correlate with
closed-loop success at candidate level, the mechanism is dead regardless of the
gradient result.

## B - Treat the latent as identity, not as prediction

**Hypothesis** (this session): the correction uses the latent's *content* but not
its *accuracy* - it behaves like "which situation is this" rather than "what will
happen". Consistently, the per-batch benefit is uncorrelated with per-batch NMSE
(r = -0.07, p = 0.36) while scrambling the latent hurts.

**If true**: Con2's objective is directionally wrong, and the right change is to
condition the action on the latent differently (for example as a token in the
prefix) rather than to predict the future better.

**Cost**: an architecture change plus a retrain; no existing cache supports the
token-level variant (spatial tokens would be ~76x the storage of the pooled
latents currently cached).

**What would change our mind**: if the latent's benefit did turn out to track
accuracy under a better-controlled comparison, this hypothesis fails and Con2 is
back in play.

## D - Re-run on the paper's 20 Clean tasks

**Hypothesis**: the results are diluted by training on the full 2,500-episode
release, most of which is not the paper's task set.

**Blocker, and it is absolute**: the release carries no task identity - the raw
parquet columns are state/action/timestamp/frame/episode/index/task_index, and
`tasks.jsonl` holds instruction strings that vary with scene randomisation, so the
20-task filter cannot be reconstructed from what is on disk. It needs an external
mapping (the simulator's task list, or a per-task original release).

**Cost**: the mapping, then the Con1 cache and the checkpoints have to be rebuilt
from scratch.

## Recommendation

Run **C first**. The entire session produced offline numbers with effect sizes
between 1% and 3% that fluctuate with training step, and the paper needs success
rate. Until we know whether flow loss tracks success rate on this benchmark, both
A and B would be built on an unvalidated metric, and D would spend a full
re-cache to answer a question we cannot currently interpret.
