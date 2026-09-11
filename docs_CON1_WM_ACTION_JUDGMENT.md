# Can the V-JEPA 2-AC world model improve Con1's action? (judgment measurement)

The proposed Con1-TTT design needed one property to be worth building: taking a
world-model gradient step at the chunk the policy *actually produced* should move
that chunk towards the demonstrated one. If it does not, the world model cannot
serve as an action-improvement oracle and no GPU should be spent on that design.

## Protocol

`scripts/dump_con1_policy_chunks.py` + `scripts/probe_con1_wm_action_judgment.py`:

1. Run the deployed Con1 policy (`pi05_libero_con1_action_adapter_40k`,
   stage-1 checkpoint 1000) on 80 (episode, frame) pairs from held-out episodes
   400-409 and record its action chunk and the demonstrated chunk.
2. For each pair, roll the world model 5 steps (1 s at 5 Hz) under the policy's
   chunk, take `grad_a ||WM(z_t, a) - z*_{t+H}||^2`, and apply one step of
   relative size `||da||/||a|| = 0.1`.
3. Compare the refined chunk with the demonstrated one.

## Result (80 samples)

| quantity | value | reading |
|---|---:|---|
| world-model energy reduction from the step | +25,732 | the step works on its own objective |
| mean distance to the demonstrated chunk - before | 1.525 | |
| mean distance to the demonstrated chunk - after | 1.564 | |
| **mean distance change** | **+0.038** | the step moves *away* from the demo |
| **fraction of samples moved closer to the demo** | **16.3%** | chance is 50% |
| mean rank of the demonstrated chunk | 0.93 | ~1 candidate (the refined one) beats it |
| policy-to-demo headroom | 0.565 | the policy is 56% away from the expert |

## Conclusion

**The world-model gradient is not an action-improvement direction.** It lowers
the model's own energy while moving the action away from the expert action in
84% of cases. Building the "world model refines the action chunk" coupling on
top of that gradient would be fitting the model's own bias, not the task.

What this does *not* contradict: the world model's **ranking** ability is real
and separately measured - with 12 candidates it places the executed chunk first
49% (0.2 s) / 67% (0.4 s) of the time against 8.3% chance, and it prefers the
demonstrated chunk over nearby perturbations. Ranking and gradient are different
objects: ranking compares *candidates against an observed future*, while the
gradient walks to the model's own optimum, which is demonstrably not the expert
action.

Consequence for the design: if Con1 is to use the world model in a way that
reaches the action, it has to use it as a **scorer of candidates**, not as a
gradient source - i.e. sample several chunks, score them against what actually
happened, and move the adapter towards the better-scoring ones (ranking-based
self-supervised adaptation). That mechanism is deployable (no labels, no goal
image, uses observed transitions), transmits to the action by construction, and
its validity can be checked offline for the ranking half and in the L5 loop for
the effect half.
