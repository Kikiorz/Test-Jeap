# Does the latent carry action information? (the root-cause measurement)

Con1 tries to let a future-latent prediction influence the action. That can only
work if the latent carries information about the action that the policy's own
observation encoding does not already have. Four coupling experiments came back
negative (action conditioning, gradient balancing, world-model gradient, learned
metric), so this measures the latent directly with held-out episodes.

## Protocol

`scripts/probe_latent_action_information.py` (linear probe) and
`scripts/train_action_relevant_latent.py` (inverse-dynamics shaping). Episodes
0-198 train / 400-439 held out, ridge fitted on train and scored on held-out
episodes. Baseline that matters: `a_t` predicted from `a_{t-1}` alone - the
action is strongly autocorrelated, so anything else must add *incremental*
information.

## Linear probe on the frozen latents (26367 train / 4562 held out)

| features | R^2 for the immediate action | R^2 for the 10-step chunk |
|---|---:|---:|
| previous action | **0.962** | **0.705** |
| z_t | 0.263 | 0.130 |
| z_{t+1} | 0.269 | 0.154 |
| z_t + z_{t+1} | 0.284 | 0.135 |
| **Delta z (what Con1 predicts)** | **0.047** | **0.033** |
| previous action + latent | 0.933 | 0.473 |

Same protocol on the V-JEPA 2-AC token mean: `z_t` -0.219, `z_{t+1}` -0.209,
`Delta z` 0.009 (chunk: -0.286 / -0.279 / 0.005). The AC space transfers even
worse across episode sets.

## Inverse-dynamics shaping

A projection `h = P_psi(z_t, z_{t+1})` trained to predict the action chunk on
the train episodes, then scored on held-out episodes (ridge fitted on train):

| features | held-out R^2 (chunk) | vs previous action |
|---|---:|---:|
| previous action only | 0.7066 | - |
| previous action + raw latent | 0.4641 | **-0.242** |
| previous action + shaped h | 0.2345 | **-0.472** |
| shaped h only | 0.1573 | - |

Target = the action *change* `a_t - a_{t-1}` (the control-relevant part, since
the absolute action is pure continuity):

| features | held-out R^2 |
|---|---:|
| previous action | 0.0085 |
| previous action + latent | 0.0122 |
| shaped h | 0.0047 |
| inverse model (direct) | -0.0136 |

## Conclusion

1. **The action is almost entirely temporal continuity** (previous action alone
   explains R^2 0.96 immediate / 0.71 chunk).
2. **The latent transition Delta z - the quantity Con1 predicts - carries almost
   no action information** (0.047 / 0.009), and adding the latent to the previous
   action makes the prediction *worse* on held-out episodes, i.e. the little
   correlation it has does not transfer across episodes/tasks.
3. **Shaping a frozen pooled latent with an inverse-dynamics objective does not
   create transferable action information** (R^2 0.005 for the action change).
4. Consequently every coupling that routes this latent into the action is
   structurally a no-op - which is exactly what the four earlier experiments
   measured. The problem is *upstream of the coupling*.

What this does not rule out, and is the natural next step: the pooled *mean* of
the teacher features may be what destroys the action-relevant structure. A
**token-level** inverse model (attention over the 256 patch tokens per frame,
using the token caches already on disk) is the one variant not yet measured, and
a representation *fine-tuned* with the inverse objective (rather than a
projection on a frozen one) is the stronger version of the same idea.
