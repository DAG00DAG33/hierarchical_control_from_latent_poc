# Privileged-Z RL Sanity Experiments

Date: 2026-06-24

## Motivation

The learned-latent RL rerun suggests that the low level is weakly sensitive to
the supplied future latent goal. This may be a representation issue, but it may
also be a conceptual issue with the high/low-level interface and the current
mostly predictable teacher trajectories.

This plan adds two simpler tests before spending more compute on learned latent
interfaces.

## Experiment A: Full Privileged-State Interface

Goal: replace learned latent `z` with normalized privileged simulator
observation state.

Use:

```text
z_t = observations_state_t
```

from the vector-consistent corpus:

```text
data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5
```

This is the 31D `rgb+state` observation state that includes robot/TCP and
T-block state. Normalize it with a standardizer fit on the training split.

Initial settings:

| Item | Value |
| --- | --- |
| Training data | 500 successful teacher trajectories/streams if enough are available |
| Validation data | held-out vector-consistent streams |
| Horizon `k` | 10 simulator steps |
| High-level update period | 10 simulator steps |
| Low-level chunk | keep one primitive action per step |
| High-level input | normalized current privileged state + normalized previous action |
| High-level target | normalized future privileged state |
| Low-level input | current privileged state, future privileged goal, previous action, time-to-go |
| Low-level target | teacher action |

Train:

1. flat privileged low level:
   `a_t = pi_flat(z_t, a_{t-1}, time)`
2. privileged goal-conditioned low level:
   `a_t = pi_low(z_t, z_{t+k}, a_{t-1}, time)`
3. deterministic high level:
   `g_t = pi_high(z_t, a_{t-1})`

Evaluate:

1. offline high-level future-state prediction error;
2. offline low-level action MAE;
3. valid-goal sensitivity with same-current-state future goals;
4. closed-loop hierarchy with learned high-level goals;
5. local RL fine-tuning using the simplest existing R1-style residual if the
   supervised privileged hierarchy is stable.

Gate:

```text
privileged goal-conditioned low-level action sensitivity should clearly exceed
the learned-latent low-level sensitivity, and learned high-level closed-loop
success should not be worse than flat privileged by more than 5 percentage
points before doing RL.
```

Interpretation:

- If privileged `z` works, learned representation/prediction is still the main
  bottleneck.
- If privileged `z` also fails to create useful goal sensitivity, the problem is
  likely the interface/training objective rather than the learned encoder.

## Experiment B: Disturbance-Rich 500-Trajectory Data

Goal: make the future goal less predictable from current state alone.

Collect or derive a 500-trajectory corpus where roughly half of the trajectories
include disturbances. The disturbance should create recovery states and local
ambiguity, so current state alone is less sufficient to infer the desired
future state and action.

Suggested mixture:

| Split | Count |
| --- | ---: |
| Clean successful teacher trajectories | 250 |
| Disturbed/recovery successful teacher trajectories | 250 |

Requirements:

- Store the same fields as the vector-consistent corpus when possible:
  `observations_state`, `simulator_states`, `executed_actions`,
  `previous_executed_actions`, success flags, DINO/proprio if needed later.
- Keep reset seeds and disturbance metadata.
- Only train from successful trajectories for the first comparison.

Evaluate:

1. same-current-state valid-goal action sensitivity for the supervised low
   level trained on clean-only 500 versus mixed 500;
2. observation/goal/previous-action block sensitivity;
3. local RL fine-tuning with the same simplest R1-style recipe if sensitivity
   improves.

Gate:

```text
mixed-data goal sensitivity should improve by at least 2x over clean-only before
running a serious RL point.
```

## Immediate Execution Order

1. Implement a compact privileged-z trainer/evaluator that consumes the
   vector-consistent `observations_state` corpus directly.
2. Run the supervised privileged-z smoke with `n=500`, `k=10`, one seed, and a
   small closed-loop/evaluation budget.
3. Compare valid-goal sensitivity against the learned-latent G1 results.
4. Only then decide whether to run the simplest RL fine-tuning on privileged-z.
5. Implement disturbed-data collection after Experiment A tells us whether the
   privileged interface is conceptually viable.

