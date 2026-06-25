# HCL Next Methods Explainer

This note explains the newer methods used after the original direct RL/PPO
experiments struggled to improve learned-high hierarchy performance.

## Context

The project is testing hierarchical control on Push-T:

```text
high level: choose a future privileged-state goal every k steps
low level: execute primitive actions conditioned on current state + held goal
```

The frozen supervised privileged-state hierarchy is already reasonably strong.
The problem is that local low-level improvement does not automatically transfer
to better full-task success. In particular, local terminal MSE and oracle-goal
performance can improve while learned-high rollouts get worse.

The newer methods therefore try to answer a narrower question:

```text
Which local action corrections are useful under the learned high-level policy's
actual state+goal distribution?
```

## Branch-Outcome Attribution

Branch-outcome attribution is a way to label local action-search improvements
by full-task outcome, not just local goal distance.

For each learned-high replan state:

1. Roll out the base low-level policy for the next `k=10` steps.
2. Sample noisy action-branch candidates.
3. Keep the candidate branch that improves local terminal MSE.
4. Continue the episode using the same learned hierarchy.
5. Compare final task outcome against the base branch.

The saved branch bank contains:

```text
conditions                  low-level inputs for the selected 10-step branch
actions                     selected branch actions
selected_base_mse           local terminal MSE from base branch
selected_best_mse           local terminal MSE from selected branch
selected_improvement_mse    base_mse - best_mse
selected_base_success       final success after base branch + continuation
selected_candidate_success  final success after selected branch + continuation
selected_success_delta      candidate_success - base_success
selected_base_return        final return after base branch + continuation
selected_candidate_return   final return after selected branch + continuation
selected_return_delta       candidate_return - base_return
sample_weights              usually based on positive return delta
```

This is not direct PPO. It is closer to:

```text
local stochastic search -> outcome attribution -> supervised distillation
```

## Return-Positive Branch Banks

A return-positive bank keeps branch candidates whose selected branch improved
final episode return by at least a threshold, commonly:

```text
selected_return_delta >= 5
```

The first strong learned-high improvement came from a single-window
return-positive bank. It improved matched hierarchy success from about `0.5713`
to a similar range, but later multi-window variants showed that return strength
alone is not enough.

## GoalNN79

`goalNN79` means:

```text
select 79 branches whose held goals are nearest to the preserve-bank held-goal
distribution
```

The preserve bank is collected from normal learned-high hierarchy rollouts.
For each candidate branch, we look at the first row of its 10-step block and
compare the held-goal slice to held goals seen in the preserve bank.

The purpose is to avoid training on branches that look useful in isolation but
are far from the learned hierarchy's normal goal distribution.

Result summary:

```text
goalNN79 improved oracle-goal / low-level diagnostics,
but did not improve learned-high hierarchy success.
```

Interpretation:

```text
Goal proximity alone helps when the high-level goal is good, but learned-high
rollouts need state context too.
```

## State+GoalNN79

`state+goalNN79` extends `goalNN79` by matching both:

```text
current normalized state + held goal
```

Specifically, it uses the first branch-row condition slice:

```text
condition[0:62]
```

This slice contains the normalized privileged state and held goal. We standardize
the slice using the preserve-bank distribution, compute nearest-neighbor MSE to
preserve-bank rows, then select the closest 79 branches.

This produced the current best learned-high checkpoint:

```text
state+goalNN79, improve_npz_weight=0.1
matched hierarchy success: [0.582, 0.594, 0.574]
mean: 0.5833
```

Why it worked better:

```text
It selects corrections that are not only outcome-positive, but also close to
the learned high-level policy's actual state+goal operating distribution.
```

## Top-Return and Hybrid Selectors

Several selectors tried to increase branch outcome strength:

```text
top-return79
state+goalNN120 top-return79
multi4 state+goalNN79
fail-to-success-only
```

These usually improved outcome-label strength or oracle-goal behavior, but did
not beat `state+goalNN79` for learned-high hierarchy success.

Main lesson:

```text
The best local correction is not necessarily the best learned-high correction.
Compatibility with the learned high-level rollout distribution matters.
```

## Local Oracle Gate

The local oracle gate evaluates both base and tuned low-level policies for the
current held goal and uses the tuned policy only if local terminal MSE is no
worse than base.

It helped oracle-goal diagnostics but hurt learned-high hierarchy success.

Interpretation:

```text
Local terminal MSE is aligned with oracle goals, but not with learned-high
task success.
```

## High-Goal Projection

Nearest-oracle-bank projection replaces a predicted learned high-level goal with
the nearest real teacher-state prototype.

This was strongly harmful.

Interpretation:

```text
The learned high-level and low-level are calibrated to the continuous predicted
goal output. Snapping to a nearby real prototype changes the control semantics
too much.
```

## Current Takeaway

Direct RL/PPO is not yet the robust solution here. The strongest signal so far
is:

```text
branch search + learned-high outcome labels + distribution-aware selection +
supervised distillation
```

The current best method is `state+goalNN79`. It is modest but real evidence that
local branch improvements can transfer to learned-high hierarchy success when
the selected branches match the learned high-level state+goal distribution.

The next natural step is a learned selector over branch features and outcome
labels, rather than more hand-written scalar selectors.
