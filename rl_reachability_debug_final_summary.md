# RL Reachability Debug Final Summary

This is the consolidated summary of the RL reachability debugging sequence.
Detailed commands, paths, and per-run notes are in
`rl_reachability_debug_experiment_log.md`; accepted result tables are in
`rl_reachability_debug_final_results.md`.

## Bottom Line

The main failure mode is not that PPO cannot reduce local goal distance. It can.
The failure mode is that improving local reachability often destroys or drifts
away from the action/contact manifold that makes the pushing task succeed.

The best current full-state low-level remains:

```text
Run 30: residual-on-Phase-C-BC PPO
oracle full-state held-subgoal success: 0.73
learned-high full-state held-subgoal success: 0.55
shuffled-goal success: 0.00
```

Run 30 proves that full-state RL can match or slightly beat Phase-C BC under
oracle subgoals when the PPO policy is constrained as a small residual around
the BC controller. It does not beat Phase-C BC under learned high-level subgoals.

The simple deployed reset-bank aggregation branch is now exhausted:

| Run | Change | Oracle success | Learned-high success | Conclusion |
| --- | --- | ---: | ---: | --- |
| Run 30 | residual-on-BC on round-2 bank | 0.73 | 0.55 | best full-state PPO |
| Run 36 | continue Run 30 on unfiltered deployed bank | 0.68 | 0.53 | worse than Run 30 |
| Run 37 | continue Run 30 on success-filtered deployed bank | 0.57 | 0.47 | worse again |

Adding deployed states, even filtered by rollout success, improved some local
training/reachability signals but did not improve task success. More deployed
reset data alone is not enough.

## What We Learned

### TCP Goals Were the Wrong Proxy

Privileged TCP PPO learned the local 10-step TCP endpoint task extremely well:

| Policy | Reward | Terminal TCP dist. | Reach eps |
| --- | --- | ---: | ---: |
| Run 2 PPO | true TCP | 0.000489 | 0.9744 |
| Run 5 PPO | learned `D_psi` | 0.000515 | 0.9725 |
| BC 1800 | imitation | 0.000258 | 0.9832 |

But those policies got near-zero full task success when deployed in the
hierarchy. The reason is conceptual: moving the TCP to the desired endpoint
does not specify the desired scene transition. TCP reachability is too weak a
subgoal for pushing.

### `D_psi` Was Valid in the Easy TCP Setting

The branch-trained privileged TCP distance ensemble was a good ranking model:

| Metric | Value |
| --- | ---: |
| terminal-distance Spearman | 0.9948 |
| reachable/unreachable AUC | 0.9999 |
| selected-branch better-than-PPO accuracy | 0.8999 |

So the failure was not simply that learned distances are useless. The problem
was the goal representation and policy/action manifold.

### Object-Pose Goals Were More Task-Relevant

Object-pose PPO without action regularization improved reachability but did not
solve the task. Adding teacher-action penalty made it task-compatible.

| Run | Goal | Teacher penalty | Oracle success |
| --- | --- | ---: | ---: |
| Phase-B BC | object pose | n/a | 0.16 |
| Run 10 | object pose PPO | 0.20 | 0.12 |
| Run 12 | object pose PPO | 0.30 | 0.16 |
| Run 13 | object pose PPO | 0.50 | 0.21 |

Run 13 was the first scratch RL low-level to beat its matching BC baseline, but
absolute success stayed low. Object pose captures the pushed object outcome, but
it underspecifies the robot/TCP configuration needed to make contact reliably.

### Full-State Goals Were the Right Semantics, but the Evaluator Had to Be Fixed

The full-state subgoal is the closest match to the original hierarchical idea:
the high level should specify the desired scene/state after the option, not just
the TCP endpoint.

The important evaluator bug was this:

```text
Wrong: hold the raw 28D full-goal feature vector fixed for k=10.
Right: hold the target future state fixed, then recompute full-goal features
       from the current state and remaining option time.
```

Once this was fixed, Phase-C time-conditioned full BC was strong:

| Policy | Goal feature semantics | Success |
| --- | --- | ---: |
| Phase-B full BC | recomputed, no time input | 0.08 |
| Phase-C full BC | recomputed, time-conditioned | 0.74 |

This changed the interpretation of earlier full-goal failures. Full-state goals
are not weak; the low-level must be conditioned on the target future state and
remaining time correctly.

### Scratch Full-State PPO Improved Local Distance but Not Task Success

Runs 19-22 trained full-state PPO with corrected held-target semantics.
Same-bank local reachability improved, but task success remained poor:

| Policy | Local terminal full-goal dist. | Oracle success | Teacher action MAE |
| --- | ---: | ---: | ---: |
| Phase-C full BC | 1.6563 | 0.69 | 0.0414 |
| Run 19 full PPO | 1.5264 | 0.00 | 0.2812 |
| Run 21 long PPO | 1.8744 | 0.04 | 0.2387 |
| Run 22 long PPO | 1.5018 | 0.01 | 0.2493 |

The key signal is the high teacher-action MAE. PPO learned actions that reduce
the full-state distance metric but are not task-compatible.

### Reset Mixtures Helped Only When BC Structure Was Added

Training on demo plus deployed reset states addressed some distribution shift,
but plain PPO still failed task success. BC structure was the turning point.

| Run | Main change | Oracle success |
| --- | --- | ---: |
| Run 23 | reset mixture, learned-high targets | 0.00 |
| Run 24 | oracle target diagnostic | 0.03 |
| Run 25 | BC warm start | 0.16 |
| Run 26 | BC warm start + BC-prior | 0.21 |
| Run 28 | stronger BC prior on aggregation bank | 0.25 |
| Run 30 | bounded residual on Phase-C BC | 0.73 |

This is the strongest evidence that reset distribution matters, but only if the
policy remains near a task-compatible BC controller.

### Residual-on-BC Was the Best Policy Structure

Run 30 parameterized the policy as a bounded residual around Phase-C full BC:

```text
action = BC_action + alpha * tanh(residual)
alpha = 0.15
residual penalty = 0.01
```

That recovered BC-level task success under oracle full-state subgoals while
keeping shuffled-goal success at zero:

| Goal source | Phase-C BC success | Run 30 success |
| --- | ---: | ---: |
| oracle full-state | 0.68 | 0.73 |
| shuffled oracle | 0.00 | 0.00 |

A larger residual radius hurt success:

| Run | Change | Oracle success |
| --- | --- | ---: |
| Run 30 | alpha 0.15, penalty 0.01 | 0.73 |
| Run 31 | alpha 0.25 | 0.65 |
| Run 32 | alpha 0.15, no penalty | 0.69 |

The policy needs freedom to correct BC, but too much freedom or weak residual
regularization damages task behavior.

### Learned-High Subgoals Are the Remaining Bottleneck

Under learned high-level full-state subgoals, Phase-C BC still beats Run 30:

| Goal source | Phase-C BC success | Run 30 success |
| --- | ---: | ---: |
| learned full-state | 0.62 | 0.55 |
| shuffled learned | 0.00 | 0.00 |

The learned-high audit showed that learned targets are not grossly wrong in
object/TCP components:

| Rollout policy | Learned-vs-oracle full L2 | Object-pose L2 | TCP L2 | Robot L2 |
| --- | ---: | ---: | ---: | ---: |
| Phase-C full BC | 0.9268 | 0.1080 | 0.0295 | 0.9106 |
| Run 30 residual-on-BC PPO | 0.9171 | 0.0919 | 0.0267 | 0.9057 |

Most error is in robot-state dimensions. However, replacing the learned robot
target with the current robot state was much worse:

| Robot target mode | Phase-C BC learned success | Run 30 learned success |
| --- | ---: | ---: |
| predicted robot target | 0.62 | 0.55 |
| current robot target | 0.07 | 0.05 |

So the robot component is imperfect but important. Removing it is not a fix.
Simple learned-target scaling also failed:

| Scale | Phase-C BC learned success | Run 30 learned success |
| ---: | ---: | ---: |
| 0.75 | 0.08 | 0.09 |
| 1.00 | 0.62 | 0.55 |
| 1.25 | 0.27 | 0.24 |
| 1.50 | 0.05 | 0.07 |

The learned-high gap is not just target scale or missing object/TCP accuracy.
It is likely the interaction between high-level target semantics, robot-state
conditioning, and contact-compatible low-level execution.

## Final Run 37 Result

Run 37 tested the specific idea of filtering deployed reset-bank batches by
rollout success before continuing Run 30.

Selected deployed batch any-success scores:

| Collector | Scores |
| --- | --- |
| Phase-C full BC | 0.6160, 0.6123, 0.6118, 0.6113 |
| Run 30 residual-on-BC PPO | 0.5923, 0.5901, 0.5850, 0.5837 |

Training improved same-bank terminal distance relative to Run 36, but task
success got worse:

| Evaluation | Phase-C BC | Run 30 | Run 36 | Run 37 |
| --- | ---: | ---: | ---: | ---: |
| oracle full-state success | 0.68-0.73 | 0.73 | 0.68 | 0.57 |
| learned-high full-state success | 0.62-0.66 | 0.55 | 0.53 | 0.47 |
| shuffled learned success | 0.00 | 0.00 | 0.00 | 0.00 |

The 50-episode checkpoint screen did not find a clearly better intermediate
checkpoint:

| Checkpoint | Learned-high success |
| --- | ---: |
| update 25 | 0.54 |
| update 50 | 0.52 |
| update 100 | 0.58 |
| update 150 | 0.58 |
| update 200 | 0.48 |

Conclusion: success-filtered deployed reset aggregation is still not the right
next step. It can improve the reset-bank quality signal and local training
metric while hurting task success.

## Recommended Stop/Continue Decision

Stop running more variants of:

```text
collect deployed states -> add/filter reset bank -> continue same residual PPO
```

The evidence from Runs 36 and 37 says this branch is optimizing the wrong thing.

The next technically plausible directions would require changing the objective
or policy constraint, not just the reset data:

1.  Stronger action/contact regularization for full-state PPO, possibly closer
    to the object-pose teacher-penalty result but without relying on online
    expert querying as the main method.
2.  A high-level target representation that predicts the robot/contact part
    more consistently, since object/TCP target quality alone is not the main
    learned-high error.
3.  A residual policy objective that explicitly preserves Phase-C BC task
    behavior while optimizing reachability only where BC is weak, rather than
    allowing PPO to improve geometric distance everywhere.

For the current proof-of-concept, the strongest defensible result is:

```text
Full-state hierarchical subgoals are the correct interface.
Phase-C full BC is a strong baseline.
Residual-on-BC PPO can match or beat BC under oracle full-state subgoals.
Learned-high deployment still favors BC, and deployed reset aggregation alone
does not close that gap.
```
