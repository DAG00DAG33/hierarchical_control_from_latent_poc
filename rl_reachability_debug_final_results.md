# RL Reachability Debug Final Results

This file summarizes accepted comparisons from the running experiment log. Full
commands, paths, and failed-run notes are in `rl_reachability_debug_experiment_log.md`.

## Key Results

### Privileged/TCP Local PPO Works but Does Not Transfer

The privileged/TCP scratch PPO sanity gate passed locally:

| Policy | Reward | Terminal TCP dist. | Reach eps | Shuffled reach | Action saturation |
| --- | --- | ---: | ---: | ---: | ---: |
| Run 2 PPO | true TCP | 0.000489 | 0.9744 | 0.1013 | 0.449 |
| Run 5 PPO | learned `D_psi` | 0.000515 | 0.9725 | 0.0958 | 0.302 |
| BC 1800 | imitation | 0.000258 | 0.9832 | 0.1978 | 0.189 |

Random shooting on the same local MDP improved over Run 2 PPO, confirming useful
10-step action sequences exist:

| Method | Terminal TCP dist. | Reach eps | Improved vs PPO |
| --- | ---: | ---: | ---: |
| PPO deterministic | 0.000602 | 0.9641 | - |
| random shooting, 32 candidates | 0.000110 | 0.9954 | 0.9578 |
| random shooting, 64 candidates | 0.000081 | 0.9968 | 0.9780 |
| random shooting, 128 candidates | 0.000063 | 0.9980 | 0.9893 |

However, the TCP RL low-levels did not transfer to full task success:

| Goal source | Low-level | Success | Final reward | Hold endpoint error |
| --- | --- | ---: | ---: | ---: |
| oracle TCP | BC 1800 | 0.66 | 0.767 | 0.0346 |
| oracle TCP | Run 2 true-distance PPO | 0.00 | 0.147 | 0.0553 |
| oracle TCP | Run 5 `D_psi` PPO | 0.00 | 0.152 | 0.0567 |
| learned TCP | BC 1800 | 0.65 | 0.758 | 0.0357 |
| learned TCP | Run 2 true-distance PPO | 0.00 | 0.141 | 0.0563 |
| learned TCP | Run 5 `D_psi` PPO | 0.01 | 0.155 | 0.0526 |

Interpretation: local TCP endpoint reachability is not enough. The RL policies
can reach short TCP endpoints but leave the task/contact manifold learned by BC.

### Branch-Trained `D_psi` Is a Valid Positive Control

The privileged/TCP branch ensemble passed the offline ranking gate:

| Gate | Result |
| --- | ---: |
| terminal-distance Spearman | 0.9948 |
| reachable/unreachable AUC | 0.9999 |
| selected-branch better than PPO accuracy | 0.8999 |
| PPO branch terminal distance | 0.000478 |
| `D_psi` selected candidate distance | 0.000081 |
| oracle random-search best distance | 0.000076 |

This validates the branch/off-policy ensemble procedure in the easy
privileged/TCP setting, but it does not by itself validate VAE/effect latent
distance rewards.

### Object-Pose PPO With Teacher-Action Penalty Beats BC Under Oracle Goals

Pure object-pose reachability PPO improved local reachability but did not
produce task success. Adding a teacher-action penalty made the scratch PPO
low-level task-compatible.

| Run | Teacher penalty | Terminal object-pose dist. | Reach | Oracle-goal success |
| --- | ---: | ---: | ---: | ---: |
| Phase-B object-pose BC | n/a | n/a | n/a | 0.16 |
| Run 9 | 0.05 | 0.0728 | n/a | 0.00 |
| Run 11 | 0.10 | 0.0646 | n/a | 0.03 |
| Run 10 | 0.20 | 0.0527 | n/a | 0.12 |
| Run 12 | 0.30 | 0.0426 | 0.4702 | 0.16 |
| Run 13 | 0.50 | 0.0414 | 0.4800 | 0.21 |

Run 13 is the first scratch RL low-level in this sequence to beat the matching
object-pose BC baseline under oracle object-pose goals.

### Learned Object-Pose High-Level Narrows but Does Not Remove the Gap

The learned object-pose high-level predictor is accurate on held-out supervised
subgoals:

| Metric | Value |
| --- | ---: |
| train episodes | 1800 |
| validation episodes | 200 |
| validation object-pose L2 | 0.0561 |
| validation xy L2 | 0.0085 m |
| validation yaw abs | 0.0517 rad |
| persistence object-pose L2 | 0.5042 |

Full rollout with learned object-pose goals:

| Goal source | Low-level | Success | Final reward | Hold object-pose dist. |
| --- | --- | ---: | ---: | ---: |
| oracle | Phase-B object-pose BC | 0.17 | 0.3647 | 0.2986 |
| oracle | Run 13 PPO | 0.20 | 0.3972 | 0.2227 |
| learned | Phase-B object-pose BC | 0.13 | 0.3404 | 0.5900 |
| learned | Run 13 PPO | 0.17 | 0.3697 | 0.3141 |
| shuffled learned | Phase-B object-pose BC | 0.04 | 0.2068 | 1.5780 |
| shuffled learned | Run 13 PPO | 0.01 | 0.1673 | 1.4187 |

Run 13 remains better than BC with learned high-level object-pose goals, but
absolute success remains low.

### Deployment-State Reachability Favors Run 13

Branching both low-levels from the same deployed full-architecture states shows
Run 13 generally reaches learned object-pose goals better than BC:

| Collector | Candidate | Shuffled | Terminal dist. | Reach eps | P90 terminal |
| --- | --- | ---: | ---: | ---: | ---: |
| BC | BC | no | 0.2757 | 0.404 | 0.8362 |
| BC | Run 13 PPO | no | 0.2358 | 0.429 | 0.7265 |
| Run 13 PPO | BC | no | 0.3042 | 0.313 | 1.0070 |
| Run 13 PPO | Run 13 PPO | no | 0.2347 | 0.301 | 0.7251 |
| BC | BC | yes | 1.0610 | 0.0716 | 3.1982 |
| BC | Run 13 PPO | yes | 1.0749 | 0.0716 | 3.2240 |
| Run 13 PPO | BC | yes | 1.0852 | 0.0971 | 3.1207 |
| Run 13 PPO | Run 13 PPO | yes | 1.0866 | 0.0835 | 3.1473 |

The threshold reach-rate is mixed on Run-13-collected states, but mean and P90
terminal distances favor Run 13. This means the low learned-goal task success is
not simply because Run 13 cannot reach learned object-pose goals from deployed
states.

### Full-State Subgoals Are Strong With Correct Held-Target Semantics

Run 16 tested the original full-state subgoal idea with scratch PPO and the same
teacher-action penalty used in Run 13.

Local held-goal reachability strongly improved:

| Metric | Initial/random | Run 16 full PPO | Shuffled-goal PPO |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 7.4381 | 1.9120 | 7.8599 |
| p50 terminal distance | 4.9755 | 0.6572 | 4.8239 |
| p90 terminal distance | 15.2186 | 3.7421 | 16.1092 |
| fraction improved | 0.7801 | 0.9708 | 0.7381 |
| action saturation | 0.0000 | 0.0058 | 0.0026 |

This is the strongest local evidence that the right subgoal semantics matter:
full-state goals are much more aligned with the desired option outcome than
TCP-only endpoint goals.

The first full-rollout evaluator held the raw 28D Phase-B `full` goal vector
fixed for `k=10`. That was the wrong protocol for Phase-B/Phase-C `full` goals,
because the vector contains velocity/rate features. The correct protocol is to
hold the target future state fixed and recompute the goal features from the
current state and remaining time.

Correct held-oracle BC audit:

| BC policy | Goal feature semantics | Success | Final reward | Teacher action MAE |
| --- | --- | ---: | ---: | ---: |
| Phase-B full BC | recomputed features, no time input | 0.08 | 0.2915 | 0.2146 |
| Phase-C full BC | recomputed features, time-conditioned | 0.74 | 0.8241 | 0.0407 |

This explains why the earlier BC full-goal held result was suspiciously low:
it was an evaluator/baseline mismatch. Full-state subgoals are not weak. The
time-conditioned full BC baseline is strong under the intended held-subgoal
hierarchy.

Run 19 retrained full-state PPO with the same recomputed held-target feature
semantics. Local reachability improved, but task success did not:

| Policy | Local terminal full-goal dist. | Oracle held success | Hold full-goal dist. | Teacher action MAE |
| --- | ---: | ---: | ---: | ---: |
| Phase-C full BC | n/a | 0.69 | 1.6358 | 0.0414 |
| Run 19 recomputed full PPO | 1.5264 | 0.00 | 5.2944 | 0.2812 |
| Run 20 recomputed full PPO, teacher penalty 1.0 | 1.4703 | 0.00 | 4.9881 | 0.2870 |
| Run 21 long PPO, 1250 total updates | 1.8744 | 0.04 | 4.4384 | 0.2387 |
| Run 19 shuffled-goal local eval | 4.5738 | n/a | n/a | n/a |

The corrected PPO objective is goal-sensitive locally, and increasing the
teacher-action penalty from `0.5` to `1.0` slightly improves local distance, but
the learned policy is still far off the teacher/contact action manifold in
rollout. Continuing to 1250 total PPO updates gives the first nonzero full-state
PPO oracle held success (`0.04`) and reduces teacher-action MAE, so scratch PPO
is not fully saturated yet.

For historical comparison only, update-period-1/full-state replanning reproduces
the old Phase-B behavior but is not the target hierarchy:

| Goal source | Low-level | Success |
| --- | --- | ---: |
| oracle, update=1 | Phase-B full BC | 0.41 |
| oracle, update=1 | Run 16 full PPO | 0.00 |
| learned, update=1 | Phase-B full BC | 0.29 |
| learned, update=1 | Run 16 full PPO | 0.00 |

## Current Conclusion

Reachability PPO is not a dead end, but the reward/interface and action-manifold
constraint matter. TCP-only goals were a weak proxy for the desired option
outcome. Object-pose goals improved task relevance, and full-state goals are the
most semantically correct local reachability target so far.

The strongest task-success result remains constrained object-pose PPO with a
teacher-action penalty. The strongest hierarchy baseline is the corrected
Phase-C time-conditioned full-state BC (`0.69-0.74` oracle held success across
seed banks). Recomputed full-state PPO still fails task success because it
drifts too far from teacher/contact actions. Longer scratch training helps but
is still far behind BC. Before switching to fine-tuning, one more 8x-style
continuation is justified; after that, the next promising direction is to use
the BC policy structurally: BC warm start, residual-on-BC full-state PPO, or a
stronger KL/behavior-cloning constraint.
