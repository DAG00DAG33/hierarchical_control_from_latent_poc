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

## Current Conclusion

Reachability PPO is not a dead end, but the reward/interface matters. Pure TCP
or pure reachability objectives can learn locally good but task-bad controllers.
The strongest current result is constrained object-pose PPO with a teacher-action
penalty: it improves local reachability and beats the matching object-pose BC
low-level under both oracle and learned object-pose high-level goals.

The remaining bottleneck is likely high-level goal quality/contact alignment or
missing robot/contact geometry in the object-pose-only interface, not a simple
low-level inability to reach learned object-pose goals.
