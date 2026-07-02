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
| Phase-C full BC | 1.6563 | 0.69 | 1.6358 | 0.0414 |
| Run 19 recomputed full PPO | 1.5264 | 0.00 | 5.2944 | 0.2812 |
| Run 20 recomputed full PPO, teacher penalty 1.0 | 1.4703 | 0.00 | 4.9881 | 0.2870 |
| Run 21 long PPO, 1250 total updates | 1.8744 | 0.04 | 4.4384 | 0.2387 |
| Run 22 long PPO, 2250 total updates | 1.5018 | 0.01 | 5.1632 | 0.2493 |
| Run 19 shuffled-goal local eval | 4.5738 | n/a | n/a | n/a |

The corrected PPO objective is goal-sensitive locally, and increasing the
teacher-action penalty from `0.5` to `1.0` slightly improves local distance, but
the learned policy is still far off the teacher/contact action manifold in
rollout. Continuing to 1250 total PPO updates gives the first nonzero full-state
PPO oracle held success (`0.04`) and reduces teacher-action MAE, so scratch PPO
is not fully saturated yet.

Continuing again to 2250 total updates improves the same-reset-bank local
distance (`1.5018`) and lowers final training-window teacher-action MAE
(`0.0837`), but held-subgoal task success remains poor (`0.01`). This is the
stop point for same-bank continuation.

Open-loop deployed-state reachability gives a more nuanced picture than the
fixed reset bank:

| Collector rollout | Candidate branch | Terminal full-goal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | 0.8351 | 0.1555 | 1.6420 | 0.8674 |
| Phase-C full BC | Run 20 full PPO | 0.8777 | 0.2302 | 2.3527 | 0.9298 |
| Phase-C full BC | Run 21 long PPO | 0.8187 | 0.1688 | 2.1240 | 0.9474 |
| Run 21 long PPO | Phase-C full BC | 10.3369 | 2.0652 | 19.6268 | 0.6777 |
| Run 21 long PPO | Run 20 full PPO | 3.2549 | 1.2234 | 5.5571 | 0.8809 |
| Run 21 long PPO | Run 21 long PPO | 3.5435 | 1.4857 | 6.1962 | 0.8848 |
| Run 22 long PPO | Phase-C full BC | 8.4377 | 1.7301 | 16.3957 | 0.6816 |
| Run 22 long PPO | Run 22 long PPO | 4.1308 | 1.2215 | 5.8609 | 0.8906 |

This argues against a pure reset-bank overfitting diagnosis: PPO is competitive
with BC on BC-generated states and much better than BC on states generated by
the PPO hierarchy itself. The remaining issue is still task compatibility:
stronger local full-goal distance has not translated into Phase-C-level success.

The next direction is reset-distribution debugging, not further same-bank
training: train/evaluate on a mixture of original demo local windows,
BC-hierarchy deployed states, and PPO-hierarchy deployed states. Online expert
branching should remain diagnostic or upper-bound evidence, not the core POC
method.

First reset-mixture result:

| Policy | Reset distribution | Local terminal dist. | Oracle held success | Hold full-goal dist. | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Run 22 long PPO | demo/teacher bank | 1.5018 | 0.01 | 5.1632 | 0.2493 |
| Run 23 reset-mixture PPO | 50% demo, 25% BC-deployed, 25% PPO-deployed | 2.1294 | 0.00 | 4.3716 | 0.2712 |

Run 23 improves deployed-state branch reachability on its own rollout
distribution:

| Collector rollout | Candidate branch | Terminal full-goal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: |
| Run 23 reset-mixture PPO | Phase-C full BC | 6.6453 | 1.7248 | 13.6863 | 0.6855 |
| Run 23 reset-mixture PPO | Run 22 long PPO | 1.9530 | 1.0845 | 3.4902 | 0.9180 |
| Run 23 reset-mixture PPO | Run 23 reset-mixture PPO | 1.8358 | 1.0318 | 3.2796 | 0.9219 |

But task success remains zero. The next diagnostic should isolate target
quality by using oracle target future states from deployed resets as an upper
bound. If that succeeds, the learned-high 28D-goal-to-pseudo-future-state
construction is the bottleneck; if it fails, the issue is action/contact
compatibility or the PPO objective.

Oracle-target reset-mixture diagnostic:

| Policy | Reset target source | Local terminal dist. | Oracle held success | Hold full-goal dist. | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Run 23 reset-mixture PPO | learned high pseudo future | 2.1294 | 0.00 | 4.3716 | 0.2712 |
| Run 24 reset-mixture PPO | oracle future state diagnostic | 2.5682 | 0.03 | 3.0692 | 0.2519 |

Oracle target states help held-goal distance and give a small success increase,
but they do not close the gap to Phase-C full BC. That points away from pure
target-quality failure and toward action/contact compatibility. The next
full-state path should use BC structurally, for example BC warm start,
residual-on-BC, or an explicit KL/BC-prior regularizer.

BC-structured reset-mixture diagnostic:

| Policy | Reset distribution | Local terminal dist. | Oracle held success | Hold full-goal dist. | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase-C full BC | demo-trained BC baseline | n/a | 0.72 | 1.7071 | 0.0421 |
| Run 22 long PPO | demo/teacher bank | 1.5018 | 0.01 | 5.1632 | 0.2493 |
| Run 23 reset-mixture PPO | 50% demo, 25% BC-deployed, 25% PPO-deployed | 2.1294 | 0.00 | 4.3716 | 0.2712 |
| Run 24 oracle-target reset PPO | reset mixture, oracle future-state diagnostic | 2.5682 | 0.03 | 3.0692 | 0.2519 |
| Run 25 BC-warm-start reset PPO | reset mixture, BC-warm-started actor | 3.6321 | 0.16 | 2.3472 | 0.1314 |
| Run 26 BC-prior reset PPO | reset mixture, BC warm start + BC-prior loss | 3.6656 | 0.21 | 2.7468 | 0.1241 |
| Run 27 iterative aggregation PPO | demo + BC-deployed + Run26-deployed bank | 1.8723 | 0.06 | 2.0819 | 0.1326 |
| Run 28 iterative aggregation BC-prior-5 PPO | same bank as Run 27, stronger BC prior | 1.9199 | 0.25 | 1.4907 | 0.0870 |
| Run 29 iterative aggregation round-2 PPO | demo + BC-deployed + Run28-deployed bank | 1.6893 | 0.25 | 1.7237 | 0.1032 |
| Run 30 residual-on-BC PPO | round-2 bank, BC base + bounded residual | 1.9028 | 0.73 | 2.7548 | 0.0513 |
| Run 31 residual alpha-0.25 PPO | Run 30 with larger residual radius | 2.2360 | 0.65 | 1.7816 | 0.0498 |
| Run 32 residual no-penalty PPO | Run 30 with residual penalty removed | 1.9913 | 0.69 | 1.6898 | 0.0421 |

Run 25 is the first full-state PPO variant with meaningful held-subgoal task
recovery, and Run 26 improves it further with an explicit BC-prior loss. Run 27
confirms that iterative reset aggregation can improve local/deployed full-goal
reachability, but task success regresses when the BC prior is too weak. Run 28
uses the same aggregation bank with a stronger BC prior and gives the best
full-state PPO result so far (`0.25` oracle held success). Run 29 improves
deployed-state reachability in a second aggregation round, but task success is
flat. Run 30 changes the policy parameterization to residual-on-BC and gives
the best full-state task result so far (`0.73` oracle held success, shuffled
success `0.00`). Run 31 increases the residual radius and recovers some branch
reachability, but held success drops to `0.65`; Run 32 removes the residual
penalty and also underperforms Run 30. The result supports the
reset-distribution-shift hypothesis, while also showing that reset coverage
alone is insufficient: the PPO policy needs stronger BC/action-manifold
structure.

Run 25 deployment-state branch reachability:

| Collector rollout | Candidate branch | Terminal full-goal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | 0.8697 | 0.1672 | 1.5919 | 0.8583 |
| Phase-C full BC | Run 22 long PPO | 0.8122 | 0.1750 | 2.1522 | 0.9553 |
| Phase-C full BC | Run 25 BC-warm-start PPO | 1.0739 | 0.3142 | 2.3017 | 0.9029 |
| Phase-C full BC | Run 26 BC-prior PPO | 0.9837 | 0.2861 | 2.0272 | 0.9316 |
| Run 25 BC-warm-start PPO | Phase-C full BC | 2.9800 | 0.6122 | 6.7456 | 0.7698 |
| Run 25 BC-warm-start PPO | Run 22 long PPO | 1.1469 | 0.3602 | 2.6900 | 0.9768 |
| Run 25 BC-warm-start PPO | Run 25 BC-warm-start PPO | 2.0967 | 0.9046 | 4.0808 | 0.8801 |
| Run 26 BC-prior PPO | Phase-C full BC | 2.5858 | 0.5424 | 6.0685 | 0.7615 |
| Run 26 BC-prior PPO | Run 22 long PPO | 1.0501 | 0.3572 | 2.5966 | 0.9596 |
| Run 26 BC-prior PPO | Run 26 BC-prior PPO | 1.6019 | 0.8024 | 3.9200 | 0.8981 |
| Run 27 iterative aggregation PPO | Phase-C full BC | 2.5061 | 0.5853 | 5.1988 | 0.7793 |
| Run 27 iterative aggregation PPO | Run 26 BC-prior PPO | 1.7118 | 0.8763 | 3.6595 | 0.8791 |
| Run 27 iterative aggregation PPO | Run 27 iterative aggregation PPO | 1.3109 | 0.6035 | 3.0399 | 0.9021 |
| Run 28 iterative aggregation BC-prior-5 PPO | Phase-C full BC | 2.3780 | 0.3135 | 5.6223 | 0.7457 |
| Run 28 iterative aggregation BC-prior-5 PPO | Run 26 BC-prior PPO | 1.6553 | 0.4794 | 3.3181 | 0.8632 |
| Run 28 iterative aggregation BC-prior-5 PPO | Run 28 iterative aggregation BC-prior-5 PPO | 1.4456 | 0.3571 | 2.9417 | 0.8825 |
| Run 29 iterative aggregation round-2 PPO | Phase-C full BC | 2.7721 | 0.5745 | 5.2880 | 0.7390 |
| Run 29 iterative aggregation round-2 PPO | Run 28 BC-prior-5 PPO | 1.5585 | 0.5494 | 2.7619 | 0.8419 |
| Run 29 iterative aggregation round-2 PPO | Run 29 round-2 PPO | 1.4397 | 0.4837 | 2.3993 | 0.8914 |
| Run 30 residual-on-BC PPO | Phase-C full BC | 1.2460 | 0.2036 | 2.3291 | 0.8281 |
| Run 30 residual-on-BC PPO | Run 29 round-2 PPO | 0.9670 | 0.2088 | 1.6343 | 0.9180 |
| Run 30 residual-on-BC PPO | Run 30 residual-on-BC PPO | 1.6468 | 0.2095 | 2.3214 | 0.8516 |
| Run 31 residual alpha-0.25 PPO | Phase-C full BC | 1.2846 | 0.2283 | 2.5737 | 0.8438 |
| Run 31 residual alpha-0.25 PPO | Run 30 residual-on-BC PPO | 1.3896 | 0.2424 | 2.4122 | 0.8571 |
| Run 31 residual alpha-0.25 PPO | Run 31 residual alpha-0.25 PPO | 1.2273 | 0.2969 | 2.3063 | 0.8781 |
| Run 32 residual no-penalty PPO | Phase-C full BC | 2.0886 | 0.2025 | 2.5489 | 0.8519 |
| Run 32 residual no-penalty PPO | Run 30 residual-on-BC PPO | 2.3258 | 0.1971 | 2.3788 | 0.8577 |
| Run 32 residual no-penalty PPO | Run 32 residual no-penalty PPO | 2.0907 | 0.1861 | 2.2640 | 0.8577 |

Do not continue plain same-bank training as the main line. Static reset mixtures
plus BC structure still trail BC, and the first iterative aggregation round only
works when the BC/action constraint is strong enough. The next scoped experiment
should keep the aggregated reset banks but change the policy constraint:
residual-on-BC PPO is now validated as the stronger task-success direction.
Oracle branching should remain diagnostic or upper-bound evidence only.

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

Corrected learned-high full-state evaluation now uses recomputed full-goal
features by converting the learned 28D high-level output into a pseudo future
state. On that metric, Run 30 is goal-sensitive but does not beat Phase-C BC:

| Goal source | Low-level | Success | Hold full-goal dist. | Teacher action MAE |
| --- | --- | ---: | ---: | ---: |
| learned | Phase-C full BC | 0.68 | 2.7514 | 0.0714 |
| learned | Run 30 residual-on-BC PPO | 0.59 | 1.6360 | 0.0664 |
| shuffled learned | Phase-C full BC | 0.00 | 17.0090 | 0.3455 |
| shuffled learned | Run 30 residual-on-BC PPO | 0.00 | 15.6677 | 0.3506 |

This separates the current bottlenecks: residual-on-BC solves oracle-subgoal
execution at BC-level success, but learned-high deployment still favors the
Phase-C BC baseline.

Learned-high target-quality audit:

| Rollout policy | Learned current dist. | Oracle current dist. | Learned-vs-oracle full L2 | Object-pose L2 | TCP L2 | Robot L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | 8.2271 | 8.7889 | 0.9268 | 0.1080 | 0.0295 | 0.9106 |
| Run 30 residual-on-BC PPO | 7.7197 | 8.7102 | 0.9171 | 0.0919 | 0.0267 | 0.9057 |

The learned high-level targets are not grossly wrong in object/TCP space and
are actually closer to the current state than oracle `t+10` targets on average.
Most learned-vs-oracle error is robot-state error. The learned-high gap is
therefore more likely target conservatism, robot-configuration mismatch, or
contact/action compatibility under learned subgoals.

The strongest task-success result remains constrained object-pose PPO with a
teacher-action penalty. The strongest hierarchy baseline is the corrected
Phase-C time-conditioned full-state BC (`0.69-0.74` oracle held success across
seed banks). Recomputed full-state PPO still fails task success after 2250 total
updates on the same reset bank. Reset-mixture plus BC warm start improves
full-state PPO task success to `0.16`, and adding a BC-prior loss improves it
to `0.21`. The first iterative aggregation round improves local/deployed
reachability but drops success to `0.06` with a weak BC prior; increasing the
BC-prior weight recovers and improves success to `0.25`. The next promising
direction is not more identical aggregation rounds; Run 29 shows that round 2
improves deployed reachability but not success. The next step should use the
aggregated reset banks with residual-on-BC. Run 30 already reaches BC-level
task success, so the next residual ablation should trade residual radius and
penalty carefully. Run 31 shows that simply increasing residual radius to
`0.25` hurts task success, and Run 32 shows that removing the residual penalty
does not recover the Run 30 win. Run 30 is the best current full-state
low-level policy.
