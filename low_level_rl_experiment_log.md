# Low-Level RL Fine-Tuning Experiment Log

This is the chronological execution record for
[`low_level_rl_tuning_plan.md`](low_level_rl_tuning_plan.md). Commands use
`configs/pusht_incremental.yaml` unless noted otherwise. Simulator interaction
counts are reported separately from demonstration counts.

## 2026-06-23 - RL-00: Plan and capability audit

- Created the active goal to execute the low-level RL fine-tuning study.
- Hardware: NVIDIA RTX 4060 Ti (16 GiB); 60 GiB disk free at start.
- Existing frozen artifacts are complete for `N_demo=500` and `N_demo=1000`,
  seeds 0-2: VAE-512 representation, deterministic high level, and
  deterministic goal-conditioned low level.
- The demonstration HDF5 contains `dino`, `proprio`, and `actions`, but no
  reset seed or complete simulator state. Exact arbitrary local resets cannot
  be reconstructed from this corpus.
- Per the plan's reset fallback, initial RL training uses Mode C full hierarchy
  rollouts. Every held 10-step goal segment supplies the local latent-progress
  objective. Stored held-out teacher windows are used for reward-scale and
  threshold audits, not claimed as online Mode A episodes.
- Frozen: DINO, VAE, all normalizers, and high-level predictor. R1 trains only
  a zero-initialized residual actor and its critic.

## 2026-06-23 - RL-01: Phase 0 implementation and frozen parity

- Implemented `hcl-poc low-level-rl` audit, R1 training, and evaluation
  commands in `src/hcl_poc/low_level_rl.py`.
- Residual PPO condition: the existing 7,065D low-level input. Actor/critic:
  two hidden layers of width 256; actor output is 3D Gaussian residual with a
  zero-initialized mean and initial log standard deviation `-2.3`.
- Frozen parameter audit: 32,718,339 parameters, zero trainable tensors.
- Teacher-window audit: 6,969 validation segments; normalized ten-step initial
  distance mean `1.754`; one-teacher-step distance mean `0.849`; 90th-percentile
  one-step goal threshold `1.421`.
- Executed-action clipping and previous-action feedback are applied after the
  residual is added to the frozen BC action.
- Unit tests: 20 passed; lint passed.

Frozen hierarchy parity on the first 100 original evaluation seeds:

| metric | new RL rollout engine | prior preliminary result |
| --- | ---: | ---: |
| task success | 0.29 | 0.28 |
| segment initial latent MSE | 1.750 | not previously logged |
| segment final latent MSE | 1.757 | not previously logged |
| segment goal-reach rate | 0.513 | not previously logged |
| action saturation | 0.043 | not previously logged |

The one-episode success difference is consistent with the same underlying
policy and validates the deployment wiring. The frozen policy has slightly
negative mean latent progress, so the local objective is not already
saturated.

## 2026-06-23 - RL-02: R1 50k smoke matrix

The first implementation propagated GAE across held-goal boundaries. This was
incorrect for the local MDP because rewards from a new goal affected actions
for the previous goal. The implementation now terminates GAE every 10 steps
while continuing the physical rollout. Checkpoints are immutable every 25k;
selection uses the development seed bank rather than on-policy training loss.

Corrected segment-MDP results on 100 development episodes:

| alpha | reward | success | final latent MSE | goal reach | residual L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.00 | frozen | 0.35 | 1.548 | 0.570 | 0.000 |
| 0.05 | progress | 0.39 | 1.680 | 0.586 | 0.007 |
| 0.05 | terminal | 0.33 | 1.609 | 0.560 | 0.007 |
| 0.05 | terminal + 0.1 task | **0.46** | **1.512** | **0.620** | 0.006 |
| 0.10 | progress | 0.33 | 1.635 | 0.566 | 0.014 |
| 0.10 | terminal | 0.37 | 1.619 | 0.547 | 0.013 |
| 0.10 | terminal + 0.1 task | 0.34 | 1.486 | 0.610 | 0.015 |
| 0.25 | progress | 0.23 | 1.626 | 0.539 | 0.034 |
| 0.25 | terminal | 0.30 | 1.534 | 0.579 | 0.037 |
| 0.25 | terminal + 0.1 task | 0.31 | 1.689 | 0.522 | 0.033 |

The apparent `alpha=0.05` task-reward gain did not replicate at 300 episodes:
frozen success was `0.363`, versus `0.293` for R1. R1 also worsened final
latent MSE (`1.459 -> 1.646`) and goal reach (`0.621 -> 0.560`). Therefore no
50k recipe passes either the local or full-hierarchy gate. The two least
aggressive `alpha=0.05` recipes proceed to the planned 500k development budget
to test whether this is insufficient optimization rather than a bad asymptote.

## 2026-06-23 - RL-03: R1 500k development decision

Three R1 variants were run to 500k or to diagnostic 50k limits and evaluated
on 300 development episodes. Immutable checkpoints were evaluated at roughly
100k-step intervals.

| method | best step | success | final latent MSE | goal reach | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| frozen | 0 | 0.363 | 1.459 | 0.621 | reference |
| R1 progress | 200,704 | 0.350 | 1.512 | 0.598 | reject |
| R1 terminal + 0.1 task | 200,704 | 0.373 | 1.539 | 0.580 | reject: tiny success gain, worse latent reach |
| R1 task only, full GAE | 500,736 | 0.290 | 1.588 | 0.562 | reject |
| R1 alpha 0.50 diagnostics | 50,176 | <=0.223 | >=1.642 | <=0.537 | reject |

R1 does not pass the local reachability gate or the full-hierarchy gate. The
best success point is only one percentage point above frozen on 300 episodes
and simultaneously worsens latent final distance and goal reach. This is not a
usable positive result.

Direct final-layer low-level fine-tuning was partially scaffolded as reusable
model code, but the executable CLI was not kept because the R1 gate failed and
the plan explicitly treats R2-R4 as follow-ons only after R1 stability. The
next serious branch should add stricter BC regularization and/or a physical
progress reward before running direct fine-tuning.

## 2026-06-23 - RL-04: R3 direct low-level fine-tuning scaffold

R1 did not produce a reliable improvement, so the next branch is a conservative
R3 variant: initialize from the deterministic BC low level, train only the final
low-policy linear layer plus policy log-standard-deviation and critic, and add
a BC action penalty against the frozen low-level action on every PPO minibatch.
The high level, VAE, DINO, normalizers, and all earlier low-level layers remain
frozen.

Implemented:

```text
hcl-poc low-level-rl train-r3
```

Key defaults:

| setting | value |
| --- | --- |
| trainable scope | low-policy final layer + logstd + critic |
| optimizer LR | 3e-5 |
| PPO rollout | 32 envs x 32 steps |
| GAE boundary | held-goal segment ends |
| reward | latent progress + terminal latent distance + optional task reward |
| regularizer | `bc_weight * ||mean_action - frozen_bc_action||^2` |

Smoke command:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r3 --n-demo 500 --seed 0 --run-name r3_bc1_terminal_smoke --steps 2048 --bc-weight 1.0 --terminal-weight 1.0 --force
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval --n-demo 500 --seed 0 --run-name r3_bc1_terminal_smoke --episodes 20 --seed-start 3200000 --force
```

Smoke result: checkpoint creation and direct-checkpoint evaluation both work.
The 20-episode eval is too small for a decision, but deterministic action drift
from frozen BC is low (`residual_l2_mean=0.0044`) and action saturation is
similar to the frozen hierarchy (`0.043`).

## 2026-06-23 - RL-05: R3 50k development sweep

The first direct R3 attempts used the same exploration scale as the residual
adapter (`initial_logstd=-2.3`, action std about `0.10`). This was too much for
direct raw-action sampling: training rollouts had sampled action deltas around
`0.16` L2 from frozen BC, even though deterministic checkpoints stayed close.
Direct R3 now uses a separate lower exploration scale:

```text
low_level_rl.direct_initial_logstd = -4.0
```

Development runs at `N_demo=500`, seed 0:

| run | checkpoint | eval eps | success | final latent MSE | goal reach | deterministic action drift | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| frozen_reference | none | 100 | 0.38 | 1.530 | 0.595 | 0.000 | reference |
| r3_bc1_terminal_50k | latest | 100 | 0.28 | 1.613 | 0.561 | 0.027 | reject |
| r3_bc1_terminal_50k | best_train_latent | 100 | 0.29 | 1.718 | 0.544 | 0.017 | reject |
| r3_bc1_task01_50k | latest | 100 | 0.25 | 1.658 | 0.555 | 0.021 | reject |
| r3_bc1_lownoise_50k | latest | 100 | 0.28 | 1.670 | 0.553 | 0.014 | reject |
| r3_bc1_lownoise_progress1_50k | latest | 100 | 0.29 | 1.508 | 0.592 | 0.013 | reject latest |
| r3_bc1_lownoise_progress1_50k | best_train_latent at 5,120 steps | 100 | 0.40 | 1.622 | 0.587 | 0.0056 | promising |

The task-progress reward is:

```text
r_task_progress = env_dense_reward_t+1 - env_dense_reward_t
```

It is used as an additional shaping term, not a replacement for the latent
goal-reaching reward. The best selected progress checkpoint was then evaluated
on the larger 300-episode development bank:

| policy | checkpoint | eval eps | success | final latent MSE | goal reach | max reward | action drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_reference_300 | none | 300 | 0.313 | 1.576 | 0.578 | 0.494 | 0.000 |
| r3_bc1_lownoise_progress1_50k_best300 | best_train_latent at 5,120 steps | 300 | 0.390 | 1.597 | 0.588 | 0.547 | 0.0056 |

Interpretation: the best low-noise R3 checkpoint improves task success and max
reward for seed 0 while keeping the deterministic policy very close to BC. The
latent goal metric is not clearly better than frozen, so this should be treated
as a task-performance signal from mild low-level adaptation rather than proof
that the latent reachability objective alone improved. Next step: repeat this
same recipe for seeds 1 and 2 before doing longer or final-budget evaluations.

## 2026-06-23 - RL-06: R3 seed confirmation at N=500

Repeated the selected R3 recipe for seeds 1 and 2:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r3 --n-demo 500 --seed <seed> --run-name r3_bc1_lownoise_progress1_50k --steps 50176 --bc-weight 1.0 --terminal-weight 1.0 --task-progress-weight 1.0 --force
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval --n-demo 500 --seed <seed> --run-name r3_bc1_lownoise_progress1_50k_best300 --episodes 300 --seed-start 3200000 --checkpoint artifacts/incremental/low_level_rl/n500/seed<seed>/r3_bc1_lownoise_progress1_50k/best_train_latent.pt --force
```

Matched frozen references were evaluated with the same 300 development seeds.

| seed | frozen success | R3 selected success | delta | frozen final latent MSE | R3 final latent MSE | R3 selected step | R3 action drift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.313 | 0.390 | +0.077 | 1.576 | 1.597 | 5,120 | 0.0056 |
| 1 | 0.313 | 0.300 | -0.013 | 1.563 | 1.507 | 39,936 | 0.0158 |
| 2 | 0.290 | 0.303 | +0.013 | 1.482 | 1.485 | 18,432 | 0.0071 |
| mean | 0.306 | 0.331 | +0.026 | 1.540 | 1.530 | - | 0.0095 |

Decision: R3 with low exploration and task-progress shaping is the first
method that improves mean success over the frozen hierarchy, but the gain is
modest and seed-dependent. It is worth testing at the confirmation budget
`N_demo=1000`, but it is not yet strong enough to justify a large final sweep
without that check.

## 2026-06-23 - RL-07: N=1000 confirmation for selected R3

Ran the selected R3 recipe at the confirmation budget:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r3 --n-demo 1000 --seed 0 --run-name r3_bc1_lownoise_progress1_50k --steps 50176 --bc-weight 1.0 --terminal-weight 1.0 --task-progress-weight 1.0 --force
```

300-episode development comparison:

| policy | success | final latent MSE | goal reach | max reward | action drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen N=1000 seed 0 | 0.553 | 1.104 | 0.780 | 0.676 | 0.000 |
| R3 selected N=1000 seed 0 | 0.493 | 1.187 | 0.753 | 0.629 | 0.012 |

Decision: the selected R3 recipe does not confirm at `N_demo=1000`. It appears
useful only as a small-data adaptation for some seeds, and can harm a stronger
frozen hierarchy. Do not scale this R3 recipe to final evaluation. Next
follow-up is to test the same task-progress shaping in the safer R1 residual
adapter, where exploration is alpha-limited and cannot directly perturb the BC
action as much as direct PPO.

## 2026-06-23 - RL-08: R1 residual with task-progress shaping

Ran an R1 follow-up using the same task-progress reward but the safer
alpha-limited residual action:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r1 --n-demo <N> --seed <seed> --run-name r1_a005_progress1_50k --steps 50176 --alpha 0.05 --terminal-weight 1.0 --task-progress-weight 1.0 --force
```

Selected checkpoint is `best_train_latent.pt`.

N=500, 300-episode development comparison:

| seed | frozen success | R1 selected success | delta | frozen final latent MSE | R1 final latent MSE | R1 selected step | R1 action drift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.313 | 0.370 | +0.057 | 1.576 | 1.542 | 39,936 | 0.0050 |
| 1 | 0.313 | 0.307 | -0.007 | 1.563 | 1.554 | 20,480 | 0.0043 |
| 2 | 0.290 | 0.300 | +0.010 | 1.482 | 1.554 | 11,264 | 0.0029 |
| mean | 0.306 | 0.326 | +0.020 | 1.540 | 1.550 | - | 0.0040 |

N=1000, seed 0 confirmation:

| policy | success | final latent MSE | goal reach | max reward | action drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen N=1000 seed 0 | 0.553 | 1.104 | 0.780 | 0.676 | 0.000 |
| R1 selected N=1000 seed 0 | 0.527 | 1.133 | 0.767 | 0.656 | 0.004 |

Decision: task-progress shaping makes R1 better than the original R1 attempts
and gives a small N=500 mean improvement, but it still does not confirm at
`N_demo=1000`. The best current conclusion is a useful negative/limited result:
low-level PPO can slightly adapt weak hierarchies when heavily constrained, but
the effect is not robust enough to use as the final method.
