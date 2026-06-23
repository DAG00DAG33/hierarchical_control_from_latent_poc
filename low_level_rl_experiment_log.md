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
