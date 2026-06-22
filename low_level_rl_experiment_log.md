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
