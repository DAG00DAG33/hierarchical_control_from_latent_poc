# VAE512 Scratch Low-Level RL Seed-0 Reward Selection

Date: 2026-06-29

## Setup

- Architecture: VAE512 deterministic hierarchy, `vae512_w2048_b1e6`
- Budgets: `N=500`, `N=1800`
- Seed: `0`
- Scratch actor: randomly initialized direct low-level policy
- Frozen modules: VAE512 encoder and learned high-level policy
- Reward distance: learned VAE512 reachability model `D_psi(z_t, z_goal)`
- Training: `100160` PPO environment steps per scratch run
- Evaluation: 100 fresh vectorized episodes from `seed_start=3450000`

Reward variants:

- `scratch_dpsi_terminal`: terminal `-D_psi`
- `scratch_dpsi_paired`: paired terminal improvement over the frozen BC low level
- `scratch_dpsi_progress`: terminal `-D_psi` plus `0.1 * D_psi` progress

## Results

| N | run | success | max reward | segment `D_psi` reduction | raw latent reduction | goal reach rate | action saturation | action delta L2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | frozen baseline | 0.300 | 0.473 | 0.0521 | 0.1284 | 0.5383 | 0.042 | 0.000 |
| 500 | scratch terminal | 0.000 | 0.129 | 0.0064 | -0.1483 | 0.0016 | 0.085 | 1.160 |
| 500 | scratch paired | 0.000 | 0.118 | -0.0034 | -0.1764 | 0.0000 | 0.128 | 1.383 |
| 500 | scratch progress | 0.010 | 0.137 | -0.0037 | -0.1264 | 0.0016 | 0.070 | 0.714 |
| 1800 | frozen baseline | 0.620 | 0.728 | 0.1079 | 0.1678 | 0.8906 | 0.049 | 0.000 |
| 1800 | scratch terminal | 0.000 | 0.123 | 0.0019 | -0.1264 | 0.0305 | 0.036 | 1.037 |
| 1800 | scratch paired | 0.000 | 0.117 | -0.0062 | -0.2297 | 0.0031 | 0.000 | 1.143 |
| 1800 | scratch progress | 0.000 | 0.136 | -0.0051 | -0.6050 | 0.0031 | 0.244 | 1.221 |

## Decision

No scratch reward variant passes the scaling gate. All scratch variants strongly
underperform the frozen VAE512 low level in task success, max task reward,
selected-distance reduction, raw latent reduction, and local goal reach rate.

Do not run the three-seed final scratch evaluation from these variants. The
next useful step would be a redesigned scratch curriculum or a different action
initialization/regularization strategy, but that is outside the reviewed
execution path for this run.
