# VAE512 Scratch Low-Level RL High-Env Reward Selection

Date: 2026-06-30

## Setup

This rerun increases the scratch RL effort relative to the first seed-0 sweep.

- Architecture: VAE512 deterministic hierarchy, `vae512_w2048_b1e6`
- Budgets: `N=500`, `N=1800`
- Seed: `0`
- Frozen modules: VAE512 encoder and learned high-level policy
- Evaluation: 100 fresh vectorized episodes from `seed_start=3450000`
- Main high-env runs: `4096` envs, `10` rollout steps, `1,024,000`
  environment steps, `25` PPO updates, `2` update epochs, `64` minibatches
- Paired runs: `2304` scratch envs plus `2304` matched frozen envs,
  `1,036,800` scratch environment steps, `45` PPO updates, `2` update epochs,
  `36` minibatches

The first attempt to run paired reward with `4096` scratch envs failed at
camera allocation because paired mode creates a second visual rollout, roughly
doubling camera count to `8192`. The paired result below uses `2304` scratch
envs, or about `4608` simultaneous visual envs total.

Reward variants:

- `terminal`: terminal `-D_psi`
- `progress`: terminal `-D_psi` plus `0.1 * D_psi` progress
- `task_mix`: progress variant plus `0.05 *` environment reward
- `paired_e2304`: paired terminal improvement over the frozen BC low level
- `raw_l2`: raw VAE latent L2 sanity ablation

## Results

| N | run | success | max reward | segment distance reduction | raw latent reduction | goal reach rate | action saturation | action delta L2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | frozen baseline | 0.300 | 0.473 | 0.0521 | 0.1284 | 0.5383 | 0.042 | 0.000 |
| 500 | terminal | 0.000 | 0.127 | 0.0390 | -0.0088 | 0.0195 | 0.000 | 0.880 |
| 500 | progress | 0.000 | 0.126 | 0.0365 | -0.0152 | 0.0445 | 0.000 | 0.802 |
| 500 | task_mix | 0.000 | 0.132 | 0.0383 | -0.0150 | 0.0312 | 0.000 | 0.821 |
| 500 | paired_e2304 | 0.000 | 0.140 | 0.0232 | -0.0108 | 0.0063 | 0.000 | 0.776 |
| 500 | raw_l2 | 0.000 | 0.123 | -0.0188 | -0.0188 | 0.0352 | 0.000 | 1.111 |
| 1800 | frozen baseline | 0.620 | 0.728 | 0.1079 | 0.1678 | 0.8906 | 0.049 | 0.000 |
| 1800 | terminal | 0.000 | 0.136 | 0.0318 | -0.0181 | 0.3078 | 0.000 | 0.812 |
| 1800 | progress | 0.000 | 0.139 | 0.0333 | -0.0359 | 0.3867 | 0.000 | 0.666 |
| 1800 | task_mix | 0.000 | 0.135 | 0.0305 | -0.0182 | 0.2617 | 0.000 | 0.759 |
| 1800 | paired_e2304 | 0.000 | 0.137 | 0.0011 | -0.0354 | 0.1961 | 0.000 | 0.984 |
| 1800 | raw_l2 | 0.000 | 0.122 | -0.0158 | -0.0158 | 0.5430 | 0.000 | 0.797 |

## Decision

No high-effort scratch variant passes the scaling gate. The larger runs improve
the learned-distance local metric compared with the earlier 32-env run, but
they still collapse task success to zero and make raw VAE latent progress
negative. The frozen low level remains far better on both local reachability
and full task success.

Do not run a three-seed final scratch evaluation from these variants.
