# VAE-512 Sample-Efficiency Experiment Log

This is the chronological execution record for
[`vae512_sample_efficiency_experiment_plan.md`](vae512_sample_efficiency_experiment_plan.md).

## 2026-06-22 - SE-00: Initial audit and implementation start

- **Plan commit:** `0a63e6c`.
- **Hardware:** NVIDIA RTX 4060 Ti, 16,380 MiB total and 15,507 MiB free at
  audit time.
- **Disk:** 82 GiB free on the workspace filesystem.
- **Existing implementation:** legacy Phase 12 is not reused as the runner. It
  hard-codes AE-256, `k=2`, one policy seed, 100 deployable episodes, and only
  10 oracle episodes. It also omits flat latent flow.
- **Reuse decision:** retain the validated VAE-512 representation,
  `k=10,U=10,H=1` deterministic hierarchy, local branch-oracle evaluator,
  DINO observation path, and common flow model. Add a separate
  `vae512_scaling` artifact/result namespace and matched flat/flow methods.
- **Primary data:** fixed 2,000-trajectory PPO corpus; nested train prefixes
  `{50,100,200,500,1000,1800}` and fixed final 200 validation trajectories.
- **Evaluation target:** 500 episodes for six deployable methods and 50 for
  the oracle, for each of three complete training seeds.

Implementation and the `N=50, seed=0` smoke benchmark are in progress. Runtime
ETA will be added only after measuring the complete smoke point.

## 2026-06-22 - SE-01: `N=50, seed=0` full-training smoke

- **Implementation commit:** `f653bb5`.
- **Manifest:** 50 nested-prefix trajectories, 2,311 transitions, fixed 200
  validation trajectories with 8,969 transitions, SHA256
  `03a45f4c83acd382153f80c0a3968b821b8dda359dd894bf50b3349faf9f7b5f`.
- **VAE-512:** 791 s, selected epoch 1, validation reconstruction `0.4623`.
- **Shared deterministic hierarchy:** 437 s, selected epoch 58, oracle and
  predicted validation action MAE `0.1340/0.1354`.
- **Flow high level:** 200 s, selected epoch 45, predicted-goal action MAE
  `0.1390`.
- **Flat latent deterministic/flow:** 89/96 s, validation action MAE
  `0.1343/0.1316`.
- **Flat observation deterministic/flow:** 171/175 s, validation action MAE
  `0.1325/0.1405`.
- **Total training wall time:** approximately 28 minutes.
- **Storage:** 318,470,959 artifact bytes (about 304 MiB).

Short rollout audit on 20 fixed deployable seeds and five oracle seeds:

| method | success | final reward |
| --- | ---: | ---: |
| deterministic hierarchy | 0.05 | 0.232 |
| flow hierarchy | 0.05 | 0.191 |
| flat latent deterministic | 0.10 | 0.225 |
| flat latent flow | 0.00 | 0.150 |
| flat observation deterministic | 0.10 | 0.244 |
| flat observation flow | 0.00 | 0.145 |
| oracle hierarchy (5 episodes) | 0.00 | 0.129 |

The low-budget values are plausible and all seven deployment paths execute.
The oracle sample is only an implementation audit and is not interpreted.

**Projected full runtime:** approximately 14-16 sequential GPU-hours for 18
training points, 54,000 deployable episodes, and 900 oracle episodes. Projected
artifact storage is about 5.5 GiB plus results. The existing 82 GiB free disk
space is sufficient.
