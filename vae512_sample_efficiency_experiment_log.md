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
