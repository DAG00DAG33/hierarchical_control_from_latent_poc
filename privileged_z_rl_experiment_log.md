# Privileged-Z RL Experiment Log

This log tracks the simpler sanity experiments proposed after the learned-latent
RL rerun showed weak future-goal sensitivity.

## 2026-06-24 - PZ-01: Plan and supervised privileged-z smoke

Added the execution plan:

```text
privileged_z_rl_experiment_plan.md
```

Implemented a compact privileged-z trainer:

```text
uv run hcl-poc rl-rerun train-privileged-z
```

The trainer consumes the vector-consistent RL rerun corpus directly and uses:

```text
z_t = observations_state_t
```

where `observations_state` is the 31D Push-T privileged observation state. It
fits standardizers on 500 successful vector streams and trains:

1. a high-level model predicting normalized future `z`;
2. a flat low-level model from current `z`, previous action, and time-to-go;
3. a goal-conditioned low-level model from current `z`, future `z`, previous
   action, and time-to-go.

Command:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-privileged-z \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-trajectories 500 --validation-trajectories 200 \
  --horizon 10 --seed 0 --epochs 40 --batch-size 4096 \
  --hidden-dim 512 --force
```

Artifact:

```text
artifacts/incremental/privileged_z/n500/seed0/privileged_z_k10.pt
artifacts/incremental/privileged_z/n500/seed0/privileged_z_k10_metrics.json
```

Offline validation:

| model | normalized MSE | normalized L2 | normalized MAE |
| --- | ---: | ---: | ---: |
| high future-z predictor | 0.2128 | 1.8553 | 0.2147 |
| flat low level | 0.0994 | 0.4377 | 0.2154 |
| goal-conditioned low level | 0.0843 | 0.4111 | 0.2034 |

Valid-goal action sensitivity for the privileged goal-conditioned low level:

| goal swap | action L2 mean | median | p90 |
| --- | ---: | ---: | ---: |
| `k=2` vs `k=10` | 0.2653 | 0.2455 | 0.5345 |
| `k=5` vs `k=10` | 0.1880 | 0.1634 | 0.3966 |
| `k=20` vs `k=10` | 0.1898 | 0.1333 | 0.4713 |
| `k=2` vs `k=20` | 0.3313 | 0.2984 | 0.6666 |

Interpretation:

This is a strong positive sanity signal for the interface concept when `z` is
privileged physical state. The learned-latent margin-scaled G1 smoke had only
`0.0255` action L2 for `k=2` versus `k=10`; privileged-z reaches `0.2653` on
the same style of valid-goal sensitivity check. This suggests the weak
learned-latent result is at least partly a representation/training-data issue,
not only an unavoidable flaw in future-state conditioning.

Next steps:

1. implement the simplest local PPO/R1-style fine-tuning on top of the
   privileged-z low level;
2. collect or derive a 500-trajectory mixed clean/disturbed corpus;
3. compare privileged-z and learned-latent goal sensitivity on clean-only versus
   mixed data before running any serious RL point.

