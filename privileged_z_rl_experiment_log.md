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

## 2026-06-24 - PZ-02: Add balanced five-expert experiment

The current task now has three privileged-z data variants:

1. clean 500 successful trajectories;
2. mixed 250 clean + 250 disturbed/recovery successful trajectories;
3. balanced five-expert data with 100 successful trajectories per expert.

Updated:

```text
privileged_z_rl_experiment_plan.md
```

The trainer now supports isolated run tags and balanced expert selection:

```text
uv run hcl-poc rl-rerun train-privileged-z \
  --run-tag five_expert \
  --selection-mode balanced_experts \
  --train-per-expert 100 \
  --validation-per-expert 40
```

This avoids overwriting clean-only artifacts and forces the five-expert
comparison to use equal data from each competent expert.

Smoke check:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-privileged-z \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-trajectories 20 --validation-trajectories 10 \
  --horizon 10 --seed 99 --epochs 1 --batch-size 1024 \
  --hidden-dim 64 --run-tag cli_smoke --force
```

Result:

```text
artifacts/incremental/privileged_z/cli_smoke/n20/seed99/privileged_z_k10.pt
```

Expert availability check:

| checkpoint | logged best success | usable as expert? |
| --- | ---: | --- |
| `artifacts/rl_pusht_official/ppo_best.pt` | 0.925 | yes |
| `artifacts/rl/ppo_best.pt` | 0.01 | no |
| `artifacts/rl_baseline/ppo_best.pt` | 0.01 | no |

Only one competent PPO expert is currently available in the repo. Experiment C
therefore needs four additional competent PPO experts to be trained or imported
before the balanced 500-trajectory dataset can be collected honestly.

## 2026-06-24 - PZ-03: Execute three privileged-z data variants

Updated the execution target to the three simplified privileged-z experiments:

1. clean 500-trajectory privileged-z interface;
2. balanced clean/disturbed 500-trajectory interface;
3. balanced five-expert 500-trajectory interface.

### PPO experts

Trained four additional PPO experts with the same official Push-T privileged
state/action setup, changing only the seed and output directory:

```text
uv run hcl-poc rl train --config configs/pusht.yaml --seed {1,2,3,4} \
  --rl-dir artifacts/rl_multiexpert/expert_seed{1,2,3,4} --no-resume
```

Expert quality:

| expert | checkpoint | deterministic eval success |
| --- | --- | ---: |
| official | `artifacts/rl_pusht_official/ppo_best.pt` | 0.91 on the standard 100-episode eval |
| seed 1 | `artifacts/rl_multiexpert/expert_seed1/ppo_best.pt` | 0.723 |
| seed 2 | `artifacts/rl_multiexpert/expert_seed2/ppo_best.pt` | 0.863 |
| seed 3 | `artifacts/rl_multiexpert/expert_seed3/ppo_best.pt` | 0.652 |
| seed 4 | `artifacts/rl_multiexpert/expert_seed4/ppo_best.pt` | 0.773 |

Seeds 1-4 all reached the PPO training target (`success_recent >= 0.90`) before
early stopping, but deterministic held-out quality varies substantially. Seed 3
is the weakest expert and should be treated as a source of style diversity, not
as a near-optimal teacher.

### Datasets

Added disturbed vector collection support:

```text
uv run hcl-poc rl-rerun collect-vector-data --disturbed --no-store-dino
```

Added merge utility:

```text
scripts/merge_privileged_z_vector_datasets.py
```

Created:

| dataset | source | successful streams used |
| --- | --- | --- |
| `data/rl_rerun/privileged_z_clean_disturbed_balanced.h5` | official clean + official disturbed | 250 clean train, 250 disturbed train, 20 validation per bucket |
| `data/rl_rerun/privileged_z_five_expert_balanced.h5` | official + seeds 1-4 | 100 train and 40 validation per expert |

### Supervised interface results

All three runs used:

```text
z = normalized 31D privileged observation state
k = 10 steps
hidden_dim = 512
epochs = 40
seed = 0
```

| run | high MSE | flat MAE | goal MAE | goal-flat MAE | k2 vs k10 sensitivity | k2 vs k20 sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean official | 0.2128 | 0.2154 | 0.2034 | -0.0120 | 0.2653 | 0.3313 |
| clean/disturbed | 0.2286 | 0.2416 | 0.2328 | -0.0088 | 0.2870 | 0.3601 |
| five expert | 0.2880 | 0.2690 | 0.2504 | -0.0185 | 0.2224 | 0.2427 |

Interpretation:

- Clean/disturbed data gives a small sensitivity improvement over clean-only.
- Five-expert data increases diversity but also increases prediction/action
  error and lowers valid-goal sensitivity. This is probably because the extra
  experts are weaker and less mutually consistent, not because multimodality is
  automatically helpful.
- The goal-conditioned low level has lower offline action MAE than the flat
  low level in all three variants, so the goal input is useful offline.

### Closed-loop smoke

Added direct evaluator:

```text
uv run hcl-poc rl-rerun eval-privileged-z --mode flat
uv run hcl-poc rl-rerun eval-privileged-z --mode hierarchy
```

100-episode closed-loop results with `seed_start=9910000`:

| run | flat success | hierarchy success | flat return | hierarchy return |
| --- | ---: | ---: | ---: | ---: |
| clean official | 0.02 | 0.00 | 17.22 | 17.33 |
| clean/disturbed | 0.00 | 0.00 | 15.82 | 17.33 |
| five expert | 0.00 | 0.00 | 13.57 | 15.57 |

This is the main negative result: offline privileged-z action prediction and
goal sensitivity do not currently translate into successful deployment. The
official PPO expert still succeeds on its standard eval path, so the failure is
not a general simulator issue.

Current hypothesis:

- The supervised low-level policy is still too brittle under closed-loop
  distribution shift.
- The high-level future-state predictor may generate physically plausible
  states that are not good control targets for the low level.
- The five-expert corpus may be noisier than helpful because some experts are
  much weaker than the official policy.

Artifacts:

```text
privileged_z_three_experiment_summary.json
artifacts/incremental/privileged_z/clean_official/n500/seed0/
artifacts/incremental/privileged_z/clean_disturbed/n500/seed0/
artifacts/incremental/privileged_z/five_expert/n500/seed0/
```
