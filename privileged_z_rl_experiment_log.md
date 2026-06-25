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

## 2026-06-24 - PZ-04: Fix low-level time-to-go training mismatch

Root cause found:

The privileged-z goal-conditioned low level was trained only on the first action
of each 10-step segment:

```text
[state_t, goal_state_t+10, previous_action_t, remaining=1.0] -> action_t
```

but deployment holds the same goal for 10 simulator steps and calls the low
level with decreasing time-to-go:

```text
remaining = 1.0, 0.9, ..., 0.1
```

This created a train/deploy mismatch for 90% of the low-level calls. The sampler
now expands each segment into all held-goal offsets:

```text
[state_t+i, goal_state_t+10, previous_action_t+i, remaining=(10-i)/10] -> action_t+i
```

for `i = 0..9`.

Retrained clean official run:

```text
artifacts/incremental/privileged_z/clean_official_multioffset/n500/seed0/privileged_z_k10.pt
```

Comparison on the same 100-episode eval seed block (`seed_start=9910000`):

| run | flat success | learned-high hierarchy success | oracle-goal hierarchy success |
| --- | ---: | ---: | ---: |
| old clean official | 0-2% | 0-3% | not measured |
| multi-offset fix | 0% | 8% | 40% |

Offline goal-low validation MAE improved from `0.2034` to `0.1002` because the
validation distribution now matches deployed low-level calls. The oracle-goal
diagnostic shows that privileged future states can work when they are reachable
teacher states. The remaining bottleneck is mostly the learned high-level
future-state predictor/target semantics, not the privileged low-level interface
itself.

## 2026-06-24 - PZ-05: Clean official n=1800 multi-offset run

Ran the same fixed privileged-z training setup with 1800 successful clean
official expert streams:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-privileged-z \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-trajectories 1800 --validation-trajectories 200 \
  --horizon 10 --seed 0 --epochs 40 --batch-size 4096 \
  --hidden-dim 512 --run-tag clean_official_multioffset --force
```

Artifact:

```text
artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10.pt
```

Offline and closed-loop comparison:

| train streams | high MSE | flat MAE | goal MAE | flat success | hierarchy success | oracle-goal success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.2127 | 0.2215 | 0.1002 | 0% | 8% | 40% |
| 1800 | 0.1273 | 0.1506 | 0.0628 | 17% | 53% | 72% |

All closed-loop numbers use the same 100-episode block:

```text
seed_start=9910000
num_envs=64
```

Interpretation:

The low 500-stream privileged-z result was not representative. Increasing the
successful clean expert data to 1800 streams substantially improves both offline
prediction and closed-loop deployment. The learned-high hierarchy now exceeds
the previously reported learned-latent 30% success reference on this eval block.

## 2026-06-24 - PZ-06: Generative privileged-z flow runs

Added a flow-matching privileged-z model family:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-privileged-z \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-trajectories {500,1800} --validation-trajectories 200 \
  --horizon 10 --seed 0 --epochs 40 --batch-size 4096 \
  --hidden-dim 512 --model-family flow --flow-steps 24 \
  --run-tag clean_official_generative --force
```

This trains flow models for:

1. high-level future privileged state;
2. flat one-step action baseline;
3. held-goal low-level action, with the same multi-offset time-to-go sampling
   used by the fixed deterministic privileged-z run.

Artifacts:

```text
artifacts/incremental/privileged_z/clean_official_generative/n500/seed0/privileged_z_k10.pt
artifacts/incremental/privileged_z/clean_official_generative/n1800/seed0/privileged_z_k10.pt
```

All closed-loop numbers use:

```text
seed_start=9910000
num_envs=64
episodes=100
```

| model | train streams | high MSE | flat MAE | goal MAE | flat success | hierarchy success | oracle-goal success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 500 | 0.2127 | 0.2215 | 0.1002 | 0% | 8% | 40% |
| flow | 500 | 0.2846 | 0.2649 | 0.1336 | 0% | 7% | 16% |
| MLP | 1800 | 0.1273 | 0.1506 | 0.0628 | 17% | 53% | 72% |
| flow | 1800 | 0.1952 | 0.1997 | 0.0869 | 2% | 24% | 55% |

Interpretation:

The generative flow variant is valid mechanically but is not better under the
current zero-noise deterministic evaluation protocol. At n=1800, oracle-goal
success remains fairly strong at 55%, but it is lower than the deterministic
MLP low-level's 72%. The learned-high generative hierarchy reaches 24%, well
below the deterministic 53%, so both flow high-level prediction and flow
low-level action sampling need tuning before this is competitive.

## 2026-06-24 - PZ-07: Experiment B/C multi-offset and generative runs

Ran the same fixed privileged-z comparison on the Experiment B and C datasets.
Both merged datasets are too small for an 1800 successful-stream run:

| dataset | successful streams | total streams |
| --- | ---: | ---: |
| `data/rl_rerun/privileged_z_clean_disturbed_balanced.h5` | 665 | 1024 |
| `data/rl_rerun/privileged_z_five_expert_balanced.h5` | 898 | 1280 |

Therefore the comparison is `n=500` only.

Experiment B split:

```text
train: 250 clean official + 250 disturbed official
validation: 20 clean official + 20 disturbed official
```

Experiment C split:

```text
train: 100 successful streams per expert, 5 experts
validation: 40 successful streams per expert, 5 experts
```

Artifacts:

```text
artifacts/incremental/privileged_z/clean_disturbed_multioffset/n500/seed0/privileged_z_k10.pt
artifacts/incremental/privileged_z/clean_disturbed_generative/n500/seed0/privileged_z_k10.pt
artifacts/incremental/privileged_z/five_expert_multioffset/n500/seed0/privileged_z_k10.pt
artifacts/incremental/privileged_z/five_expert_generative/n500/seed0/privileged_z_k10.pt
```

All closed-loop numbers use:

```text
seed_start=9910000
num_envs=64
episodes=100
```

| dataset | model | high MSE | flat MAE | goal MAE | flat success | hierarchy success | oracle-goal success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B clean/disturbed | MLP | 0.2289 | 0.2477 | 0.1232 | 0% | 9% | 36% |
| B clean/disturbed | flow | 0.2587 | 0.2938 | 0.1562 | 1% | 2% | 19% |
| C five expert | MLP | 0.2932 | 0.2864 | 0.1572 | 1% | 1% | 14% |
| C five expert | flow | 0.3620 | 0.3182 | 0.1781 | 0% | 1% | 7% |

Interpretation:

The B and C 500-stream datasets do not improve over the clean official
500-stream Experiment A result after the multi-offset fix. The five-expert
dataset is especially weak in closed loop, likely because the additional
experts are lower quality and mutually inconsistent. Flow remains worse than
MLP under zero-noise deterministic sampling on both datasets.

## 2026-06-24 - PZ-08: Residual RL on privileged-z hierarchies

Added a privileged-z residual-RL trainer:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-privileged-z-residual \
  --checkpoint <privileged_z_k10.pt> \
  --init-dataset <fresh_heldout_vector_dataset.h5> \
  --run-tag <tag> \
  --steps 32768 \
  --alpha 0.1 \
  --terminal-weight 1.0 \
  --residual-penalty-weight 0.01 \
  --learning-rate 1e-4 \
  --num-minibatches 8 \
  --force
```

The residual policy is conditioned on:

```text
normalized current privileged state,
normalized target privileged state,
normalized previous action,
normalized time-to-go
```

For residual training, fresh held-out trajectories were rerecorded and used to
initialize local rollouts. The target goal is the real recorded
`observations_state[t + 10]`, not a learned high-level prediction.

Fresh initialization datasets:

| dataset | path | success streams | total streams | expert success |
| --- | --- | ---: | ---: | ---: |
| A clean | `data/rl_rerun/privileged_z_residual_init_A_clean_n512_b4_seed9900000.h5` | 1527 | 2048 | 74.6% |
| B clean/disturbed | `data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n512_b4.h5` | 1395 | 2048 | 68.1% |
| C five expert | `data/rl_rerun/privileged_z_residual_init_C_five_expert_n512_b5.h5` | 1747 | 2560 | 68.2% |

Residual artifacts:

```text
artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n500_residual_r1/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n500_residual_r1/seed0/latest.pt
```

All comparison numbers below use the same closed-loop evaluation block:

```text
seed_start=9940000
num_envs=64
episodes=100
```

| experiment | variant | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| A clean n=1800 | base hierarchy | 47% | 41.62 | 0.0000 |
| A clean n=1800 | residual hierarchy | 48% | 41.78 | 0.0017 |
| A clean n=1800 | base oracle-goal hierarchy | 69% | 49.35 | 0.0000 |
| A clean n=1800 | residual oracle-goal hierarchy | 65% | 47.26 | 0.0018 |
| B clean/disturbed n=500 | base hierarchy | 4% | 21.07 | 0.0000 |
| B clean/disturbed n=500 | residual hierarchy | 6% | 21.14 | 0.0028 |
| B clean/disturbed n=500 | base oracle-goal hierarchy | 36% | 35.74 | 0.0000 |
| B clean/disturbed n=500 | residual oracle-goal hierarchy | 40% | 35.85 | 0.0028 |
| C five expert n=500 | base hierarchy | 3% | 19.58 | 0.0000 |
| C five expert n=500 | residual hierarchy | 1% | 18.34 | 0.0037 |
| C five expert n=500 | base oracle-goal hierarchy | 15% | 27.45 | 0.0000 |
| C five expert n=500 | residual oracle-goal hierarchy | 15% | 28.82 | 0.0037 |

Interpretation:

The residual RL runs execute end-to-end, but with this reward and alpha they
learn very small corrections. A and B move by only 1-4 percentage points on
100-episode evals; C does not improve. The low residual norms indicate that
the current residual setup is mostly a no-op around the supervised low-level
policy, not a meaningful corrective controller. The oracle-goal columns remain
the most useful diagnostic: B still has recoverable low-level capacity with
true goals, while C remains weak even with true privileged goals.

## 2026-06-24 - PZ-09: 4096-env residual RL alpha sweep

Reran residual RL with 4096 vectorized envs per PPO rollout batch and swept the
residual action scale:

```text
alpha in {0.25, 0.5, 1.0}
num_envs=4096
rollout_steps=10
samples_per_update=40960
total_steps=286720
ppo_updates=7
num_minibatches=8
minibatch_size=5120
```

Fresh 4096-env initialization datasets:

| dataset | path | success streams | total streams | expert success |
| --- | --- | ---: | ---: | ---: |
| A clean | `data/rl_rerun/privileged_z_residual_init_A_clean_n4096_b1_seed9950000.h5` | 3110 | 4096 | 75.9% |
| B clean/disturbed | `data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5` | 5334 | 8192 | 65.1% |
| C five expert | `data/rl_rerun/privileged_z_residual_init_C_five_expert_n4096_b5.h5` | 14232 | 20480 | 69.5% |

Residual artifacts:

```text
artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha025/seed0/latest.pt
artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha05/seed0/latest.pt
artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha10/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n500_residual_r1_n4096_alpha025/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n500_residual_r1_n4096_alpha05/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n500_residual_r1_n4096_alpha10/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n500_residual_r1_n4096_alpha025/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n500_residual_r1_n4096_alpha05/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n500_residual_r1_n4096_alpha10/seed0/latest.pt
```

Training end-of-run diagnostics:

| experiment | alpha | terminal distance | residual norm | reward | success seen | value loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.25 | 0.4199 | 0.0401 | 0.1034 | 0.0186 | 39.12 |
| A | 0.50 | 0.6712 | 0.0809 | 0.0531 | 0.0145 | 172.03 |
| A | 1.00 | 0.9211 | 0.1616 | 0.0030 | 0.0060 | 146.54 |
| B | 0.25 | 0.8512 | 0.0401 | 0.0610 | 0.0290 | 71.48 |
| B | 0.50 | 0.8748 | 0.0809 | 0.0563 | 0.0252 | 68.47 |
| B | 1.00 | 1.1609 | 0.1618 | -0.0010 | 0.0135 | 183.13 |
| C | 0.25 | 2.3628 | 0.0406 | -0.3408 | 0.0416 | 403.49 |
| C | 0.50 | 1.9779 | 0.0807 | -0.2638 | 0.0384 | 344.55 |
| C | 1.00 | 1.9769 | 0.1631 | -0.2637 | 0.0265 | 345.40 |

Closed-loop evaluation uses:

```text
seed_start=9940000
num_envs=64
episodes=100
```

| experiment | alpha | hierarchy success | oracle-goal success | hierarchy return | oracle return | hierarchy residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A clean n=1800 | base | 47% | 69% | 41.62 | 49.35 | 0.0000 |
| A clean n=1800 | 0.25 | 58% | 65% | 44.02 | 47.53 | 0.0053 |
| A clean n=1800 | 0.50 | 46% | 64% | 40.44 | 46.94 | 0.0133 |
| A clean n=1800 | 1.00 | 19% | 52% | 32.04 | 39.69 | 0.0298 |
| B clean/disturbed n=500 | base | 4% | 36% | 21.07 | 35.74 | 0.0000 |
| B clean/disturbed n=500 | 0.25 | 6% | 35% | 22.35 | 34.70 | 0.0053 |
| B clean/disturbed n=500 | 0.50 | 4% | 41% | 21.12 | 36.29 | 0.0113 |
| B clean/disturbed n=500 | 1.00 | 1% | 32% | 19.93 | 35.43 | 0.0255 |
| C five expert n=500 | base | 3% | 15% | 19.58 | 27.45 | 0.0000 |
| C five expert n=500 | 0.25 | 1% | 14% | 19.43 | 28.32 | 0.0074 |
| C five expert n=500 | 0.50 | 7% | 13% | 20.43 | 26.24 | 0.0131 |
| C five expert n=500 | 1.00 | 2% | 20% | 19.90 | 29.74 | 0.0281 |

Interpretation:

Using 4096 envs made the residual policies meaningfully more active than the
512-env runs. The learned residual's normalized training-time residual norm
scales roughly with alpha: about `0.04`, `0.08`, and `0.16` for alpha `0.25`,
`0.5`, and `1.0`. In closed-loop evaluation, however, aggressive residuals are
usually harmful. The best learned-high result is A with alpha `0.25`, improving
from 47% to 58%. B remains high-level limited: oracle-goal success can reach
41%, but learned-high hierarchy stays near 4-6%. C remains inconsistent; alpha
`0.5` helps learned-high hierarchy slightly, while alpha `1.0` helps oracle-goal
success to 20% but hurts learned-high deployment.

## 2026-06-24 - PZ-10: Residual RL with predicted high-level goals

Added residual-RL goal-source modes:

```text
--residual-goal-source oracle
--residual-goal-source predicted
--residual-goal-source oracle_to_predicted
```

`oracle` is the previous setup: local residual rollouts use the recorded
`observations_state[t + 10]` as the target. `predicted` uses the privileged-z
high-level model prediction from `[z_t, previous_action_t]`. `oracle_to_predicted`
uses the recorded target for the first third of training, linearly blends oracle
and predicted targets for the middle third, and uses only predicted targets for
the final third.

All runs below use the 4096-env residual setup:

```text
alpha=0.25
num_envs=4096
rollout_steps=10
samples_per_update=40960
total_steps=286720
ppo_updates=7
```

New residual artifacts:

```text
artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha025_predicted/seed0/latest.pt
artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha025_curriculum/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n500_residual_r1_n4096_alpha025_predicted/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n500_residual_r1_n4096_alpha025_curriculum/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n500_residual_r1_n4096_alpha025_predicted/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n500_residual_r1_n4096_alpha025_curriculum/seed0/latest.pt
```

Training end-of-run diagnostics:

| experiment | goal source | terminal distance | residual norm | final predicted weight | reward | success seen |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | predicted | 0.2654 | 0.0401 | 1.00 | 0.1099 | 0.0185 |
| A | oracle_to_predicted | 0.2993 | 0.0406 | 1.00 | 0.1031 | 0.0186 |
| B | predicted | 0.3214 | 0.0402 | 1.00 | 0.0786 | 0.0189 |
| B | oracle_to_predicted | 0.2962 | 0.0400 | 1.00 | 0.0836 | 0.0191 |
| C | predicted | 1.7384 | 0.0404 | 1.00 | -0.2380 | 0.0222 |
| C | oracle_to_predicted | 1.2716 | 0.0403 | 1.00 | -0.1446 | 0.0234 |

Closed-loop evaluation uses:

```text
seed_start=9940000
num_envs=64
episodes=100
```

| experiment | residual train goal | hierarchy success | oracle-goal success | hierarchy return | oracle return | hierarchy residual norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A clean n=1800 | base | 47% | 69% | 41.62 | 49.35 | 0.0000 |
| A clean n=1800 | oracle | 58% | 65% | 44.02 | 47.53 | 0.0053 |
| A clean n=1800 | predicted | 51% | 65% | 42.54 | 47.85 | 0.0050 |
| A clean n=1800 | oracle_to_predicted | 53% | 65% | 43.28 | 47.76 | 0.0056 |
| B clean/disturbed n=500 | base | 4% | 36% | 21.07 | 35.74 | 0.0000 |
| B clean/disturbed n=500 | oracle | 6% | 35% | 22.35 | 34.70 | 0.0053 |
| B clean/disturbed n=500 | predicted | 7% | 35% | 21.52 | 33.90 | 0.0058 |
| B clean/disturbed n=500 | oracle_to_predicted | 5% | 31% | 20.78 | 35.59 | 0.0052 |
| C five expert n=500 | base | 3% | 15% | 19.58 | 27.45 | 0.0000 |
| C five expert n=500 | oracle | 1% | 14% | 19.43 | 28.32 | 0.0074 |
| C five expert n=500 | predicted | 3% | 19% | 20.11 | 28.94 | 0.0083 |
| C five expert n=500 | oracle_to_predicted | 3% | 15% | 18.04 | 28.19 | 0.0066 |

Interpretation:

Training on predicted high-level goals makes the local training objective much
easier for A and B, as shown by the lower terminal distances, but that does not
translate into a large closed-loop gain. The best A result remains the
oracle-goal-trained residual at 58%. Predicted-goal training is slightly better
for B learned-high success, 7% vs 6%, but both remain very low. For C,
predicted-goal training improves oracle-goal success to 19%, but learned-high
success stays at the base 3%. The curriculum did not beat direct predicted-goal
training in these short 7-update runs.

## 2026-06-24 - PZ-11: A clean n=500 4096-env residual alpha sweep

Reran the A clean alpha sweep from PZ-09 with the 500-stream clean official
privileged-z checkpoint:

```text
artifacts/incremental/privileged_z/clean_official_multioffset/n500/seed0/privileged_z_k10.pt
```

Training setup:

```text
goal source=oracle
alpha in {0.25, 0.5, 1.0}
num_envs=4096
rollout_steps=10
samples_per_update=40960
total_steps=286720
ppo_updates=7
```

Residual artifacts:

```text
artifacts/incremental/privileged_z_residual/A_clean_n500_residual_r1_n4096_alpha025/seed0/latest.pt
artifacts/incremental/privileged_z_residual/A_clean_n500_residual_r1_n4096_alpha05/seed0/latest.pt
artifacts/incremental/privileged_z_residual/A_clean_n500_residual_r1_n4096_alpha10/seed0/latest.pt
```

Training end-of-run diagnostics:

| alpha | terminal distance | residual norm | reward | success seen |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.3802 | 0.0403 | 0.1117 | 0.0124 |
| 0.50 | 0.4955 | 0.0802 | 0.0886 | 0.0100 |
| 1.00 | 0.5447 | 0.1615 | 0.0787 | 0.0053 |

Closed-loop evaluation uses:

```text
seed_start=9940000
num_envs=64
episodes=100
```

| experiment | alpha | hierarchy success | oracle-goal success | hierarchy return | oracle return | hierarchy residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A clean n=500 | base | 6% | 40% | 22.34 | 38.50 | 0.0000 |
| A clean n=500 | 0.25 | 9% | 39% | 22.92 | 37.89 | 0.0050 |
| A clean n=500 | 0.50 | 6% | 37% | 23.64 | 37.25 | 0.0095 |
| A clean n=500 | 1.00 | 4% | 20% | 22.85 | 30.37 | 0.0309 |

Interpretation:

The n=500 base remains much weaker than n=1800 on the same eval block. Residual
RL with alpha `0.25` gives only a small learned-high improvement, 6% to 9%, and
does not improve oracle-goal success. Larger residual scales again hurt,
especially alpha `1.0`, which drops oracle-goal success from 40% to 20%.

## 2026-06-24 - PZ-12: full 500 vs 1800 residual alpha picture

Completed the missing n=1800 pieces for experiments B and C so the residual
alpha sweep can be compared at both 500 and 1800 samples.

New B/C n=1800 training data:

| dataset | path | successful trajectories | total trajectories | success rate |
| --- | --- | ---: | ---: | ---: |
| B clean | `data/rl_rerun/privileged_z_train_B_clean_n4096_b1_seed9961000.h5` | 3117 | 4096 | 76.1% |
| B disturbed | `data/rl_rerun/privileged_z_train_B_disturbed_n4096_b1_seed9962000.h5` | 2243 | 4096 | 54.8% |
| B merged | `data/rl_rerun/privileged_z_train_B_clean_disturbed_n4096_b2_seed996.h5` | 5360 | 8192 | 65.4% |
| C five-expert merged | `data/rl_rerun/privileged_z_train_C_five_expert_n4096_b5_seed996.h5` | 14241 | 20480 | 69.5% |

New B/C n=1800 base checkpoints:

```text
artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt
artifacts/incremental/privileged_z/five_expert_multioffset/n1800/seed0/privileged_z_k10.pt
```

New B/C n=1800 residual checkpoints:

```text
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n1800_residual_r1_n4096_alpha025/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n1800_residual_r1_n4096_alpha05/seed0/latest.pt
artifacts/incremental/privileged_z_residual/B_clean_disturbed_n1800_residual_r1_n4096_alpha10/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n1800_residual_r1_n4096_alpha025/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n1800_residual_r1_n4096_alpha05/seed0/latest.pt
artifacts/incremental/privileged_z_residual/C_five_expert_n1800_residual_r1_n4096_alpha10/seed0/latest.pt
```

Residual training setup:

```text
goal source=oracle
alpha in {0.25, 0.5, 1.0}
num_envs=4096
rollout_steps=10
samples_per_update=40960
total_steps=286720
ppo_updates=7
```

Closed-loop evaluation uses:

```text
seed_start=9940000
num_envs=64
episodes=100
```

### n=500 samples

| experiment | alpha | hierarchy success | oracle-goal success | hierarchy return | oracle return | hierarchy residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A clean n=500 | base | 6% | 40% | 22.34 | 38.50 | 0.0000 |
| A clean n=500 | 0.25 | 9% | 39% | 22.92 | 37.89 | 0.0050 |
| A clean n=500 | 0.50 | 6% | 37% | 23.64 | 37.25 | 0.0095 |
| A clean n=500 | 1.00 | 4% | 20% | 22.85 | 30.37 | 0.0309 |
| B clean/disturbed n=500 | base | 4% | 36% | 21.07 | 35.74 | 0.0000 |
| B clean/disturbed n=500 | 0.25 | 6% | 35% | 22.35 | 34.70 | 0.0053 |
| B clean/disturbed n=500 | 0.50 | 4% | 41% | 21.12 | 36.29 | 0.0113 |
| B clean/disturbed n=500 | 1.00 | 1% | 32% | 19.93 | 35.43 | 0.0255 |
| C five-expert n=500 | base | 3% | 15% | 19.58 | 27.45 | 0.0000 |
| C five-expert n=500 | 0.25 | 1% | 14% | 19.43 | 28.32 | 0.0074 |
| C five-expert n=500 | 0.50 | 7% | 13% | 20.43 | 26.24 | 0.0131 |
| C five-expert n=500 | 1.00 | 2% | 20% | 19.90 | 29.74 | 0.0281 |

### n=1800 samples

| experiment | alpha | hierarchy success | oracle-goal success | hierarchy return | oracle return | hierarchy residual norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A clean n=1800 | base | 47% | 69% | 41.62 | 49.35 | 0.0000 |
| A clean n=1800 | 0.25 | 58% | 65% | 44.02 | 47.53 | 0.0053 |
| A clean n=1800 | 0.50 | 46% | 64% | 40.44 | 46.94 | 0.0133 |
| A clean n=1800 | 1.00 | 19% | 52% | 32.04 | 39.69 | 0.0298 |
| B clean/disturbed n=1800 | base | 45% | 66% | 38.33 | 46.61 | 0.0000 |
| B clean/disturbed n=1800 | 0.25 | 43% | 63% | 39.05 | 45.14 | 0.0053 |
| B clean/disturbed n=1800 | 0.50 | 41% | 67% | 37.69 | 44.55 | 0.0122 |
| B clean/disturbed n=1800 | 1.00 | 35% | 65% | 35.24 | 45.74 | 0.0260 |
| C five-expert n=1800 | base | 25% | 46% | 29.32 | 40.68 | 0.0000 |
| C five-expert n=1800 | 0.25 | 28% | 42% | 29.75 | 39.13 | 0.0061 |
| C five-expert n=1800 | 0.50 | 28% | 51% | 29.97 | 41.13 | 0.0135 |
| C five-expert n=1800 | 1.00 | 24% | 44% | 28.44 | 40.17 | 0.0214 |

Interpretation:

The earlier B/C collapse was mostly a sample-count problem, not a residual-RL
environment-count problem. Moving B from 500 to 1800 raises learned-high success
from 4% to 45%, and moving C from 500 to 1800 raises it from 3% to 25%.
Residual RL is still not reliably improving the learned-high policy. For A,
alpha `0.25` is useful, 47% to 58%. For B, the n=1800 base is best in learned-high
mode, and residuals mostly reduce success. For C, alpha `0.25` and `0.50` give a
small learned-high gain, 25% to 28%, with alpha `0.50` also improving oracle-goal
success from 46% to 51%.

## 2026-06-25 - PZ-13: Matched direct paired checkpoint validation

The direct paired privileged-z runs already had local paired evals and two
closed-loop files for the hard-start checkpoint, but they were missing a
same-seed frozen baseline in the log. I evaluated the n=1800 clean
multi-offset base checkpoint on the same 200-episode seed window:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_hierarchy_seed9900000_200eps.json \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode oracle_hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_oracle_seed9900000_200eps.json \
  --force
```

Matched closed-loop comparison:

| policy | mode | success | return | action delta/residual norm |
| --- | --- | ---: | ---: | ---: |
| base n=1800 clean | learned-high hierarchy | 0.560 | 44.74 | 0.0000 |
| direct paired hard-start | learned-high hierarchy | 0.515 | 40.47 | 0.0102 |
| base n=1800 clean | oracle-goal hierarchy | 0.720 | 48.69 | 0.0000 |
| direct paired hard-start | oracle-goal hierarchy | 0.700 | 47.11 | 0.0143 |

Relevant direct checkpoint and eval artifacts:

- `artifacts/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m/seed0/latest.pt`
- `results/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m/seed0/history.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_hierarchy_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_oracle_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_hierarchy_seed9900000_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_oracle_seed9900000_200eps.json`

Interpretation:

This direct paired privileged-z checkpoint improves selected hard local starts
in the paired local metric, but it does not transfer to closed-loop success. On
the matched seed window it is worse than the frozen base in both learned-high
and oracle-goal modes. Together with PZ-12, this narrows the privileged-state
upper-bound story: residual alpha `0.25` on A clean n=1800 remains the only
clear learned-high privileged RL gain so far; more direct paired training on
hard local starts is not the next useful direction unless the local objective is
made more task-aligned.

## 2026-06-25 - PZ-14: Matched residual alpha025 validation

PZ-12 identified A clean n=1800 residual alpha `0.25` as the only clear
learned-high privileged RL gain, but that table used the older 100-episode
window starting at `9940000`. To compare residual and direct paired on the same
deployment window, I evaluated the residual checkpoint on
`9900000..9900199`, matching PZ-13.

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha025/seed0/latest.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_hierarchy_seed9900000_200eps.json \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_residual/A_clean_n1800_residual_r1_n4096_alpha025/seed0/latest.pt \
  --mode oracle_hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_oracle_seed9900000_200eps.json \
  --force
```

Matched 200-episode comparison:

| policy | mode | success | return | action delta/residual norm |
| --- | --- | ---: | ---: | ---: |
| base n=1800 clean | learned-high hierarchy | 0.560 | 44.74 | 0.0000 |
| residual alpha025 | learned-high hierarchy | 0.555 | 44.24 | 0.0053 |
| direct paired hard-start | learned-high hierarchy | 0.515 | 40.47 | 0.0102 |
| base n=1800 clean | oracle-goal hierarchy | 0.720 | 48.69 | 0.0000 |
| residual alpha025 | oracle-goal hierarchy | 0.725 | 49.01 | 0.0053 |
| direct paired hard-start | oracle-goal hierarchy | 0.700 | 47.11 | 0.0143 |

New artifacts:

- `results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_hierarchy_seed9900000_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_oracle_seed9900000_200eps.json`

Interpretation:

The residual alpha `0.25` checkpoint no longer shows a learned-high success
gain on this matched 200-episode window: `0.560` frozen versus `0.555` residual.
It is slightly positive in oracle-goal mode, `0.720` to `0.725`, and still much
less harmful than direct paired hard-start tuning. The important update is that
the previous 100-episode learned-high gain was not stable enough to treat as a
privileged RL pass. The privileged-state sanity result is now: local RL can make
small oracle-goal/local changes, but we still do not have a robust closed-loop
improvement over the frozen privileged hierarchy.
