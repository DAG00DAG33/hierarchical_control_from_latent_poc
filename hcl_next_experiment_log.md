# HCL next experiments log

## 2026-06-25 - Phase 0 manifests and VAE512 goal-use smoke

### Hypothesis

Before launching more RL, the project needs fixed `N=500` and `N=1800`
manifests/local reset banks and a fresh goal-identifiability check on the
existing VAE512 hierarchy.

### Commands

```bash
uv run hcl-poc doctor
uv run hcl-poc incremental vae-scaling-manifests --config configs/pusht_incremental.yaml
uv run python scripts/prepare_hcl_next_phase0.py
uv run python scripts/rl_rerun_valid_goal_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 2048 \
  --batch-size 512 \
  --horizons 2,5,10,20 \
  --reference-horizon 10 \
  --output results/hcl_next_phase0/goal_valid_sensitivity_n500_seed0_h2_5_10_20.json
uv run python scripts/rl_rerun_valid_goal_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 256 \
  --batch-size 128 \
  --horizons 2,5,10,20 \
  --reference-horizon 10 \
  --output results/hcl_next_phase0/goal_valid_sensitivity_n500_seed0_h2_5_10_20_smoke256.json
uv run python scripts/rl_rerun_condition_block_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 256 \
  --batch-size 128 \
  --horizon 10 \
  --output results/hcl_next_phase0/condition_block_sensitivity_n500_seed0_k10_smoke256.json
```

The 2048-sample valid-goal run was interrupted after several minutes. The
bottleneck was random HDF5 reads from the large 4096-env DINO feature dataset,
not model inference. The 256-sample result below is a smoke/debug gate, not a
final diagnostic.

### Setup

- git commit: `d218d205325d16fcdd37a0d1e2f1dd1cb9abc150`
- git dirty: yes
- machine/GPU: NVIDIA GeForce RTX 4060 Ti, 15.57 GiB total VRAM
- free disk before runs: about 22 GiB
- config: `configs/pusht_incremental.yaml`
- N trajectories: 500 and 1800 manifests created; diagnostics run for N=500
- representation: VAE512 future-state latent
- architecture: existing concat low level
- horizon k: fixed banks for 2, 5, 10, 20; condition-block smoke at k=10
- num_envs: fixed local reset banks use 4096-env validation vector dataset
- goal source: replay/oracle future states from held-out vector dataset
- result level: smoke/debug for diagnostics because samples=256

### Initial results

Environment and manifest validation passed:

| check | result |
| --- | --- |
| `hcl-poc doctor` | PushT-v1 OK, CUDA available |
| VAE scaling manifests | nested fixed-validation manifests valid |
| Phase 0 manifest files | 14 files written under `data/manifests/` |

Created fixed files:

| file group | count | notes |
| --- | ---: | --- |
| `pusht_n{500,1800}_seed0_{train,val,eval}.json` | 6 | eval is a simulator seed bank, not reused validation trajectories |
| `local_reset_bank_n{500,1800}_seed0_k{2,5,10,20}.json` | 8 | each bank references 4096 local reset episodes |

Fresh N=500 VAE512 goal-use smoke:

| metric | value |
| --- | ---: |
| mean latent goal L2, k2 vs k10 | 26.48 |
| mean latent goal L2, k5 vs k10 | 24.45 |
| mean latent goal L2, k20 vs k10 | 23.79 |
| mean action L2, k2 vs k10 | 0.024 |
| mean action L2, k5 vs k10 | 0.019 |
| mean action L2, k20 vs k10 | 0.019 |
| observation-shuffle action L2 | 0.818 |
| goal-shuffle action L2 | 0.049 |
| previous-action-shuffle action L2 | 0.077 |
| remaining-time-shuffle action L2 | 0.000 |
| observation/goal shuffle ratio | 16.73 |

### Plots / artifacts

- split and reset manifests: `data/manifests/`
- preparation script: `scripts/prepare_hcl_next_phase0.py`
- valid-goal smoke JSON:
  `results/hcl_next_phase0/goal_valid_sensitivity_n500_seed0_h2_5_10_20_smoke256.json`
- condition-block smoke JSON:
  `results/hcl_next_phase0/condition_block_sensitivity_n500_seed0_k10_smoke256.json`

### Interpretation

The existing VAE512 concat low level still fails the goal-identifiability gate.
Large changes in valid future-goal latent produce only about `0.02` action L2,
while shuffling the current observation changes actions by about `0.82` L2.
This is consistent with the previous diagnosis: the low level mostly ignores
the future goal.

The fixed manifest/reset-bank prerequisite is now in place for Phase 0, but
expensive RL should still wait for either privileged/TCP sanity runs or a
goal-conditioning architecture/representation that passes the gate.

### Next action

Run the Phase 1 RL sanity path with privileged/TCP state and oracle local goals,
or implement the reusable goal-diagnostics module/FiLM low-level path so that
VAE512 can be gated before any further learned-latent PPO.

## 2026-06-25 - Phase 1 inventory: privileged/TCP sanity base

### Hypothesis

Before launching new learned-latent PPO, reuse existing privileged/TCP artifacts
to determine whether the easier representation already gives a viable RL sanity
base.

### Command

```bash
find artifacts/incremental/privileged_z artifacts/rl_rerun results/incremental results/rl_rerun \
  -path '*privileged_z*' -type f
cat artifacts/incremental/privileged_z/clean_official_multioffset/n500/seed0/privileged_z_k10_metrics.json
cat artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10_metrics.json
cat artifacts/incremental/privileged_z/clean_official_multioffset/n500/seed0/privileged_z_k10_eval_hierarchy_n100.json
cat artifacts/incremental/privileged_z/clean_official_multioffset/n500/seed0/privileged_z_k10_eval_oracle_hierarchy_n100.json
cat artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10_eval_hierarchy_n100.json
cat artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10_eval_oracle_hierarchy_n100.json
```

### Setup

- git commit: `d218d205325d16fcdd37a0d1e2f1dd1cb9abc150`
- config: `configs/pusht_incremental.yaml`
- representation: 31D privileged Push-T observation state
- architecture: existing privileged-z MLP hierarchy
- horizon k: 10
- low-level training: multi-offset held-goal training
- data regimes inspected: `N=500`, `N=1800`
- goal source eval: learned high-level and oracle privileged future state
- result level: existing 100-episode development-bank artifacts, not new final eval

### Results

Clean official multi-offset supervised checkpoints:

| N demos | goal MAE | flat MAE | k2 vs k10 action L2 | k5 vs k10 action L2 | learned success | oracle success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.1002 | 0.2215 | 0.1271 | 0.0846 | 0.06 | 0.40 |
| 1800 | 0.0628 | 0.1506 | 0.1099 | 0.0740 | 0.47 | 0.69 |

Clean/disturbed multi-offset `N=1800` artifacts:

| variant | learned success | oracle success | mean residual norm |
| --- | ---: | ---: | ---: |
| frozen base | 0.45 | 0.66 | 0.0000 |
| residual `alpha=0.25` | 0.43 | 0.63 | 0.0053-0.0055 |

### Plots / artifacts

- `artifacts/incremental/privileged_z/clean_official_multioffset/n500/seed0/privileged_z_k10.pt`
- `artifacts/incremental/privileged_z/clean_official_multioffset/n1800/seed0/privileged_z_k10.pt`
- `artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- `artifacts/incremental/privileged_z_residual/B_clean_disturbed_n1800_residual_r1_n4096_alpha025/seed0/latest.pt`
- eval tables under `artifacts/incremental/privileged_z_eval_tables/pz12_n1800/`

### Interpretation

Privileged/TCP multi-offset training is a much healthier control interface than
the VAE512 concat low level. The `N=1800` privileged base is already strong
enough to serve as the RL sanity base: learned high-level success is about
`0.45-0.47`, and oracle-goal success is about `0.66-0.69`.

The existing residual PPO artifact with `alpha=0.25` does not improve the
`N=1800` clean/disturbed base on the 100-episode development bank. It slightly
reduces both learned and oracle success. This suggests the next RL sanity run
should focus on paired-improvement reward/local metrics, not just another
absolute-distance residual run.

### Next action

Implement or expose paired-improvement local evaluation for privileged/TCP
rollouts, using the fixed `data/manifests/local_reset_bank_*` files. After that,
rerun residual PPO only if the local paired metric shows the reward is aligned
with frozen-base improvement.

## 2026-06-25 - Phase 1 paired local evaluation for privileged/TCP residual

### Hypothesis

The existing privileged/TCP residual PPO run should only be promoted if it
improves local goal reaching over the frozen imitation base from the exact same
reset and replay/oracle goal.

### Command

Implemented:

```text
uv run hcl-poc rl-rerun eval-privileged-z-local-paired
```

Smoke command:

```bash
uv run hcl-poc --config configs/pusht_incremental.yaml \
  rl-rerun eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_residual/B_clean_disturbed_n1800_residual_r1_n4096_alpha025/seed0/latest.pt \
  --manifest results/rl_rerun/local_eval_manifest_n512_val_b1_seed20260623.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_clean_disturbed_n1800_alpha025_n512_smoke.json \
  --force
```

Fixed-bank command:

```bash
uv run hcl-poc --config configs/pusht_incremental.yaml \
  rl-rerun eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_residual/B_clean_disturbed_n1800_residual_r1_n4096_alpha025/seed0/latest.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_clean_disturbed_n1800_alpha025_k10_4096.json \
  --force
```

### Setup

- config: `configs/pusht_incremental.yaml`
- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- residual checkpoint: `B_clean_disturbed_n1800_residual_r1_n4096_alpha025/seed0/latest.pt`
- low-level training: multi-offset held-goal training
- local goal source: replay/oracle future state
- horizon k: 10
- fixed local bank: `data/manifests/local_reset_bank_n1800_seed0_k10.json`
- fixed-bank local episodes: 4096
- success epsilon: terminal normalized-state MSE `< 0.05`

### Results

| eval bank | episodes | mean paired improvement MSE | median paired improvement MSE | fraction improved | base eps success | tuned eps success | action delta L2 mean | residual norm mean | saturation mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 smoke | 512 | -0.5035 | -0.0000415 | 0.4746 | 0.9043 | 0.8867 | 0.0163 | 0.00545 | 0.00436 |
| 4096 fixed | 4096 | 0.0441 | -0.0000089 | 0.4844 | 0.8943 | 0.8923 | 0.0144 | 0.00520 | 0.00095 |

The positive mean on the 4096 bank is caused by extreme outliers:

| metric | base | tuned |
| --- | ---: | ---: |
| terminal MSE median | 0.00187 | 0.00196 |
| terminal MSE p90 | 0.05421 | 0.05479 |
| terminal MSE max | 1322.6 | 1136.7 |

### Plots / artifacts

- evaluator code: `src/hcl_poc/privileged_z.py`
- CLI wiring: `src/hcl_poc/cli.py`
- smoke JSON:
  `results/hcl_next_phase1/privileged_z_local_paired_clean_disturbed_n1800_alpha025_n512_smoke.json`
- fixed-bank JSON:
  `results/hcl_next_phase1/privileged_z_local_paired_clean_disturbed_n1800_alpha025_k10_4096.json`

### Interpretation

The existing absolute-distance residual PPO checkpoint fails the paired local
promotion gate. It improves fewer than half of fixed-bank local rollouts, makes
median terminal distance slightly worse, and slightly lowers success within the
epsilon threshold. Action saturation is low, so this is not primarily a clamp
artifact; the learned residual is just too small or misaligned to produce a
reliable paired improvement.

This supports the plan's recommendation to add an explicit paired-improvement
reward before running more expensive residual PPO. Another absolute-distance
residual run is unlikely to answer a new question.

### Next action

Implement paired-improvement reward for privileged/TCP local residual PPO:
cache or compute the frozen base terminal distance from the same local reset and
optimize `J_base - J_policy` with a small action-deviation penalty. Reuse
`eval-privileged-z-local-paired` as the promotion gate.

## 2026-06-25 - Phase 1 paired-reward PPO smoke

### Hypothesis

A paired terminal reward can be implemented in the privileged/TCP residual PPO
loop by rolling the frozen base policy from the same local reset and goal, then
rewarding `base_terminal_distance - policy_terminal_distance`.

### Command

Implemented new training options:

```text
uv run hcl-poc rl-rerun train-privileged-z-residual \
  --reward-mode progress|paired \
  --dense-progress-weight ...
```

Smoke command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-residual \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n512_b4.h5 \
  --run-tag hcl_next_paired_reward_smoke_n512 \
  --seed 0 \
  --steps 5120 \
  --alpha 0.25 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --residual-penalty-weight 0.01 \
  --learning-rate 1e-4 \
  --num-minibatches 8 \
  --update-epochs 1 \
  --checkpoint-every-updates 1 \
  --force
```

Integration eval:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_smoke_n512/seed0/latest.pt \
  --manifest results/rl_rerun/local_eval_manifest_n512_val_b1_seed20260623.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_reward_smoke_n512_eval.json \
  --force
```

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- init dataset: `privileged_z_residual_init_B_clean_disturbed_n512_b4.h5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- residual penalty weight: 0.01
- num_envs: 512
- rollout steps: 10
- total transitions: 5120, one PPO update
- update epochs: 1
- result level: implementation smoke only, not a scientific run

### Results

Training history after one update:

| metric | value |
| --- | ---: |
| mean base terminal distance | 0.6369 |
| mean policy terminal distance | 0.5672 |
| mean paired improvement | 0.0697 |
| fraction improved | 0.2773 |
| mean residual norm | 0.0401 |
| mean reward | 0.0070 |
| clip fraction | 0.1123 |

The smoke checkpoint evaluates through the paired local gate:

| eval metric | value |
| --- | ---: |
| eval episodes | 512 |
| mean paired improvement MSE | 0.0162 |
| median paired improvement MSE | -0.000064 |
| fraction improved | 0.4219 |
| base epsilon success | 0.9043 |
| tuned epsilon success | 0.8828 |

### Plots / artifacts

- paired-reward checkpoint:
  `artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_smoke_n512/seed0/latest.pt`
- paired-reward history:
  `results/incremental/privileged_z_residual/hcl_next_paired_reward_smoke_n512/seed0/history.json`
- paired local eval:
  `results/hcl_next_phase1/privileged_z_local_paired_reward_smoke_n512_eval.json`

### Interpretation

The paired-reward path is now executable and records the needed local metrics.
The one-update smoke is not a pass: it improves fewer than half of validation
local rollouts and worsens epsilon success. That is acceptable for this smoke;
its purpose was to verify the reward computation, branch-env base rollout, PPO
history schema, checkpoint save, and paired evaluator compatibility.

One implementation detail matters for future commands: this CLI has both global
and `rl-rerun`-level config arguments. For commands that need
`paths.incremental_*`, use:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml ...
```

Passing `--config` only before `rl-rerun` can be overwritten by the subcommand
default.

### Next action

Run a real paired-reward dev point with `4096` envs and at least `1M`
transitions, then evaluate with `eval-privileged-z-local-paired` on
`data/manifests/local_reset_bank_n1800_seed0_k10.json` before doing any
closed-loop task evaluation.

## 2026-06-25 - Phase 1 paired-reward 4096-env dev run

### Hypothesis

A serious `4096`-env PPO run with paired terminal reward and privileged/TCP
state may improve local goal reaching over the frozen multi-offset base, unlike
the previous absolute-distance residual run.

### Command

Training:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-residual \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5 \
  --run-tag hcl_next_paired_reward_n4096_alpha025_1m \
  --seed 0 \
  --steps 1024000 \
  --alpha 0.25 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --residual-penalty-weight 0.01 \
  --learning-rate 1e-4 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --checkpoint-every-updates 5 \
  --force
```

Fixed-bank evaluation:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_n4096_alpha025_1m/seed0/latest.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_reward_n4096_alpha025_1m_k10_4096.json \
  --force
```

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- init dataset: `privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- residual alpha: 0.25
- residual penalty weight: 0.01
- num_envs: 4096
- rollout steps: 10
- PPO batch: 40960 samples/update
- total transitions: 1,024,000
- updates: 25
- result level: dev

### Implementation Note

The first attempt failed before training with a ManiSkill GPU camera-group
buffer error because paired reward created two simultaneous `rgb+state`
4096-env simulators. The privileged/TCP local trainer and local paired evaluator
now use `obs_mode=state`, which is sufficient for 31D privileged-state reward
and avoids camera allocation. Full closed-loop visual evaluators are unchanged.

### Results

Training completed in about 237 seconds. The final update history was:

| metric | value |
| --- | ---: |
| final mean base terminal distance | 0.2101 |
| final mean policy terminal distance | 0.2214 |
| final mean paired improvement | -0.0113 |
| final fraction improved | 0.3413 |
| final mean residual norm | 0.0424 |
| final reward | -0.00113 |

Fixed 4096-bank local paired evaluation:

| metric | value |
| --- | ---: |
| base terminal MSE mean | 0.5051 |
| tuned terminal MSE mean | 0.4406 |
| base terminal MSE median | 0.00187 |
| tuned terminal MSE median | 0.00243 |
| base terminal MSE p90 | 0.05421 |
| tuned terminal MSE p90 | 0.05914 |
| mean paired improvement MSE | 0.0644 |
| median paired improvement MSE | -0.000139 |
| fraction improved | 0.3840 |
| base epsilon success | 0.8943 |
| tuned epsilon success | 0.8835 |
| action delta L2 mean | 0.0278 |
| residual norm mean | 0.0154 |
| action saturation mean | 0.00071 |

### Plots / artifacts

- checkpoint:
  `artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_n4096_alpha025_1m/seed0/latest.pt`
- training history:
  `results/incremental/privileged_z_residual/hcl_next_paired_reward_n4096_alpha025_1m/seed0/history.json`
- fixed-bank eval:
  `results/hcl_next_phase1/privileged_z_local_paired_reward_n4096_alpha025_1m_k10_4096.json`

### Interpretation

This paired-reward `alpha=0.25` dev run fails the local promotion gate. Mean MSE
looks better because a small number of extreme failures improved, but the median
rollout, p90 rollout, fraction improved, and epsilon success all worsened. The
residual is active and action saturation is low, so the failure is not just
clipping; the learned residual is not broadly improving local reachability.

Do not run closed-loop task evaluation for this checkpoint. It does not satisfy
the plan's local gate:

```text
fraction_improved > 0.55
mean_paired_improvement > 0
closed-loop success not worse
```

Only the mean-paired-improvement condition is superficially positive, and that
is outlier-driven.

### Next action

Continue the prescribed residual-alpha sweep before changing representation:
run the same paired-reward setup at a larger residual authority (`alpha=1.0` or
`alpha=0.5`) and gate with the same fixed local paired bank. If larger alpha
also worsens median/fraction-improved, stop residual PPO on privileged/TCP and
debug reward/distance normalization or try direct/partially-unfrozen low-level
training.

## 2026-06-25 - Phase 1 paired-reward alpha sweep

### Hypothesis

The failed `alpha=0.25` paired-reward residual may simply have too little
authority. Increasing residual authority should improve the fraction of local
rollouts improved if the paired reward is aligned and the residual architecture
is the right update mechanism.

### Command

The following command was run for `alpha=0.5` and `alpha=1.0`, changing only
`--alpha` and `--run-tag`:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-residual \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5 \
  --run-tag hcl_next_paired_reward_n4096_alpha05_1m \
  --seed 0 \
  --steps 1024000 \
  --alpha 0.5 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --residual-penalty-weight 0.01 \
  --learning-rate 1e-4 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --checkpoint-every-updates 5 \
  --force
```

Each checkpoint was evaluated with:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint <checkpoint>/latest.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --output results/hcl_next_phase1/<result>.json \
  --force
```

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- init dataset: `privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- residual penalty weight: 0.01
- num_envs: 4096
- rollout steps: 10
- total transitions per run: 1,024,000
- fixed local eval bank: `data/manifests/local_reset_bank_n1800_seed0_k10.json`
- base fixed-bank epsilon success: 0.8943
- base fixed-bank terminal MSE median: 0.00187
- base fixed-bank terminal MSE p90: 0.05421

### Results

| alpha | mean paired improvement | median paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE median | tuned terminal MSE p90 | action delta L2 mean | residual norm mean | saturation mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.0644 | -0.000139 | 0.3840 | 0.8835 | 0.00243 | 0.05914 | 0.0278 | 0.0154 | 0.00071 |
| 0.50 | 0.1069 | -0.000340 | 0.3533 | 0.8816 | 0.00285 | 0.05932 | 0.0405 | 0.0240 | 0.00069 |
| 1.00 | 0.0735 | -0.004005 | 0.1511 | 0.8108 | 0.00876 | 0.10508 | 0.0827 | 0.0640 | 0.00050 |

### Plots / artifacts

- `artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_n4096_alpha025_1m/seed0/latest.pt`
- `artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_n4096_alpha05_1m/seed0/latest.pt`
- `artifacts/incremental/privileged_z_residual/hcl_next_paired_reward_n4096_alpha10_1m/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_reward_n4096_alpha025_1m_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_reward_n4096_alpha05_1m_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_reward_n4096_alpha10_1m_k10_4096.json`

### Interpretation

The paired-reward residual alpha sweep fails. Larger residual authority makes
the policy more active, but the local gate gets worse:

- `fraction_improved` never exceeds 0.384, far below the `>0.55` pass criterion;
- median paired improvement is negative for every alpha;
- epsilon success is worse for every alpha;
- alpha `1.0` damages the distribution badly, with p90 terminal MSE nearly
  doubling versus the frozen base.

The positive mean paired improvements are outlier-driven and are not sufficient
evidence of a useful RL update. Action saturation remains very low, so the
failure is not explained by clamping.

### Decision

Do not run closed-loop task evaluation for these residual checkpoints. The
privileged/TCP residual PPO formulation, even with paired terminal reward,
does not currently pass the local improvement gate.

This is useful evidence: the problem is no longer only VAE goal ignoring. Even
with a healthier privileged/TCP interface, residual PPO is not producing broad
local improvements under this reward/architecture.

### Next action

Stop this residual branch and move to the next Phase 1 variant:

1. direct or partially-unfrozen privileged/TCP low-level PPO with paired reward;
2. add checkpoint-per-update outputs so local gate can select earlier updates
   instead of only `latest.pt`;
3. if direct/partial tuning also fails, debug reward/distance normalization with
   an offline local optimizer or scratch PPO before returning to learned VAE
   latents.

## 2026-06-25 - Phase 1 direct privileged/TCP PPO implementation smoke

### Hypothesis

The residual formulation may be too constrained or poorly conditioned. Before
running another serious 4096-env PPO branch, add a direct BC-initialized
privileged/TCP low-level PPO variant and verify that it can train, checkpoint,
and pass through the paired local evaluator.

This is an implementation smoke only, not scientific evidence for or against
the direct formulation.

### Commands

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-direct \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n512_b4.h5 \
  --run-tag hcl_next_direct_smoke_n512 \
  --seed 0 \
  --steps 5120 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --learning-rate 3e-5 \
  --num-minibatches 4 \
  --update-epochs 1 \
  --checkpoint-every-updates 1 \
  --train-scope final_layer \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct/hcl_next_direct_smoke_n512/seed0/latest.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_direct_smoke_n512_eval.json \
  --force

uv run pytest -q
```

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- init dataset: `privileged_z_residual_init_B_clean_disturbed_n512_b4.h5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- train scope: final low-policy layer plus log std and critic
- num_envs: 512
- rollout steps: 10
- total transitions: 5,120

### Results

Training smoke history:

| metric | value |
| --- | ---: |
| mean paired improvement | -0.00478 |
| fraction improved | 0.2988 |
| mean terminal distance | 0.03679 |
| mean base terminal distance | 0.03201 |
| mean action delta L2 | 0.02925 |
| action saturation rate | 0.00215 |
| clip fraction | 0.00840 |

Fixed-bank paired local eval:

| metric | value |
| --- | ---: |
| base epsilon success | 0.8943 |
| tuned epsilon success | 0.8943 |
| mean paired improvement MSE | 0.0209 |
| median paired improvement MSE | -0.0000067 |
| fraction improved | 0.4731 |
| tuned terminal MSE median | 0.00193 |
| tuned terminal MSE p90 | 0.05454 |
| action delta L2 mean | 0.00828 |
| action saturation frac mean | 0.00103 |

Verification:

- `uv run python -m compileall -q src/hcl_poc/privileged_z.py src/hcl_poc/cli.py`
- `uv run pytest -q`: 22 passed

### Artifacts

- `artifacts/incremental/privileged_z_direct/hcl_next_direct_smoke_n512/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct/hcl_next_direct_smoke_n512/seed0/checkpoints/step_000005120.pt`
- `results/incremental/privileged_z_direct/hcl_next_direct_smoke_n512/seed0/history.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_smoke_n512_eval.json`

### Interpretation

The direct PPO path is now implemented and mechanically verified. The evaluator
can compare either residual or direct tuned checkpoints against the same frozen
base on the fixed local reset bank.

The one-update 512-env direct smoke does not pass the local gate, as expected
for a smoke run. It made very small action changes and left epsilon success
unchanged.

### Next action

Run a serious 4096-env direct PPO dev run on the clean/disturbed init bank. Use
the same fixed local gate before any closed-loop task evaluation. If final-layer
tuning remains too weak, run `--train-scope all` with a small learning rate and
compare checkpoint-per-update local-gate metrics.

## 2026-06-25 - Phase 1 direct privileged/TCP final-layer dev run

### Hypothesis

Residual PPO may fail because the residual head is too disconnected from the
BC low-level action distribution. Direct PPO on the BC low-level policy's final
layer may allow useful goal-conditioned local corrections while keeping most of
the supervised policy fixed.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-direct \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5 \
  --run-tag hcl_next_direct_final_layer_n4096_1m \
  --seed 0 \
  --steps 1024000 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --learning-rate 3e-5 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --checkpoint-every-updates 5 \
  --train-scope final_layer \
  --force
```

Saved checkpoints at steps `204800`, `409600`, `614400`, `819200`, and
`1024000` were evaluated on the fixed `local_reset_bank_n1800_seed0_k10.json`
bank.

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- init dataset: `privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- train scope: final low-policy layer plus log std and critic
- num_envs: 4096
- rollout steps: 10
- total transitions: 1,024,000
- fixed-bank base epsilon success: 0.8943
- fixed-bank base terminal MSE median: 0.00187
- fixed-bank base terminal MSE p90: 0.05421

### Results

Fixed-bank local paired eval:

| checkpoint step | mean paired improvement | median paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE median | tuned terminal MSE p90 | action delta L2 mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 204800 | 0.0317 | -0.0000078 | 0.4790 | 0.8943 | 0.00188 | 0.05437 | 0.0104 |
| 409600 | 0.0190 | -0.0000059 | 0.4839 | 0.8901 | 0.00194 | 0.05519 | 0.0129 |
| 614400 | 0.0426 | -0.0000341 | 0.4226 | 0.8953 | 0.00192 | 0.05255 | 0.0130 |
| 819200 | 0.3637 | -0.0000180 | 0.4551 | 0.8958 | 0.00194 | 0.05481 | 0.0143 |
| 1024000 | 0.0210 | -0.0000812 | 0.3811 | 0.8950 | 0.00211 | 0.05402 | 0.0158 |

Training history final row:

- mean paired improvement: -0.1290
- fraction improved: 0.4167
- mean action delta L2: 0.0297
- action saturation rate: 0.1112
- clip fraction: 0.0811

### Artifacts

- `artifacts/incremental/privileged_z_direct/hcl_next_direct_final_layer_n4096_1m/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct/hcl_next_direct_final_layer_n4096_1m/seed0/checkpoints/`
- `results/incremental/privileged_z_direct/hcl_next_direct_final_layer_n4096_1m/seed0/history.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_n4096_1m_step204800_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_n4096_1m_step409600_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_n4096_1m_step614400_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_n4096_1m_step819200_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_n4096_1m_latest_k10_4096.json`

### Interpretation

Final-layer direct PPO fails the local improvement gate. Some checkpoints
improve the fixed-bank mean because a few catastrophic base rollouts get less
bad, but the median paired improvement is negative at every saved checkpoint
and `fraction_improved` never reaches 0.5, let alone the `>0.55` gate.

The run does not justify closed-loop task evaluation. The failure is not simply
that residual authority was too small; directly tuning the final BC action
layer still does not produce broad local improvement.

### Next action

Run the next direct variant with `--train-scope all` and a smaller learning
rate. If full low-level tuning also fails, stop privileged/TCP PPO variants and
debug the reward/local-distance formulation with an offline optimizer or scratch
local PPO before returning to learned VAE/effect latents.

## 2026-06-25 - Phase 1 direct privileged/TCP all-layer dev run

### Hypothesis

Final-layer direct tuning may be too weak because the supervised policy's hidden
features are still shaped for BC, not reachability improvement. Full low-level
tuning with a lower learning rate and small BC penalty may give PPO enough
capacity to improve the local reachability objective without immediately
destroying the BC policy.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-direct \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5 \
  --run-tag hcl_next_direct_all_layers_n4096_1m_lr1e5_bc001 \
  --seed 0 \
  --steps 1024000 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --learning-rate 1e-5 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --checkpoint-every-updates 5 \
  --train-scope all \
  --bc-weight 0.01 \
  --force
```

Saved checkpoints at steps `204800`, `409600`, `614400`, `819200`, and
`1024000` were evaluated on the fixed `local_reset_bank_n1800_seed0_k10.json`
bank.

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- init dataset: `privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- train scope: all low-policy layers plus log std and critic
- BC penalty weight: 0.01
- num_envs: 4096
- rollout steps: 10
- total transitions: 1,024,000
- fixed-bank base epsilon success: 0.8943
- fixed-bank base terminal MSE median: 0.00187
- fixed-bank base terminal MSE p90: 0.05421

### Results

Fixed-bank local paired eval:

| checkpoint step | mean paired improvement | median paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE median | tuned terminal MSE p90 | action delta L2 mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 204800 | 0.0685 | -0.0000400 | 0.4392 | 0.8967 | 0.00198 | 0.05294 | 0.0157 |
| 409600 | 0.0840 | -0.000693 | 0.2786 | 0.8801 | 0.00310 | 0.06287 | 0.0329 |
| 614400 | 0.0747 | -0.000616 | 0.3066 | 0.8838 | 0.00320 | 0.06050 | 0.0319 |
| 819200 | 0.0782 | -0.001134 | 0.2683 | 0.8833 | 0.00399 | 0.06328 | 0.0375 |
| 1024000 | 0.0711 | -0.001511 | 0.2625 | 0.8738 | 0.00448 | 0.06686 | 0.0432 |

Training history final row:

- mean paired improvement: -0.0204
- fraction improved: 0.3699
- mean action delta L2: 0.0445
- action saturation rate: 0.1161
- clip fraction: 0.3000
- approximate KL: 0.0874

### Artifacts

- `artifacts/incremental/privileged_z_direct/hcl_next_direct_all_layers_n4096_1m_lr1e5_bc001/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct/hcl_next_direct_all_layers_n4096_1m_lr1e5_bc001/seed0/checkpoints/`
- `results/incremental/privileged_z_direct/hcl_next_direct_all_layers_n4096_1m_lr1e5_bc001/seed0/history.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_n4096_1m_lr1e5_bc001_step204800_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_n4096_1m_lr1e5_bc001_step409600_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_n4096_1m_lr1e5_bc001_step614400_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_n4096_1m_lr1e5_bc001_step819200_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_n4096_1m_lr1e5_bc001_latest_k10_4096.json`

### Interpretation

All-layer direct PPO fails more clearly than final-layer tuning. It can make
larger action changes, but those changes mostly degrade the already-good local
distribution. The best checkpoint is early (`204800`) and only improves epsilon
success by about 0.24 percentage points while still having negative median
paired improvement and `fraction_improved = 0.4392`.

Later checkpoints show reward hacking/outlier behavior: the mean paired
improvement stays positive because severe base failures improve, but median
behavior and epsilon success get worse. PPO diagnostics also become less
healthy by the final update (`clip_fraction = 0.30`, `approx_kl = 0.087`).

### Decision

Do not run closed-loop task evaluation for the direct all-layer checkpoints.
The privileged/TCP PPO branch has now failed in three forms:

1. residual PPO with paired reward;
2. direct final-layer PPO with paired reward;
3. direct all-layer PPO with paired reward.

This is strong evidence that the current local PPO/reward formulation is the
next bottleneck, not just VAE goal ignoring or residual capacity.

### Next action

Stop launching PPO variants on the current reward. Before returning to learned
latents, debug the local reachability objective directly:

1. inspect distance normalization and per-dimension weights for the 31D state;
2. add an offline local optimizer / random-shooting action-sequence sanity check
   on the fixed reset bank to test whether the reward admits broad improvements;
3. if the optimizer can improve the bank, use it to diagnose PPO credit
   assignment; if it cannot, redesign the distance/reward before more PPO.

## 2026-06-25 - Phase 1 privileged/TCP local action-search sanity check

### Hypothesis

The PPO failures may mean the current 31D normalized-state distance is a bad
local objective, or they may mean PPO is failing to optimize an objective that
does contain useful improvements. Compare the frozen base policy against two
non-PPO references on the same fixed local bank:

1. replay/demo action sequence from the source trajectory;
2. best-of-32 random shooting around the frozen base action, including the base
   action sequence as candidate zero.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-action-search \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --random-candidates 32 \
  --random-noise-std 0.05 \
  --output results/hcl_next_phase1/privileged_z_local_action_search_n1800_k10_random32_std005.json \
  --force
```

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- fixed local bank: `local_reset_bank_n1800_seed0_k10.json`
- goal source: replay state at `t + k`
- horizon: 10
- random shooting: 32 candidates, action noise std 0.05, base included

### Results

| method | terminal MSE mean | terminal MSE median | terminal MSE p90 | success within epsilon |
| --- | ---: | ---: | ---: | ---: |
| frozen base | 0.5051 | 0.00187 | 0.05421 | 0.8943 |
| replay/demo actions | 0.00642 | 0.0000107 | 0.000339 | 0.9958 |
| best-of-32 random around base | 0.0699 | 0.00134 | 0.00912 | 0.9753 |

Improvement over base:

| method | mean improvement | median improvement | fraction improved |
| --- | ---: | ---: | ---: |
| replay/demo actions | 0.4986 | 0.00162 | 0.9729 |
| best-of-32 random around base | 0.4351 | 0.00000 | 0.4829 |

Verification:

- `uv run python -m compileall -q src/hcl_poc/privileged_z.py src/hcl_poc/cli.py`
- `uv run pytest -q`: 22 passed

### Artifacts

- `results/hcl_next_phase1/privileged_z_local_action_search_n1800_k10_random32_std005.json`

### Interpretation

The local target is not impossible. Replay/demo actions almost exactly reach
the fixed replay goals, so the normalized privileged-state distance is at least
consistent with the demonstration dynamics on this bank.

Random shooting around the base also improves the hard tail substantially:
epsilon success rises from `0.8943` to `0.9753`, and p90 terminal MSE drops from
`0.05421` to `0.00912`. The median improvement is zero because the base is
already very good on more than half of the bank and best-of-random keeps the
base candidate for those cases.

This reframes the PPO failure: useful action-sequence corrections exist, but
the current PPO formulations are not finding them reliably. The fixed gate
metric `fraction_improved > 0.55` may also be too strict for a base with
`~0.89` epsilon success, because many already-easy cases cannot improve much.

### Next action

Use this action-search result to debug PPO credit assignment:

1. train/evaluate on a hard-case subset where frozen base terminal MSE exceeds
   the epsilon threshold;
2. try supervised distillation from random-shooting/replay-improved actions into
   the low-level policy before PPO;
3. revise the local gate to include hard-tail metrics such as p90 terminal MSE
   and epsilon success, not only fraction improved over all starts.

## 2026-06-25 - Phase 1 hard-case subset diagnostic

### Hypothesis

The full-bank local gate may hide useful learning because the frozen base is
already successful on about 89% of starts. A hard-case subset where frozen base
terminal MSE exceeds the epsilon threshold should better reveal whether PPO is
learning corrections for cases with meaningful room to improve.

### Commands

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  create-privileged-z-hard-case-manifest \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --threshold-mse 0.05 \
  --output data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct/hcl_next_direct_all_layers_n4096_1m_lr1e5_bc001/seed0/checkpoints/step_000204800.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_step204800_hard_mse_ge_0p05.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-paired \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct/hcl_next_direct_final_layer_n4096_1m/seed0/checkpoints/step_000204800.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --output results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_step204800_hard_mse_ge_0p05.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-local-action-search \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --random-candidates 32 \
  --random-noise-std 0.05 \
  --output results/hcl_next_phase1/privileged_z_local_action_search_n1800_k10_hard_mse_ge_0p05_random32_std005.json \
  --force
```

### Setup

- source fixed bank: `local_reset_bank_n1800_seed0_k10.json`
- hard threshold: frozen base terminal MSE >= 0.05
- selected hard local starts: 433 / 4096
- hard-subset base terminal MSE median: 0.1159
- hard-subset base terminal MSE p90: 0.8888

### Results

Hard-subset PPO comparison:

| tuned checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen base | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 |
| final-layer step 204800 | 4.4204 | 0.1089 | 0.8336 | 0.5242 | 0.0855 |
| all-layer step 204800 | 3.8878 | 0.1050 | 0.6834 | 0.6074 | 0.1386 |

Hard-subset non-PPO references:

| method | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon |
| --- | ---: | ---: | ---: | ---: | ---: |
| replay/demo actions | 0.0469 | 0.0000053 | 0.00284 | 0.9954 | 0.9815 |
| best-of-32 random around base | 0.6526 | 0.01237 | 0.1867 | 0.9792 | 0.7737 |

### Artifacts

- `data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_all_layers_step204800_hard_mse_ge_0p05.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_final_layer_step204800_hard_mse_ge_0p05.json`
- `results/hcl_next_phase1/privileged_z_local_action_search_n1800_k10_hard_mse_ge_0p05_random32_std005.json`

### Interpretation

The hard-case subset changes the diagnosis. Early PPO checkpoints do learn some
useful corrections for failed local starts, especially all-layer direct tuning:
success rises from `0.0` to `0.1386`, and `fraction_improved` exceeds 0.60 on
the selected hard cases. This was hidden by the full-bank metric, where the
base is already good on most starts and broad perturbations hurt easy cases.

The gap to non-PPO action search is still large. Best-of-32 random around the
base reaches `0.7737` success on the same hard subset, and replay/demo actions
reach `0.9815`. Therefore the current policy class/action interface can express
better local behavior, but PPO is extracting only a small part of the available
hard-tail improvement.

### Next action

Do not discard the privileged/TCP branch outright. Reframe the next experiment
as hard-tail improvement:

1. distill replay or best-of-random action sequences on the hard-case subset;
2. evaluate the distilled policy on both hard and full banks to check whether
   it preserves easy-case behavior;
3. only then consider PPO initialized from that hard-tail-distilled policy.

## 2026-06-25 - Phase 1 replay distillation diagnostic

### Hypothesis

Since replay/demo action sequences nearly solve the fixed local bank, supervised
distillation from replay actions should be a stronger initializer than PPO for
hard-tail local corrections. The key question is whether local replay
distillation preserves closed-loop behavior or overfits to fixed replay branches.

### Commands

Hard-subset all-layer distillation:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --run-tag hcl_next_replay_distill_hard_mse_ge_0p05_all_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --train-scope all \
  --force
```

Full-bank all-layer distillation:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --run-tag hcl_next_replay_distill_full_k10_all_lr1e4_e100 \
  --seed 0 \
  --epochs 100 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope all \
  --force
```

Full-bank final-layer distillation:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --run-tag hcl_next_replay_distill_full_k10_final_layer_lr1e4_e100 \
  --seed 0 \
  --epochs 100 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

Each distilled checkpoint was evaluated with `eval-privileged-z-local-paired`.
The full-bank all-layer and final-layer checkpoints were also evaluated in
closed-loop `hierarchy` and `oracle_hierarchy` modes for 200 episodes with
`seed-start = 9900000`.

### Setup

- representation: 31D privileged Push-T observation state
- base checkpoint: `clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- target: replay executed actions for every held-goal offset
- horizon: 10
- optimizer: Adam, supervised action MSE
- fixed-bank base success: 0.8943
- hard-subset base success: 0.0
- closed-loop base learned-high success: 0.395
- closed-loop base oracle-goal success: 0.635

### Local Results

Full fixed bank:

| checkpoint | train scope | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | - | 0.5051 | 0.00187 | 0.05421 | - | 0.8943 | - |
| hard-only replay distill | all | 0.3571 | 0.00646 | 0.05034 | 0.2705 | 0.8997 | 0.0938 |
| full replay distill | all | 0.1002 | 0.00104 | 0.03004 | 0.6636 | 0.9312 | 0.0594 |
| full replay distill | final layer | 0.5331 | 0.00176 | 0.05036 | 0.5334 | 0.8989 | 0.0397 |

Hard subset:

| checkpoint | train scope | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen base | - | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 |
| hard-only replay distill | all | 0.4201 | 0.02483 | 0.2978 | 0.8476 | 0.6236 |
| full replay distill | all | 0.6767 | 0.03539 | 0.3129 | 0.8314 | 0.5635 |

### Closed-Loop Results

200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean action delta/residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| full replay distill, all layers | hierarchy | 0.215 | 23.35 | 0.229 |
| full replay distill, final layer | hierarchy | 0.370 | 35.31 | 0.061 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| full replay distill, all layers | oracle_hierarchy | 0.255 | 23.83 | 0.259 |
| full replay distill, final layer | oracle_hierarchy | 0.495 | 39.30 | 0.058 |

### Artifacts

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_hard_mse_ge_0p05_all_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_full_k10_all_lr1e4_e100/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_full_k10_final_layer_lr1e4_e100/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_all_lr1e4_e200_hard_mse_ge_0p05.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_all_lr1e4_e200_full_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_full_all_lr1e4_e100_full_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_full_all_lr1e4_e100_hard_mse_ge_0p05.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_full_final_layer_lr1e4_e100_full_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_replay_distill_full_hierarchy_n1800_seed9900000_e200.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_replay_distill_full_oracle_hierarchy_n1800_seed9900000_e200.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_replay_distill_full_final_layer_hierarchy_n1800_seed9900000_e200.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_replay_distill_full_final_layer_oracle_hierarchy_n1800_seed9900000_e200.json`

### Interpretation

Replay distillation confirms that local branch supervision can produce the
kind of improvement PPO failed to find. Full-bank all-layer replay distillation
passes the local gate strongly: success improves from `0.8943` to `0.9312`,
p90 MSE drops from `0.05421` to `0.03004`, and `fraction_improved = 0.6636`.

However, the same all-layer checkpoint badly damages closed-loop performance,
even with oracle high-level goals (`0.635 -> 0.255`). This means the local
replay branch distribution is not enough by itself. It overfits to fixed replay
states/goals and moves too far from the robust closed-loop action distribution.

Final-layer distillation is safer but still not good enough: learned-high
closed-loop success drops from `0.395` to `0.370`, and oracle-goal success drops
from `0.635` to `0.495`.

### Decision

Do not use these replay-distilled checkpoints for final closed-loop claims or
as-is PPO initialization. The useful result is diagnostic: supervised local
branch correction works on the fixed local objective, but preserving the
closed-loop state distribution requires either stronger regularization/mixing
or training data collected from closed-loop states.

### Next action

The next privileged/TCP experiment should be distribution-aware distillation:

1. mix replay-improvement targets with a strong base-action preservation loss;
2. train only on hard cases or use sample weights so easy cases stay close to
   the frozen base;
3. evaluate local full/hard banks and closed-loop oracle-goal before any PPO.

## 2026-06-25 - Distribution-Aware Replay Distillation

### Code Changes

`train-privileged-z-local-replay-distill` now supports weighted target mixes:

- `--preserve-manifest`: additional manifest whose target is the frozen base
  low-level action instead of the replay action.
- `--replay-weight`: per-sample weight for replay-action targets.
- `--preserve-weight`: per-sample weight for base-action preservation targets.

I also created the complement manifest:

- `data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json`

This contains the 3,663 fixed-bank starts that are not in the
`mse >= 0.05` hard manifest. The intent is to avoid directly asking the same
hard states to both copy replay and preserve the failing base action.

Verification after the code change:

```bash
uv run python -m compileall src/hcl_poc/privileged_z.py src/hcl_poc/cli.py
uv run pytest -q
```

Result: `22 passed`.

### Runs

Full-bank preservation, all layers:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10.json \
  --replay-weight 1.0 \
  --preserve-weight 5.0 \
  --run-tag hcl_next_replay_distill_hard_plus_full_preserve_w5_all_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope all \
  --force
```

Easy-case preservation, all layers:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --replay-weight 1.0 \
  --preserve-weight 1.0 \
  --run-tag hcl_next_replay_distill_hard_plus_easy_preserve_w1_all_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope all \
  --force
```

Easy-case preservation, final layer:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --replay-weight 1.0 \
  --preserve-weight 1.0 \
  --run-tag hcl_next_replay_distill_hard_plus_easy_preserve_w1_final_layer_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Local Results

Full fixed bank:

| checkpoint | train scope | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | - | 0.5051 | 0.00187 | 0.05421 | - | 0.8943 | - |
| hard + full preserve w5 | all | 0.2108 | 0.00201 | 0.05390 | 0.4512 | 0.8945 | 0.0181 |
| hard + easy preserve w1 | all | 0.0466 | 0.00184 | 0.03535 | 0.4966 | 0.9287 | 0.0341 |
| hard + easy preserve w1 | final layer | 0.2905 | 0.00201 | 0.05206 | 0.4902 | 0.8982 | 0.0299 |

Hard subset:

| checkpoint | train scope | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | - | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 | - |
| hard + full preserve w5 | all | 1.9242 | 0.1019 | 0.6579 | 0.6143 | 0.1501 | 0.0530 |
| hard + easy preserve w1 | all | 0.3772 | 0.0510 | 0.3420 | 0.8176 | 0.4850 | 0.1556 |
| hard + easy preserve w1 | final layer | 1.5418 | 0.0920 | 0.6332 | 0.6882 | 0.2610 | 0.0893 |

### Closed-Loop Results

200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean action delta/residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| hard + easy preserve w1, all layers | hierarchy | 0.225 | 25.11 | 0.168 |
| hard + easy preserve w1, final layer | hierarchy | 0.340 | 34.63 | 0.063 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| hard + easy preserve w1, all layers | oracle_hierarchy | 0.300 | 27.30 | 0.203 |
| hard + easy preserve w1, final layer | oracle_hierarchy | 0.515 | 39.69 | 0.065 |

### Artifacts

- `data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_hard_plus_full_preserve_w5_all_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_hard_plus_easy_preserve_w1_all_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_hard_plus_easy_preserve_w1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_plus_full_preserve_w5_all_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_plus_full_preserve_w5_all_hard.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_plus_easy_preserve_w1_all_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_plus_easy_preserve_w1_all_hard.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_replay_distill_hard_plus_easy_preserve_w1_all_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_replay_distill_hard_plus_easy_preserve_w1_all_200eps.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_plus_easy_preserve_w1_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_plus_easy_preserve_w1_final_layer_hard.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_replay_distill_hard_plus_easy_preserve_w1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_replay_distill_hard_plus_easy_preserve_w1_final_layer_200eps.json`

### Interpretation

The easy-preservation variant is the best fixed-bank compromise so far. It
raises full-bank success from `0.8943` to `0.9287` and hard-subset success from
`0.0` to `0.4850`, while using a smaller mean action delta than full replay
distillation (`0.0341` vs `0.0594` on the full bank).

It still fails the real closed-loop gate. All-layer training damages both
learned-high and oracle-high rollouts. Final-layer training reduces that damage
but still underperforms the frozen base (`0.340 < 0.395` learned-high,
`0.515 < 0.635` oracle-high).

The failure is now sharper: fixed replay-reset supervision can be regularized
enough to preserve easy fixed-bank states, but it still does not preserve the
closed-loop state/goal distribution. More weight sweeps on the same fixed bank
are unlikely to solve the main issue.

### Next action

Move from fixed replay-reset distillation to closed-loop-state supervision:

1. collect states/goals reached by the frozen hierarchy during closed-loop
   rollouts;
2. label those states with base actions for preservation and only add replay or
   search-improvement targets where a local diagnostic shows clear benefit;
3. gate every candidate first on oracle-high closed-loop success, not only the
   fixed local reset bank.

## 2026-06-25 - Closed-Loop Preservation Bank

### Code Changes

Added `collect-privileged-z-closed-loop-preserve-bank`, which runs the frozen
privileged hierarchy and saves the low-level condition plus the clipped frozen
base action actually executed at each closed-loop step. The distillation trainer
can now consume this NPZ through:

- `--preserve-npz`
- `--preserve-npz-weight`

This makes closed-loop state/action preservation explicit instead of relying on
fixed replay-reset states as a proxy.

Verification:

```bash
uv run python -m compileall src/hcl_poc/privileged_z.py src/hcl_poc/cli.py
uv run pytest -q
```

Result: `22 passed`.

### Collection

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  collect-privileged-z-closed-loop-preserve-bank \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 512 \
  --seed-start 9900000 \
  --num-envs 64 \
  --output data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --force
```

Collected 51,200 low-level samples. The frozen hierarchy success over this
512-episode collection was `0.4473`.

### Runs

Final-layer distillation with closed-loop preservation weight 1.0:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --replay-weight 1.0 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --run-tag hcl_next_replay_distill_hard_easy_closedloop_preserve_w1_final_layer_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

Final-layer distillation with closed-loop preservation weight 0.5:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --replay-weight 1.0 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 0.5 \
  --run-tag hcl_next_replay_distill_hard_easy_closedloop_preserve_npz05_final_layer_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Local Results

Full fixed bank:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 0.5051 | 0.00187 | 0.05421 | - | 0.8943 | - |
| hard + easy + closed preserve w1.0 | 0.4975 | 0.00187 | 0.05466 | 0.5146 | 0.8928 | 0.0203 |
| hard + easy + closed preserve w0.5 | 0.4147 | 0.00190 | 0.05288 | 0.5059 | 0.8948 | 0.0225 |

Hard subset:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 | - |
| hard + easy + closed preserve w1.0 | 3.6872 | 0.0978 | 0.6499 | 0.6744 | 0.1594 | 0.0550 |
| hard + easy + closed preserve w0.5 | 2.8461 | 0.0966 | 0.6033 | 0.6790 | 0.1917 | 0.0647 |

### Closed-Loop Results

200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean action delta/residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| hard + easy + closed preserve w1.0 | hierarchy | 0.425 | 38.40 | 0.011 |
| hard + easy + closed preserve w0.5 | hierarchy | 0.420 | 38.08 | 0.015 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| hard + easy + closed preserve w1.0 | oracle_hierarchy | 0.630 | 45.06 | 0.016 |
| hard + easy + closed preserve w0.5 | oracle_hierarchy | 0.635 | 45.12 | 0.020 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_hard_easy_closedloop_preserve_w1_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_replay_distill_hard_easy_closedloop_preserve_npz05_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_easy_closedloop_preserve_w1_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_easy_closedloop_preserve_w1_final_layer_hard.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_replay_distill_hard_easy_closedloop_preserve_w1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_replay_distill_hard_easy_closedloop_preserve_w1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_easy_closedloop_preserve_npz05_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_replay_distill_hard_easy_closedloop_preserve_npz05_final_layer_hard.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_replay_distill_hard_easy_closedloop_preserve_npz05_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_replay_distill_hard_easy_closedloop_preserve_npz05_final_layer_200eps.json`

### Interpretation

Closed-loop preservation is the first variant that improves the actual
learned-high hierarchy gate while keeping oracle-high at the frozen base level.
The gains are small but qualitatively important:

- learned-high success: `0.395 -> 0.425` for NPZ weight `1.0`
- oracle-high success: `0.635 -> 0.630`
- learned-high success: `0.395 -> 0.420` for NPZ weight `0.5`
- oracle-high success: `0.635 -> 0.635`

The tradeoff is that hard fixed-bank correction is much weaker than in
fixed-bank-only distillation. That is acceptable for now because the previous
strong local variants failed closed-loop deployment badly.

### Decision

Use closed-loop preservation as the gate for future low-level changes. The next
step should collect a larger and more diverse closed-loop preservation bank,
then add improvement targets from closed-loop states rather than only from the
single fixed replay hard subset. A useful next candidate is to collect failure
or near-failure closed-loop states, run local action search from those live
states, and distill only action-search improvements that beat the frozen base.

## 2026-06-25 - Closed-Loop Action-Search Improvement Bank

### Code Changes

Added `collect-privileged-z-closed-loop-action-search-bank`. The command runs
the frozen hierarchy in closed loop, branches at live high-level decision
states, random-shoots low-level action sequences in a separate resettable
environment, and saves only branches where the best candidate improves terminal
MSE by at least a threshold.

The local replay distillation trainer can now consume improvement targets from
NPZ files through:

- `--improve-npz`
- `--improve-npz-weight`

This separates three roles in the same final-layer distillation run:

- hard replay-reset correction;
- easy and closed-loop base-action preservation;
- closed-loop action-search improvement targets.

### Collection

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  collect-privileged-z-closed-loop-action-search-bank \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 128 \
  --seed-start 9900000 \
  --num-envs 64 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-search-batches 32 \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001.npz \
  --force
```

Summary:

| metric | value |
| --- | ---: |
| condition rows | 7190 |
| searched branches | 1280 |
| selected branches | 719 |
| selected fraction | 0.5617 |
| selected base MSE mean / median / p90 | 0.8344 / 0.1395 / 2.3075 |
| selected best MSE mean / median / p90 | 0.2059 / 0.0303 / 0.5355 |
| selected improvement MSE mean / median / p90 | 0.6285 / 0.0640 / 2.2065 |
| searched base success within epsilon | 0.5305 |
| searched best success within epsilon | 0.7625 |
| collection frozen hierarchy success | 0.4375 |

This is a useful diagnostic: random shooting from the actual closed-loop
high-level states often finds local action sequences that beat the frozen low
level.

### Runs

All runs used final-layer distillation, `epochs = 200`, `batch-size = 1024`,
`learning-rate = 1e-4`, hard replay-reset weight `0.25`, easy fixed-bank
preservation weight `1.0`, and the same closed-loop preservation/action-search
NPZ files.

| checkpoint | preserve NPZ weight | improve NPZ weight |
| --- | ---: | ---: |
| `hcl_next_closedloop_search_improve_c16_imp2_preserve_npz05_final_layer_lr1e4_e200` | 0.5 | 2.0 |
| `hcl_next_closedloop_search_improve_c16_imp1_preserve_npz05_final_layer_lr1e4_e200` | 0.5 | 1.0 |
| `hcl_next_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_lr1e4_e200` | 1.0 | 1.0 |

Representative training command for the last variant:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 1.0 \
  --run-tag hcl_next_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Closed-Loop Results

200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| closed preserve w1.0 | hierarchy | 0.425 | 38.40 | 0.011 |
| closed search imp2 preserve0.5 | hierarchy | 0.445 | 38.58 | 0.0106 |
| closed search imp1 preserve0.5 | hierarchy | 0.460 | 40.29 | 0.0089 |
| closed search imp1 preserve1.0 | hierarchy | 0.410 | 38.69 | 0.0077 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| closed preserve w1.0 | oracle_hierarchy | 0.630 | 45.06 | 0.016 |
| closed search imp2 preserve0.5 | oracle_hierarchy | 0.605 | 43.31 | 0.0129 |
| closed search imp1 preserve0.5 | oracle_hierarchy | 0.610 | 43.53 | 0.0127 |
| closed search imp1 preserve1.0 | oracle_hierarchy | 0.635 | 44.53 | 0.0120 |

### Local Results

Full fixed bank:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 0.5051 | 0.00187 | 0.05421 | - | 0.8943 | - |
| closed search imp2 preserve0.5 | 0.5182 | 0.00193 | 0.05558 | 0.4341 | 0.8901 | 0.0171 |
| closed search imp1 preserve1.0 | 0.4718 | 0.00189 | 0.05563 | 0.4644 | 0.8911 | 0.0161 |

Hard subset:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 | - |
| closed search imp2 preserve0.5 | 3.9440 | 0.1064 | 0.6331 | 0.5958 | 0.1247 | 0.0428 |
| closed search imp1 preserve1.0 | 3.9614 | 0.1072 | 0.6022 | 0.5935 | 0.1109 | 0.0392 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_imp2_preserve_npz05_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_imp1_preserve_npz05_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_imp2_preserve_npz05_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_imp2_preserve_npz05_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_imp1_preserve_npz05_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_imp1_preserve_npz05_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_imp1_preserve_npz1_final_layer_hard.json`

### Interpretation

Closed-loop action search is useful as a diagnostic and as a source of
improvement labels, but the first distillation targets are not yet robust enough
to be the new default.

The strongest learned-high result came from `improve = 1.0`, `preserve NPZ =
0.5`: `0.395 -> 0.460` hierarchy success. However, it damaged oracle-high
rollouts (`0.635 -> 0.610`), which means the low-level change is not purely a
local correction and can interfere with good high-level goals.

Increasing closed-loop preservation to `1.0` restores oracle-high success to
`0.635`, but learned-high falls to `0.410` and hard fixed-bank success is only
`0.1109`. This is safer but weaker than the preservation-only checkpoint.

### Decision

Keep `closed preserve w1.0` as the current safest deployable low-level
distillation checkpoint. Treat the action-search bank as evidence that local
improvement targets exist, but add another filter before using them for
training. The next candidate should only accept searched targets that both:

1. improve terminal MSE over the frozen base; and
2. stay close to the frozen base action sequence or preserve oracle-high
rollouts under an oracle-mode gate.

## 2026-06-25 - Action-Delta Filtered Closed-Loop Search

### Code Changes

Added `--max-action-delta-l2` to
`collect-privileged-z-closed-loop-action-search-bank`. The collector now stores
`selected_action_delta_l2` and `searched_action_delta_l2`, where each value is
the mean per-step L2 distance between the selected searched action sequence and
the frozen base action sequence over the held-goal horizon.

The purpose is to reject action-search labels that improve the branch-local MSE
only by moving far from the base low-level behavior.

Verification:

```bash
uv run python -m compileall src/hcl_poc/privileged_z.py src/hcl_poc/cli.py
uv run pytest -q
```

Result: `22 passed`.

### Filtered Collection

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  collect-privileged-z-closed-loop-action-search-bank \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 128 \
  --seed-start 9900000 \
  --num-envs 64 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.08 \
  --max-search-batches 32 \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001_delta008.npz \
  --force
```

Summary:

| metric | value |
| --- | ---: |
| condition rows | 480 |
| searched branches | 1280 |
| selected branches | 48 |
| selected fraction | 0.0375 |
| selected base MSE mean / median / p90 | 1.1074 / 1.1715 / 2.2641 |
| selected best MSE mean / median / p90 | 0.0624 / 0.0108 / 0.1058 |
| selected improvement MSE mean / median / p90 | 1.0450 / 0.8228 / 2.2609 |
| selected action delta L2 mean / median / p90 | 0.0720 / 0.0727 / 0.0788 |
| searched base success within epsilon | 0.5016 |
| searched best success within epsilon | 0.7250 |
| collection frozen hierarchy success | 0.3984 |

The action-delta filter is very strict: it keeps only `48 / 1280` searched
branches. The retained branches are high-improvement and near the intended
action-delta boundary.

### Runs

Both runs used final-layer distillation with:

- hard replay-reset weight `0.25`;
- easy fixed-bank preservation weight `1.0`;
- closed-loop preservation NPZ weight `1.0`;
- filtered improvement NPZ
  `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001_delta008.npz`;
- `epochs = 200`, `batch-size = 1024`, `learning-rate = 1e-4`.

| checkpoint | improve NPZ weight |
| --- | ---: |
| `hcl_next_closedloop_search_improve_c16_delta008_imp16_preserve_npz1_final_layer_lr1e4_e200` | 16.0 |
| `hcl_next_closedloop_search_improve_c16_delta008_imp4_preserve_npz1_final_layer_lr1e4_e200` | 4.0 |

### Closed-Loop Results

200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| closed preserve w1.0 | hierarchy | 0.425 | 38.40 | 0.011 |
| filtered search imp16 | hierarchy | 0.420 | 37.53 | 0.0074 |
| filtered search imp4 | hierarchy | 0.430 | 39.33 | 0.0077 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| closed preserve w1.0 | oracle_hierarchy | 0.630 | 45.06 | 0.016 |
| filtered search imp16 | oracle_hierarchy | 0.585 | 42.42 | 0.0124 |
| filtered search imp4 | oracle_hierarchy | 0.620 | 44.01 | 0.0119 |

### Local Results

The lower-weight filtered checkpoint was evaluated on the fixed banks.

Full fixed bank:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 0.5051 | 0.00187 | 0.05421 | - | 0.8943 | - |
| filtered search imp4 | 0.1375 | 0.00189 | 0.05480 | 0.4905 | 0.8936 | 0.0149 |

Hard subset:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 | - |
| filtered search imp4 | 0.9702 | 0.1010 | 0.5922 | 0.6143 | 0.1155 | 0.0397 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001_delta008.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_delta008_imp16_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_delta008_imp4_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_delta008_imp16_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_delta008_imp16_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_delta008_imp4_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_delta008_imp4_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_delta008_imp4_preserve_npz1_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_delta008_imp4_preserve_npz1_final_layer_hard.json`

### Interpretation

The action-distance filter improves the fixed-bank behavior dramatically, but
it still does not produce a clean closed-loop win.

`imp16` overweights a tiny high-signal filtered bank and damages oracle-high
rollouts badly (`0.635 -> 0.585`). `imp4` is much safer and gives the best hard
fixed-bank MSE so far (`4.7270 -> 0.9702`) with small mean action delta, but its
closed-loop result is only marginally better than preservation-only on
learned-high (`0.430` vs `0.425`) and slightly worse on oracle-high (`0.620` vs
`0.630`).

This strengthens the previous conclusion: fixed-bank and branch-local metrics
can look very good while still not transferring cleanly to full rollouts.

### Decision

Do not promote filtered action-search distillation over closed-loop
preservation-only. The next useful experiment should change the acceptance
criterion from an action-distance heuristic to an oracle-rollout gate: accept an
improvement target only if it improves the learned-high branch and does not
degrade an oracle-high branch from the same state/goal.

## 2026-06-25 - Oracle-Gated Closed-Loop Action Search

### Code Changes

Added `--oracle-gate-max-degradation-mse` to
`collect-privileged-z-closed-loop-action-search-bank`. When enabled, the
collector computes an oracle high-level goal from the same simulator branch
state and accepts a learned-goal action-search target only if executing the
selected action sequence does not make oracle-goal terminal MSE worse than the
frozen low-level oracle-goal branch by more than the configured tolerance.

The saved NPZ now includes oracle-gate diagnostics:

- `selected_oracle_base_mse`
- `selected_oracle_candidate_mse`
- `selected_oracle_delta_mse`
- `searched_oracle_base_mse`
- `searched_oracle_candidate_mse`
- `searched_oracle_delta_mse`

Also fixed the collector's searched-action-delta accounting by initializing the
per-branch best action sequence to the frozen base sequence. Previously,
non-improved branches kept a zero placeholder; selected rows were still valid,
but searched action-delta diagnostics for non-selected rows were not meaningful.

### Collection

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  collect-privileged-z-closed-loop-action-search-bank \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 128 \
  --seed-start 9900000 \
  --num-envs 64 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --oracle-gate-max-degradation-mse 0.0 \
  --max-search-batches 32 \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001_oraclegate0.npz \
  --force
```

Summary:

| metric | value |
| --- | ---: |
| condition rows | 2700 |
| searched branches | 1280 |
| selected branches | 270 |
| selected fraction | 0.2109 |
| selected base MSE mean / median / p90 | 1.0088 / 0.1616 / 2.7423 |
| selected best MSE mean / median / p90 | 0.2877 / 0.0443 / 0.7852 |
| selected improvement MSE mean / median / p90 | 0.7211 / 0.0600 / 2.6140 |
| selected action delta L2 mean / median / p90 | 0.1869 / 0.1525 / 0.3128 |
| selected oracle base MSE mean / median / p90 | 5.5084 / 1.0948 / 4.8297 |
| selected oracle candidate MSE mean / median / p90 | 1.1070 / 0.1754 / 1.0669 |
| selected oracle delta MSE mean / median / p90 | -4.4014 / -0.4952 / -0.0103 |
| searched oracle gate pass fraction | 0.3797 |
| collection frozen hierarchy success | 0.4297 |

The oracle gate keeps a useful middle-sized bank: stricter than unfiltered
action search (`270` vs `719` selected branches), but far less sparse than the
action-distance filter (`270` vs `48` selected branches). The selected labels
improve both the learned-goal branch and the oracle-goal branch under the
branch-local diagnostic.

### Runs

Both runs used final-layer distillation with hard replay-reset weight `0.25`,
easy fixed-bank preservation weight `1.0`, closed-loop preservation NPZ weight
`1.0`, `epochs = 200`, `batch-size = 1024`, and `learning-rate = 1e-4`.

| checkpoint | improve NPZ weight |
| --- | ---: |
| `hcl_next_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_lr1e4_e200` | 2.0 |
| `hcl_next_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_lr1e4_e200` | 1.0 |

### Closed-Loop Results

Dev seed window, 200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| closed preserve w1.0 | hierarchy | 0.425 | 38.40 | 0.011 |
| oracle-gated search imp2 | hierarchy | 0.455 | 39.95 | 0.0072 |
| oracle-gated search imp1 | hierarchy | 0.445 | 39.66 | 0.0072 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| closed preserve w1.0 | oracle_hierarchy | 0.630 | 45.06 | 0.016 |
| oracle-gated search imp2 | oracle_hierarchy | 0.675 | 46.52 | 0.0122 |
| oracle-gated search imp1 | oracle_hierarchy | 0.660 | 45.85 | 0.0114 |

Fresh seed window, 500 episodes, `seed-start = 10000000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.458 | 38.98 | 0.000 |
| oracle-gated search imp2 | hierarchy | 0.448 | 38.63 | 0.0084 |
| oracle-gated search imp1 | hierarchy | 0.464 | 39.81 | 0.0074 |
| frozen base | oracle_hierarchy | 0.626 | 45.05 | 0.000 |
| oracle-gated search imp2 | oracle_hierarchy | 0.658 | 46.37 | 0.0116 |
| oracle-gated search imp1 | oracle_hierarchy | 0.666 | 45.94 | 0.0116 |

### Local Results

Full fixed bank:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 0.5051 | 0.00187 | 0.05421 | - | 0.8943 | - |
| oracle-gated search imp2 | 0.4231 | 0.00191 | 0.05553 | 0.4795 | 0.8923 | 0.0155 |
| oracle-gated search imp1 | 0.4638 | 0.00188 | 0.05335 | 0.4893 | 0.8950 | 0.0153 |

Hard subset:

| checkpoint | terminal MSE mean | terminal MSE median | terminal MSE p90 | fraction improved | success within epsilon | action delta L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 4.7270 | 0.1159 | 0.8888 | - | 0.0000 | - |
| oracle-gated search imp2 | 3.7274 | 0.1082 | 0.6054 | 0.5935 | 0.1109 | 0.0407 |
| oracle-gated search imp1 | 4.0571 | 0.1061 | 0.5528 | 0.6005 | 0.1316 | 0.0397 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001_oraclegate0.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_base_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_oraclegate0_imp2_preserve_npz1_final_layer_hard.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_full.json`
- `results/hcl_next_phase1/privileged_z_local_paired_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_hard.json`

### Interpretation

Oracle-gated action search is the first action-search variant that improves the
dev closed-loop learned-high gate while also improving oracle-high instead of
damaging it. The branch-local oracle gate was therefore the right acceptance
criterion to add.

`imp2` has the strongest dev-window result (`0.455` learned-high and `0.675`
oracle-high), but it does not generalize on the fresh learned-high window
(`0.448` vs frozen base `0.458`). `imp1` has a smaller dev gain, but is the
first candidate that beats frozen base on both fresh learned-high and fresh
oracle-high:

- fresh learned-high: `0.458 -> 0.464`
- fresh oracle-high: `0.626 -> 0.666`

The gains are still modest and should not be treated as final evidence that the
RL/reachability formulation is solved. They do show that filtering improvement
targets with an oracle branch gate can produce a low-level update that survives
the main closed-loop deployment checks.

### Decision

Promote `hcl_next_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_lr1e4_e200`
as the current best low-level distillation candidate for the privileged
`N=1800, k=10` sanity track.

Next actions:

1. repeat oracle-gated collection on a larger/more diverse closed-loop bank;
2. evaluate the promoted checkpoint over more fresh seeds before claiming a
   stable improvement;
3. if the effect persists, move the same oracle-gated target-selection idea into
   the real RL/reachability formulation instead of continuing small supervised
   distillation sweeps.

## 2026-06-25 - Additional Fresh-Seed Validation for Oracle-Gated imp1

### Purpose

The first fresh window for
`hcl_next_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_lr1e4_e200`
was positive on both learned-high and oracle-high. To check whether that was a
stable effect rather than a seed-window artifact, I ran two more 500-episode
fresh windows at `seed-start = 10100000` and `10200000`, paired against the
frozen base checkpoint.

### Results

500 episodes per row:

| seed-start | mode | frozen base success | tuned success | delta | frozen base return | tuned return | delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000000 | hierarchy | 0.458 | 0.464 | +0.006 | 38.98 | 39.81 | +0.83 |
| 10100000 | hierarchy | 0.470 | 0.480 | +0.010 | 39.82 | 40.24 | +0.42 |
| 10200000 | hierarchy | 0.446 | 0.440 | -0.006 | 39.29 | 39.30 | +0.01 |
| 10000000 | oracle_hierarchy | 0.626 | 0.666 | +0.040 | 45.05 | 45.94 | +0.89 |
| 10100000 | oracle_hierarchy | 0.654 | 0.672 | +0.018 | 46.88 | 46.20 | -0.68 |
| 10200000 | oracle_hierarchy | 0.638 | 0.644 | +0.006 | 46.48 | 45.71 | -0.77 |

Aggregate over the three fresh windows:

| mode | frozen base success mean | tuned success mean | delta success mean | frozen base return mean | tuned return mean | delta return mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hierarchy | 0.4580 | 0.4613 | +0.0033 | 39.36 | 39.78 | +0.42 |
| oracle_hierarchy | 0.6393 | 0.6607 | +0.0213 | 46.14 | 45.95 | -0.19 |

### Artifacts

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_base_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_base_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_imp1_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`

### Interpretation

The promoted checkpoint still looks directionally useful, but the learned-high
effect is marginal across fresh seeds. It improves learned-high success in two
of three fresh windows and the mean success delta is only `+0.0033`. The
oracle-high success lift is more consistent: all three fresh windows are
positive, with mean success delta `+0.0213`.

Returns are mixed for oracle-high despite higher success, so this should not be
reported as a broad task-return improvement. The more defensible conclusion is:
the oracle gate solves the previous oracle-high damage problem and creates a
small learned-high improvement signal, but larger/more diverse target collection
is needed before claiming a robust full-policy improvement.

### Decision

Keep the same checkpoint as the current best candidate, but downgrade the
promotion language: it is the best *diagnostic* candidate, not yet a stable
improved policy. The next experiment should collect a larger oracle-gated bank
from multiple seed windows rather than continue tuning weights on the small
`n128` bank.

## 2026-06-25 - Multi-Window Oracle-Gated Bank

### Purpose

The small oracle-gated bank gave a promising but marginal learned-high effect.
This experiment tests whether more diverse oracle-gated improvement targets
help by collecting the same `n128` bank on two additional seed windows and
merging all three windows into a larger `n384` NPZ.

### Collection

Additional collection commands:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  collect-privileged-z-closed-loop-action-search-bank \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 128 \
  --seed-start 10100000 \
  --num-envs 64 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --oracle-gate-max-degradation-mse 0.0 \
  --max-search-batches 32 \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed10100000_c16_std005_min001_oraclegate0.npz \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  collect-privileged-z-closed-loop-action-search-bank \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 128 \
  --seed-start 10200000 \
  --num-envs 64 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --oracle-gate-max-degradation-mse 0.0 \
  --max-search-batches 32 \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed10200000_c16_std005_min001_oraclegate0.npz \
  --force
```

Collection summaries:

| seed-start | selected branches | searched branches | selected fraction | selected improvement MSE mean | selected action delta L2 mean | selected oracle delta MSE mean | collection base success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9900000 | 270 | 1280 | 0.2109 | 0.7211 | 0.1869 | -4.4014 | 0.4297 |
| 10100000 | 248 | 1280 | 0.1938 | 14.5466 | 0.2198 | -9.0017 | 0.4844 |
| 10200000 | 217 | 1280 | 0.1695 | 28.2455 | 0.2194 | -17.2059 | 0.5000 |
| merged | 735 | 3840 | 0.1914 | 13.5122 | 0.2076 | -9.7340 | 0.4714 |

Merged artifact:

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0.npz`

The later seed windows contain very large MSE outliers. The oracle gate still
selects branches that improve oracle-goal MSE locally, but the merged target
distribution is much heavier-tailed than the original `n128` bank.

### Run

The merged bank has `7350` low-level rows, about `2.7x` the original oracle-gated
bank. To avoid increasing the effective improvement-loss mass, I used
`--improve-npz-weight 0.5`.

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.5 \
  --run-tag hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_lr1e4_e200 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Dev seed window, 200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| oracle-gated search imp1 small bank | hierarchy | 0.445 | 39.66 | 0.0072 |
| oracle-gated search multi384 imp0.5 | hierarchy | 0.460 | 38.41 | 0.0073 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| oracle-gated search imp1 small bank | oracle_hierarchy | 0.660 | 45.85 | 0.0114 |
| oracle-gated search multi384 imp0.5 | oracle_hierarchy | 0.655 | 45.18 | 0.0122 |

Fresh seed windows, 500 episodes per row:

| seed-start | mode | frozen base success | small-bank imp1 success | multi384 imp0.5 success | multi384 delta vs base |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10000000 | hierarchy | 0.458 | 0.464 | 0.454 | -0.004 |
| 10100000 | hierarchy | 0.470 | 0.480 | 0.454 | -0.016 |
| 10200000 | hierarchy | 0.446 | 0.440 | 0.458 | +0.012 |
| 10000000 | oracle_hierarchy | 0.626 | 0.666 | 0.642 | +0.016 |
| 10100000 | oracle_hierarchy | 0.654 | 0.672 | 0.644 | -0.010 |
| 10200000 | oracle_hierarchy | 0.638 | 0.644 | 0.618 | -0.020 |

Aggregate over the three fresh windows:

| mode | frozen base success mean | small-bank imp1 success mean | multi384 imp0.5 success mean | multi384 delta vs base |
| --- | ---: | ---: | ---: | ---: |
| hierarchy | 0.4580 | 0.4613 | 0.4553 | -0.0027 |
| oracle_hierarchy | 0.6393 | 0.6607 | 0.6347 | -0.0047 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed10100000_c16_std005_min001_oraclegate0.npz`
- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed10200000_c16_std005_min001_oraclegate0.npz`
- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_imp05_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`

### Interpretation

The naive larger-bank scaling did not help. It looked competitive on the dev
seed window, but on the three fresh windows it underperformed both the frozen
base and the smaller-bank oracle-gated `imp1` checkpoint on average.

This is useful negative evidence. The problem is not just that the small bank
was too small; target distribution quality matters. The larger bank introduced
heavy-tailed local improvement labels from later seed windows, and the simple
uniform supervised loss did not turn those into a better closed-loop policy.

### Decision

Keep the smaller `n128` oracle-gated `imp1` checkpoint as the best diagnostic
candidate. Do not train further on the merged `multi384` bank without additional
filtering or weighting. The next useful change is to make target selection or
loss weighting robust to heavy-tailed branch improvements, for example by:

1. capping selected branch improvement MSE / action delta;
2. stratifying by branch difficulty instead of uniformly weighting all selected
   rows;
3. filtering out extreme selected branches whose base MSE is dominated by
   unstable simulator outliers.

## 2026-06-25 - Base-MSE-Capped Multi-Window Oracle-Gated Bank

### Purpose

The unfiltered multi-window oracle-gated bank failed fresh validation because
later seed windows contributed extreme selected-branch MSE outliers. Post-hoc
row filtering was unsafe because the existing NPZ layout stores selected
condition/action rows step-major within collection chunks and does not preserve
enough branch-to-row metadata. I added collector-side filtering instead.

### Implementation

Added `--max-base-mse` to
`collect-privileged-z-closed-loop-action-search-bank`. When provided, selected
branches must also have frozen-base terminal MSE at most that value. The value
is saved in the NPZ metadata and printed in the collection summary.

Verification after the code change:

```bash
uv run python -m compileall -q src/hcl_poc/privileged_z.py src/hcl_poc/cli.py
uv run pytest -q
```

Result: `22 passed`.

### Collection

Collected three `n128` seed windows with `--max-base-mse 5.0`,
`--max-action-delta-l2 0.25`, and `--oracle-gate-max-degradation-mse 0.0`.

| seed-start | selected branches | rows | selected fraction | selected base MSE mean / median / p90 / max | selected improvement MSE mean / median / p90 / max | selected action delta L2 mean / max | selected oracle delta MSE mean / median / p90 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9900000 | 191 | 1910 | 0.1492 | 0.6396 / 0.1265 / 2.3904 / 4.8812 | 0.4464 / 0.0417 / 2.2834 / 4.7033 | 0.1496 / 0.2494 | -7.7955 / -0.4565 / -0.0073 |
| 10100000 | 177 | 1770 | 0.1383 | 0.3952 / 0.1023 / 1.3042 / 4.2287 | 0.2644 / 0.0463 / 0.6073 / 3.8716 | 0.1423 / 0.2467 | -1.2499 / -0.2182 / -0.0092 |
| 10200000 | 168 | 1680 | 0.1313 | 0.6120 / 0.1452 / 2.5407 / 4.5555 | 0.4143 / 0.0548 / 2.2833 / 4.3340 | 0.1448 / 0.2481 | -1.5714 / -0.2263 / -0.0098 |
| merged | 536 | 5360 | 0.1396 | 0.5502 / 0.1191 / 2.1862 / 4.8812 | 0.3762 / 0.0464 / 1.5455 / 4.7033 | 0.1457 / 0.2494 | -3.6831 / -0.2579 / -0.0088 |

Merged artifact:

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025.npz`

The cap removed the severe selected-label tail from the previous merged bank
while keeping about `73%` of the selected branches (`536 / 735`).

### Run

Final-layer replay distillation used the capped merged bank with
`--improve-npz-weight 0.5`, matching the unfiltered multi-window effective
weighting choice.

Checkpoint:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

### Results

Dev seed window, 200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| frozen base | hierarchy | 0.395 | 37.02 | 0.000 |
| oracle-gated search imp1 small bank | hierarchy | 0.445 | 39.66 | 0.0072 |
| oracle-gated search multi384 imp0.5 | hierarchy | 0.460 | 38.41 | 0.0073 |
| basecap5 delta0.25 multi384 imp0.5 | hierarchy | 0.465 | 39.97 | 0.0078 |
| frozen base | oracle_hierarchy | 0.635 | 45.28 | 0.000 |
| oracle-gated search imp1 small bank | oracle_hierarchy | 0.660 | 45.85 | 0.0114 |
| oracle-gated search multi384 imp0.5 | oracle_hierarchy | 0.655 | 45.18 | 0.0122 |
| basecap5 delta0.25 multi384 imp0.5 | oracle_hierarchy | 0.685 | 46.04 | 0.0114 |

Fresh seed windows, 500 episodes per row:

| seed-start | mode | frozen base success | small-bank imp1 success | unfiltered multi384 imp0.5 success | basecap5 delta0.25 multi384 success | capped delta vs base |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10000000 | hierarchy | 0.458 | 0.464 | 0.454 | 0.454 | -0.004 |
| 10100000 | hierarchy | 0.470 | 0.480 | 0.454 | 0.460 | -0.010 |
| 10200000 | hierarchy | 0.446 | 0.440 | 0.458 | 0.476 | +0.030 |
| 10000000 | oracle_hierarchy | 0.626 | 0.666 | 0.642 | 0.644 | +0.018 |
| 10100000 | oracle_hierarchy | 0.654 | 0.672 | 0.644 | 0.672 | +0.018 |
| 10200000 | oracle_hierarchy | 0.638 | 0.644 | 0.618 | 0.658 | +0.020 |

Aggregate over the three fresh windows:

| mode | frozen base success mean | small-bank imp1 success mean | unfiltered multi384 imp0.5 success mean | basecap5 delta0.25 multi384 success mean | capped delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| hierarchy | 0.4580 | 0.4613 | 0.4553 | 0.4633 | +0.0053 |
| oracle_hierarchy | 0.6393 | 0.6607 | 0.6347 | 0.6580 | +0.0187 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed9900000_c16_std005_min001_oraclegate0_basecap5_delta025.npz`
- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed10100000_c16_std005_min001_oraclegate0_basecap5_delta025.npz`
- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n128_seed10200000_c16_std005_min001_oraclegate0_basecap5_delta025.npz`
- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`

### Interpretation

Collector-side branch capping fixed the main failure mode of the naive merged
bank. The learned-high improvement is still small and seed-dependent, but the
fresh-window mean is now above both frozen base and the previous small-bank
diagnostic candidate. Oracle-high transfer is consistently positive on all
fresh windows, although the mean remains slightly below the prior small-bank
checkpoint.

### Decision

Promote the basecap5/delta0.25 multi-window checkpoint as the best learned-high
diagnostic candidate so far, but treat the margin as modest. The result supports
continuing with capped or weighted closed-loop action-search targets rather than
more uniform training on heavy-tailed selected branches. The next useful
experiment is a small weighting sweep on this capped bank, especially
`--improve-npz-weight 1.0`, before trying any PPO fine-tuning.

### Improve-Weight 1.0 Sweep

I trained the same capped-bank final-layer recipe with
`--improve-npz-weight 1.0`.

Checkpoint:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp1_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

Dev seed window, 200 episodes, `seed-start = 9900000`:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| basecap5 delta0.25 multi384 imp0.5 | hierarchy | 0.465 | 39.97 | 0.0078 |
| basecap5 delta0.25 multi384 imp1.0 | hierarchy | 0.475 | 39.27 | 0.0068 |
| basecap5 delta0.25 multi384 imp0.5 | oracle_hierarchy | 0.685 | 46.04 | 0.0114 |
| basecap5 delta0.25 multi384 imp1.0 | oracle_hierarchy | 0.630 | 45.69 | 0.0108 |

Artifacts:

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp1_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp1_preserve_npz1_final_layer_200eps.json`

Decision: reject `imp1.0` without fresh-window validation. The learned-high dev
score improves slightly, but oracle-high drops below the frozen oracle-high base
and far below the `imp0.5` checkpoint. This suggests the capped improve signal
still needs a moderate weight; pushing it harder overfits the learned-goal
closed-loop branch and damages oracle-goal robustness.

## 2026-06-25 - Direct PPO From Capped Supervised Initialization

### Hypothesis

The capped supervised distillation checkpoint improved learned-high closed-loop
success modestly, but the earlier privileged/TCP PPO runs all started from the
older frozen BC low-level. This run tests whether paired PPO can improve when
initialized from the stronger capped supervised low-level.

### Implementation

Added `--direct-init-checkpoint` to `train-privileged-z-direct`. The command
still uses the base privileged-z checkpoint for normalizers, high-level model,
and compatibility checks, but loads the direct actor-critic state from the
provided tuned checkpoint before PPO.

A one-update smoke run verified that the direct init path was actually recorded
in the PPO recipe. An initial full run accidentally omitted the pass-through in
CLI dispatch and reproduced the old base-initialized condition; that run was
overwritten after fixing the dispatch and confirming the recipe.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-direct \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --direct-init-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5 \
  --run-tag hcl_next_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m \
  --seed 0 \
  --steps 1024000 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --learning-rate 3e-5 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --checkpoint-every-updates 5 \
  --train-scope final_layer \
  --force
```

### Setup

- representation: 31D privileged Push-T observation state
- direct init checkpoint: capped base-MSE/delta multi-window supervised
  distillation, `imp0.5`
- reward mode: paired terminal improvement
- dense progress weight: 0
- train scope: final low-policy layer plus log std and critic
- num_envs: 4096
- rollout steps: 10
- total transitions: 1,024,000
- fixed local eval bank: `data/manifests/local_reset_bank_n1800_seed0_k10.json`
- result level: dev

### Results

Training history final row:

- mean paired improvement: `0.0160`
- fraction improved: `0.4070`
- mean action delta L2: `0.0297`
- action saturation rate: `0.1252`

Fixed-bank local paired eval:

| checkpoint | mean paired improvement | median paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE median | tuned terminal MSE p90 | action delta L2 mean | saturation mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capped supervised init | 0.3428 | -0.00000293 | 0.4912 | 0.8918 | 0.00189 | 0.05585 | 0.0147 | 0.00066 |
| PPO step 204800 | 0.0476 | -0.00002582 | 0.4512 | 0.8933 | 0.00193 | 0.05462 | 0.0158 | 0.00067 |
| PPO step 409600 | 0.0533 | -0.00000870 | 0.4832 | 0.8972 | 0.00190 | 0.05150 | 0.0176 | 0.00059 |
| PPO step 614400 | 0.3637 | -0.00001065 | 0.4734 | 0.8943 | 0.00191 | 0.05289 | 0.0174 | 0.00053 |
| PPO step 819200 | 0.0577 | -0.00001702 | 0.4700 | 0.8965 | 0.00192 | 0.05203 | 0.0184 | 0.00059 |
| PPO step 1024000 | 0.0746 | -0.00001960 | 0.4612 | 0.8994 | 0.00191 | 0.05048 | 0.0189 | 0.00059 |

### Artifacts

- `artifacts/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m/seed0/checkpoints/`
- `results/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m/seed0/history.json`
- `results/hcl_next_phase1/privileged_z_local_paired_basecap5_delta025_imp05_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m_step204800_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m_step409600_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m_step614400_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m_step819200_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_final_layer_n4096_1m_latest_k10_4096.json`

### Interpretation

Starting PPO from the capped supervised checkpoint did not pass the local paired
gate. PPO slightly improves epsilon success and p90 terminal MSE at some
checkpoints, but median paired improvement remains negative and
`fraction_improved` stays below both `0.5` and the plan's `>0.55` pass
criterion. The best epsilon checkpoint is latest (`0.8994`), but it improves
only `46.1%` of local rollouts and is therefore still an outlier/threshold
tradeoff rather than broad reachability improvement.

### Decision

Do not run closed-loop task evaluation for this PPO branch. The capped
supervised checkpoint remains the best learned-high diagnostic candidate, and
paired PPO from that initialization is not a reliable local improvement method.
The next useful step is not another PPO hyperparameter sweep; it is to change
the local objective/data weighting, for example train on harder selected local
states or use stratified/quantile-weighted paired rewards so PPO does not mostly
optimize rare outlier improvements.

## 2026-06-25 - Hard-Start Direct PPO From Capped Supervised Initialization

### Hypothesis

The previous direct PPO run trained on the full local-start distribution and
did not improve most rollouts. This run tests whether PPO is useful if the
paired terminal-improvement objective is restricted to starts where the frozen
base low-level already fails, using `base_terminal_mse >= 0.05`.

### Implementation

Added `--min-base-terminal-mse` to `train-privileged-z-direct`.

For `reward-mode paired`, the trainer now computes the frozen base terminal MSE
at reset and marks only starts above the threshold as active. Inactive samples
remain in the vectorized environment rollout, but their rewards are zeroed and
they are excluded from PPO advantage normalization, minibatch updates, and
reported rollout metrics. The PPO recipe records the threshold and every
history row records `active_fraction`.

A one-update smoke run verified the direct init and hard-start mask:

- `active_fraction`: `0.1143`
- `mean_base_terminal_distance`: `2.3782`
- `mean_paired_improvement`: `0.2643`
- `fraction_improved`: `0.5769`
- `success_fraction_seen`: `0.2870`
- `action_saturation_rate`: `0.00363`

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-direct \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --direct-init-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp05_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --init-dataset data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5 \
  --run-tag hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m \
  --seed 0 \
  --steps 1024000 \
  --reward-mode paired \
  --terminal-weight 1.0 \
  --learning-rate 3e-5 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --checkpoint-every-updates 5 \
  --train-scope final_layer \
  --min-base-terminal-mse 0.05 \
  --force
```

### Setup

- direct init checkpoint: capped base-MSE/delta multi-window supervised
  distillation, `imp0.5`
- reward mode: paired terminal improvement
- active threshold: frozen base terminal MSE `>= 0.05`
- train scope: final low-policy layer plus log std and critic
- num_envs: 4096
- rollout steps: 10
- total transitions: 1,024,000
- full fixed local eval bank: `data/manifests/local_reset_bank_n1800_seed0_k10.json`
- hard fixed local eval bank: `data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json`
- result level: dev

### Training Result

Final history row:

- `active_fraction`: `0.1345`
- `mean_base_terminal_distance`: `2.1086`
- `mean_terminal_distance`: `2.1029`
- `mean_paired_improvement`: `0.00570`
- `fraction_improved`: `0.5771`
- `success_fraction_seen`: `0.2539`
- `mean_action_delta_l2`: `0.0326`
- `action_saturation_rate`: `0.00653`

### Full Fixed-Bank Local Paired Eval

Base full-bank epsilon success is `0.8943`.

| checkpoint | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO step 204800 | 0.4097 | 0.0954 | 0.4421 | 0.8933 | 0.0549 |
| PPO step 409600 | 0.4366 | 0.0685 | 0.4084 | 0.8933 | 0.0546 |
| PPO step 614400 | 0.4175 | 0.0876 | 0.4614 | 0.8894 | 0.0564 |
| PPO step 819200 | 0.4337 | 0.0713 | 0.4473 | 0.8972 | 0.0520 |
| PPO step 1024000 | 0.4230 | 0.0821 | 0.4417 | 0.8926 | 0.0548 |

### Hard Fixed-Bank Local Paired Eval

Base hard-bank epsilon success is `0.0`; base hard-bank terminal MSE mean is
`4.7270`.

| checkpoint | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| capped supervised init | 1.0231 | 3.7039 | 0.5820 | 0.1016 | 0.5898 |
| PPO step 204800 | 3.4438 | 1.2833 | 0.5820 | 0.1270 | 0.5968 |
| PPO step 409600 | 4.0454 | 0.6817 | 0.5497 | 0.1316 | 0.5466 |
| PPO step 614400 | 3.8797 | 0.8474 | 0.5635 | 0.1109 | 0.5733 |
| PPO step 819200 | 4.0431 | 0.6840 | 0.5635 | 0.1594 | 0.6565 |
| PPO step 1024000 | 3.9235 | 0.8036 | 0.5566 | 0.1201 | 0.5762 |

### Artifacts

- `artifacts/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m/seed0/checkpoints/`
- `results/incremental/privileged_z_direct/hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m/seed0/history.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000204800_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000409600_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000614400_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000819200_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_001024000_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_latest_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_supervised_basecap5_delta025_imp05_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000204800_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000409600_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000614400_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_000819200_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_step_001024000_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_local_paired_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m_latest_hard_k10.json`

### Interpretation

Hard-start PPO changed the hard-start threshold behavior, but it did not produce
a better controller. On the full fixed bank, every checkpoint remains below the
previous local improvement bar, with `fraction_improved <= 0.4614`, and success
is flat or slightly worse except for one threshold-tradeoff checkpoint.

On the hard fixed bank, PPO increases hard epsilon success at some checkpoints
up to `0.1594`, but loses most of the supervised checkpoint's tail improvement:
the capped supervised init reduces hard mean terminal MSE to `1.0231`, while
PPO checkpoints stay in the `3.44` to `4.05` range. The result is again a
threshold/outlier tradeoff rather than broad local reachability improvement.

### Decision

Do not run closed-loop task evaluation for this hard-start PPO branch. Keep the
capped supervised `imp0.5` checkpoint as the best learned-high diagnostic
candidate. The next PPO-style attempt should change the reward or sampling
objective more directly, for example stratified hard/easy minibatches with an
explicit preservation term, quantile-weighted paired improvement, or supervised
refresh against capped action-search labels before any policy-gradient updates.

## 2026-06-25 - Capped Action-Search Improve-Weight 0.25 Sweep

### Hypothesis

The `imp1.0` capped-bank distillation damaged oracle-goal robustness, while
`imp0.5` gave a modest but real fresh-window improvement. A lighter
`--improve-npz-weight 0.25` might preserve oracle behavior better while still
using the capped action-search labels enough to help the learned-high policy.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.25 \
  --run-tag hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Setup

- base checkpoint: `artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- improve bank: capped base-MSE/delta multi-window action-search bank
- improve weight: `0.25`
- preserve weights: replay hard `0.25`, easy base `1.0`, closed-loop preserve NPZ `1.0`
- train scope: final low-policy layer
- dev closed-loop window: 200 episodes, `seed-start = 9900000`
- fresh closed-loop windows: 3 x 500 episodes, `seed-start in {10000000,10100000,10200000}`

### Local Eval

Full fixed local bank:

| checkpoint | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| imp0.25 | 0.1719 | 0.3331 | 0.4631 | 0.8928 | 0.0558 |

Hard fixed local bank:

| checkpoint | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| imp0.25 | 1.1170 | 3.6100 | 0.5797 | 0.1039 | 0.5957 |
| imp0.5 reference | 1.0231 | 3.7039 | 0.5820 | 0.1016 | 0.5898 |

The local paired metrics are slightly worse than `imp0.5`; this checkpoint would
not have been selected from local eval alone.

### Closed-Loop Dev Eval

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| imp0.25 | hierarchy | 0.545 | 41.16 | 0.0068 |
| imp0.25 | oracle_hierarchy | 0.695 | 47.11 | 0.0120 |
| imp0.5 reference | hierarchy | 0.465 | 39.97 | 0.0078 |
| imp0.5 reference | oracle_hierarchy | 0.685 | 46.04 | 0.0114 |
| imp1.0 reference | hierarchy | 0.475 | 39.27 | 0.0068 |
| imp1.0 reference | oracle_hierarchy | 0.630 | 45.69 | 0.0108 |

### Fresh-Window Eval

| checkpoint | mode | seed 10000000 | seed 10100000 | seed 10200000 | mean |
| --- | --- | ---: | ---: | ---: | ---: |
| imp0.25 | hierarchy | 0.574 | 0.574 | 0.546 | 0.5647 |
| imp0.25 | oracle_hierarchy | 0.726 | 0.718 | 0.712 | 0.7187 |
| imp0.5 reference | hierarchy | 0.454 | 0.460 | 0.476 | 0.4633 |
| imp0.5 reference | oracle_hierarchy | 0.644 | 0.672 | 0.658 | 0.6580 |

### Artifacts

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_basecap5_delta025_imp025_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_basecap5_delta025_imp025_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`

### Interpretation

`imp0.25` is the strongest learned-high checkpoint so far. It improves fresh
hierarchy success by about 10 percentage points over `imp0.5` and also improves
oracle-goal success by about 6 percentage points. This is notable because the
fixed local paired metrics were not better than `imp0.5`; for this supervised
branch-bank distillation family, closed-loop validation is necessary because
the local fixed bank overweights the same hard-tail behavior and misses how the
policy interacts with learned high-level goals.

### Decision

Promote `imp0.25` as the current best privileged-state learned-high diagnostic
candidate. Do not continue PPO from this checkpoint yet; the PPO results still
show the local paired objective can destroy tail behavior. The next useful
experiment is either a small supervised neighborhood around this weight
(`0.125`, `0.375`) or a better branch-bank objective that selects labels by
closed-loop learned-high benefit rather than only local paired improvement.

## 2026-06-25 - Capped Action-Search Improve-Weight Neighborhood

### Hypothesis

After `imp0.25` became the best fresh validated checkpoint, test nearby weights
to see whether the optimum is below or above `0.25`.

### Setup

Same recipe and data as the `imp0.25` sweep, changing only
`--improve-npz-weight`:

- `imp0.125`
- `imp0.375`

`imp0.125` was evaluated only on the 200-episode dev window because it did not
clearly beat `imp0.25`. `imp0.375` beat `imp0.25` on the dev window, so it also
received three fresh 500-episode hierarchy and oracle windows.

### Dev Results

| checkpoint | hierarchy success | oracle success | hierarchy return | oracle return |
| --- | ---: | ---: | ---: | ---: |
| imp0.125 | 0.540 | 0.700 | 40.32 | 47.53 |
| imp0.25 | 0.545 | 0.695 | 41.16 | 47.11 |
| imp0.375 | 0.565 | 0.715 | 41.86 | 47.66 |

### Fresh Results

| checkpoint | mode | seed 10000000 | seed 10100000 | seed 10200000 | mean |
| --- | --- | ---: | ---: | ---: | ---: |
| imp0.25 | hierarchy | 0.574 | 0.574 | 0.546 | 0.5647 |
| imp0.25 | oracle_hierarchy | 0.726 | 0.718 | 0.712 | 0.7187 |
| imp0.375 | hierarchy | 0.582 | 0.578 | 0.522 | 0.5607 |
| imp0.375 | oracle_hierarchy | 0.722 | 0.720 | 0.706 | 0.7160 |

### Local Metrics For `imp0.375`

Full fixed local bank:

| checkpoint | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| imp0.375 | 0.1665 | 0.3386 | 0.4788 | 0.8916 | 0.0564 |

Hard fixed local bank:

| checkpoint | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| imp0.375 | 1.0879 | 3.6392 | 0.5681 | 0.0970 | 0.6126 |

### Artifacts

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0125_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0125_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0125_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_local_paired_basecap5_delta025_imp0375_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_basecap5_delta025_imp0375_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_fresh_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_fresh_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp0375_preserve_npz1_final_layer_fresh_seed10200000_500eps.json`

### Interpretation

The useful weight range is around `0.25` to `0.375`. `imp0.375` looked better
on the 200-episode dev window but did not beat `imp0.25` after fresh-window
validation. The two are close enough that the difference is probably within
seed noise, but `imp0.25` currently has the best fresh mean for both learned
high-level and oracle-goal hierarchy.

### Decision

Keep `imp0.25` as the current best checkpoint. Treat `imp0.375` as a close
alternate, not a replacement. The next useful step is no longer a scalar
improve-weight sweep; the bottleneck is label selection/objective design. The
best next branch-bank experiment is to collect or weight action-search labels
by observed closed-loop learned-high benefit, then train the same final-layer
distillation recipe against that filtered target set.

## 2026-06-25 - Fail-To-Success Action-Search Label Filter

### Hypothesis

The broad capped action-search bank improves closed-loop behavior even though
the fixed local paired gate is weak. A stricter target set may work better if it
keeps only branches where action search converts a local failure into a local
success:

```text
selected_base_mse >= 0.05
selected_best_mse <= 0.05
selected_action_delta_l2 <= 0.25
selected_oracle_delta_mse <= 0
```

This tests whether the useful labels are specifically failure-to-success local
corrections rather than all locally improving branches.

### Implementation

Added a reusable bank-filter command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  filter-privileged-z-action-search-bank \
  --input data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025.npz \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025_fail2success_eps005.npz \
  --min-base-mse 0.05 \
  --max-best-mse 0.05 \
  --max-action-delta-l2 0.25 \
  --max-oracle-delta-mse 0.0 \
  --force
```

The filter selected 163 of 536 branches, producing 1630 horizon rows.

Filtered branch summaries:

| metric | mean | median | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected base MSE | 0.6908 | 0.1091 | 2.8661 | 0.0506 | 4.7212 |
| selected best MSE | 0.0220 | 0.0190 | 0.0415 | 0.0025 | 0.0499 |
| selected improvement MSE | 0.6687 | 0.0860 | 2.8374 | 0.0112 | 4.7033 |
| selected action delta L2 | 0.1567 | 0.1523 | 0.2214 | 0.0648 | 0.2494 |
| selected oracle delta MSE | -0.8757 | -0.0857 | -0.0065 | -10.3126 | -0.0004 |

### Command

The filtered bank has fewer improve rows, so I used `--improve-npz-weight 1.0`
to keep the total improve-label influence in the same rough range as the
successful full-bank `imp0.25` run.

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025_fail2success_eps005.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 1.0 \
  --run-tag hcl_next_closedloop_search_fail2success_basecap5_delta025_imp1_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Local eval:

| bank | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full fixed local | 0.5117 | -0.0066 | 0.5042 | 0.8901 | 0.0566 |
| hard fixed local | 4.1892 | 0.5379 | 0.6189 | 0.1016 | 0.6056 |

Closed-loop dev eval:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| fail-to-success imp1 | hierarchy | 0.545 | 41.66 | 0.0070 |
| fail-to-success imp1 | oracle_hierarchy | 0.690 | 46.86 | 0.0116 |
| imp0.25 reference | hierarchy | 0.545 | 41.16 | 0.0068 |
| imp0.25 reference | oracle_hierarchy | 0.695 | 47.11 | 0.0120 |
| imp0.375 reference | hierarchy | 0.565 | 41.86 | 0.0071 |
| imp0.375 reference | oracle_hierarchy | 0.715 | 47.66 | 0.0116 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025_fail2success_eps005.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_fail2success_basecap5_delta025_imp1_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_fail2success_basecap5_delta025_imp1_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_fail2success_basecap5_delta025_imp1_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_fail2success_basecap5_delta025_imp1_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_fail2success_basecap5_delta025_imp1_preserve_npz1_final_layer_200eps.json`

### Interpretation

The strict local failure-to-success subset improves the hard-bank fraction
improved, but it worsens full-bank mean MSE and does not improve closed-loop dev
success. This suggests the closed-loop benefit of the broad capped bank is not
explained only by local threshold conversions. The filtered labels may be too
narrow and remove stabilizing or shaping examples that help the learned-high
closed-loop distribution.

### Decision

Reject this filtered bank without fresh-window validation. Keep `imp0.25` as
current best. The next label-selection experiment should not rely only on local
MSE threshold conversion; it should either collect branches under the learned
high-level closed-loop distribution with direct episode-outcome attribution, or
use a softer weighting over the broad bank rather than a hard success threshold.

## 2026-06-25 - Soft-Weighted Broad Action-Search Bank

### Hypothesis

The fail-to-success hard filter was too narrow: it improved hard local MSE but
removed many broad-bank labels that may stabilize the learned high-level
closed-loop distribution. A softer sample-weighting scheme over the full capped
action-search bank may keep the stabilizing examples while emphasizing states
where the base branch is hard and action search found a large local
improvement.

### Implementation

Added `reweight-privileged-z-action-search-bank`, which writes branch-level
`branch_sample_weights` and horizon row-level `sample_weights` into an NPZ
without dropping rows. The distillation trainer now reads optional
`sample_weights` from `--improve-npz` and multiplies them by the command-level
`--improve-npz-weight`.

The first weighting mode tested was:

```text
base_x_improvement = max(selected_base_mse / success_epsilon, 1)
                     * max(selected_improvement_mse / improvement_scale, 0)
```

with clipping to `[0.25, 4.0]` before mean normalization.

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  reweight-privileged-z-action-search-bank \
  --input data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025.npz \
  --output data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025_soft_base_x_improvement_w025_4_norm.npz \
  --mode base_x_improvement \
  --success-epsilon 0.05 \
  --improvement-scale 0.05 \
  --min-weight 0.25 \
  --max-weight 4.0 \
  --force
```

Weighted bank summary:

| field | value |
| --- | ---: |
| branches | 536 |
| horizon rows | 5360 |
| branch weight mean | 1.0000 |
| branch weight median | 0.7570 |
| branch weight p90 | 1.9767 |
| branch weight min | 0.1235 |
| branch weight max | 1.9767 |

The min is below `0.25` because normalization is applied after clipping.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025_soft_base_x_improvement_w025_4_norm.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.25 \
  --run-tag hcl_next_closedloop_search_soft_base_x_improvement_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Local eval:

| bank | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full fixed local | 0.1471 | 0.3579 | 0.4717 | 0.8916 | 0.0563 |
| hard fixed local | 0.8672 | 3.8599 | 0.5658 | 0.0901 | 0.5875 |

Closed-loop dev eval:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| soft base x improvement imp0.25 | hierarchy | 0.520 | 41.49 | 0.0076 |
| soft base x improvement imp0.25 | oracle_hierarchy | 0.700 | 46.71 | 0.0120 |
| imp0.25 reference | hierarchy | 0.545 | 41.16 | 0.0068 |
| imp0.25 reference | oracle_hierarchy | 0.695 | 47.11 | 0.0120 |
| imp0.375 reference | hierarchy | 0.565 | 41.86 | 0.0071 |
| imp0.375 reference | oracle_hierarchy | 0.715 | 47.66 | 0.0116 |

### Artifacts

- `data/manifests/privileged_z_closed_loop_action_search_hierarchy_n384_seed9900000_10100000_10200000_c16_std005_min001_oraclegate0_basecap5_delta025_soft_base_x_improvement_w025_4_norm.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_soft_base_x_improvement_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_soft_base_x_improvement_basecap5_delta025_imp025_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_soft_base_x_improvement_basecap5_delta025_imp025_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_soft_base_x_improvement_basecap5_delta025_imp025_preserve_npz1_final_layer_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_soft_base_x_improvement_basecap5_delta025_imp025_preserve_npz1_final_layer_200eps.json`

### Interpretation

The soft-weighted broad bank greatly reduces local catastrophic outliers,
especially on the hard subset, but it still lowers plain closed-loop hierarchy
success to `0.520`. The oracle score of `0.700` is not bad, but the learned
high-level path is worse than both the current best `imp0.25` and the close
`imp0.375` alternate.

This suggests the weighted labels improve local recovery in a way that does not
align with the high-level policy's closed-loop state distribution. The next
useful step is probably not a more aggressive local weighting rule. More
promising options are direct episode-outcome attribution for action-search
branches, or a separate selector/gate trained to decide when to apply the
distilled residual.

### Decision

Reject this soft-weighted candidate without fresh-window validation. Keep
`imp0.25` as the current best checkpoint.

## 2026-06-25 - Oracle Segment Gate Diagnostic

### Hypothesis

The action-search/distilled residual may be useful only on a subset of
high-level segments. If the main failure is over-applying the residual, then an
oracle segment gate should improve closed-loop success by applying the tuned
policy only when it locally beats the base policy for the held goal.

### Implementation

Added an explicit diagnostic mode to `eval-privileged-z`:

```text
--tuned-gate-mode local_oracle
```

At each high-level replan, the evaluator clones the current simulator state,
rolls out the base low-level policy and the tuned low-level policy for the next
held-goal horizon, and compares privileged normalized terminal MSE to the held
goal. The tuned policy is used for the real segment only when:

```text
tuned_terminal_mse <= base_terminal_mse + tuned_gate_max_degradation_mse
```

The default `tuned_gate_max_degradation_mse` is `0.0`, so this first diagnostic
uses a strict no-local-regression gate. This is an oracle diagnostic, not a
deployable selector.

### Commands

Current best learned-high gate:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --tuned-gate-mode local_oracle \
  --output results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_imp025_local_oracle_gate_200eps.json \
  --force
```

Soft-weight learned-high gate:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_soft_base_x_improvement_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --tuned-gate-mode local_oracle \
  --output results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_soft_base_x_improvement_local_oracle_gate_200eps.json \
  --force
```

Current best oracle-goal gate:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --mode oracle_hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --tuned-gate-mode local_oracle \
  --output results/hcl_next_phase1/privileged_z_closed_loop_oracle_imp025_local_oracle_gate_200eps.json \
  --force
```

### Results

Closed-loop dev eval:

| checkpoint | mode | gate | success | return | gate tuned fraction | gate mean paired improvement |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| imp0.25 reference | hierarchy | none | 0.545 | 41.16 | n/a | n/a |
| imp0.25 | hierarchy | local oracle | 0.495 | 39.73 | 0.3825 | -0.3358 |
| soft base x improvement reference | hierarchy | none | 0.520 | 41.49 | n/a | n/a |
| soft base x improvement | hierarchy | local oracle | 0.505 | 40.27 | 0.3845 | 0.4401 |
| imp0.25 reference | oracle_hierarchy | none | 0.695 | 47.11 | n/a | n/a |
| imp0.25 | oracle_hierarchy | local oracle | 0.705 | 47.15 | 0.4985 | 0.2409 |

### Artifacts

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_imp025_local_oracle_gate_smoke_20eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_imp025_local_oracle_gate_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_soft_base_x_improvement_local_oracle_gate_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_imp025_local_oracle_gate_200eps.json`

### Interpretation

The strict local oracle gate does not recover learned-high performance. It
hurts the current best candidate from `0.545` to `0.495`, and it also fails to
recover the weaker soft-weight candidate. This rules out a simple selector
whose target is only "does the tuned low level reduce privileged terminal MSE
to the held high-level goal?"

The oracle-goal version is different: the same gate slightly improves
oracle-goal success from `0.695` to `0.705`. That suggests the residual can be
locally useful when the high-level goal is well aligned, but local terminal MSE
to the learned high-level goal is not a sufficient selector for task success.

### Decision

Do not spend effort on a learned selector trained only from local held-goal MSE
labels. The next useful branch/outcome experiment should attach labels to
closed-loop task or segment outcome under the learned-high distribution, or it
should improve high-level goal validity/off-manifold handling before more
low-level residual training.

## 2026-06-25 - High-Level Goal Validity Diagnostic

### Hypothesis

The segment-gate diagnostic suggested that local privileged MSE to learned
high-level goals is not enough to predict task success. The next question is
whether learned high-level goals are invalid/off-manifold, or whether they are
valid-looking but semantically different from the oracle teacher continuation.

### Implementation

Added `eval-privileged-z-goal-validity`, a high-level goal diagnostic for the
privileged-state hierarchy. At each learned-high replan it:

- predicts the learned high-level goal from the current state and previous action;
- rolls a privileged PPO teacher from the same simulator state for `k=10` steps
  to produce an oracle continuation goal;
- compares learned goals to the oracle-goal manifold via nearest-neighbor
  distance over collected oracle goals;
- compares predicted-goal vs oracle-goal first low-level actions;
- rolls the frozen low-level policy to the predicted goal and to the oracle goal
  from the same state, then measures terminal MSE to both goals.

This implements the most immediately available parts of Experiment H without a
learned `D_phi` or validity discriminator.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-goal-validity \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_goal_validity_base_hierarchy_200eps.json \
  --force
```

### Results

Closed-loop learned-high success in the diagnostic run was `0.545`.

Goal validity/manifold metrics:

| metric | mean | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| predicted to matched oracle goal MSE | 0.7281 | 0.0630 | 1.9172 | 263.2875 |
| predicted to matched oracle goal L2 | 2.6978 | 1.3973 | 7.7093 | 90.3433 |
| current to predicted goal MSE | 1.2070 | 0.1713 | 1.8690 | 108.4413 |
| current to oracle goal MSE | 1.8514 | 0.4889 | 2.5807 | 265.2595 |
| predicted nearest oracle-goal MSE | 0.0808 | 0.0215 | 0.2002 | 9.1786 |
| oracle leave-one-out nearest MSE | 0.1159 | 0.0223 | 0.1666 | 28.5462 |
| random nearest oracle-goal MSE | 0.9464 | 0.9243 | 1.2570 | 1.9662 |

Low-level behavior metrics:

| metric | mean | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| predicted vs oracle first action L2 | 0.1857 | 0.0447 | 0.7090 | 2.0539 |
| predicted-goal policy terminal MSE to predicted | 0.9994 | 0.0411 | 1.9409 | 109.4581 |
| predicted-goal policy terminal MSE to oracle | 1.0506 | 0.0264 | 0.7776 | 355.7493 |
| oracle-goal policy terminal MSE to oracle | 0.5696 | 0.0129 | 0.5074 | 264.5513 |
| oracle-goal policy terminal MSE to predicted | 0.5826 | 0.0534 | 1.8774 | 65.7567 |

### Delta-Scale Mitigation Probe

Because `current -> predicted` goals were much closer than `current -> oracle`
goals, I tested the simplest H2-style mitigation at eval time:

```text
g_scaled = z_current + scale * (g_pred - z_current)
```

with no retraining.

| high-goal delta scale | learned-high success | return |
| ---: | ---: | ---: |
| 1.00 reference | 0.545 | 41.58 |
| 0.75 | 0.185 | 29.10 |
| 1.25 | 0.275 | 30.81 |
| 1.50 | 0.135 | 21.14 |

Artifacts:

- `results/hcl_next_phase1/privileged_z_goal_validity_base_hierarchy_smoke_20eps.json`
- `results/hcl_next_phase1/privileged_z_goal_validity_base_hierarchy_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_high_delta_scale075_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_high_delta_scale125_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_high_delta_scale150_200eps.json`

### Interpretation

The learned high-level goals do not look like arbitrary off-manifold vectors:
their nearest-neighbor distance to collected oracle goals is close to the
oracle leave-one-out nearest-neighbor baseline and far better than random
normalized goals. However, the matched oracle continuation error is large, and
the learned goals are usually closer to the current state than oracle teacher
continuations.

Naively scaling the high-level delta is not a fix. Both extrapolating and
shrinking the learned delta collapse success, which means the learned high-level
output is calibrated to the low-level's training distribution in a more
structured way than a scalar distance-to-current.

### Decision

Do not pursue scalar high-goal delta scaling. The next high-level mitigation
should be a real prototype/nearest-neighbor goal selection experiment or
episode-outcome-attributed branch data under the learned-high distribution. The
diagnostic also argues against treating the issue as generic off-manifold
prediction; the problem is more likely semantic goal selection/alignment.

## 2026-06-25 - Prototype High-Level Goal Projection

### Hypothesis

Experiment H proposes a prototype/nearest-neighbor high-level variant. If
learned high-level predictions are slightly off the real future-state manifold,
then projecting them to a nearby real teacher-state prototype may improve
closed-loop success while keeping the high-level semantics mostly intact.

### Implementation

Added an eval-only prototype projection to `eval-privileged-z`:

```text
--high-goal-projection nearest_oracle_bank
```

Before the evaluation starts, the command collects a separate oracle-state bank
by running the privileged PPO teacher from `--high-goal-bank-seed-start`.
During learned-high replans, the evaluator replaces:

```text
g_pred -> nearest_neighbor(g_pred, oracle_state_bank)
```

in normalized privileged-state space. This is a diagnostic H4 prototype
projection, not a trained scorer over prototypes.

### Commands

Base hierarchy:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --high-goal-projection nearest_oracle_bank \
  --high-goal-bank-episodes 200 \
  --high-goal-bank-seed-start 9800000 \
  --high-goal-bank-num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_nearest_oracle_bank_200eps.json \
  --force
```

Current best residual:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_closedloop_search_improve_c16_oraclegate0_multi384_basecap5_delta025_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --high-goal-projection nearest_oracle_bank \
  --high-goal-bank-episodes 200 \
  --high-goal-bank-seed-start 9800000 \
  --high-goal-bank-num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_imp025_nearest_oracle_bank_200eps.json \
  --force
```

### Results

| checkpoint | high-goal projection | success | return | prototype bank size | predicted-to-prototype MSE median | predicted-to-prototype MSE p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base reference | none | 0.545 | 41.58 | n/a | n/a | n/a |
| base | nearest oracle bank | 0.300 | 32.70 | 2000 | 0.0388 | 0.3597 |
| imp0.25 reference | none | 0.545 | 41.16 | n/a | n/a | n/a |
| imp0.25 | nearest oracle bank | 0.350 | 35.17 | 2000 | 0.0404 | 0.3685 |

Artifacts:

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_nearest_oracle_bank_smoke_20eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_base_nearest_oracle_bank_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_imp025_nearest_oracle_bank_200eps.json`

### Interpretation

Nearest-neighbor prototype projection is strongly harmful, even though the
chosen prototypes are close to the predicted high-level goals by normalized MSE.
This reinforces the previous delta-scaling result: the low-level and high-level
appear calibrated to the continuous learned goal output, not merely to a nearby
valid future-state manifold point.

The failure also suggests that the "validity" problem is not solved by making
goals more obviously real. A nearby real teacher-state prototype can still
change the intended local control semantics enough to break the learned
hierarchy.

### Decision

Reject nearest-oracle-bank projection as an eval-time mitigation. The next
useful H4 variant would need a trained prototype scorer or high-level trained
directly on prototype IDs. For the current RL proof-of-concept, the more direct
next step is branch/outcome attribution under the learned-high distribution,
because local/prototype geometry keeps failing to predict task-level success.

## 2026-06-25 - Learned-High Branch Outcome Attribution

### Hypothesis

Previous experiments showed that local privileged terminal MSE is a poor
selector for task-level success. The next diagnostic directly checks the
missing label: when action search finds a locally better segment under the
learned-high goal, does executing that segment and then continuing the same
learned hierarchy improve final task outcome?

### Implementation

Added `eval-privileged-z-branch-outcomes`. For each sampled learned-high
replan batch it:

- predicts the learned high-level goal;
- performs the same random-noise local action search used by the action-search
  bank collector;
- rolls two counterfactual episodes from the same simulator state:
  - base segment actions, then continue the learned hierarchy;
  - searched best segment actions, then continue the learned hierarchy;
- records final success/return deltas and splits them by the old local-MSE
  selection rule.

This is a diagnostic command. It does not write a training bank yet.

### Commands

Smoke:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 20 \
  --seed-start 9900000 \
  --num-envs 20 \
  --random-candidates 8 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.25 \
  --max-branch-batches 2 \
  --max-rollout-steps 120 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_attribution_smoke_20eps_b2_c8.json \
  --force
```

Dev windows:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9900000 \
  --num-envs 100 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.25 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_attribution_100eps_b4_c16.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9910000 \
  --num-envs 100 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.25 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_attribution_seed9910000_100eps_b4_c16.json \
  --force
```

### Results

Dev window `seed_start=9900000`:

| branch set | count | base success | candidate success | success delta | base return | candidate return | mean return delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all searched | 400 | 0.5225 | 0.5400 | +0.0175 | 44.55 | 45.60 | +1.05 |
| locally selected | 216 | 0.5093 | 0.5046 | -0.0046 | 42.42 | 42.00 | -0.41 |
| locally rejected | 184 | 0.5380 | 0.5815 | +0.0435 | 47.05 | 49.82 | +2.77 |

Dev window `seed_start=9910000`:

| branch set | count | base success | candidate success | success delta | base return | candidate return | mean return delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all searched | 400 | 0.5375 | 0.5450 | +0.0075 | 44.45 | 45.70 | +1.25 |
| locally selected | 223 | 0.5202 | 0.5202 | +0.0000 | 40.26 | 41.91 | +1.65 |
| locally rejected | 177 | 0.5593 | 0.5763 | +0.0169 | 49.72 | 50.47 | +0.75 |

The smoke run was noisier but showed the same warning sign: the locally
selected subset had negative success delta.

Artifacts:

- `results/hcl_next_phase1/privileged_z_branch_outcome_attribution_smoke_20eps_b2_c8.json`
- `results/hcl_next_phase1/privileged_z_branch_outcome_attribution_100eps_b4_c16.json`
- `results/hcl_next_phase1/privileged_z_branch_outcome_attribution_seed9910000_100eps_b4_c16.json`

### Interpretation

The searched segments can improve final task outcome on average, but the
existing local-MSE selection rule does not identify the useful subset. In both
dev windows, the locally rejected set has the larger success improvement. The
task-outcome signal is sparse: most branches leave success unchanged, and the
mean gain comes from a small imbalance between helped and hurt episodes.

This explains why prior local-MSE filters and weights were unreliable. They
optimize a real local reachability metric, but that metric is not the right
label for deciding which branch should train or override the learned hierarchy.

### Decision

Do not train another bank using local MSE thresholds as the primary label. The
next useful implementation is an outcome-attributed branch bank that stores the
searched segment actions plus `success_delta`/`return_delta`, then trains or
filters from those outcome labels instead of local terminal MSE.

## 2026-06-25 - Outcome-Attributed Success-Delta Bank Distillation

### Hypothesis

The branch-outcome attribution diagnostic showed that final task outcome labels
are better aligned than local terminal MSE labels. A first strict training bank
should keep only branches where the searched segment changes the counterfactual
continuation from failure to success:

```text
success_delta >= 0.5
```

This tests whether a small but clean set of task-outcome-positive segment
actions transfers through the same final-layer distillation recipe.

### Implementation

Extended `eval-privileged-z-branch-outcomes` with optional bank writing:

```text
--bank-output ...
--bank-min-success-delta ...
--bank-min-return-delta ...
```

The bank is compatible with `train-privileged-z-local-replay-distill`:

- `conditions`, `actions`
- local action-search metrics
- `selected_base_success`, `selected_candidate_success`, `selected_success_delta`
- `selected_base_return`, `selected_candidate_return`, `selected_return_delta`
- row `sample_weights` from positive return delta

### Commands

Bank collection:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9900000 \
  --num-envs 100 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.25 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_success_delta_pos_seed9900000_100eps_b4_c16.npz \
  --bank-min-success-delta 0.5 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_attribution_100eps_b4_c16_with_success_bank.json \
  --force
```

The bank selected 32 branches and 320 horizon rows.

Training:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_success_delta_pos_seed9900000_100eps_b4_c16.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.25 \
  --run-tag hcl_next_branch_outcome_success_delta_pos_imp025_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Bank summary:

| metric | value |
| --- | ---: |
| branches | 32 |
| horizon rows | 320 |
| success delta mean/median | 1.0 / 1.0 |
| return delta mean/median | 30.43 / 31.77 |
| return delta min/max | 0.72 / 65.19 |

Local eval:

| bank | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full fixed local | 0.3750 | 0.1301 | 0.5195 | 0.8936 | 0.0545 |
| hard fixed local | 3.2795 | 1.4475 | 0.5797 | 0.1224 | 0.5619 |

Closed-loop dev eval:

| checkpoint | mode | success | return | mean residual norm |
| --- | --- | ---: | ---: | ---: |
| branch outcome success-delta imp0.25 | hierarchy | 0.495 | 40.95 | 0.0070 |
| branch outcome success-delta imp0.25 | oracle_hierarchy | 0.675 | 46.44 | 0.0115 |
| imp0.25 reference | hierarchy | 0.545 | 41.16 | 0.0068 |
| imp0.25 reference | oracle_hierarchy | 0.695 | 47.11 | 0.0120 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_success_delta_pos_seed9900000_100eps_b4_c16.npz`
- `results/hcl_next_phase1/privileged_z_branch_outcome_attribution_100eps_b4_c16_with_success_bank.json`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_success_delta_pos_imp025_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_branch_outcome_success_delta_pos_imp025_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_branch_outcome_success_delta_pos_imp025_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_success_delta_pos_imp025_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_success_delta_pos_imp025_200eps.json`

### Interpretation

The outcome-positive bank improves hard local reset metrics, but it hurts
closed-loop learned-high and oracle-goal success. This first bank is very small
and only contains success-flip branches from one seed window, so it likely
overfits rare rescue actions or lacks enough surrounding stabilizing examples.

The result does not invalidate outcome attribution; it invalidates this narrow
success-delta-only distillation recipe. The counterfactual diagnostic remains
useful because it showed all searched branches had a small positive average
task-outcome delta, while hard local filters did not align with that outcome.

### Decision

Reject this 32-branch success-delta bank as a replacement for `imp0.25`.
Next outcome-bank attempt should either:

- collect a broader multi-window bank and weight by `return_delta` rather than
  filtering only success flips; or
- use outcome labels to train a gate/selector over candidate branches instead
  of distilling the rare rescue actions directly into the low level.

## 2026-06-25 - Return-Positive Outcome Bank Distillation

### Hypothesis

The success-flip bank was too sparse. The branch outcome diagnostic showed that
many candidate branches improve return without changing binary success, so a
broader return-positive bank may provide useful low-level updates while keeping
the closed-loop preserve bank stable.

### Commands

Bank collection:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9900000 \
  --num-envs 100 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.25 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_100eps_b4_c16.npz \
  --bank-min-return-delta 5.0 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_attribution_100eps_b4_c16_with_return_ge5_bank.json \
  --force
```

Training:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_100eps_b4_c16.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Bank summary:

| metric | value |
| --- | ---: |
| branches | 79 |
| horizon rows | 790 |
| success delta mean/median | 0.342 / 0.000 |
| return delta mean/median | 23.21 / 18.63 |
| return delta min/max | 5.06 / 65.19 |

Local eval:

| bank | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full fixed local | 0.3619 | 0.1431 | 0.5090 | 0.8931 | 0.0543 |
| hard fixed local | 3.0671 | 1.6599 | 0.6051 | 0.1247 | 0.5485 |

Closed-loop matched 3x500 eval:

| checkpoint | mode | seeds | success mean | return mean |
| --- | --- | --- | ---: | ---: |
| return-ge5 imp0.1 | hierarchy | 10000000,10100000,10200000 | 0.5713 | 43.13 |
| return-ge5 imp0.1 | oracle_hierarchy | 10000000,10100000,10200000 | 0.7300 | 48.71 |
| previous imp0.25 best | hierarchy | 10000000,10100000,10200000 | 0.5647 | 42.40 |
| previous imp0.25 best | oracle_hierarchy | 10000000,10100000,10200000 | 0.7187 | 47.59 |

Per-window success:

| mode | seed 10000000 | seed 10100000 | seed 10200000 |
| --- | ---: | ---: | ---: |
| return-ge5 hierarchy | 0.562 | 0.592 | 0.560 |
| previous imp0.25 hierarchy | 0.574 | 0.574 | 0.546 |
| return-ge5 oracle_hierarchy | 0.738 | 0.738 | 0.714 |
| previous imp0.25 oracle_hierarchy | 0.726 | 0.718 | 0.712 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_100eps_b4_c16.npz`
- `results/hcl_next_phase1/privileged_z_branch_outcome_attribution_100eps_b4_c16_with_return_ge5_bank.json`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_branch_outcome_return_ge5_imp01_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_branch_outcome_return_ge5_imp01_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_imp01_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_imp01_seed10200000_500eps.json`

### Interpretation

This is the first outcome-bank variant that improves the matched 3x500
closed-loop aggregate over the previous `imp0.25` best. The margin is small for
learned-high hierarchy success, but oracle-goal success and mean return improve
more clearly, which suggests the low-level update is useful when the goal is
good and only weakly bottlenecked by the current high-level model.

The result supports return-attributed branch banks over strict success-flip
banks. It does not yet prove a robust recipe: the bank came from one 100-episode
window, and the hierarchy gain is modest.

### Decision

Promote
`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
as the current best low-level tuned checkpoint.

Next experiment should scale this exact recipe by collecting return-positive
outcome banks from multiple seed windows, then retrain with the same conservative
`--improve-npz-weight 0.1` before evaluating on the same matched 3x500 seeds.

## 2026-06-25 - Multi-Window Return-Positive Outcome Bank

### Hypothesis

The single-window return-positive bank improved the matched 3x500 aggregate.
Collecting the same bank from several non-final seed windows may reduce
overfitting and make the low-level improvement more robust.

### Setup

Collected two additional return-positive banks with the same branch search
settings:

- `seed_start=9910000`: 71 branches, 710 rows, return delta mean 27.82,
  success delta mean 0.310
- `seed_start=9920000`: 65 branches, 650 rows, return delta mean 24.84,
  success delta mean 0.354

Merged them with the previous `seed_start=9900000` bank:

| merged bank metric | value |
| --- | ---: |
| branches | 215 |
| horizon rows | 2150 |
| return delta mean | 25.23 |
| success delta mean | 0.335 |

Merged bank artifact:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_300eps_b4_c16.npz`

Training used the same recipe and weight as the single-window run:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_300eps_b4_c16.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi3_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Local eval:

| bank | tuned terminal MSE mean | mean paired improvement | fraction improved | tuned epsilon success | tuned terminal MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full fixed local | 0.1323 | 0.3728 | 0.5103 | 0.8938 | 0.0537 |
| hard fixed local | 0.9386 | 3.7885 | 0.6374 | 0.1201 | 0.6628 |

Closed-loop matched 3x500 eval:

| checkpoint | mode | success mean | return mean |
| --- | --- | ---: | ---: |
| return-ge5 multi3 imp0.1 | hierarchy | 0.5700 | 43.06 |
| return-ge5 multi3 imp0.1 | oracle_hierarchy | 0.7267 | 47.96 |
| return-ge5 single-window imp0.1 | hierarchy | 0.5713 | 43.13 |
| return-ge5 single-window imp0.1 | oracle_hierarchy | 0.7300 | 48.71 |
| previous imp0.25 best | hierarchy | 0.5647 | 42.40 |
| previous imp0.25 best | oracle_hierarchy | 0.7187 | 47.59 |

Per-window success:

| mode | seed 10000000 | seed 10100000 | seed 10200000 |
| --- | ---: | ---: | ---: |
| return-ge5 multi3 hierarchy | 0.568 | 0.582 | 0.560 |
| return-ge5 single hierarchy | 0.562 | 0.592 | 0.560 |
| previous imp0.25 hierarchy | 0.574 | 0.574 | 0.546 |
| return-ge5 multi3 oracle_hierarchy | 0.744 | 0.732 | 0.704 |
| return-ge5 single oracle_hierarchy | 0.738 | 0.738 | 0.714 |
| previous imp0.25 oracle_hierarchy | 0.726 | 0.718 | 0.712 |

Artifacts:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_local_paired_branch_outcome_return_ge5_multi3_imp01_distill_k10_4096.json`
- `results/hcl_next_phase1/privileged_z_local_paired_branch_outcome_return_ge5_multi3_imp01_distill_hard_k10.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp01_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_imp01_seed10200000_500eps.json`

### Interpretation

The multi-window bank still beats the previous `imp0.25` reference, so the
return-positive outcome-bank idea is reproducible. It does not beat the
single-window return-positive candidate, even though local hard-bank MSE
improves much more. This is another example where local reset MSE is only a weak
proxy for closed-loop task success.

The likely issue is not bank diversity but weighting/selection. The added
branches may include more return-positive local interventions that are not
beneficial under learned high-level goals, or the return-weighted rows may shift
the low level too much on some oracle-goal windows.

### Decision

Keep the single-window return-positive checkpoint as the current best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

Do not promote the multi-window checkpoint. The next useful direction is not
just "more branches"; it is a better branch selector or a weight sweep on the
merged bank, especially below `--improve-npz-weight 0.1`.

## 2026-06-25 - Multi-Window Return Bank Improve-Weight Sweep Below 0.1

### Hypothesis

The merged return-positive bank may contain useful but noisier branches. The
`0.1` improve weight may over-apply these branches, so lower weights might keep
the broader coverage while preserving the base policy better.

### Setup

Both variants used the same merged bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_300eps_b4_c16.npz`

Only `--improve-npz-weight` changed:

- `0.05`: `hcl_next_branch_outcome_return_ge5_multi3_imp005_preserve_npz1_final_layer_lr1e4_e200`
- `0.075`: `hcl_next_branch_outcome_return_ge5_multi3_imp0075_preserve_npz1_final_layer_lr1e4_e200`

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return |
| --- | --- | ---: | ---: |
| multi3 imp0.05 | hierarchy | 0.520 | 40.92 |
| multi3 imp0.05 | oracle_hierarchy | 0.685 | 44.72 |
| multi3 imp0.075 | hierarchy | 0.555 | 40.86 |
| multi3 imp0.075 | oracle_hierarchy | 0.680 | 45.07 |
| single-window imp0.1 current best | hierarchy | 0.540 | 41.29 |
| single-window imp0.1 current best | oracle_hierarchy | 0.702 | 46.89 |

The `0.05` variant failed the quick dev gate, so it was not promoted to matched
3x500 validation.

The `0.075` variant had a better learned-high dev result but weak oracle-goal
dev result, so it received hierarchy-only matched 3x500 validation:

| checkpoint | seed 10000000 | seed 10100000 | seed 10200000 | mean | return mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi3 imp0.075 hierarchy | 0.568 | 0.574 | 0.544 | 0.5620 | 42.77 |
| multi3 imp0.1 hierarchy | 0.568 | 0.582 | 0.560 | 0.5700 | 43.06 |
| single-window imp0.1 hierarchy | 0.562 | 0.592 | 0.560 | 0.5713 | 43.13 |

Artifacts:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_imp005_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_imp0075_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp005_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_imp005_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp0075_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_imp0075_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp0075_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp0075_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_imp0075_seed10200000_500eps.json`

### Interpretation

Lowering the merged-bank improve weight does not recover the single-window
result. The useful signal is not a simple scalar balance issue. More likely, the
merged bank includes branches whose positive return label does not transfer to
the learned-high closed-loop distribution, so the next step should change
selection/conditioning instead of continuing scalar sweeps.

### Decision

Reject merged-bank improve weights `0.05` and `0.075`. Keep the single-window
return-positive `imp0.1` checkpoint as current best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

Next useful direction: train or implement an outcome-aware selector/gate, or add
bank metadata that conditions selection on the learned-high goal distribution,
rather than applying all return-positive branches as uniform distillation data.

## 2026-06-25 - Top-Return Branch Selection Diagnostic

### Hypothesis

The merged bank may fail because it includes too many weakly positive branches.
Selecting only the strongest return-positive branches, while keeping the same
branch count as the successful single-window bank, may recover or improve the
closed-loop result.

### Setup

Created a filtered NPZ from the merged return-positive bank by selecting the top
79 branches by `selected_return_delta`, preserving full 10-step branch blocks.

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_top79_return.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 79 |
| horizon rows | 790 |
| return delta mean | 43.56 |
| return delta min | 31.71 |
| success delta mean | 0.570 |
| source seed counts | 9900000: 22, 9910000: 31, 9920000: 26 |

Training used `--improve-npz-weight 0.1`, matching the current best recipe:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_top79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return |
| --- | --- | ---: | ---: |
| multi3 top79 imp0.1 | hierarchy | 0.555 | 42.35 |
| multi3 top79 imp0.1 | oracle_hierarchy | 0.675 | 46.47 |
| single-window imp0.1 current best | hierarchy | 0.540 | 41.29 |
| single-window imp0.1 current best | oracle_hierarchy | 0.702 | 46.89 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_top79_return.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_top79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_top79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_top79_imp01_200eps.json`

### Interpretation

Top-return filtering improves the learned-high dev window compared with the
current best on that same seed window, but it hurts oracle-goal success. This is
similar to the `multi3 imp0.075` pattern, which failed matched 3x500 hierarchy
validation. Therefore branch return magnitude alone is not a sufficient
selector.

The useful signal appears to depend on compatibility with the learned-high
closed-loop goal distribution, not only task return improvement under the branch
rollout.

### Decision

Reject top-return filtering as a replacement. Keep the single-window
return-positive `imp0.1` checkpoint as current best.

Next selector should include state/goal-distribution context or learn an
explicit gate from branch outcome features rather than selecting only by return
delta.

## 2026-06-25 - Preserve-Goal Nearest Branch Selection Diagnostic

### Hypothesis

The failure of top-return filtering suggests return magnitude alone is not the
right selector. A branch may be useful only if its held goal lies near the
learned-high closed-loop goal distribution. Use the closed-loop preserve bank as
a proxy for that distribution and select return-positive branches whose held
goals are nearest to preserve-bank held goals.

### Setup

Reference distribution:

- `data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz`

Source bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_300eps_b4_c16.npz`

Selection:

- take the first row of each 10-step branch block;
- extract the normalized held-goal slice from the low-level condition;
- standardize by the preserve-bank held-goal distribution;
- select the 79 branches with smallest nearest-neighbor MSE to preserve-bank
  held goals;
- preserve full 10-step branch blocks.

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_goalnn79_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 79 |
| horizon rows | 790 |
| preserve-goal NN MSE mean/max | 0.0194 / 0.0369 |
| return delta mean | 27.00 |
| success delta mean | 0.481 |
| action delta mean | 0.195 |
| source seed counts | 9900000: 40, 9910000: 28, 9920000: 11 |

Training used the current-best distillation recipe:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_goalnn79_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi3_goalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return |
| --- | --- | ---: | ---: |
| goal-NN79 imp0.1 | hierarchy | 0.555 | 41.36 |
| goal-NN79 imp0.1 | oracle_hierarchy | 0.725 | 47.16 |
| single-window imp0.1 current best | hierarchy | 0.540 | 41.29 |
| single-window imp0.1 current best | oracle_hierarchy | 0.702 | 46.89 |

Matched 3x500 eval:

| checkpoint | mode | seed 10000000 | seed 10100000 | seed 10200000 | success mean | return mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| goal-NN79 imp0.1 | hierarchy | 0.580 | 0.594 | 0.536 | 0.5700 | 43.02 |
| goal-NN79 imp0.1 | oracle_hierarchy | 0.742 | 0.750 | 0.706 | 0.7327 | 48.55 |
| single-window imp0.1 current best | hierarchy | 0.562 | 0.592 | 0.560 | 0.5713 | 43.13 |
| single-window imp0.1 current best | oracle_hierarchy | 0.738 | 0.738 | 0.714 | 0.7300 | 48.71 |
| full multi3 imp0.1 | hierarchy | 0.568 | 0.582 | 0.560 | 0.5700 | 43.06 |
| full multi3 imp0.1 | oracle_hierarchy | 0.744 | 0.732 | 0.704 | 0.7267 | 47.96 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_goalnn79_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_goalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_goalnn79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_goalnn79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_goalnn79_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_goalnn79_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_goalnn79_imp01_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_goalnn79_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_goalnn79_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_goalnn79_imp01_seed10200000_500eps.json`

### Interpretation

Preserve-goal nearest selection is the first selector that improves oracle-goal
success beyond the single-window current best, but it does not improve learned
high-level hierarchy success. This is a useful split: the selected low-level
update is beneficial when the high-level goal is good, while learned-high
rollouts remain bottlenecked by the high-level goal distribution or by
state-dependent cases not captured by goal-only nearest-neighbor selection.

The result supports adding learned-high context to selection, but goal-only
context is not sufficient to promote a new overall best.

### Decision

Do not promote goal-NN79 as the current overall best because hierarchy success
is slightly lower than the single-window `imp0.1` checkpoint. Keep:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

as the current best learned-high checkpoint.

Track goal-NN79 as the best oracle-goal/low-level diagnostic. The next selector
should use richer context than goal alone, for example state+goal nearest
neighbors or a learned selector over branch features, and should optimize the
learned-high hierarchy aggregate rather than oracle-goal success alone.

## State+goal-nearest 79-branch multi-window return-positive bank

Goal-only nearest-neighbor selection improved the oracle-goal diagnostic but
did not improve learned-high hierarchy success. I therefore repeated the
same 79-branch subset size while matching the preserve distribution on the
first branch row's low-level state plus held-goal condition slice.

### Setup

Reference distribution:

- `data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz`

Source bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_300eps_b4_c16.npz`

Selection:

- take the first row of each 10-step branch block;
- extract the low-level condition state+held-goal slice `[0:62]`;
- standardize by the preserve-bank first-row state+goal distribution;
- select the 79 branches with smallest nearest-neighbor MSE to preserve-bank
  state+goal rows;
- preserve full 10-step branch blocks.

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn79_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 79 |
| horizon rows | 790 |
| preserve state+goal NN MSE mean/max | 0.0592 / 0.1082 |
| return delta mean | 27.47 |
| success delta mean | 0.443 |
| action delta mean | 0.192 |
| source seed counts | 9900000: 31, 9910000: 28, 9920000: 20 |

Training used the same current-best distillation recipe:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn79_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return |
| --- | --- | ---: | ---: |
| state+goal-NN79 imp0.1 | hierarchy | 0.575 | 41.89 |
| state+goal-NN79 imp0.1 | oracle_hierarchy | 0.705 | 46.69 |
| goal-NN79 imp0.1 | hierarchy | 0.555 | 41.36 |
| goal-NN79 imp0.1 | oracle_hierarchy | 0.725 | 47.16 |
| single-window imp0.1 current best | hierarchy | 0.540 | 41.29 |
| single-window imp0.1 current best | oracle_hierarchy | 0.702 | 46.89 |

Matched 3x500 eval:

| checkpoint | mode | seed 10000000 | seed 10100000 | seed 10200000 | success mean | return mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| state+goal-NN79 imp0.1 | hierarchy | 0.582 | 0.594 | 0.574 | 0.5833 | 43.27 |
| state+goal-NN79 imp0.1 | oracle_hierarchy | 0.708 | 0.738 | 0.684 | 0.7100 | 47.92 |
| goal-NN79 imp0.1 | hierarchy | 0.580 | 0.594 | 0.536 | 0.5700 | 43.02 |
| goal-NN79 imp0.1 | oracle_hierarchy | 0.742 | 0.750 | 0.706 | 0.7327 | 48.55 |
| single-window imp0.1 previous best | hierarchy | 0.562 | 0.592 | 0.560 | 0.5713 | 43.13 |
| single-window imp0.1 previous best | oracle_hierarchy | 0.738 | 0.738 | 0.714 | 0.7300 | 48.71 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn79_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_seed10200000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_seed10200000_500eps.json`

### Interpretation

Adding state context to the nearest-neighbor selector flips the tradeoff seen
with goal-only selection. The learned-high hierarchy improves from the prior
best `0.5713` to `0.5833`, while oracle-goal success drops from the goal-only
diagnostic `0.7327` to `0.7100`. This suggests that the selector is now
choosing updates that better match the learned high-level rollout
distribution, even though those updates are not globally better low-level
corrections under oracle goals.

The result is a real learned-high improvement, but it also reinforces that the
current distillation recipe is strongly selector-sensitive. A next useful step
is to train the same state+goal-selected bank with a small improve-weight sweep
or to add a learned branch selector that optimizes matched learned-high
success instead of nearest-neighbor similarity alone.

### Decision

Promote state+goal-NN79 as the new current best learned-high checkpoint:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## 2026-06-25 — VAE512 FiLM Goal-Use Diagnostic and Closed-Loop Check

### Hypothesis

Replacing concat goal conditioning with FiLM/gated conditioning may make the
low-level policy actually use the future latent goal, making it a better base
for reachability RL than the current VAE512 concat hierarchy.

### Commands

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  goal-diagnostics --n-demo 1800 --candidate vae512_b1e6_film \
  --samples 5000 --horizons 2,5,10 --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate vae512_b1e6_film --goal-source learned \
  --episodes 200 --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate vae512_b1e6_film --goal-source oracle \
  --episodes 200 --force
```

### Results

Offline goal-use diagnostics for FiLM at `N=1800`:

| metric | value |
| --- | ---: |
| action MAE k10 | 0.0455 |
| max same-state goal sensitivity | 0.1266 |
| goal-shuffle action L2 | 0.2783 |
| goal-shuffle MAE gap | 0.0883 |
| frame-shuffle action L2 | 0.8213 |
| previous-action shuffle L2 | 0.1129 |
| remaining shuffle L2 | 0.0012 |

Closed-loop 200-episode check:

| checkpoint | goal source | episodes | success |
| --- | --- | ---: | ---: |
| VAE512 FiLM | learned | 200 | 0.425 |
| VAE512 FiLM | oracle | 200 | 0.535 |
| VAE512 concat n1800 seed0 | learned | 500 | 0.582 |
| VAE512 concat n1800 seed0 | oracle | 50 | 0.500 |

Artifacts:

- `results/incremental/goal_diagnostics/n1800/seed0/vae512_b1e6_film/diagnostics.json`
- `results/incremental/learned_interface/vae512_b1e6_film/seed0/learned_hierarchy_eval_200.json`
- `results/incremental/learned_interface/vae512_b1e6_film/seed0/oracle_hierarchy_eval_200.json`

### Interpretation

FiLM fixes much of the offline goal-ignoring signal: goal-shuffle action L2
increases from the concat n1800 value of `0.0740` to `0.2783`, and same-state
goal sensitivity rises from `0.0308` to `0.1266`. However, the closed-loop
learned-high hierarchy is substantially worse than the concat n1800 baseline
(`0.425` vs `0.582` success), so the stronger goal dependence is not enough by
itself.

### Decision

Do not promote the existing `vae512_b1e6_film` checkpoint as the immediate RL
base. Treat FiLM/gated conditioning as a promising architecture direction, but
it needs either retraining in the matched VAE-scaling `N=500/N=1800` setup or a
different objective before using it for expensive PPO runs.

## 2026-06-25 — Effect32 FiLM Diagnostic Fix and Closed-Loop Check

### Hypothesis

A future-effect latent may be a better real-compatible representation than a
future-state VAE latent. Existing `effect32_film` artifacts should be checked
with the same goal-use gate and a less noisy closed-loop evaluation.

### Implementation Note

The generic goal diagnostic initially treated effect-code `goals[t+h]` like a
unary state latent. That is incorrect for horizons other than the fixed training
horizon because effect-code goals are pair encodings. The diagnostic builder now
encodes the actual pair `(frame_t, frame_{t+h})` for effect candidates before
constructing the low-level condition. A regression test covers this indexing.

### Commands

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  goal-diagnostics --n-demo 1800 --candidate effect32_film \
  --samples 5000 --horizons 2,5,10 --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film --goal-source learned \
  --episodes 200 --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film --goal-source oracle \
  --episodes 200 --force
```

### Results

Corrected offline goal-use diagnostics:

| metric | value |
| --- | ---: |
| action MAE k10 | 0.0425 |
| max same-state goal sensitivity | 0.0368 |
| goal-shuffle action L2 | 0.0622 |
| goal-shuffle MAE gap | 0.0102 |
| frame-shuffle action L2 | 0.9503 |
| previous-action shuffle L2 | 0.1326 |

Closed-loop evaluation:

| checkpoint | goal source | episodes | success | max reward | final reward |
| --- | --- | ---: | ---: | ---: | ---: |
| effect32 FiLM | learned | 200 | 0.655 | 0.752 | 0.741 |
| effect32 FiLM | oracle | 200 | 0.720 | 0.804 | 0.798 |

Artifacts:

- `results/incremental/goal_diagnostics/n1800/seed0/effect32_film/diagnostics.json`
- `results/incremental/learned_interface/effect32_film/seed0/learned_hierarchy_eval_200.json`
- `results/incremental/learned_interface/effect32_film/seed0/oracle_hierarchy_eval_200.json`

### Interpretation

`effect32_film` is currently the strongest real-compatible closed-loop
hierarchy found in this pass: learned-high success is `0.655`, above concat
VAE512 n1800 seed0 at `0.582`. Oracle-goal success is also high at `0.720`.

However, the low-level goal-use gate still looks weak: goal-shuffle action L2
is only `0.0622` while frame-shuffle action L2 is `0.9503`. This suggests the
effect representation improves the imitation/high-level stack but does not by
itself solve the low-level goal-conditioning problem that blocks local
reachability RL.

### Decision

Promote `effect32_film` as the best current supervised learned-interface
baseline and as the lead for Phase 4 representation work. Do not treat it as a
passed RL base yet; the next useful work is to combine effect/reachability
latents with a stronger goal-use objective or architecture, then rerun the
corrected diagnostics before PPO.

## 2026-06-25 — Existing Candidate Goal-Use Screen

### Hypothesis

Some already-trained learned-interface candidates may have a better
closed-loop/goal-use tradeoff than the main VAE512 concat, VAE512 FiLM, or
effect32 FiLM candidates.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  goal-diagnostics --n-demo 1800 --candidate <candidate> \
  --samples 5000 --horizons 2,5,10 --force
```

Candidates screened: `vae256_b1e5`, `jepa256_r01_v1_c01`, `effect32`,
`dae256_n010`, `ae256_film`.

### Results

| candidate | learned eval100 success | goal-shuffle L2 | max goal sensitivity | frame-shuffle L2 |
| --- | ---: | ---: | ---: | ---: |
| vae256_b1e5 | 0.650 | 0.0405 | 0.0205 | 0.9653 |
| jepa256_r01_v1_c01 | 0.650 | 0.0408 | 0.0246 | 0.9687 |
| effect32 | 0.620 | 0.0279 | 0.0178 | 0.9661 |
| dae256_n010 | 0.590 | 0.0772 | 0.0286 | 0.9488 |
| ae256_film | 0.550 | 0.2506 | 0.0937 | 0.8645 |
| effect32_film | 0.690 | 0.0622 | 0.0368 | 0.9503 |
| vae512_b1e6_film | n/a | 0.2783 | 0.1266 | 0.8213 |

### Interpretation

The current candidates expose the central tradeoff:

- stronger closed-loop imitation candidates (`effect32_film`, `vae256_b1e5`,
  `jepa256_r01_v1_c01`) still have weak low-level goal response;
- stronger goal-use candidates (`vae512_b1e6_film`, `ae256_film`) have weaker
  closed-loop task success.

### Decision

Do not spend PPO budget on this candidate set as-is. The next architecture or
objective should explicitly preserve task imitation while increasing goal
dependence, rather than choosing between those two failure modes.

## 2026-06-25 — Closed-Loop Shuffled-Goal Ablation

### Hypothesis

One-step action sensitivity may underestimate goal dependence. A stronger gate
is to run the hierarchy closed-loop while shuffling predicted high-level goals
across vectorized environments at each replan. If success remains high, the
policy is effectively goal-agnostic; if success collapses, the held goal matters
in closed loop.

### Implementation

Added `goal_source=shuffled` to `learned-interface-eval`. The evaluator still
uses the learned high-level prediction, but permutes selected goals among the
currently replanning environments before the low-level rollout. The result JSON
records `shuffled_goal_l2`.

### Commands

```bash
uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film --goal-source shuffled \
  --episodes 200 --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate vae512_b1e6_film --goal-source shuffled \
  --episodes 200 --force
```

### Results

| candidate | goal source | episodes | success | max reward | final reward | shuffled-goal L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | learned | 200 | 0.655 | 0.752 | 0.741 | n/a |
| effect32_film | oracle | 200 | 0.720 | 0.804 | 0.798 | n/a |
| effect32_film | shuffled | 200 | 0.275 | 0.462 | 0.436 | 6.109 |
| vae512_b1e6_film | learned | 200 | 0.425 | 0.592 | 0.573 | n/a |
| vae512_b1e6_film | oracle | 200 | 0.535 | 0.672 | 0.665 | n/a |
| vae512_b1e6_film | shuffled | 200 | 0.010 | 0.205 | 0.140 | 24.805 |

Artifacts:

- `results/incremental/learned_interface/effect32_film/seed0/shuffled_hierarchy_eval_200.json`
- `results/incremental/learned_interface/vae512_b1e6_film/seed0/shuffled_hierarchy_eval_200.json`

### Interpretation

Both FiLM policies are closed-loop goal-dependent despite weak one-step action
sensitivity for `effect32_film`. Shuffling goals causes a large success drop:

- `effect32_film`: `0.655 -> 0.275`
- `vae512_b1e6_film`: `0.425 -> 0.010`

This means the previous action-sensitivity gate was too strict as a standalone
filter. The better gate is: closed-loop learned/oracle/shuffled evaluation plus
offline action sensitivity. Under that gate, `effect32_film` is currently the
best learned-interface base: it has strong learned/oracle success and a large
shuffled-goal penalty.

### Decision

Promote `effect32_film` from "supervised baseline only" to the leading
real-compatible candidate for the next reachability/RL experiment. Keep the
one-step diagnostic as a warning signal, not a blocker, when closed-loop
shuffled-goal ablation shows the policy actually depends on goals.

## 2026-06-25 — Effect32 FiLM Reachability Distance

### Hypothesis

The effect-code representation needs its own reachability-distance data path.
Unlike VAE/AE/JEPA latents, effect codes are pairwise future-effect embeddings,
not unary state embeddings. A useful `D_phi` should compare achieved progress
effect `(anchor -> current)` to requested target effect `(anchor -> future)`.

### Implementation

Extended `src/hcl_poc/reachability.py` so effect-code candidates build anchored
effect-progress trajectories. For each anchor frame, the cache stores a short
sequence of effects from the same anchor to later frames. The existing
`ReachabilityDistance(start, goal)` model then trains on progress pairs within
that anchored sequence.

Cache:

`artifacts/incremental/reachability_distance/effect32_film/seed0/effect_progress_h10_stride2_span41.pt`

### Commands

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-reachability-distance --candidate effect32_film \
  --horizon-steps 10 --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-reachability-distance --candidate effect32_film \
  --samples 8192 --force
```

### Results

| checkpoint | temporal MSE | temporal Spearman | near/far acc | shuffled AUC | demo decrease |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film D_phi | 0.03267 | 0.8333 | 0.9275 | 0.9074 | 0.7396 |
| VAE512 global D_phi | 0.00777 | 0.9328 | 0.9885 | 0.8745 | 0.7048 |
| VAE512 n1000 D_phi | 0.01125 | 0.9248 | 0.9866 | 0.8690 | 0.6824 |

Artifacts:

- `artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt`
- `artifacts/incremental/reachability_distance/effect32_film/seed0/metrics.json`
- `results/incremental/reachability_distance/effect32_film/seed0/eval.json`

### Interpretation

Effect-progress `D_phi` is less clean as a temporal-distance regressor than the
VAE512 state-latent `D_phi`, but it separates shuffled targets better and has
better demo-decrease accuracy. This fits the representation tradeoff: effect
codes are less like a smooth state coordinate system, but may better encode
task-relevant controllable changes.

### Decision

Keep `effect32_film` plus anchored effect-progress `D_phi` as the lead
real-compatible path for the next local reachability RL implementation. The
next code change should adapt local RL reward computation so the achieved
effect is encoded from the episode anchor to the current observation and scored
against the held target effect.

## 2026-06-25 — Effect32 FiLM Local RL Smoke with D_phi Reward

### Hypothesis

If the local RL wrapper scores achieved effect progress with the anchored
effect-progress `D_phi`, residual PPO should at least run end-to-end and show a
directional local-distance improvement over the frozen `effect32_film` policy.

### Implementation

Extended `src/hcl_poc/low_level_rl.py` so non-VAE learned-interface candidates
can be loaded by name. For effect-code candidates, each held-goal segment now
stores the anchor frame at replan time and encodes the achieved effect
`(anchor_frame -> current_frame)` at every step. This achieved effect is scored
against the held target effect with either raw effect MSE or the learned
`D_phi`.

Also added `--candidate`, `--distance-metric`, and `--reachability-checkpoint`
support to the low-level eval path so frozen baselines can be evaluated under
the same selected metric as tuned checkpoints.

### Commands

Frozen D_phi baseline:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen \
  --episodes 100 --seed-start 3400000 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

R1 residual PPO smoke:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r1 \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r1_smoke_20k \
  --steps 20000 --alpha 0.05 --terminal-weight 1.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r1_smoke_20k \
  --episodes 100 --seed-start 3400000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_smoke_20k/latest.pt \
  --force
```

### Results

100-episode eval on seed bank `3400000`:

| policy | success | max reward | D_phi reduction | raw effect reduction | reach rate | terminal AUC | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.600 | 0.711 | 0.0766 | 0.3532 | 0.734 | 0.801 | 0.0000 |
| R1 smoke 20k | 0.650 | 0.749 | 0.0957 | 0.4038 | 0.710 | 0.805 | 0.0110 |

Training latest row:

| metric | value |
| --- | ---: |
| global step | 20160 |
| mean reward | -0.0546 |
| mean latent distance | 0.6755 |
| mean terminal distance | 0.7106 |
| mean residual L2 | 0.0136 |
| clip fraction | 0.542 |
| action saturation rate | 0.000 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_smoke_20k/latest.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen/eval_100_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_smoke_20k/eval_100_seed3400000.json`

### Interpretation

The effect-code local RL path is now technically viable: it trains, evaluates,
and produces nonzero residuals without action saturation. The 20k-step smoke
improves task success and both selected/raw local distance reductions on the
100-episode bank, but the training reward is still negative and the PPO clip
fraction is high. This is a smoke result only.

### Decision

Promote this path to a serious dev run candidate, but tune PPO before spending a
large budget. The next run should use many envs, lower effective update
aggressiveness, and a longer step budget, then compare against the frozen
baseline on 300-500 episodes.

## 2026-06-25 — Effect32 FiLM 4096-Env D_phi R1 Scale Smoke

### Hypothesis

The effect-code `D_phi` residual PPO path should run at the plan-required
4096-env scale. A one-rollout run will not prove RL improvement, but it should
reveal whether the implementation is stable and whether PPO updates produce
meaningful residuals without the high clip fraction seen in the 32-env smoke.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r1 \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r1_4096_smoke_40k \
  --steps 40960 --num-envs 4096 --rollout-steps 10 \
  --num-minibatches 8 --update-epochs 2 --learning-rate 3e-5 \
  --alpha 0.05 --terminal-weight 1.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Results

100-episode eval on seed bank `3400000`:

| policy | success | max reward | D_phi reduction | raw effect reduction | reach rate | terminal AUC | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.600 | 0.711 | 0.0766 | 0.3532 | 0.734 | 0.801 | 0.0000 |
| R1 32-env smoke 20k | 0.650 | 0.749 | 0.0957 | 0.4038 | 0.710 | 0.805 | 0.0110 |
| R1 4096-env smoke 40k | 0.610 | 0.727 | 0.0767 | 0.3944 | 0.730 | 0.809 | 0.0002 |

4096-env training latest row:

| metric | value |
| --- | ---: |
| global step | 40960 |
| mean reward | -0.0376 |
| mean latent distance | 0.5745 |
| mean terminal distance | 0.5821 |
| mean residual L2 | 0.0079 |
| clip fraction | 0.0030 |
| action saturation rate | 0.2373 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_smoke_40k/latest.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_smoke_40k/eval_100_seed3400000.json`

### Interpretation

The 4096-env path is stable and fast enough, but one conservative update barely
moves the policy in closed-loop eval. It avoids the 32-env run's high clip
fraction, but eval residual L2 is nearly zero. The high train-time saturation
rate is suspicious and may mean sampled residuals are being clipped during the
rollout even though deterministic eval actions are not.

### Decision

Do not judge the formulation from this one-update run. The next dev run should
keep 4096 envs but increase policy movement carefully: more total updates,
possibly larger `alpha` or higher logstd, and monitor train/eval residual L2
and saturation. The 32-env smoke remains evidence that the reward path can
produce a positive local/task shift, but the proper conclusion needs a real
4096-env dev run.

## 2026-06-25 — Effect32 FiLM 4096-Env D_phi R1 Dev Run

### Hypothesis

The one-update 4096-env smoke was too conservative. A five-rollout dev run with
larger residual scale, lower residual penalty, and more PPO update epochs should
move the policy enough to test whether the effect-progress `D_phi` reward
improves the frozen hierarchy.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r1 \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r1_4096_dev_200k_a10 \
  --steps 204800 --num-envs 4096 --rollout-steps 10 \
  --num-minibatches 16 --update-epochs 3 \
  --learning-rate 1e-4 --initial-logstd -1.8 \
  --residual-penalty-weight 0.001 \
  --alpha 0.1 --terminal-weight 1.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Evaluated frozen, the prior 32-env smoke, and the 4096-env best-training
checkpoint on the same 300-episode seed bank:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --episodes 300 --seed-start 3400000 ...
```

### Results

300-episode eval on seed bank `3400000`:

| policy | success | max reward | D_phi reduction | raw effect reduction | reach rate | terminal AUC | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.633 | 0.740 | 0.0794 | 0.3899 | 0.735 | 0.786 | 0.0000 |
| R1 32-env smoke 20k | 0.597 | 0.707 | 0.0807 | 0.3791 | 0.701 | 0.806 | 0.0109 |
| R1 4096-env best 200k | 0.643 | 0.746 | 0.0740 | 0.3643 | 0.735 | 0.803 | 0.0034 |

Training latest row for the 4096-env dev run:

| metric | value |
| --- | ---: |
| global step | 204800 |
| mean reward | -0.0458 |
| mean latent distance | 0.5827 |
| mean terminal distance | 0.5746 |
| mean residual L2 | 0.0260 |
| clip fraction | 0.067 |
| action saturation rate | 0.0023 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_dev_200k_a10/latest.pt`
- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_dev_200k_a10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen/eval_300_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_smoke_20k/eval_300_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_dev_200k_a10_best/eval_300_seed3400000.json`

### Interpretation

The longer 4096-env dev run does not provide a convincing RL improvement. The
best-training checkpoint is only `+0.010` absolute success over frozen on 300
episodes, and its raw effect-distance reduction is worse than frozen. The prior
32-env apparent gain disappears at 300 episodes.

The implementation is useful because it verifies that effect-code local RL with
anchored `D_phi` can run at 4096 envs, but the reward/optimization is not yet
good enough to claim improvement.

### Decision

Do not promote this R1 residual recipe. The next experiment should change the
objective rather than just run longer: paired improvement against the frozen
effect32 baseline, terminal-only `D_phi` reward, or a direct low-policy update
with a BC anchor are more promising than the current dense residual-progress
reward.

## 2026-06-25 — Effect32 FiLM Terminal-Only D_phi R1 Smoke

### Hypothesis

The dense progress reward may be misleading for the learned `D_phi`. A
terminal-only objective should avoid rewarding small per-step metric artifacts
and optimize only the end-of-segment effect distance.

### Implementation

Added `distance_progress_weight` to `low_level_rl`. Existing runs keep the old
behavior with `distance_progress_weight=1.0`; terminal-only runs use
`--distance-progress-weight 0.0`.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r1 \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r1_4096_terminal_smoke_40k \
  --steps 40960 --num-envs 4096 --rollout-steps 10 \
  --num-minibatches 16 --update-epochs 3 \
  --learning-rate 1e-4 --initial-logstd -1.8 \
  --residual-penalty-weight 0.001 \
  --alpha 0.1 --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Results

300-episode eval on seed bank `3400000`:

| policy | success | max reward | D_phi reduction | raw effect reduction | reach rate | terminal AUC | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.633 | 0.740 | 0.0794 | 0.3899 | 0.735 | 0.786 | 0.0000 |
| dense R1 4096 best 200k | 0.643 | 0.746 | 0.0740 | 0.3643 | 0.735 | 0.803 | 0.0034 |
| terminal-only R1 4096 40k | 0.650 | 0.748 | 0.0678 | 0.3669 | 0.716 | 0.804 | 0.0013 |

Training latest row:

| metric | value |
| --- | ---: |
| global step | 40960 |
| mean reward | -0.0580 |
| mean latent distance | 0.5745 |
| mean terminal distance | 0.5799 |
| mean residual L2 | 0.0258 |
| clip fraction | 0.0396 |
| action saturation rate | 0.2549 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_smoke_40k/latest.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_smoke_40k/eval_300_seed3400000.json`

### Interpretation

Terminal-only `D_phi` is the best 4096-env R1 recipe so far, but the gain is
still modest: `+0.017` absolute success over frozen on 300 episodes. It improves
task success while local distance metrics get slightly worse, suggesting the
residual may be exploiting useful action changes not captured by this
effect-distance metric, or the measured local metric is not aligned enough with
task success.

### Decision

Keep terminal-only `D_phi` as the current best effect-code R1 recipe, but do not
claim solved RL fine-tuning. The next meaningful step is either a longer
terminal-only 4096-env run with checkpoint selection by dev success, or a paired
improvement reward that explicitly compares tuned versus frozen terminal
distance per segment.

## 2026-06-25 — Effect32 FiLM Longer Terminal-Only D_phi R1 Dev Run

### Hypothesis

The one-rollout terminal-only run may have been undertrained. A five-rollout
4096-env run with the same terminal-only reward should either improve over the
40k smoke or show that the early update was a transient best point.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r1 \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r1_4096_terminal_dev_200k \
  --steps 204800 --num-envs 4096 --rollout-steps 10 \
  --num-minibatches 16 --update-epochs 3 \
  --learning-rate 1e-4 --initial-logstd -1.8 \
  --residual-penalty-weight 0.001 \
  --alpha 0.1 --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Results

300-episode eval on seed bank `3400000`:

| policy | success | max reward | D_phi reduction | raw effect reduction | reach rate | terminal AUC | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.633 | 0.740 | 0.0794 | 0.3899 | 0.735 | 0.786 | 0.0000 |
| terminal-only R1 4096 40k | 0.650 | 0.748 | 0.0678 | 0.3669 | 0.716 | 0.804 | 0.0013 |
| terminal-only R1 4096 200k latest | 0.627 | 0.736 | 0.0829 | 0.3994 | 0.737 | 0.796 | 0.0035 |
| terminal-only R1 4096 200k best-train | 0.627 | 0.734 | 0.0697 | 0.3748 | 0.710 | 0.818 | 0.0035 |
| terminal-only R1 4096 200k step163840 | 0.620 | 0.730 | 0.0761 | 0.3956 | 0.717 | 0.806 | 0.0031 |

100-episode checkpoint screen:

| checkpoint | success | max reward | note |
| --- | ---: | ---: | --- |
| frozen | 0.600 | 0.711 | baseline |
| 40k smoke | 0.690 | 0.772 | best 100-episode terminal-only checkpoint |
| 200k step040960 | 0.680 | 0.772 | close to 40k smoke |
| 200k step081920 | 0.610 | 0.728 | reject |
| 200k step122880 | 0.630 | 0.741 | reject |
| 200k step163840 | 0.690 | 0.782 | failed to hold at 300 episodes |

Training latest row:

| metric | value |
| --- | ---: |
| global step | 204800 |
| mean reward | -0.0565 |
| mean latent distance | 0.5827 |
| mean terminal distance | 0.5650 |
| mean residual L2 | 0.0260 |
| clip fraction | 0.0666 |
| action saturation rate | 0.0025 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_dev_200k/latest.pt`
- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_dev_200k/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_dev_200k/eval_300_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_dev_200k_best/eval_300_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_dev_200k_step163840/eval_300_seed3400000.json`

### Interpretation

Running longer did not improve the terminal-only R1 recipe. The one-rollout
40k terminal-only checkpoint remains the best 4096-env R1 result on the
300-episode bank. Later checkpoints improve some local metrics, but task
success regresses to frozen or worse.

This points to a credit/objective issue rather than simply needing more PPO
steps. Checkpoint selection by training terminal distance is not reliable for
task success.

### Decision

Reject longer dense or terminal-only R1 residual PPO as the next main path.
Keep the 40k terminal-only checkpoint as a weak positive reference, but move to
a paired-improvement reward or direct low-policy update with BC anchoring. The
next experiment should make the objective explicitly compare tuned versus
frozen terminal outcomes instead of relying on absolute `D_phi` distance alone.

## 2026-06-25 — Effect32 FiLM Direct Low-Policy R3 Smoke

### Hypothesis

A direct final-layer low-policy update with BC anchoring may be more stable
than an additive residual policy for the effect-code `D_phi` objective.

### Implementation

Extended `DirectLowActorCritic` to support FiLM low policies by training
`low_model.output_layer` when present. The previous R3 path only supported
concat/delta policies with `low_model.policy.net[-1]`. Also fixed the CLI
override helper so `train-r3 --learning-rate` and `--initial-logstd` map to the
direct-policy config keys.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r3 \
  --candidate effect32_film --n-demo 1000 --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10 \
  --steps 40960 --num-envs 4096 --rollout-steps 10 \
  --num-minibatches 16 --update-epochs 3 \
  --learning-rate 3e-5 --initial-logstd -4.0 \
  --bc-weight 10.0 --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Results

300-episode eval on seed bank `3400000`:

| policy | success | max reward | D_phi reduction | raw effect reduction | reach rate | terminal AUC | residual/direct L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.633 | 0.740 | 0.0794 | 0.3899 | 0.735 | 0.786 | 0.0000 |
| terminal-only R1 4096 40k | 0.650 | 0.748 | 0.0678 | 0.3669 | 0.716 | 0.804 | 0.0013 |
| terminal-only R3 4096 40k bc10 | 0.643 | 0.746 | 0.0740 | 0.3971 | 0.739 | 0.805 | 0.0010 |

Training latest row:

| metric | value |
| --- | ---: |
| global step | 40960 |
| mean reward | -0.0576 |
| mean latent distance | 0.5751 |
| mean terminal distance | 0.5757 |
| mean direct delta L2 | 0.0293 |
| BC loss | 8.35e-7 |
| clip fraction | 0.0352 |
| action saturation rate | 0.2587 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/latest.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/eval_300_seed3400000.json`

### Interpretation

R3 final-layer FiLM tuning works technically, but it does not beat the simpler
R1 terminal-only smoke. The BC anchor keeps the policy close to frozen, and the
task gain remains small.

### Decision

Do not promote this R3 recipe. The strongest current effect-code RL checkpoint
remains `hcl_next_effect32_dphi_r1_4096_terminal_smoke_40k`, but even that is a
weak positive reference rather than a solved RL result. The next substantive
objective change should be paired improvement against the frozen rollout.

## Experiment E1: learned reachability distance on VAE512 latents

Implemented a lightweight reachability-distance module and CLI commands:

- `src/hcl_poc/reachability.py`
- `hcl-poc rl-rerun train-reachability-distance`
- `hcl-poc rl-rerun eval-reachability-distance`

The model uses the existing cached VAE512 latent trajectories from:

`artifacts/incremental/learned_interface/vae512_w2048_b1e6/seed0/encoded_episodes.pt`

Training target:

```text
D_phi(z_i, z_j) ~= min((j - i) / H, 1)
```

with same-trajectory future pairs plus reversed and cross-trajectory negatives.
The output is sigmoid-bounded to `[0, 1]`, matching the clipped temporal target.

Command:

```bash
env TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-reachability-distance \
  --candidate vae512_w2048_b1e6 \
  --seed 0 \
  --epochs 30 \
  --batches-per-epoch 200 \
  --batch-size 512 \
  --hidden-dim 512 \
  --depth 3 \
  --force
```

Evaluation:

```bash
env TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-reachability-distance \
  --candidate vae512_w2048_b1e6 \
  --seed 0 \
  --samples 4096 \
  --force
```

### Results

| metric | value |
| --- | ---: |
| temporal MSE | 0.00793 |
| temporal Spearman | 0.9296 |
| near mean D_phi | 0.5648 |
| far mean D_phi | 0.9886 |
| near/far accuracy | 0.9841 |
| shuffled AUC | 0.8769 |
| demo decrease accuracy | 0.7031 |

Artifacts:

- `artifacts/incremental/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt`
- `artifacts/incremental/reachability_distance/vae512_w2048_b1e6/seed0/metrics.json`
- `results/incremental/reachability_distance/vae512_w2048_b1e6/seed0/eval.json`

### Interpretation

E1 passes the non-environment validation checks from the plan: temporal
ordering is strong, near-future states are reliably closer than far-future
states, and shuffled goals are mostly separated from trajectory futures.
The weakest diagnostic is demo-decrease accuracy, which is still clearly above
chance but only around 0.70. The remaining required checks before using this as
an RL/local-control reward are correlation with privileged/TCP distance and
correlation with local rollout success.

### Matching VAE-scaling checkpoints for low-level RL

Added `--n-demo` to the reachability CLI so `D_phi` can be trained/evaluated
under the same VAE-scaling artifact tree as the frozen low-level hierarchy:

```bash
env TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-reachability-distance \
  --n-demo 1000 \
  --candidate vae512_w2048_b1e6 \
  --seed 0 \
  --epochs 30 \
  --batches-per-epoch 200 \
  --batch-size 512 \
  --hidden-dim 512 \
  --depth 3 \
  --force
```

Validation results:

| n_demo | temporal MSE | temporal Spearman | near/far acc | shuffled AUC | demo decrease acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.01540 | 0.9103 | 0.9832 | 0.8568 | 0.6658 |
| 1000 | 0.01184 | 0.9214 | 0.9827 | 0.8688 | 0.6873 |
| 1800 | 0.00793 | 0.9296 | 0.9841 | 0.8769 | 0.7031 |

Artifacts:

- `artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt`
- `artifacts/incremental/vae512_scaling/n1000/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt`

The lower-data runs degrade smoothly but still pass the offline ordering checks.
These are the correct checkpoints to use for low-level RL reward comparisons at
`n_demo=500` and `n_demo=1000`; do not mix the global `n1800` distance model
with those frozen hierarchies.

## Experiment E1 reward hook: use D_phi in learned-latent low-level RL

Added a minimal reward-metric switch to the learned-latent low-level RL rollout:

```bash
uv run hcl-poc low-level-rl ... train-r1 --distance-metric raw_l2|reachability
uv run hcl-poc low-level-rl ... train-r3 --distance-metric raw_l2|reachability
```

Default remains `raw_l2`, preserving existing behavior. With
`--distance-metric reachability`, the trainer loads the matching checkpoint:

```text
artifacts/incremental/vae512_scaling/n{n_demo}/reachability_distance/vae512_w2048_b1e6/seed{seed}/d_phi.pt
```

unless `--reachability-checkpoint` is provided. The low-level policy condition
is unchanged; only the progress and terminal distance used for reward are
swapped from raw latent MSE to learned `D_phi`.

Smoke run:

```bash
env TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_dphi_smoke_n1000_r3_1k \
  --steps 1024 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --distance-metric reachability \
  --force
```

Latest smoke metrics:

| metric | value |
| --- | ---: |
| global step | 1280 |
| mean D_phi distance | 0.7516 |
| mean terminal D_phi distance | 0.7878 |
| mean reward | -0.0646 |
| mean direct delta L2 | 0.0294 |
| action saturation rate | 0.0094 |

Artifacts:

- `artifacts/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_smoke_n1000_r3_1k/latest.pt`
- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_smoke_n1000_r3_1k/train_metrics.json`

### Interpretation

The learned-distance reward path loads and runs. This is not yet an RL result;
it is only a short integration smoke test. The next meaningful comparison is a
matched raw-L2 vs D_phi low-level RL run at the same budget and training steps,
followed by normal evaluation.

## Experiment E1 matched low-level RL comparison: raw L2 vs D_phi

Compared the existing raw-L2 low-level RL baselines against matched `D_phi`
reward runs on the same frozen `n_demo=1000`, `seed=0` VAE512 hierarchy and the
same 300-episode development evaluation:

```text
episodes: 300
seed_start: 3200000
evaluation checkpoint: best_train_latent.pt unless noted
```

Raw-L2 baselines already present:

- `artifacts/incremental/low_level_rl/n1000/seed0/r1_a005_progress1_50k/best_train_latent.pt`
- `artifacts/incremental/low_level_rl/n1000/seed0/r3_bc1_lownoise_progress1_50k/best_train_latent.pt`

New D_phi runs:

```bash
env TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r1 \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_dphi_n1000_r1_a005_50k \
  --steps 50000 \
  --alpha 0.05 \
  --terminal-weight 1.0 \
  --distance-metric reachability \
  --force
```

```bash
env TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_dphi_n1000_r3_bc1_50k \
  --steps 50000 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --distance-metric reachability \
  --force
```

Also tested two targeted R3 variants:

- `hcl_next_dphi_n1000_r3_bc5_50k`: stronger BC regularization.
- `hcl_next_dphi_n1000_r3_bc1_tw0_50k`: progress-only reward with no terminal
  D_phi penalty.

### Results

| policy | reward metric | variant | eval success | max reward | action/residual L2 | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| frozen hierarchy | none | reference | 0.553 | 0.676 | 0.000 | best overall in this set |
| R1 residual | raw L2 | alpha 0.05, 50k | 0.527 | 0.656 | 0.004 | existing baseline |
| R3 direct | raw L2 | bc 1, 50k | 0.493 | 0.629 | 0.012 | existing baseline |
| R1 residual | D_phi | alpha 0.05, 50k | 0.390 | 0.564 | 0.021 | worse |
| R3 direct | D_phi | bc 1, terminal 1, 50k best | 0.547 | 0.666 | 0.009 | best D_phi, near frozen |
| R3 direct | D_phi | bc 1, terminal 1, 50k latest | 0.480 | 0.621 | 0.023 | training drifted |
| R3 direct | D_phi | bc 5, terminal 1, 50k | 0.490 | 0.631 | 0.013 | stronger BC did not help |
| R3 direct | D_phi | bc 1, terminal 0, 50k | 0.400 | 0.564 | 0.017 | progress-only did not help |

Artifacts:

- `artifacts/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r1_a005_50k/best_train_latent.pt`
- `artifacts/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_50k/best_train_latent.pt`
- `artifacts/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc5_50k/best_train_latent.pt`
- `artifacts/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_tw0_50k/best_train_latent.pt`

Evaluation outputs:

- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r1_a005_50k/eval_300_seed3200000.json`
- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_50k/eval_300_seed3200000.json`
- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_50k_latest300/eval_300_seed3200000.json`
- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc5_50k/eval_300_seed3200000.json`
- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_tw0_50k/eval_300_seed3200000.json`

### Refreshed evaluator and rollout-success diagnostic

Fixed low-level RL evaluation so segment-distance diagnostics use the distance
metric recorded in the checkpoint recipe. Before this fix, environment success
was still valid, but D_phi-trained checkpoints would have reported raw-L2
segment distances during evaluation.

Added raw local-reach diagnostics to all low-level evals:

- `raw_segment_initial_distance`
- `raw_segment_final_distance`
- `raw_segment_distance_reduction`
- `segment_goal_reach_rate`, computed with the raw-L2 teacher threshold even
  when the selected reward metric is D_phi
- `selected_metric_terminal_reach_auc`, using lower selected terminal distance
  to predict raw local reaching

After rerunning the frozen, raw-L2, and best D_phi evals with the refreshed
evaluator, the current authoritative development comparison is:

| policy | reward metric | eval success | max reward | raw local reduction | raw reach rate | terminal metric reach AUC | action/residual L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen hierarchy | none | 0.507 | 0.640 | 0.184 | 0.759 | 1.000 | 0.000 |
| R1 residual | raw L2 | 0.510 | 0.640 | 0.164 | 0.743 | 1.000 | 0.004 |
| R3 direct | raw L2 | 0.470 | 0.617 | 0.169 | 0.767 | 1.000 | 0.012 |
| R3 direct | D_phi | 0.537 | 0.662 | 0.183 | 0.758 | 0.774 | 0.009 |

The AUC of `0.774` means D_phi terminal distance is meaningfully predictive of
raw local reaching, but not as aligned as the raw-L2 metric with the raw-L2
teacher threshold. The best D_phi R3 run is now the top current development
result in this matched set, but the margin is small enough that it needs a
larger/final-seed confirmation before promotion.

### Interpretation

E1's learned distance passes offline temporal/reachability diagnostics, the
reward hook runs, and the best R3 D_phi run gives a small development-set
improvement over the refreshed frozen/raw baselines. The result is not yet
strong: training still drifts when continuing to the latest checkpoint, and
D_phi terminal distance is only moderately aligned with raw local reaching.

This suggests the next bottleneck is not only the distance metric. The policy
update/reward formulation is still weak or partially misaligned:

- best-train selection by D_phi terminal distance does not reliably select
  higher environment success;
- R1 residual updates became too disruptive under D_phi;
- direct R3 updates can find a small improvement, but it is not robust across
  reward variants;
- removing the terminal D_phi penalty made performance worse, so the issue is
  not only terminal weighting.

Next recommended step: confirm the best D_phi R3 checkpoint on the final seed
range and compare against a privileged/TCP upper-bound or paired-improvement
reward before running broader PPO sweeps.

### Final-seed confirmation

Confirmed the best development D_phi R3 checkpoint on the final seed range:

```text
episodes: 500
seed_start: 3400000
```

Commands:

```bash
env TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_dphi_n1000_r3_bc1_50k_final500 \
  --episodes 500 \
  --seed-start 3400000 \
  --checkpoint artifacts/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_50k/best_train_latent.pt \
  --force
```

| policy | checkpoint | success | max reward | raw local reduction | raw reach rate | terminal metric reach AUC | action/residual L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen hierarchy | none | 0.512 | 0.641 | 0.165 | 0.758 | 1.000 | 0.000 |
| R1 residual raw L2 | `r1_a005_progress1_50k/best_train_latent.pt` | 0.514 | 0.649 | 0.179 | 0.766 | 1.000 | 0.004 |
| R3 direct D_phi | `hcl_next_dphi_n1000_r3_bc1_50k/best_train_latent.pt` | 0.546 | 0.669 | 0.187 | 0.773 | 0.803 | 0.009 |

Final outputs:

- `results/incremental/low_level_rl/n1000/seed0/frozen_reference_final500/eval_500_seed3400000.json`
- `results/incremental/low_level_rl/n1000/seed0/r1_a005_progress1_50k_final500/eval_500_seed3400000.json`
- `results/incremental/low_level_rl/n1000/seed0/hcl_next_dphi_n1000_r3_bc1_50k_final500/eval_500_seed3400000.json`

### Decision

Promote `hcl_next_dphi_n1000_r3_bc1_50k/best_train_latent.pt` as the current
best low-level RL checkpoint for the learned VAE512 stack. The improvement is
modest but replicated on the final seed range:

```text
D_phi R3 final success: 0.546
frozen final success:   0.512
raw R1 final success:   0.514
```

This is the first learned-distance low-level RL result in this run that improves
over both frozen and raw-L2 RL baselines on a held-out seed range. Next
recommended step: test whether the same D_phi reward improves `n_demo=500`, and
then compare against the privileged/TCP upper-bound reward.

## Experiment E1 lower-data check: n_demo=500 D_phi R3

Tested the same D_phi R3 recipe at `n_demo=500`, using the matching
VAE-scaling D_phi checkpoint:

`artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt`

Command:

```bash
env TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_dphi_n500_r3_bc1_50k \
  --steps 50000 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --distance-metric reachability \
  --force
```

Development eval:

```text
episodes: 300
seed_start: 3200000
checkpoint: best_train_latent.pt
```

| policy | reward metric | success | max reward | raw local reduction | raw reach rate | terminal metric reach AUC | action/residual L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen hierarchy | none | 0.313 | 0.494 | n/a | 0.578 | n/a | 0.000 |
| R1 residual | raw L2 | 0.370 | 0.537 | n/a | 0.603 | n/a | 0.005 |
| R3 direct | raw L2 | 0.390 | 0.547 | n/a | 0.588 | n/a | 0.006 |
| R3 direct | D_phi | 0.347 | 0.517 | 0.141 | 0.603 | 0.717 | 0.017 |

Artifacts:

- `artifacts/incremental/low_level_rl/n500/seed0/hcl_next_dphi_n500_r3_bc1_50k/best_train_latent.pt`
- `results/incremental/low_level_rl/n500/seed0/hcl_next_dphi_n500_r3_bc1_50k/eval_300_seed3200000.json`

### Interpretation

The D_phi reward improvement does not transfer directly to the lower-data
`n_demo=500` hierarchy. D_phi terminal distance is still predictive of raw local
reaching (`AUC=0.717`), but the policy update is too disruptive and trails the
existing raw-L2 R3 baseline. Keep the n1000 D_phi result as the current
positive result; do not promote the n500 D_phi run.

## Experiment C1: privileged/TCP paired-reward upper-bound check

The plan asks to validate paired-improvement reward with a privileged/TCP
distance before relying on learned distances. Existing 1M-step paired-reward
runs were already available, so I first inspected them instead of launching a
new sweep.

Base checkpoint:

`artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`

Init dataset:

`data/rl_rerun/privileged_z_residual_init_B_clean_disturbed_n4096_b2.h5`

### Existing residual paired-reward runs

| run | alpha in recipe | train mean paired improvement | train fraction improved | replay eval mean paired improvement | replay eval fraction improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hcl_next_paired_reward_n4096_alpha025_1m` | 0.25 | -0.0113 | 0.341 | 0.0644 | 0.384 |
| `hcl_next_paired_reward_n4096_alpha05_1m` | 0.50 | -0.0598 | 0.246 | 0.1069 | 0.353 |
| `hcl_next_paired_reward_n4096_alpha10_1m` | 1.00 | -0.2529 | 0.116 | 0.0735 | 0.151 |

These miss the plan's local pass criterion:

```text
fraction_improved > 0.55
mean_paired_improvement > 0
```

The replay-eval mean improvements are positive only because of heavy-tailed
outliers; medians are near zero or negative and the improved fraction is too
low.

### Existing direct paired-reward runs

Most direct paired local replay evals also miss the pass criterion, with
`fraction_improved` around `0.26-0.48`. One hard-start direct run passes the
training-batch criterion but not replay eval:

| run | train mean paired improvement | train fraction improved | replay eval mean paired improvement | replay eval fraction improved |
| --- | ---: | ---: | ---: | ---: |
| `hcl_next_direct_from_basecap5_delta025_imp05_hardmse005_final_layer_n4096_1m` | 0.0057 | 0.577 | 0.0821 | 0.442 |

### Closed-loop task check

Refreshed 200-episode closed-loop evaluations on `seed_start=9900000`:

| policy | mode | success | return | mean action delta |
| --- | --- | ---: | ---: | ---: |
| base privileged-z | hierarchy | 0.545 | 41.58 | 0.000 |
| residual paired alpha025 | hierarchy | 0.550 | 40.89 | 0.016 |
| direct paired hard-start | hierarchy | 0.515 | 40.47 | 0.010 |
| base privileged-z | oracle_hierarchy | 0.700 | 46.75 | 0.000 |
| residual paired alpha025 | oracle_hierarchy | 0.690 | 46.73 | 0.015 |
| direct paired hard-start | oracle_hierarchy | 0.700 | 47.11 | 0.014 |

Outputs:

- `results/hcl_next_phase1/privileged_z_closed_loop_base_clean_disturbed_n1800_hierarchy_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_paired_reward_alpha025_1m_hierarchy_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_hierarchy_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_base_clean_disturbed_n1800_oracle_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_paired_reward_alpha025_1m_oracle_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_oracle_200eps.json`

### Decision

Do not promote the privileged/TCP paired-reward runs. The residual run gives a
tiny hierarchy success increase on this 200-episode slice (`+0.005`) but lowers
return and fails local improvement criteria. The direct hard-start run can pass
training-batch local improvement, but replay local eval and closed-loop
hierarchy success contradict promotion.

This is useful evidence: even with privileged-state distance, the current
paired reward formulation is not reliably solving local RL. That supports the
plan's diagnosis that the PPO/local-control formulation itself still needs
work, not only the representation or distance metric.

## Experiment B: offline goal-identifiability diagnostics for VAE512 low levels

Implemented a reusable offline goal-diagnostics module:

- `src/hcl_poc/goal_diagnostics.py`
- CLI: `hcl-poc rl-rerun goal-diagnostics`

The diagnostic uses cached validation trajectories and frozen VAE512 low-level
policies. It does not run the simulator. It reports:

- same-current-state action sensitivity for valid future goals at horizons
  `2`, `5`, and `10`;
- action MAE per horizon;
- condition-block shuffle sensitivity for frame, goal, previous action, and
  remaining-time inputs.

Command template:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --n-demo 1000 \
  --samples 5000 \
  --horizons 2,5,10 \
  --force
```

### Results

| n_demo | action MAE k10 | max same-state goal sensitivity | goal-shuffle action L2 | goal-shuffle MAE gap | frame-shuffle action L2 | prev-action shuffle L2 | remaining shuffle L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.0653 | 0.0236 | 0.0489 | 0.0074 | 0.9609 | 0.0813 | 0.0009 |
| 1000 | 0.0507 | 0.0276 | 0.0649 | 0.0119 | 0.9546 | 0.1106 | 0.0012 |
| 1800 | 0.0419 | 0.0308 | 0.0740 | 0.0153 | 0.9498 | 0.1285 | 0.0013 |

Outputs:

- `results/incremental/goal_diagnostics/n500/seed0/vae512.json`
- `results/incremental/goal_diagnostics/n1000/seed0/vae512.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae512.json`

### Interpretation

The VAE512 concat low levels still mostly ignore the goal compared with the
current visual/state input:

- frame shuffle changes action by about `0.95` L2;
- goal shuffle changes action by only `0.05-0.07` L2;
- same-state valid future-goal sensitivity is only `0.02-0.03` L2;
- remaining-time sensitivity is effectively zero.

This reproduces the plan's stated failure mode: the supervised low level can
get good action MAE while remaining weakly goal-conditioned. Increasing demos
from 500 to 1800 improves action MAE and slightly increases goal sensitivity,
but it does not approach the target scale mentioned in the plan, where
privileged-state sensitivity was around `0.26`.

Decision: keep using the n1000 D_phi R3 result as a useful positive RL result,
but treat VAE512 concat as failing the full Experiment B goal-identifiability
gate. The next implementation step should be a FiLM/gated or goal-residual
low-level architecture, then rerun this same diagnostic before launching more
expensive RL.

Keep goal-NN79 as the best oracle-goal/low-level diagnostic, but do not use it
as the current learned-high checkpoint.

## State+goal-NN79 improve-weight sweep

After promoting the state+goal-NN79 `improve_npz_weight=0.1` checkpoint, I ran
a narrow weight sweep to check whether the selected branch pressure was too
weak or too strong.

### Setup

Same bank and training recipe as the promoted state+goal-NN79 run, changing
only:

```text
--improve-npz-weight 0.05
--improve-npz-weight 0.15
```

Artifacts:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp005_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp015_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| improve NPZ weight | mode | success | return | decision |
| ---: | --- | ---: | ---: | --- |
| 0.05 | hierarchy | 0.540 | 41.20 | reject |
| 0.05 | oracle_hierarchy | 0.695 | 47.14 | diagnostic only |
| 0.10 | hierarchy | 0.575 | 41.89 | keep current best |
| 0.10 | oracle_hierarchy | 0.705 | 46.69 | matched already run |
| 0.15 | hierarchy | 0.565 | 41.81 | reject |
| 0.15 | oracle_hierarchy | 0.705 | 47.09 | diagnostic only |

The `0.05` and `0.15` variants did not clear the promoted `0.10` hierarchy dev
bar, so I skipped matched 3x500 evaluations for both.

### Interpretation

For this selected branch bank, the useful update is weight-sensitive and peaks
near `0.1` in the tested range. Lower weight appears too weak to move the
learned-high rollout distribution, while higher weight does not improve the
hierarchy and does not recover the oracle-goal gap.

### Decision

Keep the state+goal-NN79 `improve_npz_weight=0.1` checkpoint as the current
learned-high best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## State+goal-NN79 high-level projection and local gate diagnostics

After promoting the state+goal-NN79 checkpoint, I reran two high-level/local
diagnostics on the new best: eval-time prototype projection from Experiment H
and local oracle gating of tuned low-level segments.

### Prototype projection

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --high-goal-projection nearest_oracle_bank \
  --high-goal-bank-episodes 200 \
  --high-goal-bank-seed-start 9800000 \
  --high-goal-bank-num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_projected_200eps.json \
  --force
```

Results:

| checkpoint | projection | success | return | bank size | predicted-to-prototype MSE median | predicted-to-prototype MSE p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| state+goal-NN79 imp0.1 | none | 0.575 | 41.89 | n/a | n/a | n/a |
| state+goal-NN79 imp0.1 | nearest oracle bank | 0.350 | 35.94 | 2000 | 0.0357 | 0.3162 |

Projection is strongly harmful again. The nearest teacher prototypes are close
in normalized privileged-state MSE, but replacing the continuous learned
high-level output changes the control target enough to collapse task success.

### Local oracle gate

I evaluated an oracle local segment gate that rolls out both base and tuned
low-level policies for the current held goal, then uses the tuned action only
when the tuned terminal MSE is no worse than the base terminal MSE plus a
tolerance.

Results on the 200-episode dev window:

| mode | gate max degradation MSE | success | return | tuned fraction | paired improvement MSE mean | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hierarchy | always tuned | 0.575 | 41.89 | 1.000 | n/a | keep |
| hierarchy | 0.00 | 0.520 | 41.24 | 0.396 | -0.364 | reject |
| hierarchy | 0.05 | 0.535 | 41.24 | 0.759 | -0.362 | reject |
| oracle_hierarchy | always tuned | 0.705 | 46.69 | 1.000 | n/a | diagnostic |
| oracle_hierarchy | 0.00 | 0.725 | 47.02 | 0.478 | 0.458 | diagnostic |

Artifacts:

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_projected_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_local_oracle_gate0_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_local_oracle_gate005_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_local_oracle_gate0_200eps.json`

### Interpretation

The projection result reinforces the earlier Experiment H conclusion: naive
nearest-prototype projection is not a valid mitigation for learned high-level
goals, even when the selected prototype is close in normalized state space.

The local gate gives a sharper diagnostic. Under learned-high goals, terminal
MSE gating rejects many tuned segments and reduces task success. Even a relaxed
`0.05` tolerance remains below always-tuned. Under oracle goals, the same gate
improves dev success from `0.705` to `0.725`, which means local MSE is useful
for oracle-goal correction but misaligned with learned-high task success. This
matches the branch-selection result: the best learned-high selector was based
on state+goal distribution matching and outcome labels, not purely local MSE.

### Decision

Keep the state+goal-NN79 always-tuned checkpoint as current learned-high best.
Do not use nearest-prototype projection or local terminal-MSE gating as an
eval-time learned-high mitigation. Future selection should optimize learned-high
closed-loop outcomes directly, or learn a selector from state+goal branch
features and task-outcome labels rather than relying on local terminal MSE.

## Hybrid state+goal proximity plus return selector

The previous selectors exposed a tradeoff:

- state+goal-NN79 gave the best learned-high success but weak oracle-goal
  success;
- goal-NN79 and return-heavy subsets improved oracle-goal behavior but did not
  improve learned-high success.

I tested a simple hybrid selector before implementing a learned selector:
restrict to branches near the preserve-bank learned-high state+goal
distribution, then choose the strongest return-improving branches within that
pool.

### Setup

Source bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_300eps_b4_c16.npz`

Reference distribution:

- `data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz`

Selection:

- compute first-row state+held-goal nearest-neighbor MSE to preserve-bank rows
  using condition slice `[0:62]`, standardized by the preserve-bank
  distribution;
- keep the closest 120 branches;
- from those, select the top 79 branches by `selected_return_delta`;
- preserve full 10-step branch blocks.

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn120_topret79_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 79 |
| horizon rows | 790 |
| preserve state+goal NN MSE mean/max | 0.0824 / 0.1525 |
| return delta mean | 35.71 |
| success delta mean | 0.532 |
| action delta mean | 0.194 |
| source seed counts | 9900000: 24, 9910000: 30, 9920000: 25 |

Training used the current-best recipe:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn120_topret79_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return |
| --- | --- | ---: | ---: |
| state+goal-NN120 top-return79 | hierarchy | 0.550 | 41.86 |
| state+goal-NN120 top-return79 | oracle_hierarchy | 0.755 | 48.47 |
| state+goal-NN79 current learned-high best | hierarchy | 0.575 | 41.89 |
| state+goal-NN79 current learned-high best | oracle_hierarchy | 0.705 | 46.69 |
| goal-NN79 oracle diagnostic | hierarchy | 0.555 | 41.36 |
| goal-NN79 oracle diagnostic | oracle_hierarchy | 0.725 | 47.16 |

The hybrid selector failed the learned-high dev gate, so I skipped matched
learned-high evaluation. Because oracle dev was strong, I ran matched oracle
3x500:

| checkpoint | mode | seed 10000000 | seed 10100000 | seed 10200000 | success mean | return mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| state+goal-NN120 top-return79 | oracle_hierarchy | 0.738 | 0.748 | 0.710 | 0.7320 | 48.25 |
| goal-NN79 oracle diagnostic | oracle_hierarchy | 0.742 | 0.750 | 0.706 | 0.7327 | 48.55 |
| state+goal-NN79 current learned-high best | oracle_hierarchy | 0.708 | 0.738 | 0.684 | 0.7100 | 47.92 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn120_topret79_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn120_topret79_imp01_seed10200000_500eps.json`

### Interpretation

Adding stronger return filtering inside a preserve-proximal state+goal pool
recovers oracle-goal quality on the dev window, but the matched oracle result
is only a near-tie with goal-NN79 and the learned-high dev result drops below
the current best. The selector therefore moves back toward the oracle-goal
tradeoff rather than improving the learned-high distribution.

This narrows the next selector direction: simple scalar mixing of proximity and
return is not enough. The useful signal likely needs either a learned selector
over branch features or additional learned-high outcome labels that are not
captured by return delta and nearest-neighbor proximity alone.

### Decision

Do not promote the hybrid selector. Keep:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

as the learned-high best, and keep goal-NN79 as the marginally best
oracle-goal diagnostic.

## Fourth branch-outcome window and multi4 state+goal selector

To check whether the state+goal selector was overfitting the three available
outcome windows, I collected one more learned-high branch-outcome window and
reran the closest-state+goal 79-branch selector over the expanded pool.

### Fresh outcome window

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9930000 \
  --num-envs 100 \
  --random-candidates 16 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.01 \
  --max-action-delta-l2 0.25 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9930000_100eps_b4_c16.npz \
  --bank-min-return-delta 5 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_return_delta_ge5_seed9930000_100eps_b4_c16.json \
  --force
```

The new window had base closed-loop success `0.480` and produced 70
return-positive branches:

| metric | value |
| --- | ---: |
| branches | 70 |
| return delta mean/median | 22.50 / 18.49 |
| success delta mean | 0.343 |
| return delta min/max | 5.04 / 54.89 |

Merged bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_9930000_400eps_b4_c16.npz`

Merged summary:

| metric | value |
| --- | ---: |
| branches | 285 |
| horizon rows | 2850 |
| return delta mean | 24.56 |
| success delta mean | 0.337 |
| action delta mean | 0.167 |
| source seed counts | 9900000: 79, 9910000: 71, 9920000: 65, 9930000: 70 |

### Multi4 selector

Selection:

- compute first-row state+held-goal nearest-neighbor MSE to preserve-bank rows
  using condition slice `[0:62]`;
- select the 79 branches closest to the preserve-bank learned-high
  state+goal distribution.

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi4_stategoalnn79_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 79 |
| horizon rows | 790 |
| preserve state+goal NN MSE mean/max | 0.0470 / 0.0895 |
| return delta mean | 27.81 |
| success delta mean | 0.430 |
| action delta mean | 0.194 |
| source seed counts | 9900000: 21, 9910000: 21, 9920000: 17, 9930000: 20 |

Training:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi4_stategoalnn79_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi4_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return | decision |
| --- | --- | ---: | ---: | --- |
| multi4 state+goal-NN79 | hierarchy | 0.535 | 41.68 | reject |
| multi4 state+goal-NN79 | oracle_hierarchy | 0.715 | 47.40 | diagnostic only |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 | keep |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 | diagnostic |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9930000_100eps_b4_c16.npz`
- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_9930000_400eps_b4_c16.npz`
- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi4_stategoalnn79_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi4_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_branch_outcome_return_delta_ge5_seed9930000_100eps_b4_c16.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi4_stategoalnn79_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi4_stategoalnn79_imp01_200eps.json`

### Interpretation

Adding a fourth outcome window made the nearest-state+goal selected bank more
distribution-close, but it reduced the outcome strength relative to the
three-window state+goal-NN79 bank and hurt learned-high dev success. More
outcome diversity alone is therefore not enough; the selector needs to preserve
the particular learned-high-improving cases rather than simply tracking the
nearest preserve distribution more tightly.

### Decision

Do not promote multi4 state+goal-NN79. Keep the multi3 state+goal-NN79
checkpoint as current learned-high best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## Multi4 fail-to-success outcome-label selector

The merged four-window branch-outcome bank contains 96 clean fail-to-success
branches:

```text
selected_base_success == 0
selected_candidate_success == 1
```

This is the most direct learned-high success label in the bank, so I tested it
as an explicit selector.

### Setup

Source bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_9930000_400eps_b4_c16.npz`

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_fail_to_success_multi4_96.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 96 |
| horizon rows | 960 |
| return delta mean/median | 33.74 / 36.61 |
| action delta mean | 0.182 |
| source seed counts | 9900000: 27, 9910000: 22, 9920000: 23, 9930000: 24 |

Training used the same current recipe:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_fail_to_success_multi4_96.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_fail_to_success_multi4_96_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return | decision |
| --- | --- | ---: | ---: | --- |
| multi4 fail-to-success | hierarchy | 0.505 | 40.83 | reject |
| multi4 fail-to-success | oracle_hierarchy | 0.695 | 45.54 | reject |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 | keep |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 | keep |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_fail_to_success_multi4_96.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_fail_to_success_multi4_96_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_fail_to_success_multi4_96_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_fail_to_success_multi4_96_imp01_200eps.json`

### Interpretation

Hard fail-to-success labels are too blunt as distillation targets. They are
strong learned-high outcome labels in the collection rollouts, but imitating
all of them hurts both learned-high and oracle-goal dev performance. This
suggests the useful branches are not simply all success flips; they must also
match the stable learned-high operating distribution. The earlier state+goal
distribution match remains the only selector that improved learned-high
matched success.

### Decision

Reject the fail-to-success bank and keep the multi3 state+goal-NN79 checkpoint
as current learned-high best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## Proximity-filtered success-flip selector

The full multi4 fail-to-success bank was too broad. I tested whether the same
hard success-flip signal becomes useful when restricted to branches that are
also close to the learned-high preserve-bank state+goal distribution.

### Setup

Source bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_seed9900000_9910000_9920000_9930000_400eps_b4_c16.npz`

Selection:

- require `selected_success_delta > 0`;
- compute nearest-neighbor MSE from the first branch row's state+held-goal
  condition slice `[0:62]` to the preserve-bank first-row state+goal
  distribution;
- select the closest 40 success-flip branches.

Filtered bank:

- `data/manifests/privileged_z_branch_outcome_success_delta_pos_multi4_stategoalnn40_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 40 |
| horizon rows | 400 |
| preserve state+goal NN MSE mean/max | 0.0547 / 0.1018 |
| return delta mean | 40.66 |
| success delta mean | 1.000 |
| action delta mean | 0.213 |
| source seed counts | 9900000: 14, 9910000: 9, 9920000: 11, 9930000: 6 |

Training used the same current recipe:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_success_delta_pos_multi4_stategoalnn40_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_success_delta_pos_multi4_stategoalnn40_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return | decision |
| --- | --- | ---: | ---: | --- |
| success-flip state+goal-NN40 | hierarchy | 0.530 | 41.88 | reject |
| success-flip state+goal-NN40 | oracle_hierarchy | 0.710 | 46.43 | reject |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 | keep |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 | keep |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_success_delta_pos_multi4_stategoalnn40_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_success_delta_pos_multi4_stategoalnn40_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_success_delta_pos_multi4_stategoalnn40_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_success_delta_pos_multi4_stategoalnn40_imp01_200eps.json`

### Interpretation

Even success-flip branches that are close to the learned-high state+goal
distribution do not improve learned-high success. This rules out the simple
"success flips plus proximity" selector. The only selector that has improved
matched learned-high success remains the multi3 state+goal-NN79 bank, whose
success labels are weaker but whose branch mix appears better aligned with the
learned-high rollout distribution.

### Decision

Reject success-flip state+goal-NN40 and keep the multi3 state+goal-NN79
checkpoint as current learned-high best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## Multi3/multi4 stable-core selector

The rejected multi4 state+goal-NN79 bank still overlapped the promoted multi3
state+goal-NN79 bank on 59 of 79 branches. I tested whether training only on
this stable intersection preserves the useful learned-high behavior while
discarding the 20 replacement branches introduced by the fourth outcome window.

### Setup

Intersection bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi4_stategoal_intersection59_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 59 |
| horizon rows | 590 |
| return delta mean | 29.29 |
| success delta mean | 0.475 |
| action delta mean | 0.210 |
| source seed counts | 9900000: 21, 9910000: 21, 9920000: 17 |

Training used the same current recipe:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi4_stategoal_intersection59_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi4_stategoal_intersection59_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return | decision |
| --- | --- | ---: | ---: | --- |
| stable intersection59 | hierarchy | 0.570 | 41.98 | reject |
| stable intersection59 | oracle_hierarchy | 0.715 | 48.12 | diagnostic only |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 | keep |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 | keep |
| multi4 state+goal-NN79 | hierarchy | 0.535 | 41.68 | rejected earlier |
| multi4 state+goal-NN79 | oracle_hierarchy | 0.715 | 47.40 | rejected earlier |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi4_stategoal_intersection59_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi4_stategoal_intersection59_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi4_stategoal_intersection59_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi4_stategoal_intersection59_imp01_200eps.json`

### Interpretation

The stable-core bank recovers most of the learned-high performance lost by the
full multi4 selector, so the fourth-window replacement branches were likely
harmful. However, the stable-core hierarchy result still does not beat the
promoted full multi3 state+goal-NN79 bank. The 20 non-overlap branches in the
multi3 bank appear to be part of the useful learned-high mix, not just noise.

The oracle-goal diagnostic improves to `0.715`, matching the full multi4 oracle
result and exceeding the current learned-high best's oracle dev score, but this
does not transfer to learned-high task success.

### Decision

Reject the stable intersection59 selector and keep the multi3 state+goal-NN79
checkpoint as current learned-high best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## Multi3 state+goal-NN79 plus fourth-window augmentation

The stable-core experiment showed that the fourth-window replacement branches
hurt learned-high performance, but it did not test whether fourth-window
branches are useful as additional data when the promoted multi3 branch set is
kept intact. I therefore built an augmentation bank that preserves all 79
branches from the current multi3 state+goal-NN79 bank and adds the 20
`seed_start=9930000` branches selected by the multi4 state+goal proximity rule.

### Setup

Augmented bank:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_preserve.npz`

Bank summary:

| metric | value |
| --- | ---: |
| branches | 99 |
| horizon rows | 990 |
| return delta mean | 26.66 |
| success delta mean | 0.414 |
| action delta mean | 0.182 |
| source seed counts | 9900000: 31, 9910000: 28, 9920000: 20, 9930000: 20 |

Training used the same final-layer distillation recipe and weights as the
current best:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_preserve.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return | decision |
| --- | --- | ---: | ---: | --- |
| multi3 + seed993 top20 | hierarchy | 0.540 | 42.06 | reject |
| multi3 + seed993 top20 | oracle_hierarchy | 0.710 | 48.09 | diagnostic only |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 | keep |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 | keep |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_return_delta_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_preserve.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_branch_outcome_return_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_branch_outcome_return_ge5_multi3_stategoalnn79_plus_multi4_seed993_top20_imp01_200eps.json`

### Interpretation

Adding the fourth-window branches is harmful even when the promoted multi3 bank
is not displaced. The oracle-goal score remains slightly above the current
learned-high best, but learned-high success drops sharply from `0.575` to
`0.540`. This suggests the fourth-window proximity-selected branches are not
just neutral extra data; under the current distillation recipe, they pull the
low level away from the learned-high operating distribution that produced the
matched 3x500 improvement.

### Decision

Reject the fourth-window augmentation. Keep the multi3 state+goal-NN79
checkpoint as current learned-high best:

`artifacts/incremental/privileged_z_direct_distill/hcl_next_branch_outcome_return_ge5_multi3_stategoalnn79_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

## Experiment G probe: oracle-low-level branch generator

The previous branch-outcome banks used random-noise local action search around
the frozen learned-high segment. Experiment G explicitly warns that random
actions are a weak branch generator, so I added a competent local branch source
to `eval-privileged-z-branch-outcomes`:

```text
--branch-source random_search|oracle_low_level
```

The new `oracle_low_level` mode rolls the privileged PPO teacher for `k=10`
steps from the current simulator state to get an oracle continuation goal, then
executes the frozen low level toward that oracle goal. For distillation, it
stores the same current states and previous actions but keeps the learned
high-level goal in the condition. This tests whether competent oracle-local
actions can correct learned-high rollouts without replacing the high level.

### Smoke

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 20 \
  --seed-start 9940000 \
  --num-envs 20 \
  --branch-source oracle_low_level \
  --random-candidates 1 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.0 \
  --max-action-delta-l2 1.0 \
  --max-branch-batches 2 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_oracle_low_level_seed9940000_20eps_b2.npz \
  --bank-min-return-delta 5 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_seed9940000_20eps_b2.json \
  --force
```

The smoke path executed and wrote an 8-branch return-positive bank.

### Main 100-episode bank

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9940000 \
  --num-envs 100 \
  --branch-source oracle_low_level \
  --random-candidates 1 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.0 \
  --max-action-delta-l2 1.0 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_oracle_low_level_return_delta_ge5_seed9940000_100eps_b4.npz \
  --bank-min-return-delta 5 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_seed9940000_100eps_b4.json \
  --force
```

Bank summary:

| metric | value |
| --- | ---: |
| branches | 100 |
| horizon rows | 1000 |
| success delta mean | 0.410 |
| return delta mean | 24.97 |
| return delta median | 20.77 |
| return delta min/max | 5.37 / 60.05 |

All-branch summary from the attribution run:

| branch set | count | success delta | return-delta median | candidate better return fraction |
| --- | ---: | ---: | ---: | ---: |
| all oracle-low-level branches | 400 | +0.015 | 0.014 | 0.518 |
| locally rejected | 281 | +0.028 | 0.109 | 0.559 |

As in earlier outcome attribution, local learned-goal MSE is not the right
selector: many oracle-low-level branches that are locally rejected still improve
return.

### Distillation and dev evaluation

Training:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_oracle_low_level_return_delta_ge5_seed9940000_100eps_b4.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_oracle_low_level_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return | decision |
| --- | --- | ---: | ---: | --- |
| oracle-low-level branch bank | hierarchy | 0.565 | 42.76 | reject |
| oracle-low-level branch bank | oracle_hierarchy | 0.685 | 46.20 | reject |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 | keep |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 | keep |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_oracle_low_level_seed9940000_20eps_b2.npz`
- `data/manifests/privileged_z_branch_outcome_oracle_low_level_return_delta_ge5_seed9940000_100eps_b4.npz`
- `results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_seed9940000_20eps_b2.json`
- `results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_seed9940000_100eps_b4.json`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_branch_return_ge5_seed9940000_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_oracle_low_level_branch_return_ge5_seed9940000_imp01_200eps.json`

### Interpretation

The new branch source confirms the Experiment G warning in a useful way:
competent oracle-low-level branches create a clean return-positive bank, but
distilling those actions into the learned-goal low level still does not beat the
selector-based current best. The oracle-goal result also drops, so this is not
only a learned-high mismatch; the training target likely mixes incompatible
semantics by asking the low level to execute oracle-goal actions while
conditioning on the learned high-level goal.

### Decision

Keep `--branch-source oracle_low_level` as an implemented diagnostic branch
generator, but do not promote the distilled checkpoint. The result suggests
that true branch data should store and train on explicit alternative branch
goals, and likely requires high-level retraining/prototype IDs, rather than
forcing oracle-local actions under the existing learned goal.

## Experiment G probe: explicit oracle branch-goal conditioning

The previous oracle-low-level branch bank used competent branch actions but
stored the learned high-level goal in the low-level condition. That is
semantically inconsistent: the action sequence was generated for an oracle
branch goal while the model was asked to associate it with the learned goal.

I added a second switch to the branch-outcome command:

```text
--branch-condition-goal-source learned_high|oracle_goal
```

For `branch_source=oracle_low_level`, `oracle_goal` stores the teacher's
10-step oracle continuation goal in the bank condition. This directly tests
Experiment G's explicit alternative branch-goal setup.

### Bank Collection

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 100 \
  --seed-start 9940000 \
  --num-envs 100 \
  --branch-source oracle_low_level \
  --branch-condition-goal-source oracle_goal \
  --random-candidates 1 \
  --random-noise-std 0.05 \
  --min-improvement-mse 0.0 \
  --max-action-delta-l2 1.0 \
  --max-branch-batches 4 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_return_delta_ge5_seed9940000_100eps_b4.npz \
  --bank-min-return-delta 5 \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_oraclegoal_seed9940000_100eps_b4.json \
  --force
```

The outcome labels are the same as the learned-goal-conditioned oracle branch
bank because only the stored training goal changed:

| metric | value |
| --- | ---: |
| branches | 100 |
| horizon rows | 1000 |
| success delta mean | 0.410 |
| return delta mean | 24.97 |
| return delta median | 20.77 |

The NPZ stores:

```text
branch_source = oracle_low_level
branch_condition_goal_source = oracle_goal
```

### Distillation

Training:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  train-privileged-z-local-replay-distill \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --manifest data/manifests/local_reset_bank_n1800_seed0_k10_hard_mse_ge_0p05.json \
  --preserve-manifest data/manifests/local_reset_bank_n1800_seed0_k10_easy_mse_lt_0p05.json \
  --preserve-npz data/manifests/privileged_z_closed_loop_preserve_hierarchy_n512_seed9900000.npz \
  --improve-npz data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_return_delta_ge5_seed9940000_100eps_b4.npz \
  --replay-weight 0.25 \
  --preserve-weight 1.0 \
  --preserve-npz-weight 1.0 \
  --improve-npz-weight 0.1 \
  --run-tag hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200 \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 1e-4 \
  --train-scope final_layer \
  --force
```

### Results

Quick 200-episode dev check on `seed_start=9900000`:

| checkpoint | mode | success | return |
| --- | --- | ---: | ---: |
| oracle-low-level explicit oracle goal | hierarchy | 0.535 | 41.54 |
| oracle-low-level explicit oracle goal | oracle_hierarchy | 0.730 | 47.17 |
| oracle-low-level learned-goal condition | hierarchy | 0.565 | 42.76 |
| oracle-low-level learned-goal condition | oracle_hierarchy | 0.685 | 46.20 |
| multi3 state+goal-NN79 current best | hierarchy | 0.575 | 41.89 |
| multi3 state+goal-NN79 current best | oracle_hierarchy | 0.705 | 46.69 |

Because oracle-goal dev success improved, I ran matched oracle 3x500:

| seed start | success | return |
| ---: | ---: | ---: |
| 10000000 | 0.692 | 45.94 |
| 10100000 | 0.730 | 48.30 |
| 10200000 | 0.710 | 48.11 |
| mean | 0.7107 | 47.45 |

Artifacts:

- `data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_return_delta_ge5_seed9940000_100eps_b4.npz`
- `results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_oraclegoal_seed9940000_100eps_b4.json`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_seed10000000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_seed10100000_500eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_seed10200000_500eps.json`

### Interpretation

Explicit oracle branch-goal conditioning fixes the semantic mismatch for
oracle-goal evaluation on the dev slice, but the improvement does not survive
matched 3x500 validation. The mean oracle success `0.7107` is essentially tied
with the current learned-high best's oracle matched score (`0.7100`) and below
the earlier goal-NN79 oracle diagnostic (`0.7327`).

The learned-high hierarchy result drops to `0.535`, which is expected because
the learned high level still emits learned goals, not oracle branch goals.

### Decision

Keep `--branch-condition-goal-source oracle_goal` as the correct way to create
explicit branch-goal banks, but do not promote this checkpoint. The result
supports the next Experiment G conclusion: explicit branch goals are necessary,
but this needs high-level/prototype training to emit those branch goals; simply
adding a small oracle-goal branch bank to the existing low-level distillation is
not enough.

## Experiment G/H bridge: explicit-branch low level plus nearest oracle prototypes

The explicit oracle-goal branch low level works only when supplied oracle-like
goals. I tested whether the existing H4 nearest-oracle prototype projection can
bridge that gap by replacing learned high-level predictions with nearest
teacher-state prototypes at eval time.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --mode hierarchy \
  --episodes 200 \
  --seed-start 9900000 \
  --num-envs 200 \
  --high-goal-projection nearest_oracle_bank \
  --high-goal-bank-episodes 200 \
  --high-goal-bank-seed-start 9800000 \
  --high-goal-bank-num-envs 200 \
  --output results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_projected_200eps.json \
  --force
```

### Results

| checkpoint | high-goal projection | success | return | predicted-to-prototype MSE median | predicted-to-prototype MSE p90 |
| --- | --- | ---: | ---: | ---: | ---: |
| explicit oracle-goal branch low level | none | 0.535 | 41.54 | n/a | n/a |
| explicit oracle-goal branch low level | nearest oracle bank | 0.315 | 34.05 | 0.0396 | 0.3297 |
| current learned-high best | none | 0.575 | 41.89 | n/a | n/a |

Artifact:

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_projected_200eps.json`

### Interpretation

Nearest-neighbor prototype projection remains harmful even with a low level
distilled on explicit oracle branch goals. The selected prototypes are close in
normalized privileged-state MSE, but replacing the continuous learned-high goal
with a nearby teacher-state prototype changes the control semantics enough to
collapse learned-high task success.

### Decision

Reject simple nearest-prototype bridging. The next high-level branch-goal path
needs a trained selector/predictor over branch goals or prototype IDs with
closed-loop outcome supervision; nearest geometry alone is not sufficient.

## Oracle-low-level all-branch selector diagnostic

The explicit oracle-goal branch bank above only kept branches with positive
return deltas. To check whether the rejection filter itself was hiding useful
selector structure, I collected the full oracle-low-level branch set with no
minimum return-delta filter and ran an offline selector diagnostic over branch
features and outcomes.

### Artifacts

- `data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_all_seed9940000_100eps_b4.npz`
- `results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_oraclegoal_all_seed9940000_100eps_b4.json`
- `results/hcl_next_phase1/privileged_z_oracle_low_level_oraclegoal_selector_offline_seed9940000_b4.json`

### Full-branch bank summary

| metric | value |
| --- | ---: |
| branches | 400 |
| rows | 4000 |
| mean success delta | 0.015 |
| mean return delta | 1.524 |
| median return delta | 0.0138 |
| p90 return delta | 25.62 |
| return-positive fraction, delta > 5 | 0.25 |
| locally rejected branch success delta | 0.028 |
| locally rejected median return delta | 0.109 |
| locally rejected better-return fraction | 0.559 |

### Offline selector results

| selector | return-positive AUC | success-positive AUC | top100 return delta mean | top100 success delta mean |
| --- | ---: | ---: | ---: | ---: |
| ridge return-positive score | 0.5826 | n/a | 5.43 | -0.01 |
| ridge success-positive score | n/a | 0.6865 | 5.78 | 0.15 |
| local improvement | 0.450 | n/a | -1.22 | -0.02 |
| negative action delta | 0.408 | n/a | 1.71 | n/a |
| low base rollout return | 0.677 | n/a | 8.65 | 0.20 |

The strongest simple signal is low base rollout return, not local action-space
improvement. Selecting the 40 lowest-base-return branches gives mean return
delta `16.40`, mean success delta `0.275`, and return-positive fraction `0.65`.

### Interpretation

The local held-goal MSE/action improvement signal is anti-correlated or weak for
task-level outcome. The branch outcome label is most predictable from whether
the current closed-loop state is already doing badly. This supports a selector
that is explicitly conditioned on closed-loop failure states, but it also means
the resulting data is narrow and likely to hurt learned-high behavior if used as
a direct low-level replay target.

## Oracle-low-level explicit-goal low-base-return top40 distillation

I distilled the explicit oracle-goal low level using the 40 oracle-low-level
branches selected by lowest base rollout return from the full branch bank.

### Artifacts

- `data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_low_base_return_top40_seed9940000_100eps_b4.npz`
- `artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_low_base_return_top40_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_low_base_return_top40_imp01_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_oracle_oracle_low_level_oraclegoal_low_base_return_top40_imp01_200eps.json`

### Selected branch bank

| metric | value |
| --- | ---: |
| branches | 40 |
| rows | 400 |
| base return mean | 6.923 |
| base return median | 6.656 |
| base return min | 1.170 |
| base return max | 17.403 |
| return delta mean | 16.403 |
| return delta median | 13.248 |
| return delta min | -10.886 |
| return delta max | 60.054 |
| success delta mean | 0.275 |
| return-positive fraction, delta > 5 | 0.65 |

### Evaluation

| mode | episodes | seed start | success | return |
| --- | ---: | ---: | ---: | ---: |
| hierarchy | 200 | 9900000 | 0.515 | 41.69 |
| oracle_hierarchy | 200 | 9900000 | 0.725 | 47.04 |

### Decision

Reject the low-base-return top40 checkpoint. The selector does identify branches
with strong oracle-goal outcome deltas, but direct replay distillation again
hurts the learned-high hierarchy (`0.515` versus current best `0.575`). The
oracle result is close to the explicit oracle-goal branch replay result but does
not exceed the best oracle-goal diagnostic.

The useful retained conclusion is narrower: low base rollout return is a good
diagnostic selector for finding high-upside branch opportunities. It is not by
itself a deployable low-level distillation recipe, because learned-high still
does not emit the branch-goal distribution that made those actions useful.

## Experiment H4: nearest explicit branch-goal projection

After rejecting the generic nearest-oracle-bank projection, I added a more
targeted projection mode for the explicit branch-goal setting:

```bash
--high-goal-projection nearest_branch_goal_bank
--high-goal-branch-bank <branch_outcome_npz>
```

This mode loads branch-bank `conditions`, takes one row per branch, and projects
each learned high-level goal to the stored branch goal with minimum average MSE
over:

```text
current normalized privileged state
learned high-level predicted goal
```

The intent was to test whether learned-high predictions can be bridged to the
same explicit oracle-goal distribution used by the branch low level, without
training a new high-level classifier.

### Evaluations

Both runs use the explicit oracle-goal branch low level:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

| branch projection bank | bank size | success | return | projected MSE median | projected MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| return-delta > 5 selected bank | 100 | 0.070 | 20.46 | 0.615 | 1.502 |
| all oracle-low-level branches | 400 | 0.100 | 23.98 | 0.257 | 1.134 |
| all oracle-low-level branches, state-only match | 400 | 0.050 | 18.96 | 0.318 | 1.889 |

Artifacts:

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_nearest_branch_goal_bank_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_nearest_branch_goal_bank_all400_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_nearest_branch_state_bank_all400_200eps.json`

### Decision

Reject nearest explicit branch-goal projection. Even the full 400-branch bank
collapses learned-high success to `0.100`, far below the unprojected explicit
branch low-level result (`0.535`) and the current learned-high best (`0.575`).
Matching only on current state is worse (`0.050`), so the failure is not just
the learned high-level goal contaminating the nearest-neighbor score.

This rules out a simple non-parametric bridge from learned high-level regression
outputs to sparse branch goals. The next high-level path should train a real
selector/policy over candidate branch goals using closed-loop outcome labels, or
collect a much denser branch bank before revisiting prototype selection.

## Experiment H4 follow-up: denser explicit branch-goal bank

To separate "nearest branch-goal projection is bad" from "the branch bank is too
sparse", I collected a denser explicit oracle-goal branch bank on fresh seeds:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml \
  eval-privileged-z-branch-outcomes \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --episodes 200 \
  --seed-start 9950000 \
  --num-envs 200 \
  --branch-source oracle_low_level \
  --branch-condition-goal-source oracle_goal \
  --min-improvement-mse 0.0 \
  --max-action-delta-l2 100.0 \
  --max-branch-batches 10 \
  --max-rollout-steps 120 \
  --bank-output data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_all_seed9950000_200eps_b10.npz \
  --output results/hcl_next_phase1/privileged_z_branch_outcome_oracle_low_level_oraclegoal_all_seed9950000_200eps_b10.json \
  --force
```

The relaxed local filters make this a true all-branch bank for this source,
rather than the earlier locally accepted subset.

### Dense bank summary

| metric | value |
| --- | ---: |
| branches | 2000 |
| rows | 20000 |
| base closed-loop success | 0.580 |
| base closed-loop return | 43.57 |
| mean success delta | 0.0195 |
| mean return delta | -0.180 |
| median return delta | 0.0 |
| p90 return delta | 17.58 |
| return delta > 5 branches | 328 |
| return delta > 10 branches | 242 |
| success-positive branches | 160 |
| top-return100 mean return delta | 53.24 |
| top-return100 mean success delta | 0.710 |

I added `scripts/filter_privileged_z_branch_bank.py` to create filtered views of
an existing branch-outcome NPZ without rerunning environment rollouts.

Filtered artifacts:

- `data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_return_delta_ge5_seed9950000_200eps_b10.npz`
- `data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_top_return100_seed9950000_200eps_b10.npz`

### Projection evaluations

All runs use the explicit oracle-goal branch low level:

- `artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`

| branch projection bank | bank size | success | return | projected MSE median | projected MSE p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse all bank, previous result | 400 | 0.100 | 23.98 | 0.257 | 1.134 |
| dense all bank | 2000 | 0.285 | 32.54 | 0.126 | 0.716 |
| dense return-delta >= 5 bank | 328 | 0.165 | 25.33 | 0.219 | 0.767 |
| dense top-return100 bank | 100 | 0.040 | 18.94 | 0.648 | 1.507 |

Artifacts:

- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_nearest_branch_goal_bank_dense2000_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_nearest_branch_goal_bank_dense_return_ge5_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_nearest_branch_goal_bank_dense_top_return100_200eps.json`

### Interpretation

Denser coverage helps: median projection MSE drops from `0.257` to `0.126`, and
success improves from `0.100` to `0.285`. But this remains far below the
unprojected learned-high baseline (`0.535` for the explicit branch low level,
`0.575` for the current best learned-high checkpoint).

Outcome filtering does not make the non-parametric bridge deployable. The
return-delta >= 5 bank is worse than the dense all-bank result, and the
top-return100 bank collapses almost completely. High-upside branch goals are
too sparse and state-specific to use as a nearest-neighbor target set.

### Decision

Reject nearest/prototype branch-goal projection even with a 2000-branch bank.
The result is useful diagnostically: branch-bank density matters, but successful
deployment needs a learned high-level selector/policy trained on branch outcomes
or a much broader branch-data distribution. Filtering to only high-return
branches makes coverage worse and is not a substitute for a selector.

## Experiment H4/H5: learned branch-goal selector

I implemented a learned branch-goal selector as the next step after rejecting
nearest-only and outcome-filter-only projection.

New artifacts/code:

- `scripts/train_privileged_z_branch_selector.py`
- `eval-privileged-z --high-goal-projection learned_branch_goal_selector`
- `--high-goal-branch-selector <selector.pt>`

The selector is a small MLP scorer. It scores candidate branch goals using:

```text
current normalized privileged state
learned high-level predicted goal
previous action
candidate branch start state
candidate branch goal
candidate previous action
state/goal/previous deltas and MSEs
candidate outcome priors from the branch bank
```

Training pairs are sampled from the dense 2000-branch bank. The target rewards
candidate return delta but penalizes state/goal mismatch:

```text
target = zscore(candidate_return_delta) - distance_penalty * pair_distance
```

This is a first learned selector over closed-loop branch outcome labels, not a
full high-level RL policy.

### Selector checkpoints

| selector | distance penalty | validation MSE | final train MSE |
| --- | ---: | ---: | ---: |
| `hcl_next_dense2000_penalty1_seed0.pt` | 1 | 0.0078 | 0.0024 |
| `hcl_next_dense2000_penalty5_seed0.pt` | 5 | 0.1468 | 0.0964 |
| `hcl_next_dense2000_penalty10_seed0.pt` | 10 | 0.7155 | 0.8980 |
| `hcl_next_dense2000_penalty20_seed0.pt` | 20 | 3.8361 | 5.3766 |

### Closed-loop evaluation

All runs use:

- base hierarchy: `artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt`
- branch low level: `artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt`
- evaluation: 200 episodes, seed start `9900000`

| high-goal selection | success | return | projected MSE median | projected MSE p90 |
| --- | ---: | ---: | ---: | ---: |
| dense nearest bank baseline | 0.285 | 32.54 | 0.126 | 0.716 |
| learned selector, penalty 1 | 0.035 | 19.47 | 0.527 | 1.512 |
| learned selector, penalty 5 | 0.135 | 24.79 | 0.226 | 0.884 |
| learned selector, penalty 10 | 0.220 | 29.36 | 0.137 | 0.643 |
| learned selector, penalty 20 | 0.250 | 31.13 | 0.083 | 0.522 |

Artifacts:

- `artifacts/incremental/privileged_z_branch_selector/hcl_next_dense2000_penalty1_seed0.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_dense2000_penalty5_seed0.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_dense2000_penalty10_seed0.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_dense2000_penalty20_seed0.pt`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_learned_branch_selector_dense2000_penalty1_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_learned_branch_selector_dense2000_penalty5_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_learned_branch_selector_dense2000_penalty10_200eps.json`
- `results/hcl_next_phase1/privileged_z_closed_loop_hierarchy_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_learned_branch_selector_dense2000_penalty20_200eps.json`

### Interpretation

The learned selector confirms the previous diagnosis. If outcome weight is too
strong, the selector chooses high-return but badly matched branch goals and task
success collapses. Increasing the distance penalty improves closed-loop success
monotonically in this sweep (`0.035 -> 0.250`) and reduces projection distance,
but even penalty 20 remains below the dense nearest-bank baseline (`0.285`) and
far below the unprojected learned-high baseline.

The learned scorer is therefore not yet useful as a high-level replacement. Its
best behavior is to approximate dense nearest-neighbor matching, which itself is
not good enough.

### Decision

Reject this first learned branch-goal selector as a promotion candidate. Keep
the implementation because it is now a reusable harness for branch-outcome
selectors. The next selector should either:

- train on actual counterfactual candidates evaluated for the same query state,
  not synthetic pair labels built from each candidate's own rollout outcome; or
- generate branch goals on-policy from the current learned-high state
  distribution and then train a selector/policy from those query-specific
  outcome labels.

## Experiment H5: query-specific branch-goal counterfactuals

The first learned branch-goal selector failed because each candidate carried its
own rollout outcome from a different source state. I added a query-specific
counterfactual collector:

- `scripts/collect_privileged_z_branch_counterfactuals.py`

For each learned-high replan state, the collector:

1. predicts the current learned high-level goal;
2. retrieves the nearest `k` candidate branch goals from the dense 2000-branch
   bank;
3. evaluates the base learned-high rollout from that exact simulator state;
4. evaluates each candidate branch goal from the same simulator state, using the
   explicit branch-goal low level for the first segment and then base learned
   hierarchy continuation;
5. saves query-specific return/success deltas.

This directly measures the selector target that the previous synthetic selector
was missing.

### Counterfactual banks

| bank | queries | candidates/query | base success | base return | nearest return delta | best-of-k return delta | positive best > 5 | best success delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed9960000_q64_k8` | 64 | 8 | 0.531 | 41.68 | -6.04 | 14.71 | 0.563 | 0.313 |
| `seed9961000_q128_k8` | 128 | 8 | 0.445 | 41.32 | -4.19 | 14.63 | 0.523 | 0.328 |

Artifacts:

- `data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9960000_q64_k8.npz`
- `data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9961000_q128_k8.npz`

### Immediate diagnostics

On the 64-query bank:

| selector | return delta | success delta |
| --- | ---: | ---: |
| nearest candidate | -6.04 | -0.109 |
| max source-return candidate | -6.63 | -0.109 |
| oracle best-of-8 | 14.71 | 0.297 |

The candidate's source-state branch return delta is essentially useless for the
new query (`corr = -0.025` with query-specific return delta). Nearest distance
is also weak (`corr = 0.079` for negative distance).

### Offline query-specific selector

I added a small offline trainer:

- `scripts/train_privileged_z_counterfactual_selector.py`

Training on the 128-query bank, split by query, gives:

| split | learned selector return delta | learned selector success delta | nearest return delta | nearest success delta | oracle return delta | oracle success delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2.70 | 0.073 | -5.07 | -0.094 | 14.13 | 0.323 |
| validation | 3.82 | 0.000 | -1.56 | 0.000 | 16.15 | 0.313 |

Artifact:

- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_seed0.pt`

### Interpretation

This is the clearest positive signal so far for the branch-goal direction:
query-specific best-of-8 candidate goals have large upside from the exact
learned-high states where nearest/prototype projection fails. The nearest
candidate is negative on average, so the issue is not simply branch-bank
coverage; it is candidate choice for the current query.

The first query-specific learned selector improves over nearest on held-out
queries (`+3.82` return delta versus `-1.56`), but captures only a small part of
the oracle best-of-8 upside (`+16.15`). This is not yet deployable, but it
validates the next direction: collect more query-specific counterfactual labels
and train/evaluate a selector from those labels, rather than using candidate
source outcomes or nearest geometry.

### Decision

Do not promote a closed-loop selector yet. Keep the counterfactual collector and
offline selector trainer as the next experimental harness. The next useful run
is a larger query-specific bank, then either:

- train a selector that can be plugged into `eval-privileged-z`, or
- collect on-policy candidate sets from the selector itself and iterate.

## Experiment H5 follow-up: larger counterfactual bank and grouped selector loss

I expanded the query-specific counterfactual bank and made the offline selector
artifact more deployment-oriented. `scripts/train_privileged_z_counterfactual_selector.py`
now stores the source checkpoint, tuned checkpoint, branch-bank path, rollout
settings, and baseline selection metrics in the selector checkpoint. I also
added a grouped best-candidate classification objective:

```bash
--loss best_ce
```

This trains the model to choose the candidate with maximum query-specific return
delta from the candidate set, instead of regressing return delta independently
for every candidate.

### Larger counterfactual bank

```bash
uv run scripts/collect_privileged_z_branch_counterfactuals.py \
  --config configs/pusht_incremental.yaml \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --branch-bank data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_all_seed9950000_200eps_b10.npz \
  --output data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9962000_q256_k8.npz \
  --seed-start 9962000 \
  --num-envs 64 \
  --query-batches 4 \
  --candidates-per-query 8 \
  --max-rollout-steps 120
```

| bank | queries | candidates/query | base success | base return | nearest return delta | nearest success delta | oracle return delta | oracle success delta | positive best > 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed9962000_q256_k8` | 256 | 8 | 0.445 | 44.07 | -0.531 | 0.004 | 13.03 | 0.219 | 0.469 |

Artifact:

- `data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9962000_q256_k8.npz`

The larger bank preserves the important conclusion: the candidate set contains
large query-specific upside, but nearest selection still does not extract it.

### Selector objectives on q256/k8

| selector objective | validation selected return delta | validation selected success delta | validation nearest return delta | validation source-return-argmax return delta | validation oracle return delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| return regression (`mse`) | 0.118 | 0.031 | -0.151 | -1.666 | 13.07 |
| grouped best-candidate CE (`best_ce`) | -3.941 | -0.063 | -0.151 | -1.666 | 13.07 |

Artifacts:

- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q256_k8_seed0.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q256_k8_bestce_seed0.pt`

### Interpretation

More query-specific data did not make the simple selector deployable. Return
regression remains slightly better than nearest on validation, but only by a
small amount (`+0.118` versus `-0.151`) and far below the oracle candidate-set
upper bound (`+13.07`). The grouped CE objective is worse, likely because the
best candidate label is too noisy/discontinuous for a small dataset and the
available features.

This narrows the next branch-goal selector direction: the problem is not just
the loss function. The selector probably needs either more query coverage, a
better candidate-generation distribution, or richer features from actual
candidate rollout prefixes/final states. The current static features
`(query, candidate goal, source outcome)` are not sufficient to recover the
best candidate.

### Decision

Keep the counterfactual collector and selector scripts, but do not spend more
time on small static-feature selector losses. The next useful experiment should
change the data/modeling problem, for example:

- collect candidate rollout prefix/final-state features for each query;
- train a selector on the actual local candidate outcome after the first
  segment, not only branch-goal identity;
- or move back to reachability/effect-latent experiments now that the
  privileged branch-goal path has exposed its current bottleneck.

## Effect32 FiLM D_phi R3 fresh-seed confirmation

After the branch-goal selector bottleneck, I returned to the real-compatible
effect/reachability path. Existing 300-episode effect32 D_phi evals on
`seed_start=3400000` showed only a small margin:

| policy | 300-episode success | max reward | raw local reduction | reach rate |
| --- | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.633 | 0.740 | 0.390 | 0.735 |
| R1 terminal smoke 40k | 0.650 | 0.749 | 0.367 | 0.716 |
| R3 terminal smoke 40k bc10 | 0.643 | 0.746 | 0.397 | 0.739 |

I first tried to run frozen/R1/R3 500-episode confirmation evals in parallel on
fresh `seed_start=3500000`, but effect-code evals repeatedly encode image
features and the parallel jobs contended heavily. I interrupted those jobs and
reran serially.

### Fresh 200-episode check

| policy | success | max reward | raw local reduction | reach rate | terminal AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.655 | 0.756 | 0.447 | 0.726 | 0.804 |
| R1 terminal smoke 40k | 0.655 | 0.755 | 0.396 | 0.726 | 0.832 |
| R3 terminal smoke 40k bc10 | 0.675 | 0.766 | 0.389 | 0.728 | 0.800 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final200_seed3500000/eval_200_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r1_4096_terminal_smoke_40k_final200_seed3500000/eval_200_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final200_seed3500000/eval_200_seed3500000.json`

### Fresh 500-episode confirmation

The R1 run tied frozen on the fresh 200-episode check, so I spent the larger
confirmation budget on frozen versus the R3 bc10 candidate.

| policy | checkpoint | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | none | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.000 |
| R3 terminal smoke 40k bc10 | `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt` | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | small direct update |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3500000/eval_500_seed3500000.json`

### Decision

Promote:

`artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt`

as the current best real-compatible effect-latent local RL checkpoint. Unlike
the earlier R1 effect result, this fresh 500-episode confirmation shows a clear
task improvement over frozen:

```text
R3 effect32 D_phi final success: 0.684
frozen effect32 final success:   0.634
```

The improvement also comes with better max reward, raw local reduction, and
reach rate. This is now the strongest evidence that the learned
effect/reachability path can improve a supervised hierarchy without privileged
state at deployment. The next useful step is to compare this checkpoint against
the supervised learned/oracle/shuffled effect32 baselines on matched final seeds
and then investigate why the R3 40k recipe works better than the longer R1 dev
runs.

## Effect32 FiLM matched supervised baseline comparison

I added `--eval-seed-start` to `learned-interface-eval` so learned/oracle/shuffled
supervised hierarchy baselines can be run on the same seed range as low-level
RL confirmations. When a custom seed is supplied, the output filename includes
the seed start to avoid overwriting default-seed results.

### Command template

```bash
uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film \
  --goal-source learned|oracle|shuffled \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force
```

### Matched fresh-seed supervised baselines

| evaluator | policy / goal source | episodes | seed start | success | max reward | final reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| learned-interface | learned | 200 | 3500000 | 0.645 | 0.742 | 0.735 |
| learned-interface | oracle | 200 | 3500000 | 0.645 | 0.746 | 0.740 |
| learned-interface | shuffled | 200 | 3500000 | 0.280 | 0.460 | 0.425 |

Artifacts:

- `results/incremental/learned_interface/effect32_film/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film/seed0/shuffled_hierarchy_eval_200_seed3500000.json`

The shuffled ablation still confirms strong closed-loop goal dependence on this
fresh seed bank (`0.645 -> 0.280`). Oracle goals do not improve success over
learned goals on this seed range, but do slightly improve reward and teacher
MAE.

### RL comparison on same seed range

The promoted RL checkpoint is evaluated through the low-level RL evaluator, not
the learned-interface evaluator, so the comparison is not a byte-identical code
path. The frozen low-level eval gives a close supervised reference on the same
wrapper:

| evaluator | policy | episodes | seed start | success | max reward | raw local reduction | reach rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low-level-rl | frozen effect32_film | 500 | 3500000 | 0.634 | 0.738 | 0.397 | 0.718 |
| low-level-rl | R3 effect32 D_phi bc10 | 500 | 3500000 | 0.684 | 0.773 | 0.410 | 0.731 |

### Interpretation

The promoted R3 effect32 D_phi checkpoint improves over the frozen low-level
reference by `+0.050` success on 500 fresh episodes. It also sits above the
200-episode learned-interface learned/oracle baselines on the same seed start
(`0.684` versus `0.645`), though that cross-evaluator comparison should be
treated as secondary evidence.

The main conclusion is robust enough for the current proof of concept:
effect32_film remains goal-dependent in closed loop, and the D_phi R3 update
can improve task performance without privileged deployment state.

### Decision

Keep `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt`
as the current best real-compatible RL checkpoint. The next technical question
is no longer whether effect32 D_phi can produce a positive result; it is why the
short R3 bc10 recipe works while longer R1 runs do not. Useful next checks:

- inspect train curves/checkpoint selection for the R3 bc10 run;
- run a matched R3 bc10 repeat with a different seed or slightly longer budget;
- compare R3 direct action deltas against R1 residual deltas to see whether R1
  is under-moving or moving in the wrong local directions.

## Effect32 FiLM R3 200k continuation diagnostic

I ran a direct follow-up to test whether the promoted short R3 recipe keeps
improving when trained for the same 200k-step budget used by the longer R1 dev
runs.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10 \
  --steps 204800 \
  --num-envs 4096 \
  --rollout-steps 10 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --bc-weight 10 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --force
```

The train curve improved its own terminal-distance selection objective by the
final update:

| step | train terminal D_phi distance | direct delta L2 | BC loss | action saturation |
| ---: | ---: | ---: | ---: | ---: |
| 40960 | 0.576 | 0.0293 | 0.000001 | 0.259 |
| 81920 | 0.612 | 0.0293 | 0.000001 | 0.094 |
| 122880 | 0.636 | 0.0292 | 0.000001 | 0.008 |
| 163840 | 0.584 | 0.0293 | 0.000003 | 0.006 |
| 204800 | 0.557 | 0.0292 | 0.000002 | 0.002 |

Then I evaluated the selected best checkpoint on the same fresh 500-episode
seed bank used for the promoted 40k checkpoint:

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_final500_seed3500000 \
  --episodes 500 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10/best_train_latent.pt \
  --distance-metric reachability \
  --force
```

### Matched 500-episode result

| policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| R3 terminal smoke 40k bc10 | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| R3 terminal dev 200k bc10 | 0.656 | 0.753 | 0.418 | 0.723 | 0.793 | 0.0026 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_final500_seed3500000/eval_500_seed3500000.json`

### Interpretation

The 200k R3 run remains better than frozen, but it gives back most of the
40k task-success gain (`0.684 -> 0.656`) even while its train terminal D_phi
distance improves (`0.576 -> 0.557`). That means the current train-selection
proxy is not aligned well enough with full-task success for long PPO runs.

The useful result is the 40k update, not the longer checkpoint. The direct R3
method can produce a small but real improvement, but further training appears
to over-optimize the local reachability proxy. For the next stage, keep the
40k checkpoint promoted and prefer either:

- early stopping/checkpoint selection by held-out rollout success or reward,
  not train terminal D_phi distance; or
- a different reachability objective that includes task-relevant local progress
  rather than only terminal latent distance.

## Effect32 FiLM R3 checkpoint-selection audit

Because the 200k R3 run saved update checkpoints, I evaluated each checkpoint
on a cheap 100-episode held-out bank to test whether held-out task success
would select a better checkpoint than train terminal D_phi distance.

### 100-episode sweep, seed start 3400000

| checkpoint step | train terminal D_phi | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40960 | 0.576 | 0.630 | 0.739 | 0.374 | 0.709 | 0.814 | 0.0009 |
| 81920 | 0.612 | 0.660 | 0.757 | 0.427 | 0.720 | 0.775 | 0.0015 |
| 122880 | 0.636 | 0.610 | 0.720 | 0.411 | 0.733 | 0.822 | 0.0016 |
| 163840 | 0.584 | 0.690 | 0.779 | 0.386 | 0.722 | 0.818 | 0.0020 |
| 204800 | 0.557 | 0.600 | 0.718 | 0.427 | 0.728 | 0.792 | 0.0027 |

The 100-episode held-out sweep would have selected the 163840-step checkpoint.
I therefore ran a 500-episode confirmation on the same fresh seed bank used for
the promoted checkpoint.

### 500-episode confirmation, seed start 3500000

| policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| R3 terminal smoke 40k bc10 | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| R3 dev 200k step163840 | 0.626 | 0.734 | 0.421 | 0.722 | 0.799 | 0.0020 |
| R3 dev 200k best-train/final | 0.656 | 0.753 | 0.418 | 0.723 | 0.793 | 0.0026 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_step000040960_eval100_seed3400000/eval_100_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_step000081920_eval100_seed3400000/eval_100_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_step000122880_eval100_seed3400000/eval_100_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_step000163840_eval100_seed3400000/eval_100_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_step000204800_eval100_seed3400000/eval_100_seed3400000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_dev_200k_bc10_step163840_final500_seed3500000/eval_500_seed3500000.json`

### Interpretation

The 100-episode held-out sweep is too noisy for reliable checkpoint selection:
it selected step 163840 at `0.690` success, but that checkpoint fell to
`0.626` on the 500-episode fresh bank. The final 200k train-selected checkpoint
is still above frozen on the 500-episode bank, but it remains weaker than the
original 40k promoted checkpoint.

Decision remains unchanged: keep the 40k R3 bc10 checkpoint promoted. The
checkpoint-selection lesson is sharper now: if early stopping is added to the
training loop, the held-out bank must be large enough to avoid selecting noise.
For this setup, 100 episodes is not enough; 500 episodes was the first bank
that clearly separated the promoted 40k checkpoint from later R3 checkpoints.

## Effect32 FiLM R3 dense-progress reward check

The 200k continuation showed that improving terminal `D_phi` distance does not
necessarily improve task success. I ran one short reward-alignment check: keep
the promoted R3 direct-last-layer recipe, but add dense `D_phi` progress reward
back on top of the terminal segment-end penalty.

### Command

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_progress_smoke_40k_bc10 \
  --steps 40960 \
  --num-envs 4096 \
  --rollout-steps 10 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --bc-weight 10 \
  --terminal-weight 1.0 \
  --distance-progress-weight 1.0 \
  --distance-metric reachability \
  --force
```

The single train row was nearly identical to the terminal-only 40k run in
distance/action terms, but the reward scale changed because the dense progress
term was active:

| mean reward | terminal D_phi distance | direct delta L2 | BC loss | action saturation |
| ---: | ---: | ---: | ---: | ---: |
| -0.0363 | 0.5757 | 0.0293 | 0.000001 | 0.259 |

### Matched 500-episode result

| policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| R3 terminal-only 40k bc10 | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| R3 terminal+progress 40k bc10 | 0.662 | 0.754 | 0.410 | 0.713 | 0.787 | 0.0010 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_progress_smoke_40k_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_progress_smoke_40k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_progress_smoke_40k_bc10_final500_seed3500000/eval_500_seed3500000.json`

### Interpretation

Adding dense `D_phi` progress did not recover the terminal-only 40k result. It
still beats frozen on the matched 500-episode bank (`0.662` versus `0.634`), but
it is weaker than terminal-only (`0.684`). The raw local distance reduction is
similar, while reach rate and terminal AUC are lower.

This supports the earlier suspicion that dense learned-metric progress can
reward local metric artifacts that are not task-useful. For effect32 R3, the
best short recipe remains terminal-only `D_phi` with a strong BC anchor.

## Effect32 FiLM R3 PPO-seed robustness check

The promoted R3 result used the same frozen effect32 hierarchy seed and the
default low-level PPO seed. To test whether the 40k terminal-only gain is
robust to PPO initialization and train rollout seeds, I added a separate
`--rl-seed-offset` option to low-level R1/R3 training. The default remains
`0`, so existing recipes are unchanged. The offset is recorded in the
checkpoint recipe and shifts both:

- the PyTorch/NumPy/random seed used for PPO initialization;
- the vectorized training environment seed window.

### Repeat commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --rl-seed-offset 1 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed1 \
  --steps 40960 \
  --num-envs 4096 \
  --rollout-steps 10 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --bc-weight 10 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --force
```

The same command was repeated with `--rl-seed-offset 2` and run name
`hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed2`.

### Matched 500-episode result

All checkpoints were evaluated on the same fresh seed bank:
`episodes=500`, `seed_start=3500000`.

| policy | PPO seed offset | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | n/a | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| R3 terminal-only 40k bc10 | 0 | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| R3 terminal-only 40k bc10 | 1 | 0.662 | 0.753 | 0.394 | 0.706 | 0.794 | 0.0011 |
| R3 terminal-only 40k bc10 | 2 | 0.610 | 0.719 | 0.402 | 0.711 | 0.806 | 0.0008 |

Across the three PPO seeds:

| metric | value |
| --- | ---: |
| mean success | 0.652 |
| min success | 0.610 |
| max success | 0.684 |
| frozen success | 0.634 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed1/best_train_latent.pt`
- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed2/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed1_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed2_final500_seed3500000/eval_500_seed3500000.json`

### Interpretation

The effect32 R3 terminal-only recipe is promising but not yet robust. The
three-seed mean is above frozen (`0.652` versus `0.634`), and two of three PPO
seeds beat frozen, but the third seed falls below frozen (`0.610`). This means
the original `0.684` should be treated as the best observed checkpoint, not as a
stable expected improvement.

Decision:

- keep `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt`
  as the best real-compatible checkpoint;
- do not claim the recipe is robust yet;
- next improvement should target variance reduction or stronger selection, for
  example evaluating multiple short PPO seeds and promoting by a sufficiently
  large held-out bank, or adding a better local/task-aligned selection signal.

## Effect32 FiLM R3 two-bank robustness check

The first PPO-seed repeat used one fresh 500-episode evaluation bank
(`seed_start=3500000`). To separate PPO-seed instability from evaluation-window
noise, I evaluated the frozen policy and all three 40k R3 PPO seeds on a second
fresh 500-episode bank:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --episodes 500 \
  --seed-start 3600000 \
  --distance-metric reachability \
  --force
```

For tuned policies, the same command was run with each `best_train_latent.pt`
checkpoint and matching output run names:

- `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10`
- `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed1`
- `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed2`

### Per-bank results

| eval seed start | policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3500000 | frozen | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| 3500000 | R3 seed 0 | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| 3500000 | R3 seed 1 | 0.662 | 0.753 | 0.394 | 0.706 | 0.794 | 0.0011 |
| 3500000 | R3 seed 2 | 0.610 | 0.719 | 0.402 | 0.711 | 0.806 | 0.0008 |
| 3600000 | frozen | 0.662 | 0.760 | 0.389 | 0.725 | 0.805 | 0.0000 |
| 3600000 | R3 seed 0 | 0.638 | 0.743 | 0.389 | 0.720 | 0.800 | 0.0010 |
| 3600000 | R3 seed 1 | 0.646 | 0.745 | 0.399 | 0.723 | 0.802 | 0.0011 |
| 3600000 | R3 seed 2 | 0.640 | 0.740 | 0.405 | 0.712 | 0.805 | 0.0008 |

### Two-bank aggregate

| policy | success values | mean success | mean max reward |
| --- | --- | ---: | ---: |
| frozen | 0.634, 0.662 | 0.648 | 0.749 |
| R3 seed 0 | 0.684, 0.638 | 0.661 | 0.758 |
| R3 seed 1 | 0.662, 0.646 | 0.654 | 0.749 |
| R3 seed 2 | 0.610, 0.640 | 0.625 | 0.730 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed1_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed2_final500_seed3600000/eval_500_seed3600000.json`

### Interpretation

The second bank is a negative robustness result: frozen reaches `0.662`, while
all three tuned R3 seeds are lower (`0.638`, `0.646`, `0.640`). Over two
500-episode banks, the best R3 seed still has a small positive mean over frozen
(`0.661` versus `0.648`), but the margin is only `+0.013` and is not stable
across evaluation windows.

This revises the effect32 R3 conclusion:

- the original `0.684` checkpoint remains the best observed real-compatible RL
  checkpoint;
- the recipe does not yet provide a robust expected improvement over frozen;
- future work should prioritize either better selection signals or a formulation
  that yields a larger effect size before spending more compute on long PPO
  training.

## Effect32 FiLM R3 paired episode-transition audit

The two-bank aggregate only shows mean success. Since the eval banks use matched
episode order, I compared per-episode success transitions between frozen and
each R3 PPO seed. This asks whether R3 consistently fixes frozen failures, or
whether it trades off gains and regressions on different episodes.

### Paired transitions versus frozen

| eval seed start | policy | frozen fail -> tuned success | frozen success -> tuned fail | net paired wins |
| ---: | --- | ---: | ---: | ---: |
| 3500000 | R3 seed 0 | 127 | 102 | +25 |
| 3500000 | R3 seed 1 | 123 | 109 | +14 |
| 3500000 | R3 seed 2 | 103 | 115 | -12 |
| 3600000 | R3 seed 0 | 102 | 114 | -12 |
| 3600000 | R3 seed 1 | 105 | 113 | -8 |
| 3600000 | R3 seed 2 | 98 | 109 | -11 |

Combined over both 500-episode banks:

| policy | improvements | regressions | net | discordant episodes | normal approx z |
| --- | ---: | ---: | ---: | ---: | ---: |
| R3 seed 0 | 229 | 216 | +13 | 445 | 0.616 |
| R3 seed 1 | 228 | 222 | +6 | 450 | 0.283 |
| R3 seed 2 | 201 | 224 | -23 | 425 | -1.116 |

### Multi-seed diversity upper bound

The three R3 seeds are highly complementary, but without a selector this is not
deployable:

| constructed outcome over R3 seeds | combined success | improvements vs frozen | regressions vs frozen | net |
| --- | ---: | ---: | ---: | ---: |
| any R3 seed succeeds | 0.947 | 324 | 25 | +299 |
| at least two R3 seeds succeed | 0.714 | 243 | 177 | +66 |
| all R3 seeds succeed | 0.279 | 91 | 460 | -369 |

The oracle `any R3 seed succeeds` number is not a policy; it is only an upper
bound showing that the PPO seeds fail on different episodes. This is useful
because it suggests there is real behavioral diversity, but the current
evaluation artifacts do not contain enough per-episode state/reward detail to
train a practical selector.

### Evaluator schema update

I updated `low-level-rl eval` to save:

- `episode_final_reward`
- `episode_max_reward`

alongside the existing `episode_success`. A 20-episode schema smoke confirmed
that the arrays are present and aggregate correctly:

| field | length | mean |
| --- | ---: | ---: |
| episode_success | 20 | 0.400 |
| episode_final_reward | 20 | 0.000 |
| episode_max_reward | 20 | 0.549 |

Artifact:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_detail_smoke_20_seed3700000/eval_20_seed3700000.json`

### Interpretation

The tuned R3 seeds do not consistently improve the same matched episodes. Each
seed creates many wins, but nearly as many regressions. That explains why mean
success is unstable across evaluation windows and why a small train-distance
improvement is not enough to promote a recipe.

The immediate next requirement for robust improvement is not a longer PPO run;
it is a selector/gating signal that can distinguish when the tuned low level is
likely to help. The new per-episode reward arrays are a small step toward that
diagnostic, but future evaluator work should also record compact initial-state
and high-level goal features if we want to train an actual gate.

## Effect32 FiLM R3 action-ensemble diagnostic

The paired transition audit showed that the three R3 PPO seeds fail on
different episodes. I tested a deployable variance-reduction baseline before
building a more complex selector: average the deterministic action outputs of
the three 40k R3 checkpoints at every control step.

I added `--ensemble-checkpoints` to `low-level-rl eval`. It currently supports
R3 direct-last-layer checkpoints with matching distance/reachability recipes.
The evaluator loads each checkpoint and executes:

```text
a_ensemble(o, g, a_prev, tau) = mean_i a_i(o, g, a_prev, tau)
```

No new training is involved.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_ensemble3_final500_seed3500000 \
  --episodes 500 \
  --seed-start 3500000 \
  --ensemble-checkpoints \
    artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
    artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed1/best_train_latent.pt \
    artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_rlseed2/best_train_latent.pt \
  --distance-metric reachability \
  --force
```

The same command was run with `seed_start=3600000`.

### Two-bank result

| policy | success values | mean success | mean max reward |
| --- | --- | ---: | ---: |
| frozen | 0.634, 0.662 | 0.648 | 0.749 |
| R3 seed 0 | 0.684, 0.638 | 0.661 | 0.758 |
| R3 seed 1 | 0.662, 0.646 | 0.654 | 0.749 |
| R3 seed 2 | 0.610, 0.640 | 0.625 | 0.730 |
| R3 action ensemble | 0.618, 0.650 | 0.634 | 0.738 |

Paired against frozen over both banks:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| R3 action ensemble | 212 | 226 | -14 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_ensemble3_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_ensemble3_final500_seed3600000/eval_500_seed3600000.json`

### Interpretation

Naive action averaging does not solve the variance problem. It reduces action
delta magnitude (`~0.0006`, smaller than each individual R3 seed), but it also
reduces task success below frozen on the two-bank aggregate (`0.634` versus
`0.648`). The R3 seed diversity is not a simple high-frequency noise problem
that can be fixed by averaging actions.

This strengthens the previous conclusion: robust improvement needs a state/goal
conditioned gate or selector, not an unconditional ensemble. The ensemble eval
path remains useful as a cheap baseline for future multi-checkpoint variants.

## Effect32 FiLM R3 residual-L2 gate diagnostic

The action-ensemble result showed that unconditional averaging is not enough.
I then added more per-episode detail to `low-level-rl eval` so simple gating
signals can be audited:

- `episode_residual_l2_mean`
- `episode_action_saturation_rate`
- `episode_raw_segment_distance_reduction`
- `episode_segment_distance_reduction`
- `episode_segment_goal_reach_rate`

A 20-episode schema smoke passed:

| field | length | mean |
| --- | ---: | ---: |
| episode_success | 20 | 0.600 |
| episode_max_reward | 20 | 0.712 |
| episode_residual_l2_mean | 20 | 0.00106 |
| episode_action_saturation_rate | 20 | 0.032 |
| episode_raw_segment_distance_reduction | 20 | 0.351 |
| episode_segment_distance_reduction | 20 | 0.069 |
| episode_segment_goal_reach_rate | 20 | 0.685 |

Artifact:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_detail_smoke_20_seed3710000/eval_20_seed3710000.json`

### One-bank diagnostic

I re-evaluated frozen and the best R3 seed on the first 500-episode bank using
new detail run names:

- `hcl_next_effect32_dphi_frozen_detail500_seed3500000`
- `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_detail500_seed3500000`

The paired transition counts were close to the previous bank:

| base success | tuned success | improvements | regressions | same success | same fail |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.632 | 0.684 | 122 | 96 | 220 | 62 |

Feature means by transition class:

| feature | improve | regress | same success | same fail |
| --- | ---: | ---: | ---: | ---: |
| residual L2 | 0.00101 | 0.00104 | 0.00098 | 0.00108 |
| saturation | 0.0309 | 0.0240 | 0.0334 | 0.0268 |
| raw local reduction | 0.441 | 0.214 | 0.500 | 0.224 |
| D_phi reduction | 0.091 | 0.050 | 0.092 | 0.032 |
| reach rate | 0.795 | 0.547 | 0.795 | 0.494 |
| max reward, post-hoc | 1.000 | 0.277 | 0.997 | 0.281 |

Among discordant episodes, high raw local reduction and high reach rate predict
tuned improvements reasonably well (`AUC=0.758` and `0.736`), but those signals
are only known after trying the tuned rollout. Residual L2 alone is weak
(`AUC=0.448`), but a conservative residual threshold is at least deployable as
an online action gate.

### Residual-L2 gate

I added an eval-only option:

```bash
--residual-l2-gate-max 0.00121
```

At each control step, if the tuned action differs from the frozen base action
by more than the threshold, the evaluator executes the frozen base action for
that environment step. The threshold came from a one-bank post-hoc scan, so the
`seed_start=3500000` result should be treated as selection data and
`seed_start=3600000` as the first held-out check.

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3500000 \
  --episodes 500 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --residual-l2-gate-max 0.00121 \
  --distance-metric reachability \
  --force
```

The same command was run with `seed_start=3600000`.

### Gate result

| eval seed start | policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3500000 | frozen | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| 3500000 | R3 ungated | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| 3500000 | R3 residual gate | 0.684 | 0.773 | 0.398 | 0.733 | 0.806 | 0.0004 |
| 3600000 | frozen | 0.662 | 0.760 | 0.389 | 0.725 | 0.805 | 0.0000 |
| 3600000 | R3 ungated | 0.638 | 0.743 | 0.389 | 0.720 | 0.800 | 0.0010 |
| 3600000 | R3 residual gate | 0.668 | 0.765 | 0.393 | 0.738 | 0.814 | 0.0004 |

Two-bank aggregate:

| policy | success values | mean success | mean max reward |
| --- | --- | ---: | ---: |
| frozen | 0.634, 0.662 | 0.648 | 0.749 |
| R3 ungated | 0.684, 0.638 | 0.661 | 0.758 |
| R3 residual gate | 0.684, 0.668 | 0.676 | 0.769 |

Paired against frozen:

| eval seed start | improvements | regressions | net |
| ---: | ---: | ---: | ---: |
| 3500000 | 114 | 89 | +25 |
| 3600000 | 106 | 103 | +3 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_detail500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_detail500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3600000/eval_500_seed3600000.json`

### Interpretation

This is the first simple deployable gate that improves the two-bank aggregate.
It preserves the original positive bank and turns the previously negative
second bank positive:

```text
frozen two-bank mean:        0.648
ungated R3 seed0 mean:      0.661
residual-gated R3 mean:     0.676
```

The result should not be overclaimed because the threshold was selected on the
first bank and validated on only one additional bank. Still, it is a useful
direction: a tiny action-delta gate reduces disruptive R3 updates while keeping
much of the upside. The next validation should test nearby thresholds and more
evaluation windows before promoting gated R3 as the best real-compatible
policy.

## Effect32 FiLM R3 residual-L2 gate third-bank validation

I evaluated the same residual-L2 gate on a third fresh 500-episode bank
(`seed_start=3700000`) to check whether the gate generalizes beyond the first
held-out bank.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3700000 \
  --episodes 500 \
  --seed-start 3700000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --residual-l2-gate-max 0.00121 \
  --distance-metric reachability \
  --force
```

Matched frozen and ungated R3 evals were also run on the same bank:

- `hcl_next_effect32_dphi_frozen_final500_seed3700000`
- `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3700000`

### Three-bank result

| eval seed start | policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3500000 | frozen | 0.634 | 0.738 | 0.397 | 0.718 | 0.806 | 0.0000 |
| 3500000 | R3 ungated | 0.684 | 0.773 | 0.410 | 0.731 | 0.805 | 0.0010 |
| 3500000 | R3 residual gate | 0.684 | 0.773 | 0.398 | 0.733 | 0.806 | 0.0004 |
| 3600000 | frozen | 0.662 | 0.760 | 0.389 | 0.725 | 0.805 | 0.0000 |
| 3600000 | R3 ungated | 0.638 | 0.743 | 0.389 | 0.720 | 0.800 | 0.0010 |
| 3600000 | R3 residual gate | 0.668 | 0.765 | 0.393 | 0.738 | 0.814 | 0.0004 |
| 3700000 | frozen | 0.622 | 0.729 | 0.393 | 0.715 | 0.811 | 0.0000 |
| 3700000 | R3 ungated | 0.638 | 0.740 | 0.399 | 0.714 | 0.789 | 0.0010 |
| 3700000 | R3 residual gate | 0.650 | 0.746 | 0.389 | 0.719 | 0.792 | 0.0004 |

Aggregate:

| policy | success values | mean success | mean max reward |
| --- | --- | ---: | ---: |
| frozen | 0.634, 0.662, 0.622 | 0.639 | 0.743 |
| R3 ungated | 0.684, 0.638, 0.638 | 0.653 | 0.752 |
| R3 residual gate | 0.684, 0.668, 0.650 | 0.667 | 0.761 |

Paired against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| R3 ungated | 338 | 317 | +21 |
| R3 residual gate | 346 | 304 | +42 |

Per-bank paired net:

| eval seed start | R3 ungated net | R3 residual gate net |
| ---: | ---: | ---: |
| 3500000 | +25 | +25 |
| 3600000 | -12 | +3 |
| 3700000 | +8 | +14 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final500_seed3700000/eval_500_seed3700000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3700000/eval_500_seed3700000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3700000/eval_500_seed3700000.json`

### Interpretation

The third bank supports the residual-L2 gate. The gate is positive on all three
500-episode windows and doubles the paired net win count relative to ungated R3
(`+42` versus `+21`). It also gives the best three-bank mean success observed
for the real-compatible effect32 path so far:

```text
frozen:             0.639
ungated R3 seed0:   0.653
residual-gated R3:  0.667
```

This is still a modest effect, but it is now more stable than the ungated R3
checkpoint. The next sensible check is a small threshold sweep around `0.00121`
on held-out banks to ensure the gain is not an overly sharp threshold artifact.

## Effect32 FiLM R3 residual-L2 gate threshold sweep

I swept nearby residual-L2 gate thresholds around `0.00121` on the same three
500-episode banks. The policy and checkpoint are unchanged; only the online
fallback threshold changes.

### Command template

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --episodes 500 \
  --seed-start 3500000|3600000|3700000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --residual-l2-gate-max THRESHOLD \
  --distance-metric reachability \
  --force
```

Tested thresholds:

```text
0.0008, 0.0010, 0.00121, 0.0014, 0.0018
```

### Three-bank aggregate

| policy | success values | mean success | mean max reward | action delta | paired improvements | paired regressions | net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.634, 0.662, 0.622 | 0.639 | 0.743 | 0.00000 | n/a | n/a | n/a |
| ungated R3 | 0.684, 0.638, 0.638 | 0.653 | 0.752 | 0.00100 | 338 | 317 | +21 |
| gate 0.0008 | 0.668, 0.668, 0.626 | 0.654 | 0.752 | 0.00026 | 337 | 315 | +22 |
| gate 0.0010 | 0.626, 0.614, 0.628 | 0.623 | 0.729 | 0.00035 | 324 | 349 | -25 |
| gate 0.00121 | 0.684, 0.668, 0.650 | 0.667 | 0.761 | 0.00044 | 346 | 304 | +42 |
| gate 0.0014 | 0.650, 0.646, 0.654 | 0.650 | 0.750 | 0.00052 | 331 | 315 | +16 |
| gate 0.0018 | 0.658, 0.634, 0.664 | 0.652 | 0.749 | 0.00068 | 335 | 316 | +19 |

Per-bank paired net:

| policy | seed 3500000 | seed 3600000 | seed 3700000 |
| --- | ---: | ---: | ---: |
| gate 0.0008 | +17 | +3 | +2 |
| gate 0.0010 | -4 | -24 | +3 |
| gate 0.00121 | +25 | +3 | +14 |
| gate 0.0014 | +8 | -8 | +16 |
| gate 0.0018 | +12 | -14 | +21 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0008_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0008_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0008_final500_seed3700000/eval_500_seed3700000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0010_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0010_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0010_final500_seed3700000/eval_500_seed3700000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0014_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0014_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0014_final500_seed3700000/eval_500_seed3700000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0018_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0018_final500_seed3600000/eval_500_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate0018_final500_seed3700000/eval_500_seed3700000.json`

### Interpretation

The threshold sweep supports the residual-L2 gate, but it also shows the effect
is threshold-sensitive. `0.00121` remains the best tested value, with the best
mean success, max reward, and paired net wins. A too-strict threshold (`0.0010`)
falls below frozen, so this is not simply "smaller updates are always better."

The practical conclusion is:

- promote residual-gated R3 at `0.00121` as the current best real-compatible
  effect32 policy variant;
- keep the claim modest because the threshold was selected from these eval
  banks;
- for a final claim, validate `0.00121` on additional fresh seed windows or
  choose the threshold by a separate held-out selector bank before final test
  reporting.

## Effect32 FiLM R3 residual-L2 gate five-bank validation

I evaluated the selected residual-L2 gate threshold on two additional fresh
500-episode banks, `seed_start=3800000` and `seed_start=3900000`, using matched
frozen and ungated R3 references.

### Command template

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --episodes 500 \
  --seed-start 3800000|3900000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --residual-l2-gate-max 0.00121 \
  --distance-metric reachability \
  --force
```

### Five-bank result

| eval seed start | frozen | R3 ungated | R3 residual gate |
| ---: | ---: | ---: | ---: |
| 3500000 | 0.634 | 0.684 | 0.684 |
| 3600000 | 0.662 | 0.638 | 0.668 |
| 3700000 | 0.622 | 0.638 | 0.650 |
| 3800000 | 0.684 | 0.652 | 0.654 |
| 3900000 | 0.654 | 0.644 | 0.658 |

Aggregate:

| policy | mean success | mean max reward | paired improvements | paired regressions | paired net |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.651 | 0.751 | n/a | n/a | n/a |
| R3 ungated | 0.651 | 0.751 | 534 | 534 | 0 |
| R3 residual gate | 0.663 | 0.758 | 548 | 519 | +29 |

Per-bank paired net:

| eval seed start | R3 ungated | R3 residual gate |
| ---: | ---: | ---: |
| 3500000 | +25 | +25 |
| 3600000 | -12 | +3 |
| 3700000 | +8 | +14 |
| 3800000 | -16 | -15 |
| 3900000 | -5 | +2 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final500_seed3800000/eval_500_seed3800000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3800000/eval_500_seed3800000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3800000/eval_500_seed3800000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final500_seed3900000/eval_500_seed3900000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3900000/eval_500_seed3900000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final500_seed3900000/eval_500_seed3900000.json`

### Interpretation

The five-bank result is positive but modest. The residual-gated policy has the
best aggregate success and reward, and it improves paired net wins over ungated
R3:

```text
frozen success mean:        0.651
ungated R3 success mean:    0.651
residual-gated R3 mean:     0.663
```

However, the new `seed_start=3800000` bank favors frozen over both tuned
variants (`0.684` frozen versus `0.654` gated). The gate reduces damage relative
to ungated R3 on most windows, but it does not eliminate regressions. This
means the gated R3 policy is the best current real-compatible variant, but the
claim should remain "small, somewhat stable improvement" rather than a decisive
RL breakthrough.

The next useful work should either:

- validate the gate on a larger final bank after fixing the threshold, or
- improve the gate from a scalar action-delta threshold to a state/goal-aware
  selector trained from the per-episode diagnostics now emitted by eval.

## Effect32 FiLM R3 residual-L2 gate fresh 1000-episode check

After selecting and validating the `0.00121` residual-L2 gate on five
500-episode windows, I ran a larger fresh 1000-episode check at
`seed_start=4000000`. This is a stricter final-style check because the gate
threshold was already fixed before this bank was evaluated.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final1000_seed4000000 \
  --episodes 1000 \
  --seed-start 4000000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --residual-l2-gate-max 0.00121 \
  --distance-metric reachability \
  --force
```

Matched frozen and ungated R3 runs were evaluated on the same seed window:

- `hcl_next_effect32_dphi_frozen_final1000_seed4000000`
- `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final1000_seed4000000`

### Result

| policy | success | max reward | raw local reduction | reach rate | terminal AUC | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.667 | 0.761 | 0.409 | 0.720 | 0.804 | 0.0000 |
| R3 ungated | 0.632 | 0.733 | 0.395 | 0.708 | 0.809 | 0.0010 |
| R3 residual gate | 0.643 | 0.744 | 0.392 | 0.716 | 0.812 | 0.0004 |

Paired against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| R3 ungated | 212 | 247 | -35 |
| R3 residual gate | 210 | 234 | -24 |

Combining the earlier five 500-episode banks with this fresh 1000-episode bank
gives a total of 3500 matched episodes:

| policy | total episodes | success |
| --- | ---: | ---: |
| frozen | 3500 | 0.656 |
| R3 ungated | 3500 | 0.646 |
| R3 residual gate | 3500 | 0.657 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final1000_seed4000000/eval_1000_seed4000000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final1000_seed4000000/eval_1000_seed4000000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final1000_seed4000000/eval_1000_seed4000000.json`

### Interpretation

This is a negative final-style validation. The residual-L2 gate reduces the
damage relative to ungated R3 on this fresh 1000-episode window, but it still
underperforms frozen (`0.643` versus `0.667`). Over all 3500 matched episodes,
gated R3 is effectively tied with frozen (`0.657` versus `0.656`), while
ungated R3 is worse (`0.646`).

This revises the previous promotion:

- the residual gate is diagnostically useful and does reduce R3 regressions;
- it is not robust enough to claim a real deployment improvement over frozen;
- the effect32 R3 result should be reported as "small, unstable, and mostly
  neutral after final-style validation";
- the next meaningful step is a state/goal-aware selector or a different
  objective that creates larger action improvements, not more scalar threshold
  tuning.

## Low-level pre-decision selector diagnostics

I extended `low-level-rl eval` with compact per-episode features that are
available before or during action selection, rather than only after the episode
outcome:

- `episode_selected_distance_mean`
- `episode_raw_distance_mean`
- `episode_base_action_l2_mean`
- `episode_previous_action_norm_l2_mean`
- `episode_replan_rate`
- `episode_initial_selected_distance`
- `episode_initial_raw_distance`
- `episode_initial_base_action_l2`
- `episode_initial_env_reward`

The `initial_*` fields are clean episode-level selector inputs because frozen
and tuned policies share the same state before the first action. The trajectory
means are still useful for diagnosis and future step-level selectors, but they
can depend on the policy after the first action.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_predetail500_seed4100000 \
  --episodes 500 \
  --seed-start 4100000 \
  --distance-metric reachability \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_predetail500_seed4100000 \
  --episodes 500 \
  --seed-start 4100000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --distance-metric reachability \
  --force
```

### Result

On this fresh 500-episode window, R3 is again worse than frozen:

| policy | success |
| --- | ---: |
| frozen | 0.656 |
| R3 ungated | 0.618 |

Paired outcome counts:

| paired outcome | episodes |
| --- | ---: |
| both success | 210 |
| both fail | 73 |
| R3 wins | 99 |
| R3 regressions | 118 |

Feature separation among the 217 discordant episodes:

| feature | AUC for R3 win | oriented AUC | R3-win mean | regression mean | best in-window gate |
| --- | ---: | ---: | ---: | ---: | --- |
| initial selected distance | 0.474 | 0.526 | 0.795 | 0.807 | 0.670 |
| initial raw distance | 0.399 | 0.601 | 1.869 | 2.038 | 0.692 |
| initial base action L2 | 0.544 | 0.544 | 1.243 | 1.222 | 0.680 |
| initial env reward | 0.500 | 0.500 | 0.000 | 0.000 | 0.618 |
| selected distance mean | 0.273 | 0.727 | 0.568 | 0.649 | 0.728 |
| raw distance mean | 0.479 | 0.521 | 0.730 | 0.773 | 0.670 |
| base action L2 mean | 0.401 | 0.599 | 0.390 | 0.434 | 0.698 |
| previous action norm L2 mean | 0.443 | 0.557 | 1.055 | 1.090 | 0.674 |
| replan rate | 0.500 | 0.500 | 0.100 | 0.100 | 0.618 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_predetail500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_predetail500_seed4100000/eval_500_seed4100000.json`

### Interpretation

The clean first-step episode features are weak. Initial raw distance is the
best of them, but its oriented discordance AUC is only `0.601`, and the
in-window gate is an optimistic upper bound rather than a validated policy.

The strongest separator is mean selected distance along the R3 trajectory
(`0.727` oriented AUC, best in-window gate `0.728` success), but this is not a
clean episode-level deployable signal because it is partly produced by already
running the tuned policy. It is still useful evidence that R3 harms episodes
where the learned reachability distance stays high during execution.

Next selector work should therefore use step-level or segment-level gating that
observes current state/goal distance during rollout, not a one-shot episode
gate from only initial state features. Scalar residual magnitude alone has
already saturated; the better selector signal appears to be "current
reachability-distance trouble" combined with conservative fallback to frozen.

## Selected-distance step gate

I added an eval-time `--selected-distance-gate-max` option. When the current
selected distance is above the threshold, eval executes the frozen base action
instead of the tuned action. Under `--distance-metric reachability`, this gates
on the learned reachability distance. The eval JSON now records:

- `selected_distance_gate_max`
- `selected_distance_gate_rate`
- `episode_selected_distance_gate_rate`

### Threshold sweep on seed_start=4100000

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate085_final500_seed4100000 \
  --episodes 500 \
  --seed-start 4100000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --selected-distance-gate-max 0.85 \
  --distance-metric reachability \
  --force
```

The same command shape was run for thresholds `0.45`, `0.55`, `0.65`, `0.75`,
and `0.85`.

| policy | success | max reward | raw local reduction | reach rate | residual L2 | fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.656 | 0.755 | 0.385 | 0.734 | 0.00000 | 0.000 |
| R3 ungated | 0.618 | 0.728 | 0.393 | 0.718 | 0.00101 | 0.000 |
| gate 0.45 | 0.642 | 0.741 | 0.399 | 0.714 | 0.00028 | 0.697 |
| gate 0.55 | 0.618 | 0.727 | 0.387 | 0.719 | 0.00044 | 0.512 |
| gate 0.65 | 0.620 | 0.729 | 0.379 | 0.718 | 0.00058 | 0.360 |
| gate 0.75 | 0.624 | 0.730 | 0.396 | 0.715 | 0.00070 | 0.263 |
| gate 0.85 | 0.664 | 0.762 | 0.418 | 0.727 | 0.00080 | 0.175 |

Threshold `0.85` was the only sweep setting that beat frozen on this window.

### Fresh validation on seed_start=4200000

I fixed the threshold at `0.85` before evaluating a fresh 500-episode window.

| policy | success | max reward | raw local reduction | reach rate | residual L2 | fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.656 | 0.755 | 0.411 | 0.730 | 0.00000 | 0.000 |
| R3 ungated | 0.658 | 0.752 | 0.408 | 0.730 | 0.00099 | 0.000 |
| gate 0.85 | 0.640 | 0.741 | 0.384 | 0.720 | 0.00078 | 0.175 |

Paired against frozen:

| window | policy | improvements | regressions | net |
| --- | --- | ---: | ---: | ---: |
| 4100000 | R3 ungated | 99 | 118 | -19 |
| 4100000 | gate 0.85 | 115 | 111 | +4 |
| 4200000 | R3 ungated | 110 | 109 | +1 |
| 4200000 | gate 0.85 | 103 | 111 | -8 |

Combined over the two 500-episode windows:

| policy | total episodes | success | improvements | regressions | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 1000 | 0.656 | - | - | - |
| R3 ungated | 1000 | 0.638 | 209 | 227 | -18 |
| gate 0.85 | 1000 | 0.652 | 218 | 222 | -4 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate045_final500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate055_final500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate065_final500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate075_final500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate085_final500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_distgatecheck500_seed4200000/eval_500_seed4200000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgatecheck500_seed4200000/eval_500_seed4200000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate085_check500_seed4200000/eval_500_seed4200000.json`

### Interpretation

This selected-distance gate is useful instrumentation and reduces the worst R3
damage on the combined two-window check, but it does not beat frozen. The fresh
validation reversed the sweep-window gain. The result is similar to the
residual-L2 gate: it can make R3 less harmful, but it is not a robust policy
improvement.

This weakens the case for hand-tuned one-dimensional gates. The next meaningful
selector should either:

- learn a multifeature segment/step selector from paired outcomes, using current
  selected distance, raw distance, base-action norm, residual norm, and local
  progress features; or
- move back to the RL objective and create a larger, cleaner policy difference
  before trying to gate it.

## Initial linear selector and vector-eval pairing caveat

I added a runnable initial linear selector to `low-level-rl eval`:

```text
--initial-selector-weights w_selected w_raw w_base_action
--initial-selector-mean mean_selected mean_raw mean_base_action
--initial-selector-std std_selected std_raw std_base_action
--initial-selector-threshold threshold
```

Feature order:

```text
initial selected distance, initial raw distance, initial base-action L2
```

The selector locks each episode to either the tuned policy or the frozen base
policy at the first step. It records:

- `initial_selector_feature_order`
- `initial_selector_weights`
- `initial_selector_mean`
- `initial_selector_std`
- `initial_selector_threshold`
- `initial_selector_tuned_rate`
- `episode_initial_selector_use_tuned`

### Offline probe

As a cheap probe, I fit a ridge linear classifier on the 4100000 window using
only discordant frozen/R3 outcomes, then chose the score threshold that
maximized the mixed success on that same training window.

Fitted selector:

```text
weights:   [-0.0167040970, -0.1439550473,  0.0884878389]
mean:      [ 0.7980023136,  1.9726019294,  1.2325225415]
std:       [ 0.1136807627,  0.6093525598,  0.2191675288]
threshold: -0.0202570547
```

The offline JSON-mixing estimate looked promising:

| window | offline selector | frozen | R3 | select-tuned rate |
| --- | ---: | ---: | ---: | ---: |
| train 4100000 | 0.702 | 0.656 | 0.618 | 0.586 |
| valid 4200000 | 0.678 | 0.656 | 0.658 | 0.588 |
| valid 4300000 | 0.660 | 0.622 | 0.658 | 0.590 |

### Direct eval

I then ran the same fixed selector as an actual mixed policy:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_check500_seed4200000 \
  --episodes 500 \
  --seed-start 4200000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --initial-selector-weights -0.0167040970 -0.1439550473 0.0884878389 \
  --initial-selector-mean 0.7980023136 1.9726019294 1.2325225415 \
  --initial-selector-std 0.1136807627 0.6093525598 0.2191675288 \
  --initial-selector-threshold -0.0202570547 \
  --distance-metric reachability \
  --force
```

The direct policy result did not validate:

| window | frozen | R3 | direct selector | tuned rate |
| --- | ---: | ---: | ---: | ---: |
| 4100000 | 0.656 | 0.618 | 0.660 | 0.590 |
| 4200000 | 0.656 | 0.658 | 0.604 | 0.617 |
| 4300000 | 0.622 | 0.658 | 0.602 | 0.611 |

Validation-only combined:

| policy | episodes | success |
| --- | ---: | ---: |
| frozen | 1000 | 0.639 |
| R3 ungated | 1000 | 0.658 |
| direct initial selector | 1000 | 0.603 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_traincheck500_seed4100000/eval_500_seed4100000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_check500_seed4200000/eval_500_seed4200000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_check500_seed4300000/eval_500_seed4300000.json`

### Interpretation

The direct mixed-policy eval is authoritative. The offline selector estimate was
too optimistic because per-episode arrays from separate vectorized closed-loop
evals are not guaranteed to be seed-aligned once policies terminate/reset at
different times. Therefore:

- aggregate success metrics remain valid;
- paired counts from separate vectorized eval JSON arrays should be treated as
  diagnostic only, not as exact paired rollouts;
- any selector must be evaluated as an actual policy inside the simulator, or
  on an explicit fixed reset bank with stable episode IDs.

This is a useful correction. It invalidates the initial linear selector as a
policy improvement and explains why offline gates looked better than direct
validation. The next infrastructure improvement should be explicit episode
identity or fixed reset-bank evaluation before training more learned selectors.

## Serial exact-seed low-level evaluator

I added `low-level-rl eval-serial`, a slower evaluator that resets one
environment explicitly for every seed and writes `episode_seed` into the JSON.
This is intended for paired selector/debug work where episode identity matters.
It uses the same effect-progress latent update as the vector low-level rollout,
not the older video helper's plain frame encoder path. I initially used a plain
`gym.make(..., num_envs=1)` env, but that produced local-distance metrics on a
different scale. I corrected it to use the same `ManiSkillVectorEnv` wrapper as
the main vector evaluator, with one explicit seed at a time.

Example command:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000 \
  --episodes 50 \
  --seed-start 4501000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --distance-metric reachability \
  --force
```

I ran a compact exact-seed diagnostic over seeds `4501000..4501049`:

| policy | success | max reward | raw local reduction | selector tuned rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.620 | 0.737 | 0.437 | - |
| R3 ungated | 0.700 | 0.785 | 0.428 | - |
| initial linear selector | 0.640 | 0.742 | 0.427 | 0.480 |

Exact paired counts against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| R3 ungated | 10 | 6 | +4 |
| initial linear selector | 5 | 4 | +1 |

All three JSON files have aligned `episode_seed` arrays:

```text
4501000, 4501001, ..., 4501049
```

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4501000/serial_eval_50_seed4501000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/serial_eval_50_seed4501000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_serial50_seed4501000/serial_eval_50_seed4501000.json`

I also added `low-level-rl compare-serial`, which refuses to compare files
unless `episode_seed` arrays match exactly. For the above runs it wrote:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/paired_vs_frozen_serial50_seed4501000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_serial50_seed4501000/paired_vs_frozen_serial50_seed4501000.json`

To make this distinction explicit, vector eval JSONs now include:

```text
eval_mode: vector_auto_reset_unpaired
episode_seed: null
```

### Interpretation

The serial evaluator fixes the immediate paired-evaluation problem for small
debug windows. It is too slow to replace vectorized final eval, but it is the
right tool for learning or auditing selectors until a proper local reset bank is
available.

This 50-seed window also reinforces that the initial linear selector is not the
right deployment policy: when exact pairing is available, it keeps only a small
fraction of R3's gains and underperforms ungated R3. Future selector work should
train/evaluate on serial exact-seed or reset-bank data, not on unaligned vector
episode arrays.

## Exact-paired initial selector fit

I added `low-level-rl fit-serial-selector`, which fits the same three-feature
initial selector only from serial exact-paired JSONs:

```text
features = [
  episode_initial_selected_distance,
  episode_initial_raw_distance,
  episode_initial_base_action_l2,
]
```

It refuses unaligned serial files, fits a ridge linear classifier on discordant
train outcomes, chooses the threshold that maximizes train mixed success, and
optionally reports an exact-paired validation mix.

I used the serial `4501000` window for fitting and a fresh serial `4502000`
window for validation:

```bash
uv run hcl-poc low-level-rl fit-serial-selector \
  --base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4501000/serial_eval_50_seed4501000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/serial_eval_50_seed4501000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4502000/serial_eval_50_seed4502000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4502000/serial_eval_50_seed4502000.json \
  --output results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/init_selector_fit_train4501000_valid4502000.json \
  --force
```

Fitted selector:

```text
weights:   [-0.5096282363, -0.1020845994, -0.3853654265]
mean:      [ 0.8209013343,  1.9272705317,  1.1766604185]
std:       [ 0.0968373716,  0.6078169942,  0.1958861202]
threshold: -0.4030666649
```

Offline exact-paired mix:

| split | frozen | R3 | selector mix | tuned rate | net vs frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 4501000 | 0.620 | 0.700 | 0.780 | 0.700 | +8 |
| validation 4502000 | 0.740 | 0.660 | 0.660 | 0.680 | -4 |

I then ran the fitted selector directly on the validation seed window with
full-precision parameters:

| policy | success | max reward | raw local reduction | tuned rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.740 | 0.808 | 0.438 | - |
| R3 ungated | 0.660 | 0.757 | 0.423 | - |
| fitted selector direct | 0.640 | 0.743 | 0.443 | 0.680 |

The direct selector chose exactly the same seeds as the offline mix, but one
episode outcome differed (`4502023`), so the direct result is slightly worse
than the offline exact-paired estimate. The conclusion is unchanged: the fitted
initial selector overfits badly and fails validation.

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4502000/serial_eval_50_seed4502000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4502000/serial_eval_50_seed4502000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/init_selector_fit_train4501000_valid4502000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_fitselector_exact_serial50_seed4502000/serial_eval_50_seed4502000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_fitselector_exact_serial50_seed4502000/paired_vs_frozen_serial50_seed4502000.json`

### Interpretation

Even with exact seed identity and a fitting utility that avoids the earlier
vector-array pairing mistake, a one-shot initial selector is not reliable. It
can overfit a 50-seed window, but it does not generalize to a fresh 50-seed
window and can underperform both frozen and ungated R3.

This pushes the next useful selector direction away from episode-initial
features and toward either:

- a larger fixed reset-bank dataset with enough samples to train a selector; or
- a step/segment-level selector that can observe current trajectory state rather
  than committing at the first action.

## Serial segment-level selector diagnostics

I extended `low-level-rl eval-serial` with per-segment arrays for selector
datasets:

- `serial_segment_episode_seed`
- `serial_segment_index`
- `serial_segment_start_step`
- `serial_segment_initial_selected_distance`
- `serial_segment_initial_raw_distance`
- `serial_segment_initial_base_action_l2`
- `serial_segment_initial_previous_action_norm_l2`
- `serial_segment_final_selected_distance`
- `serial_segment_final_raw_distance`
- `serial_segment_selected_distance_reduction`
- `serial_segment_raw_distance_reduction`
- `serial_segment_goal_reached`
- `serial_segment_residual_l2_mean`
- `serial_segment_action_saturation_rate`
- `serial_segment_distance_gate_rate`

I also added `low-level-rl compare-serial-segments`, which aligns two serial
evals by `(episode_seed, segment_index)` and reports paired segment-level local
improvement diagnostics.

### Smoke check

A 5-episode R3 smoke run produced 50 aligned ten-step segments with expected
start steps:

```text
(4503000, 0, 0), (4503000, 1, 10), ..., (4503000, 4, 40)
```

### 50-episode paired segment diagnostic

I then ran matched frozen/R3 serial evals over `4503000..4503049` and compared
500 paired segments:

| policy | task success | segment raw reduction |
| --- | ---: | ---: |
| frozen | 0.620 | 0.414 |
| R3 ungated | 0.680 | 0.417 |

Paired segment outcome:

| metric | value |
| --- | ---: |
| common segments | 500 |
| helpful R3 segments | 244 |
| harmful R3 segments | 256 |
| mean raw-reduction delta | 0.003 |

Candidate segment-start feature signal for whether R3 improves raw local
reduction over frozen:

| feature | helpful AUC | oriented AUC | corr with raw-reduction delta |
| --- | ---: | ---: | ---: |
| initial raw distance | 0.567 | 0.567 | 0.179 |
| initial base action L2 | 0.533 | 0.533 | 0.071 |
| initial selected distance | 0.527 | 0.527 | 0.074 |
| previous action norm L2 | 0.511 | 0.511 | 0.062 |
| residual L2 mean | 0.509 | 0.509 | 0.019 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segment_smoke5_seed4503000/serial_eval_5_seed4503000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/paired_segments_vs_frozen_serial50_seed4503000.json`

### Interpretation

This is a negative result for simple segment-level gating. The segment arrays
are now available and correctly aligned, but the obvious segment-start features
barely separate helpful from harmful R3 segments. Initial raw distance is the
best single signal, and its oriented AUC is only `0.567`.

The practical implication is that a deployable segment selector probably needs
more context than scalar distance/action norms, a larger fixed-bank dataset, or
an objective that creates larger candidate-policy differences before gating.

## Lower-BC R3 objective check

After the gate/selector diagnostics failed to produce a robust improvement, I
tested whether relaxing the R3 behavior-cloning anchor would create a larger,
cleaner policy difference. This keeps the same effect32 + reachability terminal
reward setup as the current best R3 checkpoint, but changes `bc_weight` from
`10.0` to `1.0`.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1 \
  --steps 40960 \
  --num-envs 4096 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 3e-5 \
  --initial-logstd -4.0 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Training summary was almost unchanged relative to `bc10`:

| run | bc weight | global step | bc loss | action saturation | clip fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| R3 bc10 | 10.0 | 40960 | 8.35e-7 | 0.259 | 0.035 |
| R3 bc1 | 1.0 | 40960 | 8.39e-7 | 0.259 | 0.035 |

I evaluated on the existing exact serial window `4503000..4503049`, using the
same frozen baseline and BC10 comparison from the segment diagnostic:

| policy | success | max reward | raw local reduction | residual L2 | action saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.620 | 0.715 | 0.414 | 0.000000 | 0.034 |
| R3 bc10 | 0.680 | 0.762 | 0.417 | 0.001031 | 0.035 |
| R3 bc1 | 0.660 | 0.758 | 0.444 | 0.001064 | 0.033 |

Exact paired counts against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| R3 bc10 | 7 | 4 | +3 |
| R3 bc1 | 6 | 4 | +2 |

Paired segment raw-reduction diagnostics:

| policy | mean raw-reduction delta | helpful segments | harmful segments |
| --- | ---: | ---: | ---: |
| R3 bc10 | 0.0030 | 244 | 256 |
| R3 bc1 | 0.0296 | 241 | 259 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1_serial50_seed4503000/serial_eval_50_seed4503000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1_serial50_seed4503000/paired_vs_frozen_serial50_seed4503000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1_serial50_seed4503000/paired_segments_vs_frozen_serial50_seed4503000.json`

### Interpretation

Lowering the BC anchor from `10` to `1` is not the missing objective fix. It
does increase mean raw local reduction on this small exact serial window
(`0.444` vs `0.417`), but it does not create a larger residual/action shift, and
task success is worse than the current `bc10` checkpoint (`0.660` vs `0.680`).
The segment-level picture is also not cleaner: BC1 has slightly larger average
local improvement but still more harmful than helpful paired segments.

This suggests the current R3 formulation is not BC-weight limited at this scale.
The next objective experiment should change the reward target itself, not just
the BC coefficient.

## 2026-06-25 - Paired terminal reward smoke

After BC-weight reduction failed, I added an R3 `--reward-mode paired` option.
Instead of rewarding absolute terminal selected distance, the tuned branch is
compared against a frozen low-level branch cloned from the same segment start
and held goal. The terminal reward is:

```text
base_next_distance - tuned_next_distance
```

This keeps the reward local, but removes the pressure to improve already-easy
segments in absolute terms.

### Smoke checks

Implementation checks:

```bash
uv run ruff check src/hcl_poc/low_level_rl.py --select F,E501
uv run python -m compileall -q src/hcl_poc/low_level_rl.py src/hcl_poc/cli.py
```

Both passed. A tiny 640-step smoke also completed and wrote paired metrics.

I then ran one larger diagnostic update:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_paired_10240_bc10 \
  --steps 10240 \
  --num-envs 1024 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 3e-5 \
  --initial-logstd -4.0 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Training metrics:

| run | global step | mean paired improvement | improved segments | tuned terminal | base terminal | direct delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired R3 bc10 | 10240 | 0.01234 | 0.522 | 0.5767 | 0.5890 | 0.0294 |

Small exact serial eval on `4504000..4504019`:

| policy | success | raw local reduction | selected reduction | residual L2 |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.500 | 0.301 | 0.031 | 0.000000 |
| paired R3 10k | 0.600 | 0.247 | 0.044 | 0.000983 |

Exact paired counts:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| paired R3 10k | 3 | 1 | +2 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial20_seed4504000/serial_eval_20_seed4504000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/serial_eval_20_seed4504000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/paired_vs_frozen_serial20_seed4504000.json`

### Interpretation

The paired reward is a better objective candidate than simply lowering BC
weight: it produced a positive train-time paired improvement and a small
positive exact-serial success delta after only one diagnostic update. The sample
is too small to call this a policy improvement, and the action shift is still
tiny, but this is the first objective change in this sequence that directly
optimizes improvement over the frozen segment policy and shows a matching
positive small-window task signal.

### Follow-up: 40k paired train

I then ran a larger paired train:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_paired_40k_bc10 \
  --steps 40960 \
  --num-envs 2048 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 3e-5 \
  --initial-logstd -4.0 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

The first strict implementation failed on longer runs because tuned and frozen
branches can terminate/reset differently. I changed paired mode to mask
desynchronized comparisons, count resyncs, and clone the base branch back from
the tuned rollout instead of aborting.

Training history:

| global step | mean paired improvement | improved segments | resync events | desynced envs | tuned terminal | base terminal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20480 | 0.01493 | 0.514 | 0 | 0 | 0.5666 | 0.5816 |
| 40960 | n/a | n/a | 1 | 2048 | 0.6069 | n/a |

The best checkpoint was selected from the positive 20480-step row.

Exact serial validation on `4505000..4505049`:

| policy | success | max reward | raw local reduction | selected reduction | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.560 | 0.683 | 0.443 | 0.094 | 0.000000 |
| paired R3 40k best | 0.560 | 0.696 | 0.485 | 0.086 | 0.000797 |

Exact paired counts:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| paired R3 40k best | 7 | 7 | 0 |

Additional artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4505000/serial_eval_50_seed4505000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/serial_eval_50_seed4505000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/paired_vs_frozen_serial50_seed4505000.json`

Interpretation: paired terminal reward is a cleaner objective than absolute
terminal distance, and it improves local raw reduction on this 50-seed window,
but the scaled run is neutral on task success. It is not yet a robust
improvement over frozen.

## 2026-06-25 - Multifeature serial segment selector diagnostic

The previous one-dimensional segment diagnostics showed weak scalar separation
between helpful and harmful R3 segments. I added a reproducible offline
segment-selector diagnostic:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  fit-serial-segment-selector \
  --base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json \
  --output results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/segment_selector_fit_train4503000_valid4506000.json \
  --ridge 1.0 \
  --force
```

The selector uses only segment-start features available before choosing tuned
or frozen:

```text
initial selected distance
initial raw distance
initial base action L2
initial previous action L2
segment start step
```

I generated a fresh exact serial validation window at `4506000..4506049` for
frozen and the R3 bc10 checkpoint. Episode-level validation on this window was
negative for ungated R3:

| policy | success | max reward | raw local reduction | selected reduction |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.720 | 0.802 | 0.451 | 0.092 |
| R3 bc10 | 0.700 | 0.783 | 0.461 | 0.090 |

Exact paired counts:

| improvements | regressions | net |
| ---: | ---: | ---: |
| 7 | 8 | -1 |

Offline segment-selector local metric:

| split | base raw reduction | R3 raw reduction | selector raw reduction | selector delta vs base | selector use R3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 4503000 | 0.414 | 0.417 | 0.478 | +0.064 | 0.796 |
| validation 4506000 | 0.451 | 0.461 | 0.515 | +0.063 | 0.788 |

The fitted selector had validation AUC `0.584` for segment-level helpfulness.
It kept 207 helpful R3 segments and 187 harmful R3 segments on validation,
which is still noisy, but the local raw-reduction aggregate improved more than
the ungated R3 aggregate.

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4506000/paired_vs_frozen_serial50_seed4506000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/segment_selector_fit_train4503000_valid4506000.json`

Interpretation:

This is the first state/goal-aware selector diagnostic that generalizes on a
held-out exact segment window for local raw reduction. It does not prove a task
success gain: the ungated R3 policy is negative on the same validation window,
and this selector has only been evaluated offline at the segment level. The next
step is to add a direct serial evaluator that applies a segment-start selector
online, then test whether the local raw-reduction gain transfers to episode
success.

## 2026-06-25 - Online multifeature serial segment selector validation

I added online `eval-serial` support for the fitted 5-feature segment selector:
at each high-level replan, the evaluator scores the segment-start features and
chooses either the tuned R3 action or the frozen low-level action for that held
goal. Initial episode selectors and segment selectors are mutually exclusive in
the evaluator to keep the policy semantics clear.

Validation used the same held-out exact serial window `4506000..4506049` and
the selector fitted above.

| policy | success | max reward | raw local reduction | segment goal reach | R3 segment use |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.720 | 0.802 | 0.451 | 0.698 | - |
| ungated R3 bc10 | 0.700 | 0.783 | 0.461 | 0.652 | 1.000 |
| online segment selector | 0.680 | 0.771 | 0.460 | 0.668 | 0.760 |

Paired episode counts for online selector vs frozen:

| improvements | regressions | net success delta |
| ---: | ---: | ---: |
| 7 | 9 | -0.040 |

Segment-level paired comparison for online selector vs frozen:

| common segments | base raw reduction | selector raw reduction | delta | helpful | harmful |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.451 | 0.460 | +0.008 | 240 | 252 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_online_serial50_seed4506000/serial_eval_50_seed4506000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_online_serial50_seed4506000/compare_vs_frozen_serial50_seed4506000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_online_serial50_seed4506000/segment_compare_vs_frozen_serial50_seed4506000.json`

Interpretation:

The offline selector result was optimistic because it selected between already
completed frozen/R3 segment outcomes. Online, changing early segments changes
later high-level goals, states, and segment distributions. The deployed selector
kept some local raw-reduction gain over frozen, but not enough to improve the
closed-loop task; it underperformed both frozen and ungated R3 on episode
success. This closes the current segment-selector branch as a deployment fix
for this checkpoint. The next useful work should focus on an objective that
creates a larger, task-aligned local effect, or a selector trained directly on
closed-loop episode outcomes rather than offline segment deltas.

## 2026-06-25 - Paired R3 with lower BC anchor

Hypothesis: the previous paired terminal reward had the right sign but the
direct R3 final-layer update was still too tightly anchored to BC. I retried the
paired objective with `bc_weight=1` instead of `10`.

The first attempt at 4096 envs failed during ManiSkill GPU camera allocation in
paired mode. Paired mode keeps both the tuned branch and frozen base branch in
memory, so I reran the same objective with 2048 envs and recorded that in the
run name.

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_paired_2048_40k_bc1 \
  --steps 40960 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 1.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --num-envs 2048 \
  --rollout-steps 10 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --force
```

Training metrics:

| step | mean paired improvement | fraction improved | direct delta L2 | saturation | BC loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20480 | 0.0149 | 0.514 | 0.0294 | 0.260 | 0.00000078 |
| 40960 | n/a | n/a | 0.0293 | 0.095 | 0.00000146 |

The best checkpoint was selected at step 20480, where paired improvement matched
the earlier `bc=10` run but saturation was much higher.

Fresh exact serial validation on `4507000..4507049`:

| policy | success | max reward | raw local reduction | segment goal reach | action saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.560 | 0.673 | 0.425 | 0.684 | 0.030 |
| paired R3 2048 40k bc1 | 0.560 | 0.673 | 0.383 | 0.656 | 0.027 |

Paired episode counts were neutral: 6 improvements, 6 regressions, success delta
`0.000`. Segment comparison was negative: 500 aligned segments, raw-reduction
delta `-0.0417`, 242 helpful and 258 harmful segments.

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_40k_bc1/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4507000/serial_eval_50_seed4507000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_40k_bc1_serial50_seed4507000/serial_eval_50_seed4507000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_40k_bc1_serial50_seed4507000/paired_vs_frozen_serial50_seed4507000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_40k_bc1_serial50_seed4507000/segment_compare_vs_frozen_serial50_seed4507000.json`

Interpretation:

Lowering the BC anchor increased action changes but mostly bought saturation
and worse local rollout behavior. The paired reward branch is not simply
underpowered by BC regularization. The next objective-side test should change
the training/evaluation target itself: use an explicit closed-loop deployment
proxy, a stronger task-aligned reachability metric, or a local reset formulation
where the paired reward is measured against cached exact base outcomes rather
than a simultaneous branch that can desynchronize.

## 2026-06-25 - Cached local-reset paired reward for rl-rerun R3

I added a paired reward mode to `rl-rerun train-local-r3`. Unlike the older
full-hierarchy paired branch, this mode does not keep a simultaneous frozen
branch in memory. For each sampled local reset it:

1. resets/replays to the demo local start;
2. rolls out the frozen low level to cache the exact terminal distance;
3. resets/replays back to the same local start;
4. trains the direct R3 policy with dense local progress plus terminal
   `base_terminal_distance - tuned_terminal_distance`.

CLI:

```bash
uv run hcl-poc rl-rerun train-local-r3 ... --reward-mode paired
```

Compatibility detail: default `--reward-mode progress` keeps the existing local
R3 recipe unchanged so old progress-mode checkpoints can still resume.

Smoke check on `data/rl_rerun/pusht_vector_state_demos_n512_b1.h5`:

| envs | steps | mean base terminal | mean tuned terminal | paired improvement | fraction improved | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 5120 | 0.5793 | 0.5789 | +0.00045 | 0.516 | 0.0008 |

The smoke checkpoint also reloaded through `eval-local-r3` on
`pusht_vector_state_demos_n512_val_b1.h5`.

One 4096-env update on `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5`:

| envs | steps | mean base terminal | mean tuned terminal | paired improvement | fraction improved | saturation | update time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 40960 | 0.5991 | 0.6089 | -0.0098 | 0.478 | 0.0079 | 288s |

Matched local validation on the existing 4096-env validation manifest
`results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json`:

| policy | initial distance | final distance | reduction | reduction fraction | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 1.0671 | 0.6020 | 0.4651 | 0.7969 | 0.0078 |
| cached-paired R3 1 update | 1.0671 | 0.6066 | 0.4605 | 0.7966 | 0.0078 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/paired_cached_smoke_1update/history.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_smoke_1update/eval_local_val512_b1_e1.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_1update_bc1/history.json`
- `results/rl_rerun/local_mode_a_audit_n4096_val_b1_n500_seed0_manifest.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_1update_bc1/eval_local_n4096_val_b1_manifest.json`

Checks:

- `uv run python -m compileall -q src/hcl_poc/rl_rerun.py src/hcl_poc/cli.py`
- `uv run ruff check src/hcl_poc/rl_rerun.py --select F821`
- `uv run pytest -q` (`32 passed`)

Interpretation:

This implements the plan's cached-base paired local-reset direction and removes
the simultaneous-branch desync/memory issue. The first 4096-env update did not
improve local validation; it was slightly worse than frozen on the matched
manifest. That is only a one-update signal, but it confirms that the next
question is learning dynamics/objective quality rather than branch mechanics.
The next run should scale this cached paired local-R3 mode for several updates,
track whether paired improvement becomes positive, and only then evaluate
closed-loop deployment.

## 2026-06-25 - Cached paired local-R3 3-update learning check

I scaled the cached local-reset paired R3 mode to three 4096-env updates:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name paired_cached_n4096_3updates_bc1 \
  --steps 122880 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --reward-mode paired \
  --num-minibatches 8 \
  --checkpoint-every-updates 1 \
  --force
```

Training signal:

| step | base terminal | tuned terminal | paired improvement | fraction improved | delta L2 | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40960 | 0.5991 | 0.6089 | -0.0098 | 0.478 | 0.0293 | 0.0079 |
| 81920 | 0.5991 | 0.6099 | -0.0107 | 0.476 | 0.0291 | 0.0075 |
| 122880 | 0.6094 | 0.6245 | -0.0151 | 0.471 | 0.0292 | 0.0087 |

Runtime was about 875s total, `~140` samples/s, with peak reserved GPU memory
around 2.4 GiB. This confirms the cached-base formulation avoids the earlier
paired-branch camera-memory failure at 4096 envs.

Matched validation on
`results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json`:

| policy | initial distance | final distance | reduction | reduction fraction | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 1.0671 | 0.6020 | 0.4651 | 0.7969 | 0.0078 |
| cached-paired R3 3 updates | 1.0671 | 0.6000 | 0.4671 | 0.7996 | 0.0079 |

Validation delta vs frozen: final distance `-0.0020`, reduction fraction
`+0.0027`. The improvement is tiny and far below a deployment-relevant effect,
but it is directionally better than the one-update checkpoint.

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_3updates_bc1/history.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_3updates_bc1/eval_local_n4096_val_b1_manifest.json`

Interpretation:

The cached paired objective is now mechanically viable at the required 4096-env
scale, but its training paired-improvement signal is negative across three
updates and the held-out local improvement is only `0.002` final-distance units.
This is not a useful RL improvement yet. If continuing this branch, the next
change should not simply be "more of the same"; it should adjust the reward or
optimization so training paired improvement becomes positive, for example by
removing dense progress from paired mode, lowering the learning rate/action
noise, or training/evaluating on a broader reset bank before attempting
closed-loop deployment.

## 2026-06-25 - Cached paired terminal-only local-R3 check

I added `--dense-progress-weight` to `rl-rerun train-local-r3`, defaulting to
`1.0` so existing progress and paired runs keep their old recipe unless the
argument is explicitly set. This lets cached paired local-R3 train on terminal
paired improvement only:

```bash
uv run hcl-poc rl-rerun train-local-r3 ... \
  --reward-mode paired \
  --dense-progress-weight 0.0
```

I first ran a 512-env smoke. It completed and recorded
`dense_progress_weight=0.0` in the recipe.

Then I ran the 4096-env terminal-only paired check by resuming the same run to
three updates:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name paired_cached_terminalonly_n4096_1update_bc1 \
  --steps 122880 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --dense-progress-weight 0.0 \
  --reward-mode paired \
  --num-minibatches 8 \
  --checkpoint-every-updates 1
```

Training signal compared with the dense-progress cached paired run:

| mode | step | paired improvement | fraction improved | terminal distance | action delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense progress | 40960 | -0.0098 | 0.478 | 0.6089 | 0.0293 |
| dense progress | 81920 | -0.0107 | 0.476 | 0.6099 | 0.0291 |
| dense progress | 122880 | -0.0151 | 0.471 | 0.6245 | 0.0292 |
| terminal only | 40960 | -0.0098 | 0.478 | 0.6089 | 0.0293 |
| terminal only | 81920 | -0.0067 | 0.487 | 0.6058 | 0.0293 |
| terminal only | 122880 | -0.0103 | 0.482 | 0.6094 | 0.0292 |

Terminal-only paired training improved the on-policy paired metric relative to
dense-progress paired mode, but it was still negative.

Matched validation on
`results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json`:

| policy | initial distance | final distance | reduction | reduction fraction |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 | 1.0671 | 0.6020 | 0.4651 | 0.7969 |
| dense-progress cached paired 3 updates | 1.0671 | 0.6000 | 0.4671 | 0.7996 |
| terminal-only cached paired 3 updates | 1.0671 | 0.6086 | 0.4585 | 0.7896 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_1update_bc1/history.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_1update_bc1/eval_local_n4096_val_b1_manifest.json`

Interpretation:

Removing dense progress makes the training objective more purely paired and
less negative, but it hurts held-out local validation after three updates. Dense
progress is not the sole cause of the weak cached-paired result. The remaining
failure looks like a weak/unstable local optimization signal: action changes
stay tiny, paired improvement remains below zero, and held-out effects are at
the noise scale. Next objective work should try lower LR/noise or broader reset
coverage, but this branch still does not justify closed-loop deployment.

## 2026-06-25 - Cached paired lower-LR/lower-noise local-R3 check

I exposed `--initial-logstd` on `rl-rerun train-local-r3`, matching the local
R1/R2 controls. Default behavior remains unchanged: if omitted, local R3 still
uses `low_level_rl.direct_initial_logstd` from config (`-4.0` here).

I then tested the lower LR/noise hypothesis using terminal-only cached paired
reward:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1 \
  --steps 122880 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --dense-progress-weight 0.0 \
  --reward-mode paired \
  --learning-rate 0.00001 \
  --initial-logstd -5.0 \
  --num-minibatches 8 \
  --checkpoint-every-updates 1 \
  --force
```

Training signal compared with terminal-only `lr=3e-5`, `initial_logstd=-4`:

| variant | step | paired improvement | fraction improved | terminal distance | action delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| lr3e-5 logstd-4 | 40960 | -0.0098 | 0.478 | 0.6089 | 0.0293 |
| lr3e-5 logstd-4 | 81920 | -0.0067 | 0.487 | 0.6058 | 0.0293 |
| lr3e-5 logstd-4 | 122880 | -0.0103 | 0.482 | 0.6094 | 0.0292 |
| lr1e-5 logstd-5 | 40960 | -0.0031 | 0.490 | 0.6022 | 0.0108 |
| lr1e-5 logstd-5 | 81920 | -0.0042 | 0.490 | 0.6034 | 0.0108 |
| lr1e-5 logstd-5 | 122880 | -0.0044 | 0.490 | 0.6138 | 0.0108 |

Lower LR/noise made the on-policy paired metric substantially less negative and
reduced action deltas by about 3x, but the signal remained below zero.

Matched validation on
`results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json`:

| policy | initial distance | final distance | reduction | reduction fraction | action delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 1.0671 | 0.6020 | 0.4651 | 0.7969 | - |
| terminal-only lr3e-5 logstd-4 | 1.0671 | 0.6086 | 0.4585 | 0.7896 | 0.0024 |
| terminal-only lr1e-5 logstd-5 | 1.0671 | 0.6081 | 0.4590 | 0.7893 | 0.0007 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/history.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/eval_local_n4096_val_b1_manifest.json`

Interpretation:

Lower LR/noise stabilizes the training objective but mostly suppresses the
policy update; held-out local validation is still worse than frozen. This makes
the cached paired local-R3 branch look bottlenecked by objective/data signal,
not by the obvious PPO noise/LR setting alone. The next useful step is likely a
broader reset bank or a different distance/task proxy rather than further
single-manifest knob tuning.

## 2026-06-25 - Broader local validation for cached paired R3

The previous local validation used one 4096-env validation entry at one
timestep. To check whether that was too narrow, I created a broader held-out
manifest on the 512-env validation dataset with 8 timesteps and 4096 total local
episodes:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  create-local-eval-manifest \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --output results/rl_rerun/local_eval_manifest_n512_val_b1_seed20260625_e8.json \
  --episodes 8 \
  --seed 20260625 \
  --horizon 10
```

Sampled timesteps: `32, 18, 27, 36, 23, 24, 6, 29`.

Matched validation:

| policy | final distance | reduction | reduction fraction | action delta |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 | 0.7092 | 0.7023 | 0.8704 | - |
| dense-progress cached paired 3 updates | 0.7086 | 0.7028 | 0.8687 | 0.0023 |
| terminal-only lr1e-5 logstd-5 | 0.7080 | 0.7034 | 0.8708 | 0.0008 |

The broader validation deltas vs frozen were:

| policy | final-distance delta | reduction delta | reduction-fraction delta |
| --- | ---: | ---: | ---: |
| dense-progress cached paired 3 updates | +0.0005 | +0.0005 | -0.0017 |
| terminal-only lr1e-5 logstd-5 | +0.0012 | +0.0012 | +0.0005 |

Here positive delta means the tuned policy was better than frozen.

Artifacts:

- `results/rl_rerun/local_eval_manifest_n512_val_b1_seed20260625_e8.json`
- `results/rl_rerun/local_mode_a_audit_n512_val_b1_n500_seed0_manifest_e8.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_3updates_bc1/eval_local_n512_val_b1_manifest_e8.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/eval_local_n512_val_b1_manifest_e8.json`

Interpretation:

Broader held-out local validation is less negative than the single 4096-env
manifest: both cached-paired variants are slightly better than frozen on mean
distance reduction. However, the absolute gains are only `0.0005` to `0.0012`
distance units over 4096 local episodes. This is still noise-scale compared
with the size of the local distances and does not justify closed-loop
deployment. It does suggest that a broader reset bank is important for judging
these tiny effects, because single-entry conclusions can change sign.

## 2026-06-25 - Local task diagnostics for cached paired R3

I added task-reward diagnostics to both frozen local Mode-A audit and tuned
local R1/R2/R3 eval outputs. The local evaluators now record:

- `final_env_reward_mean`
- `max_env_reward_mean`
- `mean_env_reward_mean`
- `task_success_once_fraction`

This keeps the existing latent-distance checks but adds a closer proxy to the
PushT task reward/success signal returned by the environment.

I reran the broader 8-timestep validation manifest with diagnostics:

| policy | final distance | final env reward | max env reward | mean env reward | success-once | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 0.7092 | 0.3979 | 0.4649 | 0.3855 | 0.2456 | - |
| dense-progress cached paired 3 updates | 0.7086 | 0.4005 | 0.4636 | 0.3873 | 0.2439 | 0.0023 |
| terminal-only lr1e-5 logstd-5 | 0.7080 | 0.4005 | 0.4672 | 0.3874 | 0.2493 | 0.0008 |

The deltas vs frozen were:

| policy | final-distance delta | final reward delta | max reward delta | mean reward delta | success-once delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-progress cached paired 3 updates | +0.0005 | +0.0027 | -0.0013 | +0.0018 | -0.0017 |
| terminal-only lr1e-5 logstd-5 | +0.0012 | +0.0026 | +0.0023 | +0.0019 | +0.0037 |

Here positive final-distance delta means the tuned policy had lower final
latent distance than frozen.

Artifacts:

- `results/rl_rerun/local_mode_a_audit_n512_val_b1_n500_seed0_manifest_e8_taskdiag.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_3updates_bc1/eval_local_n512_val_b1_manifest_e8_taskdiag.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/eval_local_n512_val_b1_manifest_e8_taskdiag.json`

Interpretation:

The lower-noise terminal-only cached-paired checkpoint is the only one that is
weakly positive across final distance, final/max/mean task reward, and
success-once on this broad local manifest. The effect is still very small:
`+0.0037` success-once over 4096 local episodes is about 15 extra local
successes, and the reward gains are around `0.002`. This supports using
task-aligned local diagnostics for checkpoint selection, but it does not change
the deployment conclusion. The current cached-paired R3 objective produces a
measurable but tiny local effect; the next experiment should increase objective
signal strength or move to a better task proxy rather than continue small
learning-rate/noise sweeps.

## 2026-06-25 - Closed-loop transfer check for lower-noise cached paired R3

The lower-noise terminal-only cached-paired checkpoint was weakly positive on
the broad local task diagnostics, so I ran a matched learned-goal closed-loop
transfer check:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4600000 \
  --num-envs 64 \
  --output results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/closed_loop_500_seed4600000.json
```

Matched result:

| policy | success | final reward | max reward | action saturation | residual/action delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 0.334 | 0.4776 | 0.5061 | 0.0462 | - |
| terminal-only lr1e-5 logstd-5 | 0.294 | 0.4480 | 0.4760 | 0.0448 | 0.0007 |

Deltas vs frozen:

| metric | delta |
| --- | ---: |
| success | -0.040 |
| final reward | -0.0296 |
| max reward | -0.0302 |

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/paired_cached_terminalonly_n4096_3updates_lr1e5_logstd5_bc1/closed_loop_500_seed4600000.json`

Interpretation:

The weakly positive broad-local diagnostics did not transfer to closed-loop
learned-goal success. The residual/action change is extremely small, but still
harmful on this 500-episode window. This is another instance of local
reachability or local task-reward proxies being insufficient for reliable
checkpoint promotion. Do not continue with small cached-paired local-R3 sweeps
unless the objective changes enough to create a much larger local effect and is
validated directly in closed loop.

## 2026-06-25 - Task-reward upper-bound debug for local R3

I added a default-off `--task-reward-weight` knob to `rl-rerun train-local-r3`
as an explicitly marked debug upper bound. It records
`debug_training_signals=["mani_skill_reward"]` when enabled and leaves the
existing reachability/paired recipes unchanged by default.

One-update debug run:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name task_reward_debug_n4096_1update_bc1_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 1 \
  --terminal-weight 0 \
  --dense-progress-weight 0 \
  --task-reward-weight 1 \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --force
```

Training update summary:

| global step | mean PPO reward | mean env reward | local terminal distance | task success diag | action delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40960 | 0.4573 | 0.4573 | 0.6022 | 0.2131 | 0.0108 |

Broad held-out local validation on the same 8-timestep manifest:

| policy | final distance | final env reward | max env reward | mean env reward | success-once | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 0.7092 | 0.3979 | 0.4649 | 0.3855 | 0.2456 | - |
| task-reward debug 1 update | 0.7085 | 0.4004 | 0.4673 | 0.3870 | 0.2493 | 0.0005 |

Closed-loop transfer on the same 500-episode learned-goal window as the
lower-noise cached-paired check:

| policy | success | final reward | max reward | action saturation | residual/action delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 0.334 | 0.4776 | 0.5061 | 0.0462 | - |
| task-reward debug 1 update | 0.306 | 0.4601 | 0.4867 | 0.0468 | 0.0005 |

Deltas vs frozen:

| metric | delta |
| --- | ---: |
| success | -0.028 |
| final reward | -0.0176 |
| max reward | -0.0194 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/eval_local_n512_val_b1_manifest_e8_taskdiag.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_500_seed4600000.json`

Interpretation:

Even direct dense task reward, used only as a diagnostic upper bound, produced
the same pattern: weakly positive broad-local metrics and negative closed-loop
transfer. This suggests the current one-segment local R3 update is not merely
limited by the reachability metric. The update is too small/misaligned at the
closed-loop decision distribution, so further local objective sweeps should be
deprioritized in favor of a structurally different training target, stronger
deployment-aligned selection, or a representation/objective that produces much
larger closed-loop-consistent action changes.

## 2026-06-25 - rl-rerun closed-loop per-episode diagnostics

The next selector/gate work should train against closed-loop outcomes, not local
reset proxies. I extended `rl-rerun eval-closed-loop-r{1,2,3}` outputs with
per-episode deployment diagnostics for both frozen and tuned branches:

- `episode_action_delta_l2_mean`
- `episode_action_delta_l2_max`
- `episode_policy_saturation_rate`
- `episode_goal_l2_initial`
- `episode_goal_l2_mean`
- `episode_high_level_decisions`

These arrays line up with `episode_success`, `episode_final_reward`, and
`episode_max_reward`, so a selector can now fit directly on paired closed-loop
wins/regressions instead of inferring from local latent-distance deltas.

Smoke command:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4610000 \
  --num-envs 8 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_diag_smoke_8_seed4610000.json
```

Smoke verification:

| branch | success | max reward | action-delta mean entries | goal-distance entries |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.375 | 0.559 | 8 | 8 |
| task-reward debug | 0.125 | 0.399 | 8 | 8 |

The frozen branch reports zero policy action delta, while the tuned branch
reports nonzero per-episode action deltas. This validates the JSON shape needed
for the next deployment-aligned selector analysis.

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_diag_smoke_8_seed4610000.json`

I then reran the same 500-episode task-reward-debug transfer window with these
diagnostics enabled:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4600000 \
  --num-envs 64 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_diag_500_seed4600000.json
```

Aggregate result matched the previous transfer check:

| policy | success | final reward | max reward | mean action delta |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 | 0.334 | 0.4776 | 0.5061 | 0.0000 |
| task-reward debug 1 update | 0.306 | 0.4601 | 0.4867 | 0.0005 |

Paired outcomes: 60 tuned wins, 74 tuned regressions, and 366 ties.

Feature separation on the 134 discordant episodes:

| tuned-branch feature | AUC for tuned win | oriented AUC | direction |
| --- | ---: | ---: | --- |
| action delta mean | 0.876 | 0.876 | high = win |
| action delta max | 0.570 | 0.570 | high = win |
| policy saturation rate | 0.749 | 0.749 | high = win |
| initial goal L2 | 0.455 | 0.545 | low = win |
| mean goal L2 | 0.702 | 0.702 | high = win |
| high-level decisions | 0.008 | 0.992 | low = win |

The high-level-decision feature is mostly an outcome/episode-length proxy, not a
clean selector input. The important result is that initial goal distance is
again weak, while online/trajectory features such as action delta and saturation
carry much stronger signal. This supports the current direction: any next gate
should be an online deployment-evaluated policy, not a pre-episode selector and
not an offline local-reset selector.

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_diag_500_seed4600000.json`

## 2026-06-25 - Online action-delta gate for rl-rerun closed-loop eval

The diagnostic bank above showed that high mean action delta separated tuned
wins from regressions better than initial state features. I added an eval-only
online gate:

```text
--action-delta-gate-min THRESHOLD
```

At each step, the evaluator computes `||a_tuned - a_base||_2`. If the value is
below the threshold, it executes the frozen base action. The gate is recorded in
the JSON via:

- `action_delta_gate_min`
- `action_delta_gate_rate`
- `episode_action_delta_gate_rate`

Smoke check:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4611000 \
  --num-envs 8 \
  --action-delta-gate-min 0.0006 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_action_delta_gate_smoke_8_seed4611000.json
```

The smoke output had a tuned-branch gate rate of `0.659`, with all new
per-episode gate arrays present.

I then swept three thresholds on the same 500-episode diagnostic window:

| policy | threshold | success | success delta | max reward | max-reward delta | gate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | - | 0.334 | - | 0.5061 | - | 0.000 |
| task-reward debug ungated | - | 0.306 | -0.028 | 0.4867 | -0.0194 | 0.000 |
| action-delta gate | 0.0006 | 0.314 | -0.020 | 0.4916 | -0.0146 | 0.732 |
| action-delta gate | 0.0008 | 0.298 | -0.036 | 0.4791 | -0.0271 | 0.858 |
| action-delta gate | 0.0010 | 0.294 | -0.040 | 0.4776 | -0.0285 | 0.928 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_action_delta_gate_smoke_8_seed4611000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_action_delta_gate0006_500_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_action_delta_gate0008_500_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_action_delta_gate0010_500_seed4600000.json`

Interpretation:

The online action-delta gate is a real deployed gate, not an offline selector,
but it still fails to beat frozen even on the window used to choose thresholds.
The best threshold (`0.0006`) only reduces the task-reward-debug loss from
`-0.028` to `-0.020` success. Stricter thresholds make the policy worse instead
of smoothly recovering frozen behavior, because step-level gating changes the
closed-loop trajectory distribution. This closes the simple action-delta gate
branch for `rl-rerun`: the remaining issue is not just identifying tiny tuned
actions, but learning an update whose useful interventions are robust enough to
survive closed-loop dynamics.

## 2026-06-25 - Oracle-goal transfer check for task-reward debug R3

The previous closed-loop transfer checks used learned high-level goals. To
separate high-level goal mismatch from low-level update damage, I ran the same
task-reward-debug local R3 checkpoint with oracle branch goals generated from
the privileged PPO teacher. I used `--oracle-copy-mode state_dict`; the reported
state-copy error was `1.19e-07`, so the branch-state copy path is accurate for
this diagnostic.

Short 100-episode learned-vs-oracle check on `seed_start=4600000`:

| goal source | policy | success | final reward | max reward | success delta |
| --- | --- | ---: | ---: | ---: | ---: |
| learned | frozen n500 | 0.350 | 0.4867 | 0.5097 | - |
| learned | task-reward debug | 0.300 | 0.4442 | 0.4711 | -0.050 |
| oracle | frozen n500 | 0.340 | 0.4934 | 0.5181 | - |
| oracle | task-reward debug | 0.380 | 0.5182 | 0.5371 | +0.040 |

The 500-episode oracle-goal check stayed positive:

| goal source | policy | success | final reward | max reward | success delta | max-reward delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| learned | frozen n500 | 0.334 | 0.4776 | 0.5061 | - | - |
| learned | task-reward debug | 0.306 | 0.4601 | 0.4867 | -0.028 | -0.0194 |
| oracle | frozen n500 | 0.334 | 0.4928 | 0.5160 | - | - |
| oracle | task-reward debug | 0.350 | 0.5061 | 0.5273 | +0.016 | +0.0113 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_state_dict_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_state_dict_500_seed4600000.json`

Interpretation:

The task-reward-debug update is not uniformly bad: under oracle goals, it gives
a small positive closed-loop effect. Under learned goals, the same update is
negative. This changes the bottleneck diagnosis for this branch. The one-step
task-reward local update can help when the target goal is generated by a strong
oracle continuation, but it interacts badly with the learned high-level goal
distribution. The next useful direction is therefore high-level/goal validity
or oracle-to-learned goal robustness, not further scalar gates around the same
learned-goal deployment.

## 2026-06-26 - Fresh learned-vs-oracle transfer validation

I repeated the 500-episode learned-vs-oracle split on a fresh closed-loop seed
window, `seed_start=4700000`, for the same task-reward-debug local R3
checkpoint.

| seed start | goal source | frozen success | tuned success | success delta | frozen max reward | tuned max reward | max-reward delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4600000 | learned | 0.334 | 0.306 | -0.028 | 0.5061 | 0.4867 | -0.0194 |
| 4600000 | oracle | 0.334 | 0.350 | +0.016 | 0.5160 | 0.5273 | +0.0113 |
| 4700000 | learned | 0.304 | 0.276 | -0.028 | 0.4835 | 0.4679 | -0.0156 |
| 4700000 | oracle | 0.314 | 0.340 | +0.026 | 0.5017 | 0.5185 | +0.0168 |

Two-window aggregate:

| goal source | mean success delta | mean max-reward delta |
| --- | ---: | ---: |
| learned | -0.028 | -0.0175 |
| oracle | +0.021 | +0.0141 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_500_seed4700000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_state_dict_500_seed4700000.json`

Interpretation:

The split replicated cleanly. The task-reward-debug local R3 update is
consistently negative under learned goals across both 500-episode windows, but
positive under privileged-teacher oracle goals. This makes the bottleneck more
specific than "the residual policy is bad": it can improve the same low-level
controller when the segment target is oracle-generated, and fails when deployed
behind the learned high-level goal distribution. The next experiment should
therefore change the goal regime or train for robustness to learned-goal error,
rather than continue scalar fallback gates for the current learned-goal policy.

## 2026-06-26 - Learned-latent goal-manifold validity diagnostic

I added an offline `rl-rerun` diagnostic command:

```text
uv run hcl-poc rl-rerun eval-learned-goal-validity
```

It samples local decisions from the vector validation dataset, compares the
high-level predicted latent goal to the replay future-goal latent, and reports
nearest-neighbor distance to the replay future-goal bank. It also measures how
much the frozen low-level action changes when the goal is swapped from predicted
to replay future.

Full validation command:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-learned-goal-validity \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 4096 \
  --sample-seed 4630000 \
  --output results/rl_rerun/goal_validity/learned_goal_validity_n500_seed0_samples4096_sample4630000.json
```

Results:

| metric | mean | median | p90 |
| --- | ---: | ---: | ---: |
| current to predicted goal L2 | 24.0533 | 23.4380 | 32.5118 |
| current to replay goal L2 | 23.6467 | 23.3056 | 32.3340 |
| predicted to replay goal L2 | 19.8424 | 19.1047 | 25.5270 |
| shuffled replay to matching replay L2 | 27.8451 | 27.5906 | 34.1877 |
| predicted nearest replay-goal L2 | 15.6886 | 15.1471 | 20.0201 |
| replay leave-one-out nearest replay-goal L2 | 14.2781 | 13.6449 | 18.7135 |
| random nearest replay-goal L2 | 25.2707 | 25.2723 | 26.1552 |
| predicted-vs-replay low-action L2 | 0.0109 | 0.0083 | 0.0198 |

Two summary ratios:

| ratio | value |
| --- | ---: |
| predicted nearest replay / replay leave-one-out nearest | 1.099 |
| predicted-to-replay / shuffled-to-replay | 0.713 |

Artifacts:

- `results/rl_rerun/goal_validity/learned_goal_validity_n500_seed0_samples256_sample4630000.json`
- `results/rl_rerun/goal_validity/learned_goal_validity_n500_seed0_samples4096_sample4630000.json`

Interpretation:

The high-level predicted goals are not obviously random or severely
off-manifold in this VAE512 latent space. Their nearest replay-goal distance is
only about `10%` worse than replay leave-one-out nearest-neighbor distance, and
much better than random Gaussian goals. The predicted goals are also closer to
their matched replay future than shuffled replay futures are.

The weak point is still control sensitivity: replacing the predicted goal with
the replay future goal changes the frozen low-level action by only `0.0109` L2
on average. This supports a narrower interpretation of Experiment H for this
checkpoint: high-level goal validity is not the dominant failure by a simple
manifold-distance test. The low-level policy remains too insensitive to
meaningful goal swaps, so the next representation/architecture work should
emphasize goal-conditioned control sensitivity, not just high-level validity
penalties.

## 2026-06-26 - Goal-sensitivity regularized R3 closed-loop transfer

I found an existing `rl-rerun` R3 checkpoint trained with in-batch valid-goal
swap sensitivity regularization:

```text
artifacts/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/latest.pt
```

Recipe:

| field | value |
| --- | ---: |
| global steps | 40960 |
| BC weight | 1.0 |
| terminal weight | 1.0 |
| learning rate | 1e-5 |
| initial logstd | -4.0 |
| goal sensitivity weight | 10.0 |
| goal sensitivity margin | 0.05 |

Training/local diagnostics:

| metric | value |
| --- | ---: |
| train mean terminal distance | 0.6089 |
| train mean action delta L2 | 0.0293 |
| train goal-swap action sensitivity L2 | 0.0319 |
| eval local initial distance | 1.0884 |
| eval local final distance | 0.6215 |
| eval local distance reduction | 0.4669 |
| eval local mean action delta L2 | 0.0011 |

I then ran fresh 500-episode closed-loop transfer checks:

| seed start | goal source | frozen success | tuned success | success delta | frozen max reward | tuned max reward | max-reward delta | action delta L2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4800000 | learned | 0.306 | 0.324 | +0.018 | 0.4954 | 0.5042 | +0.0088 | 0.000973 |
| 4800000 | oracle | 0.328 | 0.340 | +0.012 | 0.5164 | 0.5231 | +0.0068 | 0.000982 |
| 4900000 | learned | 0.294 | 0.296 | +0.002 | 0.4793 | 0.4833 | +0.0040 | 0.000940 |

Learned-goal aggregate over the two fresh windows:

| metric | mean delta |
| --- | ---: |
| success | +0.010 |
| max reward | +0.0064 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/closed_loop_learned_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/closed_loop_oracle_state_dict_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/closed_loop_learned_500_seed4900000.json`

Interpretation:

This is the first `rl-rerun` local R3 variant in the current branch that
transfers positively under learned high-level goals on fresh 500-episode
windows. The effect is small, but the sign is better than the task-reward-debug
checkpoint, which averaged `-0.028` learned-goal success delta on its two
500-episode windows. It also remains positive under oracle goals on the checked
window.

The useful change is not simply larger arbitrary actions: closed-loop action
delta is still only about `0.001` L2. The sensitivity regularizer appears to
move the tuned policy into a less brittle regime where small goal-conditioned
action changes do not hurt learned-goal deployment. This is not yet a robust
policy improvement, but it is now the best `rl-rerun` lead for the VAE512
learned-latent branch. The next validation should either run more fresh
learned-goal windows or train a larger/longer sensitivity-regularized variant,
while preserving deployment checks as the selector.

## 2026-06-26 - Longer goal-sensitivity R3 scale-up check

I trained a five-update version of the same sensitivity-regularized R3 recipe:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name goal_sensitivity_w10_m005_5update_bc1_lr1e5_logstd4 \
  --steps 204800 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --learning-rate 1e-5 \
  --initial-logstd -4.0 \
  --goal-sensitivity-weight 10.0 \
  --goal-sensitivity-margin 0.05
```

Final training metrics:

| metric | value |
| --- | ---: |
| global steps | 204800 |
| train mean terminal distance | 0.6929 |
| train mean action delta L2 | 0.0294 |
| train goal-swap action sensitivity L2 | 0.0307 |
| train task-success diagnostic rate | 0.2075 |

One-batch local eval on `pusht_vector_state_demos_n4096_val_b1.h5`:

| run | initial distance | final distance | reduction | action delta L2 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| one-update sensitivity | 1.0884 | 0.6215 | 0.4669 | 0.001065 | 0.0108 |
| five-update sensitivity | 0.8286 | 0.5700 | 0.2586 | 0.002840 | 0.0018 |

Fresh learned-goal closed-loop transfer on `seed_start=4800000`:

| run | frozen success | tuned success | success delta | max-reward delta | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| one-update sensitivity | 0.306 | 0.324 | +0.018 | +0.0088 | 0.000973 |
| five-update sensitivity | 0.306 | 0.310 | +0.004 | +0.0017 | 0.002954 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_5update_bc1_lr1e5_logstd4/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_5update_bc1_lr1e5_logstd4/history.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_5update_bc1_lr1e5_logstd4/eval_local_1batch_val_b1.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_5update_bc1_lr1e5_logstd4/closed_loop_learned_500_seed4800000.json`

Interpretation:

Longer training did not improve the sensitivity lead. The five-update variant
keeps a positive sign on the checked learned-goal window, but it is weaker than
the one-update checkpoint and uses larger closed-loop action deltas. This
repeats the broader pattern from the effect32 R3 work: pushing the local
objective longer can degrade deployment even when the local/training signals
look plausible.

For now, the one-update sensitivity checkpoint remains the best VAE512
`rl-rerun` lead. The next scale-up should not simply continue more updates with
the same objective; it should either tune the sensitivity target/BC tradeoff or
validate the one-update checkpoint on more deployment windows.

## 2026-06-26 - Broader validation of one-update sensitivity R3

I added two more fresh learned-goal 500-episode windows for the one-update
goal-sensitivity checkpoint:

```text
artifacts/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/latest.pt
```

Four-window learned-goal transfer:

| seed start | frozen success | tuned success | success delta | frozen max reward | tuned max reward | max-reward delta | action delta L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4800000 | 0.306 | 0.324 | +0.018 | 0.4954 | 0.5042 | +0.0088 | 0.000973 |
| 4900000 | 0.294 | 0.296 | +0.002 | 0.4793 | 0.4833 | +0.0040 | 0.000940 |
| 5000000 | 0.340 | 0.342 | +0.002 | 0.5155 | 0.5165 | +0.0009 | 0.000965 |
| 5100000 | 0.306 | 0.286 | -0.020 | 0.4851 | 0.4752 | -0.0099 | 0.000969 |

Aggregate:

| metric | mean |
| --- | ---: |
| success delta | +0.001 |
| max-reward delta | +0.0010 |
| action delta L2 | 0.000962 |
| action saturation rate | 0.0477 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/closed_loop_learned_500_seed5000000.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/closed_loop_learned_500_seed5100000.json`

Interpretation:

The sensitivity-regularized checkpoint is not a robust learned-goal improvement
after broader validation. The first two fresh windows looked positive, but the
fourth window is negative enough to reduce the four-window mean to essentially
neutral. This is still better than the task-reward-debug checkpoint's clearly
negative learned-goal transfer, but it is not evidence of a deployable policy
improvement.

The practical conclusion changes again: goal-sensitivity regularization is a
useful diagnostic and may reduce harm, but this specific one-update recipe
should be treated as neutral. The next experiment should adjust the
sensitivity/BC tradeoff or representation/conditioning, with deployment
validation as the primary selector.

## 2026-06-26 - Stronger sensitivity-weight R3 tradeoff check

I trained a one-update sensitivity variant that changes only the goal-swap
sensitivity weight from `10` to `30`, keeping the same dataset, BC weight,
terminal weight, learning rate, logstd, and margin:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name goal_sensitivity_w30_m005_1update_bc1_lr1e5_logstd4 \
  --steps 40960 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --learning-rate 1e-5 \
  --initial-logstd -4.0 \
  --goal-sensitivity-weight 30.0 \
  --goal-sensitivity-margin 0.05
```

The rollout-side training metrics were effectively unchanged from the
`weight=10` run because they are measured on the collected rollout before the
PPO update:

| run | terminal distance | train action delta L2 | goal-swap action sensitivity L2 | sensitivity loss |
| --- | ---: | ---: | ---: | ---: |
| weight 10 | 0.6089 | 0.029265 | 0.031907 | 0.000735 |
| weight 30 | 0.6089 | 0.029265 | 0.031910 | 0.000735 |

Aligned one-batch local eval on `pusht_vector_state_demos_n4096_val_b1.h5`:

| run | initial distance | final distance | reduction | action delta L2 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| weight 10 | 0.8286 | 0.5738 | 0.2548 | 0.000845 | 0.0020 |
| weight 30 | 0.8286 | 0.5751 | 0.2535 | 0.000834 | 0.0019 |

Fresh learned-goal closed-loop transfer on the two shared 500-episode windows:

| run | seed start | frozen success | tuned success | success delta | max-reward delta | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 10 | 4800000 | 0.306 | 0.324 | +0.018 | +0.0088 | 0.000973 |
| weight 10 | 4900000 | 0.294 | 0.296 | +0.002 | +0.0040 | 0.000940 |
| weight 30 | 4800000 | 0.306 | 0.332 | +0.026 | +0.0168 | 0.000977 |
| weight 30 | 4900000 | 0.294 | 0.280 | -0.014 | -0.0040 | 0.000948 |

Shared two-window aggregate:

| run | mean success delta | mean max-reward delta | mean action delta L2 |
| --- | ---: | ---: | ---: |
| weight 10 | +0.010 | +0.0064 | 0.000957 |
| weight 30 | +0.006 | +0.0064 | 0.000962 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w30_m005_1update_bc1_lr1e5_logstd4/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w30_m005_1update_bc1_lr1e5_logstd4/history.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w30_m005_1update_bc1_lr1e5_logstd4/eval_local_1batch_val_b1.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w30_m005_1update_bc1_lr1e5_logstd4/closed_loop_learned_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w30_m005_1update_bc1_lr1e5_logstd4/closed_loop_learned_500_seed4900000.json`

Interpretation:

Increasing the sensitivity weight from `10` to `30` does not produce a clearer
deployment improvement. It improves the first window but regresses the second,
and the two-window mean is slightly weaker than the original `weight=10`
checkpoint. Local eval is also essentially unchanged. The sensitivity objective
is therefore not simply underweighted; the current formulation can move small
goal-conditioned actions around, but it still does not create a robust
learned-goal transfer improvement.

## 2026-06-26 - Lower-BC sensitivity R3 tradeoff check

I trained a one-update sensitivity variant that changes only the BC anchor from
`1.0` to `0.3`, keeping the sensitivity weight/margin, dataset, terminal
weight, learning rate, and logstd fixed:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name goal_sensitivity_w10_m005_1update_bc03_lr1e5_logstd4 \
  --steps 40960 \
  --bc-weight 0.3 \
  --terminal-weight 1.0 \
  --learning-rate 1e-5 \
  --initial-logstd -4.0 \
  --goal-sensitivity-weight 10.0 \
  --goal-sensitivity-margin 0.05
```

Aligned one-batch local eval on `pusht_vector_state_demos_n4096_val_b1.h5`:

| run | initial distance | final distance | reduction | action delta L2 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC 1.0, weight 10 | 0.8286 | 0.5738 | 0.2548 | 0.000845 | 0.0020 |
| BC 0.3, weight 10 | 0.8286 | 0.5750 | 0.2536 | 0.000842 | 0.0020 |
| BC 1.0, weight 30 | 0.8286 | 0.5751 | 0.2535 | 0.000834 | 0.0019 |

Fresh learned-goal closed-loop transfer on `seed_start=4800000`:

| run | frozen success | tuned success | success delta | max-reward delta | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC 1.0, weight 10 | 0.306 | 0.324 | +0.018 | +0.0088 | 0.000973 |
| BC 0.3, weight 10 | 0.306 | 0.288 | -0.018 | -0.0136 | 0.000958 |
| BC 1.0, weight 30 | 0.306 | 0.332 | +0.026 | +0.0168 | 0.000977 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_1update_bc03_lr1e5_logstd4/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_1update_bc03_lr1e5_logstd4/history.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_1update_bc03_lr1e5_logstd4/eval_local_1batch_val_b1.json`
- `results/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_1update_bc03_lr1e5_logstd4/closed_loop_learned_500_seed4800000.json`

Interpretation:

Relaxing the BC anchor to `0.3` does not create larger useful interventions.
The aligned local eval is effectively unchanged, action delta is not larger,
and the deployment check flips negative on the same window where the original
`BC=1.0` sensitivity checkpoint was positive. This closes the simple
"same sensitivity objective, weaker BC" branch for this checkpoint. The next
meaningful change should alter the target formulation or representation rather
than just scale BC/sensitivity coefficients.

## 2026-06-26 - Cached paired reward plus sensitivity local-gate check

After coefficient scaling failed, I tested a target-formulation combination:
cached paired terminal improvement plus the goal-swap sensitivity regularizer.
This keeps the cached paired reward's exact-base local target, and adds the
same sensitivity loss used in the harm-reduction diagnostic.

Training command:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name paired_cached_n4096_1update_bc1_sensw10_m005 \
  --steps 40960 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --reward-mode paired \
  --num-minibatches 8 \
  --checkpoint-every-updates 1 \
  --goal-sensitivity-weight 10.0 \
  --goal-sensitivity-margin 0.05
```

Training metrics after one update:

| run | paired improvement | fraction improved | terminal distance | action delta L2 | goal-swap action sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| paired only | -0.0098 | 0.4783 | 0.6089 | 0.029265 | - |
| sensitivity only | - | - | 0.6089 | 0.029265 | 0.031907 |
| paired + sensitivity | -0.0098 | 0.4783 | 0.6089 | 0.029265 | 0.031910 |

Matched local validation on
`results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json`:

| policy | initial distance | final distance | reduction | reduction fraction | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | 1.0671 | 0.6020 | 0.4651 | 0.7969 | - |
| paired only | 1.0671 | 0.6066 | 0.4605 | 0.7966 | 0.001230 |
| paired + sensitivity | 1.0671 | 0.6047 | 0.4624 | 0.7959 | 0.001227 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_1update_bc1_sensw10_m005/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_1update_bc1_sensw10_m005/history.json`
- `results/rl_rerun/local_r3/n500/seed0/paired_cached_n4096_1update_bc1_sensw10_m005/eval_local_n4096_val_b1_manifest.json`

Interpretation:

Adding sensitivity regularization to cached paired reward slightly improves the
paired-only local final distance (`0.6066 -> 0.6047`), but it remains worse
than the frozen baseline (`0.6020`) on the matched validation manifest. The
local gate therefore fails, so I did not spend a closed-loop deployment eval on
this checkpoint. This is useful negative evidence: the paired target and
sensitivity regularizer are not obviously conflicting, but their simple sum is
still too weak to beat frozen local reaching.

## 2026-06-26 - AE256 FiLM D_phi representation screen

After the VAE512 `rl-rerun` sensitivity and paired branches remained neutral, I
screened an existing alternative representation/architecture candidate:
`ae256_film`. It had learned-interface artifacts and goal diagnostics already,
but no learned reachability distance.

I trained and evaluated D_phi:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-reachability-distance \
  --candidate ae256_film \
  --seed 0

uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-reachability-distance \
  --candidate ae256_film \
  --seed 0 \
  --force
```

Reachability-distance diagnostics:

| candidate | temporal MSE | temporal Spearman | near/far acc | shuffled AUC | demo-decrease acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| ae256_film | 0.00793 | 0.9334 | 0.9863 | 0.8678 | 0.6936 |
| effect32_film | 0.03267 | 0.8333 | 0.9275 | 0.9074 | 0.7396 |
| vae512_w2048_b1e6 | 0.00793 | 0.9296 | 0.9841 | 0.8769 | 0.7031 |

Goal-use diagnostics from the existing 5000-sample offline gate:

| candidate | goal-shuffle action L2 | max valid-goal sensitivity L2 | frame-shuffle action L2 |
| --- | ---: | ---: | ---: |
| ae256_film | 0.2506 | 0.0937 | 0.8645 |
| effect32_film | 0.0622 | 0.0368 | 0.9503 |
| vae512_b1e6_film | 0.2783 | 0.1266 | 0.8213 |

Then I ran the same 40k D_phi terminal-only final-layer R3 smoke shape used for
effect32:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate ae256_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_ae256_dphi_r3_4096_terminal_smoke_40k_bc10 \
  --steps 40960 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/ae256_film/seed0/d_phi.pt \
  --num-envs 4096
```

Training metrics:

| candidate | terminal distance | action delta L2 | saturation |
| --- | ---: | ---: | ---: |
| ae256_film R3 | 0.7728 | 0.02931 | 0.3218 |
| effect32_film R3 | 0.5757 | 0.02931 | 0.2587 |

Closed-loop check on `seed_start=3500000`, 500 episodes:

| policy | success | max reward | raw reduction | selected reduction | goal reach | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ae256_film frozen | 0.596 | 0.7100 | 0.2938 | 0.0701 | 0.8133 | 0.000000 |
| ae256_film R3 | 0.586 | 0.7057 | 0.2916 | 0.0698 | 0.8117 | 0.001307 |
| effect32_film frozen | 0.634 | 0.7378 | 0.3965 | 0.0731 | 0.7176 | 0.000000 |
| effect32_film R3 | 0.684 | 0.7725 | 0.4097 | 0.0801 | 0.7307 | 0.001006 |

Artifacts:

- `artifacts/incremental/reachability_distance/ae256_film/seed0/d_phi.pt`
- `results/incremental/reachability_distance/ae256_film/seed0/eval.json`
- `artifacts/incremental/low_level_rl/ae256_film/seed0/hcl_next_ae256_dphi_r3_4096_terminal_smoke_40k_bc10/latest.pt`
- `results/incremental/low_level_rl/ae256_film/seed0/hcl_next_ae256_dphi_r3_4096_terminal_smoke_40k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/ae256_film/seed0/hcl_next_ae256_dphi_frozen_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/ae256_film/seed0/hcl_next_ae256_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3500000/eval_500_seed3500000.json`

Interpretation:

`ae256_film` is not the next RL base despite its stronger one-step goal
sensitivity. D_phi fits the AE latent well on temporal/near-far metrics, but
the closed-loop frozen hierarchy is weaker than effect32, and the analogous
40k D_phi R3 update slightly regresses task success and local reductions.
This reinforces the earlier lesson that one-step goal sensitivity alone is not
enough; the representation also needs a deployment-useful closed-loop base and
a reachability metric whose local improvements transfer to task behavior.

## 2026-06-26 - VAE512 FiLM D_phi R3 representation screen

After `ae256_film` failed to beat effect32, I completed the same
reachability-distance/R3 screen for the existing `vae512_b1e6_film` hierarchy.
It already had learned-interface and goal-use artifacts, but no D_phi checkpoint
under `artifacts/incremental/reachability_distance/vae512_b1e6_film/seed0`.

I trained and evaluated D_phi:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-reachability-distance \
  --candidate vae512_b1e6_film \
  --seed 0

uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-reachability-distance \
  --candidate vae512_b1e6_film \
  --seed 0 \
  --force
```

Reachability-distance diagnostics:

| candidate | temporal MSE | temporal Spearman | near/far acc | shuffled AUC | demo-decrease acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| vae512_b1e6_film | 0.00793 | 0.9296 | 0.9841 | 0.8769 | 0.7031 |
| ae256_film | 0.00793 | 0.9334 | 0.9863 | 0.8678 | 0.6936 |
| effect32_film | 0.03267 | 0.8333 | 0.9275 | 0.9074 | 0.7396 |
| vae512_w2048_b1e6 | 0.00793 | 0.9296 | 0.9841 | 0.8769 | 0.7031 |

Existing VAE512 FiLM closed-loop and goal-use diagnostics before R3:

| goal source | episodes | success | max reward | final reward |
| --- | ---: | ---: | ---: | ---: |
| learned | 200 | 0.425 | 0.5916 | 0.5730 |
| oracle | 200 | 0.535 | 0.6716 | 0.6646 |
| shuffled | 200 | 0.010 | 0.2046 | 0.1404 |

| diagnostic | value |
| --- | ---: |
| goal-shuffle action L2 | 0.2783 |
| max valid-goal sensitivity L2 | 0.1266 |
| frame-shuffle action L2 | 0.8213 |
| previous-action shuffle action L2 | 0.1129 |

Then I ran the same 40k terminal-only final-layer R3 shape used for ae256 and
effect32:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate vae512_b1e6_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_vae512film_dphi_r3_4096_terminal_smoke_40k_bc10 \
  --steps 40960 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/vae512_b1e6_film/seed0/d_phi.pt \
  --num-envs 4096
```

Training metrics:

| candidate | terminal distance | action delta L2 | saturation |
| --- | ---: | ---: | ---: |
| vae512_b1e6_film R3 | 0.5207 | 0.02931 | 0.1933 |
| ae256_film R3 | 0.7728 | 0.02931 | 0.3218 |
| effect32_film R3 | 0.5757 | 0.02931 | 0.2587 |

Closed-loop check on `seed_start=3500000`, 500 episodes:

| policy | success | max reward | raw reduction | selected reduction | goal reach | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vae512_b1e6_film frozen | 0.418 | 0.5797 | 0.1495 | 0.1116 | 0.8486 | 0.000000 |
| vae512_b1e6_film R3 | 0.478 | 0.6271 | 0.1611 | 0.1150 | 0.8936 | 0.001637 |
| ae256_film frozen | 0.596 | 0.7100 | 0.2938 | 0.0701 | 0.8133 | 0.000000 |
| ae256_film R3 | 0.586 | 0.7057 | 0.2916 | 0.0698 | 0.8117 | 0.001307 |
| effect32_film frozen | 0.634 | 0.7378 | 0.3965 | 0.0731 | 0.7176 | 0.000000 |
| effect32_film R3 | 0.684 | 0.7725 | 0.4097 | 0.0801 | 0.7307 | 0.001006 |

Artifacts:

- `artifacts/incremental/reachability_distance/vae512_b1e6_film/seed0/d_phi.pt`
- `results/incremental/reachability_distance/vae512_b1e6_film/seed0/eval.json`
- `artifacts/incremental/low_level_rl/vae512_b1e6_film/seed0/hcl_next_vae512film_dphi_r3_4096_terminal_smoke_40k_bc10/latest.pt`
- `results/incremental/low_level_rl/vae512_b1e6_film/seed0/hcl_next_vae512film_dphi_r3_4096_terminal_smoke_40k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/vae512_b1e6_film/seed0/hcl_next_vae512film_dphi_frozen_final500_seed3500000/eval_500_seed3500000.json`
- `results/incremental/low_level_rl/vae512_b1e6_film/seed0/hcl_next_vae512film_dphi_r3_4096_terminal_smoke_40k_bc10_final500_seed3500000/eval_500_seed3500000.json`

Interpretation:

VAE512 FiLM is interesting diagnostically because this R3 update improves its
own weak frozen baseline by `+0.060` success and `+0.0474` max reward on the
500-episode check. But the absolute policy is still far behind effect32, and
even behind the frozen ae256 hierarchy. Its D_phi metrics are nearly identical
to the VAE512 concat metrics, so the FiLM architecture's stronger one-step
goal-use did not translate into a stronger deployment base. This leaves
`effect32_film` as the best current real-compatible representation for low-level
RL transfer despite its weaker temporal D_phi MSE.

## 2026-06-26 - Effect32 scene FiLM supervised screen

The repository already had an `effect32_scene_film` checkpoint with only
20-episode learned/oracle checks. Because `effect32_film` remains the leading
real-compatible representation, I ran the same supervised gate on the scene-only
effect variant before spending D_phi/R3 compute.

Commands:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --n-demo 1800 \
  --candidate effect32_scene_film \
  --samples 5000 \
  --horizons 2,5,10 \
  --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_scene_film \
  --goal-source learned \
  --episodes 200 \
  --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_scene_film \
  --goal-source oracle \
  --episodes 200 \
  --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_scene_film \
  --goal-source shuffled \
  --episodes 200 \
  --force
```

Offline goal-use diagnostics:

| candidate | goal-shuffle L2 | max goal sensitivity | frame-shuffle L2 | previous-action shuffle L2 |
| --- | ---: | ---: | ---: | ---: |
| effect32_scene_film | 0.0630 | 0.0352 | 0.9551 | 0.1335 |
| effect32_film | 0.0622 | 0.0368 | 0.9503 | 0.1326 |
| ae256_film | 0.2506 | 0.0937 | 0.8645 | 0.1161 |
| vae512_b1e6_film | 0.2783 | 0.1266 | 0.8213 | 0.1129 |

Closed-loop check on the default learned-interface bank
(`seed_start=2100000`, 200 episodes):

| candidate | goal source | success | max reward | final reward | shuffled goal L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| effect32_scene_film | learned | 0.590 | 0.7064 | 0.6979 | - |
| effect32_scene_film | oracle | 0.655 | 0.7551 | 0.7438 | - |
| effect32_scene_film | shuffled | 0.280 | 0.4599 | 0.4269 | 6.1045 |

Artifacts:

- `results/incremental/goal_diagnostics/n1800/seed0/effect32_scene_film/diagnostics.json`
- `results/incremental/learned_interface/effect32_scene_film/seed0/learned_hierarchy_eval_200.json`
- `results/incremental/learned_interface/effect32_scene_film/seed0/oracle_hierarchy_eval_200.json`
- `results/incremental/learned_interface/effect32_scene_film/seed0/shuffled_hierarchy_eval_200.json`

Interpretation:

`effect32_scene_film` is a supervised regression from `effect32_film`. Its
offline goal-use is essentially identical to the lead effect32 FiLM checkpoint,
but learned closed-loop success drops from `0.655` to `0.590` on the standard
200-episode screen. Shuffled-goal success collapses to `0.280`, so the policy
does use goals in closed loop, but not enough better than `effect32_film` to
justify a D_phi/R3 branch. I skipped reachability training and PPO for this
candidate.

## 2026-06-26 - Exported R3 hierarchy and fixed-seed protocol check

I added a small utility to export an R3 direct-last-layer low-level checkpoint
back into a normal learned-interface hierarchy checkpoint:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  export-direct-hierarchy \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --output artifacts/incremental/learned_interface/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_best_hierarchy.pt \
  --force
```

I also extended `incremental learned-interface-eval` with `--checkpoint` so a
specific hierarchy file can be evaluated without requiring a config candidate
entry. A direct action-equivalence check showed the exported low-level policy
and the R3 direct agent produce identical actions on identical low-level
conditions:

```text
max_abs_action_diff = 0.0
mean_abs_action_diff = 0.0
```

Then I evaluated the exported hierarchy on the same 200 fixed seeds used for
the previous effect32 learned/oracle/shuffled screen:

```bash
uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_r3_best_export \
  --checkpoint artifacts/incremental/learned_interface/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_best_hierarchy.pt \
  --goal-source learned \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_r3_best_export \
  --checkpoint artifacts/incremental/learned_interface/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_best_hierarchy.pt \
  --goal-source oracle \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_r3_best_export \
  --checkpoint artifacts/incremental/learned_interface/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_best_hierarchy.pt \
  --goal-source shuffled \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force
```

Exported hierarchy eval:

| policy | goal source | success | max reward | final reward |
| --- | --- | ---: | ---: | ---: |
| frozen hierarchy | learned | 0.645 | 0.7420 | 0.7347 |
| frozen hierarchy | oracle | 0.645 | 0.7464 | 0.7399 |
| frozen hierarchy | shuffled | 0.280 | 0.4602 | 0.4255 |
| exported R3 hierarchy | learned | 0.635 | 0.7393 | 0.7305 |
| exported R3 hierarchy | oracle | 0.605 | 0.7166 | 0.7066 |
| exported R3 hierarchy | shuffled | 0.265 | 0.4433 | 0.4065 |

Because this contradicted the earlier low-level vector eval result, I ran an
exact fixed-seed low-level serial check on the first 100 seeds of the same
window:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial100_seed3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --episodes 100 \
  --seed-start 3500000 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Fixed-seed low-level serial result:

| policy | success | max reward | raw reduction | selected reduction | goal reach | action delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen serial100 | 0.600 | 0.7085 | 0.3993 | 0.0845 | 0.7190 | 0.000000 |
| R3 serial100 | 0.670 | 0.7618 | 0.4213 | 0.0824 | 0.7300 | 0.000981 |

Paired exact-seed comparison:

| episodes | improvements | regressions | net | success delta |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 14 | 7 | +7 | +0.070 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_best_hierarchy.pt`
- `results/incremental/learned_interface/effect32_film_r3_best_export/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_r3_best_export/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_r3_best_export/seed0/shuffled_hierarchy_eval_200_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial100_seed3500000/paired_vs_frozen_serial100_seed3500000.json`

Interpretation:

The export is technically correct at the low-policy level, but the
learned-interface evaluator is not equivalent to the low-level RL evaluator for
this checkpoint. The fixed-seed low-level serial check preserves the positive
R3 sign on exact episode IDs (`+0.070` success, net `+7` paired episodes), while
the exported hierarchy path regresses both learned and oracle goal sources.
Therefore the current low-level R3 result should still be judged with
`low-level-rl eval-serial`/`eval`, not by exporting into the learned-interface
closed-loop evaluator. The oracle-goal separation remains unresolved until the
two evaluator protocols are reconciled or oracle goals are implemented directly
inside the low-level RL rollout path.

## 2026-06-26 - Oracle-goal low-level serial R3 separation

I added `--goal-source learned|oracle` to `low-level-rl eval-serial`. In oracle
mode, the exact-seed serial evaluator:

1. copies the current simulator state into a one-env branch at each low-level
   replan;
2. rolls the privileged PPO teacher forward for `horizon_steps`;
3. encodes the branch endpoint as the held local goal;
4. evaluates the frozen or tuned low-level policy with the same action path as
   regular serial eval.

The evaluator records:

- `goal_source`
- `normalized_goal_prediction_l2`
- `replay_current_state_error_max`

Smoke command:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_oracle_serial_smoke5_seed3500000 \
  --episodes 5 \
  --seed-start 3500000 \
  --goal-source oracle \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Smoke result:

| metric | value |
| --- | ---: |
| success | 0.800 |
| max reward | 0.8490 |
| raw reduction | 0.7758 |
| segment goal reach | 0.820 |
| predicted-vs-oracle goal L2 | 3.2654 |
| replay state error max | 1.19e-07 |

Then I ran the fixed 100-seed oracle-goal comparison:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_oracle_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --goal-source oracle \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_oracle_serial100_seed3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --episodes 100 \
  --seed-start 3500000 \
  --goal-source oracle \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Result:

| policy | goal source | success | max reward | raw reduction | selected reduction | goal reach | action delta L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | learned | 0.600 | 0.7085 | 0.3993 | 0.0845 | 0.7190 | 0.000000 |
| R3 | learned | 0.670 | 0.7618 | 0.4213 | 0.0824 | 0.7300 | 0.000981 |
| frozen | oracle | 0.680 | 0.7673 | 0.8155 | 0.1004 | 0.7780 | 0.000000 |
| R3 | oracle | 0.710 | 0.7924 | 0.8118 | 0.1028 | 0.7630 | 0.000978 |

Paired exact-seed comparison:

| goal source | episodes | improvements | regressions | net | success delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned | 100 | 14 | 7 | +7 | +0.070 |
| oracle | 100 | 11 | 8 | +3 | +0.030 |

Oracle-goal diagnostic means:

| policy | predicted-vs-oracle goal L2 | replay state error max |
| --- | ---: | ---: |
| frozen oracle | 3.5784 | 1.19e-07 |
| R3 oracle | 3.6382 | 1.19e-07 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_oracle_serial_smoke5_seed3500000/serial_eval_5_seed3500000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_oracle_serial100_seed3500000/serial_eval_100_seed3500000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_oracle_serial100_seed3500000/serial_eval_100_seed3500000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_oracle_serial100_seed3500000/paired_vs_frozen_oracle_serial100_seed3500000.json`

Interpretation:

Oracle local goals are a real ceiling raiser: frozen effect32 improves from
`0.600` learned-goal success to `0.680` oracle-goal success on the same exact
100 seeds. The R3 low-level update remains positive under oracle goals, but the
effect is smaller (`+0.030`, net `+3`) than under learned goals (`+0.070`, net
`+7`). This means the current bottleneck is split: learned high-level goals
leave significant performance on the table, but the low-level R3 update is not
merely compensating for learned-goal error. It still adds a small benefit when
given stronger local goals.

## 2026-06-26 - Fresh-bank validation of oracle-goal serial R3

The first 100-seed oracle-goal serial check was promising but too small to
trust. I reran the same exact-seed learned/oracle comparison on a fresh
`seed_start=3600000` bank.

Commands:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --goal-source learned \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial100_seed3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --episodes 100 \
  --seed-start 3600000 \
  --goal-source learned \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_oracle_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --goal-source oracle \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_oracle_serial100_seed3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --episodes 100 \
  --seed-start 3600000 \
  --goal-source oracle \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Fresh-bank result:

| goal source | policy | success | max reward | raw reduction | selected reduction | goal reach | action delta L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | frozen | 0.690 | 0.7781 | 0.3923 | 0.0744 | 0.7290 | 0.000000 |
| learned | R3 | 0.710 | 0.7958 | 0.4151 | 0.0739 | 0.7460 | 0.000921 |
| oracle | frozen | 0.760 | 0.8310 | 0.8320 | 0.1192 | 0.8180 | 0.000000 |
| oracle | R3 | 0.690 | 0.7846 | 0.8260 | 0.1229 | 0.7910 | 0.000964 |

Paired exact-seed comparison:

| seed start | goal source | episodes | frozen | R3 | improvements | regressions | net | success delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3500000 | learned | 100 | 0.600 | 0.670 | 14 | 7 | +7 | +0.070 |
| 3500000 | oracle | 100 | 0.680 | 0.710 | 11 | 8 | +3 | +0.030 |
| 3600000 | learned | 100 | 0.690 | 0.710 | 14 | 12 | +2 | +0.020 |
| 3600000 | oracle | 100 | 0.760 | 0.690 | 12 | 19 | -7 | -0.070 |

Two-bank aggregate:

| goal source | episodes | frozen | R3 | improvements | regressions | net | success delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | 200 | 0.645 | 0.690 | 28 | 19 | +9 | +0.045 |
| oracle | 200 | 0.720 | 0.700 | 23 | 27 | -4 | -0.020 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial100_seed3600000/serial_eval_100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial100_seed3600000/serial_eval_100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial100_seed3600000/paired_vs_frozen_serial100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_oracle_serial100_seed3600000/serial_eval_100_seed3600000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_oracle_serial100_seed3600000/serial_eval_100_seed3600000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_oracle_serial100_seed3600000/paired_vs_frozen_oracle_serial100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial_goal_source_2bank_summary.json`

Interpretation:

The fresh bank invalidates the stronger oracle-goal claim from the previous
100-seed check. Oracle goals consistently improve the frozen policy, but the R3
low-level update does not validate under oracle goals: the two-bank oracle
aggregate is `0.720 -> 0.700` with net `-4` paired episodes. Under learned
goals the R3 update remains mildly positive over these exact serial banks
(`0.645 -> 0.690`, net `+9`), but that should now be interpreted as a narrow
distribution-specific effect, not as evidence of a generally better low-level
controller. This points back to the plan's broader conclusion: the next serious
work should change the training objective or high-level/local-goal formulation,
not keep tuning this R3 checkpoint.

## 2026-06-25 - Learned-vs-oracle goal diagnostics in rl-rerun

I added a default-off closed-loop diagnostic flag:

```text
--diagnose-oracle-goals
```

When enabled, `eval-closed-loop-r{1,2,3}` generates privileged teacher branch
goals in parallel and records learned-vs-oracle goal distances without changing
the deployed policy if `--goal-source learned` is used. New fields include:

- `predicted_oracle_goal_l2_mean`
- `episode_predicted_oracle_goal_l2_initial`
- `episode_predicted_oracle_goal_l2_mean`
- `episode_current_oracle_goal_l2_initial`
- `episode_current_oracle_goal_l2_mean`

Smoke check:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4620000 \
  --num-envs 8 \
  --goal-source learned \
  --oracle-copy-mode state_dict \
  --diagnose-oracle-goals \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_oraclediag_smoke_8_seed4620000.json
```

The smoke JSON kept `goal_source=learned`, reported `diagnose_oracle_goals=true`,
and had oracle-diagnostic arrays with one entry per episode. The state-copy
error remained `1.19e-07`.

Then I ran the 100-episode learned-goal diagnostic bank on `seed_start=4600000`:

| policy | success | max reward | predicted-oracle goal L2 | current-oracle goal L2 |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 | 0.350 | 0.5097 | 27.70 | 29.77 |
| task-reward debug | 0.300 | 0.4711 | 28.21 | 29.39 |

Paired outcomes: 8 tuned wins, 13 tuned regressions, and 79 ties.

Feature separation over the 21 discordant episodes:

| tuned-branch feature | AUC for tuned win | oriented AUC | direction |
| --- | ---: | ---: | --- |
| initial predicted-oracle goal L2 | 0.452 | 0.548 | low = win |
| mean predicted-oracle goal L2 | 0.442 | 0.558 | low = win |
| initial current-oracle goal L2 | 0.529 | 0.529 | high = win |
| mean current-oracle goal L2 | 0.865 | 0.865 | high = win |
| initial current-learned goal L2 | 0.635 | 0.635 | high = win |
| mean current-learned goal L2 | 0.923 | 0.923 | high = win |
| action delta mean | 0.933 | 0.933 | high = win |
| policy saturation rate | 0.885 | 0.885 | high = win |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_oraclediag_smoke_8_seed4620000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_oraclediag_100_seed4600000.json`

Interpretation:

On this small diagnostic bank, learned-vs-oracle goal distance by itself is not
a strong separator of tuned wins and regressions. The stronger separators are
online trajectory features: whether the rollout is in a high-distance/high-action
delta/high-saturation regime. This means the learned-high failure is not simply
"predicted goal far from oracle"; it is more specifically that the tuned
low-level intervention is useful only in some difficult closed-loop regimes.
The next serious selector/objective should use online closed-loop state, not
static high-level goal error alone.

## 2026-06-25 - Online multifeature gate for rl-rerun

I added an eval-only `--goal-l2-gate-min` option to `rl-rerun`
`eval-closed-loop-r{1,2,3}`. It gates the tuned policy by the current segment's
learned-goal distance. When combined with `--action-delta-gate-min`, the tuned
action is used only when both conditions pass:

```text
||a_tuned - a_base||_2 >= action_delta_gate_min
current-to-learned-goal L2 >= goal_l2_gate_min
```

The evaluator records:

- `goal_l2_gate_min`
- `goal_l2_gate_rate`
- `episode_goal_l2_gate_rate`

Smoke command:

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4621000 \
  --num-envs 8 \
  --goal-source learned \
  --action-delta-gate-min 0.0006 \
  --goal-l2-gate-min 27 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_multigate_smoke_8_seed4621000.json
```

The smoke output had the expected action and goal gate-rate arrays.

I then tested the multifeature gate on the same 100-episode learned-goal window
used for the oracle-goal diagnostics:

| policy | action threshold | goal L2 threshold | success | success delta | max-reward delta | action gate rate | goal gate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen n500 | - | - | 0.350 | - | - | 0.000 | 0.000 |
| task-reward debug ungated | - | - | 0.300 | -0.050 | -0.0386 | 0.000 | - |
| multifeature gate | 0.0006 | 24 | 0.260 | -0.090 | -0.0586 | 0.739 | 0.233 |
| multifeature gate | 0.0006 | 27 | 0.270 | -0.080 | -0.0496 | 0.729 | 0.323 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_multigate_smoke_8_seed4621000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_multigate_a0006_g24_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_multigate_a0006_g27_100_seed4600000.json`

Interpretation:

The online multifeature gate is worse than ungated on the selection window, so I
did not spend a 500-episode validation run on it. This repeats the lesson from
the earlier selector work: per-episode or trajectory-level correlations are not
automatically useful when converted into a deployed step-level fallback policy,
because the fallback changes future states, goals, and intervention
opportunities. The next meaningful work should stop adding hand-coded gates and
instead change the training target or train a policy/selector directly in the
closed-loop distribution.

## 2026-06-26 - Nearest-training-goal projection for serial eval

I added an eval-only learned-goal projection option to `low-level-rl
eval-serial`:

```text
--goal-projection nearest_train
```

This option is only valid with `--goal-source learned`. At each high-level
replan, it takes the predicted normalized goal and replaces it with the nearest
normalized training-set goal from the learned-interface `encoded_episodes.pt`
bank. For `effect32_film` seed 0, this yields 62,472 prototype goals after
dropping the invalid effect-code prefix.

Smoke command:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_proto_serial_smoke5_seed3500000 \
  --episodes 5 \
  --seed-start 3500000 \
  --goal-source learned \
  --goal-projection nearest_train \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

The smoke completed and reported:

| episodes | success | prototypes | mean projection L2 |
| ---: | ---: | ---: | ---: |
| 5 | 0.400 | 62,472 | 1.918 |

I then ran two 100-seed banks for the frozen policy:

| seed start | learned baseline | nearest-train projection | oracle ceiling | projection vs learned |
| ---: | ---: | ---: | ---: | ---: |
| 3500000 | 0.600 | 0.640 | 0.680 | +0.040 |
| 3600000 | 0.690 | 0.680 | 0.760 | -0.010 |
| aggregate | 0.645 | 0.660 | 0.720 | +0.015 |

Paired frozen projection outcomes against learned baseline:

| seed start | improvements | regressions | net |
| ---: | ---: | ---: | ---: |
| 3500000 | 12 | 8 | +4 |
| 3600000 | 11 | 12 | -1 |
| aggregate | 23 | 20 | +3 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_proto_serial_smoke5_seed3500000/serial_eval_5_seed3500000_nearest_train.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_proto_serial100_seed3500000/serial_eval_100_seed3500000_nearest_train.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_proto_serial100_seed3600000/serial_eval_100_seed3600000_nearest_train.json`

I also checked compatibility with the existing R3 checkpoint by evaluating the
same `hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10` run with projected
goals:

| seed start | R3 learned baseline | R3 + projection | projection vs learned |
| ---: | ---: | ---: | ---: |
| 3500000 | 0.670 | 0.620 | -0.050 |
| 3600000 | 0.710 | 0.720 | +0.010 |
| aggregate | 0.690 | 0.670 | -0.020 |

Paired R3 projection outcomes against learned baseline:

| seed start | improvements | regressions | net |
| ---: | ---: | ---: | ---: |
| 3500000 | 14 | 19 | -5 |
| 3600000 | 14 | 13 | +1 |
| aggregate | 28 | 32 | -4 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/serial_eval_100_seed3500000_nearest_train.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/serial_eval_100_seed3600000_nearest_train.json`

Interpretation:

Nearest-training-goal projection is a useful diagnostic for the high-level
interface, but it is not a strong fix. It gives the frozen policy only a tiny
two-bank gain (`0.645 -> 0.660`) and remains well below the oracle frozen
ceiling (`0.720`). It also does not combine reliably with the current R3
low-level update (`0.690 -> 0.670`). This suggests the remaining learned-goal
problem is not just an off-manifold high-level output that can be repaired with
nearest-neighbor snapping; future work should either improve the high-level
model/objective directly or train/evaluate low-level adaptation in the actual
closed-loop learned-goal distribution.

## 2026-06-26 - Closed-loop outcome selector audit

I added `rl-rerun fit-closed-loop-selector`, an offline audit for selector
learnability from matched closed-loop frozen/residual JSONs. It fits a
ridge-regularized linear score on discordant episodes and evaluates the implied
per-episode frozen/residual choice on train and validation banks.

The default feature set is intentionally limited to initial features that an
episode-start selector could know:

```text
episode_action_delta_l2_initial
episode_policy_saturation_initial
episode_goal_l2_initial
```

To support that, I also added initial action-delta and initial saturation fields
to `eval-closed-loop-r{1,2,3}` result JSONs.

Fresh paired evals with the new fields:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4600000 \
  --num-envs 64 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4600000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4700000 \
  --num-envs 64 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4700000.json
```

Initial-feature selector fit:

```bash
uv run hcl-poc rl-rerun fit-closed-loop-selector \
  --train-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4600000.json \
  --validation-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4700000.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_selector_fit_train460_valid470.json \
  --force
```

Result:

| split | frozen | residual | selector | uses residual | discordant | AUC | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train 4600000 | 0.260 | 0.290 | 0.300 | 0.700 | 23 | 0.631 | 0.609 |
| validation 4700000 | 0.290 | 0.290 | 0.310 | 0.700 | 22 | 0.537 | 0.591 |

Validation details: 9 false residual regressions and 0 missed residual
improvements. This is only a weak positive offline result and does not justify a
larger direct deployment run by itself.

I also fit a deliberately non-deployable full-episode-summary selector on the
same fresh banks:

```bash
uv run hcl-poc rl-rerun fit-closed-loop-selector \
  --train-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4600000.json \
  --validation-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4700000.json \
  --feature-names episode_action_delta_l2_mean episode_action_delta_l2_max episode_policy_saturation_rate episode_goal_l2_initial episode_goal_l2_mean episode_high_level_decisions \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_summary_selector_fit_train460_valid470.json \
  --force
```

Validation result:

| selector | frozen | residual | selector | uses residual | discordant AUC | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full episode summary | 0.290 | 0.290 | 0.390 | 0.260 | 0.909 | 0.955 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initialfeatures_100_seed4700000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_selector_fit_train460_valid470.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_summary_selector_fit_train460_valid470.json`

Interpretation:

The contrast is the result. Initial deployable features barely predict which
branch wins, while full trajectory summary features predict discordant outcomes
well. So a one-shot episode-start selector is probably too weak for this
checkpoint. If selector work continues, it should use online step-level or
recurrent context, or train the selector/policy directly inside the closed-loop
distribution. Otherwise, the more promising path remains changing the training
target so the tuned policy creates a larger and more consistently useful effect.

## 2026-06-26 - Online step deployment of closed-loop selector

I added `--step-selector` to `rl-rerun eval-closed-loop-r{1,2,3}`. It consumes a
JSON produced by `fit-closed-loop-selector`, but only accepts features available
at the current step:

```text
episode_action_delta_l2_initial -> current action_delta_l2
episode_policy_saturation_initial -> current policy_saturation
episode_goal_l2_initial -> current goal_l2
```

Selectors containing non-online features fail clearly. At each action, the eval
path computes the selector score and falls back to the frozen base action when
the score is below the fitted threshold.

Smoke command:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4720000 \
  --num-envs 8 \
  --goal-source learned \
  --step-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_selector_fit_train460_valid470.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_step_selector_smoke_8_seed4720000.json
```

The smoke wrote the expected selector metadata and selected residual actions
about `0.766` of active steps.

I then evaluated the same selector on the train and validation 100-episode
banks:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4600000 \
  --num-envs 64 \
  --goal-source learned \
  --step-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_selector_fit_train460_valid470.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_step_selector_100_seed4600000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4700000 \
  --num-envs 64 \
  --goal-source learned \
  --step-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_selector_fit_train460_valid470.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_step_selector_100_seed4700000.json
```

Results:

| seed | frozen | ungated residual | online step selector | residual action rate | selector max-reward delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4600000 | 0.260 | 0.290 | 0.290 | 0.807 | +0.0264 |
| 4700000 | 0.290 | 0.290 | 0.270 | 0.804 | -0.0158 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_step_selector_smoke_8_seed4720000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_step_selector_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_step_selector_100_seed4700000.json`

Interpretation:

The online step deployment does not rescue the learned selector. It is neutral
in-sample and worse on the validation bank. Combined with the offline selector
audit, this suggests the remaining selector opportunity is not a linear
threshold over these simple state/action features. A useful selector likely
needs to be trained directly in closed loop, with temporal context and its own
intervention consequences, or the next work should return to changing the
training target so residual actions are useful more often.

## 2026-06-26 - Oracle segment selector upper-bound check

I added `--oracle-segment-selector` to `rl-rerun eval-closed-loop-r{1,2,3}`. This
is a simulator-privileged diagnostic:

1. At each high-level replan, copy the exact current simulator state.
2. Roll the frozen low-level for one held-goal segment.
3. Roll the tuned low-level for the same segment from the same state.
4. Compare final latent L2 to the current held goal.
5. Execute the tuned branch for the segment only if its counterfactual final
   latent is closer than the frozen branch.

This is not real-compatible because it uses exact simulator state copying and
counterfactual branch rollouts, but it tests whether a near-perfect local
same-state selector can turn the existing residual checkpoint into a task
improvement.

Smoke command:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 4 \
  --eval-seed-start 4730000 \
  --num-envs 4 \
  --goal-source learned \
  --oracle-segment-selector \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_smoke_4_seed4730000.json
```

The smoke completed and selected tuned segments about `0.45` of the time.

Because vector batching can change exact Push-T outcomes, I ran matched
20-episode no-selector and oracle-selector checks with the same `num-envs=10`:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 4600000 \
  --num-envs 10 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched20_seed4600000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 4600000 \
  --num-envs 10 \
  --goal-source learned \
  --oracle-segment-selector \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_20_seed4600000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 4700000 \
  --num-envs 10 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched20_seed4700000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 4700000 \
  --num-envs 10 \
  --goal-source learned \
  --oracle-segment-selector \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_20_seed4700000.json
```

Results:

| seed | frozen | ungated residual | oracle segment selector | selector residual action rate | selector decision residual rate | selector latent-distance delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4600000 | 0.350 | 0.400 | 0.350 | 0.572 | 0.575 | 0.051 |
| 4700000 | 0.100 | 0.150 | 0.200 | 0.537 | 0.537 | 0.285 |
| aggregate | 0.225 | 0.275 | 0.275 | - | - | - |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_smoke_4_seed4730000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched20_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_20_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched20_seed4700000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_20_seed4700000.json`

Interpretation:

The oracle selector confirms that same-state counterfactual branch selection can
identify local latent-distance improvements, but those choices do not translate
to a robust task gain. It improves the weaker `4700000` slice but removes the
ungated residual gain on `4600000`; aggregate success ties ungated residual.
This is strong evidence against spending more time on selectors that optimize
one-segment latent goal distance for this checkpoint. The next useful move is
to change the objective/target so the tuned branch produces larger and more
task-aligned improvements, or to train selection directly against full
closed-loop outcomes rather than local latent endpoint distance.

## 2026-06-26 - Terminal task-paired local-R3 reward check

After the oracle segment selector showed that one-segment latent-distance branch
selection was not enough, I added a new `rl-rerun train-local-r3 --reward-mode
task_paired` option. It reuses the cached frozen same-state rollout used by
latent paired reward, but the terminal reward is the tuned segment's final
ManiSkill dense reward minus the frozen segment's final ManiSkill dense reward:

```text
r_terminal = tuned_terminal_env_reward - frozen_terminal_env_reward
```

This keeps the prior `progress` and `paired` recipes unchanged and records
task-paired metrics separately in history.

Training smoke:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 1 \
  --terminal-weight 1 \
  --dense-progress-weight 0 \
  --task-reward-weight 0 \
  --reward-mode task_paired \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --force
```

Training metrics after one update:

| metric | value |
| --- | ---: |
| mean task-paired improvement | 0.00150 |
| fraction task-paired improved | 0.398 |
| terminal env reward | 0.4803 |
| frozen terminal env reward | 0.4788 |
| terminal latent distance | 0.6022 |
| action delta L2 | 0.01085 |
| task success diagnostic rate | 0.213 |

Matched local validation on
`results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json`:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
```

| policy | initial distance | final distance | reduction | action delta L2 | task success-once |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 previous baseline | 1.0671 | 0.6020 | 0.4651 | - | - |
| task-paired terminal | 1.0671 | 0.6036 | 0.4635 | 0.00042 | 0.331 |

Because the local validation did not beat frozen, I only ran a small
deployability smoke:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4740000 \
  --num-envs 4 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/closed_loop_smoke_8_seed4740000.json
```

| policy | success | final reward | max reward | residual norm |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.625 | 0.6862 | 0.6997 | 0.00000 |
| task-paired residual | 0.375 | 0.5536 | 0.5607 | 0.00039 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/closed_loop_smoke_8_seed4740000.json`

Interpretation:

The terminal task-paired objective is implemented and runnable, but this first
diagnostic is not a promotion candidate. The training task-reward delta is
barely positive, fewer than half of segments improve the terminal dense reward,
matched local latent distance remains slightly worse than frozen, and the
closed-loop smoke regresses. This adds useful negative evidence: using the
one-segment terminal dense reward as the paired target is still too weak/noisy
under the current local-R3 update. The next objective work should change the
target regime more substantially, or train directly against closed-loop outcome
rather than another one-segment local proxy.

## 2026-06-26 - Online prefix-summary selector check

### Hypothesis

The offline full-episode summary selector was a strong non-deployable upper
bound (`0.390` selector success versus `0.290` frozen/residual on validation),
while the instantaneous online step selector was weak. A middle ground is to
deploy the same summary-feature selector using cumulative prefix features that
are available online:

- action-delta mean so far;
- action-delta max so far;
- policy saturation rate so far;
- goal-L2 initial/current;
- goal-L2 mean so far;
- high-level decisions so far.

This tests whether online context, not only instantaneous features, is enough
to recover part of the full-summary selector signal.

### Implementation

I extended `eval-closed-loop-r{1,2,3} --step-selector` support so selector
features can include:

```text
episode_action_delta_l2_mean
episode_action_delta_l2_max
episode_policy_saturation_rate
episode_goal_l2_mean
episode_high_level_decisions
```

The previous initial-feature selector remains supported. The full-summary
selector file can now be loaded and evaluated online using prefix values.

Smoke:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 8 \
  --eval-seed-start 4750000 \
  --num-envs 4 \
  --goal-source learned \
  --step-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_summary_selector_fit_train460_valid470.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_prefix_summary_selector_smoke_8_seed4750000.json
```

The smoke completed and used residual actions `0.782` of the time.

### Matched 100-episode comparison

Because the older selector runs used `num_envs=64` and Push-T vectorization
changes exact outcomes, I reran matched `num_envs=20` baselines:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4600000 \
  --num-envs 20 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched_numenv20_100_seed4600000.json
```

The same command was repeated for `4700000`, and for both selector files:

- initial selector:
  `closed_loop_initial_selector_fit_train460_valid470.json`
- prefix-summary selector:
  `closed_loop_summary_selector_fit_train460_valid470.json`

Results:

| seed | policy | frozen | policy success | success delta | max-reward delta | selector residual rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4600000 | ungated residual | 0.310 | 0.320 | +0.010 | -0.0009 | 1.000 |
| 4600000 | initial-step selector | 0.310 | 0.400 | +0.090 | +0.0612 | 0.824 |
| 4600000 | prefix-summary selector | 0.310 | 0.320 | +0.010 | -0.0021 | 0.813 |
| 4700000 | ungated residual | 0.340 | 0.340 | +0.000 | -0.0009 | 1.000 |
| 4700000 | initial-step selector | 0.340 | 0.320 | -0.020 | -0.0145 | 0.838 |
| 4700000 | prefix-summary selector | 0.340 | 0.350 | +0.010 | +0.0061 | 0.819 |

Aggregates:

| policy | frozen mean | policy mean | success delta | max-reward delta | residual action rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ungated residual | 0.325 | 0.330 | +0.005 | -0.0009 | 1.000 |
| initial-step selector | 0.325 | 0.360 | +0.035 | +0.0233 | 0.831 |
| prefix-summary selector | 0.325 | 0.335 | +0.010 | +0.0020 | 0.816 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_prefix_summary_selector_smoke_8_seed4750000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched_numenv20_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_matched_numenv20_100_seed4700000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_step_selector_numenv20_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_initial_step_selector_numenv20_100_seed4700000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_prefix_summary_selector_100_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_prefix_summary_selector_100_seed4700000.json`

### Interpretation

The cumulative prefix-summary selector does not recover the strong offline
full-summary upper bound. It is only slightly above frozen and ungated residual
on this matched `num_envs=20` check.

The initial-step selector looks better under `num_envs=20`, but that conflicts
with the earlier `num_envs=64` validation where it was neutral in-sample and
worse on the validation bank. This is useful as a vectorization-sensitivity
warning, not as a robust selector result. A selector that is genuinely useful
needs to be trained and evaluated in the closed-loop intervention distribution,
not fit once from episode summaries and replayed as a simple linear online
fallback.

## 2026-06-26 - Multi-window serial segment selector check

### Hypothesis

The first serial segment selector used only one 50-episode training window
(`500` segments). It generalized offline to another exact segment window for
local raw reduction, but failed when deployed online. A larger exact segment
dataset might reduce overfitting and produce a selector whose local gain
survives direct rollout.

### Implementation

I extended `low-level-rl fit-serial-segment-selector` with repeatable paired
training inputs:

```text
--extra-base-json
--extra-candidate-json
```

The fitter now concatenates exactly aligned segment pairs across all train
windows, while keeping validation unchanged. It still refuses mismatched extra
base/candidate counts.

### Fresh validation window

I generated a new exact serial segment validation window:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4508000 \
  --episodes 50 \
  --seed-start 4508000 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4508000 \
  --episodes 50 \
  --seed-start 4508000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Multi-window fit

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  fit-serial-segment-selector \
  --base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json \
  --extra-base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json \
  --extra-candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4508000/serial_eval_50_seed4508000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4508000/serial_eval_50_seed4508000.json \
  --output results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/segment_selector_fit_train4503000_4506000_valid4508000.json \
  --ridge 1.0 \
  --force
```

Offline local-reduction metric:

| split | segments | base raw reduction | R3 raw reduction | selector raw reduction | selector delta | use R3 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train 4503000+4506000 | 1000 | 0.4328 | 0.4390 | 0.4966 | +0.0637 | 0.690 | 0.616 |
| validation 4508000 | 500 | 0.4150 | 0.4276 | 0.4899 | +0.0750 | 0.730 | 0.598 |

### Direct online validation

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_multiwin_serial50_seed4508000 \
  --episodes 50 \
  --seed-start 4508000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --segment-selector-weights -0.0038581500 -0.3627895415 0.1276507229 0.1046459749 -0.1108499840 \
  --segment-selector-mean 0.7210667729 0.9688425064 0.4701716006 1.0044081211 45.0000000000 \
  --segment-selector-std 0.2319196016 0.8249247074 0.4231470823 0.6122351289 28.7228145599 \
  --segment-selector-threshold -0.0652473196 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Direct exact-seed result:

| policy | success | max reward | segment raw-reduction delta | helpful segments | harmful segments | segment use R3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.680 | 0.778 | - | - | - | - |
| ungated R3 bc10 | 0.740 | 0.816 | +0.0126 | 272 | 228 | 1.000 |
| multi-window segment selector | 0.660 | 0.765 | +0.0026 | 252 | 210 | 0.736 |

Paired episode counts against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| ungated R3 bc10 | 7 | 4 | +3 |
| multi-window segment selector | 2 | 3 | -1 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4508000/serial_eval_50_seed4508000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4508000/serial_eval_50_seed4508000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/segment_selector_fit_train4503000_4506000_valid4508000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_multiwin_serial50_seed4508000/serial_eval_50_seed4508000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_multiwin_serial50_seed4508000/paired_vs_frozen_serial50_seed4508000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segselector_multiwin_serial50_seed4508000/segments_vs_frozen_serial50_seed4508000.json`

### Interpretation

The larger segment-selector training set makes the offline local metric look
stronger, but it still fails direct closed-loop deployment. On the fresh exact
window, ungated R3 beats frozen (`0.740` vs `0.680`), while the multi-window
selector falls below frozen (`0.660`). The selector also gives back most of the
ungated segment-level raw-reduction gain.

This reinforces the previous offline-to-online mismatch diagnosis: selecting
between already completed frozen/R3 segment outcomes is not enough, because the
deployed selector changes subsequent states, high-level goals, and segment
distributions. A larger linear segment-start selector is not the missing piece.
Further selector work should be trained in the actual intervention distribution,
or the candidate policy needs a larger and more consistently task-aligned effect.

## 2026-06-26: Short-horizon learned-interface check

After exhausting scalar gates and offline selectors, I tested the Experiment D
horizon hypothesis directly on the current effect32 FiLM hierarchy. I added
candidate-level `horizon_steps` and `update_period` overrides for learned
interfaces, then created two aliases that reuse the trained effect32
representation and high-level model while retraining only the low policy:

```text
effect32_film_h5: representation_candidate=effect32, high_level_candidate=effect32, horizon/update=5
effect32_film_h2: representation_candidate=effect32, high_level_candidate=effect32, horizon/update=2
```

Commands:

```bash
uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_h5 \
  --seed 0

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_h5 \
  --goal-source learned \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --seed 0

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_h5 \
  --goal-source oracle \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --seed 0

uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_h2 \
  --seed 0

uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_h2 \
  --goal-source learned \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --seed 0
```

Hierarchy validation:

| candidate | horizon | update | best epoch | oracle action MAE | predicted action MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film_h5 | 5 | 5 | 55 | 0.0387 | 0.0415 |
| effect32_film_h2 | 2 | 2 | 59 | 0.0375 | 0.0531 |

Matched closed-loop evaluations on `seed_start=3500000`, 200 episodes:

| candidate | goal source | success | final reward | max reward | teacher MAE | decisions/episode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film k10 | learned | 0.645 | 0.735 | 0.742 | 0.107 | 7.045 |
| effect32_film k10 | oracle | 0.645 | 0.740 | 0.746 | 0.089 | 6.960 |
| effect32_film_h5 | learned | 0.520 | 0.649 | 0.658 | 0.103 | 14.785 |
| effect32_film_h5 | oracle | 0.570 | 0.685 | 0.696 | 0.092 | 14.125 |
| effect32_film_h2 | learned | 0.315 | 0.481 | 0.501 | 0.133 | 42.180 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_h5/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_h5/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_h5/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_h5/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `artifacts/incremental/learned_interface/effect32_film_h2/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_h2/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_h2/seed0/learned_hierarchy_eval_200_seed3500000.json`

### Interpretation

Shortening the hierarchy horizon did not help. k=5 is worse than k=10 even with
oracle goals, so the regression is not only learned high-level goal error. k=2
is much worse despite frequent replanning and has worse predicted-action
validation. For the current effect32 FiLM setup, the next useful lever is not a
plain shorter horizon; it should be a changed objective or a policy/selector
trained in the closed-loop intervention distribution.

## 2026-06-26: Hard-start task-paired local R3

### Hypothesis

The uniform terminal task-paired R3 smoke produced a tiny positive terminal
reward delta but did not improve matched local validation. This run tests a
more targeted objective regime: train only on local starts where the frozen
same-state branch ends far from the held goal.

### Implementation

Added `--min-base-terminal-distance` to `rl-rerun train-local-r3`.

For `reward-mode paired` and `reward-mode task_paired`, the trainer already
computes the cached frozen terminal branch. When the threshold is provided, it
marks only environments with:

```text
base_terminal_distance >= min_base_terminal_distance
```

as active. Inactive samples stay in the vectorized rollout for simulator
synchronization, but their rewards are zeroed and they are excluded from PPO
minibatches and reported rollout metrics. Each history row records
`active_fraction`.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 1 \
  --terminal-weight 1 \
  --dense-progress-weight 0 \
  --task-reward-weight 0 \
  --reward-mode task_paired \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --min-base-terminal-distance 0.6 \
  --force
```

Matched local validation:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
```

### Results

Training metrics after one update:

| metric | value |
| --- | ---: |
| active fraction | 0.3889 |
| mean task-paired improvement | 0.00469 |
| fraction task-paired improved | 0.451 |
| terminal env reward | 0.4122 |
| frozen terminal env reward | 0.4075 |
| terminal latent distance | 0.8193 |
| action delta L2 | 0.01087 |
| task success diagnostic rate | 0.1696 |

Matched local validation on the same 4096-row validation manifest:

| policy | initial distance | final distance | reduction | action delta L2 | task success-once |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 previous baseline | 1.0671 | 0.6020 | 0.4651 | - | - |
| uniform task-paired | 1.0671 | 0.6036 | 0.4635 | 0.00042 | 0.3306 |
| hard-start task-paired | 1.0671 | 0.6093 | 0.4578 | 0.00054 | 0.3291 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json`

### Interpretation

Hard-start masking improved the in-training terminal reward delta compared with
the uniform task-paired smoke, but it did not transfer to matched local
validation. The hard-start checkpoint is worse than frozen and worse than the
uniform task-paired checkpoint on latent reduction, with only a tiny action
change. It does not justify closed-loop deployment.

This closes the simple "same objective but only hard local starts" variant for
effect32 local R3. The next objective-side experiment should use a more
deployment-aligned target than one-segment terminal dense reward, or train the
intervention policy/selector directly from closed-loop outcomes.

## 2026-06-26: Task-reward-hard task-paired local R3

### Hypothesis

The latent hard-start filter improved the task-paired training signal but
worsened matched local validation. This run targets the task-paired reward more
directly by selecting local starts where the frozen same-state terminal
ManiSkill dense reward is low.

### Implementation

Added `--max-base-terminal-env-reward` to `rl-rerun train-local-r3`. It is only
valid with `--reward-mode task_paired`. When set, active samples satisfy:

```text
base_terminal_env_reward <= max_base_terminal_env_reward
```

The active mask composes with `--min-base-terminal-distance` if both filters are
provided, but this diagnostic used only the task-reward filter.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 1 \
  --terminal-weight 1 \
  --dense-progress-weight 0 \
  --task-reward-weight 0 \
  --reward-mode task_paired \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --max-base-terminal-env-reward 0.45 \
  --force
```

Matched local validation:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
```

### Results

Training metrics after one update:

| metric | value |
| --- | ---: |
| active fraction | 0.7629 |
| mean task-paired improvement | 0.03577 |
| fraction task-paired improved | 0.5219 |
| terminal env reward | 0.3527 |
| frozen terminal env reward | 0.3169 |
| terminal latent distance | 0.6327 |
| action delta L2 | 0.01088 |
| task success diagnostic rate | 0.0643 |

Matched local validation on the same 4096-row validation manifest:

| policy | initial distance | final distance | reduction | action delta L2 | task success-once |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 previous baseline | 1.0671 | 0.6020 | 0.4651 | - | - |
| uniform task-paired | 1.0671 | 0.6036 | 0.4635 | 0.00042 | 0.3306 |
| latent-hard task-paired | 1.0671 | 0.6093 | 0.4578 | 0.00054 | 0.3291 |
| task-hard task-paired | 1.0671 | 0.6092 | 0.4578 | 0.00046 | 0.3335 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json`

### Interpretation

The task-reward hard filter substantially improves the in-training task-paired
signal and almost reaches the plan's `fraction_improved > 0.55` local bar, but
the effect does not transfer to matched local validation. The policy's action
changes are still tiny, latent reduction is worse than frozen, and local task
success-once barely changes. I skipped closed-loop deployment.

This is stronger evidence that the one-segment terminal dense-reward target can
be optimized on the sampled training starts without producing a useful held-out
local controller. The next objective should stop being a filtered version of the
same local terminal reward and move toward closed-loop outcome supervision or a
larger policy change with an explicit preservation mechanism.

## 2026-06-26: Task-hard task-paired local R3 with lower BC

### Hypothesis

The task-reward hard filter produced the strongest training signal so far, but
the policy update was still tiny. This run tests whether the BC anchor is now
the limiting factor by rerunning the same task-hard one-update diagnostic with
`bc_weight=0.3` instead of `1.0`.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 0.3 \
  --terminal-weight 1 \
  --dense-progress-weight 0 \
  --task-reward-weight 0 \
  --reward-mode task_paired \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --max-base-terminal-env-reward 0.45 \
  --force
```

Matched local validation:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
```

### Results

The rollout-side training metrics are identical to the `bc=1` task-hard run
because this one-update diagnostic records rollout metrics before the PPO update:

| metric | value |
| --- | ---: |
| active fraction | 0.7629 |
| mean task-paired improvement | 0.03577 |
| fraction task-paired improved | 0.5219 |
| terminal env reward | 0.3527 |
| frozen terminal env reward | 0.3169 |
| terminal latent distance | 0.6327 |
| action delta L2 | 0.01088 |

Matched local validation:

| policy | initial distance | final distance | reduction | action delta L2 | task success-once |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen n500 previous baseline | 1.0671 | 0.6020 | 0.4651 | - | - |
| uniform task-paired bc1 | 1.0671 | 0.6036 | 0.4635 | 0.00042 | 0.3306 |
| task-hard bc1 | 1.0671 | 0.6092 | 0.4578 | 0.00046 | 0.3335 |
| task-hard bc0.3 | 1.0671 | 0.6037 | 0.4634 | 0.00046 | 0.3367 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json`

### Interpretation

Lower BC helps relative to task-hard `bc=1`: it recovers most of the latent
reduction regression and slightly improves the local task-success diagnostic.
But it remains worse than frozen on latent reduction and does not create a
meaningful action change. I skipped closed-loop deployment.

This weakens the "BC anchor is the main blocker" explanation for the task-hard
one-segment objective. The target can be optimized on sampled starts, and lower
BC can alter the update a little, but the held-out local behavior is still not a
promotion candidate.

## 2026-06-26: Targeted subset validation for task-hard local R3

### Hypothesis

The aggregate validation for task-hard local R3 is weak, but the training target
is explicitly conditional: starts where the frozen same-state segment has low
terminal task reward. This check asks whether the update at least helps that
same target subset on held-out validation.

### Implementation

Extended `eval-local-r3` so it first runs the frozen low-level branch from the
same held-out local start, then resets/replays the same start and runs the tuned
checkpoint. The output now records base-vs-tuned deltas and subset summaries:

```text
base_final_env_reward_le_0p45
base_final_distance_ge_0p6
```

The evaluator uses the same vector environment sequentially for base and tuned
rollouts, avoiding the GPU camera allocation issue from opening two 4096-env RGB
environments at once.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json
```

### Results

Full held-out local bank, tuned minus frozen:

| policy | latent reduction delta | final reward delta | success-once delta |
| --- | ---: | ---: | ---: |
| uniform task-paired | -0.0016 | -0.0024 | -0.0024 |
| task-hard bc1 | -0.0072 | -0.0024 | +0.0005 |
| task-hard bc0.3 | -0.0017 | +0.0009 | +0.0037 |

Task-hard held-out subset (`base_final_env_reward <= 0.45`, 3150 / 4096
samples), tuned minus frozen:

| policy | latent reduction delta | final reward delta | success-once delta |
| --- | ---: | ---: | ---: |
| uniform task-paired | -0.0018 | +0.0273 | +0.0149 |
| task-hard bc1 | -0.0061 | +0.0304 | +0.0200 |
| task-hard bc0.3 | -0.0032 | +0.0314 | +0.0225 |

Latent-hard held-out subset (`base_final_distance >= 0.6`, 1606 / 4096
samples), tuned minus frozen:

| policy | latent reduction delta | final reward delta | success-once delta |
| --- | ---: | ---: | ---: |
| uniform task-paired | +0.0571 | +0.0018 | -0.0019 |
| task-hard bc1 | +0.0529 | -0.0045 | -0.0006 |
| task-hard bc0.3 | +0.0528 | -0.0054 | -0.0031 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json`

### Interpretation

The task-hard target is not pure noise: it improves terminal task reward on the
held-out task-hard subset, and weaker BC slightly improves that subset result.
However, the effect is small and comes with tradeoffs. The aggregate bank still
does not beat frozen on latent reduction, and the latent-hard subset loses task
reward under the task-hard-trained checkpoints.

This suggests local terminal dense reward can shape a tiny task-specific
correction, but the current final-layer R3 update is too small and too
single-objective. A promotion candidate likely needs either a larger policy
change with explicit preservation, or direct closed-loop outcome training rather
than another filtered one-segment reward.

## 2026-06-26: Three-update task-hard task-paired local R3

### Hypothesis

The task-hard `bc=0.3` one-update checkpoint gives the best held-out task-hard
subset reward gain so far, but the action change is tiny. This run tests whether
the same target scales with three PPO updates.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5 \
  --steps 122880 \
  --bc-weight 0.3 \
  --terminal-weight 1 \
  --dense-progress-weight 0 \
  --task-reward-weight 0 \
  --reward-mode task_paired \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --max-base-terminal-env-reward 0.45 \
  --force
```

Enriched matched local validation:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json
```

### Training Metrics

| global step | active fraction | task-paired improvement | fraction task-improved | action delta L2 | task success diagnostic |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40960 | 0.7629 | 0.03577 | 0.5219 | 0.01088 | 0.0643 |
| 81920 | 0.7629 | 0.03815 | 0.5277 | 0.01086 | 0.0636 |
| 122880 | 0.7632 | 0.03671 | 0.5186 | 0.01088 | 0.0679 |

### Held-Out Local Validation

Tuned minus frozen:

| policy | all reward delta | all success delta | task-hard reward delta | task-hard success delta | latent-hard reduction delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| task-hard bc0.3 1 update | +0.0009 | +0.0037 | +0.0314 | +0.0225 | +0.0528 |
| task-hard bc0.3 3 updates | -0.0034 | -0.0037 | +0.0293 | +0.0159 | +0.0625 |

Action delta on the full validation bank increased only from `0.00046` to
`0.00067`.

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json`

### Interpretation

Three updates do not turn the task-hard local objective into a promotion
candidate. The training task-paired metric does not scale meaningfully, action
changes remain tiny, and held-out validation shifts the tradeoff: latent-hard
reduction improves, but the target task-hard reward/success gains shrink and
the aggregate reward/success deltas become negative.

This makes the next useful direction clearer: stop extending the same
one-segment terminal reward target. Either make a larger policy change with an
explicit preservation objective, or train against closed-loop outcome labels.

## 2026-06-26: Task-hard one-update closed-loop transfer smoke

### Hypothesis

The task-hard `bc=0.3` one-update checkpoint has the best held-out local
task-hard subset gain so far. This checks whether that small local gain transfers
at all under learned-goal closed-loop execution.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4800000 \
  --num-envs 20 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_100_seed4800000.json
```

### Result

| branch | success | final reward | max reward | mean residual norm |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.230 | 0.4147 | 0.4376 | 0.000000 |
| residual | 0.250 | 0.4240 | 0.4549 | 0.000484 |
| delta | +0.020 | +0.0093 | +0.0172 | - |

High-level decisions per episode were effectively matched (`8.83` frozen,
`8.85` residual).

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_100_seed4800000.json`

### Interpretation

This is a mildly positive transfer smoke for the best task-hard local checkpoint:
same-run residual improves success by `+0.020`, final reward by `+0.0093`, and
max reward by `+0.0172`. The result is too small and too low-success to promote
on its own, but it contradicts the earlier blanket conclusion that task-hard R3
should skip all closed-loop deployment. The next check should be a larger
matched closed-loop window for this exact checkpoint before changing the
training objective again.

## 2026-06-26: Task-hard one-update larger closed-loop check

### Hypothesis

The 100-episode smoke was mildly positive but too noisy. This reruns the same
checkpoint and seed start over 500 learned-goal closed-loop episodes.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 20 \
  --goal-source learned \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json
```

### Result

| branch | success | final reward | max reward | mean residual norm |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.312 | 0.4727 | 0.4960 | 0.000000 |
| residual | 0.304 | 0.4635 | 0.4931 | 0.000482 |
| delta | -0.008 | -0.0093 | -0.0029 | - |

High-level decisions per episode were close (`8.516` frozen, `8.576`
residual).

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json`

### Interpretation

The larger matched check rejects the 100-episode smoke as a promotion signal.
The residual branch is slightly worse than frozen on success and reward over 500
episodes. Task-hard local R3 can improve the targeted held-out local subset, but
that still does not transfer into a reliable closed-loop policy improvement.

## 2026-06-26: Prefix-feature counterfactual branch selector

### Hypothesis

The previous privileged branch-goal selector used only static query/candidate
features and source-state outcome priors. It may fail because it cannot see what
actually happens when a candidate is rolled for the first held-goal segment from
the query state. I extended the counterfactual bank to save first-segment
end-state and prefix outcome features, then trained the same return-regression
selector on those richer features.

### Code Changes

- `scripts/collect_privileged_z_branch_counterfactuals.py` now saves
  `base_segment_*`, `candidate_segment_*`, and candidate prefix deltas.
- `scripts/train_privileged_z_counterfactual_selector.py` automatically appends
  these fields as optional features when present, while remaining compatible
  with older banks.

### Commands

```bash
uv run scripts/collect_privileged_z_branch_counterfactuals.py \
  --config configs/pusht_incremental.yaml \
  --checkpoint artifacts/incremental/privileged_z/clean_disturbed_multioffset/n1800/seed0/privileged_z_k10.pt \
  --residual-checkpoint artifacts/incremental/privileged_z_direct_distill/hcl_next_oracle_low_level_oraclegoal_branch_return_ge5_seed9940000_imp01_preserve_npz1_final_layer_lr1e4_e200/seed0/latest.pt \
  --branch-bank data/manifests/privileged_z_branch_outcome_oracle_low_level_oraclegoal_all_seed9950000_200eps_b10.npz \
  --output data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9963000_q128_k8_prefix.npz \
  --seed-start 9963000 \
  --num-envs 64 \
  --query-batches 2 \
  --candidates-per-query 8 \
  --max-rollout-steps 120
```

Selector training was run for seeds `0..4` with:

```bash
uv run scripts/train_privileged_z_counterfactual_selector.py \
  --input data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9963000_q128_k8_prefix.npz \
  --output artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed0.pt \
  --seed 0 \
  --epochs 200 \
  --batch-size 1024 \
  --hidden-dim 128 \
  --depth 2 \
  --learning-rate 1e-3
```

### Counterfactual Bank

| bank | queries | candidates/query | base success | base return | nearest return delta | oracle return delta | positive best > 5 | oracle success delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed9963000_q128_k8_prefix` | 128 | 8 | 0.500 | 43.79 | +1.98 | +15.87 | 0.516 | +0.313 |

### Selector Results

Validation metrics across five random query splits:

| seed | selected return delta | selected success delta | nearest return delta | nearest success delta | oracle return delta | oracle success delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -1.529 | -0.031 | -1.266 | +0.094 | +13.002 | +0.250 |
| 1 | +1.518 | +0.031 | +1.551 | +0.000 | +18.778 | +0.281 |
| 2 | +1.538 | +0.031 | +3.524 | +0.000 | +18.848 | +0.250 |
| 3 | -0.385 | +0.000 | +8.922 | +0.156 | +16.651 | +0.281 |
| 4 | -3.727 | -0.094 | -2.383 | -0.094 | +10.373 | +0.125 |
| mean | -0.517 | -0.013 | +2.070 | +0.031 | +15.530 | +0.237 |

Artifacts:

- `data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9963000_q128_k8_prefix.npz`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed0.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed1.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed2.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed3.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed4.pt`

### Interpretation

The candidate set still contains large query-specific upside, but adding
first-segment end-state and prefix outcome features did not make the small
offline selector useful. Mean validation selection is worse than nearest
selection and far below oracle best-of-8. This rejects the simplest
"add prefix/final-state features to the static scorer" fix. The next privileged
branch-goal attempt would need to change the data/modeling setup more
substantially: broader query coverage, different candidate generation, or a
selector trained directly as an online intervention policy.

## 2026-06-26: Goal diagnostics CLI for learned-interface candidates

### Hypothesis

The plan says expensive RL should be gated by goal-identifiability diagnostics.
The reusable `learned_interface_goal_diagnostics` implementation already works
for named learned-interface candidates, but the CLI only allowed
`--representation vae512`. I removed that artificial restriction and ran the
diagnostic on the current `effect32_film` hierarchy.

### Code Change

`rl-rerun goal-diagnostics` now accepts:

```text
--representation vae512|learned_interface
```

The `--candidate` argument selects the actual learned-interface artifact. For
non-VAE candidates this uses the shared artifact path under
`artifacts/incremental/learned_interface/<candidate>/seed0/`; those candidates
are not currently retrained per `N` demo bucket.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film/diagnostics.json
```

### Result

| metric | value |
| --- | ---: |
| frame shuffle action change L2 | 0.950 |
| goal shuffle action change L2 | 0.062 |
| previous-action shuffle action change L2 | 0.133 |
| remaining-time shuffle action change L2 | 0.0014 |
| max same-state horizon sensitivity L2 | 0.0368 |
| goal shuffle MAE gap | 0.0102 |

Same-state horizon sensitivity:

| comparison | action change L2 |
| --- | ---: |
| 2 vs 5 | 0.0242 |
| 2 vs 10 | 0.0368 |
| 5 vs 10 | 0.0261 |

Artifact:

- `results/incremental/goal_diagnostics/n500/seed0/effect32_film/diagnostics.json`

### Interpretation

The `effect32_film` low level is not completely goal-blind, but action selection
is still dominated by the current frame. Goal shuffle changes actions by only
about `6.5%` of the frame-shuffle action change. This reinforces the hard-gate
decision: do not spend serious PPO on a new candidate unless goal-use
diagnostics are materially stronger than this or the experiment is explicitly a
representation/objective diagnostic.

## 2026-06-26: Short-horizon goal diagnostics

### Hypothesis

The short-horizon `effect32_film_h5` and `effect32_film_h2` hierarchies failed
closed-loop evaluation. This diagnostic checks whether they failed because they
were less goal-sensitive than the k10 baseline.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_h5/diagnostics.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_h2 \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_h2/diagnostics.json
```

### Result

| candidate | goal shuffle L2 | frame shuffle L2 | previous-action shuffle L2 | max horizon sensitivity L2 | goal MAE gap | action MAE h2 | action MAE h5 | action MAE h10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| effect32_film k10 | 0.0622 | 0.9503 | 0.1326 | 0.0368 | 0.0102 | 0.0484 | 0.0449 | 0.0425 |
| effect32_film_h5 | 0.0662 | 0.9442 | 0.1309 | 0.0419 | 0.0122 | 0.0428 | 0.0413 | 0.0460 |
| effect32_film_h2 | 0.0994 | 0.9346 | 0.1284 | 0.0810 | 0.0199 | 0.0408 | 0.0465 | 0.0585 |

Artifacts:

- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_h5/diagnostics.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_h2/diagnostics.json`

### Interpretation

The short-horizon variants do not fail because their low-level policies are less
goal-sensitive. h2 is the most goal-sensitive by these offline metrics, but it
was much worse in closed loop. This is important for Phase 2 gating:
goal-identifiability is a necessary rejection gate, not a promotion criterion.
The candidate also has to preserve closed-loop imitation quality under its
training/deployment horizon.

## 2026-06-26: Aggregate goal-diagnostics gate

### Hypothesis

Per-candidate goal diagnostics are useful but easy to overread. I added an
aggregate gate report so future PPO branches can be rejected consistently before
expensive training.

### Code Change

Added:

```text
rl-rerun aggregate-goal-diagnostics
```

The command reads diagnostic JSON files, writes a JSON and Markdown table, and
classifies candidates with:

```text
offline_goal_use_pass if:
  goal_shuffle_action_change_l2 >= 0.1
  or max_goal_sensitivity_l2 >= 0.1
otherwise:
  reject_low_goal_use
```

This is intentionally a rejection gate only. Passing candidates still need
closed-loop imitation quality and local-to-task transfer evidence.

### Command

```bash
uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  aggregate-goal-diagnostics \
  --input-glob 'results/incremental/goal_diagnostics/**/diagnostics.json' \
  --output results/incremental/goal_diagnostics/gate_report.json \
  --force
```

### Result

| candidate | N | gate status | goal shuffle L2 | max horizon sensitivity L2 |
| --- | ---: | --- | ---: | ---: |
| effect32_film | 500 | reject_low_goal_use | 0.0622 | 0.0368 |
| effect32_film_h2 | 500 | reject_low_goal_use | 0.0994 | 0.0810 |
| effect32_film_h5 | 500 | reject_low_goal_use | 0.0662 | 0.0419 |
| ae256_film | 1800 | offline_goal_use_pass | 0.2506 | 0.0937 |
| dae256_n010 | 1800 | reject_low_goal_use | 0.0772 | 0.0286 |
| effect32 | 1800 | reject_low_goal_use | 0.0279 | 0.0178 |
| effect32_film | 1800 | reject_low_goal_use | 0.0622 | 0.0368 |
| effect32_scene_film | 1800 | reject_low_goal_use | 0.0630 | 0.0352 |
| jepa256_r01_v1_c01 | 1800 | reject_low_goal_use | 0.0408 | 0.0246 |
| vae256_b1e5 | 1800 | reject_low_goal_use | 0.0405 | 0.0205 |
| vae512_b1e6_film | 1800 | offline_goal_use_pass | 0.2783 | 0.1266 |

Counts:

```text
total: 11
offline_goal_use_pass: 2
reject_low_goal_use: 9
```

Artifacts:

- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Interpretation

The aggregate makes the current tradeoff explicit. Effect32 is the best observed
deployment base but fails the strict offline goal-use gate. AE/VAE FiLM variants
pass the gate but were weaker deployment bases in previous closed-loop checks.
The next candidate worth serious PPO should satisfy both requirements: stronger
goal-use than effect32 and closed-loop imitation quality near or above
effect32_film.

I checked the existing closed-loop/R3 artifacts for the two gate-passing
candidates:

| candidate | offline gate | base learned success | base oracle success | R3-window frozen success | R3 tuned success | R3 delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | reject low goal-use | 0.645 | 0.645 | 0.634 | 0.684 | +0.050 |
| ae256_film | pass | 0.550 | 0.670 | 0.596 | 0.586 | -0.010 |
| vae512_b1e6_film | pass | 0.425 | 0.535 | 0.418 | 0.478 | +0.060 |

The rows use the existing matched learned-interface evals and 500-episode R3
checks around `seed_start=3500000`. They confirm that passing the offline
goal-use gate is not enough. `vae512_b1e6_film` reacts strongly to goals and R3
improves it, but the base policy is far below `effect32_film`. `ae256_film`
passes the gate and has a better oracle ceiling, but R3 is slightly negative
against its frozen branch. No existing candidate satisfies both requirements.

## 2026-06-26: Effect32 FiLM goal-sensitivity regularizer

### Hypothesis

The aggregate gate showed that effect32 has the best deployment quality but weak
offline goal use. This run tests the simplest architecture/objective fix: keep
the effect32 representation and high level, but add a low-level supervised
regularizer that penalizes shuffled-goal predictions that stay too close to the
correct-goal prediction.

### Code Change

`train_learned_interface_hierarchy` now supports candidate-level policy
overrides:

```text
policy_batch_size
policy_batches_per_epoch
policy_epochs
policy_lr
goal_sensitivity_weight
goal_sensitivity_margin
```

If `goal_sensitivity_weight > 0`, the low-level loss is:

```text
BC_MSE + weight * mean(relu(margin - ||pi(x_goal_shuffled) - stopgrad(pi(x))||_2)^2)
```

The regularizer is opt-in and does not affect existing candidates.

Candidate:

```yaml
effect32_film_gsens:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: film
  goal_sensitivity_weight: 0.05
  goal_sensitivity_margin: 0.2
  policy_epochs: 40
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_gsens \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens/diagnostics.json

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens \
  --seed 0 \
  --episodes 200 \
  --goal-source learned \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens \
  --seed 0 \
  --episodes 200 \
  --goal-source oracle \
  --eval-seed-start 3500000 \
  --force
```

### Training Metrics

Best hierarchy epoch was `37`. Validation action errors stayed close to the
baseline:

```text
oracle_action_mae: 0.0408
predicted_action_mae: 0.0413
prediction_induced_action_l2: 0.0169
```

### Goal Diagnostics

| candidate | goal shuffle L2 | frame shuffle L2 | goal/frame | max horizon sensitivity L2 | goal MAE gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 0.0622 | 0.9503 | 0.0655 | 0.0368 | 0.0102 |
| effect32_film_gsens | 0.1149 | 0.9406 | 0.1221 | 0.0476 | 0.0279 |

The aggregate gate now counts 3 passes out of 12 diagnostics; the new candidate
is one of the passes.

### Closed-Loop Evaluation

Matched 200-episode window at `seed_start=3500000`:

| candidate | learned success | learned max reward | oracle success | oracle max reward |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.742 | 0.645 | 0.746 |
| effect32_film_gsens | 0.500 | 0.647 | 0.515 | 0.661 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens/seed0/hierarchy.pt`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens/diagnostics.json`
- `results/incremental/learned_interface/effect32_film_gsens/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens/seed0/oracle_hierarchy_eval_200_seed3500000.json`

### Interpretation

The regularizer proves that the offline gate can be moved directly: goal-shuffle
action change nearly doubled and crossed the `0.1` pass threshold. But the
closed-loop hierarchy regressed badly. This rejects the naive "increase
goal-sensitivity by margin loss" fix. Future attempts should not optimize goal
sensitivity in isolation; they need a preservation constraint or a
closed-loop/deployment-aligned objective.

## 2026-06-26: Light effect32 FiLM goal-sensitivity regularizer

### Hypothesis

The strong `effect32_film_gsens` regularizer crossed the offline goal-use gate
but damaged closed-loop deployment. This run tests whether a much lighter
regularizer finds a better tradeoff.

### Candidate

```yaml
effect32_film_gsens_light:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: film
  goal_sensitivity_weight: 0.01
  goal_sensitivity_margin: 0.1
  policy_epochs: 40
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_light \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_gsens_light \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_light/diagnostics.json

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_light \
  --seed 0 \
  --episodes 200 \
  --goal-source learned \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_light \
  --seed 0 \
  --episodes 200 \
  --goal-source oracle \
  --eval-seed-start 3500000 \
  --force
```

### Training Metrics

Best hierarchy epoch was `40`:

```text
oracle_action_mae: 0.0405
predicted_action_mae: 0.0408
prediction_induced_action_l2: 0.0123
```

### Goal/Deployment Tradeoff

| candidate | goal shuffle L2 | max horizon sensitivity L2 | gate status | learned success | oracle success |
| --- | ---: | ---: | --- | ---: | ---: |
| effect32_film | 0.0622 | 0.0368 | reject | 0.645 | 0.645 |
| effect32_film_gsens_light | 0.0736 | 0.0405 | reject | 0.570 | 0.570 |
| effect32_film_gsens | 0.1149 | 0.0476 | pass | 0.500 | 0.515 |

After including this candidate, the aggregate goal-diagnostics report has:

```text
total: 13
offline_goal_use_pass: 3
reject_low_goal_use: 10
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_light/seed0/hierarchy.pt`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_light/diagnostics.json`
- `results/incremental/learned_interface/effect32_film_gsens_light/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_light/seed0/oracle_hierarchy_eval_200_seed3500000.json`

### Interpretation

The lighter regularizer gives a smoother version of the same bad tradeoff. It
increases goal-shuffle action change only slightly (`0.0622 -> 0.0736`), does
not pass the strict offline gate, and still costs `7.5` success points on the
matched learned-goal eval. This closes the simple sensitivity-margin branch:
goal-use needs to arise from a better representation/architecture or
deployment-aligned training signal, not from a standalone action-difference
margin.

## 2026-06-26: Effect32 goal-residual low-level architecture

I added a new learned-interface low-level conditioning mode,
`goal_residual`, to test a base-plus-residual architecture:

- the base policy sees only frame features, previous action, and remaining time;
- the residual policy sees frame features, the absolute goal, previous action,
  and remaining time;
- the residual final layer is zero-initialized, so training starts from a clean
  no-goal base path.

Config:

```yaml
effect32_goal_residual:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: goal_residual
  goal_residual_scale: 1.0
  policy_epochs: 40
```

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_goal_residual \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_goal_residual \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_goal_residual/diagnostics.json
```

### Training Metrics

Best hierarchy epoch was `39`:

```text
oracle_action_mae: 0.0421
predicted_action_mae: 0.0423
prediction_induced_action_l2: 0.0040
normalized_goal_l2: 2.5864
```

### Goal-Use Screen

| candidate | goal shuffle L2 | frame shuffle L2 | previous-action shuffle L2 | max horizon sensitivity L2 | gate status |
| --- | ---: | ---: | ---: | ---: | --- |
| effect32_film | 0.0622 | 0.9503 | 0.1326 | 0.0368 | reject |
| effect32_goal_residual | 0.0218 | 0.9705 | 0.1236 | 0.0154 | reject |

20-episode smoke eval:

| candidate | learned success | oracle success |
| --- | ---: | ---: |
| effect32_film | 0.750 | 0.750 |
| effect32_goal_residual | 0.450 | 0.450 |

After including this candidate, the aggregate goal-diagnostics report has:

```text
total: 14
offline_goal_use_pass: 3
reject_low_goal_use: 11
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_goal_residual/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_goal_residual/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_goal_residual/seed0/learned_hierarchy_eval_20.json`
- `results/incremental/learned_interface/effect32_goal_residual/seed0/oracle_hierarchy_eval_20.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_goal_residual/diagnostics.json`
- `results/incremental/goal_diagnostics/gate_report.json`

### Interpretation

This closes the simple base-plus-goal-residual branch. The architecture did not
preserve effect32 imitation quality and did not increase goal dependence; it
reduced goal-shuffle action change from `0.0622` to `0.0218`. I skipped the
longer 200-episode cross-check because both the offline gate and the 20-episode
deployment smoke were worse than the existing effect32 FiLM baseline.

## 2026-06-26: Effect64 FiLM capacity check

After the sensitivity and goal-residual branches failed to preserve deployment
quality, I tested a simple effect-code capacity increase: reuse the existing
`effect64` representation and high level, but train a FiLM-conditioned low
level.

Config:

```yaml
effect64_film:
  family: conditioning_ablation
  representation_candidate: effect64
  high_level_candidate: effect64
  conditioning: film
```

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect64_film \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect64_film \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect64_film/diagnostics.json

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect64_film \
  --seed 0 \
  --episodes 200 \
  --goal-source learned \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect64_film \
  --seed 0 \
  --episodes 200 \
  --goal-source oracle \
  --eval-seed-start 3500000 \
  --force
```

### Training Metrics

Best hierarchy epoch was `57`:

```text
oracle_action_mae: 0.0384
predicted_action_mae: 0.0386
prediction_induced_action_l2: 0.0142
normalized_goal_l2: 4.1331
```

### Results

| candidate | goal shuffle L2 | frame shuffle L2 | max horizon sensitivity L2 | gate status | learned success | oracle success |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| effect32_film | 0.0622 | 0.9503 | 0.0368 | reject | 0.645 | 0.645 |
| effect64_film | 0.0823 | 0.9348 | 0.0488 | reject | 0.595 | 0.535 |

After including this candidate, the aggregate goal-diagnostics report has:

```text
total: 15
offline_goal_use_pass: 3
reject_low_goal_use: 12
```

Artifacts:

- `artifacts/incremental/learned_interface/effect64_film/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect64_film/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect64_film/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect64_film/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect64_film/diagnostics.json`
- `results/incremental/goal_diagnostics/gate_report.json`

### Interpretation

The larger effect code moves goal sensitivity in the desired direction
(`0.0622 -> 0.0823` goal-shuffle L2), but it still fails the strict gate and
loses deployment quality (`0.645 -> 0.595` learned, `0.645 -> 0.535` oracle).
This closes the simple "increase effect latent capacity" branch as a fix. Like
the sensitivity-margin branch, it improves an offline goal-use number while
making the hierarchy less useful as an RL base.

## 2026-06-26: Fresh validation of prefix-summary online selector

The current summary had a weakly positive 100-episode signal for the
prefix-summary `--step-selector`, which uses online prefix approximations of the
non-deployable full-episode selector features. I ran a larger fresh
500-episode validation at `seed_start=4800000`, plus a matched ungated eval, to
check whether this is a real online-selector direction.

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --step-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_summary_selector_fit_train460_valid470.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_prefix_summary_selector_500_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_500_seed4800000.json
```

Results:

| policy | success | final reward | max reward | residual action rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.306 | 0.4642 | 0.4954 | 0.000 |
| ungated task-reward residual | 0.298 | 0.4585 | 0.4871 | 1.000 |
| prefix-summary step selector | 0.292 | 0.4538 | 0.4830 | 0.806 |

Selector details:

```text
features:
  episode_action_delta_l2_mean
  episode_action_delta_l2_max
  episode_policy_saturation_rate
  episode_goal_l2_initial
  episode_goal_l2_mean
  episode_high_level_decisions
mean residual norm: 0.000490
action saturation: 0.0456
```

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_prefix_summary_selector_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_500_seed4800000.json`

Interpretation:

The larger fresh window rejects the small positive 100-episode prefix-selector
signal. The selector still uses the residual branch most of the time and is
slightly worse than ungated residual, which is itself worse than frozen. This
closes the current linear online-selector branch for the task-reward-debug R3
checkpoint. A useful selector likely needs direct training in the closed-loop
intervention distribution, or the objective needs to produce a larger
task-aligned residual before selection is worth revisiting.

## 2026-06-26: Task-hard residual oracle-goal transfer check

The best task-hard local-R3 checkpoint (`bc=0.3`,
`max_base_terminal_env_reward=0.45`) failed the learned-goal 500-episode
closed-loop check. I ran oracle-goal closed-loop evals to separate high-level
goal quality from low-level objective quality.

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 200 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --goal-source oracle \
  --oracle-copy-mode state_dict \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_oracle_200_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --goal-source oracle \
  --oracle-copy-mode state_dict \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_oracle_500_seed4800000.json
```

Results:

| goal source | episodes | frozen success | residual success | success delta | final reward delta | max reward delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | 500 | 0.312 | 0.304 | -0.008 | -0.0093 | -0.0029 |
| oracle | 200 | 0.355 | 0.375 | +0.020 | +0.0185 | +0.0182 |
| oracle | 500 | 0.372 | 0.372 | +0.000 | +0.0006 | +0.0019 |

Additional 500-episode oracle diagnostics:

```text
mean residual norm: 0.000491
action saturation: 0.0469
```

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_oracle_200_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_oracle_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json`

Interpretation:

Oracle goals raise the frozen ceiling and remove the learned-goal regression,
but they do not turn this task-hard residual into a robust improvement. The
200-episode oracle gain was another small-window false lead; the 500-episode
result ties frozen success and only adds a tiny reward delta. This means the
task-hard objective is not failing only because the learned high level emits bad
goals. The low-level update itself is still too small or too weakly aligned with
closed-loop task success.

## 2026-06-26: Task-reward oracle segment selector diagnostic

The existing `--oracle-segment-selector` chose between frozen and tuned
branches by counterfactual one-segment terminal latent distance. I added:

```text
--oracle-segment-selector-metric latent_distance|env_reward
```

The new `env_reward` mode chooses the tuned branch when its counterfactual
one-segment terminal normalized dense reward is higher than the frozen branch.
This is still non-deployable, but it tests whether the previous oracle selector
failed because latent distance was the wrong local branch-selection proxy.

Commands:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 4600000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_20_seed4600000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 4700000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_20_seed4700000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4800000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_100_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4800000 \
  --num-envs 20 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_numenv20_100_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_500_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 20 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_numenv20_500_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4900000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_500_seed4900000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4900000 \
  --num-envs 20 \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_numenv20_500_seed4900000.json
```

### Results

20-episode smoke windows:

| seed | frozen | task-reward oracle selector | success delta | residual action rate |
| ---: | ---: | ---: | ---: | ---: |
| 4600000 | 0.300 | 0.250 | -0.050 | 0.519 |
| 4700000 | 0.300 | 0.350 | +0.050 | 0.581 |

Matched 100-episode check:

| policy | success | final reward | max reward | residual action rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.230 | 0.4147 | 0.4376 | 0.000 |
| ungated residual | 0.270 | 0.4177 | 0.4537 | 1.000 |
| task-reward oracle selector | 0.250 | 0.4208 | 0.4505 | 0.493 |

Matched 500-episode checks:

| seed | policy | success | final reward | max reward | residual action rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4800000 | frozen | 0.312 | 0.4727 | 0.4960 | 0.000 |
| 4800000 | ungated residual | 0.298 | 0.4553 | 0.4850 | 1.000 |
| 4800000 | task-reward oracle selector | 0.322 | 0.4753 | 0.5047 | 0.483 |
| 4900000 | frozen | 0.304 | 0.4532 | 0.4859 | 0.000 |
| 4900000 | ungated residual | 0.284 | 0.4404 | 0.4727 | 1.000 |
| 4900000 | task-reward oracle selector | 0.302 | 0.4563 | 0.4852 | 0.470 |

Two-window aggregate:

| policy | success | final reward | max reward |
| --- | ---: | ---: | ---: |
| frozen | 0.308 | 0.4630 | 0.4910 |
| ungated residual | 0.291 | 0.4478 | 0.4788 |
| task-reward oracle selector | 0.312 | 0.4658 | 0.4949 |

The selector's counterfactual branch diagnostics on the 100-episode check were:

```text
mean terminal latent-distance improvement selected metric trace: -0.0099
mean terminal env-reward improvement selected metric trace: -0.00019
mean residual norm: 0.000481
action saturation: 0.0404
```

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_20_seed4600000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_20_seed4700000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_100_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_numenv20_100_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_numenv20_500_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_500_seed4900000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_ungated_numenv20_500_seed4900000.json`

### Interpretation

The 100-episode result was another small-window false signal: the selector
looked worse than ungated on success despite better final reward. At 500
episodes on the first window, however, the task-reward oracle selector was
positive and beat both frozen and ungated residual. A fresh second 500-episode
window weakened that to approximately tied with frozen, but still clearly
better than ungated residual. This suggests one-segment task reward is a better
branch-selection proxy than one-segment latent distance for this checkpoint,
mostly because it suppresses harmful residual interventions.

This is still an upper-bound diagnostic, not a deployable policy: it requires
counterfactual simulator rollouts of frozen and tuned branches at the current
state. The useful next direction is to approximate this kind of task-aligned
closed-loop branch decision with deployable online state/history features, or to
train the residual/selector directly in that intervention distribution. The
effect size over frozen is currently tiny (`+0.004` success over two windows),
so this is a harm-reduction lead rather than a solved improvement.

## 2026-06-26 - Oracle segment selector trace export

### Hypothesis

The task-reward oracle segment selector is useful only as a non-deployable
upper-bound unless it can produce supervised labels and online features for a
future deployable selector.

### Implementation

I added `oracle_segment_selector_trace` to residual closed-loop eval results
when `--oracle-segment-selector` is enabled. The trace is column-oriented and
records one row per high-level replan:

- episode index and step index
- oracle residual/base choice
- frozen and tuned one-segment latent distances
- frozen and tuned one-segment terminal normalized dense rewards
- action-delta, saturation, goal-distance, and high-level-decision prefix
  features available before executing the current action

### Validation command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 5000000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5000000.json
```

### Results

| metric | value |
| --- | ---: |
| episodes | 20 |
| frozen success | 0.450 |
| selector success | 0.450 |
| success delta | 0.000 |
| final reward delta | -0.0041 |
| max reward delta | +0.0072 |
| selector residual action rate | 0.454 |
| selector decision residual rate | 0.444 |
| oracle decisions | 162 |
| trace rows | 162 |

The trace contains the expected keys:

```text
action_delta_l2_prefix_max
action_delta_l2_prefix_mean
base_env_reward
base_latent_distance
choose_residual
env_reward_delta
episode_index
goal_l2
goal_l2_prefix_mean
high_level_decisions
latent_distance_delta
policy_saturation_prefix_rate
step_index
tuned_env_reward
tuned_latent_distance
```

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5000000.json`

### Interpretation

The smoke run validates the trace plumbing: decision count and trace length
match exactly. This does not change the policy conclusion yet, but it creates
the dataset shape needed to train or fit a deployable branch selector against
the task-reward oracle labels.

## 2026-06-26 - Trace-fitted deployable segment selector

### Hypothesis

If the task-reward oracle selector's branch decisions are predictable from
online prefix features, a deployable segment selector should imitate those
decisions well enough to reduce residual harm without simulator
counterfactuals.

### Implementation

I added:

- `hcl-poc rl-rerun fit-oracle-segment-selector`, which fits a linear ridge
  selector from `oracle_segment_selector_trace` rows and writes the same
  loadable selector JSON shape as the existing closed-loop selector.
- `eval-closed-loop-r{1,2,3} --segment-selector`, which scores the selector
  once at each high-level replan and holds the base/residual choice for the
  segment. This matches the oracle trace decision timing better than
  `--step-selector`, which gates each action after computing the candidate
  residual action.

Default fitted features:

```text
episode_action_delta_l2_mean
episode_action_delta_l2_max
episode_policy_saturation_rate
episode_goal_l2_initial
episode_goal_l2_mean
episode_high_level_decisions
```

### Commands

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml fit-oracle-segment-selector \
  --train-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5000000.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/oracle_segment_trace_selector_smoke20_seed5000000.json \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 5000000 \
  --num-envs 20 \
  --segment-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/oracle_segment_trace_selector_smoke20_seed5000000.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_trace_segment_selector_smoke20_seed5000000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 5100000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5100000.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml fit-oracle-segment-selector \
  --train-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5000000.json \
  --validation-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5100000.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/oracle_segment_trace_selector_train20_5000000_valid20_5100000.json \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 5100000 \
  --num-envs 20 \
  --segment-selector results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/oracle_segment_trace_selector_train20_5000000_valid20_5100000.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_trace_segment_selector_train20_5000000_valid20_5100000.json
```

### Results

Trace imitation:

| split | decisions | oracle residual rate | selector residual rate | AUC | accuracy | selected reward gap vs oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train seed 5000000 | 162 | 0.444 | 0.179 | 0.637 | 0.611 | -0.00522 |
| valid seed 5100000 | 194 | 0.557 | 0.299 | 0.507 | 0.505 | -0.00123 |

Closed-loop segment-selector smokes:

| eval seed | frozen success | segment-selector success | success delta | final reward delta | residual action rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5000000, train selector | 0.450 | 0.250 | -0.200 | -0.1435 | 0.185 |
| 5100000, train5000000 selector | 0.150 | 0.200 | +0.050 | +0.0233 | 0.269 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/oracle_segment_trace_selector_smoke20_seed5000000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_trace_segment_selector_smoke20_seed5000000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5100000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/oracle_segment_trace_selector_train20_5000000_valid20_5100000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_trace_segment_selector_train20_5000000_valid20_5100000.json`

### Interpretation

The trace-fitted selector path works mechanically, but the simple deployable
prefix features do not predict the task-reward oracle decisions on a fresh
20-episode trace. Validation AUC is essentially chance (`0.507`), and the
closed-loop smokes are too noisy to outweigh that. This weakens the idea that a
linear selector over the current prefix features can recover the oracle
task-reward branch decision for this checkpoint.

The next selector attempt should either use richer deployable state features
from the current observation/latent and goal, or train the residual in a way
that makes the branch effect larger and easier to classify. A larger trace-only
dataset may still be useful for feature probing, but it should not be promoted
to long closed-loop evaluation unless validation AUC moves clearly above
chance.

## 2026-06-26 - Max-reward and success oracle selector metrics

### Hypothesis

The previous task-reward oracle selector used only the terminal dense reward
after one segment. A more deployment-aligned non-deployable selector might do
better if it chooses by max dense reward within the segment or by whether either
branch reaches task success during the segment.

### Implementation

I extended `--oracle-segment-selector-metric` with:

```text
env_max_reward
success
```

The oracle branch trace now also records:

```text
base_env_max_reward
tuned_env_max_reward
env_max_reward_delta
base_success_once
tuned_success_once
success_once_delta
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 5200000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_max_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envmaxreward_20_seed5200000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 20 \
  --eval-seed-start 5200000 \
  --num-envs 20 \
  --oracle-segment-selector \
  --oracle-segment-selector-metric success \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_success_20_seed5200000.json
```

### Results

| selector metric | frozen success | selector success | success delta | max reward delta | residual action rate | decision residual rate | branch max-reward delta | branch success-once delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| env_max_reward | 0.450 | 0.450 | 0.000 | -0.0052 | 0.468 | 0.457 | +0.0093 | +0.013 |
| success | 0.450 | 0.450 | 0.000 | 0.0000 | 0.000 | 0.000 | -0.0082 | -0.013 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envmaxreward_20_seed5200000.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_success_20_seed5200000.json`

### Interpretation

The new metrics work mechanically and expose richer branch outcomes, but the
smoke does not reveal a stronger oracle upper bound. `env_max_reward` sees
positive one-segment counterfactual deltas but does not improve closed-loop
success or max reward on this small window. `success` is too sparse: it chose
the residual branch zero times. This reinforces the previous conclusion that
one-segment branch selection signals are not the main missing ingredient for
this checkpoint.

## 2026-06-26 - Full-window max-reward oracle selector validation

### Hypothesis

The 20-episode max-reward oracle selector smoke was inconclusive. A 500-episode
window should tell whether choosing the tuned branch by one-segment max dense
reward is a real upper-bound branch-selection signal.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --oracle-copy-mode state_dict \
  --oracle-segment-selector \
  --oracle-segment-selector-metric env_max_reward \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envmaxreward_500_seed4800000.json
```

### Results

| metric | frozen | max-reward oracle selector | delta |
| --- | ---: | ---: | ---: |
| success | 0.306 | 0.294 | -0.012 |
| final dense reward | 0.4642 | 0.4555 | -0.0087 |
| max dense reward | 0.4954 | 0.4859 | -0.0095 |

Selector diagnostics:

| residual action rate | decision residual rate | decisions | branch max-reward delta | branch reward delta | branch success-once delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.476 | 0.472 | 4309 | +0.0010 | +0.0008 | +0.0012 |

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envmaxreward_500_seed4800000.json`

### Interpretation

The full-window validation rejects the max-reward oracle selector for this
checkpoint. Even with non-deployable branch rollouts, the selected tuned branch
has slightly positive one-segment counterfactual deltas but worse closed-loop
success, final reward, and max reward. This closes the simple one-segment
max-reward upper-bound selector path; the next objective/selector work needs a
longer-horizon or directly closed-loop training signal.

## 2026-06-26 - Full-window success oracle selector validation

### Hypothesis

The one-segment success selector was too sparse on the 20-episode smoke, but a
500-episode validation can check whether sparse success deltas become useful at
larger scale.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --oracle-copy-mode state_dict \
  --oracle-segment-selector \
  --oracle-segment-selector-metric success \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_success_500_seed4800000.json
```

### Results

| metric | frozen | success oracle selector | delta |
| --- | ---: | ---: | ---: |
| success | 0.306 | 0.306 | 0.000 |
| final dense reward | 0.4642 | 0.4642 | -0.00003 |
| max dense reward | 0.4954 | 0.4954 | -0.00001 |

Selector diagnostics:

| residual action rate | decision residual rate | decisions | branch max-reward delta | branch reward delta | branch success-once delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0036 | 0.0045 | 4266 | -0.0002 | -0.0007 | -0.0005 |

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_success_500_seed4800000.json`

### Interpretation

The success selector is a near no-op at full-window scale. It preserves frozen
success because it almost never selects the residual branch, not because it
finds useful interventions. Together with the max-reward validation, this closes
the simple one-segment task-oracle selector metrics for this checkpoint.

## 2026-06-26 - Per-sample local eval export for proxy analysis

### Hypothesis

The plan calls for validating reachability/local metrics against task outcomes.
Existing local eval JSONs only saved aggregate means, making proxy analysis
indirect. Exporting bounded per-sample local outcomes should make it possible to
measure whether local distance improvements actually align with task reward and
success deltas.

### Implementation

I added `--include-samples` to:

```text
eval-local-r1
eval-local-r2
eval-local-r3
```

When enabled, local eval writes a `sample_metrics` block with one row per local
sample:

```text
initial_distance
base_final_distance
final_distance
distance_reduction_delta_vs_base
base/final/max/mean dense rewards and deltas
base/tuned success-once flags and delta
mean_action_delta_l2
episode_index
env_index
```

### Commands

The first attempt used the full 4096-env validation bank and was interrupted
after several minutes in DINO preprocessing. I then validated the same path on
the smaller 512-env validation bank:

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_1_seed0.json
```

### Results

Aggregate smoke metrics:

| metric | value |
| --- | ---: |
| sampled local episodes | 512 |
| initial distance mean | 0.8632 |
| frozen final distance mean | 0.5688 |
| tuned final distance mean | 0.5553 |
| raw-distance delta vs frozen | +0.0135 |
| final dense-reward delta vs frozen | +0.0079 |
| success-once delta vs frozen | +0.0039 |

Per-sample proxy correlations:

| pair | Pearson r |
| --- | ---: |
| raw-distance delta vs final dense-reward delta | 0.052 |
| raw-distance delta vs max dense-reward delta | 0.0019 |
| action-delta mean vs final dense-reward delta | 0.014 |
| frozen final distance vs final dense-reward delta | 0.022 |
| initial distance vs final dense-reward delta | 0.035 |

Success-delta counts:

```text
-1: 11
 0: 488
+1: 13
```

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_1_seed0.json`

### Interpretation

The export works and gives the missing per-sample substrate for proxy analysis.
On this smoke, raw local distance improvement is only weakly correlated with
task dense-reward improvement, and success deltas are almost balanced. This is
additional evidence that local raw reachability is not a reliable deployment
proxy for the current checkpoint. The next useful use of this infrastructure is
to compare raw L2, `D_phi`, dense reward, and success on the same per-sample
local bank instead of relying only on aggregate means.

## 2026-06-26 - Same-sample D_phi local proxy export

### Hypothesis

Raw local L2 and learned `D_phi` may disagree on whether a tuned policy improved
the local rollout. Adding optional `D_phi` distances to the same local sample
export lets us compare raw reachability, learned reachability, dense reward,
and success on identical reset samples.

### Implementation

I added `--reachability-checkpoint` to `eval-local-r{1,2,3}`. When provided,
local eval reports:

```text
initial_reachability_distance_mean
base_final_reachability_distance_mean
final_reachability_distance_mean
base_reachability_reduction_mean
reachability_reduction_mean
reachability_reduction_delta_vs_base
```

With `--include-samples`, the same fields are also exported per sample.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_dphi_1_seed0.json
```

### Results

Aggregate same-sample metrics:

| metric | value |
| --- | ---: |
| sampled local episodes | 512 |
| raw-distance delta vs frozen | +0.0135 |
| D_phi delta vs frozen | -0.0091 |
| final dense-reward delta vs frozen | +0.0079 |
| success-once delta vs frozen | +0.0039 |
| initial D_phi distance | 0.8485 |
| frozen final D_phi distance | 0.7909 |
| tuned final D_phi distance | 0.8000 |

Per-sample proxy correlations:

| pair | Pearson r |
| --- | ---: |
| raw-distance delta vs final dense-reward delta | 0.052 |
| D_phi delta vs final dense-reward delta | 0.017 |
| raw-distance delta vs max dense-reward delta | 0.0019 |
| D_phi delta vs max dense-reward delta | -0.0157 |
| raw-distance delta vs D_phi delta | 0.139 |
| action-delta mean vs D_phi delta | 0.0084 |

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_dphi_1_seed0.json`

### Interpretation

The optional `D_phi` local eval path works and exposes a concrete mismatch: the
same checkpoint can look slightly positive under raw L2 and slightly negative
under learned reachability on the same reset samples. Neither proxy correlates
meaningfully with dense task-reward deltas on this smoke. This strengthens the
case that the next objective work should use same-sample proxy audits before
promoting a local metric to PPO reward or checkpoint selection.

## 2026-06-26 - Local sample proxy audit command

### Hypothesis

The per-sample local eval exports should be audited through a repeatable command
rather than ad hoc Python snippets, so raw L2, `D_phi`, action magnitude, and
state-difficulty proxies can be compared consistently.

### Implementation

I added:

```bash
uv run hcl-poc rl-rerun audit-local-sample-proxies \
  --local-json ... \
  --output ...
```

The audit reports, for every available proxy:

- mean and standard deviation
- Pearson correlation with final dense-reward delta
- Pearson correlation with max dense-reward delta
- mean proxy value on success-improved and success-regressed samples
- AUC for classifying positive versus negative success deltas

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_dphi_proxy_audit.json \
  --force
```

### Results

Aggregate:

| metric | value |
| --- | ---: |
| rows | 512 |
| final dense-reward delta mean | +0.00794 |
| max dense-reward delta mean | +0.00266 |
| success delta mean | +0.00391 |
| success delta counts | `-1: 11`, `0: 488`, `+1: 13` |

Proxy audit:

| proxy | mean | corr final reward delta | corr max reward delta | success-delta AUC |
| --- | ---: | ---: | ---: | ---: |
| raw-distance delta | +0.0135 | 0.052 | 0.0019 | 0.469 |
| D_phi delta | -0.0091 | 0.0168 | -0.0157 | 0.483 |
| mean action delta | 0.00037 | 0.0145 | 0.0572 | 0.636 |
| initial raw distance | 0.863 | 0.0346 | 0.102 | 0.643 |
| frozen final raw distance | 0.569 | 0.0217 | 0.0302 | 0.538 |

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n512_dphi_proxy_audit.json`

### Interpretation

This audit formalizes the proxy mismatch: on this smoke, raw and learned
reachability deltas are both below chance for separating success improvements
from success regressions among the discordant samples. Initial difficulty and
action magnitude look more predictive, but the discordant subset is only 24
samples, so this is a feature-design hint rather than a policy conclusion. The
main practical gain is that future candidate checkpoints can now be screened
with the same proxy audit before expensive closed-loop validation.

## 2026-06-26 - Comparative proxy audit on task-hard checkpoint

### Hypothesis

The same-sample proxy audit should be useful for comparing candidate local
checkpoints, not just diagnosing one checkpoint. The task-hard `bc=0.3`
one-update checkpoint is a good contrast because it previously showed the best
task-hard subset local signal while remaining weak in broader closed-loop
transfer.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n512_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n512_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n512_dphi_proxy_audit.json \
  --force
```

### Results

Same 512-sample bank comparison:

| checkpoint | final reward delta | max reward delta | success delta | raw delta | D_phi delta | raw success AUC | D_phi success AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task-reward debug | +0.0079 | +0.0027 | +0.0039 | +0.0135 | -0.0091 | 0.469 | 0.483 |
| task-hard bc0.3 | +0.0106 | +0.0130 | +0.0195 | +0.0089 | -0.0083 | 0.333 | 0.597 |

Task-hard proxy correlations:

| proxy | corr final reward delta | corr max reward delta | success-delta AUC |
| --- | ---: | ---: | ---: |
| raw-distance delta | 0.0908 | 0.0264 | 0.333 |
| D_phi delta | 0.0614 | 0.0579 | 0.597 |
| mean action delta | -0.0390 | -0.0069 | 0.431 |
| initial raw distance | -0.0275 | -0.0305 | 0.340 |

Success-delta counts for task-hard:

```text
-1: 8
 0: 486
+1: 18
```

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n512_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n512_dphi_proxy_audit.json`

### Interpretation

The audit distinguishes the two checkpoints: task-hard has a stronger
same-sample task signal than task-reward-debug on this 512-bank smoke. But raw
local distance still misranks the success-discordant samples, and `D_phi` only
partly helps. Dense-reward correlations stay weak for both proxies. This makes
the next objective clearer: use the proxy audit as a promotion gate, but do not
select checkpoints from raw L2 or D_phi alone unless the audit improves on a
larger bank.

## 2026-06-26 - Full-bank task-hard proxy audit

### Hypothesis

The 512-bank task-hard proxy audit was positive on aggregate task reward and
success, but the discordant-success subset was small. A full 4096-env
same-sample audit should tell whether that signal survives a larger validation
bank.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --force
```

### Results

512-bank versus 4096-bank task-hard audit:

| bank | rows | final reward delta | max reward delta | success delta counts | raw delta | D_phi delta | raw success AUC | D_phi success AUC |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| n512 val | 512 | +0.0106 | +0.0130 | `-1: 8`, `0: 486`, `+1: 18` | +0.0089 | -0.0083 | 0.333 | 0.597 |
| n4096 val | 4096 | -0.0024 | +0.0010 | `-1: 88`, `0: 3914`, `+1: 94` | -0.0071 | -0.0042 | 0.562 | 0.526 |

Full-bank proxy correlations:

| proxy | corr final reward delta | corr max reward delta | success-delta AUC |
| --- | ---: | ---: | ---: |
| raw-distance delta | 0.114 | 0.023 | 0.562 |
| D_phi delta | 0.0489 | 0.0185 | 0.526 |
| mean action delta | 0.0319 | 0.0303 | 0.577 |
| initial raw distance | 0.0078 | 0.0002 | 0.483 |
| frozen final raw distance | 0.0393 | 0.0057 | 0.543 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json`

### Interpretation

The 512-bank positive signal does not survive the full 4096-env same-sample
audit. The task-hard checkpoint is slightly negative on final dense reward and
negative under both raw L2 and D_phi local distance, with success nearly tied.
Proxy AUCs are only weakly above chance. This is a useful correction: the
same-sample audit should be run at the full validation-bank scale before using a
checkpoint as evidence for a better local objective or before promoting it to
closed-loop evaluation.

## 2026-06-26 - Full-bank task-reward-debug proxy audit

### Hypothesis

The task-hard full-bank result should be compared against the task-reward-debug
checkpoint on the same 4096-env validation bank. Otherwise the comparison mixes
full-bank evidence for one checkpoint with 512-bank smoke evidence for the
other.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --force
```

### Results

Full 4096-bank comparison:

| checkpoint | final reward delta | max reward delta | success delta counts | raw delta | D_phi delta | raw success AUC | D_phi success AUC |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| task-reward debug | -0.0020 | -0.0008 | `-1: 90`, `0: 3921`, `+1: 85` | -0.0020 | +0.0005 | 0.531 | 0.503 |
| task-hard bc0.3 | -0.0024 | +0.0010 | `-1: 88`, `0: 3914`, `+1: 94` | -0.0071 | -0.0042 | 0.562 | 0.526 |

Task-reward-debug full-bank proxy correlations:

| proxy | corr final reward delta | corr max reward delta | success-delta AUC |
| --- | ---: | ---: | ---: |
| raw-distance delta | 0.0987 | 0.0165 | 0.531 |
| D_phi delta | 0.0534 | 0.0166 | 0.503 |
| mean action delta | 0.0220 | 0.0181 | 0.549 |
| initial raw distance | -0.0086 | -0.0092 | 0.462 |
| frozen final raw distance | 0.0250 | -0.0009 | 0.523 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json`

### Interpretation

The symmetric full-bank comparison closes the apparent local positives for both
candidate checkpoints. Task-reward-debug is slightly negative on final reward,
max reward, and success; task-hard is slightly negative on final reward and
local distances, with only a tiny success edge. Raw L2 and D_phi have weak
proxy value at best. This points away from selecting among these checkpoints
with scalar local metrics and toward changing the objective or using richer
deployment-aligned proxy features.

## 2026-06-26 - Local proxy audit comparison command

### Hypothesis

The full-bank proxy audits should be comparable by a repeatable command rather
than by manually stitching individual JSON summaries. This makes checkpoint
promotion gates easier to audit as more local-R3 variants are added.

### Command

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml compare-local-proxy-audits \
  --audit-json \
    results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --name task_reward_debug taskhard_bc03 \
  --output results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_taskreward_vs_taskhard.json \
  --force
```

### Results

| checkpoint | final reward delta | max reward delta | success delta | raw success AUC | D_phi success AUC | positive task gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task-reward debug | -0.0020 | -0.0008 | -0.0012 | 0.531 | 0.503 | false |
| task-hard bc0.3 | -0.0024 | +0.0010 | +0.0015 | 0.562 | 0.526 | false |

Rankings:

- final reward delta: task-reward debug, task-hard bc0.3
- max reward delta: task-hard bc0.3, task-reward debug
- success delta: task-hard bc0.3, task-reward debug
- best proxy success AUC: task-hard bc0.3, task-reward debug

Artifact:

- `results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_taskreward_vs_taskhard.json`

### Interpretation

The comparison command preserves the same conclusion as the manual read. The
task-hard `bc=0.3` checkpoint has the better success, max-reward, and proxy-AUC
ranking, but neither checkpoint passes the positive task gate because both are
negative on final dense reward. The current promotion rule should therefore
reject both and move effort toward objective changes or richer deployment-aligned
selector features instead of promoting either checkpoint.

## 2026-06-26 - D_phi reward local-R3 smoke

### Hypothesis

The plan calls for using the learned reachability distance as an RL reward, not
only as an evaluation metric. If raw VAE L2 is the wrong local reward, replacing
the local-R3 reward distance with `D_phi` should improve the full-bank local
proxy audit.

### Implementation

I added `train-local-r3 --reward-distance-metric raw_l2|reachability` and
`--reachability-checkpoint`. With `reachability`, the training loop uses `D_phi`
for dense progress, terminal distance, and paired local-distance rewards.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name dphi_reward_n4096_1update_bc1_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 1 \
  --terminal-weight 1 \
  --dense-progress-weight 1 \
  --reward-mode progress \
  --reward-distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --checkpoint-every-updates 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml compare-local-proxy-audits \
  --audit-json \
    results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --name task_reward_debug taskhard_bc03 dphi_reward \
  --output results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_taskreward_taskhard_dphi.json \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json
```

### Results

Training final row:

| global step | reward distance | mean reward | mean D_phi distance | terminal D_phi distance | action delta | saturation | task success diagnostic |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40960 | reachability | -0.0736 | 0.8402 | 0.8024 | 0.0108 | 0.0075 | 0.213 |

Full-bank local proxy audit:

| checkpoint | final reward delta | max reward delta | success delta | raw delta | D_phi delta | raw success AUC | D_phi success AUC | positive task gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| task-reward debug | -0.0020 | -0.0008 | -0.0012 | -0.0020 | +0.0005 | 0.531 | 0.503 | false |
| task-hard bc0.3 | -0.0024 | +0.0010 | +0.0015 | -0.0071 | -0.0042 | 0.562 | 0.526 | false |
| D_phi reward | +0.0026 | +0.0005 | +0.0007 | +0.0002 | +0.0035 | 0.549 | 0.500 | true |

Closed-loop learned-goal validation:

| episodes | frozen success | D_phi reward success | success delta | final reward delta | max reward delta | residual norm | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.300 | 0.300 | 0.000 | -0.0007 | -0.0011 | 0.00048 | 0.0475 |
| 500 | 0.306 | 0.302 | -0.004 | -0.0046 | -0.0052 | 0.00048 | 0.0446 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json`
- `results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_taskreward_taskhard_dphi.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_100_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json`

### Interpretation

Using `D_phi` as the reward distance is mechanically working and gives a better
full-bank local proxy result than the previous task-reward/task-hard local-R3
candidates. It is the first candidate in this comparison to pass the simple
positive local task gate. But the update is extremely small in closed-loop
deployment and does not improve learned-goal task success. The next D_phi reward
check should increase the effect size deliberately, for example by lowering BC
weight, increasing updates, or using D_phi paired reward, while keeping the
full-bank local audit and 500-episode closed-loop validation as promotion gates.

## 2026-06-26 - D_phi reward lower-BC effect-size check

### Hypothesis

The first D_phi reward run passed the full-bank local gate but had very small
closed-loop action changes. Lowering the BC anchor from `1.0` to `0.3` might
increase useful D_phi-driven policy change while keeping the same stable LR and
initial action noise.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name dphi_reward_n4096_1update_bc03_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 0.3 \
  --terminal-weight 1 \
  --dense-progress-weight 1 \
  --reward-mode progress \
  --reward-distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --checkpoint-every-updates 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml compare-local-proxy-audits \
  --audit-json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --name dphi_bc1 dphi_bc03 \
  --output results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_dphi_bc1_vs_bc03.json \
  --force
```

### Results

Training final row:

| run | bc weight | mean reward | mean D_phi distance | terminal D_phi distance | action delta | saturation | task success diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D_phi bc1 | 1.0 | -0.0736 | 0.8402 | 0.8024 | 0.0108 | 0.0075 | 0.213 |
| D_phi bc0.3 | 0.3 | -0.0736 | 0.8402 | 0.8024 | 0.0108 | 0.0075 | 0.213 |

Full-bank local proxy comparison:

| checkpoint | final reward delta | max reward delta | success delta | raw delta | D_phi delta | raw success AUC | D_phi success AUC | positive task gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D_phi bc1 | +0.0026 | +0.0005 | +0.0007 | +0.0002 | +0.0035 | 0.549 | 0.500 | true |
| D_phi bc0.3 | -0.0034 | -0.0032 | -0.0049 | +0.0003 | -0.0018 | 0.518 | 0.535 | false |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc03_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json`
- `results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_dphi_bc1_vs_bc03.json`

### Interpretation

Lowering BC weight did not increase useful effect size. The training metrics
were effectively unchanged because the BC loss is already tiny in this
one-update final-layer setup, and the held-out full-bank local audit regressed
below the `bc=1` D_phi run. Since `bc=0.3` failed the local promotion gate, I did
not run closed-loop validation. The next D_phi reward check should change the
optimization regime more substantially, such as more updates or a paired D_phi
terminal reward, rather than simply weakening this already-small BC term.

## 2026-06-26 - D_phi reward 3-update check

### Hypothesis

The one-update D_phi reward run passed the full-bank local gate but had tiny
closed-loop action changes. Since lowering BC failed, the next clean effect-size
test is to keep the stable `bc=1`, `lr=1e-5`, `logstd=-5` setup and run three
PPO updates.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name dphi_reward_n4096_3update_bc1_lr1e5_logstd5 \
  --steps 122880 \
  --bc-weight 1 \
  --terminal-weight 1 \
  --dense-progress-weight 1 \
  --reward-mode progress \
  --reward-distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --checkpoint-every-updates 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml compare-local-proxy-audits \
  --audit-json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --name dphi_1update dphi_3update \
  --output results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_dphi_1update_vs_3update.json \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 100 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/closed_loop_learned_100_seed4800000.json

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-closed-loop-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/latest.pt \
  --n-demo 500 \
  --seed 0 \
  --episodes 500 \
  --eval-seed-start 4800000 \
  --num-envs 64 \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json
```

### Results

Training history:

| update | global step | mean reward | mean D_phi distance | terminal D_phi distance | action delta | saturation | task success diagnostic |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40960 | -0.0736 | 0.8402 | 0.8024 | 0.0108 | 0.0075 | 0.213 |
| 2 | 81920 | -0.0721 | 0.8396 | 0.7951 | 0.0108 | 0.0077 | 0.212 |
| 3 | 122880 | -0.0724 | 0.8352 | 0.7957 | 0.0108 | 0.0078 | 0.215 |

Full-bank local proxy comparison:

| checkpoint | final reward delta | max reward delta | success delta | raw delta | D_phi delta | positive task gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D_phi 1 update | +0.0026 | +0.0005 | +0.0007 | +0.0002 | +0.0035 | true |
| D_phi 3 updates | +0.0024 | +0.0009 | +0.0012 | +0.0026 | +0.0025 | true |

Closed-loop learned-goal validation:

| run | episodes | frozen success | tuned success | success delta | final reward delta | max reward delta | residual norm | saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D_phi 3 updates | 100 | 0.300 | 0.330 | +0.030 | +0.0271 | +0.0232 | 0.00090 | 0.0453 |
| D_phi 3 updates | 500 | 0.306 | 0.302 | -0.004 | +0.0007 | -0.0032 | 0.00091 | 0.0431 |
| D_phi 1 update | 500 | 0.306 | 0.302 | -0.004 | -0.0046 | -0.0052 | 0.00048 | 0.0446 |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json`
- `results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_dphi_1update_vs_3update.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/closed_loop_learned_100_seed4800000.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json`

### Interpretation

Three D_phi reward updates improve the full-bank local proxy audit relative to
one update on success delta, max-reward delta, and raw-distance delta, and the
100-episode closed-loop smoke looked promising. The 500-episode validation did
not confirm a robust deployment improvement: success stayed slightly below
frozen, and max reward was still negative. This is the strongest D_phi-reward
local candidate so far, but it remains a local improvement without a reliable
learned-goal closed-loop gain. The next check should change the deployment
alignment of the reward, for example paired D_phi terminal improvement or a
longer-horizon/closed-loop objective, rather than simply adding more identical
updates.

## 2026-06-26 - Paired D_phi terminal reward check

### Hypothesis

The progress-only D_phi reward improves held-out local metrics but does not
transfer reliably to learned-goal closed-loop deployment. A paired terminal
reward should be more deployment-aligned because it rewards improvement over the
frozen low-level from the same local reset and goal.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 \
  --seed 0 \
  --run-name dphi_paired_n4096_1update_bc1_lr1e5_logstd5 \
  --steps 40960 \
  --bc-weight 1 \
  --terminal-weight 1 \
  --dense-progress-weight 1 \
  --reward-mode paired \
  --reward-distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --learning-rate 1e-5 \
  --initial-logstd -5 \
  --checkpoint-every-updates 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r3 \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --episodes 1 \
  --include-samples \
  --reachability-checkpoint artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-local-sample-proxies \
  --local-json results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml compare-local-proxy-audits \
  --audit-json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
    results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json \
  --name dphi_progress_1update dphi_progress_3update dphi_paired_1update \
  --output results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_dphi_progress_vs_paired.json \
  --force
```

### Results

Training final row:

| reward | mean reward | mean D_phi distance | terminal D_phi distance | base terminal D_phi distance | paired D_phi improvement | fraction paired improved | action delta | saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paired D_phi | +0.0063 | 0.8402 | 0.8024 | 0.7987 | -0.0037 | 0.488 | 0.0108 | 0.0075 |

Full-bank local proxy comparison:

| checkpoint | final reward delta | max reward delta | success delta | raw delta | D_phi delta | raw success AUC | D_phi success AUC | positive task gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D_phi progress 1 update | +0.0026 | +0.0005 | +0.0007 | +0.0002 | +0.0035 | 0.549 | 0.500 | true |
| D_phi progress 3 updates | +0.0024 | +0.0009 | +0.0012 | +0.0026 | +0.0025 | 0.545 | 0.497 | true |
| paired D_phi 1 update | -0.0008 | +0.0012 | +0.0017 | -0.0011 | -0.0063 | 0.555 | 0.504 | false |

Artifacts:

- `artifacts/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/latest.pt`
- `results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/history.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_1_seed0.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/local_eval_samples_n4096_dphi_proxy_audit.json`
- `results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_dphi_progress_vs_paired.json`

### Interpretation

The paired D_phi terminal reward is runnable, but this one-update checkpoint
does not pass the local promotion gate. Training already showed negative paired
D_phi improvement against the cached frozen terminal distance, and the held-out
full-bank audit regressed final dense reward and D_phi reduction. I did not run
closed-loop validation. This weakens the idea that simply changing the terminal
reward from absolute D_phi to paired D_phi is enough; the next reward experiment
should likely change horizon/deployment alignment more substantially rather than
rerun the same one-segment local objective.

## 2026-06-26 - D_phi reward goal-use diagnostics

### Hypothesis

The D_phi reward checkpoints improve some full-bank local scalar metrics but do
not reliably improve learned-goal closed-loop deployment. If the core issue is
still weak goal conditioning, the D_phi checkpoints should have nearly the same
goal sensitivity as frozen.

### Commands

```bash
uv run python scripts/rl_rerun_condition_block_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 4096 \
  --batch-size 512 \
  --horizon 10 \
  --policy dphi_progress_1=artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --policy dphi_progress_3=artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/latest.pt \
  --policy dphi_paired_1=artifacts/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_goal_diagnostics_condition_block_n4096.json

uv run python scripts/rl_rerun_valid_goal_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 4096 \
  --batch-size 512 \
  --horizons 2,5,10,20 \
  --reference-horizon 10 \
  --policy dphi_progress_1=artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --policy dphi_progress_3=artifacts/rl_rerun/local_r3/n500/seed0/dphi_reward_n4096_3update_bc1_lr1e5_logstd5/latest.pt \
  --policy dphi_paired_1=artifacts/rl_rerun/local_r3/n500/seed0/dphi_paired_n4096_1update_bc1_lr1e5_logstd5/latest.pt \
  --output results/rl_rerun/local_r3/n500/seed0/dphi_reward_goal_diagnostics_valid_goal_n4096.json
```

### Results

Condition-block shuffle action L2:

| policy | observation | goal | previous action | remaining |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.807 | 0.0467 | 0.0738 | 0.000 |
| D_phi progress 1 update | 0.812 | 0.0461 | 0.0742 | 0.000 |
| D_phi progress 3 updates | 0.807 | 0.0467 | 0.0725 | 0.000 |
| paired D_phi 1 update | 0.814 | 0.0468 | 0.0716 | 0.000 |

Same-state valid future-goal action L2:

| policy | k2 vs k10 | k5 vs k10 | k20 vs k10 | k2 vs k20 |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.02316 | 0.01691 | 0.01862 | 0.02419 |
| D_phi progress 1 update | 0.02316 | 0.01691 | 0.01862 | 0.02419 |
| D_phi progress 3 updates | 0.02316 | 0.01690 | 0.01861 | 0.02418 |
| paired D_phi 1 update | 0.02316 | 0.01691 | 0.01862 | 0.02419 |

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_goal_diagnostics_condition_block_n4096.json`
- `results/rl_rerun/local_r3/n500/seed0/dphi_reward_goal_diagnostics_valid_goal_n4096.json`

### Interpretation

The D_phi reward checkpoints do not materially change goal usage. Observation
shuffle remains about `17x` larger than goal shuffle, and valid same-state goal
swaps are indistinguishable from frozen. This explains why the D_phi updates can
move full-bank local scalar metrics but fail to produce reliable learned-goal
closed-loop gains: the policy still mostly acts from current observation and
previous action, not from the future goal. The next productive branch should
return to representation/architecture or closed-loop intervention training,
not more one-segment scalar reward tuning on this goal-insensitive low level.

## 2026-06-26 - Fixed-seed 500-episode learned-interface base check

### Hypothesis

The offline goal-use gate currently promotes `ae256_film` and
`vae512_b1e6_film`, but earlier learned-interface deployment checks were
smaller than the recent 500-episode R3 windows. If one of these is a serious RL
base, it should preserve effect32-level learned-goal closed-loop success on the
same `seed_start=3500000` window.

### Commands

```bash
for candidate in ae256_film vae512_b1e6_film; do
  for source in learned oracle shuffled; do
    TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
      --config configs/pusht_incremental.yaml \
      --candidate "${candidate}" \
      --goal-source "${source}" \
      --episodes 500 \
      --eval-seed-start 3500000 \
      --force
  done
done

for source in learned oracle shuffled; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film \
    --goal-source "${source}" \
    --episodes 500 \
    --eval-seed-start 3500000 \
    --force
done
```

### Results

| candidate | goal source | success | final reward | max reward | high decisions | teacher MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | learned | 0.650 | 0.741 | 0.748 | 6.874 | 0.0996 |
| effect32_film | oracle | 0.694 | 0.777 | 0.782 | 6.660 | 0.0824 |
| effect32_film | shuffled | 0.312 | 0.443 | 0.476 | 8.598 | 0.2038 |
| ae256_film | learned | 0.544 | 0.657 | 0.671 | 7.296 | 0.1184 |
| ae256_film | oracle | 0.642 | 0.742 | 0.749 | 7.024 | 0.0751 |
| ae256_film | shuffled | 0.034 | 0.174 | 0.232 | 9.852 | 0.2802 |
| vae512_b1e6_film | learned | 0.438 | 0.580 | 0.595 | 7.948 | 0.1167 |
| vae512_b1e6_film | oracle | 0.532 | 0.662 | 0.671 | 7.542 | 0.0812 |
| vae512_b1e6_film | shuffled | 0.018 | 0.159 | 0.214 | 9.912 | 0.2867 |

Artifacts:

- `results/incremental/learned_interface/effect32_film/seed0/learned_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/effect32_film/seed0/oracle_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/effect32_film/seed0/shuffled_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/ae256_film/seed0/learned_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/ae256_film/seed0/oracle_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/ae256_film/seed0/shuffled_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/vae512_b1e6_film/seed0/learned_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/vae512_b1e6_film/seed0/oracle_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/vae512_b1e6_film/seed0/shuffled_hierarchy_eval_500_seed3500000.json`

### Interpretation

The strict offline goal-use gate identifies real goal dependence: AE/VAE FiLM
collapse almost completely under shuffled goals, while effect32 still gets
`0.312` shuffled success. However, that stronger dependence does not produce a
better learned-goal hierarchy. `effect32_film` remains the best base on this
fixed 500-episode window (`0.650` learned success), ahead of `ae256_film`
(`0.544`) and `vae512_b1e6_film` (`0.438`).

This reinforces the current selection rule: goal-use diagnostics are useful as
a rejection gate, but promotion requires closed-loop learned-goal deployment
quality. No archived candidate currently combines AE/VAE-style goal dependence
with effect32-level imitation quality, so the next representation branch should
explicitly optimize for both rather than selecting on goal sensitivity alone.

## 2026-06-26 - Delta/relation conditioning goal-use screen

### Hypothesis

The repository already had trained `delta` and `relation` conditioning variants
for AE256 and VAE512. Their 20-episode smokes were not obviously bad, so they
were a cheap way to check whether a non-FiLM conditioning mode can preserve
deployment quality while improving goal use.

### Commands

```bash
for candidate in ae256_delta ae256_relation vae512_b1e6_delta vae512_b1e6_relation; do
  TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
    --config configs/pusht_incremental.yaml \
    goal-diagnostics \
    --representation learned_interface \
    --candidate "${candidate}" \
    --n-demo 1800 \
    --seed 0 \
    --samples 5000 \
    --horizons 2,5,10 \
    --output "results/incremental/goal_diagnostics/n1800/seed0/${candidate}/diagnostics.json"
done

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  aggregate-goal-diagnostics \
  --input-glob 'results/incremental/goal_diagnostics/**/diagnostics.json' \
  --output results/incremental/goal_diagnostics/gate_report.json \
  --force
```

### Results

| candidate | conditioning | goal shuffle L2 | frame shuffle L2 | max horizon sensitivity L2 | gate status |
| --- | --- | ---: | ---: | ---: | --- |
| ae256_delta | delta | 0.0589 | 0.9547 | 0.0340 | reject low goal-use |
| ae256_relation | relation | 0.0707 | 0.7603 | 0.0287 | reject low goal-use |
| vae512_b1e6_delta | delta | 0.0599 | 0.9513 | 0.0355 | reject low goal-use |
| vae512_b1e6_relation | relation | 0.0620 | 0.8163 | 0.0270 | reject low goal-use |

After adding these rows, the aggregate gate report is:

```text
total: 19
offline_goal_use_pass: 3
reject_low_goal_use: 16
```

Artifacts:

- `results/incremental/goal_diagnostics/n1800/seed0/ae256_delta/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/ae256_relation/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae512_b1e6_delta/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae512_b1e6_relation/diagnostics.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Interpretation

The cheap delta/relation branch is closed. These variants can look acceptable in
tiny deployment smokes, but their low-level policies still barely react to valid
future goals. None reaches the strict `0.1` gate on either block-shuffle goal
action change or same-state horizon sensitivity. I skipped larger closed-loop
evals because the whole purpose of this screen was to avoid scaling candidates
that fail the low-level goal-use gate.

## 2026-06-26 - Remaining archived hierarchy goal-use sweep

### Hypothesis

Several trained hierarchy checkpoints still had no goal-diagnostics entry. A
complete archived sweep could reveal a hidden candidate that already has better
low-level goal usage, avoiding a new training branch.

### Code Change

Older `encoded_episodes.pt` artifacts store validation goals under
`validation[*].goals` instead of top-level `validation_goals`. I updated
`_load_encoded_validation_goals` to support both schemas and still fail clearly
if neither is present.

### Commands

```bash
for candidate in \
  ae256_control dae256_n005 dae512_w2048_n005 effect16 effect64 \
  jepa256_predonly_v1_c001 jepa256_r001_v1_c001 jepa256_r1_v10_c01 \
  vae256_b1e6 vae256_b1e7 vae512_w2048_b1e6 vae512_w2048_b1e7; do
  TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
    --config configs/pusht_incremental.yaml \
    goal-diagnostics \
    --representation learned_interface \
    --candidate "${candidate}" \
    --n-demo 1800 \
    --seed 0 \
    --samples 5000 \
    --horizons 2,5,10 \
    --output "results/incremental/goal_diagnostics/n1800/seed0/${candidate}/diagnostics.json" \
    --force
done

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  aggregate-goal-diagnostics \
  --input-glob 'results/incremental/goal_diagnostics/**/diagnostics.json' \
  --output results/incremental/goal_diagnostics/gate_report.json \
  --force
```

### Results

No newly screened candidate passed. The aggregate report now contains:

```text
total: 31
offline_goal_use_pass: 3
reject_low_goal_use: 28
```

Passing rows remain unchanged:

| candidate | N | conditioning | goal shuffle L2 | max horizon sensitivity L2 |
| --- | ---: | --- | ---: | ---: |
| effect32_film_gsens | 500 | film | 0.1149 | 0.0476 |
| ae256_film | 1800 | film | 0.2506 | 0.0937 |
| vae512_b1e6_film | 1800 | film | 0.2783 | 0.1266 |

Best newly screened rows:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | status |
| --- | ---: | ---: | --- |
| dae512_w2048_n005 | 0.0799 | 0.0310 | reject low goal-use |
| vae512_w2048_b1e7 | 0.0789 | 0.0317 | reject low goal-use |
| ae256_control | 0.0784 | 0.0277 | reject low goal-use |
| dae256_n005 | 0.0780 | 0.0272 | reject low goal-use |
| vae512_w2048_b1e6 | 0.0740 | 0.0308 | reject low goal-use |

Artifacts:

- `results/incremental/goal_diagnostics/n1800/seed0/ae256_control/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/dae256_n005/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/dae512_w2048_n005/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/effect16/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/effect64/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/jepa256_predonly_v1_c001/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/jepa256_r001_v1_c001/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/jepa256_r1_v10_c01/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae256_b1e6/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae256_b1e7/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae512_w2048_b1e6/diagnostics.json`
- `results/incremental/goal_diagnostics/n1800/seed0/vae512_w2048_b1e7/diagnostics.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/goal_diagnostics.py
```

### Interpretation

The archived candidate search is now mostly exhausted. No hidden VAE, DAE,
JEPA, or effect-code checkpoint passes the low-level goal-use gate. The only
passes are still the direct sensitivity-regularized effect32 policy, which
damaged deployment, and the AE/VAE FiLM policies, which use the goal but remain
weaker learned-goal bases than effect32. The next useful representation work
needs to train for both properties explicitly; there is not an existing
checkpoint to promote.

## 2026-06-26 - Effect32 FiLM low-frame-dropout check

### Hypothesis

The low-level policy may ignore goals because the current observation is an
easy shortcut for one-step BC. If so, randomly dropping the current-frame block
during low-level training should increase goal dependence while preserving the
same deployment interface. This is a more direct shortcut test than adding an
action-difference margin to shuffled goals.

### Code Change

Added `low_frame_dropout_prob` to learned-interface hierarchy training. It is
opt-in per candidate and only applies to low-level BC training samples; high
level training, validation, and deployment use the normal unmasked frame.

Candidate:

```yaml
effect32_film_frame_drop25:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: film
  low_frame_dropout_prob: 0.25
  policy_epochs: 40
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_frame_drop25 \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_frame_drop25 \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_frame_drop25/diagnostics.json \
  --force

for source in learned oracle; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film_frame_drop25 \
    --goal-source "${source}" \
    --episodes 200 \
    --eval-seed-start 3500000 \
    --force
done
```

### Results

Training validation:

| metric | value |
| --- | ---: |
| best epoch | 40 |
| low frame dropout prob | 0.25 |
| oracle action MAE | 0.0427 |
| predicted action MAE | 0.0432 |
| prediction-induced action L2 | 0.0194 |

Goal-use/deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.0622 | 0.0368 | 0.645 | 0.645 |
| effect32_film_frame_drop25 | 0.1121 | 0.0615 | 0.490 | 0.465 |

The 20-episode smoke was also weak: learned `0.35`, oracle `0.45`.

After adding this candidate, the aggregate gate report is:

```text
total: 32
offline_goal_use_pass: 4
reject_low_goal_use: 28
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_frame_drop25/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_frame_drop25/seed0/hierarchy_metrics.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_frame_drop25/diagnostics.json`
- `results/incremental/learned_interface/effect32_film_frame_drop25/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_frame_drop25/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py
```

### Interpretation

Frame dropout confirms the shortcut diagnosis: weakening the current-observation
path raises goal-shuffle action change enough to pass the offline gate
(`0.0622 -> 0.1121`). But it also hurts closed-loop imitation badly
(`0.645 -> 0.490` learned success on the same 200-episode seed window). This is
not a promotable base, and it suggests the next successful approach needs a
counterfactual or deployment-aligned signal that teaches when the goal matters,
not just a generic reduction in observation reliance.

## 2026-06-26 - Effect32 FiLM scene-frame-dropout check

### Hypothesis

Full-frame dropout removes both scene/object features and proprio/current robot
state. A more targeted shortcut test is to drop only the scene prefix while
keeping the proprio tail. If the deployment regression came mostly from losing
local robot state, scene-only dropout should preserve more imitation quality
while retaining the goal-use gain.

### Code Change

Extended the low-frame-dropout hook with
`low_frame_dropout_keep_tail_dim`. During dropout samples, the dataset zeros
only `frame[:, :-keep_tail_dim]` and preserves the tail. The candidate uses
`keep_tail_dim=21`, matching the configured Push-T proprio dimension.

Candidate:

```yaml
effect32_film_scene_drop25:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: film
  low_frame_dropout_prob: 0.25
  low_frame_dropout_keep_tail_dim: 21
  policy_epochs: 40
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_scene_drop25 \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_scene_drop25 \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_scene_drop25/diagnostics.json \
  --force

for source in learned oracle; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film_scene_drop25 \
    --goal-source "${source}" \
    --episodes 200 \
    --eval-seed-start 3500000 \
    --force
done
```

### Results

Training validation:

| metric | value |
| --- | ---: |
| best epoch | 36 |
| low frame dropout prob | 0.25 |
| keep tail dim | 21 |
| oracle action MAE | 0.0437 |
| predicted action MAE | 0.0441 |
| prediction-induced action L2 | 0.0191 |

Goal-use/deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.0622 | 0.0368 | 0.645 | 0.645 |
| effect32_film_frame_drop25 | 0.1121 | 0.0615 | 0.490 | 0.465 |
| effect32_film_scene_drop25 | 0.1141 | 0.0616 | 0.510 | 0.560 |

The 20-episode smoke was learned `0.70`, oracle `0.55`; the longer fixed-seed
check showed that this was optimistic.

After adding this candidate, the aggregate gate report is:

```text
total: 33
offline_goal_use_pass: 5
reject_low_goal_use: 28
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_scene_drop25/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_scene_drop25/seed0/hierarchy_metrics.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_scene_drop25/diagnostics.json`
- `results/incremental/learned_interface/effect32_film_scene_drop25/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_scene_drop25/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py
```

### Interpretation

Scene-only dropout is directionally better than full-frame dropout, especially
for oracle goals (`0.560` vs `0.465`), but it still fails the central
requirement: learned-goal deployment remains far below baseline effect32 FiLM
(`0.510` vs `0.645`). The observation shortcut diagnosis is now stronger, but
generic dropout is not enough. The next attempt should use counterfactual
same-state goals or closed-loop/intervention data so the policy learns a useful
goal-conditioned correction rather than a weaker imitation policy.

## 2026-06-26 - Effect32 FiLM auxiliary scene-dropout check

### Hypothesis

The primary scene-dropout objective weakened deployment because the low policy
was trained on corrupted inputs as its main BC distribution. A cleaner
preservation test is to keep normal clean-input BC as the primary loss and add
scene-dropout BC only as an auxiliary loss. This should preserve effect32-style
imitation quality better while still discouraging the observation shortcut.

### Code Change

Added an opt-in low-frame-dropout auxiliary loss:

```text
low_frame_dropout_aux_prob
low_frame_dropout_aux_keep_tail_dim
low_frame_dropout_aux_weight
```

For each low-level batch, the training loop always optimizes clean BC. If the
auxiliary weight is positive, it also masks a copy of the frame block with the
configured probability and adds weighted BC on that masked copy.

Candidate:

```yaml
effect32_film_scene_drop_aux05:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: film
  low_frame_dropout_aux_prob: 0.25
  low_frame_dropout_aux_keep_tail_dim: 21
  low_frame_dropout_aux_weight: 0.5
  policy_epochs: 40
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_scene_drop_aux05 \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_scene_drop_aux05 \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_scene_drop_aux05/diagnostics.json \
  --force

for source in learned oracle; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film_scene_drop_aux05 \
    --goal-source "${source}" \
    --episodes 200 \
    --eval-seed-start 3500000 \
    --force
done
```

### Results

Training validation:

| metric | value |
| --- | ---: |
| best epoch | 40 |
| auxiliary dropout prob | 0.25 |
| auxiliary keep tail dim | 21 |
| auxiliary loss weight | 0.5 |
| oracle action MAE | 0.0410 |
| predicted action MAE | 0.0415 |
| prediction-induced action L2 | 0.0143 |

Goal-use/deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.0622 | 0.0368 | 0.645 | 0.645 |
| effect32_film_scene_drop25 | 0.1141 | 0.0616 | 0.510 | 0.560 |
| effect32_film_scene_drop_aux05 | 0.0864 | 0.0481 | 0.500 | 0.570 |

After adding this candidate, the aggregate gate report is:

```text
total: 34
offline_goal_use_pass: 5
reject_low_goal_use: 29
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_scene_drop_aux05/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_scene_drop_aux05/seed0/hierarchy_metrics.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_scene_drop_aux05/diagnostics.json`
- `results/incremental/learned_interface/effect32_film_scene_drop_aux05/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_scene_drop_aux05/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py
```

### Interpretation

The auxiliary version preserves validation MAE better than primary scene
dropout, but it does not solve the deployment tradeoff. Goal-shuffle action L2
rises only to `0.0864`, below the strict `0.1` gate, and learned-goal success is
still only `0.500`. The result narrows the conclusion: the shortcut cannot be
fixed by generic masked reconstruction-style auxiliary BC. The missing signal is
not just "act well when observation is partially hidden"; it must teach which
goal-conditioned correction improves the closed-loop outcome from the same
state.

## 2026-06-26 - Task-hard R3 goal-use diagnostic

### Hypothesis

The task-hard local R3 checkpoint gave the strongest targeted local task-reward
signal among the recent real-compatible R3 variants. If that target regime is
meaningfully different from the D_phi progress/paired variants, it might also
increase the low-level policy's dependence on the future goal.

### Commands

```bash
TQDM_DISABLE=1 uv run python scripts/rl_rerun_condition_block_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 4096 \
  --batch-size 512 \
  --horizon 10 \
  --policy taskhard_bc03=artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/goal_diagnostics_condition_block_n4096.json

TQDM_DISABLE=1 uv run python scripts/rl_rerun_valid_goal_sensitivity.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --n-demo 500 \
  --seed 0 \
  --samples 4096 \
  --batch-size 512 \
  --horizons 2,10 \
  --policy taskhard_bc03=artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt \
  --output results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/goal_diagnostics_valid_goal_n4096.json
```

### Results

Condition-block shuffle action L2:

| policy | observation | goal | previous action | remaining |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.8357 | 0.0444 | 0.0726 | 0.0000 |
| task-hard bc0.3 | 0.8363 | 0.0450 | 0.0729 | 0.0000 |

Valid same-state future-goal action L2:

| policy | k=2 vs k=10 action L2 | action L2 / goal L2 |
| --- | ---: | ---: |
| frozen | 0.020705 | 0.000734 |
| task-hard bc0.3 | 0.020706 | 0.000734 |

The mean latent goal separation for the valid-goal swap was `25.32`.

Artifacts:

- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/goal_diagnostics_condition_block_n4096.json`
- `results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/goal_diagnostics_valid_goal_n4096.json`

### Interpretation

The task-hard objective does not fix goal use. Its action response to goal
shuffling is still about `5%` of the observation-shuffle response, and valid
future-goal swaps are numerically indistinguishable from frozen. This closes the
strongest task-hard local objective as a goal-identifiability fix: it can shape
small terminal task-reward deltas on selected local starts, but it does not
teach a materially more goal-conditioned low-level correction.

## 2026-06-26 - D_phi-ranked nearest goal projection smoke

### Hypothesis

Raw nearest-training-goal projection keeps high-level goals on the replay
manifold, but it chooses prototypes by latent L2 to the predicted goal. A more
control-aligned projection is to use raw L2 only to get a small candidate set,
then choose the candidate with lowest learned reachability distance
`D_phi(current, candidate_goal)`.

### Implementation

Added a serial eval-only projection mode:

```text
--goal-projection nearest_train_dphi
--goal-projection-topk 32
```

The evaluator loads the same normalized training-goal prototype bank used by
`nearest_train`, selects the top-k raw-L2 nearest prototypes to the high-level
prediction, then ranks those candidates with the configured reachability
checkpoint.

### Commands

R3 residual, two 20-episode prefixes:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10 \
  --episodes 20 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/latest.pt \
  --goal-projection nearest_train_dphi \
  --goal-projection-topk 32 \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10 \
  --episodes 20 \
  --seed-start 3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/latest.pt \
  --goal-projection nearest_train_dphi \
  --goal-projection-topk 32 \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Frozen, matched two 20-episode prefixes:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_dphi_proto_serial20_seed3500000 \
  --episodes 20 \
  --seed-start 3500000 \
  --goal-projection nearest_train_dphi \
  --goal-projection-topk 32 \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml eval-serial \
  --n-demo 500 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_dphi_proto_serial20_seed3600000 \
  --episodes 20 \
  --seed-start 3600000 \
  --goal-projection nearest_train_dphi \
  --goal-projection-topk 32 \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

I also attempted the 100-episode R3 `seed_start=3500000` run first, but
interrupted it after several minutes and switched to 20-episode matched smokes.

### Results

The table compares the new D_phi-ranked projection against the first 20 episodes
of existing no-projection and raw nearest-train 100-episode serial files on the
same two seed windows.

| policy | projection | episodes | success | final reward | max reward |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen | none | 40 | 0.700 | 0.6791 | 0.7852 |
| frozen | nearest_train | 40 | 0.600 | 0.6292 | 0.7136 |
| frozen | nearest_train_dphi | 40 | 0.675 | 0.7353 | 0.7679 |
| R3 | none | 40 | 0.625 | 0.6301 | 0.7383 |
| R3 | nearest_train | 40 | 0.600 | 0.6018 | 0.7150 |
| R3 | nearest_train_dphi | 40 | 0.650 | 0.6106 | 0.7425 |

Projection diagnostics:

| policy | projection L2 | projection D_phi |
| --- | ---: | ---: |
| frozen nearest_train_dphi | 2.1398 | 0.5536 |
| R3 nearest_train_dphi | 2.1101 | 0.5780 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_dphi_proto_serial20_seed3500000/serial_eval_20_seed3500000_nearest_train_dphi.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_dphi_proto_serial20_seed3600000/serial_eval_20_seed3600000_nearest_train_dphi.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/serial_eval_20_seed3500000_nearest_train_dphi.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/serial_eval_20_seed3600000_nearest_train_dphi.json`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/low_level_rl.py src/hcl_poc/cli.py
python3 -m json.tool <each new serial eval JSON>
```

### Interpretation

`D_phi` ranking is better than raw nearest-goal projection, especially for
avoiding the reward/success loss from naive prototype snapping. It is not a
promotion path yet: over the matched 40-episode smoke it does not beat the
no-projection frozen success and only gives a small mixed R3 signal. This
supports the narrower conclusion that reachability-aware goal projection is a
useful high-level diagnostic, but high-level on-manifold projection alone does
not solve the low-level/R3 reliability problem.

## 2026-06-26 - Baseline-initialized goal-sensitivity fine-tune

### Hypothesis

The previous goal-sensitivity margin loss raised offline goal use but damaged
deployment because the low-level policy was trained from scratch under the new
objective. This test starts from the working `effect32_film` low policy and
applies a short, low-learning-rate goal-sensitivity fine-tune. The goal is to
preserve imitation quality while nudging the low-level toward stronger goal
dependence.

I first tried an `effect32_delta` conditioning alias, but that branch is not
deployable for effect-code latents: evaluation requires a unary current effect,
which is undefined for the effect representation. I removed the config entry and
kept the direction focused on a deployable FiLM checkpoint.

### Code Change

`train_learned_interface_hierarchy` now supports:

```text
low_init_candidate
```

When set, the trainer loads the source checkpoint's low-level weights after
constructing a compatible low policy. It fails clearly if frame/goal/hidden
dimensions or conditioning mode differ.

Candidate:

```yaml
effect32_film_gsens_ft:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32
  conditioning: film
  low_init_candidate: effect32_film
  goal_sensitivity_weight: 0.05
  goal_sensitivity_margin: 0.2
  policy_lr: 1.0e-5
  policy_epochs: 10
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_gsens_ft \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft/diagnostics.json \
  --force

for source in learned oracle; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film_gsens_ft \
    --goal-source "${source}" \
    --episodes 200 \
    --eval-seed-start 3500000 \
    --force
done
```

### Results

Training validation:

| metric | value |
| --- | ---: |
| best epoch | 10 |
| low init candidate | `effect32_film` |
| oracle action MAE | 0.0362 |
| predicted action MAE | 0.0367 |
| prediction-induced action L2 | 0.0131 |

Goal-use/deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.0622 | 0.0368 | 0.645 | 0.645 |
| effect32_film_gsens | 0.1149 | 0.0476 | 0.500 | 0.515 |
| effect32_film_gsens_ft | 0.0919 | 0.0399 | 0.550 | 0.675 |

After adding this candidate, the aggregate gate report is:

```text
total: 35
offline_goal_use_pass: 5
reject_low_goal_use: 30
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft/seed0/hierarchy_metrics.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft/diagnostics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py
```

### Interpretation

Baseline initialization helps compared with training the sensitivity-regularized
policy from scratch. It preserves offline action MAE and improves oracle-goal
closed-loop success above the baseline on the fixed 200-episode window
(`0.645 -> 0.675`). But it still misses the strict goal-use gate
(`0.0919 < 0.1`) and learned-goal deployment remains well below the baseline
(`0.645 -> 0.550`). This is a useful diagnostic: stronger low-level goal use
can help when the goal is good, but the current learned high-level goals do not
support the same gain. The next representation work should treat high-level
goal quality and low-level goal sensitivity as a coupled problem, not optimize
the low-level margin alone.

## 2026-06-26 - D_phi projection on goal-sensitivity fine-tune

### Hypothesis

The baseline-initialized goal-sensitivity fine-tune improved oracle-goal
closed-loop success but hurt learned-goal success. If the issue is partly
high-level goal quality, reachability-ranked goal projection might recover some
of the learned-goal deployment loss for the more goal-sensitive low-level.

### Commands

Two matched 20-episode serial windows, with and without `nearest_train_dphi`
projection:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft \
  --seed 0 \
  --run-name hcl_next_effect32_film_gsens_ft_serial20_seed3500000 \
  --episodes 20 \
  --seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft \
  --seed 0 \
  --run-name hcl_next_effect32_film_gsens_ft_serial20_seed3500000 \
  --episodes 20 \
  --seed-start 3500000 \
  --goal-projection nearest_train_dphi \
  --goal-projection-topk 32 \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

The same pair was repeated for `seed-start=3600000`.

### Results

| seed start | projection | success | final reward | max reward |
| ---: | --- | ---: | ---: | ---: |
| 3500000 | none | 0.550 | 0.5385 | 0.6822 |
| 3500000 | nearest_train_dphi | 0.750 | 0.7726 | 0.8167 |
| 3600000 | none | 0.700 | 0.6973 | 0.7982 |
| 3600000 | nearest_train_dphi | 0.550 | 0.5854 | 0.6910 |
| aggregate | none | 0.625 | 0.6179 | 0.7402 |
| aggregate | nearest_train_dphi | 0.650 | 0.6790 | 0.7538 |

Projection diagnostics:

| projection | projection L2 | projection D_phi |
| --- | ---: | ---: |
| nearest_train_dphi aggregate | 2.0310 | 0.5787 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film_gsens_ft/seed0/hcl_next_effect32_film_gsens_ft_serial20_seed3500000/serial_eval_20_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft/seed0/hcl_next_effect32_film_gsens_ft_serial20_seed3500000/serial_eval_20_seed3500000_nearest_train_dphi.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft/seed0/hcl_next_effect32_film_gsens_ft_serial20_seed3600000/serial_eval_20_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft/seed0/hcl_next_effect32_film_gsens_ft_serial20_seed3600000/serial_eval_20_seed3600000_nearest_train_dphi.json`

### Interpretation

The coupled high/low diagnostic is directionally useful but not promotable.
`D_phi` projection improves the aggregate `gsens_ft` serial smoke
(`0.625 -> 0.650` success and better reward), but the effect flips sign between
the two 20-episode windows. It also still fails to beat the original
no-projection `effect32_film` smoke on the same prefixes (`0.700` success).
This supports the same broader conclusion: reachability-aware high-level goal
repair and low-level goal sensitivity can interact, but the current pieces do
not yet combine into a robust learned-goal hierarchy.

## 2026-06-26 - Action-aware high-level fine-tune

### Hypothesis

`effect32_film_gsens_ft` improved oracle-goal deployment but hurt learned-goal
deployment. This suggests that the low-level change can be useful when goals
are good, but the reused high-level goal predictor is not aligned with the more
goal-sensitive low-level. A direct coupled test is to freeze the
`effect32_film_gsens_ft` low-level and fine-tune only the high-level so that its
predicted goals make the frozen low-level match demonstration actions.

### Code Change

`train_learned_interface_hierarchy` now supports:

```text
high_init_candidate
freeze_low_policy
high_goal_mse_weight
high_action_loss_weight
```

The trainer samples high-level replans plus low-level offsets from the same
held-goal segment. When `high_action_loss_weight > 0`, it substitutes the
predicted high-level goal into the low-level condition and backpropagates action
MSE through the frozen low-level into the high-level model. The normal high
goal-MSE target remains active.

Candidate:

```yaml
effect32_film_gsens_ft_highact:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  freeze_low_policy: true
  high_action_loss_weight: 100.0
  policy_lr: 1.0e-5
  policy_epochs: 10
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact \
  --seed 0 \
  --force

for source in learned oracle; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film_gsens_ft_highact \
    --goal-source "${source}" \
    --episodes 200 \
    --eval-seed-start 3500000 \
    --force
done

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_gsens_ft_highact \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact/diagnostics.json \
  --force
```

### Results

Training validation:

| metric | value |
| --- | ---: |
| best epoch | 7 |
| high init candidate | `effect32` |
| low init candidate | `effect32_film_gsens_ft` |
| frozen low policy | true |
| high action loss weight | 100 |
| normalized goal L2 | 2.5467 |
| oracle action MAE | 0.0362 |
| predicted action MAE | 0.0365 |
| prediction-induced action L2 | 0.0128 |

Fixed 200-episode deployment check:

| candidate | learned success | learned max reward | oracle success | oracle max reward | oracle goal L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.7420 | 0.645 | 0.7464 | 3.391 |
| effect32_film_gsens_ft | 0.550 | 0.6793 | 0.675 | 0.7733 | 3.292 |
| effect32_film_gsens_ft_highact | 0.595 | 0.7130 | 0.675 | 0.7733 | 3.264 |

Goal-use gate:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | gate status |
| --- | ---: | ---: | --- |
| effect32_film_gsens_ft_highact | 0.0919 | 0.0399 | reject low goal-use |

After adding this candidate, the aggregate gate report is:

```text
total: 36
offline_goal_use_pass: 5
reject_low_goal_use: 31
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact/diagnostics.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py
```

### Interpretation

The coupled high-level action loss works mechanically and moves in the intended
direction. It reduces validation goal error, improves learned-goal success over
the low-level-only fine-tune (`0.550 -> 0.595`), and preserves the oracle-goal
gain (`0.675`). But it still remains below the original `effect32_film`
learned-goal baseline (`0.645`). This narrows the next step: high-level
action-aware tuning is useful, but this mild 10-epoch high-only version is not
strong enough. Future coupled work should either tune high and low jointly in a
closed-loop/intervention distribution or use a stronger high-level objective
than demonstration action MSE through a frozen low-level.

## 2026-06-26 - Stronger action-aware high-level fine-tune

### Hypothesis

The first high-action run recovered part of the learned-goal loss but remained
below baseline. Its training curve was still slowly improving, and the
high-action loss was small relative to the normal high-level goal MSE. This run
tests a stronger high-level-only objective while keeping the same frozen
`effect32_film_gsens_ft` low-level.

### Candidate

```yaml
effect32_film_gsens_ft_highact_strong:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact_strong
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  freeze_low_policy: true
  high_goal_mse_weight: 0.3
  high_action_loss_weight: 300.0
  policy_lr: 1.0e-5
  policy_epochs: 20
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-run \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --force

for source in learned oracle; do
  TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
    --config configs/pusht_incremental.yaml \
    --candidate effect32_film_gsens_ft_highact_strong \
    --goal-source "${source}" \
    --episodes 200 \
    --eval-seed-start 3500000 \
    --force
done

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --representation learned_interface \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --samples 5000 \
  --horizons 2,5,10 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact_strong/diagnostics.json \
  --force
```

### Results

Training validation:

| metric | value |
| --- | ---: |
| best epoch | 20 |
| high goal MSE weight | 0.3 |
| high action loss weight | 300 |
| normalized goal L2 | 2.5489 |
| oracle action MAE | 0.0362 |
| predicted action MAE | 0.0365 |
| prediction-induced action L2 | 0.0128 |

Fixed 200-episode deployment check:

| candidate | learned success | learned max reward | oracle success | oracle max reward | oracle goal L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.7420 | 0.645 | 0.7464 | 3.391 |
| effect32_film_gsens_ft_highact | 0.595 | 0.7130 | 0.675 | 0.7733 | 3.264 |
| effect32_film_gsens_ft_highact_strong | 0.640 | 0.7403 | 0.675 | 0.7733 | 3.255 |

Fixed 500-episode learned-goal check:

| candidate | learned success | final reward | max reward | teacher action MAE |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.650 | 0.7410 | 0.7484 | 0.0996 |
| ae256_film | 0.544 | 0.6572 | 0.6707 | 0.1184 |
| vae512_b1e6_film | 0.438 | 0.5798 | 0.5952 | 0.1167 |
| effect32_film_gsens_ft_highact_strong | 0.652 | 0.7455 | 0.7523 | 0.0895 |

Goal-use gate:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | gate status |
| --- | ---: | ---: | --- |
| effect32_film_gsens_ft_highact_strong | 0.0919 | 0.0399 | reject low goal-use |

After adding this candidate, the aggregate gate report is:

```text
total: 37
offline_goal_use_pass: 5
reject_low_goal_use: 32
```

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_500_seed3500000.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact_strong/diagnostics.json`
- `results/incremental/goal_diagnostics/gate_report.json`
- `results/incremental/goal_diagnostics/gate_report.md`

### Interpretation

The stronger high-action objective is the best coupled learned-interface result
so far. It nearly matches the original baseline on the 200-episode learned
check and slightly beats the baseline on the fixed 500-episode learned check
(`0.652` versus `0.650`) while keeping the `effect32_film_gsens_ft` oracle-goal
gain. The margin is too small to call robust, and the low-level still fails the
strict offline goal-use gate because the low policy itself is unchanged.

Still, this is a meaningful lead: action-aware high-level tuning can make a more
goal-sensitive low-level deployable again. The next validation should test this
candidate on a fresh 500-episode learned-goal window, and if it survives, the
next implementation step should move from high-only tuning to a joint high/low
coupled objective or closed-loop intervention training.

## 2026-06-26 - Fresh-window validation for strong action-aware high-level tuning

### Hypothesis

The fixed `seed_start=3500000` 500-episode check for
`effect32_film_gsens_ft_highact_strong` was only slightly above the original
`effect32_film` baseline. A fresh 500-episode learned-goal window should tell
whether that small gain is a one-window artifact or a reproducible lead.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3600000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3600000 \
  --force
```

### Results

Fresh 500-episode learned-goal window:

| candidate | success | final reward | max reward | teacher action MAE |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.666 | 0.7524 | 0.7618 | 0.0965 |
| effect32_film_gsens_ft_highact_strong | 0.672 | 0.7620 | 0.7674 | 0.0834 |

Two-window 1000-episode aggregate over `seed_start=3500000` and `3600000`:

| candidate | success | final reward | max reward | teacher action MAE |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.658 | 0.7467 | 0.7551 | 0.0981 |
| effect32_film_gsens_ft_highact_strong | 0.662 | 0.7538 | 0.7598 | 0.0865 |

Artifacts:

- `results/incremental/learned_interface/effect32_film/seed0/learned_hierarchy_eval_500_seed3600000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_500_seed3600000.json`

### Interpretation

The strong action-aware high-level candidate survives the first fresh-window
check. The margin remains small (`+0.004` success over 1000 episodes), but the
reward and teacher-action metrics also move in the right direction. This is now
the strongest real-compatible learned-interface lead in the current branch:
action-aware high-level tuning can recover deployment quality for the more
goal-sensitive low-level while slightly improving over the original baseline.

This is still not a final policy claim. The next useful step is either a larger
final-style validation window or an implementation step toward joint high/low
coupled training, using this result as the justification.

## 2026-06-26 - 1000-episode validation for strong action-aware high-level tuning

### Hypothesis

The strong action-aware high-level candidate was positive on two 500-episode
learned-goal windows, but the aggregate margin was still small. A fresh
1000-episode same-window comparison should tell whether the lead survives a
larger final-style validation.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film \
  --goal-source learned \
  --episodes 1000 \
  --eval-seed-start 3700000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 1000 \
  --eval-seed-start 3700000 \
  --force
```

### Results

Fresh 1000-episode learned-goal window:

| candidate | success | final reward | max reward | teacher action MAE |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.635 | 0.7306 | 0.7399 | 0.0978 |
| effect32_film_gsens_ft_highact_strong | 0.645 | 0.7386 | 0.7463 | 0.0902 |

Aggregate over `seed_start=3500000`, `3600000`, and `3700000`:

| candidate | episodes | success | final reward | max reward | teacher action MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 2000 | 0.6465 | 0.7386 | 0.7475 | 0.0980 |
| effect32_film_gsens_ft_highact_strong | 2000 | 0.6535 | 0.7462 | 0.7530 | 0.0883 |

Artifacts:

- `results/incremental/learned_interface/effect32_film/seed0/learned_hierarchy_eval_1000_seed3700000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_1000_seed3700000.json`

### Interpretation

The strong action-aware high-level candidate now has a replicated positive
learned-goal signal over 2000 fixed-seed episodes. The success margin is still
modest (`+0.007` absolute), but all task-quality proxies move in the same
direction: final reward, max reward, and teacher-action MAE. This is the first
real-compatible learned-interface candidate in the current branch that both
retains the improved oracle-goal low-level behavior and beats the original
effect32 FiLM learned-goal baseline on larger validation.

The next useful work should either validate this candidate against the R3/local
RL path or turn this high-only objective into a joint high/low coupled training
recipe. It is now strong enough to be a candidate base for follow-up RL, but the
margin is still too small to call the overall RL proof-of-concept solved.

## 2026-06-26 - Low-level-RL serial compatibility for strong action-aware candidate

### Hypothesis

The strong action-aware high-level candidate is the best learned-interface lead
so far, but the next plan-aligned question is whether it is compatible with the
`low-level-rl` serial evaluator and whether the earlier R3/projection machinery
still looks useful around this candidate.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_frozen_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_frozen_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --goal-projection nearest_train_dphi \
  --goal-projection-topk 32 \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Results

Same serial 100-episode window, `seed_start=3500000`:

| policy | projection | success | final reward | max reward | segment goal reach | action saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original effect32_film frozen | none | 0.600 | 0.6237 | 0.7085 | 0.719 | 0.0325 |
| old effect32_film R3 checkpoint | none | 0.670 | 0.6416 | 0.7618 | 0.730 | 0.0288 |
| effect32_film_gsens_ft_highact_strong frozen | none | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.0458 |
| effect32_film_gsens_ft_highact_strong frozen | nearest_train_dphi | 0.610 | 0.6438 | 0.7213 | 0.673 | 0.0428 |

Paired success counts:

| comparison | wins | losses | net |
| --- | ---: | ---: | ---: |
| highact strong frozen vs original frozen | 17 | 10 | +7 |
| highact strong frozen vs old R3 checkpoint | 13 | 13 | 0 |
| nearest_train_dphi vs no projection for highact strong | 8 | 14 | -6 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3500000/serial_eval_100_seed3500000_nearest_train_dphi.json`

### Interpretation

The high-action candidate is compatible with the local-RL serial path and
matches the old R3 checkpoint's task success on the same fixed reset bank
without applying any learned residual. It also has a much better final reward
than both original effect32_film frozen and the old R3 checkpoint on this
window.

Reusing the original effect32_film D_phi projection hurts this candidate. The
projection result is worse in success, final reward, max reward, segment goal
reach, and paired wins/losses against the unprojected high-action candidate.

This makes the next RL step narrower: use
`effect32_film_gsens_ft_highact_strong` as the base candidate for follow-up
local-RL experiments, but do not add the old nearest-neighbor D_phi goal
projection by default.

## 2026-06-26 - R3 smoke on strong action-aware candidate

### Hypothesis

The high-action candidate already beats the original frozen effect32_film base
in learned-interface and serial checks. If the old terminal-only D_phi R3 recipe
was limited by a weak base candidate, applying the same R3 update to
`effect32_film_gsens_ft_highact_strong` should improve serial closed-loop
deployment more reliably.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10 \
  --steps 40960 \
  --num-envs 4096 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_frozen_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --force
```

### Training Result

Final R3 training row:

| metric | value |
| --- | ---: |
| global step | 40960 |
| mean reward | -0.04065 |
| mean latent distance | 0.6136 |
| mean terminal distance | 0.4065 |
| mean direct delta L2 | 0.2645 |
| BC loss | 0.0000515 |
| action saturation rate | 0.4028 |

The local terminal-distance proxy improved substantially relative to the old
effect32 R3 run (`0.4065` here vs `0.5757` in
`hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10`), but the action
distribution moved much more during training.

### Serial Results

Same candidate, matched 100-episode serial windows:

| seed start | policy | success | final reward | max reward | segment goal reach | action saturation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3500000 | frozen highact strong | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.0458 |
| 3500000 | R3 highact strong | 0.700 | 0.6688 | 0.7874 | 0.755 | 0.0255 |
| 3600000 | frozen highact strong | 0.770 | 0.7199 | 0.8385 | 0.791 | 0.0400 |
| 3600000 | R3 highact strong | 0.610 | 0.6386 | 0.7273 | 0.744 | 0.0223 |

Two-window aggregate:

| policy | episodes | success | final reward | max reward | segment goal reach | action saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen highact strong | 200 | 0.720 | 0.7146 | 0.7976 | 0.7565 | 0.0429 |
| R3 highact strong | 200 | 0.655 | 0.6537 | 0.7574 | 0.7495 | 0.0239 |

Paired R3-vs-frozen success counts:

| seed start | wins | losses | net |
| ---: | ---: | ---: | ---: |
| 3500000 | 13 | 10 | +3 |
| 3600000 | 7 | 23 | -16 |
| aggregate | 20 | 33 | -13 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3600000/serial_eval_100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_4096_terminal_smoke_40k_bc10_serial100_seed3600000/serial_eval_100_seed3600000.json`

### Interpretation

This is a negative R3 result. The first serial bank looked mildly positive, but
the fresh bank was strongly negative and the two-window aggregate is worse than
the frozen high-action candidate across success, final reward, max reward, and
paired wins/losses.

The important diagnostic is the mismatch between local and deployment metrics:
the R3 objective found a much lower terminal D_phi distance during training, but
that local proxy did not preserve closed-loop robustness. This is the same
failure mode seen in earlier R3 branches, now reproduced on the stronger
high-action base.

Do not promote this R3 checkpoint. The current best deployable candidate remains
the frozen `effect32_film_gsens_ft_highact_strong` learned-interface policy.
The next useful implementation direction is deployment-coupled high/low
training, not another scalar terminal-D_phi R3 residual.

## 2026-06-26 - Joint high/low action-through-low fine-tune

### Hypothesis

The high-only action-through-low candidate recovered learned-goal deployment for
the more goal-sensitive low level, but it froze the low policy. A small joint
variant may improve further if the high level keeps the action-through-low loss
while the low level continues a gentle BC plus goal-sensitivity update from the
same initialized low policy.

### Implementation

Changed `train_learned_interface_hierarchy` so `high_action_loss_weight` no
longer requires `freeze_low_policy`. When the low policy is trainable, the
trainer temporarily sets the low-policy parameters to `requires_grad=False`
inside the high-level action-through-low loss. This preserves gradients from the
low-policy output back to the predicted high-level goal, but prevents that loss
from updating low-policy parameters. The low policy is then updated separately
by its BC, frame-dropout auxiliary, and goal-sensitivity losses.

Added candidate:

```yaml
effect32_film_gsens_ft_highact_joint:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact_joint
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  high_goal_mse_weight: 0.3
  high_action_loss_weight: 300.0
  goal_sensitivity_weight: 0.02
  goal_sensitivity_margin: 0.2
  policy_lr: 1.0e-5
  policy_epochs: 20
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_joint \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_joint \
  --goal-source learned \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_joint \
  --goal-source oracle \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_joint \
  --seed 0 \
  --samples 4096 \
  --horizons 2,5,10,20 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact_joint/diagnostics.json \
  --force
```

### Training Metrics

Best epoch: `18`.

| metric | value |
| --- | ---: |
| validation normalized goal L2 | 2.5281 |
| validation oracle action MAE | 0.0361 |
| validation predicted action MAE | 0.0363 |
| validation prediction-induced action L2 | 0.0110 |
| last low train MSE | 0.000731 |
| last low goal-sensitivity loss | 0.00585 |

### Closed-Loop Results

Matched 200-episode window, `seed_start=3500000`:

| candidate | learned success | learned final | learned max | learned teacher MAE | oracle success | oracle final | oracle max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.7347 | 0.7420 | 0.1073 | 0.645 | 0.7399 | 0.7464 |
| effect32_film_gsens_ft | 0.550 | 0.6662 | 0.6793 | 0.1026 | 0.675 | 0.7657 | 0.7733 |
| effect32_film_gsens_ft_highact_strong | 0.640 | 0.7316 | 0.7403 | 0.0958 | 0.675 | 0.7657 | 0.7733 |
| effect32_film_gsens_ft_highact_joint | 0.635 | 0.7260 | 0.7358 | 0.1034 | 0.615 | 0.7173 | 0.7270 |

Goal-use diagnostic:

| candidate | frame shuffle L2 | goal shuffle L2 | max goal sensitivity L2 |
| --- | ---: | ---: | ---: |
| highact strong | 0.9386 | 0.0919 | 0.0399 |
| highact joint | 0.9381 | 0.0812 | 0.0598 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_joint/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_joint/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_joint/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_joint/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact_joint/diagnostics.json`

### Interpretation

Reject this joint recipe. It does not improve learned-goal deployment over
`highact_strong`, and it damages the oracle-goal low-level result substantially
(`0.675 -> 0.615`). The goal diagnostic also does not show a cleaner
goal-conditioned low policy: goal-shuffle L2 drops from `0.0919` to `0.0812`,
still below the gate threshold.

The failure is useful: simply unfreezing the low policy under the same
BC+sensitivity objective is too weak a constraint. The current best candidate
remains frozen `effect32_film_gsens_ft_highact_strong`. A better next joint
variant should anchor the low policy's oracle-goal action/closed-loop behavior
while tuning high-level compatibility, instead of letting low-level BC drift
erase the oracle-goal improvement.

## 2026-06-26 - Anchored joint high/low action-through-low fine-tune

### Hypothesis

The naive joint high/low fine-tune damaged oracle-goal low-level behavior. A
simple low-policy anchor may preserve the useful initialized low-level behavior
while still allowing a small joint update under the action-through-low high-level
loss and low-level goal-sensitivity objective.

### Implementation

Added optional `low_anchor_loss_weight` to `train_learned_interface_hierarchy`.
When the weight is positive, the trainer keeps a frozen copy of the initialized
low policy and adds an MSE penalty between the trainable low policy and the
anchor low policy on the same low-level training inputs. This is deliberately a
small offline action anchor; it does not add a new rollout evaluator.

Added candidate:

```yaml
effect32_film_gsens_ft_highact_joint_anchor:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact_joint_anchor
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  high_goal_mse_weight: 0.3
  high_action_loss_weight: 300.0
  low_anchor_loss_weight: 10.0
  goal_sensitivity_weight: 0.02
  goal_sensitivity_margin: 0.2
  policy_lr: 1.0e-5
  policy_epochs: 20
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_joint_anchor \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_joint_anchor \
  --goal-source learned \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_joint_anchor \
  --goal-source oracle \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  goal-diagnostics \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_joint_anchor \
  --seed 0 \
  --samples 4096 \
  --horizons 2,5,10,20 \
  --output results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact_joint_anchor/diagnostics.json \
  --force
```

### Training Metrics

Best epoch: `18`.

| metric | value |
| --- | ---: |
| validation normalized goal L2 | 2.5443 |
| validation oracle action MAE | 0.0361 |
| validation predicted action MAE | 0.0364 |
| validation prediction-induced action L2 | 0.0125 |
| last low train MSE | 0.000925 |
| last low anchor loss | 0.00000749 |
| last low goal-sensitivity loss | 0.00501 |

The anchor strongly constrained drift on the sampled low-level training inputs,
but the validation action-through-low metrics were not better than the previous
joint run.

### Results

Matched 200-episode window, `seed_start=3500000`:

| candidate | learned success | learned final | learned max | learned teacher MAE | oracle success | oracle final | oracle max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.7347 | 0.7420 | 0.1073 | 0.645 | 0.7399 | 0.7464 |
| effect32_film_gsens_ft_highact_strong | 0.640 | 0.7316 | 0.7403 | 0.0958 | 0.675 | 0.7657 | 0.7733 |
| effect32_film_gsens_ft_highact_joint | 0.635 | 0.7260 | 0.7358 | 0.1034 | 0.615 | 0.7173 | 0.7270 |
| effect32_film_gsens_ft_highact_joint_anchor | 0.595 | 0.7006 | 0.7113 | 0.0975 | 0.635 | 0.7395 | 0.7455 |

Goal-use diagnostic:

| candidate | frame shuffle L2 | goal shuffle L2 | max goal sensitivity L2 |
| --- | ---: | ---: | ---: |
| highact strong | 0.9386 | 0.0919 | 0.0399 |
| highact joint | 0.9381 | 0.0812 | 0.0598 |
| highact joint anchor | 0.9349 | 0.0878 | 0.0653 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_joint_anchor/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_joint_anchor/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_joint_anchor/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_joint_anchor/seed0/oracle_hierarchy_eval_200_seed3500000.json`
- `results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_ft_highact_joint_anchor/diagnostics.json`

### Interpretation

Reject this anchored joint recipe. The anchor prevents the worst oracle-goal
collapse from the naive joint run (`0.615 -> 0.635`), but it is still below
`highact_strong` oracle performance (`0.675`) and it substantially hurts
learned-goal deployment (`0.640 -> 0.595` versus `highact_strong`).

The offline same-input anchor is too indirect. It can constrain sampled action
drift while still failing to preserve the closed-loop behavior that matters, and
it does not fix the low-level goal-use gate. The next coupled objective should
anchor deployment behavior more directly, for example by preserving oracle-goal
closed-loop or serial-segment behavior while optimizing learned-goal
action-through-low compatibility.

## 2026-06-26 - Frozen-low action-only high-level fine-tune

### Hypothesis

The best successful branch so far freezes the goal-sensitive low policy and
trains the high level through that frozen low policy. The previous strong
candidate still included a high-level goal-MSE term. That term may be
over-constraining the high level toward demonstration future-goal latents instead
of the goals that make the frozen low policy match demonstration actions.

### Configuration

Added candidate:

```yaml
effect32_film_gsens_ft_highact_actiononly:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact_actiononly
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  freeze_low_policy: true
  high_goal_mse_weight: 0.0
  high_action_loss_weight: 300.0
  policy_lr: 1.0e-5
  policy_epochs: 20
```

This is identical to `effect32_film_gsens_ft_highact_strong` except that
`high_goal_mse_weight` is zero instead of `0.3`. Because the low policy is the
same frozen `effect32_film_gsens_ft` low policy, the oracle-goal low-level
behavior and low-level goal-use diagnostics are inherited from
`highact_strong`.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 200 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3600000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 1000 \
  --eval-seed-start 3700000 \
  --force
```

### Training Metrics

Best epoch: `17`.

| metric | action-only | highact-strong |
| --- | ---: | ---: |
| validation normalized goal L2 | 2.6673 | 2.5489 |
| validation oracle action MAE | 0.0362 | 0.0362 |
| validation predicted action MAE | 0.03645 | 0.03646 |
| validation prediction-induced action L2 | 0.01299 | 0.01278 |

The offline validation metrics are nearly tied. Action-only predicts goals a bit
farther from the supervised future-goal target, as expected.

### Results

Initial 200-episode learned-goal screen:

| candidate | success | final reward | max reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.7347 | 0.7420 | 0.1073 |
| effect32_film_gsens_ft_highact_strong | 0.640 | 0.7316 | 0.7403 | 0.0958 |
| effect32_film_gsens_ft_highact_actiononly | 0.650 | 0.7383 | 0.7496 | 0.0936 |

500-episode windows:

| seed start | candidate | success | final reward | max reward | teacher MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3500000 | effect32_film | 0.650 | 0.7410 | 0.7484 | 0.0996 |
| 3500000 | highact_strong | 0.652 | 0.7455 | 0.7523 | 0.0895 |
| 3500000 | actiononly | 0.668 | 0.7550 | 0.7637 | 0.0915 |
| 3600000 | effect32_film | 0.666 | 0.7524 | 0.7618 | 0.0965 |
| 3600000 | highact_strong | 0.672 | 0.7620 | 0.7674 | 0.0834 |
| 3600000 | actiononly | 0.670 | 0.7581 | 0.7648 | 0.0850 |

1000-episode final-style window:

| candidate | success | final reward | max reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.635 | 0.7306 | 0.7399 | 0.0978 |
| highact_strong | 0.645 | 0.7386 | 0.7463 | 0.0902 |
| actiononly | 0.661 | 0.7481 | 0.7564 | 0.0958 |

Aggregate over all three windows:

| candidate | episodes | success | final reward | max reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 2000 | 0.6465 | 0.7386 | 0.7475 | 0.0980 |
| highact_strong | 2000 | 0.6535 | 0.7462 | 0.7530 | 0.0883 |
| actiononly | 2000 | 0.6650 | 0.7523 | 0.7603 | 0.0920 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_200_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_500_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_500_seed3600000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_1000_seed3700000.json`

### Interpretation

This is the new best learned-goal candidate in the current branch. Removing the
explicit high-level goal-MSE term improves closed-loop success and rewards over
the previous `highact_strong` candidate on the 2000-episode aggregate. The gain
is not explained by lower aggregate teacher-action MAE, because `highact_strong`
is still slightly better on that scalar. The important signal is that allowing
the high level to choose action-compatible goals, unconstrained by direct
future-goal MSE, improves deployment with the frozen goal-sensitive low policy.

Treat `effect32_film_gsens_ft_highact_actiononly` as the current frozen-low base
for the next serial/RL compatibility checks. It keeps the same low-level oracle
behavior as `highact_strong`, but has better learned-goal closed-loop results.

## 2026-06-26 - Action-only high-level serial compatibility check

### Hypothesis

The action-only high-level candidate is the best standard learned-interface
evaluator result. Before using it as an R3 base, it should pass the same
`low-level-rl eval-serial` compatibility check used for `highact_strong`.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --seed 0 \
  --run-name hcl_next_highact_actiononly_frozen_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --seed 0 \
  --run-name hcl_next_highact_actiononly_frozen_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --force
```

### Results

Matched serial windows:

| seed start | candidate | success | final reward | max reward | segment goal reach | action saturation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3500000 | highact_strong | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.0458 |
| 3500000 | actiononly | 0.660 | 0.6161 | 0.7562 | 0.742 | 0.0479 |
| 3600000 | highact_strong | 0.770 | 0.7199 | 0.8385 | 0.791 | 0.0400 |
| 3600000 | actiononly | 0.720 | 0.6943 | 0.8058 | 0.767 | 0.0421 |

Two-window aggregate:

| candidate | episodes | success | final reward | max reward | segment goal reach | action saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| highact_strong | 200 | 0.720 | 0.7146 | 0.7976 | 0.7565 | 0.0429 |
| actiononly | 200 | 0.690 | 0.6552 | 0.7810 | 0.7545 | 0.0450 |

Paired action-only-vs-highact-strong success counts:

| seed start | wins | losses | net |
| ---: | ---: | ---: | ---: |
| 3500000 | 15 | 16 | -1 |
| 3600000 | 7 | 12 | -5 |
| aggregate | 22 | 28 | -6 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_actiononly/seed0/hcl_next_highact_actiononly_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_actiononly/seed0/hcl_next_highact_actiononly_frozen_serial100_seed3600000/serial_eval_100_seed3600000.json`

### Interpretation

This is a compatibility caveat, not a full rejection of action-only. The
standard learned-interface evaluator favors action-only over `highact_strong` on
the 2000-episode aggregate, but the serial evaluator favors `highact_strong` on
the same first two seed banks used for prior R3 checks.

The disagreement matters because the R3/local-RL path uses the serial evaluator.
Do not immediately replace `highact_strong` with action-only for R3 experiments.
The next useful step is to inspect the evaluator/protocol difference, or keep
`highact_strong` as the conservative local-RL base while treating action-only as
the best standard learned-interface deployment candidate.

## 2026-06-26 - Learned-interface evaluator num-env audit

### Hypothesis

The disagreement between standard learned-interface evaluation and
`low-level-rl eval-serial` may come from evaluator protocol rather than from the
candidate itself. The learned-interface evaluator defaults to vectorized raw
ManiSkill envs (`num_envs=16`), while `low-level-rl eval-serial` uses one env
through `_visual_env`, which wraps ManiSkill with `ManiSkillVectorEnv`.

### Implementation

Added an opt-in `--eval-num-envs` argument to `learned-interface-eval`. When
provided, it overrides `learned_interface.evaluation.num_envs` and appends
`_envs{N}` to the output filename, preserving existing result files.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 100 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 100 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 100 \
  --eval-seed-start 3500000 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 100 \
  --eval-seed-start 3500000 \
  --force
```

### Results

Same first 100-seed bank, `seed_start=3500000`:

| evaluator | num envs | candidate | success | final reward | max reward | teacher MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| learned-interface | 16 | highact_strong | 0.690 | 0.7661 | 0.7734 | 0.1001 |
| learned-interface | 16 | actiononly | 0.710 | 0.7819 | 0.7921 | 0.0931 |
| learned-interface | 1 | highact_strong | 0.670 | 0.7492 | 0.7566 | 0.0937 |
| learned-interface | 1 | actiononly | 0.660 | 0.7497 | 0.7562 | 0.0893 |
| low-level-rl serial | 1 | highact_strong | 0.670 | 0.7092 | 0.7566 | - |
| low-level-rl serial | 1 | actiononly | 0.660 | 0.6161 | 0.7562 | - |

Artifacts:

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json`

### Interpretation

The `num_envs=1` learned-interface evaluator reproduces the serial success
ordering and exact max rewards. The default vectorized learned-interface
evaluator does not. This means the action-only lead is currently a vectorized
evaluator lead, not a robust single-env/serial deployment lead.

For follow-up RL, keep `highact_strong` as the conservative base because the
R3/local-RL path uses serial evaluation. Future learned-interface validation
tables should report `eval_num_envs`, and any promoted RL base should pass a
single-env or serial check.

## 2026-06-26 - Goal-MSE 0.1 high-action interpolation

### Hypothesis

Action-only (`high_goal_mse_weight=0.0`) is best under the default vectorized
learned-interface evaluator but worse under single-env/serial evaluation.
`highact_strong` (`high_goal_mse_weight=0.3`) remains better for serial/RL. A
middle value may keep some action-compatible goal freedom while preserving the
single-env behavior that matters for R3.

### Configuration

Added candidate:

```yaml
effect32_film_gsens_ft_highact_goal01:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact_goal01
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  freeze_low_policy: true
  high_goal_mse_weight: 0.1
  high_action_loss_weight: 300.0
  policy_lr: 1.0e-5
  policy_epochs: 20
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_goal01 \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_goal01 \
  --goal-source learned \
  --episodes 100 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_goal01 \
  --seed 0 \
  --run-name hcl_next_highact_goal01_frozen_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --force
```

### Training Metrics

| candidate | goal MSE weight | best epoch | validation goal L2 | predicted action MAE | prediction-induced action L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| actiononly | 0.0 | 17 | 2.6673 | 0.03645 | 0.01299 |
| goal01 | 0.1 | 17 | 2.5796 | 0.03646 | 0.01290 |
| highact_strong | 0.3 | 20 | 2.5489 | 0.03646 | 0.01278 |

The interpolation behaves as expected offline: goal L2 moves between action-only
and `highact_strong`, while action-through-low metrics remain nearly tied.

### Results

First 100-seed screen, `seed_start=3500000`:

| evaluator | candidate | success | final reward | max reward | teacher MAE / segment reach |
| --- | --- | ---: | ---: | ---: | ---: |
| learned-interface envs=1 | actiononly | 0.660 | 0.7497 | 0.7562 | 0.0893 |
| learned-interface envs=1 | goal01 | 0.660 | 0.7479 | 0.7536 | 0.0840 |
| learned-interface envs=1 | highact_strong | 0.670 | 0.7492 | 0.7566 | 0.0937 |
| low-level-rl serial | actiononly | 0.660 | 0.6161 | 0.7562 | 0.742 |
| low-level-rl serial | goal01 | 0.660 | 0.6742 | 0.7536 | 0.745 |
| low-level-rl serial | highact_strong | 0.670 | 0.7092 | 0.7566 | 0.722 |

Paired serial counts:

| comparison | wins | losses | net |
| --- | ---: | ---: | ---: |
| goal01 vs highact_strong | 12 | 13 | -1 |
| goal01 vs actiononly | 10 | 10 | 0 |

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_goal01/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_goal01/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_goal01/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_goal01/seed0/hcl_next_highact_goal01_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`

### Interpretation

Reject this interpolation point as a serial/RL base. It improves final reward
over action-only in serial mode, but not enough to beat `highact_strong`; success
is still lower and paired counts are slightly negative. This narrows the current
candidate choice: use `highact_strong` for conservative local-RL work, and keep
action-only only as a vectorized learned-interface lead.

## 2026-06-26 - Paired terminal-D_phi R3 on high-action base

### Hypothesis

The absolute terminal-`D_phi` R3 update on
`effect32_film_gsens_ft_highact_strong` improved the local training proxy but
hurt serial deployment. A paired reward that directly scores tuned-vs-frozen
terminal reachability might reduce proxy over-optimization and produce a safer
low-level update.

### Commands

The first 4096-env paired run failed at reset with GPU camera allocation because
paired mode creates a second synchronized rollout:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_paired_4096_terminal_40k_bc10 \
  --steps 40960 \
  --num-envs 4096 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

I reran the smoke with 2048 paired envs. The first 2048-env run exposed a
paired-branch sync bug: the bootstrap value call can trigger a high-level replan
on the tuned rollout after an update, but the frozen paired branch was not
copied after that mutation. I patched `train_direct_low_rl` to copy the paired
branch after bootstrap replans and reran under a fresh run name:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10 \
  --steps 40960 \
  --num-envs 2048 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Serial checks:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/best_train_latent.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/best_train_latent.pt \
  --force
```

### Training Metrics

The fixed run stayed synchronized. The best checkpoint is still the
intermediate paired-positive checkpoint at 20480 steps because the final row's
mean paired improvement was much smaller.

| global step | mean paired improvement | fraction paired improved | tuned terminal D_phi | base terminal D_phi | action saturation | paired desynced envs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20480 | 0.0907 | 0.5869 | 0.4029 | 0.4935 | 0.4062 | 0 |
| 40960 | 0.0161 | 0.4854 | 0.5889 | 0.6049 | 0.1917 | 0 |

### Serial Results

Matched two-window serial comparison against the frozen high-action base:

| seed start | policy | success | final reward | max reward | segment reach | segment final distance | raw segment reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3500000 | frozen | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.5225 | 0.3890 |
| 3500000 | pairedsync R3 | 0.640 | 0.6469 | 0.7428 | 0.747 | 0.5974 | 0.3963 |
| 3600000 | frozen | 0.770 | 0.7199 | 0.8385 | 0.791 | 0.4565 | 0.3732 |
| 3600000 | pairedsync R3 | 0.680 | 0.6704 | 0.7739 | 0.777 | 0.5888 | 0.3879 |

Aggregate:

| policy | episodes | success | final reward | max reward | segment reach |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 200 | 0.720 | 0.7146 | 0.7976 | 0.756 |
| pairedsync R3 | 200 | 0.660 | 0.6587 | 0.7584 | 0.762 |

Paired counts across both windows:

| metric | wins | losses | ties |
| --- | ---: | ---: | ---: |
| final reward | 54 | 63 | 83 |
| max reward | 39 | 46 | 115 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3600000/serial_eval_100_seed3600000.json`

### Interpretation

Reject this paired R3 checkpoint. The paired local training objective found a
checkpoint with positive tuned-vs-frozen `D_phi` improvement, and serial segment
reach was slightly higher in aggregate, but full-task serial success, final
reward, and max reward all dropped. This is the same proxy-transfer failure as
the absolute terminal-`D_phi` run, now under a paired objective. The immediate
next lever should be a better checkpoint selector/local objective or a different
counterfactual/evaluation formulation, not scaling this paired recipe.

## 2026-06-26 - Oracle-goal diagnostic for paired high-action R3

### Hypothesis

The paired-R3 checkpoint may fail under learned goals because the high-level
prediction distribution is bad, not because the low-level update is intrinsically
unhelpful. Re-evaluating frozen and paired-R3 with oracle serial goals separates
learned high-level quality from low-level/task-transfer behavior.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_frozen_oracle_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --goal-source oracle \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_oracle_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --goal-source oracle \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/best_train_latent.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_frozen_oracle_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --goal-source oracle \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_oracle_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --goal-source oracle \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/best_train_latent.pt \
  --force
```

### Results

Matched comparison against the learned-goal serial windows:

| seed start | goal source | policy | success | final reward | max reward | segment reach | segment final distance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3500000 | learned | frozen | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.5225 |
| 3500000 | learned | pairedsync R3 | 0.640 | 0.6469 | 0.7428 | 0.747 | 0.5974 |
| 3500000 | oracle | frozen | 0.660 | 0.6215 | 0.7560 | 0.792 | 0.3380 |
| 3500000 | oracle | pairedsync R3 | 0.650 | 0.5858 | 0.7482 | 0.793 | 0.6042 |
| 3600000 | learned | frozen | 0.770 | 0.7199 | 0.8385 | 0.791 | 0.4565 |
| 3600000 | learned | pairedsync R3 | 0.680 | 0.6704 | 0.7739 | 0.777 | 0.5888 |
| 3600000 | oracle | frozen | 0.680 | 0.5994 | 0.7767 | 0.812 | 0.3322 |
| 3600000 | oracle | pairedsync R3 | 0.700 | 0.6273 | 0.7926 | 0.833 | 0.5747 |

Aggregate:

| goal source | policy | episodes | success | final reward | max reward | segment reach | segment final distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | frozen | 200 | 0.720 | 0.7146 | 0.7976 | 0.756 | 0.4895 |
| learned | pairedsync R3 | 200 | 0.660 | 0.6587 | 0.7584 | 0.762 | 0.5931 |
| oracle | frozen | 200 | 0.670 | 0.6105 | 0.7663 | 0.802 | 0.3351 |
| oracle | pairedsync R3 | 200 | 0.675 | 0.6065 | 0.7704 | 0.813 | 0.5895 |

Paired counts:

| goal source | metric | wins | losses | ties |
| --- | --- | ---: | ---: | ---: |
| learned | final reward | 54 | 63 | 83 |
| learned | max reward | 39 | 46 | 115 |
| oracle | final reward | 66 | 75 | 59 |
| oracle | max reward | 47 | 43 | 110 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_oracle_serial100_seed3500000/serial_eval_100_seed3500000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_oracle_serial100_seed3500000/serial_eval_100_seed3500000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_oracle_serial100_seed3600000/serial_eval_100_seed3600000_oracle_goals.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_oracle_serial100_seed3600000/serial_eval_100_seed3600000_oracle_goals.json`

### Interpretation

Oracle serial goals greatly improve local segment reach for the frozen
high-action base (`0.756 -> 0.802`) but reduce task-level final reward and
success (`0.720 -> 0.670`). For this candidate, oracle one-segment goals are not
a better closed-loop task policy than the learned high-level goals.

The paired-R3 checkpoint is only neutral under oracle goals: success changes
from `0.670` to `0.675`, final reward is slightly lower, max reward is slightly
higher, and paired final-reward counts are negative (`66` wins / `75` losses).
This weakens the hypothesis that paired-R3 failed only because learned
high-level goals were poor. The low-level update still improves local reach-like
metrics without creating a reliable task-level gain.

## 2026-06-26 - Hindsight branch-selector upper bound for high-action paired R3

### Hypothesis

If frozen and paired-R3 make complementary closed-loop mistakes, then an
episode-level hindsight selector should outperform both. If the upper bound is
weak, deployable selector work is probably not worth pursuing on this checkpoint.

### Command

Computed directly from the exact paired serial JSONs for learned and oracle
goals on seed windows `3500000` and `3600000`.

### Results

Here `success` is the same max-reward threshold used by the serial evaluator.
The final-reward selector chooses the branch with better episode final reward;
the max-reward selector chooses the branch with better episode max reward.

| goal source | policy / selector | final reward | max reward | success |
| --- | --- | ---: | ---: | ---: |
| learned | frozen | 0.7146 | 0.7976 | 0.720 |
| learned | pairedsync R3 | 0.6587 | 0.7584 | 0.660 |
| learned | hindsight final-reward selector | 0.7956 | - | - |
| learned | hindsight max-reward selector | 0.7537 | 0.8616 | 0.805 |
| oracle | frozen | 0.6105 | 0.7663 | 0.670 |
| oracle | pairedsync R3 | 0.6065 | 0.7704 | 0.675 |
| oracle | hindsight final-reward selector | 0.7194 | - | - |
| oracle | hindsight max-reward selector | 0.6708 | 0.8571 | 0.795 |

### Interpretation

There is real branch complementarity at the episode-outcome level. The
paired-R3 branch is bad on average, but a non-deployable max-reward oracle
selector would improve learned-goal success by `+0.085` over frozen and oracle
goal success by `+0.125` over frozen. This does not rescue the checkpoint as a
policy because the selector uses full-episode future information, but it keeps
the selector direction alive. The selector target needs to be closed-loop
task-outcome labels, not local segment reachability or terminal `D_phi`.

## 2026-06-26 - Initial selector cross-split audit for high-action paired R3

### Hypothesis

The hindsight upper bound shows branch complementarity, but a deployable selector
needs to predict useful branch choices from pre-decision features. The existing
`low-level-rl fit-serial-selector` command fits a three-feature initial selector
from exact paired serial success labels. If this simple selector transfers across
seed windows, it is worth validating online; if it does not, selector work needs
richer closed-loop context.

### Commands

```bash
uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml \
  fit-serial-selector \
  --base-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3500000/serial_eval_100_seed3500000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3600000/serial_eval_100_seed3600000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3600000/serial_eval_100_seed3600000.json \
  --output results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_initial_selector_learned_train3500000_valid3600000.json \
  --force

uv run hcl-poc low-level-rl --config configs/pusht_incremental.yaml \
  fit-serial-selector \
  --base-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3600000/serial_eval_100_seed3600000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3600000/serial_eval_100_seed3600000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3500000/serial_eval_100_seed3500000.json \
  --output results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_initial_selector_learned_train3600000_valid3500000.json \
  --force
```

I repeated the same cross-split fit with the oracle-goal JSONs from the previous
diagnostic.

### Results

Selector features:

```text
episode_initial_selected_distance
episode_initial_raw_distance
episode_initial_base_action_l2
```

| goal source | train window | validation window | train frozen | train R3 | train selector | validation frozen | validation R3 | validation selector | validation use R3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | 3500000 | 3600000 | 0.670 | 0.640 | 0.700 | 0.770 | 0.680 | 0.740 | 0.690 |
| learned | 3600000 | 3500000 | 0.770 | 0.680 | 0.780 | 0.670 | 0.640 | 0.690 | 0.240 |
| oracle | 3500000 | 3600000 | 0.660 | 0.650 | 0.710 | 0.680 | 0.700 | 0.710 | 0.480 |
| oracle | 3600000 | 3500000 | 0.680 | 0.700 | 0.720 | 0.660 | 0.650 | 0.690 | 0.540 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_initial_selector_learned_train3500000_valid3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_initial_selector_learned_train3600000_valid3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_initial_selector_oracle_train3500000_valid3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_initial_selector_oracle_train3600000_valid3500000.json`

### Interpretation

Reject this simple initial selector for learned-goal deployment. It overfits in
both directions and has a two-split validation average of `0.715`, slightly
below the frozen learned-goal average of `0.720`. The oracle-goal selector is
consistently mildly positive, but oracle goals are not the deployment setting
and the oracle-goal frozen baseline is itself worse than learned-goal frozen on
these windows. The selector direction still needs richer online context or
direct closed-loop intervention training; the existing three initial features
are not enough.

## 2026-06-26 - Weaker BC anchor for high-action paired R3

### Hypothesis

The paired-R3 branch may be too close to the frozen low policy to create a useful
closed-loop effect. Reducing the BC anchor from `10.0` to `1.0` should allow
larger action changes while keeping the paired terminal-`D_phi` objective.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc1 \
  --steps 40960 \
  --num-envs 2048 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 1.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

### Results

Comparison against the previous `bc_weight=10.0` pairedsync run:

| run | step | mean paired improvement | fraction improved | tuned terminal D_phi | base terminal D_phi | direct delta L2 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bc10 | 20480 | 0.0907 | 0.5869 | 0.4029 | 0.4935 | 0.2650 | 0.4062 |
| bc10 | 40960 | 0.0161 | 0.4854 | 0.5889 | 0.6049 | 0.2640 | 0.1917 |
| bc1 | 20480 | 0.0907 | 0.5869 | 0.4029 | 0.4935 | 0.2650 | 0.4062 |
| bc1 | 40960 | 0.0176 | 0.4888 | 0.5874 | 0.6049 | 0.2639 | 0.1894 |

Both runs selected the 20480-step checkpoint as best. The bc1 and bc10 best
agent tensors differ only slightly:

| comparison | value |
| --- | ---: |
| global step | 20480 for both |
| max absolute tensor delta | 0.000482 |
| average per-tensor mean absolute delta | 0.0000159 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc1/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc1/train_metrics.json`

### Interpretation

Reject this as a meaningfully new deployment candidate. Weakening the BC anchor
did not produce larger action changes or a materially different paired training
signal. Since the bc10 best checkpoint already failed serial deployment, and bc1
is nearly identical at the selected step, I skipped another serial evaluation.
The next objective change needs to alter the target/regime more substantially
than just reducing the final-layer BC coefficient.

## 2026-06-26 - Dense task-reward diagnostic on high-action direct-low R3

### Hypothesis

The learned reachability rewards may be the bottleneck. As a diagnostic only,
train the same high-action direct-low R3 update against privileged dense task
reward with local distance rewards disabled. If even the task reward cannot
produce a useful update in this local setup, the issue is the local update
formulation rather than just the learned reachability metric.

### Commands

Training:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_2048_40k_bc1 \
  --steps 40960 \
  --num-envs 2048 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 1.0 \
  --terminal-weight 0.0 \
  --distance-progress-weight 0.0 \
  --task-reward-weight 1.0 \
  --task-progress-weight 0.0 \
  --reward-mode absolute \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --force
```

Serial eval used `latest.pt`, not `best_train_latent.pt`, because the generic
direct-low checkpoint selector still ranks by terminal reachability distance,
while this diagnostic optimizes dense task reward:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_2048_40k_bc1_latest_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_40k_bc1/latest.pt \
  --force
```

### Training Metrics

| step | mean reward | terminal D_phi | direct delta L2 | saturation | bc loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20480 | 0.1285 | 0.4029 | 0.2650 | 0.4062 | 0.000047 |
| 40960 | 0.1865 | 0.5885 | 0.2640 | 0.1956 | 0.000130 |

The in-training dense reward increased, but the terminal reachability distance
got worse. This is expected because the local distance reward was disabled.

### Serial Result

First 100-seed learned-goal window, `seed_start=3500000`:

| policy | success | final reward | max reward | segment reach | segment final distance | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen highact_strong | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.5225 | 0.00000 |
| pairedsync D_phi R3 | 0.640 | 0.6469 | 0.7428 | 0.747 | 0.5974 | 0.00579 |
| task-reward latest | 0.580 | 0.6101 | 0.6976 | 0.693 | 0.6211 | 0.01051 |

Paired final-reward counts versus frozen:

| candidate | wins | losses | ties |
| --- | ---: | ---: | ---: |
| pairedsync D_phi R3 | 25 | 31 | 44 |
| task-reward latest | 30 | 36 | 34 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_40k_bc1/latest.pt`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_40k_bc1/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_40k_bc1_latest_serial100_seed3500000/serial_eval_100_seed3500000.json`

### Interpretation

Reject this local direct-low task-reward update. It creates a larger serial
residual than the paired `D_phi` checkpoint, but the larger changes hurt task
success, final reward, max reward, and segment reach. The failure is therefore
not simply that `D_phi` is the wrong scalar reward; the current local direct-low
update can also overfit or misapply privileged dense task reward. A useful next
target likely needs longer-horizon/closed-loop intervention training or a
different policy update structure, not another one-segment scalar target.

## 2026-06-26 - Longer-rollout task-reward diagnostic on high-action R3

### Hypothesis

The dense task-reward run failed partly because PPO credit was cut at every
held-goal segment. A longer rollout with `segment_terminates_gae=False` should
let task reward propagate across several high-level replans, making the update
closer to closed-loop training while still using the direct-low R3 machinery.

### Commands

Training:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae \
  --steps 102400 \
  --num-envs 2048 \
  --rollout-steps 50 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 1.0 \
  --terminal-weight 0.0 \
  --distance-progress-weight 0.0 \
  --task-reward-weight 1.0 \
  --task-progress-weight 0.0 \
  --reward-mode absolute \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --no-segment-terminate-gae \
  --force
```

Continuation to a second update:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae \
  --steps 204800 \
  --num-envs 2048 \
  --rollout-steps 50 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 1.0 \
  --terminal-weight 0.0 \
  --distance-progress-weight 0.0 \
  --task-reward-weight 1.0 \
  --task-progress-weight 0.0 \
  --reward-mode absolute \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film/seed0/d_phi.pt \
  --no-segment-terminate-gae
```

Serial eval:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_roll50_102k_bc1_noseggae_latest_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae/latest.pt \
  --force
```

After the second update, I ran a distinct serial eval for the 204800-step
`latest.pt` checkpoint:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_roll50_204k_bc1_noseggae_latest_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae/latest.pt \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_taskreward_roll50_204k_bc1_noseggae_latest_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae/latest.pt \
  --force
```

### Training Metrics

| run | step | mean reward | terminal D_phi | direct delta L2 | saturation | segment terminates GAE |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| roll10 task reward | 20480 | 0.1285 | 0.4029 | 0.2650 | 0.4062 | true |
| roll10 task reward | 40960 | 0.1865 | 0.5885 | 0.2640 | 0.1956 | true |
| roll50 task reward | 102400 | 0.2121 | 0.6198 | 0.2640 | 0.1438 | false |
| roll50 task reward | 204800 | 0.2130 | 0.6238 | 0.2638 | 0.1461 | false |

The second update barely changed the in-training mean reward, but its serial
behavior was different enough to evaluate.

### Serial Result

First 100-seed learned-goal window, `seed_start=3500000`:

| policy | success | final reward | max reward | segment reach | segment final distance | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen highact_strong | 0.670 | 0.7092 | 0.7566 | 0.722 | 0.5225 | 0.00000 |
| roll10 task reward | 0.580 | 0.6101 | 0.6976 | 0.693 | 0.6211 | 0.01051 |
| roll50 task reward, 102400 | 0.640 | 0.6171 | 0.7436 | 0.743 | 0.6095 | 0.00907 |
| roll50 task reward, 204800 | 0.680 | 0.6556 | 0.7755 | 0.761 | 0.6077 | 0.01391 |

Paired final-reward counts versus frozen:

| candidate | wins | losses | ties |
| --- | ---: | ---: | ---: |
| roll10 task reward | 30 | 36 | 34 |
| roll50 task reward, 102400 | 26 | 40 | 34 |
| roll50 task reward, 204800 | 30 | 32 | 38 |

The first window looked positive on success and max reward at 204800 steps, so I
ran the matched second seed window. Two-window aggregate:

| policy | episodes | success | final reward | max reward | segment reach | residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen highact_strong | 200 | 0.720 | 0.7146 | 0.7976 | 0.756 | 0.00000 |
| pairedsync D_phi R3 | 200 | 0.660 | 0.6587 | 0.7584 | 0.762 | 0.00561 |
| roll50 task reward, 204800 | 200 | 0.680 | 0.6329 | 0.7759 | 0.765 | 0.01366 |

Two-window paired counts versus frozen:

| candidate | metric | wins | losses | ties |
| --- | --- | ---: | ---: | ---: |
| roll50 task reward, 204800 | final reward | 58 | 71 | 71 |
| roll50 task reward, 204800 | max reward | 42 | 40 | 118 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae/latest.pt`
- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae/step_000102400.pt`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_2048_roll50_102k_bc1_noseggae/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_roll50_102k_bc1_noseggae_latest_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_roll50_204k_bc1_noseggae_latest_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_taskreward_roll50_204k_bc1_noseggae_latest_serial100_seed3600000/serial_eval_100_seed3600000.json`

### Checkpointing Fix

This continuation exposed a resume bug in both `train_residual_rl` and
`train_direct_low_rl`: when resuming an existing run, `best_score` was reset to
`-inf`, so the first resumed update could overwrite `best_train_latent.pt` even
if it was worse than previous history. I fixed both trainers to restore
`best_score` from loaded `history` before continuing, and restored this run's
`best_train_latent.pt` to the true 102400-step best checkpoint.

### Interpretation

Longer credit assignment is the strongest local-RL direction in this high-action
branch so far, but it is still below frozen over two windows. At 204800 steps it
beats paired `D_phi` on success, max reward, and segment reach, and it briefly
beats frozen on the first seed window. The second window rejects it as a robust
policy improvement: aggregate success is `0.680` versus frozen `0.720`, and
final reward is much worse. This suggests multi-replan credit is more promising
than one-segment scalar reachability, but the current direct-low local update
still does not solve closed-loop deployment.

## 2026-06-26 - Larger single-env learned-interface evaluator check

### Hypothesis

The `actiononly` high-level lead may be an artifact of the default vectorized
learned-interface evaluator. The first 100-episode `eval_num_envs=1` check
favored `highact_strong`; a larger matched 500-episode single-env check on the
same `seed_start=3500000` window should tell whether that ordering was just
small-sample noise.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force
```

### Results

Matched 500-episode single-env learned-interface window:

| candidate | success | final reward | max reward | teacher MAE | high-level decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| highact_strong | 0.690 | 0.7720 | 0.7784 | 0.0877 | 6.786 |
| actiononly | 0.684 | 0.7684 | 0.7755 | 0.0846 | 6.746 |

Artifacts:

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_500_seed3500000_envs1.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_500_seed3500000_envs1.json`

### Interpretation

The larger single-env learned-interface check confirms the conservative
ordering from the 100-episode envs=1 audit: `highact_strong` is slightly better
than `actiononly` under the single-env protocol used as a proxy for serial/RL
compatibility. This does not erase the vectorized learned-interface lead for
`actiononly`, but it means the lead is evaluator-protocol dependent. I did not
run the second 500-episode single-env window because each run takes roughly
20 minutes and this first larger window already supports the current policy
choice: keep `highact_strong` as the serial/RL base and treat `actiononly` as a
vectorized learned-interface lead until the protocol difference is understood.

## 2026-06-26 - Single-env learned-interface versus serial metric audit

### Hypothesis

The single-env learned-interface evaluator and `low-level-rl eval-serial` may
be running the same trajectories but reporting different final-reward semantics.
If so, per-episode success and max reward should match exactly, while
`episode_final_reward` may differ on episodes that terminate early.

### Inputs

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_actiononly/seed0/hcl_next_highact_actiononly_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`

### Results

Per-episode array comparison:

| candidate | success exact matches | max-reward exact matches | final-reward exact matches | final-reward mean diff |
| --- | ---: | ---: | ---: | ---: |
| highact_strong | 100 / 100 | 100 / 100 | 94 / 100 | +0.0400 |
| actiononly | 100 / 100 | 100 / 100 | 80 / 100 | +0.1336 |

The final-reward mismatches are success episodes. Example rows:

| candidate | seed | serial steps | learned final | serial final | max reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| highact_strong | 3500014 | 100 | 1.000 | 0.3280 | 1.000 |
| actiononly | 3500004 | 100 | 1.000 | 0.3364 | 1.000 |

### Interpretation

The single-env learned-interface evaluator and serial evaluator agree on the
trajectory-level success and max-reward outcomes for this window. The
`final_reward` discrepancy is a metric/protocol semantics issue: raw
learned-interface evaluation stops when the environment terminates at success,
whereas serial evaluation uses `ManiSkillVectorEnv(ignore_terminations=True)`
and records the later reward at the 100-step horizon. Cross-protocol comparisons
should therefore use success and max reward, not final reward. This explains
the large serial final-reward penalty for `actiononly` without changing the
conservative base choice, because success and max reward still do not show a
single-env/serial advantage over `highact_strong`.

## 2026-06-26 - Learned-interface evaluator vectorization sweep

### Hypothesis

The action-only lead may depend on the learned-interface evaluator's default
`num_envs=16`. If it is a robust deployment improvement, it should not flip
erratically as the same evaluator is run with smaller vectorization levels.

### Commands

```bash
for envs in 2 4 8; do
  for candidate in \
    effect32_film_gsens_ft_highact_strong \
    effect32_film_gsens_ft_highact_actiononly; do
    TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
      --config configs/pusht_incremental.yaml \
      --candidate "${candidate}" \
      --goal-source learned \
      --episodes 100 \
      --eval-seed-start 3500000 \
      --eval-num-envs "${envs}" \
      --force
  done
done
```

I compared these new results with the existing `eval_num_envs=1` and default
`eval_num_envs=16` files.

### Results

| eval num envs | highact success | actiononly success | highact final | actiononly final | highact max | actiononly max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.670 | 0.660 | 0.7492 | 0.7497 | 0.7566 | 0.7562 |
| 2 | 0.650 | 0.650 | 0.7305 | 0.7370 | 0.7448 | 0.7473 |
| 4 | 0.630 | 0.690 | 0.7227 | 0.7673 | 0.7342 | 0.7775 |
| 8 | 0.640 | 0.600 | 0.7285 | 0.7065 | 0.7387 | 0.7156 |
| 16 | 0.690 | 0.710 | 0.7661 | 0.7819 | 0.7734 | 0.7921 |

Artifacts:

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs2.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs2.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs4.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs4.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs8.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs8.json`

### Interpretation

The evaluator-vectorization effect is not monotonic. `actiononly` wins at
`eval_num_envs=4` and `16`, ties success at `2`, and loses at `1` and `8`.
This makes the default vectorized lead too protocol-sensitive for RL-base
promotion. The single-env/serial-compatible evidence remains the more
conservative deployment criterion, so `highact_strong` stays the local-RL base
despite `actiononly` being the best candidate under the default vectorized
learned-interface evaluator.

## 2026-06-26 - Reproducible eval-vectorization comparison command

### Hypothesis

The vectorization sensitivity should be captured by a repeatable JSON artifact
rather than an ad hoc Python snippet. If the same seeds have unstable outcomes
across `eval_num_envs`, the comparison should show many per-index success flips
and large max-reward differences relative to the single-env reference.

### Implementation

Added:

```bash
uv run hcl-poc incremental learned-interface-compare-evals \
  --eval-json ... \
  --name ... \
  --output ...
```

The command reads two or more learned-interface eval JSONs with matching episode
counts, treats the first file as reference, and writes scalar summaries plus
per-index agreement metrics for success, final reward, and max reward.

### Commands

```bash
uv run hcl-poc incremental learned-interface-compare-evals \
  --config configs/pusht_incremental.yaml \
  --eval-json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs2.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs4.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000_envs8.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_100_seed3500000.json \
  --name envs1 envs2 envs4 envs8 envs16 \
  --output results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/eval_num_envs_sensitivity_100_seed3500000.json \
  --force

uv run hcl-poc incremental learned-interface-compare-evals \
  --config configs/pusht_incremental.yaml \
  --eval-json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs2.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs4.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000_envs8.json \
    results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_100_seed3500000.json \
  --name envs1 envs2 envs4 envs8 envs16 \
  --output results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/eval_num_envs_sensitivity_100_seed3500000.json \
  --force
```

### Results

Reference is `eval_num_envs=1` for each candidate:

| candidate | comparison | success delta | success flips | max-reward mean abs diff |
| --- | --- | ---: | ---: | ---: |
| highact_strong | envs=2 | -0.020 | 32 | 0.2425 |
| highact_strong | envs=4 | -0.040 | 40 | 0.3026 |
| highact_strong | envs=8 | -0.030 | 47 | 0.3503 |
| highact_strong | envs=16 | +0.020 | 42 | 0.3125 |
| actiononly | envs=2 | -0.010 | 33 | 0.2437 |
| actiononly | envs=4 | +0.030 | 39 | 0.2870 |
| actiononly | envs=8 | -0.060 | 48 | 0.3471 |
| actiononly | envs=16 | +0.050 | 45 | 0.3291 |

Artifacts:

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/eval_num_envs_sensitivity_100_seed3500000.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/eval_num_envs_sensitivity_100_seed3500000.json`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py src/hcl_poc/cli.py
uv run hcl-poc incremental learned-interface-compare-evals --help
```

### Interpretation

This confirms that evaluator vectorization changes same-index outcomes, not
just aggregate estimates. Relative to `eval_num_envs=1`, 32-48 of 100 success
labels flip depending on candidate and env count, and max-reward mean absolute
differences are large. Promotion comparisons must pin the evaluator protocol;
for serial/local-RL work, use the single-env/serial-compatible protocol rather
than the default vectorized learned-interface evaluator.

## 2026-06-26 - Second-window 500-episode single-env base validation

### Hypothesis

The first 500-episode single-env validation favored `highact_strong` over
`actiononly`, but only by a small margin. A matched second 500-episode
single-env window at `seed_start=3600000` should determine whether action-only
recovers under the serial-compatible protocol.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3600000 \
  --eval-num-envs 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_actiononly \
  --goal-source learned \
  --episodes 500 \
  --eval-seed-start 3600000 \
  --eval-num-envs 1 \
  --force
```

### Results

Matched 500-episode single-env windows:

| candidate | seed start | success | final reward | max reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| highact_strong | 3500000 | 0.690 | 0.7720 | 0.7784 | 0.0877 |
| highact_strong | 3600000 | 0.690 | 0.7738 | 0.7790 | 0.0866 |
| actiononly | 3500000 | 0.684 | 0.7684 | 0.7755 | 0.0846 |
| actiononly | 3600000 | 0.690 | 0.7734 | 0.7810 | 0.0877 |

Two-window aggregate:

| candidate | episodes | success | final reward | max reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| highact_strong | 1000 | 0.690 | 0.7729 | 0.7787 | 0.0872 |
| actiononly | 1000 | 0.687 | 0.7709 | 0.7782 | 0.0861 |

Artifacts:

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_500_seed3600000_envs1.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_actiononly/seed0/learned_hierarchy_eval_500_seed3600000_envs1.json`

### Interpretation

The second single-env window is essentially tied, and the two-window
single-env aggregate still slightly favors `highact_strong` on success, final
reward, and max reward. This confirms that `actiononly` is not a
serial-compatible replacement for `highact_strong` despite its default
vectorized learned-interface lead. Keep `highact_strong` as the conservative
base for serial/local-RL work.

## 2026-06-27 - Learned-interface reset vectorization audit

### Hypothesis

The learned-interface evaluator vectorization sensitivity may come from reset
state mismatch rather than policy dynamics alone. If raw ManiSkill reset seeds
are not invariant to `num_envs`, the same seed index will start from a different
simulator state when the evaluator batch size changes.

### Implementation

Added:

```bash
uv run hcl-poc incremental learned-interface-audit-reset-vectorization \
  --seed-start ... \
  --episodes ... \
  --eval-num-envs ... \
  --output ...
```

The command reproduces the raw learned-interface evaluator reset pattern for
each requested `eval_num_envs`, records `env.unwrapped.get_state()`, and compares
same-seed state vectors against the first env-count argument as reference.

### Command

```bash
uv run hcl-poc incremental learned-interface-audit-reset-vectorization \
  --config configs/pusht_incremental.yaml \
  --seed-start 3500000 \
  --episodes 16 \
  --eval-num-envs 1 2 4 8 16 \
  --output results/incremental/learned_interface/reset_vectorization_audit_seed3500000_n16.json \
  --force
```

### Results

Reference is `eval_num_envs=1`:

| eval num envs | changed seeds | mean max-abs state diff | max max-abs state diff |
| ---: | ---: | ---: | ---: |
| 2 | 16 / 16 | 0.4811 | 1.9681 |
| 4 | 16 / 16 | 0.6172 | 1.6495 |
| 8 | 16 / 16 | 0.4777 | 1.1450 |
| 16 | 16 / 16 | 0.6299 | 1.6305 |

Artifact:

- `results/incremental/learned_interface/reset_vectorization_audit_seed3500000_n16.json`

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py src/hcl_poc/cli.py
uv run hcl-poc incremental learned-interface-audit-reset-vectorization --help
python3 -m json.tool results/incremental/learned_interface/reset_vectorization_audit_seed3500000_n16.json
```

### Interpretation

The vectorized learned-interface evaluator is not producing matched seed
conditions across `eval_num_envs`. All 16 checked seeds reset to different
state vectors when the env count changes. This explains why same-index outcomes
and candidate rankings are unstable across vectorization settings. For future
promotion gates, either pin the evaluator protocol or treat vectorized and
single-env evaluations as different seed distributions, not matched
comparisons. This further supports using the single-env/serial-compatible
protocol for local-RL base selection.

## 2026-06-27 - Matched-reset vectorized learned-interface diagnostic

### Hypothesis

If the vectorized learned-interface instability is mainly caused by reset-state
mismatch, then forcing a vectorized env to the corresponding `num_envs=1` reset
states should reproduce the single-env evaluator much more closely.

### Implementation

Added `learned-interface-eval --eval-reset-mode serial_state`. In this mode the
evaluator creates a temporary single-env reset for each evaluation seed, stores
`env.unwrapped.get_state()`, resets the vectorized student/branch envs as usual,
then calls `env.unwrapped.set_state(...)` and `get_obs()` before rollout.

The default is still `--eval-reset-mode raw`.

### Commands

Single-env reference:

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 20 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force
```

Vectorized matched-reset diagnostic:

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_strong \
  --goal-source learned \
  --episodes 20 \
  --eval-seed-start 3500000 \
  --eval-num-envs 4 \
  --eval-reset-mode serial_state \
  --force
```

### Results

Artifacts:

- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_20_seed3500000_envs1.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_strong/seed0/learned_hierarchy_eval_20_seed3500000_envs4_resetserial_state.json`

Comparison:

| metric | exact matches | max abs diff | mean abs diff |
| --- | ---: | ---: | ---: |
| episode success | 17 / 20 | 1.0 | 0.1500 |
| episode max reward | 14 / 20 | 0.7086 | 0.1049 |
| episode final reward | 14 / 20 | 0.7390 | 0.1099 |

Aggregate:

| eval mode | num envs | success | final reward | max reward |
| --- | ---: | ---: | ---: | ---: |
| raw | 1 | 0.700 | 0.7699 | 0.7857 |
| serial_state | 4 | 0.850 | 0.8798 | 0.8901 |

Focused probe on seeds `3500000..3500003`:

| check | max abs diff | mean abs diff |
| --- | ---: | ---: |
| raw vector reset state vs serial reset state | 1.1450 | - |
| vector state after `set_state` vs serial reset state | 2.38e-7 | - |
| observation state after `set_state` | 8.34e-7 | 2.97e-8 |
| RGB after `set_state` | 2 pixels | 2.54e-5 |
| DINO/state frame input after `set_state` | 0.0080 | 0.00125 |
| first high-level goal | 0.0065 | 0.00165 |
| first low-level action | 0.00105 | 0.00039 |
| next simulator state after identical first action | 5.65e-4 | 1.62e-5 |

### Verification

```bash
uv run python -m py_compile src/hcl_poc/learned_interface.py src/hcl_poc/cli.py
uv run hcl-poc incremental learned-interface-eval --help
```

### Interpretation

Overwriting vectorized env state is feasible and removes the large reset-state
mismatch, but it is not enough to reproduce the single-env learned-interface
evaluator. Rendering/feature extraction differs slightly after `set_state`, the
first model outputs differ slightly, and even stepping identical first actions
introduces small simulator-state differences in batched mode. These small
differences can compound across a closed-loop rollout.

Treat `--eval-reset-mode serial_state` as an audit/debugging tool only. It does
not replace the single-env/serial-compatible promotion protocol. The
conservative RL base remains `effect32_film_gsens_ft_highact_strong`.

## 2026-06-27 - Larger exact-serial segment-selector replication

### Hypothesis

The previous exact-serial segment selector may have failed online because it
was trained on only 50 episodes. A larger exact-seed train/validation pair might
make the segment-start gate less noisy and improve closed-loop task success.

### Commands

Train-window exact serial evals:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 1000 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_frozen_segmentselector_serial100_seed4510000 \
  --episodes 100 \
  --seed-start 4510000 \
  --distance-metric reachability \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 1000 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4510000 \
  --episodes 100 \
  --seed-start 4510000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt \
  --distance-metric reachability \
  --force
```

Validation-window exact serial evals used the same commands with
`seed-start=4511000` and matching run names.

Selector fit:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  fit-serial-segment-selector \
  --base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial100_seed4510000/serial_eval_100_seed4510000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4510000/serial_eval_100_seed4510000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial100_seed4511000/serial_eval_100_seed4511000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4511000/serial_eval_100_seed4511000.json \
  --output results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4510000/segment_selector_fit_train4510000_valid4511000.json \
  --force
```

Online validation used the fitted weights:

```text
weights:    [0.026956, -0.292065, 0.167119, -0.033937, -0.130110]
mean:       [0.711870, 0.922547, 0.449592, 0.983873, 45.0]
std:        [0.226132, 0.845738, 0.428169, 0.599908, 28.722815]
threshold:  -0.099340
```

### Results

Offline local selector metrics:

| split | segments | base raw reduction | R3 raw reduction | selector raw reduction | selector delta vs base | selector use R3 | selector AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train 4510000 | 1000 | 0.4071 | 0.4198 | 0.4703 | +0.0632 | 0.749 | 0.599 |
| validation 4511000 | 1000 | 0.4407 | 0.4527 | 0.5108 | +0.0701 | 0.714 | 0.594 |

Online validation on `4511000..4511099`:

| policy | success | final reward | max reward | raw local reduction | reach rate | residual L2 | R3 segment use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.660 | 0.6802 | 0.7611 | 0.4407 | 0.714 | 0.000000 | - |
| ungated R3 | 0.650 | 0.6760 | 0.7502 | 0.4527 | 0.713 | 0.001034 | 1.000 |
| online segment selector | 0.660 | 0.6810 | 0.7554 | 0.4307 | 0.722 | 0.000731 | 0.748 |

Paired against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| ungated R3 | 10 | 11 | -1 |
| online segment selector | 11 | 11 | 0 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial100_seed4510000/serial_eval_100_seed4510000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4510000/serial_eval_100_seed4510000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial100_seed4511000/serial_eval_100_seed4511000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4511000/serial_eval_100_seed4511000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_serial100_seed4510000/segment_selector_fit_train4510000_valid4511000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_segmentselector_fit4510000_online_serial100_seed4511000/serial_eval_100_seed4511000.json`

### Interpretation

The larger exact-serial dataset reproduces the old pattern. The segment-start
linear selector improves held-out local raw reduction offline, but online
deployment only ties frozen task success and does not preserve the offline local
raw-reduction gain. The deployed selector changes later states and goals, so
the completed-segment labels remain a poor online intervention target.

This further rejects simple offline linear segment gating for the current R3
checkpoint. The next selector attempt needs closed-loop intervention training
or a larger/more task-aligned residual effect, not just more exact serial
segments for the same five-feature linear gate.

## 2026-06-27 - Effect32 long-credit task-reward R3 diagnostic

### Hypothesis

The high-action task-reward diagnostic showed that cutting GAE at every held-goal
segment can make dense task-reward R3 too myopic. The original `effect32_film`
base is still the canonical real-compatible effect-latent checkpoint, so I
tested whether a one-update task-reward R3 run with 50-step rollouts and
`segment_terminates_gae=False` gives a stronger deployment-aligned effect there.

### Command

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_taskreward_2048_roll50_102k_bc1_noseggae \
  --steps 102400 \
  --num-envs 2048 \
  --rollout-steps 50 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --bc-weight 1 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --task-reward-weight 1.0 \
  --reward-mode absolute \
  --distance-metric reachability \
  --no-segment-terminate-gae \
  --force
```

Exact serial deployment smoke:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 1000 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_taskreward_roll50_102k_bc1_noseggae_serial100_seed4511000 \
  --episodes 100 \
  --seed-start 4511000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_taskreward_2048_roll50_102k_bc1_noseggae/latest.pt \
  --distance-metric reachability \
  --force
```

### Results

Training produced one update:

| global step | mean reward | terminal D_phi | action saturation | BC loss |
| ---: | ---: | ---: | ---: | ---: |
| 102400 | 0.2450 | 0.5901 | 0.075 | 5.17e-7 |

Exact serial validation on `4511000..4511099`:

| policy | success | final reward | max reward | raw local reduction | reach rate | residual L2 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.660 | 0.6802 | 0.7611 | 0.4407 | 0.714 | 0.000000 | 0.0357 |
| terminal D_phi R3 | 0.650 | 0.6760 | 0.7502 | 0.4527 | 0.713 | 0.001034 | 0.0339 |
| task-reward roll50 R3 | 0.650 | 0.6409 | 0.7466 | 0.4347 | 0.706 | 0.001128 | 0.0394 |

Paired against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| terminal D_phi R3 | 10 | 11 | -1 |
| task-reward roll50 R3 | 13 | 14 | -1 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_taskreward_2048_roll50_102k_bc1_noseggae/latest.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_taskreward_2048_roll50_102k_bc1_noseggae/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_taskreward_roll50_102k_bc1_noseggae_serial100_seed4511000/serial_eval_100_seed4511000.json`

### Interpretation

The long-credit dense task-reward update is not a promotion candidate on
`effect32_film`. It ties terminal-D_phi R3 on success in this smoke but has much
worse final reward, lower raw local reduction, and lower reach rate. The paired
win/regression balance is also neutral-negative. This rejects the simple
effect32 version of the long-credit task-reward objective; the objective problem
is not fixed by letting dense task reward backpropagate across five held-goal
segments.

The next objective change needs either a different target distribution or a
larger closed-loop/intervention training setup, not another one-update dense
task-reward variant of the same direct-low R3 recipe.

## 2026-06-27 - Effect32 long-credit paired terminal reward check

The previous paired terminal reward runs used one-segment PPO rollouts
(`rollout_steps=10`). Since the task-reward roll50 variant also failed, I ran
the cleaner paired terminal objective with five held-goal segments of credit
assignment and segment-boundary GAE disabled:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film \
  --n-demo 1000 \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_paired_2048_roll50_102k_bc10_noseggae \
  --steps 102400 \
  --num-envs 2048 \
  --rollout-steps 50 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --bc-weight 10 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --no-segment-terminate-gae \
  --force
```

Exact serial deployment smoke:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 1000 \
  --candidate effect32_film \
  --seed 0 \
  --run-name hcl_next_effect32_dphi_r3_paired_roll50_102k_bc10_noseggae_serial100_seed4511000 \
  --episodes 100 \
  --seed-start 4511000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_roll50_102k_bc10_noseggae/latest.pt \
  --distance-metric reachability \
  --force
```

### Results

Training produced one synchronized paired update:

| global step | mean paired improvement | improved segments | terminal D_phi | base terminal D_phi | resync events | desynced envs | BC loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 102400 | 0.00592 | 0.509 | 0.5901 | 0.5960 | 0 | 0 | 4.68e-7 |

Exact serial validation on `4511000..4511099`:

| policy | success | final reward | max reward | raw local reduction | reach rate | residual L2 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.660 | 0.6802 | 0.7611 | 0.4407 | 0.714 | 0.000000 | 0.0357 |
| terminal D_phi R3 | 0.650 | 0.6760 | 0.7502 | 0.4527 | 0.713 | 0.001034 | 0.0339 |
| task-reward roll50 R3 | 0.650 | 0.6409 | 0.7466 | 0.4347 | 0.706 | 0.001128 | 0.0394 |
| paired roll50 R3 | 0.620 | 0.6091 | 0.7256 | 0.4180 | 0.708 | 0.001067 | 0.0362 |

Paired against frozen:

| policy | improvements | regressions | net |
| --- | ---: | ---: | ---: |
| paired roll50 R3 | 6 | 10 | -4 |

Artifacts:

- `artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_roll50_102k_bc10_noseggae/latest.pt`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_2048_roll50_102k_bc10_noseggae/train_metrics.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_roll50_102k_bc10_noseggae_serial100_seed4511000/serial_eval_100_seed4511000.json`
- `results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_roll50_102k_bc10_noseggae_serial100_seed4511000/paired_vs_frozen_serial100_seed4511000.json`

### Interpretation

Long-credit paired terminal reward gives a small positive synchronized training
signal, but it transfers worse than both frozen and the earlier one-segment
paired/terminal-D_phi variants on the matched validation slice. The failure is
not from paired rollout desynchronization in this run; the recorded resync and
desynced counts are both zero.

This rejects the simple "same paired terminal objective, longer GAE horizon"
branch for `effect32_film`. The next objective change should not be another
rollout-length tweak of direct-low R3; it needs a different intervention
distribution or a deployment-level training/selection loop.

## 2026-06-27 - High-action paired R3 segment-selector audit

The high-action pairedsync checkpoint is the one branch where the hindsight
episode selector showed real complementarity, but the three-feature initial
selector was not robust enough. I tested the existing five-feature segment-start
selector on the learned-goal high-action exact serial windows.

### Commands

Fit on `3500000..3500099`, validate offline on `3600000..3600099`:

```bash
uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  fit-serial-segment-selector \
  --base-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json \
  --candidate-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3500000/serial_eval_100_seed3500000.json \
  --validation-base-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_frozen_serial100_seed3600000/serial_eval_100_seed3600000.json \
  --validation-candidate-json results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10_serial100_seed3600000/serial_eval_100_seed3600000.json \
  --output results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_segment_selector_learned_train3500000_valid3600000.json \
  --force
```

Deploy the selector online on the validation window:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_segmentselector_train350_online_serial100_seed3600000 \
  --episodes 100 \
  --seed-start 3600000 \
  --checkpoint artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_2048_terminal_40k_bc10/best_train_latent.pt \
  --segment-selector-weights -0.1537081748 -0.1537081748 0.1060052738 0.0405058116 -0.0682906508 \
  --segment-selector-mean 0.9115276933 0.9115276933 0.4361685514 0.9485784769 45.0000000000 \
  --segment-selector-std 0.8321719170 0.8321719170 0.4352999032 0.6072300076 28.7228145599 \
  --segment-selector-threshold -0.0300638266 \
  --force
```

### Results

Offline segment-selector local metric:

| split | segments | base raw reduction | R3 raw reduction | selector raw reduction | selector delta vs base | selector use R3 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train 3500000 | 1000 | 0.3890 | 0.3963 | 0.4595 | +0.0705 | 0.651 | 0.603 |
| validation 3600000 | 1000 | 0.3732 | 0.3879 | 0.4284 | +0.0552 | 0.717 | 0.615 |

Online exact serial validation on `3600000..3600099`:

| policy | success | final reward | max reward | raw local reduction | reach rate | residual L2 | selector use R3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.770 | 0.7199 | 0.8385 | 0.3732 | 0.791 | 0.000000 | - |
| ungated pairedsync R3 | 0.680 | 0.6704 | 0.7739 | 0.3879 | 0.777 | 0.005424 | - |
| segment selector | 0.710 | 0.6628 | 0.7937 | 0.3869 | 0.784 | 0.004284 | 0.793 |

Paired counts:

| comparison | improvements | regressions | net | success delta |
| --- | ---: | ---: | ---: | ---: |
| selector vs frozen | 7 | 13 | -6 | -0.060 |
| selector vs ungated R3 | 5 | 2 | +3 | +0.030 |

Artifacts:

- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/highact_pairedsync_segment_selector_learned_train3500000_valid3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_segmentselector_train350_online_serial100_seed3600000/serial_eval_100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_segmentselector_train350_online_serial100_seed3600000/paired_vs_frozen_serial100_seed3600000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_segmentselector_train350_online_serial100_seed3600000/paired_vs_ungated_serial100_seed3600000.json`

### Interpretation

The richer segment-start selector partially rescues the high-action pairedsync
checkpoint relative to ungated R3, but it still trails frozen by six net
successes and lower final/max reward. The offline local gain again does not map
cleanly to deployment success. This keeps the general selector diagnosis
unchanged: retrospective local or segment-start linear selectors are not enough;
the next selector attempt needs direct closed-loop/intervention training or a
substantially richer online policy.

## 2026-06-27 - High-action candidate-specific D_phi identity check

The high-action pairedsync R3 runs used the base
`effect32_film/seed0/d_phi.pt` reachability checkpoint. Since cached
reachability latents also existed under
`effect32_film_gsens_ft_highact_strong`, I checked whether training a
candidate-specific D_phi changed the reward metric.

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  train-reachability-distance \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc rl-rerun \
  --config configs/pusht_incremental.yaml \
  eval-reachability-distance \
  --candidate effect32_film_gsens_ft_highact_strong \
  --seed 0 \
  --force
```

I then reran the high-action pairedsync 40k recipe with only the checkpoint path
changed:

```bash
TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  train-r3 \
  --candidate effect32_film_gsens_ft_highact_strong \
  --n-demo 500 \
  --seed 0 \
  --run-name hcl_next_highact_strong_r3_pairedsync_candidate_dphi_2048_terminal_40k_bc10 \
  --steps 40960 \
  --num-envs 2048 \
  --rollout-steps 10 \
  --num-minibatches 16 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --initial-logstd -1.8 \
  --bc-weight 10.0 \
  --terminal-weight 1.0 \
  --distance-progress-weight 0.0 \
  --reward-mode paired \
  --distance-metric reachability \
  --reachability-checkpoint artifacts/incremental/reachability_distance/effect32_film_gsens_ft_highact_strong/seed0/d_phi.pt \
  --force
```

### Results

Reachability eval:

| D_phi | temporal MSE | temporal Spearman | near/far acc | shuffled AUC | demo decrease acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| base effect32 | 0.03267 | 0.8333 | 0.9275 | 0.9075 | 0.7396 |
| high-action path | 0.03216 | 0.8321 | 0.9219 | 0.9083 | 0.7327 |

The apparent metric difference is not a real independent checkpoint:

| comparison | result |
| --- | ---: |
| D_phi model max tensor delta | 0.0 |
| D_phi model mean tensor delta | 0.0 |
| paired R3 best-agent max tensor delta vs old base-D_phi pairedsync run | 0.0 |
| paired R3 best-agent mean tensor delta vs old base-D_phi pairedsync run | 0.0 |

The cached reachability metadata explains why:

```text
base candidate:      effect32_film
high-action path:    effect32_film_gsens_ft_highact_strong
representation ckpt: artifacts/incremental/learned_interface/effect32/seed0/representation.pt
encoder_type:        effect
```

Both D_phi checkpoints train from the same effect representation and identical
encoded goal caches. The R3 training history is consequently identical to the
old pairedsync run:

| global step | mean paired improvement | fraction improved | tuned terminal D_phi | base terminal D_phi | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20480 | 0.0907 | 0.5869 | 0.4029 | 0.4935 | 0.4062 |
| 40960 | 0.0161 | 0.4854 | 0.5889 | 0.6049 | 0.1917 |

Artifacts:

- `artifacts/incremental/reachability_distance/effect32_film_gsens_ft_highact_strong/seed0/d_phi.pt`
- `results/incremental/reachability_distance/effect32_film_gsens_ft_highact_strong/seed0/eval.json`
- `artifacts/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_candidate_dphi_2048_terminal_40k_bc10/best_train_latent.pt`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_strong/seed0/hcl_next_highact_strong_r3_pairedsync_candidate_dphi_2048_terminal_40k_bc10/train_metrics.json`

### Interpretation

This closes the "metric mismatch" hypothesis for the high-action pairedsync
branch. The candidate-specific D_phi path is just a different artifact location
for the same learned effect-space distance. The high-action R3 failure is not
explained by accidentally using the base `effect32_film` D_phi checkpoint; the
next reward/representation experiment needs a genuinely different encoder or
distance target, not another alias over the same effect representation.

## 2026-06-27 - High-level oracle-action anchor diagnostic

The previous joint high/low anchor only constrained the trainable low policy on
offline low-level inputs. It did not preserve the deployment behavior that made
`highact_strong` useful. I added a deployment-closer high-level anchor: while
fine-tuning the high level through the frozen goal-sensitive low policy, penalize
the predicted-goal low action for drifting away from the same low policy's action
under the demonstration/oracle future goal.

### Code change

`train_learned_interface_hierarchy` now accepts:

```text
high_oracle_action_anchor_weight
```

When this weight is positive, the high-level update adds:

```text
high_oracle_action_anchor_weight *
MSE(low(current, predicted_high_goal), low(current, oracle_future_goal))
```

The low policy remains frozen for this loss, so the term updates only the high
policy and directly anchors the predicted goal's induced action to oracle-goal
low behavior.

### Candidate

```yaml
effect32_film_gsens_ft_highact_oracleanchor:
  family: conditioning_ablation
  representation_candidate: effect32
  high_level_candidate: effect32_film_gsens_ft_highact_oracleanchor
  conditioning: film
  high_init_candidate: effect32
  low_init_candidate: effect32_film_gsens_ft
  freeze_low_policy: true
  high_goal_mse_weight: 0.0
  high_action_loss_weight: 300.0
  high_oracle_action_anchor_weight: 300.0
  policy_lr: 1.0e-5
  policy_epochs: 20
```

### Commands

```bash
TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-train-hierarchy \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_oracleanchor \
  --seed 0 \
  --force

TQDM_DISABLE=1 uv run hcl-poc incremental learned-interface-eval \
  --config configs/pusht_incremental.yaml \
  --candidate effect32_film_gsens_ft_highact_oracleanchor \
  --goal-source learned \
  --episodes 100 \
  --eval-seed-start 3500000 \
  --eval-num-envs 1 \
  --force

TQDM_DISABLE=1 uv run hcl-poc low-level-rl \
  --config configs/pusht_incremental.yaml \
  eval-serial \
  --n-demo 500 \
  --candidate effect32_film_gsens_ft_highact_oracleanchor \
  --seed 0 \
  --run-name hcl_next_highact_oracleanchor_frozen_serial100_seed3500000 \
  --episodes 100 \
  --seed-start 3500000 \
  --force
```

### Results

Offline validation:

| candidate | normalized goal L2 | oracle action MAE | predicted action MAE | predicted-vs-oracle action L2 |
| --- | ---: | ---: | ---: | ---: |
| highact_strong | 2.5489 | 0.0362 | 0.0366 | 0.0157 |
| actiononly | 2.6673 | 0.0362 | 0.0366 | 0.0180 |
| oracleanchor | 2.5576 | 0.0362 | 0.0365 | 0.0126 |

The anchor achieved the intended offline effect: it reduced predicted-vs-oracle
low-action drift while preserving the one-step predicted action MAE.

Conservative single-env learned-interface screen on `3500000..3500099`:

| candidate | success | final reward | max reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: |
| highact_strong | 0.670 | 0.7492 | 0.7566 | 0.0937 |
| actiononly | 0.660 | 0.7497 | 0.7562 | 0.0893 |
| goal01 | 0.660 | 0.7479 | 0.7536 | 0.0840 |
| oracleanchor | 0.630 | 0.7246 | 0.7349 | 0.0979 |

Matching exact serial screen:

| candidate | success | final reward | max reward | raw local reduction | reach rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| highact_strong | 0.670 | 0.7092 | 0.7566 | 0.3890 | 0.722 |
| actiononly | 0.660 | 0.6161 | 0.7562 | 0.3721 | 0.742 |
| goal01 | 0.660 | 0.6742 | 0.7536 | 0.3786 | 0.745 |
| oracleanchor | 0.630 | 0.6780 | 0.7349 | 0.3803 | 0.737 |

Paired against `highact_strong`, oracleanchor had `9` improvements,
`13` regressions, net `-4`, and success delta `-0.04`.

Artifacts:

- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_oracleanchor/seed0/hierarchy.pt`
- `artifacts/incremental/learned_interface/effect32_film_gsens_ft_highact_oracleanchor/seed0/hierarchy_metrics.json`
- `results/incremental/learned_interface/effect32_film_gsens_ft_highact_oracleanchor/seed0/learned_hierarchy_eval_100_seed3500000_envs1.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_oracleanchor/seed0/hcl_next_highact_oracleanchor_frozen_serial100_seed3500000/serial_eval_100_seed3500000.json`
- `results/incremental/low_level_rl/effect32_film_gsens_ft_highact_oracleanchor/seed0/hcl_next_highact_oracleanchor_frozen_serial100_seed3500000/paired_vs_highact_strong_serial100_seed3500000.json`

### Interpretation

The oracle-action anchor improves the offline action-preservation proxy but
hurts deployment. This rejects the simple frozen-low high-level anchor variant:
keeping predicted-goal actions close to oracle-goal actions in the offline
held-goal data is not enough to preserve closed-loop/serial behavior. The
conservative serial/RL base remains `effect32_film_gsens_ft_highact_strong`.
The next coupled objective needs a stronger deployment-level signal, not another
one-step action-space anchor.

## 2026-06-27 - Prefix counterfactual selector grouped-loss check

### Hypothesis

The prefix-feature counterfactual branch selector previously used return
regression on each candidate. Since deployment chooses one candidate per query,
a grouped best-candidate cross-entropy objective might better match the branch
selection problem.

### Command

I trained five seeds on the existing prefix counterfactual bank:

```bash
for s in 0 1 2 3 4; do
  TQDM_DISABLE=1 uv run scripts/train_privileged_z_counterfactual_selector.py \
    --input data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9963000_q128_k8_prefix.npz \
    --output artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_bestce_seed${s}.pt \
    --seed ${s} \
    --epochs 200 \
    --batch-size 1024 \
    --hidden-dim 128 \
    --depth 2 \
    --learning-rate 1e-3 \
    --loss best_ce
done
```

### Results

Validation metrics across five random query splits:

| seed | selected return delta | selected success delta | nearest return delta | nearest success delta | oracle return delta | oracle success delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -5.387 | -0.062 | -1.266 | +0.094 | +13.002 | +0.250 |
| 1 | +2.606 | +0.031 | +1.551 | +0.000 | +18.778 | +0.281 |
| 2 | -0.821 | -0.031 | +3.524 | +0.000 | +18.848 | +0.250 |
| 3 | +1.893 | +0.062 | +8.922 | +0.156 | +16.651 | +0.281 |
| 4 | -0.361 | -0.062 | -2.383 | -0.094 | +10.373 | +0.125 |
| mean | -0.414 | -0.013 | +2.070 | +0.031 | +15.530 | +0.237 |

Artifacts:

- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_bestce_seed0.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_bestce_seed1.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_bestce_seed2.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_bestce_seed3.pt`
- `artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_bestce_seed4.pt`

### Interpretation

Grouped best-candidate training does not rescue the prefix-feature static
selector. It is marginally better than the previous regression selector on
mean selected return (`-0.414` versus `-0.517`), but selected success remains
negative and it still loses clearly to nearest-candidate selection. The large
oracle best-of-8 gap remains, so the candidate set has useful alternatives, but
this small offline scorer cannot choose them reliably. This further supports
the current direction: branch/counterfactual work needs broader query coverage,
different candidate generation, or an online/intervention-trained selector,
not another static per-candidate scoring loss on this bank.
