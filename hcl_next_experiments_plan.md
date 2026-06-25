# Next Experiments for Hierarchical Control from Latent / Reachability RL

**Project:** hierarchical future-state / future-effect imitation learning with local RL fine-tuning on Push-T proof of concept.  
**Main goal:** learn a hierarchical policy from limited demonstrations, then improve the low-level controller with RL using a *reachability-style reward*, without requiring a hand-designed task reward.  
**Target future application:** longer real-robot manipulation task with mostly real robot demonstrations, limited data, difficult or impossible full-task RL, and no access to complete privileged object state at deployment.

---

## 0. Executive summary for a new person

This repository currently tests a hierarchical control idea on the simplified ManiSkill `PushT-v1` environment. The long-term target is not Push-T itself; Push-T is a debugging environment for a much harder real-robot task where:

- flat RL from scratch is expected to be too sample inefficient;
- imitation learning from limited demonstrations should produce a reasonable initial policy;
- a hierarchy may help by separating slow future-goal selection from fast local control;
- the low-level controller should later be improved with RL using only a reachability reward, not a full task reward.

The current hierarchy is approximately:

```text
observation encoder: z_t = E(o_t)

high level:          H(o_t, a_{t-1}) -> goal latent z_{t+k}

low level:           pi(o_t, z_goal, a_{t-1}, remaining_time) -> primitive action a_t
```

The current best imitation results are decent: the learned hierarchy is competitive with flat baselines. The failure is the RL fine-tuning stage. The most important current diagnosis is that the supervised low-level policy can minimize one-step imitation loss while mostly ignoring the future goal. In the learned-latent experiments, changing the future goal caused tiny action changes compared with changing the current observation. Therefore PPO/reachability fine-tuning has little useful goal-conditioned behavior to improve.

The next experiments should **not** be small hyperparameter sweeps of the current VAE-latent PPO setup. They should test the core assumptions:

1. Can reachability RL train or improve a local goal-conditioned policy at all?
2. Does the low-level policy actually use the future goal?
3. Is raw VAE latent L2 the wrong reward metric?
4. Can a learned reachability distance from trajectories replace `||z_reached - z_goal||`?
5. Is a future-effect latent better than a future-state latent?
6. Can the method be made compatible with mostly real robot trajectories, where privileged rock/object poses are unavailable?

A key implementation rule for all RL experiments: **use many parallel environments**. For serious PPO runs, use **4096 or more vectorized environments** if the simulator/GPU can handle it. Do not draw conclusions from 32, 64, or a few hundred envs except for smoke tests and debugging.

---

## 1. Repository/code structure overview

The project uses `uv` and the CLI entry point:

```bash
uv run hcl-poc <command> ...
```

Important files/directories:

```text
configs/
  pusht.yaml
  pusht_incremental.yaml
  pusht_spatial*.yaml

src/hcl_poc/
  cli.py                    # CLI definitions; add new experiment commands here
  config.py                 # config loading
  data.py                   # dataset preparation and feature loading
  features.py               # visual/proprio feature helpers
  models.py                 # core model modules
  flow.py                   # flow-matching utilities
  learned_interface.py      # VAE/learned interface training and eval
  privileged_z.py           # privileged-state hierarchy experiments
  low_level_rl.py           # older low-level residual/direct RL experiments
  rl_rerun.py               # newer exact-reset/local-rerun RL formulation
  rl.py                     # privileged PPO teacher training/eval
  incremental.py            # many staged imitation/representation experiments
  vae_scaling.py            # final VAE512 sample-efficiency sweep

artifacts/
  trained models/checkpoints

data/
  generated datasets/manifests

results/
  raw JSON metrics, eval outputs, plots, videos

logs*.txt, *_experiment_log.md, *_final_results.md
  chronological experiment logs and summaries
```

Useful existing commands/examples:

```bash
# Environment sanity check
uv run hcl-poc doctor

# VAE512 final imitation sweep commands are documented in README.md
uv run hcl-poc incremental vae-scaling-manifests --config configs/pusht_incremental.yaml
scripts/run_vae_scaling_sweep.sh train
scripts/run_vae_scaling_sweep.sh eval

# Existing RL rerun audits and datasets
uv run hcl-poc rl-rerun local-reset-audit --config configs/pusht_incremental.yaml
uv run hcl-poc rl-rerun algorithm-audit --config configs/pusht_incremental.yaml
uv run hcl-poc rl-rerun throughput-benchmark --config configs/pusht_incremental.yaml --num-envs 512,1024,2048,4096,8192

# Existing privileged-z training/eval
uv run hcl-poc rl-rerun train-privileged-z --config configs/pusht_incremental.yaml ...
uv run hcl-poc rl-rerun eval-privileged-z --config configs/pusht_incremental.yaml ...
uv run hcl-poc rl-rerun train-privileged-z-residual --config configs/pusht_incremental.yaml ...
```

When adding new experiments, prefer adding new subcommands under `rl-rerun` or `incremental` instead of one-off notebooks. Every experiment should write a JSON result file with enough metadata to reproduce it.

---

## 2. Current known results and failure modes

### 2.1 Imitation learning is not the main failure

The VAE512 hierarchy is competitive with flat baselines in the final sample-efficiency experiment. At 8k demonstrations, the deterministic hierarchy reached about `0.69` success, similar to the reachable branch oracle and modestly above flat deterministic baselines. However, across the full data-efficiency curve, flat full-observation deterministic control is still comparable or slightly better by area-under-curve. Therefore the honest current imitation conclusion is:

> The hierarchy can match flat BC on Push-T and may help at high data budgets, but Push-T is too short/simple to prove strong hierarchical sample-efficiency gains.

### 2.2 RL fine-tuning with learned VAE latent goals is the main failure

The learned-latent PPO/rerun experiments generally do not improve task success. Diagnostics show that the low level barely reacts to the goal:

```text
observation shuffle action change: ~0.805 L2
goal shuffle action change:       ~0.048-0.050 L2
valid k=2/5/10/20 goals:          goal L2 ~24-27, action change only ~0.016-0.022
learned-vs-oracle goal swap:      goal L2 ~25, action change only ~0.033
```

Interpretation:

- The supervised low level can solve one-step BC mostly from the current observation and previous action.
- The future goal is statistically redundant in the current dataset/formulation.
- A policy that ignores the goal cannot be reliably improved by a goal-distance reward.

### 2.3 Privileged-state experiments show the idea is not dead

When the latent is replaced by 31D privileged Push-T state, valid-goal sensitivity becomes much larger. Early privileged-state experiments looked bad, but a major train/deployment mismatch was found: the low level was initially trained only on the first action of a 10-step held-goal segment while deployed on all 10 offsets.

After fixing this with **multi-offset training**, results improved substantially:

```text
old privileged clean 500-stream run:   ~0-3% hierarchy success
multi-offset clean 500 streams:        ~8% hierarchy, ~40% oracle-goal hierarchy
multi-offset clean 1800 streams:       ~53% hierarchy, ~72% oracle-goal hierarchy
```

The key lesson:

> If the low level is deployed with a held goal for k steps, it must be trained on all held-goal offsets `i = 0..k-1`, not only the first offset.

### 2.4 Raw VAE L2 is probably not a good reachability reward

The current learned-latent reward assumes:

```math
D(z_t, z_g) = ||z_t - z_g||_2
```

A VAE latent may be smooth for reconstruction, but it is not guaranteed to align with:

- control effort;
- number of steps to reach a state;
- contact changes;
- object pose progress;
- local controllability;
- action-conditioned reachability.

The next experiments should keep a compact high-level latent goal, but replace raw Euclidean VAE distance with a learned distance/value/reachability function.

---


## 3. Pre-experiment prerequisites: data regimes, checkpoints, and baselines

Before running the new RL/reachability experiments, first create a clean set of imitation checkpoints and evaluation banks. A new person should not start by launching PPO. They should first reproduce the supervised imitation stack and confirm the frozen base policies are healthy.

### 3.1 Standard demonstration counts

Use two main data regimes throughout the proof of concept:

```text
N = 500 trajectories      # limited-data regime: imitation has some success, but is not saturated
N = 1800 trajectories     # plenty-data regime: imitation is much stronger and less noisy
```

These two regimes should be used consistently across representation, architecture, and RL experiments.

Interpretation:

- `N=500` is the important limited-data setting. It is the closest Push-T analogue to the final thesis situation: limited real-robot demonstrations, some imitation success, but room for improvement.
- `N=1800` is the stability/upper-data setting. It should reduce noise and tell us whether a formulation works when the imitation policy is already reasonably competent.
- Optional larger settings such as `8k` demos are useful for final scaling plots, but they are not the main debugging regimes for the next experiments.

When reporting results, always indicate the data count in the table:

```text
representation | architecture | N demos | seed | success | local reach metric | notes
```

Do not mix `N=500` and `N=1800` checkpoints in the same comparison unless explicitly stated.

### 3.2 Fixed splits and evaluation banks

For each `N`, create and save fixed train/validation/evaluation splits:

```text
data/manifests/pusht_n500_seed0_train.json
data/manifests/pusht_n500_seed0_val.json
data/manifests/pusht_n500_seed0_eval.json

data/manifests/pusht_n1800_seed0_train.json
data/manifests/pusht_n1800_seed0_val.json
data/manifests/pusht_n1800_seed0_eval.json
```

Also save fixed local-reset banks for local reaching and RL rerun evaluations:

```text
local_reset_bank_n500_seed0_k2.json
local_reset_bank_n500_seed0_k5.json
local_reset_bank_n500_seed0_k10.json
local_reset_bank_n500_seed0_k20.json

local_reset_bank_n1800_seed0_k2.json
local_reset_bank_n1800_seed0_k5.json
local_reset_bank_n1800_seed0_k10.json
local_reset_bank_n1800_seed0_k20.json
```

The local reset bank should store enough information to reproduce the exact local start state, the replay/oracle future goal, the previous action, and metadata about the source trajectory/time index.

Use fresh held-out evaluation seeds for final closed-loop task success. Do not tune directly on the final eval bank.

### 3.3 What must be trained before running each experiment family

The following supervised/frozen artifacts should exist before launching serious RL.

#### A. Flat imitation baselines

Train at both `N=500` and `N=1800`:

```text
flat_full_obs_deterministic
flat_latent_deterministic, if using latent observations
optional flat_flow or Gaussian baseline, only after deterministic baseline is stable
```

Purpose:

- establish the non-hierarchical imitation baseline;
- measure whether the hierarchy actually helps imitation;
- provide a sanity check for full-task success.

Required eval:

```text
closed-loop task success
mean return
action MAE on validation set
videos for representative successes/failures
```

#### B. Representation encoders / goal encoders

Depending on the experiment, train or prepare:

```text
privileged state encoder: identity / normalization only
VAE512 encoder: existing E_VAE(o) baseline
reachability encoder E_theta(o): Experiment E2
future-goal encoder G(o_{t+k}): Experiment F1
future-effect encoder E_effect(o_t, o_{t+k}, k): Experiment F2
```

For learned encoders, save:

```text
encoder checkpoint
normalization statistics
training config
validation metrics
embedding diagnostics
```

Do not use a learned encoder in RL until its offline diagnostics pass the relevant checks, especially temporal ordering and correct-vs-shuffled separation for reachability/effect latents.

#### C. Supervised hierarchical checkpoints

For every representation and architecture being tested, train the hierarchy at both data regimes:

```text
N=500:
  high-level H
  low-level pi_low concat
  low-level pi_low FiLM/gated
  optional base+goal-residual low-level

N=1800:
  same set
```

Training must use the multi-offset held-goal dataset:

```text
for i in 0..k-1:
    input  = [current obs/state at t+i, held goal at t+k, previous action, remaining time]
    target = action at t+i
```

For each checkpoint, evaluate:

```text
action MAE
closed-loop learned-high hierarchy success
closed-loop oracle-goal hierarchy success
correct-goal vs shuffled-goal local reaching
goal sensitivity diagnostics
```

Only checkpoints that pass the goal-identifiability gate should be used as RL bases.

#### D. Frozen base policies for paired reward

For paired-improvement RL, cache the frozen base policy rollouts on the local reset banks:

```text
base_final_state
base_final_observation
base_final_goal_distance
base_action_sequence
base_success_within_epsilon
```

Cache these separately for:

```text
N=500, k=2/5/10/20
N=1800, k=2/5/10/20
each representation
each low-level architecture
```

This avoids repeatedly rolling out the frozen policy inside every PPO update and makes paired reward/evaluation reproducible.

#### E. Reachability distance checkpoints, when used

Experiment E introduces `D_phi`. Before using it as an RL reward, train and validate:

```text
D_phi over VAE512 latent            # E1
joint encoder + D_phi               # E2
optional Bellman-style D_phi         # E3, later
```

Required validation before RL:

```text
temporal distance correlation
reachable-within-k classification AUC
D_phi decreases along teacher rollouts
D_phi separates correct future from shuffled future
D_phi correlates with privileged/TCP local reaching where available
```

Then rerun Experiments C/D using `D_phi` as the distance metric.

### 3.4 Recommended checkpoint naming

Use names that make comparisons unambiguous:

```text
artifacts/pusht/n500/seed0/vae512/film/k10/low_bc.pt
artifacts/pusht/n500/seed0/vae512/film/k10/high_bc.pt
artifacts/pusht/n500/seed0/vae512/film/k10/goal_diag.json
artifacts/pusht/n500/seed0/vae512/film/k10/base_rollouts.pt

artifacts/pusht/n1800/seed0/privileged/film/k10/low_bc.pt
artifacts/pusht/n1800/seed0/reachability_e2/film/k10/d_phi.pt
```

At minimum, every result path should encode:

```text
environment, N, seed, representation, architecture, horizon k, reward type, RL variant
```

### 3.5 Minimal supervised pretraining matrix

Do not start with the full Cartesian product. Start with this matrix:

| representation | architecture | N demos | horizon | purpose |
| --- | --- | ---: | ---: | --- |
| privileged/TCP | FiLM | 500 | 10 | clean RL sanity base |
| privileged/TCP | FiLM | 1800 | 10 | strong RL sanity base |
| VAE512 | concat | 500 | 10 | current-style weak baseline |
| VAE512 | concat | 1800 | 10 | current-style strong-data baseline |
| VAE512 | FiLM | 500 | 10 | test architecture fix |
| VAE512 | FiLM | 1800 | 10 | test architecture fix |
| VAE512 + D_phi | FiLM | 500 | 10 | learned-distance test after E1 |
| VAE512 + D_phi | FiLM | 1800 | 10 | learned-distance test after E1 |
| effect/reachability latent | FiLM | 500 | 10 | real-compatible formulation |
| effect/reachability latent | FiLM | 1800 | 10 | real-compatible formulation |

After this works, run horizon sweeps `k in {2,5,10,20}` only for the most promising representations/architectures.


## 4. Non-negotiable rules for all next experiments

### 3.1 Use enough parallel environments

For serious PPO/RL runs:

```text
num_envs >= 4096
rollout_steps usually 10 for local horizon k=10
samples_per_update >= 40960
```

Use 32/64/128 envs only for:

- smoke tests;
- debugging shape errors;
- checking videos;
- quick validation of a new command.

Do not use small-env runs to decide whether an RL formulation works. Previous runs show that 4096-env PPO made residual policies meaningfully more active than 512-env runs.

Before every new RL family, run or update a throughput benchmark:

```bash
uv run hcl-poc rl-rerun throughput-benchmark \
  --config configs/pusht_incremental.yaml \
  --num-envs 512,1024,2048,4096,8192
```

Pick the largest stable value. If `8192` is stable and not memory-bound, use it. Otherwise use `4096`.

### 3.2 Always separate smoke, dev, and final eval

Use three levels:

```text
smoke:     1-2 updates, small episodes, may use 64-512 envs
dev:       real run, 4096+ envs, 100-300 eval episodes
final:     multi-seed, fresh eval seed bank, 500+ eval episodes where possible
```

Do not report smoke numbers as conclusions.

### 3.3 Every RL experiment must compare against the frozen base policy

For every local RL run, report:

```text
frozen base local final distance
RL-tuned local final distance
delta = base - tuned
base task success
RL task success
base oracle-goal success
RL oracle-goal success
```

The main claim is improvement over imitation, so every metric should be paired against the frozen imitation policy.

### 3.4 Always test correct-goal vs shuffled-goal

A useful low level must perform worse when goals are shuffled.

Required evaluations:

```text
correct oracle/replay goal
shuffled goal within batch
no-goal/zero-goal ablation when applicable
predicted high-level goal
```

If correct and shuffled goals perform similarly, the low level is not genuinely goal-conditioned. Do not run expensive RL until this is fixed.

### 3.5 Always use multi-offset low-level training

For horizon `k`, a demonstration segment is:

```text
s_t, s_{t+1}, ..., s_{t+k}
a_t, a_{t+1}, ..., a_{t+k-1}
```

The low-level training samples must include all offsets:

```text
for i in 0..k-1:
    input  = [current_obs/state at t+i,
              held_goal at t+k,
              previous_action at t+i,
              remaining_time = (k-i)/k]
    target = action at t+i
```

Do not train only the first action of each segment.

### 3.6 Log action saturation and residual magnitude

If actions are sampled from a Gaussian and then clamped, PPO gradients may not match executed actions. Log:

```text
fraction of action dimensions clipped/saturated
mean action norm
mean residual norm if using residual policy
mean residual/base action ratio
policy log_std
KL, entropy, clip fraction, approximate KL
value loss and explained variance
```

Saturation above roughly 5-10% is suspicious.

---

## 5. Metrics and result schema

Every experiment should write a JSON file with at least:

```json
{
  "experiment_name": "...",
  "git_commit": "...",
  "config_path": "configs/pusht_incremental.yaml",
  "seed": 0,
  "num_train_trajectories": 500,
  "num_envs": 4096,
  "rollout_steps": 10,
  "total_steps": 1048576,
  "dataset": "...",
  "checkpoint_paths": {"base": "...", "tuned": "..."},
  "representation": "vae512 | privileged_state | temporal_reachability | effect_latent | ...",
  "goal_source_train": "oracle | predicted | replay | shuffled | mixed",
  "goal_source_eval": "oracle | predicted | replay | shuffled",
  "reward": "...",
  "local_metrics": {
    "base_final_distance_mean": 0.0,
    "tuned_final_distance_mean": 0.0,
    "paired_improvement_mean": 0.0,
    "correct_goal_distance": 0.0,
    "shuffled_goal_distance": 0.0,
    "success_within_epsilon": 0.0
  },
  "closed_loop_metrics": {
    "base_success": 0.0,
    "tuned_success": 0.0,
    "base_return": 0.0,
    "tuned_return": 0.0,
    "oracle_goal_success": 0.0,
    "predicted_goal_success": 0.0,
    "shuffled_goal_success": 0.0
  },
  "policy_diagnostics": {
    "goal_sensitivity_mean": 0.0,
    "obs_shuffle_action_l2": 0.0,
    "goal_shuffle_action_l2": 0.0,
    "action_saturation_frac": 0.0,
    "residual_norm_mean": 0.0
  }
}
```

Also save:

- training history JSON;
- evaluation episode outcomes;
- videos for 8-16 representative episodes;
- a small Markdown summary table for each experiment family.

---


## 6. Experiment logging and final reporting protocol

Every experiment family must maintain two Markdown files:

```text
<experiment_family>_experiment_log.md      # chronological running log
<experiment_family>_final_results.md       # cleaned final report after aggregation
```

The running log is not optional. It should be updated as experiments are run, not reconstructed from memory later.

### 6.1 Running experiment log

The running log should include one entry per run or run group:

````markdown
## YYYY-MM-DD — short run name

### Hypothesis
What this run is supposed to test.

### Command
```bash
uv run hcl-poc ...
```

### Setup
- git commit:
- machine/GPU:
- config:
- N trajectories: 500 or 1800
- representation:
- architecture:
- horizon k:
- num_envs:
- seeds:
- base checkpoint:
- reward:

### Results
Small table of the important metrics.

### Plots / artifacts
- learning curve:
- local distance histogram:
- success plot:
- videos:
- JSON result path:

### Interpretation
What changed compared with the baseline? Did it pass/fail the gate?

### Next action
What should be tried next and why.
````

Include failed runs. Failed runs are often the most useful evidence for this project.

### 6.2 Tables and plots to add when useful

Use plots whenever they make trends easier to understand. At minimum, generate plots for:

```text
PPO learning curve: paired improvement / local distance / task success vs environment steps
correct-goal vs shuffled-goal local distance distributions
goal sensitivity by horizon k
D_phi temporal ordering / calibration curves
N=500 vs N=1800 comparisons
horizon sweep k=2/5/10/20
```

Useful tables:

```text
single-run metric table for each run group
multi-seed mean ± std table for final results
N=500 vs N=1800 table
representation/architecture ablation table
reward metric comparison table
```

### 6.3 Final results document

After an experiment family is complete, write a cleaned final report:

```text
<experiment_family>_final_results.md
```

It should include:

```markdown
# Final results: <experiment family>

## Question
The exact research question.

## Setup
Datasets, N, seeds, horizons, representations, architectures, reward, env count.

## Main results
Final tables with mean ± std where possible.

## Plots
Key plots with short captions.

## Qualitative observations
Videos/failure modes/reward hacking/saturation/instability.

## Conclusion
What this proves or disproves.

## Decision
Keep / modify / discard this formulation.

## Follow-up experiments
Only the next useful experiments, not every possible variation.
```

The final report should separate:

```text
confirmed conclusions
likely explanations
speculation / next hypotheses
```

### 6.4 Required metadata in every log and final report

Always write:

```text
N trajectories used for imitation pretraining: 500 or 1800
whether the low level was multi-offset trained
whether the goal was oracle/replay, predicted, shuffled, or zero
whether RL used privileged/TCP, raw VAE L2, or learned D_phi reward
num_envs used for PPO, with 4096+ for serious runs
whether results are smoke/dev/final
```

If a run used fewer than 4096 envs, explicitly label it as smoke/debug unless there is a hardware reason and the limitation is documented.


# Experiment A — RL sanity: can reachability RL work at all?

## Goal

Determine whether the local reachability RL formulation can train or improve a low-level goal-conditioned controller when representation is not the bottleneck.

This should be the first serious experiment. It answers:

> If we use privileged/TCP state and oracle local goals, can PPO reduce local final distance or improve success?

If this fails, do not debug VAE latents yet. The RL formulation, reward, or implementation is still wrong or too weak.

## Representation

Use the easiest meaningful state:

```text
z_t = privileged Push-T state, currently 31D observations_state
```

Optionally also test a smaller hand-selected physical state:

```text
z_t = [pusher position, block pose, block velocity/contact-ish features]
```

For the excavator analogue, this corresponds to using robot/TCP/effect state as a diagnostic, not as final deployable input.

## Goal source

Use oracle/replay local goals from recorded trajectories:

```text
g = z_{t+k}^{demo}
```

Start with `k=10` to match current hierarchy. Later sweep `k in {2,5,10,20}`.

## Reward formulation

Use normalized progress plus terminal distance:

```math
d_i = D(z_i, g)

r_i = (d_i - d_{i+1}) / (d_0 + eps)

r_{terminal} = -lambda_terminal * min(d_H / (d_0 + eps), clip_max)
```

Use a weighted physical distance. Do not use raw unweighted state L2 if different dimensions have very different scales. Start with normalized z dimensions and/or task-specific weights.

Suggested first values:

```text
lambda_terminal = 1.0
clip_max = 5.0
eps = 1e-6
horizon H = 10
```

## Policies to compare

Run all four:

```text
A1: scratch PPO low-level
A2: BC-initialized PPO, whole low-level trainable with small LR
A3: frozen BC + goal-conditioned residual PPO
A4: goal-fusion/FiLM layers trainable, trunk mostly frozen
```

For A3 residual:

```math
a_t = a_base(z_t, g, a_{t-1}, tau) + alpha * r_theta(z_t, g, a_{t-1}, tau)
```

Sweep:

```text
alpha in {0.1, 0.25, 0.5, 1.0}
```

Previous 4096-env privileged residual runs suggest `alpha=0.25` is often safer than larger values.

## PPO scale

Serious runs:

```text
num_envs >= 4096
rollout_steps = 10
samples_per_update >= 40960
total_steps >= 1M for dev, more if learning curve is still improving
num_minibatches = 8 or 16
```

Smoke runs may use smaller env count, but do not draw conclusions from them.

## Required metrics

Local held-out reset bank:

```text
base final distance
RL final distance
paired improvement = base - RL
success within distance epsilon
correct-goal vs shuffled-goal distance
```

Closed-loop full task:

```text
base hierarchy success
RL hierarchy success
base oracle-goal success
RL oracle-goal success
return
```

Policy diagnostics:

```text
goal sensitivity
residual/action norm
action saturation
KL / entropy / value loss
```

## Pass criteria

At least one RL variant should satisfy:

```text
RL local final distance < frozen base local final distance
correct goals clearly outperform shuffled goals
no severe action saturation
closed-loop task success not worse than base
```

A strong pass is:

```text
closed-loop task success improves by >= 5-10 percentage points over base
```

## Interpretation

- If scratch PPO works but BC-initialized PPO fails, the BC initialization is creating a goal-ignoring local optimum.
- If BC PPO works but scratch fails, imitation initialization is helpful and the previous failure is likely representation/goal-use related.
- If all variants fail with privileged state and oracle goals, debug the local-reset/reward/PPO implementation before doing learned latents.

---

# Experiment B — Low-level goal-identifiability gate

## Goal

Before any expensive RL, verify that the supervised low-level policy actually uses the goal.

Current learned VAE-latent low levels mostly ignore the goal. This experiment turns goal usage into a hard gate.

## Training setup

Use multi-offset supervised training for every representation:

```text
for each trajectory and segment start t:
    g = encode(o_{t+k})
    for i in 0..k-1:
        input = [o_{t+i}, g, previous_action_{t+i}, remaining=(k-i)/k]
        target = a_{t+i}
```

Test representations:

```text
B1: privileged state goal
B2: existing VAE512 future-state goal
B3: temporal-reachability latent from Experiment E
B4: effect latent from Experiment F
```

Test low-level architectures:

```text
arch 1: concat baseline
arch 2: FiLM/gated goal conditioning
arch 3: base policy + goal residual
```

## Architecture details

### Arch 1: concat baseline

```math
a_t = pi([obs_t, g, a_{t-1}, tau])
```

This is the current-style baseline and is easiest for the network to ignore `g`.

### Arch 2: FiLM/gated low level

```math
h_0 = f_obs(obs_t, a_{t-1}, tau)

for each layer l:
    gamma_l, beta_l = f_goal_l(g)
    h_l = activation(W_l h_{l-1})
    h_l = (1 + gamma_l) * h_l + beta_l

a_t = W_out h_L
```

Initialize FiLM close to identity:

```text
gamma initialized near 0
beta initialized near 0
```

This keeps early behavior stable while forcing goal information into every layer.

### Arch 3: base + goal residual

```math
a_base = pi_base(obs_t, a_{t-1})

delta_a = r_theta(obs_t, g, a_{t-1}, tau)

a_t = a_base + alpha * delta_a
```

Use this to isolate goal-dependent correction. Do not make `alpha` so tiny that residuals are always negligible.

## Required diagnostics

### Same-state valid-goal action sensitivity

For the same current state, compare actions under valid future goals:

```math
S(k1,k2) = ||pi(o_t, g_{t+k1}) - pi(o_t, g_{t+k2})||_2
```

Evaluate at:

```text
k pairs: (2,5), (2,10), (5,10), (10,20), (2,20)
```

### Condition-block shuffle

Measure action change from shuffling each input block:

```text
shuffle current observation/state
shuffle goal
shuffle previous action
shuffle remaining time
```

A healthy goal-conditioned low level should not have goal-shuffle action change near zero.

### Closed-loop local reaching

From the same reset bank:

```text
correct oracle/replay goals
shuffled goals
zero/no goals
predicted high-level goals
```

Report final distance and success-within-epsilon.

### Full closed-loop task

Evaluate:

```text
flat baseline
learned-high hierarchy
oracle-goal hierarchy
shuffled-goal hierarchy
```

## Pass criteria

Do not run expensive RL unless:

```text
correct-goal local final distance < shuffled-goal local final distance
same-state goal sensitivity is materially larger than the current VAE512 result
oracle-goal hierarchy is meaningfully better than shuffled/no-goal hierarchy
```

For learned VAE512, current goal sensitivity is around `0.02-0.05` action L2. A useful target is at least several times larger, ideally approaching the privileged-state sensitivity scale where `k=2` vs `k=10` was around `0.26`.

---

# Experiment C — Paired-improvement RL reward

## Goal

The current reward asks the policy to minimize absolute distance to the goal. A more thesis-aligned reward asks whether the tuned policy improves over the frozen imitation policy from the same reset and same goal.

This directly matches the claim:

> RL fine-tuning improves the imitation policy without a hand-designed task reward.

## Formulation

For every local reset and goal, run the frozen base policy once or use a cached base rollout:

```math
J_base = D(z_H^{base}, g)
```

Run the tuned policy:

```math
J_pi = D(z_H^{pi}, g)
```

Use a terminal paired reward:

```math
R = J_base - J_pi - lambda_a * sum_t ||a_t^{pi} - a_t^{base}||_2^2
```

Optionally add dense progress:

```math
r_t = eta * (D(z_t,g) - D(z_{t+1},g)) / (D(z_0,g)+eps)
```

and terminal paired reward at the end:

```math
r_H += J_base - J_pi
```

## Why this matters

Absolute distance punishes the policy for hard or imperfect goals. Paired reward normalizes by difficulty because the base policy sees the same reset and goal.

This is especially important when high-level goals are predicted and sometimes imperfect.

## Variants

Test the paired reward with distance metrics in stages:

```text
C1: privileged/TCP distance       # available immediately; sanity/upper-bound metric
C2: raw VAE L2, as a control      # available immediately; expected weak baseline
C3: learned reachability D_phi    # requires Experiment E first
```

Use the same policy/residual architecture in each case. Do not wait for `D_phi` to test paired reward itself; first validate the paired reward with privileged/TCP distance.

## Required metrics

```text
mean paired improvement J_base - J_pi
fraction of local rollouts improved
mean action deviation from base
closed-loop task success delta
oracle-goal success delta
predicted-goal success delta
```

## Pass criteria

A good local RL method should improve the majority of local rollouts:

```text
fraction_improved > 0.55
mean_paired_improvement > 0
closed-loop success not worse
```

A strong result is:

```text
mean_paired_improvement > 0
AND task success improves by >= 5 percentage points
```

---

# Experiment D — Horizon curriculum

## Dependency note

This is a **sweep/wrapper experiment**, not a new representation or reward by itself. Run it first with rewards that already exist, for example privileged/TCP distance or raw latent L2. After Experiment E has implemented the learned reachability distance `D_phi`, rerun the same horizon sweep using `D_phi`.

In other words, Experiment D should not block on Experiment E. Experiment E provides an additional distance metric that can later be plugged into D.

## Goal

Determine the local horizon at which goal-conditioned control and RL are reliable.

The current system often uses:

```text
k = 10 steps
```

but longer horizons may be too hard for the low level or representation. Short horizons may be easier and can be chained by the high level.

## Setup

Run Experiments A/B/C for:

```text
k in {2, 5, 10, 20}
```

For each `k`, train with matching multi-offset samples:

```text
i = 0..k-1
remaining = (k-i)/k
```

## Metrics

For each horizon:

```text
offline low-level action MAE
goal sensitivity
correct-vs-shuffled local final distance
oracle-goal closed-loop success
learned-high closed-loop success
RL paired improvement
```

## Interpretation

- If `k=2` and `k=5` work but `k=10` fails, use a shorter local horizon.
- If all horizons fail with learned latent but privileged state works, representation/metric is the bottleneck.
- If all horizons fail even with privileged state, RL/local-control formulation is the bottleneck.

## Recommendation

Do not force a long local horizon early. For the excavator project, a reliable short-horizon effect controller may be more useful than a speculative 20-step latent goal.

---

# Experiment E — Learned reachability distance from trajectories

## Goal

Replace raw `||z_reached - z_goal||` with a learned distance/value function trained from trajectories.

This is one of the most important real-robot-compatible ideas. It can be trained from ordered demonstrations without privileged object labels.

The high level still outputs a compact latent goal. We are replacing the **distance metric**, not requiring the high level to generate a full future observation.

Current assumption:

```math
D(z_t,z_g) = ||z_t-z_g||_2
```

New assumption:

```math
D(z_t,z_g) = D_phi(z_t,z_g)
```

where `D_phi` is trained to approximate temporal/control reachability.

## Version E1: add learned distance on top of existing VAE512

Keep the existing encoder:

```math
z_t = E_{VAE}(o_t)
```

Train:

```math
D_phi(z_i, z_j) approx min((j-i)/H, 1)
```

for pairs from the same trajectory with `j >= i`.

Use negatives from:

```text
reversed pairs
pairs from different trajectories
far-future pairs clipped to 1
shuffled goals
```

### Loss option: temporal distance regression

```math
y_{ij} = min((j-i)/H, 1)

L_dist = (D_phi(z_i,z_j) - y_{ij})^2
```

Constrain output:

```math
D_phi >= 0
```

using `softplus` or positive output head.

### Loss option: binary reachability classifier

```math
R_phi(z_i,z_j,k) = P(z_j reachable within k steps from z_i)
```

Positive:

```text
0 < j-i <= k
```

Negative:

```text
j-i > k, reversed, different trajectory, shuffled
```

Loss:

```math
L = - y log R_phi - (1-y) log(1-R_phi)
```

Convert to distance for reward:

```math
D_phi = -log(R_phi + eps)
```

## Version E2: train encoder and distance jointly

Instead of relying on VAE geometry, train:

```math
z_t = E_theta(o_t)
D_phi(z_i,z_j) approx temporal distance / reachability
```

Add regularizers to avoid collapse:

```text
contrastive negatives
unit-norm embeddings
variance/covariance regularization similar to VICReg
optional reconstruction auxiliary, low weight
optional inverse-dynamics auxiliary
```

Suggested loss:

```math
L = L_reachability
  + lambda_contrast * L_contrastive
  + lambda_var * L_variance
  + lambda_inv * L_inverse_action
  + lambda_recon * L_reconstruction_optional
```

Do not let reconstruction dominate. The goal is reachability geometry, not pretty reconstruction.

## Version E3: Bellman-style distance

Train a goal-conditioned value/distance:

```math
D_phi(o_g, o_g) = 0
D_phi(o_t, o_g) approx 1 + D_phi(o_{t+1}, o_g)
```

for demo transitions until the goal is reached.

This is more theoretically aligned but easier to destabilize. Do it after E1/E2.

## RL reward using learned distance

Given high-level goal `g`:

```math
r_t = D_phi(E(o_t), g) - D_phi(E(o_{t+1}), g)
```

Normalized version:

```math
r_t = (D_phi(E(o_t),g) - D_phi(E(o_{t+1}),g)) / (D_phi(E(o_0),g)+eps)
```

Terminal:

```math
r_H -= lambda_terminal * D_phi(E(o_H),g)
```

Paired version:

```math
R = D_phi(E(o_H^{base}),g) - D_phi(E(o_H^{pi}),g)
    - lambda_a * sum_t ||a_t^pi - a_t^base||^2
```

## Required validation before RL

A learned distance must pass these checks:

```text
same-trajectory temporal distance monotonicity
near future closer than far future
future closer than random trajectory state
current-to-goal distance decreases along demo trajectory
correlation with privileged/TCP distance where available
correlation with local rollout success
```

Quantitative diagnostics:

```text
Spearman correlation between D_phi and time-to-goal
AUC for reachable-within-k classification
mean D_phi(o_t,o_{t+k}) along demos vs shuffled pairs
D_phi decrease along teacher rollout
D_phi decrease/failure along bad rollout
```

## Main comparison

Run identical RL/local reaching with:

```text
reward 1: raw VAE L2
reward 2: learned D_phi over VAE z
reward 3: jointly learned reachability latent + D_phi
reward 4: privileged/TCP distance upper bound
```

## Pass criteria

A good learned distance should:

```text
rank demo future states in the correct temporal order
make correct goals clearly better than shuffled goals
produce better local RL improvement than raw VAE L2
avoid obvious reward hacking in videos
```

---

# Experiment F — Future-effect latent instead of future-state latent

## Goal

Test whether the high-level/low-level interface should represent the desired **future effect/change**, not the full future state.

The current approach uses:

```math
z_{t+k} = E_state(o_{t+k})
```

A future-state latent may encode irrelevant details. A future-effect latent should encode controllable change:

```math
e_{t,k} = E_effect(o_t, o_{t+k}, a_{t-1}, k)
```

or a simplified version:

```math
g_{t+k} = G(o_{t+k})
```

with losses that make it useful for action/reachability, not reconstruction.

## Why this matters for the real robot

The final robot may not have privileged rock pose/shape labels. But it can record:

```text
current observation o_t
future observation o_{t+k}
robot proprioception
actions
```

A future-effect latent can be trained from observation pairs without requiring full object state labels or future image generation.

## Version F1: goal encoder from future observation only

Train:

```math
g_{t+k} = G(o_{t+k})
```

But train `G` with action/reachability losses instead of only reconstruction.

Low level:

```math
pi(a_{t+i} | o_{t+i}, g_{t+k}, a_{t+i-1}, tau_i)
```

High level:

```math
H(o_t, a_{t-1}, k) -> g_{t+k}
```

## Version F2: effect encoder from current and future observation

Train:

```math
e_{t,k} = E_effect(o_t, o_{t+k}, k)
```

Low level:

```math
pi(a_{t+i} | o_{t+i}, e_{t,k}, a_{t+i-1}, tau_i)
```

High level:

```math
H(o_t, a_{t-1}, k) -> e_{t,k}
```

This is not a full future observation. It is a compact code for the desired transition/effect.

## Losses

Use a combination:

```math
L = L_low_action
  + lambda_high * L_high_predict
  + lambda_reach * L_reachability
  + lambda_inv * L_inverse_action
  + lambda_contrast * L_temporal_contrast
```

Where:

```math
L_low_action = ||pi(o_{t+i}, goal, a_{t+i-1}, tau_i) - a_{t+i}^{demo}||^2
```

For deterministic action. If using flow matching later, replace MSE with the flow loss, but keep the diagnostics unchanged.

High-level prediction:

```math
L_high_predict = ||H(o_t,a_{t-1},k) - stopgrad(goal_{t,k})||^2
```

or contrastive/cosine loss if embeddings are normalized.

Reachability:

```math
D_phi(E(o_t), goal_{t,k}) approx k/H
```

Inverse-action auxiliary:

```math
q(goal, o_t) -> action chunk or first action
```

This encourages the latent to keep action-relevant information.

## Architecture

Use FiLM/gated low-level conditioning as default. Compare concat only as a baseline.

## Required diagnostics

```text
low-level action MAE
same-state valid-goal action sensitivity
correct-vs-shuffled local reaching
oracle-goal hierarchy success
learned-high hierarchy success
high-level prediction error
D_phi temporal ordering if using reachability loss
```

## Pass criteria

Effect latent is promising if it beats VAE512 on at least two of:

```text
goal sensitivity
oracle-goal closed-loop success
learned-high closed-loop success
local RL paired improvement
correct-vs-shuffled gap
```

---

# Experiment G — Simulator branch bank for true alternative goals

## Goal

Create data where the same or near-same current state has multiple reachable future goals requiring different actions.

This directly attacks the main identifiability problem: in a single deterministic demonstration, the future goal is often redundant with the current state.

## Important constraint

Do **not** rely mainly on random actions. Random actions produce poor, noisy, often irrelevant data.

Use competent local branch generators:

```text
privileged PPO teacher from reset state
scripted local controller
CEM/MPC local planner
multiple competent experts
teleop/scripted short branches
```

## Branch dataset format

For each reset state `s_t`, collect several branches:

```text
branch j:
    same initial state s_t
    local target/effect g_j
    actions a_t^{j}, ..., a_{t+k-1}^{j}
    final observation/state o_{t+k}^{j}
```

Store:

```text
initial simulator state or replay state
current observation/features
goal representation
action sequence
final observation/features
success/reachability flags
branch generator ID
```

## Push-T branch targets

Start simple:

```text
pusher endpoint targets around current pusher position
block pose targets from short teacher/MPC branches
teacher continuation endpoints from perturbed local objectives
```

## Training use

Train low level on:

```math
(s_t, g_j, a_{t:t+k-1}^{j})
```

with multi-offset expansion:

```math
(s_{t+i}^{j}, g_j, a_{t+i-1}^{j}, tau_i) -> a_{t+i}^{j}
```

## Metrics

```text
same-state different-goal action sensitivity
correct branch goal final distance
wrong branch/shuffled goal final distance
oracle-goal closed-loop success
learned-high success if high level is trained on branch data
```

## Pass criteria

Branch data is useful if:

```text
same-state action sensitivity increases substantially
correct branch goals outperform wrong/shuffled branch goals
oracle-goal hierarchy improves over non-branch data
```

## Real-robot relevance

Exact branching from the real robot will be hard. The simulator branch bank is mainly a proof-of-concept diagnostic. For the final robot, approximate branch data can be collected as short local teleop branches from similar/canonical states, but the main real-compatible method should still be learned reachability/effect latents from ordinary trajectories.

---

# Experiment H — High-level goal validity and off-manifold goals

## Goal

High-level predicted goals may be off the manifold of real encoded future observations. Then `D_phi` or the low level may behave unpredictably.

This experiment checks whether high-level predictions are valid control targets.

## Diagnostics

For predicted goals:

```math
hat_g = H(o_t,a_{t-1},k)
```

Compute:

```text
nearest-neighbor distance to encoded real future goals
D_phi(current, predicted_goal)
D_phi(current, true_future_goal)
D_phi(predicted_goal, true_future_goal) if defined
low-level action under predicted vs true goal
closed-loop local final distance to predicted goal and true goal
```

Also train a goal validity discriminator:

```math
C(g) = P(g came from real encoded observation/future effect)
```

Negatives:

```text
high-level predictions
random latent vectors
interpolations/extrapolations
shuffled deltas
```

## Mitigation variants

Test:

```text
H1: direct goal prediction
H2: predict delta: g_hat = z_t + DeltaH(z_t)
H3: unit-normalized goal embeddings with cosine loss
H4: nearest-neighbor/prototype goal selection from a memory bank
H5: high-level goal validity penalty
```

Prototype version:

```text
High level predicts scores over candidate future goals from a bank.
Choose top-1 or sample top-k.
```

This avoids off-manifold vectors, but may be less elegant.

## Metrics

```text
validity discriminator score
nearest-neighbor distance
high-level prediction error
learned-high closed-loop success
oracle-goal closed-loop success
predicted-vs-oracle goal gap
```

## Pass criteria

High-level predictions should be close enough to real goal manifold that:

```text
learned-high success is not dramatically below oracle-goal success
predicted goals pass validity checks better than random/shuffled vectors
low-level does not produce unstable actions under predicted goals
```

---

# Experiment I — Flow matching only where it is actually needed

## Goal

Flow matching was not competitive in the current privileged-state zero-noise deterministic evaluations. Do not make flow matching a dependency for all next experiments.

Use deterministic MLPs for the first formulation tests. Add flow matching only where multimodality is clearly causing deterministic averaging.

## Recommended use

### Low-level action policy

Start with deterministic or Gaussian policy for RL:

```text
deterministic MLP for BC
Gaussian/tanh-Gaussian for PPO
```

Do not fine-tune a flow-matching low-level with PPO until the deterministic/Gaussian version works.

### High-level future goal prediction

Flow may be useful here if multiple future goals are valid from the same state.

Test:

```text
I1: deterministic high level
I2: flow high level, sample N goals
I3: choose best sampled goal using D_phi or a learned critic/reachability score
```

## Metrics

```text
high-level prediction diversity
best-of-N local reachability
closed-loop success with sampled goals
comparison to deterministic high level
```

## Pass criteria

Flow should only be kept if:

```text
best-of-N or sampled high-level goals improve closed-loop success
OR deterministic high-level predictions are clearly averaging incompatible futures
```

---

# Experiment J — Real-data-compatible pipeline test

## Goal

Test a version of the method that does not require privileged object/rock pose labels or exact real-world branching.

This is the most relevant pipeline for the final excavator task.

## Allowed data

Use only what a real robot can plausibly record:

```text
observations: RGB, point cloud, proprioception, robot state
actions
trajectory ordering/time
optional robot TCP/end-effector state
```

Do not use:

```text
privileged block/rock poses as training labels
exact object identities unless perception provides them
simulator-only full state for the main method
```

Privileged data can still be used for diagnostics and upper bounds.

## Formulation

Train:

```math
x_t = E_current(o_t)

g_{t,k} = G_goal(o_{t+k})         # or E_effect(o_t,o_{t+k})

D_phi(x_t, g_{t,k}) = learned reachability distance

H(x_t, a_{t-1}, k) -> g_hat_{t,k}

pi(a_t | o_t, g_hat, a_{t-1}, tau) -> action
```

RL reward:

```math
r_t = D_phi(E_current(o_t), g_hat) - D_phi(E_current(o_{t+1}), g_hat)
```

or paired version:

```math
R = D_phi(E(o_H^{base}), g_hat) - D_phi(E(o_H^{pi}), g_hat)
    - lambda_a * action_deviation
```

## Dataset construction

Use ordinary trajectories. From each trajectory sample pairs:

```text
(o_t, o_{t+k}) for k in {2,5,10,20}
```

Create:

```text
low-level multi-offset samples
high-level prediction samples
reachability metric pair samples
negative/shuffled pairs
```

## Metrics

```text
reachability temporal ordering
low-level correct-vs-shuffled gap
oracle future observation goal vs high-level predicted goal
local RL paired improvement
closed-loop task success
```

## Pass criteria

This pipeline is promising if it achieves a significant fraction of the privileged-state upper bound without privileged object labels.

Example target:

```text
learned real-compatible pipeline >= 60-70% of privileged oracle-goal performance
and improves over frozen imitation after RL
```

---

# Experiment K — Visual/point-cloud representation for unordered objects/rocks

## Goal

Prepare for the excavator/rock task where object pose/shape may be an unordered set and not directly available as privileged labels.

This is not the first Push-T experiment, but it should guide representation choices.

## Candidate representations

### K1: raw observation encoder

```text
RGB/DINO/spatial features + proprioception
```

### K2: point-cloud local patch encoder

```text
local point cloud around end-effector or workspace region
PointNet/PointTransformer/sparse conv encoder
```

### K3: unordered object/rock set encoder, if segmentation exists

For detected rocks/clusters:

```math
r_i = [position/keypoints/shape features/color/features]

z_rocks = rho(sum_i phi(r_i))
```

or use attention pooling:

```text
Set Transformer / Perceiver-style object tokens
```

### K4: selected-object/effect token

If the task involves manipulating one object at a time, learn or provide a selected local target token:

```text
selected rock patch/keypoint + robot state + local scene context
```

This can make goals much easier than encoding the whole unordered scene.

## Recommendation

Do not make the first proof of concept depend on perfect rock pose/shape labels. Use privileged rock state only as an oracle diagnostic in sim.

For the deployable method, prefer:

```text
robot state + local visual/point-cloud embedding + learned reachability/effect goal
```

---

# Experiment L — Larger/longer task selection after Push-T

## Goal

Once the formulation works on Push-T, test a longer task where hierarchy matters more.

Push-T is too short to strongly prove hierarchical benefits. The next environment should have:

```text
long horizon
several meaningful subgoals
limited demonstration setting
possible local resets in sim for debugging
no full navigation if avoidable
```

## Candidate task properties

For the excavator/robot application, prefer tasks like:

```text
pick/sort/manipulate several rocks or objects without navigation
local uncovering + grasping + placing
structured workspace with multiple objects
```

Avoid early tasks with heavy soil dynamics if sim mismatch dominates.

## Transfer the same gates

Before RL on the longer task, require:

```text
supervised hierarchy matches flat BC
oracle-goal low-level works
correct goals beat shuffled goals
learned D_phi ranks trajectory futures correctly
RL improves local reachability over frozen BC
```

Only then claim full-task RL fine-tuning results.

---

## 7. Suggested implementation order

Do not implement everything at once. Use this order.

### Phase 0 — Prepare supervised bases and documentation

1. Create fixed `N=500` and `N=1800` manifests and local reset banks.
2. Train the required flat and hierarchical imitation checkpoints for the minimal pretraining matrix.
3. Run closed-loop imitation evaluation, oracle-goal evaluation, and goal-identifiability diagnostics.
4. Cache frozen base policy rollouts for paired reward.
5. Create the running Markdown log for the experiment family before launching RL.

### Phase 1 — Make RL sanity pass

1. Implement/clean Experiment A with privileged/TCP state.
2. Use `num_envs >= 4096`.
3. Compare scratch, BC-initialized, residual, and partially unfrozen policies.
4. Add paired reward from Experiment C.
5. Produce a short report: does reachability RL improve local goal reaching when representation is perfect?

### Phase 2 — Make goal identifiability a hard gate

1. Implement Experiment B diagnostics as reusable functions.
2. Run them on existing VAE512, privileged state, and any new representation.
3. Refuse to launch expensive RL for a model that fails correct-vs-shuffled or goal-sensitivity gates.

### Phase 3 — Replace raw VAE L2

1. Implement Experiment E1: learned `D_phi` on top of existing VAE512.
2. Use `D_phi` in local RL reward.
3. Compare directly against raw VAE L2 and privileged/TCP distance.

### Phase 4 — Learn better goal representations

1. Implement Experiment E2: jointly learned temporal-reachability latent.
2. Implement Experiment F: future-effect latent.
3. Use FiLM/gated low-level conditioning by default.
4. Compare with VAE512 using the same diagnostics and eval banks.

### Phase 5 — Add true branch/counterfactual data in sim

1. Implement simulator branch bank only after the above gates exist.
2. Use competent branch generators, not random actions.
3. Test whether same-state alternative goals fix goal ignoring.

### Phase 6 — Real-data-compatible version

1. Remove privileged object labels.
2. Use only observations, proprioception, actions, and trajectory order.
3. Keep privileged metrics only as hidden diagnostics/upper bounds.
4. Test on a longer task.

---

## 8. Concrete implementation tasks

### Task 1: Reusable goal-sensitivity module

Add a module, for example:

```text
src/hcl_poc/goal_diagnostics.py
```

Functions:

```python
def same_state_valid_goal_sensitivity(policy, batch, horizons): ...
def condition_block_shuffle_sensitivity(policy, batch): ...
def evaluate_correct_vs_shuffled_goals(envs, policy, reset_bank, goal_encoder): ...
def summarize_goal_diagnostics(metrics) -> dict: ...
```

Expose via CLI:

```bash
uv run hcl-poc rl-rerun goal-diagnostics --checkpoint ... --representation ...
```

### Task 2: Learned reachability distance module

Add:

```text
src/hcl_poc/reachability.py
```

Core classes:

```python
class ReachabilityEncoder(nn.Module): ...
class ReachabilityDistance(nn.Module): ...
class ReachabilityClassifier(nn.Module): ...
```

Training dataset:

```python
class TrajectoryPairDataset(Dataset):
    # returns o_i, o_j, delta_steps, same_traj flag, reachable label
```

CLI:

```bash
uv run hcl-poc rl-rerun train-reachability-distance \
  --dataset ... \
  --encoder vae512|joint|effect \
  --horizon 10 \
  --negatives shuffled,reversed,far \
  --batch-size 4096

uv run hcl-poc rl-rerun eval-reachability-distance \
  --checkpoint ... \
  --dataset ...
```

### Task 3: RL reward abstraction

Refactor reward computation so new rewards can be swapped cleanly.

Suggested interface:

```python
class LocalGoalReward:
    def reset(self, base_policy=None, reset_batch=None, goal_batch=None): ...
    def distance(self, obs_or_z, goal): ...
    def dense_reward(self, prev_obs, next_obs, goal, initial_distance): ...
    def terminal_reward(self, final_obs, goal): ...
```

Implement:

```text
RawL2Reward
NormalizedRawL2Reward
PrivilegedWeightedDistanceReward
LearnedReachabilityReward
PairedImprovementReward
```

### Task 4: FiLM/gated low-level architecture

Add or extend model definitions:

```python
class GoalFiLMLowLevel(nn.Module): ...
class GoalResidualLowLevel(nn.Module): ...
```

Make architecture selectable from config/CLI:

```text
low_level_arch: concat | film | residual
```

### Task 5: Experiment runner and aggregation

Every new experiment family should have:

```text
run command
aggregate command
Markdown report generator
```

Example:

```bash
uv run hcl-poc rl-rerun run-reachability-suite --config configs/pusht_incremental.yaml ...
uv run hcl-poc rl-rerun aggregate-reachability-suite --results-dir ...
```

---

## 9. Final reporting tables

For each experiment family, produce tables like these.

### Table 1: goal identifiability

| representation | arch | N demos | action MAE | goal sensitivity | obs shuffle L2 | goal shuffle L2 | correct local dist | shuffled local dist |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| privileged | concat | | | | | | |
| privileged | FiLM | | | | | | |
| VAE512 | concat | | | | | | |
| VAE512 | FiLM | | | | | | |
| reachability latent | FiLM | | | | | | |
| effect latent | FiLM | | | | | | |

### Table 2: RL sanity

| representation | init | reward | N demos | num_envs | base local dist | RL local dist | paired improvement | base success | RL success |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| privileged | scratch | progress | 4096 | | | | | |
| privileged | BC init | progress | 4096 | | | | | |
| privileged | residual | paired | 4096 | | | | | |

### Table 3: learned distance comparison

| reward metric | N demos | temporal corr | correct-vs-shuffled gap | local RL improvement | task success delta | reward hacking observed? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw VAE L2 | | | | | |
| VAE + learned D_phi | | | | | |
| joint reachability latent | | | | | |
| privileged/TCP upper bound | | | | | |

### Table 4: high-level bottleneck

| representation | N demos | oracle-goal success | learned-high success | gap | high pred error | validity score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| privileged | | | | | |
| VAE512 | | | | | |
| reachability latent | | | | | |
| effect latent | | | | | |

---

## 10. What success would look like

A convincing proof-of-concept should show this chain:

1. Flat BC is decent but limited.
2. Hierarchical BC is at least competitive with flat BC.
3. The low-level truly uses the goal: correct goals beat shuffled goals.
4. A learned reachability/effect representation gives a better control metric than raw VAE L2.
5. Local RL improves goal-reaching over the frozen imitation policy using only reachability reward.
6. This local improvement transfers to full-task success.
7. The best version does not depend on privileged object labels, except for diagnostics/upper bounds.

The minimum strong claim would be:

```text
Using a trajectory-learned reachability metric and goal-conditioned low-level architecture,
RL fine-tuning improves a hierarchical imitation policy over its frozen BC version,
without using a hand-designed task reward.
```

---

## 11. Things not to over-invest in yet

Do not spend too much time on:

```text
small PPO hyperparameter sweeps of the current VAE512 raw-L2 reward
flow-matching low-level RL before deterministic/Gaussian low-level RL works
32/64-env PPO conclusions
one-step action MAE as the main model-selection metric
privileged object-state solutions that cannot transfer to real robot data
```

These may be useful later, but they are not the bottleneck right now.

---

## 12. One-sentence working hypothesis

The project should move from:

```text
future VAE state + raw latent L2 reward + concat low level
```

to:

```text
future effect/reachability latent + learned distance D_phi + FiLM/residual goal-conditioned low level + paired reachability RL
```

while keeping privileged-state experiments as oracle diagnostics and using many parallel environments for all serious PPO runs.
