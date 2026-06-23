# Push-T Low-Level RL Rerun Plan: Reset-Capable Data, Massive Parallelism, and Explicit Gates

## 1. Purpose

The previous low-level RL attempt is not a definitive test of the idea. It fell back to full-hierarchy rollouts because the demonstration HDF5 did not contain reset seeds or complete simulator states. It also used a small PPO collection size (`32 envs x 32 steps`) and skipped the flow-policy RL branches.

This rerun is designed to answer the actual question:

```text
Can low-level RL make fixed high-level latent goals more reachable?
```

The central change is:

```text
regenerate demonstration data with loadable simulator states,
retrain all imitation models on that regenerated data,
then train low-level RL from exact local goal-reaching resets
with as many GPU-parallel environments as reasonably fit.
```

The RL reward must remain project-clean:

```text
allowed: latent goal distance, latent progress, action/residual regularization
disallowed: ManiSkill dense reward, task success, privileged object state, task progress
```

Task success and dense reward are evaluation metrics only.

---

## 2. Why the Previous Result Is Not Enough

The previous RL run had several limitations:

1. The HDF5 corpus stored DINO features, proprioception, and actions, but not reset seeds or complete simulator state. Therefore exact local demonstration resets were not possible.
2. Training used full hierarchy rollouts instead of the intended local 10-step goal-reaching MDP.
3. The implemented PPO rollout was only `32 envs x 32 steps`, which is very small for GPU simulation.
4. R2 residual-flow and R4 direct-flow low-level tuning were not run.
5. Some of the best-looking variants used task-progress shaping, which is not part of the final project idea.
6. The previous results gave only small gains at `N_demo=500` and degraded the stronger `N_demo=1000` hierarchy.

This rerun should not be considered optional if the thesis will make a statement about RL fine-tuning. The previous result should be described as preliminary and limited.

---

# 3. Fixed High-Level Experimental Claim

The target architecture remains:

```text
VAE-512 future-state interface
k = 10
U = 10
H = 1
```

At every high-level decision:

$$
g_i = H_{high}(o_{t_i}, a_{t_i-1})
$$

The low level receives:

$$
x_t =
[
o_t,
g_i,
a_{t-1}^{exec},
tau_t
]
$$

and outputs:

$$
a_t in [-1,1]^3
$$

The RL policy should improve the low-level mapping:

$$
x_t -> a_t
$$

without changing:

- the VAE;
- the high-level predictor;
- the high-level timing;
- DINO;
- the demonstration budget.

---

# 4. Demonstration Data Must Be Regenerated

## 4.1 Reason

The old data cannot support exact local resets. Regenerating data is required.

Because the new data will differ from the old data, the old trained checkpoints must not be reused for the main comparison. Otherwise the supervised models and RL models would not have seen the same data distribution.

## 4.2 New dataset requirements

Collect a new successful PPO-teacher corpus with exactly the same environment/controller setup:

```text
task: ManiSkill PushT-v1
backend: CUDA PhysX
controller: pd_ee_delta_pos
control frequency: 20 Hz
episode limit: 100 controls
teacher: deterministic privileged PPO teacher
```

For every trajectory and timestep, store:

```text
trajectory_id
reset_seed
timestep
full simulator state from env.get_state() or equivalent
initial reset state
teacher action raw
teacher action clipped/executed
previous executed action
RGB frame or enough information to rerender it
DINO feature
21D proprioception
VAE input vector
reward/success for evaluation only
done/terminated/truncated flags from the simulator
```

The critical field is the **full simulator state**. It must be sufficient to restore the exact physical state, including:

- robot/articulation state;
- object pose and velocity;
- all actor poses and velocities;
- controller internal state if applicable;
- any hidden simulator state that affects contact evolution, if accessible.

## 4.3 State-load validation

Before training anything, validate state loading.

For at least 1,000 random `(trajectory, timestep)` pairs:

1. Load stored simulator state `s_t`.
2. Rerender observation.
3. Recompute DINO/proprio/VAE latent.
4. Compare to stored values.
5. Execute the next 10 stored teacher actions.
6. Compare the resulting observations/latents/rewards to the stored trajectory.

Required gate:

```text
state_load_observation_error <= 1e-5 for low-dimensional state/proprio
VAE latent error <= 1e-4 normalized MSE
10-step replay final VAE latent MSE <= 1e-3
action/reward/success parity visually audited on samples
```

If this fails, do not start RL. Fix state serialization first.

## 4.4 Dataset sizes

Regenerate at least:

```text
N_train = 1000 successful trajectories
N_val   = 200 successful trajectories
N_test  = separate evaluation reset seeds, not stored in training file
```

Primary budgets:

```text
N_demo = 500
N_demo = 1000
```

Use nested prefixes:

```text
first 500 train trajectories
first 1000 train trajectories
```

Do not mix old and new data in the main comparison.

---

# 5. Retrain All Imitation Components on the New Data

For every `N_demo` and seed:

```text
seed in {0,1,2}
```

train from scratch:

1. VAE-512 representation.
2. Deterministic high-level future-latent predictor.
3. Deterministic low-level BC policy.
4. Action-flow low-level policy.
5. Optional flat observation and flat latent baselines.

No old checkpoint may be used in the main comparison.

## Gate 5A: supervised parity

Before RL, the newly trained frozen hierarchy must reproduce reasonable imitation performance.

Approximate expected range from previous scaling:

| N demo | old det hierarchy | old flow hierarchy |
| ---: | ---: | ---: |
| 500 | ~0.30 | ~0.30 |
| 1000 | ~0.51 | ~0.53 |

Gate:

```text
new frozen hierarchy is within +/- 0.10 absolute success of the old reference
or the difference is explained by dataset/seed statistics
```

If the new frozen hierarchy is much worse, do not start RL. Debug retraining, data quality, and checkpoint selection.

---

# 6. RL Environment: Local Goal-Reaching MDP

## 6.1 Primary RL mode: exact local reset

The main RL environment is a 10-step local goal-reaching MDP.

On reset:

1. Sample a training trajectory.
2. Sample a timestep `t` such that `t+10` exists.
3. Load simulator state `s_t`.
4. Set previous executed action to the stored `a_{t-1}^{exec}`.
5. Set held goal:
   $$
   g = z_{t+10}^{demo}
   $$
   for Mode A, or:
   $$
   g = H_{high}(o_t, a_{t-1}^{exec})
   $$
   for Mode B.
6. Roll for exactly 10 primitive controls.

The first main training mode is Mode A because the goal is guaranteed reachable by the teacher.

Then run Mode B because it matches deployment.

## 6.2 Secondary RL mode: full hierarchy

Full hierarchy rollouts are not the first training mode. They are used only after local training passes gates.

In full hierarchy mode:

1. Start from normal task reset.
2. Every 10 controls, predict:
   $$
   g_i = H_{high}(o_{t_i},a_{t_i-1})
   $$
3. Hold `g_i` for 10 primitive steps.
4. Train or evaluate the low-level RL policy.

## 6.3 Recovery-state local resets

After clean local training works, add local resets from:

- failed frozen-hierarchy rollouts;
- artificially perturbed states;
- recovery corpus states, if state-loadable.

Still use latent-only reward.

---

# 7. Reward: Project-Clean Latent Reachability Only

## 7.1 Current latent and goal

Let:

$$
z_t = mu_{VAE}(o_t)
$$

and let the held goal be:

$$
g
$$

Use normalized latents:

$$
zbar_t, gbar
$$

## 7.2 Distance

$$
d_t =
(1/512) || zbar_t - gbar ||_2^2
$$

## 7.3 Reward

Dense progress:

$$
r_t^{progress}=d_t-d_{t+1}
$$

Terminal penalty at step 10:

$$
r_T^{final}=-lambda_f d_T
$$

Optional latent-only success bonus:

$$
r_T^{bonus}=lambda_b * 1[d_T < epsilon_d]
$$

where `epsilon_d` is chosen from teacher validation segments, not from task success.

Regularization:

$$
r_t^{reg}
=
-lambda_delta ||Delta a_t||^2
-lambda_a ||a_t||^2
-lambda_s ||a_t-a_{t-1}||^2
$$

Full initial reward:

$$
r_t =
(d_t-d_{t+1})
+
1[t=T](-lambda_f d_T)
+
r_t^{reg}
$$

Initial weights:

```text
lambda_f      = 1.0
lambda_delta  = 0.01 for residual methods
lambda_a      = 0.001
lambda_s      = 0.001
lambda_bonus  = 0.0 initially
```

## 7.4 Disallowed training signals

Do not use for training:

```text
ManiSkill dense reward
task success
object pose
object distance to target
privileged simulator state
TCP/object probes
task progress
human-designed Push-T progress
```

These may be logged and reported only as evaluation diagnostics.

---

# 8. Termination, Truncation, and Advantage Handling

This is a critical implementation gate.

## 8.1 Definitions

Use the following convention:

```text
terminated = true MDP terminal; value bootstrap should be zero.
truncated  = time-limit or artificial cutoff; value bootstrap should be used.
```

## 8.2 Local 10-step goal-reaching episodes

For the local MDP, the 10-step segment is the whole episode.

At step 10:

```text
terminated = True
truncated  = False
bootstrap  = False
```

Reason: the finite-horizon local goal-reaching task ends at the 10-step goal deadline. We do not want value from a new sampled goal to affect actions for the previous goal.

If early latent-goal success is used:

```text
terminated = True
truncated  = False
```

but initially do **not** use early termination. Always run 10 steps for simpler learning and cleaner metrics.

## 8.3 Full task episode time limit

For a normal 100-step Push-T episode ending only due to time limit:

```text
terminated = False
truncated  = True
bootstrap  = True
```

If the simulator has a true failure/invalid state:

```text
terminated = True
truncated  = False
bootstrap  = False
```

If task success is detected during evaluation, it can be logged. Do not use success termination for training unless that is explicitly an evaluation-only wrapper.

## 8.4 Held-goal segment boundaries inside full hierarchy rollouts

If training in full hierarchy mode and the physical rollout continues across goals, do not let GAE propagate across held-goal boundaries.

There are two valid implementations:

### Preferred: segment-as-episode buffer

Collect each 10-step segment as a separate RL episode for PPO, even if the simulator continues physically afterward.

At each goal boundary:

```text
advantage_done = True
env_reset = False
bootstrap = False for local latent objective
```

This requires custom buffer logic. Do not abuse environment `done` if it forces a simulator reset.

### Alternative: true reset at every segment

At every 10-step segment boundary, reset the simulator to a newly sampled state from the state-loadable dataset. This is simpler and preferred for Mode A/B.

## 8.5 rsl_rl-specific audit

If using `rsl_rl`, explicitly verify:

1. `dones` passed to the rollout buffer reflect true terminal local episodes.
2. `time_outs` are only set for time-limit truncations where bootstrapping is desired.
3. Local 10-step goal episodes are not accidentally marked as `time_outs`.
4. GAE does not bootstrap from the next sampled goal.
5. Reset states after `done` are sampled independently from the state dataset.
6. Episode length statistics correspond to 10 steps for local training.
7. Reward normalization, value normalization, and observation normalization do not mix training and evaluation data improperly.
8. The final transition of a local episode has no value leakage from the next reset.

Gate:

```text
unit test with deterministic toy reward proves returns equal hand-computed 10-step returns
```

Do not run expensive RL before this gate passes.

---

# 9. Massive Parallelism and Throughput Gate

## 9.1 Goal

Use as many environments as reasonably fit in GPU memory and maintain stable throughput.

The previous `32 envs x 32 steps` setting is too small for a serious GPU-parallel RL conclusion.

## 9.2 Throughput benchmark

Before training, run a throughput sweep.

Test:

```text
num_envs in {128, 256, 512, 1024, 2048, 4096, 8192, 16384}
rollout_len in {10, 16, 32, 64}
```

Do not assume all will fit. Measure.

For each point log:

```text
GPU memory allocated
GPU utilization
steps/sec simulator only
steps/sec sim + render
steps/sec sim + render + DINO
steps/sec sim + render + DINO + VAE + policy
wall-clock per PPO update
crashes/NaNs
```

## 9.3 Required minimum

Target:

```text
minimum serious setting: >= 512 envs
preferred setting: >= 2048 envs
ideal setting: >= 8192 envs if memory allows
```

Effective PPO batch:

$$
B = N_env * T_rollout
$$

Required:

```text
B >= 32,768 for development
B >= 65,536 preferred
```

Examples:

```text
512 envs x 64 steps    = 32,768 samples/update
1024 envs x 64 steps   = 65,536 samples/update
2048 envs x 32 steps   = 65,536 samples/update
8192 envs x 16 steps   = 131,072 samples/update
16384 envs x 10 steps  = 163,840 samples/update
```

## 9.4 If DINO is the bottleneck

If DINO inference prevents large environment counts:

1. batch DINO over all envs;
2. use fp16/bfloat16;
3. use `torch.no_grad()`;
4. use `torch.compile` if stable;
5. profile render versus DINO versus VAE;
6. consider storing local observations for teacher actions only for audits, but not for on-policy RL transitions.

Do not silently reduce to 32 envs. If 32 envs is the maximum, document the bottleneck and treat the result as small-scale only.

---

# 10. PPO / rsl_rl Configuration Gate

The exact implementation can use `rsl_rl`, but the following must be audited.

## 10.1 Initial PPO settings

For local 10-step episodes:

```text
num_envs: largest stable from throughput gate
num_steps_per_env: 10 or 20
episode_length: 10
gamma: 0.99
gae_lambda: 0.95
ppo_epochs: 3 to 5
minibatches: choose so minibatch size >= 4096
learning_rate: 3e-4 for residual, 1e-4 or 3e-5 for direct fine-tune
clip_param: 0.2
entropy_coef: small, e.g. 0.0 to 0.005
value_loss_coef: 1.0
max_grad_norm: 1.0
normalize_advantages: True
```

For residual policies:

```text
initial residual mean = 0
initial logstd around -2.3 or lower
residual scale alpha in {0.05, 0.10, 0.25}
```

For direct policies:

```text
initial logstd <= -4.0
small learning rate
BC regularization required
```

## 10.2 PPO gate metrics

During the first 100k steps, log:

```text
mean return
mean final latent distance
goal reach rate
policy KL
clip fraction
entropy
value loss
explained variance
action saturation
residual/action drift from BC
NaN count
reset/state-load failures
```

Abort if:

```text
NaNs appear
clip fraction > 0.5 for many updates
action saturation doubles relative to frozen policy
residual/action drift grows before latent distance improves
value loss explodes
state-load failures occur
```

---

# 11. RL Methods to Run

## R1: residual deterministic low-level PPO

Base:

$$
a_t^{BC}=pi_{BC}(x_t)
$$

Residual:

$$
a_t=
clip(a_t^{BC}+alpha tanh(Delta a_t), -1, 1)
$$

Run first.

## R2: residual flow low-level PPO

Pretrain an action-flow low-level policy on the regenerated data.

Use zero-noise endpoint as base:

$$
a_t^{flow}=F(x_t, epsilon=0)
$$

Residual:

$$
a_t=
clip(a_t^{flow}+alpha tanh(Delta a_t), -1, 1)
$$

Run regardless of R1 outcome after the throughput and local-reset gates pass. The previous decision to skip it is not valid for this rerun.

## R3: direct deterministic low-level fine-tuning

Start from BC low-level.

Test:

```text
final layer only
last two layers
all layers only if final/last-two are stable
```

Use BC action regularization.

## R4: direct flow low-level tuning

Do not use PPO unless final-action log probability is correctly implemented.

Preferred first direct-flow method:

```text
zero-noise flow endpoint as deterministic actor
critic trained on latent-reaching reward
flow-matching + BC regularization on demo batches
```

Alternative:

```text
Q-weighted flow matching
```

Run R4 only after R2 establishes that the flow low-level base is stable.

---

# 12. Training Schedule and Gates

## Phase A: data/state gate

Pass before training models.

Required:

```text
new dataset has full simulator states
state-load 10-step replay passes tolerance
nested 500/1000 splits defined
old checkpoints forbidden in main comparison
```

## Phase B: supervised retraining gate

Pass before RL.

For each `N_demo in {500,1000}` and seed:

```text
train VAE/high/low from scratch
evaluate frozen hierarchy
compare to previous scaling reference
```

Gate:

```text
frozen hierarchy is not catastrophically worse than old reference
```

## Phase C: throughput gate

Pass before RL.

Required:

```text
largest stable num_envs found
effective PPO batch >= 32,768
DINO/render/VAE bottleneck identified
```

If effective batch is below 32,768, label all results as small-scale exploratory.

## Phase D: RL correctness gate

Pass before long RL.

Required:

```text
termination/truncation unit tests pass
GAE hand-computation unit test passes
local episode length = 10
no value leakage across goals
frozen policy reproduced in RL wrapper
zero residual exactly matches frozen policy
```

## Phase E: R1 local-goal gate at N=500

Train with Mode A local demo-goal episodes.

Interaction budget:

```text
1M steps smoke/development
```

Pass if:

```text
final latent distance improves by >= 25%
goal reach rate improves by >= 15 percentage points
no task-success degradation in evaluation-only full rollouts
```

If not passed, tune reward scale/residual scale and continue up to:

```text
5M steps
```

Do not move to full hierarchy training until this passes.

## Phase F: R1 local predicted-goal gate

Train with Mode B.

Pass if:

```text
latent reach improves by >= 20%
full-hierarchy evaluation improves by >= 5 points
```

## Phase G: R1 full-hierarchy gate

Train or fine-tune with Mode C only after local gates pass.

Pass at `N_demo=500` if:

```text
success_RL_500 >= success_frozen_500 + 0.10
```

and:

```text
clean evaluation success does not drop by > 5 points
```

## Phase H: flow gates

Run R2 and R4 after R1 local gates pass.

R2 pass:

```text
R2 >= R1 on local latent reach
or
R2 improves recovery/disturbed evaluation
```

R4 pass:

```text
R4 >= max(R1,R2) on local latent reach or task success
and does not damage imitation action quality
```

## Phase I: N=1000 confirmation

Repeat only the best two methods.

Pass if:

```text
success_RL_1000 >= success_frozen_1000 + 0.05
```

If N=500 passes but N=1000 fails, report RL as a small-data improvement only.

## Phase J: final multi-seed evaluation

For final methods:

```text
3 seeds
500 unseen clean episodes per seed
500 disturbed episodes per seed
local goal-reaching evaluation on held-out states
branch-oracle goal evaluation
```

Use final test seeds only once.

---

# 13. Evaluation Protocol

## 13.1 Checkpoint selection

For project-clean selection, choose checkpoints by validation latent reachability:

```text
final latent distance
goal reach rate
residual/action regularization
```

Do not select by task success on the final evaluation bank.

A separate analysis table may show which checkpoint would have been selected by task success, but it must be labeled as proof-of-concept-only.

## 13.2 Required evaluations

For every serious checkpoint:

1. Local demo-goal evaluation.
2. Local predicted-goal evaluation.
3. Full hierarchy clean evaluation.
4. Disturbed evaluation.
5. Recovery-state evaluation.
6. Branch-oracle goal evaluation.
7. Frozen baseline with identical seeds.

## 13.3 Metrics

Main latent metrics:

```text
initial latent distance
final latent distance
distance reduction
goal reach rate
per-step latent progress
```

Main task metrics, evaluation only:

```text
success
final reward
max reward
episode length
failure mode
```

Safety metrics:

```text
residual/action drift
action saturation
action smoothness
BC action distance
NaN/reset failures
```

RL diagnostics:

```text
steps/sec
policy KL
clip fraction
entropy
value loss
explained variance
advantage mean/std
```

---

# 14. Final Decision Rules

## Strong positive

Claim low-level RL works if:

```text
N=500: +10 points success over frozen
N=1000: +5 points success over frozen
local latent reach improves >=25%
3-seed result positive
no clean degradation
```

## Useful partial positive

Claim limited usefulness if:

```text
N=500 improves strongly
N=1000 does not improve
```

Interpretation:

```text
RL helps weak low-level policies but is not needed once imitation data is sufficient.
```

## Negative but credible

Only claim negative if all are true:

```text
state-loadable local reset data was used
>=512 envs and effective PPO batch >=32768
local latent reward was clean
R1 and R2 were tested
at least one direct fine-tuning method was tested
N=500 and N=1000 were evaluated
termination/truncation audits passed
```

If these conditions are not met, the result is preliminary, not a real negative.

---

# 15. Deliverables

Produce:

```text
rl_rerun_state_dataset_spec.md
rl_rerun_state_load_audit.md
rl_rerun_throughput_benchmark.csv
rl_rerun_algorithm_audit.md
rl_rerun_experiment_log.md
rl_rerun_final_results.md
rl_rerun_learning_curves.png
rl_rerun_failure_videos/
```

The final results must explicitly state:

```text
number of environments
rollout length
effective batch size
total RL steps
wall-clock time
GPU memory
whether task reward was used in training
whether local resets were exact
whether terminations/truncations were handled correctly
```

---

# 16. Summary of What Must Change Relative to the Previous Run

| Issue in previous run | Required change |
| --- | --- |
| No simulator states in dataset | Regenerate state-loadable dataset |
| Old checkpoints used | Retrain VAE/high/low from new data |
| Full hierarchy fallback | Train first on exact local 10-step goal resets |
| 32 environments | Use largest stable GPU-parallel env count |
| Task-progress shaping | Keep it evaluation-only or separate ablation |
| R2/R4 skipped | Run flow residual and direct-flow branches after gates |
| GAE bug discovered mid-run | Add termination/truncation unit tests before RL |
| No strong final gates | Use explicit gates before concluding positive or negative |
