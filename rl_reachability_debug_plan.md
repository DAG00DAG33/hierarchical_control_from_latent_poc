# RL Reachability Debugging Plan for Hierarchical Control from Latents

## 1. Purpose

The current imitation-learning experiments have explored the main flat, VAE512, TCP, and effect32 hierarchy variants. The remaining uncertainty is the RL stage:

```text
Can online RL train or fine-tune a low-level controller so that fixed high-level latent goals become more reachable?
```

The current evidence does **not** justify the conclusion that reachability-based RL is impossible. It suggests that the current formulation has not yet passed the necessary sanity gates.

This plan orders the next experiments from easiest to hardest so that each failure is interpretable.

The desired progression is:

```text
1. PPO mechanics audit
2. privileged/TCP local scratch PPO
3. random shooting / CEM local upper bound
4. improved learned distance ensemble D_psi
5. low-dimensional PPO with D_psi
6. full visual/VAE scratch PPO
7. scratch versus pretraining comparison
8. full hierarchy deployment
```

Do not start with the full visual hierarchy again. It is too hard and too ambiguous.

---

## 2. Current Working Diagnosis

There are three plausible failure explanations:

1. **RL is not actually improving reachability.**
2. **Reachability improves locally but does not improve task success.**
3. **The low-level policy is weakly goal-conditioned, so there is little useful handle for a goal-reaching reward.**

The current evidence points mostly to explanations 1 and 3.

The VAE512 and some learned-latent low levels react much more to the current observation than to the future goal. That means the supervised low level can solve one-step imitation largely from the current frame and previous action, with weak dependence on the high-level goal.

The current RL failures should therefore be treated as:

```text
preliminary evidence that the current RL formulation is not yet improving reachability,
not final evidence that reachability-based RL is useless.
```

---

## 3. Correct Interpretation of PPO Step Counts

For high-env PPO runs, distinguish:

```text
environment samples
PPO rollout/update cycles
optimizer gradient steps
minibatch size
```

Example high-env run:

```text
4096 envs * 10 rollout steps = 40,960 samples per PPO update
1,024,000 total env steps / 40,960 = 25 PPO update cycles
2 update epochs * 64 minibatches = 128 optimizer steps per PPO update
25 PPO updates * 128 optimizer steps = 3,200 gradient steps
```

This is not tiny in gradient-step count, but it is still only 25 fresh-data policy iterations.

Example old small-env run:

```text
32 envs * 10 rollout steps = 320 samples per PPO update
100,160 env steps / 320 = 313 PPO update cycles
313 * 4 epochs * 8 minibatches = 10,016 gradient steps
```

This has many more fresh-policy iterations, but each update uses a very small and noisy batch.

A serious next PPO configuration should combine large batches with enough update cycles:

```text
num_envs = 4096 or more if stable
rollout_len = 10 for local episodes
samples/update = num_envs * rollout_len
num_minibatches in {8, 16}
update_epochs in {3, 5}
PPO updates >= 250 for a real development run
PPO updates >= 500 for a serious run if learning is observed
```

Avoid using 64 minibatches as the default with a 40,960-sample rollout, because it gives minibatches of only about 640 samples. Prefer minibatches of a few thousand samples.

---

## 4. Non-Negotiable Audits Before New RL Runs

## 4.1 Low-level input update audit

During a 10-step held-goal period:

```text
current observation must update every primitive step
held goal must stay fixed
remaining time must update every primitive step
previous action must be the clipped executed action
```

The intended low-level policy is:

$$
a_t = \pi_{low}(o_t, g_i, a_{t-1}^{exec}, \tau_t)
$$

not:

$$
a_t = \pi_{low}(o_{t_i}, g_i, a_{t_i-1}^{exec}, \tau_i)
$$

for all ten steps.

Required tests:

| Test | Expected result |
| --- | --- |
| live current observation vs cached start observation | live should perform better or at least change actions |
| live previous action vs cached previous action | actions should change if previous action is used |
| live remaining time vs constant remaining time | actions should change if time conditioning is used |
| goal shuffle | should hurt if the policy uses goals |
| observation shuffle | should strongly hurt |

## 4.2 Termination and truncation audit

For local 10-step goal-reaching episodes:

```text
terminated = True
truncated  = False
bootstrap  = False
```

Reason: the 10-step local goal deadline is the terminal condition of the local MDP. Do not bootstrap into the next sampled goal.

For normal full-task time limits:

```text
terminated = False
truncated  = True
bootstrap  = True
```

If using `rsl_rl`, verify:

```text
dones are true local terminals
time_outs are only time-limit truncations
GAE does not leak across new sampled goals
local episode length is exactly 10
reward/value normalization is not mixing train/eval incorrectly
```

Required unit test:

```text
A deterministic toy 10-step reward sequence must produce the same returns as a hand calculation.
```

## 4.3 PPO diagnostics audit

Every PPO run must log:

```text
mean return
terminal distance
goal reach rate
policy KL
clip fraction
entropy
value loss
explained variance
action saturation
action/residual drift from base
NaN count
reset/state-load failures
```

Abort or debug if:

```text
NaNs appear
clip fraction is high for many updates
action saturation rises sharply
residual/action drift grows before distance improves
value loss explodes
state-load failures occur
```

---

## 5. Required Metrics for Every Experiment

## 5.1 Local reachability metrics

On a fixed local reset bank, report:

```text
initial distance to goal
terminal distance to goal
distance reduction
goal reach rate under epsilon
p50/p90/p99 terminal distance
fraction improved over frozen base
paired improvement relative to frozen base
action saturation
action magnitude or residual magnitude
```

## 5.2 Full rollout metrics

Even if the training reward is local, evaluate full behavior.

For each low-level checkpoint, evaluate:

```text
learned high-level goals
oracle/replay goals
shuffled goals
```

Report:

```text
task success
final reward
max reward
episode length
failure mode
```

Task success and environment reward are evaluation metrics only unless explicitly marked as a proof-of-concept ablation.

## 5.3 Success rate with a matching high-level policy

For every low-level representation, measure full hierarchy success using a high-level policy trained for that same representation.

| Low-level representation | Required high-level success metric |
| --- | --- |
| privileged/TCP state | success using high level trained to predict privileged/TCP future goals |
| VAE512 latent | success using high level trained to predict VAE512 future latents |
| effect32 latent | success using high level trained to predict effect32 goals |
| new reachability latent | success using high level trained for that latent |

This is not a pure reachability metric because it includes high-level prediction error. However, it is still a main metric of interest because it answers:

```text
If this low level were used in the actual hierarchy, how well would it work?
```

Always pair this with oracle-goal success:

```text
learned-goal success  = full deployable hierarchy
oracle-goal success   = low-level ceiling with good goals
shuffled-goal success = goal-use sanity check
```

---

# 6. Experiment Order

## Phase 1: PPO From Zero on Privileged/TCP Local Reaching

### Purpose

Prove that the PPO loop can solve the easiest local goal-reaching problem.

### Setup

```text
input: current privileged state
goal: future privileged/TCP state
horizon: k = 10
reset: exact local state reset
policy init: random
reward: true local state/TCP distance
no VAE
no DINO
no learned distance metric
no high-level prediction
```

This is the most important next experiment.

### Reward

Use a simple low-dimensional distance:

$$
d_t = \|s_t^{priv} - g^{priv}\|_W^2
$$

Reward:

$$
r_t = d_t - d_{t+1}
$$

Terminal:

$$
r_T = -d_T
$$

### Training configuration

Start with:

```text
num_envs = 4096 or largest stable
rollout_len = 10
PPO updates = 250 initially
increase to 500 if learning
num_minibatches = 8 or 16
update_epochs = 3 or 5
```

### Gates

Pass if:

```text
terminal distance clearly decreases
goal reach rate increases strongly
correct-goal performance is much better than shuffled-goal performance
policy does not collapse to saturated actions
```

Also evaluate:

```text
full hierarchy success using a privileged/TCP high-level predictor
oracle-goal success using privileged/TCP goals
```

If this fails, stop and debug PPO/reward before touching VAE or visual inputs.

---

## Phase 2: Random Shooting / CEM Upper Bound on the Same Local MDP

### Purpose

Determine whether useful local action sequences exist and whether PPO is finding them.

This is a diagnostic optimizer, not necessarily a final method.

### Random shooting

For each local reset:

1. Start from the same state as PPO.
2. Sample many 10-step action sequences around the frozen/base policy.
3. Simulate each sequence in a branch environment.
4. Measure terminal distance.
5. Keep the best sequence.

Candidate sequence:

$$
a_{t:t+9}^{(i)} = a_{base,t:t+9} + \epsilon^{(i)}
$$

Test:

```text
num_candidates in {32, 64, 128, 256}
noise_std in {0.025, 0.05, 0.1}
```

### CEM

If random shooting helps, optionally run CEM:

1. sample action sequences;
2. keep the top elite fraction;
3. fit a Gaussian to elites;
4. resample;
5. repeat 3-5 iterations.

### Interpretation

| Outcome | Meaning |
| --- | --- |
| CEM/random improves but PPO fails | PPO/training is the bottleneck |
| CEM/random also fails | reward/distance/action space is likely bad |
| PPO approaches CEM/random | PPO loop is working |

Run this on:

```text
full local bank
hard subset where frozen base fails
```

---

## Phase 3: Train a Better Learned Distance Ensemble

### Purpose

Raw VAE L2 and a single learned distance model can be unreliable and exploitable. Train a robust reachability metric before using it as a PPO reward.

### Ensemble definition

Train:

$$
D_{\psi_j}(z_t, g, \tau), \quad j=1,\ldots,M
$$

Use conservative distance:

$$
D_{ens}(z,g,\tau) = \mu_D(z,g,\tau) + \beta \sigma_D(z,g,\tau)
$$

where:

```text
mu_D    = ensemble mean
sigma_D = ensemble disagreement / uncertainty
beta    = uncertainty penalty, e.g. 0.5 or 1.0
```

### Training data

Do not train only on successful teacher demonstrations.

Use:

1. demonstration teacher windows;
2. frozen low-level local rollouts;
3. failed frozen hierarchy states;
4. random-shooting/CEM branches;
5. replay/demo branches;
6. early PPO rollouts;
7. shuffled or invalid goals as negatives;
8. action-search improved and non-improved branches.

The metric must see good, bad, reachable, unreachable, and off-policy examples.

### Targets to train

Train multiple heads or separate variants:

| Target | Meaning |
| --- | --- |
| terminal distance after a branch | regression |
| reach within 10 steps | binary classification |
| improvement over frozen base | paired regression |
| branch A better than branch B | pairwise ranking |
| ensemble uncertainty | OOD / hackability indicator |

### Validation gates

Before using `D_psi` as PPO reward, verify:

```text
D_psi ranks replay/demo branches better than bad random branches
D_psi ranks CEM/random-search selected branches better than base branches
D_psi correlates with actual terminal rollout distance on held-out branches
D_psi separates reachable within-k from unreachable/shuffled goals
ensemble uncertainty rises on off-policy/weird states
candidate selection by D_psi improves actual rollout distance
```

If these fail, do not use `D_psi` for PPO.

---

## Phase 4: Low-Dimensional PPO With Learned Distance

### Purpose

Test whether the learned distance can drive control when perception is not the bottleneck.

### Setup

```text
input: current privileged state
goal: VAE/effect latent goal
reward: D_psi ensemble
policy init: random or simple privileged policy
no DINO input
exact local resets
```

This isolates the learned metric.

### Gates

Pass if:

```text
D_psi distance improves
actual rollout terminal distance also improves
oracle-goal success improves
success using the matching high-level policy is not degraded
```

If `D_psi` improves but real rollout distance or success worsens, the metric is hackable or misaligned.

---

## Phase 4B: Reset-Distribution Debugging for Full-State Low-Level PPO

### Purpose

If long same-bank full-state PPO improves local reset-bank reachability but still
has poor held-subgoal task success, do not keep extending the same reset-bank
training indefinitely. The next hypothesis is reset-distribution shift:

```text
The low level trains on states close to teacher/demo rollouts, but deployment
creates shifted states after previous imperfect subgoals.
```

This phase tests whether training on deployed hierarchy states improves actual
held-subgoal success.

### Constraint: no online expert relabeling as the core method

For the proof of concept, avoid DAgger-style online expert-action relabeling as
the main solution. In the real setting, an expert will not be available.

Allowed use of oracle/teacher branching:

```text
diagnostic only
upper-bound evaluation only
```

Do not make online expert action labels the central POC method. If a
teacher-action penalty is used, treat it as regularization from already
available demonstrations and report it explicitly.

### Main reset mixture experiment

Collect low-level reset states from actual held-subgoal hierarchy rollouts and
train/evaluate the same full-state PPO objective on a mixture:

| Source | Fraction |
| --- | ---: |
| original demo/teacher local windows | 50% |
| Phase-C BC hierarchy deployed states | 25% |
| current PPO hierarchy deployed states, e.g. Run 21/22 | 25% |

Use held full-state subgoals and recomputed goal features as in the corrected
full-state evaluator. The POC-friendly version should use the subgoal target
already available to the hierarchy, not new online expert action labels.

Recommended variants:

| Variant | Regularization |
| --- | --- |
| reset mixture, no teacher-action penalty | tests pure reachability on the new distribution |
| reset mixture, existing teacher-action penalty | tests whether demo regularization preserves task-compatible actions |
| reset mixture, KL or BC-prior to Phase-C low level | tests a less direct action-manifold constraint |

### Disturbed demo/oracle reset ablation

As a cheaper controlled variant, start from demo local windows and perturb the
state before the 10-step low-level episode:

```text
small object-pose perturbations
small TCP perturbations
1-3 random, BC, or PPO warm-up steps before the subgoal starts
action noise before subgoal start
```

For small perturbations, keep the original future demo/full-state target as the
goal. For larger perturbations, use oracle teacher branching only as a diagnostic
upper bound to measure whether a good recovery target exists.

### Diagnostic: oracle target from deployed states

Optionally branch the teacher from deployed hierarchy states to compute the
ideal `t+10` full-state target. Use this to answer:

```text
If the low level had the right local target from these bad deployed states,
could it recover?
```

This separates reset-distribution failure from goal-quality failure, but it is
not the main POC method.

### Success criteria

Pass if the reset-mixture policy improves:

```text
held-subgoal task success
terminal full-goal distance on deployed states
correct-vs-shuffled goal separation
```

It must not collapse into generic corrective motions. Shuffled-goal reachability
should remain clearly worse than correct-goal reachability.

### Post-Run-22/25 decision rule

Run 22 was the stop point for simply extending PPO on the original demo reset
bank: same-bank local metrics improved, but held-subgoal task success remained
poor. Do not make "train longer on the same reset bank" the next main branch.

Run 25 showed that reset-mixture training plus a BC-warm-started actor can
recover nonzero full-state PPO held-subgoal success, but it is still below the
Phase-C full BC baseline. The next main experiments should therefore keep
changing the reset-state distribution while adding stronger BC structure:

| Next option | Purpose | Expert use |
| --- | --- | --- |
| residual-on-Phase-C-BC PPO on reset mixture | preserve BC contact behavior while learning corrections | no online expert labels |
| KL/BC-prior regularized PPO on reset mixture | keep actions near the BC manifold without freezing the policy | no online expert labels |
| disturbed demo/deployed resets | train recovery around shifted states cheaply | no online expert labels for small perturbations |
| oracle target from deployed states | upper-bound whether correct local targets solve recovery | diagnostic only |

Preferred next run:

```text
reset distribution:
  50% original demo local windows
  25% Phase-C BC hierarchy deployed states
  25% current PPO hierarchy deployed states

policy structure:
  residual-on-Phase-C full BC or explicit KL/BC-prior regularizer

evaluation:
  held full-state oracle subgoals
  learned-high subgoals when available
  deployed-state branch reachability
  correct-vs-shuffled goal separation
  teacher-action MAE only as diagnostic, not as online relabeling
```

If this still gives poor task success, debug reachability specifically on
states generated by rolling out the hierarchy itself. Measure whether the
low-level reaches high-level-generated full-state subgoals from those deployed
states better than Phase-C BC, and whether it preserves task-compatible contact
actions.

### Iterative reset-bank aggregation

If a static reset mixture with BC warm start or BC-prior regularization still
trails the Phase-C BC baseline, move to an iterative dataset aggregation loop.
This is the preferred next branch after Run 26.

Algorithm:

```text
initial bank:
  expert/demo local windows
  Phase-C BC hierarchy deployed trajectories with modest task success

round i:
  train or continue BC-structured PPO on current reset bank
  deploy the learned low level inside the hierarchy with the learned high-level policy
  record the resulting hierarchy rollout states
  record the held full-state subgoals produced by the hierarchy
  add these learned-policy deployed states to the reset bank
  continue training from the current PPO checkpoint
```

Important constraint:

```text
Do not use online expert action relabeling as the core method.
The main target for deployed states is the subgoal already produced by the hierarchy.
Teacher/oracle branches are allowed only for diagnostics or upper bounds.
```

Recommended first aggregation run:

| Component | Setting |
| --- | --- |
| base checkpoint | latest BC-structured PPO, e.g. Run 26 |
| bank round 0 | demo windows + Phase-C BC deployed states + Run 26 deployed states |
| target source | learned high-level full-state subgoal |
| continuation | load Run 26 checkpoint and normalizers |
| regularization | keep BC-prior loss; test residual-on-BC later |
| evaluation | held oracle subgoals, learned-high subgoals, deployed-state branch reachability, shuffled goals |

Stop criteria for an aggregation round:

```text
oracle held success improves
learned-high success improves or stays neutral
shuffled-goal success remains low
deployed-state terminal full-goal distance improves on the newest rollout distribution
```

---

## Phase 5: Full Visual/VAE PPO From Zero

### Purpose

Test whether PPO from zero can learn the real visual/proprio goal-conditioned low-level controller.

### Setup

```text
input: DINO/proprio current observation
goal: VAE512 or effect latent
reward: D_psi ensemble
policy init: random
local 10-step reset
```

Use serious training:

```text
num_envs = 4096 or largest stable
rollout_len = 10
PPO updates = 250-500 minimum
```

### Important: retrain the distance ensemble

The distance ensemble used here must be retrained or adapted for the full visual/VAE low-level setting.

Reason:

```text
A distance model trained for privileged inputs or privileged-state branch data is not the same as a distance model for the full visual/VAE low-level setting.
The full visual low level sees DINO/proprio observations and VAE/effect goals.
The metric must be trained on the same style of states, goals, and on-policy rollouts that the visual policy will encounter.
```

Training data for the visual/VAE `D_psi` should include:

```text
visual/DINO/proprio rollouts
VAE latents from visual observations
frozen visual low-level rollouts
scratch visual PPO rollouts
random/CEM branches in the visual setting
failed full visual hierarchy states
shuffled-goal negatives
```

Do not reuse a privileged-only distance model as the main reward for the visual low-level experiment.

### Gates

Pass if:

```text
local reachability improves over frozen visual low level
correct-goal performance exceeds shuffled-goal performance
full hierarchy success with learned high-level improves or remains neutral
oracle-goal success improves
```

---

## Phase 6: Scratch Versus Pretraining

### Purpose

Once scratch PPO works locally, test whether imitation pretraining helps or hurts.

Compare:

| Initialization | Meaning |
| --- | --- |
| scratch | can PPO learn goal use from zero? |
| BC warm start, full policy trainable | does imitation help? |
| residual on frozen BC | conservative adaptation |
| final-layer tuning | small adaptation |
| all-layer tuning | aggressive adaptation |

Do not evaluate only task success. Report:

```text
goal sensitivity
correct-goal vs shuffled-goal local performance
local terminal distance
oracle-goal full hierarchy success
learned-high full hierarchy success
action saturation
action drift from BC
```

Interpretation:

| Result | Meaning |
| --- | --- |
| scratch learns goal use, BC does not | pretraining creates goal-ignoring inertia |
| BC warm start wins | imitation helps exploration/stability |
| residual wins | BC is mostly good; only small corrections needed |
| all fail | reward/metric/representation is still wrong |

---

## Phase 7: Full Hierarchy Deployment

Only run this after local reachability passes.

Evaluate:

```text
learned high-level goals
oracle/replay high-level goals
shuffled goals
disturbed resets
recovery states
clean task resets
```

For each low-level representation, use the matching high-level predictor trained for that representation.

Required table:

| Policy | Learned-goal success | Oracle-goal success | Shuffled-goal success |
| --- | ---: | ---: | ---: |
| frozen BC low level | | | |
| scratch PPO low level | | | |
| residual PPO low level | | | |
| BC warm-start PPO low level | | | |

The learned-vs-oracle gap indicates whether the remaining bottleneck is high-level prediction or low-level reachability.

---

# 7. Immediate Run List

## Run 1: PPO mechanics audit

```text
goal: prove rollout buffer, done/truncation, GAE, and input update semantics are correct
cost: low
```

## Run 2: Privileged/TCP scratch PPO local

```text
goal: prove PPO can learn local 10-step goal reaching from zero
input: privileged state + privileged goal
reward: true state/TCP distance
num_envs: 4096+
updates: 250
```

## Run 3: Random shooting / CEM on privileged local bank

```text
goal: quantify local action-improvement upper bound
compare: frozen, replay/demo, random shooting, CEM, PPO
```

## Run 4: D_psi ensemble training

```text
goal: produce a less hackable reachability metric
data: demos + frozen rollouts + random/CEM branches + failures + early PPO
```

## Run 5: Low-dimensional PPO with D_psi

```text
goal: test learned metric without visual/perception bottleneck
input: privileged state
goal/reward: VAE/effect latent + D_psi ensemble
```

## Run 6: Full visual/VAE scratch PPO

```text
goal: test real visual low-level from zero
requires: visual/VAE-specific D_psi ensemble
```


## Run 6B: Visual/VAE scratch PPO privileged-distance curriculum

```text
goal: determine whether visual scratch PPO can learn when the reward distance is privileged/perfect
input: DINO/proprio + VAE512 goal
low-level BC demos: 0
VAE/high-level demos: 1800
reward variants: privileged distance -> calibrated distance -> non-privileged D_psi ensemble
required eval: expert PPO, oracle goal, learned-high N=1800, shuffled goal
```

## Run 7: Pretraining ablation

```text
goal: compare scratch vs BC warm-start vs residual
run only after scratch works locally
```

## Run 8: Full hierarchy evaluation

```text
goal: measure actual task success with matching high-level predictor
run only after local reachability passes
```

---

# 8. Decision Rules

## Strong positive

Claim reachability PPO works if:

```text
local reachability improves clearly over frozen base
correct-goal performance exceeds shuffled-goal performance
oracle-goal full hierarchy success improves
learned-high full hierarchy success improves or remains neutral
result holds across seeds
```

## PPO implementation failure

Conclude PPO/training loop is the bottleneck if:

```text
random/CEM improves the local bank
replay/demo actions improve the local bank
but PPO from zero fails on privileged/TCP local reaching
```

## Distance metric failure

Conclude the distance metric is the bottleneck if:

```text
PPO works with true privileged distance
but fails with D_psi or raw VAE distance
and D_psi fails offline ranking/candidate-selection tests
```

## Representation or goal-conditioning failure

Conclude representation/conditioning is the bottleneck if:

```text
PPO works with privileged state and true distance
PPO works with D_psi in low-dimensional setup
but full visual/VAE PPO fails or ignores goals
```

## Pretraining failure

Conclude pretraining hurts goal use if:

```text
scratch PPO learns goal-sensitive behavior
but BC warm-start/residual policies remain goal-insensitive or worse
```

## Reachability-task mismatch

Only conclude reachability does not help task success if:

```text
local reachability reliably improves
oracle-goal full hierarchy improves or remains strong
but learned-high task success does not improve
```

Do not claim reachability-task mismatch before proving actual local reachability improvement.

---

# 9. Reporting Requirements

Maintain:

```text
rl_reachability_debug_experiment_log.md
rl_reachability_debug_final_results.md
```

Every log entry should include:

```text
hypothesis
command
git commit
dataset / reset bank
num_envs
rollout length
samples/update
PPO updates
gradient steps
reward
termination/truncation setting
frozen baseline
local metrics
full hierarchy metrics
interpretation
next action
```

Required plots:

```text
terminal distance vs PPO update
goal reach rate vs PPO update
policy KL and clip fraction vs update
correct-goal vs shuffled-goal performance
success with matching high-level predictor
D_psi predicted distance vs actual rollout distance
ensemble uncertainty vs rollout error
```

---

# 10. Main Takeaway

Do not ask first:

```text
Can the full visual hierarchy be improved by PPO right away?
```

Ask, in order:

```text
1. Can PPO solve the simplest local goal-reaching MDP?
2. Are local improvements available according to random shooting / CEM?
3. Can a learned distance metric rank those improvements reliably?
4. Can PPO optimize that learned distance in a low-dimensional setting?
5. Can the same idea survive full visual/VAE inputs?
6. Does pretraining help or hurt goal-conditioned RL?
7. Does improved local reachability improve full hierarchy success?
```

Only after these gates pass or fail should the project make a strong conclusion about reachability-based low-level RL.


## 4.3 Oracle success rate is a required metric

For every serious checkpoint, measure success with **oracle goals** in addition to learned high-level goals.

The oracle-goal evaluation replaces the high-level model output with the true reachable future goal for the current representation:

| Representation | Oracle goal used for evaluation |
| --- | --- |
| privileged/TCP | future privileged/TCP state from teacher branch or replay |
| VAE512 | future VAE512 latent from teacher branch or replay |
| effect32 | future effect32 target from teacher branch/replay or encoded future pair, depending on the interface |
| learned reachability latent | future goal encoded with the same learned representation |

The required success metrics are:

```text
learned-high success
oracle-goal success
shuffled-goal success
```

Interpretation:

| Pattern | Meaning |
| --- | --- |
| oracle improves but learned does not | high-level prediction or goal calibration is the bottleneck |
| learned and oracle both improve | low-level reachability improvement transfers to deployment |
| local reach improves but oracle success does not | reachability metric is misaligned or local training distribution is wrong |
| oracle success improves but learned success drops | low level became sensitive to good goals but incompatible with learned goals |
| shuffled success close to correct-goal success | low level is not using the goal enough |

Do not claim that low-level RL helped the hierarchy unless oracle-goal success is at least neutral and preferably improved.

# 8. Reporting Requirements

## 8.1 Mandatory Markdown experiment log

Keep a running Markdown experiment log throughout the entire RL debugging phase.

Required file:

```text
rl_reachability_debug_experiment_log.md
```

This file must be updated as experiments are run, not reconstructed at the end.

Each entry should include:

```text
date
hypothesis
command
git commit
dataset/reset bank
representation
high-level goal source
oracle goal source
num_envs
rollout length
samples per PPO update
PPO update cycles
gradient steps
minibatch size
reward
termination/truncation handling
checkpoint path
local reachability metrics
learned-high success
oracle-goal success
shuffled-goal success
interpretation
next action
```

Failed runs must be logged. In this project, failed runs are often the most useful evidence.

Also maintain a cleaned final report:

```text
rl_reachability_debug_final_results.md
```

The final report should summarize only accepted comparisons, tables, plots, and conclusions.


## 4.4 Data-budget interpretation for scratch low-level RL

For **scratch low-level RL**, the usual demonstration budget does not train the low-level policy directly. The low level is trained from simulator interaction.

Therefore, separate these quantities in every table:

| Quantity | Meaning |
| --- | --- |
| `N_high` | number of recorded trajectories used to train the representation and high-level model |
| `N_low_BC` | number of recorded trajectories used to pretrain the low level; for scratch RL this is `0` |
| `N_RL_steps` | number of online simulator transitions used to train the low level |

For the main visual/VAE scratch PPO experiments, use:

```text
N_high = 1800 recorded trajectories
N_low_BC = 0
```

That means the VAE representation and high-level predictor are the same strong pretrained hierarchy components, but the low-level policy starts from random initialization and learns only from RL.

Use the high-level model trained with `1800` recorded trajectories for learned-high success. This avoids mixing the scratch low-level question with a weak high-level predictor.

Oracle-goal success should also be reported. In oracle mode, the high-level goal is replaced by the true reachable future goal from a teacher branch or replay, so it is the cleanest measure of low-level reachability under a perfect high-level goal.

For perspective, every oracle-goal success table must also report the success rate of the privileged expert PPO policy used to generate the oracle branch on the same evaluation seed bank. This expert success is the practical upper bound for oracle-goal evaluation.

Required columns:

| metric | description |
| --- | --- |
| `expert_ppo_success` | success of the privileged teacher/expert on the same eval seeds |
| `oracle_goal_success` | success of the tested low level with oracle future goals |
| `learned_high_success_N1800` | success using the 1800-demo pretrained high-level model |
| `shuffled_goal_success` | goal-use sanity check |


## Phase 5B: Visual/VAE Scratch PPO With Privileged-Distance Curriculum

### Purpose

This is an important bridge experiment for the visual/VAE scratch PPO path.

The final project-compatible reward should not use privileged state. However, for debugging it is useful to ask:

```text
Can the visual/VAE scratch low level learn if the reward distance is perfect or privileged?
```

If scratch PPO fails even with a privileged distance metric, the issue is likely the RL loop, policy architecture, action space, or exploration.

If scratch PPO works with privileged distance but fails with the non-privileged learned distance, the bottleneck is the learned distance metric.

### Setup

Use the same low-level observation and action interface as the real visual/VAE scratch policy:

```text
low-level input:
  current DINO/proprio observation
  VAE512 future latent goal
  previous executed action
  remaining time

policy initialization:
  random low-level policy

representation/high-level:
  VAE512 encoder trained from N_high = 1800 demonstrations
  high-level predictor trained from N_high = 1800 demonstrations

low-level demonstrations:
  N_low_BC = 0
```

The only difference between variants is the reward distance.

### Reward variants

Run the following in order.

#### Variant A: privileged physical distance

Use privileged simulator state only for the reward:

```text
reward distance = structured privileged state/TCP/object distance to the oracle future state
```

The policy input does **not** receive privileged state. Only the reward uses it.

This is a proof-of-concept/debugging reward, not the final method.

#### Variant B: privileged-to-latent calibrated distance

Use privileged state to train a calibrated distance model, then freeze it:

```text
D_priv_calibrated(z_t, g, o_t) -> reachability distance
```

The training labels come from privileged branch outcomes, but the deployed reward model consumes non-privileged observations/latents.

This is still not fully real-compatible if trained from privileged labels, but it tests whether privileged supervision can produce a usable reward model.

#### Variant C: non-privileged learned distance ensemble

Use the project-compatible learned distance ensemble:

```text
D_ens(z_t, g, tau) = mean(D_j) + beta * std(D_j)
```

This ensemble must be trained or adapted for the full visual/VAE low-level setting, not reused from privileged-state experiments.

Training data should include:

```text
visual/DINO/proprio rollouts
VAE latents from visual observations
frozen visual low-level rollouts
scratch visual PPO rollouts
random/CEM branches in the visual setting
failed visual hierarchy states
shuffled-goal negatives
```

### Training curriculum

Run:

1. Visual scratch PPO with privileged physical distance.
2. Continue or restart with privileged-to-latent calibrated distance.
3. Continue or restart with non-privileged learned distance ensemble.

Use both continuation and restart comparisons:

| transition | purpose |
| --- | --- |
| privileged reward -> learned reward continuation | tests curriculum transfer |
| learned reward from scratch | tests whether curriculum is necessary |
| privileged reward only | upper-bound debugging baseline |

### Required success evaluation

For every checkpoint, evaluate:

```text
expert_ppo_success
oracle_goal_success
learned_high_success_N1800
shuffled_goal_success
```

Where:

- `expert_ppo_success` is the privileged PPO teacher success on the same eval bank;
- `oracle_goal_success` uses perfect future VAE goals from teacher branch/replay;
- `learned_high_success_N1800` uses the high-level predictor trained from 1800 recorded trajectories;
- `shuffled_goal_success` tests whether the low level actually uses the goal.

### Interpretation

| Outcome | Meaning |
| --- | --- |
| privileged-distance PPO fails | RL loop/policy/action-space problem |
| privileged-distance PPO works, learned-distance PPO fails | distance metric problem |
| privileged curriculum works but learned from scratch fails | learned reward is usable but exploration/curriculum matters |
| oracle success improves but learned-high success does not | high-level prediction/calibration bottleneck |
| learned-high and oracle both improve | true low-level reachability improvement transfers to hierarchy |
| shuffled-goal success remains high | low level is not actually using the goal |

### Gate

Do not claim that visual/VAE scratch PPO works unless:

```text
oracle_goal_success improves over the frozen or scratch baseline
learned_high_success_N1800 is neutral or improved
shuffled_goal_success is clearly lower than correct-goal success
expert_ppo_success is reported on the same eval bank
```


Additional required fields for scratch low-level RL logs:

```text
N_high
N_low_BC
N_RL_steps
expert_ppo_success
oracle_goal_success
learned_high_success_N1800
shuffled_goal_success
reward_distance_type
whether privileged information was used in training reward
whether privileged information was used in policy input
```

For scratch visual/VAE PPO, explicitly state:

```text
low-level policy uses zero demonstration trajectories for BC pretraining
VAE/high-level are trained with 1800 demonstrations
expert PPO success is the oracle upper-bound reference
```
