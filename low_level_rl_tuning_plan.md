# Push-T Low-Level RL Fine-Tuning Plan

## 1. Objective

The imitation-learning studies have now tested the main learned interfaces and data-scaling behavior. The next question is whether **low-level RL fine-tuning** can improve the hierarchy by making the low level better at reaching the latent goal provided by a fixed high-level model.

The core idea is:

```text
freeze the learned representation
freeze the high-level goal model
fine-tune only the low-level controller
optimize local reachability of the held latent goal
```

The main target is the selected VAE-512 future-state interface:

```text
interface: VAE-512 posterior mean
candidate: vae512_w2048_b1e6
k = 10
U = 10
H = 1
goal: future VAE latent predicted by the high level
```

The low level receives the current visual/proprio observation, the held future latent goal, the previous executed action, and the remaining-time fraction. It outputs one `pd_ee_delta_pos` action at 20 Hz.

The RL objective is not to learn a new high-level policy. It is to improve:

```text
current state + high-level latent goal -> action
```

so that the robot reaches the high-level goal more reliably, especially from off-nominal states.

---

## 2. Why Low-Level RL Now?

The VAE-512 sample-efficiency experiment shows that the hierarchy is viable but still leaves room for improvement.

At the primary RL data budgets:

| trajectories | det hierarchy | flow hierarchy | branch oracle | flat obs det |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 0.301 | 0.304 | 0.240 | 0.315 |
| 1,000 | 0.507 | 0.525 | 0.533 | 0.501 |

The 500-trajectory setting is the main RL setting because performance is low enough to leave substantial room for improvement. The 1,000-trajectory setting is the confirmation setting because imitation is already much stronger but still below saturation.

The goal of RL is therefore:

```text
primary: improve the 500-trajectory hierarchy
secondary: confirm at 1,000 trajectories
```

Do not use extra demonstration trajectories for the main comparison. Online simulator interaction is allowed and must be reported separately as RL environment steps.

---

## 3. Fixed Data and Frozen Components

### 3.1 Demonstration budgets

Run the main experiments at:

```text
N_demo = 500
```

Run confirmation experiments at:

```text
N_demo = 1000
```

Use the exact same nested trajectory prefixes and fixed validation split from the VAE-512 sample-efficiency experiment.

For each `N_demo`, use training seeds:

```text
seed in {0, 1, 2}
```

The seed controls the VAE, high level, low level, RL initialization, and any policy noise.

### 3.2 Frozen components

For the main low-level RL experiments, freeze:

1. Frozen DINOv2-small spatial image features.
2. VAE-512 encoder and decoder.
3. VAE normalizers.
4. High-level future-latent predictor.
5. High-level timing:
   ```text
   k = 10
   U = 10
   ```
6. Previous-action semantics.
7. Action clipping semantics.
8. Environment/controller:
   ```text
   ManiSkill PushT-v1
   CUDA PhysX
   pd_ee_delta_pos
   20 Hz
   ```

The high-level model must not be updated during these experiments.

### 3.3 Trainable components

The trainable component is the low-level action policy.

The plan tests four low-level RL variants:

| ID | Method | Trainable part | Purpose |
| --- | --- | --- | --- |
| R1 | residual RL on deterministic BC low level | small residual adapter only | safest first method |
| R2 | residual RL on flow low level | small residual adapter over flow endpoint | tests whether flow base helps |
| R3 | direct deterministic low-level fine-tuning | full or partial deterministic low-level MLP | stronger but riskier |
| R4 | direct flow low-level fine-tuning | low-level action-flow model | tests flow policy tuning directly |

The first serious experiment should be R1. R2-R4 should only be interpreted after R1 is stable.

---

# 4. Low-Level Goal-Reaching MDP

## 4.1 Observation

At primitive step `t`, the RL policy receives the same low-level condition as the imitation low level:

```text
x_t = [
  current 6549D DINO+proprio observation,
  held future latent goal g,
  previous executed action,
  normalized remaining time tau
]
```

where:

```text
g = high-level predicted VAE latent goal
```

or, in oracle/local-goal diagnostics:

```text
g = true future VAE latent from a demonstration window or branch rollout
```

The current latent is:

$$
z_t = \mu_{\mathrm{VAE}}(o_t)
$$

The held goal is:

$$
g_i = \hat{z}_{t_i+10}
$$

for high-level decision time `t_i`.

## 4.2 Action

The environment action is:

$$
a_t \in [-1,1]^3
$$

and is interpreted by ManiSkill as a `pd_ee_delta_pos` command.

Every final action must be clipped before execution:

$$
a_t^{exec}=\operatorname{clip}(a_t,-1,1)
$$

The next previous-action input must be the clipped executed action.

## 4.3 Episode types

Use three RL rollout modes.

### Mode A: local demonstration-goal episodes

Sample a causal demonstration trajectory and a valid timestep `t`.

Initial state:

```text
s_t from the demonstration
```

Goal:

$$
g = z_{t+10}^{demo}
$$

Roll the low level for 10 primitive steps.

This is the cleanest local goal-reaching objective. The goal is known to be reachable by the teacher from the sampled state.

### Mode B: local predicted-goal episodes

Sample a causal demonstration state `s_t`.

Goal:

$$
g = H_{\mathrm{high}}(o_t,a_{t-1})
$$

Roll the low level for 10 primitive steps.

This matches deployment better than Mode A, because the goal is the high-level model's actual predicted latent.

### Mode C: full hierarchy episodes

Run the full frozen high-level hierarchy from a normal reset.

Every 10 steps:

$$
g_i = H_{\mathrm{high}}(o_{t_i},a_{t_i-1})
$$

Hold the goal for 10 primitive steps while the low-level RL policy acts every step.

This is the final deployment setting.

## 4.4 Reset implementation

Preferred order:

1. Use exact reset-and-replay to reach stored demonstration states when possible.
2. Use direct state copy only if state-copy and transition parity are verified.
3. If exact local resets are too expensive, use full-episode hierarchy rollouts for RL and reserve exact local resets for validation.

State-copy or reset errors must be recorded. Do not train on invalid local states.

---

# 5. Reward Design

The reward should primarily train the low level to reach the held latent goal.

## 5.1 Latent distance

Use normalized VAE posterior means.

Define:

$$
d_t =
\frac{1}{D}
\left\|
\bar{z}_t-\bar{g}
\right\|_2^2
$$

where:

- `D = 512`;
- `bar` indicates the same latent normalization used by the low-level policy;
- `g` is the held high-level goal.

## 5.2 Dense progress reward

Reward improvement toward the goal:

$$
r_t^{progress}
=
d_t-d_{t+1}
$$

This is positive when the low level moves closer to the goal.

## 5.3 Terminal segment reward

At the end of a 10-step goal segment:

$$
r_T^{final}
=
-\lambda_d d_T
$$

Optionally add a goal-reaching bonus:

$$
r_T^{bonus}
=
\lambda_b \mathbf{1}[d_T < \epsilon_d]
$$

Choose `epsilon_d` from the validation distribution of successful teacher 10-step segments, not by hand.

## 5.4 Action and residual penalties

Maybe add some regularization on the residual action


## 5.5 Task reward as secondary signal

Should be only a side experiment, it is not the main goal of the architecture, but would be interesting to see if it is better.

The main reward should be local latent-goal reaching.

Do not use task reward from the environment:

$$
r_t =
w_p r_t^{progress}
+
w_f r_t^{final}
+
w_b r_t^{bonus}
+
w_{success}\mathbf{1}[\mathrm{task\ success}]
$$

Recommended initial weights:

```text
w_p       = 1.0
w_f       = 1.0
w_b       = 0.0 initially, then tune
w_success = 0.0 only in full-episode mode as an ablation test
```

Do not tune these weights on the final evaluation seed bank.

## 5.6 Reward-hacking checks

Latent distance may not perfectly reflect task progress. Therefore every RL checkpoint must also report:

- task success;
- final and maximum environment reward;
- object pose progress if available;
- TCP/object physical probes if available;
- action saturation;
- whether the policy reaches the latent goal while harming task success.

A policy that improves latent distance but reduces task success is not a success.

---

# 6. Method R1: Residual RL on Deterministic Low Level

## 6.1 Policy

Let the frozen BC low level be:

$$
a_t^{BC}
=
\pi_{\mathrm{BC}}(x_t)
$$

Train a small residual policy:

$$
\Delta a_t
=
\pi_{\mathrm{res}}(x_t)
$$

Execute:

$$
a_t
=
\operatorname{clip}
\left(
a_t^{BC}
+
\alpha \tanh(\Delta a_t),
-1,
1
\right)
$$

where:

```text
alpha in {0.05, 0.10, 0.25, 0.50}
```

Start with:

```text
alpha = 0.10
```

## 6.2 Architecture

Use a small MLP residual adapter:

```text
input: same low-level condition x_t
hidden: width 256 or 512, depth 2 or 3
output: 3D residual
```

Initialize the final layer near zero so the initial policy is almost exactly the BC policy.

## 6.3 Algorithm

Recommended first algorithm:

```text
PPO with Gaussian residual policy
```

## 6.4 Training curriculum

Use four stages.

### Stage R1.0: local oracle-goal sanity

Rollout mode:

```text
Mode A: local demonstration-goal episodes
```

Episode length:

```text
10 primitive steps
```

Goal:

```text
g = z_demo(t+10)
```

Train only the residual.

Gate:

```text
mean final latent distance improves by >= 20%
task success in full hierarchy does not need to improve yet
```

### Stage R1.1: local predicted-goal training

Rollout mode:

```text
Mode B: local predicted-goal episodes
```

Goal:

```text
g = H_high(o_t, a_{t-1})
```

Gate:

```text
final latent distance improves over frozen BC
actions remain close to BC
```

### Stage R1.2: full hierarchy training

Rollout mode:

```text
Mode C: full hierarchy episodes
```

Goal source:

```text
frozen learned high level
```

Gate:

```text
full-episode success improves over frozen hierarchy
```

### Stage R1.3: disturbance and recovery resets

Add resets from:

- states in failed frozen-hierarchy rollouts;
- states in the recovery corpus;
- states after artificial action perturbations.

Gate:

```text
disturbed/recovery success improves
clean success does not drop by more than 5 percentage points
```

## 6.5 Main comparison

For `N_demo=500`, compare:

1. Frozen deterministic hierarchy.
2. R1 residual after Stage R1.0.
3. R1 residual after Stage R1.1.
4. R1 residual after Stage R1.2.
5. R1 residual after Stage R1.3.

Repeat the final selected R1 recipe for `N_demo=1000`.

---

# 7. Method R2: Residual RL on Flow Low Level

## 7.1 Motivation

The sample-efficiency experiment compared deterministic and flow high levels, but the low level was deterministic. This method tests whether an action-flow low level can be used as the base policy and improved with residual RL.

## 7.2 Pretraining an action-flow low level

Train a low-level action-flow policy with the same condition:

```text
[current observation, held VAE goal, previous action, remaining fraction]
```

Target:

```text
teacher action at the current offset
```

Use rectified conditional flow matching over 3D actions.

Evaluate both:

1. zero-noise deterministic endpoint;
2. stochastic sampled action.

For contact control, the main base action should be the zero-noise endpoint unless stochastic sampling is clearly better.

## 7.3 Residual on top of flow endpoint

Let:

$$
a_t^{flow}
=
\pi_{\mathrm{flow}}^{zero}(x_t)
$$

Train:

$$
a_t
=
\operatorname{clip}
\left(
a_t^{flow}
+
\alpha\tanh(\Delta a_t),
-1,
1
\right)
$$

Use the same R1 curriculum.

## 7.4 Gate

R2 is useful only if it beats R1 or shows better recovery robustness.

If flow base actions are noisier or less stable than deterministic BC, do not continue with R2.

---

# 8. Method R3: Direct Deterministic Low-Level Fine-Tuning

## 8.1 Motivation

Residual RL may be too conservative. Direct fine-tuning can modify the full low-level policy but risks destroying imitation behavior.

## 8.2 Policy

Initialize from the deterministic imitation low level:

$$
a_t = \pi_\theta(x_t)
$$

Fine-tune either:

1. last layer only;
2. last two layers;
3. all layers.

## 8.3 Algorithm

Use off-policy deterministic actor-critic:

```text
TD3 or SAC-style deterministic actor with exploration noise
```

or PPO with a Gaussian policy initialized around the deterministic mean.

If using PPO, policy mean starts at the BC action and standard deviation starts small.

## 8.4 Regularization

Keep the policy close to the imitation low level:

$$
\mathcal{L}_{BC-reg}
=
\lambda_{BC}
\left\|
\pi_\theta(x_t)
-
\pi_{BC}(x_t)
\right\|_2^2
$$

on replayed demonstration states and recent on-policy states.

Also use:

```text
KL or action L2 penalty to BC
gradient clipping
small learning rate
early stopping on clean validation success
```

## 8.5 Gate

R3 must beat R1 by at least 3 percentage points on full hierarchy success to justify the extra instability.

If it improves disturbed success but loses more than 5 clean-success points, reject.

---

# 9. Method R4: Direct Flow Low-Level Fine-Tuning

## 9.1 Motivation

This tests the requested direct RL tuning of the flow policy.

The action-flow low level is initialized from imitation and then tuned directly by RL.

## 9.2 Important algorithmic issue

The current rectified flow implementation does not necessarily provide a simple PPO-style log probability for the final sampled action.

Therefore, do **not** start with PPO unless a correct log-probability implementation exists.

Use one of these safer options.

## 9.3 Option R4-A: deterministic actor-critic on zero-noise endpoint

Treat the zero-noise flow endpoint as a deterministic actor:

$$
a_t
=
F_\theta(x_t,\epsilon=0)
$$

Train a critic:

$$
Q_\psi(x_t,a_t)
$$

Update the flow parameters through:

$$
\nabla_\theta Q_\psi(x_t,F_\theta(x_t,0))
$$

Keep a flow-matching regularizer on demonstration batches:

$$
\mathcal{L}
=
-\mathbb{E}[Q_\psi(x,F_\theta(x,0))]
+
\lambda_{FM}\mathcal{L}_{flow}
+
\lambda_{BC}\left\|
F_\theta(x,0)-a_{BC}
\right\|^2
$$

## 9.4 Option R4-B: Q-weighted flow matching

Collect an RL replay buffer.

Train a critic.

Fine-tune the flow by weighting action targets with estimated advantage:

$$
w_i=\exp
\left(
\frac{A(x_i,a_i)}{\eta}
\right)
$$

and minimizing:

$$
\mathcal{L}_{QFM}
=
w_i
\mathcal{L}_{flow}(x_i,a_i)
$$

This is more conservative than directly optimizing the flow endpoint.

## 9.5 Option R4-C: residual flow

Keep the flow model frozen and train a residual adapter, as in R2. This is technically not direct flow tuning but is a strong safety baseline.

## 9.6 Direct-flow experiment order

Run:

1. train action-flow low-level imitation baseline;
2. evaluate zero-noise and stochastic flow;
3. run R2 residual flow;
4. run R4-A direct endpoint actor-critic;
5. only if needed, run R4-B Q-weighted flow matching.

## 9.7 Gate

Direct flow tuning is useful if:

```text
success >= R1 success
or
recovery success > R1 recovery success without clean degradation
```

If it is unstable or loses clean performance, use residual deterministic RL as the main method.

---

# 10. Training Budgets

## 10.1 Demonstration budgets

Primary:

```text
N_demo = 500
```

Confirmation:

```text
N_demo = 1000
```

Do not use 1,800 or 8,000 for the main RL tuning study unless needed for an upper-bound sanity check.

## 10.2 RL interaction budgets

Use online simulator transitions, reported separately from demonstration data.

For each `N_demo` and training seed:

### Smoke budget

```text
50k environment steps
```

Purpose:

- check reward signs;
- verify no collapse;
- check residual magnitude;
- verify logging.

### Development budget

```text
500k environment steps
```

Purpose:

- compare R1/R2/R3/R4;
- tune reward weights and residual scale.

### Final budget

```text
2M environment steps
```

Purpose:

- final selected R1 and best flow-tuning method;
- three training seeds;
- full evaluation.

If runtime allows:

```text
5M environment steps
```

as an upper-bound curve for the best method only.

## 10.3 Checkpointing

Save checkpoints every:

```text
25k environment steps
```

Evaluate every:

```text
100k environment steps
```

Use a validation seed bank separate from the final seed bank.

---

# 11. Evaluation Protocol

## 11.1 Final evaluation seeds

Use an unseen evaluation bank:

```text
500 episodes per deployable method
```

Use the same 500 seeds for all methods and all training seeds.

Do not reuse the seed bank used to tune reward weights or select checkpoints.

## 11.2 Evaluation conditions

Evaluate every final checkpoint on:

### E1. Clean full episodes

Normal Push-T resets.

### E2. Local goal-reaching episodes

Exact local resets from held-out teacher trajectories. Measure whether the low level reaches the held latent goal in 10 steps.

### E3. Disturbed episodes

Inject perturbation bursts during evaluation, without training-time labels.

### E4. Recovery-state resets

Start from states in the stored recovery corpus or from failed frozen-hierarchy rollouts.

### E5. Branch-oracle goals

Replace the learned high-level goal with an exact branch future latent.

This tests whether RL improved the low-level interface independently of high-level prediction.

## 11.3 Required baselines

For every `N_demo`, compare:

1. Frozen deterministic hierarchy.
2. Frozen flow hierarchy.
3. Flat observation deterministic policy.
4. R1 residual deterministic low-level.
5. R2 residual flow low-level.
6. R3 direct deterministic fine-tune.
7. R4 direct flow fine-tune.
8. Branch-oracle versions of the frozen and RL-tuned low levels.

---

# 12. Metrics

## 12.1 Main metrics

- task success;
- final normalized reward;
- maximum normalized reward;
- local latent goal final distance;
- latent distance reduction over each 10-step segment;
- fraction of segments reaching a latent threshold;
- recovery success;
- disturbed success.

## 12.2 Safety and stability metrics

- clean success drop relative to frozen hierarchy;
- action saturation rate;
- residual magnitude;
- action smoothness;
- distance from BC action;
- number of high-level decisions per episode;
- rollout length;
- divergence from demonstration distribution.

## 12.3 Algorithm diagnostics

For actor-critic methods:

- critic loss;
- Q-value scale;
- actor loss;
- entropy if applicable;
- replay buffer composition;
- on-policy versus replay distribution;
- correlation between predicted Q and realized latent progress.

For flow methods:

- flow-matching loss before and after RL;
- zero-noise action MAE;
- stochastic sample variance;
- Q-weighted action distribution;
- whether direct flow tuning damages imitation action quality.

---

# 13. Success Gates

## 13.1 Local reachability gate

A method passes local reachability if:

```text
mean final latent distance decreases by >= 25%
and
segment goal-reaching rate increases by >= 15 percentage points
```

relative to the frozen low level.

## 13.2 Full hierarchy gate at 500 demonstrations

A method is successful if:

```text
success_RL_500 >= success_frozen_500 + 0.10
```

Preferred:

```text
success_RL_500 >= 0.45
```

Strong:

```text
success_RL_500 >= 0.50
```

## 13.3 Full hierarchy gate at 1000 demonstrations

A method is successful if:

```text
success_RL_1000 >= success_frozen_1000 + 0.05
```

Preferred:

```text
success_RL_1000 >= 0.60
```

Strong:

```text
success_RL_1000 >= 0.65
```

## 13.4 Clean-preservation gate

A method fails if:

```text
clean success drops by more than 5 percentage points
```

relative to the frozen imitation hierarchy at the same `N_demo`.

## 13.5 Flow-tuning gate

Direct flow tuning is worth keeping only if it matches or beats residual deterministic RL on at least one of:

- clean full-episode success;
- disturbed success;
- recovery success;
- local latent-goal reaching.

Otherwise, keep residual deterministic RL as the main method.

---

# 14. Experiment Schedule

## Phase 0: Implementation audits

1. Add low-level RL environment wrapper.
2. Verify latent distance reward on stored teacher windows.
3. Verify exact reset/replay for local goal episodes.
4. Verify the frozen hierarchy reproduces the reported baseline at `N=500`.
5. Verify residual action starts near zero.
6. Verify RL actions are clipped and previous action uses executed action.
7. Verify no high-level or VAE parameters receive gradients.

## Phase 1: R1 residual RL at `N=500`

1. Smoke run, 50k steps.
2. Development run, 500k steps.
3. Sweep residual scale:
   ```text
   alpha in {0.05, 0.10, 0.25}
   ```
4. Sweep reward:
   ```text
   progress only
   progress + terminal distance
   progress + terminal + weak task reward
   ```
5. Choose one recipe.

## Phase 2: R2/R3/R4 at `N=500`

1. Train action-flow low-level imitation baseline.
2. Run R2 residual flow.
3. Run R3 direct deterministic fine-tune.
4. Run R4 direct flow endpoint actor-critic.
5. Compare against R1.

## Phase 3: Final `N=500`

For the best two methods:

1. Train 3 seeds.
2. Use 2M RL steps per seed.
3. Evaluate 500 episodes per seed.
4. Report clean, disturbed, recovery, and branch-oracle results.

## Phase 4: Confirmation at `N=1000`

Repeat only:

1. frozen hierarchy;
2. best residual method;
3. best direct-flow/direct-fine-tune method.

Use the same RL step budget unless learning curves show early saturation.

## Phase 5: Optional upper-bound

If `N=500` and `N=1000` are positive, test:

```text
N_demo = 1800
```

only for the best method to see whether the improvement persists at stronger imitation performance.

---

# 15. Required Tables and Plots

## Table 1: Baselines

| N demo | method | success | final reward | local latent distance | recovery success |
| ---: | --- | ---: | ---: | ---: | ---: |

## Table 2: RL methods at 500 demos

| method | RL steps | success | gain | local distance gain | clean drop | recovery success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Table 3: RL methods at 1000 demos

| method | RL steps | success | gain | local distance gain | clean drop | recovery success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Table 4: Flow-specific diagnostics

| method | zero-noise MAE | sample std | task success | recovery success | stability |
| --- | ---: | ---: | ---: | ---: | --- |

## Plot 1: Success versus RL environment steps

Curves:

- frozen baseline;
- residual deterministic;
- residual flow;
- direct deterministic;
- direct flow.

## Plot 2: Latent goal distance versus RL environment steps

Report final 10-step segment distance and progress.

## Plot 3: Clean versus recovery performance

Show whether RL improves recovery without destroying clean behavior.

## Plot 4: Residual magnitude and action saturation

Detect reward hacking and unstable policies.

---

# 16. Interpretation Rules

## If residual RL improves local goal reaching and task success

Conclusion:

```text
The high-level latent interface is useful, and the remaining bottleneck was low-level reachability under distribution shift.
```

This supports using low-level RL fine-tuning in the final project.

## If residual RL improves latent reaching but not task success

Conclusion:

```text
The latent distance reward is not aligned enough with task success.
```

Next:

- use action-sensitive latent distance;
- add task/progress reward;
- use decoded/probed physical progress;
- revise the goal representation.

## If direct fine-tuning beats residual RL

Conclusion:

```text
The base imitation low level is too restrictive, and larger policy adaptation is needed.
```

But only accept this if clean performance remains stable.

## If direct flow tuning beats deterministic residual RL

Conclusion:

```text
Flow policy fine-tuning provides a useful low-level improvement, possibly due to better exploration or multimodal action representation.
```

## If RL fails at 500 but works at 1000

Conclusion:

```text
The learned high-level goals or base low-level policy are too weak at 500 demos for stable RL.
```

In this case, the thesis should present 1000 demos as the practical minimum for RL fine-tuning.

## If all RL methods fail

Conclusion:

```text
The current latent-goal reward or low-level interface is insufficient for RL fine-tuning.
```

Next steps would be:

- switch to TCP/effect32 goal;
- add physical progress reward;
- collect intervention/recovery data;
- use model-based or planner-guided low-level improvement.

---

# 17. Main Deliverables

The researcher should produce:

1. `low_level_rl_experiment_log.md`
   - one entry per run;
   - commands, seeds, RL steps, reward config, metrics, and diagnosis.

2. `low_level_rl_config_table.csv`
   - all reward weights, algorithms, residual scales, and trainable modules.

3. `low_level_rl_final_results.md`
   - final summary with tables and plots.

4. `low_level_rl_learning_curves.png`
   - success and latent reachability over RL steps.

5. `low_level_rl_failure_videos/`
   - videos before and after RL on the same reset seeds.

6. `low_level_rl_policy_checkpoints/`
   - frozen baselines and selected RL-tuned policies.

---

# 19. Do Not Do Initially

Do not initially:

- fine-tune the VAE encoder;
- fine-tune DINO;
- fine-tune the high-level predictor;
- jointly train high and low levels;
- use extra demonstrations beyond the fixed `N_demo`;
- select checkpoints using the final evaluation seed bank;
- optimize only task success without local latent-goal diagnostics;
- trust latent-distance improvement unless task success/recovery also improves.

The first RL result should answer one clean question:

```text
Can low-level RL make the fixed high-level latent goals more reachable?
```
