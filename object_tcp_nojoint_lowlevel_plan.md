# Object+TCP No-Joint Low-Level Experiments

## 1. Purpose

The previous RL reachability experiments show that PPO can reduce local goal distance, but unconstrained local reachability often drifts away from the contact/action manifold that makes Push-T succeed.

The current best low-level RL result is a bounded residual around the Phase-C full-state BC low level:

```text
Run 30: residual-on-Phase-C-BC PPO
oracle full-state held-subgoal success: about 0.73
learned-high full-state held-subgoal success: about 0.55
shuffled-goal success: 0.00
```

However, this result used a full-state goal that includes robot joint-state components. The new hypothesis is:

> Future joint positions/velocities should not be part of the low-level goal or reward because the action is an end-effector/TCP displacement command. If the object state and TCP state reach the target, the subgoal is achieved even if the robot joints differ from the demonstration.

This plan tests a new low-level interface:

```text
current object + current TCP + previous action + remaining time
held future object + future TCP target
no current joint positions
no future joint targets
no joint-position loss
```

The high-level model is **not** changed in this plan. Learned-goal evaluation uses the existing 1800-demo full-state high-level predictor, but projects its output down to the object+TCP subgoal representation before sending it to the low level.

---

## 2. Core Experimental Claim

The low-level controller should solve:

```text
given current object/TCP state and a held future object/TCP target,
choose pd_ee_delta_pos actions that move the object and TCP toward that target
while preserving task-compatible contact behavior.
```

The policy should not be asked to match future robot joint positions.

The main comparison is:

```text
Phase-C full-state BC
vs
object+TCP no-joint BC
vs
object+TCP no-joint residual PPO
vs
object+TCP no-joint hard-gated residual PPO
```

High level remains frozen.

---

# 3. Representation and Goal Formulation

## 3.1 State variables

Let the privileged simulator state contain:

- T-object position:
  $$
  p^T_t = (x^T_t, y^T_t)
  $$
- T-object yaw:
  $$
  \theta^T_t
  $$
- T-object velocity, if available:
  $$
  v^T_t, \omega^T_t
  $$
- TCP/end-effector position:
  $$
  p^{tcp}_t = (x^{tcp}_t, y^{tcp}_t, z^{tcp}_t)
  $$
- TCP velocity, if available:
  $$
  v^{tcp}_t
  $$

The no-joint low-level must **not** receive:

```text
joint positions q
joint velocities qdot
future joint target
joint target error
joint velocity target
```

The main no-joint current state is:

$$
s^{ot}_t =
[
p^T_t,
\sin \theta^T_t,
\cos \theta^T_t,
v^T_t,
\omega^T_t,
p^{tcp}_t,
v^{tcp}_t
]
$$

If velocities are noisy or unavailable, run a no-velocity ablation:

$$
s^{ot,no\_vel}_t =
[
p^T_t,
\sin \theta^T_t,
\cos \theta^T_t,
p^{tcp}_t
]
$$

Do not include joints in the main branch.

## 3.2 Held target

At high-level decision time \(t_i\), the held target is the projected future object+TCP state:

$$
y^*_{i} =
[
p^T_{t_i+k},
\sin \theta^T_{t_i+k},
\cos \theta^T_{t_i+k},
v^T_{t_i+k},
\omega^T_{t_i+k},
p^{tcp}_{t_i+k},
v^{tcp}_{t_i+k}
]
$$

For \(k=10\), this is a 0.5 second future target at 20 Hz.

The target \(y^*_i\) remains fixed during the low-level option.

## 3.3 Recomputed goal features

Do **not** simply concatenate the raw target at every primitive step.

At primitive step \(t_i+j\), where \(j \in \{0,\ldots,k-1\}\), define remaining steps:

$$
r_j = k-j
$$

and remaining time fraction:

$$
\tau_j = \frac{r_j}{k}
$$

Recompute relative features from the current state to the held target.

Object position error:

$$
\Delta p^T_j = p^{T,*}_i - p^T_{t_i+j}
$$

Object yaw error:

$$
\Delta \theta^T_j =
\mathrm{wrap}(\theta^{T,*}_i - \theta^T_{t_i+j})
$$

Represent yaw error as:

$$
[\sin(\Delta \theta^T_j), \cos(\Delta \theta^T_j)]
$$

Desired object velocity-to-goal:

$$
\bar v^T_j =
\frac{\Delta p^T_j}{r_j \Delta t}
$$

Desired yaw rate-to-goal:

$$
\bar \omega^T_j =
\frac{\Delta \theta^T_j}{r_j \Delta t}
$$

TCP position error:

$$
\Delta p^{tcp}_j =
p^{tcp,*}_i - p^{tcp}_{t_i+j}
$$

Desired TCP velocity-to-goal:

$$
\bar v^{tcp}_j =
\frac{\Delta p^{tcp}_j}{r_j \Delta t}
$$

The low-level goal feature is:

$$
g^{ot}_{i,j} =
[
y^*_i,
\Delta p^T_j,
\sin(\Delta \theta^T_j),
\cos(\Delta \theta^T_j),
\bar v^T_j,
\bar \omega^T_j,
\Delta p^{tcp}_j,
\bar v^{tcp}_j,
\tau_j
]
$$

If the absolute target \(y^*_i\) causes overfitting or redundancy, run an ablation using only the relative features:

$$
g^{ot,rel}_{i,j} =
[
\Delta p^T_j,
\sin(\Delta \theta^T_j),
\cos(\Delta \theta^T_j),
\bar v^T_j,
\bar \omega^T_j,
\Delta p^{tcp}_j,
\bar v^{tcp}_j,
\tau_j
]
$$

## 3.4 Low-level policy input

The main low-level input is:

$$
x_{i,j} =
[
s^{ot}_{t_i+j},
g^{ot}_{i,j},
a^{exec}_{t_i+j-1}
]
$$

where \(a^{exec}_{t-1}\) is the clipped action actually executed by the environment/controller.

There are no joint positions in \(x_{i,j}\).

---

# 4. Distance / Reward Formulation

## 4.1 Object+TCP distance

The no-joint local distance is:

$$
d_t =
w_{obj}
\left\|
\frac{p^T_t - p^{T,*}}{\sigma_{p^T}}
\right\|_2^2
+
w_{yaw}
\left(
1-\cos(\theta^T_t-\theta^{T,*})
\right)
+
w_{tcp}
\left\|
\frac{p^{tcp}_t - p^{tcp,*}}{\sigma_{p^{tcp}}}
\right\|_2^2
$$

Optional velocity terms:

$$
+
w_{vobj}
\left\|
\frac{v^T_t-v^{T,*}}{\sigma_{v^T}}
\right\|_2^2
+
w_{vtcp}
\left\|
\frac{v^{tcp}_t-v^{tcp,*}}{\sigma_{v^{tcp}}}
\right\|_2^2
$$

Initial weights:

```text
w_obj   = 1.0
w_yaw   = 0.5
w_tcp   = 0.5
w_vobj  = 0.1
w_vtcp  = 0.1
```

Ablation weights:

```text
no velocity:
  w_vobj = 0.0
  w_vtcp = 0.0

object-focused:
  w_obj = 1.0
  w_yaw = 0.5
  w_tcp = 0.2

balanced:
  w_obj = 1.0
  w_yaw = 0.5
  w_tcp = 1.0
```

There is no joint distance term:

$$
w_q = 0
$$

## 4.2 PPO reward

For RL, use project-clean local reachability reward:

$$
r_t =
(d_t-d_{t+1})
+
\mathbf{1}[t=T](-\lambda_T d_T)
-
\lambda_\Delta \|\Delta a_t\|_2^2
-
\lambda_a \|a_t\|_2^2
-
\lambda_s \|a_t-a_{t-1}\|_2^2
$$

Initial weights:

```text
lambda_T     = 1.0
lambda_delta = 0.01 for residual policies
lambda_a     = 0.001
lambda_s     = 0.001
```

No environment task reward is used for training.

No task success reward is used for training.

No privileged target pose beyond the object/TCP subgoal is used for training.

Task success and environment reward are evaluation-only metrics.

---

# 5. High-Level Handling

## 5.1 High level is frozen

Do not update:

```text
high-level predictor
representation encoder
DINO
VAE
```

The high-level model is used only to produce learned-goal evaluation targets.

## 5.2 Oracle goal evaluation

For oracle evaluation, roll or replay the teacher from the current state for \(k\) steps and project the resulting future state to object+TCP:

$$
y^{*,oracle}_i =
P_{ot}(s^{teacher}_{t_i+k})
$$

where \(P_{ot}\) removes all joint-state components.

## 5.3 Learned goal evaluation

For learned-goal evaluation, use the existing 1800-demo full-state high-level predictor:

$$
\hat{s}_{t_i+k} = H_{full}(o_{t_i}, a_{t_i-1})
$$

Then project it:

$$
\hat{y}^{*,learned}_i =
P_{ot}(\hat{s}_{t_i+k})
$$

The high-level model is not retrained.

This answers:

```text
If we keep the high level fixed and only change the low level's interface/reward,
does the low level improve?
```

## 5.4 Shuffled goal evaluation

For shuffled-goal sanity checks, shuffle the projected object+TCP goals across environments:

$$
\tilde{y}^{*}_i =
\mathrm{shuffle}(y^{*}_i)
$$

A useful low level should fail or strongly degrade under shuffled goals.

---

# 6. Phase A — Feature and Dataset Preparation

## 6.1 Implement object+TCP projection

Add a utility:

```python
project_state_to_object_tcp(state) -> object_tcp_state
```

It should extract:

```text
object position
object yaw sin/cos
object velocity if available
object angular velocity if available
TCP position
TCP velocity if available
```

It must not return joint positions or joint velocities.

## 6.2 Implement goal-feature recomputation

Add:

```python
make_object_tcp_goal_features(current_state, target_object_tcp_state, remaining_steps, dt)
```

This returns the recomputed goal feature \(g^{ot}_{i,j}\).

Unit tests:

1. If current state equals target state, position/yaw/TCP errors are zero.
2. If remaining steps decreases, velocity-to-goal features scale correctly.
3. Shuffling target states changes goal features.
4. Joint values in the source state do not affect the projected output.

## 6.3 Normalizers

Fit normalizers on the training split only:

```text
current object+TCP state normalizer
goal-feature normalizer
previous-action normalizer
action normalizer
```

Do not reuse full-state normalizers.

## 6.4 Reset banks

Use or create the following banks.

### Demo bank

```text
teacher/demo local windows
target = projected future object+TCP at t+k
```

### Mixed deployed bank

Start with the bank analogous to Run 30:

```text
50% demo/teacher windows
25% Phase-C full BC deployed states
25% current PPO/residual deployed states, or Run 30 states if available
```

For the no-joint experiments, after training no-joint BC, create a no-joint deployed bank:

```text
50% demo/teacher windows
25% Phase-C full BC deployed states
25% no-joint BC deployed states
```

Do not use online expert action labels.

Oracle branches are diagnostic only.

---

# 7. Phase B — Train Object+TCP No-Joint BC Low Level

## 7.1 Training data

Use successful teacher trajectories.

For each segment \(t:t+k\), train on all offsets:

```text
for j in 0..k-1:
    current = state[t+j]
    target  = project_object_tcp(state[t+k])
    goal_features = recompute_goal_features(current, target, remaining=k-j)
    input = [current_object_tcp, goal_features, previous_executed_action]
    label = clipped_executed_teacher_action[t+j]
```

This is multi-offset held-goal training.

Do not train only the first action of the segment.

## 7.2 Model

Use the same base MLP family as the previous Phase-C low level:

```text
input: object+TCP current state + recomputed object+TCP goal + previous action
hidden layers: 4
width: 512
activation: SiLU
output: 3D pd_ee_delta_pos action
```

Train deterministic BC first.

Optional later:

```text
Gaussian/action-flow low level
FiLM conditioning
```

But do not start with those.

## 7.3 Optimization

Initial values:

```text
optimizer: AdamW
learning rate: 3e-4
batch size: 512 or 1024
batches/epoch: 200
epochs: 60
weight decay: same as previous low-level BC if used
```

Checkpoint selection:

```text
validation action MAE
```

Do not select by closed-loop success.

## 7.4 Evaluation

Evaluate:

1. Oracle projected object+TCP goals.
2. Learned projected object+TCP goals from frozen 1800-demo full high level.
3. Shuffled projected object+TCP goals.

Compare against:

```text
Phase-C time-conditioned full BC
Phase-B object-pose BC if useful
Run 30 residual-on-BC if available
```

Required table:

| Policy | Oracle success | Learned success | Shuffled success | Hold object+TCP distance | Teacher action MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | | | | | |
| object+TCP no-joint BC | | | | | |

## 7.5 Gates

Promote no-joint BC if:

```text
oracle success >= 0.8 * Phase-C full BC oracle success
shuffled success <= 0.05
learned success is not catastrophically worse than Phase-C full BC
```

If no-joint BC fails badly, run these ablations before RL:

1. Add current joint state as context only, but still no future joint target.
2. Use object+TCP relative goal only.
3. Change object/TCP distance weights.
4. Increase horizon to k=20.

Do not add future joint targets back unless all no-joint variants fail.

---

# 8. Phase C — Residual PPO on Object+TCP No-Joint BC

## 8.1 Policy

Let the frozen no-joint BC low level be:

$$
a^{BC}_t =
\pi_{BC}(x_t)
$$

Train a residual policy:

$$
u_t =
\pi_{res}(x_t)
$$

Execute:

$$
a_t =
\mathrm{clip}
\left(
a^{BC}_t
+
\alpha \tanh(u_t),
-1,
1
\right)
$$

Initial setting:

```text
alpha = 0.15
residual_penalty = 0.01
```

This matches the successful Run 30 structure.

## 8.2 PPO setup

Use exact local 10-step episodes.

```text
num_envs: largest stable, target >=4096
rollout_len: k = 10
samples/update: num_envs * 10
num_minibatches: 8 or 16
update_epochs: 3
PPO updates: 250 dev, 500 serious if learning
gamma: 0.99
gae_lambda: 0.95
clip_param: 0.2
entropy_coef: 0.0 to 0.005
max_grad_norm: 1.0
```

For the local 10-step MDP:

```text
terminated = True
truncated = False
bootstrap = False
```

Do not let value estimates bootstrap into the next sampled goal.

## 8.3 Training reward

Use the object+TCP distance from Section 4.

The reward is:

$$
r_t =
(d_t-d_{t+1})
+
\mathbf{1}[t=T](-d_T)
-
0.01 \|\alpha \tanh(u_t)\|^2
-
0.001 \|a_t\|^2
-
0.001 \|a_t-a_{t-1}\|^2
$$

No task reward.

No online expert action relabeling.

No high-level update.

## 8.4 Evaluation

Evaluate every saved checkpoint on:

```text
oracle projected object+TCP goals
learned projected object+TCP goals
shuffled projected object+TCP goals
local reset bank
deployed-state branch bank
```

Required comparisons:

| Policy | Oracle success | Learned success | Shuffled success | Local distance | Teacher action MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-joint BC | | | | | |
| no-joint residual PPO | | | | | |
| Phase-C full BC | | | | | |
| Run 30 full residual PPO | | | | | |

## 8.5 Gates

Pass if:

```text
oracle success >= no-joint BC oracle success + 0.03
learned success is neutral or improved
shuffled success remains <= 0.05
teacher-action MAE does not increase significantly
residual L2 remains small
```

If oracle success improves but learned success drops, the low-level is becoming incompatible with learned high-level projected goals.

---

# 9. Phase D — Hard-State Gated Residual PPO

## 9.1 Motivation

The previous reset-bank aggregation experiments showed that changing all states can hurt task success. The residual should act mostly on states where BC is predicted to fail.

## 9.2 Precompute BC difficulty

For every local reset sample \((s,g)\), roll the frozen no-joint BC policy for \(k\) steps and compute terminal distance:

$$
d_{BC}(s,g)
$$

Compute distribution quantiles over the training reset bank:

```text
q50 = median d_BC
q75 = 75th percentile d_BC
q90 = 90th percentile d_BC
```

## 9.3 Continuous hard gate

Define:

$$
w_{hard}(s,g)
=
\mathrm{clip}
\left(
\frac{d_{BC}(s,g)-q_{50}}{q_{90}-q_{50}},
0,
1
\right)
$$

Execute:

$$
a_t =
\mathrm{clip}
\left(
a^{BC}_t
+
w_{hard}(s,g)
\alpha\tanh(u_t),
-1,
1
\right)
$$

This makes the residual inactive on easy states and active on hard states.

## 9.4 Alternative sharp gate

A sharper ablation:

$$
w_{hard}(s,g)=
\mathbf{1}[d_{BC}(s,g)>q_{75}]
$$

## 9.5 Reward

Use weighted reachability reward:

$$
r_t =
w_{hard}(s,g)(d_t-d_{t+1})
+
\mathbf{1}[t=T]w_{hard}(s,g)(-d_T)
-
\lambda_{easy}(1-w_{hard})\|\Delta a_t\|^2
-
\lambda_{hard}w_{hard}\|\Delta a_t\|^2
$$

Initial values:

```text
lambda_easy = 0.05
lambda_hard = 0.01
```

This explicitly says:

```text
change behavior on hard states
preserve BC on easy states
```

## 9.6 Training schedule

Start from the Phase C residual checkpoint if it passed, or from zero residual on no-joint BC.

Run:

```text
250 updates dev
500 updates serious if dev improves
```

## 9.7 Gates

Pass if:

```text
hard-subset terminal distance improves
easy-subset terminal distance does not degrade
oracle success improves or stays neutral
learned success improves or stays neutral
shuffled success remains low
```

Required table:

| Policy | Easy local dist. | Hard local dist. | Oracle success | Learned success | Shuffled success |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-joint BC | | | | | |
| residual everywhere | | | | | |
| hard-gated residual | | | | | |

---

# 10. Phase E — Horizon Sweep

## 10.1 Motivation

A 10-step option is only 0.5 seconds at 20 Hz. The low level may need more time to establish contact and reach the object+TCP subgoal without aggressive or task-bad behavior.

## 10.2 Horizons

Test:

```text
k in {10, 20, 30}
U = k
H = 1
```

For each \(k\), retrain:

1. object+TCP no-joint BC;
2. residual PPO;
3. hard-gated residual PPO if residual PPO is promising.

Do not compare different horizons without retraining the high/low data formulation.

## 10.3 High-level target handling

The high level is still frozen. For learned-goal evaluation:

- use the existing full-state high-level if only \(k=10\);
- if evaluating \(k=20\) or \(k=30\), either:
  - train a horizon-specific high-level predictor for evaluation only, or
  - restrict non-\(k=10\) experiments to oracle-goal low-level evaluation.

Since the high level should not be changed in the main branch, the first horizon sweep should focus on oracle-goal low-level evaluation.

## 10.4 Gates

Longer horizon is useful if:

```text
oracle success improves
local reachability improves
shuffled success remains low
action saturation and teacher-action MAE do not increase
```

If oracle success improves but learned-goal success cannot be evaluated fairly, report it as a low-level horizon result, not a deployable hierarchy result.

---

# 11. Phase F — Optional Context Ablations

Run only if no-joint BC fails.

## 11.1 Current joints as context only

Input:

```text
current object+TCP
current joint state q/qdot
object+TCP future goal
previous action
remaining time
```

Reward/goal:

```text
object+TCP only
no future joint target
no joint distance
```

This tests whether current robot configuration helps action selection without forcing future joint matching.

## 11.2 No-velocity object+TCP

Input and goal:

```text
object pose
TCP position
previous action
remaining time
```

No object velocity or TCP velocity.

## 11.3 Relative-goal-only

Use only relative errors and velocity-to-goal features, not absolute target state.

---

# 12. Logging Requirements

Keep a running Markdown log:

```text
object_tcp_nojoint_experiment_log.md
```

Update after every run.

Each entry must include:

```text
hypothesis
command
git commit
dataset/reset bank
goal formulation
current input formulation
whether joints are present anywhere
normalizers
num_envs
rollout_len
PPO updates
gradient steps
reward weights
termination/truncation handling
oracle success
learned projected-goal success
shuffled success
local object+TCP distance
teacher-action MAE
residual L2
interpretation
next action
```

Also produce a final report:

```text
object_tcp_nojoint_final_results.md
```

with accepted comparisons only.

---

# 13. Final Decision Rules

## Positive

The no-joint low-level direction is positive if:

```text
object+TCP no-joint BC is close to Phase-C full BC under oracle goals
and/or
hard-gated residual PPO improves oracle or learned success over no-joint BC
while shuffled-goal success remains near zero
```

## Negative

Call it negative only if:

```text
no-joint BC is much worse than Phase-C full BC
current-joint-context ablation also fails
residual and hard-gated residual do not improve
oracle goals do not help
```

Do not reject no-joint based only on learned-goal performance, because the learned high-level is frozen and may not be calibrated for the projected object+TCP interface.

## Main scientific interpretation

If no-joint works:

```text
Future robot joint targets were an unnecessary or harmful part of the low-level subgoal. The low-level should optimize object and TCP outcomes because the action is a TCP displacement command.
```

If no-joint fails:

```text
Even though the action is a TCP displacement command, robot configuration/state provides important information for contact-compatible local control. Future joint matching should still remain excluded, but current robot state may need to be included as context.
```
