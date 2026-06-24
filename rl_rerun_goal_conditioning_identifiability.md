# RL Rerun Goal-Conditioning Identifiability Note

Date: 2026-06-24

## Question

Why do the low-level policies barely react to future latent goals, even when the
goals are valid and clearly different in latent space?

## Evidence

The post-RL diagnostics show a consistent pattern:

| Diagnostic | Main result |
| --- | --- |
| Learned-vs-oracle swap | mean goal L2 `25.02`, tuned action change `0.033` L2 |
| Same-state `k=9/10/11` valid goals | mean goal L2 `~16`, action change `~0.0085` L2 |
| Same-state `k=2/5/10/20` valid goals | mean goal L2 `~24-27`, action change `~0.016-0.022` L2 |
| Condition-block shuffle | observation shuffle action change `~0.805` L2, goal shuffle `~0.048-0.050` L2 |
| Action-block prediction | observation-only raw action L2 `0.227`, goal-only `0.404`; adding goal to observation improves only `0.020` |

R3 fine-tuning and the RR-48 goal-sensitivity hinge do not materially change
these values relative to the frozen low level.

## Identifiability Issue

The current low-level target is one-step deterministic teacher imitation:

```text
a_t = pi_teacher(s_t)
```

The future goal is supplied as:

```text
g_t(k) = E(o_{t+k})
```

For a fixed current state `s_t`, the deterministic teacher action is the same no
matter which valid future horizon is selected:

```text
pi_teacher(s_t, g_t(k=2))  -> a_t
pi_teacher(s_t, g_t(k=5))  -> a_t
pi_teacher(s_t, g_t(k=10)) -> a_t
pi_teacher(s_t, g_t(k=20)) -> a_t
```

because the teacher policy does not condition on `g_t`. The supervised label is
therefore invariant to the goal for all same-current-state horizon variants.

This means one-step distillation has no direct statistical reason to learn a
strong causal map from future goal to action. The policy can minimize action
error by using current observation and previous action, and the goal becomes a
weak correlational feature.

The RL reward used in the rerun was goal-distance progress, but the trainable
scope was small and the initialized policy was already current-state dominated.
The observed RL updates improved some local latent metrics but did not create a
new strong goal-conditioned control interface.

## Consequence

Small final-layer tweaks, generic goal-sensitivity penalties, or more seeds of
the same R3 setup are unlikely to solve the learned-interface problem. The
objective itself needs to make future goals behaviorally identifiable.

## Recommended Next Experiments

### 1. Multi-step goal-reaching training with policy rollouts

Train the low level as a local goal-reaching controller over a rollout horizon,
not just as a one-step action imitator.

Use a loss or reward that depends on whether the executed rollout reaches the
supplied future goal:

```text
minimize || z_{t+H}^{student} - g_t ||^2
```

This makes different goals require different behavior through closed-loop
outcomes. It also aligns with the actual deployment interface.

### 2. Counterfactual branch data

Create multiple reachable futures from the same current state using different
valid branch rollouts, not only different horizons of the same deterministic
teacher trajectory.

Useful branch sources:

- teacher with small action perturbations;
- recovery teacher from nearby perturbed states;
- short MPC-style or sampled-action branches accepted only if dynamically valid;
- teacher branches toward different local contact strategies if available.

Store coherent tuples:

```text
(s_t, g_t^branch, action or action sequence that actually reaches g_t^branch)
```

This gives the low level examples where the same current state can pair with
different goals and different desired behavior.

### 3. Architecture bottleneck or goal-gated low level

Reduce the easy current-observation shortcut and make the goal affect the action
path more directly:

- factor the policy into current-state encoder, goal encoder, and explicit
  relative-goal module;
- FiLM/gating from goal features into the action trunk;
- residual policy whose residual path is goal-only or goal-dominant;
- train with observation dropout/noise only after verifying it does not destroy
  basic control.

This should be tested with the same valid-goal sensitivity diagnostics before
running expensive closed-loop deployment.

### 4. Promote only if diagnostics move

Before a full 1M-transition or 500-episode evaluation, require:

```text
same-state valid-goal action sensitivity increases substantially
condition-block goal sensitivity becomes comparable to previous-action sensitivity
local goal-reaching improves without large action saturation
```

The existing R3 family does not meet these criteria.
