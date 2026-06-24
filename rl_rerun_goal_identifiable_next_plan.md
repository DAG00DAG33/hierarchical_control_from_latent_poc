# Goal-Identifiable Low-Level Training Plan

Date: 2026-06-24

## Purpose

The RL rerun showed that the current low level is weakly goal-conditioned. The
main issue is not just high-level prediction error: one-step deterministic
teacher imitation makes the future goal weakly identifiable as a causal input.

This plan defines the next low-level experiments. Do not run another expensive
R3-style final-layer PPO point until the cheap diagnostics in this plan move.

## Starting Evidence

| Artifact | Result |
| --- | --- |
| `rl_rerun_goal_conditioning_identifiability.md` | one-step deterministic labels are invariant to same-current-state horizon changes |
| `rl_rerun_valid_goal_sensitivity_seed0_2048.json` | `k=9/10/11` valid goals change actions only `~0.0085` L2 |
| `rl_rerun_valid_goal_sensitivity_seed0_2048_wide.json` | `k=2/5/10/20` valid goals change actions only `~0.016-0.022` L2 |
| `rl_rerun_condition_block_sensitivity_seed0_2048.json` | observation-block shuffle is `~16-17x` more influential than goal-block shuffle |
| `rl_rerun_action_block_prediction_seed0_8192_2048.json` | obs-only action prediction is much better than goal-only; obs+goal adds little |

## Fixed Requirements

- Use the same VAE-512 learned interface unless a phase explicitly changes it.
- Use the vector-consistent state dataset for any local reset experiment.
- Keep task reward, task success, object pose, and task progress out of training
  rewards unless an experiment is explicitly labeled privileged diagnostic.
- Use the existing valid-goal sensitivity scripts as promotion gates.
- Treat closed-loop success as final evaluation only, not as the first screen.

## Phase G0: Label-Identifiability Baseline

Goal: make the weak-identifiability issue executable as a regression check.

Required artifacts:

- a compact command or script that reports:
  - same-current-state `k=2/5/10/20` goal L2;
  - action-label equality for the deterministic teacher first action;
  - frozen/R3 action sensitivity to those goals.

Gate:

```text
Documented only. This phase is diagnostic and already effectively passed by RR-49/RR-50/RR-53.
```

Decision:

Do not continue same-label one-step action cloning as the main way to create a
future-goal interface.

## Phase G1: Multi-Step Goal-Reaching Local Training

Goal: make the supplied future goal matter through rollout outcome.

Training objective:

```text
roll out low-level policy for H steps from s_t toward g_t
minimize || z_{t+H}^{student} - g_t ||^2
plus action regularization and optional BC anchor
```

Initial settings:

| Parameter | Value |
| --- | --- |
| goal horizon `k` | `10` |
| rollout chunk `H` | `10` |
| dataset | `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5` |
| envs | `4096` if stable, otherwise largest exact-reset vector corpus available |
| trainable scope | start with final layer/residual; expand only if diagnostics do not move |
| reward/loss | latent terminal distance and latent progress only |

Variants:

1. residual deterministic low level with larger goal-dominant residual path;
2. direct low level with goal-gated action trunk;
3. current low level with observation dropout/noise during rollout training.

Required cheap diagnostics after each smoke:

- same-state valid-goal sensitivity at `k=9/10/11`;
- wide valid-goal sensitivity at `k=2/5/10/20`;
- condition-block sensitivity;
- one-batch local goal-reaching metric.

Promotion gate to a serious run:

```text
goal-block shuffle action L2 >= 0.15
same-state k=2 vs k=10 action L2 >= 0.08
one-batch local final distance no worse than frozen + 0.02
action saturation <= 5%
```

Only if these pass, run a 1.024M-transition point.

## Phase G2: Counterfactual Branch Dataset

Goal: create training examples where the same current state can pair with
different reachable future goals and different behavior.

Branch sources:

1. deterministic teacher branch;
2. teacher branch after small valid first-action perturbations;
3. recovery teacher from nearby perturbed simulator states;
4. short sampled-action branch accepted only if it remains valid and improves a
   measurable latent or TCP objective.

Store each branch as:

```text
current simulator state
current observation and latent
future goal observation and latent
previous action
executed branch action sequence
teacher/recovery metadata
validity flags
```

Coherence rule:

Every goal and action sequence must originate from the same copied current
state. Do not mix nominal teacher futures with learner/recovery actions.

Cheap branch dataset gate:

```text
for at least 2048 current states:
  >= 2 valid distinct goals per state
  mean pairwise goal L2 >= 10
  mean first-action pairwise L2 >= 0.05 or mean action-sequence L2 >= 0.15
  replay/copy state error == 0 for replay-backed branches
```

If this gate fails, the branch generator is not producing useful counterfactual
supervision.

## Phase G3: Goal-Gated Architecture Smoke

Goal: force a stronger path from future goal to action without destroying basic
control.

Candidate architecture:

```text
state_embed = f_o(observation, previous_action)
goal_embed  = f_g(goal, current_latent)
relative    = f_r(goal - current_latent)
action      = f_a(state_embed, goal_embed, relative)
```

Recommended mechanisms:

- FiLM or multiplicative gating from goal features into the action trunk;
- residual action head whose residual is goal-dominant;
- optional observation dropout during training;
- action anchor to the frozen low level for stability.

Smoke training data:

- start with the existing teacher corpus;
- add G2 counterfactual branches only after G2 passes.

Promotion diagnostics:

```text
condition-block goal shuffle action L2 approaches previous-action shuffle L2
same-state valid-goal sensitivity increases at least 3x over frozen
action-block prediction with goal-gated model shows obs+goal benefit >= 0.05 raw L2
```

## Phase G4: Closed-Loop Evaluation

Run only after G1/G2/G3 diagnostics improve.

Development:

```text
100 episodes
same eval seeds as previous R3 diagnostics
compare frozen, previous best R3, and new candidate
```

Final:

```text
500 episodes
fresh seed bank
at least two policy seeds if development result is positive
```

Required comparisons:

- frozen hierarchy;
- previous best R3 direct low-level checkpoint;
- new goal-identifiable low-level candidate;
- replay-oracle goals if compute allows.

Promotion gate:

```text
new candidate success >= frozen success + 0.05 on development bank
new candidate not worse than frozen on disturbed/recovery diagnostic
valid-goal sensitivity remains improved after training
```

If development is mixed, do not run final 500-episode evaluation.

## Immediate Next Action

Implement Phase G1 as the first code change:

1. add a local multi-step goal-reaching trainer that actually rolls the policy
   for `H=10` before scoring the supplied goal;
2. run a one-update `4096 x 10` smoke;
3. evaluate the existing diagnostic scripts;
4. promote only if the goal-sensitivity gates move.

This is the smallest next step that directly targets the identified failure
mode.
