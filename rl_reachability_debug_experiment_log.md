# RL Reachability Debug Experiment Log

This is the running log for `rl_reachability_debug_plan.md`.

## 2026-06-30 - Goal Activation

Hypothesis:

The previous visual/VAE scratch PPO failures are not enough to conclude that
reachability-based RL is impossible. The next experiments should start with
mechanics audits and the simplest local goal-reaching MDP before returning to
full visual/VAE PPO.

Plan source:

`rl_reachability_debug_plan.md`

Execution status:

Starting with Run 1, PPO mechanics audit.

## 2026-06-30 - Run 1: PPO Mechanics Audit

Hypothesis:

Before running more RL, verify that the low-level rollout semantics match the
intended local MDP: current observations update every primitive step, held
goals stay fixed for the 10-step segment, previous action and remaining time
enter the policy input, and local terminal GAE does not bootstrap into the next
goal segment.

Command:

```bash
uv run python scripts/rl_reachability_mechanics_audit.py \
  --config configs/pusht_incremental.yaml \
  --n-demo 1800 \
  --seed 0 \
  --num-envs 32 \
  --output results/incremental/rl_reachability_debug/run1_mechanics_audit.json
```

Dataset/reset bank:

- VAE512 deterministic hierarchy, `vae512_w2048_b1e6`
- `N_high = 1800`, seed `0`
- 32 vectorized visual envs from `seed_start=3900000`
- one held-goal local segment of length 10

Input update audit:

| Perturbation | Mean action L2 vs live | Max action L2 vs live |
| --- | ---: | ---: |
| cached start observation | 0.5641 | 0.8891 |
| cached previous action | 0.0980 | 0.2081 |
| constant remaining time | 0.0061 | 0.0107 |
| shuffled goal | 0.1285 | 0.1682 |
| shuffled observation | 0.5578 | 0.7428 |

Branch terminal raw distance after 10 steps:

| Branch | Terminal raw distance |
| --- | ---: |
| live | 0.9440 |
| cached start observation | 2.8622 |
| cached previous action | 1.0260 |
| constant remaining time | 0.9637 |
| shuffled goal | 1.4906 |
| shuffled observation | 1.6860 |

GAE unit check:

- local terminal returns: `[0.908406675, 0.95535, 1.0]`
- hand-computed local terminal returns: `[0.908406675, 0.95535, 1.0]`
- max absolute error: `0.0`
- truncation/bootstrap variant changes returns: `true`

Interpretation:

Run 1 passes the mechanics gate. The low-level condition uses live current
observations, previous actions, remaining time, and held goals. Observation
shuffling and cached observations have a large effect, goal shuffling has a
clear effect, and local terminal GAE does not bootstrap across the 10-step
goal boundary in the toy check.

Next action:

Proceed to Run 2: privileged/TCP scratch PPO local reaching, unless a separate
issue is found while preparing that environment.

## 2026-06-30 - Run 2: Privileged/TCP Scratch PPO Local Reaching

Hypothesis:

If PPO and the local reward are basically viable, a random policy should learn
the easiest 10-step low-dimensional reaching problem before we try VAE or visual
inputs again.

Setup:

- input: normalized privileged state, normalized TCP endpoint/velocity goal,
  normalized previous action, remaining local time
- policy: random scratch MLP actor-critic
- local horizon: `10`
- envs: `4096`
- PPO updates: `250`
- samples/update: `40960`
- total env steps: `10,240,000`
- optimizer steps: `250 * 3 epochs * 8 minibatches = 6000`
- reward: TCP squared-distance progress plus terminal TCP squared-distance
  penalty
- reset bank: `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5`
- reset replay check: mean live/reference state L2 `0.0`

Commands:

```bash
uv run python -m hcl_poc.cli --config configs/pusht_incremental.yaml \
  rl-rerun collect-vector-data \
  --output data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --num-envs 4096 \
  --batches 2 \
  --max-steps 60 \
  --seed-start 9500000 \
  --no-store-dino

uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --updates 250 \
  --eval-episodes 2 \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 25 \
  --output-dir results/incremental/rl_reachability_debug/run2_privileged_tcp \
  --force
```

Fixed-bank local evaluation:

| Mode | Initial distance | Terminal distance | Reduction | Reach rate | Improved fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial random mean policy | 0.078806 | 0.072764 | 0.006042 | 0.0059 | 0.9784 |
| trained correct goal | 0.078806 | 0.000489 | 0.078317 | 0.9744 | 0.9999 |
| trained shuffled goal | 0.078806 | 0.036897 | 0.041909 | 0.1013 | 0.8141 |

Terminal distance quantiles:

| Mode | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| initial random mean policy | 0.067389 | 0.132611 | 0.198787 |
| trained correct goal | 0.000177 | 0.000807 | 0.007111 |
| trained shuffled goal | 0.023942 | 0.092444 | 0.158354 |

PPO diagnostics:

| Metric | First update | Last update |
| --- | ---: | ---: |
| global env steps | 40,960 | 10,240,000 |
| mean terminal distance | 0.019954 | 0.000721 |
| reach rate | 0.1631 | 0.9487 |
| policy KL | 0.0047 | 0.0066 |
| clip fraction | 0.0494 | 0.0837 |
| entropy | 1.2528 | -0.7190 |
| value loss | 0.02309 | 0.000041 |
| explained variance | -27.516 | 0.912 |
| NaN count | 0 | 0 |
| action saturation | 0.0201 | 0.4633 |

Interpretation:

Run 2 passes the main local reachability gate. PPO can solve the easy
privileged/TCP local MDP from scratch with large parallel batches and enough
policy iterations. The correct-goal policy is far better than the shuffled-goal
condition on the same reset bank, so the learned controller is using the goal.

The main caveat is action saturation: deterministic eval saturation is about
`0.45`, and the last training update saturation is about `0.46`. This is not an
immediate failure because terminal distance and correct-vs-shuffled separation
are strong, but the next diagnostic should check whether the learned behavior is
mostly bang-bang control and whether a smaller action scale, action penalty, or
residual formulation preserves the reachability gain with less saturation.

Next action:

Proceed to the next gated diagnostic in the plan: random shooting / CEM on the
same privileged/TCP local reset bank, using the trained PPO result as a reference
point. Also consider a short Run 2b ablation with an action penalty or lower
initial/action scale if saturation is judged too high.

## 2026-06-30 - Run 3: Privileged/TCP Random Shooting Pilot

Hypothesis:

On the same easy local MDP, random action-sequence search should reveal whether
there is still meaningful local improvement above the trained PPO controller.

Setup:

- reset bank: same `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5`
- eval references: `1` vector reset reference, `4096` local episodes
- base sequence: deterministic Run 2 PPO policy
- random shooting: Gaussian noise around the PPO 10-step action sequence
- noise std: `0.05`
- candidate counts: `32`, `64`, `128`
- selection metric: terminal TCP squared distance to the same held goal

Command:

```bash
uv run python scripts/rl_reachability_tcp_random_shooting.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --checkpoint results/incremental/rl_reachability_debug/run2_privileged_tcp/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt \
  --eval-refs 1 \
  --random-candidates 32 64 128 \
  --noise-stds 0.05 \
  --output results/incremental/rl_reachability_debug/run3_tcp_random_shooting_pilot.json
```

Results:

| Method | Terminal distance | Reach rate | Improved vs PPO | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO deterministic | 0.000602 | 0.9641 | - | 0.000199 | 0.000918 | 0.010293 |
| shooting, 32 candidates | 0.000110 | 0.9954 | 0.9578 | 0.000023 | 0.000225 | 0.001618 |
| shooting, 64 candidates | 0.000081 | 0.9968 | 0.9780 | 0.000014 | 0.000170 | 0.001076 |
| shooting, 128 candidates | 0.000063 | 0.9980 | 0.9893 | 0.000009 | 0.000126 | 0.000786 |

Interpretation:

Random shooting improves substantially over the already successful PPO policy,
so useful 10-step action sequences do exist and the local objective is not
saturated. PPO is good enough to pass the local sanity gate, but it does not
reach the local action-search upper bound.

This suggests the next question is no longer "can PPO learn reachability at all?"
for privileged/TCP. It can. The next useful diagnostics are:

1. whether saturation can be reduced without losing most reachability;
2. whether CEM improves materially beyond random shooting;
3. whether these action-search branches provide good off-policy examples for
   the planned `D_psi` ensemble.
