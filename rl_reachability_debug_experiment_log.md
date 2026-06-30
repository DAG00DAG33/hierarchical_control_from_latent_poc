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
