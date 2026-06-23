# RL Rerun Algorithm Audit

Date: 2026-06-23

This audit covers the Phase D correctness gate from
`low_level_rl_rerun_state_parallel_plan.md` before starting expensive low-level
RL training.

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml algorithm-audit --n-demo 1000 --seed 0 --output results/rl_rerun/algorithm_audit.json
```

Tracked JSON result:

```text
rl_rerun_algorithm_audit.json
```

## Result

Gate: **pass**

| check | result |
| --- | ---: |
| local horizon is 10 steps | pass |
| update period is 10 steps | pass |
| GAE returns match hand-computed 10-step returns | pass |
| terminal local step does not bootstrap | pass |
| nonterminal variant would bootstrap | pass |
| zero residual matches frozen clipped policy | pass |

## GAE Unit Test

The deterministic toy episode uses rewards:

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

with:

```text
gamma = 1
lambda = 1
values = 0
terminated_after_step[9] = True
next_value = 999
```

Expected and computed returns:

```text
[55, 54, 52, 49, 45, 40, 34, 27, 19, 10]
```

Maximum absolute error: `0.0`.

The last return is `10.0`, proving the terminal 10th step does not bootstrap
from the next sampled goal. If the last step is incorrectly treated as
nonterminal, the synthetic `next_value=999` changes the last return by `999`,
which confirms the test is sensitive to value leakage.

## Frozen Policy / Zero Residual

The audit samples `episode_000000`, timestep `5` from
`data/rl_rerun/pusht_state_demos.h5`, builds the local Mode-A condition:

```text
z_t, g=z_{t+10}, previous executed action, remaining=1.0
```

and compares:

```text
clip(pi_BC(condition), -1, 1)
```

against:

```text
clip(pi_BC(condition) + 0.1 * tanh(0), -1, 1)
```

Maximum executed-action difference: `0.0`.

The unclipped frozen action exceeded the action box by `0.0134` on this sample,
so future audits and RL wrappers should consistently compare executed clipped
actions when checking equivalence to the deployed frozen hierarchy.

## Remaining Phase D Work

This audit verifies the algorithmic GAE cutoff and zero-residual equivalence.
The next implementation step is the actual exact local reset-and-replay RL
environment:

1. reset by sampling a stored trajectory/timestep;
2. replay stored executed actions from the original reset to that timestep;
3. set previous executed action from the dataset;
4. hold `z_{t+10}` for exactly 10 primitive steps;
5. terminate the local MDP at step 10 with no value bootstrap.
