# RL Rerun Local Mode-A Audit

Date: 2026-06-23

This audit verifies the local 10-step Mode-A MDP before PPO training. It uses
the vector-consistent corpus and the frozen supervised hierarchy.

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml local-mode-a-audit --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --episodes 2 --output results/rl_rerun/local_mode_a_audit_n512_b1_seed0.json
```

Tracked JSON:

```text
rl_rerun_local_mode_a_audit_n512_b1_seed0.json
```

## Setup

| item | value |
| --- | ---: |
| vector envs | 512 |
| sampled local episodes | 1024 |
| local horizon | 10 |
| frozen hierarchy | `n=1000, seed=0` |
| vector corpus | `pusht_vector_state_demos_n512_b1.h5` |

At reset, the audit:

1. resets the 512-env CUDA vector env with the stored vector batch seed;
2. replays stored teacher actions to a sampled shared timestep `t`;
3. sets the previous action from `previous_executed_actions[t]`;
4. uses the stored Mode-A goal frame at `t+10`;
5. runs the frozen low-level policy for exactly 10 primitive steps;
6. computes clean latent distance progress only.

## Result

Gate: **pass**

| metric | value |
| --- | ---: |
| initial latent distance mean | 1.351 |
| final latent distance mean | 1.082 |
| mean distance reduction | 0.270 |
| median distance reduction | 0.247 |
| fraction with reduced distance | 0.746 |
| action saturation rate | 0.008 |
| task success diagnostic fraction | 0.452 |

## Interpretation

The frozen supervised low-level policy already moves toward the reachable
Mode-A latent goal on most sampled local episodes, but the average remaining
latent distance is still substantial. This is a valid baseline for R1 local PPO:
the RL objective should improve local latent reach while preserving the low
action saturation observed here.
