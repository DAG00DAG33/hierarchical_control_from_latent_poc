# RL Rerun Local Reset Audit

Date: 2026-06-23

This audit checks whether the regenerated successful PPO corpus can be used for
exact local 10-step resets under the high-throughput vectorized CUDA setting
required by the RL rerun plan.

## Commands

Single-env replay:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml local-reset-audit --n-demo 1000 --seed 0 --num-envs 1 --batches 8 --output results/rl_rerun/local_reset_audit_numenv1.json
```

Vector replay:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml local-reset-audit --n-demo 1000 --seed 0 --num-envs 16 --batches 8 --output results/rl_rerun/local_reset_audit.json
```

Tracked JSON results:

```text
rl_rerun_local_reset_audit_numenv1.json
rl_rerun_local_reset_audit_vector16.json
```

## Results

| audit | sampled resets | state max error | obs-state max error | frame MSE max | previous-action max error | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `num_envs=1` | 8 | 0.0 | 0.0 | 5.57e-6 | 0.0 | pass |
| `num_envs=16` | 128 | 6.18 | 3.35 | 0.509 | 0.0 | fail |

## Diagnosis

The existing corpus was collected with `num_envs=1`. Warm reset-and-replay is
exact when replayed in a new single-env CUDA simulator, but the same stored reset
seeds do not reproduce those initial states inside a multi-env CUDA simulator.

The mismatch appears immediately after vector reset, before any replayed action:

```text
seed_arg 920001                 per-env max reset errors: [0.0667, 1.0945, 0.7465, 0.2308]
seed_arg [920001 repeated]      per-env max reset errors: [0.0667, 1.0945, 0.7465, 0.2308]
seed_arg [920001..920004]       per-env max reset errors: [0.0667, 1.0945, 0.7465, 0.2308]
```

Setting the stored single-env initial physical state into a vector env gives
near-exact immediate state equality, but replay still diverges:

```text
after set_state0 max error: 2.38e-7
after 10-step replay max error: 0.089996
```

This means direct physical state assignment still does not restore all hidden
CUDA/contact/controller state needed for exact future replay.

A separate check confirmed that vector resets are reproducible across fresh
vector env instances when the vector configuration and base seed are identical:

```text
num_envs=16, reset seed=123456, max reset difference across new envs: 0.0
```

## Decision

The current single-env corpus is valid for supervised training and exact
single-env local reset audits, but it is not valid for the high-throughput
parallel local-reset RL gate.

Before running serious local PPO, regenerate a vector-consistent reset/replay
dataset using the same `num_envs` regime intended for local RL. Store at least:

```text
vector_num_envs
vector_batch_seed
vector_env_index
executed actions per stream
observations/features per stream
success flags per stream
```

Then the local RL reset should load an entire vector batch by resetting with the
stored vector batch seed and replaying each stream's stored actions to a shared
timestep. This keeps the hidden CUDA simulator state consistent without relying
on arbitrary intermediate `set_state()`.
