# RL Rerun Vector Dataset Spec

Date: 2026-06-23

The single-env state-loadable corpus is exact only for `num_envs=1`. For
high-throughput local PPO, data must be collected and replayed in the same
vectorized CUDA reset regime. This file documents the vector-consistent corpus
format added for that purpose.

## Collection Commands

Pilot:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-vector-data --num-envs 16 --batches 2 --max-steps 60 --seed-start 9600000 --output data/rl_rerun/pusht_vector_state_demos_pilot.h5 --force
```

Development-scale batch:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-vector-data --num-envs 512 --batches 1 --max-steps 60 --seed-start 9700000 --output data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --force
```

## HDF5 Layout

Root:

```text
meta/
batch_000000/
batch_000001/
...
```

`meta` attributes:

```text
source = vector_consistent_privileged_ppo
checkpoint = artifacts/rl_pusht_official/ppo_best.pt
num_envs
batches
max_steps
seed_start
sim_backend = physx_cuda
control_mode = pd_ee_delta_pos
obs_mode = rgb+state
store_dino
git_commit
git_dirty
```

Each `batch_*` group stores one full CUDA vector rollout:

```text
attrs:
  batch_seed
  num_envs
  max_steps
  success_count

datasets:
  simulator_states          [T+1, N, 79]
  observations_state        [T+1, N, 31]
  proprio                   [T+1, N, 21]
  dino                      [T+1, N, 6528]
  raw_actions               [T, N, 3]
  executed_actions          [T, N, 3]
  previous_executed_actions [T, N, 3]
  rewards                   [T, N]
  terminated                [T, N]
  truncated                 [T, N]
  success                   [T, N]
  success_once              [N]
```

## Exact Local Reset Rule

To recreate a local training state at shared timestep `t`:

1. Create a vector CUDA env with the same `num_envs`.
2. Reset with the stored `batch_seed`.
3. Replay `executed_actions[0:t]` for every stream in the vector batch.
4. Use `previous_executed_actions[t]`.
5. Use the stored Mode-A local goal from `dino[t+10] + proprio[t+10]`.
6. Run the local MDP for exactly 10 primitive steps.

Do not mix streams from different vector batches in one reset. Do not use the
single-env corpus for high-throughput vector local PPO.

## Audits

| corpus | size | streams | successful streams | gate |
| --- | ---: | ---: | ---: | --- |
| `pusht_vector_state_demos_pilot.h5` | 46 MB | 32 | 20 | pass |
| `pusht_vector_state_demos_n512_b1.h5` | 0.72 GB | 512 | 388 | pass |

Tracked audit JSON:

```text
rl_rerun_vector_state_audit_pilot.json
rl_rerun_vector_state_audit_n512_b1.json
```

Both audits verify zero error for:

```text
current simulator state
current observation state
current DINO+proprio frame
future goal simulator state at t+10
future goal observation state at t+10
future goal DINO+proprio frame at t+10
previous executed action
```
