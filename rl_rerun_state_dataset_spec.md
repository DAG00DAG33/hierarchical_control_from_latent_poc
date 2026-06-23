# RL Rerun State Dataset Specification

This file defines the regenerated dataset required by
[`low_level_rl_rerun_state_parallel_plan.md`](low_level_rl_rerun_state_parallel_plan.md).
The dataset is intentionally separate from the old prepared PPO dataset because
old files do not contain loadable simulator states.

## File

Default path:

```text
data/rl_rerun/pusht_state_demos.h5
```

The file stores successful deterministic privileged PPO teacher trajectories
from:

```text
env_id: PushT-v1
obs_mode: rgb+state
sim_backend: physx_cuda
control_mode: pd_ee_delta_pos
control_freq: 20
```

## Metadata

Top-level group:

```text
/meta
```

Required attributes:

| attribute | meaning |
| --- | --- |
| `dataset_type` | `rl_rerun_state_loadable_pusht` |
| `env_id` | ManiSkill environment id |
| `obs_mode` | collection observation mode |
| `control_mode` | action/controller mode |
| `sim_backend` | simulator backend |
| `control_freq` | control frequency in Hz |
| `teacher_checkpoint` | privileged PPO checkpoint path |
| `dino_model` | DINO model name |
| `dino_feature_type` | DINO feature type |
| `dino_spatial_pool` | spatial pooling setting |
| `state_shape` | flattened `env.unwrapped.get_state()` shape |
| `action_shape` | action shape |
| `git_commit` | collection code commit |
| `git_dirty` | whether collection code had uncommitted changes |

## Episode Groups

Each successful trajectory is stored as:

```text
/episode_000000
/episode_000001
...
```

Required attributes:

| attribute | meaning |
| --- | --- |
| `trajectory_id` | integer trajectory id |
| `reset_seed` | environment reset seed |
| `success` | whether teacher succeeded |
| `length` | number of executed actions |

Required datasets:

| dataset | shape | dtype | description |
| --- | ---: | --- | --- |
| `timesteps` | `[T]` | int32 | action timestep indices |
| `simulator_states` | `[T + 1, S]` | float32 | flattened simulator states before action 0 and after every action |
| `observations_state` | `[T + 1, D_state]` | float32 | flattened low-dimensional state observations |
| `proprio` | `[T + 1, 21]` | float32 | proprioception slice used by visual policies |
| `dino` | `[T + 1, D_dino]` | float32 | DINO features for observations at states |
| `raw_actions` | `[T, 3]` | float32 | raw deterministic teacher output |
| `clipped_actions` | `[T, 3]` | float32 | teacher action clipped to env action box |
| `executed_actions` | `[T, 3]` | float32 | action sent to the simulator |
| `previous_executed_actions` | `[T, 3]` | float32 | previous executed action, zero for timestep 0 |
| `rewards` | `[T]` | float32 | normalized dense reward, evaluation-only |
| `terminated` | `[T]` | bool | simulator terminated flag |
| `truncated` | `[T]` | bool | simulator truncated flag |
| `success` | `[T]` | bool | per-step success flag |

Optional datasets:

| dataset | shape | dtype | description |
| --- | ---: | --- | --- |
| `rgb` | `[T + 1, H, W, 3]` | uint8 | RGB frames if storage allows |

## Validation Gates

Before supervised retraining or RL:

1. Reset the environment with the stored `reset_seed`.
2. Restore a sampled `simulator_states[t]` with `env.unwrapped.set_state`.
3. Verify low-dimensional state/proprio parity against stored values.
4. Recompute DINO and verify the feature error is small enough for numerical
   and preprocessing tolerance.
5. Execute the next 10 stored `executed_actions`.
6. Compare final state, reward sequence, success flags, and VAE latent.

Required initial implementation gates:

```text
state/proprio max absolute error <= 1e-5
10-step replay final simulator-state max absolute error <= 1e-5
10-step replay reward max absolute error <= 1e-5
```

VAE latent replay tolerance is checked after the regenerated VAE is trained.

## Phase A Pilot Finding

On CUDA PhysX, `env.unwrapped.get_state()` restores the immediate physical
state and low-dimensional observation to numerical tolerance, but direct
intermediate `set_state()` does not always reproduce future contact dynamics.
The public PandaStick controller state is currently empty:

```text
env.unwrapped.agent.get_controller_state() == {}
```

Therefore the exact local reset procedure for this rerun is:

```text
env.reset(seed=stored_reset_seed)
for j in 0 .. t-1:
    env.step(stored_executed_action[j])
start local 10-step episode at timestep t
```

This reset-and-replay procedure reproduced state, reward, and success exactly
on the pilot audit. Direct arbitrary `set_state()` should be treated as an
optimization candidate only if a complete hidden simulator/controller state API
is found.
