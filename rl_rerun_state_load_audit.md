# RL Rerun State-Load Audit

This audit validates Phase A of
[`low_level_rl_rerun_state_parallel_plan.md`](low_level_rl_rerun_state_parallel_plan.md).

## Summary

ManiSkill `PushT-v1` exposes a CUDA simulator state tensor with shape `[1, 79]`.
Restoring this tensor exactly restores low-dimensional observations, but direct
intermediate `set_state()` does not exactly reproduce future contact dynamics
for all timesteps. Reset-and-replay from the stored reset seed and teacher
actions is exact on the pilot dataset.

This means the regenerated dataset can support exact local goal resets through:

```text
reset(seed)
replay stored executed actions up to timestep t
start the local 10-step RL episode
```

Direct arbitrary state load should not be used for the main local RL MDP unless
a richer hidden simulator/controller state becomes available.

## Pilot Dataset

```text
data/rl_rerun/pusht_state_demos_pilot.h5
```

Contents:

| item | value |
| --- | ---: |
| successful trajectories | 2 |
| collection attempts | 3 |
| state shape | 79 |
| state observation dim | 31 |
| DINO dim | 6528 |
| RGB stored | false |
| file size | 1.1M |

## Direct `set_state()` Audit

Direct restore gives exact state/observation parity at the restored timestep,
but future replay can diverge:

| metric | value |
| --- | ---: |
| restore state max abs error | `1.19e-7` |
| restore observation/proprio max abs error | `2.09e-7` |
| 10-step replay state max abs error | `1.13` |
| 10-step replay reward max abs error | `0.665` |
| success mismatches | 1 |

Diagnosis: direct `set_state()` restores actor/articulation state but does not
restore every hidden quantity needed for identical future contact/controller
dynamics. The public `agent.get_controller_state()` returns an empty dict for
this controller, so the missing state is not currently recoverable through that
API.

## Reset-And-Replay Audit

Warm-start reset-and-replay was exact on 20 sampled local windows:

| metric | value |
| --- | ---: |
| restore state max abs error | `0.0` |
| restore observation/proprio max abs error | `0.0` |
| 10-step replay state max abs error | `0.0` |
| 10-step replay reward max abs error | `0.0` |
| success mismatches | 0 |

DINO recomputation on 5 sampled windows:

| metric | value |
| --- | ---: |
| DINO MSE mean | `3.62e-6` |
| DINO MSE max | `4.53e-6` |

## Gate Decision

Phase A is not passed for arbitrary direct `set_state()` local resets.

Phase A is passed for exact reset-and-replay local resets from stored reset seed
and action history. The RL local environment should use reset-and-replay unless
future work identifies a complete hidden simulator/controller state API.

## Full Dataset Audit

After the pilot, the full rerun corpus was collected:

```text
data/rl_rerun/pusht_state_demos.h5
```

| item | value |
| --- | ---: |
| successful trajectories | 1200 |
| collection attempts | 1498 |
| file size | 1.3 GB |
| state shape | 79 |
| state observation dim | 31 |
| DINO dim | 6528 |
| sim backend | `physx_cuda` |

Warm-start reset-and-replay audit on 1000 random windows:

| metric | value |
| --- | ---: |
| sampled windows | 1000 |
| horizon | 10 |
| state restore max abs error | `0.0` |
| observation/proprio restore max abs error | `0.0` |
| 10-step replay state max abs error | `0.0` |
| reward max abs error | `0.0` |
| success mismatches | 0 |
| DINO MSE mean | `3.93e-6` |
| DINO MSE max | `2.51e-5` |

Full Phase A decision: pass for exact reset-and-replay local resets. The next
phase can define nested 500/1000 prefixes and retrain the supervised VAE/high
level/low level components from this regenerated corpus.
