# RL Rerun Experiment Log

This log tracks execution of
[`low_level_rl_rerun_state_parallel_plan.md`](low_level_rl_rerun_state_parallel_plan.md).
The previous low-level RL study is treated as preliminary because it did not
have exact local simulator resets and used only `32 envs x 32 steps`.

## 2026-06-23 - RR-00: Plan intake

Active objective:

```text
Regenerate state-loadable PPO demonstration data, retrain imitation models on
the regenerated data, then run local-reset low-level RL with large GPU
parallelism and clean latent-only rewards.
```

Important changes relative to the prior RL attempt:

- Main RL training must use exact 10-step local goal-reaching resets before
  full-hierarchy rollouts.
- Training rewards may use only latent distance/progress and action
  regularization. ManiSkill dense reward, task success, object pose, and
  hand-designed task progress are evaluation-only diagnostics.
- New supervised checkpoints must be trained from the regenerated data; old
  VAE/high/low checkpoints are forbidden for the main comparison.
- R2 residual-flow is required after local-reset gates pass. R4 direct-flow is
  required only after R2 establishes a stable flow base.
- A serious negative result requires state-loadable data, at least 512 parallel
  environments or an explicitly documented bottleneck, clean local latent
  reward, R1 and R2 tests, one direct fine-tuning method, N=500 and N=1000
  evaluation, and termination/GAE audits.

Immediate Phase A tasks:

1. Verify ManiSkill `PushT-v1` exposes a simulator state that can be saved and
   restored under CUDA PhysX.
2. Implement a regenerated HDF5 corpus that stores reset seed, simulator state,
   teacher actions, previous executed action, DINO/proprio features, rewards,
   and flags.
3. Validate state loading by replaying stored teacher actions from randomly
   sampled `(trajectory, timestep)` states.

## 2026-06-23 - RR-01: CUDA state round-trip smoke

Ran a direct ManiSkill `PushT-v1` CUDA PhysX state restore smoke with
`obs_mode=rgb+state`, `control_mode=pd_ee_delta_pos`, and one environment.

Result:

| check | value |
| --- | ---: |
| device | `cuda` |
| flattened state shape | `(1, 79)` |
| state dict keys | `actors`, `articulations` |
| restore state max abs error | `1.19e-7` |
| restore low-dimensional observation max abs error | `0.0` |
| one-step replay state max abs error | `1.19e-7` |
| one-step replay reward abs error | `0.0` |
| terminated/truncated parity | true |

Conclusion: the simulator exposes a state tensor that can be restored accurately
enough for Phase A. Next step is to collect a pilot HDF5 with these states and
validate multi-step replay from stored intermediate states.

## 2026-06-23 - RR-02: Pilot state dataset and replay audit

Implemented:

```text
hcl-poc rl-rerun collect-state-data
hcl-poc rl-rerun audit-state-data
```

Pilot collection command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-state-data --episodes 2 --output data/rl_rerun/pusht_state_demos_pilot.h5 --seed-start 910000 --max-attempts 80 --force
```

Pilot dataset:

| item | value |
| --- | ---: |
| successful trajectories | 2 |
| attempts | 3 |
| state shape | 79 |
| state observation dim | 31 |
| DINO dim | 6528 |
| file size | 1.1M |

Direct `set_state()` restores the immediate state/observation but does not
produce exact future replay from arbitrary intermediate timesteps:

| metric | value |
| --- | ---: |
| direct restore state max error | `1.19e-7` |
| direct restore proprio max error | `2.09e-7` |
| direct 10-step replay state max error | `1.13` |
| direct reward max error | `0.665` |

Reset-and-replay from the stored reset seed and stored executed actions is
exact:

| metric | value |
| --- | ---: |
| warm-start restore state max error | `0.0` |
| warm-start 10-step replay state max error | `0.0` |
| warm-start reward max error | `0.0` |
| warm-start success mismatches | 0 |
| recomputed DINO MSE mean/max | `3.62e-6` / `4.53e-6` |

Diagnosis: `env.unwrapped.get_state()` does not fully capture hidden
contact/controller state for direct intermediate replay. The public
`agent.get_controller_state()` returns `{}` for this controller. Exact local
resets are still possible using reset-and-replay, but arbitrary direct
`set_state()` should not be used for RL training.

Detailed audit: [`rl_rerun_state_load_audit.md`](rl_rerun_state_load_audit.md).

## 2026-06-23 - RR-03: Full state-loadable dataset and Phase A gate

Full collection command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-state-data --episodes 1200 --output data/rl_rerun/pusht_state_demos.h5 --seed-start 920000 --max-attempts 24000 --force
```

Full dataset:

| item | value |
| --- | ---: |
| successful trajectories | 1200 |
| collection attempts | 1498 |
| file size | 1.3 GB |
| state shape | 79 |
| state observation dim | 31 |
| DINO dim | 6528 |
| sim backend | `physx_cuda` |
| teacher checkpoint | `artifacts/rl_pusht_official/ppo_best.pt` |

Full warm-start audit command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-state-data --dataset data/rl_rerun/pusht_state_demos.h5 --samples 1000 --horizon 10 --seed 42 --warm-start-replay --recompute-dino
```

Result:

| metric | value |
| --- | ---: |
| sampled windows | 1000 |
| state restore max abs error | `0.0` |
| observation/proprio restore max abs error | `0.0` |
| 10-step replay state max abs error | `0.0` |
| reward max abs error | `0.0` |
| success mismatches | 0 |
| DINO MSE mean | `3.93e-6` |
| DINO MSE max | `2.51e-5` |

Phase A gate decision: passed for exact reset-and-replay local resets. Direct
arbitrary `set_state()` remains rejected for intermediate timesteps because it
does not reproduce future contact dynamics.

## 2026-06-23 - RR-04: Phase B supervised training wiring

Implemented:

```text
hcl-poc rl-rerun ensure-action-aliases
hcl-poc rl-rerun train-supervised
```

`ensure-action-aliases` adds an HDF5 hard link:

```text
actions -> executed_actions
```

for every episode, allowing the existing VAE-512 learned-interface loaders to
read the regenerated state dataset without duplicating action arrays.

`train-supervised` builds a rerun-specific config:

```text
prepared_path = data/rl_rerun/pusht_state_demos.h5
artifact root = artifacts/rl_rerun/vae512_scaling/n<N>
result root   = results/rl_rerun/vae512_scaling/n<N>
```

It then writes the nested train/validation manifest, trains the VAE-512
representation and deterministic high/low hierarchy from scratch, and evaluates
the frozen learned-goal hierarchy.

## 2026-06-23 - RR-05: Phase B first supervised rerun checkpoint

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-supervised --n-demo 500 --seed 0 --eval-episodes 100 --force
```

Artifacts:

```text
artifacts/rl_rerun/vae512_scaling/n500/learned_interface/vae512_w2048_b1e6/seed0/
results/rl_rerun/vae512_scaling/n500/learned_interface/vae512_w2048_b1e6/seed0/learned_hierarchy_eval_100.json
```

Dataset split:

| item | value |
| --- | ---: |
| train trajectories | 500 |
| train transitions | 22518 |
| validation trajectories | 200 |
| validation transitions | 9005 |
| equivalent behavior seconds | 1125.9 |

Evaluation result:

| metric | value |
| --- | ---: |
| episodes | 100 |
| success | 0.24 |
| final reward | 0.404 |
| max reward | 0.432 |
| teacher action MAE | 0.192 |
| action saturation rate | 0.043 |
| offline oracle action MAE | 0.0609 |
| offline predicted action MAE | 0.0622 |
| offline normalized goal L2 | 21.35 |
| representation reconstruction MSE | 0.108 |

Gate interpretation: this is close enough to the previous `n=500` learned
hierarchy reference to continue Phase B, but it is not a strong result. The next
check is the matched `n=1000, seed=0` rerun; if that does not recover expected
performance, inspect whether the regenerated teacher corpus distribution differs
from the older successful-data corpus before starting low-level RL.

## 2026-06-23 - RR-06: Phase B `n=1000, seed=0` supervised rerun

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-supervised --n-demo 1000 --seed 0 --eval-episodes 100 --force
```

Artifacts:

```text
artifacts/rl_rerun/vae512_scaling/n1000/learned_interface/vae512_w2048_b1e6/seed0/
results/rl_rerun/vae512_scaling/n1000/learned_interface/vae512_w2048_b1e6/seed0/learned_hierarchy_eval_100.json
```

Dataset split:

| item | value |
| --- | ---: |
| train trajectories | 1000 |
| train transitions | 44020 |
| validation trajectories | 200 |
| validation transitions | 9005 |
| equivalent behavior seconds | 2201.0 |

Evaluation result:

| metric | value |
| --- | ---: |
| episodes | 100 |
| success | 0.46 |
| final reward | 0.577 |
| max reward | 0.598 |
| teacher action MAE | 0.155 |
| action saturation rate | 0.049 |
| offline oracle action MAE | 0.0477 |
| offline predicted action MAE | 0.0487 |
| offline normalized goal L2 | 23.15 |
| representation reconstruction MSE | 0.0569 |
| representation active dimensions | 350 |

Interpretation: increasing the regenerated teacher corpus from 500 to 1000
trajectories gives a clear gain in online success and offline reconstruction /
action prediction. This argues against an obvious data-format regression in the
new corpus. Phase B should continue with additional supervised seeds before the
local-reset RL variants are treated as comparable.
