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
