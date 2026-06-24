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

## 2026-06-23 - RR-07: Phase B `n=500, seed=1` supervised rerun

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-supervised --n-demo 500 --seed 1 --eval-episodes 100 --force
```

Evaluation result:

| metric | value |
| --- | ---: |
| episodes | 100 |
| success | 0.31 |
| final reward | 0.467 |
| max reward | 0.487 |
| teacher action MAE | 0.182 |
| action saturation rate | 0.046 |
| offline oracle action MAE | 0.0611 |
| offline predicted action MAE | 0.0620 |
| offline normalized goal L2 | 21.95 |
| representation reconstruction MSE | 0.109 |
| representation active dimensions | 512 |

Interim `n=500` seed summary:

| seed | success | final reward | max reward | teacher action MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.24 | 0.404 | 0.432 | 0.192 |
| 1 | 0.31 | 0.467 | 0.487 | 0.182 |

Interpretation: `n=500` has moderate seed variation but remains in the same
performance band. Continue with `seed=2` before using this as the matched
supervised baseline for local low-level RL.

## 2026-06-23 - RR-08: Phase B `n=500, seed=2` supervised rerun

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-supervised --n-demo 500 --seed 2 --eval-episodes 100 --force
```

Evaluation result:

| metric | value |
| --- | ---: |
| episodes | 100 |
| success | 0.29 |
| final reward | 0.440 |
| max reward | 0.472 |
| teacher action MAE | 0.190 |
| action saturation rate | 0.044 |
| offline oracle action MAE | 0.0602 |
| offline predicted action MAE | 0.0613 |
| offline normalized goal L2 | 22.66 |
| representation reconstruction MSE | 0.108 |
| representation active dimensions | 512 |

Completed `n=500` supervised rerun summary:

| seed | success | final reward | max reward | teacher action MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.24 | 0.404 | 0.432 | 0.192 |
| 1 | 0.31 | 0.467 | 0.487 | 0.182 |
| 2 | 0.29 | 0.440 | 0.472 | 0.190 |
| mean | 0.28 | 0.437 | 0.463 | 0.188 |
| sample SD | 0.036 | 0.032 | 0.028 | 0.005 |

Interpretation: the regenerated `n=500` supervised hierarchy is stable across
three seeds but only moderately successful. It is suitable as a low-data matched
baseline for Phase B, not as a strong final system.

## 2026-06-23 - RR-09: Phase B `n=1000, seed=1` supervised rerun

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-supervised --n-demo 1000 --seed 1 --eval-episodes 100 --force
```

Evaluation result:

| metric | value |
| --- | ---: |
| episodes | 100 |
| success | 0.48 |
| final reward | 0.598 |
| max reward | 0.613 |
| teacher action MAE | 0.147 |
| action saturation rate | 0.045 |
| offline oracle action MAE | 0.0477 |
| offline predicted action MAE | 0.0487 |
| offline normalized goal L2 | 22.47 |
| representation reconstruction MSE | 0.0586 |
| representation active dimensions | 512 |

Interim `n=1000` seed summary:

| seed | success | final reward | max reward | teacher action MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.46 | 0.577 | 0.598 | 0.155 |
| 1 | 0.48 | 0.598 | 0.613 | 0.147 |

Interpretation: the first two `n=1000` rerun seeds are consistent and clearly
above the `n=500` band. Complete `seed=2` before Phase B gate judgment.

## 2026-06-23 - RR-10: Phase B `n=1000, seed=2` supervised rerun and gate

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-supervised --n-demo 1000 --seed 2 --eval-episodes 100 --force
```

Evaluation result:

| metric | value |
| --- | ---: |
| episodes | 100 |
| success | 0.43 |
| final reward | 0.563 |
| max reward | 0.582 |
| teacher action MAE | 0.153 |
| action saturation rate | 0.046 |
| offline oracle action MAE | 0.0470 |
| offline predicted action MAE | 0.0482 |
| offline normalized goal L2 | 20.63 |
| representation reconstruction MSE | 0.0640 |
| representation active dimensions | 512 |

Completed `n=1000` supervised rerun summary:

| seed | success | final reward | max reward | teacher action MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.46 | 0.577 | 0.598 | 0.155 |
| 1 | 0.48 | 0.598 | 0.613 | 0.147 |
| 2 | 0.43 | 0.563 | 0.582 | 0.153 |
| mean | 0.457 | 0.580 | 0.598 | 0.152 |
| sample SD | 0.025 | 0.018 | 0.016 | 0.004 |

Phase B supervised gate decision: pass for proceeding to local-reset low-level
RL development. The regenerated state/replay corpus produces a reproducible
VAE-512 learned-interface hierarchy, and `n=1000` improves clearly over `n=500`
in online success, final reward, max reward, and teacher-action MAE. This is not
a final system result; it is the matched supervised baseline that the exact
local-reset RL methods must compare against.

## 2026-06-23 - RR-11: Phase C bounded throughput benchmark

Implemented:

```text
hcl-poc rl-rerun throughput-benchmark
```

The command writes a CSV with simulator-only, render, render+DINO, and full
DINO+VAE+high+low policy throughput. It catches failed points and records the
crash/error in the row so the sweep can identify the largest stable setting.

Smoke command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml throughput-benchmark --num-envs 128 --rollout-lens 10 --n-demo 1000 --seed 0 --output results/rl_rerun/rl_rerun_throughput_benchmark_smoke.csv
```

Bounded sweep command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml throughput-benchmark --num-envs 128,256,512,1024 --rollout-lens 10,32,64 --n-demo 1000 --seed 0 --output results/rl_rerun/rl_rerun_throughput_benchmark.csv
```

Tracked CSV deliverable:

```text
rl_rerun_throughput_benchmark.csv
```

Selected results:

| num envs | rollout len | batch | sim-only steps/s | render steps/s | DINO steps/s | full-stack steps/s | PPO update wall-clock |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 64 | 8192 | 8965 | 5184 | 395 | 387 | 21.2 s |
| 256 | 64 | 16384 | 16798 | 7258 | 410 | 391 | 41.9 s |
| 512 | 64 | 32768 | 30249 | 8896 | 416 | 407 | 80.6 s |
| 1024 | 32 | 32768 | 46747 | 10168 | 427 | 417 | 78.5 s |
| 1024 | 64 | 65536 | 44151 | 10150 | 429 | 420 | 155.9 s |

No tested point crashed or produced NaNs. The effective-batch requirement is met
at `512 x 64` and `1024 x 32`. The limiting stage is DINO/full-stack inference:
sim-only throughput scales to tens of thousands of steps/s, while the full stack
stays near 400 env-steps/s. Larger sweeps can still be run to find a memory
limit, but they will not materially improve wall-clock throughput unless the
visual feature pipeline is optimized or cached for local-reset RL.

Phase C decision for development: use `512 x 64` or `1024 x 32` as the first
serious GPU-parallel settings. Treat any smaller setting as exploratory. Before
expensive RL, run the Phase D correctness checks for local 10-step episodes and
GAE value leakage.

## 2026-06-23 - RR-12: Phase D algorithm audit

Implemented:

```text
hcl-poc rl-rerun algorithm-audit
```

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml algorithm-audit --n-demo 1000 --seed 0 --output results/rl_rerun/algorithm_audit.json
```

Tracked deliverables:

```text
rl_rerun_algorithm_audit.md
rl_rerun_algorithm_audit.json
```

Result: pass.

| check | value |
| --- | ---: |
| horizon steps | 10 |
| update period | 10 |
| GAE hand-computed max error | 0.0 |
| terminal last return with `next_value=999` | 10.0 |
| nonterminal bootstrap sensitivity | 999.0 |
| zero residual executed-action error | 0.0 |
| unclipped frozen action-box overshoot | 0.0134 |

Interpretation: the local 10-step episode semantics and no-bootstrap terminal
cutoff are correct in the standalone audit. The zero-residual check must compare
executed clipped actions, because the frozen BC policy can slightly exceed the
action box before deployment clipping. The remaining Phase D implementation
work is the exact reset-and-replay local RL environment itself.

## 2026-06-23 - RR-13: Phase D local reset audit

Implemented:

```text
hcl-poc rl-rerun local-reset-audit
```

Results:

| audit | sampled resets | state max error | obs-state max error | frame MSE max | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `num_envs=1` | 8 | 0.0 | 0.0 | 5.57e-6 | pass |
| `num_envs=16` | 128 | 6.18 | 3.35 | 0.509 | fail |

Detailed audit:

```text
rl_rerun_local_reset_audit.md
rl_rerun_local_reset_audit_numenv1.json
rl_rerun_local_reset_audit_vector16.json
```

Diagnosis: the successful teacher corpus was collected with `num_envs=1`.
Reset-and-replay is exact in a new single-env CUDA simulator, but the stored
single-env reset seeds do not reproduce the same initial states inside a
multi-env CUDA simulator. The mismatch appears immediately after vector reset,
before replay. Directly setting the stored initial physical states into the
vector env gives immediate equality but replay still diverges, so hidden CUDA
state is still missing.

Decision: do not run high-throughput local PPO on the current single-env corpus.
Regenerate a vector-consistent local reset corpus using the same vectorized CUDA
reset regime intended for RL. The dataset must store vector batch seed,
`vector_num_envs`, stream index, actions, and features so exact local resets can
reset a whole vector batch and replay each stream to a shared timestep.

## 2026-06-23 - RR-14: Vector-consistent reset corpus

Implemented:

```text
hcl-poc rl-rerun collect-vector-data
hcl-poc rl-rerun audit-vector-data
```

The new corpus stores full vector CUDA batches with `batch_seed`, stream-wise
actions, simulator states, DINO features, proprioception, and success flags.
The local reset rule is: recreate the same vector env size, reset with the
stored vector `batch_seed`, and replay all streams in that batch to a shared
timestep. Streams from different vector batches must not be mixed in one reset.

Detailed spec:

```text
rl_rerun_vector_dataset_spec.md
```

Pilot collection:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-vector-data --num-envs 16 --batches 2 --max-steps 60 --seed-start 9600000 --output data/rl_rerun/pusht_vector_state_demos_pilot.h5 --force
```

Pilot audit:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-vector-data --dataset data/rl_rerun/pusht_vector_state_demos_pilot.h5 --batches 4 --seed 0 --horizon 10 --output results/rl_rerun/vector_state_audit_pilot.json
```

Development-scale collection:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-vector-data --num-envs 512 --batches 1 --max-steps 60 --seed-start 9700000 --output data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --force
```

Development-scale audit:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml audit-vector-data --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --batches 4 --seed 1 --horizon 10 --output results/rl_rerun/vector_state_audit_n512_b1.json
```

Results:

| corpus | size | streams | successful streams | current state error | goal state error | frame MSE | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pilot `16 x 2 x 60` | 46 MB | 32 | 20 | 0.0 | 0.0 | 0.0 | pass |
| dev `512 x 1 x 60` | 0.72 GB | 512 | 388 | 0.0 | 0.0 | 0.0 | pass |

Tracked audit outputs:

```text
rl_rerun_vector_state_audit_pilot.json
rl_rerun_vector_state_audit_n512_b1.json
```

Interpretation: the vector-consistent collection fixes the hidden-state problem
found in RR-13. The `512 x 60` corpus is exact for current state and local
Mode-A future goal replay at `t+10`, matching the first serious parallelism
setting from the throughput gate. The next implementation step is to build the
Mode-A local PPO environment on top of this vector batch reset rule.

## 2026-06-23 - RR-15: Local Mode-A frozen-policy audit

Implemented:

```text
hcl-poc rl-rerun local-mode-a-audit
```

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml local-mode-a-audit --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --episodes 2 --output results/rl_rerun/local_mode_a_audit_n512_b1_seed0.json
```

Tracked outputs:

```text
rl_rerun_local_mode_a_audit.md
rl_rerun_local_mode_a_audit_n512_b1_seed0.json
```

Result:

| metric | value |
| --- | ---: |
| sampled local episodes | 1024 |
| horizon | 10 |
| initial latent distance mean | 1.351 |
| final latent distance mean | 1.082 |
| mean distance reduction | 0.270 |
| median distance reduction | 0.247 |
| fraction with reduced distance | 0.746 |
| action saturation rate | 0.008 |
| task success diagnostic fraction | 0.452 |

Interpretation: the exact vector local reset environment is usable and the
frozen supervised low-level policy usually moves toward the reachable Mode-A
goal. This establishes the local baseline for R1 residual PPO. The R1 reward
should use the same clean latent progress and terminal distance terms, not
ManiSkill task reward or task-progress shaping.

## 2026-06-23 - RR-16: R1 local PPO smoke

Implemented:

```text
hcl-poc rl-rerun train-local-r1
hcl-poc rl-rerun eval-local-r1
```

Training command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r1 --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --run-name smoke_32k --steps 262144 --alpha 0.1 --terminal-weight 1.0
```

Evaluation commands:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml local-mode-a-audit --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --episodes 4 --output results/rl_rerun/local_mode_a_audit_n512_b1_seed0_eval4.json
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r1 --checkpoint artifacts/rl_rerun/local_r1/n1000/seed0/smoke_32k/latest.pt --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --episodes 4 --output results/rl_rerun/local_r1/n1000/seed0/smoke_32k/eval_local_4_after262k.json
```

Tracked outputs:

```text
rl_rerun_local_r1_smoke.md
rl_rerun_local_r1_smoke_262k_history.json
rl_rerun_local_mode_a_audit_n512_b1_seed0_eval4.json
rl_rerun_local_r1_smoke_262k_eval4.json
```

Training summary:

| metric | first update | last update |
| --- | ---: | ---: |
| global step | 32768 | 262144 |
| mean terminal distance | 1.233 | 1.079 |
| action saturation rate | 0.071 | 0.034 |
| clip fraction | 0.058 | 0.118 |
| explained variance | -0.534 | 0.449 |

Paired local evaluation on 2048 local episodes:

| policy | final distance | distance reduction | reduction fraction | saturation |
| --- | ---: | ---: | ---: | ---: |
| frozen BC low-level | 1.131 | 0.415 | 0.812 | 0.0219 |
| R1 residual, 262k | 1.138 | 0.408 | 0.799 | 0.0207 |

Decision: R1 smoke validates the local PPO code path but does not pass the
local-goal improvement gate. The deterministic residual remains very small
(`mean_residual_norm = 0.00335`) and mostly reproduces the frozen controller.
Next tuning should try lower residual penalty, larger `alpha`, and/or larger
residual learning rate, and should evaluate on fixed local validation resets
during training.

## 2026-06-23 - RR-17: R1 local PPO alpha/penalty tuning

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r1 --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --run-name alpha025_nopenalty_262k --steps 262144 --alpha 0.25 --terminal-weight 1.0 --residual-penalty-weight 0.0 --force
```

Evaluation:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml eval-local-r1 --checkpoint artifacts/rl_rerun/local_r1/n1000/seed0/alpha025_nopenalty_262k/latest.pt --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --episodes 4 --output results/rl_rerun/local_r1/n1000/seed0/alpha025_nopenalty_262k/eval_local_4.json
```

Tracked outputs:

```text
rl_rerun_local_r1_alpha025_nopenalty_262k_history.json
rl_rerun_local_r1_alpha025_nopenalty_262k_eval4.json
```

Paired local evaluation on 2048 local episodes:

| policy | final distance | distance reduction | reduction fraction | residual norm | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen BC low-level | 1.131 | 0.415 | 0.812 | n/a | 0.0219 |
| R1 alpha 0.10, penalty 0.01 | 1.138 | 0.408 | 0.799 | 0.00335 | 0.0207 |
| R1 alpha 0.25, penalty 0.00 | 1.135 | 0.411 | 0.804 | 0.00886 | 0.0213 |

Decision: increasing residual authority and removing the residual penalty
increased deterministic residual usage but still did not pass the local-goal
gate. The result remains slightly below the frozen controller. Before spending
on 1M+ steps, improve the training setup: use multiple vector batches, add
fixed validation evaluation during training, and consider a stronger residual
mean optimization signal.

## 2026-06-23 - RR-18: Align PPO rollouts with the local MDP

The PPO rollout is now exactly one complete local episode:

```text
future-goal horizon: 10 simulator steps
rollout steps per environment: 10
terminal mask: true after step 10
```

This removes concatenated local episodes and partial episodes at PPO update
boundaries. Each collected trajectory has one initial state, one reachable
10-step future goal, and exactly ten policy decisions.

## 2026-06-23 - RR-19: Large 10-step vector-width benchmark

Benchmark:

| environments | samples/update | full stack steps/s | estimated rollout/update | result |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 20480 | 391.5 | 52.3 s | pass |
| 4096 | 40960 | 392.1 | 104.5 s | pass |
| 8192 | 81920 | n/a | n/a | GPU camera-group allocation failure |

The full stack includes CUDA simulation, rendering, DINO, VAE, and policy
inference. DINO/rendering saturates throughput near 400 environment steps/s,
but 4096 environments is stable and provides a sufficiently large one-episode
PPO batch. Select `num_envs=4096`.

## 2026-06-23 - RR-20: 4096-environment exact-replay corpus

Collected:

```text
data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5
vector environments per batch: 4096
independent vector batches: 2
stored simulator steps per stream: 60
total streams: 8192
file size: 12 GB
collection time: 25 min 26 s
```

The initial exact replay audit passed with zero error for simulator state,
state observation, DINO/proprio frame, previous action, and the stored
10-step future goal. The audit implementation was then changed to sample
distinct vector batches whenever the requested audit count does not exceed
the number of stored batches.

## 2026-06-23 - RR-21: Explicit frozen low-level baseline

The primary R1 baseline is the frozen supervised low-level controller, not the
privileged PPO expert and not the complete learned hierarchy. It receives the
same current observation latent, reachable 10-step Mode-A future latent,
previous action, and remaining-horizon input as the residual RL policy. The
only difference is that its RL residual is identically zero.

Evaluation on one complete 4096-environment reset batch:

| metric | frozen BC low level |
| --- | ---: |
| sampled local episodes | 4096 |
| initial latent distance | 1.466 |
| final latent distance | 1.105 |
| mean latent-distance reduction | 0.361 |
| fraction ending closer to the goal | 0.793 |
| action saturation rate | 0.0144 |
| task-success diagnostic fraction | 0.314 |

The task-success fraction is secondary because each local episode starts from
an arbitrary teacher-trajectory state and lasts only ten steps. The R1 gate is
a paired comparison on identical starting states and reachable goals:

```text
RL residual final distance <= frozen BC final distance
RL residual distance-reduction fraction >= frozen BC fraction
```

For system-level context, the complete frozen `n=1000` learned hierarchy has
100-episode success rates of `0.46`, `0.48`, and `0.43` for policy seeds 0, 1,
and 2 respectively (`mean=0.457`, sample SD `0.025`). Seed 0 is the frozen
checkpoint used by the current local R1 run.

## 2026-06-23 - RR-22: Aligned 4096-environment R1 result

Training:

```text
method: deterministic residual PPO, Mode A
environments: 4096
rollout/episode horizon: 10
samples per PPO update: 40960
minibatches: 8
minibatch size: 5120
updates: 8
total transitions: 327680
residual scale alpha: 0.25
residual penalty: 0.0
```

Paired evaluation across both stored vector batches, 8192 local episodes:

| policy | final distance | mean reduction | reduction fraction | saturation |
| --- | ---: | ---: | ---: | ---: |
| frozen BC low level | 1.2328 | 0.5331 | 0.8649 | 0.0344 |
| aligned R1 residual | 1.2250 | 0.5409 | 0.8701 | 0.0339 |

The residual improves final distance by `0.0078` and the distance-improvement
fraction by `0.0052`. Its deterministic mean residual norm is `0.00868`.
This is the first positive paired R1 result, but the margin is small and the
same two vector batches supplied training resets. Validate on a separately
collected vector seed before extending the run to one million transitions.

## 2026-06-23 - RR-23: Held-out local evaluation manifest

Added `create-local-eval-manifest` and `--manifest` support to both frozen and
R1 local evaluators. A manifest records:

```text
dataset
vector batch
batch reset seed
local start timestep
future-goal horizon
number of vector environments
```

All paired policy comparisons must now use the same manifest. This prevents
absolute baseline metrics from changing merely because the evaluators sampled
different start timesteps.

## 2026-06-23 - RR-24: Independent 4096-environment validation

Collected a held-out vector corpus from a new reset seed:

```text
dataset: data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5
batch seed: 9900000
environments: 4096
stored steps: 60
size: 5.8 GB
```

Exact replay passed with zero simulator-state, observation, DINO/proprio-frame,
previous-action, and 10-step goal error.

Paired manifest:

```text
results/rl_rerun/local_eval_manifest_n4096_val_b1_seed20260623.json
batch: batch_000000
timestep: 34
local episodes: 4096
```

Held-out result:

| policy | final distance | mean reduction | reduction fraction | saturation |
| --- | ---: | ---: | ---: | ---: |
| frozen BC low level | 1.08587 | 0.36427 | 0.79102 | 0.01130 |
| aligned R1 residual | 1.08302 | 0.36712 | 0.79468 | 0.01050 |

The R1 residual generalizes a small positive gain: final distance improves by
`0.00285` and distance-improvement frequency by `0.00366`. This is evidence
that the result is not only memorization of the two training vector seeds, but
it is far below the Phase E target of 25% lower final distance and 15
percentage points higher goal reach. Treat this checkpoint as a successful
correctness/smoke result, not as a passed R1 local-goal gate.

## 2026-06-23 - RR-25: Serious N=500 R1 setup

The Phase E gate is defined at `N_demo=500`; the preceding aligned run used
`N_demo=1000` and therefore remains an implementation smoke test.

Trainer changes for the serious run:

```text
rollout length: exactly 10
environments: 4096
samples/update: 40960
residual learning rate: explicit CLI override, 3e-4
value-loss coefficient: 1.0
gradient-norm limit: 1.0
periodic checkpoint snapshots: every 5 PPO updates
checkpoint recipe: records all PPO and actor-critic hyperparameters
```

Collected an independent, lower-cost checkpoint-selection bank:

```text
dataset: data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5
batch seed: 9950000
environments: 512
stored steps: 60
```

Its exact replay audit passes. Periodic checkpoints will be selected using
latent reachability on this bank, never task success.

## 2026-06-23 - RR-26: N=500 aligned R1 at 1M transitions

Training completed:

```text
run: aligned10_n4096_lr3e4_alpha025_nopenalty_1m
n_demo: 500
RL-state corpus: data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5
environments: 4096
rollout horizon: 10
updates: 25
transitions: 1,024,000
learning rate: 3e-4
alpha: 0.25
residual penalty: 0.0
```

Training terminal latent distance improved from `1.071` to `0.711`, but the
decision must use held-out evaluation.

Held-out 512-env local checkpoint selection:

| checkpoint | final distance | mean reduction | reduction fraction | residual norm |
| --- | ---: | ---: | ---: | ---: |
| frozen BC | 0.6073 | 0.5193 | 0.8301 | 0.0000 |
| 204800 | 0.6095 | 0.5172 | 0.8477 | 0.0094 |
| 409600 | 0.6083 | 0.5184 | 0.8281 | 0.0127 |
| 614400 | 0.6042 | 0.5225 | 0.8418 | 0.0155 |
| 819200 | 0.5967 | 0.5300 | 0.8379 | 0.0171 |
| 1024000 | 0.5912 | 0.5354 | 0.8418 | 0.0188 |

The final checkpoint is best by held-out final latent distance, improving over
frozen by `0.0161` absolute (`2.65%`). This is positive but still far below
the Phase E gate (`25%` final-distance improvement and `15` percentage point
goal-reach improvement).

Closed-loop paired evaluation on 100 full Push-T episodes:

| checkpoint | frozen success | residual success | success delta | final reward delta | max reward delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 204800 | 0.34 | 0.31 | -0.03 | -0.0235 | -0.0218 |
| 409600 | 0.34 | 0.35 | +0.01 | +0.0048 | +0.0090 |
| 614400 | 0.34 | 0.34 | 0.00 | -0.0021 | -0.0023 |
| 819200 | 0.34 | 0.34 | 0.00 | -0.0036 | +0.0029 |
| 1024000 | 0.34 | 0.31 | -0.03 | -0.0221 | -0.0225 |

Interpretation: local latent reachability and closed-loop task success are not
monotonic together. The best latent checkpoint is not the best deployed
checkpoint. The only positive closed-loop signal is small (`+1` success on 100
episodes) and too weak to pass the full-hierarchy gate. Before R2, run the
requested ablation: train R1 on a separate ~500-stream expert-state corpus
that is disjoint from the BC training trajectories.

R2 remains on the active task list. After the disjoint-state R1 ablation and
closed-loop checks are logged, run the residual flow low-level PPO branch from
the rerun plan instead of dropping it based on the deterministic R1 result.

## 2026-06-23 - RR-27: Disjoint-state R1 ablation

Question: does RL improve more if its reset states come from expert trajectories
not used to train the frozen `N=500` low-level BC policy?

Constraint: the strict "episodes 500--999 from the single-env corpus" version
cannot be used for large-vector PPO because previous reset audits showed that
single-env intermediate states do not replay exactly after loading into a
vectorized simulator. The vector-valid ablation uses a separately collected
512-stream expert corpus with exact vector reset/replay:

```text
BC training data: first 500 single-env expert trajectories
RL local reset corpus: data/rl_rerun/pusht_vector_state_demos_n512_b1.h5
RL corpus source: fresh vector expert rollouts, not the BC training file split
environments: 512
rollout horizon: 10
minibatches: 1
learning rate: 3e-4
alpha: 0.25
residual penalty: 0.0
transitions: 1,024,000
```

Training diagnostics did not improve monotonically: terminal latent distance
started at `0.607` and ended at `0.619`, while residual norm grew from `0.0396`
to `0.0672`.

Held-out 512-env local checkpoint selection:

| checkpoint | final distance | mean reduction | reduction fraction | residual norm |
| --- | ---: | ---: | ---: | ---: |
| frozen BC | 0.6073 | 0.5193 | 0.8301 | 0.0000 |
| 204800 | 0.5958 | 0.5309 | 0.8438 | 0.0230 |
| 409600 | 0.6163 | 0.5103 | 0.8438 | 0.0333 |
| 614400 | 0.6359 | 0.4907 | 0.8281 | 0.0425 |
| 819200 | 0.6250 | 0.5016 | 0.8281 | 0.0461 |
| 1024000 | 0.6363 | 0.4903 | 0.8340 | 0.0518 |

The best local checkpoint is early, at `204800` transitions. It improves held-
out final latent distance by `0.0115` absolute (`1.90%`) and improves the
distance-reduction fraction by `1.37` percentage points.

Closed-loop paired evaluation for that best local checkpoint on the same 100
deployment seeds:

| policy | success | final reward | max reward | saturation |
| --- | ---: | ---: | ---: | ---: |
| frozen BC low level | 0.34 | 0.4946 | 0.5163 | 0.0444 |
| disjoint-state R1 | 0.30 | 0.4544 | 0.4820 | 0.0388 |

Conclusion: using a separate vector expert-state corpus did not solve the R1
problem. It can improve local latent reach slightly at an early checkpoint, but
the deployed closed-loop hierarchy gets worse. The deterministic residual R1
branch remains below the plan gates. Proceed to R2 residual flow low-level PPO
as the next active branch.

## 2026-06-23 - RR-28: R2 residual-flow plumbing and smoke

Clarification on the last R1 run: the disjoint-state R1 ablation used the
available exact-reset `512`-stream vector corpus, with rollout horizon `10`, so
the PPO batch was only `5120` samples/update. That satisfies the minimum
parallel environment count from the plan, but it is much smaller than the
serious aligned R1 run (`4096 x 10 = 40960` samples/update). Interpret the
disjoint result as an ablation, not as the strongest possible R1 PPO setting.

Implemented R2 as a matched variant of the R1 local PPO loop:

- train a zero-noise action-flow low-level base on the same learned-interface
  low-level condition used by the deterministic policy;
- use the flow endpoint as the frozen base action;
- train a residual actor-critic on top with the same clean latent progress and
  terminal-distance reward as R1;
- keep local rollouts exactly `10` steps long;
- disallow ManiSkill reward, task success, object pose, and task progress in
  training.

Commands:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-low-flow-base --n-demo 500 --seed 0

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r2 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 \
  --n-demo 500 --seed 0 --run-name smoke_10k --steps 10240 \
  --alpha 0.25 --terminal-weight 1.0 --residual-penalty-weight 0.0 \
  --learning-rate 0.0003 --num-minibatches 1 --checkpoint-every-updates 1 \
  --flow-checkpoint artifacts/rl_rerun/local_r2/n500/seed0/low_flow_base/low_flow.pt --force
```

Low-flow base:

| metric | value |
| --- | ---: |
| best epoch | 51 |
| zero-noise validation action MAE | 0.0640 |
| flow steps | 24 |
| condition dim | 7065 |
| train time | 253 s |

Smoke local held-out eval on the existing 512-env manifest:

| policy | final latent distance | reduction | improved fraction | saturation |
| --- | ---: | ---: | ---: | ---: |
| deterministic frozen baseline | 0.6073 | 0.5193 | 0.8301 | 0.0100 |
| R2 smoke, 10k transitions | 0.6607 | 0.4659 | 0.7949 | 0.0150 |

The 10k smoke checkpoint has mean residual norm `0.0056`, so this mostly
measures the frozen zero-noise flow base. The flow base is locally worse than
the deterministic low-level before serious residual tuning. Next run R2 on the
4096-stream corpus with the same `4096 x 10` batch size used for the serious R1
run.

## 2026-06-23 - RR-29: Serious N=500 R2 residual-flow PPO

Run:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r2 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 --seed 0 \
  --run-name aligned10_n4096_lr3e4_alpha025_nopenalty_1m \
  --steps 1024000 --alpha 0.25 --terminal-weight 1.0 \
  --residual-penalty-weight 0.0 --learning-rate 0.0003 \
  --num-minibatches 8 --checkpoint-every-updates 5 \
  --flow-checkpoint artifacts/rl_rerun/local_r2/n500/seed0/low_flow_base/low_flow.pt --force
```

Configuration:

| item | value |
| --- | ---: |
| environments | 4096 |
| rollout horizon | 10 |
| samples/update | 40960 |
| minibatches | 8 |
| minibatch size | 5120 |
| total transitions | 1,024,000 |
| base policy | zero-noise action flow |
| flow steps | 24 |

Training diagnostics:

| step | train mean distance | train terminal distance | residual norm | saturation |
| ---: | ---: | ---: | ---: | ---: |
| 40,960 | 1.9320 | 1.0915 | 0.0398 | 0.2088 |
| 860,160 | 0.9323 | 0.6604 | 0.0419 | 0.0121 |
| 901,120 | 0.9891 | 0.7193 | 0.0421 | 0.0240 |
| 942,080 | 1.0611 | 0.7359 | 0.0421 | 0.0177 |
| 983,040 | 0.9884 | 0.7239 | 0.0419 | 0.0208 |
| 1,024,000 | 1.0404 | 0.7370 | 0.0418 | 0.0170 |

Held-out 512-env local checkpoint selection:

| checkpoint | final distance | mean reduction | reduction fraction | residual norm | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| flow base smoke-like | 0.6607 | 0.4659 | 0.7949 | 0.0056 | 0.0150 |
| 204800 | 0.6408 | 0.4858 | 0.8145 | 0.0099 | 0.0150 |
| 409600 | 0.6435 | 0.4831 | 0.8086 | 0.0139 | 0.0135 |
| 614400 | 0.6444 | 0.4822 | 0.8262 | 0.0183 | 0.0127 |
| 819200 | 0.6377 | 0.4890 | 0.8340 | 0.0189 | 0.0127 |
| 1024000 | 0.6267 | 0.4999 | 0.8359 | 0.0207 | 0.0129 |

The best R2 local checkpoint is the final checkpoint. It improves the held-out
flow-base local final distance by `0.0340` absolute (`5.15%`) and improves the
reduction fraction by `4.10` percentage points. However, it remains worse than
the deterministic frozen low-level in final latent distance (`0.6267` versus
`0.6073`), though the reduction fraction is slightly higher (`0.8359` versus
`0.8301`).

Closed-loop paired evaluation for the best R2 checkpoint on 100 deployment
seeds:

| policy | success | final reward | max reward | saturation |
| --- | ---: | ---: | ---: | ---: |
| frozen flow-base low level | 0.28 | 0.4438 | 0.4783 | 0.0427 |
| R2 residual flow low level | 0.23 | 0.3919 | 0.4357 | 0.0439 |

R2 closed-loop deltas:

| metric | delta |
| --- | ---: |
| success | -0.05 |
| final reward | -0.0519 |
| max reward | -0.0426 |

Conclusion: R2 successfully improves local latent reaching relative to its weak
flow base, but the improvement does not transfer to full deployment. The flow
base itself is worse than the deterministic low-level (`28%` versus the R1
deterministic frozen baseline of `34%` on the same deployment seed range), and
the residual-tuned flow policy further degrades closed-loop success. R2 does
not pass the local or closed-loop gates.

## 2026-06-23 - RR-30: R3 direct deterministic low-level last-layer tuning

Because R1 and R2 did not pass, the next plan branch is R3: directly tune the
deterministic low-level policy starting from BC. The first R3 variant follows
the plan's conservative setting: tune only the final low-policy layer plus
actor log-std and critic, with BC action regularization.

Implementation:

- exact-reset local Mode-A data path, same as R1/R2;
- one local episode per PPO rollout (`10` steps);
- deterministic low-level final layer trainable;
- earlier low-level layers, high-level model, representation encoder, and
  normalizers frozen;
- PPO samples the final action directly;
- BC regularization keeps the deterministic mean near the frozen BC action;
- training reward remains clean latent progress plus terminal latent distance;
- no ManiSkill reward, task success, object pose, or task progress in training.

Smoke on 512-env exact-reset corpus:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 \
  --n-demo 500 --seed 0 --run-name smoke_10k_bc1 --steps 10240 \
  --bc-weight 1.0 --terminal-weight 1.0 --learning-rate 0.00003 \
  --num-minibatches 1 --checkpoint-every-updates 1 --force
```

Held-out 512-env smoke eval:

| policy | final distance | mean reduction | reduction fraction | action delta | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen deterministic low level | 0.6073 | 0.5193 | 0.8301 | 0.0000 | 0.0100 |
| R3 smoke, 10k transitions | 0.6020 | 0.5246 | 0.8379 | 0.0030 | 0.0092 |

Serious run:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 --seed 0 --run-name aligned10_n4096_lr3e5_bc1_1m \
  --steps 1024000 --bc-weight 1.0 --terminal-weight 1.0 \
  --learning-rate 0.00003 --num-minibatches 8 --checkpoint-every-updates 5 --force
```

Configuration:

| item | value |
| --- | ---: |
| environments | 4096 |
| rollout horizon | 10 |
| samples/update | 40960 |
| minibatches | 8 |
| minibatch size | 5120 |
| total transitions | 1,024,000 |
| learning rate | 3e-5 |
| BC weight | 1.0 |

Training diagnostics were noisy; later updates sometimes saturated more and
had worse terminal latent distance. Therefore checkpoint selection used the
fixed held-out 512-env local manifest.

Held-out 512-env local checkpoint selection:

| checkpoint | final distance | mean reduction | reduction fraction | action delta | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| frozen BC | 0.6073 | 0.5193 | 0.8301 | 0.0000 | 0.0100 |
| 204800 | 0.5920 | 0.5346 | 0.8438 | 0.0039 | 0.0102 |
| 409600 | 0.6034 | 0.5232 | 0.8457 | 0.0068 | 0.0109 |
| 614400 | 0.5903 | 0.5364 | 0.8457 | 0.0067 | 0.0123 |
| 819200 | 0.6035 | 0.5231 | 0.8340 | 0.0081 | 0.0115 |
| 1024000 | 0.5851 | 0.5415 | 0.8379 | 0.0093 | 0.0096 |

The best local checkpoint is the final checkpoint. It improves final latent
distance by `0.0222` absolute (`3.65%`) relative to the frozen deterministic
low level, below the R1/R3 local gate target but clearly better than R1/R2.

Closed-loop paired evaluations on 100 deployment seeds:

| checkpoint | frozen success | tuned success | success delta | final reward delta | max reward delta | action delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 204800 | 0.34 | 0.34 | 0.00 | -0.0008 | +0.0012 | 0.0037 |
| 614400 | 0.34 | 0.29 | -0.05 | -0.0346 | -0.0334 | 0.0066 |
| 1024000 | 0.34 | 0.29 | -0.05 | -0.0363 | -0.0347 | 0.0096 |

Conclusion: R3 last-layer tuning gives the strongest local latent-reaching
improvement so far, but too much tuning degrades full-task deployment. The
early `204800` checkpoint is task-neutral and slightly improves max reward,
while later locally better checkpoints hurt success by 5 percentage points.
R3 last-layer with `bc_weight=1.0` does not pass the full closed-loop gate.
Next R3 variants should keep the useful local signal but reduce deployment
drift, for example by increasing BC regularization or using a smaller direct
learning rate and selecting by paired closed-loop performance.

## 2026-06-23 - RR-31: Constrained R3 variants

R3 `lr=3e-5, bc_weight=1.0` showed useful local latent improvement but degraded
closed-loop deployment when action drift grew. Tested two constrained variants.

### Smoke variants

Held-out 512-env local smoke eval:

| variant | final distance | mean reduction | reduction fraction | action delta | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen deterministic low level | 0.6073 | 0.5193 | 0.8301 | 0.0000 | 0.0100 |
| `lr=3e-5, bc=1`, 10k | 0.6020 | 0.5246 | 0.8379 | 0.0030 | 0.0092 |
| `lr=3e-5, bc=10`, 10k | 0.6120 | 0.5146 | 0.8145 | 0.0030 | 0.0096 |
| `lr=1e-5, bc=1`, 10k | 0.6036 | 0.5231 | 0.8281 | 0.0014 | 0.0092 |

Increasing BC weight to `10` over-constrained the update and removed the local
gain. Lowering the direct learning rate to `1e-5` preserved a small local gain
with much lower deterministic action drift in the smoke run, so it was selected
for a serious run.

### Serious lower-LR R3 run

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 --seed 0 --run-name aligned10_n4096_lr1e5_bc1_1m \
  --steps 1024000 --bc-weight 1.0 --terminal-weight 1.0 \
  --learning-rate 0.00001 --num-minibatches 8 --checkpoint-every-updates 5 --force
```

Configuration:

| item | value |
| --- | ---: |
| environments | 4096 |
| rollout horizon | 10 |
| samples/update | 40960 |
| total transitions | 1,024,000 |
| learning rate | 1e-5 |
| BC weight | 1.0 |

Held-out 512-env local checkpoint selection:

| checkpoint | final distance | mean reduction | reduction fraction | action delta | saturation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| frozen BC | 0.6073 | 0.5193 | 0.8301 | 0.0000 | 0.0100 |
| 204800 | 0.6040 | 0.5226 | 0.8320 | 0.0036 | 0.0096 |
| 409600 | 0.5950 | 0.5316 | 0.8516 | 0.0052 | 0.0102 |
| 614400 | 0.5957 | 0.5310 | 0.8379 | 0.0053 | 0.0090 |
| 819200 | 0.5932 | 0.5334 | 0.8301 | 0.0061 | 0.0104 |
| 1024000 | 0.5996 | 0.5270 | 0.8359 | 0.0062 | 0.0096 |

This variant gives a smaller local gain than `lr=3e-5`, but keeps action drift
lower. The locally best final-distance checkpoint is `819200`; the best
reduction-fraction checkpoint is `409600`.

Closed-loop paired evaluations on 100 deployment seeds:

| run | checkpoint | frozen success | tuned success | success delta | final reward delta | max reward delta | action delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr=3e-5, bc=1` | 204800 | 0.34 | 0.34 | 0.00 | -0.0008 | +0.0012 | 0.0037 |
| `lr=3e-5, bc=1` | 614400 | 0.34 | 0.29 | -0.05 | -0.0346 | -0.0334 | 0.0066 |
| `lr=3e-5, bc=1` | 1024000 | 0.34 | 0.29 | -0.05 | -0.0363 | -0.0347 | 0.0096 |
| `lr=1e-5, bc=1` | 409600 | 0.34 | 0.38 | +0.04 | +0.0307 | +0.0315 | 0.0045 |
| `lr=1e-5, bc=1` | 819200 | 0.34 | 0.38 | +0.04 | +0.0272 | +0.0250 | 0.0054 |

Conclusion: the lower-LR R3 variant is the first positive closed-loop low-level
RL result. It improves paired success by 4 percentage points and improves final
and max reward, while keeping deterministic action drift near `0.005`. This
still does not reach the original `+10` point full-hierarchy gate, but it is a
meaningful positive signal and suggests direct last-layer tuning is more useful
than residual R1 or residual-flow R2 for this setup.

## 2026-06-23 - RR-32: N=1000 R3 smoke screen

The rerun plan asks for checking both `N=500` and `N=1000`. Before launching a
full `4096 x 10` R3 run at `N=1000`, screened the two R3 variants that were
most informative at `N=500` on the cheap 512-env local manifest.

Commands:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml local-mode-a-audit \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5 \
  --n-demo 1000 --seed 0 --episodes 1 \
  --manifest results/rl_rerun/local_eval_manifest_n512_val_b1_seed20260623.json \
  --output results/rl_rerun/local_mode_a_audit_n512_val_b1_n1000_seed0_manifest.json

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 \
  --n-demo 1000 --seed 0 --run-name smoke_10k_lr1e5_bc1 \
  --steps 10240 --bc-weight 1.0 --terminal-weight 1.0 \
  --learning-rate 0.00001 --num-minibatches 1 --checkpoint-every-updates 1 --force

uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 \
  --n-demo 1000 --seed 0 --run-name smoke_10k_lr3e5_bc1 \
  --steps 10240 --bc-weight 1.0 --terminal-weight 1.0 \
  --learning-rate 0.00003 --num-minibatches 1 --checkpoint-every-updates 1 --force
```

Held-out 512-env local results:

| policy | final distance | mean reduction | reduction fraction | action delta | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `N=1000` frozen deterministic low level | 1.1175 | 0.3906 | 0.8359 | 0.0000 | 0.0152 |
| `N=1000` R3, `lr=1e-5`, 10k | 1.1249 | 0.3832 | 0.8184 | 0.0011 | 0.0141 |
| `N=1000` R3, `lr=3e-5`, 10k | 1.1193 | 0.3888 | 0.8223 | 0.0029 | 0.0137 |

Both `N=1000` R3 smokes are locally worse than the frozen `N=1000` low level.
Decision: do not spend a full 4096-env 1M-transition run on these exact
`N=1000` R3 settings. The positive low-level RL signal currently exists only
for `N=500`, lower-LR direct last-layer tuning.

## 2026-06-24 - RR-33: Failure-video deliverable

Added rerun-specific video recording for the exact policy path used in the
closed-loop R1/R2/R3 evaluations:

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml record-videos \
  --checkpoint artifacts/rl_rerun/local_r3/n500/seed0/aligned10_n4096_lr1e5_bc1_1m/checkpoints/step_000409600.pt \
  --n-demo 500 --seed 0 --episodes 6 --eval-seed-start 10000 \
  --mode both --output-dir rl_rerun_failure_videos --force
```

Generated 12 videos:

```text
rl_rerun_failure_videos/frozen/seed10000_step409600_success1_final1.000_max1.000.mp4
rl_rerun_failure_videos/frozen/seed10001_step409600_success1_final1.000_max1.000.mp4
rl_rerun_failure_videos/frozen/seed10002_step409600_success1_final1.000_max1.000.mp4
rl_rerun_failure_videos/frozen/seed10003_step409600_success0_final0.258_max0.265.mp4
rl_rerun_failure_videos/frozen/seed10004_step409600_success0_final0.262_max0.323.mp4
rl_rerun_failure_videos/frozen/seed10005_step409600_success0_final0.182_max0.224.mp4
rl_rerun_failure_videos/tuned/seed10000_step409600_success1_final1.000_max1.000.mp4
rl_rerun_failure_videos/tuned/seed10001_step409600_success0_final0.326_max0.329.mp4
rl_rerun_failure_videos/tuned/seed10002_step409600_success1_final1.000_max1.000.mp4
rl_rerun_failure_videos/tuned/seed10003_step409600_success0_final0.028_max0.093.mp4
rl_rerun_failure_videos/tuned/seed10004_step409600_success1_final1.000_max1.000.mp4
rl_rerun_failure_videos/tuned/seed10005_step409600_success0_final0.201_max0.225.mp4
```

The set intentionally includes both successes and failures for frozen and tuned
policies, so the action-level qualitative differences can be inspected directly.

## 2026-06-24 - RR-34: R3 lower-LR seed1 confirmation

The best seed0 R3 setting was a small direct update:

```text
R3 direct last-layer, N=500, lr=1e-5, bc_weight=1.0
```

To check whether the seed0 result was just a single-policy-seed artifact, ran
the same serious `4096 x 10` local PPO setup for policy seed1.

```text
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r3 \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --n-demo 500 --seed 1 --run-name aligned10_n4096_lr1e5_bc1_1m \
  --steps 1024000 --bc-weight 1.0 --terminal-weight 1.0 \
  --learning-rate 0.00001 --num-minibatches 8 --checkpoint-every-updates 5 --force
```

Configuration:

| item | value |
| --- | ---: |
| environments | 4096 |
| rollout horizon | 10 |
| samples/update | 40960 |
| total transitions | 1,024,000 |
| learning rate | 1e-5 |
| BC weight | 1.0 |

Before the serious run, the cheap 10k local smoke screen was:

| seed | frozen final distance | 10k tuned final distance | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.6073 | 0.6036 | -0.0037 |
| 1 | 0.6299 | 0.6247 | -0.0051 |
| 2 | 0.6836 | 0.6913 | +0.0078 |

Seed1 passed the same cheap final-distance screen as seed0. Seed2 did not, so
it has not been promoted to a serious `4096`-env run yet.

Held-out 512-env local checkpoint selection for seed1:

| checkpoint | final distance | reduction fraction | action delta |
| ---: | ---: | ---: | ---: |
| frozen BC | 0.6299 | 0.8379 | 0.0000 |
| 204800 | 0.6310 | 0.8203 | 0.0026 |
| 409600 | 0.6342 | 0.8379 | 0.0049 |
| 614400 | 0.6171 | 0.8555 | 0.0045 |
| 819200 | 0.6344 | 0.8359 | 0.0055 |
| 1024000 | 0.6226 | 0.8477 | 0.0055 |

The local-selected checkpoint is `614400`.

Closed-loop paired evaluation on 100 deployment seeds:

| seed | checkpoint | frozen success | tuned success | success delta | final reward delta | max reward delta | action delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.34 | 0.38 | +0.04 | +0.0307 | +0.0315 | 0.0045 |
| 1 | 614400 | 0.39 | 0.40 | +0.01 | +0.0224 | +0.0156 | 0.0043 |

Conclusion: the lower-LR R3 update remains positive on a second policy seed, but
the deployment gain is smaller. This supports the direction of the result while
also weakening the claim: current evidence is a modest, not gate-passing,
improvement. A final multi-seed claim would still need the third policy seed
and a larger evaluation budget, but seed2 failed the cheap local final-distance
screen and should not be promoted automatically without a new reason.
