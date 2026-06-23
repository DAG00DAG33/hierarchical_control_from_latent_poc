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
