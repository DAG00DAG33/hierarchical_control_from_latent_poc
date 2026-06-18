# Push-T Incremental Experiment Log

This is the running lab notebook for
[`pusht_incremental_experiment_plan.md`](pusht_incremental_experiment_plan.md).
It is updated after every meaningful experiment, including failed runs.

## Logging Rules

Every experiment entry records:

- date and experiment ID;
- Git commit and configuration;
- hypothesis;
- dataset type: `query_dataset` or `causal_dataset`;
- dataset size in trajectories and transitions/state queries;
- exact command;
- training and evaluation seeds;
- success and secondary KPIs;
- gate decision;
- diagnosis and next action.

State-query labels are never treated as causal transitions. Any experiment
using action chunks, future states, or world-model targets must explicitly
identify its causal rollout source.

## Current Status

| Item | Status |
| --- | --- |
| Active phase | Phase 0: teacher and pipeline sanity |
| Gate | Copied teacher policy matches PPO success within 1 percentage point |
| Gate state | Not yet evaluated |
| Current blocker | None |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Free disk at start | 113 GB |
| Canonical controller | `pd_ee_delta_pos`, 20 Hz |
| Canonical final evaluation | 500 episodes, fixed reset seeds |

## Prior Evidence

The previous end-to-end experiment established useful diagnostics but does not
satisfy the new phase gates:

- privileged PPO deterministic evaluation previously measured about 86.3%
  success over 256 episodes;
- the PPO checkpoint records a 92.5% recent training success metric, which is
  not interchangeable with deterministic held-out evaluation;
- the old `bc_state` result reached about 46% success;
- old `bc_state` is not a valid Phase 1 implementation because it uses only
  successful deterministic teacher rollouts and predicts an 8-step action
  chunk;
- visual and hierarchical policies reached only a few percent success;
- the final 512D reconstruction-regularized latent passed the static pose probe
  but was not validated for velocity, contact, inverse dynamics, or the new
  oracle-hierarchy gate.

These results are baselines and debugging evidence only. They will not be used
to skip phases.

## Phase 0 Checklist

| Check | Evidence required | Status |
| --- | --- | --- |
| 0.1 Original PPO evaluation | Deterministic success and reward on canonical seeds | Pending |
| 0.1 Downstream evaluator parity | Same actor through scalar student evaluation path | Pending |
| 0.2 Copied actor | Exact weight equality and success within 1 percentage point | Pending |
| 0.3 Action semantics | Raw, clipped, stored, executed, normalized comparisons | Pending |
| 0.4 Temporal alignment | `s_t, a_t, s_{t+1}` audit and +/-1 action comparison | Pending |
| 0.5 One-state overfit | Nearly zero normalized action error | Pending |
| 0.5 One-trajectory overfit | Nearly zero normalized action error and replay check | Pending |
| 0.5 Ten-trajectory overfit | Training error and closed-loop behavior | Pending |

## Experiment Entries

### 2026-06-18 - P0-I00: Repository and resource inventory

- **Git base:** `b03fa2c`
- **Configuration inspected:** `configs/pusht.yaml`
- **Hypothesis:** Existing artifacts are sufficient to begin Phase 0 without
  retraining the PPO teacher.
- **Dataset type:** None.
- **Commands:** `nvidia-smi`, `df -h`, `hcl-poc rl status`, source and artifact
  inventory.
- **Evidence:**
  - CUDA is available.
  - GPU has approximately 15.4 GB free.
  - Disk has 113 GB free.
  - `artifacts/rl_pusht_official/ppo_best.pt` exists.
  - PPO checkpoint observation/action dimensions and exact evaluation behavior
    still need runtime verification.
  - Existing prepared visual HDF5 files do not store full privileged state or
    next state, so they cannot by themselves support the Phase 0 alignment
    audit or later causal privileged experiments.
- **Gate decision:** Phase 0 remains open.
- **Next action:** Implement one canonical Phase 0 command that evaluates the
  PPO actor through both vector and scalar downstream paths, copies actor
  weights into an independent student wrapper, records causal state/action
  transitions, audits alignment, and runs the overfit ladder.

### 2026-06-18 - P0-D01: First Phase 0 debug run

- **Git state:** Uncommitted Phase 0 implementation based on `b03fa2c`.
- **Configuration:** `configs/pusht_incremental.yaml`, 50 evaluation episodes.
- **Hypothesis:** The existing scalar downstream evaluator reproduces the PPO
  vector evaluator.
- **Dataset type:** New `causal_dataset` audit with 10 successful trajectories,
  489 transitions.
- **Command:** `uv run hcl-poc incremental phase0 --config
  configs/pusht_incremental.yaml --episodes 50 --force`
- **Results:**
  - copied actor output max error: exactly 0;
  - copied actor and teacher scalar success: both 46%;
  - stored versus queried clipped action MAE at shift 0:
    `6.86e-8`;
  - shift -1/+1 MAE: both approximately `0.173`;
  - exact simulator transition replay max state error: 0;
  - raw teacher actions outside bounds: 21.9% of audit transitions;
  - clipped versus executed action MAE: 0;
  - one-state normalized MAE: `9.76e-4`;
  - one-trajectory normalized MAE: `9.88e-4`;
  - one-trajectory closed-loop replay success: 0/1;
  - ten-trajectory training-initialization success: 2/10.
- **Gate decision:** Failed. The initial implementation incorrectly marked the
  gate as passed because it compared only teacher/student equality and omitted
  the known-teacher-performance requirement.
- **Diagnosis:** State/action alignment and causal replay are correct, but the
  scalar evaluator used `physx_cpu`, while PPO was trained and evaluated with
  `physx_cuda`.
- **Next action:** Compare the same scalar actor on CPU and CUDA, correct the
  gate, and make CUDA the canonical downstream backend if the discrepancy is
  confirmed.

### 2026-06-18 - P0-D02: Simulator-backend isolation

- **Hypothesis:** The teacher-success collapse is caused by CPU versus CUDA
  simulation dynamics rather than reset seeds or policy wrapping.
- **Dataset type:** None.
- **Commands:** Scalar teacher evaluation for 100 episodes on CPU and CUDA,
  with seed starts 10000 and 50000; vector PPO evaluation for 500 episodes.
- **Results:**
  - canonical vector CUDA evaluation: 83.8% success over 500 episodes;
  - scalar CUDA evaluation: 83% success over 100 episodes;
  - scalar CPU evaluation: 46% over seeds starting at 10000;
  - scalar CPU evaluation: 40% over seeds starting at 50000.
- **Gate decision:** CPU downstream evaluation is invalid for this experiment.
- **Diagnosis:** The same bit-exact actor loses roughly 40 percentage points
  when the simulator backend changes to CPU. Reset seed range does not explain
  the discrepancy.
- **Next action:** Use explicit CUDA simulation for all canonical policy
  evaluation and causal collection. Require downstream teacher success of at
  least 80%, vector/scalar agreement within 5 percentage points, and copied
  actor agreement within 1 percentage point.

### Plan amendment - Phase 6

Phase 6 now includes a reconstruction-only autoencoder with zero world-model
prediction loss. This will isolate whether the action-conditioned temporal
objective adds useful control-state structure beyond current-observation
reconstruction.
