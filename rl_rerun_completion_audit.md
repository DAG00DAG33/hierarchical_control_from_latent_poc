# RL Rerun Completion Audit

Date: 2026-06-24

This file audits the current repository state against
`low_level_rl_rerun_state_parallel_plan.md` and the later runtime reduction
decisions from the experiment thread. It is intentionally stricter than the
compact final report: it separates completed evidence from skipped or weakened
requirements.

## Scope Decision

The original plan asks for 3 policy seeds and 500 clean plus 500 disturbed
episodes per final seed. During execution this was reduced because runtime was
large. The current result should therefore be read as a serious development
result, not a final statistically powered claim.

## Required Deliverables

| Deliverable | Status | Evidence |
| --- | --- | --- |
| `rl_rerun_state_dataset_spec.md` | done | dataset schema and reset-and-replay procedure documented |
| `rl_rerun_state_load_audit.md` | done | full dataset reset-and-replay audit summarized |
| `rl_rerun_throughput_benchmark.csv` | done | throughput benchmark exported at repo root |
| `rl_rerun_algorithm_audit.md` | done | Phase D algorithm audit documented |
| `rl_rerun_experiment_log.md` | done | chronological log through RR-35 |
| `rl_rerun_final_results.md` | done | compact final result report |
| `rl_rerun_learning_curves.png` | done | summary plot exported |
| `rl_rerun_failure_videos/` | done | paired frozen/tuned videos for the best R3 seed0 checkpoint |

## Gate Evidence

| Requirement | Status | Evidence |
| --- | --- | --- |
| State-loadable teacher data | pass | `data/rl_rerun/pusht_state_demos.h5`, 1200 successful teacher trajectories |
| Exact reset/replay on single-env corpus | pass | `results/rl_rerun/state_load_audit.json`: 1000 samples, horizon 10, replay state error `0.0`, reward error `0.0`, success mismatches `0` |
| Exact vector local reset corpus | pass | `results/rl_rerun/vector_state_audit_n4096_b2.json`: `4096` envs, horizon `10`, current/goal state and observation errors `0.0` |
| Throughput large enough for serious PPO | pass | selected `4096 x 10 = 40960` samples/update; `8192` failed camera allocation |
| Clean local RL reward | pass | final report and code path use latent progress/terminal distance only; no task reward, success, object pose, or task progress in training |
| Local rollout length matches subgoal horizon | pass | algorithm audit and histories use `10` steps |
| GAE terminates at local segment boundary | pass | `results/rl_rerun/algorithm_audit.json`: `gate_pass=true`, hand-computed GAE max error `0.0`, terminal step does not bootstrap |
| Frozen/zero residual equivalence | pass | `zero_residual_max_abs_action_error=0.0` in algorithm audit |
| R1 residual deterministic tested | pass | serious `4096`-env N=500 R1 run, 1.024M transitions |
| R2 residual flow tested | pass | serious `4096`-env N=500 R2 run, 1.024M transitions |
| Direct deterministic fine-tuning tested | pass | R3 `lr=3e-5` and `lr=1e-5` serious N=500 runs |
| N=1000 checked | partial | cheap exact-reset R3 screen failed; full 4096-env N=1000 run skipped |
| R4 direct-flow tested | skipped by plan condition | R2 did not establish a stable flow base; final report documents this |
| Final multi-seed 500-episode evaluation | partial | two serious R3 seeds evaluated on 100 episodes; seed2 failed cheap screen; no 500-episode final bank |
| Disturbed/recovery/branch-oracle evaluations | not run | omitted under runtime reduction; no final gate claim is made |

## Main Quantitative Result

Best method: R3 direct deterministic last-layer tuning, `N=500`,
`lr=1e-5`, `bc_weight=1.0`.

| Policy seed | Selected checkpoint | Frozen success | Tuned success | Delta | Episodes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.34 | 0.38 | +0.04 | 100 |
| 1 | 614400 | 0.39 | 0.40 | +0.01 | 100 |

The result is positive but below the original `+0.10` success gate.

## Negative/Positive Claim Status

Strong positive claim: **not supported**.

Reason: R3 improves two development seeds, but the success gains are only
`+0.04` and `+0.01`, and no 500-episode final bank was run.

Strong negative claim: **not supported**.

Reason: the plan's credible-negative conditions are mostly satisfied for local
R1/R2/R3, but N=1000 was only screened cheaply and disturbed/recovery/branch
oracle evaluations were not run.

Supported conclusion:

> Exact local resets and large vector batches invalidate the earlier weak RL
> run as a definitive negative. Residual R1/R2 did not solve the problem. A
> small direct update to the deterministic low-level final layer gives a modest
> positive deployment signal at N=500, but the result is below gate and should
> be treated as preliminary.

## Remaining Work For A Final Claim

1. Run a final 500-episode evaluation for any method intended to be claimed as
   positive.
2. Add wall-clock and GPU-memory telemetry to serious RL histories.
3. Decide whether to spend a full run on seed2 despite its failed cheap local
   screen, or explicitly stop at two-seed development evidence.
4. If pursuing a negative claim, run the omitted disturbed, recovery-state, and
   branch-oracle evaluations.
5. If pursuing N=1000, design a new R3/R2 setting because the current cheap
   R3 screen was locally worse than frozen.
