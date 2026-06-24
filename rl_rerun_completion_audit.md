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
large. After the two 100-episode development-bank R3 signals, one selected
checkpoint was evaluated on a fresh 500-episode clean bank. That fresh bank was
negative, so a full multi-seed positive evaluation was not pursued.

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
| Fresh 500-episode clean evaluation | fail for selected R3 seed0 | frozen `0.306`, tuned `0.282`, delta `-0.024` on seeds `20000-20499` |
| Final multi-seed 500-episode evaluation | stopped after negative fresh bank | two serious R3 seeds evaluated on 100 dev episodes; selected seed0 failed fresh 500-bank eval; seed2 failed cheap screen |
| Disturbed/recovery/branch-oracle evaluations | not run | omitted under runtime reduction; no final gate claim is made |
| RL wall-clock and GPU telemetry | implemented for future runs | R1/R2/R3 history writers now record update/run wall time, sample rates, and peak CUDA memory; verified by `telemetry_smoke_1update` |

## Main Quantitative Result

Best development-bank method: R3 direct deterministic last-layer tuning,
`N=500`, `lr=1e-5`, `bc_weight=1.0`.

| Policy seed | Selected checkpoint | Frozen success | Tuned success | Delta | Episodes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.34 | 0.38 | +0.04 | 100 |
| 1 | 614400 | 0.39 | 0.40 | +0.01 | 100 |

Fresh-bank check:

| Policy seed | Selected checkpoint | Eval seeds | Frozen success | Tuned success | Delta | Episodes |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 409600 | `20000-20499` | 0.306 | 0.282 | -0.024 | 500 |

The fresh-bank result is negative for the selected method.

## Negative/Positive Claim Status

Strong positive claim: **not supported**.

Reason: R3 improves two 100-episode development banks, but the selected seed0
checkpoint loses `2.4` percentage points on a fresh 500-episode bank.

Strong negative claim: **not supported**.

Reason: the plan's credible-negative conditions are mostly satisfied for local
R1/R2/R3, but N=1000 was only screened cheaply and disturbed/recovery/branch
oracle evaluations were not run.

Supported conclusion:

> Exact local resets and large vector batches invalidate the earlier weak RL
> run as a definitive negative. Residual R1/R2 did not solve the problem. A
> small direct update to the deterministic low-level final layer gives better
> local latent reaching and small development-bank positives, but the selected
> checkpoint does not improve fresh closed-loop deployment.

## Remaining Work For A Final Claim

1. Do not claim the current R3 setting as positive; design a new method or
   checkpoint-selection rule before spending a full multi-seed final bank.
2. Re-run any serious RL point whose final report needs wall-clock and GPU
   memory, because telemetry is now implemented but old histories do not contain
   retrospective measurements.
3. Decide whether to spend a full run on seed2 despite its failed cheap local
   screen, or explicitly stop at two-seed development evidence.
4. If pursuing a negative claim, run the omitted disturbed, recovery-state, and
   branch-oracle evaluations.
5. If pursuing N=1000, design a new R3/R2 setting because the current cheap
   R3 screen was locally worse than frozen.
