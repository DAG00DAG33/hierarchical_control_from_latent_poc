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
checkpoint per serious R3 seed was evaluated on a fresh 500-episode clean bank.
The same two checkpoints were also evaluated on a fresh 500-episode disturbed
bank. Both final-budget averages were non-positive, so a full positive
evaluation was not pursued.

## Required Deliverables

| Deliverable | Status | Evidence |
| --- | --- | --- |
| `rl_rerun_state_dataset_spec.md` | done | dataset schema and reset-and-replay procedure documented |
| `rl_rerun_state_load_audit.md` | done | full dataset reset-and-replay audit summarized |
| `rl_rerun_throughput_benchmark.csv` | done | throughput benchmark exported at repo root |
| `rl_rerun_algorithm_audit.md` | done | Phase D algorithm audit documented |
| `rl_rerun_experiment_log.md` | done | chronological log through RR-46 |
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
| Fresh 500-episode clean evaluation | no robust gain | seed0 delta `-0.024`, seed1 delta `+0.020`, mean delta `-0.002` on seeds `20000-20499` |
| Final multi-seed 500-episode evaluation | stopped after near-zero two-seed result | two serious R3 seeds evaluated on fresh 500-episode banks; seed2 failed cheap screen |
| Disturbed/recovery evaluation | fail at final budget | 500-episode disturbed mean success delta `-0.014`, recovery delta `-0.015`; 100-episode diagnostic was optimistic |
| Branch-oracle evaluation | bounded diagnostic only | 100-episode replay-oracle checks for two selected R3 seeds: deltas `+0.01` and `+0.02`, mean `+0.015`; replay state error `0.0`; no 500-episode gate claim |
| Matched learned-vs-oracle seed bank | diagnostic only | on eval seeds `50000-50099`, tuned learned-goal mean success `0.330` versus tuned replay-oracle mean success `0.380`; suggests high-level goal quality remains a bottleneck |
| Learned-vs-oracle goal mismatch | diagnostic only | 20-episode audits show mean future-goal L2 `25.02`, but tuned low-level action changes only `0.033` L2 when swapping learned goals for oracle goals |
| Goal sensitivity | diagnostic only | tuned action response is `0.00117` L2 per unit latent-goal L2, essentially identical to frozen `0.00118`; R3 did not make the low-level more goal-sensitive |
| Same-state valid-goal sensitivity | diagnostic only | same-trajectory `k=9/10/11` future latents differ by mean L2 `~16`, but frozen/R3 actions change only `~0.0085` L2; RR-48 sensitivity hinge does not improve this metric |
| Fast state-dict branch copy | fail for oracle use | current state/action parity is near `1e-6`, but 10-step same-action branch rollout still has mean future-goal L2 error `5.19`; replay remains the trusted oracle path |
| RL wall-clock telemetry | partial, retrospectively recovered where logs exist | `rl_rerun_recovered_wallclock.csv` recovers raw-log wall-clock for R2/R3 serious runs; R1 raw logs are missing and old histories have no timing fields |
| RL GPU-memory telemetry | implemented for future runs | R1/R2/R3 history writers now record peak CUDA memory; verified by `telemetry_smoke_1update`, but old serious runs do not contain retrospective memory traces |

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
| 1 | 614400 | `20000-20499` | 0.296 | 0.316 | +0.020 | 500 |
| mean | n/a | `20000-20499` | 0.301 | 0.299 | -0.002 | 500 each |

The fresh-bank result is mixed and averages to no improvement.

Disturbed/recovery diagnostic:

| Policy seed | Selected checkpoint | Frozen success | Tuned success | Delta | Frozen recovery | Tuned recovery | Recovery delta | Episodes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.32 | 0.33 | +0.01 | 0.29 | 0.31 | +0.02 | 100 |
| 1 | 614400 | 0.30 | 0.35 | +0.05 | 0.29 | 0.32 | +0.03 | 100 |
| mean | n/a | 0.31 | 0.34 | +0.03 | 0.29 | 0.315 | +0.025 | 100 each |

Fresh disturbed/recovery check:

| Policy seed | Selected checkpoint | Frozen success | Tuned success | Delta | Frozen recovery | Tuned recovery | Recovery delta | Episodes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.292 | 0.260 | -0.032 | 0.284 | 0.254 | -0.030 | 500 |
| 1 | 614400 | 0.254 | 0.258 | +0.004 | 0.250 | 0.250 | 0.000 | 500 |
| mean | n/a | 0.273 | 0.259 | -0.014 | 0.267 | 0.252 | -0.015 | 500 each |

## Negative/Positive Claim Status

Strong positive claim: **not supported**.

Reason: R3 improves two 100-episode development banks, but the two selected
checkpoints average `-0.2` percentage points on fresh 500-episode banks.

Strong negative claim: **not supported**.

Reason: the plan's credible-negative conditions are mostly satisfied for local
R1/R2/R3, but N=1000 was only screened cheaply and branch-oracle evaluation was
only run as a bounded 100-episode replay diagnostic.

Supported conclusion:

> Exact local resets and large vector batches invalidate the earlier weak RL
> run as a definitive negative. Residual R1/R2 did not solve the problem. A
> small direct update to the deterministic low-level final layer gives better
> local latent reaching and small development-bank positives, but it does not
> improve fresh clean or disturbed closed-loop deployment on average across the
> two serious seeds.

Additional diagnostic conclusion:

> On a matched 100-episode seed bank, replay-oracle goals outperform learned
> high-level goals for the tuned R3 low level (`0.380` versus `0.330` success),
> so high-level goal quality or compounding goal error is part of the remaining
> bottleneck. However, the learned-versus-oracle goal mismatch audit shows that
> large latent-goal errors only weakly affect low-level actions (`0.00117`
> action-L2 per latent-goal-L2), and R3 tuning does not improve this sensitivity
> over frozen. The current learned-interface bottleneck is therefore both
> high-level goal prediction and weak useful goal conditioning in the low level.

## Remaining Work For A Final Claim

1. Do not claim the current R3 setting as positive; design a new method or
   checkpoint-selection rule before spending a full multi-seed final bank.
2. Re-run any serious RL point whose final report needs wall-clock and GPU
   memory, because telemetry is now implemented but old histories do not contain
   retrospective measurements.
3. Decide whether a new method is needed before spending a full run on seed2,
   since seed2 already failed the cheap local screen and the two fresh-bank
   seeds average to no gain.
4. If pursuing a negative claim, run the branch-oracle evaluation at final
   budget; the current 100-episode replay diagnostic is not enough.
5. If pursuing N=1000, design a new R3/R2 setting because the current cheap
   R3 screen was locally worse than frozen.
6. If pursuing another learned-interface attempt, prioritize stronger
   goal-conditioned low-level training or architecture changes that measurably
   increase valid-goal sensitivity before spending large deployment budgets.
   RR-48 tried a first simple in-batch goal-swap sensitivity hinge; it did not
   reach the action-sensitivity margin and worsened the cheap local metric, so
   that exact setting should not be promoted to a serious run. RR-49 then
   confirmed on same-state `k=9/10/11` valid goals that the hinge did not
   improve the actual nearby-goal sensitivity bottleneck.
