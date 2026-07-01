# RL Reachability Debug Experiment Log

This is the running log for `rl_reachability_debug_plan.md`.

## 2026-06-30 - Goal Activation

Hypothesis:

The previous visual/VAE scratch PPO failures are not enough to conclude that
reachability-based RL is impossible. The next experiments should start with
mechanics audits and the simplest local goal-reaching MDP before returning to
full visual/VAE PPO.

Plan source:

`rl_reachability_debug_plan.md`

Execution status:

Starting with Run 1, PPO mechanics audit.

## 2026-06-30 - Run 1: PPO Mechanics Audit

Hypothesis:

Before running more RL, verify that the low-level rollout semantics match the
intended local MDP: current observations update every primitive step, held
goals stay fixed for the 10-step segment, previous action and remaining time
enter the policy input, and local terminal GAE does not bootstrap into the next
goal segment.

Command:

```bash
uv run python scripts/rl_reachability_mechanics_audit.py \
  --config configs/pusht_incremental.yaml \
  --n-demo 1800 \
  --seed 0 \
  --num-envs 32 \
  --output results/incremental/rl_reachability_debug/run1_mechanics_audit.json
```

Dataset/reset bank:

- VAE512 deterministic hierarchy, `vae512_w2048_b1e6`
- `N_high = 1800`, seed `0`
- 32 vectorized visual envs from `seed_start=3900000`
- one held-goal local segment of length 10

Input update audit:

| Perturbation | Mean action L2 vs live | Max action L2 vs live |
| --- | ---: | ---: |
| cached start observation | 0.5641 | 0.8891 |
| cached previous action | 0.0980 | 0.2081 |
| constant remaining time | 0.0061 | 0.0107 |
| shuffled goal | 0.1285 | 0.1682 |
| shuffled observation | 0.5578 | 0.7428 |

Branch terminal raw distance after 10 steps:

| Branch | Terminal raw distance |
| --- | ---: |
| live | 0.9440 |
| cached start observation | 2.8622 |
| cached previous action | 1.0260 |
| constant remaining time | 0.9637 |
| shuffled goal | 1.4906 |
| shuffled observation | 1.6860 |

GAE unit check:

- local terminal returns: `[0.908406675, 0.95535, 1.0]`
- hand-computed local terminal returns: `[0.908406675, 0.95535, 1.0]`
- max absolute error: `0.0`
- truncation/bootstrap variant changes returns: `true`

Interpretation:

Run 1 passes the mechanics gate. The low-level condition uses live current
observations, previous actions, remaining time, and held goals. Observation
shuffling and cached observations have a large effect, goal shuffling has a
clear effect, and local terminal GAE does not bootstrap across the 10-step
goal boundary in the toy check.

Next action:

Proceed to Run 2: privileged/TCP scratch PPO local reaching, unless a separate
issue is found while preparing that environment.

## 2026-06-30 - Run 2: Privileged/TCP Scratch PPO Local Reaching

Hypothesis:

If PPO and the local reward are basically viable, a random policy should learn
the easiest 10-step low-dimensional reaching problem before we try VAE or visual
inputs again.

Setup:

- input: normalized privileged state, normalized TCP endpoint/velocity goal,
  normalized previous action, remaining local time
- policy: random scratch MLP actor-critic
- local horizon: `10`
- envs: `4096`
- PPO updates: `250`
- samples/update: `40960`
- total env steps: `10,240,000`
- optimizer steps: `250 * 3 epochs * 8 minibatches = 6000`
- reward: TCP squared-distance progress plus terminal TCP squared-distance
  penalty
- reset bank: `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5`
- reset replay check: mean live/reference state L2 `0.0`

Commands:

```bash
uv run python -m hcl_poc.cli --config configs/pusht_incremental.yaml \
  rl-rerun collect-vector-data \
  --output data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --num-envs 4096 \
  --batches 2 \
  --max-steps 60 \
  --seed-start 9500000 \
  --no-store-dino

uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --updates 250 \
  --eval-episodes 2 \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 25 \
  --output-dir results/incremental/rl_reachability_debug/run2_privileged_tcp \
  --force
```

Fixed-bank local evaluation:

| Mode | Initial distance | Terminal distance | Reduction | Reach rate | Improved fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial random mean policy | 0.078806 | 0.072764 | 0.006042 | 0.0059 | 0.9784 |
| trained correct goal | 0.078806 | 0.000489 | 0.078317 | 0.9744 | 0.9999 |
| trained shuffled goal | 0.078806 | 0.036897 | 0.041909 | 0.1013 | 0.8141 |

Terminal distance quantiles:

| Mode | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| initial random mean policy | 0.067389 | 0.132611 | 0.198787 |
| trained correct goal | 0.000177 | 0.000807 | 0.007111 |
| trained shuffled goal | 0.023942 | 0.092444 | 0.158354 |

PPO diagnostics:

| Metric | First update | Last update |
| --- | ---: | ---: |
| global env steps | 40,960 | 10,240,000 |
| mean terminal distance | 0.019954 | 0.000721 |
| reach rate | 0.1631 | 0.9487 |
| policy KL | 0.0047 | 0.0066 |
| clip fraction | 0.0494 | 0.0837 |
| entropy | 1.2528 | -0.7190 |
| value loss | 0.02309 | 0.000041 |
| explained variance | -27.516 | 0.912 |
| NaN count | 0 | 0 |
| action saturation | 0.0201 | 0.4633 |

Interpretation:

Run 2 passes the main local reachability gate. PPO can solve the easy
privileged/TCP local MDP from scratch with large parallel batches and enough
policy iterations. The correct-goal policy is far better than the shuffled-goal
condition on the same reset bank, so the learned controller is using the goal.

The main caveat is action saturation: deterministic eval saturation is about
`0.45`, and the last training update saturation is about `0.46`. This is not an
immediate failure because terminal distance and correct-vs-shuffled separation
are strong, but the next diagnostic should check whether the learned behavior is
mostly bang-bang control and whether a smaller action scale, action penalty, or
residual formulation preserves the reachability gain with less saturation.

Next action:

Proceed to the next gated diagnostic in the plan: random shooting / CEM on the
same privileged/TCP local reset bank, using the trained PPO result as a reference
point. Also consider a short Run 2b ablation with an action penalty or lower
initial/action scale if saturation is judged too high.

## 2026-06-30 - Run 3: Privileged/TCP Random Shooting Pilot

Hypothesis:

On the same easy local MDP, random action-sequence search should reveal whether
there is still meaningful local improvement above the trained PPO controller.

Setup:

- reset bank: same `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5`
- eval references: `1` vector reset reference, `4096` local episodes
- base sequence: deterministic Run 2 PPO policy
- random shooting: Gaussian noise around the PPO 10-step action sequence
- noise std: `0.05`
- candidate counts: `32`, `64`, `128`
- selection metric: terminal TCP squared distance to the same held goal

Command:

```bash
uv run python scripts/rl_reachability_tcp_random_shooting.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --checkpoint results/incremental/rl_reachability_debug/run2_privileged_tcp/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt \
  --eval-refs 1 \
  --random-candidates 32 64 128 \
  --noise-stds 0.05 \
  --output results/incremental/rl_reachability_debug/run3_tcp_random_shooting_pilot.json
```

Results:

| Method | Terminal distance | Reach rate | Improved vs PPO | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO deterministic | 0.000602 | 0.9641 | - | 0.000199 | 0.000918 | 0.010293 |
| shooting, 32 candidates | 0.000110 | 0.9954 | 0.9578 | 0.000023 | 0.000225 | 0.001618 |
| shooting, 64 candidates | 0.000081 | 0.9968 | 0.9780 | 0.000014 | 0.000170 | 0.001076 |
| shooting, 128 candidates | 0.000063 | 0.9980 | 0.9893 | 0.000009 | 0.000126 | 0.000786 |

Interpretation:

Random shooting improves substantially over the already successful PPO policy,
so useful 10-step action sequences do exist and the local objective is not
saturated. PPO is good enough to pass the local sanity gate, but it does not
reach the local action-search upper bound.

This suggests the next question is no longer "can PPO learn reachability at all?"
for privileged/TCP. It can. The next useful diagnostics are:

1. whether saturation can be reduced without losing most reachability;
2. whether CEM improves materially beyond random shooting;
3. whether these action-search branches provide good off-policy examples for
   the planned `D_psi` ensemble.

## 2026-06-30 - Run 4 Preparation: Existing D_phi Checkpoints

Finding:

Existing VAE512 reachability checkpoints are present, including:

- `artifacts/incremental/vae512_scaling/n500/reachability_distance/vae512_w2048_b1e6/seed{0,1,2}/d_phi.pt`
- `artifacts/incremental/vae512_scaling/n1800/reachability_distance/vae512_w2048_b1e6/seed{0,1,2}/d_phi.pt`
- `artifacts/incremental/reachability_distance/vae512_w2048_b1e6/seed0/d_phi.pt`

These are the older demo-temporal single-model `D_phi` checkpoints from
`src/hcl_poc/reachability.py`, not the branch/off-policy ensemble specified in
the new plan.

Existing VAE512 `D_phi` validation summary:

| Dataset | Seed | Temporal MSE | Temporal Spearman | Near/Far acc. | Shuffled AUC | Demo decrease acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N=500 | 0 | 0.01459 | 0.9154 | 0.9827 | 0.8597 | 0.6577 |
| N=500 | 1 | 0.01422 | 0.9137 | 0.9824 | 0.8472 | 0.6558 |
| N=500 | 2 | 0.01520 | 0.9132 | 0.9805 | 0.8598 | 0.6470 |
| N=1800 | 0 | 0.00777 | 0.9328 | 0.9885 | 0.8745 | 0.7048 |
| N=1800 | 1 | 0.00903 | 0.9263 | 0.9854 | 0.8600 | 0.6938 |
| N=1800 | 2 | 0.00880 | 0.9287 | 0.9868 | 0.8641 | 0.6951 |

Interpretation:

These checkpoints are useful baselines and may form an initialization or
comparison point, but they do not satisfy the Run 4 validation gate because they
were not trained or validated on frozen-policy failures, PPO branches,
random-shooting/CEM branches, shuffled invalid goals, or off-policy states.

Next action:

Build a branch dataset for the new `D_psi` ensemble. The Run 3 random-shooting
diagnostic already shows that selected branches are meaningfully better than PPO
branches; the next useful artifact is a saved branch dataset containing start
state/latent, goal, PPO terminal outcome, random-search terminal outcome, and
negative/shuffled goals so the ensemble can be validated on actual rollout
ranking instead of demo temporal distance alone.

## 2026-06-30 - Run 4: Privileged/TCP Branch D_psi Ensemble

Hypothesis:

Before using a learned distance as an RL reward, verify that an ensemble can
rank actual rollout branches on held-out local resets, including PPO branches,
random-search branches, selected best branches, and shuffled-goal negatives.

Scope:

This is a privileged/TCP sanity-check ensemble on the same low-dimensional MDP
as Runs 2-3. It validates the branch/off-policy ensemble machinery, but it is
not yet the final VAE/effect-latent `D_psi` reward model.

Branch dataset:

- path: `data/rl_reachability_debug/run4_tcp_branch_dataset_c64_ref2.npz`
- local episodes: `8192`
- random candidates per episode: `64`
- noise std: `0.05`
- PPO terminal distance mean: `0.000494`
- random candidate terminal distance mean: `0.020927`
- random-search best terminal distance mean: `0.0000759`
- PPO reach rate: `0.9747`
- random-search best reach rate: `0.9983`
- best improved vs PPO: `0.9768`

Dataset command:

```bash
uv run python scripts/rl_reachability_tcp_branch_dataset.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --checkpoint results/incremental/rl_reachability_debug/run2_privileged_tcp/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt \
  --eval-refs 2 \
  --random-candidates 64 \
  --output data/rl_reachability_debug/run4_tcp_branch_dataset_c64_ref2.npz
```

Ensemble:

- members: `3`
- target: `log1p(distance * 1000)`
- train samples: `88679`
- validation samples: `110592`
- validation split: held-out local episodes
- checkpoint: `artifacts/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble/tcp_dpsi_ensemble.pt`

Training command:

```bash
uv run python scripts/rl_reachability_tcp_dpsi_ensemble.py \
  --dataset data/rl_reachability_debug/run4_tcp_branch_dataset_c64_ref2.npz \
  --members 3 \
  --epochs 16 \
  --target-scale 1000 \
  --max-train-candidate-samples 65536 \
  --output results/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble.json \
  --output-dir artifacts/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble
```

Validation gates:

| Gate | Metric | Result |
| --- | ---: | --- |
| correlates with actual terminal distance | Spearman `0.9948` | pass |
| separates reachable vs unreachable branches | AUC `0.9999` | pass |
| ranks random-search selected branch better than PPO | accuracy `0.8999` | pass |
| D_psi-selected candidate improves actual rollout distance | `0.000478 -> 0.000081` | pass |
| D_psi-selected candidate is near oracle random-search best | oracle gap `0.0000046` | pass |
| uncertainty rises on shuffled goals | std ratio `27.20x` | pass |

Candidate-selection comparison on held-out branches:

| Selector | Actual terminal distance |
| --- | ---: |
| PPO branch | 0.000478 |
| D_psi-selected random candidate | 0.000081 |
| oracle best random candidate | 0.000076 |
| random candidate mean | 0.021063 |

Interpretation:

The branch/off-policy ensemble gate passes for the privileged/TCP local MDP.
The learned ensemble is not merely fitting demo temporal distance: it ranks
held-out branch outcomes, selects random candidates that nearly match the oracle
best candidate, separates reachable/unreachable branches, and has much higher
uncertainty on shuffled-goal inputs.

Main caveat:

Because this model sees privileged TCP state and a TCP goal, the metric is much
easier than the intended VAE/effect-latent `D_psi`. It should be treated as a
positive control for the ensemble/data/validation procedure. The next step in
the plan is the low-dimensional PPO-with-learned-distance test; for a strict
VAE/effect reward test, we still need the corresponding visual/latent branch
dataset.

## 2026-06-30 - Run 5: Privileged/TCP PPO with Learned D_psi Reward

Hypothesis:

If the branch-trained `D_psi` ensemble is a useful reward model, PPO trained
from scratch with the learned distance should still solve the local TCP
reachability task, while possibly reducing the action saturation seen in the
true-distance Run 2 PPO.

Command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5 \
  --num-envs 4096 \
  --updates 250 \
  --num-steps 10 \
  --local-horizon 10 \
  --reward-mode progress_terminal \
  --reward-distance-source dpsi \
  --dpsi-checkpoint artifacts/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble/tcp_dpsi_ensemble.pt \
  --terminal-reward-weight 10.0 \
  --progress-reward-weight 5.0 \
  --entropy-coef 0.01 \
  --eval-refs 1 \
  --output-dir results/incremental/rl_reachability_debug/run5_tcp_dpsi_ppo
```

Local reachability result on the fixed evaluation bank:

| Policy | Reward distance | Terminal TCP distance | Reach rate eps=0.0025 | Shuffled-goal reach | Eval action saturation |
| --- | --- | ---: | ---: | ---: | ---: |
| initial random policy | n/a | 0.072764 | 0.0059 | n/a | n/a |
| Run 2 PPO | true TCP | 0.000489 | 0.9744 | 0.1013 | 0.449 |
| Run 5 PPO | learned D_psi | 0.000515 | 0.9725 | 0.0958 | 0.302 |

Additional Run 5 fixed-bank details:

- terminal TCP distance p50/p90/p99: `0.000248 / 0.001096 / 0.004828`
- action L2 mean: `0.756`
- final training-window terminal TCP distance: `0.000910`
- final training-window reach rate: `0.916`
- final training-window action saturation: `0.330`
- PPO updates: `250`
- environment steps: `10,240,000`

Interpretation:

The learned-distance reward preserves the local reachability behavior: it is
nearly tied with the true-distance reward on terminal TCP distance and reach
rate. It also reduces action saturation compared with Run 2, but the local
task is still much easier than the full task because the reset bank starts from
states sampled from successful demonstrations and the goal is only a short TCP
endpoint.

## 2026-06-30 - Run 5b: Full-Task Success Comparison for BC vs RL Low-Level Policies

Question:

Does the RL-trained low-level policy improve full task success when paired with
oracle TCP subgoals or the learned 1800-trajectory privileged high-level
predictor?

Important measurement correction:

An initial comparison incorrectly reported `0.0` success for the 1800-trajectory
BC low-level policy. That was a success-accounting bug in the new evaluator:
it read only live `info["success"]`, while the existing hierarchy evaluators
use `final_info["episode"]["success_once"]`. After matching the existing
success bookkeeping, BC returns to the expected nonzero range.

Policy note:

Run 3 was a random-shooting action-selection diagnostic around the Run 2 PPO
policy, not a separately saved low-level policy. The trainable RL low-level
comparison here is therefore:

- Run 2: true TCP-distance PPO low-level policy
- Run 5: learned `D_psi`-reward PPO low-level policy
- BC baseline: 1800-trajectory privileged TCP low-level policy

Command:

```bash
uv run python scripts/rl_reachability_tcp_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle learned \
  --output results/incremental/rl_reachability_debug/run5_low_level_success_comparison_100_fixed.json
```

Results:

| Goal source | Low-level policy | Success | Final reward | Max reward | Mean length | Hold endpoint error | Teacher action MAE | Action saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle TCP endpoint | BC 1800 | 0.66 | 0.767 | 0.770 | 64.6 | 0.0346 | 0.0461 | 0.083 |
| oracle TCP endpoint | Run 2 true-distance PPO | 0.00 | 0.147 | 0.186 | 100.0 | 0.0553 | 0.2800 | 0.133 |
| oracle TCP endpoint | Run 5 D_psi PPO | 0.00 | 0.152 | 0.193 | 100.0 | 0.0567 | 0.2701 | 0.089 |
| learned high-level endpoint | BC 1800 | 0.65 | 0.758 | 0.761 | 67.4 | 0.0357 | 0.0551 | 0.077 |
| learned high-level endpoint | Run 2 true-distance PPO | 0.00 | 0.141 | 0.185 | 100.0 | 0.0563 | 0.3331 | 0.106 |
| learned high-level endpoint | Run 5 D_psi PPO | 0.01 | 0.155 | 0.194 | 99.2 | 0.0526 | 0.2894 | 0.063 |

For the oracle rows, the learned high-level predictor's endpoint error against
the oracle endpoint was `0.0204 m` when paired with BC, but around
`0.105-0.110 m` on the RL-policy rollouts because those policies move the
system into a different state distribution.

Interpretation:

The RL low-level policies solve the local TCP reachability probe but do not
improve full task success. In fact, they are far worse than the 1800-trajectory
BC low-level policy in closed loop. The likely issue is not local endpoint
reachability in isolation; it is distribution/behavior mismatch. The RL
policies produce actions with much larger teacher-action MAE and keep episodes
alive until timeout, so they can hit short TCP endpoints without following the
task-relevant contact dynamics that the BC policy preserves.

## 2026-06-30 - Run 5c: Local Reachability Comparison for BC vs RL Low-Level Policies

Question:

Is the 1800-trajectory BC low-level policy locally more reachable than the RL
low-level policies on the same reset bank used by Runs 2 and 5?

Command:

```bash
uv run python scripts/rl_reachability_tcp_local_policy_compare.py \
  --eval-refs 1 \
  --include-shuffled \
  --output results/incremental/rl_reachability_debug/run5_local_policy_compare_ref1.json
```

The comparison uses the same 4096-env fixed local reset reference and 10-step
TCP endpoint targets as the PPO local evaluations. Reach uses the same squared
TCP-distance threshold, `eps=0.0025`.

Results:

| Goal condition | Low-level policy | Reach rate | Mean terminal squared distance | Mean TCP error | P90 squared distance | Action saturation | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true goal | BC 1800 | 0.9832 | 0.000258 | 0.0104 m | 0.000470 | 0.189 | 0.0452 |
| true goal | Run 2 true-distance PPO | 0.9597 | 0.000514 | 0.0163 m | 0.000887 | 0.411 | 0.3466 |
| true goal | Run 5 D_psi PPO | 0.9861 | 0.000384 | 0.0164 m | 0.000845 | 0.279 | 0.3084 |
| shuffled goal | BC 1800 | 0.1978 | 0.012417 | 0.0971 m | 0.027450 | 0.167 | 0.1747 |
| shuffled goal | Run 2 true-distance PPO | 0.8662 | 0.001511 | 0.0253 m | 0.003638 | 0.426 | 0.4997 |
| shuffled goal | Run 5 D_psi PPO | 0.8699 | 0.001346 | 0.0248 m | 0.003381 | 0.275 | 0.4474 |

Interpretation:

BC is better on the stricter local reachability diagnostics: it has the lowest
mean endpoint error, the lowest P90 endpoint error, and much lower teacher
action error. Run 5 has a marginally higher threshold reach rate than BC
(`0.9861` vs `0.9832`), but only because the threshold is loose enough that
both policies are near saturation; its mean endpoint error is still worse.

The shuffled-goal rows are the clearest warning sign. BC reach drops from
`0.9832` to `0.1978`, so it is actually using the goal. The RL policies still
reach around `0.87` on shuffled goals, meaning they learned a strong local
default motion from the successful-demonstration reset bank rather than a
robust goal-conditioned controller. This explains why local PPO reachability
does not transfer to full task success.

## 2026-06-30 - Run 6: Larger Reset Bank and Longer D_psi Training

Motivation:

The 250-update runs could be undertrained and the original reset bank had only
two highly correlated vector batches. Run 6 increases both:

- reset bank: `2 x 4096` teacher trajectories -> `8 x 4096`
- valid horizon-10 reset references: `417,792` -> `1,671,168`
- PPO updates: `250` -> `1000`
- environment steps: `10.24M` -> `40.96M`

New reset bank:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml collect-vector-data \
  --output data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --batches 8 \
  --max-steps 60 \
  --seed-start 9600000 \
  --no-store-dino \
  --force
```

Long PPO runs:

| Run | Reward distance | Reward mode | Updates | Env steps | Runtime |
| --- | --- | --- | ---: | ---: | ---: |
| Run 6 true-TCP | true TCP squared distance | progress + terminal | 1000 | 40.96M | 8396s |
| Run 6 D_psi | learned D_psi | progress + terminal | 1000 | 40.96M | 8382s |

Local fixed-bank evaluation:

| Policy | Terminal squared distance | Reach rate | P50 | P90 | P99 | Eval action saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 6 true-TCP | 0.000450 | 0.9760 | 0.000173 | 0.001114 | 0.003813 | 0.321 |
| Run 6 D_psi | 0.000330 | 0.9901 | 0.000209 | 0.000630 | 0.002497 | 0.333 |

Local comparison against the 1800-trajectory BC low level on the same `b8`
bank:

| Goal condition | Low-level policy | Reach rate | Mean terminal squared distance | Mean TCP error | P90 squared distance | P99 squared distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true goal | BC 1800 | 0.9834 | 0.000261 | 0.0105 m | 0.000463 | 0.004142 | 0.0456 |
| true goal | Run 6 true-TCP | 0.9752 | 0.000460 | 0.0175 m | 0.001045 | 0.004014 | 0.3591 |
| true goal | Run 6 D_psi | 0.9895 | 0.000343 | 0.0156 m | 0.000675 | 0.002589 | 0.3498 |
| shuffled goal | BC 1800 | 0.2285 | 0.011329 | 0.0921 m | 0.025965 | 0.063640 | 0.1672 |
| shuffled goal | Run 6 true-TCP | 0.9205 | 0.001013 | 0.0230 m | 0.001987 | 0.014034 | 0.4657 |
| shuffled goal | Run 6 D_psi | 0.9301 | 0.000989 | 0.0215 m | 0.001627 | 0.015300 | 0.4774 |

Full-task success comparison:

| Goal source | Low-level policy | Success | Final reward | Max reward | Mean length | Hold endpoint error | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle TCP endpoint | BC 1800 | 0.64 | 0.753 | 0.756 | 63.9 | 0.0354 | 0.0484 |
| oracle TCP endpoint | Run 6 true-TCP | 0.00 | 0.140 | 0.185 | 100.0 | 0.0567 | 0.3147 |
| oracle TCP endpoint | Run 6 D_psi | 0.00 | 0.152 | 0.202 | 100.0 | 0.0563 | 0.2680 |
| learned high-level endpoint | BC 1800 | 0.62 | 0.740 | 0.742 | 68.2 | 0.0356 | 0.0519 |
| learned high-level endpoint | Run 6 true-TCP | 0.02 | 0.146 | 0.195 | 98.8 | 0.0679 | 0.3658 |
| learned high-level endpoint | Run 6 D_psi | 0.00 | 0.150 | 0.190 | 100.0 | 0.0557 | 0.2925 |

Interpretation:

Longer training and the larger reset bank improve local reachability. The
D_psi run now slightly beats BC on the loose reach threshold and has better
P90/P99 squared distance than the true-TCP run. However, BC is still more
precise in mean TCP error and much closer to the teacher action distribution.

The task-success gap remains large. The RL policies still fail in full
rollouts even with oracle TCP endpoints, so the issue is not only high-level
prediction error. The local shuffled-goal comparison also remains poor for RL:
the policies are still too successful when goals are shuffled, suggesting a
strong default motion on the demonstration reset distribution rather than
robust goal-selective control.

## 2026-06-30 - Run 7: D_psi Reward-Mode Selection

Motivation:

The progress-plus-terminal `D_psi` reward improved local metrics but did not
improve task success. Run 7 tests the reward variants specified in the scratch
RL plan before deciding whether to train any one variant longer:

- terminal-only `D_psi`
- progress-only `D_psi`
- paired terminal advantage over the frozen 1800-trajectory BC low level

Implementation note:

`bc_advantage_terminal` precomputes the frozen BC low-level terminal `D_psi`
distance on the same reset and goal in a branch environment, then gives the RL
policy terminal reward:

```text
D_psi(BC_terminal, goal) - D_psi(RL_terminal, goal)
```

Queue command:

```bash
setsid bash scripts/run_reachability_reward_variants.sh \
  > results/incremental/rl_reachability_debug/run7_reward_variants_queue.nohup.log \
  2>&1 < /dev/null &
```

Startup status:

The queue launched successfully and started the terminal-only variant first.
The variants run sequentially to avoid GPU/simulator contention.

Decision rule before moving on:

If the reward variants plus a longer run of the best variant still give poor
full-task success, do not advance to the next planned experiment. Instead debug
the deployment-distribution reachability gap directly:

1. Measure local reachability to goals generated by the learned high-level
   policy, comparing BC and the best RL low-level policy.
2. Collect reset states from actual deployed hierarchy rollouts rather than
   only from teacher/demo reset banks.
3. On those deployed reset states, evaluate whether the RL low level reaches
   the high-level goals better than BC.
4. Split the measurement by goal source: oracle TCP endpoint, learned high
   endpoint, and shuffled learned endpoint.

This is required because Run 6 already shows that high local reachability on
teacher/demo reset banks can coexist with near-zero task success.

Prepared debug tool:

`scripts/rl_reachability_deployment_reachability_eval.py` implements the first
version of this deployment-distribution check. It deploys one or more collector
hierarchies, snapshots states at learned high-level replanning points, then
branch-evaluates candidate low-level policies on the learned high-level TCP
endpoint goals from those actual rollout states. It reports true-goal and
shuffled-goal reachability for BC, true-distance PPO, and D_psi PPO policies.

Default intended use after Run 7:

```bash
uv run python scripts/rl_reachability_deployment_reachability_eval.py \
  --collector-policies bc1800 run5_dpsi_ppo \
  --candidate-policies bc1800 run2_true_tcp_ppo run5_dpsi_ppo \
  --decisions 200 \
  --num-envs 10 \
  --output results/incremental/rl_reachability_debug/run7_deployment_reachability_eval.json
```

Prepared post-training evaluator:

`scripts/summarize_reachability_reward_variants.py` reads the three Run 7
metrics files and ranks variants by local reachability score. The companion
`scripts/run_reachability_reward_variant_evals.sh` then runs local BC-vs-RL
comparisons and full-task success evaluations for terminal-only, progress-only,
and BC-advantage D_psi checkpoints. This should be run only after the detached
Run 7 training queue has completed all three variants.

Run 7 update:

The original detached queue stalled during the terminal-only variant. The
terminal-only process wrote history through update `52/1000` and then stopped
updating files while still consuming CPU/GPU. The last healthy terminal-only
history row was:

- update: `52`
- global step: `2,129,920`
- mean terminal distance: `0.002574`
- local reach rate: `0.6731`
- action saturation: `0.0413`
- NaNs: `0`
- elapsed: `429s`

Because this was a hang rather than a completed training run, terminal-only is
marked unstable/incomplete for now. The process was terminated and the remaining
variants were launched separately with wall-clock timeout guards.

Progress-only D_psi completed successfully:

| Metric | Value |
| --- | ---: |
| updates | 1000 |
| env steps | 40.96M |
| eval terminal squared distance | 0.000339 |
| eval reach rate | 0.9872 |
| eval p50/p90/p99 squared distance | 0.000156 / 0.000806 / 0.002845 |
| eval action saturation | 0.2735 |
| runner shuffled-goal reach | 0.0919 |
| final train terminal squared distance | 0.000201 |
| final train reach rate | 0.9902 |
| elapsed | 8402s |

BC-advantage D_psi was relaunched separately with an `8h` timeout guard:

```bash
timeout 8h uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 1000 \
  --horizon 10 \
  --reward-mode bc_advantage_terminal \
  --reward-distance-source dpsi \
  --dpsi-checkpoint artifacts/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble/tcp_dpsi_ensemble.pt \
  --checkpoint-every-updates 250 \
  --eval-episodes 2 \
  --output-dir results/incremental/rl_reachability_debug/run7_dpsi_bc_advantage_b8_u1000 \
  --force
```

Run 7 final reward-variant results:

| Variant | Status | Updates | Eval terminal sq dist | Eval reach | Eval p90/p99 sq dist | Eval action saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| terminal-only D_psi | incomplete/hung | 52 | 0.002574 train last | 0.6731 train last | n/a | 0.0413 train last |
| progress-only D_psi | complete | 1000 | 0.000339 | 0.9872 | 0.000806 / 0.002845 | 0.2735 |
| BC-advantage terminal D_psi | complete | 1000 | 0.001448 | 0.8599 | 0.003155 / 0.012149 | 0.2909 |

Local reset-bank comparison on the larger b8 reset bank:

| Goal split | Policy | Reach | Terminal sq dist | TCP error | Action saturation |
| --- | --- | ---: | ---: | ---: | ---: |
| true goal | BC 1800 | 0.9834 | 0.000261 | 0.0105 m | 0.2337 |
| true goal | Run 6 true-TCP | 0.9752 | 0.000460 | 0.0175 m | 0.3238 |
| true goal | Run 7 progress D_psi | 0.9849 | 0.000336 | 0.0142 m | 0.2693 |
| shuffled goal | BC 1800 | 0.2285 | 0.011329 | 0.0921 m | 0.1949 |
| shuffled goal | Run 6 true-TCP | 0.9205 | 0.001013 | 0.0230 m | 0.3304 |
| shuffled goal | Run 7 progress D_psi | 0.9237 | 0.000930 | 0.0208 m | 0.2729 |
| true goal | Run 7 BC-advantage D_psi | 0.9059 | 0.001175 | 0.0277 m | 0.2905 |
| shuffled goal | Run 7 BC-advantage D_psi | 0.8356 | 0.002136 | 0.0331 m | 0.3013 |

Full-task success with oracle and learned high-level goals:

| Variant | Goal source | BC 1800 success | Run 6 true-TCP success | RL variant success |
| --- | --- | ---: | ---: | ---: |
| Run 7 progress D_psi eval | oracle TCP endpoint | 0.67 | 0.00 | 0.01 |
| Run 7 progress D_psi eval | learned high endpoint | 0.59 | 0.02 | 0.00 |
| Run 7 BC-advantage eval | oracle TCP endpoint | 0.69 | 0.00 | 0.00 |
| Run 7 BC-advantage eval | learned high endpoint | 0.68 | 0.02 | 0.00 |

The reward variants did not solve task success. Progress-only is the best
completed local reachability variant, but its full-task success remains near
zero. Per the stop rule, the next step was deployment-distribution debugging
instead of advancing to other planned experiments.

Deployment-distribution reachability for Run 7 progress-only:

This diagnostic deploys a collector hierarchy, snapshots learned high-level
replanning states, then branch-evaluates candidate low levels on the learned
high-level TCP endpoint from those actual rollout states. The same branch
rollout also records the environment task reward over the low-level segment.

| Collector | Candidate | Goal | Reach | Terminal sq dist | TCP error | Terminal reward | Max reward |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| BC 1800 | BC 1800 | learned | 0.9561 | 0.000993 | 0.0184 m | 0.2767 | 0.2978 |
| BC 1800 | BC 1800 | shuffled learned | 0.5610 | 0.009021 | 0.0644 m | 0.2706 | 0.2805 |
| BC 1800 | Run 6 true-TCP | learned | 0.9366 | 0.000506 | 0.0165 m | 0.2794 | 0.2934 |
| BC 1800 | Run 6 true-TCP | shuffled learned | 0.8780 | 0.001849 | 0.0239 m | 0.2469 | 0.2716 |
| BC 1800 | Run 7 progress D_psi | learned | 0.9756 | 0.000380 | 0.0144 m | 0.2680 | 0.2795 |
| BC 1800 | Run 7 progress D_psi | shuffled learned | 0.9171 | 0.000739 | 0.0189 m | 0.2432 | 0.2688 |
| Run 7 progress D_psi | BC 1800 | learned | 0.4800 | 0.009509 | 0.0726 m | 0.1533 | 0.1614 |
| Run 7 progress D_psi | BC 1800 | shuffled learned | 0.2500 | 0.019532 | 0.1089 m | 0.1495 | 0.1586 |
| Run 7 progress D_psi | Run 6 true-TCP | learned | 0.8900 | 0.001459 | 0.0276 m | 0.1466 | 0.1534 |
| Run 7 progress D_psi | Run 6 true-TCP | shuffled learned | 0.8000 | 0.003191 | 0.0352 m | 0.1411 | 0.1535 |
| Run 7 progress D_psi | Run 7 progress D_psi | learned | 0.9350 | 0.000792 | 0.0229 m | 0.1479 | 0.1548 |
| Run 7 progress D_psi | Run 7 progress D_psi | shuffled learned | 0.8500 | 0.002816 | 0.0326 m | 0.1431 | 0.1550 |

Interpretation:

The best RL low level can reach learned high-level TCP endpoints from both
BC-collected and RL-collected hierarchy states. However, it also reaches
shuffled learned endpoints nearly as well, so it is still weakly
goal-selective. More importantly, when the collector is the RL hierarchy, all
candidate branch rollouts have much lower task reward than from BC-collected
states. This suggests the RL low level is driving the system into a bad
deployment state distribution even while satisfying short-horizon TCP endpoint
reachability.

The current hypothesis is that TCP endpoint reachability is an insufficient
training target for this low-level controller. The RL policy learns a generic
motion that reaches endpoints but does not preserve the contact/object dynamics
that make the BC/teacher hierarchy solve PushT. Longer endpoint-only PPO is
unlikely to fix task success unless the reward or diagnostic target includes
task-relevant interaction state, stronger goal selectivity, or imitation-style
constraints on the low-level behavior.

## 2026-07-01 - Run 7: Low-Level Input Ablation Audit

Motivation:

The plan requires checking whether the low-level controller is actually using
live observations, previous action, remaining time, and goals. Earlier results
showed near-zero task success despite strong local TCP endpoint reachability,
so the key question is whether the RL policy ignores goals or whether it
follows TCP goals in a task-misaligned way.

Diagnostic:

`scripts/rl_reachability_low_input_ablation.py` reuses the local reset-bank
evaluation but runs each low-level policy under controlled input ablations:

- live inputs
- cached start observation
- cached previous action
- constant remaining time
- shuffled goal
- shuffled observation

For shuffled goals it reports both distance to the original reference goal and
distance to the commanded shuffled goal. This avoids conflating "goal ignored"
with "policy followed the wrong goal."

Command:

```bash
uv run python scripts/rl_reachability_low_input_ablation.py \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --run2-low results/incremental/rl_reachability_debug/run6_true_tcp_b8_u1000/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt \
  --run5-low results/incremental/rl_reachability_debug/run7_dpsi_progress_b8_u1000/privileged_tcp_ppo_progress_n4096_seed0/latest.pt \
  --eval-refs 2 \
  --output results/incremental/rl_reachability_debug/run7_low_input_ablation_b8_ref2.json
```

Results on the same two b8 references used by the local comparison:

| Policy | Ablation | Original-goal reach | Original terminal sq dist | Commanded-goal reach | Commanded terminal sq dist | Action delta from live | Saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BC 1800 | live | 0.9834 | 0.000261 | 0.9834 | 0.000261 | 0.0000 | 0.2337 |
| BC 1800 | cached start observation | 0.0260 | 0.062012 | 0.0260 | 0.062012 | 0.9583 | 0.4874 |
| BC 1800 | cached previous action | 0.7197 | 0.002116 | 0.7197 | 0.002116 | 0.1937 | 0.2348 |
| BC 1800 | constant remaining time | 0.8220 | 0.001693 | 0.8220 | 0.001693 | 0.0854 | 0.1965 |
| BC 1800 | shuffled goal | 0.2975 | 0.013693 | 0.2285 | 0.011329 | 0.3581 | 0.1949 |
| BC 1800 | shuffled observation | 0.0415 | 0.058948 | 0.0415 | 0.058948 | 1.1211 | 0.3068 |
| Run 6 true-TCP | live | 0.9752 | 0.000460 | 0.9752 | 0.000460 | 0.0000 | 0.3238 |
| Run 6 true-TCP | cached start observation | 0.0361 | 0.037900 | 0.0361 | 0.037900 | 1.3250 | 0.8431 |
| Run 6 true-TCP | cached previous action | 0.9192 | 0.000964 | 0.9192 | 0.000964 | 0.2363 | 0.3466 |
| Run 6 true-TCP | constant remaining time | 0.7034 | 0.002140 | 0.7034 | 0.002140 | 0.3467 | 0.3022 |
| Run 6 true-TCP | shuffled goal | 0.0944 | 0.036725 | 0.9205 | 0.001013 | 1.3656 | 0.3304 |
| Run 6 true-TCP | shuffled observation | 0.0260 | 0.074295 | 0.0260 | 0.074295 | 1.6906 | 0.7502 |
| Run 7 progress D_psi | live | 0.9849 | 0.000336 | 0.9849 | 0.000336 | 0.0000 | 0.2693 |
| Run 7 progress D_psi | cached start observation | 0.0375 | 0.027410 | 0.0375 | 0.027410 | 1.0879 | 0.7717 |
| Run 7 progress D_psi | cached previous action | 0.8840 | 0.001133 | 0.8840 | 0.001133 | 0.2116 | 0.2586 |
| Run 7 progress D_psi | constant remaining time | 0.7632 | 0.001880 | 0.7632 | 0.001880 | 0.2540 | 0.2361 |
| Run 7 progress D_psi | shuffled goal | 0.0923 | 0.036822 | 0.9237 | 0.000930 | 1.1665 | 0.2729 |
| Run 7 progress D_psi | shuffled observation | 0.0361 | 0.059667 | 0.0361 | 0.059667 | 1.4048 | 0.6189 |

Interpretation:

The low-level policies do use live observations. Caching the start observation
or shuffling observations collapses reachability for all policies, especially
the RL policies where action saturation rises sharply under bad observations.
Previous action and remaining-time conditioning matter, but less than the
current observation and goal.

The RL policies are not simply goal-ignoring on the local TCP objective. When
given shuffled goals, original-goal reach drops to about `0.09`, while
commanded shuffled-goal reach stays high (`0.92` for Run 6 true-TCP and
`0.924` for Run 7 progress D_psi). This means the RL low levels are strongly
able to chase arbitrary TCP endpoint commands.

That result sharpens the failure diagnosis: endpoint-goal conditioning works
locally, but it is too permissive and task-misaligned. BC is less capable of
reaching arbitrary shuffled endpoints, yet succeeds in the full task. The RL
policies are better short-horizon TCP servos but worse task controllers.
Future RL rewards should therefore include task-relevant object/contact state
or an imitation/action-distribution constraint, rather than only more
endpoint-distance optimization.

## 2026-07-01 - Run 7: Full-Rollout Shuffled Goal Metric

Motivation:

The plan requires full rollout metrics for learned, oracle/replay, and
shuffled goals. Earlier Run 7 full-task evaluations included oracle and learned
TCP endpoints but not shuffled full-rollout goals.

Implementation:

`scripts/rl_reachability_tcp_full_success_eval.py` now supports two additional
goal sources:

- `shuffled_oracle`: oracle TCP endpoints are computed per env, then shuffled
  across envs before being sent to the low level.
- `shuffled_learned`: learned high-level TCP endpoints are shuffled across
  envs before being sent to the low level.

Command:

```bash
uv run python scripts/rl_reachability_tcp_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle learned shuffled_oracle shuffled_learned \
  --run2-low results/incremental/rl_reachability_debug/run6_true_tcp_b8_u1000/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt \
  --run5-low results/incremental/rl_reachability_debug/run7_dpsi_progress_b8_u1000/privileged_tcp_ppo_progress_n4096_seed0/latest.pt \
  --output results/incremental/rl_reachability_debug/run7_progress_full_success_with_shuffled_100.json
```

Results:

| Goal source | Policy | Success | Final reward | Max reward | Mean length | Hold endpoint error | Action saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | BC 1800 | 0.65 | 0.7605 | 0.7627 | 63.5 | 0.0359 | 0.0845 |
| oracle | Run 6 true-TCP | 0.00 | 0.1404 | 0.1850 | 100.0 | 0.0567 | 0.1371 |
| oracle | Run 7 progress D_psi | 0.01 | 0.1476 | 0.1995 | 99.5 | 0.0644 | 0.0620 |
| learned | BC 1800 | 0.63 | 0.7469 | 0.7490 | 68.0 | 0.0336 | 0.0757 |
| learned | Run 6 true-TCP | 0.02 | 0.1466 | 0.1947 | 98.8 | 0.0679 | 0.1369 |
| learned | Run 7 progress D_psi | 0.00 | 0.1442 | 0.1904 | 100.0 | 0.0578 | 0.0456 |
| shuffled oracle | BC 1800 | 0.03 | 0.2076 | 0.2545 | 98.0 | 0.1408 | 0.0872 |
| shuffled oracle | Run 6 true-TCP | 0.00 | 0.1262 | 0.1541 | 100.0 | 0.0618 | 0.1237 |
| shuffled oracle | Run 7 progress D_psi | 0.00 | 0.1227 | 0.1535 | 100.0 | 0.0590 | 0.0448 |
| shuffled learned | BC 1800 | 0.02 | 0.2071 | 0.2573 | 98.8 | 0.1234 | 0.0766 |
| shuffled learned | Run 6 true-TCP | 0.00 | 0.1215 | 0.1580 | 100.0 | 0.0690 | 0.1189 |
| shuffled learned | Run 7 progress D_psi | 0.00 | 0.1192 | 0.1526 | 100.0 | 0.0654 | 0.0398 |

Interpretation:

BC behaves as expected: full-task success drops from about `0.63-0.65` to
`0.02-0.03` when full-rollout goals are shuffled. The RL policies have near
zero task success even under correct oracle/learned goals, so the shuffled
full-rollout rows cannot distinguish much further; they mainly confirm that
the RL failure is not caused by high-level goal prediction alone.

Combined with the local input ablation, the picture is now consistent:

```text
RL low level can chase TCP endpoints locally,
but endpoint chasing does not produce successful PushT behavior.
```

The next useful training experiment should change the local reward or policy
constraint, not simply increase endpoint-only PPO length.
