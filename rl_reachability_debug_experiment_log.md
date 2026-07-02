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

## 2026-07-01 - Run 8: Privileged Object-Pose Local PPO

Motivation:

The plan's Phase 1 asks for privileged/state local reaching, not only TCP
endpoint reaching. Runs 6 and 7 showed that a low-level policy can become a
strong TCP endpoint servo while still failing PushT. Run 8 therefore changes
the local goal and reward to the object's privileged pose:

```text
goal_type = object_pose
goal = [object_x, object_y, sin(object_yaw), cos(object_yaw)]
reward distance = squared distance in that object-pose goal space
```

This is still an oracle local-reset experiment, not a deployable hierarchy,
because the available learned high-level predictor in `phase_f` is TCP-only.

Implementation:

`scripts/rl_reachability_privileged_tcp_ppo.py` now supports:

- `--goal-type {tcp, object_pose, object, robot, full}`
- `--reward-distance-source true_goal`

The default remains TCP and existing `D_psi`/BC-advantage paths are guarded as
TCP-only because their learned metric and frozen BC baseline were trained for
the TCP interface.

Command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type object_pose \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run8_object_pose_b8_u250 \
  --force
```

Training budget:

```text
4096 envs * 10 steps/update * 250 updates = 10.24M environment steps
8 minibatches * 3 epochs * 250 updates = 6000 optimizer steps
```

Results:

| Metric | Initial random policy | Trained object-pose PPO | Trained shuffled-goal eval |
| --- | ---: | ---: | ---: |
| object-pose initial distance | 0.5897 | 0.5897 | 0.5897 |
| terminal object-pose distance | 0.5690 | 0.1517 | 0.3422 |
| distance reduction | 0.0207 | 0.4380 | 0.2475 |
| reach under epsilon 0.0025 | 0.2410 | 0.3160 | 0.2744 |
| p50 terminal distance | 0.1357 | 0.0120 | 0.0269 |
| p90 terminal distance | 1.9496 | 0.4894 | 1.0402 |
| p99 terminal distance | 3.2821 | 1.9135 | 3.8756 |
| action saturation | 0.0000 | 0.1214 | 0.1584 |
| action L2 | 0.0057 | 0.7342 | 0.7440 |

Final PPO diagnostics:

| Diagnostic | Value |
| --- | ---: |
| final train terminal object-pose distance | 0.3992 |
| final train reach under epsilon | 0.0974 |
| mean return per step | 0.0256 |
| policy KL | 0.00846 |
| clip fraction | 0.1132 |
| value loss | 0.0624 |
| explained variance | 0.8225 |
| action saturation | 0.3090 |
| NaN count | 0 |
| elapsed | 2009s |

Interpretation:

Object-pose PPO gives a positive local learning signal: eval terminal distance
drops from `0.5690` to `0.1517`, and correct-goal performance is clearly
better than shuffled-goal performance (`0.1517` versus `0.3422` terminal
distance). This is a stronger result than the TCP-only endpoint experiments for
the specific question of whether PPO can use a privileged task-relevant goal.

However, this is not yet a deployable hierarchy result. We do not currently
have a matching learned object-pose high-level predictor wired into the full
rollout evaluator, and the existing BC low level is TCP-conditioned. The next
experiment should either train/evaluate the matching object-pose high-level
interface or add object/contact terms to the TCP-interface reward while keeping
the deployable TCP high-level policy fixed.

## 2026-07-01 - Run 8: Object-Pose Oracle Full Rollout

Motivation:

Run 8 showed that PPO can improve a local privileged object-pose objective, but
the relevant ceiling metric is whether that low level helps full PushT rollouts
when the object-pose goals are good. This evaluation uses oracle object-pose
goals generated by rolling the privileged teacher forward in a branch
environment at every high-level decision point.

Baseline:

The matching object-pose BC low level already exists in:

```text
artifacts/incremental/pre_rl/phase_b/k10/seed0/oracle_goal_decomposition.pt
```

That model is not time-conditioned, but it is the available supervised
object-pose low-level baseline for the same goal representation. The Run 8 PPO
low level is time-conditioned and uses the `object_pose` goal normalizer from
the local PPO run.

Command:

```bash
uv run python scripts/rl_reachability_object_pose_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle shuffled_oracle \
  --output results/incremental/rl_reachability_debug/run8_object_pose_full_success_100.json
```

Results:

| Goal source | Low-level policy | Success | Final reward | Max reward | Mean length | Hold object-pose distance | Teacher action MAE | Action saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle object-pose | Phase-B object-pose BC | 0.18 | 0.3740 | 0.3908 | 89.4 | 0.3005 | 0.1777 | 0.0569 |
| oracle object-pose | Run 8 object-pose PPO | 0.00 | 0.1407 | 0.1544 | 100.0 | 0.2606 | 0.5181 | 0.0098 |
| shuffled oracle object-pose | Phase-B object-pose BC | 0.01 | 0.1796 | 0.2215 | 99.5 | 1.4569 | 0.3134 | 0.0735 |
| shuffled oracle object-pose | Run 8 object-pose PPO | 0.00 | 0.1265 | 0.1485 | 100.0 | 1.8466 | 0.5709 | 0.0447 |

Interpretation:

Oracle object-pose goals help the supervised object-pose BC baseline somewhat
(`0.18` success), and shuffled object-pose goals mostly remove that success
(`0.01`). This confirms the evaluator is sensitive to the object-pose goal.

The Run 8 PPO low level has lower mean hold object-pose distance than the BC
baseline under oracle goals (`0.2606` versus `0.3005`) but still has zero task
success and much larger teacher-action MAE (`0.5181` versus `0.1777`). This is
the same failure pattern seen in the TCP endpoint experiments:

```text
improving short-horizon goal reachability alone is not enough;
the learned action distribution is task-destructive.
```

The next aligned experiment should explicitly penalize action drift or train a
reward/critic that includes task progress/contact dynamics, rather than only
changing the goal representation.

## 2026-07-01 - Run 9: Object-Pose PPO With Teacher-Action Penalty

Motivation:

Run 8 showed the same core failure as the TCP runs: the RL low level improved
short-horizon goal distance but moved far from the teacher action
distribution. Run 9 adds a simple opt-in imitation penalty to the local PPO
reward:

```text
reward = progress_terminal_object_pose_reward
         - 0.05 * mean_abs(clipped_action - teacher_action)
```

Implementation:

`scripts/rl_reachability_privileged_tcp_ppo.py` now supports:

```text
--teacher-action-penalty-weight
```

The default is `0.0`, so previous runs are unchanged. When the weight is
nonzero, the runner loads the privileged teacher and logs train-time
`teacher_action_mae`.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type object_pose \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 0.05 \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run9_object_pose_teacher_penalty_b8_u250 \
  --force
```

Local object-pose eval:

| Metric | Run 8 object-pose PPO | Run 9 + teacher penalty |
| --- | ---: | ---: |
| terminal object-pose distance | 0.1517 | 0.0728 |
| distance reduction | 0.4380 | 0.5169 |
| reach under epsilon 0.0025 | 0.3160 | 0.4074 |
| p50 terminal distance | 0.0120 | 0.0055 |
| p90 terminal distance | 0.4894 | 0.1536 |
| p99 terminal distance | 1.9135 | 1.2180 |
| action saturation | 0.1214 | 0.1197 |
| action L2 | 0.7342 | 0.5813 |
| final train teacher-action MAE | n/a | 0.2723 |

Final PPO diagnostics:

| Diagnostic | Value |
| --- | ---: |
| updates | 250 |
| env steps | 10.24M |
| final train terminal object-pose distance | 0.1829 |
| final train reach under epsilon | 0.1213 |
| mean return per step | 0.0552 |
| policy KL | 0.0114 |
| clip fraction | 0.1580 |
| value loss | 0.0382 |
| explained variance | 0.8753 |
| action saturation | 0.2691 |
| NaN count | 0 |
| elapsed | 2014s |

Oracle object-pose full rollout:

```bash
uv run python scripts/rl_reachability_object_pose_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle shuffled_oracle \
  --run8-low results/incremental/rl_reachability_debug/run9_object_pose_teacher_penalty_b8_u250/privileged_object_pose_ppo_progress_terminal_n4096_seed0/latest.pt \
  --run8-low-name run9_object_pose_teacher_penalty_ppo \
  --output results/incremental/rl_reachability_debug/run9_object_pose_teacher_penalty_full_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Mean length | Hold object-pose distance | Teacher action MAE | Action L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle object-pose | Phase-B object-pose BC | 0.17 | 0.3650 | 0.3840 | 90.2 | 0.2721 | 0.1896 | 0.4863 |
| oracle object-pose | Run 9 teacher-penalty PPO | 0.00 | 0.1546 | 0.1979 | 100.0 | 0.3184 | 0.2939 | 0.3575 |
| shuffled oracle object-pose | Phase-B object-pose BC | 0.01 | 0.1805 | 0.2225 | 99.5 | 1.5011 | 0.3160 | 0.5419 |
| shuffled oracle object-pose | Run 9 teacher-penalty PPO | 0.00 | 0.1239 | 0.1648 | 100.0 | 1.8513 | 0.4196 | 0.3456 |

Interpretation:

The teacher-action penalty improves the local object-pose objective even beyond
Run 8 and reduces action magnitude/teacher-action drift, but it still does not
recover full-task success. Under oracle object-pose goals, Run 9 has lower
teacher MAE than Run 8 (`0.2939` versus `0.5181`) but remains worse than the
Phase-B BC baseline (`0.1896`) and gets `0.00` success.

This narrows the issue further: a weak imitation penalty is not enough. The
policy still does not reproduce the task-relevant interaction strategy, even
when it reaches object-pose goals locally. The next useful variants are either
a much stronger action-distribution constraint/warm start, or a reward/critic
that directly scores task progress and contact dynamics during the branch.

## 2026-07-01 - Run 10: Stronger Teacher-Action Penalty

Motivation:

Run 9 used a weak action penalty (`0.05`) and reduced teacher-action drift but
still got zero full-task success. Run 10 increases the teacher-action penalty
to `0.2` while keeping the object-pose reward and training budget fixed.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type object_pose \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 0.2 \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run10_object_pose_teacher_penalty02_b8_u250 \
  --force
```

Local object-pose eval:

| Metric | Run 8 no penalty | Run 9 penalty 0.05 | Run 10 penalty 0.2 |
| --- | ---: | ---: | ---: |
| terminal object-pose distance | 0.1517 | 0.0728 | 0.0527 |
| distance reduction | 0.4380 | 0.5169 | 0.5370 |
| reach under epsilon 0.0025 | 0.3160 | 0.4074 | 0.4334 |
| p50 terminal distance | 0.0120 | 0.0055 | 0.0041 |
| p90 terminal distance | 0.4894 | 0.1536 | 0.1062 |
| p99 terminal distance | 1.9135 | 1.2180 | 0.9304 |
| action saturation | 0.1214 | 0.1197 | 0.1207 |
| action L2 | 0.7342 | 0.5813 | 0.6151 |
| final train teacher-action MAE | n/a | 0.2723 | 0.2324 |

Final PPO diagnostics:

| Diagnostic | Value |
| --- | ---: |
| updates | 250 |
| env steps | 10.24M |
| final train terminal object-pose distance | 0.1429 |
| final train reach under epsilon | 0.1382 |
| mean return per step | 0.0304 |
| policy KL | 0.0158 |
| clip fraction | 0.2081 |
| value loss | 0.0393 |
| explained variance | 0.8546 |
| action saturation | 0.2570 |
| NaN count | 0 |
| elapsed | 2025s |

Oracle object-pose full rollout:

```bash
uv run python scripts/rl_reachability_object_pose_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle shuffled_oracle \
  --run8-low results/incremental/rl_reachability_debug/run10_object_pose_teacher_penalty02_b8_u250/privileged_object_pose_ppo_progress_terminal_n4096_seed0/latest.pt \
  --run8-low-name run10_object_pose_teacher_penalty02_ppo \
  --output results/incremental/rl_reachability_debug/run10_object_pose_teacher_penalty02_full_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Mean length | Hold object-pose distance | Teacher action MAE | Action L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle object-pose | Phase-B object-pose BC | 0.17 | 0.3679 | 0.3849 | 90.3 | 0.2935 | 0.1851 | 0.4794 |
| oracle object-pose | Run 10 teacher-penalty PPO | 0.12 | 0.2982 | 0.3293 | 94.3 | 0.2558 | 0.2406 | 0.3585 |
| shuffled oracle object-pose | Phase-B object-pose BC | 0.02 | 0.1822 | 0.2254 | 99.0 | 1.3863 | 0.3147 | 0.5551 |
| shuffled oracle object-pose | Run 10 teacher-penalty PPO | 0.00 | 0.1479 | 0.1665 | 100.0 | 1.6080 | 0.3660 | 0.2994 |

Interpretation:

Run 10 is the first scratch RL low-level variant in this debug sequence that
recovers nontrivial full-task success: `0.12` under oracle object-pose goals.
It still trails the supervised object-pose BC baseline (`0.17`) and remains
sensitive to shuffled object-pose goals (`0.00` success), but this is a
meaningful improvement over Runs 8 and 9, which both had `0.00` success.

The key difference is not just better object-pose distance. Run 10 also lowers
teacher-action drift enough to preserve some task interaction behavior while
still improving local reachability. This supports the current diagnosis:

```text
reachability reward needs an action-distribution/task-interaction constraint;
pure local goal reaching optimizes the wrong controller behavior.
```

Next useful variants:

- sweep stronger penalties around `0.2` (for example `0.1`, `0.3`, `0.5`);
- warm-start from the object-pose BC low level instead of scratch;
- add task reward or contact/progress reward during local branches.

## 2026-07-01 - Run 11/12: Teacher-Action Penalty Sweep

Motivation:

Run 10 recovered nontrivial task success with penalty `0.2`. Runs 11 and 12
bracket that setting with penalties `0.1` and `0.3` while keeping the same
object-pose reward, PPO setup, dataset, and training budget.

Training commands:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type object_pose \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight {0.1 or 0.3} \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run{11 or 12}_object_pose_teacher_penalty*_b8_u250 \
  --force
```

Local object-pose eval:

| Run | Penalty | Terminal distance | Reach | P90 distance | Shuffled terminal distance | Train teacher MAE | Clip fraction | Eval action L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 8 | 0.00 | 0.1517 | 0.3160 | 0.4894 | 0.3422 | n/a | 0.1132 | 0.7342 |
| Run 9 | 0.05 | 0.0728 | 0.4074 | 0.1536 | 0.3271 | 0.2723 | 0.1580 | 0.5813 |
| Run 11 | 0.10 | 0.0646 | 0.4311 | 0.1265 | 0.3063 | 0.2510 | 0.2050 | 0.6026 |
| Run 10 | 0.20 | 0.0527 | 0.4334 | 0.1062 | 0.2813 | 0.2324 | 0.2081 | 0.6151 |
| Run 12 | 0.30 | 0.0426 | 0.4702 | 0.0790 | 0.2613 | 0.1836 | 0.2002 | 0.6695 |

Oracle object-pose full rollout:

| Run | Low-level policy | Success | Final reward | Max reward | Hold object-pose distance | Teacher action MAE | Action L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-B baseline | object-pose BC | 0.16-0.18 | 0.355-0.374 | 0.372-0.391 | 0.268-0.301 | 0.178-0.190 | 0.472-0.486 |
| Run 8 | no-penalty PPO | 0.00 | 0.1407 | 0.1544 | 0.2606 | 0.5181 | 0.5940 |
| Run 9 | penalty 0.05 PPO | 0.00 | 0.1546 | 0.1979 | 0.3184 | 0.2939 | 0.3575 |
| Run 11 | penalty 0.10 PPO | 0.03 | 0.2285 | 0.2506 | 0.2650 | 0.2590 | 0.3164 |
| Run 10 | penalty 0.20 PPO | 0.12 | 0.2982 | 0.3293 | 0.2558 | 0.2406 | 0.3585 |
| Run 12 | penalty 0.30 PPO | 0.16 | 0.3533 | 0.3720 | 0.2073 | 0.1766 | 0.3761 |

Run 12 shuffled-goal full rollout:

| Goal source | Policy | Success | Final reward | Max reward | Hold object-pose distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| shuffled oracle object-pose | Phase-B object-pose BC | 0.02 | 0.1848 | 0.2296 | 1.4296 | 0.3272 |
| shuffled oracle object-pose | Run 12 penalty 0.30 PPO | 0.00 | 0.1514 | 0.1803 | 1.5550 | 0.3250 |

Interpretation:

The penalty sweep shows a clear monotonic trend over the tested range:

```text
higher teacher-action penalty -> lower teacher-action drift,
better local object-pose reachability,
and better oracle-goal full-task success.
```

Run 12 is the best scratch RL low-level so far. With oracle object-pose goals
it matches the Phase-B object-pose BC baseline's success (`0.16`) while having
lower hold object-pose distance (`0.2073` versus `0.2819` in the same eval)
and comparable teacher-action MAE (`0.1766` versus `0.1814`). Shuffled
object-pose goals still remove success, so the full rollout behavior is goal
sensitive.

This is the strongest evidence so far that PPO can improve local reachability
without destroying task behavior, but only when constrained strongly enough
toward the teacher/action manifold. The next scoped experiment should test
whether a warm start from the object-pose BC baseline or an even stronger
penalty (`0.5`) can surpass the BC baseline rather than merely match it.

## 2026-07-01 - Run 13: Teacher-Action Penalty 0.5

Motivation:

Run 12 matched the object-pose BC baseline at penalty `0.3`. Run 13 tests the
next stronger penalty, `0.5`, to see whether scratch PPO can surpass the
supervised object-pose BC baseline while still improving object-pose
reachability.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type object_pose \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 0.5 \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run13_object_pose_teacher_penalty05_b8_u250 \
  --force
```

Local object-pose eval:

| Run | Penalty | Terminal distance | Reach | P90 distance | Shuffled terminal distance | Train teacher MAE | Eval action saturation | Eval action L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 12 | 0.3 | 0.0426 | 0.4702 | 0.0790 | 0.2613 | 0.1836 | 0.1671 | 0.6695 |
| Run 13 | 0.5 | 0.0414 | 0.4800 | 0.0754 | 0.2531 | 0.1662 | 0.1947 | 0.6742 |

Final PPO diagnostics:

| Diagnostic | Value |
| --- | ---: |
| updates | 250 |
| env steps | 10.24M |
| final train terminal object-pose distance | 0.1255 |
| final train reach under epsilon | 0.1746 |
| mean return per step | -0.0028 |
| policy KL | 0.0151 |
| clip fraction | 0.1998 |
| value loss | 0.0651 |
| explained variance | 0.7794 |
| action saturation | 0.3375 |
| NaN count | 0 |
| elapsed | 2033s |

Oracle object-pose full rollout:

```bash
uv run python scripts/rl_reachability_object_pose_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle shuffled_oracle \
  --run8-low results/incremental/rl_reachability_debug/run13_object_pose_teacher_penalty05_b8_u250/privileged_object_pose_ppo_progress_terminal_n4096_seed0/latest.pt \
  --run8-low-name run13_object_pose_teacher_penalty05_ppo \
  --output results/incremental/rl_reachability_debug/run13_object_pose_teacher_penalty05_full_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Mean length | Hold object-pose distance | Teacher action MAE | Action saturation | Action L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle object-pose | Phase-B object-pose BC | 0.16 | 0.3561 | 0.3739 | 91.0 | 0.2839 | 0.1792 | 0.0500 | 0.4703 |
| oracle object-pose | Run 13 penalty 0.5 PPO | 0.21 | 0.4015 | 0.4110 | 89.0 | 0.2315 | 0.1649 | 0.0544 | 0.3785 |
| shuffled oracle object-pose | Phase-B object-pose BC | 0.01 | 0.1786 | 0.2227 | 99.5 | 1.4147 | 0.3127 | 0.0723 | 0.5509 |
| shuffled oracle object-pose | Run 13 penalty 0.5 PPO | 0.00 | 0.1549 | 0.1784 | 100.0 | 1.6670 | 0.3319 | 0.0431 | 0.3444 |

Interpretation:

Run 13 is the first scratch RL low-level in this sequence to beat the matching
supervised object-pose BC baseline under oracle object-pose goals:

```text
Phase-B object-pose BC: 0.16 success
Run 13 penalty 0.5 PPO: 0.21 success
```

It also has better final/max reward, lower hold object-pose distance, lower
teacher-action MAE, and lower action L2 than the BC baseline in the same
evaluation. Shuffled object-pose goals still remove success, so the behavior is
not merely a goal-independent default policy.

The result confirms the central debugging conclusion: PPO can improve the
low-level controller, but only when the reachability objective is paired with a
strong enough action-manifold constraint. Pure reachability learns a task-bad
servo; constrained reachability can now surpass the supervised object-pose
baseline under oracle goals.

Remaining gap:

This is still an oracle-goal result. To make it deployable, the next scoped
step is to train/evaluate a learned high-level object-pose predictor or to port
the same constrained-RL idea back to the deployable TCP high-level interface
with task/contact-aware reward terms.

## 2026-07-01 - Run 14: Learned Object-Pose High-Level Predictor

Motivation:

Run 13 beat the matching object-pose BC low-level under oracle object-pose
subgoals. The plan also requires the deployability check with a matching
learned high-level policy. Run 14 trains a deterministic object-pose high-level
predictor from the same Phase-7 privileged teacher episodes, then evaluates
Phase-B object-pose BC and Run 13 PPO under oracle, learned, and shuffled
learned subgoals.

Training command:

```bash
uv run python scripts/rl_reachability_object_pose_high_predictor.py \
  --output artifacts/incremental/rl_reachability_debug/object_pose_high_predictor/seed0/predictor.pt
```

High-level validation:

| Metric | Value |
| --- | ---: |
| train episodes | 1800 |
| validation episodes | 200 |
| train samples | 53115 |
| validation samples | 5901 |
| validation object-pose L2 | 0.0561 |
| validation xy L2 | 0.0085 m |
| validation yaw abs | 0.0517 rad |
| persistence object-pose L2 | 0.5042 |
| persistence xy L2 | 0.0597 m |

Full-rollout command:

```bash
uv run python scripts/rl_reachability_object_pose_full_success_eval.py \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle learned shuffled_learned \
  --high-checkpoint artifacts/incremental/rl_reachability_debug/object_pose_high_predictor/seed0/predictor.pt \
  --run8-low results/incremental/rl_reachability_debug/run13_object_pose_teacher_penalty05_b8_u250/privileged_object_pose_ppo_progress_terminal_n4096_seed0/latest.pt \
  --run8-low-name run13_object_pose_teacher_penalty05_ppo \
  --output results/incremental/rl_reachability_debug/run14_object_pose_learned_high_full_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold object-pose distance | Selected goal initial distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-B object-pose BC | 0.17 | 0.3647 | 0.3837 | 0.2986 | 0.1922 | 0.1835 |
| oracle | Run 13 penalty 0.5 PPO | 0.20 | 0.3972 | 0.4063 | 0.2227 | 0.1303 | 0.1604 |
| learned | Phase-B object-pose BC | 0.13 | 0.3404 | 0.3581 | 0.5900 | 0.5296 | 0.1643 |
| learned | Run 13 penalty 0.5 PPO | 0.17 | 0.3697 | 0.3811 | 0.3141 | 0.2694 | 0.1688 |
| shuffled learned | Phase-B object-pose BC | 0.04 | 0.2068 | 0.2439 | 1.5780 | 1.6085 | 0.3036 |
| shuffled learned | Run 13 penalty 0.5 PPO | 0.01 | 0.1673 | 0.1985 | 1.4187 | 1.4060 | 0.2957 |

Interpretation:

The learned object-pose high-level predictor is strong in supervised validation,
but the full-rollout learned goals are farther from the current state than the
oracle goals and reduce task success. Even so, Run 13 PPO remains better than
the Phase-B object-pose BC baseline with the learned high-level (`0.17` vs
`0.13` success), and shuffled learned goals largely remove success. This keeps
the main conclusion intact: the constrained object-pose PPO low-level is useful
and goal-sensitive, but the deployable stack still has a high-level distribution
gap relative to oracle object-pose subgoals.

Next scoped debugging if learned-goal success stays low:

1. Measure reachability to learned high-level goals in the states produced by
   full architecture deployment, not only on reset/teacher branch states.
2. Compare those deployment-state reachability numbers directly against the
   Phase-B object-pose BC low-level.
3. Only continue to new plan experiments after this rollout-state reachability
   gap is explained.

## 2026-07-01 - Run 15: Object-Pose Reachability from Deployed States

Motivation:

Run 14 learned-goal task success remained low even though Run 13 PPO beat the
BC low-level under learned high-level goals. This diagnostic follows the user
requested check: collect states produced by deploying the architecture, then
branch both candidate low-level policies from those exact states toward the same
learned high-level object-pose goals.

Command:

```bash
uv run python scripts/rl_reachability_object_pose_deployment_reachability_eval.py \
  --decisions 512 \
  --num-envs 16 \
  --output results/incremental/rl_reachability_debug/run15_object_pose_deployment_reachability_512.json
```

Setup:

- high-level goal source: learned object-pose predictor from Run 14
- collectors: Phase-B object-pose BC, Run 13 penalty `0.5` PPO
- candidates branched from each collected replan state: same two low-levels
- decisions per collector: about `512`
- branch horizon: `10` primitive steps
- reach epsilon: object-pose squared distance `<= 0.01`
- shuffled learned goals are included as a goal-use control

Results:

| Collector trajectory | Candidate low-level | Shuffled | Initial dist. | Terminal dist. | Reach eps | Improved | P90 terminal | Action sat. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-B BC | Phase-B BC | no | 0.4784 | 0.2757 | 0.404 | 0.584 | 0.8362 | 0.046 |
| Phase-B BC | Run 13 PPO | no | 0.4784 | 0.2358 | 0.429 | 0.594 | 0.7265 | 0.101 |
| Phase-B BC | Phase-B BC | yes | 1.2610 | 1.0610 | 0.0716 | 0.569 | 3.1982 | 0.089 |
| Phase-B BC | Run 13 PPO | yes | 1.2610 | 1.0749 | 0.0716 | 0.590 | 3.2240 | 0.105 |
| Run 13 PPO | Phase-B BC | no | 0.3859 | 0.3042 | 0.313 | 0.571 | 1.0070 | 0.036 |
| Run 13 PPO | Run 13 PPO | no | 0.3859 | 0.2347 | 0.301 | 0.557 | 0.7251 | 0.070 |
| Run 13 PPO | Phase-B BC | yes | 1.2043 | 1.0852 | 0.0971 | 0.536 | 3.1207 | 0.059 |
| Run 13 PPO | Run 13 PPO | yes | 1.2043 | 1.0866 | 0.0835 | 0.551 | 3.1473 | 0.077 |

Interpretation:

From identical deployed architecture states and identical learned high-level
goals, Run 13 PPO has better mean and P90 object-pose terminal distance than
the Phase-B object-pose BC low-level. This holds both on BC-collected states
and on Run-13-collected states. The threshold reach-rate metric is mixed on
Run-13-collected states (`0.301` vs `0.313`), but the distributional metrics
favor Run 13 (`0.2347` mean terminal distance vs `0.3042`, and `0.7251` P90 vs
`1.0070`).

Shuffled learned goals are much harder and have low reach rates, so the
diagnostic still shows goal sensitivity. The remaining low task success is
therefore not explained by the Run 13 low-level failing to reach learned
object-pose goals from deployed states. The more likely bottleneck is that the
learned high-level object-pose goals are not sufficiently task/contact aligned,
or that object-pose-only subgoals omit task-relevant robot/contact geometry.

## 2026-07-01 - Run 16: Full-State Subgoal PPO

Motivation:

The original hierarchy idea is that the high level should output the desired
future scene/robot state after the low-level option, not only the future TCP
position. The TCP-only RL policies improved endpoint reachability without
improving task success because a good future end-effector position does not
imply the object/contact state was reached. Run 16 tests the same constrained
scratch PPO recipe as Run 13, but with the `full` Phase-B goal representation.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type full \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 0.5 \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run16_full_goal_teacher_penalty05_b8_u250 \
  --force
```

Local full-goal result:

| Metric | Initial/random | Run 16 PPO | Shuffled-goal PPO |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 7.4381 | 1.9120 | 7.8599 |
| distance reduction | 6.9701 | 12.4961 | 6.5483 |
| fraction improved | 0.7801 | 0.9708 | 0.7381 |
| p50 terminal distance | 4.9755 | 0.6572 | 4.8239 |
| p90 terminal distance | 15.2186 | 3.7421 | 16.1092 |
| action saturation | 0.0000 | 0.0058 | 0.0026 |
| action L2 | 0.0068 | 0.5322 | 0.5378 |

Training diagnostics:

| Diagnostic | Value |
| --- | ---: |
| updates | 250 |
| env steps | 10.24M |
| final train terminal full-goal distance | 3.7594 |
| mean return per step | 1.0142 |
| policy KL | 0.0137 |
| clip fraction | 0.1789 |
| value loss | 2.5507 |
| explained variance | 0.9248 |
| action saturation | 0.0174 |
| teacher action MAE | 0.2849 |
| NaN count | 0 |
| elapsed | 2034s |

The local result is strongly goal-sensitive: the trained policy reduces held
full-goal distance far more than the shuffled-goal control. This supports the
core hypothesis that full-state subgoals are a better RL objective than TCP-only
subgoals.

Learned full-goal high-level:

```bash
uv run python scripts/rl_reachability_goal_high_predictor.py \
  --goal-type full \
  --output artifacts/incremental/rl_reachability_debug/full_goal_high_predictor/seed0/predictor.pt
```

| Metric | Value |
| --- | ---: |
| train episodes | 1800 |
| validation episodes | 200 |
| validation full-goal L2 | 0.8019 |
| validation full-goal MSE | 0.0522 |
| validation p90 full-goal L2 | 1.5780 |
| persistence full-goal L2 | 3.8790 |
| persistence p90 full-goal L2 | 6.2743 |

Raw-goal held evaluation:

This was the first held-subgoal check I ran. It holds the raw 28D Phase-B
`full` goal vector fixed for `k=10` primitive steps.

```bash
uv run python scripts/rl_reachability_goal_full_success_eval.py \
  --goal-type full \
  --goal-dim 28 \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle learned shuffled_learned \
  --high-checkpoint artifacts/incremental/rl_reachability_debug/full_goal_high_predictor/seed0/predictor.pt \
  --rl-low results/incremental/rl_reachability_debug/run16_full_goal_teacher_penalty05_b8_u250/privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt \
  --rl-low-name run16_full_goal_teacher_penalty05_ppo \
  --output results/incremental/rl_reachability_debug/run16_full_goal_full_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Selected goal initial distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-B full BC | 0.01 | 0.1724 | 0.2226 | 13.0115 | 14.5368 | 0.2710 |
| oracle | Run 16 full PPO | 0.00 | 0.1239 | 0.1668 | 5.2004 | 10.6121 | 0.3163 |
| learned | Phase-B full BC | 0.01 | 0.1633 | 0.2115 | 20.9570 | 22.9705 | 0.3146 |
| learned | Run 16 full PPO | 0.00 | 0.1311 | 0.1729 | 4.3042 | 6.5982 | 0.3486 |
| shuffled learned | Phase-B full BC | 0.00 | 0.1113 | 0.1669 | 19.9949 | 20.7079 | 0.3937 |
| shuffled learned | Run 16 full PPO | 0.00 | 0.1073 | 0.1438 | 13.8111 | 15.4410 | 0.4528 |

Correction:

This is **not** the correct Phase-B `full` held-subgoal protocol. The Phase-B
`full` goal vector contains velocity/rate features in addition to target object,
TCP, and joint state. The old project evaluator keeps the target future state
fixed, but recomputes the goal features at every primitive step using the
current state and remaining time. Holding the raw 28D vector fixed gives the BC
policy an out-of-distribution condition and explains the suspicious near-zero
BC success.

Official held-goal BC protocol audit:

I reran the existing project evaluator with `goal_type=full`, `k=10`, and
`goal_update_period=10`. This uses `fixed_endpoint_recomputed_features`
semantics: the future target state is held, while the Phase-B goal features are
recomputed each step.

| BC policy | Time conditioned | Success | Final reward | Max reward | Teacher action MAE | Object error | TCP error | Yaw error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-B full BC | no | 0.08 | 0.2915 | 0.3187 | 0.2146 | 0.0204 m | 0.1087 m | 0.1454 rad |
| Phase-C full BC | yes | 0.74 | 0.8241 | 0.8258 | 0.0407 | 0.0179 m | 0.0662 m | 0.1299 rad |

The time-conditioned Phase-C full BC checkpoint is:

```text
artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt
```

Its validation action MAE is `0.0305`, and the held-oracle success is `0.74`.
This confirms that full-state subgoals are not inherently bad; they are strong
when the low-level is trained/evaluated with the correct held-target semantics
and remaining-time conditioning.

Non-hierarchical one-step replan diagnostic:

For comparison with the old Phase-B table, I also ran `--update-period 1`. This
replans every primitive step, so it is **not** the hierarchy we are trying to
test. It is a receding-horizon oracle/high-level diagnostic only.

| Goal source | Low-level policy | Success | Final reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| oracle, update=1 | Phase-B full BC | 0.41 | 0.5759 | 9.0369 | 0.0974 |
| oracle, update=1 | Run 16 full PPO | 0.00 | 0.1132 | 6.4169 | 0.4106 |
| learned, update=1 | Phase-B full BC | 0.29 | 0.4863 | 7.2604 | 0.1117 |
| learned, update=1 | Run 16 full PPO | 0.00 | 0.1116 | 6.0406 | 0.4222 |

This cross-check explains why older Phase-B `full` rows looked much stronger:
they used `action_horizon_steps=1`. That result is useful historically, but the
held-subgoal evaluation with recomputed features and time conditioning is the
correct BC baseline for the hierarchical controller.

Next action:

Do not abandon full-state goals. The corrected audit shows the opposite:
time-conditioned full-state BC is very strong under held oracle goals. The next
scoped RL experiment should retrain/evaluate full-state PPO with the same
held-target/recomputed-feature semantics as Phase C, rather than holding the raw
Phase-B `full` vector fixed.

## 2026-07-01 - Run 19: Full-State PPO With Recomputed Held-Goal Features

Motivation:

Run 16 trained PPO with a fixed raw `full` goal vector. That was inconsistent
with the Phase-C held-target semantics because `full` contains error-to-go
velocity/rate features. Run 19 retrains full-state PPO while holding the target
future state fixed and recomputing the full-goal feature vector at every
primitive step from the current state and remaining time.

Code change:

`scripts/rl_reachability_privileged_tcp_ppo.py` now supports:

```text
--recompute-held-goal-features
```

When enabled:

- local reset stores the target future state;
- each primitive step recomputes `goal = f(current_state, target_future_state, remaining_steps)`;
- goal normalizers are fit over all offsets `1..k`, matching Phase-C style data;
- shuffled-goal eval shuffles the held target future state, not only the raw goal vector.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type full \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 0.5 \
  --recompute-held-goal-features \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run19_full_goal_recomputed_teacher_penalty05_b8_u250 \
  --force
```

Local full-goal result:

| Metric | Initial/random | Run 19 PPO | Shuffled-goal PPO |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 7.4366 | 1.5264 | 4.5738 |
| distance reduction | 6.9715 | 12.8817 | 9.8344 |
| fraction improved | 0.7559 | 0.9727 | 0.8095 |
| p50 terminal distance | 4.9705 | 0.6987 | 3.2515 |
| p90 terminal distance | 15.2401 | 3.2267 | 8.6473 |
| action saturation | 0.0000 | 0.0221 | 0.0202 |
| action L2 | 0.0073 | 0.5404 | 0.5665 |

Training diagnostics:

| Diagnostic | Value |
| --- | ---: |
| updates | 250 |
| env steps | 10.24M |
| recomputed goal normalizer samples | 16.71M |
| final train terminal full-goal distance | 2.8175 |
| mean return per step | 1.1971 |
| policy KL | 0.0154 |
| clip fraction | 0.1972 |
| value loss | 3.9498 |
| explained variance | 0.8322 |
| action saturation | 0.0712 |
| teacher action MAE | 0.2960 |
| NaN count | 0 |
| elapsed | 2006s |

Corrected held-target oracle rollout:

```bash
uv run python scripts/rl_reachability_goal_full_success_eval.py \
  --goal-type full \
  --goal-dim 28 \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle shuffled_oracle \
  --recompute-held-goal-features \
  --bc-low artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt \
  --rl-low results/incremental/rl_reachability_debug/run19_full_goal_recomputed_teacher_penalty05_b8_u250/privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt \
  --rl-low-name run19_full_goal_recomputed_teacher_penalty05_ppo \
  --output results/incremental/rl_reachability_debug/run19_full_goal_recomputed_oracle_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.69 | 0.7897 | 0.7918 | 1.6358 | 0.0414 |
| oracle | Run 19 recomputed full PPO | 0.00 | 0.1587 | 0.1879 | 5.2944 | 0.2812 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3724 | 0.4002 |
| shuffled oracle | Run 19 recomputed full PPO | 0.00 | 0.1141 | 0.1476 | 10.2356 | 0.4011 |

Interpretation:

The corrected full-state PPO objective improves local reachability and preserves
goal sensitivity, but it still does not produce a task-useful low-level policy.
The comparison against the correct Phase-C full BC baseline is stark:

```text
Phase-C full BC oracle held success: 0.69
Run 19 recomputed full PPO oracle held success: 0.00
```

The main failure mode is still action/contact-manifold drift. Run 19 has much
higher teacher-action MAE than Phase-C BC (`0.281` vs `0.041`) and much worse
task reward, despite reducing full-goal distance locally. The next scoped PPO
variant should not change the goal semantics again; it should constrain the
policy harder, for example with a stronger teacher-action penalty, BC warm
start, or residual-on-BC formulation.

## 2026-07-01 - Run 20: Recomputed Full-State PPO With Teacher Penalty 1.0

Motivation:

Run 19 used the corrected full-state held-target semantics, but still drifted
far from the Phase-C full BC action/contact manifold. Run 20 repeats the same
setup with a stronger teacher-action penalty (`1.0` instead of `0.5`) to test
whether a simple penalty increase preserves local reachability while reducing
teacher-action drift.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type full \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 1.0 \
  --recompute-held-goal-features \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run20_full_goal_recomputed_teacher_penalty10_b8_u250 \
  --force
```

Local full-goal result:

| Metric | Initial/random | Run 19 penalty 0.5 | Run 20 penalty 1.0 | Run 20 shuffled |
| --- | ---: | ---: | ---: | ---: |
| terminal full-goal distance | 7.4366 | 1.5264 | 1.4703 | 4.4992 |
| distance reduction | 6.9715 | 12.8817 | 12.9379 | 9.9089 |
| fraction improved | 0.7559 | 0.9727 | 0.9688 | 0.8101 |
| p50 terminal distance | 4.9705 | 0.6987 | 0.6708 | 3.2720 |
| p90 terminal distance | 15.2401 | 3.2267 | 2.9677 | 8.3182 |
| action saturation | 0.0000 | 0.0221 | 0.0157 | 0.0100 |
| action L2 | 0.0073 | 0.5404 | 0.5277 | 0.5458 |

Training diagnostics:

| Diagnostic | Value |
| --- | ---: |
| updates | 250 |
| env steps | 10.24M |
| optimizer minibatch steps | 6000 |
| final train terminal full-goal distance | 2.8243 |
| mean return per step | 1.0635 |
| policy KL | 0.0175 |
| clip fraction | 0.2253 |
| value loss | 5.3138 |
| explained variance | 0.8589 |
| action saturation | 0.0523 |
| train teacher action MAE | 0.2801 |
| NaN count | 0 |
| elapsed | 2018s |

Corrected held-target oracle rollout:

```bash
uv run python scripts/rl_reachability_goal_full_success_eval.py \
  --goal-type full \
  --goal-dim 28 \
  --episodes 100 \
  --num-envs 10 \
  --goal-sources oracle shuffled_oracle \
  --recompute-held-goal-features \
  --bc-low artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt \
  --rl-low results/incremental/rl_reachability_debug/run20_full_goal_recomputed_teacher_penalty10_b8_u250/privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt \
  --rl-low-name run20_full_goal_recomputed_teacher_penalty10_ppo \
  --output results/incremental/rl_reachability_debug/run20_full_goal_recomputed_oracle_success_100.json
```

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.72 | 0.8105 | 0.8119 | 1.8087 | 0.0414 |
| oracle | Run 20 recomputed full PPO | 0.00 | 0.1342 | 0.1690 | 4.9881 | 0.2870 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.4041 | 0.4002 |
| shuffled oracle | Run 20 recomputed full PPO | 0.00 | 0.1131 | 0.1448 | 10.6965 | 0.4034 |

Interpretation:

Increasing the teacher-action penalty from `0.5` to `1.0` slightly improves
local full-goal reachability and reduces local action saturation, but it does
not solve the full-rollout task failure. In corrected held-oracle rollout,
Phase-C full BC still dominates:

```text
Phase-C full BC: success 0.72, hold distance 1.81, teacher MAE 0.041
Run 20 full PPO: success 0.00, hold distance 4.99, teacher MAE 0.287
```

The simple teacher-action penalty is not enough. The next scoped experiment
should use the Phase-C full BC policy structurally: BC warm start, residual
policy on top of BC, or a much more conservative KL/behavior-cloning constraint.

## 2026-07-01 - Run 21: Longer Recomputed Full-State PPO

Motivation:

The user pointed out that the recomputed full-state PPO runs may simply need
much longer training before moving to fine-tuning or residual variants. Run 21
continues Run 20 for `1000` additional PPO updates, keeping the same corrected
held-target semantics and teacher-action penalty `1.0`.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 1000 \
  --horizon 10 \
  --goal-type full \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 1.0 \
  --recompute-held-goal-features \
  --init-checkpoint results/incremental/rl_reachability_debug/run20_full_goal_recomputed_teacher_penalty10_b8_u250/privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 100 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run21_full_goal_recomputed_penalty10_continue_u1000 \
  --force
```

Training budget:

| Quantity | Value |
| --- | ---: |
| additional PPO updates | 1000 |
| previous Run 20 updates | 250 |
| total PPO updates from scratch | 1250 |
| additional env steps | 40.96M |
| total env steps from scratch | 51.20M |
| additional optimizer minibatch steps | 24000 |
| total optimizer minibatch steps from scratch | 30000 |

Local fixed-bank result:

| Metric | Phase-C full BC | Run 20 | Run 21 | Run 21 shuffled |
| --- | ---: | ---: | ---: | ---: |
| terminal full-goal distance | 1.6563 | 1.4703 | 1.8744 | 4.7948 |
| p50 terminal distance | 0.1399 | 0.6708 | 0.5893 | 3.3125 |
| p90 terminal distance | 2.0872 | 2.9677 | 2.9423 | 8.3058 |
| p99 terminal distance | 29.9696 | 16.5195 | 30.0081 | 40.8741 |
| fraction improved | 0.9315 | 0.9688 | 0.9764 | 0.8072 |
| action saturation | 0.1412 | 0.0157 | 0.0096 | 0.0058 |
| action L2 | 0.7385 | 0.5277 | 0.5367 | 0.5397 |

Training-window diagnostics at the end of Run 21:

| Diagnostic | Value |
| --- | ---: |
| global env steps | 51.20M |
| train terminal full-goal distance | 0.2055 |
| train teacher action MAE | 0.1038 |
| mean return per step | 0.3363 |
| policy KL | 0.0463 |
| clip fraction | 0.3957 |
| value loss | 0.1279 |
| explained variance | 0.9826 |
| action saturation | 0.0005 |
| NaN count | 0 |
| elapsed | 8387s |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.72 | 0.8101 | 0.8115 | 1.6113 | 0.0420 |
| oracle | Run 21 long PPO | 0.04 | 0.2172 | 0.2408 | 4.4384 | 0.2387 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3987 | 0.4002 |
| shuffled oracle | Run 21 long PPO | 0.00 | 0.1187 | 0.1431 | 12.2622 | 0.4025 |

Interpretation:

Longer training helps, but not nearly enough yet. Compared with Run 20, Run 21
reduces rollout teacher-action MAE (`0.287 -> 0.239`), improves hold full-goal
distance (`4.99 -> 4.44`), and produces the first nonzero full-state PPO oracle
held success (`0.04`). The fixed-bank local mean distance gets worse because
the tail worsens, but median/P90 improve slightly.

This supports continuing the scratch PPO training before moving to fine-tuning:
the policy is not saturated. However, it remains far behind Phase-C full BC
(`0.72` success, `0.042` teacher MAE), so if an 8x-style continuation still
fails, the next step should be BC warm start or residual-on-BC.

## 2026-07-01 - Full-State Open-Loop Deployment Reachability Audit

Motivation:

The reset-bank local distances above may overstate or understate useful
reachability if PPO overfits to stored reset states. This audit rolls the
hierarchy open-loop with held subgoals, records the deployed decision states,
then branches candidate low-level policies from exactly those states toward
the oracle 10-step terminal future state.

Commands:

```bash
uv run python scripts/rl_reachability_full_bc_local_eval.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --references 8 \
  --output results/incremental/rl_reachability_debug/phasec_full_bc_local_eval_ref8.json

uv run python scripts/rl_reachability_full_deployment_reachability_eval.py \
  --config configs/pusht_incremental.yaml \
  --decisions 512 \
  --num-envs 16 \
  --output results/incremental/rl_reachability_debug/run21_full_deployment_reachability_512.json
```

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7650 | 0.8351 | 0.1555 | 1.6420 | 0.8674 |
| Phase-C full BC | Run 20 full PPO | no | 8.7650 | 0.8777 | 0.2302 | 2.3527 | 0.9298 |
| Phase-C full BC | Run 21 long PPO | no | 8.7650 | 0.8187 | 0.1688 | 2.1240 | 0.9474 |
| Run 21 long PPO | Phase-C full BC | no | 10.4716 | 10.3369 | 2.0652 | 19.6268 | 0.6777 |
| Run 21 long PPO | Run 20 full PPO | no | 10.4716 | 3.2549 | 1.2234 | 5.5571 | 0.8809 |
| Run 21 long PPO | Run 21 long PPO | no | 10.4716 | 3.5435 | 1.4857 | 6.1962 | 0.8848 |
| Phase-C full BC | Phase-C full BC | yes | 8.6000 | 8.4209 | 5.4644 | 18.9098 | 0.5088 |
| Phase-C full BC | Run 20 full PPO | yes | 8.6000 | 3.0239 | 1.9148 | 7.2498 | 0.9123 |
| Phase-C full BC | Run 21 long PPO | yes | 8.6000 | 3.1619 | 1.6356 | 6.9080 | 0.9435 |
| Run 21 long PPO | Phase-C full BC | yes | 12.5410 | 20.6092 | 12.7459 | 39.2882 | 0.3809 |
| Run 21 long PPO | Run 20 full PPO | yes | 12.5410 | 7.4669 | 5.5013 | 14.5999 | 0.8555 |
| Run 21 long PPO | Run 21 long PPO | yes | 12.5410 | 7.8267 | 5.6721 | 14.9396 | 0.8457 |

Interpretation:

On states generated by the Phase-C full BC hierarchy, Run 21 is locally
competitive with BC (`0.8187` vs `0.8351` mean terminal full-goal distance),
although BC has slightly better median/P90. On states generated by the Run 21
hierarchy itself, BC performs very poorly (`10.3369` mean terminal distance),
while the PPO policies still reduce distance to about `3.3-3.5`. This means
the PPO low level is not simply overfitting the fixed reset bank; it is better
than BC at reaching oracle full-state subgoals from its own deployed state
distribution.

The bad news is that shuffled goals are still improved substantially by PPO,
especially on BC-collected states. PPO is partly learning a generic corrective
motion prior in addition to goal-conditioned reachability. Therefore, the next
long continuation is still justified, but task success should be judged with
held-subgoal rollout metrics, not reset-bank terminal distance alone.

## 2026-07-01 - Run 22: Second Longer Recomputed Full-State PPO Continuation

Motivation:

Run 21 was not saturated and showed lower rollout teacher-action MAE than Run
20, so Run 22 continued the same corrected full-state PPO objective for another
`1000` updates. This is the stop point for same-reset-bank continuation unless
held-subgoal task success improves substantially.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5 \
  --num-envs 4096 \
  --updates 1000 \
  --horizon 10 \
  --goal-type full \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 1.0 \
  --recompute-held-goal-features \
  --init-checkpoint results/incremental/rl_reachability_debug/run21_full_goal_recomputed_penalty10_continue_u1000/privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 100 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run22_full_goal_recomputed_penalty10_continue2_u1000 \
  --force
```

Training budget:

| Quantity | Value |
| --- | ---: |
| additional PPO updates | 1000 |
| total PPO updates from Run 20 scratch init | 2250 |
| additional env steps | 40.96M |
| total env steps from Run 20 scratch init | 92.16M |
| additional optimizer minibatch steps | 24000 |
| total optimizer minibatch steps from Run 20 scratch init | 54000 |
| elapsed | 2h 19m 41s |

Local fixed-bank result:

| Metric | Phase-C full BC | Run 21 | Run 22 | Run 22 shuffled |
| --- | ---: | ---: | ---: | ---: |
| terminal full-goal distance | 1.6563 | 1.8744 | 1.5018 | 4.4356 |
| p50 terminal distance | 0.1399 | 0.5893 | 0.5154 | 3.1201 |
| p90 terminal distance | 2.0872 | 2.9423 | 2.7939 | 7.8654 |
| p99 terminal distance | 29.9696 | 30.0081 | 20.5832 | 32.7371 |
| fraction improved | 0.9315 | 0.9764 | 0.9844 | 0.8164 |
| action saturation | 0.1412 | 0.0096 | 0.0050 | 0.0053 |
| action L2 | 0.7385 | 0.5367 | 0.5382 | 0.5499 |

Training-window diagnostics at the end of Run 22:

| Diagnostic | Value |
| --- | ---: |
| train teacher action MAE | 0.0837 |
| mean return per step | 0.3624 |
| clip fraction | 0.3896 |
| explained variance | 0.9839 |
| action saturation | 0.0008 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.69 | 0.7898 | 0.7915 | 1.6297 | 0.0410 |
| oracle | Run 22 long PPO | 0.01 | 0.1540 | 0.1934 | 5.1632 | 0.2493 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3789 | 0.4002 |
| shuffled oracle | Run 22 long PPO | 0.00 | 0.1133 | 0.1458 | 8.6494 | 0.3897 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7485 | 0.8645 | 0.1353 | 1.8000 | 0.8850 |
| Phase-C full BC | Run 21 long PPO | no | 8.7485 | 0.8135 | 0.1675 | 2.2483 | 0.9591 |
| Phase-C full BC | Run 22 long PPO | no | 8.7485 | 0.7904 | 0.1597 | 2.1517 | 0.9669 |
| Run 22 long PPO | Phase-C full BC | no | 9.8928 | 8.4377 | 1.7301 | 16.3957 | 0.6816 |
| Run 22 long PPO | Run 21 long PPO | no | 9.8928 | 5.4068 | 1.3543 | 7.0409 | 0.8809 |
| Run 22 long PPO | Run 22 long PPO | no | 9.8928 | 4.1308 | 1.2215 | 5.8609 | 0.8906 |

Interpretation:

Run 22 improves same-reset-bank local reachability over Run 21 and has a much
lower final training-window teacher-action MAE (`0.0837`). This still does not
transfer to held-subgoal task success: oracle full-state success is only
`0.01`, far below Phase-C BC (`0.69`), and hold full-goal distance is worse
than both BC and Run 21. The deployed-state branch audit still argues against
pure fixed-bank overfitting, because Run 22 reaches oracle targets better than
BC from Run-22-collected states. The stronger conclusion is that additional
same-bank training is the wrong next lever.

Next action:

Follow Phase 4B of `rl_reachability_debug_plan.md`: train/evaluate full-state
PPO on a reset mixture containing original demo windows, BC-hierarchy deployed
states, and PPO-hierarchy deployed states. Treat oracle teacher branching from
deployed states as a diagnostic or upper bound, not the core proof-of-concept
method.

## 2026-07-01 - Run 23: Full-State PPO on Learned-High Reset Mixture

Motivation:

Run 22 showed that continuing on the same teacher/demo reset bank improves local
metrics but not task success. Run 23 changes the reset-state distribution:

| Source | Batches | Fraction |
| --- | ---: | ---: |
| original demo/teacher local windows | 8 | 50% |
| Phase-C full BC deployed hierarchy states | 4 | 25% |
| Run 22 PPO deployed hierarchy states | 4 | 25% |

The deployed-state targets do not use online teacher branching. They use the
learned full-state high-level predictor and convert its 28D full-goal output
into a pseudo future privileged state so the low level can still use recomputed
full-goal features. This is a POC-friendly approximation; oracle target
branching remains a diagnostic only.

Dataset:

```text
data/rl_reachability_debug/full_reset_mixture_demo8_bc4_run22_4.h5
```

Implementation changes:

- `scripts/rl_reachability_collect_full_reset_mixture.py` creates the mixture
  HDF5 file.
- `scripts/rl_reachability_privileged_tcp_ppo.py` now supports optional
  `valid_starts` and `target_future_states` datasets.
- Run 23 reuses Run 22 normalizers via `--use-init-normalizers`, so the loaded
  policy sees the same input scaling as before.

Training command:

```bash
uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
  --config configs/pusht_incremental.yaml \
  --dataset data/rl_reachability_debug/full_reset_mixture_demo8_bc4_run22_4.h5 \
  --num-envs 4096 \
  --updates 250 \
  --horizon 10 \
  --goal-type full \
  --reward-mode progress_terminal \
  --reward-distance-source true_goal \
  --teacher-action-penalty-weight 1.0 \
  --recompute-held-goal-features \
  --init-checkpoint results/incremental/rl_reachability_debug/run22_full_goal_recomputed_penalty10_continue2_u1000/privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt \
  --use-init-normalizers \
  --num-minibatches 8 \
  --update-epochs 3 \
  --checkpoint-every-updates 50 \
  --eval-episodes 8 \
  --output-dir results/incremental/rl_reachability_debug/run23_full_reset_mixture_learned_high_penalty10_u250 \
  --force
```

Mixture fixed-bank local result:

| Metric | Run 23 initial | Run 23 trained | Run 23 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 2.3732 | 2.1294 | 5.1502 |
| p50 terminal distance | 1.4899 | 1.3197 | 4.2200 |
| p90 terminal distance | 4.3697 | 3.8530 | 8.2776 |
| p99 terminal distance | 20.4742 | 17.0256 | 25.5973 |
| fraction improved | 0.9464 | 0.9600 | 0.7555 |
| action saturation | 0.0034 | 0.0033 | 0.0025 |
| action L2 | 0.5401 | 0.5639 | 0.5792 |

Training-window diagnostics at the end of Run 23:

| Diagnostic | Value |
| --- | ---: |
| train teacher action MAE | 0.2822 |
| mean return per step | 0.3080 |
| clip fraction | 0.4952 |
| explained variance | 0.8812 |
| action saturation | 0.0059 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.74 | 0.8237 | 0.8258 | 1.5759 | 0.0406 |
| oracle | Run 23 reset-mixture PPO | 0.00 | 0.1461 | 0.1892 | 4.3716 | 0.2712 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3783 | 0.4002 |
| shuffled oracle | Run 23 reset-mixture PPO | 0.01 | 0.1224 | 0.1518 | 9.4336 | 0.3960 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.6567 | 0.9078 | 0.1658 | 1.8327 | 0.8613 |
| Phase-C full BC | Run 22 long PPO | no | 8.6567 | 0.8008 | 0.1618 | 2.1821 | 0.9512 |
| Phase-C full BC | Run 23 reset-mixture PPO | no | 8.6567 | 0.7509 | 0.1987 | 1.9158 | 0.9297 |
| Run 23 reset-mixture PPO | Phase-C full BC | no | 9.2914 | 6.6453 | 1.7248 | 13.6863 | 0.6855 |
| Run 23 reset-mixture PPO | Run 22 long PPO | no | 9.2914 | 1.9530 | 1.0845 | 3.4902 | 0.9180 |
| Run 23 reset-mixture PPO | Run 23 reset-mixture PPO | no | 9.2914 | 1.8358 | 1.0318 | 3.2796 | 0.9219 |

Interpretation:

The reset mixture does what it was meant to do locally: Run 23 improves
terminal full-goal distance on the mixture bank, and deployed-state branch
distance on its own rollout distribution is much better than BC and better than
Run 22. However, task success remains zero and teacher-action MAE remains high.

This means reset coverage alone is not sufficient in this formulation. The
next diagnostic should isolate target quality: train or evaluate the same
reset-mixture idea with oracle target future states from deployed resets as an
upper bound. If oracle-target reset mixture still fails, the bottleneck is more
likely action/contact compatibility or the PPO objective. If it succeeds, the
learned-high 28D-goal-to-pseudo-future-state target construction is the main
problem.

## 2026-07-01 - Run 24: Oracle-Target Reset-Mixture Diagnostic

Motivation:

Run 23 used deployed reset states but learned-high pseudo future targets. Run
24 is an upper-bound diagnostic: use teacher/oracle branches to label deployed
reset states with the actual reachable `t+10` future state. This is not the
main POC method, but it separates target-quality failure from reset-distribution
failure.

Dataset:

```text
data/rl_reachability_debug/full_reset_mixture_oracle_demo4_bc2_run22_2.h5
```

Mixture:

| Source | Batches | Fraction |
| --- | ---: | ---: |
| original demo/teacher local windows | 4 | 50% |
| Phase-C full BC deployed hierarchy states | 2 | 25% |
| Run 22 PPO deployed hierarchy states | 2 | 25% |

Oracle-target mixture fixed-bank local result:

| Metric | Run 24 initial | Run 24 trained | Run 24 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 2.5107 | 2.5682 | 6.3283 |
| p50 terminal distance | 1.2624 | 1.2992 | 4.6700 |
| p90 terminal distance | 3.8567 | 3.9914 | 10.1929 |
| p99 terminal distance | 32.9147 | 35.1254 | 46.9478 |
| fraction improved | 0.9640 | 0.9660 | 0.7339 |
| action saturation | 0.0036 | 0.0153 | 0.0227 |
| action L2 | 0.5374 | 0.5625 | 0.5937 |

Training-window diagnostics at the end of Run 24:

| Diagnostic | Value |
| --- | ---: |
| train teacher action MAE | 0.3367 |
| mean return per step | 0.1393 |
| clip fraction | 0.4868 |
| explained variance | 0.8088 |
| action saturation | 0.0381 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.74 | 0.8234 | 0.8251 | 1.5749 | 0.0411 |
| oracle | Run 24 oracle-target reset PPO | 0.03 | 0.1591 | 0.1953 | 3.0692 | 0.2519 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.4141 | 0.4002 |
| shuffled oracle | Run 24 oracle-target reset PPO | 0.00 | 0.1137 | 0.1449 | 12.0599 | 0.4193 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7790 | 0.9057 | 0.1609 | 1.8721 | 0.8643 |
| Phase-C full BC | Run 22 long PPO | no | 8.7790 | 0.8075 | 0.1752 | 2.1652 | 0.9554 |
| Phase-C full BC | Run 24 oracle-target reset PPO | no | 8.7790 | 0.8986 | 0.1860 | 2.3411 | 0.9419 |
| Run 24 oracle-target reset PPO | Phase-C full BC | no | 8.1222 | 4.7505 | 1.6733 | 11.1948 | 0.6230 |
| Run 24 oracle-target reset PPO | Run 22 long PPO | no | 8.1222 | 2.4427 | 1.2820 | 4.1023 | 0.8672 |
| Run 24 oracle-target reset PPO | Run 24 oracle-target reset PPO | no | 8.1222 | 2.1048 | 1.0311 | 3.8327 | 0.9082 |

Interpretation:

Oracle targets improve held-rollout full-goal distance over Run 23
(`4.3716 -> 3.0692`) and give slightly higher oracle success (`0.00 -> 0.03`),
but the policy remains far below Phase-C full BC (`0.74`). This means target
quality matters, but it is not the whole problem. The stronger remaining
failure is action/contact compatibility: Run 24 still has high teacher-action
MAE and very low task reward despite better full-goal distance.

Next action:

Do not continue plain scratch PPO on reset mixtures as the main line. The next
useful full-state experiment should constrain the policy structurally toward
Phase-C BC, for example BC warm start, residual-on-BC, or a KL/BC-prior
regularizer. A stronger scalar teacher-action penalty is possible as a quick
ablation, but the previous results suggest simple penalties are a weak tool.

## 2026-07-01 - Run 25: BC-Warm-Started Reset-Mixture Full-State PPO

Motivation:

Run 22 showed that longer same-bank PPO improves some local metrics but does
not transfer to held-subgoal task success. Runs 23 and 24 showed that deployed
reset coverage and oracle targets help local/deployed reachability, but plain
scratch PPO remains far off the action/contact behavior of Phase-C full BC.
Run 25 keeps the reset-mixture direction but initializes the trainable PPO
actor from the Phase-C full BC low-level policy.

This keeps the experiment aligned with the POC constraint: deployed reset
states are used to address distribution shift, while oracle branching is only
used for diagnostics. The core training target is still the hierarchy's
available held full-state subgoal, not DAgger-style online expert action
relabeling.

Dataset:

```text
data/rl_reachability_debug/full_reset_mixture_demo8_bc4_run22_4.h5
```

Mixture:

| Source | Batches | Fraction |
| --- | ---: | ---: |
| original demo/teacher local windows | 8 | 50% |
| Phase-C full BC deployed hierarchy states | 4 | 25% |
| Run 22 PPO deployed hierarchy states | 4 | 25% |

BC warm start:

| Field | Value |
| --- | ---: |
| checkpoint | `artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt` |
| warm-start steps | 1000 |
| batch size | 8192 |
| learning rate | 0.001 |
| initial action MSE to BC | 0.2371 |
| final action MSE to BC | 0.0022 |

Reset-mixture fixed-bank local result:

| Metric | Run 25 initial | Run 25 trained | Run 25 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 5.3252 | 3.6321 | 8.8288 |
| p50 terminal distance | 1.5708 | 1.7027 | 6.7154 |
| p90 terminal distance | 12.1632 | 6.9932 | 15.6308 |
| p99 terminal distance | 59.8115 | 35.4761 | 47.8307 |
| fraction improved | 0.7678 | 0.8653 | 0.5837 |
| action saturation | 0.1384 | 0.0445 | 0.0359 |
| action L2 | 0.7965 | 0.6843 | 0.6340 |

Training-window diagnostics at the end of Run 25:

| Diagnostic | Value |
| --- | ---: |
| train teacher action MAE | 0.1090 |
| mean return per step | 0.4181 |
| clip fraction | 0.1271 |
| policy KL | 0.0095 |
| explained variance | 0.9609 |
| action saturation | 0.0023 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.72 | 0.8094 | 0.8115 | 1.7071 | 0.0421 |
| oracle | Run 25 BC-warm-start reset-mixture PPO | 0.16 | 0.3941 | 0.4071 | 2.3472 | 0.1314 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3954 | 0.4002 |
| shuffled oracle | Run 25 BC-warm-start reset-mixture PPO | 0.02 | 0.1439 | 0.1755 | 11.2030 | 0.3397 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.9909 | 0.8697 | 0.1672 | 1.5919 | 0.8583 |
| Phase-C full BC | Run 22 long PPO | no | 8.9909 | 0.8122 | 0.1750 | 2.1522 | 0.9553 |
| Phase-C full BC | Run 25 BC-warm-start PPO | no | 8.9909 | 1.0739 | 0.3142 | 2.3017 | 0.9029 |
| Run 25 BC-warm-start PPO | Phase-C full BC | no | 8.0484 | 2.9800 | 0.6122 | 6.7456 | 0.7698 |
| Run 25 BC-warm-start PPO | Run 22 long PPO | no | 8.0484 | 1.1469 | 0.3602 | 2.6900 | 0.9768 |
| Run 25 BC-warm-start PPO | Run 25 BC-warm-start PPO | no | 8.0484 | 2.0967 | 0.9046 | 4.0808 | 0.8801 |

Interpretation:

Run 25 is the first full-state PPO variant in this thread with meaningful
held-subgoal task recovery (`0.16` oracle success versus `0.00-0.04` for the
plain full-state PPO/reset-mixture variants). It also has much lower
teacher-action MAE than Runs 23/24. However, it remains far below Phase-C full
BC (`0.72` on the same 100-episode oracle bank), and its deployed-state branch
reachability is not uniformly better than Run 22.

The evidence now supports two constraints for the next experiments:

1. Do not return to simply training longer on the original demo reset bank.
   The low-level must see shifted deployed hierarchy states.
2. Do not use online expert action relabeling as the main POC path. Oracle
   targets/branches are useful as diagnostics or upper bounds, but the main
   training method should use reset mixtures, disturbed resets, BC priors,
   residual-on-BC, or KL-to-BC structure.

Next action:

Prefer a stronger BC-structured reset-distribution experiment over another
plain scratch continuation. The most direct candidates are residual-on-Phase-C
BC PPO on the same reset mixture, an explicit KL/BC-prior regularizer on the
PPO actor, or a disturbed-reset variant that trains recovery around demo and
deployed states without requiring new online expert action labels.

## 2026-07-01 - Run 26: BC-Prior Reset-Mixture Full-State PPO

Motivation:

Run 25 showed that BC warm start helps but does not keep the PPO actor close
enough to the Phase-C full BC action/contact manifold. Run 26 keeps the same
reset-mixture data and BC warm start, removes the online teacher-action penalty,
and adds an offline BC-prior loss during PPO updates:

```text
loss = PPO loss + bc_prior_weight * MSE(actor_mean(condition), Phase-C-BC(condition))
```

The BC prior uses the fixed Phase-C full BC checkpoint and the same state,
full-state goal, previous-action, and remaining-time condition as PPO. It does
not query an online expert.

Reset-mixture fixed-bank local result:

| Metric | Run 26 initial | Run 26 trained | Run 26 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 5.3252 | 3.6656 | 8.6369 |
| p50 terminal distance | 1.5708 | 1.6724 | 6.5977 |
| p90 terminal distance | 12.1632 | 7.2180 | 15.1600 |
| fraction improved | 0.7678 | 0.8568 | 0.5936 |
| action saturation | 0.1384 | 0.0362 | 0.0362 |
| action L2 | 0.7965 | 0.6985 | 0.6628 |

Training-window diagnostics at the end of Run 26:

| Diagnostic | Value |
| --- | ---: |
| BC-prior weight | 1.0 |
| train BC-prior MSE | 0.0123 |
| mean return per step | 0.5182 |
| clip fraction | 0.1401 |
| policy KL | 0.0102 |
| action saturation | 0.0032 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.74 | 0.8242 | 0.8256 | 1.6095 | 0.0394 |
| oracle | Run 26 BC-prior reset-mixture PPO | 0.21 | 0.4302 | 0.4397 | 2.7468 | 0.1241 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3951 | 0.4002 |
| shuffled oracle | Run 26 BC-prior reset-mixture PPO | 0.00 | 0.1387 | 0.1606 | 10.6828 | 0.3596 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.8494 | 0.8616 | 0.1661 | 1.7163 | 0.8594 |
| Phase-C full BC | Run 22 long PPO | no | 8.8494 | 0.7998 | 0.1695 | 2.1502 | 0.9609 |
| Phase-C full BC | Run 25 BC-warm-start PPO | no | 8.8494 | 1.0737 | 0.3155 | 2.2884 | 0.9121 |
| Phase-C full BC | Run 26 BC-prior PPO | no | 8.8494 | 0.9837 | 0.2861 | 2.0272 | 0.9316 |
| Run 26 BC-prior PPO | Phase-C full BC | no | 7.2687 | 2.5858 | 0.5424 | 6.0685 | 0.7615 |
| Run 26 BC-prior PPO | Run 22 long PPO | no | 7.2687 | 1.0501 | 0.3572 | 2.5966 | 0.9596 |
| Run 26 BC-prior PPO | Run 25 BC-warm-start PPO | no | 7.2687 | 1.7256 | 0.7293 | 4.2324 | 0.8827 |
| Run 26 BC-prior PPO | Run 26 BC-prior PPO | no | 7.2687 | 1.6019 | 0.8024 | 3.9200 | 0.8981 |

Interpretation:

Run 26 improves held-subgoal success over Run 25 (`0.16 -> 0.21`) and keeps
shuffled-goal success at zero, so the BC-prior direction is useful. It still
does not approach the Phase-C full BC baseline (`0.74`) and its own deployed
states remain a harder distribution than the original demo/BC reset bank.

Next action:

Move from a static reset mixture to iterative reset-bank aggregation:

1. Start from expert/demo local windows plus Phase-C BC deployed trajectories.
2. Train/continue PPO from the BC-structured checkpoint.
3. Deploy the learned low level with the learned high-level policy.
4. Record those deployed hierarchy states and the hierarchy's own held
   full-state subgoals.
5. Add them to the reset bank and continue training.
6. Repeat for multiple aggregation rounds.

This is not DAgger-style online expert relabeling. The main labels are the
subgoals already produced by the hierarchy. Oracle branches may still be used
only as diagnostics or upper bounds.

## 2026-07-01 - Run 27: Iterative Reset Aggregation Round 1

Motivation:

Run 26 still trailed Phase-C BC. Per the reset-distribution hypothesis, Run 27
starts the iterative aggregation loop: collect reset states from the current
learned hierarchy, add them to the reset bank, and continue BC-structured PPO
from the previous checkpoint.

Dataset:

```text
data/rl_reachability_debug/full_reset_agg_round1_demo8_bc4_run26_4.h5
```

Mixture:

| Source | Batches |
| --- | ---: |
| original demo/teacher local windows | 8 |
| Phase-C full BC deployed hierarchy states | 4 |
| Run 26 BC-prior PPO deployed hierarchy states | 4 |

The deployed targets use the learned high-level full-state subgoal converted to
a pseudo future state. No online expert action labels are used.

Training setup:

| Field | Value |
| --- | --- |
| init checkpoint | Run 26 latest |
| normalizers | Run 26 latest |
| reward | true full-goal progress + terminal |
| regularization | BC-prior loss, weight 1.0 |
| PPO updates | 250 |
| envs | 4096 |

Round-1 aggregation-bank local result:

| Metric | Run 27 initial | Run 27 trained | Run 27 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 2.5236 | 1.8723 | 7.1062 |
| p50 terminal distance | 0.9220 | 0.5393 | 5.3328 |
| p90 terminal distance | 4.3332 | 3.0827 | 13.1209 |
| fraction improved | 0.9368 | 0.9578 | 0.6475 |
| action saturation | 0.0440 | 0.0665 | 0.0499 |
| action L2 | 0.6367 | 0.6489 | 0.6364 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.74 | 0.8235 | 0.8251 | 1.6095 | 0.0425 |
| oracle | Run 27 iterative aggregation PPO | 0.06 | 0.3079 | 0.3228 | 2.0819 | 0.1326 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3946 | 0.4002 |
| shuffled oracle | Run 27 iterative aggregation PPO | 0.00 | 0.1325 | 0.1544 | 10.3694 | 0.3570 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7948 | 0.9012 | 0.1849 | 1.7518 | 0.8779 |
| Phase-C full BC | Run 26 BC-prior PPO | no | 8.7948 | 1.0211 | 0.3279 | 2.1136 | 0.9419 |
| Phase-C full BC | Run 27 iterative aggregation PPO | no | 8.7948 | 0.7323 | 0.2451 | 1.4923 | 0.9419 |
| Run 27 iterative aggregation PPO | Phase-C full BC | no | 7.6443 | 2.5061 | 0.5853 | 5.1988 | 0.7793 |
| Run 27 iterative aggregation PPO | Run 26 BC-prior PPO | no | 7.6443 | 1.7118 | 0.8763 | 3.6595 | 0.8791 |
| Run 27 iterative aggregation PPO | Run 27 iterative aggregation PPO | no | 7.6443 | 1.3109 | 0.6035 | 3.0399 | 0.9021 |

Interpretation:

The iterative reset-bank idea works for the local metric: Run 27 improves
terminal full-goal distance on the new aggregation bank and on deployed-state
branch audits, including its own rollout distribution. However, held-oracle
task success regresses from Run 26 (`0.21 -> 0.06`) despite a better hold-goal
distance (`2.7468 -> 2.0819`).

This is important evidence: more in-distribution reset states improve geometric
reachability but can still damage task-compatible contact behavior. The next
aggregation round should not simply add more Run 27 states and continue with
the same objective. It should add a stronger action/contact constraint, such
as residual-on-Phase-C-BC PPO or a stronger BC-prior/KL schedule, while keeping
the iterative reset-bank collection mechanism.

## 2026-07-02 - Run 28: Iterative Aggregation Round 1 With Stronger BC Prior

Motivation:

Run 27 improved geometric reachability but regressed task success. Run 28 tests
whether the first aggregation bank needs a stronger action/contact constraint:
same dataset and Run 26 initialization as Run 27, but increase
`bc_prior_weight` from `1.0` to `5.0`.

Round-1 aggregation-bank local result:

| Metric | Run 28 initial | Run 28 trained | Run 28 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 2.5236 | 1.9199 | 7.4422 |
| p50 terminal distance | 0.9220 | 0.5005 | 5.5369 |
| p90 terminal distance | 4.3332 | 3.5163 | 13.7794 |
| fraction improved | 0.9368 | 0.9391 | 0.6330 |
| action saturation | 0.0440 | 0.0802 | 0.0500 |
| action L2 | 0.6367 | 0.6808 | 0.6588 |

Training-window diagnostics at the end of Run 28:

| Diagnostic | Value |
| --- | ---: |
| BC-prior weight | 5.0 |
| train BC-prior MSE | 0.0093 |
| mean return per step | 0.6839 |
| clip fraction | 0.1749 |
| policy KL | 0.0135 |
| action saturation | 0.1820 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.66 | 0.7682 | 0.7706 | 1.7542 | 0.0423 |
| oracle | Run 28 iterative aggregation BC-prior-5 PPO | 0.25 | 0.4756 | 0.4836 | 1.4907 | 0.0870 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3820 | 0.4002 |
| shuffled oracle | Run 28 iterative aggregation BC-prior-5 PPO | 0.01 | 0.1391 | 0.1649 | 10.3251 | 0.3398 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7160 | 0.9091 | 0.1657 | 1.7433 | 0.8721 |
| Phase-C full BC | Run 26 BC-prior PPO | no | 8.7160 | 0.9802 | 0.2900 | 2.0345 | 0.9341 |
| Phase-C full BC | Run 27 iterative aggregation PPO | no | 8.7160 | 0.6992 | 0.2085 | 1.5093 | 0.9360 |
| Phase-C full BC | Run 28 iterative aggregation BC-prior-5 PPO | no | 8.7160 | 0.7426 | 0.2221 | 1.4575 | 0.9031 |
| Run 28 iterative aggregation BC-prior-5 PPO | Phase-C full BC | no | 7.5663 | 2.3780 | 0.3135 | 5.6223 | 0.7457 |
| Run 28 iterative aggregation BC-prior-5 PPO | Run 26 BC-prior PPO | no | 7.5663 | 1.6553 | 0.4794 | 3.3181 | 0.8632 |
| Run 28 iterative aggregation BC-prior-5 PPO | Run 27 iterative aggregation PPO | no | 7.5663 | 1.2134 | 0.3605 | 2.5594 | 0.8902 |
| Run 28 iterative aggregation BC-prior-5 PPO | Run 28 iterative aggregation BC-prior-5 PPO | no | 7.5663 | 1.4456 | 0.3571 | 2.9417 | 0.8825 |

Interpretation:

Run 28 is the strongest full-state PPO result so far: oracle held success
improves to `0.25`, teacher-action MAE drops to `0.0870`, and shuffled-goal
success remains near zero. The stronger BC prior fixes much of Run 27's task
success regression while retaining the local/deployed reachability gains from
the aggregation bank.

The policy is still below Phase-C full BC on task success, so the iterative
reset-bank direction should continue, but with the stronger BC prior (or a
residual-on-BC formulation) rather than the weaker Run 27 objective.

## 2026-07-02 - Run 29: Iterative Aggregation Round 2 With Stronger BC Prior

Motivation:

Run 28 was the best full-state PPO result so far, so Run 29 applies one more
iteration of the user's proposed aggregation loop: collect states from deploying
Run 28 in the hierarchy, add those states to a new reset bank, and continue
training from Run 28 with `bc_prior_weight=5.0`.

Dataset:

```text
data/rl_reachability_debug/full_reset_agg_round2_demo8_bc4_run28_4.h5
```

Mixture:

| Source | Batches |
| --- | ---: |
| original demo/teacher local windows | 8 |
| Phase-C full BC deployed hierarchy states | 4 |
| Run 28 BC-prior-5 PPO deployed hierarchy states | 4 |

Round-2 aggregation-bank local result:

| Metric | Run 29 initial | Run 29 trained | Run 29 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 1.6456 | 1.6893 | 6.9968 |
| p50 terminal distance | 0.3542 | 0.3432 | 5.0104 |
| p90 terminal distance | 2.6809 | 2.7063 | 13.7057 |
| fraction improved | 0.9336 | 0.9424 | 0.6466 |
| action saturation | 0.0799 | 0.0920 | 0.0540 |
| action L2 | 0.6588 | 0.6520 | 0.6311 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.72 | 0.8103 | 0.8118 | 1.6403 | 0.0404 |
| oracle | Run 29 iterative aggregation round-2 PPO | 0.25 | 0.4739 | 0.4810 | 1.7237 | 0.1032 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.4139 | 0.4002 |
| shuffled oracle | Run 29 iterative aggregation round-2 PPO | 0.00 | 0.1361 | 0.1564 | 11.2801 | 0.3579 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.8953 | 0.9622 | 0.1676 | 1.8047 | 0.8824 |
| Phase-C full BC | Run 28 BC-prior-5 PPO | no | 8.8953 | 0.8528 | 0.2410 | 1.6351 | 0.9070 |
| Phase-C full BC | Run 29 round-2 PPO | no | 8.8953 | 0.7233 | 0.2241 | 1.4255 | 0.9526 |
| Run 29 round-2 PPO | Phase-C full BC | no | 8.0961 | 2.7721 | 0.5745 | 5.2880 | 0.7390 |
| Run 29 round-2 PPO | Run 28 BC-prior-5 PPO | no | 8.0961 | 1.5585 | 0.5494 | 2.7619 | 0.8419 |
| Run 29 round-2 PPO | Run 29 round-2 PPO | no | 8.0961 | 1.4397 | 0.4837 | 2.3993 | 0.8914 |

Interpretation:

Round 2 improves deployed-state branch reachability over Run 28, but it does
not improve held-subgoal task success (`0.25 -> 0.25`) and worsens
teacher-action MAE (`0.0870 -> 0.1032`). This suggests the iterative reset-bank
mechanism is useful for distribution coverage, but repeating it with the same
direct PPO objective is now saturating.

Next action:

Do not keep adding aggregation rounds with the same direct-action PPO objective.
The next useful experiment should keep the aggregated reset banks but change
the policy parameterization/constraint, especially residual-on-Phase-C-BC PPO
or a more explicit KL-to-BC objective. The target is to preserve the Run
28/29 deployed-state reachability gains while closing the remaining task-success
gap to Phase-C full BC.

## 2026-07-02 - Run 30: Residual-on-BC PPO on Round-2 Aggregation Bank

Motivation:

Run 29 showed that repeating direct-action PPO aggregation improves
deployed-state reachability but does not improve task success. Run 30 changes
the policy parameterization to preserve the Phase-C full BC action/contact
manifold:

```text
action = Phase-C-BC(condition) + alpha * tanh(residual_policy(condition))
alpha = 0.15
residual_penalty_weight = 0.01
```

The residual policy starts near zero residual, so the initial controller is
approximately Phase-C full BC. No online expert action labels are used.

Dataset:

```text
data/rl_reachability_debug/full_reset_agg_round2_demo8_bc4_run28_4.h5
```

Round-2 aggregation-bank local result:

| Metric | Run 30 initial | Run 30 trained | Run 30 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 1.9494 | 1.9028 | 12.1846 |
| p50 terminal distance | 0.2527 | 0.2795 | 8.6451 |
| p90 terminal distance | 3.1583 | 3.1486 | 23.9891 |
| fraction improved | 0.8625 | 0.8708 | 0.5034 |
| action saturation | 0.0995 | 0.0898 | 0.0668 |
| action L2 | 0.7245 | 0.7189 | 0.7449 |
| residual L2 | 0.0013 | 0.0294 | 0.0307 |

Training-window diagnostics at the end of Run 30:

| Diagnostic | Value |
| --- | ---: |
| mean terminal full-goal distance | 1.2088 |
| residual L2 mean | 0.0356 |
| residual penalty mean | 0.0005 |
| mean return per step | 0.6447 |
| clip fraction | 0.0358 |
| policy KL | 0.0035 |
| NaN count | 0 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.68 | 0.7833 | 0.7852 | 1.6737 | 0.0422 |
| oracle | Run 30 residual-on-BC PPO | 0.73 | 0.8152 | 0.8166 | 2.7548 | 0.0513 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.4001 | 0.4002 |
| shuffled oracle | Run 30 residual-on-BC PPO | 0.00 | 0.1299 | 0.1672 | 27.7202 | 0.3966 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7709 | 0.9136 | 0.1579 | 1.7506 | 0.8733 |
| Phase-C full BC | Run 29 round-2 PPO | no | 8.7709 | 0.6506 | 0.2012 | 1.3600 | 0.9415 |
| Phase-C full BC | Run 30 residual-on-BC PPO | no | 8.7709 | 1.0805 | 0.1780 | 2.0009 | 0.8772 |
| Run 30 residual-on-BC PPO | Phase-C full BC | no | 9.3221 | 1.2460 | 0.2036 | 2.3291 | 0.8281 |
| Run 30 residual-on-BC PPO | Run 29 round-2 PPO | no | 9.3221 | 0.9670 | 0.2088 | 1.6343 | 0.9180 |
| Run 30 residual-on-BC PPO | Run 30 residual-on-BC PPO | no | 9.3221 | 1.6468 | 0.2095 | 2.3214 | 0.8516 |

Interpretation:

Run 30 is the best full-state task-success result so far: `0.73` oracle held
success, slightly above the Phase-C full BC baseline on the same eval bank
(`0.68`), with shuffled-goal success at zero. This supports the hypothesis that
the low-level should preserve BC contact behavior while learning small
goal-conditioned corrections.

However, Run 30 does not dominate Run 29 on terminal full-goal distance in the
deployed-state branch audit. That means the earlier direct PPO policies learned
stronger geometric reachability, while residual-on-BC preserved task-compatible
contact behavior. The next useful experiment should try to recover more of
Run 29's geometric reachability without losing Run 30's task success, for
example with a larger residual radius (`alpha=0.25`) or a smaller residual
penalty, while keeping the residual-on-BC parameterization.

## 2026-07-02 - Run 31: Residual-on-BC With Larger Residual Radius

Motivation:

Run 30 reached BC-level task success but did not dominate direct PPO on
deployed-state geometric reachability. Run 31 keeps the same round-2
aggregation bank and residual penalty, but increases the residual radius:

```text
alpha: 0.15 -> 0.25
residual_penalty_weight: 0.01
```

Round-2 aggregation-bank local result:

| Metric | Run 31 initial | Run 31 trained | Run 31 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 1.9132 | 2.2360 | 11.9395 |
| p50 terminal distance | 0.2519 | 0.3048 | 8.4822 |
| p90 terminal distance | 3.1179 | 3.5195 | 23.6102 |
| fraction improved | 0.8638 | 0.8676 | 0.5081 |
| action saturation | 0.1026 | 0.0621 | 0.0548 |
| action L2 | 0.7244 | 0.7119 | 0.7390 |
| residual L2 | 0.0022 | 0.0435 | 0.0486 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.73 | 0.8159 | 0.8175 | 1.6519 | 0.0430 |
| oracle | Run 31 residual-on-BC alpha-0.25 PPO | 0.65 | 0.7635 | 0.7648 | 1.7816 | 0.0498 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3897 | 0.4002 |
| shuffled oracle | Run 31 residual-on-BC alpha-0.25 PPO | 0.00 | 0.1230 | 0.1621 | 24.2242 | 0.4022 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.7782 | 0.9490 | 0.1741 | 1.7580 | 0.8660 |
| Phase-C full BC | Run 30 residual-on-BC PPO | no | 8.7782 | 1.1371 | 0.1825 | 2.0638 | 0.8602 |
| Phase-C full BC | Run 31 residual alpha-0.25 PPO | no | 8.7782 | 0.8724 | 0.1867 | 1.8268 | 0.8621 |
| Run 31 residual alpha-0.25 PPO | Phase-C full BC | no | 8.3151 | 1.2846 | 0.2283 | 2.5737 | 0.8438 |
| Run 31 residual alpha-0.25 PPO | Run 30 residual-on-BC PPO | no | 8.3151 | 1.3896 | 0.2424 | 2.4122 | 0.8571 |
| Run 31 residual alpha-0.25 PPO | Run 31 residual alpha-0.25 PPO | no | 8.3151 | 1.2273 | 0.2969 | 2.3063 | 0.8781 |

Interpretation:

Increasing `alpha` to `0.25` recovers some deployed-state branch reachability
relative to Run 30, but it loses the main win: held-oracle task success drops
from `0.73` to `0.65`. The smaller Run 30 residual radius is the better
task-success setting. If more reachability is needed, the next ablation should
change the penalty or training schedule more gently instead of simply allowing
larger residuals.

## 2026-07-02 - Run 32: Residual-on-BC With No Residual Penalty

Motivation:

Run 31 showed that increasing residual radius to `0.25` hurts task success.
Run 32 returns to Run 30's smaller residual radius but removes the residual
penalty:

```text
alpha = 0.15
residual_penalty_weight: 0.01 -> 0.0
```

Round-2 aggregation-bank local result:

| Metric | Run 32 initial | Run 32 trained | Run 32 shuffled |
| --- | ---: | ---: | ---: |
| terminal full-goal distance | 1.9494 | 1.9913 | 12.3200 |
| p50 terminal distance | 0.2527 | 0.2727 | 8.6975 |
| p90 terminal distance | 3.1583 | 3.1441 | 24.2540 |
| fraction improved | 0.8625 | 0.8701 | 0.5009 |
| action saturation | 0.0995 | 0.1026 | 0.0752 |
| action L2 | 0.7245 | 0.7205 | 0.7459 |
| residual L2 | 0.0013 | 0.0275 | 0.0292 |

Corrected held-target oracle rollout:

| Goal source | Low-level policy | Success | Final reward | Max reward | Hold full-goal distance | Teacher action MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle | Phase-C time-conditioned full BC | 0.70 | 0.7969 | 0.7984 | 1.4735 | 0.0397 |
| oracle | Run 32 residual-on-BC no-penalty PPO | 0.69 | 0.7907 | 0.7916 | 1.6898 | 0.0421 |
| shuffled oracle | Phase-C time-conditioned full BC | 0.00 | 0.1266 | 0.1649 | 26.3827 | 0.4002 |
| shuffled oracle | Run 32 residual-on-BC no-penalty PPO | 0.00 | 0.1304 | 0.1655 | 24.4013 | 0.3902 |

Open-loop deployed-state terminal full-goal distance:

| Collector rollout | Candidate branch | Shuffled | Initial dist. | Terminal dist. | P50 | P90 | Improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-C full BC | Phase-C full BC | no | 8.6750 | 0.8705 | 0.1613 | 1.7297 | 0.8353 |
| Phase-C full BC | Run 30 residual-on-BC PPO | no | 8.6750 | 0.9644 | 0.1807 | 1.9803 | 0.8508 |
| Phase-C full BC | Run 32 residual no-penalty PPO | no | 8.6750 | 0.8580 | 0.1781 | 1.8485 | 0.8450 |
| Run 32 residual no-penalty PPO | Phase-C full BC | no | 9.9301 | 2.0886 | 0.2025 | 2.5489 | 0.8519 |
| Run 32 residual no-penalty PPO | Run 30 residual-on-BC PPO | no | 9.9301 | 2.3258 | 0.1971 | 2.3788 | 0.8577 |
| Run 32 residual no-penalty PPO | Run 32 residual no-penalty PPO | no | 9.9301 | 2.0907 | 0.1861 | 2.2640 | 0.8577 |

Interpretation:

Removing the residual penalty does not improve the important metrics. It lowers
held-oracle task success relative to Run 30 (`0.73 -> 0.69`) and does not
improve round-2 local reachability. Run 30 remains the best residual-on-BC
setting from this set of ablations.
