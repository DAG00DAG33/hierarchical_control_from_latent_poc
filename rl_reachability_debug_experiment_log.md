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
