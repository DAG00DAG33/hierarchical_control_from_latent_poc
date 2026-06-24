# RL Rerun Final Results

This report summarizes the low-level RL rerun from
`low_level_rl_rerun_state_parallel_plan.md`. The chronological execution log is
`rl_rerun_experiment_log.md`; this file is the compact results view.

## Main Result

The best low-level RL variant was direct deterministic low-level last-layer
tuning:

```text
R3 direct last-layer, N=500, lr=1e-5, bc_weight=1.0
```

It improved paired 100-episode closed-loop Push-T success on the two serious
policy seeds tested:

| Policy seed | Selected checkpoint | Frozen success | Tuned success | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.34 | 0.38 | +0.04 |
| 1 | 614400 | 0.39 | 0.40 | +0.01 |

This is a positive signal, but it does not pass the original full-hierarchy
gate of `+0.10` success.

## Required Run Facts

| Item | Value |
| --- | --- |
| Simulator backend | ManiSkill CUDA PhysX (`physx_cuda`) |
| Main RL environment count | `4096` parallel vector envs |
| Local rollout length | `10` steps |
| Effective PPO batch | `4096 x 10 = 40960` samples/update |
| Serious RL budget | `1,024,000` transitions per main run |
| Action space | `pd_ee_delta_pos`, 3D continuous |
| Local goal horizon | `10` simulator steps |
| Task reward used in training | No |
| Task success used in training | No |
| Object pose/task progress used in training | No |
| Training reward | latent progress minus terminal latent distance |
| Exact local resets | Passed on vector-consistent corpora |
| Termination/GAE handling | One complete local episode per rollout; GAE terminates at the 10-step segment boundary |
| GPU memory | Not captured as a numeric time-series; `4096` envs was stable, `8192` failed camera-group allocation |
| Wall-clock | Not stored in JSON for the serious RL runs; this is an instrumentation gap |

## Data And Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/rl_rerun/pusht_state_demos.h5` | 1200 single-env state-loadable teacher trajectories |
| `data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5` | main 4096-env exact-reset RL corpus |
| `data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5` | independent 4096-env validation corpus |
| `data/rl_rerun/pusht_vector_state_demos_n512_val_b1.h5` | cheap fixed checkpoint-selection corpus |
| `rl_rerun_vector_state_audit_n4096_b2.json` | exact replay audit for the main corpus |
| `rl_rerun_throughput_rollout10_large.csv` | 10-step throughput benchmark |
| `rl_rerun_failure_videos/` | paired frozen/tuned deployment videos for the best R3 checkpoint |

The single-env corpus replays exactly in a single-env CUDA simulator, but
single-env intermediate states are not vector-reset equivalent. Serious local
RL therefore uses vector-consistent corpora collected with the same vector width
used for reset/replay.

The main `4096`-env RL corpus is already disjoint from the supervised low-level
BC demonstrations by reset seed: the BC demos use `920001-921498`, while the
main RL corpus uses vector batches seeded at `9800000` and `9800001`. The
`512`-env disjoint-state R1 row below is an extra independent-state ablation,
not the only disjoint-state RL test; its PPO batch is much smaller
(`512 x 10 = 5120`) than the serious runs (`4096 x 10 = 40960`).

## Supervised Baselines

The rerun retrained the VAE/high/low hierarchy from the regenerated data.

| N trajectories | Seeds | Closed-loop success mean | Sample SD |
| ---: | ---: | ---: | ---: |
| 500 | 3 | 0.280 | 0.036 |
| 1000 | 3 | 0.457 | 0.025 |

These are supervised frozen-hierarchy baselines, not RL-tuned results.

## RL Results

All main RL rows below use exact 10-step local resets. Closed-loop deployment
uses paired 100-episode evaluation seeds.

| Method | N | Main envs | Steps | Best local final distance | Closed-loop success | Delta vs frozen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen deterministic low level | 500 | n/a | 0 | 0.6073 | 0.34 | 0.00 |
| R1 residual deterministic, best task checkpoint | 500 | 4096 | 1.024M | 0.6083 | 0.35 | +0.01 |
| R1 disjoint-state ablation | 500 | 512 | 1.024M | 0.5958 | 0.30 | -0.04 |
| R2 residual flow low level | 500 | 4096 | 1.024M | 0.6267 | 0.23 | -0.05 vs flow base |
| R3 direct last-layer, lr=3e-5 | 500 | 4096 | 1.024M | 0.5851 | 0.29 at local-best | -0.05 |
| R3 direct last-layer, lr=1e-5, seed0 | 500 | 4096 | 1.024M | 0.5932 | 0.38 | +0.04 |
| R3 direct last-layer, lr=1e-5, seed1 | 500 | 4096 | 1.024M | 0.6171 | 0.40 | +0.01 |

The summary plot is:

![RL rerun learning curves](rl_rerun_learning_curves.png)

## Candidate Details

### R1: Residual Deterministic Low Level

Base action:

```text
a = frozen_low(condition) + alpha * tanh(residual)
```

Best closed-loop R1 checkpoint improved success only from `0.34` to `0.35`.
The locally best checkpoint did not transfer to deployment.

### R2: Residual Flow Low Level

The frozen base was a zero-noise endpoint from a low-level action-flow model
trained on the same low-level condition as the deterministic policy.

The flow base itself was weaker than the deterministic low level. R2 improved
local latent reaching relative to the flow base but worsened full deployment:

```text
frozen flow base success: 0.28
R2 tuned success:        0.23
```

R2 did not establish a stable flow base, so R4 direct-flow tuning was not run.

### R3: Direct Deterministic Low-Level Tuning

The tuned actor is the deterministic low-level policy itself. Only the final
low-policy layer, actor log-std, and critic were trainable. A BC regularizer
penalized deviation from the frozen low-level action.

`lr=3e-5` gave the best local final distance but overfit the local objective and
hurt deployment. Reducing the direct learning rate to `1e-5` preserved a smaller
local gain and improved closed-loop success on two policy seeds:

| Seed | Checkpoint | Frozen success | Tuned success | Final reward delta | Max reward delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 409600 | 0.34 | 0.38 | +0.0307 | +0.0315 |
| 1 | 614400 | 0.39 | 0.40 | +0.0224 | +0.0156 |

Seed2 failed the cheap 10k local final-distance screen (`0.6913` tuned versus
`0.6836` frozen), so it has not been promoted to a serious `4096`-env run.

### N=1000 R3 Screen

The promising R3 variants were smoke-tested at `N=1000` before launching a full
4096-env run. Both were locally worse than the frozen `N=1000` low level:

| Policy | Final distance | Reduction fraction |
| --- | ---: | ---: |
| N=1000 frozen | 1.1175 | 0.8359 |
| N=1000 R3 lr=1e-5, 10k | 1.1249 | 0.8184 |
| N=1000 R3 lr=3e-5, 10k | 1.1193 | 0.8223 |

The full `N=1000` R3 run was skipped because the cheap exact-reset screen
failed.

## Gate Decisions

| Gate | Decision | Evidence |
| --- | --- | --- |
| State-loadable data | Pass | exact reset/replay audits pass on vector-consistent corpora |
| Supervised retraining | Pass | N=500 and N=1000 frozen hierarchies retrained and evaluated |
| Throughput | Pass | `4096 x 10` stable, batch `40960`; `8192` fails allocation |
| RL correctness | Pass for local PPO setup | no task reward/progress in training; 10-step segment boundary |
| R1 local gate | Fail | local gains far below 25% target |
| R2 flow gate | Fail | flow base weak; residual degrades deployment |
| R3 direct tuning | Positive but below final gate | `+0.04` on seed0 and `+0.01` on seed1, not `+0.10` |
| N=1000 confirmation | Not passed | smoke variants locally worse than frozen N=1000 |
| Final multi-seed RL gate | Partial only | two serious policy seeds tested; third seed failed cheap screen; no 500-episode final eval |

## Interpretation

The rerun invalidates the earlier weak RL attempt as a definitive negative:
using exact local resets and large vector batches matters. However, residual
low-level PPO did not solve the problem. Directly tuning the deterministic
low-level final layer is more promising and produced the first positive
closed-loop RL results, but the improvement is modest and not yet robust enough
for a final claim.

The current best scientific conclusion is:

> Low-level RL can improve the learned-interface hierarchy slightly when it is
> constrained to a small direct update of the deterministic low-level policy,
> but the observed `+4` and `+1` point success gains are below the planned gate
> and did not reproduce in the cheap `N=1000` screen.

## Remaining Instrumentation Gaps

- Serious RL histories should store wall-clock time per update.
- GPU memory should be sampled during training, not inferred from allocation
  success/failure.
- A final positive claim would require at least three policy seeds and a larger
  evaluation budget for the selected R3 setting.

## Videos

Representative paired videos for the best R3 checkpoint are in
`rl_rerun_failure_videos/`.

```text
checkpoint: artifacts/rl_rerun/local_r3/n500/seed0/aligned10_n4096_lr1e5_bc1_1m/checkpoints/step_000409600.pt
evaluation seeds: 10000-10005
modes: frozen, tuned
```

The filenames include `success`, `final`, and `max` reward fields. The set
contains both successes and failures for qualitative inspection.
