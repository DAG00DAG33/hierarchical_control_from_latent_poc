# VAE-512 Sample-Efficiency Experiment Log

This is the chronological execution record for
[`vae512_sample_efficiency_experiment_plan.md`](vae512_sample_efficiency_experiment_plan.md).

## 2026-06-22 - SE-00: Initial audit and implementation start

- **Plan commit:** `0a63e6c`.
- **Hardware:** NVIDIA RTX 4060 Ti, 16,380 MiB total and 15,507 MiB free at
  audit time.
- **Disk:** 82 GiB free on the workspace filesystem.
- **Existing implementation:** legacy Phase 12 is not reused as the runner. It
  hard-codes AE-256, `k=2`, one policy seed, 100 deployable episodes, and only
  10 oracle episodes. It also omits flat latent flow.
- **Reuse decision:** retain the validated VAE-512 representation,
  `k=10,U=10,H=1` deterministic hierarchy, local branch-oracle evaluator,
  DINO observation path, and common flow model. Add a separate
  `vae512_scaling` artifact/result namespace and matched flat/flow methods.
- **Primary data:** fixed 2,000-trajectory PPO corpus; nested train prefixes
  `{50,100,200,500,1000,1800}` and fixed final 200 validation trajectories.
- **Evaluation target:** 500 episodes for six deployable methods and 50 for
  the oracle, for each of three complete training seeds.

Implementation and the `N=50, seed=0` smoke benchmark are in progress. Runtime
ETA will be added only after measuring the complete smoke point.

## 2026-06-22 - SE-01: `N=50, seed=0` full-training smoke

- **Implementation commit:** `f653bb5`.
- **Manifest:** 50 nested-prefix trajectories, 2,311 transitions, fixed 200
  validation trajectories with 8,969 transitions, SHA256
  `03a45f4c83acd382153f80c0a3968b821b8dda359dd894bf50b3349faf9f7b5f`.
- **VAE-512:** 791 s, selected epoch 1, validation reconstruction `0.4623`.
- **Shared deterministic hierarchy:** 437 s, selected epoch 58, oracle and
  predicted validation action MAE `0.1340/0.1354`.
- **Flow high level:** 200 s, selected epoch 45, predicted-goal action MAE
  `0.1390`.
- **Flat latent deterministic/flow:** 89/96 s, validation action MAE
  `0.1343/0.1316`.
- **Flat observation deterministic/flow:** 171/175 s, validation action MAE
  `0.1325/0.1405`.
- **Total training wall time:** approximately 28 minutes.
- **Storage:** 318,470,959 artifact bytes (about 304 MiB).

Short rollout audit on 20 fixed deployable seeds and five oracle seeds:

| method | success | final reward |
| --- | ---: | ---: |
| deterministic hierarchy | 0.05 | 0.232 |
| flow hierarchy | 0.05 | 0.191 |
| flat latent deterministic | 0.10 | 0.225 |
| flat latent flow | 0.00 | 0.150 |
| flat observation deterministic | 0.10 | 0.244 |
| flat observation flow | 0.00 | 0.145 |
| oracle hierarchy (5 episodes) | 0.00 | 0.129 |

The low-budget values are plausible and all seven deployment paths execute.
The oracle sample is only an implementation audit and is not interpreted.

**Projected full runtime:** approximately 14-16 sequential GPU-hours for 18
training points, 54,000 deployable episodes, and 900 oracle episodes. Projected
artifact storage is about 5.5 GiB plus results. The existing 82 GiB free disk
space is sufficient.

## 2026-06-22 - SE-02: Complete training sweep

- **Runner commit:** `7954d81`; aggregation implementation commit `0f14f3b`.
- **Command:** `scripts/run_vae_scaling_sweep.sh train`.
- **Completed points:** all 18 combinations of budgets
  `{50,100,200,500,1000,1800}` and training seeds `{0,1,2}`.
- **Artifacts per point:** VAE-512 representation, shared deterministic
  high/low hierarchy, flow-matching high level, and deterministic/flow flat
  policies for both latent and full-observation inputs.
- **Manifest audit:** 18/18 point manifests exist. Nested train prefixes and
  the fixed final-200 validation split pass `vae-scaling-manifests` validation.
- **No effect interface:** no learned effect-interface model is trained or
  loaded by this runner.
- **Failures/reruns:** none. The `N=50, seed=0` smoke artifacts were reused;
  all other points were trained exactly once.
- **Summed component training time per point (minutes, seeds 0/1/2):**
  `N=50`: 32.7/32.6/32.8; `N=100`: 32.3/32.9/33.0; `N=200`:
  32.5/33.0/33.1; `N=500`: 32.9/33.1/33.5; `N=1000`:
  33.0/33.4/33.6; `N=1800`: 33.4/35.0/36.0.
- **Storage after training:** 6.7 GiB; 75 GiB remained free.

The fixed-seed rollout sweep starts only after this completeness audit. Final
deployable runs use 500 episodes per point; local branch-oracle runs use 50.

## 2026-06-22 - SE-03: Preliminary seed-0 rollout sweep

- **Reason:** inspect the complete learning-curve shape before spending the
  full three-seed, 500-episode evaluation budget.
- **Command:** `SEEDS=0 EPISODES=100 ORACLE_EPISODES=10
  scripts/run_vae_scaling_sweep.sh eval`.
- **Evaluation seeds:** fixed bank beginning at 2,200,000 for every point.
- **Budget:** 100 episodes per deployable method and 10 per local branch
  oracle. Oracle values in this pass are explicitly not used for conclusions.
- **Output:** `results/incremental/vae512_scaling/preliminary_seed0_100/`.

| trajectories | det hierarchy | flow hierarchy | flat latent det | flat latent flow | flat obs det | flat obs flow | oracle (10) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.03 | 0.03 | 0.02 | 0.03 | 0.01 | 0.00 | 0.00 |
| 100 | 0.04 | 0.07 | 0.05 | 0.00 | 0.06 | 0.01 | 0.00 |
| 200 | 0.09 | 0.14 | 0.02 | 0.06 | 0.10 | 0.07 | 0.10 |
| 500 | 0.28 | 0.31 | 0.14 | 0.08 | 0.29 | 0.22 | 0.10 |
| 1000 | 0.51 | 0.48 | 0.24 | 0.18 | 0.48 | 0.26 | 0.40 |
| 1800 | 0.46 | 0.56 | 0.44 | 0.32 | 0.54 | 0.34 | 0.40 |

The preliminary curve justifies continuing the complete evaluation. Learned
hierarchies are competitive with the strongest flat observation baseline at
1,000-1,800 trajectories, while the three-seed variance and reliable oracle
gap remain unresolved.

## 2026-06-22 - SE-04: Prior 0.72 result discrepancy audit

The full evaluation was paused after the user flagged that preliminary
full-data VAE success was below the previously reported `0.72`.

### Artifact and configuration parity

- The prior and newly trained `N=1800, seed=0` VAE encoder and decoder tensors
  are bit-identical.
- All cached train and validation latent arrays are exactly equal, element by
  element: 1,800 train trajectories and 200 validation trajectories.
- Both runs use 80,472 train transitions, `k=10`, `U=10`, `H=1`, concat goal
  conditioning, posterior-mean deployment, and the same DINO/proprio/action
  path.
- The prior hierarchy selected epoch 57; the new hierarchy selected epoch 54.
  Offline predicted-action MAE is effectively tied (`0.03889` prior versus
  `0.03848` new), but the policy tensors differ.

### Learned checkpoint x evaluation-bank cross-check

| checkpoint | development bank 2,100,000 | unseen bank 2,200,000 |
| --- | ---: | ---: |
| prior selected VAE hierarchy | 0.72 | 0.56 |
| new `N=1800, seed=0` hierarchy | 0.64 | 0.46 |

Each cross-check uses 100 episodes. This attributes roughly 16-18 percentage
points to the finite evaluation bank and 8-10 points to the hierarchy
retraining realization.

### Policy-seed and low-level oracle check

The three newly trained full-data learned hierarchies score `0.46, 0.43,
0.51` on the unseen 100-seed bank: mean `0.467`, sample SD `0.040`.

On the unseen 50-seed oracle bank, their corresponding low levels score
`0.50, 0.56, 0.50`: mean `0.520`, sample SD `0.035`. The prior selected low
level scores `0.68` on that same unseen bank. On the old bank, the prior
reported oracle score was `0.76` over 100 episodes while the new seed-0 low
scores `0.62` over 50.

### Decision

No encoder, data-split, horizon, or deployment wiring error was found. The
previous `0.72` is a development result selected after screening many
candidates on the reused 2,100,000 seed bank and from a particularly strong
low-level training realization. Offline action MAE does not expose the
closed-loop difference.

Keep the unseen 2,200,000 bank for the final comparison and use 500 episodes
plus three training seeds as planned. Retain the prior selected checkpoint as
a labeled diagnostic reference; do not mix it into the independent-seed
average.

## 2026-06-22 - SE-05: Final evaluation and decision

- **Evaluation:** all 18 `(budget, policy seed)` points completed.
- **Per point:** six deployable methods x 500 fixed unseen episodes; one local
  branch oracle x 50 episodes.
- **Seed bank:** 2,200,000 through 2,200,499 for deployable methods; first 50
  of the same bank for the oracle.
- **Completeness:** the aggregator validated all seven methods, expected
  episode counts, and retained episode-level outcome arrays before plotting.
- **Aggregate output:** `results/incremental/vae512_scaling/aggregate/`.
- **Tracked report:** `VAE512_SAMPLE_EFFICIENCY_FINAL_RESULTS.md`.
- **Tracked plots:** `docs/results/vae512_scaling/`.
- **Videos:** full-data seed-2 learned and branch-oracle successes/failures
  under the point's `videos/` directory.

Full-data mean success +/- sample SD across policy seeds:

| method | success |
| --- | ---: |
| deterministic hierarchy | `0.565 +/- 0.025` |
| flow hierarchy | `0.556 +/- 0.005` |
| flat latent deterministic | `0.457 +/- 0.017` |
| flat latent flow | `0.357 +/- 0.008` |
| flat observation deterministic | `0.571 +/- 0.048` |
| flat observation flow | `0.401 +/- 0.049` |
| branch oracle (50 episodes/seed) | `0.520 +/- 0.035` |

The flow hierarchy has the highest normalized log-data AULC (`0.259`), versus
`0.255` for flat observation deterministic and `0.251` for deterministic
hierarchy. The differences are too small relative to three-seed uncertainty
to claim improved sample efficiency. No method reaches 70% mean success.

**Final decision:** the learned VAE future-state interface is viable and
recovers the performance lost by flat control from the current VAE latent,
but it does not robustly outperform deterministic flat control from the full
observation. The experiment therefore does not support the main
sample-efficiency thesis.

## 2026-06-22 - SE-06: Extend the data curve to 4,000 and 8,000

- Added 6,200 successful trajectories from the same privileged PPO expert,
  controller, observation pipeline, and deterministic teacher policy.
- Collection required 7,796 attempts (`79.5%` retained success rate) and took
  15.5 minutes with 64 vectorized CUDA environments.
- Preserved the original first 1,800 training episodes and copied the original
  final-200 validation split unchanged.
- Extended dataset: 8,000 train plus 200 validation episodes, 8.4 GiB.
- New transition counts: 177,972 at `N=4000`; 354,589 at `N=8000`.
- Trained all six new `(budget, seed)` points from scratch with no warm starts.
- Ran a seed-0 preliminary screen on 100 deployable and 10 oracle episodes,
  then the final 500/50 evaluation for all three policy seeds.
- Recorded six learned-goal and six branch-oracle videos from the full-data
  seed-0 hierarchy, including successful and failed episodes.

Final success mean +/- sample SD:

| trajectories | det hierarchy | flow hierarchy | oracle | flat latent det | flat latent flow | flat obs det | flat obs flow |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 0.654 +/- 0.046 | 0.621 +/- 0.040 | 0.640 +/- 0.072 | 0.570 +/- 0.025 | 0.413 +/- 0.038 | **0.679 +/- 0.037** | 0.483 +/- 0.035 |
| 8,000 | **0.692 +/- 0.026** | 0.659 +/- 0.017 | 0.693 +/- 0.070 | 0.639 +/- 0.020 | 0.464 +/- 0.014 | 0.649 +/- 0.034 | 0.455 +/- 0.032 |

The extension changes the full-data conclusion: deterministic hierarchy is
the strongest deployable method at 8,000 trajectories. It still does not win
the normalized complete-curve AULC (`0.365` versus `0.369` for deterministic
flat observation), so the evidence supports a learned-interface benefit at
scale but not a robust data-efficiency advantage over the entire range.
