# Push-T Pre-RL Experiment Log

This log records execution of
[`pusht_pre_rl_next_experiments_plan.md`](pusht_pre_rl_next_experiments_plan.md).
State-query data is always reported separately from causal transitions.

## 2026-06-20 - A-I01: Statistical-replication runner

- **Objective:** Re-evaluate the full-data ordering across three independent
  policy seeds before changing the interface or dataset.
- **Methods:** Direct visual BC, direct visual flow, matched flat AE-256 BC,
  exact local branch-oracle hierarchy, deterministic learned hierarchy, and
  conditional-flow learned hierarchy.
- **Causal data:** The same first 1,800 successful PPO trajectories for every
  method and seed; the same last 200 trajectories for validation; 80,472
  causal training transitions; no DAgger/state-query samples.
- **Protocol:** Policy seeds `{0,1,2}`; 200 paired evaluation episodes per
  deployable method and seed; 50 paired exact-replay oracle episodes per seed;
  reset seeds begin at `1500000`.
- **Implementation:** Added resumable `pre-rl-a-run` and
  `pre-rl-a-aggregate` commands. Seed 0 reuses only its Phase 12 training
  checkpoints, while every 200/50-episode result is newly evaluated on the
  Phase A seed range. Seeds 1 and 2 train independent checkpoints in the same
  isolated 1,800-trajectory artifact root.
- **Uncertainty:** Per-policy success includes a 95% Wilson interval. The
  aggregate reports mean and sample standard deviation across training seeds,
  plus a pooled Wilson interval as a secondary evaluation-noise summary.
- **Commands:**

  ```bash
  for SEED in 0 1 2; do
    uv run hcl-poc incremental pre-rl-a-run \
      --config configs/pusht_incremental.yaml --seed "$SEED"
  done

  uv run hcl-poc incremental pre-rl-a-aggregate \
    --config configs/pusht_incremental.yaml
  ```

- **Status:** Implementation complete; execution pending.

## 2026-06-20 - A-D01: Matched-flat lazy-training inference-mode bug

- **Command:** `uv run hcl-poc incremental pre-rl-a-run --config
  configs/pusht_incremental.yaml --seed 0`.
- **Observed behavior:** Fresh 200-episode visual BC and visual-flow
  evaluations completed (`0.600` and `0.585` success), then matched-flat
  latent training failed on its first backward pass because
  `evaluate_phase7_matched_flat_latent_policy` decorated the entire function
  with `torch.inference_mode()`.
- **Diagnosis:** Earlier calls silently depended on an already existing latent
  BC checkpoint. Phase A is the first path that legitimately requested lazy
  training through this evaluator.
- **Fix:** Keep evaluation inference-only but explicitly disable inference
  mode around checkpoint preparation/training. Completed result files are
  retained and the resumable runner will reuse them.
- **Data impact:** None. No partial checkpoint was written and no result was
  included in an aggregate.

## 2026-06-20 - A-R01: Seed 0 replication

- **Command:** `uv run hcl-poc incremental pre-rl-a-run --config
  configs/pusht_incremental.yaml --seed 0`.
- **Checkpoints:** Reused the independently trained Phase 12 seed-0 visual,
  AE-256, low-level, deterministic-high, and flow-high checkpoints. Trained
  the previously missing matched-flat AE-256 head from the same 1,800 causal
  trajectories. No state-query data was used.
- **Evaluation:** Deployable methods use 200 episodes and the oracle uses 50;
  all begin at seed `1500000`. Exact branch replay error is `0.0`.

| method | success | 95% Wilson CI | final reward | max reward | validation action MAE | rollout teacher MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| visual BC | `0.600` | `[0.531, 0.665]` | `0.601` | `0.711` | `0.0382` | n/a |
| visual flat flow | `0.585` | `[0.516, 0.651]` | `0.579` | `0.701` | `0.0375` | n/a |
| matched flat latent | `0.505` | `[0.436, 0.574]` | `0.621` | `0.634` | `0.0325` | n/a |
| exact branch oracle | `0.620` | `[0.482, 0.741]` | `0.743` | `0.745` | `0.0266` | `0.0334` |
| deterministic hierarchy | `0.300` | `[0.241, 0.367]` | `0.364` | `0.475` | `0.0392` | `0.1457` |
| generative hierarchy | `0.335` | `[0.273, 0.403]` | `0.364` | `0.498` | `0.0401` | `0.1389` |

- **Preliminary diagnosis:** The point estimates retain
  `oracle > direct flat > matched latent flat > learned hierarchy`. The oracle
  interval overlaps direct visual BC because only 50 expensive episodes are
  available; the three-training-seed aggregate remains necessary.
- **Runtime:** The exact 50-episode branch replay took about 9.6 minutes. The
  other seed-0 evaluations and matched-flat training took about 6 minutes.
- **Status:** Seed 0 complete; no Phase A gate decision until seeds 1 and 2.

## 2026-06-20 - A-D02: Learned-hierarchy lazy-training inference-mode bug

- **Command:** The sequential Phase A seed-1/seed-2 command documented in
  A-I01.
- **Observed behavior:** Seed 1 completed visual/latent training, matched-flat
  evaluation, low-level training, and all 50 exact oracle episodes. It then
  failed when the decorated Phase 8 evaluator lazily trained the new seed's
  deterministic predictor with autograd disabled.
- **Diagnosis:** This is the same latent evaluator assumption found in A-D01,
  also present in the Phase 8 and Phase 9 evaluation entry points because
  historical evaluations normally followed explicit training commands.
- **Fix:** Explicitly leave inference mode around every lazy Phase 8/9
  checkpoint preparation path, including optional DAgger/adapted/robust paths.
  Rollout computation remains under inference mode.
- **Data impact:** None. The valid seed-1 oracle result and all completed
  checkpoints/results are retained. No Phase A aggregate has been generated.

## 2026-06-20 - A-G01: Full-budget statistical replication gate

- **Commands:** Three `pre-rl-a-run` commands for seeds `{0,1,2}`, followed by
  `pre-rl-a-aggregate`, exactly as recorded in A-I01.
- **Data:** Every model uses the same 1,800 clean successful causal
  trajectories (80,472 transitions) and fixed 200-trajectory validation set.
  Each seed retrains visual BC, visual flow, AE-256, matched latent flat,
  oracle low level, deterministic high level, and generative high level from
  its own initialization. No state-query data is used.
- **Evaluation:** The same 200 reset seeds beginning at `1500000` for every
  deployable policy; the first 50 for each exact-replay oracle. Exact replay
  state error is zero. The table reports mean and sample standard deviation
  across training seeds.

| method | seed successes | mean +/- training-seed SD | pooled 95% Wilson CI | mean final reward | mean max reward |
| --- | --- | ---: | --- | ---: | ---: |
| visual BC | `0.600, 0.595, 0.595` | `0.597 +/- 0.003` | `[0.557, 0.635]` | `0.582` | `0.707` |
| visual flat flow | `0.585, 0.555, 0.605` | `0.582 +/- 0.025` | `[0.542, 0.620]` | `0.600` | `0.700` |
| matched flat latent | `0.505, 0.490, 0.460` | `0.485 +/- 0.023` | `[0.445, 0.525]` | `0.604` | `0.619` |
| exact branch oracle | `0.620, 0.760, 0.700` | `0.693 +/- 0.070` | `[0.615, 0.762]` | `0.793` | `0.794` |
| deterministic hierarchy | `0.300, 0.345, 0.335` | `0.327 +/- 0.024` | `[0.290, 0.365]` | `0.373` | `0.496` |
| generative hierarchy | `0.335, 0.390, 0.325` | `0.350 +/- 0.035` | `[0.313, 0.389]` | `0.358` | `0.511` |

- **Action-error diagnosis:** Oracle rollout teacher MAE is stable at
  `0.0320-0.0334`; deterministic learned hierarchy MAE is `0.137-0.151`, and
  generative hierarchy MAE is `0.131-0.141`. Learned-goal error therefore
  remains a large control error under every training seed.
- **Gate:** Pass. The mean ordering is
  `oracle (0.693) > best flat (0.597) > best learned hierarchy (0.350)`.
  This ordering also holds in point estimates for each policy seed. The pooled
  oracle and visual-BC confidence intervals overlap slightly, so the exact
  oracle advantage is not claimed as a precise effect size; the large
  flat-versus-learned gap is reproducible.
- **Decision:** Continue to Phase B with high priority on determining whether
  the oracle benefit is future object effect, future robot motion, or motor
  waypoint leakage. Representation redesign remains justified because the
  learned hierarchy does not overlap the direct flat methods after replication.
- **Artifacts:** Tracked aggregate JSON and plot are
  `docs/results/pre_rl/phase_a_aggregate.json` and
  `docs/results/pre_rl/phase_a_success_across_seeds.png`.
- **Remaining Phase A deliverable:** Representative rollout videos will be
  generated with the shared pre-RL video recorder alongside Phase B videos so
  success/failure selection and rendering are implemented once.

## 2026-06-20 - B-I01: Oracle-information decomposition implementation

- **Objective:** Determine whether oracle performance comes from a desired
  future object effect, future robot/TCP motion, or both.
- **Goal types:** Full 28D object/TCP/joint/contact goal, 20D robot goal, 6D
  TCP goal, 7D object pose/velocity goal, and 4D object-pose-only goal. Every
  policy also receives the same complete current 31D privileged state and
  previous action. A matched flat current-state policy is included.
- **Horizons:** `k={2,5,10,20}` at 20 Hz with one-step low-level actions.
- **Data:** Existing 1,800/200 clean successful privileged causal split,
  deterministic clipped teacher actions, no learner-state queries. Each
  representation has a separately fitted condition normalizer and otherwise
  identical width-256, depth-4 MLP training.
- **Oracle correctness:** Evaluation uses reset-and-exact-replay from each
  policy's current student state before rolling the teacher. It does not use a
  nominal trajectory or direct state copy.
- **Added diagnostics:** Per-goal validation action MAE, valid shuffled-goal
  action sensitivity, teacher-action MAE, action saturation, exact replay
  error, and one-step object/yaw/TCP error toward the supplied branch target.
- **Commands:** `pre-rl-b-train`, `pre-rl-b-eval`, and
  `pre-rl-b-aggregate`. The aggregate writes the required
  `oracle_goal_decomposition.csv` using structured CSV serialization and a
  success-by-goal/horizon plot.
- **Execution plan:** Run 20-episode smoke evaluations first. Promote all
  valid settings to the required 100 fixed episodes after checking state
  slicing, replay equality, and goal sensitivity.

## 2026-06-20 - B-G01: Oracle-information decomposition smoke sweep

- **Commands:** `pre-rl-b-train` and `pre-rl-b-eval --episodes 20` for
  `k={2,5,10,20}`, followed by `pre-rl-b-aggregate --episodes 20`.
- **Evaluation:** All variants use the same 20 reset seeds beginning at
  `1600000`. Exact replay current-state error is zero for every branch-goal
  rollout. These small samples are screening results, not final effect-size
  estimates.
- **Success rates:**

| horizon | flat | full | robot | TCP | object | object pose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k=2` (0.10 s) | `0.25` | `0.75` | `0.60` | `0.50` | `0.40` | `0.25` |
| `k=5` (0.25 s) | `0.20` | `0.55` | `0.30` | `0.25` | `0.15` | `0.25` |
| `k=10` (0.50 s) | `0.00` | `0.40` | `0.35` | `0.50` | `0.20` | `0.20` |
| `k=20` (1.00 s) | `0.20` | `0.20` | `0.15` | `0.05` | `0.25` | `0.10` |

- **Information diagnosis:** At the strongest horizon (`k=2`), robot state
  retains 80% of full-oracle success and TCP retains 67%, whereas object state
  retains 53%. At `k=10`, TCP is the strongest point estimate. Future robot
  motion therefore explains substantially more of the oracle advantage than
  future object state alone; the current full-state oracle is partly a motor
  waypoint interface rather than a pure desired-effect interface.
- **Horizon diagnosis:** Full-oracle performance falls from `0.75` at `k=2`
  to `0.20` at `k=20`, while one-step TCP subgoal error grows from `0.021 m`
  to `0.116 m`. One second is too long for this one-step low-level formulation.
  The `k=20` object/full ratio exceeds one only because both estimates are weak
  and their Wilson intervals are broad; it is not treated as a gate pass.
- **Dataset constraint:** A 20-step future exists in only 1,954 successful
  episodes. `k=20` therefore used 1,754 train and 200 validation episodes,
  versus the requested 1,800/200 and the full split at shorter horizons. The
  loader now records requested, usable, and effective counts and only applies
  this explicit cap for Phase B.
- **Decision:** Use larger evaluation budgets only for promising diagnostic
  settings: the short full/robot/TCP interfaces and the `k=10` TCP result.
  Do not spend development budget expanding every clearly weak `k=20` or
  object-only setting. Phase C should focus on whether short future TCP/robot
  goals provide genuine temporal abstraction rather than leaked teacher
  waypoints.
- **Artifacts:** `results/incremental/pre_rl/phase_b/phase_b_aggregate_20.json`,
  `oracle_goal_decomposition.csv`, and
  `oracle_goal_decomposition_20.png` (tracked copies under
  `docs/results/pre_rl/phase_b/`).

## 2026-06-20 - C-D01: Fixed-offset held-goal failure and correction

- **Initial test:** Reused the fixed-`k` Phase B TCP policies while holding one
  oracle branch goal for `U={1,2,5,10}` primitive steps. The first
  implementation cached the six-dimensional TCP feature vector directly.
- **Bug found:** TCP "velocity" is a displacement-over-horizon feature derived
  from the current state and target endpoint. Caching it made the feature stale
  after the first action. The corrected implementation holds the raw reachable
  endpoint state and recomputes derived features from each current observation
  and the remaining time. This does not invoke the high level again.
- **Corrected fixed-offset result:** At `k=5`, success was `0.30, 0.25, 0.05,
  0.05` for `U=1,2,5,10` against flat `0.30`. At `k=10`, it was `0.20, 0.20,
  0.15, 0.00` against flat `0.35`. Exact replay errors remained zero.
- **Diagnosis:** Even coherent endpoint preprocessing becomes out of
  distribution after one held action because each Phase B low level was
  trained only at exactly `k` remaining steps.

## 2026-06-20 - C-G01: Multi-offset time-conditioned oracle smoke gate

- **Method:** Train TCP low-level policies on every future offset from 1 through
  `k`, repeating the coherent teacher action label for each reachable teacher
  endpoint. Append normalized time-to-go `offset/k`. At deployment, hold the
  endpoint for `U` steps, recompute its derived TCP feature from the observed
  current state, decrement time-to-go, and reobserve before every action.
- **Data:** Same clean successful privileged trajectories as Phase B, no state
  queries. `k=5` expands 71,115 causal transitions across five offsets; `k=10`
  expands them across ten offsets. Architecture remains a width-256, depth-4
  deterministic MLP with 100 epochs, batch size 4,096, and learning rate
  `3e-4`.
- **20-episode smoke result:**

| horizon | flat | `U=1` | `U=2` | `U=5` | `U=10` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `k=5` | `0.30` | `0.60` | `0.50` | `0.50` | `0.15` |
| `k=10` | `0.35` | `0.55` | `0.55` | `0.55` | `0.75` |

- **Temporal abstraction:** `k=10,U=10` uses 6.05 high-level decisions per
  episode on average versus 56.15 primitive actions, while reaching `0.75`
  success and `0.0486` teacher-action MAE. This is a genuine 0.5-second held
  future target rather than per-step indirect action prediction.
- **Decision:** Phase C has a strong positive smoke signal. Confirm only the
  selected `k=10` family on 100 fixed episodes before choosing the interface;
  do not train `k=20`, whose Phase B oracle was already weak.

## 2026-06-20 - C-G02: 100-episode temporal-abstraction confirmation

- **Configuration:** Time-conditioned multi-offset TCP policy, `k=10`,
  `H=1`, fixed evaluation seeds beginning at `1700000`, exact local teacher
  branch endpoints, and endpoint holds `U={1,2,5,10}`. No additional training
  or state-query data was introduced after C-G01.

| method | success | 95% Wilson CI | final reward | teacher MAE | high-level decisions/episode |
| --- | ---: | --- | ---: | ---: | ---: |
| flat privileged | `0.16` | `[0.101, 0.244]` | `0.381` | `0.1704` | `0.00` |
| TCP oracle, `U=1` | `0.69` | `[0.594, 0.772]` | `0.783` | `0.0620` | `59.67` |
| TCP oracle, `U=2` | `0.71` | `[0.615, 0.790]` | `0.798` | `0.0553` | `29.19` |
| TCP oracle, `U=5` | `0.69` | `[0.594, 0.772]` | `0.786` | `0.0525` | `12.46` |
| TCP oracle, `U=10` | `0.81` | `[0.722, 0.875]` | `0.866` | `0.0554` | `6.25` |

- **Gate:** Pass. `k=10,U=10` represents a meaningful 0.5-second physical
  target, makes materially fewer decisions than the primitive controller, and
  exceeds both the matched flat policy and per-step oracle on this evaluation
  set. Exact replay error remains zero.
- **Selected interface for subsequent experiments:** TCP endpoint,
  `k=10`, `U=10`, and one-step low-level action horizon `H=1`. The action-chunk
  ablation remains secondary because goal holding already supplies temporal
  abstraction without open-loop primitive execution.

## 2026-06-20 - D-G01: Causal recovery corpus and equal-budget views

- **Collection:** 1,000 CUDA episodes, 100 steps each, seeds beginning at
  `1800000`. Each episode contains 1-3 causal action bursts from directional
  bias, action hold, 1-3-step delay, or 0.7/1.3 scaling. The teacher resumes
  from the reached state; no state restoration occurs.
- **Stored signals:** Raw RGB, state, proprioception, executed behavior action,
  same-state deterministic teacher query, perturbation/burst metadata,
  recovery intervals/completions, reward, and success. Behavior and recovery
  labels are explicitly separate.
- **Recoverability:** 2,035 bursts, 46.1% overall recovery. Bias/hold/delay/
  scaling recover at 50.6%/37.1%/40.7%/56.3%. Every family remains usable;
  eight-step hold and delay are the hardest at 24.4% and 31.1%.
- **Visual preparation:** Frozen `facebook/dinov2-small` spatial features
  (`6528D`) plus `21D` proprioception for all 100,000 transitions. Prepared
  file size is 2.5 GB; 97 GB disk remains.
- **Equal-budget views:** Fixed 80,000-transition manifests are clean 80/0k,
  mixed-25 60/20k, mixed-50 40/40k, and recovery-heavy 30/50k clean/
  off-nominal. Recovery episodes 800-999 and the existing last 200 clean
  episodes are held out.
- **Specification:** Full schema, hashes, perturbation parameters, label
  semantics, and split definitions are in `recovery_dataset_spec.md`.
- **Next:** Train matched flat and hierarchical candidates from each manifest
  using teacher-query recovery labels first; retain executed-action imitation
  as a separate ablation.

## 2026-06-20 - D-G02: Equal-budget direct visual BC comparison

- **Method:** History-1 direct spatial-DINO/proprioception BC, identical
  width-512 three-hidden-layer architecture, 80,000 transitions per variant,
  and deterministic teacher-query labels on recovery states. Previous action
  is always the action actually executed. All variants use the same clean and
  recovery validation sets and 100 fixed clean/disturbed evaluation seeds.
- **Initial 50-epoch comparison:**

| dataset | clean validation MAE | recovery validation MAE | clean success | disturbed success | recovery success |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | `0.0427` | `0.0797` | `0.66` | `0.60` | `0.53` |
| mixed-25 | `0.0443` | `0.0643` | `0.54` | `0.51` | `0.43` |
| mixed-50 | `0.0478` | `0.0636` | `0.53` | `0.51` | `0.43` |
| recovery-heavy | `0.0489` | `0.0640` | `0.40` | `0.39` | `0.38` |

- **Undertraining check:** Mixed-25 validation was still improving, so it was
  retrained for 100 epochs. MAE improved to `0.0417` clean and `0.0622`
  recovery. Closed-loop success improved to `0.59` clean, `0.55` disturbed,
  and `0.50` recovery, but remained below clean-only by 7, 5, and 3 percentage
  points respectively.
- **Interpretation:** Teacher-query recovery data materially improves offline
  recovery-state action prediction but does not improve direct visual BC under
  causal disturbances. Larger recovery fractions increasingly hurt nominal
  control. The clean policy already generalizes to these moderate action
  bursts better than the mixed policies.
- **Decision:** Select clean data for the direct visual BC reference. Keep
  mixed-25 as the only recovery-data candidate for later matched hierarchy
  tests; do not spend hierarchy/flow budget on mixed-50 or recovery-heavy
  unless a later representation diagnostic provides a specific reason.

## 2026-06-20 - E-G01: AE/VAE/raw goal-geometry audit

- **Data:** Existing fixed 12,000-state Phase 6 probe corpus. Geometry uses
  20,000 random state pairs and 500 nearest-neighbor queries against 3,000
  references. Latent dimensions are standardized before Euclidean comparison,
  matching downstream condition normalization.
- **Pair geometry:** Object-XY Spearman correlation is `0.627/0.704/0.713`
  for raw/AE/VAE; TCP-XY is `0.796/0.701/0.673`; teacher-action distance is
  `0.579/0.628/0.621`. VAE is smoother for object state, while raw DINO is most
  direct for TCP and AE aligns best with action differences.
- **Neighborhood control consistency:** Teacher-action MAE of nearest states is
  `0.1007` raw, `0.0671` AE, and `0.0774` VAE. Contact match is 92.6%, 94.8%,
  and 93.8%. AE is the strongest local control neighborhood.
- **Interpolation:** VAE decoded interpolation has lower linear-reference MSE
  (`0.0459`) than AE (`0.0556`), but existing matched latent BC succeeds at
  only `0.37` for VAE versus `0.53` for AE.
- **Decision:** Keep AE-256 for current-state compression, but do not use raw
  AE subtraction as the preferred goal interface. Carry forward the compact
  time-conditioned TCP endpoint/effect interface validated in Phase C. Full
  interpretation is in `representation_geometry_report.md`.

## 2026-06-20 - F-G01: Learned privileged TCP effect interface

- **Factorization:** Complete privileged current state remains the low-level
  input, while the future goal is reduced to a learned 3D TCP endpoint. The
  validated multi-offset low level derives time-to-go velocity features and
  holds each endpoint for `U=10` actions at `k=10`.
- **High-level training:** Deterministic width-512, depth-4 MLP on 53,115 clean
  causal samples; input is current 31D state plus previous action, output is
  the TCP position 10 steps ahead. Training uses 100 epochs, batch size 1,024,
  and learning rate `3e-4`.
- **Prediction:** Held-out TCP endpoint L2 error is `0.0170 m`, versus
  `0.1899 m` for persistence.
- **Closed loop:** 100 episodes on seeds beginning at `1920000` produce `0.69`
  success (95% Wilson `[0.594, 0.772]`), final reward `0.784`, teacher-action
  MAE `0.0579`, and 9.16 high-level decisions per episode. A prior 20-episode
  smoke sample happened to score 1.00 and is not used as the estimate.
- **Interpretation:** A compact, physically meaningful future TCP effect is
  both predictable and controllable. This closes most of the interface
  uncertainty without requiring latent subtraction. The next test replaces
  privileged current state in the high and low policies with AE-256/DINO plus
  proprioception while retaining the same endpoint target.

## 2026-06-21 - F-G02: Matched visual factorized TCP hierarchy

- **Methods:** Two matched deterministic hierarchies use the same 1,800 clean
  causal training trajectories, 200 validation trajectories, `k=10`, `U=10`,
  `H=1`, 100 evaluation seeds, previous executed action, and 3D future TCP
  endpoint. The only difference is current representation: raw normalized
  spatial DINO plus proprioception (`6549D`) or frozen AE-256.
- **Training:** Separate width-512, depth-4 high and low MLPs. The high level
  predicts the 10-step endpoint. The low level is trained over every remaining
  offset 1-10 and receives endpoint, recomputed velocity, previous action, and
  normalized time-to-go. Both models use 60 epochs, 200 batches/epoch, batch
  size 512, and learning rate `3e-4`.
- **Offline metrics:**

| representation | endpoint L2 | oracle-goal action MAE | predicted-goal action MAE | induced action L2 |
| --- | ---: | ---: | ---: | ---: |
| raw DINO+prop | `0.0173 m` | `0.0444` | `0.0445` | `0.0015` |
| AE-256 | `0.0180 m` | `0.0391` | `0.0396` | `0.0072` |

- **Closed loop:** Evaluation uses 16 CUDA environments consistently because
  a 64-versus-16 comparison exposed batch-size sensitivity in vectorized
  simulation/inference. Raw succeeds at `0.71` (95% Wilson
  `[0.615, 0.790]`), final reward `0.781`, teacher-action MAE `0.0879`, and
  6.42 high-level decisions/episode. AE-256 succeeds at `0.53`, final reward
  `0.639`, and teacher-action MAE `0.1267`.
- **Decision:** Select raw spatial DINO plus proprioception for the factorized
  hierarchy. AE-256 compresses static state well but loses rollout-relevant
  information for this interface.

## 2026-06-21 - G-G01: Matched oracle gap and on-policy endpoint audit

- **Protocol:** Exact local teacher branches are regenerated from each current
  student state every 10 primitive steps. Learned and oracle policies use the
  identical raw visual low level, 100 seeds beginning at `1930000`, and 16
  environments. Replay current-state error is exactly zero.
- **Result:** Learned endpoint success is `0.71`; matched branch-oracle endpoint
  success is also `0.71`. Both Wilson intervals are `[0.615, 0.790]`. The
  Phase F/G requirement of at least 80% oracle performance passes with a ratio
  of 1.00.
- **Distribution shift:** Endpoint L2 grows from `0.0173 m` on held-out teacher
  states to `0.0368 m` on hierarchy-visited states. Despite that increase, the
  learned hierarchy retains the oracle success rate. Learned and oracle
  teacher-action MAE are `0.0879` and `0.0765`.
- **Interface conclusion:** The remaining gap is not high-level endpoint
  prediction for this selected interface. The principal residual is the
  visual low-level policy relative to the privileged low-level ceiling and
  direct flat variability, not an oracle-to-learned high-level gap.

## 2026-06-21 - D-D02: Exact matched flat/hierarchy recovery comparison

- **Fairness correction:** The first direct BC recovery sweep used 80,000
  one-step samples, while the hierarchy requires valid 10-step future windows.
  A dedicated manifest now gives both methods the exact same 60,000 current
  state queries. Clean uses 60,000 clean queries; mixed-25 uses 45,000 clean
  and 15,000 coherent recovery queries. Validation uses 6,969 clean and 8,149
  recovery queries.
- **Causal semantics:** Recovery current state, previous executed action,
  teacher-query target action, and future TCP endpoint all come from one
  uninterrupted recovery trajectory. The disturbed evaluation schedule is
  precomputed globally, so it does not depend on vector batch size.
- **Training:** Flat policies are width-512 three-hidden-layer MLPs trained for
  100 epochs. Hierarchies use separate width-512, depth-4 high/low MLPs,
  `k=10`, `U=10`, `H=1`, 60 epochs, 200 batches/epoch, batch size 512, and
  learning rate `3e-4`. All methods use raw spatial DINO plus proprioception.
- **Closed loop:**

| method | data | clean success | disturbed success | recovery success |
| --- | --- | ---: | ---: | ---: |
| flat | clean | `0.48` | `0.44` | `0.31` |
| flat | mixed-25 | `0.45` | `0.44` | `0.36` |
| hierarchy | clean | `0.45` | `0.43` | `0.41` |
| hierarchy | mixed-25 | `0.35` | `0.36` | `0.34` |

- **Offline/closed-loop mismatch:** Mixed hierarchy endpoint L2 improves from
  `0.0407 m` to `0.0341 m`, and predicted-goal action MAE improves from
  `0.0798` to `0.0657`, yet clean success falls 10 points and disturbed
  success falls 7 points.
- **Gate decision:** The clean hierarchy passes the matched-interface
  tolerance: it is within 0.03 clean success and 0.01 disturbed success of
  the clean flat policy. Recovery-rich data does not pass the improvement
  gate. Select the clean 1,800-trajectory corpus for the final interface and
  treat recovery-data expansion as an RL/on-policy collection question rather
  than a supervised pretraining improvement.
- **Artifacts:** `docs/results/pre_rl/phase_d/matched/` contains all six result
  JSON files, `clean_vs_recovery_rich.csv`, and
  `clean_vs_recovery_rich.png`.

## 2026-06-21 - G-G02: Predictor distribution and action-sensitivity audit

- **Command:** `uv run hcl-poc incremental pre-rl-g-tcp-diagnostics --config
  configs/pusht_incremental.yaml`
- **Protocol:** The selected raw-DINO TCP predictor was evaluated on 5,000
  held-out teacher queries, 5,000 coherent recovery queries, 126 high-level
  decisions from 20 flat-policy rollouts, and 140 decisions from 20 hierarchy
  rollouts. Online goals use exact replay teacher branches from the current
  student state.
- **Metrics:**

| distribution | endpoint L2 | induced low action L2 | Spearman |
| --- | ---: | ---: | ---: |
| held-out teacher | `0.0175 m` | `0.0016` | `0.710` |
| coherent recovery | `0.0611 m` | `0.0080` | `0.871` |
| flat visited | `0.0348 m` | `0.0045` | `0.743` |
| hierarchy visited | `0.0372 m` | `0.0047` | `0.795` |

- **Diagnosis:** Prediction error increases off the teacher distribution and
  is largest on recovery states. The action-sensitive error remains small on
  hierarchy states, consistent with learned and branch-oracle success both
  being `0.71`. The high-level predictor is not the current closed-loop
  bottleneck.
- **Artifacts:** `docs/results/pre_rl/phase_g/` contains the complete JSON,
  summary CSV, and endpoint-error versus induced-action plot.

## 2026-06-21 - H-I01: Multimodality and camera decision

- **Phase H decision:** Skip generative high-level experiments. The
  deterministic raw TCP predictor reaches exactly the same 100-episode success
  as the reachable branch oracle (`0.71`), leaving no measured oracle gap for
  multimodal candidate generation to close. The deterministic teacher corpus
  also provides one continuation per state rather than evidence of distinct
  valid modes.
- **Phase I decision:** Skip base-plus-wrist-camera retraining. Base-camera
  spatial DINO plus proprioception already passes the learned/oracle gate.
  Hierarchy-state endpoint error is `0.0372 m`, and exact goals do not improve
  success, so perception is not the demonstrated bottleneck.
- **Next action:** Begin RL only as a frozen-policy residual recovery study,
  as specified in `pre_rl_summary.md`.

## 2026-06-21 - F-G03: Representative learned/oracle videos

- **Learned videos:** Seeds 1,930,000-1,930,009 produced seven successes and
  three failures. Files include success and final/max reward in their names.
- **Paired oracle diagnostic:** Exact branch-oracle videos were recorded for
  learned failures 1,930,006, 1,930,007, and 1,930,009. The oracle also fails
  on the first two and succeeds on 1,930,009.
- **Interpretation:** Some failures are endpoint-prediction dependent, but
  others persist under exact reachable goals. This is consistent with equal
  aggregate learned/oracle success and supports low-level residual recovery RL
  rather than high-level-only RL.
- **Path:**
  `results/incremental/pre_rl/phase_f/raw_tcp/seed0/videos/`.
