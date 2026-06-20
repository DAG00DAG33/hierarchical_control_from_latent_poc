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
