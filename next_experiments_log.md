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
