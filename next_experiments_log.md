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
