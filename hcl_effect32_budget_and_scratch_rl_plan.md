# Effect32 Budget Scaling and Scratch Low-Level RL Plan

This plan replaces the current local-RL iteration path with two reviewable
experiments:

1. Measure whether the compact `effect32_film` interface keeps its advantage
   across the same demonstration-budget protocol used by the VAE512 final
   sample-efficiency plot.
2. Test whether a low-level policy trained from scratch with RL can become more
   goal-sensitive and more reachable than the BC-initialized/fine-tuned low
   level.

No experiment should be launched from this file until the protocol below is
reviewed and cleaned up.

## Context

The README plot reports the final VAE512 sample-efficiency protocol:

```text
budgets: 50, 100, 200, 500, 1000, 1800, 4000, 8000 trajectories
training seeds: 0, 1, 2
deployable evaluation: 500 fresh episodes per point
oracle diagnostic: 50 fresh episodes per point
data split: nested train prefixes plus fixed 200-trajectory validation set
```

The `effect32_film` development result was strong under the older learned
interface protocol:

```text
effect32_film learned: 0.69 over 100 fixed episodes, seed0
effect32_film oracle:  0.72 over 100 fixed episodes, seed0
```

The more recent HCL-next logs show `effect32_film` around `0.645` on
500-episode matched banks, while some low-level RL comparisons mention a
frozen baseline near `0.45` in narrower diagnostic contexts. These numbers
should not be compared directly with the README plot until they use the same
data budget, evaluation bank, episode count, and checkpoint-selection rule.

The first experiment below is therefore mostly a protocol-alignment experiment:
put `effect32_film` on the same plot as the final VAE512 sweep.

## Experiment A: Effect32 FiLM Budget Sweep

### Question

Does `effect32_film` improve sample efficiency over the final VAE512 hierarchy
and flat baselines when every representation and policy is retrained from
scratch at each data budget?

### Main Protocol

Use the exact VAE512 budget protocol:

```text
N:     50, 100, 200, 500, 1000, 1800, 4000, 8000
seeds: 0, 1, 2
eval:  500 learned-high deployable episodes per seed
oracle diagnostic: 50 branch-oracle episodes per seed
```

For each `(N, seed)`:

1. Use the nested VAE512 data manifest for the same `N`.
2. Train `effect32` representation from scratch on that budget.
3. Prepare encoded episodes for that representation.
4. Train the deterministic `effect32_film` hierarchy from scratch.
5. Evaluate learned-high deployment for 500 episodes.
6. Evaluate branch-oracle goals for 50 episodes.
7. Run goal diagnostics on at least the key budgets `N=500`, `N=1800`, and
   `N=8000`.

The important point is that the `effect32` encoder must be retrained per
budget. Reusing the current shared
`artifacts/incremental/learned_interface/effect32/seed0/` artifact would leak
full-budget representation training into small-budget points.

### Current Code Status

The existing VAE512 sweep is close but not directly reusable:

```text
scripts/run_vae_scaling_sweep.sh
uv run hcl-poc incremental vae-scaling-train
uv run hcl-poc incremental vae-scaling-eval
uv run hcl-poc incremental vae-scaling-aggregate
```

That path is currently hardwired to:

```text
VAE_CANDIDATE = "vae512_w2048_b1e6"
experiment = "vae512_sample_efficiency"
output dirs = artifacts/incremental/vae512_scaling/
              results/incremental/vae512_scaling/
```

So this experiment needs a small implementation step before the sweep:

1. Generalize the scaling config helper to accept a learned-interface
   candidate and an experiment/output name, or add a narrow
   `effect32-scaling-*` command family.
2. Reuse the existing nested manifest validation and extended 8k dataset.
3. Store new outputs separately, for example:

```text
artifacts/incremental/effect32_film_scaling/n{N}/...
results/incremental/effect32_film_scaling/n{N}/...
docs/results/effect32_film_scaling/...
```

4. Generate both an `effect32_film`-only plot and a combined plot that overlays
   `effect32_film` on the existing VAE512 deployable success curves. Once the
   sweep is final, update the README plot so it includes all previous VAE512
   architectures plus `effect32_film`.

### Staging

Run this in three stages. The protocol check is the first serious run after a
single smoke point; do not jump directly to the full eight-budget sweep.

1. **Pipeline smoke:** `N=500`, `seed=0`, learned eval `20`, oracle eval `20`.
   This validates that budgeted effect-code training, FiLM hierarchy training,
   and evaluation all use the intended directories.
2. **Protocol check:** `N=500` and `N=1800`, all three seeds, learned eval
   `500`, oracle eval `50`. This directly resolves the `0.45`/`0.6`/`0.65`
   comparison confusion under one protocol.
3. **Full sweep:** all eight budgets and all three seeds.

### Expected Outputs

Produce:

```text
results/incremental/effect32_film_scaling/aggregate/aggregate.json
results/incremental/effect32_film_scaling/aggregate/summary.csv
results/incremental/effect32_film_scaling/aggregate/combined_comparison.csv
docs/results/effect32_film_scaling/success_deployable_effect32_only.png
docs/results/effect32_film_scaling/success_deployable_with_vae512.png
docs/results/effect32_film_scaling/success_oracle.png
docs/results/effect32_film_scaling/goal_diagnostics.csv
```

After the full sweep is accepted, update:

```text
docs/results/vae512_scaling/success_deployable.png
README.md
```

The README figure should remain the main combined comparison: all previous
VAE512 sample-efficiency architectures plus the new `effect32_film` curve.

The main comparison should report:

- success mean +/- seed SD at each `N`
- normalized area under the learning curve
- learned-vs-oracle gap at each `N`
- goal sensitivity diagnostics at `N=500`, `N=1800`, `N=8000`
- whether `effect32_film` is better at small budgets, large budgets, or only
  under the older 100-episode development protocol

### Decision Rule

Promote `effect32_film` as a real sample-efficiency result only if at least one
of these is true under the final 500-episode protocol:

- it improves normalized AUC over the VAE512 deterministic hierarchy by a
  meaningful margin, or
- it clearly improves the low-budget region (`N <= 1800`) without giving back
  the gain at high budgets, or
- it has a smaller learned-vs-oracle gap and stronger goal sensitivity at
  matched success.

If it only ties VAE512 around `N=1800`, it is still useful, but the conclusion
should be that compact action-aware effect codes match the strong VAE state
interface rather than dominate it.

## Experiment B: Scratch Low-Level RL

### Question

Can a low-level policy trained from scratch with RL learn to use the supplied
goal more strongly than the BC low level, improving local reachability and
closed-loop task success?

### Why This Is Different From Current R3

The current R3 training path is not scratch RL. `DirectLowActorCritic` deep
copies the BC low model, freezes almost all of it, trains only the final layer
plus log standard deviation and critic, and optionally adds a BC loss:

```text
trainable_scope = low_policy_final_layer_plus_logstd_and_critic
```

Setting `--bc-weight 0` would remove the BC loss but would still start from the
BC policy and keep most of the low-level network frozen. The scratch experiment
needs a new mode.

### Proposed Training Modes

Implement one explicit new mode rather than overloading R3 semantics:

```text
low-level-rl train-scratch
```

or add an explicit initialization flag:

```text
low-level-rl train-r3 --init-mode scratch_full_low
```

The scratch agent should:

- use the same low-level condition as the deployed hierarchy:
  `[current frame, goal, previous action, remaining fraction]`
- initialize the actor randomly, not from `frozen.low_model`
- train the full actor network, not only the final layer
- keep the representation encoder and high-level policy fixed
- use demonstration data only for reset/goal sampling, not for BC loss
- fail clearly if a required representation or hierarchy checkpoint is absent

The existing BC low level remains the baseline, not an initialization.

One implementation caveat: the current `low-level-rl train-r3` CLI only exposes
`--n-demo` choices `500` and `1000`. This experiment needs budgeted
learned-interface checkpoints at `N=500` and `N=1800`, so the new scratch path
should not inherit that parser restriction.

### Budgets

Run only the two budgets that answer the immediate question:

```text
N=500
N=1800
```

For each budget, use the best available `effect32_film` checkpoint from
Experiment A. If Experiment A has not yet finished, use the current shared
`effect32_film` checkpoint only for code smoke tests and label the result as
non-final.

### Training Curriculum

Stage the RL work so failures are cheap:

1. **Scratch sanity smoke**
   - budget: `N=500`, seed0
   - envs: 512 or 1024
   - steps: short smoke only
   - goal: verify the policy produces nontrivial actions and the reward is
     wired correctly

2. **Seed0 reward-selection runs**
   - budgets: `N=500`, `N=1800`
   - seed: 0 only
   - train on reachable local branch goals
   - use `D_psi` as the learned distance/reachability signal
   - compare reward variants before scaling:
     - pure terminal `D_psi`
     - paired terminal improvement over the frozen BC low level
     - terminal `D_psi` plus a small progress term
     - optional small task-reward mixture only if the first three are unstable

3. **Scaled scratch run**
   - budgets: `N=500`, `N=1800`
   - reward: only the most promising seed0 reward variant from the previous
     stage
   - seeds: 0, 1, 2
   - final evaluation: run this three-seed final evaluation only for the
     selected reward variant, not for every reward candidate
   - use substantially more steps than the R3 fine-tune, because learning from
     scratch is a harder exploration problem

4. **Optional curriculum if scratch fails immediately**
   - start from oracle local goals with short horizons
   - then move to the normal 10-step goal horizon
   - do not add BC action loss unless we explicitly decide the experiment is
     no longer "from scratch"

### Metrics

Measure local reachability and full deployment separately.

Local metrics:

- terminal distance to goal in the learned effect space
- improvement over the frozen BC low level on matched local branches
- success under oracle local goals
- action saturation rate
- action magnitude and policy entropy

Closed-loop task metrics:

- learned-high success over 500 fresh episodes
- oracle-goal success over 100 to 500 episodes
- matched frozen-vs-scratch episode deltas
- final reward and max reward

Goal-use diagnostics:

- `goal_shuffle_action_change_l2`
- `frame_shuffle_action_change_l2`
- `max_same_state_horizon_sensitivity_l2`
- shuffled-goal closed-loop success

The key success condition is not only higher task success. The scratch low
level should also show stronger goal sensitivity than the BC low level, because
the whole hypothesis is that RL can learn to actually use the goal.

### Baselines

Compare against:

- frozen `effect32_film` BC hierarchy at the same budget
- best current R3/fine-tuned low-level checkpoint, if available for the same
  budget and protocol
- VAE512 deterministic hierarchy at the same budget from the README sweep
- oracle-goal frozen hierarchy, to estimate the reachable ceiling

### Decision Rule

Scale scratch RL beyond seed0 only if it passes both gates:

1. Local reachability improves over frozen on matched branches without large
   action saturation.
2. Full deployment is at least neutral on a fresh 500-episode learned-high
   evaluation, or oracle-goal success improves enough to show a real low-level
   reachability gain.

Reject or redesign the scratch objective if it only improves latent/effect
distance while reducing task success, matching the failure mode seen in longer
R3 training.

## Confirmed Execution Order

The current decisions are:

- start the effect32 budget sweep with the `N=500`/`N=1800` protocol check
  after one smoke point
- produce both effect32-only plots and a combined README plot
- use `D_psi` for scratch RL reward design
- try several seed0 reward variants at `N=500` and `N=1800`
- run the three-seed scratch RL final evaluation only for the most promising
  reward variant

Execution order:

```text
first: implement effect32_film scaling support
then:  smoke N=500 seed0 with short evals
then:  run N=500 and N=1800 for seeds 0,1,2 under final eval episodes
then:  review before launching the full 8-budget sweep
then:  implement scratch low-level RL only after the budgeted effect32
       checkpoints exist for N=500 and N=1800
then:  run seed0 scratch RL reward-selection variants at N=500 and N=1800
then:  run three-seed scratch RL final eval only for the selected reward
```
