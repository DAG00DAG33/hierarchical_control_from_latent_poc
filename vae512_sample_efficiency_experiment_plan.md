# VAE-512 Sample-Efficiency Comparison Plan

## 1. Objective

Measure how policy performance scales with the number of successful causal
training trajectories when the selected learned interface is the 512D VAE
future-state latent.

The comparison must answer:

1. Does a VAE-512 hierarchy use demonstrations more efficiently than flat
   control?
2. Does deterministic or flow-matching prediction work better at the high
   level?
3. How much performance is lost relative to an exact reachable future-latent
   oracle?
4. Is compression into the VAE latent helpful or harmful for flat policies?
5. Are any gains consistent across three complete training seeds?

The action-aware learned effect interface is excluded from this experiment.
The only learned high/low interface is the selected VAE-512 future state.

---

## 2. Fixed Reference Configuration

### 2.1 Environment

```text
environment: PushT-v1
backend: CUDA PhysX
controller: pd_ee_delta_pos
control frequency: 20 Hz
episode limit: 100 controls
reward: normalized dense
```

### 2.2 Observation

The full observation is the existing 6,549D vector:

```text
6,528D frozen facebook/dinov2-small spatial RGB feature
+ 21D non-privileged robot/TCP proprioception
```

DINO remains frozen in every method. No method receives T-object privileged
state at deployment.

### 2.3 VAE interface

Use the selected representation family:

```text
candidate: vae512_w2048_b1e6
latent dimension: 512
hidden width: 2,048
beta: 1e-6
KL warmup: 20,000 optimizer steps
free bits: 0.01 per dimension
```

Deployment always uses the posterior mean:

$$
z_t=\mu_\phi(o_t)
$$

Do not use posterior samples as policy inputs.

### 2.4 Hierarchical timing

Keep the successful learned-interface timing fixed:

```text
k = 10 future controls = 0.50 s
U = 10 controls between high-level updates
H = 1 primitive action
```

The low level observes the current frame every control and receives previous
executed action plus remaining-time fraction. The future goal is held for ten
controls.

---

## 3. Data Budgets and Splits

Use the successful PPO trajectories collected with the fixed expert and
downstream action space:

```text
data/prepared/pusht_ppo_dino_spatial_proprio_tcp.h5
```

Keep the original final 200 trajectories as one fixed validation set for every
run. Preserve the original first 1,800 trajectories exactly and append new
successful expert rollouts. Use nested training prefixes:

$$
N\in\{50,100,200,500,1000,1800,4000,8000\}
$$

The subset for budget `N` must be identical across methods and training seeds.
A larger subset must contain every trajectory in each smaller subset.

For every point, record:

- number of causal trajectories;
- number of causal transitions;
- equivalent behavior time at 20 Hz;
- validation trajectories and transitions;
- exact trajectory identifiers or deterministic prefix definition;
- SHA256 or equivalent fingerprint of the split manifest.

No DAgger, recovery queries, privileged state-query samples, or extra
trajectories may enter the primary comparison.

---

## 4. Training Seeds and Sharing Rules

Use complete training seeds:

```text
seed in {0, 1, 2}
```

A training seed controls:

- VAE initialization and minibatch sampling;
- policy initialization and minibatch sampling;
- flow-matching noise and time sampling;
- checkpoint selection tie breaking.

For each `(N, seed)`:

1. Train one VAE-512 from scratch.
2. Use that frozen VAE for every latent-input method at that point.
3. Train one hierarchical low-level policy from scratch and share it between
   deterministic, flow, and oracle high-level evaluations.
4. Train every high-level and flat policy independently.
5. Fit normalizers using only the `N` training trajectories.

Do not transfer checkpoints from a larger budget, another seed, or the
existing 1,800-trajectory result. Warm starting would invalidate the learning
curve.

The sharing above removes irrelevant training variation inside a point while
the three complete seeds measure representation and policy training variance.

---

## 5. Required Methods

Seven methods are required. The learned effect interface must not appear in
training, evaluation, or plots.

### M1. Deterministic VAE hierarchy

High level:

$$
\hat z_{t+10}=H_{det}(o_t,a_{t-1})
$$

Use the established depth-4, width-512 MLP and normalized future-latent MSE.

Low level:

$$
a_t=\pi_{low}(o_t,\hat z_{base+10},a_{t-1},\tau)
$$

Use the selected absolute-goal concatenation controller from
`vae512_w2048_b1e6`. The held goal and time-to-go sampling must match its
original training protocol.

### M2. Flow-matching VAE hierarchy

Replace only the deterministic high level with a conditional flow model:

$$
\hat z_{t+10}\sim H_{flow}(z\mid o_t,a_{t-1})
$$

The VAE and hierarchical low level must be identical to M1 for the same
`(N, seed)`. Use the repository's rectified conditional flow implementation,
width 512, depth 4, 64D sinusoidal time embedding, and 24 Euler integration
steps unless an implementation audit shows the selected prior experiment used
different fixed values.

Use one reproducible Gaussian initial sample per high-level decision. Key the
noise stream by training seed, evaluation seed, episode, and decision index.
Do not select among multiple samples using privileged reward or state.

### M3. Reachable branch-oracle VAE hierarchy

At every high-level update:

1. Copy or exactly replay the current student simulator state into a branch.
2. Roll the deterministic privileged teacher for 10 controls.
3. Render the reached observation.
4. Encode it with the same frozen VAE posterior mean.
5. Supply that reachable future latent to the shared hierarchical low level.

This is a diagnostic and is not deployable. Verify exact branch-state and
transition parity before running the sweep.

### M4. Deterministic flat latent policy

Input:

```text
[VAE posterior mean z_t, previous executed action]
```

Output one deterministic action with a width-512, depth-4 MLP. Train with
normalized action MSE.

### M5. Generative flat latent policy

Input condition:

```text
[VAE posterior mean z_t, previous executed action]
```

Generate one action with conditional rectified flow matching. Use the same
flow width, depth, time embedding, Euler steps, action normalization, and
reproducible noise convention as M2.

### M6. Deterministic flat full-observation policy

Input:

```text
[full 6,549D observation, previous executed action]
```

Output one deterministic action with the same width-512, depth-4 MLP family as
M4. Train with normalized action MSE.

### M7. Generative flat full-observation policy

Condition the same rectified action-flow architecture as M5 on:

```text
[full 6,549D observation, previous executed action]
```

M5 versus M7 isolates VAE compression. M4 versus M5 and M6 versus M7 isolate
deterministic versus generative action prediction. M1 versus M2 isolates the
high-level prediction family.

---

## 6. Fairness Requirements

Use identical values across matched methods for:

- trajectory subsets and validation set;
- evaluation seeds;
- action clipping;
- previous executed-action semantics;
- current frame extraction and DINO model;
- normalizer fitting rules;
- hidden width and depth where the model role is matched;
- optimizer, learning rate, batch size, and checkpoint-selection budget where
  practical;
- maximum episode length and success definition.

Every action must be inverse-normalized and clipped to the environment action
box before execution. The next previous-action input must be the clipped,
actually executed action.

Report model parameter counts and training optimizer steps. If parameter
counts differ materially because the raw observation is much larger, report
both the main architecture-matched result and parameter count; do not silently
change widths to equalize capacity.

---

## 7. Checkpoint Selection and Leakage Prevention

Select checkpoints using only the fixed validation trajectories.

Recommended criteria:

| component | selection metric |
| --- | --- |
| VAE | validation reconstruction objective |
| deterministic flat policies | validation physical action MAE |
| action-flow policies | validation flow-matching loss, with action MAE as diagnostic |
| hierarchical low level | validation oracle-goal physical action MAE |
| deterministic high level | validation predicted-goal induced action error |
| flow high level | validation sampled-goal induced action error |

Closed-loop evaluation results must never select epochs, hyperparameters, or
random seeds.

Before the full sweep, run one `N=50, seed=0` smoke point and audit:

- no validation leakage;
- VAE and normalizers are newly trained inside the point directory;
- all seven methods load the intended point-specific artifacts;
- learned hierarchy and oracle share exactly the same low level;
- deterministic and flow methods use the intended inference mode;
- result manifests record all artifact paths and Git/config metadata.

---

## 8. Evaluation Protocol

### 8.1 Deployable methods

Evaluate M1, M2, M4, M5, M6, and M7 on:

```text
500 episodes per (method, N, training seed)
```

Use one fixed bank of 500 environment seeds for every method, budget, and
training seed. This gives paired episode-level comparisons.

For stochastic flow policies, use fixed reproducible policy-noise streams.
Environment seed and policy-noise seed must be logged separately.

### 8.2 Oracle method

The branch oracle is substantially more expensive. Evaluate M3 on:

```text
50 episodes per (N, training seed)
```

Use the first 50 seeds from the same 500-seed bank. If measured runtime is
acceptable after the first two budget points, the oracle may be increased to
100 episodes, but it must not block completion of the deployable comparison.
Every table and plot must show the oracle's smaller episode count.

### 8.3 Runtime discipline

Run evaluations vectorized where valid. Do not poll long-running training or
evaluation jobs repeatedly. Save atomic per-run result files so interrupted
runs resume without repeating completed work.

Before launching the sweep, record:

- GPU model and free memory;
- free disk space;
- measured training time for the smoke point;
- measured deployable and oracle rollout throughput;
- projected total runtime and storage.

---

## 9. Metrics

### Primary

- episode success rate;
- mean and sample standard deviation across the three training seeds;
- per-run 95% Wilson interval;
- final normalized reward;
- maximum normalized reward.

### Control diagnostics

- teacher action MAE where the privileged teacher is queried for diagnostics;
- action saturation rate;
- decisions per episode;
- policy inference latency;
- branch generation latency for the oracle;
- VAE future-latent prediction L2;
- prediction-induced low-level action L2;
- oracle-versus-learned hierarchy gap.

### Sample-efficiency summaries

Compute using trajectories and transitions:

$$
N_{50}=\min\{N:\mathrm{Success}(N)\ge0.50\}
$$

$$
N_{70}=\min\{N:\mathrm{Success}(N)\ge0.70\}
$$

and area under the success curve over log data size:

$$
\mathrm{AULC}=\int \mathrm{Success}(\log N)\,d\log N
$$

Use the mean across training seeds for threshold and AULC summaries. State
when a method never reaches a threshold.

---

## 10. Required Plots

### P1. Main deployable learning curve

- x-axis: training trajectories;
- y-axis: success rate;
- six deployable methods;
- point: mean across three training seeds;
- band or error bar: sample standard deviation across seeds;
- annotate `500 evaluation episodes per seed`;
- use either categorical spacing matching the reference plot or a log x-axis;
  do not imply linear spacing between budgets.

### P2. Hierarchy-focused curve

Plot:

- deterministic VAE hierarchy;
- flow VAE hierarchy;
- branch-oracle VAE hierarchy.

Use a visually distinct dashed oracle curve and state its smaller evaluation
budget in the legend/caption.

### P3. Flat observation/representation ablation

Use facets or paired panels:

- deterministic versus flow;
- VAE latent versus full observation.

### P4. Learned-to-oracle gap

For deterministic and flow high levels, plot success relative to oracle and
future-latent prediction/induced-action error versus trajectory budget.

### P5. Reward curves

Produce final-reward and maximum-reward versions of the main deployable plot.

All plots must use percentages or fractions consistently, include seed counts,
and avoid uncertainty bands that extend outside `[0,1]`.

---

## 11. Required Tables

### T1. Data and compute

| trajectories | transitions | behavior seconds | train time by component | rollout time |

### T2. Success by method and budget

Report mean, sample SD, each individual training-seed value, and evaluation
episode count.

### T3. Reward and control diagnostics

Report final/max reward, action MAE, saturation, latency, and decisions per
episode.

### T4. Sample efficiency

| method | N50 trajectories/transitions | N70 trajectories/transitions | AULC |

### T5. Architecture and hyperparameters

Record exact input dimensions, parameter counts, optimizer settings, epochs,
optimizer steps, flow integration settings, and checkpoint criteria.

---

## 12. Statistical Reporting

The unit of training variation is the complete training seed, not the episode.
For each plotted point:

```text
mean success across 3 independently trained seeds
+/- sample SD across those 3 seeds
```

Also retain the 500 episode-level outcomes per run. Use paired bootstrap or
paired success differences over the fixed environment seed bank as a
secondary within-seed comparison. Do not treat 1,500 pooled episodes as 1,500
independent training replicates.

With only three training seeds, avoid strong significance claims from small
differences. Report effect sizes and all seed values.

---

## 13. Execution Order

1. Implement an isolated VAE-512 scaling experiment namespace and manifests.
2. Add deterministic and flow flat policies for VAE latent and full
   observation inputs.
3. Add deterministic, flow, and reachable-oracle VAE hierarchies sharing one
   low level per `(N, seed)`.
4. Add unit tests for nested splits, artifact isolation, model input shapes,
   flow sampling reproducibility, and previous executed-action handling.
5. Run `N=50, seed=0` smoke training and short rollout checks.
6. Measure runtime, GPU memory, and disk usage; write an ETA to the experiment
   log.
7. Run all representations and policies for seeds 0-2, budget by budget.
8. Evaluate every deployable method on 500 fixed episodes.
9. Evaluate the oracle on 50 fixed episodes per point.
10. Aggregate only after validating result completeness and seed identity.
11. Generate required plots, tables, representative videos, and failure
    summaries.
12. Update README and a dedicated final report with exact commands, artifacts,
    data, architectures, hyperparameters, and conclusions.

Prefer completing all methods for one `(N, seed)` before moving on, so partial
results remain interpretable. Commit implementation, smoke validation,
completed budget blocks, and final reporting as separate checkpoints.

---

## 14. Experiment Log Requirements

Maintain a dedicated chronological log from the first implementation change.
For every run record:

- date and Git commit;
- command and config;
- budget and trajectory manifest;
- training seed and evaluation seed bank;
- representation, policy, and low/high checkpoint paths;
- hyperparameters and parameter counts;
- training duration and selected epoch;
- offline validation metrics;
- closed-loop metrics and episode count;
- failures, reruns, and reasons for pruning or changing anything.

Never overwrite a failed result without preserving the failure diagnosis.

---

## 15. Decision Rules

The main thesis is supported if either learned VAE hierarchy:

1. reaches `0.50` or `0.70` success with fewer trajectories than every flat
   deployable baseline; or
2. has materially higher AULC across the complete nested-budget curve.

The hierarchy is useful but not more sample efficient if it matches the best
flat method at 1,800 trajectories without improving N50, N70, or AULC.

The high level is the bottleneck if the oracle curve is consistently above
both learned hierarchy curves.

The VAE representation helps flat control if flat latent policies outperform
their matched full-observation policies. It hurts if the reverse is stable
across seeds.

Flow matching helps only if it improves the matched deterministic method
across multiple budgets or materially improves AULC. A single noisy point is
not sufficient.

A scientifically useful negative result is:

> VAE-512 gives strong full-data performance, but its hierarchy does not
> improve demonstration sample efficiency relative to matched flat latent or
> full-observation policies.

---

## 16. Completion Criteria

The experiment is complete when:

- all eight budgets and three training seeds exist for all six deployable
  methods;
- every deployable point has 500 fixed-seed episodes;
- every oracle point has at least 50 fixed-seed episodes;
- no learned effect-interface artifact is used;
- manifests prove nested data and artifact isolation;
- all required plots and tables are generated;
- representative success/failure videos are available for the full-data
  candidates;
- the log and final report document exact training and deployment details;
- README states whether VAE hierarchy improves sample efficiency.
