# Final Push-T Results and Candidate Specifications

## 1. Scope

This document specifies the five candidates in the final Phase 12
sample-efficiency comparison. It records:

- the data available to each candidate;
- the model architecture and optimization hyperparameters;
- checkpoint selection;
- the exact closed-loop deployment procedure;
- the experiments that motivated the final choices;
- the final results and their limitations.

The five reported candidates are:

1. direct visual deterministic behavioral cloning (`visual_bc`);
2. direct visual one-step flow matching (`visual_flat_flow`);
3. future-latent hierarchy with an online privileged branch oracle
   (`oracle_hierarchy`);
4. future-latent hierarchy with a deterministic learned high level
   (`deterministic_hierarchy`);
5. future-latent hierarchy with a conditional-flow high level
   (`generative_hierarchy`).

The oracle is a diagnostic, not a deployable policy: it runs the privileged
teacher online to construct a reachable future goal. The other four methods
use only RGB, robot proprioception, and their previous executed action at
deployment.

## 2. Common Task and Data

### 2.1 Environment

| setting | value |
| --- | --- |
| environment | ManiSkill `PushT-v1` |
| simulator backend | CUDA PhysX |
| control mode | `pd_ee_delta_pos` |
| control frequency | 20 Hz |
| action | 3D end-effector delta-position command |
| action bounds | `[-1, 1]` in every dimension |
| reward | ManiSkill normalized dense reward |
| episode limit | 100 control steps |

Every predicted action is converted from its training normalization and then
clipped to the environment action box before execution. The normalized
previous-action input is updated with this clipped, actually executed action.
It is initialized from the zero action at every reset.

### 2.2 Teacher and causal trajectories

The downloaded demonstration actions did not replay successfully with the
installed simulator and selected controller. The project therefore trained a
privileged PPO teacher in the same `pd_ee_delta_pos` action space and collected
2,000 successful trajectories from it. The final policies imitate the
teacher's deterministic, clipped actions.

The prepared dataset is:

```text
data/prepared/pusht_ppo_dino_spatial_proprio_tcp.h5
```

The last 200 trajectories are always the validation set. Phase 12 uses nested
prefixes of the first 1,800 trajectories for training. Thus every larger
budget contains all trajectories from every smaller budget, and validation is
constant.

| training trajectories | unique causal transitions | behavior time at 20 Hz | valid `k=2` low-level samples |
| ---: | ---: | ---: | ---: |
| 50 | 2,311 | 115.55 s | 2,211 |
| 100 | 4,507 | 225.35 s | 4,307 |
| 200 | 8,834 | 441.70 s | 8,434 |
| 500 | 22,367 | 1,118.35 s | 21,367 |
| 1,000 | 44,605 | 2,230.25 s | 42,605 |
| 1,800 | 80,472 | 4,023.60 s | 76,872 |

The fixed validation set contains 8,969 transitions and 8,569 valid `k=2`
samples. Phase 12 uses no DAgger or separately collected state-query samples;
`state_query_samples=0` for every budget. This is important because the final
horizontal axis measures only causal trajectories/transitions.

### 2.3 Visual observation

The policy observation is formed as follows:

1. Read the base-camera RGB image.
2. Run frozen `facebook/dinov2-small`.
3. Retain the 384D CLS token.
4. Adaptively pool the patch-token grid to `4 x 4`, giving 16 additional 384D
   tokens.
5. Flatten and concatenate these values into a 6,528D spatial DINO feature.
6. Append the first 21 dimensions of the ManiSkill state observation. These
   are robot/TCP proprioceptive quantities and exclude the goal and T-object
   privileged state.

The resulting frame is 6,549D. DINO is frozen in every experiment. Features
were precomputed for training and recomputed online from RGB during rollout.

### 2.4 Normalization and sampling

Frame, action, and latent standardizers are fit from the training subset only
and are independently refit at each trajectory budget. Validation data never
contributes normalization statistics.

Training datasets sample a trajectory and valid timestep randomly with
replacement. Therefore the number of optimization examples is fixed by the
schedule rather than by the number of unique transitions. Smaller budgets are
seen more times. This keeps optimization effort constant while changing only
the amount of unique causal behavior, but it can increase overfitting at small
budgets.

All final models use policy seed 0, AdamW with its PyTorch default weight decay
(`0.01`), no learning-rate scheduler, and SiLU MLPs. Every schedule runs for
its full epoch count and retains the best validation checkpoint.

## 3. Shared Architecture Components

### 3.1 MLP definition

`MLP(in, out, width, depth)` contains `depth` blocks of:

```text
Linear -> SiLU
```

followed by a final linear output layer. A depth-4 MLP therefore has four
hidden layers, not four linear layers in total.

### 3.2 Flow model

Both flow candidates use rectified conditional flow matching. For target
`x_1`, Gaussian noise `eps`, and `t ~ Uniform(0,1)`:

```text
x_t = (1 - t) eps + t x_1
target velocity = x_1 - eps
```

The network receives `[x_t, sinusoidal_time_embedding(t), condition]`. The
time embedding is 64D. It is a depth-4 SiLU MLP. Deployment integrates the
learned velocity with 24 explicit Euler steps.

### 3.3 AE-256 observation representation

All three hierarchical candidates use the same representation family,
retrained independently at every budget.

Encoder:

```text
6549 -> 1024 -> 1024 -> 1024 -> 256
```

Decoder:

```text
256 -> 1024 -> 1024 -> 1024 -> 6549
```

Every hidden layer uses SiLU. The encoder has 9,068,800 parameters; the
training-only decoder has 9,075,093. The decoder is discarded at deployment.

Despite the generic Phase 6 training code supporting an action-conditioned
world model, the selected `ae_recon` variant has:

```text
world-model prediction weight = 0
latent variance regularization weight = 0
reconstruction weight = 0.1
VAE KL weight = 0
```

The reconstruction term is computed separately for the 6,528 DINO dimensions
and 21 proprioceptive dimensions:

```text
L_recon = MSE(DINO_hat, DINO) + MSE(proprio_hat, proprio)
L_AE = 0.1 * L_recon
```

This prevents the much larger DINO vector from numerically overwhelming
proprioception. Both the current and sampled future frame are reconstructed in
each training item.

AE training hyperparameters:

| hyperparameter | value |
| --- | ---: |
| hidden width / latent dimension | 1,024 / 256 |
| batch size | 512 |
| batches per epoch | 400 |
| epochs | 60 |
| optimizer steps | 24,000 |
| sampled items | 12,288,000 |
| learning rate | `3e-4` |
| validation items per epoch | 8,192 |
| checkpoint criterion | lowest validation reconstruction loss |

The representation sampler still draws offsets from `{1,2,4,8}`, but for the
AE-only loss these offsets merely select two frames to reconstruct. Actions
and the world model do not affect the selected encoder.

### 3.4 Shared low-level policy

The hierarchical low-level policy is a depth-4, width-1,024 MLP:

```text
input  = [z_t, z_(t+2) - z_t, normalized(a_(t-1))]  # 256 + 256 + 3 = 515
output = normalized(a_t)                              # 3
```

It has 3,680,259 parameters. The action horizon is one step (`H=1`), while the
goal horizon is two steps (`k=2`, 0.10 s), satisfying `H < k`.

The policy is trained only on coherent windows from the successful causal
teacher trajectories. For each valid timestep, the frozen AE encodes current
and two-step future observations, and the current teacher action is the label.
There is no goal dropout and no artificial goal corruption in Phase 12.

| hyperparameter | value |
| --- | ---: |
| batch size | 512 |
| batches per epoch | 300 |
| epochs | 80 |
| optimizer steps | 24,000 |
| sampled items | 12,288,000 |
| learning rate | `3e-4` |
| loss | normalized action MSE |
| checkpoint criterion | lowest full validation normalized action MSE |

At 1,800 trajectories its held-out action MAE is `0.0266` with the correct
future goal, versus `0.4235` with shuffled goals. This large gap verifies that
the policy uses the goal rather than ignoring it.

The final Phase 12 low level is the base causal policy described above, not
the stronger Phase 7 coherent-DAgger checkpoint. The DAgger checkpoint was
excluded because it adds learner-state queries whose amount cannot be counted
as causal trajectories on the final sample-efficiency axis.

## 4. Final Candidate Specifications

### 4.1 Direct visual deterministic BC

**Purpose.** Establish the simplest direct-observation imitation baseline.

**Condition.** One current 6,549D visual/proprio frame plus the normalized
previous action, for 6,552 input dimensions. No temporal history beyond the
previous action is used.

**Architecture.** A three-hidden-layer MLP:

```text
6552 -> 512 -> 512 -> 512 -> 3
```

It uses SiLU and has 3,881,987 parameters.

**Training.** It uses every state-action pair in the selected causal training
trajectories.

| hyperparameter | value |
| --- | ---: |
| batch size | 512 |
| batches per epoch | 500 |
| epochs | 50 |
| optimizer steps | 25,000 |
| sampled items | 12,800,000 |
| learning rate | `3e-4` |
| loss | normalized action MSE |
| validation queries | up to 10,000 (all 8,969 are available) |
| checkpoint criterion | lowest raw-action validation MAE |

**Deployment.** Encode current RGB with DINO, append proprioception and the
previous clipped action, run the MLP once, inverse-normalize, clip, and execute.
The policy is deterministic.

**Why this design.** In development, single-frame concatenation reached 65%
success, compared with 56% for a two-frame concat MLP and 55% for a two-frame
GRU. Longer history was therefore rejected as extra cost without evidence of
benefit.

### 4.2 Direct visual one-step flow

**Purpose.** Provide the matched flat flow policy requested by the original
hypothesis, without introducing a learned latent bottleneck.

**Condition and target.** The same 6,552D condition as visual BC. The flow
target is the normalized 3D current teacher action.

**Architecture.** `FlowModel(sample_dim=3, cond_dim=6552, hidden_dim=512)`.
Including the 64D time embedding, its MLP input is 6,619D. It has four 512D
hidden layers and 4,178,947 parameters.

**Training objective.** Standard flow matching is augmented with a
deterministic endpoint term. For the first 256 items in every batch, the model
is differentiably integrated for four Euler steps from zero noise and the
endpoint is regressed to the action:

```text
L = L_flow_matching + 20 * MSE(zero_noise_endpoint, action)
```

| hyperparameter | value |
| --- | ---: |
| batch size | 512 |
| batches per epoch | 500 |
| epochs | 80 |
| optimizer steps | 40,000 |
| sampled items | 20,480,000 |
| learning rate | `3e-4` |
| endpoint sub-batch / train integration steps | 256 / 4 |
| deployment integration steps | 24 |
| validation interval | 5 epochs |
| checkpoint criterion | lowest zero-noise raw-action validation MAE |

**Deployment.** The initial flow state is always the zero vector, not random
Gaussian noise. After 24 Euler steps, the endpoint is inverse-normalized,
clipped, and executed. This makes the reported flow policy deterministic.

**Why this design.** Random flow samples were too noisy for contact control.
The zero-noise endpoint with consistency weight 20 brought the privileged
flow within 3 percentage points of privileged BC, then produced 66% versus
65% for visual BC during full-budget development. The same recipe was frozen
for Phase 12.

### 4.3 Future-latent oracle hierarchy

**Purpose.** Test the low-level future-state interface independently of the
difficulty of learning a high-level predictor.

**Trained components.** AE-256 plus the shared causal low-level MLP. There is
no trained high-level model.

**Online goal construction.** At every student step:

1. Reset a branch simulator to the same episode seed.
2. Replay the complete sequence of actions already executed by the student.
3. Verify that the replayed physical state matches the current student state.
4. Run the deterministic privileged PPO teacher for two branch steps.
5. Render the reached branch observation and encode it with frozen DINO and
   AE-256 to obtain `g_t`.
6. Run the low level on `[z_t, g_t-z_t, a_(t-1)]`.
7. Execute one student action and recompute a fresh branch goal.

Exact replay is required because direct simulator-state copying reproduced
visible actor state but not hidden contact/solver history. Direct copies
showed transition errors up to 7 mm in contact; reset-and-replay gives zero
measured replay-state error.

**Deployment status.** This is not deployable without the privileged PPO
teacher and additional branch simulator. Its measured branch-generation cost
is orders of magnitude larger than the learned policies.

**Why `k=2`.** Exact-oracle development tested `k={2,5,10,20}`. `k=2` was the
shortest horizon that passed the interface gate and had the lowest held-out
teacher-action MAE. The promising 10-episode `k=10` result did not demonstrate
an advantage large enough to justify the longer contact-control horizon;
`k=20` degraded clearly.

### 4.4 Deterministic future-latent hierarchy

**Purpose.** Test the deployable hierarchy with the simplest learned high
level.

**Condition and target.** The high level receives the normalized current
AE-256 latent and normalized previous action:

```text
condition = [normalize(z_t), normalize(a_(t-1))]  # 259D
target    = normalize(z_(t+2))                    # absolute future latent
```

It does not receive current or future teacher actions. The low level later
converts the predicted absolute goal into `g_t-z_t`.

**Architecture.** A depth-4, width-1,024 MLP:

```text
259 -> 1024 -> 1024 -> 1024 -> 1024 -> 256
```

It has 3,677,440 parameters.

| hyperparameter | value |
| --- | ---: |
| history | 1 |
| future horizon | 2 steps / 0.10 s |
| batch size | 512 |
| batches per epoch | 300 |
| epochs | 60 |
| optimizer steps | 18,000 |
| sampled items | 9,216,000 |
| learning rate | `3e-4` |
| loss | normalized future-latent MSE |
| validation samples | 10,000 requested; 8,569 available |
| checkpoint criterion | lowest validation latent MSE |

**Deployment.** Encode the live observation, predict one absolute future
latent, feed its displacement to the shared low level, clip and execute one
action, then replan at the next control step. Both encoder and low level are
frozen.

**Why this version.** Development rejected longer histories (`L=2,4,8`), a
delta prediction target, `k=5`, old and fresh high-level DAgger, low-level
adaptation, flat/branch action blending, action-consistency fine-tuning, and a
192D AE. Several reduced offline error but none improved the base model's
closed-loop success. The final controlled sweep therefore uses the simplest
absolute, single-history model rather than selecting on a misleading offline
metric.

### 4.5 Generative future-latent hierarchy

**Purpose.** Test whether conditional flow matching avoids deterministic
regression toward an average future latent.

**Condition and target.** Exactly the same 259D normalized
`[z_t,a_(t-1)]` condition and normalized absolute `z_(t+2)` target as the
deterministic high level.

**Architecture.** `FlowModel(sample_dim=256, cond_dim=259,
hidden_dim=1024)`. With the 64D time embedding, the MLP input is 579D. It has
four 1,024D hidden layers and 4,005,120 parameters.

**Training objective.** Conditional flow matching plus a zero-noise endpoint
loss on 128 items per batch:

```text
L = L_flow_matching + 10 * MSE(zero_noise_endpoint, z_(t+2))
```

| hyperparameter | value |
| --- | ---: |
| batch size | 512 |
| batches per epoch | 300 |
| epochs | 60 |
| optimizer steps | 18,000 |
| sampled items | 9,216,000 |
| learning rate | `3e-4` |
| endpoint sub-batch / train integration steps | 128 / 4 |
| deployment integration steps | 24 |
| validation interval | 10 epochs |
| checkpoint criterion | lowest zero-noise endpoint MSE on 2,048 held-out samples |

**Deployment.** The reported candidate starts from zero flow noise, performs
24 Euler steps, inverse-normalizes the absolute future latent, and uses the
shared low level. It replans every control step and is deterministic.

**Why zero noise and endpoint weight 10.** The model overfit one and ten
trajectories, showing the implementation could learn the mapping. At larger
scale, unregularized zero-noise endpoints were poor. Endpoint weights 1 and 10
improved the central endpoint, with 10 selected by held-out endpoint error.
Fresh random samples were much worse: on the full-data development model,
random sampling achieved 5% success on 20 episodes versus 42% for the
zero-noise endpoint on 100 episodes. Sample diversity primarily represented
error rather than useful alternative futures.

## 5. Controlled Phase 12 Procedure

For each budget in `{50,100,200,500,1000,1800}`, the runner creates an
isolated artifact root and trains all components from seed 0. It does not
warm-start a larger budget from a smaller one. The fixed validation set and
evaluation seeds are shared across budgets and methods.

All deployable candidates are evaluated on 100 episodes beginning at reset
seed `1200000`, using up to 64 vector environments. The oracle uses the first
10 of those episode seeds because exact branch replay dominates runtime.
Consequently, oracle error bars are much wider. The reported standard error is
the Bernoulli sample standard deviation divided by `sqrt(n)`.

The protocol measures one training seed only. It supports paired comparisons
over environment initializations, but it does not estimate training-seed
variance.

## 6. Final Results

### 6.1 Success versus causal data

| trajectories | transitions | visual BC | flat flow | oracle hierarchy | deterministic hierarchy | generative hierarchy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 2,311 | 0.00 | 0.03 | 0.00 | 0.01 | 0.03 |
| 100 | 4,507 | 0.05 | 0.07 | 0.10 | 0.02 | 0.05 |
| 200 | 8,834 | 0.10 | 0.20 | 0.40 | 0.08 | 0.03 |
| 500 | 22,367 | 0.29 | 0.28 | 0.80 | 0.14 | 0.23 |
| 1,000 | 44,605 | 0.44 | 0.49 | 0.80 | 0.22 | 0.25 |
| 1,800 | 80,472 | 0.60 | 0.62 | 0.70 | 0.37 | 0.42 |

The corresponding plot is
[`docs/results/incremental_sample_efficiency.png`](docs/results/incremental_sample_efficiency.png).

### 6.2 Dense reward at the largest budget

| candidate | success | standard error | final reward | maximum reward |
| --- | ---: | ---: | ---: | ---: |
| visual BC | 0.60 | 0.049 | 0.582 | 0.707 |
| visual flat flow | 0.62 | 0.049 | 0.598 | 0.730 |
| oracle hierarchy (10 episodes) | 0.70 | 0.145 | 0.800 | 0.801 |
| deterministic hierarchy | 0.37 | 0.048 | 0.389 | 0.525 |
| generative hierarchy | 0.42 | 0.049 | 0.389 | 0.567 |

### 6.3 Threshold and learning-curve summaries

| candidate | first measured `>=50%` | first measured `>=70%` | AULC over log transitions |
| --- | ---: | ---: | ---: |
| visual BC | 80,472 | not reached | 0.807 |
| visual flat flow | 80,472 | not reached | 0.940 |
| oracle hierarchy | 22,367 | 22,367 | 1.754 |
| deterministic hierarchy | not reached | not reached | 0.444 |
| generative hierarchy | not reached | not reached | 0.538 |

The oracle threshold is only a coarse estimate because each point has 10
episodes. In particular, the observed drop from 0.80 at 500/1,000 trajectories
to 0.70 at 1,800 is well inside the wide sampling uncertainty and should not
be interpreted as a reliable negative scaling trend.

## 7. Interpretation

### 7.1 What worked

The exact local-oracle hierarchy is strong evidence that a reachable future
latent is a useful control interface. At 500 trajectories it achieves 8/10
success while both direct visual policies are below 30%. Phase 7's larger
development evaluation also showed that a correctly defined local branch goal
substantially outperforms the old nominal-trajectory goal once the student
deviates.

The direct visual policies scale predictably with causal data. At 1,800
trajectories, visual flow is slightly better than deterministic BC in success
and dense reward, but the difference is smaller than one evaluation standard
error.

### 7.2 What failed

Neither learned high level preserves the oracle's advantage. At 1,800
trajectories, deterministic and generative hierarchies trail flat visual flow
by 25 and 20 percentage points. They also have lower final and maximum reward,
so the failure is not an artifact of the binary success threshold.

The full-data deterministic high level has future-latent L2 error `13.96`
versus persistence error `25.25`, and its predictions are better than
persistence on 97.9% of held-out teacher samples. Nevertheless, replacing the
true goal with that prediction raises low-level action MAE from `0.0266` to
`0.0392` offline and causes much larger compounding error in rollout.

The full-data flow high level is similar offline: zero-noise latent L2 is
`14.06`, predicted-goal action MAE is `0.0401`, stochastic mean L2 is `33.61`,
and best-of-four stochastic L2 is `31.89`. The stochastic samples are less
accurate than its deterministic central endpoint.

### 7.3 Failure attribution

The evidence rules out several simpler explanations:

- Exact reset-and-replay gives zero branch state error, so the positive oracle
  is not based on an unreachable nominal goal or incorrect state copy.
- Goal shuffling strongly changes low-level actions, so the low level does not
  ignore its goal.
- AE-256 probes retain object pose, TCP pose, velocity, contact, inverse
  dynamics, and reward information.
- A privileged structured predictor retained 98.2% of its matched oracle
  success (`0.54` versus `0.55`), so deterministic future prediction and the
  hierarchy as such are feasible.
- Longer history, longer horizon, delta targets, high-level DAgger, low-level
  adaptation, action-consistency loss, and a smaller latent did not recover
  learned-latent performance.
- Training the low level on the measured flow-goal error distribution reduced
  supervised error but lowered closed-loop success.

The remaining bottleneck is therefore specific to predicting a
control-compatible goal in the reconstruction latent. Uniform 256D latent
loss gives equal importance to nuisance/reconstruction directions and to the
directions to which the low-level policy is sensitive. Moderate average
prediction error then becomes compounding contact-control error.

## 8. Final Conclusion

The experiment gives a positive result for the **future-state interface** and
a negative result for the **deployable sample-efficiency hypothesis**.

The oracle shows that locally reachable future latents can make the low-level
problem substantially more data efficient. However, neither deterministic MSE
nor conditional flow matching learns future latents accurately enough in the
control-relevant directions to realize that advantage without privileged
online branch generation.

The most direct next experiment is not another low-level robustness loss. It
is to change the high-level goal representation or objective so that distance
is physically and control aligned: for example, predict a compact structured
task latent, learn a controllability-aware metric, or jointly learn a smaller
goal bottleneck while preserving the already validated current-state
representation.

## 9. Reproduction and Sources

Final sweep:

```bash
CONFIG=configs/pusht_incremental.yaml

for N in 50 100 200 500 1000 1800; do
  uv run hcl-poc incremental phase12-run \
    --config "$CONFIG" \
    --n-trajectories "$N" \
    --episodes 100 \
    --eval-seed-start 1200000
done

uv run hcl-poc incremental phase12-plot --config "$CONFIG"
```

Canonical sources:

- hyperparameters: [`configs/pusht_incremental.yaml`](configs/pusht_incremental.yaml);
- implementation: `src/hcl_poc/incremental.py`, `src/hcl_poc/models.py`, and
  `src/hcl_poc/flow.py`;
- complete experiment history:
  [`INCREMENTAL_EXPERIMENT_LOG.md`](INCREMENTAL_EXPERIMENT_LOG.md);
- machine-readable final curves:
  [`docs/results/incremental_sample_efficiency.json`](docs/results/incremental_sample_efficiency.json).
