# Push-T Pre-RL Diagnostic and Improvement Plan

## 1. Objective

The current proof of concept establishes two important but conflicting results:

- A **reachable oracle future latent** can be a strong low-level interface.
- Once the future latent must be predicted by a learned high-level model, the hierarchy performs substantially worse than the direct visual baselines.

At the largest data budget:

| Method | Success |
| --- | ---: |
| Direct visual BC | 0.60 |
| Direct visual flow | 0.62 |
| Oracle future-latent hierarchy | 0.70 over 10 episodes |
| Deterministic learned hierarchy | 0.37 |
| Generative learned hierarchy | 0.42 |

The oracle result has high uncertainty because it was evaluated on only 10 episodes, but it is still evidence that the interface may be useful. The learned high-level result shows that the deployable hierarchy is not yet ready for RL.

The purpose of this plan is to determine:

1. What information makes the oracle goal useful.
2. Whether the current hierarchy is genuinely temporally abstract.
3. Whether recovery-rich, teleoperation-like data changes the comparison.
4. Whether the AE latent is an appropriate high-level goal representation.
5. Whether the problem is the representation, the high-level predictor, the low-level interface, or the task/horizon itself.
6. Whether RL should be started, and which components should be frozen when it begins.

Do not begin the main RL study until the gates at the end of this document are met or a clear negative conclusion is reached.

---

# 2. General Experimental Rules

## 2.1 Canonical environment

Use the existing canonical setup:

| Setting | Value |
| --- | --- |
| Environment | ManiSkill `PushT-v1` |
| Backend | CUDA PhysX |
| Controller | `pd_ee_delta_pos` |
| Control frequency | 20 Hz |
| Episode limit | 100 actions |
| Base image encoder | Frozen DINOv2-small spatial features |
| Current learned representation | `ae_recon_z256` |
| Clean causal dataset | 1,800 train trajectories and 200 fixed validation trajectories |

Do not mix CPU and CUDA results.

## 2.2 Fair comparison rule

Every flat and hierarchical policy in a comparison must use:

- the same causal trajectories;
- the same training/validation split;
- the same visual observations;
- the same proprioception;
- the same action labels;
- the same reset seeds;
- comparable optimization budgets;
- separately fitted normalization statistics;
- no privileged data at deployment unless explicitly labeled as an oracle diagnostic.

## 2.3 Representation retraining rule

For any experiment that changes the available causal dataset, retrain the encoder from scratch.

Examples:

- clean data versus recovery-rich data;
- 500 trajectories versus 1,800 trajectories;
- base camera versus base plus wrist camera;
- different ratios of clean and recovery data.

Do not reuse the full-data AE when making a claim about representation sample efficiency.

For cheap interface diagnostics that do not change the data distribution, the frozen existing AE may be reused initially. Any final comparison must retrain it.

## 2.4 Evaluation budgets

Use:

- 20 episodes for smoke tests;
- 100 fixed-seed episodes for development decisions;
- at least 3 training seeds and 200 evaluation episodes per seed for the selected final comparisons;
- at least 50 episodes for expensive online oracle evaluations at the selected settings.

Report binomial confidence intervals for success.

## 2.5 Required logging

Every experiment must record:

- Git commit;
- command;
- configuration;
- dataset name and construction;
- number of causal trajectories and transitions;
- representation checkpoint;
- policy seed;
- evaluation seeds;
- success;
- final and maximum reward;
- action MAE;
- subgoal-reaching error;
- runtime;
- videos for at least 10 representative successes and failures;
- gate decision;
- diagnosis and next action.

---

# 3. Phase A: Confirm the Final Performance Gap Statistically

## Goal

Verify that the observed gap is not mainly training-seed or evaluation noise.

## Methods

At the 1,800-trajectory budget, retrain:

1. Direct visual BC.
2. Direct visual flow.
3. Matched flat AE-latent policy.
4. Deterministic learned hierarchy.
5. Generative learned hierarchy.

Use:

- 3 training seeds;
- 200 evaluation episodes per seed;
- identical evaluation reset seeds.

Evaluate the oracle hierarchy for at least 50 episodes using the selected 256D encoder and low-level policy.

## KPIs

- Mean success across training seeds.
- Standard deviation across training seeds.
- Binomial confidence interval within each seed.
- Final and maximum reward.
- Action MAE.
- Oracle-to-learned hierarchy gap.
- Flat-to-hierarchy gap.

## Gate

Proceed if the qualitative result remains:

$$
\mathrm{Success}_{\mathrm{oracle}}
>
\mathrm{Success}_{\mathrm{flat}}
>
\mathrm{Success}_{\mathrm{learned\ hierarchy}}
$$

If the learned hierarchy overlaps the flat baseline after multiple seeds, reduce the priority of representation redesign and proceed first to the recovery-data experiment.

---

# 4. Phase B: Determine What Makes the Oracle Goal Useful

## Goal

Determine whether the oracle future state communicates:

- the desired future physical effect;
- the expert's future robot motion;
- or both.

This is the highest-priority conceptual diagnostic.

The current oracle goal is based on a latent encoding of both future scene information and future robot proprioception. At the current short horizon, future TCP state may nearly reveal the next action.

## B1. Privileged structured oracle decomposition

Use the same exact local branch oracle as in Phase 7.

At each student state, roll the privileged teacher forward from that exact state and construct the following future-goal variants.

### Full structured goal

$$
g_t^{\mathrm{full}}
=
\left[
p_{t+k}^{T},
\sin\theta_{t+k}^{T},
\cos\theta_{t+k}^{T},
v_{t+k}^{T},
\omega_{t+k}^{T},
p_{t+k}^{\mathrm{TCP}},
v_{t+k}^{\mathrm{TCP}},
q_{t+k},
\dot q_{t+k},
c_{t+k}
\right]
$$

### Robot-only goal

$$
g_t^{\mathrm{robot}}
=
\left[
p_{t+k}^{\mathrm{TCP}},
v_{t+k}^{\mathrm{TCP}},
q_{t+k},
\dot q_{t+k}
\right]
$$

### TCP-only goal

$$
g_t^{\mathrm{TCP}}
=
\left[
p_{t+k}^{\mathrm{TCP}},
v_{t+k}^{\mathrm{TCP}}
\right]
$$

### Object-only goal

$$
g_t^{\mathrm{object}}
=
\left[
p_{t+k}^{T},
\sin\theta_{t+k}^{T},
\cos\theta_{t+k}^{T},
v_{t+k}^{T},
\omega_{t+k}^{T}
\right]
$$

### Object-pose-only goal

$$
g_t^{\mathrm{object\ pose}}
=
\left[
p_{t+k}^{T},
\sin\theta_{t+k}^{T},
\cos\theta_{t+k}^{T}
\right]
$$

The current-state input should remain the complete privileged current state for this first diagnostic.

Train an otherwise identical deterministic low-level policy for each goal representation.

## B2. Horizon sweep

Run the decomposition at:

$$
k\in\{2,5,10,20\}
$$

corresponding to:

| Steps | Time |
| ---: | ---: |
| 2 | 0.10 s |
| 5 | 0.25 s |
| 10 | 0.50 s |
| 20 | 1.00 s |

Keep the low-level action horizon at:

$$
H=1
$$

for the first sweep.

## B3. Required comparisons

For each horizon, compare:

1. Flat privileged current-state policy.
2. Full structured oracle.
3. Robot-only oracle.
4. TCP-only oracle.
5. Object-only oracle.
6. Object-pose-only oracle.

## KPIs

- Closed-loop success.
- Final and maximum reward.
- Teacher action MAE.
- Physical subgoal-reaching error.
- Action sensitivity to each goal component.
- Performance relative to the flat privileged policy.
- Performance relative to the full oracle.

## Interpretation

### Outcome 1: Robot-only or TCP-only explains nearly all oracle performance

The current oracle is mainly communicating a motor waypoint or action proxy.

The interface is not yet a clean desired-effect interface.

Next step:

- separate robot and scene encoders;
- remove future robot information from the high-level target;
- test a scene/effect-only goal.

### Outcome 2: Object-only performs close to the full oracle

This strongly supports the intended formulation.

The high level can specify what should happen to the object while the low level chooses how to move the robot.

### Outcome 3: Object-only works only at larger horizons

This is plausible and desirable.

At short horizons, the object may barely move while the TCP already changes. A meaningful desired-effect interface may require:

$$
k\geq5
$$

### Outcome 4: Only the full goal works

The interface may need both:

- desired physical effect;
- suggested motor realization.

Consider a factorized interface rather than one entangled future latent.

## Gate

Before RL, require either:

- object/scene-only oracle success at least 80% of full-oracle success; or
- a clear documented decision that future robot motion is intentionally part of the interface.

---

# 5. Phase C: Test Genuine Temporal Abstraction

## Goal

Determine whether the hierarchy provides useful abstraction beyond an indirect next-action prediction.

The current final system uses:

$$
H=1,\qquad k=2
$$

and recomputes a new high-level goal every control step.

This is only 0.10 seconds into the future and is not strongly hierarchical.

## C1. Oracle horizon and update-frequency sweep

Using the most promising goal representation from Phase B, test:

$$
k\in\{5,10,20\}
$$

and high-level update periods:

$$
U\in\{1,2,5,10\}
$$

where $$U$$ is the number of primitive actions for which one high-level goal is held.

The low level must still reobserve the environment every primitive step.

For example:

- high level predicts one goal;
- low level acts toward that goal for $$U$$ steps in closed loop;
- high level replans afterward.

## C2. Action chunk sweep

After the one-step low level is stable, test:

$$
H\in\{1,2,4\}
$$

with:

$$
H<k
$$

Execute no more than:

$$
E\in\{1,2\}
$$

actions before replanning the low-level chunk.

## C3. Baselines

Compare:

1. Flat one-step policy.
2. Flat action-chunk policy.
3. Oracle hierarchy with a goal updated every step.
4. Oracle hierarchy with a held goal.
5. Learned hierarchy with the same horizon and update period.

## KPIs

- Success.
- Number of high-level decisions per episode.
- Low-level subgoal error.
- Recovery after disturbances.
- Inference latency.
- Success versus horizon.
- Success versus goal-hold period.
- Success versus action-chunk length.

## Gate

The hierarchy should be tested at a setting where:

- the high level makes materially fewer decisions than the low level;
- oracle performance remains competitive;
- the goal represents a physical future that changes meaningfully.

Preferred target:

$$
k\geq10
$$

and:

$$
U\geq2
$$

If no oracle representation works beyond $$k=2$$, Push-T may be too reactive to validate the intended hierarchy.

---

# 6. Phase D: Create Teleoperation-Like Recovery Data

## Goal

Test the hierarchy and flat baseline on causal data containing:

- imperfect commands;
- learner-like deviations;
- genuine expert recoveries;
- no simulator reset inside a recovery segment.

The final clean comparison remains valid and fair, but clean deterministic teacher data is less representative of real teleoperation.

## D1. Data-generation procedure

Run the deterministic PPO teacher.

At random times, inject one perturbation burst for a short window. After the burst ends, return control to the teacher and let it recover from the resulting state.

The complete sequence must remain causal:

$$
s_{t+1}
=
f
\left(
s_t,a_t^{\mathrm{executed}}
\right)
$$

Do not restore the nominal trajectory after the disturbance.

## D2. Perturbation types

Implement at least four realistic perturbation families.

### Correlated directional bias

For a burst of length $$B$$:

$$
a_t^{\mathrm{executed}}
=
\operatorname{clip}
\left(
a_t^{E}+b+\epsilon_t
\right)
$$

where:

- $$b$$ is fixed during the burst;
- $$\epsilon_t$$ is small temporally correlated noise.

### Action hold

Repeat the previous command for several steps:

$$
a_t^{\mathrm{executed}}=a_{t-1}^{\mathrm{executed}}
$$

### Action delay

Execute an older action:

$$
a_t^{\mathrm{executed}}=a_{t-d}^{E}
$$

for small delay $$d$$.

### Action scaling

$$
a_t^{\mathrm{executed}}
=
\alpha a_t^{E}
$$

with either overshoot or undershoot.

Optional:

- small T-block impulse;
- small TCP displacement;
- temporary action dropout.

## D3. Burst parameters

Initial sweep:

| Parameter | Values |
| --- | --- |
| Burst duration | 2, 4, 8 steps |
| Burst probability | 1-3 per episode |
| Bias magnitude | 5%, 10%, 20% of action range |
| Delay | 1-3 steps |
| Scaling | 0.7, 1.3 |

Reject perturbation levels from which the teacher almost never recovers.

## D4. Dataset variants

Create equal-transition-budget datasets:

1. `clean`
   - 100% clean successful teacher trajectories.

2. `mixed_25`
   - 75% clean transitions;
   - 25% perturbation/recovery transitions.

3. `mixed_50`
   - 50% clean transitions;
   - 50% perturbation/recovery transitions.

4. `recovery_heavy`
   - majority recovery and off-nominal transitions.

Keep a fixed clean validation set and a separate recovery validation set.

## D5. Labels

Store per timestep:

- executed action;
- unperturbed teacher action;
- perturbation type;
- perturbation start/end;
- recovery start;
- recovery completion;
- success;
- physical state;
- RGB;
- proprioception.

For policy imitation, use the action that was actually executed.

For learning the desired recovery policy after the perturbation ends, also create a separate teacher-action query view labeled by the deterministic teacher.

Do not mix the two meanings silently.

## D6. Training comparison

For each dataset variant, retrain from scratch:

1. Direct visual BC.
2. Direct visual flow.
3. AE or selected scene encoder.
4. Matched flat latent policy.
5. Oracle hierarchy.
6. Deterministic hierarchy.
7. Generative hierarchy only if multimodality is present.

Use the exact same trajectories for the flat and hierarchical methods.

## D7. Evaluation distributions

Evaluate on:

1. Clean initial states.
2. Action-noise bursts.
3. Action delay.
4. Object/TCP disturbances.
5. States sampled from failed learned-policy rollouts.

## KPIs

- Clean success.
- Disturbed success.
- Recovery success.
- Time to recover.
- Maximum reward drop after perturbation.
- Fraction returning to the nominal success region.
- Flat-versus-hierarchy gap.
- Oracle-versus-learned hierarchy gap.

## Gate

Before RL, establish whether recovery-rich causal data:

- improves both methods similarly;
- preferentially improves the hierarchy;
- or hurts nominal performance.

This experiment is the closest simulation analogue to eventual real teleoperation data.

---

# 7. Phase E: Test the Latent Geometry and Goal-Comparison Assumption

## Goal

Determine whether the current AE latent is suitable as a **future goal space**, not merely as a current-state compression.

The AE performed best as a control bottleneck, but reconstruction alone does not guarantee:

- local smoothness;
- meaningful subtraction;
- Euclidean reachability;
- task-relevant interpolation;
- predictable future geometry.

A standard VAE also does not guarantee those properties.

## E1. Absolute versus delta conditioning

Compare identical low-level policies using:

### Absolute pair

$$
a_t
=
\pi
\left(
z_t,z_{t+k},a_{t-1}
\right)
$$

### Delta goal

$$
a_t
=
\pi
\left(
z_t,z_{t+k}-z_t,a_{t-1}
\right)
$$

### Learned relation network

$$
r_t
=
R_\psi
\left(
z_t,z_{t+k}
\right)
$$

$$
a_t
=
\pi
\left(
z_t,r_t,a_{t-1}
\right)
$$

Use the same parameter budget where practical.

The relation-network version does not assume that subtraction is meaningful.

## E2. Representation variants

At the full data budget, compare:

1. Balanced deterministic AE.
2. Weakly regularized VAE.
3. Denoising AE.
4. Contractive AE if implementation cost is reasonable.
5. AE plus inverse-dynamics auxiliary loss.
6. AE plus local temporal-consistency loss.
7. Separate scene and robot encoders.

### Suggested VAE sweep

Use:

$$
\beta\in
\left\{
10^{-6},
10^{-5},
10^{-4}
\right\}
$$

Use the posterior mean for all policies.

Record:

- reconstruction;
- KL;
- number of active latent dimensions;
- oracle control;
- learned high-level prediction;
- closed-loop success.

Do not select a VAE based only on static probes.

## E3. Local smoothness tests

For real state pairs, measure:

### Physical versus latent distance

- latent distance versus object pose distance;
- latent distance versus TCP pose distance;
- latent distance versus control steps between states;
- latent distance versus action effort.

### Local perturbation consistency

Apply small physical perturbations and measure whether latent changes scale smoothly.

### Interpolation validity

Interpolate:

$$
z(\alpha)
=
(1-\alpha)z_1+\alpha z_2
$$

Decode and inspect whether intermediate states are physically plausible.

### Nearest-neighbor consistency

Retrieve nearest dataset states in latent space and compare:

- object state;
- contact;
- robot state;
- teacher action.

## E4. Control-sensitivity analysis

Measure how sensitive the low-level action is to goal errors.

For policy:

$$
a_t
=
\pi_{\mathrm{low}}
\left(
z_t,g_t
\right)
$$

estimate the Jacobian:

$$
J_g
=
\frac{\partial a_t}{\partial g_t}
$$

Measure:

- largest singular values;
- sensitive latent directions;
- high-level prediction error projected onto those directions.

Define:

$$
e_t^{\mathrm{sensitive}}
=
\left\|
J_g
\left(
\hat{g}_t-g_t
\right)
\right\|_2
$$

This may predict rollout failure better than raw latent L2.

## Gate

Choose the goal representation using:

- closed-loop oracle performance;
- learned-goal closed-loop performance;
- robustness to small goal errors;
- not static probe quality alone.

---

# 8. Phase F: Separate Current-State Representation from Goal Representation

## Goal

Avoid requiring one latent to simultaneously be:

- a complete current observation representation;
- a future prediction target;
- a goal difference space;
- a low-level control interface.

Use a rich current-state representation and a smaller goal/effect representation.

## F1. Factorized current state

Construct:

$$
h_t
=
\left[
h_t^{\mathrm{scene}},
h_t^{\mathrm{robot}}
\right]
$$

where:

- $$h_t^{\mathrm{scene}}$$ comes from image/scene encoding;
- $$h_t^{\mathrm{robot}}$$ comes from proprioception.

The low level receives the full current representation.

## F2. Scene-only future goal

Use:

$$
g_t
=
E_{\mathrm{scene}}
\left(
I_{t+k}
\right)
$$

The future goal must not contain future robot proprioception.

Train:

$$
a_t
=
\pi_{\mathrm{low}}
\left(
h_t,g_t,a_{t-1}
\right)
$$

## F3. Compact effect code

Learn:

$$
e_t
=
E_g
\left(
h_t,h_{t+k}
\right)
$$

with:

$$
e_t\in\mathbb{R}^{d_e}
$$

Test:

$$
d_e\in\{16,32,64\}
$$

The low level receives:

$$
a_t
=
\pi_{\mathrm{low}}
\left(
h_t,e_t,a_{t-1}
\right)
$$

The high level predicts:

$$
\hat{e}_t
=
F_{\mathrm{high}}
\left(
h_t,a_{t-1}
\right)
$$

## F4. Effect-code training objectives

Compare:

1. Action prediction.
2. Future scene reconstruction.
3. Contact prediction.
4. Object-state probe preservation.
5. Inverse dynamics.
6. Variance/covariance regularization.

A possible combined objective is:

$$
\mathcal{L}_{\mathrm{effect}}
=
\lambda_a\mathcal{L}_{\mathrm{action}}
+
\lambda_s\mathcal{L}_{\mathrm{scene}}
+
\lambda_c\mathcal{L}_{\mathrm{contact}}
+
\lambda_v\mathcal{L}_{\mathrm{variance}}
+
\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{covariance}}
$$

Do not add every term initially. Start with:

- action prediction;
- future scene reconstruction;
- variance regularization.

## F5. Structured-state bridge

Before relying on a learned effect encoder, test a structured goal:

$$
g_t^{\mathrm{structured}}
=
\left[
p_{t+k}^{T},
\sin\theta_{t+k}^{T},
\cos\theta_{t+k}^{T},
v_{t+k}^{T},
\omega_{t+k}^{T}
\right]
$$

Then train a visual predictor of this structured target.

This provides a strong interpretable bridge between the privileged object-only oracle and a fully learned effect code.

## KPIs

- Oracle-interface success.
- Learned-predictor success.
- Oracle-to-learned gap.
- High-level prediction error.
- Goal sensitivity.
- Recovery performance.
- Dimension versus performance.
- Amount of future robot information retained.

## Gate

A promising goal representation should satisfy:

$$
\mathrm{Success}_{\mathrm{learned\ goal}}
\geq
0.8
\,
\mathrm{Success}_{\mathrm{matched\ oracle}}
$$

and should not require future robot state unless that is an explicit design choice.

---

# 9. Phase G: Diagnose the High-Level Predictor Directly

## Goal

Determine why the learned high-level target fails even when average latent and action errors appear reasonable.

## G1. Structured prediction first

Predict the future structured object state before predicting a learned latent.

Compare:

1. Oracle structured object goal.
2. Deterministic predicted structured object goal.
3. Generative predicted structured object goal.

If the deterministic structured predictor nearly matches its oracle, the main issue is the learned latent semantics.

## G2. Error decomposition

For every predicted goal, measure errors in:

- object position;
- object yaw;
- object velocity;
- TCP position if included;
- contact;
- task progress;
- low-level action after conditioning on the prediction.

## G3. On-policy versus offline prediction

Evaluate high-level prediction on:

1. Held-out teacher states.
2. Flat-policy visited states.
3. Hierarchy visited states.
4. Perturbed recovery states.

The high-level model may have good offline accuracy but fail badly on its own state distribution.

## G4. Manifold diagnostics

For each predicted goal:

- nearest real future-goal distance;
- decoder reconstruction plausibility;
- probe plausibility;
- reachability estimate;
- low-level action sensitivity.

## G5. Predict only sensitive goal components

Use the low-level goal Jacobian to weight prediction loss:

$$
\mathcal{L}_{\mathrm{sensitive}}
=
\left\|
J_g
\left(
\hat{g}_t-g_t
\right)
\right\|_2^2
$$

A simpler approximation is per-dimension weighting from empirical action sensitivity.

## G6. Candidate generation and selection

If the high level is genuinely multimodal:

1. Sample several goals.
2. Reject off-manifold candidates.
3. Score reachability.
4. Select the best candidate.

Do not evaluate random stochastic samples without candidate selection if sample diversity mainly reflects model error.

## Gate

Do not proceed to RL until the learned high level reaches at least 80% of the success of the corresponding oracle representation.

---

# 10. Phase H: Test Whether Multimodality Actually Exists

## Goal

Determine whether flow matching is justified for the high level.

The current deterministic PPO dataset usually contains one continuation per state. That is insufficient to identify multiple valid future modes.

## H1. Collect multiple continuations

From the same or closely matched initial states, create several causal continuations using:

- small stochastic expert perturbations;
- different contact approaches;
- different recovery strategies;
- different sides of the object;
- teleoperation-like error bursts followed by recovery.

## H2. Quantify multimodality

For a fixed current-state neighborhood, measure the distribution of:

- future object position;
- future object yaw;
- future TCP position;
- future effect code;
- teacher action.

Use clustering only after verifying that differences are physically meaningful.

## H3. Model comparison

Compare:

1. Deterministic regression.
2. Gaussian predictor.
3. Mixture-density network.
4. Conditional flow matching.

Evaluate:

- best-of-one performance;
- best-of-N performance;
- sample validity;
- sample diversity;
- closed-loop success;
- candidate-selection requirement.

## Decision

If the data does not contain clear multimodality, use the deterministic high level for the proof of concept and reserve flow matching for the richer excavator dataset.

---

# 11. Phase I: Camera and Observation Follow-Up

## Goal

Test additional visual information only after the interface diagnostics above.

## Experiment

Compare:

1. Base camera only.
2. Base plus wrist camera.

Retrain:

- image features;
- representation;
- flat policy;
- hierarchy.

Use synchronized camera frames.

## KPIs

- Object/contact probes.
- Flat success.
- Oracle hierarchy success.
- Learned hierarchy success.
- Recovery performance.
- Compute and latency.

## Gate

Adopt the wrist camera only if it improves closed-loop control, not merely static probes.

---

# 12. Priority Order

Execute the work in this order.

## Priority 1: Statistical replication

- Three training seeds at the full data budget.
- Larger oracle evaluation.
- Confirm the real performance gap.

## Priority 2: Oracle-information decomposition

- Full, robot-only, TCP-only, object-only, and object-pose-only structured goals.
- Horizons 2, 5, 10, and 20.
- Determine whether the oracle is an effect target or a motor-plan leak.

## Priority 3: Meaningful temporal abstraction

- Larger horizons.
- Hold goals for multiple low-level steps.
- Test whether Push-T supports the intended hierarchy at all.

## Priority 4: Teleoperation-like causal recovery data

- Correlated perturbation bursts.
- Teacher recovery without reset.
- Equal-data flat versus hierarchy comparison.
- Retrain all encoders for each data condition.

## Priority 5: Goal-space semantics

- Absolute versus delta versus learned relation.
- Separate scene and robot encoders.
- Scene-only goal.
- Compact effect code.
- Structured-state bridge.

## Priority 6: Representation regularization

- VAE beta sweep.
- Denoising AE.
- Inverse-dynamics and local consistency objectives.
- Select using closed-loop control, not probes alone.

## Priority 7: Multimodality test

- Multiple continuations from the same or nearby states.
- Determine whether a generative high level is necessary.

## Priority 8: Wrist camera

- Only if perception remains a demonstrated bottleneck.

---

# 13. RL Readiness Gates

Do not begin the main RL experiments until the following have been evaluated.

## Gate 1: Reproducible performance gap

The baseline and hierarchy results are stable across at least three training seeds.

## Gate 2: Oracle semantics understood

It is known whether the oracle benefit comes from:

- future robot motion;
- future object/scene effect;
- or both.

## Gate 3: Meaningful temporal abstraction

The selected hierarchy operates beyond the trivial:

$$
H=1,\qquad k=2,\qquad U=1
$$

setting, unless there is a documented reason that Push-T cannot support a longer hierarchy.

## Gate 4: Recovery-rich comparison completed

Flat and hierarchical methods have been trained on the same causal teleoperation-like recovery dataset.

## Gate 5: Goal representation validated

The selected goal representation has:

- strong oracle performance;
- robustness to small goal errors;
- meaningful physical semantics;
- no accidental reliance on future robot motion unless intentional.

## Gate 6: Learned high level closes most of the oracle gap

Require:

$$
\mathrm{Success}_{\mathrm{learned\ hierarchy}}
\geq
0.8
\,
\mathrm{Success}_{\mathrm{matched\ oracle}}
$$

If this is not reached, document the exact bottleneck before RL.

## Gate 7: RL objective chosen

Specify whether RL will optimize:

- low-level latent goal reaching;
- high-level terminal task success;
- high-level learned progress;
- candidate reranking;
- or residual correction.

Do not jointly fine-tune high and low levels in the first RL experiment.

---

# 14. Recommended First RL Configuration After the Gates

When the gates are met:

1. Freeze the image encoder.
2. Freeze the current-state representation.
3. Freeze the low-level policy for the first high-level RL experiment.
4. Keep the imitation high-level generator as a proposal distribution.
5. Train a high-level critic or candidate selector.
6. Keep generated goals near the demonstrated goal distribution.
7. Use the simulator task reward only after the interface is validated.
8. Compare against the frozen imitation hierarchy and flat baseline.

For low-level RL:

1. Freeze the high-level goal source.
2. Reset from causal demonstration and perturbation states.
3. Optimize local subgoal reaching.
4. Start with residual RL or a small adapter rather than modifying the full policy.
5. Evaluate whether recovery improves without degrading nominal behavior.

---

# 15. Required Final Deliverables

The researcher should produce:

1. `next_experiments_log.md`
   - one entry per experiment;
   - commands, seeds, metrics, diagnosis, and decision.

2. `oracle_goal_decomposition.csv`
   - one row per goal type, horizon, seed, and evaluation condition.

3. `recovery_dataset_spec.md`
   - exact perturbation process;
   - data ratios;
   - causal guarantees;
   - dataset statistics.

4. `representation_geometry_report.md`
   - AE/VAE/denoising/relation-network comparisons;
   - smoothness, interpolation, and sensitivity diagnostics.

5. `pre_rl_summary.md`
   - final selected interface;
   - selected horizon;
   - selected dataset;
   - selected representation;
   - remaining oracle gap;
   - explicit RL recommendation.

6. Plots:
   - success by oracle-goal information;
   - success by horizon and update period;
   - clean versus recovery-rich data;
   - oracle versus learned goal performance;
   - latent error versus action-sensitive error;
   - success across training seeds.

---

# 16. Main Decision Tree

## If object-only oracle works well

Proceed with:

- scene/effect-only future goal;
- separate current robot representation;
- compact effect code;
- high-level prediction of the effect code.

## If only robot/TCP oracle works

Decide explicitly whether the interface should be:

- a future motor waypoint; or
- redesigned to express physical effect.

Do not describe it as an effect-only hierarchy without further evidence.

## If structured object goal works but learned scene goal fails

The representation is the bottleneck.

Prioritize:

- structured target prediction;
- separate scene encoder;
- compact effect code;
- sensitivity-aware representation learning.

## If even structured privileged goals fail

The low-level goal-conditioned formulation or chosen horizon is the bottleneck.

Do not proceed to high-level RL.

## If recovery-rich data closes the gap

Use causal intervention/recovery data as a central part of the final thesis method.

## If recovery-rich data helps flat and hierarchy equally

The hierarchy does not yet provide a recovery-specific advantage. Focus on temporal abstraction or the high-level target.

## If no meaningful hierarchy works in Push-T

Treat Push-T as a pipeline/debugging benchmark only and move the central hierarchy claim to a longer-horizon task where subgoals are genuinely useful.
