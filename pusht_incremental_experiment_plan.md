# Push-T Future-Latent Hierarchy: Incremental Experiment Plan

## 1. Purpose

This document defines the experiment sequence for the Push-T proof of concept.

The plan is intentionally incremental:

1. Verify the teacher and data pipeline.
2. Make privileged-state imitation work.
3. Make privileged-state flow matching work.
4. Make visual deterministic imitation work.
5. Make visual flow matching work.
6. Validate the learned latent representation.
7. Validate the future-latent interface with oracle subgoals.
8. Train and validate the high-level future-state predictor.
9. Combine both levels.
10. Run final sample-efficiency and hierarchy comparisons.

The main rule is:

> Do not move to a harder phase until the previous phase reaches its success gate or the reason for failure is understood.

The current privileged PPO teacher reaches approximately 86.3% success. Privileged-state BC reaches approximately 46%, while visual and hierarchical policies remain much lower. Therefore, the immediate goal is not to improve the hierarchy. The immediate goal is to build a trustworthy imitation-learning stack that can closely reproduce the teacher before adding perception and hierarchy.

---

## 2. Global experiment rules

### 2.1 Use one canonical evaluation protocol

All policies should be evaluated with:

- identical environment version;
- identical controller;
- identical action clipping;
- identical observation preprocessing;
- identical reset seed list;
- identical maximum episode length;
- deterministic evaluation unless stochasticity is explicitly being tested;
- at least 500 evaluation episodes for final results;
- at least 3 training seeds for final comparisons.

For fast debugging, 50-100 episodes are acceptable, but they should not be used for final claims.

### 2.2 Track the same KPIs in every phase

Primary KPI:

- task success rate.

Secondary KPIs:

- final normalized dense reward;
- maximum normalized dense reward;
- mean episode length;
- action MAE against teacher labels;
- action saturation rate;
- rollout divergence from teacher;
- inference latency;
- training loss;
- evaluation variance across seeds.

### 2.3 Store complete experiment metadata

For every run, store:

- Git commit;
- configuration file;
- environment version;
- simulator backend;
- controller;
- observation fields;
- normalization statistics;
- dataset identifier;
- number of trajectories;
- number of transitions;
- random seed;
- checkpoint;
- success/reward metrics;
- evaluation initial-state seeds.

### 2.4 Use gates, not intuition

Each phase below includes a minimum gate.

Do not proceed merely because the loss decreases or rollout videos “look somewhat better.”

### 2.5 Keep state-query data and causal trajectory data separate

The project needs two different dataset types. They must not be confused.

#### A. State-query dataset

A state-query sample is:

$$\left(s,\pi_{\mathrm{PPO}}^{\mathrm{det}}(s)\right)$$

It answers:

> Given this state, what action would the deterministic teacher choose now?

The state can come from:

- a deterministic teacher rollout;
- a stochastic teacher rollout;
- a student rollout;
- a failed rollout;
- a perturbed simulator state.

The state-query dataset does **not** claim that the relabeled action generated the stored next state.

It is valid for:

- one-step deterministic BC;
- one-step flow-policy distillation;
- DAgger;
- action-prediction diagnostics.

It is not valid for:

- world-model training;
- future-state prediction;
- inverse dynamics using the stored next state;
- action-chunk supervision;
- high-level future-latent training;
- trajectory replay.

If an old trajectory contains:

$$\left(s_t,a_t^{\mathrm{old}},s_{t+1}^{\mathrm{old}}\right)$$

and the state is relabeled with:

$$a_t^{\mathrm{label}}=\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)$$

the relabeled tuple:

$$\left(s_t,a_t^{\mathrm{label}},s_{t+1}^{\mathrm{old}}\right)$$

is generally not dynamically consistent because:

$$s_{t+1}^{\mathrm{old}}=f\left(s_t,a_t^{\mathrm{old}}\right)$$

rather than:

$$s_{t+1}^{\mathrm{old}}=f\left(s_t,a_t^{\mathrm{label}}\right)$$

#### B. Causal trajectory dataset

A causal trajectory is collected by actually executing the policy in the simulator:

$$a_t=\pi_{\mathrm{teacher}}^{\mathrm{det}}(s_t)$$

$$s_{t+1}=f\left(s_t,a_t\right)$$

The resulting trajectory is:

$$\tau=\left(s_0,a_0,s_1,a_1,\ldots,s_T\right)$$

Every transition is dynamically consistent.

The causal trajectory dataset is required for:

- action chunks;
- temporal observation windows;
- flow matching over action sequences;
- world-model learning;
- inverse dynamics;
- future-state labels;
- oracle future-latent training;
- high-level future-state prediction;
- trajectory replay.

#### C. Naming convention

Use explicit dataset names in code and reports:

- `query_dataset`: independent state and deterministic-teacher-action pairs;
- `causal_dataset`: simulator rollouts with consistent state-action-next-state transitions.

Never call relabeled state queries “trajectories.”

---


# Phase 0: Teacher and pipeline sanity

## Goal

Prove that the policy-evaluation and data pipelines can reproduce the original privileged PPO teacher without changing its behavior.

## Experiments

### 0.1 Evaluate the PPO teacher through the downstream evaluation path

Run the original PPO actor using exactly the same:

- environment construction;
- observation extraction;
- normalization;
- action clipping;
- control mode;
- wrapper stack;
- evaluation seed list;

used by BC and flow policies.

### 0.2 Copy the PPO actor weights into a student wrapper

Create a student policy with the same network architecture as the PPO actor.

Copy the actor weights exactly and evaluate it through the student-policy code path.

### 0.3 Verify stored action semantics

For every stored state $$s_t$$, query the deterministic PPO actor:

$$a_t^{\mathrm{teacher}}=\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)$$

Compare it with:

$$a_{t-1}^{\mathrm{stored}}$$

$$a_t^{\mathrm{stored}}$$

$$a_{t+1}^{\mathrm{stored}}$$

Measure:

$$e_{\mathrm{action}}=\left\|a_t^{\mathrm{stored}}-a_t^{\mathrm{teacher}}\right\|$$

Also compare:

- raw teacher output;
- clipped teacher output;
- stored action;
- executed action;
- normalized and denormalized action.

### 0.4 Check observation alignment

Verify that the stored observation paired with $$a_t$$ is the observation before executing $$a_t$$.

Explicitly inspect:

$$\left(s_t,a_t,s_{t+1}\right)$$

and rule out an off-by-one error.

### 0.5 Overfit one state and one trajectory

Train deterministic BC on:

- one state-action pair;
- one complete trajectory;
- ten trajectories.

The student should nearly exactly reproduce the labels.

## KPIs

- Teacher success through original evaluation path.
- Teacher success through downstream student evaluation path.
- Copied-weight student success.
- Mean action error between stored and deterministic teacher actions.
- Best temporal alignment among $$a_{t-1}$$, $$a_t$$, and $$a_{t+1}$$.
- One-state and one-trajectory training error.
- Closed-loop replay success from training initial states.

## Potential problems

- Observation normalization mismatch.
- Action clipping mismatch.
- Raw action stored but clipped action executed.
- Stochastic teacher actions stored while deterministic evaluation is used.
- Controller mismatch.
- Action scale mismatch.
- Off-by-one state-action alignment.
- Different environment wrappers for PPO and BC.
- Incorrect terminal handling.

## Success gate

Proceed only if:

- the PPO teacher reaches approximately its known 86.3% success through the downstream evaluation path;
- a copied-weight student reaches within 1 percentage point of the teacher;
- state-action alignment is verified;
- one-state and one-trajectory overfitting gives nearly zero normalized action error.

If this phase fails, stop all learning experiments and fix the pipeline.

---

# Phase 1: Deterministic privileged-state BC

## Goal

Train a simple deterministic policy from privileged state that approaches the PPO teacher without DAgger.

## Dataset preparation

Phase 1 uses a **state-query dataset**, not a relabeled causal trajectory dataset.

Take states from the existing dataset and query the deterministic PPO actor:

$$a_t^{\mathrm{label}}=\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)$$

Store only the supervised pair:

$$\left(s_t,a_t^{\mathrm{label}}\right)$$

Do not attach the original stored next state to the new label.

The relabeled sample is valid because Phase 1 trains a one-step policy mapping:

$$s_t\longrightarrow a_t$$

It is not being used to claim:

$$s_{t+1}=f\left(s_t,a_t^{\mathrm{label}}\right)$$

Prepare two state-query subsets:

1. Queries from all available teacher-visited states.
2. Queries from states belonging only to successful trajectories.

The all-state query dataset should be the main teacher-distillation dataset because it contains broader state coverage.

Keep the original causal trajectories unchanged for later temporal experiments.

## Model

Start with the same actor architecture as the PPO teacher.

Input:

$$s_t^{\mathrm{privileged}}$$

Output:

$$\hat{a}_t$$

Loss:

$$\mathcal{L}_{\mathrm{BC}}=\left\|\hat{a}_t-a_t^{\mathrm{label}}\right\|_2^2$$

Use one-step actions first.

## Experiments

### 1.1 Same-architecture student

Train the same MLP architecture as the PPO actor.

### 1.2 Capacity sweep

Only if needed, compare:

- teacher-sized network;
- half-size network;
- double-size network.

### 1.3 Dataset-size sweep

Use nested subsets:

$$N\in\{50,100,200,500,1000,2000\}$$

Report both trajectory and transition counts.

### 1.4 Successful-only versus all teacher states

Test whether filtering to successful trajectories harms state coverage.

### 1.5 Deterministic-label versus recorded-action labels

This is an important one-step distillation diagnostic.

Compare two state-query datasets:

- $$\left(s_t,\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)\right)$$;
- $$\left(s_t,a_t^{\mathrm{stored}}\right)$$.

Do not compare them as trajectory datasets, and do not reuse the original $$s_{t+1}$$ with the deterministic relabel.

## KPIs

- Closed-loop success.
- Action MAE on held-out teacher states.
- Action correlation per dimension.
- Fraction of actions near bounds.
- Success on training initializations.
- Success on held-out initializations.
- Difference between relabeled and original-action training.
- Success versus number of transitions.

## Potential problems

- Teacher policy depends on observation normalization not reproduced in BC.
- Successful-only filtering removes difficult and recovery states.
- MSE averages inconsistent labels.
- BC loss is low but small action errors compound.
- The environment is highly sensitive to contact timing.
- The dataset terminates at success and underrepresents near-failure states.

## Success gate

Minimum gate:

$$\mathrm{Success}_{\mathrm{privileged\ BC}}\geq0.70$$

Target gate:

$$\mathrm{Success}_{\mathrm{privileged\ BC}}\geq0.75\text{--}0.80$$

The student should be within approximately 5-10 percentage points of the teacher.

Do not proceed to visual policies if privileged BC remains near 46%.

---

# Phase 2: Privileged DAgger and recovery data

## Goal

Close the remaining covariate-shift gap between privileged BC and the PPO teacher.

## DAgger state-query collection

The standard DAgger dataset is a **state-query dataset**.

At iteration $$i$$:

1. Roll out the student or a teacher-student mixture.
2. At every visited state $$s_t$$, query the deterministic PPO teacher.
3. Add the independent supervised pair:

$$\left(s_t,\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)\right)$$

to `query_dataset`.
4. Retrain or continue training the one-step student.

The student action may have generated the next state, but the teacher label is only the action the teacher would choose at the current state. No new causal transition is implied by the relabel.

Use a teacher-mixture schedule if necessary:

$$a_t=
\begin{cases}
a_t^{\mathrm{teacher}} & \text{with probability }\beta_i \\
a_t^{\mathrm{student}} & \text{otherwise}
\end{cases}$$

with:

$$\beta_i\rightarrow0$$

## Causal recovery-trajectory collection

Temporal models require new simulator rollouts rather than relabeled old trajectories.

To create recovery trajectories:

1. Reset to a state from a valid causal trajectory.
2. Apply a controlled perturbation:

$$s_t'=s_t+\epsilon$$

3. Start a new rollout from $$s_t'$$.
4. Let the deterministic teacher act:

$$a_j=\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_j)$$

5. Step the simulator:

$$s_{j+1}=f\left(s_j,a_j\right)$$

6. Store the new causal sequence:

$$\tau^{\mathrm{recovery}}=\left(s_t',a_t,s_{t+1},a_{t+1},\ldots\right)$$

Possible perturbations:

- TCP position;
- TCP velocity;
- T-block position;
- T-block orientation;
- T-block velocity;
- previous action;
- contact configuration, when feasible.

Use these new causal recovery trajectories later for:

- action chunks;
- visual temporal windows;
- world-model learning;
- future-latent labels;
- hierarchy training.

Do not create a recovery trajectory by changing actions in an old stored trajectory while keeping its old future states.

## Suggested schedule

Run 5-10 DAgger iterations.

Per iteration, collect at least:

- 50,000 learner-visited states for debugging;
- 100,000-200,000 states if GPU vectorization makes this practical.

## KPIs

- Success after each DAgger iteration.
- Action MAE on learner-visited states.
- Dataset size after each iteration.
- Fraction of learner states outside the original demonstration distribution.
- Recovery success after controlled perturbations.
- Teacher intervention rate if mixed control is used.
- Success versus DAgger iteration.

## Potential problems

- Teacher cannot recover from far out-of-distribution states.
- Dataset becomes dominated by easy repeated states.
- Retraining forgets original behavior.
- Perturbations are unrealistic.
- Too little DAgger data per iteration.
- Teacher labels are queried with different normalization from the student input.

## Success gate

Target:

$$\mathrm{Success}_{\mathrm{privileged\ DAgger}}\geq0.80$$

Stretch target:

$$\mathrm{Success}_{\mathrm{privileged\ DAgger}}\geq0.83$$

Recovery test target:

- at least 80% recovery from small perturbations that the teacher can recover from.

Once achieved, freeze this privileged deterministic policy as the main teacher-distillation baseline.

---

# Phase 3: Privileged flow-matching policy

## Goal

Verify that the flow-matching implementation can reproduce a task already solved by deterministic privileged BC.

## Important principle

Start with the easiest formulation.

Do not begin with 8-step action chunks.

## Dataset usage

For the one-step flow policy, use the same `query_dataset` as one-step deterministic BC:

$$\left(s_t,\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)\right)$$

For every multi-step action-chunk experiment, switch to `causal_dataset`.

A valid action-chunk sample must come from a trajectory that was actually executed:

$$\left(s_t,a_{t:t+H-1}\right)$$

with all intermediate transitions generated by those same stored actions.

Do not construct action chunks from independently relabeled states.

## Experiments

### 3.1 One-step privileged flow policy

Input:

$$s_t^{\mathrm{privileged}}$$

Output:

$$a_t$$

Use the same DAgger state-query dataset as the deterministic BC policy.

No next state or action sequence is needed for this one-step experiment.

### 3.2 Flow overfitting ladder

For one-step flow, verify overfitting on:

- one state-action query;
- ten state-action queries;
- one hundred state-action queries.

For action-chunk flow, verify overfitting separately on:

- one causal trajectory;
- ten causal trajectories;
- one hundred causal trajectories.

### 3.3 Sampling sanity

For a fixed state, sample multiple actions.

Measure:

- sample mean;
- sample variance;
- teacher action distance;
- percentage of samples outside action limits before clipping;
- effect of integration-step count.

### 3.4 Action-chunk progression

Only after one-step flow works, test:

$$H\in\{2,4,8\}$$

Execute:

- one action before replanning;
- two actions before replanning.

### 3.5 Compare flow and deterministic BC

Use identical:

- observations;
- data;
- model capacity where possible;
- evaluation states.

## KPIs

- One-step action MAE.
- Negative log-likelihood or flow-matching validation loss.
- Closed-loop success.
- Success versus integration steps.
- Sample variance at fixed states.
- Action saturation rate.
- Inference latency.
- Success versus chunk length.
- Success versus number of executed actions before replanning.

## Potential problems

- Incorrect flow integration direction.
- Training and inference interpolation mismatch.
- Action normalization mismatch.
- Too much stochasticity at evaluation.
- Too few integration steps.
- Long chunks amplify compounding error.
- Policy samples multimodal actions when the teacher is effectively unimodal.
- Clipping hides sampler instability.

## Success gate

One-step privileged flow should reach:

$$\mathrm{Success}_{\mathrm{privileged\ flow}}\geq\mathrm{Success}_{\mathrm{privileged\ BC}}-0.05$$

Preferred target:

- within 2-3 percentage points of deterministic privileged BC.

Do not use flow matching in the visual or hierarchical system until this gate is reached.

---

# Phase 4: Visual deterministic BC

## Goal

Train a deterministic visual policy that reaches a substantial fraction of privileged-policy performance.

## Initial observation

Use:

- base-camera RGB;
- spatial DINO features;
- robot proprioception;
- previous actions;
- a temporal observation window.

A single frame is likely insufficient because it does not expose:

- object velocity;
- angular velocity;
- contact transitions;
- recent pusher motion.

## Temporal experiments

Test:

$$L\in\{1,2,4,8\}$$

At 20 Hz, these correspond to:

- 0.05 seconds;
- 0.10 seconds;
- 0.20 seconds;
- 0.40 seconds.

Input:

$$o_{t-L+1:t}$$

and optionally:

$$a_{t-L:t-1}$$

## Temporal architecture experiments

Compare:

1. Feature concatenation.
2. GRU.
3. Small temporal transformer.
4. 1D temporal convolution.

Start with concatenation and GRU.

## Visual encoder experiments

Compare different options with probing.

Do not assume frozen DINO is optimal for synthetic low-resolution imagery.

## DAgger

Use the same iterative state-query DAgger framework as in Phase 2.

The teacher receives privileged state.

The student receives visual history plus proprioception.

At each learner-visited state, store:

$$\left(o_{t-L+1:t},a_{t-L:t-1},\pi_{\mathrm{PPO}}^{\mathrm{det}}(s_t)\right)$$

The observation history must come from the actual learner rollout that reached $$s_t$$.

For one-step visual BC, this is a valid supervised query.

For visual action chunks, future-state labels, or world-model training, collect separate causal teacher or recovery rollouts. Do not append deterministic teacher labels to a learner trajectory and then treat the learner trajectory's future states as if the teacher actions had generated them.

## Probes

Train probes from the visual history representation for:

- T-block position;
- T-block yaw;
- T-block linear velocity;
- T-block angular velocity;
- TCP position;
- TCP velocity;
- contact state;
- normalized dense reward;
- success probability.

## KPIs

- Closed-loop success.
- Final and maximum reward.
- Action MAE on held-out and learner-visited states.
- Position and yaw probe error.
- Velocity probe error.
- Contact classification AUROC.
- Improvement per DAgger iteration.
- Generalization to held-out initial states.
- Sensitivity to history length.

## Potential problems

- Current image contains pose but not velocity/contact information.
- Frozen DINO discards task-specific details.
- Proprioception and DINO features use incompatible scales.
- DAgger dataset is too small.
- Student sees a different temporal context during collection and training.
- Contact-point precision is below what Push-T requires.
- The policy overfits to camera appearance rather than geometry.

## Wrist-camera decision

Do not add a wrist camera immediately.

Add it only if:

- temporal base-camera BC plateaus;
- object/contact probes show missing local information;
- privileged BC and flow baselines are already strong.

Then compare:

1. Base camera only.
2. Base plus wrist.

## Success gate

Minimum useful gate:

$$\mathrm{Success}_{\mathrm{visual\ BC}}\geq0.50$$

Target gate:

$$\mathrm{Success}_{\mathrm{visual\ BC}}\geq0.60\text{--}0.70$$

Also require:

- clear improvement over the current 6%;
- no major train/evaluation preprocessing mismatch;
- velocity/contact probes meaningfully above baseline.

---

# Phase 5: Visual flat flow policy

## Goal

Show that the visual flow policy works at least as well as the visual deterministic BC baseline.

## Experiments

### 5.1 One-step visual flow

Use the exact same input representation as the best visual BC model.

### 5.2 Small action chunks

Test:

$$H\in\{2,4\}$$

Only test $$H=8$$ after shorter chunks work.

### 5.3 Replanning frequency

Compare executing:

- one action;
- two actions;
- full short chunk.

### 5.4 Deterministic versus stochastic sampling

Evaluate:

- fixed noise seed;
- multiple sampled actions;
- averaged action;
- best-of-N using a simple critic only as an optional diagnostic.

## KPIs

- Success.
- Action MAE.
- Final/max reward.
- Sample variance.
- Inference latency.
- Success versus chunk length.
- Success versus replanning interval.
- Gap to deterministic visual BC.

## Potential problems

- Flow model does not benefit from multimodality in this task.
- Sampler variance hurts contact precision.
- Action chunk is too long.
- Flow loss decreases while closed-loop behavior remains poor.
- Model capacity is spent modeling irrelevant action modes.

## Success gate

Require:

$$\mathrm{Success}_{\mathrm{visual\ flow}}\geq\mathrm{Success}_{\mathrm{visual\ BC}}-0.05$$

Preferred target:

- match or slightly outperform visual BC;
- preserve or improve robustness under perturbations.

Do not proceed to the hierarchy with a weak flat flow model.

---

# Phase 6: Latent representation validation

## Goal

Produce a latent state that preserves the information required for future-state prediction, low-level control, and high-level generation.

All world-model, inverse-dynamics, and multi-step representation training in this phase must use `causal_dataset`.

Every sample must preserve:

$$s_{t+1}=f\left(s_t,a_t\right)$$

Relabeled state-query data can be used for action probes, but not as synthetic transitions.

## Representation candidates

Compare:

1. Raw spatial DINO plus proprioception.
2. 128-dimensional learned latent.
3. 256-dimensional learned latent.
4. 512-dimensional learned latent.
5. Learned latent with reconstruction.
6. Learned latent without reconstruction.
7. Reconstruction-only autoencoder with no world-model prediction loss.

Keep the rest of the training recipe fixed when sweeping dimension.

## World-model objective

Base:

$$\mathcal{L}_{E}=
\mathcal{L}_{\mathrm{prediction}}
+
\lambda_{\mathrm{SIG}}\mathcal{L}_{\mathrm{SIGReg}}$$

Reconstruction variant:

$$\mathcal{L}_{E}=
\mathcal{L}_{\mathrm{prediction}}
+
\lambda_{\mathrm{SIG}}\mathcal{L}_{\mathrm{SIGReg}}
+
\lambda_{\mathrm{recon}}\mathcal{L}_{\mathrm{reconstruction}}$$

Reconstruction-only autoencoder ablation:

$$\mathcal{L}_{E}=
\lambda_{\mathrm{recon}}\mathcal{L}_{\mathrm{reconstruction}}$$

For this ablation, set the world-model prediction-loss weight to zero. It
tests whether the action-conditioned temporal objective adds useful
control-state structure beyond simply compressing and reconstructing the
current observation.


## Required probes

Train held-out probes for:

- T position;
- T yaw;
- T linear velocity;
- T angular velocity;
- TCP pose;
- TCP velocity;
- contact state;
- reward/progress;
- inverse dynamics;
- forward dynamics.

Inverse-dynamics probe:

$$\hat{a}_{t:t+H-1}=I\left(z_t,z_{t+k}\right)$$

## Latent geometry diagnostics

Measure:

$$D_{\mathrm{future}}=\left\|z_{t+k}-z_t\right\|_2$$

Nearest-neighbor inspection:

- retrieve nearest real latent states;
- verify similar object pose, velocity, and contact state.

Distance correlation:

- latent distance versus physical pose distance;
- latent distance versus task progress;
- latent distance versus number of control steps between states.

## Potential problems

- Latent retains pose but loses velocity.
- Latent retains observation reconstruction details but not control information.
- Latent dimension is too small.
- Large latent is hard for the high-level generator.
- Representation is anisotropic and noise calibration becomes meaningless.
- Reconstruction dominates and preserves irrelevant features.
- Inverse dynamics makes the latent too action-specific.

## Success gate

Minimum probe targets:

- T position MAE no worse than approximately 1 cm;
- T yaw MAE no worse than approximately 10 degrees;
- meaningful velocity prediction above a mean baseline;
- contact AUROC above 0.80;
- inverse-dynamics probe clearly better than predicting the mean action.

Preferred target:

- latent probes within approximately 2x the error of raw spatial DINO probes.

Control target:

- a flat policy using the latent should achieve at least 80-90% of the success of the corresponding direct-observation policy.

---

# Phase 7: Oracle future-latent low-level policy

## Goal

Validate the future-latent interface before training the high-level generator.

Use only causal trajectories collected by actually executing the policy.

The low level receives the true encoded future state from the same causal trajectory:

$$g_t=z_{t+k}^{\mathrm{true}}$$

The corresponding action target:

$$a_{t:t+H-1}$$

must come from that same trajectory.

## Important horizon constraint

Require:

$$H<k$$

The current setup with $$H=8$$ and $$k\in\{1,2,5\}$$ violates the intended hierarchy.

Start with:

$$H\in\{1,2,4\}$$

and:

$$k\in\{10,20\}$$

At 20 Hz:

- action chunk: 0.05-0.20 seconds;
- subgoal horizon: 0.50-1.00 seconds.

## Experiments

### 7.1 Privileged oracle hierarchy

Use privileged future state as the goal.

### 7.2 Learned-latent oracle hierarchy

Use true future latent:

$$z_{t+k}^{\mathrm{true}}$$

### 7.3 Goal-conditioning intervention tests

Correct goal:

$$\pi_{\mathrm{low}}\left(z_t,z_{t+k}\right)$$

Shuffled goal:

$$\pi_{\mathrm{low}}\left(z_t,z_{j+k}\right)$$

Zero or mean goal:

$$\pi_{\mathrm{low}}\left(z_t,0\right)$$

Goal-sensitivity score:

$$S_g=
\mathbb{E}_{g_1,g_2}
\left[
\left\|
\pi_{\mathrm{low}}(z_t,g_1)
-
\pi_{\mathrm{low}}(z_t,g_2)
\right\|_2
\right]$$

### 7.4 Horizon sweep

Test subgoal horizons:

$$k\in\{1, 2, 5,10,20\}$$

## KPIs

- Oracle hierarchy success.
- Subgoal-reaching error.
- Action MAE with true future goals.
- Correct-goal versus shuffled-goal performance gap.
- Goal-sensitivity score.
- Success versus subgoal horizon.
- Success versus action-chunk length.
- Success versus high-level update frequency.

## Potential problems

- Low level ignores the future goal.
- Future latent includes irrelevant details.
- Goal horizon is too short to add hierarchy.
- Goal horizon is too long for low-level reachability.
- Action chunk extends beyond the subgoal horizon.
- Current state already predicts the action so strongly that the goal is unused.

## Success gate

Require:

- oracle future-latent policy reaches at least the performance of the flat visual flow policy;
- shuffled or zero goals significantly reduce performance;
- correct goals improve action error and closed-loop success;
- the low level demonstrably uses the goal.

Preferred target:

$$\mathrm{Success}_{\mathrm{oracle\ hierarchy}}\geq\mathrm{Success}_{\mathrm{flat\ visual\ flow}}$$

If oracle hierarchy is worse, do not train the high level yet.

---

# Phase 8: Deterministic high-level future predictor

## Goal

Verify that future latent prediction is possible before introducing generative flow matching.

Train only on future pairs extracted from causal trajectories:

$$\left(z_{t-L+1:t},z_{t+k}\right)$$

Do not use future states from an old trajectory after replacing its actions with new teacher labels.

## Model

Input:

- temporal current latent history;
- optional previous actions;
- no future actions.

Output:

$$\hat{z}_{t+k}=f_{\mathrm{high}}\left(z_{t-L+1:t}\right)$$

Loss:

$$\mathcal{L}_{\mathrm{high}}=
\left\|
\hat{z}_{t+k}-z_{t+k}
\right\|_2^2$$

## Experiments

### 8.1 Predict structured state first

Predict privileged future T pose and TCP state.

This establishes whether future prediction is fundamentally feasible.

### 8.2 Predict learned latent

Predict $$z_{t+k}$$.

### 8.3 Temporal-history sweep

Use:

$$L\in\{1,2,4,8\}$$

### 8.4 Horizon sweep

Use:

$$k\in\{1, 2, 5,10,20\}$$

### 8.5 Decode and probe predicted latents

Apply the same probes used for real latents to:

$$\hat{z}_{t+k}$$

Measure predicted:

- T pose;
- velocity;
- contact;
- task progress.

### 8.6 Nearest-neighbor manifold test

For every predicted latent, find its nearest real latent.

Measure:

$$D_{\mathrm{NN}}=
\min_{z_i\in\mathcal{D}}
\left\|
\hat{z}_{t+k}-z_i
\right\|_2$$

## KPIs

- Latent prediction MSE.
- Structured future-state error.
- Probe error on predicted latents.
- Nearest-real-latent distance.
- Low-level action MAE conditioned on predicted latents.
- Closed-loop hierarchy success using deterministic predictions.
- Gap between oracle and predicted-subgoal hierarchy.

## Potential problems

- Future is multimodal and MSE averages modes.
- Predicted latent is off-manifold.
- Temporal input lacks velocity/contact information.
- High-level horizon is too long.
- Prediction error is small in latent distance but large in task-relevant variables.

## Success gate

Require:

- predicted latents remain close to the real-latent manifold;
- predicted pose/contact probes remain meaningful;
- low-level action error with predicted goals is no more than approximately 2x oracle-goal action error;
- deterministic hierarchy achieves a substantial fraction of oracle hierarchy performance.

Suggested minimum:

$$\mathrm{Success}_{\mathrm{det\ hierarchy}}
\geq0.70\cdot
\mathrm{Success}_{\mathrm{oracle\ hierarchy}}$$

---

# Phase 9: Generative high-level flow model

## Goal

Replace deterministic future regression with a model that can represent multiple valid future states.

## Model

Train:

$$\hat{z}_{t+k}\sim
p_{\mathrm{high}}\left(z_{t+k}\mid z_{t-L+1:t}\right)$$

Use conditional flow matching.

## Experiments

### 9.1 Overfit small datasets

Overfit:

- one trajectory;
- ten trajectories;
- one hundred trajectories.

### 9.2 Compare deterministic and generative predictions

Evaluate:

- latent error;
- manifold distance;
- diversity;
- downstream success.

### 9.3 Sample quality diagnostics

For each current state, generate multiple future latents.

Measure:

- sample diversity;
- nearest-real-latent distance;
- decoded physical plausibility;
- downstream low-level action consistency.

### 9.4 Best-of-N diagnostic

For analysis only, sample $$N$$ future goals and select the one nearest to a real trajectory continuation or with the highest learned reachability score.

Do not make best-of-N selection part of the primary method unless justified.

## KPIs

- Downstream hierarchy success.
- Sample diversity.
- Off-manifold rate.
- Oracle-versus-generated action MAE.
- Oracle-versus-generated success gap.
- Deterministic-versus-generative success.
- High-level inference latency.

## Potential problems

- Flow sampler produces off-manifold latent combinations.
- Diversity comes from irrelevant dimensions.
- Model samples physically plausible but task-irrelevant futures.
- Low-level policy is brittle to generator errors.
- High-dimensional latent is difficult to model.

## Success gate

Require:

- generative hierarchy is at least as good as deterministic hierarchy;
- sampled latents pass structured probes;
- sampled-goal low-level action error remains close to oracle-goal error;
- generated future diversity is reflected in meaningful physical alternatives.

---

# Phase 10: Train the low level on actual high-level errors

## Goal

Make the low-level policy robust to the real distribution of generated future goals without teaching it to ignore the goal.

## Important change

Do not use arbitrary isotropic latent noise such as:

$$\epsilon\sim\mathcal{N}\left(0,0.5^2I\right)$$

until its scale is calibrated.

In 512 dimensions, its expected norm is approximately:

$$0.5\sqrt{512}\approx11.3$$

which may be much larger than the true future-state displacement.

## Diagnostics

Measure:

$$D_{\mathrm{future}}=
\left\|z_{t+k}-z_t\right\|_2$$

$$D_{\mathrm{high\ error}}=
\left\|\hat{z}_{t+k}-z_{t+k}\right\|_2$$

$$D_{\mathrm{noise}}=
\left\|\epsilon\right\|_2$$

Plot all three distributions.

## Robustness methods

Compare:

1. No subgoal corruption.
2. Empirical high-level residual noise.
3. Direct training on generated high-level latents.
4. Small interpolation between real and generated latents.
5. Covariance-matched residual sampling.

Empirical residual model:

$$\epsilon\sim
\mathcal{N}
\left(
0,
\alpha^2\Sigma_{\mathrm{high\ residual}}
\right)$$

## Goal-use tests

Repeat:

- shuffled goal;
- zero goal;
- goal sensitivity;
- action change under different valid goals.

## KPIs

- Low-level action MAE with oracle goals.
- Low-level action MAE with generated goals.
- Closed-loop hierarchy success.
- Correct-goal versus shuffled-goal gap.
- Goal-sensitivity score.
- Noise norm relative to true future displacement.
- Oracle-versus-generated success gap.

## Potential problems

- Noise is so large that the low level learns to ignore the goal.
- Generated goals contain structured bias not captured by Gaussian noise.
- Low-level robustness reduces goal sensitivity.
- High-level errors are multimodal.

## Success gate

Require:

- generated-goal action MAE no more than 1.5x oracle-goal action MAE;
- correct-goal performance clearly exceeds shuffled/zero-goal performance;
- generated-goal hierarchy success reaches at least 80% of oracle hierarchy success.

---

# Phase 11: Complete hierarchy

## Goal

Evaluate the complete high-level and low-level system against strong flat baselines.

## Main methods

1. Privileged deterministic BC.
2. Privileged flow policy.
3. Visual deterministic BC.
4. Visual flat flow policy.
5. Oracle future-latent hierarchy.
6. Deterministic predicted-latent hierarchy.
7. Generative predicted-latent hierarchy.

## Fair-comparison requirements

All visual methods must use:

- the same visual encoder;
- the same temporal history;
- the same proprioception;
- the same training trajectories;
- the same evaluation seeds;
- comparable parameter counts where possible.

## KPIs

- Success.
- Final/max reward.
- Sample efficiency.
- Inference latency.
- Robustness to perturbations.
- Oracle-versus-generated hierarchy gap.
- High-level and low-level individual failure rates.
- Goal sensitivity.
- Success versus horizon.
- Success versus dataset size.

## Failure categorization

For every failed episode, classify:

- perception error;
- wrong high-level subgoal;
- unreachable subgoal;
- low-level failure to reach a valid subgoal;
- contact instability;
- compounding error;
- timeout;
- action saturation;
- sampler instability.

Use rollout videos and structured logs for a representative sample.

## Success gate

The hierarchy should demonstrate at least one clear advantage:

- higher success than flat flow;
- better sample efficiency;
- better recovery;
- better performance at longer horizons;
- improved robustness to perturbations.

If it only matches the flat policy, report that honestly and use the ablations to explain why.

---

# Phase 12: Final sample-efficiency experiments

## Goal

Test the thesis claim under controlled data budgets.

## Dataset sizes

For temporal and hierarchical methods, use nested subsets of the same causal trajectory dataset:

$$N\in\{50,100,200,500,1000,2000\}$$

For one-step BC and DAgger diagnostics, additionally report the number of independent state queries.

Report separately:

- causal trajectories;
- causal transitions;
- state-query samples;
- equivalent causal behavior time.

Do not equate one state query with one additional simulated transition.

Equivalent time:

$$T_{\mathrm{data}}=
\frac{N_{\mathrm{transitions}}}{20}$$

for 20 Hz control.

## Methods

At minimum:

- visual deterministic BC;
- visual flat flow;
- oracle hierarchy;
- deterministic hierarchy;
- generative hierarchy.

## Seeds

Use at least:

$$3$$

training seeds per method and dataset size.

Use 500 evaluation episodes per seed.

## Main plot

Horizontal axis:

$$N_{\mathrm{transitions}}$$

Vertical axis:

$$\mathrm{Success}$$

Use a logarithmic horizontal axis.

## Sample-efficiency summaries

Number of samples required for 50% success:

$$N_{50}=
\min
\left\{
N:
\mathrm{Success}(N)\geq0.5
\right\}$$

Number of samples required for 70% success:

$$N_{70}=
\min
\left\{
N:
\mathrm{Success}(N)\geq0.7
\right\}$$

Area under the learning curve:

$$\mathrm{AULC}
=
\int
\mathrm{Success}(\log N)
\,d\log N$$

## Final success criterion

The strongest positive result would be:

> The future-latent hierarchy reaches a target success rate with fewer demonstration transitions than a capacity-matched flat flow policy.

A useful negative result would be:

> Oracle future subgoals help, but learned future-subgoal generation removes the advantage.

That still identifies the high-level generator as the central bottleneck.

---

# Optional Phase 13: Wrist camera

## Goal

Test whether local contact visibility improves control after the base pipeline is working.

## Preconditions

Only start this phase after:

- privileged BC succeeds;
- privileged flow succeeds;
- temporal base-camera BC reaches a stable useful success rate;
- visual probes show a remaining local-contact information gap.

## Experiments

1. Base camera only.
2. Wrist camera only.
3. Base plus wrist.
4. Base plus wrist with temporal history.

## KPIs

- Visual BC success.
- Contact probe AUROC.
- Object-pose probe error.
- Recovery success after contact disturbances.
- Inference cost.

## Potential problems

- Wrist camera occlusion.
- Tool dominates the image.
- Two-camera feature dimension becomes unnecessarily large.
- Camera synchronization mismatch.
- The added view improves probes but not control.

---

# Experiment priority summary

## Immediate next experiments

1. Run PPO teacher through the BC evaluation path.
2. Copy PPO weights into a student wrapper.
3. Verify state-action temporal alignment.
4. Build a state-query dataset by querying deterministic PPO actions on stored states.
5. Retrain one-step privileged BC using only state-action queries.
6. Run multi-iteration privileged DAgger and add learner-visited state queries.
7. Collect fresh causal deterministic-teacher and perturb-and-recover trajectories.
8. Train one-step privileged flow from state queries.
9. Train action-chunk flow only from the fresh causal trajectories.

## Experiments to postpone

- More high-level hierarchy sweeps.
- Large arbitrary latent noise.
- More latent-horizon sweeps.
- Wrist-camera policy training.
- Direct flow RL fine-tuning.
- Sim-to-real alignment.
- Exact real-to-sim reconstruction.
- Further claims about hierarchy sample efficiency.

---

# Compact phase-gate table

| Phase | Main experiment | Minimum gate |
| --- | --- | --- |
| 0 | Pipeline sanity | Copied teacher policy matches PPO success within 1 percentage point |
| 1 | Privileged one-step BC from state queries | At least 70% success |
| 2 | Privileged DAgger plus fresh causal recovery rollouts | At least 80% success |
| 3 | Privileged flow: one-step from queries, chunks from causal trajectories | Within 5 percentage points of privileged BC |
| 4 | Visual deterministic BC | At least 50% success |
| 5 | Visual flat flow | Within 5 percentage points of visual BC |
| 6 | Latent validation | Pose, velocity, contact, and inverse-dynamics probes pass |
| 7 | Oracle hierarchy | At least matches flat visual flow |
| 8 | Deterministic high level | At least 70% of oracle-hierarchy success |
| 9 | Generative high level | At least matches deterministic hierarchy |
| 10 | Robust low level | Generated-goal success at least 80% of oracle-goal success |
| 11 | Complete hierarchy | Clear advantage in success, robustness, or sample efficiency |
| 12 | Final scaling | Three seeds, nested data subsets, 500 episodes per seed |

---

# Final decision rule

The hierarchy should only be judged after all of the following are true:

- privileged teacher distillation works;
- privileged flow matching works;
- visual flat control works;
- the latent preserves controllable state;
- the low level benefits from oracle future goals;
- the low level demonstrably uses the goal;
- predicted high-level latents remain on the data manifold.

Until then, low hierarchy success is not evidence against the future-latent idea. It is evidence that one of the prerequisite components is not yet strong enough.
