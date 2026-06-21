# Learned Interface Experiments for the Push-T Hierarchy

## 1. Motivation

The current Push-T proof of concept produced a strong result for a compact **future TCP endpoint** interface:

```text
current visual/proprio state -> future TCP endpoint -> time-conditioned low-level controller
```

This interface is useful, predictable, and deployable. However, it is also a **future motor waypoint**, not a learned future scene/effect latent.

The original learned AE-latent interface performed poorly in the deployable hierarchy, but this does not conclusively prove that learned latent interfaces are bad. The previous AE and VAE experiments were not exhaustive. In particular:

- the AE latent is not guaranteed to have meaningful geometry;
- the VAE used only one beta setting and may have been under-capacity or over-regularized;
- world-model / LeWorldModel-like objectives may need better hyperparameter sweeps;
- probe metrics are useful but insufficient;
- every promising representation must be tested in closed loop;
- the best learned interface should be compared against the TCP endpoint interface.

This plan defines a new set of experiments to test whether a learned latent/effect space can become a high/low-level interface competitive with the explicit TCP endpoint.

---

## 2. Main Research Question

Can we learn an interface variable:

$$
g_t
$$

such that:

1. it is predicted by the high level from current visual/proprio observations;
2. it is useful for the low level in closed loop;
3. it supports temporal abstraction over approximately 0.5 seconds;
4. it does not merely encode the next primitive action;
5. it performs close to the explicit TCP endpoint interface.

The benchmark to beat is the selected TCP endpoint hierarchy:

```text
k = 10
U = 10
H = 1
goal = future 3D TCP endpoint
current representation = raw spatial DINO + proprioception
```

---

## 3. Reference Results and Baselines

Use these as fixed references.

| Method | Deployable | Success | Notes |
| --- | ---: | ---: | --- |
| Direct visual BC | yes | ~0.60 | Three-seed mean from clean 1,800-trajectory corpus |
| Direct visual flow | yes | ~0.58 | Three-seed mean from clean 1,800-trajectory corpus |
| Original learned AE-latent hierarchy | yes | ~0.33-0.35 | Failed deployable learned-latent interface |
| Original AE-latent oracle hierarchy | no | ~0.69 | Oracle future latent is informative |
| Selected learned TCP hierarchy | yes | ~0.71 | Current best deployable hierarchy |
| Selected TCP branch oracle | no | ~0.71 matched setting, 0.81 earlier oracle setting | Shows high-level TCP prediction is not current bottleneck |

The target for a new learned latent interface is:

```text
minimum: >= flat visual BC
strong:  >= 0.9 * selected TCP hierarchy
ideal:   >= selected TCP hierarchy
```

Numerically:

```text
minimum: >= 0.60
strong:  >= 0.64
ideal:   >= 0.71
```

---

## 4. General Rules

### 4.1 Data

Primary training data:

```text
1,800 clean successful PPO causal trajectories
200 held-out validation trajectories
```

Use the existing prepared spatial-DINO/proprio dataset unless the experiment explicitly changes the observation stream.

For every representation experiment:

- train the representation from scratch;
- train the low-level policy from scratch;
- train the high-level predictor from scratch;
- use the same clean causal split;
- use the same evaluation seeds;
- keep DINO frozen unless explicitly testing visual encoder fine-tuning.

Do not use state-query data for future-goal or world-model training.

### 4.2 Closed-loop testing is mandatory

A representation is not selected from probes alone.

Every candidate must be evaluated in three stages:

1. **Representation probes**
2. **Oracle-goal low-level closed loop**
3. **Learned high-level closed loop**

The key metric is closed-loop success, not reconstruction loss or probe accuracy.

### 4.3 Interface settings

Use the same temporal abstraction as the selected TCP hierarchy:

```text
k = 10
U = 10
H = 1
```

where:

- `k=10`: goal is 10 control steps into the future;
- `U=10`: one high-level goal is held for 10 primitive controls;
- `H=1`: low level outputs one primitive action at a time.

The low-level policy should receive a time-to-go signal or equivalent offset conditioning, because the same goal is held while remaining time changes.

### 4.4 Evaluation protocol

For screening:

```text
20 episodes
1 policy seed
```

For serious candidates:

```text
100 episodes
1 policy seed
```

For final candidates:

```text
3 training seeds
200 evaluation episodes per seed
```

Report Wilson confidence intervals for success.

---

# 5. Candidate Interface Families

## Family A: Improved VAE latents

Goal:

```text
test whether a better-tuned VAE gives a smoother and more useful latent goal space than AE
```

Candidate interface:

$$
g_t = \mu_\phi(o_{t+k})
$$

where:

$$
q_\phi(z_t|o_t)=\mathcal{N}(\mu_\phi(o_t),\sigma_\phi(o_t)^2)
$$

Use the posterior mean for all downstream policies.

## Family B: Denoising / contractive reconstruction latents

Goal:

```text
make nearby observations map to nearby latents and improve local smoothness
```

Candidate interface:

$$
g_t = E(o_{t+k})
$$

with training corruptions or Jacobian regularization.

## Family C: LeWorldModel / JEPA-style predictive latents

Goal:

```text
learn a latent space optimized for predictable future dynamics rather than only reconstruction
```

Candidate interface:

$$
g_t = E(o_{t+k})
$$

with a prediction model:

$$
\hat{z}_{t+k}=F(z_t,a_{t:t+k-1})
$$

or a high-level predictor:

$$
\hat{z}_{t+k}=H(z_t,a_{t-1})
$$

Use anti-collapse regularization.

## Family D: Action-aware effect latents

Goal:

```text
learn a compact latent that preserves exactly what matters for low-level control
```

Candidate interface:

$$
e_t = E_g(o_t,o_{t+k})
$$

or:

$$
g_t = E_g(o_{t+k})
$$

The latent is trained with auxiliary action, inverse-dynamics, and future-effect objectives.

## Family E: Scene-only learned latents

Goal:

```text
test if a learned space can represent desired object/scene effect without future robot proprioception
```

Candidate interface:

$$
g_t = E_{\mathrm{scene}}(I_{t+k})
$$

The current input still contains robot proprioception, but the future goal should not contain future robot/TCP state.

## Family F: Learned latent plus TCP auxiliary supervision

Goal:

```text
bridge between explicit TCP endpoint and learned latent
```

Candidate interface:

$$
g_t = E(o_{t+k})
$$

with an auxiliary decoder:

$$
\hat{p}_{TCP,t+k}=D_{TCP}(g_t)
$$

This tests whether the latent can recover the benefits of TCP while still being more general.

---

# 6. Standard Closed-Loop Architecture for Learned Interfaces

For each candidate representation, train the following components.

## 6.1 Encoder

$$
z_t = E(o_t)
$$

or, for factorized encoders:

$$
z_t = [z_t^{scene}, z_t^{robot}]
$$

## 6.2 Low-level oracle-goal policy

The low level receives current observation representation, a future goal, previous action, and time-to-go:

$$
a_t =
\pi_{\mathrm{low}}
\left(
h_t,
g,
a_{t-1},
\tau
\right)
$$

where:

```text
h_t = current representation
g   = held future interface target
tau = normalized remaining time-to-go
```

Train over all offsets:

$$
j = 1,\ldots,k
$$

For each training window:

- endpoint/future goal is fixed at timestep `t+k`;
- sampled current state can be `t`, `t+1`, ..., `t+k-1`;
- remaining time-to-go is updated accordingly;
- target action is the teacher action at the sampled current state.

This mirrors the successful TCP endpoint training scheme.

## 6.3 Learned high-level predictor

The high level predicts the future interface target:

$$
\hat{g}_t =
\pi_{\mathrm{high}}(h_t,a_{t-1})
$$

During deployment:

1. predict a new goal every `U=10` actions;
2. hold the goal;
3. recompute derived relative features and time-to-go every action;
4. low level acts every primitive step.

## 6.4 Oracle version

The oracle version uses the real future goal from a local teacher branch or from the held-out causal teacher trajectory, depending on the diagnostic.

For final oracle tests, prefer the local branch oracle from the current student state.

---

# 7. Stage 1: VAE Hyperparameter and Capacity Sweep

## 7.1 Purpose

The previous VAE result used only one beta value and one capacity setting. It had competitive probes but poor closed-loop control. This stage tests whether VAE was prematurely rejected.

## 7.2 Model variants

Test latent dimensions:

```text
z_dim in {128, 256, 512}
```

Test encoder/decoder widths:

```text
width in {1024, 2048}
```

Test beta values:

```text
beta in {0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4}
```

`beta=0` is the deterministic AE control.

Use a KL warmup schedule:

```text
warmup_steps in {0, 5k, 20k}
```

Use free bits:

```text
free_bits in {0.0, 0.01, 0.05}
```

Do not run the full grid initially.

## 7.3 Initial sweep

Run:

| z dim | width | beta | warmup | free bits |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1024 | 0 | 0 | 0 |
| 256 | 1024 | 1e-7 | 20k | 0.01 |
| 256 | 1024 | 1e-6 | 20k | 0.01 |
| 256 | 1024 | 1e-5 | 20k | 0.01 |
| 256 | 1024 | 1e-4 | 20k | 0.01 |
| 512 | 2048 | 1e-7 | 20k | 0.01 |
| 512 | 2048 | 1e-6 | 20k | 0.01 |
| 512 | 2048 | 1e-5 | 20k | 0.01 |

## 7.4 Required metrics

Representation metrics:

- reconstruction MSE for DINO and proprio separately;
- KL total;
- KL per dimension;
- number of active dimensions;
- posterior variance statistics;
- latent norm distribution.

Probe metrics:

- object pose;
- object velocity;
- TCP pose;
- TCP velocity;
- contact AUROC;
- inverse action MAE;
- forward action-conditioned prediction.

Geometry metrics:

- nearest-neighbor teacher-action MAE;
- latent distance versus object distance;
- latent distance versus TCP distance;
- latent distance versus teacher-action distance;
- interpolation plausibility;
- local perturbation smoothness.

Closed-loop metrics:

- matched flat latent policy success;
- oracle-goal hierarchy success;
- learned-goal hierarchy success;
- oracle-to-learned gap.

## 7.5 Gate

Promote a VAE candidate only if it satisfies at least one of:

```text
oracle-goal success > AE oracle-goal success
learned-goal success > AE learned-goal success
learned-goal success >= 0.60
```

Reject if:

- active dimensions collapse;
- closed-loop success remains below deterministic AE;
- probes improve but closed-loop worsens.

---

# 8. Stage 2: Denoising and Contractive AE

## 8.1 Purpose

A VAE encourages distributional regularity, but it does not guarantee local control smoothness. A denoising or contractive AE may be more directly useful.

## 8.2 Denoising AE

Train:

$$
\tilde{o}_t = o_t + \epsilon
$$

$$
z_t = E(\tilde{o}_t)
$$

$$
\hat{o}_t = D(z_t)
$$

Loss:

$$
\mathcal{L}
=
\mathcal{L}_{recon}(D(E(\tilde{o}_t)), o_t)
$$

Use separate noise scales:

```text
DINO noise std in {0.005, 0.01, 0.02}
proprio noise std in {0.005, 0.01, 0.02}
```

## 8.3 Temporal denoising AE

Use nearby frames as positive recon targets:

$$
D(E(o_t)) \approx o_{t+\delta}
$$

with:

```text
delta in {-1, 0, +1}
```

This encourages invariance to tiny temporal changes but may remove velocity/contact information, so test carefully.

## 8.4 Contractive AE

Add a Jacobian penalty:

$$
\mathcal{L}
=
\mathcal{L}_{recon}
+
\lambda_J
\left\|
\frac{\partial E(o)}{\partial o}
\right\|_F^2
$$

Test:

```text
lambda_J in {1e-6, 1e-5, 1e-4}
```

If exact Jacobian is expensive, use Hutchinson approximation.

## 8.5 Gate

Promote only if closed-loop oracle or learned hierarchy improves. Do not promote solely based on smoother interpolation.

---

# 9. Stage 3: LeWorldModel / JEPA-Style Predictive Latents

## 9.1 Purpose

Test whether a predictive latent with anti-collapse regularization creates a better high-level goal space than reconstruction-only latents.

## 9.2 Base objective

Encoder:

$$
z_t = E(o_t)
$$

Predictor:

$$
\hat{z}_{t+k}=F(z_t,a_{t:t+k-1})
$$

Target:

$$
z_{t+k}^{target} = \mathrm{stopgrad}(E_{\mathrm{target}}(o_{t+k}))
$$

Prediction loss:

$$
\mathcal{L}_{pred}
=
\left\|
\hat{z}_{t+k}
-
z_{t+k}^{target}
\right\|_2^2
$$

Use an EMA target encoder:

$$
E_{\mathrm{target}}
\leftarrow
mE_{\mathrm{target}}+(1-m)E
$$

Anti-collapse regularization:

```text
VICReg / SIGReg / variance-covariance regularization
```

Full loss:

$$
\mathcal{L}
=
\lambda_{pred}\mathcal{L}_{pred}
+
\lambda_{var}\mathcal{L}_{var}
+
\lambda_{cov}\mathcal{L}_{cov}
+
\lambda_{recon}\mathcal{L}_{recon}
$$

## 9.3 Hyperparameter sweep

Test:

```text
z_dim in {128, 256, 512}
predictor_hidden in {1024, 2048}
ema_momentum in {0.99, 0.995, 0.999}
lambda_pred in {1.0}
lambda_var in {1, 10}
lambda_cov in {0.01, 0.1, 1}
lambda_recon in {0, 0.01, 0.1, 1.0}
horizon_offsets in {{1,2,4,8}, {1,2,5,10}, {5,10,20}}
```

Do not use only the previous default setting.

## 9.4 Important ablations

Run:

1. Predictive only, no reconstruction.
2. Predictive plus weak reconstruction.
3. Predictive plus balanced reconstruction.
4. Predictive plus inverse dynamics.
5. Predictive plus action consistency.
6. Predictive plus stop-gradient target but no EMA.
7. Predictive plus EMA target.

## 9.5 Closed-loop tests

For each promoted candidate:

1. matched flat latent policy;
2. oracle-goal hierarchy;
3. learned high-level hierarchy;
4. held-goal hierarchy with `k=10,U=10`.

## 9.6 Gate

Promote a world-model candidate only if:

```text
oracle-goal hierarchy success >= AE oracle-goal success
or
learned-goal hierarchy success >= AE learned-goal success + 0.05
```

If probes improve but closed-loop does not, reject as interface representation.

---

# 10. Stage 4: Action-Aware Effect Codes

## 10.1 Purpose

The latent interface should preserve what matters for choosing actions. Full reconstruction may preserve nuisance information, while pure prediction may discard control-sensitive details.

This stage learns a compact effect code:

$$
e_t = E_g(o_t,o_{t+k})
$$

or:

$$
g_t = E_g(o_{t+k})
$$

that is explicitly trained to be useful for low-level control.

## 10.2 Pairwise effect encoder

Define:

$$
e_{t,k}=E_g(h_t,h_{t+k},k)
$$

Low level:

$$
a_t=\pi_{\mathrm{low}}(h_t,e_{t,k},a_{t-1},\tau)
$$

High level:

$$
\hat{e}_{t,k}=H(h_t,a_{t-1},k)
$$

This avoids assuming:

$$
e_{t,k}=z_{t+k}-z_t
$$

## 10.3 Training objectives

Start with:

$$
\mathcal{L}
=
\lambda_a
\left\|
\pi_{\mathrm{low}}(h_t,e_{t,k})-a_t
\right\|^2
+
\lambda_{var}\mathcal{L}_{var}
+
\lambda_{cov}\mathcal{L}_{cov}
$$

Then add optional decoders:

- future TCP decoder;
- future object pose decoder;
- future contact decoder;
- future scene decoder.

## 10.4 Dimensions

Test:

```text
effect_dim in {8, 16, 32, 64, 128}
```

## 10.5 Key risk

If the effect code is trained only through action prediction, it may become an action latent rather than a future-state/effect latent.

Therefore require auxiliary decoders:

```text
decode future TCP
decode future object pose
decode contact
```

and test whether the code remains meaningful.

## 10.6 Gate

This is the most promising learned-space alternative to TCP.

Promote if:

```text
learned effect-code hierarchy >= 0.64
strong candidate if >= 0.70
```

Also require:

```text
oracle effect-code hierarchy >= selected TCP learned hierarchy - 0.05
```

---

# 11. Stage 5: Scene-Only and Object-Effect Learned Goals

## 11.1 Purpose

Test whether a learned latent can represent desired scene/object effect without future robot motion.

## 11.2 Scene-only encoder

Train:

$$
z_t^{scene}=E_{scene}(I_t)
$$

The current low level receives:

$$
[h_t^{scene}, h_t^{robot}]
$$

but the goal contains only:

$$
g_t=z_{t+k}^{scene}
$$

## 11.3 Object/effect supervised auxiliary targets

Add auxiliary decoders from the goal latent:

- T object pose;
- T object velocity;
- contact;
- task progress;
- maybe TCP only as an auxiliary, not as the main target.

## 11.4 Variants

Compare:

1. scene-only AE;
2. scene-only VAE;
3. scene-only denoising AE;
4. scene-only predictive latent;
5. scene-only effect code.

## 11.5 Gate

Promote only if:

```text
oracle scene-goal hierarchy >= 0.60
learned scene-goal hierarchy >= 0.50
```

This is allowed to be lower than TCP, because it is more aligned with the original conceptual goal.

If it fails, document that Push-T at this horizon requires motor-waypoint information.

---

# 12. Stage 6: Goal-Conditioning Architecture Ablations

## 12.1 Purpose

The representation might be reasonable, but the low-level may be using it poorly. Test relation architectures before rejecting a latent space.

## 12.2 Conditioning variants

For each promising latent representation, compare:

### Concatenation

$$
a_t=\pi([h_t,g_t,a_{t-1},\tau])
$$

### Delta

$$
a_t=\pi([h_t,g_t-h_t,a_{t-1},\tau])
$$

only where dimensions align.

### Learned relation

$$
r_t=R(h_t,g_t,\tau)
$$

$$
a_t=\pi([h_t,r_t,a_{t-1},\tau])
$$

### FiLM conditioning

Use the goal to modulate hidden activations:

$$
h_{\ell+1}=\gamma(g)\odot f(W h_\ell)+\beta(g)
$$

### Cross-attention

Treat current tokens and goal tokens separately:

```text
current tokens attend to goal tokens
```

This is especially relevant for scene/patch-based goals.

## 12.3 Gate

If a latent fails only under delta/concat but succeeds under relation or attention conditioning, keep it as a valid learned interface.

---

# 13. Stage 7: Sensitivity-Aware High-Level Prediction

## 13.1 Purpose

Uniform latent MSE may optimize nuisance dimensions. We need prediction errors that matter for the low level.

## 13.2 Low-level sensitivity

Estimate:

$$
J_g=
\frac{\partial a}{\partial g}
$$

Compute:

$$
e_{\mathrm{sensitive}}
=
\left\|
J_g(\hat{g}-g)
\right\|
$$

## 13.3 Sensitivity-weighted predictor loss

Train high level with:

$$
\mathcal{L}_{high}
=
\left\|
W(\hat{g}-g)
\right\|^2
$$

where:

$$
W
$$

is derived from empirical goal-dimension action sensitivity.

Alternative:

$$
\mathcal{L}_{high}
=
\left\|
\pi_{\mathrm{low}}(h_t,\hat{g})
-
\pi_{\mathrm{low}}(h_t,g)
\right\|^2
$$

## 13.4 Gate

Use only if:

- oracle latent hierarchy works;
- learned latent hierarchy fails;
- prediction error is concentrated in high-sensitivity directions.

---

# 14. Stage 8: Candidate Selection Rather Than Single Prediction

## 14.1 Purpose

A learned latent high level may be multimodal. Single deterministic regression may predict an average future that is not useful.

## 14.2 Candidate generation

Generate:

$$
g^{(1)},\ldots,g^{(M)}
$$

using:

- VAE prior samples;
- conditional flow;
- mixture density network;
- nearest-neighbor future retrieval;
- stochastic perturbations around the deterministic prediction.

## 14.3 Candidate scoring

Score candidates using:

1. low-level reachability estimate;
2. action-sensitivity plausibility;
3. decoder/probe plausibility;
4. future progress predictor;
5. optional learned value.

## 14.4 Gate

Only pursue if there is empirical multimodality:

- multiple valid futures from similar states;
- deterministic predictor underperforms oracle;
- sampled candidates include valid alternatives.

Do not use stochastic sampling if diversity is just model noise.

---

# 15. Prioritized Execution Plan

## Tier 1: Fast, high-information experiments

Run first.

1. VAE capacity/beta sweep with closed-loop oracle and learned hierarchy.
2. Relation-network conditioning for AE/VAE latents.
3. Denoising AE.
4. LeWorldModel with weak/balanced reconstruction and EMA target.
5. Compact action-aware effect code.

## Tier 2: Conceptual experiments

Run after Tier 1.

1. Scene-only learned latent.
2. Object-effect latent with auxiliary physical decoders.
3. Sensitivity-weighted high-level prediction.
4. Candidate selection.

## Tier 3: Expensive final verification

Run only for the best 2-3 learned interfaces.

1. Three training seeds.
2. 200 episodes per seed.
3. Clean and disturbed evaluation.
4. Comparison against selected TCP hierarchy and flat BC/flow.
5. Videos of successes and failures.

---

# 16. Minimal First Batch

To avoid an unbounded sweep, start with this exact batch.

## Batch 1A: VAE revisit

```text
z_dim=256,width=1024,beta=0
z_dim=256,width=1024,beta=1e-7,warmup=20k,free_bits=0.01
z_dim=256,width=1024,beta=1e-6,warmup=20k,free_bits=0.01
z_dim=256,width=1024,beta=1e-5,warmup=20k,free_bits=0.01
z_dim=512,width=2048,beta=1e-7,warmup=20k,free_bits=0.01
z_dim=512,width=2048,beta=1e-6,warmup=20k,free_bits=0.01
```

For each:

- train representation;
- run probes;
- train time-conditioned low level with oracle goals;
- train high level;
- evaluate learned hierarchy.

## Batch 1B: Relation conditioning

Use the best AE and best VAE from Batch 1A.

Compare low-level conditioning:

```text
concat absolute
delta
relation MLP
FiLM
```

## Batch 1C: Denoising AE

```text
z_dim=256,width=1024,noise=0.005
z_dim=256,width=1024,noise=0.01
z_dim=512,width=2048,noise=0.005
```

## Batch 1D: LeWorldModel-like objective

```text
z_dim=256
lambda_recon in {0.01,0.1,1.0}
lambda_var in {1,10}
lambda_cov in {0.01,0.1}
ema_momentum=0.99
horizon_offsets={1,2,5,10}
```

Run only 4-6 combinations first.

## Batch 1E: Compact effect code

```text
effect_dim in {16,32,64}
object/TCP/contact auxiliary decoders enabled
low-level action objective enabled
```

This may be the most likely learned-space alternative to explicit TCP.

---

# 17. Result Tables to Produce

## Table 1: Representation diagnostics

| model | z dim | beta/reg | recon | KL/active dims | object probe | TCP probe | contact | inverse action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Table 2: Geometry diagnostics

| model | NN action MAE | latent-object corr | latent-TCP corr | latent-action corr | interpolation score |
| --- | ---: | ---: | ---: | ---: | ---: |

## Table 3: Closed-loop interface performance

| model | oracle success | learned success | oracle/learned gap | final reward | teacher MAE |
| --- | ---: | ---: | ---: | ---: | ---: |

## Table 4: Comparison to TCP

| interface | deployable success | oracle success | decisions/episode | conceptual type |
| --- | ---: | ---: | ---: | --- |
| TCP endpoint | 0.71 | 0.71 | ~6 | motor waypoint |
| learned latent candidate | TBD | TBD | TBD | learned goal space |

---

# 18. Decision Rules

## Strong learned latent interface

A learned latent interface is strong if:

```text
learned success >= 0.70
oracle success >= 0.70
learned/oracle ratio >= 0.85
```

This would make it competitive with TCP.

## Useful but weaker learned interface

A learned latent interface is still useful if:

```text
learned success >= 0.60
oracle success >= 0.65
```

This would match or beat flat visual BC and support the hierarchy idea.

## Representation bottleneck

If:

```text
oracle success is high
learned success is low
```

then the low-level can use the latent, but the high level cannot predict it.

Work on high-level prediction, sensitivity-weighted losses, or candidate selection.

## Low-level interface bottleneck

If:

```text
oracle success is low
```

then the latent is not useful to the low level, regardless of the high-level predictor.

Work on representation or goal-conditioning architecture.

## Probe mismatch

If probes are good but closed-loop is bad, reject the representation for now.

Closed-loop success has priority.

---

# 19. Recommended Interpretation Language

Use this language if the experiments succeed:

```text
The explicit TCP endpoint is not the only viable high/low-level interface.
With appropriate regularization and closed-loop selection, a learned latent/effect
space can approach the performance of the TCP waypoint while retaining a more
general state-based interface.
```

Use this language if they fail:

```text
The learned latent spaces tested here produced reasonable probes but failed to
match the closed-loop performance of the explicit TCP endpoint. For Push-T, the
most reliable interface is a compact physical waypoint. Learned future-state
interfaces remain an open direction for richer tasks where the relevant effect
cannot be captured by a simple TCP target.
```

---

# 20. Main Warning

Do not select a latent representation because it is visually smooth, has good VAE interpolation, or performs well on probes.

The only valid selection criterion is:

```text
closed-loop oracle success + learned high-level success + robustness
```

The TCP endpoint became strong because it passed closed-loop tests, not because it was elegant. Learned latent spaces must be held to the same standard.
