# Learned Interface Experiment Log

This log records the experiments defined in
`learned_interface_experiment_plan.md`. Closed-loop success is the selection
criterion. The fixed reference is the raw-visual TCP endpoint hierarchy:
`k=10`, `U=10`, `H=1`, learned success `0.71`.

## 2026-06-21 - LI-00: Common held-goal pipeline and AE-256 control

- **Candidate:** `ae256_control`
- **Data:** 1,800 clean causal teacher trajectories; 200 fixed validation
  trajectories; frozen spatial DINO plus 21D proprioception.
- **Representation:** deterministic AE, 256D latent, width 1,024, three-hidden
  layer encoder/decoder, reconstruction weight `0.1`, 60 epochs, 400
  batches/epoch, batch size 512, AdamW `3e-4`.
- **Interface architecture:** current input remains the full normalized 6,549D
  visual/proprio vector. The future goal is the AE encoding of the observation
  10 steps ahead. A width-512 depth-4 high level predicts the normalized
  future latent. A separate width-512 depth-4 low level receives current raw
  input, held future latent, previous executed action, and normalized
  time-to-go. It is trained over offsets 1-10 while the future endpoint of
  each sampled window remains fixed.
- **Representation validation:** total reconstruction `0.05148`, DINO
  reconstruction `0.04841`, proprio reconstruction `0.00307`; all 256 latent
  dimensions active.
- **Probes:** object position MAE `0.00255/0.00301 m`, yaw MAE `0.0518 rad`,
  TCP position MAE `0.00310/0.00374 m`, contact AUROC `0.9928`, inverse-action
  MAE `0.0191`.
- **Offline control:** oracle action MAE `0.0383`, predicted-goal action MAE
  `0.0390`, prediction-induced action L2 `0.0119`.
- **Closed loop:** 20 fixed seeds beginning at 2,100,000. Learned goal success
  `0.55` (Wilson 95% `[0.342, 0.742]`); exact local branch-oracle latent
  success `0.65` (`[0.433, 0.819]`). Replay current-state error is exactly
  zero.
- **Decision:** The corrected `k=10`, `U=10` held-goal architecture is much
  stronger than the old AE-latent hierarchy, but the candidate does not yet
  meet the `0.60` learned-success gate. It is the deterministic control for
  the VAE sweep and remains eligible for relation-conditioning ablations.

## 2026-06-21 - LI-01: VAE-256 beta `1e-7`

- **Candidate:** `vae256_b1e7`
- **Representation:** 256D posterior-mean interface, width 1,024, beta
  `1e-7`, 20,000-step KL warmup, and 0.01 free bits per dimension. Downstream
  policies always use the posterior mean.
- **Representation validation:** reconstruction `0.04689`, DINO `0.04398`,
  proprio `0.00291`; KL total `1102.8`, mean KL/dimension `4.31`; posterior
  variance mean `0.00052`; all 256 dimensions active.
- **Probe comparison:** object position `0.00252/0.00269 m`, yaw `0.0482 rad`,
  TCP position `0.00291/0.00349 m`, contact AUROC `0.9927`, inverse-action MAE
  `0.0193`. These are slightly better than the AE on several static probes but
  essentially tied on inverse action.
- **Offline control:** oracle action MAE `0.0380`, predicted-goal action MAE
  `0.0385`, induced action L2 `0.0131`.
- **20-episode screen:** learned `0.65`, oracle `0.55`; uncertainty was too
  large and the ordering was implausible, so the candidate was promoted to
  100 episodes.
- **100-episode development:** learned `0.59` (Wilson
  `[0.492,0.681]`), oracle `0.65` (`[0.553,0.736]`). Final rewards are `0.697`
  and `0.745`. Exact replay state error is zero.
- **Decision:** The candidate improves learned success by four points over the
  new AE control and has a six-point oracle-to-learned gap. It misses the
  useful threshold by one point but satisfies the oracle `0.65` criterion.
  Promote it to relation-conditioning and sensitivity-aware prediction while
  continuing the beta sweep.

## 2026-06-21 - LI-02: VAE-256 beta `1e-6`

- **Representation:** KL total falls from `1102.8` to `699.2`; posterior
  variance rises to `0.00558`; all 256 dimensions remain active.
  Reconstruction remains similar at `0.04656`.
- **Probes:** inverse-action MAE worsens to `0.0219`; pose/contact probes remain
  strong.
- **20-episode closed loop:** learned `0.55`, oracle `0.60`.
- **Decision:** Reject as inferior to beta `1e-7` and the AE control in learned
  success. The additional regularity at this beta does not improve control.

## 2026-06-21 - LI-03: VAE-256 beta `1e-5`

- **Representation:** reconstruction worsens to `0.05655`; KL total falls to
  `247.2`; posterior variance rises to `0.435`; only 144 dimensions have
  standard deviation above 0.1. Inverse-action probe worsens to `0.0288`.
- **20-episode screen:** learned `0.80`, oracle `0.70`, triggering the
  mandatory 100-episode development evaluation.
- **100-episode development:** learned `0.65` (Wilson
  `[0.553,0.736]`), oracle `0.69` (`[0.594,0.772]`), learned/oracle ratio
  `0.942`. Final rewards are `0.745` and `0.774`.
- **Interpretation:** Moderate information compression improves closed-loop
  control despite worse reconstruction and probes. This is direct evidence
  that the latent geometry/control interface matters more than static
  recoverability.
- **Decision:** Promote as the leading learned interface. It is useful
  (`>=0.60`), meets the plan's `>=0.64` strong comparison target, and is six
  percentage points below the selected TCP hierarchy. Continue with capacity
  and conditioning tests before final multi-seed evaluation.

## 2026-06-21 - LI-04: VAE-512 width 2,048 beta `1e-7`

- **Representation:** 512D posterior-mean interface, width 2,048, beta
  `1e-7`, 20,000-step KL warmup, and 0.01 free bits per dimension. Training
  otherwise matches the 256D sweep.
- **Representation validation:** reconstruction improves to `0.04133`
  (`0.03726` DINO and `0.00407` proprio); KL total is `1665.5`, mean
  KL/dimension is `3.25`, posterior variance mean is `0.00175`, and all 512
  dimensions are active.
- **Probes:** object position `0.00254/0.00288 m`, yaw `0.0490 rad`, TCP
  position `0.00405/0.00449 m`, contact AUROC `0.9903`, and inverse-action
  MAE `0.0238`. The larger representation improves reconstruction but
  worsens the control-relevant TCP and inverse-action probes.
- **Offline control:** oracle action MAE `0.0380`, predicted-goal action MAE
  `0.0388`, prediction-induced action L2 `0.0164`.
- **20-episode screen:** learned success `0.40` (Wilson
  `[0.219,0.613]`); local branch-oracle success `0.45`
  (`[0.258,0.658]`). Replay current-state error remains exactly zero.
- **Decision:** Reject. It is worse than both the deterministic AE control and
  every promoted 256D VAE in closed loop. The result supports the hypothesis
  that excess latent capacity preserves nuisance variation and makes the
  future-state interface harder to use, despite lower reconstruction error.

## 2026-06-21 - LI-05: VAE-512 width 2,048 beta `1e-6`

- **Representation:** 512D posterior-mean interface, width 2,048, beta
  `1e-6`, 20,000-step KL warmup, and 0.01 free bits per dimension.
- **Representation validation:** reconstruction `0.04879` (`0.04404` DINO,
  `0.00474` proprio); KL total `1055.5`, mean KL/dimension `2.06`, posterior
  variance mean `0.0304`, and all 512 dimensions active.
- **Probes:** object position `0.00277/0.00309 m`, yaw `0.0538 rad`, TCP
  position `0.00427/0.00488 m`, contact AUROC `0.9908`, and inverse-action
  MAE `0.0265`. These diagnostics are worse than the best 256D candidates,
  reinforcing that probe ranking alone is not a reliable interface selector.
- **Offline control:** oracle action MAE `0.03819`, predicted-goal action MAE
  `0.03889`, prediction-induced action L2 `0.01592`, and normalized
  high-level goal L2 `19.59`.
- **20-episode screen:** learned success `0.90`, oracle success `0.70`. The
  inverted ordering required promotion because sampling uncertainty was
  large.
- **100-episode development:** learned success `0.72` (Wilson
  `[0.625,0.799]`) with final reward `0.786`; local branch-oracle success
  `0.76` (`[0.668,0.833]`) with final reward `0.825`. The learned/oracle
  success ratio is `0.947`, and exact branch replay error is zero.
- **Decision:** Promote as the Batch 1A winner. It reaches the plan's strong
  learned-interface threshold (`>=0.70`) and slightly exceeds the fixed TCP
  reference (`0.71`) on the same 100 evaluation seeds. It becomes the VAE
  source for relation-conditioning ablations and final multi-seed
  verification. The contrast with the otherwise identical beta `1e-7`
  candidate (`0.40` learned at 20 episodes) shows that useful stochastic
  regularization depends strongly on beta, not just capacity.

## 2026-06-21 - Batch 1A decision

The six-candidate minimal VAE batch is complete. Closed-loop learned/oracle
success was:

| candidate | screen learned | screen oracle | 100-episode learned | 100-episode oracle |
| --- | ---: | ---: | ---: | ---: |
| `ae256_control` | 0.55 | 0.65 | - | - |
| `vae256_b1e7` | 0.65 | 0.55 | 0.59 | 0.65 |
| `vae256_b1e6` | 0.55 | 0.60 | - | - |
| `vae256_b1e5` | 0.80 | 0.70 | 0.65 | 0.69 |
| `vae512_w2048_b1e7` | 0.40 | 0.45 | - | - |
| `vae512_w2048_b1e6` | 0.90 | 0.70 | **0.72** | **0.76** |

Selection is based on closed-loop success, not reconstruction or probe rank.
The AE remains the deterministic control; `vae512_w2048_b1e6` is the selected
learned representation.

## 2026-06-21 - LI-06: VAE-512 goal-conditioning ablation

All variants reuse the exact `vae512_w2048_b1e6` representation, normalized
latent goals, high-level predictor, data split, and evaluation seeds. Only
the low-level conditioning architecture is retrained.

| conditioning | learned success | oracle success | oracle/predicted action MAE | induced action L2 |
| --- | ---: | ---: | ---: | ---: |
| absolute concat | 0.90 | 0.70 | 0.0382 / 0.0389 | 0.0159 |
| latent delta | 0.60 | 0.60 | 0.0372 / 0.0377 | 0.0171 |
| relation MLP | 0.70 | 0.70 | **0.0323 / 0.0325** | **0.0111** |
| FiLM | 0.55 | 0.45 | 0.0403 / 0.0424 | 0.0537 |

These are 20-episode screens; the absolute-concat row is its original screen
for a matched comparison.

- **Delta:** replaces the absolute future latent with normalized
  `z_future-z_current`. It loses 30 points of learned success despite slightly
  better offline action MAE.
- **Relation MLP:** encodes `(z_current,z_future,time-to-go)` into a learned
  512D relation before the action MLP. It gives the best offline action
  diagnostics but remains below absolute concatenation in closed loop.
- **FiLM:** modulates four hidden layers from the future latent, initialized
  as identity modulation. It has the largest sensitivity to high-level
  prediction error and the worst oracle result.
- **Decision:** Retain absolute concatenation as the selected VAE interface.
  Keep relation MLP as a secondary candidate because it matches `0.70` on
  both goal sources and has materially better action diagnostics, but do not
  promote delta or FiLM.

## 2026-06-21 - LI-07: AE-256 goal-conditioning ablation

As in LI-06, all variants reuse the exact AE representation and high-level
predictor. Only the low-level conditioning architecture changes.

| conditioning | learned success | oracle success | oracle/predicted action MAE | induced action L2 |
| --- | ---: | ---: | ---: | ---: |
| absolute concat | 0.55 | 0.65 | 0.0383 / 0.0390 | 0.0119 |
| latent delta | 0.65 | 0.60 | 0.0370 / 0.0375 | 0.0130 |
| relation MLP | 0.65 | 0.75 | **0.0321 / 0.0324** | **0.0084** |
| FiLM | 0.70 | 0.75 | 0.0378 / 0.0400 | 0.0354 |

The table uses 20-episode screens. FiLM was promoted because it had the
highest learned success:

- **FiLM 100-episode development:** learned `0.55` (Wilson
  `[0.452,0.644]`), oracle `0.67` (`[0.573,0.754]`).
- **Interpretation:** Relation and FiLM improve the AE's oracle interface on
  the small screen, but the apparent learned FiLM gain disappears at adequate
  sample size. Relation again has the best offline action metrics, which do
  not translate directly into learned closed-loop ranking.
- **Decision:** Do not replace the selected VAE absolute-concat controller.
  AE-FiLM remains a useful negative control: a better oracle low-level
  interface cannot compensate for its weaker learned deployment.

## 2026-06-21 - LI-08: Denoising AE-256 sweep

The encoder receives normalized DINO and proprioception corrupted by
independent Gaussian noise and reconstructs the clean observation. All other
representation, hierarchy, and evaluation settings match the AE control.

| candidate | noise std | recon | inverse action MAE | screen learned | screen oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dae256_n005` | 0.005 | 0.05151 | 0.01902 | 0.60 | 0.65 |
| `dae256_n010` | 0.010 | 0.05072 | 0.01923 | 0.80 | 0.50 |

The `noise=0.01` inversion triggered 100-episode development evaluation:

- learned success `0.59` (Wilson `[0.492,0.681]`);
- oracle success `0.52` (`[0.423,0.615]`);
- final rewards `0.700` and `0.659`.

**Decision:** Reject both 256D denoising candidates. Small input noise leaves
the static probes and reconstruction essentially unchanged, and neither
candidate improves on the selected VAE after adequate closed-loop sampling.

## 2026-06-21 - LI-09: Denoising AE-512

- **Candidate:** `dae512_w2048_n005`, 512D latent, width 2,048, normalized
  input noise standard deviation `0.005`.
- **Representation:** reconstruction `0.04468` (`0.04057` DINO, `0.00411`
  proprio); all dimensions active. Object yaw probe is `0.0482 rad`, contact
  AUROC `0.9917`, and inverse-action MAE `0.0227`.
- **Offline control:** oracle/predicted action MAE `0.03821/0.03889`,
  prediction-induced action L2 `0.01511`.
- **20-episode closed loop:** learned success `0.60`; oracle success `0.75`.
- **Decision:** Reject. The oracle low level is competitive, but learned
  deployment is 12 points below the selected VAE and the oracle result does
  not exceed the VAE's 100-episode `0.76`. Denoising alone does not provide a
  better future-state interface.

## 2026-06-21 - Batch 1C decision

No denoising candidate is promoted. Across 256D and 512D models, modest input
noise changes reconstruction and probes little, while closed-loop learned
success remains unstable or inferior to `vae512_w2048_b1e6`.

## 2026-06-21 - LI-10: JEPA-style predictive latent sweep

All candidates use a 256D online encoder, an EMA target encoder
(`momentum=0.99`), an action-sequence GRU predictor, horizons
`{1,2,5,10}`, and VICReg variance/covariance regularization. The target future
latent is stop-gradient. The objective sweep changes reconstruction,
variance, and covariance weights.

| candidate | recon / var / cov | prediction MSE | recon MSE | inverse action MAE | screen learned/oracle |
| --- | --- | ---: | ---: | ---: | ---: |
| `jepa256_predonly_v1_c001` | 0 / 1 / 0.01 | 0.217 | - | 0.0575 | 0.40 / 0.70 |
| `jepa256_r001_v1_c001` | 0.01 / 1 / 0.01 | 0.221 | 0.311 | 0.0561 | 0.75 / 0.75 |
| `jepa256_r01_v1_c01` | 0.1 / 1 / 0.1 | 0.428 | 0.213 | 0.0538 | 0.80 / 0.55 |
| `jepa256_r1_v10_c01` | 1 / 10 / 0.1 | 0.478 | 0.117 | 0.0404 | 0.60 / 0.60 |

All 256 dimensions remain active. Predictive-only training preserves object
pose and contact but removes substantial one-step action information. More
reconstruction improves inverse dynamics while worsening predictive loss.

The two promising screens were promoted:

- **Weak reconstruction, 100 episodes:** learned `0.61` (Wilson
  `[0.512,0.700]`), oracle `0.66` (`[0.563,0.746]`).
- **Balanced reconstruction, 100 episodes:** learned `0.65` (`[0.553,0.736]`),
  oracle `0.58` (`[0.482,0.672]`).

Training used a 60-epoch ceiling. The first two candidates selected epochs
6-7, motivating validation early stopping with patience 10; the balanced
candidate stopped after 28 epochs. The strong-reconstruction objective
continued to the 60-epoch ceiling, so the stopping rule did not truncate a
still-improving run.

**Decision:** Reject the JEPA candidates as primary interfaces. Weak
reconstruction is useful but remains 11 points below the selected VAE in
learned success. Predictive loss alone is insufficient; some reconstruction
is necessary, but increasing it does not produce a monotonic control gain.

## 2026-06-21 - LI-11: Compact action-aware effect codes

The effect encoder receives the normalized current and horizon-end observation
plus the normalized horizon, and produces a pairwise code:

```text
effect = E([h_t, h_t+10, 1.0])
```

The representation objective combines normalized one-step action prediction,
VICReg variance/covariance regularization, and auxiliary future object,
TCP, and contact prediction. The 2,000-trajectory DINO file does not retain
privileged object state. A frozen observation probe was therefore trained on
the separate 12,000-sample causal Phase 6 probe dataset and used to generate
observation-derived pseudo-labels for the trajectory file. Its validation
quality is:

- object position RMSE: `0.00597 m`;
- object yaw MAE: `0.0702 rad`;
- TCP position RMSE: `0.01033 m`;
- contact accuracy/AUROC: `0.9621 / 0.9931`.

The probe is used only while training the representation auxiliary heads.
Neither privileged state nor probe output is supplied to the deployed high or
low policy.

All candidates use width 512, `lambda_action=1`,
`lambda_auxiliary=1`, `lambda_var=1`, `lambda_cov=0.01`, 200 batches
per epoch, a 40-epoch ceiling, and validation early stopping with patience 10.
The high/low hierarchy remains `k=10`, `U=10`, `H=1`.

| effect dim | action MSE | auxiliary MSE | active dims | screen learned | screen oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.0266 | 0.00529 | 16 | 0.65 | 0.60 |
| 32 | **0.0229** | 0.00605 | 32 | **0.65** | **0.80** |
| 64 | 0.0283 | 0.00701 | 64 | 0.45 | 0.50 |

`effect32` was promoted to 100 episodes:

- learned success `0.62` (Wilson `[0.522,0.709]`);
- oracle success `0.56` (`[0.462,0.653]`);
- learned/oracle final reward `0.725 / 0.690`;
- learned/oracle teacher action MAE `0.0906 / 0.0772`.

The 20-episode oracle result was not stable. The low policy is trained on
nominal teacher states, while online oracle effects are generated from
learner-visited states. One-step teacher imitation also makes the future effect
optional because the current observation alone predicts the action well.

**Decision:** Select 32D as the effect-code capacity, reject 16D and 64D, and
continue low-level goal-conditioning/goal-use work before claiming the oracle
gate. The deployable 32D result is useful (`0.62`) but remains below the
selected TCP interface (`0.71`).
