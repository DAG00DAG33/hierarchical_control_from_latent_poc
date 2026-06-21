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
