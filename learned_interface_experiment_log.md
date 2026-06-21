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
