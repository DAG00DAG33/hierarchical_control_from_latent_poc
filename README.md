# Push-T Future-Latent Hierarchy POC

This repository implements a Push-T proof of concept for a two-level control
architecture where the high-level model predicts a future latent state and the
low-level policy uses that latent as a subgoal.

The intended experiment compares:

- a flat flow-matching action policy;
- hierarchical flow policies with high-level horizons of `0.05s`, `0.10s`, and `0.25s`;
- nested training sets with `50`, `100`, and `200` demonstration trajectories.

The observation is a frozen DINOv2 RGB feature plus non-privileged robot
proprioception. The latent encoder is trained with a separate action-conditioned
multi-horizon world model, and that world model is not used as the high-level
hierarchical model.

## Setup

```bash
uv sync --python 3.11
uv run hcl-poc doctor
uv run hcl-poc rl status --config configs/pusht.yaml
```

Privileged PPO training uses ManiSkill vectorized simulation and expects CUDA.
If `uv run hcl-poc doctor` reports `CUDA available: False`, fix the NVIDIA
driver/runtime before running the full experiment. CPU is only used for smoke
tests.

On the current machine, the observed CUDA failure was a kernel-module mismatch:
the system was booted into `6.17.0-35-generic` while NVIDIA modules were only
installed for `6.17.0-22-generic`. The matching package is:

```bash
sudo apt-get update
sudo apt-get install -y \
  linux-modules-nvidia-580-open-6.17.0-35-generic \
  linux-modules-nvidia-580-open-generic-hwe-24.04
sudo modprobe nvidia
nvidia-smi
```

If apt downloads 1187-byte files or reports `NOSPLIT`, the ETH guest network is
intercepting package requests; authenticate at `enter-guest-net.ethz.ch` first.

## Data

```bash
uv run hcl-poc rl train --config configs/pusht.yaml
uv run hcl-poc rl status --config configs/pusht.yaml
uv run hcl-poc rl eval --config configs/pusht.yaml
uv run hcl-poc data prepare --config configs/pusht.yaml
```

The current dataset source is a privileged-state PPO policy trained in this
repository with the same `pd_ee_delta_pos` action space used by the downstream
policies. The collector runs that PPO policy in `rgb+state` mode, keeps only
successful causal rollouts, extracts frozen DINOv2-S/14 RGB features, stores
`qpos`, `qvel`, and `tcp_pose` as proprioception, and writes the prepared HDF5
dataset under `data/`.

Collection has a quality gate: the PPO checkpoint must reach
`rl.collect_min_success` in privileged-state evaluation before any downstream
DINO/proprio dataset is generated.

This avoids the downloaded ManiSkill Push-T trajectories because, in the
current simulator install, their saved actions do not reproduce the successful
rollouts unless the environment state is overwritten during replay.

## Training And Evaluation

Run the staged proof of concept:

```bash
uv run hcl-poc run-sweep --config configs/pusht.yaml --profile staged
uv run hcl-poc report --config configs/pusht.yaml
```

The staged profile runs one seed across all data sizes and methods first. Add
more seeds after checking runtime and variance:

```bash
uv run hcl-poc run-sweep --config configs/pusht.yaml --profile full
uv run hcl-poc report --config configs/pusht.yaml
```

To run only the diagnostic flat baseline that conditions directly on the
normalized DINO/proprio observation, bypassing the learned latent encoder:

```bash
uv run hcl-poc train flat_obs --config configs/pusht.yaml --n-traj 50 --seed 0
uv run hcl-poc eval flat_obs --config configs/pusht.yaml --n-traj 50 --seed 0
```

To record inspection videos for a trained direct-observation flat policy:

```bash
uv run hcl-poc video flat_obs --config configs/pusht.yaml --n-traj 200 --seed 0 --episodes 4
```

Two imitation-learning diagnostics are also available:

```bash
uv run hcl-poc train bc_obs --config configs/pusht.yaml --n-traj 1000 --seed 0
uv run hcl-poc eval bc_obs --config configs/pusht.yaml --n-traj 1000 --seed 0

uv run hcl-poc train bc_obs_1step --config configs/pusht.yaml --n-traj 1000 --seed 0
uv run hcl-poc eval bc_obs_1step --config configs/pusht.yaml --n-traj 1000 --seed 0

uv run hcl-poc train bc_obs_dagger --config configs/pusht.yaml --n-traj 1000 --seed 0
uv run hcl-poc eval bc_obs_dagger --config configs/pusht.yaml --n-traj 1000 --seed 0

uv run hcl-poc train bc_pose --config configs/pusht.yaml --n-traj 1000 --seed 0
uv run hcl-poc eval bc_pose --config configs/pusht.yaml --n-traj 1000 --seed 0

uv run hcl-poc train bc_state --config configs/pusht.yaml --n-traj 1000 --seed 0
uv run hcl-poc eval bc_state --config configs/pusht.yaml --n-traj 1000 --seed 0
```

`bc_obs` is deterministic behavioral cloning from the same DINO/proprio input
used by the visual policies. `bc_obs_1step` removes the auxiliary 8-step action
chunk target and predicts only the next action. `bc_obs_dagger` rolls out the
one-step visual BC policy, labels those visited states with the privileged PPO
teacher, and retrains on demonstrations plus those DAgger labels. `bc_pose`
trains a supervised DINO-to-object-pose/goal bottleneck and then clones from
predicted pose plus proprioception. `bc_state` distills the privileged PPO
teacher from the full simulator state and is only a diagnostic; it is not part
of the final non-privileged method.

## Method

The encoder maps the current RGB/proprio observation to a latent state:

```text
z_t = E_o(DINO(rgb_t), proprio_t)
```

The encoder is trained with an action-conditioned multi-horizon world model:

```text
z_hat_{t+k} = F_dyn(z_t, A(a_t, ..., a_{t+k-1}), k)
```

The world model receives actions and is used only to shape the latent space.
After representation training, `E_o` is frozen.

The hierarchical high-level model is separate and does not receive actions:

```text
g_t ~ p_high(z_{t+k} | z_t)
```

The low-level policy receives the current latent and generated future latent:

```text
a_{t:t+H-1} ~ pi_low(a_{t:t+H-1} | z_t, g_t)
```

Both high-level subgoal generation and low-level action generation use
conditional flow matching.

## Results

Results are written by `hcl-poc report` into `results/ppo_main/`.

Current status on June 17, 2026:

- The privileged PPO expert was trained in this repo and reached `0.863`
  deterministic success over 256 evaluation episodes.
- The prepared DINO/proprio dataset contains 2000 successful PPO rollouts.
- The latent-conditioned flat baseline and the hierarchical runs currently
  produce zero closed-loop success, including the smaller `0.05s`, `0.10s`,
  and `0.25s` spatial-DINO hierarchy horizons.
- A direct-observation flat baseline (`flat_obs`) was added to test whether the
  learned WM latent encoder is the main failure point. It reduces the
  flow-matching training loss substantially. With more data it reached one
  successful rollout out of 50 at 1000 trajectories, but the 2000-trajectory
  run dropped back to zero under the current sampler/evaluator.
- `flat_obs` inspection videos were generated for n=200, n=1000, and n=2000 in
  `results/videos/`. Contact sheets for one n=1000 and one n=2000 rollout are
  in `results/videos/contact_sheets/`.
- The PPO teacher emits some raw actions outside the `Box(-1, 1, shape=(3,))`
  action space. Clipping the teacher does not change privileged PPO success
  (`0.863` raw and clipped), so downstream training and evaluation now clip
  actions to the environment bounds explicitly.
- Deterministic BC from DINO/proprio (`bc_obs`) remains near zero success, while
  deterministic BC from full privileged state (`bc_state`) reaches `0.46`
  success with 1000 teacher rollouts. This shows that plain BC can solve a
  substantial fraction of Push-T when object/goal state is available; the
  current DINO CLS-token observation is the main bottleneck before returning to
  hierarchy.
- One-step visual BC does not materially change the result, so the 8-step chunk
  target is not the main cause of failure. A single DAgger iteration with 5000
  PPO-labeled learner-visited states improves success from `0.02` to `0.04` and
  max reward from about `0.20` to `0.24`, but it remains far below privileged
  state BC.
- A supervised DINO pose/goal bottleneck reaches about `1 cm` held-out position
  error and `8 deg` yaw error, but BC from predicted pose plus proprioception
  gets `0.00` success. Compressing the image to this noisy low-dimensional
  bottleneck is worse than using DINO features directly.
- Spatial DINO features plus one-step BC are currently the strongest visual
  success result (`0.06`). Adding one DAgger iteration keeps success at `0.06`
  but improves final reward to `0.273` and max reward to `0.296`.
- The spatial direct-observation flow baseline reaches `0.04` success at 1000
  trajectories, so the flow sampler is not outperforming deterministic BC yet.
- A pose probe on the spatial features confirms that the observation contains
  the T pose, but the learned latent does not. A small MLP from spatial DINO
  predicts T position to about `4 mm` MAE and yaw to `4.5 deg` MAE. The same
  probe from the learned `z` is essentially baseline: about `3.6 cm`/`4.6 cm`
  position MAE and `30 deg` yaw MAE.
- Shortening the world-model training horizons did not fix the latent. With
  `horizons_steps: [1, 2, 5]`, the latent pose probe still gives about `3.6 cm`
  x MAE, `4.3 cm` y MAE, and `29.4 deg` yaw MAE. Training the same setup for
  90 epochs instead of 30 also does not help (`29.3 deg` yaw MAE), so the issue
  is not simply too few encoder epochs. A one-step-only world model is only
  marginally better (`28.9 deg` yaw MAE).
- Adding an intrinsic reconstruction regularizer to the world-model encoder is
  the first representation change that clearly helps. A 64D latent with
  reconstruction weight `0.1` improves the pose probe to `2.4 cm`/`3.2 cm` and
  `27.2 deg`. Increasing the latent to 512D, hidden width to 512, and
  reconstruction weight to `1.0` improves the frozen-latent probe to `8.2 mm`
  x MAE, `9.0 mm` y MAE, and `9.2 deg` yaw MAE. This is within about 2x of the
  spatial DINO probe (`3.8 mm`, `4.3 mm`, `4.5 deg`) without adding any direct
  pose-supervised encoder loss.

| Method | Trajectories | Success | Final reward | Max reward |
| --- | ---: | ---: | ---: | ---: |
| flat latent | 50 | 0.00 | 0.120 | 0.149 |
| flat latent | 100 | 0.00 | 0.122 | 0.151 |
| flat latent, spatial DINO | 1000 | 0.00 | 0.120 | 0.148 |
| flat latent, spatial DINO, 512D recon WM | 1000 | 0.08 | 0.214 | 0.255 |
| hier, spatial DINO, 0.05s | 1000 | 0.00 | 0.119 | 0.152 |
| hier, spatial DINO, 0.10s | 1000 | 0.00 | 0.113 | 0.150 |
| hier, spatial DINO, 0.25s | 1000 | 0.00 | 0.115 | 0.150 |
| flat obs | 50 | 0.00 | 0.123 | 0.175 |
| flat obs | 100 | 0.00 | 0.104 | 0.170 |
| flat obs | 200 | 0.00 | 0.124 | 0.188 |
| flat obs | 1000 | 0.02 | 0.144 | 0.204 |
| flat obs | 2000 | 0.00 | 0.116 | 0.170 |
| flat obs, spatial DINO | 1000 | 0.04 | 0.206 | 0.255 |
| BC obs | 1000 | 0.00 | 0.110 | 0.176 |
| BC obs | 2000 | 0.02 | 0.137 | 0.197 |
| BC obs, 1-step | 1000 | 0.02 | 0.142 | 0.196 |
| BC obs, DAgger | 1000 | 0.04 | 0.158 | 0.236 |
| BC predicted pose | 1000 | 0.00 | 0.119 | 0.158 |
| BC obs, spatial DINO | 1000 | 0.02 | 0.179 | 0.219 |
| BC obs, spatial DINO, 1-step | 1000 | 0.06 | 0.219 | 0.256 |
| BC obs, spatial DINO, DAgger | 1000 | 0.06 | 0.273 | 0.296 |
| BC privileged state | 1000 | 0.46 | 0.582 | 0.594 |

The direct-observation result and the probe suggest the original learned WM
latent was a major bottleneck: spatial DINO itself contains the object pose,
while the original action-conditioned world-model latent discarded it. The
512D reconstruction-regularized WM latent largely fixes this diagnostic and
also improves latent flat control from `0.00` to `0.08` success. The control
result is still below privileged-state BC and only slightly above direct
spatial-DINO policies, so the next useful hierarchy runs should use this
stronger encoder while continuing to treat the low-level policy itself as a
possible bottleneck.

A quick supervised probe checked whether the DINO CLS token contains the
T-block pose. A small MLP was trained from DINOv2-S/14 CLS features to
`obj_pose[x, y, sin(yaw), cos(yaw)]` on 4000 teacher rollout frames. On 800
held-out frames it reached `1.0 cm` x MAE, `1.2 cm` y MAE, and `9.0 deg` yaw
MAE. This means the CLS token is not blind to the T pose, but the remaining
error may still be too coarse for contact-rich pushing. Spatial DINO features
can be enabled with `configs/pusht_spatial.yaml`; they use CLS plus a 4x4
pooled patch-token grid and write to separate data/artifact/result paths. The
first spatial BC run improved reward but still only reached `0.02` success with
1000 trajectories, so the current failure is not explained by the CLS token
being completely pose-blind.
