# Push-T Future-Latent Hierarchy POC

This repository implements a Push-T proof of concept for a two-level control
architecture where the high-level model predicts a future latent state and the
low-level policy uses that latent as a subgoal.

The intended experiment compares:

- a flat flow-matching action policy;
- hierarchical flow policies with high-level horizons of `0.25s`, `1s`, and `4s`;
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
- The latent-conditioned flat baseline and the first hierarchical staged runs
  currently produce zero closed-loop success.
- A direct-observation flat baseline (`flat_obs`) was added to test whether the
  learned WM latent encoder is the main failure point. It reduces the
  flow-matching training loss substantially. With more data it reached one
  successful rollout out of 50 at 1000 trajectories, but the 2000-trajectory
  run dropped back to zero under the current sampler/evaluator.
- Four `flat_obs` n=200 inspection videos were generated in `results/videos/`.

| Method | Trajectories | Success | Final reward | Max reward |
| --- | ---: | ---: | ---: | ---: |
| flat latent | 50 | 0.00 | 0.120 | 0.149 |
| flat latent | 100 | 0.00 | 0.122 | 0.151 |
| flat obs | 50 | 0.00 | 0.123 | 0.175 |
| flat obs | 100 | 0.00 | 0.104 | 0.170 |
| flat obs | 200 | 0.00 | 0.124 | 0.188 |
| flat obs | 1000 | 0.02 | 0.144 | 0.204 |
| flat obs | 2000 | 0.00 | 0.116 | 0.170 |

The direct-observation result suggests the learned WM latent is not the only
issue. The action policy objective/sampling or closed-loop imitation setup needs
debugging before spending more compute on hierarchical ablations.
