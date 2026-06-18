# Push-T POC: Implementation, Experiments, and Results

This document records the implementation changes, debugging work, ablations,
validation, and measured results for the Push-T future-latent hierarchy proof
of concept. It complements the shorter overview in [README.md](README.md).

Status date: June 18, 2026.

## 1. Objective

The project tests whether a hierarchical imitation-learning policy can use
demonstrations more efficiently than a flat policy on ManiSkill `PushT-v1`.

The intended hierarchy is:

```text
RGB image + proprioception
            |
            v
frozen DINOv2 image features
            |
            v
observation encoder E_o
            |
            v
current latent z_t
       /           \
      v             v
high-level flow     low-level flow
z_t -> future z     (z_t, future z) -> action chunk
```

The action-conditioned world model used to train `E_o` is separate from the
high-level policy:

```text
world-model training:
    z_hat_{t+k} = F_dyn(z_t, action_sequence, k)

hierarchical policy:
    g_t ~ p_high(z_{t+k} | z_t)
    actions ~ pi_low(actions | z_t, g_t)
```

The high-level policy does not receive actions. The world model does receive
actions, but is discarded after representation training.

## 2. Current Main Configuration

The strongest and most extensively evaluated configuration is
[`configs/pusht_spatial_wm_recon512_lownoise.yaml`](configs/pusht_spatial_wm_recon512_lownoise.yaml).

| Component | Current setting |
| --- | --- |
| Simulator | ManiSkill 3.0.1, `PushT-v1` |
| Control | `pd_ee_delta_pos`, 20 Hz |
| Observation | RGB plus non-privileged robot proprioception |
| Image encoder | Frozen `facebook/dinov2-small` |
| Visual feature | CLS token plus pooled 4x4 spatial patch grid |
| Proprioception | `qpos`, `qvel`, and `tcp_pose` |
| Latent dimension | 512 |
| World-model hidden width | 512 |
| World-model horizons | 1, 2, and 5 control steps |
| World-model epochs | 30 |
| Reconstruction weight | 1.0 |
| Action chunk | 8 steps |
| Flow integration steps | 12 |
| Flow-policy epochs | 25 |
| Hierarchy horizons | 0.05, 0.10, and 0.25 seconds |
| Low-level subgoal noise | Standard deviation 0.5 |
| Training-set sizes | 50, 100, 200, 1000, and 2000 trajectories |
| Final evaluation | 500 episodes per policy and seed |
| Policy seeds | 0, 1, and 2 through n=1000; 0 and 1 at n=2000 |

At 20 Hz, the hierarchy horizons correspond to 1, 2, and 5 control steps.

## 3. Implementation Changes

### 3.1 Project scaffold and experiment tooling

The initial implementation added:

- a Python 3.11 project managed with `uv`;
- configuration-driven training and evaluation;
- HDF5 trajectory preparation and loading;
- frozen DINOv2 feature extraction;
- the action-conditioned world model and observation encoder;
- conditional flow-matching flat, high-level, and low-level policies;
- sweep orchestration, JSON metrics, CSV summaries, and plots;
- smoke tests and runtime environment checks.

The main commands are exposed through `hcl-poc`:

```bash
uv run hcl-poc doctor
uv run hcl-poc data prepare --config <config>
uv run hcl-poc train <method> --config <config> --n-traj <n> --seed <seed>
uv run hcl-poc eval <method> --config <config> --n-traj <n> --seed <seed>
uv run hcl-poc report --config <config>
```

### 3.2 Demonstration replay investigation

The downloaded ManiSkill demonstration actions did not reliably reproduce the
recorded successful behavior in the installed simulator. The trajectories only
looked correct when their saved simulator states were restored directly.

Clipping those actions was rejected as the primary fix because clipping a
non-reproducible trajectory does not preserve its success. Instead, the data
source was replaced with a locally trained teacher using the exact downstream
action space.

### 3.3 Privileged PPO teacher

A privileged-state PPO policy was implemented and aligned with the ManiSkill
baseline setup:

- vectorized CUDA simulation;
- normalized dense reward;
- partial resets during training;
- deterministic evaluation;
- `pd_ee_delta_pos`, matching the imitation policies;
- a collection quality gate before demonstrations are accepted.

The final teacher reached `0.863` deterministic success over 256 evaluation
episodes. The collector retains only successful causal rollouts.

The teacher sometimes outputs values outside the environment's
`Box(-1, 1, shape=(3,))` action space. Evaluating the teacher with and without
explicit clipping gave the same `0.863` success, so action clipping is now
applied consistently during downstream data preparation, training, and
evaluation.

### 3.4 Prepared dataset

The current spatial-DINO dataset contains 2000 successful teacher rollouts:

```text
data/prepared/pusht_ppo_dino_spatial_proprio_tcp.h5
```

For each step it stores:

- frozen DINOv2 spatial features;
- robot proprioception;
- the teacher action.

The successful trajectories are not all 100 steps long:

| Dataset | Trajectories | Minimum | Maximum | Mean |
| --- | ---: | ---: | ---: | ---: |
| Current prepared teacher dataset | 2000 | 11 | 100 | 44.72 |
| Earlier raw recorded dataset | 218 | 40 | 100 | 97.24 |

At 20 Hz, the current training demonstrations average about 2.24 seconds.
Successful collection episodes terminate when success occurs; they are not
padded to 100 actions.

### 3.5 Direct-observation and imitation diagnostics

Several diagnostic policies were added to isolate the source of the original
zero-success result:

- `flat_obs`: flow matching directly from DINO plus proprioception, bypassing
  the learned world-model latent;
- `bc_obs`: deterministic behavioral cloning from the same visual input;
- `bc_obs_1step`: predicts one action instead of an 8-step action chunk;
- `bc_obs_dagger`: one DAgger iteration using the privileged PPO teacher;
- `bc_pose`: predicts object/goal pose from DINO and clones from that pose
  bottleneck plus proprioception;
- `bc_state`: clones the teacher from full privileged state.

These tests established that:

- action chunking was not the primary failure;
- deterministic imitation itself can work when privileged state is available;
- the original visual and latent representations were the dominant early
  bottlenecks;
- DAgger improves the visited-state distribution but does not close the large
  gap to privileged-state BC.

### 3.6 DINO observation changes

The original observation used only the DINOv2 CLS token. A pose probe showed
that CLS contains useful object information, but only at approximately:

- 1.0 cm x-position MAE;
- 1.2 cm y-position MAE;
- 9.0 degrees yaw MAE.

The representation was changed to include a 4x4 adaptive pooling of DINO patch
tokens in addition to CLS. A small held-out MLP probe from these spatial
features reached:

- 3.8 mm x-position MAE;
- 4.1 mm y-position MAE;
- 3.8 degrees yaw MAE.

The probe audit included train/validation metrics, linear and MLP probes, and a
shuffled-label control. The shuffled-label model failed on held-out data,
confirming that the useful result was not just probe memorization.

### 3.7 World-model representation ablations

The original 64D latent discarded nearly all recoverable T pose information.
The following changes were tested without adding direct pose supervision to
the encoder:

| World-model variant | Held-out x MAE | Held-out y MAE | Held-out yaw MAE |
| --- | ---: | ---: | ---: |
| Original 64D latent | 3.62 cm | 4.57 cm | 30.3 deg |
| Short horizons `[1,2,5]` | 3.57 cm | 4.33 cm | 29.4 deg |
| Short horizons, 90 epochs | 3.60 cm | 4.65 cm | 29.3 deg |
| One-step-only horizon | 3.55 cm | 4.08 cm | 28.9 deg |
| 64D, reconstruction weight 0.1 | 2.37 cm | 3.20 cm | 27.2 deg |
| 512D/512 width, no reconstruction | 3.64 cm | 4.77 cm | 28.7 deg |
| 512D/512 width, reconstruction weight 1.0 | 8.2 mm | 9.0 mm | 9.2 deg |
| Spatial DINO input probe | 3.8 mm | 4.1 mm | 3.8 deg |

The reconstruction objective reconstructs the encoder input features. It does
not use T pose labels. The comparison with the 512D model without
reconstruction shows that capacity alone did not produce the improvement.

### 3.8 Smaller hierarchy horizons

The first hierarchy used 0.25, 1, and 4 second subgoals. Those horizons were
too long relative to this small contact-rich task. The final set is:

```text
0.05 s, 0.10 s, 0.25 s
```

This corresponds to predicting 1, 2, or 5 control steps into the future.

### 3.9 Low-level hierarchy robustness

An offline hierarchy diagnostic compared the low-level policy under two
conditions:

1. the true future latent from the demonstration;
2. a future latent sampled by the high-level flow model.

Without subgoal perturbation during training:

| Horizon | Oracle-subgoal action MAE | Sampled-subgoal action MAE |
| --- | ---: | ---: |
| 0.05 s | 0.075 | 0.337 |
| 0.10 s | 0.071 | 0.333 |
| 0.25 s | 0.069 | 0.382 |

The low-level policy was accurate for oracle subgoals but brittle to the
high-level model's sampled latent distribution.

Training the low-level policy with latent subgoal noise reduced the sampled
subgoal action MAE to:

| Horizon | Oracle-subgoal action MAE | Sampled-subgoal action MAE |
| --- | ---: | ---: |
| 0.05 s | 0.095 | 0.097 |
| 0.10 s | 0.092 | 0.095 |
| 0.25 s | 0.093 | 0.103 |

This is the `low_subgoal_noise_std: 0.5` setting in the final configuration.

### 3.10 Evaluation protocol

Early iterations used 50 evaluation episodes for fast diagnosis. Because
success rates are low, the final protocol was increased to:

- 500 episodes per method, dataset size, and seed;
- seeds 0, 1, and 2 through 1000 trajectories;
- seeds 0 and 1 at 2000 trajectories;
- fixed evaluation initializations starting at seed 10000;
- a maximum of 100 actions per episode;
- success, final normalized dense reward, maximum normalized dense reward,
  and inference latency as recorded KPIs.

At 20 Hz, a full 100-step rollout represents 5 seconds of simulated time.

### 3.11 Camera and video inspection

ManiSkill supports cameras mounted to robot links. Push-T uses the custom
`panda_stick` robot, so its built-in fixed `base_camera` was supplemented in a
test environment with a camera mounted on `panda_hand`.

A centered wrist camera was mostly occluded by the pushing tool. Offsetting it
8 cm from the hand and pointing it along the stick toward the table gave a
usable view of the T.

Generated artifacts include:

- one exact-state replay of a successful expert trajectory from the wrist
  camera:
  `results/videos/ppo_expert_traj0_wrist_camera.mp4`;
- 20 seed-0 policy videos with base and wrist views side by side:
  `results/ppo_spatial_wm_recon512_lownoise/videos/dual_camera_seed0/`.
- contact sheets comparing flat-policy progress and all n=2000 methods:
  `results/ppo_spatial_wm_recon512_lownoise/videos/contact_sheets/`.

The policy videos cover all five dataset sizes, the flat latent policy, and all
three hierarchy horizons. They use evaluation seed 10000 and contain 100 action
steps plus the initial frame, or 101 frames at 20 FPS.

The dual-camera renderer was used as an inspection utility and has not been
added to the public `hcl-poc video` CLI.

## 4. Results

### 4.1 Diagnostic policy results

The following results are from the earlier 50-episode diagnostic phase unless
otherwise stated:

| Method | Trajectories | Success | Final reward | Max reward |
| --- | ---: | ---: | ---: | ---: |
| BC privileged state | 1000 | 0.46 | 0.582 | 0.594 |
| BC spatial DINO, 1-step | 1000 | 0.06 | 0.219 | 0.256 |
| BC spatial DINO, DAgger | 1000 | 0.06 | 0.273 | 0.296 |
| Flat flow, spatial DINO observation | 1000 | 0.04 | 0.206 | 0.255 |
| Flat latent, 512D reconstruction WM | 1000 | 0.08 | 0.214 | 0.255 |
| Hierarchy, noisy low, 0.05 s | 1000 | 0.10 | 0.232 | 0.276 |
| Hierarchy, noisy low, 0.10 s | 1000 | 0.04 | 0.198 | 0.233 |
| Hierarchy, noisy low, 0.25 s | 1000 | 0.02 | 0.174 | 0.219 |

These small evaluations were useful for model selection but are too noisy for
final claims.

### 4.2 Current 500-episode multi-seed results

Values below are mean +/- sample standard deviation across available policy
seeds. Each seed is evaluated for 500 episodes.

| Trajectories | Method | Seeds complete | Success | Final reward | Max reward |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | flat | 3 | 0.002 +/- 0.002 | 0.114 +/- 0.004 | 0.158 +/- 0.005 |
| 50 | hier 0.05 s | 3 | 0.002 +/- 0.002 | 0.115 +/- 0.011 | 0.161 +/- 0.018 |
| 50 | hier 0.10 s | 3 | 0.002 +/- 0.003 | 0.117 +/- 0.003 | 0.150 +/- 0.010 |
| 50 | hier 0.25 s | 3 | 0.001 +/- 0.001 | 0.111 +/- 0.003 | 0.144 +/- 0.008 |
| 100 | flat | 3 | 0.001 +/- 0.001 | 0.117 +/- 0.002 | 0.155 +/- 0.005 |
| 100 | hier 0.05 s | 3 | 0.004 +/- 0.004 | 0.121 +/- 0.003 | 0.163 +/- 0.007 |
| 100 | hier 0.10 s | 3 | 0.001 +/- 0.002 | 0.120 +/- 0.003 | 0.167 +/- 0.009 |
| 100 | hier 0.25 s | 3 | 0.000 +/- 0.000 | 0.119 +/- 0.002 | 0.160 +/- 0.003 |
| 200 | flat | 3 | 0.011 +/- 0.008 | 0.121 +/- 0.009 | 0.171 +/- 0.014 |
| 200 | hier 0.05 s | 3 | 0.006 +/- 0.004 | 0.121 +/- 0.004 | 0.164 +/- 0.006 |
| 200 | hier 0.10 s | 3 | 0.006 +/- 0.007 | 0.123 +/- 0.006 | 0.166 +/- 0.012 |
| 200 | hier 0.25 s | 3 | 0.009 +/- 0.006 | 0.125 +/- 0.004 | 0.173 +/- 0.007 |
| 1000 | flat | 3 | 0.034 +/- 0.009 | 0.175 +/- 0.017 | 0.220 +/- 0.014 |
| 1000 | hier 0.05 s | 3 | 0.040 +/- 0.009 | 0.179 +/- 0.014 | 0.220 +/- 0.010 |
| 1000 | hier 0.10 s | 3 | 0.037 +/- 0.017 | 0.182 +/- 0.022 | 0.221 +/- 0.018 |
| 1000 | hier 0.25 s | 3 | 0.041 +/- 0.011 | 0.178 +/- 0.008 | 0.221 +/- 0.012 |
| 2000 | flat | 2 | 0.050 +/- 0.014 | 0.194 +/- 0.009 | 0.234 +/- 0.008 |
| 2000 | hier 0.05 s | 2 | 0.041 +/- 0.001 | 0.186 +/- 0.005 | 0.226 +/- 0.005 |
| 2000 | hier 0.10 s | 2 | 0.039 +/- 0.001 | 0.187 +/- 0.001 | 0.224 +/- 0.004 |
| 2000 | hier 0.25 s | 2 | 0.040 +/- 0.011 | 0.185 +/- 0.009 | 0.226 +/- 0.011 |

The n=2000 seed-2 run was intentionally stopped before completion. No partial
seed-2 checkpoint or metric is included in the aggregates.

The generated KPI figures are:

- `docs/results/success_vs_trajectories.png`;
- `docs/results/final_reward_vs_trajectories.png`;
- `docs/results/max_reward_vs_trajectories.png`;
- `docs/results/inference_latency_s_vs_trajectories.png`.

### 4.3 Interpretation

The current evidence supports the following conclusions:

1. **The task and action space are learnable.** The privileged PPO teacher
   reaches 86.3% success, and privileged-state BC reaches 46%.
2. **The original latent was defective.** It discarded object pose even though
   spatial DINO retained it.
3. **Reconstruction regularization was necessary.** Increasing latent capacity
   without reconstruction did not improve the probe.
4. **The stronger latent improves control, but absolute success remains low.**
   Final 500-episode results rise with dataset size, reaching approximately
   3-5% at 1000-2000 trajectories.
5. **The hierarchy has not demonstrated a data-efficiency advantage.** Across
   the current final results it is generally comparable to, and sometimes
   worse than, the flat latent policy.
6. **Low-level sampled-subgoal brittleness was real and was fixed offline.**
   Subgoal noise closed the oracle-versus-sampled action-error gap, but did not
   produce a large closed-loop success gain.
7. **The remaining failure is broader than one encoder issue.** Visual
   precision, compounding imitation error, flow sampling, and high-level
   subgoal quality remain plausible contributors.

The proof of concept currently does not support the claim that the hierarchy
requires fewer demonstrations than the flat baseline.

## 5. KPI Definitions

- **Success:** fraction of episodes satisfying ManiSkill's Push-T success
  condition.
- **Final reward:** normalized dense reward on the final transition.
- **Max reward:** maximum normalized dense reward reached during an episode,
  averaged across evaluation episodes.
- **Inference latency:** mean wall-clock policy inference time per action.
- **Pose-probe position MAE:** held-out mean absolute error for T center x/y.
- **Pose-probe yaw MAE:** held-out wrapped angular error in degrees.
- **Hierarchy action MAE:** offline mean absolute error between predicted and
  demonstration action chunks.

Success is the primary task KPI. Final and maximum reward are useful when
success is sparse, but they are not substitutes for completing the task.

## 6. Validation Performed

### Automated checks

Executed on June 18, 2026:

```text
uv run pytest -q
5 passed

uv run ruff check src tests
All checks passed
```

The tests cover durable model and data contracts, including the requirement
that the world model consumes actions.

### Runtime checks

The following runtime paths have been exercised:

- ManiSkill environment creation;
- CPU smoke training and evaluation;
- CUDA PPO training and vectorized evaluation;
- successful PPO trajectory collection;
- HDF5 dataset preparation and loading;
- DINO CLS and spatial feature extraction;
- world-model, flat-flow, high-level, and low-level training;
- all diagnostic BC variants;
- policy evaluation and report generation;
- video encoding and decode checks;
- mounted wrist-camera rendering with the CPU simulation backend.

### Probe validity checks

Representation probes were retrained independently for each frozen encoder.
They used disjoint train/validation frame sets. The overfitting audit compared:

- linear and nonlinear probes;
- training and validation error;
- real and shuffled labels.

The strong spatial-DINO result generalized to validation data, while the
original latent stayed near the mean-pose baseline on both train and
validation data.

## 7. Reproducing the Current Experiment

Install and validate:

```bash
uv sync --python 3.11
uv run hcl-poc doctor
```

Prepare data after the privileged teacher is available:

```bash
uv run hcl-poc data prepare \
  --config configs/pusht_spatial_wm_recon512_lownoise.yaml
```

Train one complete policy set:

```bash
CONFIG=configs/pusht_spatial_wm_recon512_lownoise.yaml
N=1000
SEED=0

uv run hcl-poc train encoder --config "$CONFIG" --n-traj "$N" --seed "$SEED"
uv run hcl-poc train flat --config "$CONFIG" --n-traj "$N" --seed "$SEED"

for H in 0.05 0.10 0.25; do
  uv run hcl-poc train high \
    --config "$CONFIG" --n-traj "$N" --seed "$SEED" --horizon-s "$H"
  uv run hcl-poc train low \
    --config "$CONFIG" --n-traj "$N" --seed "$SEED" --horizon-s "$H"
done
```

Evaluate with the final episode count:

```bash
uv run hcl-poc eval flat \
  --config "$CONFIG" --n-traj "$N" --seed "$SEED" --episodes 500

for H in 0.05 0.10 0.25; do
  uv run hcl-poc eval hier \
    --config "$CONFIG" --n-traj "$N" --seed "$SEED" \
    --horizon-s "$H" --episodes 500
done
```

## 8. Artifact Map

| Artifact | Location |
| --- | --- |
| Current config | `configs/pusht_spatial_wm_recon512_lownoise.yaml` |
| Prepared dataset | `data/prepared/pusht_ppo_dino_spatial_proprio_tcp.h5` |
| Current checkpoints | `artifacts/ppo_spatial_wm_recon512_lownoise/` |
| Current metrics | `results/ppo_spatial_wm_recon512_lownoise/` |
| Representation probes | `results/probes/` |
| Hierarchy diagnostics | `results/diagnostics/` |
| Earlier rollout videos | `results/videos/` |
| Dual-camera seed-0 videos | `results/ppo_spatial_wm_recon512_lownoise/videos/dual_camera_seed0/` |
| Video contact sheets | `docs/results/flat_progress_seed0.png`, `docs/results/n2000_methods_seed0.png` |
| Packaged metrics and videos | `results/ppo_spatial_wm_recon512_lownoise.zip` |
| Long-running command logs | `run_logs/` |

Large datasets, checkpoints, logs, and videos are local experiment artifacts
and are intentionally not committed to Git.

## 9. Known Limitations

- The n=2000 aggregate has two policy seeds rather than three.
- Success remains low enough that even 500 episodes leave nontrivial
  uncertainty.
- The same 2000-trajectory teacher dataset is subsampled for all training-set
  sizes; this controls data nesting but does not test independent collection
  sets.
- Only one task, robot, action space, DINO backbone, and flow architecture have
  been studied.
- The wrist camera was tested for visualization only. Current policies still
  consume the fixed base camera.
- DAgger was tested as a limited diagnostic, not as a full iterative
  data-aggregation study.
- The current result does not establish a hierarchy data-efficiency benefit.
