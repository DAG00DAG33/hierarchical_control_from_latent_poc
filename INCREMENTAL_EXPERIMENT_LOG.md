# Push-T Incremental Experiment Log

This is the running lab notebook for
[`pusht_incremental_experiment_plan.md`](pusht_incremental_experiment_plan.md).
It is updated after every meaningful experiment, including failed runs.

## Logging Rules

Every experiment entry records:

- date and experiment ID;
- Git commit and configuration;
- hypothesis;
- dataset type: `query_dataset` or `causal_dataset`;
- dataset size in trajectories and transitions/state queries;
- exact command;
- training and evaluation seeds;
- success and secondary KPIs;
- gate decision;
- diagnosis and next action.

State-query labels are never treated as causal transitions. Any experiment
using action chunks, future states, or world-model targets must explicitly
identify its causal rollout source.

## Current Status

| Item | Status |
| --- | --- |
| Active phase | Phase 6: latent representation capacity and objective study |
| Gate | Encoder should preserve task-relevant state and support flat latent control before Phase 7 |
| Gate state | Phases 0-5 passed; Phase 6 reopened for smaller-latent and WM-objective ablations |
| Current blocker | None |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Free disk at start | 113 GB |
| Canonical controller | `pd_ee_delta_pos`, 20 Hz |
| Canonical final evaluation | 500 episodes, fixed reset seeds |
| Development gate evaluation | 100 episodes unless a cheaper diagnostic is sufficient |

## Prior Evidence

The previous end-to-end experiment established useful diagnostics but does not
satisfy the new phase gates:

- privileged PPO deterministic evaluation previously measured about 86.3%
  success over 256 episodes;
- the PPO checkpoint records a 92.5% recent training success metric, which is
  not interchangeable with deterministic held-out evaluation;
- the old `bc_state` result reached about 46% success;
- old `bc_state` is not a valid Phase 1 implementation because it uses only
  successful deterministic teacher rollouts and predicts an 8-step action
  chunk;
- visual and hierarchical policies reached only a few percent success;
- the final 512D reconstruction-regularized latent passed the static pose probe
  but was not validated for velocity, contact, inverse dynamics, or the new
  oracle-hierarchy gate.

These results are baselines and debugging evidence only. They will not be used
to skip phases.

## Phase 0 Checklist

| Check | Evidence required | Status |
| --- | --- | --- |
| 0.1 Original PPO evaluation | Deterministic success and reward on canonical seeds | Passed |
| 0.1 Downstream evaluator parity | Same actor through scalar student evaluation path | Passed |
| 0.2 Copied actor | Exact weight equality and success within 1 percentage point | Passed |
| 0.3 Action semantics | Raw, clipped, stored, executed, normalized comparisons | Passed |
| 0.4 Temporal alignment | `s_t, a_t, s_{t+1}` audit and +/-1 action comparison | Passed |
| 0.5 One-state overfit | Nearly zero normalized action error | Passed |
| 0.5 One-trajectory overfit | Nearly zero normalized action error and replay check | Passed |
| 0.5 Ten-trajectory overfit | Training error and closed-loop behavior | Diagnostic complete |

## Experiment Entries

### 2026-06-18 - P0-I00: Repository and resource inventory

- **Git base:** `b03fa2c`
- **Configuration inspected:** `configs/pusht.yaml`
- **Hypothesis:** Existing artifacts are sufficient to begin Phase 0 without
  retraining the PPO teacher.
- **Dataset type:** None.
- **Commands:** `nvidia-smi`, `df -h`, `hcl-poc rl status`, source and artifact
  inventory.
- **Evidence:**
  - CUDA is available.
  - GPU has approximately 15.4 GB free.
  - Disk has 113 GB free.
  - `artifacts/rl_pusht_official/ppo_best.pt` exists.
  - PPO checkpoint observation/action dimensions and exact evaluation behavior
    still need runtime verification.
  - Existing prepared visual HDF5 files do not store full privileged state or
    next state, so they cannot by themselves support the Phase 0 alignment
    audit or later causal privileged experiments.
- **Gate decision:** Phase 0 remains open.
- **Next action:** Implement one canonical Phase 0 command that evaluates the
  PPO actor through both vector and scalar downstream paths, copies actor
  weights into an independent student wrapper, records causal state/action
  transitions, audits alignment, and runs the overfit ladder.

### 2026-06-18 - P0-D01: First Phase 0 debug run

- **Git state:** Uncommitted Phase 0 implementation based on `b03fa2c`.
- **Configuration:** `configs/pusht_incremental.yaml`, 50 evaluation episodes.
- **Hypothesis:** The existing scalar downstream evaluator reproduces the PPO
  vector evaluator.
- **Dataset type:** New `causal_dataset` audit with 10 successful trajectories,
  489 transitions.
- **Command:** `uv run hcl-poc incremental phase0 --config
  configs/pusht_incremental.yaml --episodes 50 --force`
- **Results:**
  - copied actor output max error: exactly 0;
  - copied actor and teacher scalar success: both 46%;
  - stored versus queried clipped action MAE at shift 0:
    `6.86e-8`;
  - shift -1/+1 MAE: both approximately `0.173`;
  - exact simulator transition replay max state error: 0;
  - raw teacher actions outside bounds: 21.9% of audit transitions;
  - clipped versus executed action MAE: 0;
  - one-state normalized MAE: `9.76e-4`;
  - one-trajectory normalized MAE: `9.88e-4`;
  - one-trajectory closed-loop replay success: 0/1;
  - ten-trajectory training-initialization success: 2/10.
- **Gate decision:** Failed. The initial implementation incorrectly marked the
  gate as passed because it compared only teacher/student equality and omitted
  the known-teacher-performance requirement.
- **Diagnosis:** State/action alignment and causal replay are correct, but the
  scalar evaluator used `physx_cpu`, while PPO was trained and evaluated with
  `physx_cuda`.
- **Next action:** Compare the same scalar actor on CPU and CUDA, correct the
  gate, and make CUDA the canonical downstream backend if the discrepancy is
  confirmed.

### 2026-06-18 - P0-D02: Simulator-backend isolation

- **Hypothesis:** The teacher-success collapse is caused by CPU versus CUDA
  simulation dynamics rather than reset seeds or policy wrapping.
- **Dataset type:** None.
- **Commands:** Scalar teacher evaluation for 100 episodes on CPU and CUDA,
  with seed starts 10000 and 50000; vector PPO evaluation for 500 episodes.
- **Results:**
  - canonical vector CUDA evaluation: 83.8% success over 500 episodes;
  - scalar CUDA evaluation: 83% success over 100 episodes;
  - scalar CPU evaluation: 46% over seeds starting at 10000;
  - scalar CPU evaluation: 40% over seeds starting at 50000.
- **Gate decision:** CPU downstream evaluation is invalid for this experiment.
- **Diagnosis:** The same bit-exact actor loses roughly 40 percentage points
  when the simulator backend changes to CPU. Reset seed range does not explain
  the discrepancy.
- **Next action:** Use explicit CUDA simulation for all canonical policy
  evaluation and causal collection. Require downstream teacher success of at
  least 80%, vector/scalar agreement within 5 percentage points, and copied
  actor agreement within 1 percentage point.

### Plan amendment - Phase 6

Phase 6 now includes a reconstruction-only autoencoder with zero world-model
prediction loss. This will isolate whether the action-conditioned temporal
objective adds useful control-state structure beyond current-observation
reconstruction.

### Evaluation-budget amendment

Use 50-100 episodes for phase gates, debugging, and model selection. Reserve
500-episode evaluations for final results and claims in Phase 12. A
500-episode Phase 0 run was stopped before completion and replaced with a
100-episode gate run.

### 2026-06-18 - P0-G01: Final Phase 0 gate run

- **Git base:** `4c27722`
- **Configuration:** `configs/pusht_incremental.yaml`
- **Dataset type:** Fresh CUDA `causal_dataset`, 10 successful trajectories,
  515 transitions.
- **Command:** `uv run hcl-poc incremental phase0 --config
  configs/pusht_incremental.yaml --episodes 100 --force`
- **Evaluation seeds:** 10000-10099.
- **Results:**
  - canonical vector CUDA teacher: 81% success;
  - downstream scalar CUDA teacher: 83% success;
  - independent copied actor: 83% success;
  - copied actor output max absolute error: exactly 0;
  - vector/scalar success gap: 2 percentage points;
  - copied-teacher success gap: 0 percentage points;
  - action alignment shift 0 MAE: `6.87e-8`;
  - action alignment shift -1/+1 MAE: approximately `0.199`;
  - causal simulator transition replay max state error: exactly 0;
  - raw teacher actions outside action bounds: 19.0%;
  - clipped-to-executed action MAE: exactly 0;
  - normalization round-trip max error: `1.19e-7`;
  - one-state fit action MAE: `1.76e-5`;
  - one-trajectory fit action MAE: `2.61e-6`;
  - ten-trajectory fit action MAE: `5.06e-4`.
- **Closed-loop overfit diagnostic:**
  - one-trajectory student: 0/1 success despite micro-scale supervised error;
  - ten-trajectory student: 7/10 success on training initializations.
- **Interpretation:** Push-T contact behavior is sensitive enough that tiny
  action errors can leave the exact training-state manifold. This is not an
  alignment or replay bug: the copied actor and exact stored-action replay
  both preserve behavior. Phase 1 must use broader all-state queries, and
  Phase 2 DAgger remains necessary even if held-out action MAE is small.
- **Gate decision:** Passed. All explicit Phase 0 gate checks pass.
- **Next action:** Build separate all-state and successful-only
  `query_dataset` files using deterministic teacher labels, then train
  same-architecture one-step privileged BC.

### 2026-06-18 - P1-D01: All-state clipped-label BC

- **Dataset type:** `query_dataset`.
- **Data:** 2,600 teacher episodes, 127,517 state queries; 2,190 successful
  episodes; 41,000 queries from failed episodes.
- **Training subset:** First 2,000 episodes, 97,546 queries.
- **Validation:** Disjoint 200 episodes, 9,966 queries.
- **Model:** Teacher-sized 3x256 Tanh MLP, raw privileged state input.
- **Labels:** Clipped deterministic teacher actions.
- **Training:** 100 epochs, Adam `3e-4`.
- **Results:**
  - held-out action MAE: `0.0471`;
  - held-out action RMSE: `0.0708`;
  - per-dimension correlations: `0.980`, `0.990`, `0.975`;
  - closed-loop success: 9/100;
  - final/max normalized reward: `0.291` / `0.312`.
- **Gate decision:** Failed.
- **Diagnosis:** High action correlation is insufficient; absolute errors are
  still large for contact control. Validation loss is still decreasing at the
  final epoch.

### 2026-06-18 - P1-D02: All-state raw-label BC

- **Change:** Replaced clipped labels with smooth raw deterministic actor
  outputs; clipping remained in the environment execution path.
- **Results:**
  - held-out action MAE: `0.0503`;
  - closed-loop success: 10/100;
  - final/max normalized reward: `0.311` / `0.330`.
- **Gate decision:** Failed.
- **Diagnosis:** Clipping-label nonsmoothness is not the primary bottleneck.
  Optimization remains underfit after 100 epochs.
- **Next action:** Normalize privileged inputs, raise learning rate to `1e-3`,
  and train for 300 epochs before changing architecture or data.

### 2026-06-18 - P1-G01: Normalized all-state privileged BC

- **Dataset type:** `query_dataset`.
- **Data:** First 2,000 teacher episodes, 97,546 state queries, including
  states from both successful and failed teacher episodes.
- **Model:** Teacher-sized 3x256 Tanh MLP.
- **Input:** Standardized 31D privileged state.
- **Labels:** Raw deterministic PPO actor output; actions clipped only when
  executed in the environment.
- **Training:** 300 epochs, Adam `1e-3`, batch size 4096.
- **Results:**
  - best validation MSE: `0.00118`;
  - held-out action MAE/RMSE: `0.0204` / `0.0344`;
  - per-dimension correlations: `0.997`, `0.998`, `0.994`;
  - closed-loop success: 78/100;
  - final/max normalized reward: `0.848` / `0.849`;
  - teacher success on the same downstream protocol: 83/100.
- **Gate decision:** Passed. This exceeds the 70% minimum and is within five
  percentage points of the teacher on the development evaluation.
- **Key finding:** Input normalization and sufficient optimization changed
  success from 10% to 78% without changing the architecture or dataset.

### 2026-06-18 - P1-A01: Successful-only state queries

- **Dataset type:** `query_dataset`.
- **Data:** 1,900 successful episodes, 75,022 state queries; disjoint 200
  successful validation episodes.
- **Recipe:** Same normalized raw-label model as P1-G01.
- **Results:**
  - held-out action MAE: `0.0208`;
  - closed-loop success: 72/100;
  - final/max normalized reward: `0.807` / `0.810`.
- **Interpretation:** Successful-only and all-state validation MAE are nearly
  identical, but all-state training is six percentage points better in closed
  loop. Failed teacher episodes add behaviorally useful coverage not captured
  by aggregate one-step MAE.
- **Phase 1 conclusion:** Use all teacher states and raw deterministic labels
  as the privileged BC baseline. Proceed to learner-visited DAgger queries.

### 2026-06-18 - P2-D01: DAgger iteration 1, random restart

- **Dataset type:** 50,000 new learner-visited `query_dataset` samples.
- **Learner rollout diagnostic:** 882 completed vector episodes, 85.9%
  success on the collector distribution; teacher-versus-learner action MAE
  `0.0232`.
- **Training:** Aggregated 147,546 queries, but restarted the student from
  random weights.
- **Result:** 75/100 success; held-out base action MAE `0.0248`.
- **Gate decision:** Failed.
- **Diagnosis:** Random-restart aggregation forgot part of the Phase 1
  function and worsened the original held-out distribution.

### 2026-06-18 - P2-D02: DAgger fine-tuning without forgetting

- **Change:** Initialize each iteration from the preceding policy, preserve its
  input normalizer, lower learning rate to `1e-4`, and keep the pre-update
  checkpoint unless base validation improves.
- **Iteration 1 result:** 77% success, held-out action MAE `0.0191`.
- **Iteration 2 result:** 77% success, held-out action MAE `0.0180`.
- **Gate decision:** Still below the 80% target, but learner-state teacher
  disagreement continued decreasing.

### 2026-06-18 - P2-G01: DAgger iteration 3

- **Dataset type:** Three cumulative 50,000-query DAgger sets plus 97,546 base
  queries; 247,546 total training queries.
- **Learner-state teacher disagreement by iteration:** `0.0232`, `0.0205`,
  `0.0189` action MAE.
- **Results:**
  - held-out base action MAE: `0.0170`;
  - closed-loop success: 82/100;
  - final/max normalized reward: `0.877` / `0.878`.
- **Policy gate decision:** Passed.

### 2026-06-18 - P2-G02: Controlled perturbation recovery

- **Perturbations:** T-block x/y Gaussian noise with 1 cm standard deviation
  and yaw noise with 5 degree standard deviation, each clipped at 2 standard
  deviations.
- **Samples:** 128 perturbed simulator states drawn from valid causal teacher
  trajectories.
- **Results:**
  - deterministic teacher recovery: 106/128, 82.8%;
  - learner recovery over all perturbations: 75.0%;
  - learner recovery conditional on teacher recoverability: 87/106, 82.1%.
- **Causal output:** 106 fresh teacher recovery trajectories in
  `data/incremental/phase2_recovery/iteration_03_seed0.h5`.
- **Causal audit:** Every stored episode has one more simulator state than
  action, and each action was actually executed to produce the stored next
  state.
- **Recovery gate decision:** Passed.
- **Phase 2 conclusion:** Freeze iteration 3 as the privileged deterministic
  baseline and proceed to one-step privileged flow matching.

### 2026-06-18 - P3-D01: Raw-action one-step privileged flow

- **Dataset type:** Aggregated `query_dataset` from Phase 1 plus three DAgger
  iterations, 247,546 training queries.
- **Model:** Conditional flow matching, one-step action target, 3D action
  sample, 31D privileged state condition, 3x256 MLP.
- **Labels:** Raw deterministic PPO actor outputs.
- **Evaluation:** 100 episodes, fixed reset seeds 10000-10099.
- **Result:** 67% success with one sample; 75% success when averaging eight
  flow samples.
- **Diagnostics:**
  - held-out action MAE: `0.0329`;
  - sample mean action MAE on 256 fixed states: `0.0247`;
  - single-sample action MAE on the same states: `0.0315`;
  - sample action standard deviation mean: `0.0260`.
- **Gate decision:** Failed. Sampling noise and endpoint error were too large
  for contact control.

### 2026-06-18 - P3-D02: Clipped-action targets and deterministic zero-noise evaluation

- **Change:** Train on clipped deterministic teacher actions, because these
  are the actions actually executed by the simulator.
- **Change:** Evaluate by integrating from zero noise instead of sampling or
  averaging stochastic actions.
- **Result:** 75% success.
- **Diagnostics:** Clipped labels reduce out-of-bounds samples but do not by
  themselves close the gate.
- **Gate decision:** Failed.

### 2026-06-18 - P3-D03: Endpoint-consistency loss

- **Change:** Added a deterministic endpoint-consistency loss by integrating
  the flow from zero noise on a 512-sample sub-batch and penalizing distance to
  the teacher action.
- **Attempt:** Weight `1.0` did not help; success dropped to 68%.
- **Diagnosis:** Checkpoint selection still used stochastic sample averaging
  while evaluation used zero-noise integration. The auxiliary loss was also
  too weak relative to the flow-matching loss.

### 2026-06-18 - P3-G01: Consistency-trained clipped privileged flow

- **Final Phase 3 recipe:**
  - clipped deterministic teacher action targets;
  - conditional flow matching plus endpoint-consistency loss;
  - endpoint-consistency weight `20.0`;
  - 4 integration steps for the differentiable training endpoint loss;
  - 24 integration steps for evaluation;
  - zero-noise deterministic evaluation;
  - checkpoint selection by zero-noise validation action MAE.
- **Results:**
  - closed-loop success: 79/100;
  - Phase 2 BC reference success: 82/100;
  - gate margin: flow is 3 percentage points below BC, within the 5 point gate;
  - final/max normalized reward: `0.855` / `0.855`;
  - held-out zero-noise action MAE/RMSE: `0.0254` / `0.0379`;
  - fixed-state random-sample action std mean: `0.0273`;
  - fixed-state sample-mean action MAE: `0.0196`;
  - fixed-state single-sample action MAE: `0.0297`.
- **Gate decision:** Passed.
- **Interpretation:** Pure stochastic sampling remains noisier than
  deterministic BC. For this mostly unimodal teacher, the useful flow policy is
  the zero-noise deterministic flow endpoint. This is acceptable for the Phase
  3 gate because the flow implementation can now solve the privileged
  one-step task within the required margin, but later visual/hierarchical flow
  phases should track sampling variance explicitly.
- **Next action:** Start Phase 4 with temporal visual deterministic BC using
  the successful PPO causal visual dataset as the first supervised source.

### 2026-06-18 - P4-I01: Temporal visual deterministic BC implementation

- **Observation:** Base-camera RGB encoded with frozen spatial DINOv2-small,
  concatenated with proprioception from the `rgb+state` observation and a
  previous-action history.
- **Dataset:** Existing successful PPO causal dataset
  `data/prepared/pusht_ppo_dino_spatial_proprio_tcp.h5`.
- **Training split:** 1,800 successful teacher episodes for training and 200
  held-out successful teacher episodes for validation.
- **Model path:** Added Phase 4 commands:
  - `incremental phase4-train`;
  - `incremental phase4-eval`;
  - `incremental phase4-probe`.
- **Evaluator:** Vectorized `rgb+state` environment wrapped with
  `ManiSkillVectorEnv`, batched DINO encoding, deterministic closed-loop
  evaluation on 100 episodes.
- **Implementation note:** The first visual evaluator attempt used an
  unwrapped vector env and did not emit `final_info`; this was fixed before
  the real runs.

### 2026-06-18 - P4-G01: Offline visual BC gate

- **Concat, L=1 result:**
  - closed-loop success: 65/100;
  - final/max normalized reward: `0.630` / `0.753`;
  - held-out action MAE/RMSE: `0.0382` / `0.0593`;
  - action correlation per dimension: `0.989`, `0.994`, `0.985`.
- **Concat, L=2 result:**
  - closed-loop success: 56/100;
  - final/max normalized reward: `0.569` / `0.686`;
  - held-out action MAE/RMSE: `0.0420` / `0.0654`.
- **GRU, L=2 result:**
  - closed-loop success: 55/100;
  - final/max normalized reward: `0.584` / `0.685`;
  - held-out action MAE/RMSE: `0.0404` / `0.0619`.
- **History sweep decision:** Full-budget `L=4` was started but stopped after
  one epoch because the epoch time was about 45 seconds and both full-budget
  `L=2` models underperformed `L=1`. The current best policy is the
  single-frame spatial-DINO concat model. Longer histories remain a later
  optimization, not a blocker for the Phase 4 gate.
- **Gate decision:** Passed. The best visual deterministic BC result is 65%,
  above the 50% minimum useful gate and within the target 60-70% band.

### 2026-06-18 - P4-G02: Visual-history probe

- **Probe input:** `L=2` concat visual-history representation, using the same
  DINO/proprio/action-history normalization as the Phase 4 policy.
- **Samples:** 8,000 teacher-rollout states, 6,400 train and 1,600 validation.
- **Label fix:** The first probe used the wrong TCP state indices and then
  compared each state against itself for finite-difference velocities. The
  corrected state layout is:
  - TCP pose starts at state index 14;
  - goal position starts at state index 21;
  - object pose starts at state index 24.
- **Corrected continuous probe MAE versus mean baseline:**
  - object x/y: `0.0032` / `0.0040` m vs `0.0243` / `0.0246` m;
  - object yaw: `0.274` rad vs `2.391` rad;
  - object vx/vy: `0.0270` / `0.0242` m/s vs `0.0536` / `0.0614` m/s;
  - object yaw rate: `0.275` rad/s vs `0.586` rad/s;
  - TCP x/y: `0.0055` / `0.0064` m vs `0.0498` / `0.0700` m;
  - TCP vx/vy: `0.0564` / `0.0419` m/s vs `0.2055` / `0.2305` m/s.
- **Corrected contact probe:** 95.9% accuracy vs 65.0% majority baseline,
  AUROC `0.994`.
- **Probe gate support:** Pose, velocity, and contact diagnostics are all
  meaningfully above baseline.
- **Phase 4 conclusion:** Freeze `concat_h1/seed0` as the current visual
  deterministic BC baseline and proceed to visual flow matching.

### 2026-06-18 - P5-I01: Visual one-step flow implementation

- **Input representation:** Same as best Phase 4 policy, `concat_h1`: current
  spatial-DINO/proprio frame plus previous-action slot.
- **Dataset:** Same 1,800/200 successful PPO causal split as Phase 4.
- **Training:** Conditional flow matching over one-step 3D actions with the
  Phase 3 deterministic endpoint recipe:
  - clipped executed teacher actions;
  - endpoint-consistency weight `20.0`;
  - 4-step differentiable endpoint loss during training;
  - 24-step zero-noise deterministic endpoint evaluation.
- **Reference:** Phase 4 `concat_h1/seed0` visual BC, 65/100 success.

### 2026-06-18 - P5-G01: Visual flow gate

- **Closed-loop result:** 66/100 success.
- **Gate threshold:** Visual BC minus 5 percentage points = 60%.
- **Final/max normalized reward:** `0.631` / `0.758`.
- **Held-out zero-noise action MAE/RMSE:** `0.0375` / `0.0566`.
- **Action correlation per dimension:** `0.990`, `0.994`, `0.986`.
- **Comparison:** Flow is 1 percentage point above the deterministic visual BC
  reference on the same 100 evaluation seeds.
- **Gate decision:** Passed.
- **Phase 5 conclusion:** Freeze `phase5/concat_h1/seed0` as the current flat
  visual flow baseline and proceed to Phase 6 representation validation. The
  Phase 6 plan already includes the requested reconstruction-only autoencoder
  ablation with no world-model prediction loss.

### 2026-06-18 - P6-I01: Representation validation implementation

- **Dataset:** Same successful PPO causal visual dataset as Phases 4-5,
  `data/prepared/pusht_ppo_dino_spatial_proprio_tcp.h5`.
- **Representation inputs:** Spatial DINOv2-small base-camera RGB features plus
  the first 21 proprioceptive state dimensions.
- **Implemented Phase 6 commands:**
  - `incremental phase6-train`;
  - `incremental phase6-probe`;
  - `incremental phase6-control-{train,eval}`;
  - `incremental phase6-flow-{train,eval}`;
  - `incremental phase6-dagger-{collect,train,eval}`.
- **Representation variants:** world-model plus reconstruction,
  world-model without reconstruction, and reconstruction-only autoencoder.
  The reconstruction-only autoencoder has zero world-model prediction loss, as
  requested in the Phase 6 plan.
- **Probe labels:** Object pose/velocity, TCP pose/velocity, contact, reward,
  inverse dynamics, and forward next-label prediction. Yaw probes use
  sin/cos targets and report angular MAE to avoid wrap artifacts.

### 2026-06-18 - P6-D01: Initial 512D autoencoder probe passed but control failed

- **Initial representation:** `ae_recon_z512`, hidden width 512, unweighted
  full-observation reconstruction.
- **Probe result after yaw fix:** Passed all static probe gates:
  object x/y `0.0028` / `0.0031` m, yaw `0.052` rad, contact AUROC `0.990`,
  inverse-dynamics action MAE `0.0577` vs mean baseline `0.2107`.
- **Control result:** Latent deterministic BC reached 35/100 success after
  adding previous-action conditioning. Latent zero-noise flow reached 42/100.
  Both missed the 80% of direct visual-flow target, where direct visual flow is
  66/100.
- **Diagnosis:** The latent probe was too weak as a control diagnostic. Feeding
  decoded AE reconstructions into the already-successful Phase 4 visual BC gave
  action MAE `0.0677`, compared with `0.0382` on true visual inputs. The
  reconstruction loss was dominated by thousands of DINO dimensions and
  underweighted the 21D proprio tail.

### 2026-06-18 - P6-D02: Capacity-only change was insufficient

- **Experiment:** 1024D reconstruction-only autoencoder with the original
  hidden width.
- **Result:** Latent zero-noise flow reached 31/100 success and held-out MAE
  `0.0512`, worse than the 512D version.
- **Diagnosis:** Increasing latent dimensionality alone did not fix the
  information bottleneck because the encoder/decoder hidden width and
  unbalanced reconstruction objective still limited action-relevant
  reconstruction.

### 2026-06-18 - P6-G01: Balanced proprio reconstruction fixes the representation

- **Final representation:** `ae_recon_z512`, hidden width 1024, reconstruction
  loss computed as DINO-feature MSE plus proprio-tail MSE over the final 21
  input dimensions.
- **Observation:** This is still pure observation reconstruction. It does not
  add a pose, reward, or action loss to the encoder.
- **Reconstruction diagnostic:** Phase 4 visual BC on decoded inputs improved
  from `0.0677` action MAE to `0.0410`; true inputs remain `0.0382`.
- **Final probe result:** Passed all Phase 6 probe gates:
  - object x/y MAE: `0.0025` / `0.0027` m;
  - object yaw MAE: `0.0488` rad;
  - object vx/vy/yaw-rate MAE: `0.0155` / `0.0177` m/s / `0.164` rad/s;
  - TCP x/y MAE: `0.0030` / `0.0037` m;
  - TCP vx/vy MAE: `0.0254` / `0.0322` m/s;
  - contact AUROC: `0.994`;
  - reward MAE: `0.0218` vs mean baseline `0.3479`;
  - inverse-dynamics action MAE: `0.0183` vs mean baseline `0.2107`.

### 2026-06-18 - P6-G02: Latent flat-control gate with DAgger

- **Before DAgger:** Balanced latent BC reached 44/100 success with held-out
  action MAE `0.0322`; balanced latent flow reached 42/100 with held-out MAE
  `0.0381`. This showed that one-step held-out action MAE was no longer the
  limiting diagnostic; closed-loop distribution shift remained.
- **DAgger collection:** 200 latent-policy episodes, state-query semantics only:
  visited RGB/state observations were labeled by the privileged PPO teacher and
  used only for action-head imitation, not for world-model or future-state
  targets.
- **DAgger training:** Fixed the balanced 512D encoder and trained a latent
  deterministic BC head on causal teacher data plus relabeled visited states,
  with query samples repeated 4x.
- **Closed-loop result:** 59/100 success.
- **Reference:** Direct visual flow is 66/100 on the same 100 evaluation seeds.
- **Gate:** Passed the 80% control target (`59/66 = 89%`). It does not pass the
  stricter 90% target by a small margin.
- **Final/max normalized reward:** `0.637` / `0.700`.
- **Held-out action MAE/RMSE:** `0.0355` / `0.0619`.
- **Phase 6 conclusion at this point:** The learned 512D latent preserved the
  task-relevant state variables and could support a flat latent controller that
  retained most of the direct-observation visual-flow performance.
- **Later amendment:** Phase 6 was reopened before Phase 7 to study whether
  512D is unnecessarily large and whether the world-model objective helps the
  encoder.

### 2026-06-18 - P6-I02: Force flag propagation for representation probes

- **Issue:** `phase6-probe --force` retrained probe heads but still reused an
  existing encoder checkpoint because the force flag did not propagate into
  `train_phase6_representation`.
- **Impact:** Objective-ablation probes could silently compare stale encoders.
- **Fix:** Pass `force` through `_phase6_representations` into
  `train_phase6_representation`.
- **Validation:** Re-ran the affected WM+reconstruction 512D probe after the
  fix. The fresh result is used in the ablation table below.

### 2026-06-18 - P6-S01: Reconstruction-only latent capacity sweep

- **Hypothesis:** 512D may be larger than necessary; representation probes may
  pass at much smaller dimensions, but closed-loop control may reveal the true
  information threshold.
- **Variant:** `ae_recon`, hidden width 1024, balanced DINO/proprio
  reconstruction, zero world-model prediction loss.
- **Dataset type:** Successful PPO `causal_dataset`, 1,800 train episodes and
  200 validation episodes for encoder/control; separate 12,000-sample causal
  probe dataset.
- **Commands:** `phase6-probe --variant ae_recon --latent-dim {256,192,128,64,32,16}
  --force`, `phase6-control-eval --variant ae_recon --latent-dim
  {256,192,128,64,32,16} --force`, plus `phase6-dagger-eval` for 128D, 192D,
  and 256D.
- **Probe results:**

| dim | obj x/y MAE m | yaw MAE rad | inv action MAE | reward MAE | contact AUROC |
| --- | --- | --- | --- | --- | --- |
| 512 | `0.0025/0.0027` | `0.0488` | `0.0183` | `0.0218` | `0.994` |
| 256 | `0.0027/0.0028` | `0.0512` | `0.0179` | `0.0243` | `0.994` |
| 192 | `0.0026/0.0029` | `0.0524` | `0.0184` | `0.0250` | `0.994` |
| 128 | `0.0027/0.0029` | `0.0506` | `0.0192` | `0.0268` | `0.994` |
| 64 | `0.0028/0.0031` | `0.0542` | `0.0228` | `0.0355` | `0.995` |
| 32 | `0.0033/0.0035` | `0.0600` | `0.0286` | `0.0555` | `0.993` |
| 16 | `0.0040/0.0041` | `0.0843` | `0.0342` | `0.0703` | `0.989` |

- **Control results:**

| dim | BC success | BC max reward | BC action MAE | DAgger success | DAgger max reward | DAgger action MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | `0.44` | `0.589` | `0.0322` | `0.59` | `0.700` | `0.0355` |
| 256 | `0.53` | `0.651` | `0.0325` | `0.60` | `0.710` | `0.0367` |
| 192 | `0.50` | `0.625` | `0.0324` | `0.50` | `0.634` | `0.0377` |
| 128 | `0.37` | `0.524` | `0.0332` | `0.50` | `0.628` | `0.0393` |
| 64 | `0.27` | `0.458` | `0.0357` | - | - | - |
| 32 | `0.28` | `0.453` | `0.0386` | - | - | - |
| 16 | `0.12` | `0.306` | `0.0422` | - | - | - |

- **Interpretation:** Static probes remain deceptively strong down to 64D and
  formally pass even at 16D, but control-relevant information starts degrading
  below 256D. The 192D and 128D encoders both have good pose/inverse probes,
  yet one DAgger iteration reaches only 50/100 for each. The 256D encoder is
  the smallest tested latent that matches or slightly exceeds the 512D
  closed-loop result, reaching 60/100 after DAgger against the 66/100 direct
  visual-flow reference.
- **Decision:** Use `ae_recon_z256` as the current Phase 6 default. Keep 512D
  as a capacity reference, not the main representation.

### 2026-06-18 - P6-S02: World-model objective ablation

- **Hypothesis:** The action-conditioned world-model loss may improve latent
  dynamics for hierarchical control, or it may hurt current-state information
  needed by the low-level controller.
- **Variants:**
  - `wm_recon`: action-conditioned multi-horizon world-model loss plus balanced
    reconstruction;
  - `wm_norecon`: action-conditioned world-model loss only, no reconstruction.
- **Important separation:** This is the encoder-training world model, which
  takes actions as input. It is not the later hierarchical high-level model,
  which should map current latent state to future latent state without actions.
- **Probe results:**

| variant | dim | obj x/y MAE m | yaw MAE rad | inv action MAE | reward MAE | contact AUROC | gate support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ae_recon` | 512 | `0.0025/0.0027` | `0.0488` | `0.0183` | `0.0218` | `0.994` | pass |
| `ae_recon` | 128 | `0.0027/0.0029` | `0.0506` | `0.0192` | `0.0268` | `0.994` | pass |
| `wm_recon` | 512 | `0.0055/0.0058` | `0.1052` | `0.0503` | `0.1306` | `0.982` | pass |
| `wm_norecon` | 512 | `0.0216/0.0185` | `0.3254` | `0.1716` | `0.3015` | `0.870` | fail |
| `wm_recon` | 128 | `0.0075/0.0072` | `0.1430` | `0.0770` | `0.1809` | `0.971` | pass |
| `wm_norecon` | 128 | `0.0193/0.0210` | `0.3330` | `0.1747` | `0.3090` | `0.764` | fail |

- **Interpretation:** WM-only is not enough; it fails core pose/yaw/velocity
  support at both 128D and 512D. Adding reconstruction makes the WM variant
  usable by the formal gates, but it is consistently worse than
  reconstruction-only AE on pose, yaw, inverse dynamics, and reward. Under the
  current recipe, the action-conditioned temporal objective does not help the
  encoder and appears to trade away current-state detail.
- **Decision:** Do not move to Phase 7 with a WM-trained encoder. Continue
  with reconstruction-only `ae_recon_z256` unless a later Phase 6 diagnostic
  finds a concrete reason to revisit the WM objective.
