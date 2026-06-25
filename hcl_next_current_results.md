# HCL Next Current Results

This is the current short-form result summary for the experiments in
`hcl_next_experiment_log.md`. It records the latest conclusion after the
effect32/reachability RL validation, not just the best intermediate run.

## Main Answer

The real-compatible effect32 + learned reachability path produced small
positive-looking runs, but final-style validation shows it is not yet a robust
RL improvement over the frozen hierarchy.

The best single observed checkpoint remains:

```text
artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt
```

However, after more evaluation windows, the honest conclusion is:

```text
frozen hierarchy:        0.656 success over 3500 matched episodes
ungated R3 D_phi update: 0.646 success over 3500 matched episodes
residual-gated R3:       0.657 success over 3500 matched episodes
```

So the residual gate makes the R3 update mostly neutral instead of harmful, but
it does not establish a strong deployment improvement.

## What Worked

### Effect32 FiLM has real goal dependence

Matched supervised evaluations on `seed_start=3500000`:

| goal source | episodes | success | max reward |
| --- | ---: | ---: | ---: |
| learned | 200 | 0.645 | 0.742 |
| oracle | 200 | 0.645 | 0.746 |
| shuffled | 200 | 0.280 | 0.460 |

The shuffled collapse shows this interface is not simply ignoring goals in
closed loop.

### Short R3 D_phi update can find positive windows

The original 500-episode fresh check looked promising:

| policy | seed start | episodes | success | max reward |
| --- | ---: | ---: | ---: | ---: |
| frozen effect32_film | 3500000 | 500 | 0.634 | 0.738 |
| R3 terminal-only 40k bc10 | 3500000 | 500 | 0.684 | 0.773 |

This is still the best observed real-compatible RL checkpoint, but it did not
remain stable under broader validation.

### Residual-L2 gate reduces some damage

A simple eval-time gate executes the frozen base action whenever the tuned
action differs from the base action by more than `0.00121` L2. It improved the
five-bank aggregate:

| policy | five 500-episode windows | mean success | mean max reward |
| --- | --- | ---: | ---: |
| frozen | 3500000, 3600000, 3700000, 3800000, 3900000 | 0.651 | 0.751 |
| ungated R3 | same | 0.651 | 0.751 |
| residual-gated R3 | same | 0.663 | 0.758 |

But the final-style 1000-episode window was negative:

| policy | seed start | episodes | success | max reward |
| --- | ---: | ---: | ---: | ---: |
| frozen | 4000000 | 1000 | 0.667 | 0.761 |
| ungated R3 | 4000000 | 1000 | 0.632 | 0.733 |
| residual-gated R3 | 4000000 | 1000 | 0.643 | 0.744 |

## What Failed

### Longer R3 training over-optimizes the local proxy

The 200k R3 continuation improved train terminal `D_phi` distance but hurt task
success compared with the 40k checkpoint:

| policy | success | max reward | raw local reduction |
| --- | ---: | ---: | ---: |
| frozen | 0.634 | 0.738 | 0.397 |
| R3 40k bc10 | 0.684 | 0.773 | 0.410 |
| R3 200k bc10 | 0.656 | 0.753 | 0.418 |

This means training terminal `D_phi` distance is not a reliable checkpoint
selector for full-task success.

### Dense D_phi progress did not help

Adding dense `D_phi` progress to the terminal-only R3 recipe weakened the
500-episode result:

| policy | success | max reward |
| --- | ---: | ---: |
| R3 terminal-only 40k bc10 | 0.684 | 0.773 |
| R3 terminal+progress 40k bc10 | 0.662 | 0.754 |

Dense learned-metric progress appears to reward local metric artifacts that are
not consistently task-useful.

### PPO seed variation is high

Three PPO seeds for the same 40k R3 recipe gave:

| PPO seed offset | success on seed_start=3500000 |
| ---: | ---: |
| 0 | 0.684 |
| 1 | 0.662 |
| 2 | 0.610 |

Action averaging across these checkpoints also failed:

```text
R3 action ensemble two-bank mean: 0.634
frozen two-bank mean:             0.648
```

The issue is not simple high-frequency action noise.

## Current Interpretation

The proof of concept has now shown something useful but narrower than the
original hope:

1. A real-compatible effect latent plus learned reachability distance can
   produce local RL updates that sometimes improve task success.
2. The improvements are small and unstable across PPO seeds and evaluation
   windows.
3. The current local proxy (`D_phi` terminal distance) is not aligned enough
   with full-task success for long training or reliable checkpoint selection.
4. Simple scalar gating can reduce harm, but not enough to establish a robust
   improvement over frozen imitation.

### Initial state features are too weak for a one-shot gate

I added pre-decision eval fields for state/goal distance, base action norm,
previous action norm, replan rate, and first-step versions of those signals.
A fresh matched 500-episode diagnostic at `seed_start=4100000` was negative for
R3:

| policy | success |
| --- | ---: |
| frozen | 0.656 |
| R3 terminal-only 40k bc10 | 0.618 |

Among discordant episodes, the clean initial features only weakly separated
R3 wins from regressions:

| feature | oriented AUC |
| --- | ---: |
| initial raw distance | 0.601 |
| initial base action L2 | 0.544 |
| initial selected distance | 0.526 |

The stronger diagnostic was mean selected distance along the R3 trajectory
(`0.727` oriented AUC), which suggests the next selector should be step-level or
segment-level and observe current reachability-distance trouble during rollout.
A one-shot episode gate from initial state alone is probably too weak.

### Step-level distance gating also did not validate

I added `--selected-distance-gate-max`, which falls back to the frozen base
action whenever the current selected distance is above the threshold. A sweep on
`seed_start=4100000` found one positive setting:

| policy | success | fallback rate |
| --- | ---: | ---: |
| frozen | 0.656 | 0.000 |
| R3 ungated | 0.618 | 0.000 |
| selected-distance gate 0.85 | 0.664 | 0.175 |

But a fresh 500-episode validation at `seed_start=4200000` did not hold:

| policy | success | fallback rate |
| --- | ---: | ---: |
| frozen | 0.656 | 0.000 |
| R3 ungated | 0.658 | 0.000 |
| selected-distance gate 0.85 | 0.640 | 0.175 |

Combined over these two 500-episode windows:

| policy | success | improvements | regressions | net |
| --- | ---: | ---: | ---: | ---: |
| R3 ungated | 0.638 | 209 | 227 | -18 |
| selected-distance gate 0.85 | 0.652 | 218 | 222 | -4 |

So current-distance gating reduces harm, but like residual-L2 gating it remains
mostly neutral and does not establish a robust improvement over frozen.

### Offline paired selectors were over-optimistic

I added a runnable initial linear selector that chooses frozen or R3 at the
first step of each episode from three features: initial selected distance,
initial raw distance, and initial base-action L2. An offline selector trained on
the 4100000 window looked promising when mixing separate frozen/R3 JSON arrays,
but the direct mixed-policy eval failed:

| window | frozen | R3 | direct selector |
| --- | ---: | ---: | ---: |
| 4100000 | 0.656 | 0.618 | 0.660 |
| 4200000 | 0.656 | 0.658 | 0.604 |
| 4300000 | 0.622 | 0.658 | 0.602 |

This exposed an important evaluation caveat: per-episode arrays from separate
vectorized closed-loop evals are not guaranteed to be seed-aligned once policies
terminate and reset at different times. Aggregate success metrics are still
valid, but paired counts from separate vector eval arrays are diagnostic only.
Selector policies must be evaluated directly in the simulator or on an explicit
fixed reset bank with stable episode IDs.

I added `low-level-rl eval-serial` for this purpose. It uses the same
`ManiSkillVectorEnv` wrapper as the vector evaluator, resets one environment per
explicit seed, and writes `episode_seed` to the JSON. On a compact 50-seed debug
window (`4501000..4501049`), exact pairing worked:

| policy | success | improvements | regressions | net |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.620 | - | - | - |
| R3 ungated | 0.700 | 10 | 6 | +4 |
| initial linear selector | 0.640 | 5 | 4 | +1 |

This confirms that serial exact-seed eval is the right small-window tool for
selector debugging. It also further weakens the initial linear selector: with
reliable pairing, it underperforms ungated R3 on this window.

I also added `low-level-rl compare-serial`, which writes exact paired counts
only after verifying matching `episode_seed` arrays. Vector eval JSONs are now
marked `eval_mode: vector_auto_reset_unpaired` with `episode_seed: null`.

I then added `low-level-rl fit-serial-selector` and fit a three-feature initial
selector on exact paired serial data. It overfit:

| split | frozen | R3 | selector |
| --- | ---: | ---: | ---: |
| train 4501000 | 0.620 | 0.700 | 0.780 |
| validation 4502000 offline mix | 0.740 | 0.660 | 0.660 |
| validation 4502000 direct run | 0.740 | 0.660 | 0.640 |

So even after fixing the pairing methodology, a one-shot initial selector is
not reliable. The next selector attempt should use a larger fixed reset-bank
dataset or move to step/segment-level decisions.

I added serial segment-level arrays and `low-level-rl compare-serial-segments`
to test the step/segment direction on a 50-episode exact-seed window
(`4503000..4503049`). R3 improved task success on that small window
(`0.680` vs frozen `0.620`), but segment-level local labels were nearly
balanced:

| segment metric | value |
| --- | ---: |
| paired segments | 500 |
| helpful R3 segments | 244 |
| harmful R3 segments | 256 |
| mean raw-reduction delta | 0.003 |

The best simple segment-start feature was initial raw distance, with only
`0.567` oriented AUC for helpful-vs-harmful R3 segments. This makes simple
segment-level scalar gating look weak too.

### Lowering the R3 BC weight did not fix the objective

I trained a matching R3 40k run with `bc_weight=1.0` instead of the current best
`bc_weight=10.0`. On the exact serial `4503000..4503049` window:

| policy | success | raw local reduction | residual L2 | net vs frozen |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.620 | 0.414 | 0.000000 | - |
| R3 bc10 | 0.680 | 0.417 | 0.001031 | +3 |
| R3 bc1 | 0.660 | 0.444 | 0.001064 | +2 |

BC1 increased mean raw local reduction but did not create a larger action shift
or better task success. This points away from "just reduce BC weight" and toward
changing the reward target/objective itself.

### Paired terminal reward is cleaner but not enough yet

I added an R3 `--reward-mode paired` option that clones a frozen low-level
branch at each synchronized segment start and rewards:

```text
base_next_distance - tuned_next_distance
```

This directly optimizes improvement over the frozen segment policy instead of
absolute terminal distance. A first 10240-step diagnostic was small but
positive:

| run | mean paired improvement | improved segments | tuned terminal | base terminal | direct delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| paired R3 bc10 | 0.01234 | 0.522 | 0.5767 | 0.5890 | 0.0294 |

On an exact 20-seed serial smoke window (`4504000..4504019`):

| policy | success | improvements | regressions | net |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.500 | - | - | - |
| paired R3 10k | 0.600 | 3 | 1 | +2 |

This was not enough evidence to update the main best-policy claim, so I scaled
the paired objective to a 40k diagnostic. The best checkpoint came from the
positive 20480-step row:

| global step | mean paired improvement | improved segments | resync events |
| ---: | ---: | ---: | ---: |
| 20480 | 0.01493 | 0.514 | 0 |
| 40960 | n/a | n/a | 1 |

On a fresh exact 50-seed serial window (`4505000..4505049`), the 40k paired
checkpoint was neutral on task success:

| policy | success | improvements | regressions | net |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.560 | - | - | - |
| paired R3 40k best | 0.560 | 7 | 7 | 0 |

It did improve raw local reduction (`0.485` vs `0.443`) and max reward
slightly (`0.696` vs `0.683`), but not success. The paired objective is cleaner
than absolute terminal distance, but by itself it still does not establish a
robust policy improvement.

### Privileged-state sanity narrows the RL bottleneck

The privileged-z experiments remain the upper-bound diagnostic where
representation should not be the main bottleneck. Earlier, the residual A-clean
n=1800 run looked positive on a 100-episode window:

| policy | learned-high success | oracle-goal success | note |
| --- | ---: | ---: | --- |
| frozen privileged-z n=1800 | 0.470 | 0.690 | 100-episode PZ window |
| residual alpha 0.25 | 0.580 | 0.650 | improves learned-high, hurts oracle-goal |

I then checked both residual alpha `0.25` and the stronger-looking direct paired
hard-start checkpoint against a same-seed frozen baseline on
`9900000..9900199`:

| policy | mode | success | return |
| --- | --- | ---: | ---: |
| frozen privileged-z n=1800 | learned-high | 0.560 | 44.74 |
| residual alpha 0.25 | learned-high | 0.555 | 44.24 |
| direct paired hard-start | learned-high | 0.515 | 40.47 |
| frozen privileged-z n=1800 | oracle-goal | 0.720 | 48.69 |
| residual alpha 0.25 | oracle-goal | 0.725 | 49.01 |
| direct paired hard-start | oracle-goal | 0.700 | 47.11 |

So privileged state confirms the idea is not completely dead, but neither the
previous residual gain nor the stronger direct paired local improvements
currently establish a robust learned-high deployment improvement. The current
bottleneck is not only visual representation; it is the objective/deployment
alignment of local RL updates.

### Multifeature segment gating has local signal

I added `low-level-rl fit-serial-segment-selector`, which fits an offline linear
selector from exact paired serial segment data using only segment-start features:
initial selected distance, initial raw distance, base action L2, previous action
L2, and segment start step.

Training on `4503000..4503049` and validating on `4506000..4506049`:

| split | base raw reduction | R3 raw reduction | selector raw reduction | selector delta vs base | selector use R3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.414 | 0.417 | 0.478 | +0.064 | 0.796 |
| validation | 0.451 | 0.461 | 0.515 | +0.063 | 0.788 |

Validation AUC for segment helpfulness was only `0.584`, but the aggregate local
raw-reduction gain held out. On the same validation window, ungated R3 was still
negative for episode success (`0.700` vs frozen `0.720`), so this is not yet a
deployment win. It is evidence that multifeature segment-start gating is more
promising than the earlier scalar gates.

Online deployment of the same selector on `4506000..4506049` did not transfer to
task success:

| policy | success | max reward | raw local reduction | segment goal reach | R3 segment use |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.720 | 0.802 | 0.451 | 0.698 | - |
| ungated R3 bc10 | 0.700 | 0.783 | 0.461 | 0.652 | 1.000 |
| online segment selector | 0.680 | 0.771 | 0.460 | 0.668 | 0.760 |

The online selector still slightly improved local raw reduction over frozen
(`+0.008` over 500 aligned segments), but regressed episode success (`-0.040`,
7 improved episodes, 9 regressed). The offline segment selector is therefore a
useful diagnostic, not a deployable fix for this checkpoint.

### Lower BC paired reward did not fix the objective

I retried paired R3 with a lower BC anchor (`bc_weight=1`, 2048 envs because the
4096-env paired branch exceeded GPU camera allocation). The best step again had
mean paired improvement `0.0149` and fraction improved `0.514`, but with much
higher training saturation (`0.260`).

Fresh exact serial validation on `4507000..4507049`:

| policy | success | max reward | raw local reduction | segment goal reach |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.560 | 0.673 | 0.425 | 0.684 |
| paired R3 2048 40k bc1 | 0.560 | 0.673 | 0.383 | 0.656 |

Episode success tied frozen, but local raw reduction regressed by `-0.0417` over
500 aligned segments. This argues against "BC regularization is just too
strong" as the main explanation for the weak paired-reward result.

### Cached local-reset paired reward is implemented

I added `--reward-mode paired` to `rl-rerun train-local-r3`. It measures the
frozen base terminal distance by replaying the exact same local reset before the
tuned rollout, then uses terminal `base - tuned` improvement without keeping a
simultaneous frozen branch in memory. This directly addresses the previous
paired-branch desync and GPU camera allocation issue.

One 4096-env update is mechanically successful but not yet useful:

| run | envs | steps | train paired improvement | train fraction improved | validation final distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen local n500 | 4096 | - | - | - | 0.6020 |
| cached-paired local R3 bc1 dense | 4096 | 40960 | -0.0098 | 0.478 | 0.6066 |
| cached-paired local R3 bc1 dense | 4096 | 122880 | -0.0151 | 0.471 | 0.6000 |
| cached-paired local R3 bc1 terminal-only | 4096 | 122880 | -0.0103 | 0.482 | 0.6086 |
| cached-paired local R3 lr1e-5 logstd-5 | 4096 | 122880 | -0.0044 | 0.490 | 0.6081 |

The matched validation manifest had identical initial distance (`1.0671`), so
the one-update cached-paired policy was slightly worse than frozen locally. The
three-update dense-progress checkpoint was slightly better on the held-out local
manifest (`final distance 0.6000` vs frozen `0.6020`), but the effect is tiny
and the training paired-improvement signal became more negative across updates.
Terminal-only paired training improved the training paired metric relative to
dense progress, but failed held-out validation. This is mechanically useful
infrastructure, not yet a convincing RL improvement. Lowering LR/noise made the
training paired metric less negative and reduced action deltas, but also failed
held-out validation.

Broader local validation on 8 held-out timesteps from
`pusht_vector_state_demos_n512_val_b1.h5` changes the sign but not the scale:

| policy | final distance | final reward | max reward | success-once |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 | 0.7092 | 0.3979 | 0.4649 | 0.2456 |
| dense-progress cached paired 3 updates | 0.7086 | 0.4005 | 0.4636 | 0.2439 |
| terminal-only lr1e-5 logstd-5 | 0.7080 | 0.4005 | 0.4672 | 0.2493 |

The cached paired variants are slightly better than frozen on this broader
mean-distance metric. After adding task-reward diagnostics, the lower-noise
terminal-only checkpoint is weakly positive across final distance, final/max
environment reward, mean environment reward, and success-once. The gain is still
tiny (`+0.0037` success-once and about `+0.002` reward), so it is useful as a
selection diagnostic but too small to justify closed-loop deployment.

A direct 500-episode learned-goal closed-loop transfer check confirmed that this
local signal is not enough:

| policy | success | final reward | max reward |
| --- | ---: | ---: | ---: |
| frozen n500 | 0.334 | 0.4776 | 0.5061 |
| terminal-only lr1e-5 logstd-5 | 0.294 | 0.4480 | 0.4760 |

The tuned checkpoint loses `-0.040` success and about `-0.030` reward despite
the weakly positive local diagnostics. Treat cached-paired local R3 as
diagnostically useful infrastructure, not as a candidate deployment policy.

I also added a default-off task-reward debug knob to `rl-rerun train-local-r3`
and ran a one-update dense-task-reward upper-bound check. It showed the same
pattern:

| policy | local success-once | closed-loop success | closed-loop max reward |
| --- | ---: | ---: | ---: |
| frozen n500 | 0.2456 | 0.334 | 0.5061 |
| task-reward debug 1 update | 0.2493 | 0.306 | 0.4867 |

So even using the environment reward directly as a diagnostic did not make the
current local R3 update transfer. The problem is not only raw latent distance;
the current one-segment local update/selection loop is not deployment-aligned
enough.

The `rl-rerun` closed-loop evaluator now emits per-episode action-delta,
policy-saturation, replan goal-distance, and high-level-decision diagnostics
alongside episode success/reward. This does not improve the policy by itself,
but it is the right substrate for the next gate: any selector should be fit and
validated against paired closed-loop wins/regressions, not local reset metrics.

On the 500-episode task-reward-debug diagnostic bank, the tuned branch had 60
wins, 74 regressions, and 366 ties versus frozen. Initial goal distance was weak
for separating wins from regressions (`0.545` oriented AUC), while online or
trajectory features were stronger: mean action delta `0.876`, policy saturation
rate `0.749`, and mean goal distance `0.702`. This reinforces that a one-shot
initial gate is the wrong tool; a candidate gate must be online and evaluated
directly in closed loop.

I tested the simplest online version, `--action-delta-gate-min`, which executes
the frozen action unless the current tuned/base action delta exceeds a threshold.
It reduced harm but did not beat frozen:

| policy | threshold | success | max reward | gate rate |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 | - | 0.334 | 0.5061 | 0.000 |
| task-reward debug ungated | - | 0.306 | 0.4867 | 0.000 |
| action-delta gate | 0.0006 | 0.314 | 0.4916 | 0.732 |
| action-delta gate | 0.0008 | 0.298 | 0.4791 | 0.858 |
| action-delta gate | 0.0010 | 0.294 | 0.4776 | 0.928 |

Since this gate cannot beat frozen even on the threshold-selection window, the
next useful step is not another scalar gate. The tuned interventions themselves
need to become larger and more robustly deployment-aligned.

A learned-vs-oracle goal split gives a sharper diagnosis for the task-reward
debug branch:

| goal source | frozen success | tuned success | delta | tuned max reward delta |
| --- | ---: | ---: | ---: | ---: |
| learned goals, 500 eps | 0.334 | 0.306 | -0.028 | -0.0194 |
| oracle goals, 500 eps | 0.334 | 0.350 | +0.016 | +0.0113 |

The local update can help slightly when the goal is generated by the privileged
teacher continuation, but it hurts under the learned high-level goal
distribution. For this branch, the next useful target is high-level goal
validity or robustness to learned-goal errors, not another scalar action gate.

I added `--diagnose-oracle-goals` to `rl-rerun` closed-loop eval so learned-goal
rollouts can record oracle branch goals without changing the deployed policy.
On a 100-episode learned-goal diagnostic bank, predicted-vs-oracle goal distance
was only a weak separator of tuned wins/regressions (`0.558` oriented AUC). The
stronger signals were online rollout features: mean current-learned goal
distance `0.923`, action delta `0.933`, policy saturation `0.885`, and mean
current-oracle goal distance `0.865`. So the issue is not just off-manifold
learned goals; useful low-level interventions appear concentrated in difficult
online regimes.

I tried turning those online features into a direct multifeature gate using
`--action-delta-gate-min 0.0006` plus `--goal-l2-gate-min`. On the same
100-episode learned-goal window, both tested thresholds were worse than
ungated:

| policy | goal L2 threshold | success | success delta |
| --- | ---: | ---: | ---: |
| frozen n500 | - | 0.350 | - |
| task-reward debug ungated | - | 0.300 | -0.050 |
| multifeature gate | 24 | 0.260 | -0.090 |
| multifeature gate | 27 | 0.270 | -0.080 |

This closes the hand-coded gate branch for this checkpoint. The next useful work
should alter the training target or train a selector/policy directly in the
closed-loop distribution.

## Current Best Policies

Best observed real-compatible checkpoint:

```text
artifacts/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10/best_train_latent.pt
```

Best current real-compatible deployment variant:

```text
same checkpoint, evaluated with --residual-l2-gate-max 0.00121
```

But the recommended report wording is:

```text
Residual-gated R3 is approximately tied with frozen after final-style
validation. It is diagnostically useful, but not a robust policy improvement.
```

## Recommended Next Work

Stop spending compute on scalar threshold tuning for this checkpoint.

The next useful directions are:

1. Add a state/goal-aware gate.
   The eval path now records per-episode success, reward, residual magnitude,
   saturation, local progress, and compact pre-decision state/goal features.
   Initial episode features and scalar current-distance gating are both weak,
   and the exact-serial multifeature segment selector improved held-out local
   raw reduction offline but failed online. Further gate work should train
   directly against closed-loop episode outcomes or use a richer policy/context
   model; offline local segment deltas are not enough.

2. Improve the objective so the tuned policy creates a larger effect.
   The current R3 updates are tiny. Reducing BC weight from 10 to 1 did not
   solve this, and paired `bc=1` made fresh-window local raw reduction worse.
   Paired terminal reward is cleaner and improved local raw reduction in the
   `bc=10` training window, but the 40k exact serial check was success-neutral.
   Cached local-reset paired reward is now implemented and avoids simultaneous
   branch desync, but three 4096-env updates still showed negative training
   paired improvement and only a tiny held-out local gain. Terminal-only paired
   reward improved the training signal but regressed validation. Lower
   LR/action noise improved the training metric but suppressed useful held-out
   action changes. A broader reset-bank validation changed the sign of the local
   mean effect but kept it near zero; task-reward diagnostics are also only
   weakly positive and the best local candidate failed a 500-episode closed-loop
   transfer check. A direct task-reward debug upper bound also failed transfer.
   The next objective check should change the target regime, not simply scale
   the same formulation: move toward a stronger deployment-aligned signal than
   one-segment local reachability or local task reward alone.
   The privileged direct hard-start check shows that large selected-local gains
   can still hurt closed-loop deployment, so checkpoint selection needs
   deployment evidence or a better local-to-task proxy.

3. Revisit horizon/representation only after the gate/objective question.
   The current effect32 interface is goal-dependent, so the main bottleneck is
   not simply "low level ignores the goal"; it is reliable improvement without
   damaging already-good frozen behavior.

## Key Artifacts

Experiment log:

```text
hcl_next_experiment_log.md
```

Evaluator/CLI support:

```text
src/hcl_poc/low_level_rl.py
src/hcl_poc/cli.py
```

Recent useful eval outputs:

```text
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_final1000_seed4000000/eval_1000_seed4000000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_final1000_seed4000000/eval_1000_seed4000000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_gate00121_final1000_seed4000000/eval_1000_seed4000000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_predetail500_seed4100000/eval_500_seed4100000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_predetail500_seed4100000/eval_500_seed4100000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate085_final500_seed4100000/eval_500_seed4100000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_distgatecheck500_seed4200000/eval_500_seed4200000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgatecheck500_seed4200000/eval_500_seed4200000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_distgate085_check500_seed4200000/eval_500_seed4200000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_traincheck500_seed4100000/eval_500_seed4100000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_check500_seed4200000/eval_500_seed4200000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_check500_seed4300000/eval_500_seed4300000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4501000/serial_eval_50_seed4501000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/serial_eval_50_seed4501000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_serial50_seed4501000/serial_eval_50_seed4501000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/paired_vs_frozen_serial50_seed4501000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_initselector_serial50_seed4501000/paired_vs_frozen_serial50_seed4501000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_serial50_seed4502000/serial_eval_50_seed4502000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4502000/serial_eval_50_seed4502000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_serial50_seed4501000/init_selector_fit_train4501000_valid4502000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_fitselector_exact_serial50_seed4502000/serial_eval_50_seed4502000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/serial_eval_50_seed4503000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/paired_segments_vs_frozen_serial50_seed4503000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1/train_metrics.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc1_serial50_seed4503000/serial_eval_50_seed4503000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/train_metrics.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/serial_eval_20_seed4504000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_10240_bc10/paired_vs_frozen_serial20_seed4504000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/train_metrics.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/serial_eval_50_seed4505000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_paired_40k_bc10/paired_vs_frozen_serial50_seed4505000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_frozen_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentselector_serial50_seed4506000/serial_eval_50_seed4506000.json
results/incremental/low_level_rl/effect32_film/seed0/hcl_next_effect32_dphi_r3_4096_terminal_smoke_40k_bc10_segmentdetail_serial50_seed4503000/segment_selector_fit_train4503000_valid4506000.json
results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_hierarchy_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_oracle_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_hierarchy_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_oracle_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_hierarchy_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_oracle_200eps.json
```
