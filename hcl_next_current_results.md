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

### Paired terminal reward is the next objective candidate

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

This is not enough evidence to update the main best-policy claim, but it is the
first reward-target change in this sequence with both a positive paired
train-time signal and a matching small exact-serial task signal. The next
compute should scale this paired objective to the 40k setting and validate on a
larger exact serial window before broad vector eval.

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
   Initial episode features and scalar current-distance gating are both weak.
   A next gate should not be trained from unaligned vector-eval arrays. First
   add explicit episode identity or a fixed reset-bank evaluator, then train a
   multifeature step/segment selector. The exact-serial initial selector already
   failed validation, and simple segment-start features are weak, so avoid more
   one-dimensional gates.

2. Improve the objective so the tuned policy creates a larger effect.
   The current R3 updates are tiny. Reducing BC weight from 10 to 1 did not
   solve this. The new paired terminal reward is the current best candidate:
   scale it from the 10240-step diagnostic to the 40k setting and validate on
   at least a 50-seed exact serial window.

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
```
