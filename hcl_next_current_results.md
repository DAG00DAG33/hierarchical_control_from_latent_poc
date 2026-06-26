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

A fixed-seed serial replay check on the first 100 seeds of that same window
kept the positive sign:

| policy | seed start | episodes | success | paired improvements | paired regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen effect32_film | 3500000 | 100 | 0.600 | - | - |
| R3 terminal-only 40k bc10 | 3500000 | 100 | 0.670 | 14 | 7 |

I added an export path that converts an R3 direct-low checkpoint back into a
normal learned-interface hierarchy checkpoint. The exported low policy is
action-identical to the direct agent on identical conditions, but evaluating it
through `learned-interface-eval` on 200 fixed seeds regressed learned and oracle
success (`0.645 -> 0.635` learned, `0.645 -> 0.605` oracle). Treat that as an
evaluator-protocol caveat rather than a low-policy weight mismatch: the low-level
serial evaluator still shows a paired positive result on exact seeds.

I then added oracle-goal support directly to `low-level-rl eval-serial`. The
first 100-seed bank suggested R3 still helped under oracle goals, but a fresh
100-seed bank reversed that. Over the two exact-seed banks:

| goal source | frozen | R3 | paired improvements | paired regressions | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned | 0.645 | 0.690 | 28 | 19 | +9 |
| oracle | 0.720 | 0.700 | 23 | 27 | -4 |

So the high-level learned goal is a real bottleneck, and oracle local goals
raise the frozen ceiling. But the R3 low-level update does **not** validate
under oracle goals; its apparent oracle-goal gain was a one-bank false lead.
The remaining learned-goal R3 gain is small and should be treated as
distribution-specific rather than a robust low-level improvement.

I also tested an eval-only nearest-training-goal projection for learned serial
goals. Each high-level prediction is snapped to the nearest normalized
training-set goal from the learned-interface encoded replay bank. This produced
62,472 effect32 prototypes and moved predicted goals by roughly `1.4-1.5`
normalized L2 on average.

Two exact-seed banks:

| policy | seed starts | success | paired improvements | paired regressions | net |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen learned-goal baseline | 3500000, 3600000 | 0.645 | - | - | - |
| frozen + nearest-train projection | 3500000, 3600000 | 0.660 | 23 | 20 | +3 |
| oracle frozen ceiling | 3500000, 3600000 | 0.720 | - | - | - |
| R3 learned-goal baseline | 3500000, 3600000 | 0.690 | - | - | - |
| R3 + nearest-train projection | 3500000, 3600000 | 0.670 | 28 | 32 | -4 |

The projection is therefore a useful high-level diagnostic, not a strong policy
fix. It closes only a small fraction of the learned-vs-oracle gap for the frozen
policy and hurts the current R3 aggregate.

I then added a more plan-I-style eval-only variant, `nearest_train_dphi`: first
take the top-k nearest training goals by raw latent L2, then select the one with
lowest learned reachability distance `D_phi(current, goal)`. On two matched
20-episode prefixes of the existing serial seed windows (`3500000`, `3600000`,
top-k `32`), this beat raw nearest-goal projection but still did not clearly
beat no projection:

| policy | projection | episodes | success | final reward | max reward |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen | none | 40 | 0.700 | 0.6791 | 0.7852 |
| frozen | nearest_train | 40 | 0.600 | 0.6292 | 0.7136 |
| frozen | nearest_train_dphi | 40 | 0.675 | 0.7353 | 0.7679 |
| R3 | none | 40 | 0.625 | 0.6301 | 0.7383 |
| R3 | nearest_train | 40 | 0.600 | 0.6018 | 0.7150 |
| R3 | nearest_train_dphi | 40 | 0.650 | 0.6106 | 0.7425 |

This is a better projection diagnostic than raw nearest-neighbor snapping, but
not a promotion path yet. It suggests `D_phi` can choose less harmful
on-manifold goals than raw nearest L2, while also confirming that high-level
goal projection alone is not enough to make the R3 residual reliable.

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

### Shorter held-goal horizons were worse

I exposed `rl-rerun goal-diagnostics` for named learned-interface candidates,
not only the VAE512 scaling alias, and ran it on the current `effect32_film`
hierarchy. This is a diagnostic for the shared
`artifacts/incremental/learned_interface/effect32_film/seed0/hierarchy.pt`
artifact; the non-VAE learned-interface path is not currently split into
separate `N=500` and `N=1800` checkpoints.

The goal-use gate remains weak:

| metric | value |
| --- | ---: |
| frame shuffle action change L2 | 0.950 |
| goal shuffle action change L2 | 0.062 |
| previous-action shuffle action change L2 | 0.133 |
| max same-state horizon sensitivity L2 | 0.0368 |
| goal shuffle MAE gap | 0.0102 |

So `effect32_film` is goal-dependent enough that shuffled goals hurt closed-loop
performance, but the low-level action is still dominated by the current frame.
This supports the current decision to avoid more expensive RL on candidates that
do not pass a stronger goal-use gate.

The existing diagnostic archive also shows why this gate cannot be the only
selector. Some candidates have much stronger offline goal sensitivity than
`effect32_film`, but they were weaker deployment bases:

| candidate | goal shuffle L2 | frame shuffle L2 | max horizon sensitivity L2 |
| --- | ---: | ---: | ---: |
| effect32_film | 0.062 | 0.950 | 0.0368 |
| ae256_film | 0.251 | 0.865 | 0.0937 |
| vae512_b1e6_film | 0.278 | 0.821 | 0.1266 |

So goal-use diagnostics should be treated as a hard rejection gate, not a
promotion criterion. A candidate still needs closed-loop imitation quality and
local-to-task transfer before PPO is worth scaling.

I added an aggregate diagnostics gate command that applies this conservative
rule across all diagnostic JSONs:

```text
offline_goal_use_pass if goal_shuffle_action_change_l2 >= 0.1
or max_goal_sensitivity_l2 >= 0.1
```

On the current archive plus the new dropout checks, the baseline-initialized
goal-sensitivity fine-tune, and the action-aware high-level fine-tunes, five of
thirty-seven
diagnostics pass:

| status | candidates |
| --- | --- |
| offline goal-use pass | `effect32_film_frame_drop25`, `effect32_film_gsens`, `effect32_film_scene_drop25`, `ae256_film`, `vae512_b1e6_film` |
| reject low goal-use | all other archived hierarchy candidates checked so far, including `effect32_film_gsens_ft`, `effect32_film_gsens_ft_highact`, and `effect32_film_gsens_ft_highact_strong` |

This formalizes the current decision rule. Effect32 remains the best observed
deployment base, but it fails the strict offline goal-use gate; AE/VAE FiLM
pass the gate but were weaker deployment bases in prior checks. The next
candidate worth serious PPO needs both: pass this gate and preserve closed-loop
imitation quality.

Fixed-seed 500-episode learned-interface evals confirm that neither current
gate-passing candidate meets the second requirement:

| candidate | offline gate | learned success | oracle success | shuffled success | learned final reward |
| --- | --- | ---: | ---: | ---: | ---: |
| effect32_film | reject low goal-use | 0.650 | 0.694 | 0.312 | 0.741 |
| ae256_film | pass | 0.544 | 0.642 | 0.034 | 0.657 |
| vae512_b1e6_film | pass | 0.438 | 0.532 | 0.018 | 0.580 |

AE/VAE FiLM collapse much more under shuffled goals, so they are genuinely more
goal-dependent than effect32. They still underperform effect32 on learned-goal
deployment. The promising region is therefore not occupied by any current
candidate. We need a representation/architecture that keeps the AE/VAE-style
goal sensitivity while preserving effect32-level closed-loop imitation quality.

I also checked the already-trained AE/VAE `delta` and `relation` conditioning
variants. They had decent 20-episode deployment smokes, but all four fail the
offline goal-use gate:

| candidate | conditioning | goal shuffle L2 | max horizon sensitivity L2 | status |
| --- | --- | ---: | ---: | --- |
| ae256_delta | delta | 0.0589 | 0.0340 | reject |
| ae256_relation | relation | 0.0707 | 0.0287 | reject |
| vae512_b1e6_delta | delta | 0.0599 | 0.0355 | reject |
| vae512_b1e6_relation | relation | 0.0620 | 0.0270 | reject |

So the only archived AE/VAE conditioning mode that meaningfully uses the goal is
FiLM, and FiLM's stronger goal dependence still does not recover effect32-level
learned-goal performance.

I then completed the same 5k-sample gate for the remaining archived hierarchy
checkpoints that had not been in the report. None passed. The best new rows were
still below the `0.1` goal-shuffle threshold:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | status |
| --- | ---: | ---: | --- |
| dae512_w2048_n005 | 0.0799 | 0.0310 | reject |
| vae512_w2048_b1e7 | 0.0789 | 0.0317 | reject |
| ae256_control | 0.0784 | 0.0277 | reject |
| dae256_n005 | 0.0780 | 0.0272 | reject |
| vae512_w2048_b1e6 | 0.0740 | 0.0308 | reject |

This makes the current archive search fairly complete: there is no hidden
candidate that already combines strong low-level goal usage with effect32-level
closed-loop quality.

I then added opt-in low-level frame-dropout training modes. The first zeros the
whole normalized current-frame block for 25% of low-level BC samples; the second
zeros only the scene/current-object prefix and keeps the 21D proprio tail. Both
confirm that suppressing the observation shortcut can force more goal use, but
both still damage deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.0622 | 0.0368 | 0.645 | 0.645 |
| effect32_film_frame_drop25 | 0.1121 | 0.0615 | 0.490 | 0.465 |
| effect32_film_scene_drop25 | 0.1141 | 0.0616 | 0.510 | 0.560 |
| effect32_film_scene_drop_aux05 | 0.0864 | 0.0481 | 0.500 | 0.570 |

The result is useful diagnostically: goal ignoring is partly an observation
shortcut problem. Keeping proprio improves the result slightly over full-frame
dropout, but the dropout family repeats the same bad tradeoff as the
goal-sensitivity margin loss: it passes the offline gate by weakening the
learned-goal imitation policy. Making scene dropout auxiliary to clean BC
preserves validation MAE better but does not pass the gate and still regresses
closed-loop learned-goal success.

I then added a direct low-level goal-sensitivity regularizer as an opt-in
learned-interface policy loss and trained `effect32_film_gsens`, which reuses
the effect32 representation and high level but penalizes shuffled-goal actions
that remain too close to the correct-goal action. This successfully moved the
offline gate but hurt deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.062 | 0.0368 | 0.645 | 0.645 |
| effect32_film_gsens_light | 0.074 | 0.0405 | 0.570 | 0.570 |
| effect32_film_gsens | 0.115 | 0.0476 | 0.500 | 0.515 |
| effect32_film_gsens_ft | 0.092 | 0.0399 | 0.550 | 0.675 |

So the missing ingredient is not merely "make actions change when the goal is
shuffled." A lighter margin produces a smoother tradeoff, but still loses
`7.5` success points for a small goal-use gain and does not pass the strict
offline gate. A stronger margin passes the gate but loses `14.5` success points.
I then tried initializing from the baseline `effect32_film` low policy and
fine-tuning for only 10 epochs with the stronger margin loss at `1e-5`. This
preserved offline action MAE and improved oracle-goal success (`0.645 ->
0.675`), but learned-goal success still fell to `0.550` and the strict
goal-use gate was still missed (`0.092 < 0.1`). The useful diagnosis is that
low-level goal sensitivity can help when goals are good; with the current
learned high-level goals, the same low-level change does not transfer. The next
representation/architecture attempt should couple high-level goal quality with
low-level goal use, not optimize the low-level margin alone.

I tested that coupling directly with `nearest_train_dphi` projection on two
20-episode serial windows. Projection improved `effect32_film_gsens_ft` on
aggregate (`0.625 -> 0.650` success, max reward `0.740 -> 0.754`), but the sign
flipped across the two seed windows and the projected policy still did not beat
the original no-projection `effect32_film` smoke (`0.700` success). So
reachability-aware goal repair and low-level sensitivity can interact, but this
combination is still a diagnostic rather than a promotable hierarchy.

I then added an action-aware high-level fine-tune. It initializes the high model
from `effect32`, freezes the `effect32_film_gsens_ft` low model, and
backpropagates demonstration action MSE through that frozen low model into the
predicted high-level goal while retaining the normal future-goal MSE. This
recovered part of the learned-goal deployment loss:

| candidate | learned success | learned max reward | oracle success | oracle max reward | oracle goal L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| effect32_film | 0.645 | 0.742 | 0.645 | 0.746 | 3.391 |
| effect32_film_gsens_ft | 0.550 | 0.679 | 0.675 | 0.773 | 3.292 |
| effect32_film_gsens_ft_highact | 0.595 | 0.713 | 0.675 | 0.773 | 3.264 |

This confirms that high-level action-aware tuning is directionally useful, but
the mild high-only version is still below the original learned-goal baseline.
It should be treated as evidence for coupled high/low training, not as a
promotion candidate.

I then strengthened that high-level-only objective by reducing the high goal-MSE
weight, increasing the action-through-low loss, and training for 20 epochs. This
is the best coupled learned-interface result so far:

| candidate | learned success 500 | final reward 500 | max reward 500 | teacher MAE 500 |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.650 | 0.7410 | 0.7484 | 0.0996 |
| ae256_film | 0.544 | 0.6572 | 0.6707 | 0.1184 |
| vae512_b1e6_film | 0.438 | 0.5798 | 0.5952 | 0.1167 |
| effect32_film_gsens_ft_highact_strong | 0.652 | 0.7455 | 0.7523 | 0.0895 |

The margin is tiny, so this is not yet a robust improvement claim. But it is a
real lead: a more goal-sensitive low-level can be made deployable again by
training the high-level against the frozen low-level's induced action error.
The next validation should use a fresh 500-episode learned-goal window before
promoting it, and the next implementation step should consider joint high/low
coupled training if the fresh window holds.

I then tested an effect32 "base + goal residual" low-level architecture,
`effect32_goal_residual`, where a no-goal base policy predicts the action and a
zero-initialized goal-conditioned residual can correct it. This preserved a
clean base path but collapsed further toward ignoring the goal:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | 20-episode learned success | 20-episode oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.062 | 0.0368 | 0.750 | 0.750 |
| effect32_goal_residual | 0.0218 | 0.0154 | 0.450 | 0.450 |

The aggregate goal-use gate now has `3` pass / `11` reject / `14` total, and
`effect32_goal_residual` is rejected for low goal use. I skipped the longer
200-episode cross-check because both the offline diagnostic and the 20-episode
screen were worse than the baseline. This closes the simple residual-addition
architecture branch: separating a base path from a goal residual did not force
useful goal-conditioned corrections.

I also tested an `effect64_film` capacity variant that reuses the existing
64-dimensional effect-code representation and high level with FiLM low-level
conditioning. It increased offline goal-shuffle response compared with
`effect32_film`, but not enough to pass the gate, and it weakened deployment:

| candidate | goal shuffle L2 | max horizon sensitivity L2 | learned success | oracle success |
| --- | ---: | ---: | ---: | ---: |
| effect32_film | 0.062 | 0.0368 | 0.645 | 0.645 |
| effect64_film | 0.082 | 0.0488 | 0.595 | 0.535 |

The aggregate goal-use gate now has `3` pass / `12` reject / `15` total. This
closes the simple effect-code capacity increase as a fix: larger effect latents
move the offline goal-use metric in the right direction, but again trade away
the closed-loop quality needed before PPO scaling is justified.

I added candidate-level `horizon_steps` / `update_period` overrides for learned
interfaces and trained short-horizon aliases of the effect32 FiLM interface:

```text
effect32_film_h5: representation_candidate=effect32, high_level_candidate=effect32, horizon/update=5
effect32_film_h2: representation_candidate=effect32, high_level_candidate=effect32, horizon/update=2
```

Matched closed-loop evaluations on `seed_start=3500000`, 200 episodes:

| candidate | goal source | success | final reward | max reward | teacher MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| effect32_film k10 | learned | 0.645 | 0.735 | 0.742 | 0.107 |
| effect32_film k10 | oracle | 0.645 | 0.740 | 0.746 | 0.089 |
| effect32_film_h5 | learned | 0.520 | 0.649 | 0.658 | 0.103 |
| effect32_film_h5 | oracle | 0.570 | 0.685 | 0.696 | 0.092 |
| effect32_film_h2 | learned | 0.315 | 0.481 | 0.501 | 0.133 |

The k=5 low policy is worse even with oracle goals, and k=2 is much worse
despite frequent replanning. This rejects the simple "shorter hierarchy horizon"
hypothesis for the current effect32 FiLM setup. The remaining issue is not just
that k=10 goals are too far away; changing horizon without changing the
objective/closed-loop training distribution degrades deployment.

Goal diagnostics on the short-horizon aliases clarify the failure: they do not
fail because the low-level ignores goals more. In fact, h2 has the strongest
offline goal sensitivity of the three:

| candidate | goal shuffle L2 | frame shuffle L2 | max horizon sensitivity L2 | action MAE at h=10 |
| --- | ---: | ---: | ---: | ---: |
| effect32_film k10 | 0.062 | 0.950 | 0.0368 | 0.0425 |
| effect32_film_h5 | 0.066 | 0.944 | 0.0419 | 0.0460 |
| effect32_film_h2 | 0.099 | 0.935 | 0.0810 | 0.0585 |

So the short-horizon regression is a deployment/training-distribution problem,
not a simple goal-use problem. Increasing low-level goal sensitivity without
maintaining closed-loop imitation quality can make the hierarchy worse.

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

I also extended the privileged branch-goal counterfactual harness so each
candidate rollout saves first-segment end-state and prefix outcome features.
This tests whether a selector can use actual candidate prefix evidence, not
only static `(query, candidate goal, source outcome)` features. On a fresh
`q128/k8` bank, the candidate set still had large oracle upside, but the small
MLP selector did not extract it:

| selector over 5 query-split seeds | validation return delta | validation success delta |
| --- | ---: | ---: |
| learned selector with prefix features | -0.517 | -0.013 |
| nearest candidate | +2.070 | +0.031 |
| oracle best-of-8 | +15.530 | +0.237 |

So prefix/end-state features are not sufficient in the current static selector
form. The remaining privileged branch-goal direction would need a better
candidate-generation distribution, more query coverage, or a selector trained
as an online/intervention policy rather than another small offline scorer.

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

I then extended `fit-serial-segment-selector` to train on multiple exact serial
windows and fit on the union of `4503000..4503049` plus `4506000..4506049`
(`1000` aligned segments). Offline validation on a fresh exact window
`4508000..4508049` still looked locally positive:

| split | segments | base raw reduction | R3 raw reduction | selector raw reduction | selector use R3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 4503000+4506000 | 1000 | 0.433 | 0.439 | 0.497 | 0.690 |
| validation 4508000 | 500 | 0.415 | 0.428 | 0.490 | 0.730 |

But direct online deployment on `4508000..4508049` failed:

| policy | success | max reward | raw-reduction delta vs frozen | segment use R3 |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.680 | 0.778 | - | - |
| ungated R3 bc10 | 0.740 | 0.816 | +0.0126 | 1.000 |
| multi-window segment selector | 0.660 | 0.765 | +0.0026 | 0.736 |

The larger exact segment dataset did not fix the offline-to-online mismatch.
This makes the selector conclusion stronger: selecting from completed segment
outcomes is still not the right training target because the deployed selector
changes later states/goals. A useful selector needs closed-loop intervention
training or a much larger effect from the candidate policy.

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
debug branch. I repeated the 500-episode check on a fresh seed window, and the
sign replicated:

| seed start | goal source | frozen success | tuned success | success delta | max-reward delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4600000 | learned | 0.334 | 0.306 | -0.028 | -0.0194 |
| 4600000 | oracle | 0.334 | 0.350 | +0.016 | +0.0113 |
| 4700000 | learned | 0.304 | 0.276 | -0.028 | -0.0156 |
| 4700000 | oracle | 0.314 | 0.340 | +0.026 | +0.0168 |

The local update can help slightly when the goal is generated by the privileged
teacher continuation, with a two-window mean delta of `+0.021` success and
`+0.0141` max reward. It hurts under the learned high-level goal distribution,
with a two-window mean delta of `-0.028` success and `-0.0175` max reward. For
this branch, the next useful target is high-level goal validity or robustness to
learned-goal errors, not another scalar action gate.

I added `eval-learned-goal-validity` to test the simplest high-level
off-manifold hypothesis on the `N=500` VAE512 learned-latent hierarchy. On 4096
sampled validation decisions, predicted goals were fairly close to the replay
future-goal manifold:

| metric | mean |
| --- | ---: |
| predicted nearest replay-goal L2 | 15.689 |
| replay leave-one-out nearest replay-goal L2 | 14.278 |
| random nearest replay-goal L2 | 25.271 |
| predicted-to-replay goal L2 | 19.842 |
| shuffled-to-replay goal L2 | 27.845 |
| predicted-vs-replay low-action L2 | 0.0109 |

So the learned high-level goals are not obviously random/off-manifold by this
nearest-neighbor test. The low-level remains the sharper failure: even a large
predicted-vs-replay goal difference causes only a tiny action change.

An existing R3 checkpoint trained with explicit goal-swap sensitivity
regularization initially looked like the first positive `rl-rerun` learned-goal
transfer lead:

```text
artifacts/rl_rerun/local_r3/n500/seed0/goal_sensitivity_w10_m005_smoke_10k/latest.pt
```

Broader fresh closed-loop checks:

| seed start | goal source | frozen success | tuned success | success delta | max-reward delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4800000 | learned | 0.306 | 0.324 | +0.018 | +0.0088 |
| 4800000 | oracle | 0.328 | 0.340 | +0.012 | +0.0068 |
| 4900000 | learned | 0.294 | 0.296 | +0.002 | +0.0040 |
| 5000000 | learned | 0.340 | 0.342 | +0.002 | +0.0009 |
| 5100000 | learned | 0.306 | 0.286 | -0.020 | -0.0099 |

Across the four learned-goal windows, the mean delta is only `+0.001` success
and `+0.0010` max reward. This is effectively neutral, not a robust policy
improvement. It is still better than the task-reward-debug checkpoint's
negative learned-goal transfer, but it should be treated as a diagnostic lead
rather than a deployable improvement.

I trained a five-update version of the same sensitivity objective. It stayed
positive on the checked learned-goal window, but was weaker than the one-update
checkpoint:

| run | success delta | max-reward delta | action delta L2 |
| --- | ---: | ---: | ---: |
| one-update sensitivity, seed 4800000 | +0.018 | +0.0088 | 0.000973 |
| five-update sensitivity, seed 4800000 | +0.004 | +0.0017 | 0.002954 |

So the current conclusion is narrower: sensitivity regularization reduces the
damage relative to the task-reward-debug branch, but neither the one-update nor
the five-update sensitivity checkpoint establishes reliable learned-goal
improvement.

I also tested a stronger sensitivity-weight variant, changing only
`goal_sensitivity_weight` from `10` to `30`. It did not improve the tradeoff:

| run | shared windows | mean success delta | mean max-reward delta |
| --- | --- | ---: | ---: |
| weight 10 | 4800000, 4900000 | +0.010 | +0.0064 |
| weight 30 | 4800000, 4900000 | +0.006 | +0.0064 |

The weight-30 run improved one window and regressed the other, with nearly
identical action magnitude. The current sensitivity formulation appears useful
as a harm-reduction diagnostic, but not sufficient as the main training target.

I also tested loosening the BC anchor from `1.0` to `0.3` while keeping
`goal_sensitivity_weight=10`. This did not create larger useful interventions:

| run | seed 4800000 success delta | max-reward delta | action delta L2 |
| --- | ---: | ---: | ---: |
| BC 1.0, weight 10 | +0.018 | +0.0088 | 0.000973 |
| BC 0.3, weight 10 | -0.018 | -0.0136 | 0.000958 |
| BC 1.0, weight 30 | +0.026 | +0.0168 | 0.000977 |

The aligned local eval was also nearly identical for `BC=1.0` and `BC=0.3`.
So simple BC/sensitivity coefficient scaling is not enough; the next useful
change should alter the target formulation or representation.

I then combined cached paired reward with the sensitivity regularizer. It helped
relative to paired-only locally, but still failed the frozen local gate:

| policy | matched local final distance | reduction |
| --- | ---: | ---: |
| frozen n500 | 0.6020 | 0.4651 |
| paired only | 0.6066 | 0.4605 |
| paired + sensitivity | 0.6047 | 0.4624 |

Because paired+sensitivity remained worse than frozen on the matched local
manifest, I skipped closed-loop deployment for that checkpoint. The next useful
change should be a stronger target or representation change, not a linear sum
of the current paired and sensitivity losses.

I screened `ae256_film` and then rechecked `vae512_b1e6_film` with the same
D_phi/R3 recipe. Both VAE-style representations have strong temporal D_phi
metrics, but neither beats effect32 as a deployment base:

| candidate | temporal Spearman | near/far acc | shuffled AUC | demo-decrease acc |
| --- | ---: | ---: | ---: | ---: |
| ae256_film | 0.933 | 0.986 | 0.868 | 0.694 |
| effect32_film | 0.833 | 0.928 | 0.907 | 0.740 |
| vae512_b1e6_film | 0.930 | 0.984 | 0.877 | 0.703 |

The analogous 40k D_phi R3 smokes gave:

| policy | success | max reward | raw reduction |
| --- | ---: | ---: | ---: |
| ae256_film frozen | 0.596 | 0.710 | 0.294 |
| ae256_film R3 | 0.586 | 0.706 | 0.292 |
| vae512_b1e6_film frozen | 0.418 | 0.580 | 0.150 |
| vae512_b1e6_film R3 | 0.478 | 0.627 | 0.161 |
| effect32_film frozen | 0.634 | 0.738 | 0.397 |
| effect32_film R3 | 0.684 | 0.773 | 0.410 |

So `ae256_film` is not a better RL base, and VAE512 FiLM is a weak base even
though R3 improves it on this window. Stronger one-step goal sensitivity or
clean D_phi temporal structure is not enough when the frozen closed-loop policy
and local-to-task transfer are weaker.

I also screened the existing `effect32_scene_film` checkpoint. Its offline
goal-use was effectively the same as `effect32_film` (`goal-shuffle L2 0.063`,
max goal sensitivity 0.035), but the learned closed-loop policy was weaker:
`0.590` success learned, `0.655` oracle, and `0.280` shuffled over 200 episodes.
That makes it a regression from `effect32_film`, so I skipped D_phi/R3 for it.

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

I then added an offline `rl-rerun fit-closed-loop-selector` audit. It fits a
ridge linear selector from matched closed-loop frozen/residual outcome labels and
validates on a separate matched bank. With only initial, deployable-at-episode
features (`initial action delta`, `initial policy saturation`, `initial goal
L2`), the selector was weak:

| selector features | train seed | validation seed | frozen | residual | selector | discordant AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial only | 4600000 | 4700000 | 0.290 | 0.290 | 0.310 | 0.537 |

The non-deployable episode-summary upper bound on the same banks was much
stronger:

| selector features | train seed | validation seed | frozen | residual | selector | discordant AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full episode summary | 4600000 | 4700000 | 0.290 | 0.290 | 0.390 | 0.909 |

This sharpens the diagnosis: outcome information exists in the rollout
trajectory, but not enough in first-decision features. A useful selector likely
needs online recurrent/step-level context or must be trained as part of the
closed-loop policy, not as an episode-start switch.

I also deployed the initial-feature selector as an online step selector in
`eval-closed-loop-r3`, recomputing the same three features at every action and
falling back to the frozen action when the selector score is negative. This did
not transfer:

| seed | frozen | ungated residual | online step selector | residual action rate |
| ---: | ---: | ---: | ---: | ---: |
| 4600000 | 0.260 | 0.290 | 0.290 | 0.807 |
| 4700000 | 0.290 | 0.290 | 0.270 | 0.804 |

So the current learned linear selector is not a useful online controller. The
selector path should now move beyond episode-level outcome fitting: either train
a policy/selector directly online, or change the RL objective first.

As an upper-bound selector diagnostic, I added `--oracle-segment-selector` to
the closed-loop eval. At each high-level replan it copies the simulator state,
rolls frozen and tuned for one held-goal segment, and executes the branch whose
counterfactual final latent is closer to the current held goal. This is not
real-compatible, but it tests whether perfect local same-state branch selection
would help.

Matched 20-seed slices:

| seed | frozen | ungated residual | oracle segment selector | selector residual rate |
| ---: | ---: | ---: | ---: | ---: |
| 4600000 | 0.350 | 0.400 | 0.350 | 0.572 |
| 4700000 | 0.100 | 0.150 | 0.200 | 0.537 |

The two-slice aggregate ties ungated residual (`0.275`) and remains only above
frozen (`0.225`). Even a privileged one-segment latent-distance oracle is not a
robust selector for task success here. This points back to the objective: the
tuned branch needs to create a larger, more task-aligned effect, not merely be
selected by a sharper local latent-distance heuristic.

I then extended the oracle segment selector with
`--oracle-segment-selector-metric env_reward`, which chooses the tuned branch
when its counterfactual one-segment terminal normalized dense reward exceeds
the frozen branch. The first matched 100-episode check at `seed_start=4800000`
was mixed:

| policy | success | final reward | max reward | residual action rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.230 | 0.4147 | 0.4376 | 0.000 |
| ungated residual | 0.270 | 0.4177 | 0.4537 | 1.000 |
| task-reward oracle selector | 0.250 | 0.4208 | 0.4505 | 0.493 |

Scaling the same check to 500 matched episodes changed the sign:

| policy | success | final reward | max reward | residual action rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.312 | 0.4727 | 0.4960 | 0.000 |
| ungated residual | 0.298 | 0.4553 | 0.4850 | 1.000 |
| task-reward oracle selector | 0.322 | 0.4753 | 0.5047 | 0.483 |

A fresh 500-episode window at `seed_start=4900000` weakened but did not erase
the selector benefit over ungated residual:

| policy | success | final reward | max reward | residual action rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.304 | 0.4532 | 0.4859 | 0.000 |
| ungated residual | 0.284 | 0.4404 | 0.4727 | 1.000 |
| task-reward oracle selector | 0.302 | 0.4563 | 0.4852 | 0.470 |

Across these two 500-episode windows, the task-reward oracle selector is
approximately tied with frozen on success (`0.312` vs `0.308`) and slightly
better on final reward (`0.4658` vs `0.4630`), while ungated residual is clearly
harmful (`0.291` success). So one-segment task reward is a better oracle branch
selector than latent distance, mostly because it suppresses harmful residual
interventions. This is still not deployable: it requires counterfactual
simulator rollouts from the current state. Treat it as evidence that
task-aligned closed-loop branch selection can reduce harm, not as a policy
candidate.

I added per-replan trace output for the oracle segment selector so this
non-deployable diagnostic can generate labels for a future deployable selector.
Each residual rollout with `--oracle-segment-selector` now records the oracle
choice, both branch outcomes, and online prefix features available before the
current action. A 20-episode smoke at `seed_start=5000000` produced 162 trace
rows with matched decision count, `0.45` frozen/residual success, and selector
residual action rate `0.454`. The trace artifact is:

```text
results/rl_rerun/local_r3/n500/seed0/task_reward_debug_n4096_1update_bc1_lr1e5_logstd5/closed_loop_oracle_segment_selector_envreward_trace_20_seed5000000.json
```

I also added a deployable trace-fitted segment selector path:
`fit-oracle-segment-selector` fits a linear selector from oracle trace labels,
and `eval-closed-loop-r{1,2,3} --segment-selector` scores it once per replan
using the same prefix features. A train-20/valid-20 smoke did not validate the
simple prefix feature set: validation AUC was `0.507`, accuracy `0.505`, and
selected one-segment reward was still below the oracle by `0.00123`. The
20-episode closed-loop validation was noisy-positive (`0.15 -> 0.20` success),
but this is not credible given the chance-level oracle-imitation metric.

I then widened the non-deployable oracle segment selector metric from terminal
one-segment dense reward to also support one-segment max dense reward and
success-within-segment. On a 20-episode smoke at `seed_start=5200000`,
`env_max_reward` selected residual for `46.8%` of actions and had positive
counterfactual branch deltas (`+0.0093` max reward, `+0.013` success-once), but
closed-loop task success stayed tied with frozen (`0.45`) and max reward was
slightly lower. The sparse `success` metric selected residual `0%` of the time
on the same smoke. This suggests the currently available one-segment task
signals are still too weak/sparse as an upper-bound selector. A 500-episode
`env_max_reward` validation at `seed_start=4800000` confirmed that diagnosis:
the selector reached `0.294` success versus frozen `0.306`, with final reward
delta `-0.0087` and max reward delta `-0.0095`, despite positive one-segment
counterfactual deltas inside the selector trace. Optimizing one-segment max
reward is therefore not a stronger upper-bound branch selector for this
checkpoint. A matching 500-episode `success` selector was essentially inactive
(`0.36%` residual action rate), tying frozen success at `0.306` with negligible
negative reward deltas. The sparse success branch signal also does not provide
a useful upper bound.

To make local-to-task proxy checks less indirect, I added
`eval-local-r{1,2,3} --include-samples`, which exports per-sample local
distances, dense rewards, success flags, and action deltas under
`sample_metrics`. A 512-env one-entry smoke on the task-reward debug checkpoint
validated the format and gave another weak proxy signal: local raw-distance
improvement versus final dense-reward improvement had Pearson correlation
`0.052`, versus max-reward improvement `0.0019`, and success deltas were nearly
balanced (`-1: 11`, `0: 488`, `+1: 13`). This supports the current diagnosis
that local raw reachability deltas are a poor deployment proxy for this
checkpoint.

I then extended the same local sample export with optional `D_phi` distances
via `--reachability-checkpoint`. On the same 512-env smoke, the checkpoint
improved raw distance (`+0.0135`) but worsened learned reachability distance
(`-0.0091`), while final dense reward still improved slightly (`+0.0079`).
Per-sample correlations with final dense-reward delta were weak for both raw
distance (`0.052`) and `D_phi` (`0.017`), and raw-vs-`D_phi` improvement
correlation was only `0.139`. This makes the local proxy mismatch directly
measurable on identical reset samples.

I added `rl-rerun audit-local-sample-proxies` to turn these sample exports into
repeatable proxy audits. On the 512-sample D_phi smoke, raw-distance delta had
success-improvement AUC `0.469` and `D_phi` delta had AUC `0.483`; both are
below chance on the small discordant-success subset. Initial difficulty
features were more predictive (`initial_distance` AUC `0.643`), suggesting that
where an intervention is attempted may matter more than the current raw or
learned local-distance delta.

I then ran the same audit on the contrasting task-hard `bc=0.3` one-update
checkpoint. On the same 512-sample bank, it had a stronger task signal than the
task-reward-debug checkpoint (`+0.0106` final reward, `+0.0195` success-once),
but raw local distance again did not explain success deltas well: raw AUC was
`0.333`, while `D_phi` AUC was `0.597`. Dense-reward correlations remained weak
for both (`0.091` raw, `0.061` D_phi). This suggests the same-sample audit can
distinguish checkpoint behavior, but neither local proxy is currently strong
enough to trust alone.

Scaling that task-hard audit to the full 4096-env validation bank changed the
sign of the aggregate local result: final reward delta was `-0.0024`, raw
distance delta `-0.0071`, and `D_phi` delta `-0.0042`, with success almost tied
(`+0.0015`). Raw and `D_phi` success AUCs were only weakly above chance
(`0.562` and `0.526`). So the 512-bank task-hard positive was not stable; the
larger same-sample proxy audit agrees with the broader conclusion that this
checkpoint is not a robust promotion candidate.

I also ran the full 4096-env same-sample audit for the task-reward-debug
checkpoint to make the comparison symmetric. It was also slightly negative:
final reward delta `-0.0020`, max reward delta `-0.0008`, success delta
`-0.0012`, raw distance delta `-0.0020`, and `D_phi` delta `+0.0005`. Raw and
`D_phi` success AUCs were `0.531` and `0.503`. With both candidate checkpoints
on the full bank, neither local raw L2 nor learned `D_phi` provides a reliable
promotion signal. I added `rl-rerun compare-local-proxy-audits` to keep these
comparisons reproducible; the first full-bank comparison ranks task-hard higher
by success delta, max-reward delta, and best proxy AUC, but both candidates fail
the positive final-task-signal gate because final dense reward is negative.
The comparison artifact is
`results/rl_rerun/local_r3/n500/seed0/local_proxy_audit_comparison_n4096_taskreward_vs_taskhard.json`.

I then added `D_phi` as an actual `train-local-r3` reward distance via
`--reward-distance-metric reachability --reachability-checkpoint ...`, not just
as an evaluation metric. A one-update 4096-env D_phi-reward smoke was stable and
produced the first positive full-bank local task gate in this group: final
reward delta `+0.0026`, max reward delta `+0.0005`, success delta `+0.0007`,
raw-distance delta `+0.0002`, and `D_phi` delta `+0.0035`. However, learned-goal
closed-loop validation did not transfer: on 500 episodes at `seed_start=4800000`
the tuned policy reached `0.302` success versus frozen `0.306`, with final
reward delta `-0.0046` and max reward delta `-0.0052`. This is still useful:
`D_phi` is a better local reward than raw/task-hard variants under the full-bank
proxy audit, but the resulting update is tiny and not yet a deployment
improvement. I then tried the smallest stronger-effect variant, lowering
`bc_weight` from `1.0` to `0.3` while keeping the same one-update D_phi reward
setup. That failed the full-bank local gate: final reward delta `-0.0034`, max
reward delta `-0.0032`, success delta `-0.0049`, and `D_phi` delta `-0.0018`.
So simple BC weakening does not solve the D_phi effect-size problem.
Extending the stable `bc=1` D_phi reward setup to three updates kept the
full-bank local gate positive (`+0.0024` final reward, `+0.0009` max reward,
`+0.0012` success, `+0.0025` D_phi reduction) and doubled the closed-loop
residual norm, but still failed the 500-episode learned-goal promotion check:
success stayed at `0.302` versus frozen `0.306`, with max reward delta
`-0.0032`. More D_phi updates improve local metrics but still do not create a
robust deployment improvement.
I then tested paired D_phi terminal improvement (`reward_mode=paired` with
`reward_distance_metric=reachability`). It failed the full-bank local promotion
gate: final reward delta `-0.0008` and D_phi delta `-0.0063`, even though max
reward and success deltas were slightly positive. The training paired D_phi
improvement was also negative (`-0.0037`, `48.8%` improved), so this checkpoint
was not promoted to closed-loop validation.
I also ran goal-use diagnostics across frozen, D_phi progress, and paired-D_phi
checkpoints. They are effectively identical: goal-block shuffle changes actions
by only `0.046-0.047` L2 while observation shuffle is about `0.81`, and valid
same-state future-goal swaps remain around `0.023` L2 for `k=2` versus `k=10`.
So D_phi reward changes local scalar outcomes without fixing the underlying
goal-conditioning bottleneck.

I then added a `task_paired` local-R3 reward mode. It reuses the cached frozen
same-state rollout, but compares terminal ManiSkill dense reward instead of
terminal latent distance:

```text
r_terminal = tuned_terminal_env_reward - frozen_terminal_env_reward
```

A one-update terminal-only diagnostic (`bc=1`, `lr=1e-5`, `logstd=-5`,
`dense_progress_weight=0`) was runnable but weak:

| metric | value |
| --- | ---: |
| train task-paired improvement | 0.00150 |
| train fraction task-improved | 0.398 |
| train terminal env reward | 0.4803 |
| train frozen terminal env reward | 0.4788 |
| matched local final distance | 0.6036 |
| matched local action delta L2 | 0.00042 |
| matched local task success-once fraction | 0.331 |

The matched local final distance is still worse than frozen (`0.6020`) and only
slightly better than the previous paired+sensitivity local result (`0.6047`).
An 8-episode deployability smoke was negative: frozen success `0.625` versus
task-paired residual `0.375`, with final reward `0.6862 -> 0.5536`. This is not
a promotion candidate. It is useful mostly as infrastructure and evidence that
one-segment terminal dense reward alone is still too weak/noisy under the
current local-R3 update.

I then added `--min-base-terminal-distance` to `rl-rerun train-local-r3` so
paired/task-paired updates can train only on local starts where the frozen
branch ends far from the held goal. A one-update hard-start task-paired run
with `min_base_terminal_distance=0.6` selected `38.9%` of samples and improved
the in-training terminal reward delta versus the uniform task-paired smoke:

| metric | uniform task-paired | hard-start task-paired |
| --- | ---: | ---: |
| train task-paired improvement | 0.00150 | 0.00469 |
| train fraction task-improved | 0.398 | 0.451 |
| matched local final distance | 0.6036 | 0.6093 |
| matched local reduction | 0.4635 | 0.4578 |
| matched local action delta L2 | 0.00042 | 0.00054 |
| matched local task success-once fraction | 0.331 | 0.329 |

This is a useful negative target-regime check: hard-start masking makes the
training signal less noisy, but it still does not transfer to the matched local
validation bank and remains below frozen (`0.6020` final distance,
`0.4651` reduction). I skipped closed-loop deployment for this checkpoint.

I then added a task-difficulty filter,
`--max-base-terminal-env-reward`, for `task_paired` local R3. This selects starts
where the frozen same-state segment has low terminal ManiSkill dense reward,
which is closer to the task-paired target than latent terminal distance. With
`max_base_terminal_env_reward=0.45`, the one-update training signal was much
stronger:

| metric | uniform | latent-hard | task-hard |
| --- | ---: | ---: | ---: |
| active fraction | 1.000 | 0.389 | 0.763 |
| train task-paired improvement | 0.00150 | 0.00469 | 0.03577 |
| train fraction task-improved | 0.398 | 0.451 | 0.522 |
| matched local final distance | 0.6036 | 0.6093 | 0.6092 |
| matched local reduction | 0.4635 | 0.4578 | 0.4578 |
| matched local task success-once fraction | 0.331 | 0.329 | 0.333 |

This is informative but still negative. Task-hard filtering gives a real
in-training task-reward improvement, but the learned update does not transfer to
the held-out local validation bank and remains worse than frozen on latent
distance reduction. I also skipped closed-loop deployment for this checkpoint.

I also reran the same task-hard setup with a weaker BC anchor (`bc_weight=0.3`).
Because the training metrics are collected before the PPO update in this
one-update diagnostic, the rollout-side task-paired metrics match the `bc=1`
run. The resulting checkpoint did improve over task-hard `bc=1` on matched local
validation, but still did not beat frozen:

| policy | matched local final distance | reduction | action delta L2 | task success-once |
| --- | ---: | ---: | ---: | ---: |
| frozen n500 previous baseline | 0.6020 | 0.4651 | - | - |
| uniform task-paired bc1 | 0.6036 | 0.4635 | 0.00042 | 0.331 |
| task-hard bc1 | 0.6092 | 0.4578 | 0.00046 | 0.333 |
| task-hard bc0.3 | 0.6037 | 0.4634 | 0.00046 | 0.337 |

So lower BC recovers most of the latent-distance regression from task-hard
filtering and slightly improves local task-success diagnostics, but it remains a
tiny policy change and below the frozen local-distance baseline. I skipped
closed-loop deployment.

I then extended `eval-local-r3` to run the frozen low-level branch from the same
held-out local starts before evaluating the tuned checkpoint. This gives direct
base-vs-tuned local deltas and subset summaries for the task-hard filter
(`base_final_env_reward <= 0.45`) and latent-hard filter
(`base_final_distance >= 0.6`). On the same validation manifest:

| policy | all reward delta | all success delta | task-hard reward delta | task-hard success delta | latent-hard reward delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform task-paired | -0.0024 | -0.0024 | +0.0273 | +0.0149 | +0.0018 |
| task-hard bc1 | -0.0024 | +0.0005 | +0.0304 | +0.0200 | -0.0045 |
| task-hard bc0.3 | +0.0009 | +0.0037 | +0.0314 | +0.0225 | -0.0054 |

This narrows the diagnosis: the task-hard target is not pure noise. It improves
terminal task reward on the held-out starts it was meant to target. But the
gain is small, action changes remain tiny, latent reduction worsens, and the
latent-hard subset loses task reward. This is still not a deployment candidate;
it is evidence that local task-reward filtering can shape a very small
task-specific correction but not a robust low-level improvement.

I then ran the best-looking task-hard setup for three PPO updates instead of
one (`bc=0.3`, `max_base_terminal_env_reward=0.45`). Training did not scale the
target signal much: task-paired improvement stayed around `0.036-0.038`,
fraction improved stayed near `0.52`, and action deltas remained tiny. Enriched
held-out validation showed a tradeoff shift rather than a promotion:

| policy | all reward delta | all success delta | task-hard reward delta | task-hard success delta | latent-hard reduction delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| task-hard bc0.3 1 update | +0.0009 | +0.0037 | +0.0314 | +0.0225 | +0.0528 |
| task-hard bc0.3 3 updates | -0.0034 | -0.0037 | +0.0293 | +0.0159 | +0.0625 |

Longer training increased action delta only slightly (`0.00046 -> 0.00067`)
and improved latent-hard reduction, but it gave back task reward and success on
the target task-hard subset. This reinforces that repeatedly optimizing the same
one-segment task-paired target changes the local tradeoff; it still does not
produce a robust low-level improvement.

I then ran learned-goal closed-loop transfer checks for the best one-update
task-hard checkpoint (`bc=0.3`, `max_base_terminal_env_reward=0.45`) from
`seed_start=4800000`. A compact 100-episode smoke was mildly positive, but the
larger 500-episode matched window reversed the sign:

| episodes | branch | success | final reward | max reward |
| ---: | --- | ---: | ---: | ---: |
| 100 | frozen | 0.230 | 0.4147 | 0.4376 |
| 100 | task-hard residual | 0.250 | 0.4240 | 0.4549 |
| 100 | delta | +0.020 | +0.0093 | +0.0172 |
| 500 | frozen | 0.312 | 0.4727 | 0.4960 |
| 500 | task-hard residual | 0.304 | 0.4635 | 0.4931 |
| 500 | delta | -0.008 | -0.0093 | -0.0029 |

The 100-episode improvement was therefore a small-window false lead. The
task-hard local objective does shape its targeted local subset, but it still
does not validate as a closed-loop policy improvement.

I then ran the same task-hard checkpoint under oracle high-level goals. The
first 200-episode oracle check was mildly positive, but the matched 500-episode
window was neutral:

| goal source | episodes | frozen success | residual success | success delta | final reward delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned | 500 | 0.312 | 0.304 | -0.008 | -0.0093 |
| oracle | 200 | 0.355 | 0.375 | +0.020 | +0.0185 |
| oracle | 500 | 0.372 | 0.372 | +0.000 | +0.0006 |

So the task-hard residual is not simply blocked by learned high-level goal
quality. Oracle goals raise the frozen ceiling and remove most of the damage,
but the tuned low level still does not produce a robust improvement. This points
back to the local objective/effect size rather than high-level prediction as the
main blocker for this checkpoint.

I then checked whether the task-hard target at least fixed the low-level
goal-conditioning bottleneck. It did not. On the same 4096-sample diagnostic
used for the D_phi checkpoints, the task-hard residual's condition-block
sensitivity was essentially unchanged from frozen: observation shuffle action
L2 was `0.8363`, goal shuffle was `0.0450`, previous-action shuffle was
`0.0729`, and remaining-time shuffle was `0.0000`. Valid same-state future-goal
swaps were also unchanged: `k=2` versus `k=10` action L2 was `0.020706` for
task-hard versus `0.020705` for frozen, despite mean latent goal separation
`25.32`. This closes the strongest task-hard local objective as a
goal-identifiability fix. It can shape small terminal task-reward deltas, but
it does not make the deployed low-level policy materially more goal-sensitive.

I also tested whether the strong non-deployable full-episode summary selector
could be made deployable by using online prefix approximations of its features
(`action_delta_l2` mean/max so far, saturation rate so far, goal-L2 mean so far,
and high-level decisions so far). The evaluator now accepts these cumulative
features in `--step-selector`.

Matched `num_envs=20`, 100-episode windows:

| seed | frozen | ungated residual | initial-step selector | prefix-summary selector |
| ---: | ---: | ---: | ---: | ---: |
| 4600000 | 0.310 | 0.320 | 0.400 | 0.320 |
| 4700000 | 0.340 | 0.340 | 0.320 | 0.350 |
| mean | 0.325 | 0.330 | 0.360 | 0.335 |

The prefix-summary selector only gives `+0.010` over frozen and does not recover
the offline full-summary upper bound. The initial-step selector looks better in
this matched `num_envs=20` check, but the same selector was previously neutral
or worse with `num_envs=64`. Treat that as vectorization-sensitive diagnostic
evidence, not a robust online selector. The broader conclusion still holds:
linear selectors over simple online features are not enough; a real selector
would need to be trained/evaluated in the closed-loop intervention distribution.

I then ran a larger fresh 500-episode validation at `seed_start=4800000` for
the prefix-summary selector and a matched ungated eval. It rejected the small
positive 100-episode signal:

| policy | success | final reward | max reward | residual action rate |
| --- | ---: | ---: | ---: | ---: |
| frozen | 0.306 | 0.4642 | 0.4954 | 0.000 |
| ungated task-reward residual | 0.298 | 0.4585 | 0.4871 | 1.000 |
| prefix-summary step selector | 0.292 | 0.4538 | 0.4830 | 0.806 |

So online prefix selection does not rescue this checkpoint. It uses the
residual branch most of the time and is slightly worse than simply deploying the
ungated residual on this fresh window. This closes the current linear
online-selector branch for task-reward-debug R3; the next selector attempt needs
training in the closed-loop intervention distribution, not another linear gate
fit from retrospective outcome labels.

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
   raw reduction offline but failed online. A closed-loop outcome selector using
   only initial deployable features also barely validated, while a non-deployable
   full-episode summary selector was strong. Further gate work should therefore
   use online step/recurrent context or train a selector/policy directly in the
   closed-loop distribution; offline local segment deltas and initial switches
   are not enough. A sim-privileged one-segment oracle selector also tied
   ungated residual on two matched slices, so local latent-distance branch
   choice alone is unlikely to be the missing ingredient.

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
   transfer check. A direct task-reward debug upper bound also failed transfer,
   and a terminal task-paired reward mode produced only tiny local action
   changes with a negative 8-episode deployability smoke. Hard-start masking for
   task-paired local R3 increased the training reward delta but made matched
   local validation worse than both frozen and the uniform task-paired smoke. A
   task-reward hard filter produced a much stronger training reward delta, but
   it also failed matched local validation. Weakening BC for that task-hard
   target recovered some local validation performance but still stayed below
   frozen with tiny action changes. Targeted held-out subset validation shows
   task-hard local R3 does improve the task-hard subset's terminal task reward,
   but only slightly and with a latent-hard tradeoff. Extending the best
   task-hard setup to three updates increases latent-hard reduction but reduces
   task-hard reward/success gains.
   The next objective check should change the target regime, not simply scale
   the same formulation: move toward a stronger deployment-aligned signal than
   one-segment local reachability or local task reward alone.
   The privileged direct hard-start check shows that large selected-local gains
   can still hurt closed-loop deployment, so checkpoint selection needs
   deployment evidence or a better local-to-task proxy.

3. Revisit representation only after the gate/objective question.
   The current effect32 interface is goal-dependent, so the main bottleneck is
   not simply "low level ignores the goal"; it is reliable improvement without
   damaging already-good frozen behavior. Oracle serial goals and nearest-train
   goal projection show that learned high-level goal quality matters, but simple
   projection only gives a tiny frozen gain and does not combine well with R3.
   A direct short-horizon check with k=5 and k=2 was worse than the current k=10
   hierarchy, so horizon shortening is not the next lever unless paired with a
   different training distribution or objective.

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
results/incremental/goal_diagnostics/n500/seed0/effect32_film/diagnostics.json
artifacts/incremental/learned_interface/effect32_film_h5/seed0/hierarchy.pt
results/incremental/learned_interface/effect32_film_h5/seed0/learned_hierarchy_eval_200_seed3500000.json
results/incremental/learned_interface/effect32_film_h5/seed0/oracle_hierarchy_eval_200_seed3500000.json
results/incremental/goal_diagnostics/n500/seed0/effect32_film_h5/diagnostics.json
artifacts/incremental/learned_interface/effect32_film_h2/seed0/hierarchy.pt
results/incremental/learned_interface/effect32_film_h2/seed0/learned_hierarchy_eval_200_seed3500000.json
results/incremental/goal_diagnostics/n500/seed0/effect32_film_h2/diagnostics.json
results/incremental/goal_diagnostics/gate_report.json
results/incremental/goal_diagnostics/gate_report.md
artifacts/incremental/learned_interface/effect32_film_gsens/seed0/hierarchy.pt
results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens/diagnostics.json
results/incremental/learned_interface/effect32_film_gsens/seed0/learned_hierarchy_eval_200_seed3500000.json
results/incremental/learned_interface/effect32_film_gsens/seed0/oracle_hierarchy_eval_200_seed3500000.json
artifacts/incremental/learned_interface/effect32_film_gsens_light/seed0/hierarchy.pt
results/incremental/goal_diagnostics/n500/seed0/effect32_film_gsens_light/diagnostics.json
results/incremental/learned_interface/effect32_film_gsens_light/seed0/learned_hierarchy_eval_200_seed3500000.json
results/incremental/learned_interface/effect32_film_gsens_light/seed0/oracle_hierarchy_eval_200_seed3500000.json
artifacts/incremental/learned_interface/effect32_goal_residual/seed0/hierarchy.pt
artifacts/incremental/learned_interface/effect32_goal_residual/seed0/hierarchy_metrics.json
results/incremental/learned_interface/effect32_goal_residual/seed0/learned_hierarchy_eval_20.json
results/incremental/learned_interface/effect32_goal_residual/seed0/oracle_hierarchy_eval_20.json
results/incremental/goal_diagnostics/n500/seed0/effect32_goal_residual/diagnostics.json
artifacts/incremental/learned_interface/effect64_film/seed0/hierarchy.pt
artifacts/incremental/learned_interface/effect64_film/seed0/hierarchy_metrics.json
results/incremental/learned_interface/effect64_film/seed0/learned_hierarchy_eval_200_seed3500000.json
results/incremental/learned_interface/effect64_film/seed0/oracle_hierarchy_eval_200_seed3500000.json
results/incremental/goal_diagnostics/n500/seed0/effect64_film/diagnostics.json
artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/latest.pt
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/history.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_hard06_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/latest.pt
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/history.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/latest.pt
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/history.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_100_seed4800000.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/closed_loop_learned_500_seed4800000.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc1_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_1update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json
artifacts/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/latest.pt
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/history.json
results/rl_rerun/local_r3/n500/seed0/task_paired_terminal_taskhard045_n4096_3update_bc03_lr1e5_logstd5/eval_local_n4096_val_b1_manifest_with_base.json
results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_hierarchy_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_base_clean_n1800_oracle_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_hierarchy_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_residual_alpha025_n1800_oracle_seed9900000_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_hierarchy_200eps.json
results/hcl_next_phase1/privileged_z_closed_loop_direct_paired_hardmse005_oracle_200eps.json
data/manifests/privileged_z_branch_counterfactuals_dense2000_seed9963000_q128_k8_prefix.npz
artifacts/incremental/privileged_z_branch_selector/hcl_next_counterfactual_q128_k8_prefix_seed0.pt
```
