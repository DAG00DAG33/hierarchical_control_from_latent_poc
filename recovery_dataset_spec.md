# Phase D Recovery Dataset Specification

## Purpose

This dataset is the causal, simulation-side analogue of imperfect
teleoperation. A deterministic privileged PPO teacher controls Push-T, short
action perturbations are executed, and the teacher subsequently acts from the
state actually reached. Simulator state is never restored to a nominal
trajectory.

## Source Corpus

- Environment: `PushT-v1`
- Backend: canonical CUDA/PhysX
- Controller: `pd_ee_delta_pos`
- Control frequency: 20 Hz
- Episodes: 1,000
- Steps per episode: 100
- Reset seeds: 1,800,000 through 1,800,999
- Bursts per episode: uniformly 1 through 3
- Raw path: `data/incremental/pre_rl_phase_d_recovery_1000.h5`
- Raw SHA-256: `d80f45c457b4c6cc87cb2e4572bcab492e7db89d0e7c934b3b7b7c442db9aa4c`

Every timestep stores RGB `(128,128,3)`, 31D simulator state, 21D
proprioception, executed action, deterministic teacher action queried at the
same visited state, reward, success, perturbation metadata, and recovery
metadata.

`executed_actions` and `teacher_actions` have different meanings:

- `executed_actions` is the causal behavior action. During a burst it contains
  the deliberately imperfect command.
- `teacher_actions` is the deterministic recovery-policy query at that same
  state. It is the supervised target for learning desired recovery behavior.

Training code must select one label view explicitly. It must never silently
mix them.

## Perturbations

Four families are sampled approximately uniformly:

| ID | Family | Parameters |
| ---: | --- | --- |
| 1 | Correlated directional bias | 5%, 10%, or 20% of action range; fixed direction plus AR noise |
| 2 | Action hold | Previous executed command |
| 3 | Action delay | Teacher command delayed by 1, 2, or 3 steps |
| 4 | Action scaling | Scale 0.7 or 1.3 |

Burst duration is 2, 4, or 8 steps. Per-episode `bursts/` tables preserve the
start, end, type, direction, and scalar parameters. Per-step arrays preserve
the active burst ID and boundaries.

## Recoverability Audit

The corpus contains 2,035 bursts. A burst counts as recovered if the teacher
returns to the success region after its end and before the next burst or
episode timeout.

| Family | Bursts | Recovery rate | Mean successful recovery time |
| --- | ---: | ---: | ---: |
| Directional bias | 539 | 50.6% | 19.2 steps |
| Action hold | 520 | 37.1% | 14.3 steps |
| Action delay | 491 | 40.7% | 15.6 steps |
| Action scaling | 485 | 56.3% | 18.5 steps |
| **All** | **2,035** | **46.1%** | - |

Eight-step hold and delay are the hardest retained settings, at 24.4% and
31.1% recovery. None of the four families is almost never recoverable, so all
remain in the training corpus. There are 9,532 perturbation steps and 55,726
post-burst recovery steps.

## Frozen Visual Features

- Path: `data/incremental/pre_rl_phase_d_recovery_dino_1000.h5`
- SHA-256: `011d8a216a61e9ca38b23008bb9e19657f05ec3f36f9addc63d573747632ac3a`
- Encoder: `facebook/dinov2-small`
- Feature type: spatial, pooled with `spatial_pool=4`
- Feature dimension: 6,528
- Proprioception dimension: 21

Features were extracted once from the stored RGB. The prepared file retains
state, both action-label views, perturbation masks, recovery masks, rewards,
and success alongside DINO features.

## Equal-Budget Manifests

- Path: `data/incremental/pre_rl_phase_d_manifests.h5`
- SHA-256: `554a1f4761cee80992c90a4ee0b780d8e4cd4c0b915791ca74a11ed525b1d4d4`
- Sampling seed: 1,810,000
- Training budget: 80,000 transitions per variant

| Variant | Clean transitions | Off-nominal transitions | Total |
| --- | ---: | ---: | ---: |
| `clean` | 80,000 | 0 | 80,000 |
| `mixed_25` | 60,000 | 20,000 | 80,000 |
| `mixed_50` | 40,000 | 40,000 | 80,000 |
| `recovery_heavy` | 30,000 | 50,000 | 80,000 |

Off-nominal means a perturbation-active or post-burst recovery-active
transition. Recovery training uses episodes 0-799. Episodes 800-999 form the
fixed 20,000-transition recovery validation set. The clean validation set is
the same final 200 clean episodes used by the existing Phase 4 experiments
(8,969 transitions).

Each manifest stores `(episode_index, timestep)` pairs rather than duplicating
2.5 GB of features. Variants may share source transitions, but every method
within one variant must use exactly the same manifest.

## Intended Comparisons

For each manifest, train cleanly initialized flat and hierarchical methods
with identical samples. Primary recovery experiments use the teacher-query
label view. The executed-action view is a separate behavior-cloning ablation
that measures the consequence of imitating imperfect operator commands.

Report clean success, disturbed success, recovery success/time, nominal reward
loss, and the flat-to-hierarchy and oracle-to-learned gaps. Development uses
100 episodes or fewer; larger budgets are reserved for the final selected
comparison.

## Matched Hierarchy Manifests

The factorized hierarchy requires uninterrupted future windows. A second
manifest therefore applies the same causal semantics while requiring every
recovery sample to remain inside `recovery_active` from the current step
through the 10-step future TCP endpoint.

- Path: `data/incremental/pre_rl_phase_d_hierarchy_manifests.h5`
- SHA-256: `cbf65f61c7e7dd6d33d6cb7ca0a584108d8cb183e4a2259ade116d75036243d9`
- Sampling seed: 1,815,000
- Horizon: 10 control steps (0.50 s)
- Training budget: 60,000 current-state queries per variant
- Validation: 6,969 clean and 8,149 coherent recovery queries

| Variant | Clean queries | Coherent recovery queries | Total |
| --- | ---: | ---: | ---: |
| `clean` | 60,000 | 0 | 60,000 |
| `mixed_25` | 45,000 | 15,000 | 60,000 |

Both the direct flat policy and factorized hierarchy use these exact
`(episode_index, timestep)` references. On recovery samples, the previous
action is the causally executed action, the supervised action is the
deterministic teacher query at the same state, and the hierarchy's future TCP
goal comes from the same uninterrupted teacher recovery trajectory.

## Matched Development Result

All entries below use one policy seed and 100 fixed evaluation episodes. The
disturbed schedule is generated once for all episodes, independent of vector
batch size.

| Method | Training data | Clean success | Disturbed success | Recovery success |
| --- | --- | ---: | ---: | ---: |
| Flat visual BC | clean | 0.48 | 0.44 | 0.31 |
| Flat visual BC | mixed-25 | 0.45 | 0.44 | 0.36 |
| Factorized TCP hierarchy | clean | 0.45 | 0.43 | 0.41 |
| Factorized TCP hierarchy | mixed-25 | 0.35 | 0.36 | 0.34 |

Recovery data improves the flat policy's recovery rate by five percentage
points without improving disturbed success, while it degrades the hierarchy
on all three closed-loop metrics. Better mixed-hierarchy offline errors
(`0.0341 m` endpoint L2 versus `0.0407 m`) therefore do not predict better
closed-loop behavior. Clean data remains the selected training distribution.
