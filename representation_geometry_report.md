# Representation Geometry Report

## Question

Phase E asks whether the observation representation is a useful future-goal
space, rather than only a compact current-state input. The key tests are local
smoothness, physical distance alignment, nearest-neighbor consistency,
interpolation, and closed-loop control.

This report compares:

- raw spatial DINO plus proprioception (`6549D`);
- deterministic reconstruction AE (`256D`);
- weakly regularized reconstruction VAE (`256D`, posterior mean).

All geometry metrics use the same 12,000-state Phase 6 probe corpus. Pairwise
correlations use 20,000 fixed random pairs. Nearest-neighbor metrics use 500
queries against 3,000 references after per-dimension standardization.

## Geometry Results

| Metric | Raw DINO+prop | AE-256 | VAE-256 |
| --- | ---: | ---: | ---: |
| Pair latent distance vs object XY | 0.627 | 0.704 | **0.713** |
| Pair latent distance vs object yaw | 0.587 | 0.636 | **0.657** |
| Pair latent distance vs TCP XY | **0.796** | 0.701 | 0.673 |
| Pair latent distance vs teacher action | 0.579 | **0.628** | 0.621 |
| One-step latent change vs TCP motion | **0.889** | 0.834 | 0.885 |
| One-step latent change vs action effort | 0.848 | 0.880 | **0.898** |
| Decoded interpolation linear MSE | n/a | 0.0556 | **0.0459** |

Values in the first six rows are Spearman correlations. The VAE has somewhat
better object-distance and local-motion geometry, while raw DINO retains the
strongest direct TCP geometry. The AE aligns best with pairwise teacher-action
distance.

## Nearest-Neighbor Results

| Metric | Raw DINO+prop | AE-256 | VAE-256 |
| --- | ---: | ---: | ---: |
| Object XY error | **4.60 mm** | 4.87 mm | **4.34 mm** |
| Object yaw error | 0.0283 rad | 0.0260 rad | **0.0234 rad** |
| TCP XY error | 9.53 mm | **8.84 mm** | 8.86 mm |
| Teacher-action MAE | 0.1007 | **0.0671** | 0.0774 |
| Contact match | 92.6% | **94.8%** | 93.8% |

All representations preserve local physical state reasonably well. AE-256 is
the best neighborhood for control action and contact, despite not having the
smoothest aggregate geometry.

## Existing Probe and Control Evidence

Both AE-256 and VAE-256 pass the Phase 6 static probes. Their object position
MAE is approximately 2.5-2.8 mm, object-yaw MAE approximately 0.05 rad, and
contact AUROC above 0.99. Those probes do not distinguish them strongly.

Closed-loop control does:

| Representation | Matched latent BC success |
| --- | ---: |
| AE-256 | **0.53** |
| VAE-256 | 0.37 |

The VAE's smoother distance and interpolation metrics do not translate into
better control. It should not replace AE-256 solely to make the latent look
more regular.

## Goal-Interface Implication

Phase B showed that the useful oracle information is mainly future robot/TCP
motion, not object state alone. Phase C then found a stronger factorized
interface: rich current state plus a compact future TCP endpoint, trained over
multiple remaining offsets and held for 0.5 seconds.

That interface avoids requiring the AE latent to simultaneously be:

- a complete current-state encoding;
- a future prediction target;
- a meaningful subtraction space;
- and a low-level motor waypoint.

The successful `k=10,U=10` structured TCP oracle reaches 0.81 success over 100
episodes. This is stronger evidence than the modest AE/VAE geometry
differences.

## Decision

1. Keep AE-256 as the rich current-state compression baseline because it has
   the strongest established latent control and action-neighborhood behavior.
2. Do not select VAE-256 based on static smoothness; its control result is
   materially worse.
3. Use a factorized future TCP/effect code for the high-level/low-level
   interface instead of raw AE subtraction.
4. Retain raw spatial DINO plus proprioception as the direct flat reference and
   as an input to a learned compact effect encoder.
5. Evaluate learned high-level predictions in physical TCP error and projected
   low-level action error, not latent L2 alone.

Machine-readable results are under `docs/results/pre_rl/phase_e/`.
