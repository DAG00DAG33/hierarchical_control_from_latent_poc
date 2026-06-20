# Push-T Future-Latent Hierarchy POC

This repository tests whether a reachable future latent state is a useful
interface between high- and low-level imitation policies on ManiSkill
`PushT-v1`. The final experiment compares direct visual policies, an exact
local-oracle hierarchy, and hierarchies with learned deterministic or
generative high levels.

The result separates two claims:

1. **Interface:** A low-level policy can use a reachable future latent. This is
   supported by the exact local-oracle hierarchy.
2. **Deployable hierarchy:** A learned high level can predict a sufficiently
   control-compatible future latent. This is not supported by the current
   implementation.

The full gated plan is in
[pusht_incremental_experiment_plan.md](pusht_incremental_experiment_plan.md),
and every experiment and failed intervention is recorded in
[INCREMENTAL_EXPERIMENT_LOG.md](INCREMENTAL_EXPERIMENT_LOG.md). The earlier
prototype is retained in [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).

## Method

The environment uses `pd_ee_delta_pos` control at 20 Hz. Observations contain
frozen `facebook/dinov2-small` spatial RGB features and non-privileged robot
proprioception. The selected Phase 6 representation is a 256D
reconstruction-only autoencoder latent:

```text
z_t = E_o(DINO(rgb_t), proprio_t)
```

No object pose labels are used to train the encoder. World-model-only,
world-model plus reconstruction, AE, VAE, and multiple latent dimensions were
tested before selecting this representation. Held-out pose probes are
diagnostics only.

At each step, the low level receives the current latent, previous action, and
a future-latent displacement two control steps (0.10 s) ahead:

```text
a_t = pi_low(z_t, g_t - z_t, a_{t-1})
```

Three sources of `g_t` are compared:

- **Exact local oracle:** Copy the current student simulator state, roll the
  privileged teacher for two steps, then encode the reached observation.
- **Deterministic high level:** Predict the absolute future latent from `z_t`.
- **Generative high level:** Conditional flow matching in future-latent space;
  the reported policy uses its zero-noise endpoint.

The oracle branch is regenerated from the student's current state every step.
It never uses a nominal precomputed trajectory after the student deviates.
State-copy, action, transition, RGB, DINO-feature, and latent parity were
validated before evaluating the interface.

## Setup

Dependencies are managed with `uv`; training and simulator evaluation require
CUDA.

```bash
uv sync --python 3.11
uv run hcl-poc doctor
```

The final incremental configuration is
[`configs/pusht_incremental.yaml`](configs/pusht_incremental.yaml).

## Data

The original downloaded demonstrations did not replay successfully under the
installed simulator/controller combination. A privileged PPO teacher was
therefore trained using the same downstream action space. The prepared HDF5
dataset contains 2,000 successful causal trajectories with frozen DINO
features, proprioception, simulator state for diagnostics, and teacher
actions. The final 200 trajectories are a fixed validation set; training uses
nested prefixes of the first 1,800.

To rebuild the incremental teacher dataset and its basic checks:

```bash
CONFIG=configs/pusht_incremental.yaml

uv run hcl-poc incremental phase0 --config "$CONFIG"
uv run hcl-poc incremental phase1-collect --config "$CONFIG"
uv run hcl-poc incremental phase1-train --config "$CONFIG"
uv run hcl-poc incremental phase1-eval --config "$CONFIG"
```

Subsequent phase commands are listed by `uv run hcl-poc incremental --help`.
They are intentionally gated; follow the plan and log when reproducing the
representation and oracle diagnostics from scratch.

## Final Sweep

Each Phase 12 budget retrains visual BC, visual action flow, AE-256, the oracle
low level, deterministic high level, and generative high level in an isolated
artifact directory. No checkpoint is transferred between data budgets.

```bash
CONFIG=configs/pusht_incremental.yaml

for N in 50 100 200 500 1000 1800; do
  uv run hcl-poc incremental phase12-run \
    --config "$CONFIG" \
    --n-trajectories "$N" \
    --episodes 100 \
    --eval-seed-start 1200000
done

uv run hcl-poc incremental phase12-plot --config "$CONFIG"
```

All deployable methods use one policy seed and the same 100 evaluation seeds.
Exact branch replay is much slower and uses 10 of those seeds per budget. The
plot reports binomial standard errors. This reduced protocol does not measure
training-seed robustness.

## Results

| Trajectories | Transitions | Visual BC | Flat flow | Oracle hierarchy | Deterministic hierarchy | Generative hierarchy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 2,311 | 0.00 | 0.03 | 0.00 | 0.01 | 0.03 |
| 100 | 4,507 | 0.05 | 0.07 | 0.10 | 0.02 | 0.05 |
| 200 | 8,834 | 0.10 | 0.20 | 0.40 | 0.08 | 0.03 |
| 500 | 22,367 | 0.29 | 0.28 | 0.80 | 0.14 | 0.23 |
| 1000 | 44,605 | 0.44 | 0.49 | 0.80 | 0.22 | 0.25 |
| 1800 | 80,472 | 0.60 | 0.62 | 0.70 | 0.37 | 0.42 |

![Success versus causal training transitions](docs/results/incremental_sample_efficiency.png)

The plotted values and protocol metadata are available as
[`docs/results/incremental_sample_efficiency.json`](docs/results/incremental_sample_efficiency.json).

The exact oracle first reaches 50% measured success at 22,367 transitions;
both direct visual policies first reach it at 80,472. This supports the
future-state interface as a useful temporal abstraction, but the oracle uses a
privileged teacher online and is not deployable. Its 10-episode points also
have substantially wider uncertainty.

The learned hierarchy does not inherit the oracle's data-efficiency advantage.
At 1,800 trajectories, deterministic and generative hierarchies reach 0.37 and
0.42 success, below visual BC at 0.60 and flat flow at 0.62. Detailed Phase
8-10 diagnostics show that future-latent prediction remains the dominant
bottleneck: the predicted goals have moderate offline error but induce
compounding closed-loop action error. Robust low-level training on measured
high-level errors did not recover performance.

The main deployable sample-efficiency hypothesis is therefore negative. The
oracle-interface result is positive and identifies a narrower next research
problem: learn compact, physically aligned future goals without losing the
low-level control advantage.

Large datasets, checkpoints, detailed raw result JSON, and videos remain local
and are ignored by Git.
