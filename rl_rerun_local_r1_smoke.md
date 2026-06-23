# RL Rerun Local R1 Smoke

Date: 2026-06-23

This is the first residual deterministic low-level PPO run on the exact
vector-consistent Mode-A local MDP.

## Training

Command:

```bash
uv run hcl-poc rl-rerun --config configs/pusht_incremental.yaml train-local-r1 --dataset data/rl_rerun/pusht_vector_state_demos_n512_b1.h5 --n-demo 1000 --seed 0 --run-name smoke_32k --steps 262144 --alpha 0.1 --terminal-weight 1.0
```

Checkpoint:

```text
artifacts/rl_rerun/local_r1/n1000/seed0/smoke_32k/latest.pt
```

Tracked history:

```text
rl_rerun_local_r1_smoke_262k_history.json
```

## Recipe

| item | value |
| --- | ---: |
| method | R1 residual deterministic low-level PPO |
| local mode | Mode A, stored reachable `z_{t+10}` |
| vector envs | 512 |
| rollout steps | 64 |
| samples/update | 32768 |
| minibatch size | 4096 |
| updates | 8 |
| total samples | 262144 |
| residual scale `alpha` | 0.1 |
| residual penalty | 0.01 |
| terminal distance weight | 1.0 |

Training reward:

```text
d_t - d_{t+1} - terminal_weight * d_final - residual_penalty
```

The trainer does not use ManiSkill reward, task success, object pose, or
hand-designed task progress as training signals.

## Training Metrics

| metric | first update | last update |
| --- | ---: | ---: |
| mean terminal distance | 1.233 | 1.079 |
| mean distance | 1.557 | 1.390 |
| mean residual norm | 0.0159 | 0.0161 |
| action saturation rate | 0.071 | 0.034 |
| approx KL | 0.0076 | 0.0106 |
| clip fraction | 0.058 | 0.118 |
| value loss | 0.258 | 0.088 |
| explained variance | -0.534 | 0.449 |

Training diagnostics improved, but this did not transfer to deterministic
held-out local evaluation.

## Evaluation

Frozen and residual policies were evaluated on the same 4 sampled vector-local
resets, using 2048 local episodes total.

Tracked evaluation files:

```text
rl_rerun_local_mode_a_audit_n512_b1_seed0_eval4.json
rl_rerun_local_r1_smoke_262k_eval4.json
```

| policy | final distance | distance reduction | reduction fraction | saturation |
| --- | ---: | ---: | ---: | ---: |
| frozen BC low-level | 1.131 | 0.415 | 0.812 | 0.0219 |
| R1 residual, 262k | 1.138 | 0.408 | 0.799 | 0.0207 |

## Interpretation

This smoke run validates the exact vector local PPO machinery and produces
stable PPO metrics, but it does **not** pass the R1 local-goal improvement gate.
The deterministic residual mean remains very small at evaluation time
(`mean_residual_norm = 0.00335`), so the policy mostly reproduces the frozen
controller.

Next tuning ideas:

1. lower or remove the residual penalty for the first local PPO runs;
2. increase residual scale `alpha` to `0.25`;
3. use a larger learning rate for the residual mean;
4. evaluate during training on fixed local validation resets, because training
   terminal distance alone was optimistic here;
5. collect more than one `512`-env vector batch to reduce overfitting to a
   single reset batch.
