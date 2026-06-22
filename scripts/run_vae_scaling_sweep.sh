#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
config="${CONFIG:-configs/pusht_incremental.yaml}"
read -r -a budgets <<< "${BUDGETS:-50 100 200 500 1000 1800}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
episodes="${EPISODES:-500}"
oracle_episodes="${ORACLE_EPISODES:-50}"

if [[ "$stage" != "train" && "$stage" != "eval" && "$stage" != "all" ]]; then
  printf 'Usage: %s [train|eval|all]\n' "$0" >&2
  exit 2
fi

for budget in "${budgets[@]}"; do
  for seed in "${seeds[@]}"; do
    if [[ "$stage" == "train" || "$stage" == "all" ]]; then
      uv run hcl-poc incremental vae-scaling-train \
        --config "$config" \
        --n-trajectories "$budget" \
        --seed "$seed"
    fi
    if [[ "$stage" == "eval" || "$stage" == "all" ]]; then
      uv run hcl-poc incremental vae-scaling-eval \
        --config "$config" \
        --n-trajectories "$budget" \
        --seed "$seed" \
        --episodes "$episodes" \
        --oracle-episodes "$oracle_episodes"
    fi
  done
done
