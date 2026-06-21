#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
config="${CONFIG:-configs/pusht_incremental.yaml}"
budgets=(50 100 200 500 1000 1800)
seeds=(0 1 2)

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
        --episodes 500 \
        --oracle-episodes 50
    fi
  done
done
