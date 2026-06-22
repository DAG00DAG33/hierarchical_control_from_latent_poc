#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
config="${CONFIG:-configs/pusht_incremental.yaml}"
read -r -a budgets <<< "${BUDGETS:-50 100 200 500 1000 1800 4000 8000}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
episodes="${EPISODES:-500}"
oracle_episodes="${ORACLE_EPISODES:-50}"
log_dir="${LOG_DIR:-results/incremental/vae512_scaling/run_logs}"
mkdir -p "$log_dir"

run_logged() {
  local label="$1"
  shift
  local log_path="$log_dir/${label}.log"
  printf 'start %s (log: %s)\n' "$label" "$log_path"
  if ! "$@" >"$log_path" 2>&1; then
    printf 'failed %s; last log lines:\n' "$label" >&2
    tail -n 40 "$log_path" >&2
    return 1
  fi
  printf 'done %s\n' "$label"
}

if [[ "$stage" != "train" && "$stage" != "eval" && "$stage" != "all" ]]; then
  printf 'Usage: %s [train|eval|all]\n' "$0" >&2
  exit 2
fi

for budget in "${budgets[@]}"; do
  for seed in "${seeds[@]}"; do
    if [[ "$stage" == "train" || "$stage" == "all" ]]; then
      run_logged "train_n${budget}_seed${seed}" \
        uv run hcl-poc incremental vae-scaling-train \
        --config "$config" \
        --n-trajectories "$budget" \
        --seed "$seed"
    fi
    if [[ "$stage" == "eval" || "$stage" == "all" ]]; then
      run_logged "eval_n${budget}_seed${seed}_d${episodes}_o${oracle_episodes}" \
        uv run hcl-poc incremental vae-scaling-eval \
        --config "$config" \
        --n-trajectories "$budget" \
        --seed "$seed" \
        --episodes "$episodes" \
        --oracle-episodes "$oracle_episodes"
    fi
  done
done
