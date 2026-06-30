#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_ROOT="results/incremental/rl_reachability_debug"
DATASET="data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5"
BASE_TRUE="$RESULT_ROOT/run6_true_tcp_b8_u1000/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt"

variant_ckpt() {
  local dir="$1"
  printf "%s/%s/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt" "$RESULT_ROOT" "$dir"
}

run_eval_pair() {
  local name="$1"
  local ckpt="$2"
  local local_out="$RESULT_ROOT/run7_${name}_local_policy_compare_b8_ref2.json"
  local full_out="$RESULT_ROOT/run7_${name}_full_success_100.json"
  uv run python scripts/rl_reachability_tcp_local_policy_compare.py \
    --dataset "$DATASET" \
    --run2-low "$BASE_TRUE" \
    --run5-low "$ckpt" \
    --eval-refs 2 \
    --include-shuffled \
    --output "$local_out" \
    > "${local_out%.json}.log" 2>&1
  uv run python scripts/rl_reachability_tcp_full_success_eval.py \
    --episodes 100 \
    --num-envs 10 \
    --goal-sources oracle learned \
    --run2-low "$BASE_TRUE" \
    --run5-low "$ckpt" \
    --output "$full_out" \
    > "${full_out%.json}.log" 2>&1
}

uv run python scripts/summarize_reachability_reward_variants.py \
  --output "$RESULT_ROOT/run7_reward_variants_summary.json"

run_eval_pair terminal "$(variant_ckpt run7_dpsi_terminal_b8_u1000)"
run_eval_pair progress "$(variant_ckpt run7_dpsi_progress_b8_u1000)"
run_eval_pair bc_advantage "$(variant_ckpt run7_dpsi_bc_advantage_b8_u1000)"
