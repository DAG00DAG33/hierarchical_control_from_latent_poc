#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5"
DPSI="artifacts/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble/tcp_dpsi_ensemble.pt"
RESULT_ROOT="results/incremental/rl_reachability_debug"

run_variant() {
  local mode="$1"
  local out_dir="$2"
  local log="$RESULT_ROOT/${out_dir}.log"
  echo "[$(date -Is)] start ${mode}" | tee -a "$RESULT_ROOT/run7_reward_variants_queue.log"
  uv run python scripts/rl_reachability_privileged_tcp_ppo.py \
    --config configs/pusht_incremental.yaml \
    --dataset "$DATASET" \
    --num-envs 4096 \
    --updates 1000 \
    --horizon 10 \
    --reward-mode "$mode" \
    --reward-distance-source dpsi \
    --dpsi-checkpoint "$DPSI" \
    --checkpoint-every-updates 250 \
    --eval-episodes 2 \
    --output-dir "$RESULT_ROOT/$out_dir" \
    --force > "$log" 2>&1
  echo "[$(date -Is)] done ${mode}" | tee -a "$RESULT_ROOT/run7_reward_variants_queue.log"
}

run_variant terminal run7_dpsi_terminal_b8_u1000
run_variant progress run7_dpsi_progress_b8_u1000
run_variant bc_advantage_terminal run7_dpsi_bc_advantage_b8_u1000
