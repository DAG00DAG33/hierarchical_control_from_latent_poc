#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/pusht_incremental.yaml}
CANDIDATE=${CANDIDATE:-vae512_w2048_b1e6}
BUDGETS=${BUDGETS:-"500 1800"}
SEED=${SEED:-0}
TRAIN_STEPS=${TRAIN_STEPS:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_SEED_START=${EVAL_SEED_START:-3450000}
NUM_ENVS=${NUM_ENVS:-32}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-10}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-8}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
INITIAL_LOGSTD=${INITIAL_LOGSTD:--1.5}
LOG_DIR=${LOG_DIR:-results/incremental/low_level_rl/scratch_reward_selection_logs}
FORCE=${FORCE:-0}

mkdir -p "$LOG_DIR"

force_flag=()
if [[ "$FORCE" == "1" ]]; then
  force_flag=(--force)
fi

run_variant() {
  local n_demo=$1
  local name=$2
  local reward_mode=$3
  local terminal_weight=$4
  local progress_weight=$5
  local prefix="${LOG_DIR}/n${n_demo}_seed${SEED}_${name}"

  uv run hcl-poc low-level-rl --config "$CONFIG" train-scratch \
    --n-demo "$n_demo" \
    --candidate "$CANDIDATE" \
    --seed "$SEED" \
    --run-name "$name" \
    --steps "$TRAIN_STEPS" \
    --terminal-weight "$terminal_weight" \
    --distance-progress-weight "$progress_weight" \
    --reward-mode "$reward_mode" \
    --distance-metric reachability \
    --num-envs "$NUM_ENVS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --num-minibatches "$NUM_MINIBATCHES" \
    --update-epochs "$UPDATE_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --initial-logstd "$INITIAL_LOGSTD" \
    "${force_flag[@]}" \
    > "${prefix}_train.log" 2>&1

  uv run hcl-poc low-level-rl --config "$CONFIG" eval \
    --n-demo "$n_demo" \
    --candidate "$CANDIDATE" \
    --seed "$SEED" \
    --run-name "$name" \
    --episodes "$EVAL_EPISODES" \
    --seed-start "$EVAL_SEED_START" \
    --distance-metric reachability \
    "${force_flag[@]}" \
    > "${prefix}_eval.log" 2>&1
}

for n_demo in $BUDGETS; do
  run_variant "$n_demo" scratch_dpsi_terminal absolute 1.0 0.0
  run_variant "$n_demo" scratch_dpsi_paired paired 1.0 0.0
  run_variant "$n_demo" scratch_dpsi_progress absolute 1.0 0.1
done
