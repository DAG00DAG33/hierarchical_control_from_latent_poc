#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/pusht_incremental.yaml}
CANDIDATE=${CANDIDATE:-vae512_w2048_b1e6}
BUDGETS=${BUDGETS:-"500 1800"}
SEED=${SEED:-0}
VARIANTS=${VARIANTS:-"terminal paired progress task_mix raw_l2"}
TRAIN_STEPS=${TRAIN_STEPS:-1024000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_SEED_START=${EVAL_SEED_START:-3450000}
NUM_ENVS=${NUM_ENVS:-4096}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-10}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-64}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
INITIAL_LOGSTD=${INITIAL_LOGSTD:--1.5}
LOG_DIR=${LOG_DIR:-results/incremental/low_level_rl/scratch_high_env_reward_selection_logs}
FORCE=${FORCE:-0}

mkdir -p "$LOG_DIR"

force_flag=()
if [[ "$FORCE" == "1" ]]; then
  force_flag=(--force)
fi

run_one() {
  local n_demo=$1
  local variant=$2
  local name="scratch_high_${variant}"
  local reward_mode="absolute"
  local distance_metric="reachability"
  local terminal_weight="1.0"
  local progress_weight="0.0"
  local task_reward_weight="0.0"
  local task_progress_weight="0.0"

  case "$variant" in
    terminal)
      ;;
    paired)
      reward_mode="paired"
      ;;
    progress)
      progress_weight="0.1"
      ;;
    task_mix)
      progress_weight="0.1"
      task_reward_weight="0.05"
      ;;
    raw_l2)
      distance_metric="raw_l2"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 2
      ;;
  esac

  local prefix="${LOG_DIR}/n${n_demo}_seed${SEED}_${name}"
  uv run hcl-poc low-level-rl --config "$CONFIG" train-scratch \
    --n-demo "$n_demo" \
    --candidate "$CANDIDATE" \
    --seed "$SEED" \
    --run-name "$name" \
    --steps "$TRAIN_STEPS" \
    --terminal-weight "$terminal_weight" \
    --distance-progress-weight "$progress_weight" \
    --task-reward-weight "$task_reward_weight" \
    --task-progress-weight "$task_progress_weight" \
    --reward-mode "$reward_mode" \
    --distance-metric "$distance_metric" \
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
    --distance-metric "$distance_metric" \
    "${force_flag[@]}" \
    > "${prefix}_eval.log" 2>&1
}

for n_demo in $BUDGETS; do
  for variant in $VARIANTS; do
    run_one "$n_demo" "$variant"
  done
done
