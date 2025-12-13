#!/bin/bash

# Configuration
# Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" # Or local path
DATA_DIR="$(pwd)"
TRAIN_DATA="$DATA_DIR/train_data.jsonl"
TEST_DATA="$DATA_DIR/test_data.jsonl"
OUTPUT_DIR="output_grpo"

# WandB Configuration
export WANDB_PROJECT="grpo-trader"

# Ray Configuration
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

# Generate data if not exists
# Generate data if not exists or empty
# Always regenerate to ensure format is correct
echo "Regenerating data to ensure correct format..."
rm -f "$TRAIN_DATA" "$TEST_DATA"
python3 -m grpo_trader.slime_adapter.gen_data --output_dir "$DATA_DIR" || exit 1

echo "Checking data files..."
ls -l "$TRAIN_DATA" "$TEST_DATA"
wc -l "$TRAIN_DATA" "$TEST_DATA"

# Run Slime Training
echo "Starting GRPO Training with Slime..."

# Ensure PYTHONPATH includes current directory for custom modules
export PYTHONPATH=$PYTHONPATH:.

python3 Slime/train.py \
    --model-name $MODEL_PATH \
    --hf-checkpoint $MODEL_PATH \
    --train-backend fsdp \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-num-gpus 0 \
    --colocate \
    --advantage-estimator grpo \
    --prompt-data $TRAIN_DATA \
    --input-key prompt \
    --metadata-key metadata \
    --custom-rm-path grpo_trader.slime_adapter.reward.reward_func \
    --lr 1e-6 \
    --num-rollout 1000 \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 2 \
    --global-batch-size 4 \
    --num-steps-per-rollout 1 \
    --num-steps-per-rollout 1 \
    --rollout-max-response-len 512 \
    --rollout-max-prompt-len 8192 \
    --rollout-max-context-len 16384 \
    --sglang-mem-fraction-static 0.5 \
    --save $OUTPUT_DIR \
    --save-interval 50 \
    --use-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-group "grpo-experiment" \
    --eval-interval 10 \
    --eval-prompt-data test_split $TEST_DATA \
    --n-samples-per-eval-prompt 4
