#!/bin/bash

# Configuration
# Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" # Or local path
DATA_DIR="."
TRAIN_DATA="$DATA_DIR/train_data.jsonl"
TEST_DATA="$DATA_DIR/test_data.jsonl"
OUTPUT_DIR="output_grpo"

# WandB Configuration
export WANDB_PROJECT="grpo-trader"

# Ray Configuration
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Generate data if not exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Generating data..."
    python3 -m grpo_trader.slime_adapter.gen_data --output_dir "$DATA_DIR"
fi

# Run Slime Training
echo "Starting GRPO Training with Slime..."

# Ensure PYTHONPATH includes current directory for custom modules
export PYTHONPATH=$PYTHONPATH:.

python3 Slime/train.py \
    --model-name $MODEL_PATH \
    --hf-checkpoint $MODEL_PATH \
    --train-backend fsdp \
    --distributed-backend gloo \
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
    --num-rollout 100 \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 2 \
    --global-batch-size 4 \
    --num-steps-per-rollout 1 \
    --rollout-max-response-len 512 \
    --sglang-mem-fraction-static 0.5 \
    --save $OUTPUT_DIR \
    --save-interval 50 \
    --use-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-group "grpo-experiment" \
    --eval-interval 10 \
    --eval-prompt-data test_split $TEST_DATA \
    --n-samples-per-eval-prompt 4
