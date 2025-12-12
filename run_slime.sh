#!/bin/bash

# Configuration
# Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" # Or local path
DATA_DIR="."
TRAIN_DATA="$DATA_DIR/train_data.jsonl"
TEST_DATA="$DATA_DIR/test_data.jsonl"
OUTPUT_DIR="output_grpo"

# WandB Configuration
export WANDB_API_KEY="fe5c554ccf23a3cf456386d1759680f75166adb5"
export WANDB_PROJECT="grpo-trader"

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
    --use-hf-config-for-megatron \
    --num-layers 24 \
    --hidden-size 896 \
    --num-attention-heads 14 \
    --seq-length 2048 \
    --max-position-embeddings 32768 \
    --advantage-estimator grpo \
    --prompt-data $TRAIN_DATA \
    --input-key prompt \
    --metadata-key metadata \
    --custom-rm-path grpo_trader.slime_adapter.reward:reward_func \
    --lr 1e-6 \
    --num-rollout 100 \
    --rollout-batch-size 4 \
    --n-samples-per-prompt 4 \
    --global-batch-size 16 \
    --num-steps-per-rollout 1 \
    --rollout-max-response-len 512 \
    --save $OUTPUT_DIR \
    --use-wandb \
    --wandb-project $WANDB_PROJECT \
    --eval-interval 10 \
    --eval-prompt-data test_split $TEST_DATA \
    --n-samples-per-eval-prompt 4
