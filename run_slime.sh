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

python3 -m slime.train \
    --model_name_or_path $MODEL_PATH \
    --method grpo \
    --prompt_data $TRAIN_DATA \
    --input_key prompt \
    --metadata_key metadata \
    --custom_rm_path grpo_trader.slime_adapter.reward:reward_func \
    --learning_rate 1e-6 \
    --num_rollouts 100 \
    --rollout_batch_size 4 \
    --n_samples_per_prompt 4 \
    --max_new_tokens 512 \
    --output_dir $OUTPUT_DIR \
    --zero_stage 2 \
    --gradient_checkpointing \
    --report_to wandb \
    --eval_interval 10 \
    --eval_prompt_data test_split $TEST_DATA \
    --n_samples_per_eval_prompt 4
