#!/bin/bash

# Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" # Or local path
DATA_PATH="train_data.jsonl"
OUTPUT_DIR="output_grpo"

# Generate data if not exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Generating data..."
    python3 -m grpo_trader.slime_adapter.gen_data --output "$DATA_PATH"
fi

# Run Slime Training
# Note: This assumes you are in the Slime docker container or have Slime installed
# and 'train.py' is in the python path or current directory.
# We use 'python3 -m slime.train' if installed as package, or point to the script.

# Assuming we are running from the root of the repo and Slime is installed
# We need to pass the custom reward function path

echo "Starting GRPO Training with Slime..."

python3 -m slime.train \
    --model_name_or_path $MODEL_PATH \
    --method grpo \
    --prompt_data $DATA_PATH \
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
    --gradient_checkpointing
