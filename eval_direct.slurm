#!/bin/bash
#SBATCH --job-name=eval-direct
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --partition=training
#SBATCH --container-image=slimerl/slime:latest
#SBATCH --container-mounts=./:/workspace

# Setup
mkdir -p logs
cd /workspace

# Environment
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONUNBUFFERED=1
# We likely don't need Slime PYTHONPATH for this, but keeping it doesn't hurt if we use shared utils
export PYTHONPATH=$PYTHONPATH:$(pwd)/Slime:$(pwd)/Megatron-LM:$(pwd)

# Install dependencies
pip install yfinance pandas --break-system-packages

# SGLang direct evaluation
MODEL_PATH="local_ckpt/hf_model_iter_0000500"
TEST_DATA="test_data.jsonl"
OUTPUT_DIR="eval_results_direct"

echo "Starting Direct Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data: $TEST_DATA"

python3 scripts/eval_direct.py \
    --model-path $MODEL_PATH \
    --test-data $TEST_DATA \
    --output-dir $OUTPUT_DIR \
    --tp-size 1

echo "Evaluation complete."
