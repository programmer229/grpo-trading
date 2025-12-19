#!/bin/bash
#SBATCH --job-name=eval-grpo-trader
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --partition=training
#SBATCH --container-image=slimerl/slime:latest
#SBATCH --container-mounts=./:/workspace

# 1. Setup Environment
mkdir -p logs
cd /workspace

if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Environment variables
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export HF_HOME="/workspace/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0 # Disable just in case
mkdir -p "$HF_HOME"

# Install dependencies
echo "Installing dependencies..."
pip install yfinance pandas --break-system-packages

# Install Slime (Clone to get train.py)
if [ ! -d "Slime" ]; then
    git clone https://github.com/THUDM/Slime.git
fi
pip install -e Slime --break-system-packages

# Install Megatron-LM (Required dependency)
if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git
fi
export PYTHONPATH=$PYTHONPATH:$(pwd)/Megatron-LM

# Install local package
pip install -e . --break-system-packages

# Patch Slime
echo "Patching Slime..."
python3 patch_slime.py

# Debug: Print python path and script content
echo "PYTHONPATH: $PYTHONPATH"

# 2. Key Configurations
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
CHECKPOINT_PATH="output_grpo/iter_0001000"
TEST_DATA="test_data.jsonl"

echo "Starting Evaluation..."
echo "Model: $MODEL_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data: $TEST_DATA"

# 3. Run Evaluation Script
# Using Slime arguments to ensure model loads correctly
# Note: we use --num-rollout 0 or similar to just init, but 'generate' is called manually in script.
# Actually, the script creates the manager itself. We just need to pass the config.

python3 scripts/eval_and_trade.py \
    --model-name $MODEL_PATH \
    --hf-checkpoint $MODEL_PATH \
    --load $CHECKPOINT_PATH \
    --train-backend fsdp \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-num-gpus 0 \
    --colocate \
    --prompt-data $TEST_DATA \
    --input-key prompt \
    --metadata-key metadata \
    --rollout-batch-size 8 \
    --global-batch-size 8 \
    --num-rollout 16 \
    --rollout-max-response-len 2048 \
    --sglang-mem-fraction-static 0.6 \
    --apply-chat-template \
    --sglang-disable-cuda-graph \
    --sglang-device cuda
    # So we should KEEP prompt-data so manager initializes with it.

