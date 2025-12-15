#!/bin/bash
#SBATCH --job-name=grpo-trader
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=8
#SBATCH --time=04:00:00
#SBATCH --partition=training
#SBATCH --container-image=slimerl/slime:latest
#SBATCH --container-mounts=./:/workspace

# Ensure logs directory exists
mkdir -p logs

# Navigate to workspace
cd /workspace

# Load secrets if present
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

export PYTHONUNBUFFERED=1

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

# Install current package in editable mode so 'grpo_trader' is found
pip install -e . --break-system-packages

# Reset Slime repo to fix any corruption from previous patches
echo "Resetting Slime repository..."
cd Slime
git checkout .
cd ..

# Patch Slime for single-GPU FSDP
echo "Patching Slime..."
python3 patch_slime.py

# Run the training script
bash run_slime.sh
