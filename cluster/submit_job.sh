#!/bin/bash
#SBATCH --job-name=grpo-trader
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=1
#SBATCH --container-image=slimerl/slime:latest
#SBATCH --container-mounts=./:/workspace

# Ensure logs directory exists
mkdir -p logs

# Navigate to workspace
cd /workspace

# Install dependencies
echo "Installing dependencies..."
pip install yfinance pandas

# Install Slime (if not present or to ensure latest)
# Using git+https to install directly
pip install git+https://github.com/THUDM/Slime.git

# Install current package in editable mode so 'grpo_trader' is found
pip install -e .

# Run the training script
bash run_slime.sh
