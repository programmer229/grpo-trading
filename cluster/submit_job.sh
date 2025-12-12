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

# Install dependencies if needed (though Slime container should have most)
# pip install -r grpo_trader/requirements.txt # If you had extra deps

# Run the training script
bash run_slime.sh
