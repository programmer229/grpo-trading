#!/bin/bash
#SBATCH --job-name=download-assets
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --partition=training
#SBATCH --container-image=slimerl/slime:latest
#SBATCH --container-mounts=./:/workspace

mkdir -p logs
cd /workspace
export PYTHONUNBUFFERED=1

pip install huggingface_hub --break-system-packages

python3 download_qwen4b.py
