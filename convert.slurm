#!/bin/bash
#SBATCH --job-name=convert-checkpoint
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --partition=training
#SBATCH --container-image=slimerl/slime:latest
#SBATCH --container-mounts=./:/workspace

# Setup
mkdir -p logs
cd /workspace

# Environment
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)/Slime:$(pwd)/Megatron-LM:$(pwd)

# Install dependencies (ensure coherence)
pip install yfinance pandas --break-system-packages
if [ ! -d "Slime" ]; then
    git clone https://github.com/THUDM/Slime.git
fi
pip install -e Slime --break-system-packages
if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git
fi
pip install -e . --break-system-packages

# Patch Slime (required for reading if patched during training? likely yes)
python3 patch_slime.py

# Configuration
# Configuration
INPUT_DIR="local_ckpt/iter_0000500"
OUTPUT_DIR="local_ckpt/hf_model_iter_0000500"
REPO_ID="Qwen/Qwen3-4B-Thinking-2507"
ORIGIN_HF="base_qwen_4b"

echo "Downloading base model assets to $ORIGIN_HF..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO_ID', local_dir='$ORIGIN_HF', allow_patterns=['*.json', '*.txt', '*.model', '*.py', '*.tiktoken'])"

echo "Converting $INPUT_DIR to $OUTPUT_DIR..."

# Generate common.pt
echo "Generating mock common.pt..."
python3 mock_common_pt.py \
    --output-dir $INPUT_DIR \
    --model-name $ORIGIN_HF

python3 Slime/tools/convert_torch_dist_to_hf.py \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --origin-hf-dir $ORIGIN_HF \
    --force

echo "Conversion complete."
