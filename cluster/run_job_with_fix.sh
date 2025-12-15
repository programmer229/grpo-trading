#!/bin/bash

# 1. Create the directories in your home folder
mkdir -p $HOME/.enroot/data $HOME/.enroot/cache $HOME/.enroot/runtime

# 2. Create Enroot Configuration File (More robust than env vars)
mkdir -p $HOME/.config/enroot
echo "Creating $HOME/.config/enroot/enroot.conf..."
# Resolve variables to absolute paths before writing
DATA_PATH="$HOME/.enroot/data"
CACHE_PATH="$HOME/.enroot/cache"
RUNTIME_PATH="/tmp/enroot-runtime-$USER"

cat > $HOME/.config/enroot/enroot.conf <<EOF
ENROOT_DATA_PATH $DATA_PATH
ENROOT_CACHE_PATH $CACHE_PATH
ENROOT_RUNTIME_PATH $RUNTIME_PATH
EOF

# 3. Export variables just in case
export ENROOT_DATA_PATH="$DATA_PATH"
export ENROOT_CACHE_PATH="$CACHE_PATH"
export ENROOT_RUNTIME_PATH="$RUNTIME_PATH"

# 4. Submit the job
echo "Submitting job..."
sbatch --export=ALL cluster/submit_job.sh
