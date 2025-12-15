#!/bin/bash

# 1. Create the directories in your home folder
mkdir -p $HOME/.enroot/data $HOME/.enroot/cache $HOME/.enroot/runtime

# 2. Create Enroot Configuration File (More robust than env vars)
mkdir -p $HOME/.config/enroot
echo "Creating $HOME/.config/enroot/enroot.conf..."
cat > $HOME/.config/enroot/enroot.conf <<EOF
ENROOT_DATA_PATH $HOME/.enroot/data
ENROOT_CACHE_PATH $HOME/.enroot/cache
ENROOT_RUNTIME_PATH $HOME/.enroot/runtime
EOF

# 3. Export variables just in case
export ENROOT_DATA_PATH=$HOME/.enroot/data
export ENROOT_CACHE_PATH=$HOME/.enroot/cache
export ENROOT_RUNTIME_PATH=$HOME/.enroot/runtime

# 4. Submit the job
echo "Submitting job..."
sbatch --export=ALL cluster/submit_job.sh
