#!/bin/bash

# 1. Create the directories in your home folder
mkdir -p $HOME/.enroot/data $HOME/.enroot/cache $HOME/.enroot/runtime

# 2. Export the variables so the system knows where to look
export ENROOT_DATA_PATH=$HOME/.enroot/data
export ENROOT_CACHE_PATH=$HOME/.enroot/cache
export ENROOT_RUNTIME_PATH=$HOME/.enroot/runtime

# 3. Submit the job, EXPLICITLY passing these variables
echo "Submitting job with ENROOT paths set to $HOME/.enroot..."
sbatch --export=ALL,ENROOT_DATA_PATH=$ENROOT_DATA_PATH,ENROOT_CACHE_PATH=$ENROOT_CACHE_PATH,ENROOT_RUNTIME_PATH=$ENROOT_RUNTIME_PATH cluster/submit_job.sh
