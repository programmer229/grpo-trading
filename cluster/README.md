# Cluster Deployment Guide

This guide explains how to run the GRPO Trader on the cluster using the provided infrastructure.

## 1. Local Setup

### SSH Configuration
Add the following to your local `~/.ssh/config`:

```toml
Host bastion
    HostName 31.12.82.146
    User tufa

Host slurm
    ForwardAgent yes
    HostName 10.100.0.253
    User <your_username>
    ProxyJump bastion
```

Replace `<your_username>` with your actual cluster username.

### SSH Key
Generate a key pair if you haven't:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Send the public key (`~/.ssh/id_ed25519.pub`) to the cluster admin (Kev).

Add your key to the agent:
```bash
ssh-add ~/.ssh/id_ed25519
```

## 2. Deploying Code

SSH into the login node:
```bash
ssh slurm
```

Clone the repository (or rsync your local code):
```bash
git clone https://github.com/programmer229/grpo-trading.git
cd grpo-trading
```

## 3. Running the Job

We use `sbatch` with a Slime container to run the training.

### Submit Job
```bash
sbatch cluster/submit_job.sh
```

This script will:
1. Request 1 GPU.
2. Launch the `slimerl/slime:latest` container.
3. Mount the current directory to `/workspace`.
4. Execute `run_slime.sh`.

### Monitor Job
Check status:
```bash
squeue
```

Check logs (once running):
```bash
tail -f logs/grpo-trader-<job_id>.out
```

## 4. Interactive Session (Debugging)

If you need to debug inside the container:

1. Start an interactive session:
   ```bash
   srun --gpus=1 --pty bash --login
   ```

2. Start the container (using Enroot/Pyxis manually or just running the script if supported interactively):
   *Note: The cluster docs suggest using `sbatch` for containers, but for interactive debugging you might need to check `enroot list` as per the docs.*

   To debug the training script directly:
   ```bash
   sbatch --container-image=slimerl/slime:latest --container-mounts=./:/workspace --wrap="bash" --pty ... # (Consult cluster docs for exact interactive container syntax if srun doesn't support --container-image directly)
   ```
   
   *Alternative based on provided docs:*
   1. `srun --gpus=1 --pty bash --login`
   2. (Inside node) `enroot start ...` (See cluster docs for specific interactive container flow if `srun --container-image` isn't available).
