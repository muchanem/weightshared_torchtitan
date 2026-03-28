#!/bin/bash
#SBATCH --job-name=colm_combined_sharing
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4320
#SBATCH --account=optim
#SBATCH --qos=h100_alignment_shared
#SBATCH --output=/home/sanaelotfi/combined_sharing/colm_logs/%x_%j.out
#SBATCH --error=/home/sanaelotfi/combined_sharing/colm_logs/%x_%j.err
# Usage: sbatch launch_colm.sh <config_name>
# Example: sbatch launch_colm.sh colm_300m_5_conservative

PROJECT_DIR="/home/sanaelotfi/combined_sharing"
CONFIG_NAME=$1

if [ -z "$CONFIG_NAME" ]; then
    echo "Error: No config name provided"
    echo "Usage: sbatch launch_colm.sh <config_name>"
    exit 1
fi

mkdir -p /home/sanaelotfi/combined_sharing/colm_logs

export WANDB_API_KEY=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
export WANDB_TEAM="weight-sharing"
export WANDB_PROJECT="colm-experiments"
export WANDB_RUN_NAME="$CONFIG_NAME"
export TRITON_CACHE_DIR=/checkpoint/optim/sanaelotfi/triton_cache
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Enable NCCL debug logging for 1B configs
if [[ "$CONFIG_NAME" == *"_1b_"* ]]; then
    export NCCL_DEBUG=WARN
fi

echo "Starting job at $(date)"
echo "Config: $CONFIG_NAME"
start_time=$(date +%s)

# Activate environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua_251209

cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    -m torchtitan.train --module qwen3 --config "$CONFIG_NAME" \
    || { echo "Training script failed"; exit 1; }

echo "Job completed at $(date)"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $((elapsed_time / 3600))h $(((elapsed_time % 3600) / 60))m $((elapsed_time % 60))s"
