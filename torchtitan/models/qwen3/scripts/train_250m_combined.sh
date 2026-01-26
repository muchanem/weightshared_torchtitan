#!/usr/bin/bash
# Train 250M Qwen3 with combined weight sharing (attention + layer + factorized embeddings)
# This script sets up wandb logging and runs distributed training

set -ex

# ============================================================================
# Configuration
# ============================================================================

# Number of GPUs (default: 4 for this config)
NGPU=${NGPU:-4}

# Wandb configuration
export WANDB_TEAM="weight-sharing"
export WANDB_PROJECT="combined-runs"
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"250m-combined-$(date +%Y%m%d-%H%M%S)"}

# Output and checkpoint directories
# Set CHECKPOINT_DIR to override the default checkpoint location
# Checkpoints will be saved to: ${OUTPUT_DIR}/${CHECKPOINT_FOLDER}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/250m_combined"}
CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER:-"checkpoints"}

# Config file
CONFIG_FILE="torchtitan/models/qwen3/train_configs/250m_combined.toml"

# Logging rank
export LOG_RANK=${LOG_RANK:-0}

# ============================================================================
# Run training
# ============================================================================

echo "Starting 250M combined weight sharing training"
echo "  Wandb: ${WANDB_TEAM}/${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Checkpoints: ${OUTPUT_DIR}/${CHECKPOINT_FOLDER}"
echo "  GPUs: ${NGPU}"
#TORCH_USE_CUDA_DSA=1 \
#CUDA_LAUNCH_BLOCKING=1 \
#TORCH_DISTRIBUTED_DEBUG="DETAIL" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train \
    --job.config_file ${CONFIG_FILE} \
    --job.dump_folder ${OUTPUT_DIR} \
    --checkpoint.folder ${CHECKPOINT_FOLDER} \
    "$@"
