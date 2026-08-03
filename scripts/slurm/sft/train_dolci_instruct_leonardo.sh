#!/usr/bin/env bash
#SBATCH --job-name=dolci-instruct-sft
#SBATCH --nodes=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

# Dolci Instruct SFT training on Leonardo (CINECA).
# Trains OLMo-3 7B on the Dolci-Instruct-SFT dataset (~1.8B tokens, 2 epochs).
#
# Prerequisites:
#   1. Prepare tokenized data: sbatch prepare_dolci_instruct_data.sh
#   2. Convert checkpoint:     sbatch convert_checkpoint_leonardo.sh
#   3. Clone OLMo-core with its venv set up.
#
# Usage:
#   mkdir -p logs
#   sbatch train_dolci_instruct_leonardo.sh
#
# Environment variables (optional):
#   OLMOCORE_PATH: Path to OLMo-core clone
#   DATASET_PATH: Path to tokenized dataset
#   BASE_CKPT: Path to OLMo-core native checkpoint (model_and_optim dir)
#   SAVE_FOLDER: Override checkpoint save directory
#   LEARNING_RATE: Learning rate (default: 8e-5 for Instruct SFT)
#   NUM_EPOCHS: Number of training epochs (default: 2)
#   SEQ_LEN: Sequence length (default: 32768)

set -euo pipefail

# --- Paths ---
WORK_DIR="${WORK}/ytahtah0"
OLMOCORE_PATH="${OLMOCORE_PATH:-/leonardo/home/userexternal/ytahtah0/OLMo-core}"
DATASET_PATH="${DATASET_PATH:-${WORK_DIR}/data/dolci_instruct_sft_tokenized}"
BASE_CKPT="${BASE_CKPT:-${WORK_DIR}/models/OLMo-3-7B-Think-SFT-olmocore/model_and_optim}"

# Training config (OLMo-3 paper Table 47)
RUN_NAME="${RUN_NAME:-reproduce_olmo3_instruct_sft}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1048576}"  # ~1M tokens
GPUS_PER_NODE=4
SAVE_FOLDER="${SAVE_FOLDER:-${WORK_DIR}/checkpoints/${RUN_NAME}}"

# HuggingFace offline
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# OLMo-core needs to be on PYTHONPATH
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"

# Shared filesystem flag (required by OLMo-core for local checkpointing)
export OLMO_SHARED_FS=1

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# WandB offline mode (compute nodes have no internet)
export WANDB_MODE=offline

# Activate OLMo-core venv
source "${OLMOCORE_PATH}/.venv/bin/activate"

# --- Multi-node torchrun setup ---
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT
MASTER_PORT=$((60000 + SLURM_JOB_ID % 5000))

echo "=== Dolci Instruct SFT Training (Leonardo) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "OLMo-core: ${OLMOCORE_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Base checkpoint: ${BASE_CKPT}"
echo "Save folder: ${SAVE_FOLDER}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Sequence length: ${SEQ_LEN}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE} tokens"
echo "Epochs: ${NUM_EPOCHS}"
echo "================================================"

# Get the path to the training script (same directory as this launcher).
# NOTE: BASH_SOURCE[0] points to SLURM's spool copy, not the original, so
# we use SLURM_SUBMIT_DIR (the directory where sbatch was invoked).
export TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/Olmo-3-7B-SFT-slurm.py"

mkdir -p "$SAVE_FOLDER"

# Export all variables so they're available inside srun's bash -c.
# SLURM_NODEID must be expanded per-node (not in the sbatch shell), so we
# use bash -c with single quotes to defer its expansion to each srun task.
export GPUS_PER_NODE MASTER_ADDR MASTER_PORT
export TRAIN_SCRIPT RUN_NAME BASE_CKPT WORK_DIR DATASET_PATH
export SEQ_LEN GLOBAL_BATCH_SIZE LEARNING_RATE NUM_EPOCHS SAVE_FOLDER

srun --kill-on-bad-exit=1 bash -c 'exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$SLURM_NODEID" \
    "$TRAIN_SCRIPT" train \
    "$RUN_NAME" \
    "$BASE_CKPT" \
    --root_dir="$WORK_DIR" \
    --dataset_path="$DATASET_PATH" \
    --seq_len="$SEQ_LEN" \
    --num_nodes="$SLURM_NNODES" \
    --gpus_per_node="$GPUS_PER_NODE" \
    --global_batch_size="$GLOBAL_BATCH_SIZE" \
    --lr="$LEARNING_RATE" \
    --max_epochs="$NUM_EPOCHS" \
    --save_folder="$SAVE_FOLDER"'

echo "=== Training complete ==="
echo "Checkpoints saved to: $SAVE_FOLDER"
