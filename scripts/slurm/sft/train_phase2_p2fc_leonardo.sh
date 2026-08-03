#!/usr/bin/env bash
#SBATCH --job-name=phase2-p2fc-fc-sft
#SBATCH --nodes=4
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

# Phase 2 Experiment P2-FC: FC-only SFT with full curated FC mix (588,937 samples).
#
# Dolci(203,035) + GS(23,427) + v1-dedup(200,310) + TxT360-AMS(162,165).
# Full curated FC ceiling. FC-only recipe. Compare against all add-one
# experiments and K' (backward ablation for TxT360).
# FC-only recipe: LR 5e-5, batch 256K, 1 epoch, 4 nodes.
#
# Prerequisites:
#   1. Prepare tokenized data: sbatch prepare_phase2_p2fc_data.sh
#   2. OLMo-core checkpoint must be available
#
# Usage:
#   mkdir -p logs
#   sbatch train_phase2_p2fc_leonardo.sh

set -euo pipefail

# --- Paths ---
WORK_DIR="${WORK}/ytahtah0"
OLMOCORE_PATH="${OLMOCORE_PATH:-/leonardo/home/userexternal/ytahtah0/OLMo-core}"
DATASET_PATH="${DATASET_PATH:-${WORK_DIR}/data/phase2_p2fc_tokenized}"
BASE_CKPT="${BASE_CKPT:-${WORK_DIR}/models/OLMo-3-7B-Think-SFT-olmocore/model_and_optim}"

# Training config — FC-only recipe
RUN_NAME="${RUN_NAME:-phase2_p2fc_fc_sft}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEQ_LEN="${SEQ_LEN:-32768}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-262144}"  # 256K tokens
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

echo "=== Phase 2 P2-FC: Full Curated FC Mix SFT Training (Leonardo) ==="
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
echo "Max epochs: ${MAX_EPOCHS}"
echo "====================================================================="

export TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/Olmo-3-7B-SFT-slurm.py"

mkdir -p "$SAVE_FOLDER"

export GPUS_PER_NODE MASTER_ADDR MASTER_PORT
export TRAIN_SCRIPT RUN_NAME BASE_CKPT WORK_DIR DATASET_PATH
export SEQ_LEN GLOBAL_BATCH_SIZE LEARNING_RATE MAX_EPOCHS SAVE_FOLDER

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
    --max_epochs="$MAX_EPOCHS" \
    --save_folder="$SAVE_FOLDER"'

echo "=== Training complete ==="
echo "Checkpoints saved to: $SAVE_FOLDER"
