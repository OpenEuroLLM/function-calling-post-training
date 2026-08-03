#!/usr/bin/env bash
#SBATCH --job-name=prepare-dolci-instruct
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Converts allenai/Dolci-Instruct-SFT dataset to OLMo-core tokenized format.
#
# Usage:
#   mkdir -p logs  # Create logs directory first
#   sbatch prepare_dolci_instruct_data.sh
#
# Resume after interruption:
#   Just resubmit the same script - it will automatically resume from checkpoint.
#
# Environment variables (optional):
#   OUTPUT_DIR: Where to save tokenized data (default: ./data/dolci_instruct_sft_tokenized)
#   TOKENIZER: HuggingFace tokenizer to use (default: allenai/Olmo-3-7B-Instruct-SFT)

set -euo pipefail

# --- Paths: keep everything on $WORK, not $HOME ---
WORK_DIR="${WORK}/ytahtah0"
mkdir -p "$WORK_DIR"

# HuggingFace caches → $WORK (downloaded models + datasets from HF Hub)
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORK_DIR}/.cache/huggingface/datasets"

# Only set offline AFTER you've downloaded everything once with online mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /leonardo/home/userexternal/ytahtah0/open-instruct/.venv/bin/activate

# Configuration - adjust paths as needed
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/dolci_instruct_sft_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"

# Open-instruct's processed dataset cache → $WORK
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

# HuggingFace cache directories (optional, but recommended to set)
echo "=== Dolci Instruct Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Tokenizer: $TOKENIZER"
echo "Output: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
  --tokenizer_name_or_path "$TOKENIZER" \
  --dataset_mixer_list allenai/Dolci-Instruct-SFT 1.0 \
  --output_dir "$OUTPUT_DIR" \
  --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
  --max_seq_length 32768 \
  --visualize True \
  --resume \
  --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
