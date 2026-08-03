#!/usr/bin/env bash
#SBATCH --job-name=prepare-dolci-it-only
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Experiment D (Q3): Tokenize IT-only Dolci data (Dolci-Instruct-SFT minus Tool-Use).
#
# Two steps:
#   1. Filter: remove Tool-Use IDs from Dolci-Instruct-SFT → parquet file
#   2. Tokenize: convert parquet to OLMo-core format (same pipeline as Experiment A)
#
# Prerequisites:
#   - Both allenai/Dolci-Instruct-SFT and allenai/Dolci-Instruct-SFT-Tool-Use
#     must be cached in HF_HOME (download once with online mode first).
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_dolci_it_only_data.sh

set -euo pipefail

# --- Paths: keep everything on $WORK, not $HOME ---
WORK_DIR="${WORK}/ytahtah0"
mkdir -p "$WORK_DIR"

# HuggingFace caches → $WORK
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORK_DIR}/.cache/huggingface/datasets"

# Only set offline AFTER you've downloaded everything once with online mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /leonardo/home/userexternal/ytahtah0/open-instruct/.venv/bin/activate

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-/leonardo/home/userexternal/ytahtah0/open-instruct}"
FILTERED_PARQUET="${WORK_DIR}/data/dolci_it_only.parquet"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/dolci_it_only_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Dolci IT-Only Data Preparation (Experiment D) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Tokenizer: $TOKENIZER"
echo "Filtered parquet: $FILTERED_PARQUET"
echo "Output: $OUTPUT_DIR"
echo "======================================================"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

# Step 1: Filter out Tool-Use samples
if [ -f "$FILTERED_PARQUET" ]; then
    echo "Filtered parquet already exists at $FILTERED_PARQUET, skipping filter step."
else
    echo "Step 1: Filtering Dolci-Instruct-SFT to remove Tool-Use samples..."
    python scripts/data/filter_dolci_it_only.py \
        --output_path "$FILTERED_PARQUET"
fi

# Step 2: Tokenize (same pipeline as Experiment A)
echo "Step 2: Tokenizing IT-only data..."
python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$FILTERED_PARQUET" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
    --max_seq_length 32768 \
    --visualize True \
    --resume \
    --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
