#!/usr/bin/env bash
#SBATCH --job-name=prepare-phase2-centerpiece-random-v2
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Phase 2 Wave 1 v2 (clean-label) centerpiece_random_v2: tokenize the 5-dataset
# centerpiece with a uniform random subsample matched to the AMS_v2 volume
# (drawn from the raw centerpiece as a whole, seed=42). 889,325 samples
# (== centerpiece_AMS_v2 count). Matched-volume control for AMS_v2/turndrop_v2.
# Built by fcanalysis/scripts/prepare_centerpiece_v2_data.py.
#
# Walltime bumped to 36h (vs the v1 centerpieces' 24h); --resume as backstop.
#
# Prerequisites:
#   - Dolci-format JSONL at $PHASE2_DIR/centerpiece_random_v2.jsonl
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_phase2_centerpiece_random_v2_data.sh

set -euo pipefail

WORK_DIR="${WORK}/ytahtah0"
mkdir -p "$WORK_DIR"

export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${WORK_DIR}/.cache/huggingface/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /leonardo/home/userexternal/ytahtah0/open-instruct/.venv/bin/activate

PROJECT_ROOT="${PROJECT_ROOT:-/leonardo/home/userexternal/ytahtah0/open-instruct}"
PHASE2_DIR="${PHASE2_DIR:-${WORK_DIR}/data/phase2_dolci_format}"
CENTERPIECE_JSONL="${PHASE2_DIR}/centerpiece_random_v2.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/phase2_centerpiece_random_v2_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Phase 2 centerpiece_random_v2: 5-dataset Centerpiece (clean-label-matched random) FC Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Centerpiece JSONL: $CENTERPIECE_JSONL"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"

if [ ! -f "$CENTERPIECE_JSONL" ]; then
    echo "ERROR: $CENTERPIECE_JSONL not found."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$CENTERPIECE_JSONL" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
    --max_seq_length 32768 \
    --visualize True \
    --resume \
    --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
