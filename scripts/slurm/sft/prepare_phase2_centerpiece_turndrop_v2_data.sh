#!/usr/bin/env bash
#SBATCH --job-name=prepare-phase2-centerpiece-turndrop-v2
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Phase 2 Wave 1 v2 (clean 2-stage labels) centerpiece_turndrop_v2: tokenize the
# 5-dataset centerpiece with unjustified turn-range drop (4 categories:
# ANTI_MANUAL_SOLVE, ANTI_UNJUSTIFIED_REFUSAL, ANTI_PRESSURE_CAVE,
# OTHER_UNJUSTIFIED) applied to v1, v2.IA, TxT360 using the corrected
# fcanalysis-clean labels. 891,441 samples (1,385 fully-dropped; 5,800 modified
# in place). dolci + graphsyn pass through unfiltered. Built by
# fcanalysis/scripts/prepare_v2_components.py + prepare_centerpiece_v2_data.py.
#
# Walltime bumped to 36h (vs the v1 centerpieces' 24h); --resume as backstop.
#
# Prerequisites:
#   - Dolci-format JSONL at $PHASE2_DIR/centerpiece_turndrop_v2.jsonl
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_phase2_centerpiece_turndrop_v2_data.sh

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
CENTERPIECE_JSONL="${PHASE2_DIR}/centerpiece_turndrop_v2.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/phase2_centerpiece_turndrop_v2_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Phase 2 centerpiece_turndrop_v2: 5-dataset Centerpiece (clean-label turn-drop) FC Data Preparation ==="
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
