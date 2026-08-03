#!/usr/bin/env bash
#SBATCH --job-name=prepare-phase2-centerpiece-raw
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Phase 2 Wave 1 Experiment centerpiece_raw: Tokenize the 5-dataset centerpiece
# (no AMS filter applied). Single JSONL containing pre-concatenated + shuffled
# (dolci + graphsyn + v1_dedup + v2_interactive_agent + txt360_full) at 892,826
# samples.
#
# Built by fcanalysis/scripts/prepare_centerpiece_data.py (concat with
# SUBSAMPLE_SEED=42, SHUFFLE_SEED=42), then converted to Dolci format and
# uploaded to RfKnowledge/phase2-dolci-format.
#
# Tests the aggregate AMS-vs-Random comparison at centerpiece scale; pair with
# centerpiece_AMS and centerpiece_random. See notes/phase2_experiments_log.md
# §"Wave 1 Centerpiece Design".
#
# Prerequisites:
#   - Dolci-format JSONL at $PHASE2_DIR/centerpiece_raw.jsonl
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_phase2_centerpiece_raw_data.sh

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
CENTERPIECE_JSONL="${PHASE2_DIR}/centerpiece_raw.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/phase2_centerpiece_raw_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Phase 2 centerpiece_raw: 5-dataset Centerpiece FC Data Preparation ==="
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
