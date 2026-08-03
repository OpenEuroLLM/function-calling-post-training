#!/usr/bin/env bash
#SBATCH --job-name=prepare-phase2-a1prime
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Phase 2 Experiment A1' (step-matched A1): Tokenize a step-matched subsample
# of the (Dolci + GraphSyn) pool, totaling 203,035 samples — exactly matching
# P1's sample count.
#
# Why this exists: A1 (Dolci 203K + GS 23K = 226K) trains for ~12% more steps
# than P1 (Dolci 203K). A1 vs P1 thus confounds "GS signal" with "extra training
# steps." A1' eliminates that confound by sub-sampling the merged pool to 203K
# total. A1' vs P1 is volume-matched, step-matched, and recipe-identical;
# only composition varies. A1 vs A1' decomposes A1's forward delta into
# "GS-specific signal" (A1' - P1) vs "additional-step contribution"
# (A1 - A1').
#
# Composition of A1' (built by scripts/data/build_a1prime_step_matched.py
# with seed=42):
#   - 182,013 Dolci samples (89.65%)
#   - 21,022 GraphSyn samples (10.35%)
#   - Total: 203,035 (= P1)
#   - Preserves the original Dolci/GS proportion (10.34%) within 0.01%.
#
# Prerequisites:
#   - Dolci-format JSONL at $PHASE2_DIR/a1prime_step_matched.jsonl
#     Generated locally by build_a1prime_step_matched.py from
#     phase2_dolci_format/{dolci,graphsyn}.jsonl, then uploaded to HF
#     and downloaded to $PHASE2_DIR.
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_phase2_a1prime_data.sh

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
PHASE2_DIR="${PHASE2_DIR:-${WORK_DIR}/data/phase2}"
A1PRIME_JSONL="${PHASE2_DIR}/a1prime_step_matched.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/phase2_a1prime_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Phase 2 A1': Step-matched (Dolci + GS) FC Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "A1' JSONL: $A1PRIME_JSONL"
echo "Output: $OUTPUT_DIR"
echo "==================================================================="

if [ ! -f "$A1PRIME_JSONL" ]; then
    echo "ERROR: $A1PRIME_JSONL not found."
    echo "Run: python scripts/data/build_a1prime_step_matched.py \\"
    echo "    --dolci <phase2_dolci_format>/dolci.jsonl \\"
    echo "    --graphsyn <phase2_dolci_format>/graphsyn.jsonl \\"
    echo "    --output $A1PRIME_JSONL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$A1PRIME_JSONL" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
    --max_seq_length 32768 \
    --visualize True \
    --resume \
    --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
