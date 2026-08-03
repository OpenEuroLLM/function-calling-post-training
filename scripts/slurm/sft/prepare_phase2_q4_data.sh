#!/usr/bin/env bash
#SBATCH --job-name=prepare-phase2-q4
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Phase 2 Experiment Q4: Tokenize curated Dolci FC + v1-dedup-random (385,927 samples).
#
# Q4 is the v1 analog of Q2 (the TxT360 random control). It tests AMS-targeted
# vs random removal at fixed sample volume on v1, parallel to A3 vs Q2 on TxT360.
#
#   A2  = Dolci + v1_dedup (200,310)            total 403,345
#   A2' = Dolci + v1_dedup_ams (182,892)        total 385,927   AMS-targeted removal
#   Q4  = Dolci + v1_dedup_random (182,892)     total 385,927   random removal
#
# Compare A2' vs Q4 (volume-matched, step-matched, recipe-identical; only
# the identity of the 17,418 removed v1 samples differs). Strengthens Claim 4
# ("AMS-targeted filtering > random") with multi-dataset evidence beyond
# the TxT360-only A3 vs Q2 test.
#
# Composition under random sampling (seed=42): of 17,418 AMS-flagged samples
# in v1_dedup, ~15,907 are kept in Q4 (vs A2' which keeps zero); ~166,985
# non-AMS samples kept in Q4 (vs 182,892 in A2').
#
# Prerequisites:
#   - Dolci-format JSONL at $PHASE2_DIR/{dolci,v1_dedup_random}.jsonl
#     v1_dedup_random.jsonl is built by scripts/data/build_q4_v1_random.py
#     from the Dolci-format v1_dedup.jsonl with seed=42.
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_phase2_q4_data.sh

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
DOLCI_JSONL="${PHASE2_DIR}/dolci.jsonl"
V1_RANDOM_JSONL="${PHASE2_DIR}/v1_dedup_random.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/phase2_q4_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Phase 2 Q4: Curated Dolci + v1-dedup-random FC Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Dolci JSONL: $DOLCI_JSONL"
echo "v1-dedup-random JSONL: $V1_RANDOM_JSONL"
echo "Output: $OUTPUT_DIR"
echo "==========================================================================="

for f in "$DOLCI_JSONL" "$V1_RANDOM_JSONL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$DOLCI_JSONL" 1.0 "$V1_RANDOM_JSONL" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
    --max_seq_length 32768 \
    --visualize True \
    --resume \
    --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
