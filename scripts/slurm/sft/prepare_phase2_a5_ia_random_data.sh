#!/usr/bin/env bash
#SBATCH --job-name=prepare-phase2-a5-ia-random
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --mem=128G

# Phase 2 Wave 0 Experiment A5_ia_random: Tokenize curated Dolci FC +
# v2.interactive_agent random subsample (458,516 samples).
#
# Random-removal control for AMS filtering on v2.interactive_agent. Random
# subsample of v2_interactive_agent.jsonl to 255,481 lines (matching the AMS-filtered
# variant volume) with seed=42 — same 23,382 sample volume removed as A5_ia_ams,
# but randomly selected instead of AMS-targeted. A5_ia_ams vs A5_ia_random is
# volume-matched, step-matched, recipe-identical; the AMS-vs-random comparison
# extends Claim 4 to v2.IA.
#
# Prerequisites:
#   - Dolci-format JSONLs at $PHASE2_DIR/{dolci,v2_interactive_agent_random}.jsonl
#     v2_interactive_agent_random produced by build_random_subsample.py
#     (Q4-pattern; see notes/initial_paper_recap.md "Reproduction asymmetry note").
#
# Usage:
#   mkdir -p logs
#   sbatch prepare_phase2_a5_ia_random_data.sh

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
V2_IA_RANDOM_JSONL="${PHASE2_DIR}/v2_interactive_agent_random.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/data/phase2_a5_ia_random_tokenized}"
TOKENIZER="${TOKENIZER:-allenai/Olmo-3-7B-Instruct-SFT}"
DATASET_CACHE_DIR="${WORK_DIR}/local_dataset_cache"

echo "=== Phase 2 A5_ia_random: Curated Dolci + v2.interactive_agent-random FC Data Preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Dolci JSONL: $DOLCI_JSONL"
echo "v2.interactive_agent_random JSONL: $V2_IA_RANDOM_JSONL"
echo "Output: $OUTPUT_DIR"
echo "==============================================================================================="

for f in "$DOLCI_JSONL" "$V2_IA_RANDOM_JSONL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path "$TOKENIZER" \
    --dataset_mixer_list "$DOLCI_JSONL" 1.0 "$V2_IA_RANDOM_JSONL" 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
    --max_seq_length 32768 \
    --visualize True \
    --resume \
    --checkpoint_interval 50000

echo "=== Data preparation complete ==="
echo "Output saved to: $OUTPUT_DIR"
