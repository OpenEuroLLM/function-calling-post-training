#!/usr/bin/env bash
#SBATCH --job-name=eval-exp-d-v2
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

# Experiment D (v2): IT-only SFT — trimmed olmo3:adapt × 3 runs.
#
# Model: Dolci-Instruct-SFT minus Tool-Use (~1.9M IT-only samples, 2 epochs, step 2346).
# Checkpoint: dolci_it_only_sft_hf (converted from OLMo-core).
#
# v2: Trimmed task list (10 benchmarks, ~112 leaf tasks). Removed OMEGA, AIME,
#     LiveCodeBench, IFBench to fit within 12h. New output dir to avoid task
#     index conflicts with v1 results.
#
# 3 runs for variance estimation.
#
# Usage:
#   cd /leonardo/home/userexternal/ytahtah0/open-instruct/scripts/slurm/sft
#   mkdir -p logs
#   sbatch --cpus-per-task=32 eval_exp_d_v2.sh

set -uo pipefail

SCRIPT_DIR="/leonardo/home/userexternal/ytahtah0/open-instruct/scripts/slurm/sft"
WORK_DIR="${WORK}/ytahtah0"
OLMES_PATH="/leonardo/home/userexternal/ytahtah0/olmes"
MODEL_PATH="${WORK_DIR}/models/dolci_it_only_sft_hf"
OUTPUT_BASE="${WORK_DIR}/eval-results/exp_d_it_only_v2"

# Offline — all artifacts must be pre-cached
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_CACHE_DIR="${WORK_DIR}/.cache/tiktoken"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${OLMES_PATH}/.venv/bin/activate"

echo "=== Experiment D (v2): IT-only SFT — trimmed olmo3:adapt × 3 runs ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Model: $MODEL_PATH"
echo "Output base: $OUTPUT_BASE"
echo "Tasks: 10 benchmarks (~112 leaf tasks)"
echo "Max length: 32768 (via --model-args)"
echo "GPUs: 4 (tensor parallel, auto-detected)"
echo "====================================================================="

# Trimmed task list: 10 benchmarks (~112 leaf tasks).
# Removed from v1: omega, aime:2024, aime:2025, ifbench, livecodebench_codegeneration
TASKS=(
    "popqa::olmo3:adapt"
    "gsm8k::olmo3:adapt"
    "ifeval::olmo3:adapt"
    "gpqa::olmo3:adapt"
    "mmlu:cot::olmo3:adapt"
    "bbh:cot::olmo3:adapt"
    "agi_eval_english::olmo3:adapt"
    "minerva_math::olmo3:adapt"
    "codex_humanevalplus::olmo3:adapt"
    "mbppplus::olmo3:adapt"
)

FAILED_COUNT=0

for RUN in 1 2 3; do
    OUTPUT_DIR="${OUTPUT_BASE}/run_${RUN}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "--- Run ${RUN}/3 ---"
    echo "Output: $OUTPUT_DIR"
    echo "Start time: $(date)"

    if python "${SCRIPT_DIR}/olmes_eval_wrapper.py" \
        --model "$MODEL_PATH" \
        --model-type vllm \
        --model-args '{"trust_remote_code": true, "max_length": 32768, "gpu_memory_utilization": 0.8, "enforce_eager": true}' \
        --task "${TASKS[@]}" \
        --batch-size auto \
        --output-dir "$OUTPUT_DIR"; then
        echo "Run ${RUN} complete: $(date)"
    else
        echo "WARNING: Run ${RUN} failed with exit code $? at $(date)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "=== All 3 runs attempted ==="
echo "Results saved to: $OUTPUT_BASE/run_{1,2,3}"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "WARNING: ${FAILED_COUNT} run(s) had failures (some tasks may have completed)."
    exit 1
fi
