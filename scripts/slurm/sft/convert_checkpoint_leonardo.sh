#!/usr/bin/env bash
#SBATCH --job-name=convert-ckpt-hf2olmocore
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

# Converts a HuggingFace checkpoint to OLMo-core native format for SFT training.
#
# Prerequisites:
#   1. Download the HF model on the login node first (compute nodes have no internet):
#        export HF_HOME="/leonardo_work/OELLM_prod2026/ytahtah0/.cache/huggingface"
#        uv run python -c "from huggingface_hub import snapshot_download; \
#          snapshot_download('allenai/OLMo-3-7B-Think-SFT', \
#          local_dir='/leonardo_work/OELLM_prod2026/ytahtah0/models/OLMo-3-7B-Think-SFT')"
#   2. Have OLMo-core cloned with its venv set up.
#
# Usage:
#   mkdir -p logs
#   sbatch convert_checkpoint_leonardo.sh
#
# Environment variables (optional):
#   HF_CKPT_PATH: Path to HF checkpoint (default: $WORK/ytahtah0/models/OLMo-3-7B-Think-SFT)
#   OUTPUT_PATH: Where to save converted checkpoint (default: $WORK/ytahtah0/models/OLMo-3-7B-Think-SFT-olmocore)
#   MODEL_ARCH: OLMo-core model architecture (default: olmo3_7b)
#   OLMOCORE_PATH: Path to OLMo-core clone (default: /leonardo/home/userexternal/ytahtah0/OLMo-core)

set -euo pipefail

# --- Paths ---
WORK_DIR="${WORK}/ytahtah0"
OLMOCORE_PATH="${OLMOCORE_PATH:-/leonardo/home/userexternal/ytahtah0/OLMo-core}"
HF_CKPT_PATH="${HF_CKPT_PATH:-${WORK_DIR}/models/OLMo-3-7B-Think-SFT}"
OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/OLMo-3-7B-Think-SFT-olmocore}"
MODEL_ARCH="${MODEL_ARCH:-olmo3_7b}"

# HuggingFace offline (model already downloaded)
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# OLMo-core on PYTHONPATH + activate venv
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
source "${OLMOCORE_PATH}/.venv/bin/activate"

echo "=== Checkpoint Conversion: HF → OLMo-core ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Input (HF): $HF_CKPT_PATH"
echo "Output (OLMo-core): $OUTPUT_PATH"
echo "Model arch: $MODEL_ARCH"
echo "================================================"

mkdir -p "$(dirname "$OUTPUT_PATH")"

python "${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_from_hf.py" \
  -i "$HF_CKPT_PATH" \
  -o "$OUTPUT_PATH" \
  -m "$MODEL_ARCH" \
  -t dolma2 \

echo "=== Conversion complete ==="
echo "OLMo-core checkpoint saved to: $OUTPUT_PATH"
echo "model_and_optim dir: $OUTPUT_PATH/model_and_optim"
ls -la "$OUTPUT_PATH/"
