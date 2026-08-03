#!/usr/bin/env bash
#SBATCH --job-name=convert-ckpt-olmocore2hf
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=oellm_prod2026
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus=a100:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail

# --- Paths ---
WORK_DIR="${WORK}/ytahtah0"
OLMOCORE_PATH="/leonardo/home/userexternal/ytahtah0/OLMo-core"

# # Exp A
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/reproduce_olmo3_instruct_sft/step3252}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/OLMo-3-7B-Instruct-SFT-reproduce}"

# # Exp B
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_fc_sft/step1801}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_fc_sft_hf}"

# # Exp C
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/nemotron_fc_sft/step2229}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/nemotron_fc_sft_hf}"

# # Exp D
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_it_only_sft/step2346}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_it_only_sft_hf}"

# # Exp E
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/mixed_fc_sft/step4032}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/mixed_fc_sft_hf}"

# # Exp F
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_instruct_plus_nemotron_sft/step4366}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_instruct_plus_nemotron_sft_hf}"

# # Exp G
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/filtered_mixed_fc_sft/step3794}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/filtered_mixed_fc_sft_hf}"

# # Exp H
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_graphsyn_fc_sft/step2928}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_graphsyn_fc_sft_hf}"

# # Exp I
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_graphsyn_nemotron_fc_sft/step4919}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_graphsyn_nemotron_fc_sft_hf}"

# # Exp J
# INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_instruct_plus_graphsyn_sft/step3814}"
# OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_instruct_plus_graphsyn_sft_hf}"

# Exp K
INPUT_PATH="${INPUT_PATH:-${WORK_DIR}/checkpoints/dolci_instruct_plus_graphsyn_nemotron_sft/step4810}"
OUTPUT_PATH="${OUTPUT_PATH:-${WORK_DIR}/models/dolci_instruct_plus_graphsyn_nemotron_sft_hf}"

# HuggingFace offline (tokenizer already cached in Step 1)
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# OLMo-core on PYTHONPATH + activate venv
export PYTHONPATH="${OLMOCORE_PATH}/src:${PYTHONPATH:-}"
source "${OLMOCORE_PATH}/.venv/bin/activate"

echo "=== Checkpoint Conversion: OLMo-core → HF ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Input (OLMo-core): $INPUT_PATH"
echo "Output (HF): $OUTPUT_PATH"
echo "Tokenizer: allenai/olmo-3-tokenizer-instruct-dev"
echo "================================================"

mkdir -p "$(dirname "$OUTPUT_PATH")"

python "${OLMOCORE_PATH}/src/examples/huggingface/convert_checkpoint_to_hf.py" \
    -i "$INPUT_PATH" \
    -o "$OUTPUT_PATH" \
    -t allenai/olmo-3-tokenizer-instruct-dev

echo "=== Conversion complete ==="
echo "HF checkpoint saved to: $OUTPUT_PATH"
ls -la "$OUTPUT_PATH/"
