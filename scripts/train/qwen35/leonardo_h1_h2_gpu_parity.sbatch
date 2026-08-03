#!/bin/bash
#SBATCH --job-name=qwen35-h1h2-parity
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail

G2_EXTRA_PERSONAL_PATHS=()
qwen35_repo_real=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${QWEN35_REPO:?Set QWEN35_REPO before submission}")
[[ "$qwen35_repo_real" == /leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/* ]]
source "$qwen35_repo_real/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

"$QWEN35_VENV/bin/python" scripts/train/qwen35/capture_qwen35_hardware_inventory.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/h0_hardware_inventory.json" \
    --repo-root "$QWEN35_REPO" \
    --identity-file "$QWEN35_CODE_MANIFEST" \
    --identity-file "$QWEN35_MODEL_MANIFEST" \
    --identity-file "$QWEN35_NUMPY_DATA/manifest.json" \
    --identity-file "$QWEN35_PACK_SCHEDULE"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_reference.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --fixture-jsonl tests/fixtures/qwen35_fc/fixture.jsonl \
    --report-output "$QWEN35_OUTPUT_DIR/base_reference_and_conversion_parity.json" \
    --tokenizer-name-or-path "$QWEN35_MODEL_SNAPSHOT" \
    --model-parity \
    --model-name-or-path "$QWEN35_MODEL_SNAPSHOT" \
    --require-conditional-text-conversion-parity

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h1_report_r16.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --h1-report "$QWEN35_OUTPUT_DIR/base_reference_and_conversion_parity.json" \
    --validation-output "$QWEN35_OUTPUT_DIR/base_reference_and_conversion_independent_validation.json"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_selective_loss_r16.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/selective_liger_loss_qualification.json"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h2_report_r16.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --h2-report "$QWEN35_OUTPUT_DIR/selective_liger_loss_qualification.json" \
    --validation-output "$QWEN35_OUTPUT_DIR/selective_liger_loss_independent_validation.json"
