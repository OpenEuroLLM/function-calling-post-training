#!/bin/bash
#SBATCH --job-name=qwen35-h3-ddp-ga
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail
export QWEN35_OMP_NUM_THREADS=8

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

"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/validate_qwen35_full_model_ddp_ga.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --accumulation-steps 2 \
    --report-output "$QWEN35_OUTPUT_DIR/full_model_ddp_ga_qualification.json"

"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/validate_qwen35_ddp_loss_normalization.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/ddp_target_normalization_qualification.json"

"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/validate_qwen35_accelerate_schedule_sharding.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/accelerate_schedule_sharding_qualification.json" \
    --schedule-length 40
