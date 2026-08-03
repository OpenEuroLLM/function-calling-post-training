#!/bin/bash
#SBATCH --job-name=qwen35-g2b-c00-dist
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

# Staged G2-B contract only. Do not submit without the live accounting review.
set -euo pipefail
export QWEN35_OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

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
    scripts/train/qwen35/train_qwen35_sft.py \
    --model_name_or_path "$QWEN35_MODEL_SNAPSHOT" \
    --model_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --numpy_data_dir "$QWEN35_NUMPY_DATA" \
    --expected_arm_id C00 \
    --pack_schedule_path "$QWEN35_PACK_SCHEDULE" \
    --expected_schedule_sha256 "$QWEN35_SCHEDULE_SHA256" \
    --sequence_length 32768 \
    --drop_last false \
    --verify_data_hashes true \
    --hardware_qualification_manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --hardware_profile true \
    --require_no_dense_logits true \
    --output_dir "$QWEN35_OUTPUT_DIR" \
    --bf16 true \
    --fp16 false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --average_tokens_across_devices true \
    --train_sampling_strategy sequential \
    --learning_rate 2e-5 \
    --optim adamw_torch_fused \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_steps 10 \
    --stop_after_steps 5 \
    --expected_final_global_step 5 \
    --exact_metrics_sync_interval 1 \
    --logging_steps 1 \
    --save_steps 5 \
    --save_total_limit 1 \
    --gradient_checkpointing true \
    --remove_unused_columns false \
    --ddp_find_unused_parameters false \
    --report_to none \
    --seed 3407 \
    --data_seed 3407

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_hardware_profile.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --hardware-profile "$QWEN35_OUTPUT_DIR/qwen35_cuda_hardware_profile.json" \
    --trace "$QWEN35_OUTPUT_DIR/qwen35_cuda_profiler_trace.json" \
    --report-output "$QWEN35_OUTPUT_DIR/qwen35_h4_hardware_profile_validation.json"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_nccl_profile.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --trace "$QWEN35_OUTPUT_DIR/qwen35_cuda_profiler_trace.json" \
    --world-size 4 \
    --num-nodes 1 \
    --report-output "$QWEN35_OUTPUT_DIR/qwen35_nccl_profile.json"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_g2_artifacts.py \
    --output-dir "$QWEN35_OUTPUT_DIR" \
    --expected-initial-step 0 \
    --expected-final-step 5 \
    --expected-world-size 4 \
    --expected-gradient-accumulation 2 \
    --expected-schedule-sha256 "$QWEN35_SCHEDULE_SHA256" \
    --expected-qualification-manifest-sha256 "$QWEN35_QUALIFICATION_SHA256" \
    --require-hardware-profile
