#!/bin/bash
#SBATCH --job-name=q35-r18h4
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail

: "${QWEN35_H4_CONTRACT:?Set the frozen R18 H4 machine contract}"
: "${QWEN35_H4_CONTRACT_SHA256:?Set the frozen R18 H4 machine-contract digest}"
: "${QWEN35_H4_FOUR_SCHEDULE:?Set the frozen four-update schedule}"
: "${QWEN35_H4_FOUR_SCHEDULE_SHA256:?Set the four-update embedded schedule digest}"
: "${QWEN35_H4_THIRTEEN_SCHEDULE:?Set the frozen thirteen-update schedule}"
: "${QWEN35_H4_THIRTEEN_SCHEDULE_SHA256:?Set the thirteen-update embedded schedule digest}"
: "${QWEN35_H4_CHUNK_SIZE:?Set one of 128, 256, 512, or 1024}"

case "$QWEN35_H4_CHUNK_SIZE" in
    128|256|512|1024) ;;
    *) echo "Refusing unknown H4 chunk candidate: $QWEN35_H4_CHUNK_SIZE" >&2; exit 2 ;;
esac
test "$QWEN35_H4_CONTRACT_SHA256" = adb6f08012d4893320e9ffc0f2d1753c48ceb744c51e98a3e6d10d7ff328fa33
test "$QWEN35_H4_FOUR_SCHEDULE_SHA256" = ae87fd292468b429a8f90830b799b26e48f7c8226c025125e1659df993c6b36b
test "$QWEN35_H4_THIRTEEN_SCHEDULE_SHA256" = 377b1aa408e6cdeb9c6e66fbc51906866e1cdcfcb2b939aab2ffa731b32c0738
test "$(sha256sum "$QWEN35_H4_CONTRACT" | cut -d' ' -f1)" = "$QWEN35_H4_CONTRACT_SHA256"
test "$(sha256sum "$QWEN35_H4_FOUR_SCHEDULE" | cut -d' ' -f1)" = 27dd7c182df0702109305643595561a41a48c07dcac1d86fb139286a2bf340c1
test "$(sha256sum "$QWEN35_H4_THIRTEEN_SCHEDULE" | cut -d' ' -f1)" = f13fbe8c9b8f6c724aaa48c2598bc197dad0278c4bf6345a59479cbb780d0591

G2_EXTRA_PERSONAL_PATHS=("$QWEN35_H4_CONTRACT" "$QWEN35_H4_THIRTEEN_SCHEDULE")
qwen35_repo_real=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${QWEN35_REPO:?Set QWEN35_REPO}")
[[ "$qwen35_repo_real" == /leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/* ]]
source "$qwen35_repo_real/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

test "$QWEN35_PACK_SCHEDULE" = "$QWEN35_H4_FOUR_SCHEDULE"
test "$QWEN35_SCHEDULE_SHA256" = "$QWEN35_H4_FOUR_SCHEDULE_SHA256"
profile_dir=$QWEN35_OUTPUT_DIR/profiler
timing_dir=$QWEN35_OUTPUT_DIR/timing
identity=$QWEN35_OUTPUT_DIR/h4_candidate_job_identity.json
profile_validation=$QWEN35_OUTPUT_DIR/h4_profiler_assay_validation.json
candidate_validation=$QWEN35_OUTPUT_DIR/h4_candidate_automated_validation.json

"$QWEN35_VENV/bin/python" scripts/train/qwen35/capture_qwen35_h4_job_identity_r18.py \
    --repo-root "$QWEN35_REPO" \
    --code-manifest "$QWEN35_CODE_MANIFEST" \
    --expected-code-commit "$QWEN35_EXPECTED_CODE_COMMIT" \
    --runtime-report "$QWEN35_VENV/runtime-import-report.json" \
    --model-snapshot "$QWEN35_MODEL_SNAPSHOT" \
    --model-manifest "$QWEN35_MODEL_MANIFEST" \
    --numpy-data "$QWEN35_NUMPY_DATA" \
    --four-update-schedule "$QWEN35_H4_FOUR_SCHEDULE" \
    --thirteen-update-schedule "$QWEN35_H4_THIRTEEN_SCHEDULE" \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --h4-contract "$QWEN35_H4_CONTRACT" \
    --candidate-chunk-size "$QWEN35_H4_CHUNK_SIZE" \
    --report-output "$identity"

common_arguments=(
    --model_name_or_path "$QWEN35_MODEL_SNAPSHOT"
    --model_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
    --numpy_data_dir "$QWEN35_NUMPY_DATA"
    --expected_arm_id C00
    --sequence_length 32768
    --drop_last false
    --verify_data_hashes true
    --selected_loss_chunk_size "$QWEN35_H4_CHUNK_SIZE"
    --use_checkpointed_chunked_selected_loss true
    --use_liger_fused_linear_cross_entropy false
    --hash_text_conversion_tensors true
    --hardware_qualification_manifest "$QWEN35_QUALIFICATION_MANIFEST"
    --h4_qualification_contract "$QWEN35_H4_CONTRACT"
    --h4_preregistration_closure_sha256 fd1db81c568c473a417c2936f3492288a8d0188057fe548696a2211e26bb8080
    --require_no_dense_logits true
    --require_forward_loss_audit true
    --bf16 true
    --fp16 false
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 8
    --average_tokens_across_devices true
    --train_sampling_strategy sequential
    --learning_rate 2e-5
    --optim adamw_torch_fused
    --adam_beta1 0.9
    --adam_beta2 0.95
    --adam_epsilon 1e-8
    --weight_decay 0.1
    --max_grad_norm 1.0
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --exact_metrics_sync_interval 1
    --logging_steps 1
    --save_total_limit 1
    --gradient_checkpointing true
    --remove_unused_columns false
    --report_to none
    --disable_tqdm true
    --seed 3407
    --data_seed 3407
)

"$QWEN35_VENV/bin/python" scripts/train/qwen35/train_qwen35_sft.py \
    "${common_arguments[@]}" \
    --pack_schedule_path "$QWEN35_H4_FOUR_SCHEDULE" \
    --expected_schedule_sha256 "$QWEN35_H4_FOUR_SCHEDULE_SHA256" \
    --h4_assay profiler \
    --hardware_profile true \
    --cuda_event_step_timing false \
    --output_dir "$profile_dir" \
    --max_steps 4 \
    --expected_final_global_step 4 \
    --save_steps 4

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h4_profile_r18.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --h4-contract "$QWEN35_H4_CONTRACT" \
    --hardware-profile "$profile_dir/qwen35_cuda_hardware_profile.json" \
    --trace "$profile_dir/qwen35_cuda_profiler_trace.json" \
    --memory-snapshot "$profile_dir/qwen35_cuda_memory_snapshot.pickle" \
    --candidate-chunk-size "$QWEN35_H4_CHUNK_SIZE" \
    --report-output "$profile_validation"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/train_qwen35_sft.py \
    "${common_arguments[@]}" \
    --pack_schedule_path "$QWEN35_H4_THIRTEEN_SCHEDULE" \
    --expected_schedule_sha256 "$QWEN35_H4_THIRTEEN_SCHEDULE_SHA256" \
    --h4_assay timing \
    --hardware_profile false \
    --cuda_event_step_timing true \
    --output_dir "$timing_dir" \
    --max_steps 13 \
    --expected_final_global_step 13 \
    --save_steps 13

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h4_candidate_r18.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --h4-contract "$QWEN35_H4_CONTRACT" \
    --candidate-chunk-size "$QWEN35_H4_CHUNK_SIZE" \
    --profile-output-dir "$profile_dir" \
    --timing-output-dir "$timing_dir" \
    --four-update-schedule "$QWEN35_H4_FOUR_SCHEDULE" \
    --thirteen-update-schedule "$QWEN35_H4_THIRTEEN_SCHEDULE" \
    --profile-validation "$profile_validation" \
    --job-identity "$identity" \
    --expected-code-commit "$QWEN35_EXPECTED_CODE_COMMIT" \
    --report-output "$candidate_validation"

scontrol show job --oneliner "$SLURM_JOB_ID" >"$QWEN35_OUTPUT_DIR/slurm_job_pre_exit.txt"
echo R18_H4_CANDIDATE_AUTOMATED_VALIDATION_PASSED_PENDING_MANUAL_KERNEL_REVIEW
