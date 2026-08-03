#!/bin/bash
#SBATCH --job-name=q35-r18h6r2
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail

: "${QWEN35_H6_R2_CONTRACT:?Set the frozen R18 H6 r2 machine contract}"
: "${QWEN35_H6_R2_HUMAN_PROTOCOL:?Set the frozen R18 H6 r2 human protocol}"
: "${QWEN35_H5_FINAL_CLOSURE:?Set the independently closed H5 authority}"
: "${QWEN35_H6_R1_FAILURE_CLOSURE:?Set the retained H6 r1 failure closure}"
: "${QWEN35_H6_R1_FAILED_COMPARISON:?Set the retained H6 r1 failed comparison}"
: "${QWEN35_H6_R2_PREREGISTRATION_CLOSURE:?Set the external H6 r2 preregistration closure}"
: "${QWEN35_REPO:?Set the clean staged H6 r2 source repository}"
: "${QWEN35_PACK_SCHEDULE:?Set the frozen ten-update schedule}"
: "${QWEN35_SCHEDULE_SHA256:?Set the embedded schedule digest}"
: "${QWEN35_QUALIFICATION_SHA256:?Set the R18 qualification-manifest digest}"

test "$(sha256sum "$QWEN35_H6_R2_CONTRACT" | awk '{print $1}')" = \
    7ef1590617f453b6d70afdec185ca7f70574d14e5a791b5ace96103e1d55383d
test "$(sha256sum "$QWEN35_H6_R2_HUMAN_PROTOCOL" | awk '{print $1}')" = \
    35ed7330ba4fc111d4c667f0aaf28b8a57e340d2d0dc83c0e5cb2a12c0fa7c9f
test "$(sha256sum "$QWEN35_H5_FINAL_CLOSURE" | awk '{print $1}')" = \
    0d38dfdb814ed0facca07ca92fd28421b8fc66b405e755b14cd8b13708580e4a
test "$(sha256sum "$QWEN35_H6_R1_FAILURE_CLOSURE" | awk '{print $1}')" = \
    bf6dd4e1be1e2bf3ac3c3ca184061af8cdc17f4015f73a3abf8d80c959690e17
test "$(sha256sum "$QWEN35_H6_R1_FAILED_COMPARISON" | awk '{print $1}')" = \
    0784f1a6b94c63775564c0d3166c855e26f2c417356a309055bb6369ad4cce01
test "$(sha256sum "$QWEN35_H6_R2_PREREGISTRATION_CLOSURE" | awk '{print $1}')" = \
    ad5df3dc8c669ee04ce9cda81ca4e3ccca2693682eba090e621b31ca929e6c3b
test "$(sha256sum "$QWEN35_PACK_SCHEDULE" | awk '{print $1}')" = \
    4e8d4a9fb1a1fd0c92abff0321e71355e066c5723f626bed847fd799704e0c2c
test "$QWEN35_SCHEDULE_SHA256" = 6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281
test "$QWEN35_QUALIFICATION_SHA256" = 679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0
test "$(sha256sum "$QWEN35_VENV/runtime-import-report.json" | awk '{print $1}')" = \
    728e6a5d618db6ea7f321fe8013133e5590bc89616ec646d3dba99b7848266cf
test "$(sha256sum "$QWEN35_MODEL_MANIFEST" | awk '{print $1}')" = \
    017de528d444839a04929fc68b45242169675a917f8249baa295b9b969618e39

G2_EXTRA_PERSONAL_PATHS=(
    "$QWEN35_H6_R2_CONTRACT"
    "$QWEN35_H6_R2_HUMAN_PROTOCOL"
    "$QWEN35_H5_FINAL_CLOSURE"
    "$QWEN35_H6_R1_FAILURE_CLOSURE"
    "$QWEN35_H6_R1_FAILED_COMPARISON"
    "$QWEN35_H6_R2_PREREGISTRATION_CLOSURE"
)
source "$QWEN35_REPO/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=8

hardware_inventory=$QWEN35_OUTPUT_DIR/h6_r2_hardware_inventory.json
"$QWEN35_VENV/bin/python" scripts/train/qwen35/capture_qwen35_hardware_inventory.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$hardware_inventory" \
    --repo-root "$QWEN35_REPO" \
    --identity-file "$QWEN35_CODE_MANIFEST" \
    --identity-file "$QWEN35_MODEL_MANIFEST" \
    --identity-file "$QWEN35_NUMPY_DATA/manifest.json" \
    --identity-file "$QWEN35_PACK_SCHEDULE"

training_common=(
    --model_name_or_path "$QWEN35_MODEL_SNAPSHOT"
    --model_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
    --numpy_data_dir "$QWEN35_NUMPY_DATA"
    --expected_arm_id C00
    --sequence_length 32768
    --drop_last false
    --verify_data_hashes true
    --pack_schedule_path "$QWEN35_PACK_SCHEDULE"
    --expected_schedule_sha256 6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281
    --selected_loss_chunk_size 512
    --use_checkpointed_chunked_selected_loss true
    --use_liger_fused_linear_cross_entropy false
    --hash_text_conversion_tensors true
    --hardware_qualification_manifest "$QWEN35_QUALIFICATION_MANIFEST"
    --require_no_dense_logits true
    --require_forward_loss_audit true
    --hardware_profile false
    --cuda_event_step_timing false
    --bf16 true
    --fp16 false
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 2
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
    --max_steps 10
    --expected_final_global_step 10
    --exact_metrics_sync_interval 1
    --logging_steps 1
    --save_steps 5
    --save_total_limit 2
    --gradient_checkpointing true
    --remove_unused_columns false
    --report_to none
    --disable_tqdm true
    --seed 3407
    --data_seed 3407
)

continuous_dir=$QWEN35_OUTPUT_DIR/continuous
"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/train_qwen35_sft.py \
    "${training_common[@]}" \
    --output_dir "$continuous_dir" \
    --expected_initial_global_step 0

resume_checkpoint=$continuous_dir/checkpoint-5
test -d "$resume_checkpoint"
test -f "$resume_checkpoint/model.safetensors"
test -f "$resume_checkpoint/optimizer.pt"
test -f "$resume_checkpoint/scheduler.pt"
test -f "$resume_checkpoint/trainer_state.json"
for rank in 0 1 2 3; do test -f "$resume_checkpoint/rng_state_${rank}.pth"; done

resumed_dir=$QWEN35_OUTPUT_DIR/resumed
"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/train_qwen35_sft.py \
    "${training_common[@]}" \
    --output_dir "$resumed_dir" \
    --resume_from_checkpoint "$resume_checkpoint" \
    --expected_initial_global_step 5

comparison=$QWEN35_OUTPUT_DIR/h6_r2_checkpoint_comparison.json
"$QWEN35_VENV/bin/python" scripts/train/qwen35/compare_qwen35_checkpoints.py \
    --continuous-checkpoint "$continuous_dir/checkpoint-10" \
    --resumed-checkpoint "$resumed_dir/checkpoint-10" \
    --report-output "$comparison" \
    --atol 0 \
    --rtol 0

validator_common=(
    --h6-r2-contract "$QWEN35_H6_R2_CONTRACT"
    --human-protocol "$QWEN35_H6_R2_HUMAN_PROTOCOL"
    --h5-final-closure "$QWEN35_H5_FINAL_CLOSURE"
    --r1-failure-closure "$QWEN35_H6_R1_FAILURE_CLOSURE"
    --r1-failed-comparison "$QWEN35_H6_R1_FAILED_COMPARISON"
    --preregistration-closure "$QWEN35_H6_R2_PREREGISTRATION_CLOSURE"
    --source-repository "$QWEN35_REPO"
    --source-code-manifest "$QWEN35_CODE_MANIFEST"
    --expected-source-head "$QWEN35_EXPECTED_CODE_COMMIT"
    --schedule "$QWEN35_PACK_SCHEDULE"
    --numpy-data "$QWEN35_NUMPY_DATA"
    --runtime-report "$QWEN35_VENV/runtime-import-report.json"
    --model-manifest "$QWEN35_MODEL_MANIFEST"
    --hardware-inventory "$hardware_inventory"
    --output-dir "$QWEN35_OUTPUT_DIR"
    --checkpoint-comparison "$comparison"
)
"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h6_r18_r2.py \
    --mode producer \
    "${validator_common[@]}" \
    --report-output "$QWEN35_OUTPUT_DIR/h6_r2_producer_validation.json"

scontrol show job --oneliner "$SLURM_JOB_ID" >"$QWEN35_OUTPUT_DIR/slurm_job_pre_exit.txt"
echo R18_H6_R2_PRODUCER_PASSED_PENDING_INDEPENDENT_CLOSURE
