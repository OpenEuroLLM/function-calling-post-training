#!/bin/bash
#SBATCH --job-name=q35-r18h5
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:45:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail

: "${QWEN35_H5_CONTRACT:?Set the frozen R18 H5 machine contract}"
: "${QWEN35_H5_HUMAN_PROTOCOL:?Set the frozen R18 H5 human protocol}"
: "${QWEN35_H5_PREREGISTRATION_CLOSURE:?Set the frozen H5 preregistration closure}"
: "${QWEN35_H5_HARNESS_AMENDMENT:?Set the frozen H5 harness amendment}"
: "${QWEN35_H5_HARNESS_HUMAN_AMENDMENT:?Set the frozen H5 human harness amendment}"
: "${QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE:?Set the retained H5 attempt-01 failure closure}"
: "${QWEN35_H5_HARNESS_AMENDMENT_R2:?Set the frozen H5 harness amendment R2}"
: "${QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2:?Set the frozen H5 human harness amendment R2}"
: "${QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE:?Set the retained H5 attempt-02 failure closure}"
: "${QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC:?Set the retained H5 reload-type diagnostic}"
: "${QWEN35_H4_SET_VALIDATION:?Set the final H4 exact-set validation}"
: "${QWEN35_H4_FINAL_CLOSURE:?Set the final H4 independent closure}"
: "${QWEN35_H5_SCHEDULE_VALIDATION:?Set the independent H5 schedule validation}"
: "${QWEN35_H5_SCHEDULE_CLOSURE:?Set the frozen H5 schedule closure}"
: "${QWEN35_H5_SCHEDULE_STAGING:?Set the passing schedule-loader staging marker}"
: "${QWEN35_H5_SCHEDULE_SOURCE_CODE_MANIFEST:?Set the schedule-loader source code manifest}"
: "${QWEN35_REPO:?Set the clean staged H5 source repository}"
: "${QWEN35_PACK_SCHEDULE:?Set the frozen ten-update H5 schedule}"
: "${QWEN35_SCHEDULE_SHA256:?Set the embedded H5 schedule digest}"
: "${QWEN35_QUALIFICATION_SHA256:?Set the R18 qualification-manifest digest}"

test "$(sha256sum "$QWEN35_H5_CONTRACT" | awk '{print $1}')" = 74476804f5bb72f2212394f7d1646b31d1f3e2df8ca0f65cda261997f2efebb1
test "$(sha256sum "$QWEN35_H5_HUMAN_PROTOCOL" | awk '{print $1}')" = fd35b116ccd7a4f806f577bd16a752d1e3149166751ce9101f79739e98d9468d
test "$(sha256sum "$QWEN35_H5_PREREGISTRATION_CLOSURE" | awk '{print $1}')" = 1b7fc59e698f38192a8a4019a6ff54baa4fab2672376375a87f7864343101148
test "$(sha256sum "$QWEN35_H5_HARNESS_AMENDMENT" | awk '{print $1}')" = d5f1521240ab5f9054f75ee50cf8659711edfdc022a176ab99c902572807c7f5
test "$(sha256sum "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT" | awk '{print $1}')" = 315b6bbdd594972becd4cf0d6515c46db25fec5ec3bd45a76fc72f524d681a5b
test "$(sha256sum "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE" | awk '{print $1}')" = 5ef82ccdc46325c97040d85019d57bd5a63f2f3029a39c3e55a03fc089cc1d80
test "$(sha256sum "$QWEN35_H5_HARNESS_AMENDMENT_R2" | awk '{print $1}')" = 0ffb0610d6bbef4bc42c3b0c6ecee9327214fb77a4d219440af069fbde09a057
test "$(sha256sum "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2" | awk '{print $1}')" = 3e976e767b2c5b4897f804a18b54c42da8427a6585ad05319d7a3ab27568bc5b
test "$(sha256sum "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE" | awk '{print $1}')" = 256a2867af66ba8b162d3ee5f9b22b41e7405ecab6cb4e669b4dbb26da6414e9
test "$(sha256sum "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC" | awk '{print $1}')" = 891c28dc4484780349ea05bf388b70c5e7ecbd2c64dc633154ccbe0f9895ccf1
test "$(sha256sum "$QWEN35_H4_SET_VALIDATION" | awk '{print $1}')" = 5650915894222a3ba50b8dc3b2049db8983c90cf43842e7993b42b8eaa018cda
test "$(sha256sum "$QWEN35_H4_FINAL_CLOSURE" | awk '{print $1}')" = d363c5868ebe440b0540431b536cb0baff061c64e633a725a79fa6277538945d
test "$(sha256sum "$QWEN35_H5_SCHEDULE_VALIDATION" | awk '{print $1}')" = 836acb3d91782b649e6d54b9a7a8a1db2a3833954e38a667c2f8881305c21e80
test "$(sha256sum "$QWEN35_H5_SCHEDULE_CLOSURE" | awk '{print $1}')" = 10afebd0c2ed11820ea61b4d4263173d331a0622bd120710578c293d11e7c463
test "$(sha256sum "$QWEN35_H5_SCHEDULE_STAGING" | awk '{print $1}')" = cfecaac6f6df22bfab6230b286b6e07d27d0d6c6533a67d15f33c4e8a09573d1
test "$(sha256sum "$QWEN35_H5_SCHEDULE_SOURCE_CODE_MANIFEST" | awk '{print $1}')" = e154006acc996b90402e6c6d67f1f5dc421b50e9f8e4608435b58ce82fba8257
test "$(sha256sum "$QWEN35_PACK_SCHEDULE" | awk '{print $1}')" = 4e8d4a9fb1a1fd0c92abff0321e71355e066c5723f626bed847fd799704e0c2c
test "$QWEN35_SCHEDULE_SHA256" = 6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281
test "$QWEN35_QUALIFICATION_SHA256" = 679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0

G2_EXTRA_PERSONAL_PATHS=(
    "$QWEN35_H5_CONTRACT"
    "$QWEN35_H5_HUMAN_PROTOCOL"
    "$QWEN35_H5_PREREGISTRATION_CLOSURE"
    "$QWEN35_H5_HARNESS_AMENDMENT"
    "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT"
    "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE"
    "$QWEN35_H5_HARNESS_AMENDMENT_R2"
    "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2"
    "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE"
    "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC"
    "$QWEN35_H4_SET_VALIDATION"
    "$QWEN35_H4_FINAL_CLOSURE"
    "$QWEN35_H5_SCHEDULE_VALIDATION"
    "$QWEN35_H5_SCHEDULE_CLOSURE"
    "$QWEN35_H5_SCHEDULE_STAGING"
    "$QWEN35_H5_SCHEDULE_SOURCE_CODE_MANIFEST"
)
source "${QWEN35_REPO:?Set QWEN35_REPO}/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG_FILE="$QWEN35_OUTPUT_DIR/nccl_debug.%h.%p.log"
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=8

validator_common=(
    --h5-contract "$QWEN35_H5_CONTRACT"
    --human-protocol "$QWEN35_H5_HUMAN_PROTOCOL"
    --preregistration-closure "$QWEN35_H5_PREREGISTRATION_CLOSURE"
    --harness-amendment "$QWEN35_H5_HARNESS_AMENDMENT"
    --harness-human-amendment "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT"
    --attempt01-failure-closure "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE"
    --harness-amendment-r2 "$QWEN35_H5_HARNESS_AMENDMENT_R2"
    --harness-human-amendment-r2 "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2"
    --attempt02-failure-closure "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE"
    --reload-type-diagnostic "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC"
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST"
    --h4-set-validation "$QWEN35_H4_SET_VALIDATION"
    --h4-final-closure "$QWEN35_H4_FINAL_CLOSURE"
    --schedule-validation "$QWEN35_H5_SCHEDULE_VALIDATION"
    --schedule-closure "$QWEN35_H5_SCHEDULE_CLOSURE"
    --schedule-staging "$QWEN35_H5_SCHEDULE_STAGING"
    --schedule-source-code-manifest "$QWEN35_H5_SCHEDULE_SOURCE_CODE_MANIFEST"
    --source-repository "$QWEN35_REPO"
    --source-code-manifest "$QWEN35_CODE_MANIFEST"
    --expected-source-head "$QWEN35_EXPECTED_CODE_COMMIT"
    --schedule "$QWEN35_PACK_SCHEDULE"
    --numpy-data "$QWEN35_NUMPY_DATA"
    --runtime-report "$QWEN35_VENV/runtime-import-report.json"
    --model-manifest "$QWEN35_MODEL_MANIFEST"
    --output-dir "$QWEN35_OUTPUT_DIR"
)

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h5_r18.py \
    --mode capture "${validator_common[@]}" \
    --report-output "$QWEN35_OUTPUT_DIR/h5_immutable_input_and_hardware_inventory.json"

"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/validate_qwen35_accelerate_schedule_sharding_r18_h5.py \
    --h5-contract "$QWEN35_H5_CONTRACT" \
    --human-protocol "$QWEN35_H5_HUMAN_PROTOCOL" \
    --preregistration-closure "$QWEN35_H5_PREREGISTRATION_CLOSURE" \
    --harness-amendment "$QWEN35_H5_HARNESS_AMENDMENT" \
    --harness-human-amendment "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT" \
    --attempt01-failure-closure "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE" \
    --harness-amendment-r2 "$QWEN35_H5_HARNESS_AMENDMENT_R2" \
    --harness-human-amendment-r2 "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2" \
    --attempt02-failure-closure "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE" \
    --reload-type-diagnostic "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC" \
    --schedule "$QWEN35_PACK_SCHEDULE" \
    --report-output "$QWEN35_OUTPUT_DIR/h5_accelerate_schedule_sharding_preflight.json"

"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/validate_qwen35_ddp_loss_normalization_r18_h5.py \
    --h5-contract "$QWEN35_H5_CONTRACT" \
    --human-protocol "$QWEN35_H5_HUMAN_PROTOCOL" \
    --preregistration-closure "$QWEN35_H5_PREREGISTRATION_CLOSURE" \
    --harness-amendment "$QWEN35_H5_HARNESS_AMENDMENT" \
    --harness-human-amendment "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT" \
    --attempt01-failure-closure "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE" \
    --harness-amendment-r2 "$QWEN35_H5_HARNESS_AMENDMENT_R2" \
    --harness-human-amendment-r2 "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2" \
    --attempt02-failure-closure "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE" \
    --reload-type-diagnostic "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC" \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/h5_non_liger_ddp_normalization_preflight.json"

training_dir=$QWEN35_OUTPUT_DIR/training
"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/train_qwen35_sft.py \
    --model_name_or_path "$QWEN35_MODEL_SNAPSHOT" \
    --model_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --numpy_data_dir "$QWEN35_NUMPY_DATA" \
    --expected_arm_id C00 \
    --sequence_length 32768 \
    --drop_last false \
    --verify_data_hashes true \
    --pack_schedule_path "$QWEN35_PACK_SCHEDULE" \
    --expected_schedule_sha256 6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281 \
    --selected_loss_chunk_size 512 \
    --use_checkpointed_chunked_selected_loss true \
    --use_liger_fused_linear_cross_entropy false \
    --hash_text_conversion_tensors true \
    --hardware_qualification_manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --require_no_dense_logits true \
    --require_forward_loss_audit true \
    --hardware_profile true \
    --cuda_event_step_timing true \
    --output_dir "$training_dir" \
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
    --expected_initial_global_step 0 \
    --expected_final_global_step 5 \
    --exact_metrics_sync_interval 1 \
    --logging_steps 1 \
    --save_steps 5 \
    --save_total_limit 1 \
    --gradient_checkpointing true \
    --remove_unused_columns false \
    --report_to none \
    --disable_tqdm true \
    --seed 3407 \
    --data_seed 3407

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h5_nccl_trace_r18.py \
    --h5-contract "$QWEN35_H5_CONTRACT" \
    --human-protocol "$QWEN35_H5_HUMAN_PROTOCOL" \
    --preregistration-closure "$QWEN35_H5_PREREGISTRATION_CLOSURE" \
    --harness-amendment "$QWEN35_H5_HARNESS_AMENDMENT" \
    --harness-human-amendment "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT" \
    --attempt01-failure-closure "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE" \
    --harness-amendment-r2 "$QWEN35_H5_HARNESS_AMENDMENT_R2" \
    --harness-human-amendment-r2 "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2" \
    --attempt02-failure-closure "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE" \
    --reload-type-diagnostic "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC" \
    --hardware-profile "$training_dir/qwen35_cuda_hardware_profile.json" \
    --trace "$training_dir/qwen35_cuda_profiler_trace.json" \
    --sanitized-trace-output "$training_dir/qwen35_cuda_profiler_trace_sanitized.json" \
    --report-output "$QWEN35_OUTPUT_DIR/h5_nccl_exact_event_catalog.json"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h5_checkpoint_reload_r18.py \
    --h5-contract "$QWEN35_H5_CONTRACT" \
    --human-protocol "$QWEN35_H5_HUMAN_PROTOCOL" \
    --preregistration-closure "$QWEN35_H5_PREREGISTRATION_CLOSURE" \
    --harness-amendment "$QWEN35_H5_HARNESS_AMENDMENT" \
    --harness-human-amendment "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT" \
    --attempt01-failure-closure "$QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE" \
    --harness-amendment-r2 "$QWEN35_H5_HARNESS_AMENDMENT_R2" \
    --harness-human-amendment-r2 "$QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2" \
    --attempt02-failure-closure "$QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE" \
    --reload-type-diagnostic "$QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC" \
    --checkpoint "$training_dir/checkpoint-5" \
    --file-manifest-output "$QWEN35_OUTPUT_DIR/h5_checkpoint_5_file_manifest.json" \
    --report-output "$QWEN35_OUTPUT_DIR/h5_checkpoint_5_reload_validation.json"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h5_r18.py \
    --mode producer "${validator_common[@]}" \
    --report-output "$QWEN35_OUTPUT_DIR/h5_producer_validation.json"

scontrol show job --oneliner "$SLURM_JOB_ID" >"$QWEN35_OUTPUT_DIR/slurm_job_pre_exit.txt"
echo R18_H5_PRODUCER_PASSED_PENDING_SLURM_AND_INDEPENDENT_CLOSURE
