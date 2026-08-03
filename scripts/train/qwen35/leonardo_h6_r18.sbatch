#!/bin/bash
#SBATCH --job-name=q35-r18h6
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

: "${QWEN35_H6_CONTRACT:?Set the frozen R18 H6 machine contract}"
: "${QWEN35_H6_HUMAN_PROTOCOL:?Set the frozen R18 H6 human protocol}"
: "${QWEN35_H5_FINAL_CLOSURE:?Set the independently closed H5 authority}"
: "${QWEN35_H6_PREREGISTRATION_CLOSURE:?Set the external H6 preregistration closure}"
: "${QWEN35_H5_RUN:?Set the accepted H5 attempt-03 output root}"
: "${QWEN35_REPO:?Set the clean staged H6 source repository}"
: "${QWEN35_PACK_SCHEDULE:?Set the frozen ten-update H5/H6 schedule}"
: "${QWEN35_SCHEDULE_SHA256:?Set the embedded H5/H6 schedule digest}"
: "${QWEN35_QUALIFICATION_SHA256:?Set the R18 qualification-manifest digest}"

test "$(sha256sum "$QWEN35_H6_CONTRACT" | awk '{print $1}')" = \
    376d2a210c35a791b79743f5354808d767303ae422f907ca0fbf0c42b7361e10
test "$(sha256sum "$QWEN35_H6_HUMAN_PROTOCOL" | awk '{print $1}')" = \
    2add03918c475fe67066acb9d7b1acad4fb50600f68cd968987cd76e75a87081
test "$(sha256sum "$QWEN35_H5_FINAL_CLOSURE" | awk '{print $1}')" = \
    0d38dfdb814ed0facca07ca92fd28421b8fc66b405e755b14cd8b13708580e4a
test "$(sha256sum "$QWEN35_H6_PREREGISTRATION_CLOSURE" | awk '{print $1}')" = \
    47ff866893804a86617b700e13cbbe11d2c3d6015df7581f2fb784e94944f335
test "$(sha256sum "$QWEN35_PACK_SCHEDULE" | awk '{print $1}')" = \
    4e8d4a9fb1a1fd0c92abff0321e71355e066c5723f626bed847fd799704e0c2c
test "$QWEN35_SCHEDULE_SHA256" = 6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281
test "$QWEN35_QUALIFICATION_SHA256" = 679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0
test "$(sha256sum "$QWEN35_VENV/runtime-import-report.json" | awk '{print $1}')" = \
    728e6a5d618db6ea7f321fe8013133e5590bc89616ec646d3dba99b7848266cf
test "$(sha256sum "$QWEN35_MODEL_MANIFEST" | awk '{print $1}')" = \
    017de528d444839a04929fc68b45242169675a917f8249baa295b9b969618e39

h5_checkpoint=$QWEN35_H5_RUN/training/checkpoint-5
test "$(sha256sum "$QWEN35_H5_RUN/h5_checkpoint_5_file_manifest.json" | awk '{print $1}')" = \
    76aee3cd63d1a0bf6291f65cf0b065c102f6710d497de9ae3782e75bcf67cfa9
test "$(sha256sum "$QWEN35_H5_RUN/h5_checkpoint_5_reload_validation.json" | awk '{print $1}')" = \
    8d3b16e0f993e3a44d4e42f003ca0cf0997fc8c415d1cf65d53e86be6ce386fb
test "$(sha256sum "$QWEN35_H5_RUN/training/qwen35_exact_metrics.jsonl" | awk '{print $1}')" = \
    693fcc630fff09fea6a9d42fdbd2843f8f4b42f1827d621b1474e601ae8fb848
test "$(sha256sum "$h5_checkpoint/model.safetensors" | awk '{print $1}')" = \
    11b8000cca8d0c0049b3607728814e5268ad1dab15cda17403527b98814e026f

G2_EXTRA_PERSONAL_PATHS=(
    "$QWEN35_H6_CONTRACT"
    "$QWEN35_H6_HUMAN_PROTOCOL"
    "$QWEN35_H5_FINAL_CLOSURE"
    "$QWEN35_H6_PREREGISTRATION_CLOSURE"
    "$QWEN35_H5_RUN"
)
source "$QWEN35_REPO/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=8

"$QWEN35_VENV/bin/python" -c '
import sys
from pathlib import Path
from scripts.train.qwen35.validate_qwen35_h6_r18 import validate_checkpoint_file_manifest
report=validate_checkpoint_file_manifest(Path(sys.argv[1]), Path(sys.argv[2]))
if report.get("status") != "passed": raise SystemExit("H5 checkpoint file-manifest validation failed")
' "$h5_checkpoint" "$QWEN35_H5_RUN/h5_checkpoint_5_file_manifest.json"

hardware_inventory=$QWEN35_OUTPUT_DIR/h6_hardware_inventory.json
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
    --save_steps 10
    --save_total_limit 1
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

resumed_dir=$QWEN35_OUTPUT_DIR/resumed
"$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    scripts/train/qwen35/train_qwen35_sft.py \
    "${training_common[@]}" \
    --output_dir "$resumed_dir" \
    --resume_from_checkpoint "$h5_checkpoint" \
    --expected_initial_global_step 5

comparison=$QWEN35_OUTPUT_DIR/h6_checkpoint_comparison.json
"$QWEN35_VENV/bin/python" scripts/train/qwen35/compare_qwen35_checkpoints.py \
    --continuous-checkpoint "$continuous_dir/checkpoint-10" \
    --resumed-checkpoint "$resumed_dir/checkpoint-10" \
    --report-output "$comparison" \
    --atol 0 \
    --rtol 0

validator_common=(
    --h6-contract "$QWEN35_H6_CONTRACT"
    --human-protocol "$QWEN35_H6_HUMAN_PROTOCOL"
    --h5-final-closure "$QWEN35_H5_FINAL_CLOSURE"
    --preregistration-closure "$QWEN35_H6_PREREGISTRATION_CLOSURE"
    --source-repository "$QWEN35_REPO"
    --source-code-manifest "$QWEN35_CODE_MANIFEST"
    --expected-source-head "$QWEN35_EXPECTED_CODE_COMMIT"
    --schedule "$QWEN35_PACK_SCHEDULE"
    --numpy-data "$QWEN35_NUMPY_DATA"
    --runtime-report "$QWEN35_VENV/runtime-import-report.json"
    --model-manifest "$QWEN35_MODEL_MANIFEST"
    --hardware-inventory "$hardware_inventory"
    --h5-run "$QWEN35_H5_RUN"
    --output-dir "$QWEN35_OUTPUT_DIR"
    --checkpoint-comparison "$comparison"
)
"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_h6_r18.py \
    --mode producer \
    "${validator_common[@]}" \
    --report-output "$QWEN35_OUTPUT_DIR/h6_producer_validation.json"

scontrol show job --oneliner "$SLURM_JOB_ID" >"$QWEN35_OUTPUT_DIR/slurm_job_pre_exit.txt"
echo R18_H6_PRODUCER_PASSED_PENDING_SLURM_AND_INDEPENDENT_CLOSURE
