#!/bin/bash
#SBATCH --job-name=qwen35-h7-t8
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:45:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail
export QWEN35_OMP_NUM_THREADS=8
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

: "${QWEN35_T4_OUTPUT:?Set QWEN35_T4_OUTPUT to the accepted H7 T4 output}"
G2_EXTRA_PERSONAL_PATHS=("$QWEN35_T4_OUTPUT")
qwen35_repo_real=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${QWEN35_REPO:?Set QWEN35_REPO before submission}")
[[ "$qwen35_repo_real" == /leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/* ]]
source "$qwen35_repo_real/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 bash -c '
"$QWEN35_VENV/bin/python" scripts/train/qwen35/capture_qwen35_hardware_inventory.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/h0_hardware_inventory_node${SLURM_PROCID}.json" \
    --repo-root "$QWEN35_REPO" \
    --identity-file "$QWEN35_CODE_MANIFEST" \
    --identity-file "$QWEN35_MODEL_MANIFEST" \
    --identity-file "$QWEN35_NUMPY_DATA/manifest.json" \
    --identity-file "$QWEN35_PACK_SCHEDULE"
'

"$QWEN35_VENV/bin/python" scripts/train/qwen35/combine_qwen35_hardware_inventories.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --inventory "$QWEN35_OUTPUT_DIR/h0_hardware_inventory_node0.json" \
    --inventory "$QWEN35_OUTPUT_DIR/h0_hardware_inventory_node1.json" \
    --expected-nodes 2 \
    --expected-gpus-per-node 4 \
    --report-output "$QWEN35_OUTPUT_DIR/h0_hardware_inventory.json"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 bash -c '
node_cache="$G2_JOB_TMP/node-$SLURM_PROCID"
mkdir -p "$node_cache/triton" "$node_cache/torch_extensions"
export TRITON_CACHE_DIR="$node_cache/triton"
export TORCH_EXTENSIONS_DIR="$node_cache/torch_extensions"
exec "$QWEN35_VENV/bin/torchrun" \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
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
    --cuda_event_step_timing true \
    --require_no_dense_logits true \
    --output_dir "$QWEN35_OUTPUT_DIR" \
    --bf16 true \
    --fp16 false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
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
    --max_steps 13 \
    --expected_final_global_step 13 \
    --exact_metrics_sync_interval 1 \
    --logging_steps 1 \
    --save_steps 13 \
    --save_total_limit 1 \
    --gradient_checkpointing true \
    --remove_unused_columns false \
    --ddp_find_unused_parameters false \
    --report_to none \
    --seed 3407 \
    --data_seed 3407
'

"$QWEN35_VENV/bin/python" scripts/train/qwen35/validate_qwen35_g2_artifacts.py \
    --output-dir "$QWEN35_OUTPUT_DIR" \
    --expected-initial-step 0 \
    --expected-final-step 13 \
    --expected-world-size 8 \
    --expected-gradient-accumulation 1 \
    --expected-schedule-sha256 "$QWEN35_SCHEDULE_SHA256" \
    --expected-qualification-manifest-sha256 "$QWEN35_QUALIFICATION_SHA256"

"$QWEN35_VENV/bin/python" scripts/train/qwen35/compare_qwen35_topologies.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --four-gpu-output "$QWEN35_T4_OUTPUT" \
    --eight-gpu-output "$QWEN35_OUTPUT_DIR" \
    --report-output "$QWEN35_OUTPUT_DIR/qwen35_four_vs_eight_gpu_topology_comparison.json"
