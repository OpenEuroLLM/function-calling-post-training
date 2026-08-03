#!/bin/bash
#SBATCH --job-name=q35-r18-h3
#SBATCH --account=aifac_f02_434
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail
export QWEN35_OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

: "${QWEN35_H3_MANIFEST:?Set QWEN35_H3_MANIFEST}"
: "${QWEN35_H3_HUMAN_PROTOCOL:?Set QWEN35_H3_HUMAN_PROTOCOL}"
: "${QWEN35_H3_PREREG_CLOSURE:?Set QWEN35_H3_PREREG_CLOSURE}"
: "${QWEN35_H2_INDEPENDENT_VALIDATION:?Set QWEN35_H2_INDEPENDENT_VALIDATION}"
: "${QWEN35_H3_HARNESS_AMENDMENT:?Set QWEN35_H3_HARNESS_AMENDMENT}"
: "${QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL:?Set QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL}"
: "${QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE:?Set QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE}"
: "${QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE:?Set QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE}"

G2_EXTRA_PERSONAL_PATHS=(
    "$QWEN35_H3_MANIFEST"
    "$QWEN35_H3_HUMAN_PROTOCOL"
    "$QWEN35_H3_PREREG_CLOSURE"
    "$QWEN35_H2_INDEPENDENT_VALIDATION"
    "$QWEN35_H3_HARNESS_AMENDMENT"
    "$QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL"
    "$QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE"
    "$QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE"
)
qwen35_repo_real=$(readlink -f -- "${QWEN35_REPO:?Set QWEN35_REPO before submission}")
[[ "$qwen35_repo_real" == /leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/* ]]
source "$qwen35_repo_real/scripts/train/qwen35/g2_job_guard.sh"
g2_prepare_common

test "$(sha256sum "$QWEN35_H3_MANIFEST" | cut -d' ' -f1)" = \
    95aec699d2bab81c5eb3094d2048f997f137faa624dbb7128f92b32134b8abf4
test "$(sha256sum "$QWEN35_H3_HUMAN_PROTOCOL" | cut -d' ' -f1)" = \
    a50aaeecdb17a48f431902127d7d80b6760df6b1b93a7081c89607902522aaab
test "$(sha256sum "$QWEN35_H3_PREREG_CLOSURE" | cut -d' ' -f1)" = \
    70fb3569c0b46eab9bbb7474ca8e3f8b09c524b389b80dd0dbe49b82768dae85
test "$(sha256sum "$QWEN35_H2_INDEPENDENT_VALIDATION" | cut -d' ' -f1)" = \
    cd01fe72f7f73c3d2496390a8ccbee2718f78b7843208ccc626d1c7a51fed176
test "$(sha256sum "$QWEN35_H3_HARNESS_AMENDMENT" | cut -d' ' -f1)" = \
    35618b046ec7120866125189b7a512c3ee38b4b81e075f67d1c880ccd71896a1
test "$(sha256sum "$QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL" | cut -d' ' -f1)" = \
    a95bc1b725af200ce9ff82d099a07c96bdc251b3c725dfe6b5dfb306bd392430
test "$(sha256sum "$QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE" | cut -d' ' -f1)" = \
    e6a9caf4a5eb4d6bb16e3d96ec6a5837983ff2d6549be28704488c729207cc78
test "$(sha256sum "$QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE" | cut -d' ' -f1)" = \
    c0ff9961da3d8407d0f4b8b6f44f427bee960a6a0012f797d8c14c5c184cd990

if find "$QWEN35_REPO" -type f -name '*.pyc' -print -quit | grep -q .; then
    echo "Refusing H3 source tree containing Python bytecode" >&2
    exit 2
fi

"$QWEN35_VENV/bin/python" scripts/train/qwen35/capture_qwen35_hardware_inventory.py \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report-output "$QWEN35_OUTPUT_DIR/h0_hardware_inventory.json" \
    --repo-root "$QWEN35_REPO" \
    --identity-file "$QWEN35_CODE_MANIFEST" \
    --identity-file "$QWEN35_MODEL_MANIFEST" \
    --identity-file "$QWEN35_NUMPY_DATA/manifest.json" \
    --identity-file "$QWEN35_PACK_SCHEDULE" \
    --identity-file "$QWEN35_H3_MANIFEST" \
    --identity-file "$QWEN35_H3_HUMAN_PROTOCOL" \
    --identity-file "$QWEN35_H3_PREREG_CLOSURE" \
    --identity-file "$QWEN35_H2_INDEPENDENT_VALIDATION" \
    --identity-file "$QWEN35_H3_HARNESS_AMENDMENT" \
    --identity-file "$QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL" \
    --identity-file "$QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE" \
    --identity-file "$QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE"

for scenario in P4x2 B4x4; do
    for chunk_size in 128 256 512 1024; do
        run_dir="$QWEN35_OUTPUT_DIR/${scenario}_c$(printf '%04d' "$chunk_size")"
        "$QWEN35_VENV/bin/torchrun" --standalone --nproc_per_node=4 \
            scripts/train/qwen35/validate_qwen35_ddp_ga_r18_h3.py \
            --h3-manifest "$QWEN35_H3_MANIFEST" \
            --r18-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
            --human-protocol "$QWEN35_H3_HUMAN_PROTOCOL" \
            --harness-amendment "$QWEN35_H3_HARNESS_AMENDMENT" \
            --harness-amendment-human-protocol "$QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL" \
            --harness-amendment-preregistration-closure "$QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE" \
            --attempt01-failure-closure "$QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE" \
            --scenario-id "$scenario" \
            --chunk-size "$chunk_size" \
            --output-dir "$run_dir" \
            2>&1 | tee "$QWEN35_OUTPUT_DIR/${scenario}_c$(printf '%04d' "$chunk_size").log"
        test -s "$run_dir/h3_report.json"
        test -s "$run_dir/h3_evidence.safetensors"
        sha256sum "$run_dir/h3_report.json" "$run_dir/h3_evidence.safetensors" \
            >"$run_dir/producer_artifacts.sha256"
    done
done

"$QWEN35_VENV/bin/python" - "$QWEN35_OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = []
for scenario in ("P4x2", "B4x4"):
    for chunk_size in (128, 256, 512, 1024):
        path = root / f"{scenario}_c{chunk_size:04d}" / "h3_report.json"
        value = json.loads(path.read_text())
        if value.get("status") != "passed" or value.get("scenario_id") != scenario or value.get("chunk_size") != chunk_size:
            raise SystemExit(f"incomplete producer report: {path}")
        reports.append({"chunk_size": chunk_size, "path": str(path), "scenario_id": scenario})
output = {
    "allowed_conclusion": "Producer completed eight reports; independent H3 validation is still required and H4 remains blocked.",
    "artifact": "qwen35_r18_h3_producer_set_summary",
    "reports": reports,
    "schema_version": 1,
    "scientific_training_authorized": False,
    "status": "producer_complete_validation_pending",
}
(root / "h3_producer_set_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
PY

if find "$QWEN35_REPO" -type f -name '*.pyc' -print -quit | grep -q .; then
    echo "H3 execution wrote Python bytecode into the immutable source tree" >&2
    exit 2
fi
