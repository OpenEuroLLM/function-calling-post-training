#!/bin/bash
# Shared fail-closed guards for the three staged Leonardo G2 wrappers.

g2_personal_root=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3

g2_canonical_path() {
    python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

g2_require_personal_path() {
    local label=$1
    local value=$2
    case "$(g2_canonical_path "$value")" in
        "$g2_personal_root"/*) ;;
        *) echo "Refusing $label outside personal AIFAC work root: $value" >&2; return 2 ;;
    esac
}

g2_pin_python_import_environment() {
    : "${QWEN35_REPO:?Set QWEN35_REPO to the clean staged Open-Instruct tree}"
    # A repository script executed by filename receives its own script directory
    # as sys.path[0], not the repository root. Pin the only permitted project
    # import root explicitly and reject inherited user/PYTHONHOME overlays.
    unset PYTHONHOME
    export PYTHONNOUSERSITE=1
    local repo_real
    repo_real=$(g2_canonical_path "$QWEN35_REPO")
    export PYTHONPATH="$repo_real"
}

g2_cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "${QWEN35_OUTPUT_DIR:-}" && -d "$QWEN35_OUTPUT_DIR" ]]; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
            >"$QWEN35_OUTPUT_DIR/nvidia_compute_apps_at_exit.csv" 2>&1
        printf '{"exit_code":%d,"slurm_job_id":"%s"}\n' \
            "$exit_code" "${SLURM_JOB_ID:-unset}" >"$QWEN35_OUTPUT_DIR/g2_job_exit.json"
    fi
    case "${G2_JOB_TMP:-}" in
        "$g2_personal_root"/tmp/*) rm -rf -- "$G2_JOB_TMP" ;;
        "") ;;
        *) echo "Refusing unsafe G2 scratch cleanup path: $G2_JOB_TMP" >&2; exit_code=2 ;;
    esac
    exit "$exit_code"
}

g2_prepare_common() {
    : "${SLURM_JOB_ID:?G2 wrappers must run inside an allocated Slurm job}"
    : "${QWEN35_VENV:?Set QWEN35_VENV to the reviewed pinned Leonardo environment}"
    : "${QWEN35_REPO:?Set QWEN35_REPO to the clean staged Open-Instruct tree}"
    : "${QWEN35_CODE_MANIFEST:?Set QWEN35_CODE_MANIFEST to the staged SHA-256 manifest}"
    : "${QWEN35_EXPECTED_CODE_COMMIT:?Set QWEN35_EXPECTED_CODE_COMMIT to the reviewed staged commit}"
    : "${QWEN35_QUALIFICATION_MANIFEST:?Set QWEN35_QUALIFICATION_MANIFEST to the frozen manifest overlay}"
    : "${QWEN35_QUALIFICATION_SHA256:?Set QWEN35_QUALIFICATION_SHA256 to the overlay digest}"
    : "${QWEN35_NUMPY_DATA:?Set QWEN35_NUMPY_DATA to the independently verified C00 directory}"
    : "${QWEN35_PACK_SCHEDULE:?Set QWEN35_PACK_SCHEDULE to the frozen C00 pack schedule JSON}"
    : "${QWEN35_SCHEDULE_SHA256:?Set QWEN35_SCHEDULE_SHA256 to the embedded schedule digest}"
    : "${QWEN35_HF_HOME:?Set QWEN35_HF_HOME to the personal pinned model cache}"
    : "${QWEN35_MODEL_SNAPSHOT:?Set QWEN35_MODEL_SNAPSHOT to the pinned local model revision}"
    : "${QWEN35_MODEL_MANIFEST:?Set QWEN35_MODEL_MANIFEST to its symlink-aware SHA-256 manifest}"
    : "${QWEN35_OUTPUT_DIR:?Set QWEN35_OUTPUT_DIR under the personal AIFAC work root}"

    if [[ "${SLURM_JOB_ACCOUNT:-}" != "aifac_f02_434" ]]; then
        echo "Refusing non-personal Slurm account: ${SLURM_JOB_ACCOUNT:-unset}" >&2
        return 2
    fi
    g2_require_personal_path QWEN35_VENV "$QWEN35_VENV"
    g2_require_personal_path QWEN35_REPO "$QWEN35_REPO"
    g2_require_personal_path QWEN35_CODE_MANIFEST "$QWEN35_CODE_MANIFEST"
    g2_require_personal_path QWEN35_QUALIFICATION_MANIFEST "$QWEN35_QUALIFICATION_MANIFEST"
    g2_require_personal_path QWEN35_NUMPY_DATA "$QWEN35_NUMPY_DATA"
    g2_require_personal_path QWEN35_PACK_SCHEDULE "$QWEN35_PACK_SCHEDULE"
    g2_require_personal_path QWEN35_HF_HOME "$QWEN35_HF_HOME"
    g2_require_personal_path QWEN35_MODEL_SNAPSHOT "$QWEN35_MODEL_SNAPSHOT"
    g2_require_personal_path QWEN35_MODEL_MANIFEST "$QWEN35_MODEL_MANIFEST"
    g2_require_personal_path QWEN35_OUTPUT_DIR "$QWEN35_OUTPUT_DIR"
    for extra_path in "${G2_EXTRA_PERSONAL_PATHS[@]}"; do
        g2_require_personal_path extra_G2_path "$extra_path"
    done
    # Create the job-private cache before the first venv Python import.  A
    # copied staging tree may otherwise execute an ignored source-tree .pyc,
    # and an ordinary import would write new bytecode into the immutable tree.
    mkdir -p "$g2_personal_root/logs" "$g2_personal_root/tmp"
    G2_JOB_TMP="$g2_personal_root/tmp/${SLURM_JOB_NAME:-qwen35-g2}-${SLURM_JOB_ID}"
    if [[ -e "$G2_JOB_TMP" ]]; then
        echo "Refusing pre-existing per-job scratch: $G2_JOB_TMP" >&2
        return 2
    fi
    mkdir -p \
        "$G2_JOB_TMP/tmp" \
        "$G2_JOB_TMP/xdg" \
        "$G2_JOB_TMP/triton" \
        "$G2_JOB_TMP/torch_extensions" \
        "$G2_JOB_TMP/pycache"
    export G2_JOB_TMP
    export PYTHONPYCACHEPREFIX="$G2_JOB_TMP/pycache"
    trap g2_cleanup EXIT INT TERM

    g2_pin_python_import_environment

    if [[ ! -x "$QWEN35_VENV/bin/python" ]]; then
        echo "Pinned environment has no executable Python" >&2
        return 2
    fi
    if [[ ! -f "$QWEN35_VENV/.environment_complete" || ! -f "$QWEN35_VENV/runtime-import-report.json" ]]; then
        echo "Pinned environment lacks its completed-build evidence" >&2
        return 2
    fi
    "$QWEN35_VENV/bin/python" -c '
import json,sys
from pathlib import Path
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
report=json.load(open(sys.argv[1]))
qualification,_=load_qualification_manifest(Path(sys.argv[2]))
runtime=qualification["runtime_pins"]
expected_versions={
    "accelerate":runtime["accelerate_version"],
    "causal-conv1d":runtime["causal_conv1d_version"],
    "flash-attn":runtime["flash_attn_version"],
    "flash-linear-attention":runtime["flash_linear_attention_version"],
    "fla-core":runtime["fla_core_version"],
    "numpy":runtime["numpy_version"],
    "torch":runtime["torch_version"],
    "torchvision":runtime["torchvision_version"],
    "transformers":runtime["transformers_version"],
}
if "liger_version" in runtime:
    expected_versions["liger-kernel"]=runtime["liger_version"]
if report.get("status") != "passed": raise SystemExit("runtime import report did not pass")
observed_versions=report.get("packages",{})
if any(observed_versions.get(package) != version for package,version in expected_versions.items()):
    raise SystemExit("runtime required package-version contract drift")
allowed_extra={"liger-kernel"} if runtime.get("liger_execution_allowed") is False else set()
if not set(observed_versions) - set(expected_versions) <= allowed_extra:
    raise SystemExit("runtime package-set contract drift")
if report.get("torch_cuda_build") != runtime["torch_cuda_build"]:
    raise SystemExit("runtime CUDA build contract drift")
source_commits={"transformers":runtime["transformers_commit"]}
if "liger_commit" in runtime:
    source_commits["liger-kernel"]=runtime["liger_commit"]
for package, commit in source_commits.items():
    source=report.get("source_pins",{}).get(package,{})
    if source.get("expected_commit") != commit or commit not in source.get("source_url",""):
        raise SystemExit(f"runtime source pin drift for {package}")
if "liger_commit" in runtime:
    liger_source=report["source_pins"]["liger-kernel"]
    if liger_source.get("runtime_import_mode") != runtime["liger_import_mode"]:
        raise SystemExit("Liger runtime import-mode drift")
    if liger_source.get("checkout_head") != runtime["liger_commit"] or liger_source.get("checkout_clean") is not True:
        raise SystemExit("Liger runtime checkout identity drift")
elif runtime.get("liger_execution_allowed") is not False:
    raise SystemExit("R18 runtime does not explicitly forbid Liger execution")
for package, expected in runtime.get("native_extensions", {}).items():
    source=report.get("source_pins",{}).get(package,{})
    if source.get("source_url") != expected["sdist_url"]:
        raise SystemExit(f"native-extension source URL drift for {package}")
    if source.get("sdist_sha256") != expected["sdist_sha256"]:
        raise SystemExit(f"native-extension source hash drift for {package}")
    if source.get("build_mode") != expected["build_mode"]:
        raise SystemExit(f"native-extension build-mode drift for {package}")
    abi=report.get("native_extension_abi",{}).get(package,{})
    if abi.get("import_status") != "passed" or abi.get("module") != expected["module"]:
        raise SystemExit(f"native-extension import preflight failed for {package}")
    if abi.get("maximum_glibc_version") > expected["maximum_glibc_version"]:
        raise SystemExit(f"native-extension GLIBC ABI drift for {package}")
' "$QWEN35_VENV/runtime-import-report.json" "$QWEN35_QUALIFICATION_MANIFEST"
    if [[ ! -f "$QWEN35_CODE_MANIFEST" || ! -f "$QWEN35_MODEL_MANIFEST" ]]; then
        echo "Code or model checksum manifest is missing" >&2
        return 2
    fi
    if [[ ! -f "$QWEN35_PACK_SCHEDULE" ]]; then
        echo "Frozen Qwen3.5 pack schedule is missing" >&2
        return 2
    fi
    if [[ ! -f "$QWEN35_QUALIFICATION_MANIFEST" ]]; then
        echo "Frozen Qwen3.5 hardware-qualification manifest is missing" >&2
        return 2
    fi
    qualification_digest=$(sha256sum "$QWEN35_QUALIFICATION_MANIFEST" | awk '{print $1}')
    if [[ "$qualification_digest" != "$QWEN35_QUALIFICATION_SHA256" ]]; then
        echo "Hardware-qualification manifest digest drift" >&2
        return 2
    fi
    "$QWEN35_VENV/bin/python" -c '
import sys
from pathlib import Path
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
value,_=load_qualification_manifest(Path(sys.argv[1]))
if value.get("protocol_id") not in {"qwen35-hardware-qualification-r16", "qwen35-hardware-qualification-r17", "qwen35-hardware-qualification-r18"}:
    raise SystemExit("unexpected hardware-qualification protocol")
status=value.get("status")
if status != "ready_for_execution":
    raise SystemExit(f"hardware qualification is not ready: {status!r}")
if value.get("scope",{}).get("slurm_account") != "aifac_f02_434":
    raise SystemExit("hardware-qualification account drift")
' "$QWEN35_QUALIFICATION_MANIFEST"
    schedule_digest=$("$QWEN35_VENV/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["schedule_sha256"])' "$QWEN35_PACK_SCHEDULE")
    if [[ "$schedule_digest" != "$QWEN35_SCHEDULE_SHA256" ]]; then
        echo "Pack schedule embedded digest does not match QWEN35_SCHEDULE_SHA256" >&2
        return 2
    fi
    if [[ -e "$QWEN35_OUTPUT_DIR" ]]; then
        echo "Refusing pre-existing G2 output: $QWEN35_OUTPUT_DIR" >&2
        return 2
    fi

    cd "$QWEN35_MODEL_SNAPSHOT"
    sha256sum --check --strict "$QWEN35_MODEL_MANIFEST"
    cd "$QWEN35_REPO"
    actual_code_commit=$(git rev-parse HEAD)
    if [[ "$actual_code_commit" != "$QWEN35_EXPECTED_CODE_COMMIT" ]]; then
        echo "Staged source commit drift: $actual_code_commit != $QWEN35_EXPECTED_CODE_COMMIT" >&2
        return 2
    fi
    sha256sum --check --strict "$QWEN35_CODE_MANIFEST"
    mkdir -p "$QWEN35_OUTPUT_DIR"
    export HF_HOME="$QWEN35_HF_HOME"
    export TMPDIR="$G2_JOB_TMP/tmp"
    export XDG_CACHE_HOME="$G2_JOB_TMP/xdg"
    export TRITON_CACHE_DIR="$G2_JOB_TMP/triton"
    export TORCH_EXTENSIONS_DIR="$G2_JOB_TMP/torch_extensions"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${QWEN35_OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
    date --iso-8601=seconds >"$QWEN35_OUTPUT_DIR/job_started_at.txt"
    hostname >"$QWEN35_OUTPUT_DIR/hostname.txt"
    scontrol show job --oneliner "$SLURM_JOB_ID" >"$QWEN35_OUTPUT_DIR/slurm_job_start.txt"
    "$QWEN35_VENV/bin/python" --version >"$QWEN35_OUTPUT_DIR/python_version.txt" 2>&1
}
