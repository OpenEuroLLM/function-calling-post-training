#!/bin/bash
#SBATCH --job-name=q35-r18-h2-val-v1
#SBATCH --account=aifac_f02_434
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/logs/%x-%j.err

set -euo pipefail

: "${QWEN35_VENV:?}"
: "${QWEN35_REPO:?}"
: "${QWEN35_CODE_MANIFEST:?}"
: "${QWEN35_EXPECTED_CODE_COMMIT:?}"
: "${QWEN35_QUALIFICATION_MANIFEST:?}"
: "${QWEN35_R18_H2_REPORT:?}"
: "${QWEN35_AMENDMENT_MANIFEST:?}"
: "${QWEN35_AMENDMENT_HUMAN_PROTOCOL:?}"
: "${QWEN35_AMENDMENT_PREREG_CLOSURE:?}"
: "${QWEN35_OUTPUT_DIR:?}"

test "$SLURM_JOB_ACCOUNT" = aifac_f02_434
for path in \
    "$QWEN35_VENV" \
    "$QWEN35_REPO" \
    "$QWEN35_CODE_MANIFEST" \
    "$QWEN35_QUALIFICATION_MANIFEST" \
    "$QWEN35_R18_H2_REPORT" \
    "$QWEN35_AMENDMENT_MANIFEST" \
    "$QWEN35_AMENDMENT_HUMAN_PROTOCOL" \
    "$QWEN35_AMENDMENT_PREREG_CLOSURE" \
    "$QWEN35_OUTPUT_DIR"; do
    real=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$path")
    [[ "$real" == /leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/* ]]
done

test -x "$QWEN35_VENV/bin/python"
test "$(git -C "$QWEN35_REPO" rev-parse HEAD)" = "$QWEN35_EXPECTED_CODE_COMMIT"
test -z "$(git -C "$QWEN35_REPO" status --porcelain)"
if find "$QWEN35_REPO" -type f -name '*.pyc' -print -quit | grep -q .; then
    echo "Refusing validator source tree containing pre-existing Python bytecode" >&2
    exit 2
fi
test "$(sha256sum "$QWEN35_QUALIFICATION_MANIFEST" | cut -d' ' -f1)" = 679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0
test "$(sha256sum "$QWEN35_R18_H2_REPORT" | cut -d' ' -f1)" = d402d24ad9661f05abfbec02b92c5cb022fc392d953a24920072a4487a08e50d
test "$(wc -c < "$QWEN35_R18_H2_REPORT" | tr -d ' ')" = 253195551
test "$(sha256sum "$QWEN35_AMENDMENT_MANIFEST" | cut -d' ' -f1)" = a210578e9e40ef6b3e06646ccd6fdfac0fc30625349913b18b2b622cacb5a566
test "$(sha256sum "$QWEN35_AMENDMENT_HUMAN_PROTOCOL" | cut -d' ' -f1)" = 57bc83df87aee2f078407a687348a54a6158f4e21d9bc0c59a2bc6d4756894b4
test "$(sha256sum "$QWEN35_AMENDMENT_PREREG_CLOSURE" | cut -d' ' -f1)" = 36e02a2a1dbcb50bd496a5beaa9c3e0b9a1a69db2e40168500817b396b24a2ca
(
    cd "$QWEN35_REPO"
    sha256sum --quiet -c "$QWEN35_CODE_MANIFEST"
)
test ! -e "$QWEN35_OUTPUT_DIR"
mkdir -p "$QWEN35_OUTPUT_DIR/pycache"

export PYTHONPATH="$QWEN35_REPO"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$QWEN35_OUTPUT_DIR/pycache"
unset PYTHONHOME

"$QWEN35_VENV/bin/python" "$QWEN35_REPO/scripts/train/qwen35/validate_qwen35_h2_report_r18_amended_v1.py" \
    --qualification-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --report "$QWEN35_R18_H2_REPORT" \
    --amendment-manifest "$QWEN35_AMENDMENT_MANIFEST" \
    --human-protocol "$QWEN35_AMENDMENT_HUMAN_PROTOCOL" \
    --preregistration-closure "$QWEN35_AMENDMENT_PREREG_CLOSURE" \
    --code-manifest "$QWEN35_CODE_MANIFEST" \
    --validation-output "$QWEN35_OUTPUT_DIR/h2_independent_validation_amended_v1.json"

if find "$QWEN35_REPO" -type f -name '*.pyc' -print -quit | grep -q .; then
    echo "Validator execution wrote Python bytecode into the source tree" >&2
    exit 2
fi
