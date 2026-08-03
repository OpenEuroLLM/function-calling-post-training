#!/bin/bash
#SBATCH --job-name=q35-r18-h3-val
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
: "${QWEN35_H3_MANIFEST:?}"
: "${QWEN35_H3_HUMAN_PROTOCOL:?}"
: "${QWEN35_H3_PREREG_CLOSURE:?}"
: "${QWEN35_H2_INDEPENDENT_VALIDATION:?}"
: "${QWEN35_H3_HARNESS_AMENDMENT:?}"
: "${QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL:?}"
: "${QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE:?}"
: "${QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE:?}"
: "${QWEN35_H3_VALIDATOR_AMENDMENT:?}"
: "${QWEN35_H3_VALIDATOR_AMENDMENT_HUMAN_PROTOCOL:?}"
: "${QWEN35_H3_VALIDATOR_AMENDMENT_PREREG_CLOSURE:?}"
: "${QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT:?}"
: "${QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_PROTOCOL:?}"
: "${QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREG_CLOSURE:?}"
: "${QWEN35_H3_VALIDATOR_V2_FAILURE_CLOSURE:?}"
: "${QWEN35_H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE:?}"
: "${QWEN35_H3_PRODUCER_ROOT:?}"
: "${QWEN35_OUTPUT_DIR:?}"

test "$SLURM_JOB_ACCOUNT" = aifac_f02_434
for path in \
    "$QWEN35_VENV" \
    "$QWEN35_REPO" \
    "$QWEN35_CODE_MANIFEST" \
    "$QWEN35_QUALIFICATION_MANIFEST" \
    "$QWEN35_H3_MANIFEST" \
    "$QWEN35_H3_HUMAN_PROTOCOL" \
    "$QWEN35_H3_PREREG_CLOSURE" \
    "$QWEN35_H2_INDEPENDENT_VALIDATION" \
    "$QWEN35_H3_HARNESS_AMENDMENT" \
    "$QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL" \
    "$QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE" \
    "$QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE" \
    "$QWEN35_H3_VALIDATOR_AMENDMENT" \
    "$QWEN35_H3_VALIDATOR_AMENDMENT_HUMAN_PROTOCOL" \
    "$QWEN35_H3_VALIDATOR_AMENDMENT_PREREG_CLOSURE" \
    "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT" \
    "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_PROTOCOL" \
    "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREG_CLOSURE" \
    "$QWEN35_H3_VALIDATOR_V2_FAILURE_CLOSURE" \
    "$QWEN35_H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE" \
    "$QWEN35_H3_PRODUCER_ROOT" \
    "$QWEN35_OUTPUT_DIR"; do
    real=$(readlink -f -- "$path")
    [[ "$real" == /leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/* ]]
done

test -x "$QWEN35_VENV/bin/python"
test "$(git -C "$QWEN35_REPO" rev-parse HEAD)" = "$QWEN35_EXPECTED_CODE_COMMIT"
test -z "$(git -C "$QWEN35_REPO" status --porcelain)"
if find "$QWEN35_REPO" -type f -name '*.pyc' -print -quit | grep -q .; then
    echo "Refusing H3 validator source tree containing Python bytecode" >&2
    exit 2
fi
test "$(sha256sum "$QWEN35_QUALIFICATION_MANIFEST" | cut -d' ' -f1)" = \
    679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0
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
test "$(sha256sum "$QWEN35_H3_VALIDATOR_AMENDMENT" | cut -d' ' -f1)" = \
    bc648147a2af821544fc30554186c05b0758cb9780bf58887eae88f766b66644
test "$(sha256sum "$QWEN35_H3_VALIDATOR_AMENDMENT_HUMAN_PROTOCOL" | cut -d' ' -f1)" = \
    1934d64a3714648a4144f36a1cd2cd445a46ce68e08cab733a2c10fd5daca81d
test "$(sha256sum "$QWEN35_H3_VALIDATOR_AMENDMENT_PREREG_CLOSURE" | cut -d' ' -f1)" = \
    d9ce018a7c110ad550ce6523a21835a4bcd48c42614492e6179ffe780c63f6ef
test "$(sha256sum "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT" | cut -d' ' -f1)" = \
    1090502a52feeed6cbf878018a2c251f02ce0a1431f8974b4e59ea13be950dcf
test "$(sha256sum "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_PROTOCOL" | cut -d' ' -f1)" = \
    90d26f74ef51c1ccee8cdbc5e75ae3708a172bc65918912295c07edfd70cc671
test "$(sha256sum "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREG_CLOSURE" | cut -d' ' -f1)" = \
    77e09a4b862e58bd496bfe839eee0653724914fed5ac4449ed12a901a021c27e
test "$(sha256sum "$QWEN35_H3_VALIDATOR_V2_FAILURE_CLOSURE" | cut -d' ' -f1)" = \
    9946450917786e13cb297b319769d06aa302c0a3b2e62eda5b43df59b69011aa
test "$(sha256sum "$QWEN35_H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE" | cut -d' ' -f1)" = \
    497be6ced00350270cf77dfe89a15313c1f451b5e416985d12fba4533712e1a4
(
    cd "$QWEN35_REPO"
    sha256sum --quiet --check --strict "$QWEN35_CODE_MANIFEST"
)

test ! -e "$QWEN35_OUTPUT_DIR"
mkdir -p "$QWEN35_OUTPUT_DIR/pycache"
export PYTHONPATH="$QWEN35_REPO"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$QWEN35_OUTPUT_DIR/pycache"
unset PYTHONHOME

"$QWEN35_VENV/bin/python" "$QWEN35_REPO/scripts/train/qwen35/validate_qwen35_h3_reports_r18.py" \
    --h3-manifest "$QWEN35_H3_MANIFEST" \
    --r18-manifest "$QWEN35_QUALIFICATION_MANIFEST" \
    --human-protocol "$QWEN35_H3_HUMAN_PROTOCOL" \
    --preregistration-closure "$QWEN35_H3_PREREG_CLOSURE" \
    --harness-amendment "$QWEN35_H3_HARNESS_AMENDMENT" \
    --harness-amendment-human-protocol "$QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL" \
    --harness-amendment-preregistration-closure "$QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE" \
    --attempt01-failure-closure "$QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE" \
    --validator-amendment "$QWEN35_H3_VALIDATOR_AMENDMENT" \
    --validator-amendment-human-protocol "$QWEN35_H3_VALIDATOR_AMENDMENT_HUMAN_PROTOCOL" \
    --validator-amendment-preregistration-closure "$QWEN35_H3_VALIDATOR_AMENDMENT_PREREG_CLOSURE" \
    --validator-consistency-amendment "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT" \
    --validator-consistency-amendment-human-protocol "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_PROTOCOL" \
    --validator-consistency-amendment-preregistration-closure "$QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREG_CLOSURE" \
    --validator-v2-failure-closure "$QWEN35_H3_VALIDATOR_V2_FAILURE_CLOSURE" \
    --validator-failure-forensic-closure "$QWEN35_H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE" \
    --h2-independent-validation "$QWEN35_H2_INDEPENDENT_VALIDATION" \
    --producer-root "$QWEN35_H3_PRODUCER_ROOT" \
    --repo-root "$QWEN35_REPO" \
    --code-manifest "$QWEN35_CODE_MANIFEST" \
    --validation-output "$QWEN35_OUTPUT_DIR/h3_independent_validation.json"

if find "$QWEN35_REPO" -type f -name '*.pyc' -print -quit | grep -q .; then
    echo "H3 validator execution wrote Python bytecode into the immutable source tree" >&2
    exit 2
fi
