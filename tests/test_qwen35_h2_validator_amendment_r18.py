from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.train.qwen35.validate_qwen35_h2_report_r18_amended_v1 import (
    AMENDMENT_SHA256,
    HUMAN_PROTOCOL_SHA256,
    PRODUCER_COMMIT,
    QUALIFICATION_SHA256,
    REPORT_SHA256,
    REPORT_SIZE_BYTES,
    VALIDATOR_SOURCE_FILES,
    _verify_producer_source,
)

ROOT = Path(__file__).parents[1]
AMENDMENT = ROOT / "scripts/train/qwen35/qwen35_h2_validator_amendment_r18_v1.json"
VALIDATION_WRAPPER = ROOT / "scripts/train/qwen35/leonardo_validate_h2_report_r18_amended_v1.sbatch"
PRODUCER_SOURCE_HASHES = {
    "open_instruct/qwen35_chunked_loss.py": "295f5452878a6bcc15c446000fa061fe93b6f6d8df8448f264552c4db7918d90",
    "open_instruct/qwen35_qualification_r18.py": "92471174eb19c9e38cc03ab6313a36b773e5181cb1791d7ac255ab74e9c15d26",
    "open_instruct/qwen35_qualification_r18_report.py": "c559ce3ae1a18c696d13aba6cd60c7b8ad755c35b59c5da34501d271dc01ac2e",
    "scripts/train/qwen35/validate_qwen35_chunked_loss_r18.py": "125e85a6e91dc0a7847f4495566b913ee9a43421c0f0b2f71558205d7027dced",
    "scripts/train/qwen35/validate_qwen35_h2_report_r18.py": "2d512f4e99808197fc2d80d318ba5a485dc188631b236facb8d8688446b19ee3",
}


def producer_report_stub() -> dict:
    return {
        "source_attestation": {
            "git_commit": PRODUCER_COMMIT,
            "git_worktree_clean": True,
            "source_files_sha256": copy.deepcopy(PRODUCER_SOURCE_HASHES),
        }
    }


def test_amendment_manifest_binds_immutable_report_and_protocol() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["protocol_manifest_sha256"] == QUALIFICATION_SHA256
    assert amendment["human_protocol"]["sha256"] == HUMAN_PROTOCOL_SHA256
    assert amendment["immutable_input"] == {
        "report_sha256": REPORT_SHA256,
        "report_size_bytes": REPORT_SIZE_BYTES,
        "producer_commit": PRODUCER_COMMIT,
        "slurm_job_id": "49845033",
        "slurm_account": "aifac_f02_434",
        "slurm_state": "FAILED",
        "slurm_exit_code": "1:0",
        "producer_report_status": "passed",
        "producer_successor_gate_authorized": True,
        "producer_scientific_training_authorized": False,
    }
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == AMENDMENT_SHA256


def test_amended_validator_reconstructs_exact_producer_bytes_from_git_objects() -> None:
    observed = _verify_producer_source(ROOT, producer_report_stub())
    assert observed == PRODUCER_SOURCE_HASHES


def test_amended_validator_rejects_producer_git_object_hash_drift() -> None:
    report = producer_report_stub()
    report["source_attestation"]["source_files_sha256"]["open_instruct/qwen35_chunked_loss.py"] = "0" * 64
    with pytest.raises(ValueError, match="Git-object source bytes drift"):
        _verify_producer_source(ROOT, report)


def test_amended_validator_source_closure_paths_exist_and_are_unique() -> None:
    assert len(VALIDATOR_SOURCE_FILES) == len(set(VALIDATOR_SOURCE_FILES))
    assert all((ROOT / relative).is_file() for relative in VALIDATOR_SOURCE_FILES)


def test_validation_wrapper_isolates_python_bytecode_from_source_tree() -> None:
    source = VALIDATION_WRAPPER.read_text()
    assert "#SBATCH --account=aifac_f02_434" in source
    assert "#SBATCH --partition=lrd_all_serial" in source
    assert "#SBATCH --mem=24G" in source
    assert "pre-existing Python bytecode" in source
    assert 'export PYTHONPYCACHEPREFIX="$QWEN35_OUTPUT_DIR/pycache"' in source
    assert "Validator execution wrote Python bytecode into the source tree" in source
