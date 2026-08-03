#!/usr/bin/env python3
"""Validation-only R18 H2 amendment for JSON object member-order semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_report import validate_h2_chunked_report

AMENDMENT_SHA256 = "a210578e9e40ef6b3e06646ccd6fdfac0fc30625349913b18b2b622cacb5a566"
PREREGISTRATION_CLOSURE_SHA256 = "36e02a2a1dbcb50bd496a5beaa9c3e0b9a1a69db2e40168500817b396b24a2ca"
HUMAN_PROTOCOL_SHA256 = "57bc83df87aee2f078407a687348a54a6158f4e21d9bc0c59a2bc6d4756894b4"
REPORT_SHA256 = "d402d24ad9661f05abfbec02b92c5cb022fc392d953a24920072a4487a08e50d"
REPORT_SIZE_BYTES = 253_195_551
PRODUCER_COMMIT = "1af07d8f498595840034ba1210008058415aee9a"
QUALIFICATION_SHA256 = "679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0"
VALIDATOR_SOURCE_FILES = (
    "open_instruct/qwen35_qualification_r18_report.py",
    "scripts/train/qwen35/qwen35_h2_validator_amendment_r18_v1.json",
    "scripts/train/qwen35/validate_qwen35_h2_report_r18_amended_v1.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--amendment-manifest", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--preregistration-closure", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser.parse_args()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=text
    )
    return completed.stdout.strip() if text else completed.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_preregistration(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.amendment_manifest) != AMENDMENT_SHA256:
        raise ValueError("R18 H2 validator amendment manifest digest drift")
    if sha256_file(args.human_protocol) != HUMAN_PROTOCOL_SHA256:
        raise ValueError("R18 H2 validator amendment human-protocol digest drift")
    if sha256_file(args.preregistration_closure) != PREREGISTRATION_CLOSURE_SHA256:
        raise ValueError("R18 H2 validator amendment preregistration-closure digest drift")
    amendment = json.loads(args.amendment_manifest.read_text())
    if (
        amendment.get("schema_version") != 1
        or amendment.get("amendment_id") != "qwen35-r18-h2-independent-validator-amendment-v1"
        or amendment.get("status") != "preregistered_before_corrected_validator_implementation_or_execution"
        or amendment.get("protocol_id") != "qwen35-hardware-qualification-r18"
        or amendment.get("protocol_manifest_sha256") != QUALIFICATION_SHA256
        or amendment.get("human_protocol")
        != {
            "path": "methodology/qwen35_hardware_qualification_r18_h2_validator_amendment_v1_20260719.md",
            "sha256": HUMAN_PROTOCOL_SHA256,
        }
    ):
        raise ValueError("R18 H2 validator amendment identity drift")
    immutable = amendment.get("immutable_input", {})
    if (
        immutable.get("report_sha256") != REPORT_SHA256
        or immutable.get("report_size_bytes") != REPORT_SIZE_BYTES
        or immutable.get("producer_commit") != PRODUCER_COMMIT
        or immutable.get("slurm_job_id") != "49845033"
        or immutable.get("slurm_account") != "aifac_f02_434"
        or immutable.get("producer_report_status") != "passed"
        or immutable.get("producer_successor_gate_authorized") is not True
        or immutable.get("producer_scientific_training_authorized") is not False
    ):
        raise ValueError("R18 H2 validator amendment immutable-input drift")
    if amendment.get("execution") != {
        "device": "cpu",
        "slurm_account": "aifac_f02_434",
        "automatic_successor": False,
        "allowed_successor_on_pass": "H3_only",
        "scientific_training_authorized": False,
    }:
        raise ValueError("R18 H2 validator amendment execution-scope drift")
    closure = json.loads(args.preregistration_closure.read_text())
    if closure != {
        "artifact": "qwen35_r18_h2_validator_amendment_v1_preregistration_closure",
        "baseline_source_commit": PRODUCER_COMMIT,
        "corrected_validator_executed": False,
        "gpu_reexecution_authorized": False,
        "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
        "immutable_h2_report_sha256": REPORT_SHA256,
        "immutable_h2_report_size_bytes": REPORT_SIZE_BYTES,
        "machine_amendment_sha256": AMENDMENT_SHA256,
        "protocol_manifest_sha256": QUALIFICATION_SHA256,
        "schema_version": 1,
        "scientific_training_authorized": False,
        "status": "frozen_before_corrected_validator_implementation_or_execution",
    }:
        raise ValueError("R18 H2 validator preregistration closure content drift")
    return amendment


def _verify_producer_source(root: Path, report: dict[str, Any]) -> dict[str, str]:
    source = report.get("source_attestation", {})
    if source.get("git_commit") != PRODUCER_COMMIT or source.get("git_worktree_clean") is not True:
        raise ValueError("R18 H2 producer source identity drift")
    expected = source.get("source_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("R18 H2 producer source hash map missing")
    _git(root, "cat-file", "-e", f"{PRODUCER_COMMIT}^{{commit}}")
    observed = {
        relative: _sha256_bytes(_git(root, "show", f"{PRODUCER_COMMIT}:{relative}", text=False))
        for relative in expected
    }
    if observed != expected:
        raise ValueError("R18 H2 producer Git-object source bytes drift")
    return observed


def main() -> None:
    args = parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("validation-only R18 H2 amendment must not initialize CUDA")
    amendment = _validate_preregistration(args)
    if args.report.stat().st_size != REPORT_SIZE_BYTES or sha256_file(args.report) != REPORT_SHA256:
        raise ValueError("R18 H2 immutable input report bytes drift")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if qualification_sha256 != QUALIFICATION_SHA256:
        raise ValueError("R18 H2 qualification manifest digest drift")

    root = _source_root()
    validator_commit = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain") != "":
        raise ValueError("amended R18 H2 validation requires a clean immutable validator worktree")
    _git(root, "merge-base", "--is-ancestor", PRODUCER_COMMIT, validator_commit)
    validator_source_hashes = {relative: sha256_file(root / relative) for relative in VALIDATOR_SOURCE_FILES}
    report = json.loads(args.report.read_text())
    producer_source_hashes = _verify_producer_source(root, report)
    validation = validate_h2_chunked_report(
        report, qualification=qualification, expected_manifest_sha256=qualification_sha256
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("validation-only R18 H2 amendment initialized CUDA")
    output = {
        "artifact": "qwen35_h2_r18_independent_validation_amended_v1",
        "schema_version": 1,
        "status": "passed",
        "amendment": {
            "amendment_id": amendment["amendment_id"],
            "machine_manifest_sha256": AMENDMENT_SHA256,
            "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
            "preregistration_closure_sha256": PREREGISTRATION_CLOSURE_SHA256,
        },
        "qualification_manifest_sha256": qualification_sha256,
        "report": {
            "path": str(args.report.resolve()),
            "size_bytes": args.report.stat().st_size,
            "sha256": REPORT_SHA256,
        },
        "producer_source": {
            "commit": PRODUCER_COMMIT,
            "source_files_sha256": producer_source_hashes,
        },
        "validator_source": {
            "commit": validator_commit,
            "worktree_clean": True,
            "source_files_sha256": validator_source_hashes,
            "code_manifest_path": str(args.code_manifest.resolve()),
            "code_manifest_sha256": sha256_file(args.code_manifest),
        },
        "execution": {"device": "cpu", "cuda_initialized": False},
        "validation": validation,
        "successor_gate_authorized": "H3_only",
        "scientific_training_authorized": False,
    }
    _write_json_atomic(args.validation_output, output)
    print(json.dumps({"output": str(args.validation_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
