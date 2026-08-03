#!/usr/bin/env python3
"""Independently validate a complete successful R18 H2 CUDA report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_report import validate_h2_chunked_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser.parse_args()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    args = parse_args()
    qualification, manifest_sha256 = load_qualification_manifest(args.qualification_manifest)
    report = json.loads(args.report.read_text())
    source = report.get("source_attestation", {})
    root = _source_root()
    if source.get("git_commit") != _git_output(root, "rev-parse", "HEAD"):
        raise ValueError("R18 independent validator source commit differs from the producer attestation")
    if _git_output(root, "status", "--porcelain") != "":
        raise ValueError("R18 independent validation requires a clean immutable source worktree")
    expected_hashes = source.get("source_files_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("R18 report has no source-file hash map")
    observed_hashes = {relative: sha256_file(root / relative) for relative in expected_hashes}
    if observed_hashes != expected_hashes:
        raise ValueError("R18 producer/validator source-file bytes differ from the report attestation")
    validation = validate_h2_chunked_report(
        report, qualification=qualification, expected_manifest_sha256=manifest_sha256
    )
    output = {
        "artifact": "qwen35_h2_r18_independent_validation",
        "schema_version": 1,
        "status": "passed",
        "qualification_manifest_sha256": manifest_sha256,
        "report_path": str(args.report.resolve()),
        "report_sha256": sha256_file(args.report),
        "source_commit": source["git_commit"],
        "source_files_sha256": observed_hashes,
        "validation": validation,
        "successor_gate_authorized": True,
        "scientific_training_authorized": False,
    }
    _write_json_atomic(args.validation_output, output)
    print(json.dumps({"output": str(args.validation_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
