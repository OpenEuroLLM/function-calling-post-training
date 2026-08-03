#!/usr/bin/env python3
"""Independently validate a saved R17 matched-reference H2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_instruct.qwen35_qualification_r17 import load_qualification_manifest, validate_h2_liger_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-failed-evidence", action="store_true")
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, manifest_sha256 = load_qualification_manifest(args.qualification_manifest)
    report = json.loads(args.report.read_text())
    validation = validate_h2_liger_report(
        report,
        qualification=qualification,
        expected_manifest_sha256=manifest_sha256,
        require_numerical_pass=not args.allow_failed_evidence,
    )
    output = {
        "artifact": "qwen35_h2_r17_independent_validation",
        "schema_version": 1,
        "status": validation["status"],
        "qualification_manifest_sha256": manifest_sha256,
        "report_path": str(args.report.resolve()),
        "validation": validation,
        "successor_gate_authorized": validation["numerical_status"] == "passed",
        "scientific_training_authorized": False,
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.validation_output), "status": output["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
