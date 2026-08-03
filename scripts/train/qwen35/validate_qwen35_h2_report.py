#!/usr/bin/env python3
"""Independently validate a saved Qwen3.5 selective-Liger H2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_instruct.qwen35_qualification import load_qualification_manifest, sha256_file, validate_h2_liger_report
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--h2-report", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h2_report = json.loads(args.h2_report.read_text())
    validation = validate_h2_liger_report(
        h2_report,
        qualification=qualification,
        expected_manifest_sha256=qualification_sha256,
    )
    output = {
        "artifact": "qwen35_h2_independent_validation",
        "schema_version": 1,
        "status": "passed",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "h2_report_path": str(args.h2_report.resolve()),
        "h2_report_sha256": sha256_file(args.h2_report),
        "validation": validation,
    }
    write_json_atomic(args.validation_output, output)
    print(json.dumps(output, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
