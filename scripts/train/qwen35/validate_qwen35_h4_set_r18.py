#!/usr/bin/env python3
"""Independently close the primary R18 H4 four-candidate set or authorize its single timing repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification_r18_h4 import (
    load_h4_contract,
    load_strict_json,
    select_chunk_size,
    sha256_file,
    timing_statistics,
)
from open_instruct.qwen35_training import write_json_atomic


def parse_keyed_path(value: str) -> tuple[int, Path]:
    key, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected CHUNK_SIZE=PATH")
    try:
        chunk_size = int(key)
    except ValueError as error:
        raise argparse.ArgumentTypeError("chunk size must be an integer") from error
    return chunk_size, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h4-contract", type=Path, required=True)
    parser.add_argument("--candidate-report", type=parse_keyed_path, action="append", required=True)
    parser.add_argument("--kernel-audit", type=parse_keyed_path, action="append", required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _indexed(rows: list[tuple[int, Path]], expected: list[int], *, label: str) -> dict[int, Path]:
    result = dict(rows)
    if len(result) != len(rows) or list(sorted(result)) != sorted(expected) or len(rows) != len(expected):
        raise ValueError(f"H4 {label} candidate set drift")
    return result


def validate(args: argparse.Namespace) -> dict[str, Any]:
    h4, h4_sha256 = load_h4_contract(args.h4_contract)
    expected = h4["candidate_chunk_sizes_in_execution_order"]
    candidate_paths = _indexed(args.candidate_report, expected, label="candidate-report")
    kernel_paths = _indexed(args.kernel_audit, expected, label="kernel-audit")
    rows = []
    report_hashes = {}
    kernel_hashes = {}
    job_ids = set()
    source_commits = set()
    qualification_hashes = set()
    any_unstable = False
    for chunk_size in expected:
        candidate = load_strict_json(candidate_paths[chunk_size])
        kernel = load_strict_json(kernel_paths[chunk_size])
        if (
            candidate.get("artifact") != "qwen35_r18_h4_candidate_automated_validation"
            or candidate.get("status") != "automated_candidate_passed_pending_manual_kernel_mapping"
            or candidate.get("candidate_chunk_size") != chunk_size
            or candidate.get("h4_contract_sha256") != h4_sha256
            or candidate.get("eligible_pending_manual_kernel_mapping") is not True
            or candidate.get("slurm_account") != "aifac_f02_434"
        ):
            raise ValueError(f"H4 candidate {chunk_size} automated validation drift")
        if (
            kernel.get("artifact") != "qwen35_r18_h4_final_kernel_audit"
            or kernel.get("status") != "passed_kernel_mapping_only_H4_set_validation_still_required"
            or kernel.get("candidate_chunk_size") != chunk_size
            or kernel.get("h4_contract_sha256") != h4_sha256
            or kernel.get("profile_validation_sha256") != candidate.get("profile_validation_sha256")
        ):
            raise ValueError(f"H4 candidate {chunk_size} kernel audit drift")
        measured = candidate.get("measured_synchronized_update_seconds")
        recomputed = timing_statistics(measured)
        if candidate.get("timing_statistics") != recomputed:
            raise ValueError(f"H4 candidate {chunk_size} timing statistics drift")
        unstable = recomputed["coefficient_of_variation"] > float(
            h4["timing_selection"]["maximum_coefficient_of_variation"]
        )
        if candidate.get("timing_coefficient_of_variation_exceeds_threshold") is not unstable:
            raise ValueError(f"H4 candidate {chunk_size} timing CV disposition drift")
        any_unstable |= unstable
        rows.append(
            {
                "chunk_size": chunk_size,
                "eligible": True,
                "measured_update_seconds": measured,
                "timing_statistics": recomputed,
            }
        )
        job_ids.add(candidate.get("slurm_job_id"))
        source_commits.add(candidate.get("source_commit"))
        qualification_hashes.add(candidate.get("qualification_manifest_sha256"))
        report_hashes[str(chunk_size)] = sha256_file(candidate_paths[chunk_size])
        kernel_hashes[str(chunk_size)] = sha256_file(kernel_paths[chunk_size])
    if None in job_ids or len(job_ids) != 4:
        raise ValueError("H4 primary candidates must come from four distinct Slurm jobs")
    if len(source_commits) != 1 or len(qualification_hashes) != 1:
        raise ValueError("H4 primary candidates mix source commits or qualification manifests")

    base = {
        "artifact": "qwen35_r18_h4_primary_set_independent_validation",
        "automatic_successor": False,
        "candidate_report_sha256": report_hashes,
        "candidate_rows": rows,
        "h4_contract_sha256": h4_sha256,
        "kernel_audit_sha256": kernel_hashes,
        "qualification_manifest_sha256": next(iter(qualification_hashes)),
        "schema_version": 1,
        "scientific_training_authorized": False,
        "slurm_job_ids": sorted(job_ids),
        "source_commit": next(iter(source_commits)),
        "timing_set": "primary",
    }
    if any_unstable:
        return {
            **base,
            "allowed_successor": None,
            "selection": None,
            "single_complete_four_candidate_timing_repeat_authorized": True,
            "status": "timing_repeat_required_H4_not_passed",
        }
    selection = select_chunk_size(rows, h4)
    return {
        **base,
        "allowed_successor": "H5_only",
        "selection": selection,
        "single_complete_four_candidate_timing_repeat_authorized": False,
        "status": "passed_H5_only_authorized",
    }


def main() -> int:
    args = parse_args()
    if args.report_output.exists():
        raise FileExistsError(args.report_output)
    report = validate(args)
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
