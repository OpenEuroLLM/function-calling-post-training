#!/usr/bin/env python3
"""Fail closed unless the exact R18 H4 accelerator-event set and required components were reviewed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification_r18_h4 import load_h4_contract, load_strict_json, sha256_file
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h4-contract", type=Path, required=True)
    parser.add_argument("--profile-validation", type=Path, required=True)
    parser.add_argument("--reviewed-mapping", type=Path, required=True)
    parser.add_argument("--candidate-chunk-size", type=int, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    h4, h4_sha256 = load_h4_contract(args.h4_contract)
    profile = load_strict_json(args.profile_validation)
    mapping = load_strict_json(args.reviewed_mapping)
    if args.candidate_chunk_size not in h4["candidate_chunk_sizes_in_execution_order"]:
        raise ValueError("R18 H4 kernel finalizer received an unknown candidate")
    if (
        profile.get("status") != "automated_profile_passed_pending_manual_kernel_mapping"
        or profile.get("h4_contract_sha256") != h4_sha256
        or profile.get("candidate_chunk_size") != args.candidate_chunk_size
    ):
        raise ValueError("R18 H4 profile validation identity/status drift")
    expected_mapping_scalars = {
        "artifact": "qwen35_r18_h4_reviewed_kernel_mapping",
        "schema_version": 1,
        "review_status": "reviewed_against_exact_trace_and_pinned_source_before_H4_disposition",
        "h4_contract_sha256": h4_sha256,
        "trace_sha256": profile["trace_sha256"],
        "candidate_chunk_size": args.candidate_chunk_size,
        "liger_execution_observed": False,
    }
    for key, expected in expected_mapping_scalars.items():
        if mapping.get(key) != expected:
            raise ValueError(f"R18 H4 reviewed mapping drift for {key}")

    observed_rows = profile.get("trace_inventory", {}).get("observed_accelerator_events")
    reviewed_rows = mapping.get("accelerator_events")
    if not isinstance(observed_rows, list) or not isinstance(reviewed_rows, list):
        raise ValueError("R18 H4 observed or reviewed accelerator event rows are absent")
    observed_by_name = {row["exact_name"]: row for row in observed_rows}
    reviewed_by_name = {row.get("exact_name"): row for row in reviewed_rows}
    if len(observed_by_name) != len(observed_rows) or len(reviewed_by_name) != len(reviewed_rows):
        raise ValueError("R18 H4 observed or reviewed accelerator event names are duplicated")
    if set(observed_by_name) != set(reviewed_by_name):
        raise ValueError(
            "R18 H4 reviewed accelerator-event set drift: "
            f"missing={sorted(set(observed_by_name) - set(reviewed_by_name))[:10]}, "
            f"extra={sorted(set(reviewed_by_name) - set(observed_by_name))[:10]}"
        )
    allowed_components = set(h4["kernel_path"]["required_components"])
    observed_components = set()
    for name, observed in observed_by_name.items():
        reviewed = reviewed_by_name[name]
        if (
            reviewed.get("observed_categories") != observed["categories"]
            or reviewed.get("observed_count") != observed["count"]
            or reviewed.get("observed_duration_microseconds") != observed["duration_microseconds"]
        ):
            raise ValueError(f"R18 H4 reviewed count/category/duration drift for {name!r}")
        if reviewed.get("disposition") != "allowed":
            raise AssertionError(f"R18 H4 manual review rejected accelerator event {name!r}")
        components = reviewed.get("semantic_components")
        if not isinstance(components, list) or not components or len(components) != len(set(components)):
            raise ValueError(f"R18 H4 accelerator event {name!r} has invalid semantic components")
        if not set(components) <= allowed_components:
            raise ValueError(f"R18 H4 accelerator event {name!r} has an unregistered semantic component")
        for key in ("source_identity", "source_file_or_implementation_family", "rationale"):
            if not isinstance(reviewed.get(key), str) or not reviewed[key].strip():
                raise ValueError(f"R18 H4 accelerator event {name!r} lacks {key}")
        if "liger" in name.lower() or "liger" in reviewed["source_identity"].lower():
            raise AssertionError(f"R18 H4 observed or mapped a forbidden Liger event: {name!r}")
        observed_components.update(components)

    all_trace_names = {
        row["exact_name"] for row in profile.get("trace_inventory", {}).get("observed_all_event_names", [])
    }
    component_evidence = mapping.get("required_component_evidence")
    if not isinstance(component_evidence, dict) or set(component_evidence) != allowed_components:
        raise ValueError("R18 H4 required-component evidence set drift")
    for component in sorted(allowed_components):
        rows = component_evidence[component]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"R18 H4 component {component!r} has no exact trace evidence")
        for row in rows:
            name = row.get("observed_exact_event_name")
            if name not in all_trace_names:
                raise ValueError(f"R18 H4 component {component!r} cites an unobserved trace event {name!r}")
            for key in ("source_identity", "source_file_or_implementation_family", "rationale"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"R18 H4 component {component!r} evidence lacks {key}")
            if "liger" in str(name).lower() or "liger" in row["source_identity"].lower():
                raise AssertionError(f"R18 H4 component evidence cites forbidden Liger execution: {name!r}")
        observed_components.add(component)
    missing = sorted(allowed_components - observed_components)
    if missing:
        raise AssertionError(f"R18 H4 reviewed trace lacks required components: {missing}")

    return {
        "artifact": "qwen35_r18_h4_final_kernel_audit",
        "candidate_chunk_size": args.candidate_chunk_size,
        "h4_contract_sha256": h4_sha256,
        "observed_accelerator_event_count": profile["trace_inventory"]["accelerator_event_count"],
        "observed_components": sorted(observed_components),
        "profile_validation_sha256": sha256_file(args.profile_validation),
        "required_components": sorted(allowed_components),
        "reviewed_distinct_accelerator_event_names": len(reviewed_rows),
        "reviewed_mapping_sha256": sha256_file(args.reviewed_mapping),
        "schema_version": 1,
        "status": "passed_kernel_mapping_only_H4_set_validation_still_required",
        "trace_sha256": profile["trace_sha256"],
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
