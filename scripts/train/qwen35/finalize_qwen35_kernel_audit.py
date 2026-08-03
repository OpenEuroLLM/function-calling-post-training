#!/usr/bin/env python3
"""Fail closed unless every observed GPU kernel is reviewed and mapped to pinned source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_instruct.qwen35_qualification import load_qualification_manifest, sha256_file
from open_instruct.qwen35_training import write_json_atomic

ALLOWED_COMPONENTS = {
    "attention",
    "gated_delta_recurrence",
    "causal_convolution",
    "fused_cross_entropy",
    "matrix_multiplication",
    "optimizer",
    "collective",
    "memory_or_runtime",
    "elementwise_or_reduction",
}
REQUIRED_COMPONENTS = {
    "attention",
    "gated_delta_recurrence",
    "causal_convolution",
    "fused_cross_entropy",
    "matrix_multiplication",
    "optimizer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--profile-validation", type=Path, required=True)
    parser.add_argument("--reviewed-mapping", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    profile = json.loads(args.profile_validation.read_text())
    mapping = json.loads(args.reviewed_mapping.read_text())
    if profile.get("qualification_manifest_sha256") != qualification_sha256:
        raise ValueError("profile validation qualification-manifest drift")
    if mapping.get("qualification_manifest_sha256") != qualification_sha256:
        raise ValueError("reviewed kernel mapping qualification-manifest drift")
    if mapping.get("trace_sha256") != profile.get("trace_sha256"):
        raise ValueError("reviewed kernel mapping trace drift")
    if mapping.get("review_status") != "reviewed_pre_h4_disposition":
        raise ValueError("kernel mapping does not carry the required review status")
    observed = set(profile.get("kernel_name_counts", {}))
    reviewed = mapping.get("kernels")
    if not isinstance(reviewed, dict) or set(reviewed) != observed:
        missing = sorted(observed - set(reviewed or {}))
        extra = sorted(set(reviewed or {}) - observed)
        raise ValueError(
            f"reviewed kernel-name set differs from the trace; missing={missing[:10]}, extra={extra[:10]}"
        )
    components = set()
    for kernel, row in reviewed.items():
        if row.get("allowed") is not True:
            raise AssertionError(f"review rejected kernel {kernel!r}")
        component = row.get("component")
        if component not in ALLOWED_COMPONENTS:
            raise ValueError(f"kernel {kernel!r} has unknown component {component!r}")
        if not row.get("source_repository") or not row.get("source_identity") or not row.get("rationale"):
            raise ValueError(f"kernel {kernel!r} lacks source identity or rationale")
        components.add(component)
    missing_components = sorted(REQUIRED_COMPONENTS - components)
    if missing_components:
        raise AssertionError(f"reviewed kernel mapping lacks required components: {missing_components}")
    report = {
        "artifact": "qwen35_h4_final_kernel_audit",
        "schema_version": 1,
        "status": "passed",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "profile_validation_sha256": sha256_file(args.profile_validation),
        "reviewed_mapping_sha256": sha256_file(args.reviewed_mapping),
        "trace_sha256": profile["trace_sha256"],
        "reviewed_kernel_count": len(reviewed),
        "observed_components": sorted(components),
        "required_components": sorted(REQUIRED_COMPONENTS),
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
