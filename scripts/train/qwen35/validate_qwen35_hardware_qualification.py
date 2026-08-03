#!/usr/bin/env python3
"""Independent H9 closure audit for the complete Qwen3.5 R1 hardware campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification import (
    load_qualification_manifest,
    sha256_file,
    validate_h1_reference_report,
    validate_h2_liger_report,
)
from open_instruct.qwen35_training import write_json_atomic

REQUIRED_ARTIFACTS = {
    "h1_inventory",
    "h1_reference",
    "h2_liger",
    "h3_inventory",
    "h3_full_ddp_ga",
    "h3_isolated_ddp",
    "h3_schedule_sharding",
    "h4_inventory",
    "h4_profile",
    "h4_profile_validation",
    "h4_kernel_audit",
    "h5_inventory",
    "h5_output_validation",
    "h5_nccl_profile",
    "h6_continuous_inventory",
    "h6_continuous_output_validation",
    "h6_resumed_inventory",
    "h6_resumed_output_validation",
    "h6_resume_parity",
    "h7_t4_inventory",
    "h7_t4_output_validation",
    "h7_t8_inventory",
    "h7_t8_output_validation",
    "h7_topology_comparison",
    "h7_t8_profile_inventory",
    "h7_t8_profile_output_validation",
    "h7_t8_nccl_profile",
    "h8_inventory",
    "h8_coarse_output_validation",
    "h8_checkpoint_comparison",
    "h8_reporting_overhead",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def read_artifacts(campaign_path: Path, campaign: dict) -> tuple[dict[str, Any], dict[str, str]]:
    rows = campaign.get("artifacts")
    if not isinstance(rows, dict) or set(rows) != REQUIRED_ARTIFACTS:
        raise ValueError(
            f"campaign artifact set drift: missing={sorted(REQUIRED_ARTIFACTS - set(rows or {}))}, "
            f"extra={sorted(set(rows or {}) - REQUIRED_ARTIFACTS)}"
        )
    values = {}
    hashes = {}
    for label, row in rows.items():
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError(f"campaign artifact {label} has no path/hash")
        path = Path(row["path"])
        if not path.is_absolute():
            path = campaign_path.parent / path
        path = path.resolve()
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(f"campaign artifact hash drift for {label}")
        values[label] = json.loads(path.read_text())
        hashes[label] = digest
    return values, hashes


def require_status(values: dict[str, Any], label: str, expected: str = "passed") -> None:
    if values[label].get("status") != expected:
        raise AssertionError(
            f"campaign artifact {label} status is {values[label].get('status')!r}, expected {expected!r}"
        )


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    campaign = json.loads(args.campaign_manifest.read_text())
    if campaign.get("schema_version") != 1 or campaign.get("campaign_id") != "qwen35-hardware-qualification-r15":
        raise ValueError("unsupported hardware-qualification campaign manifest")
    if campaign.get("qualification_manifest_sha256") != qualification_sha256:
        raise ValueError("campaign uses a different qualification manifest")
    if campaign.get("gate_order") != [f"H{i}" for i in range(10)]:
        raise ValueError("campaign gate order drift")
    if campaign.get("slurm_account") != qualification["scope"]["slurm_account"]:
        raise ValueError("campaign Slurm-account drift")
    values, hashes = read_artifacts(args.campaign_manifest.resolve(), campaign)

    indirectly_bound = {"h6_resume_parity", "h8_checkpoint_comparison"}
    for label, value in values.items():
        if label not in indirectly_bound and value.get("qualification_manifest_sha256") != qualification_sha256:
            raise AssertionError(f"campaign artifact {label} is not bound to the frozen qualification manifest")
    if values["h6_resumed_output_validation"].get("resume_parity_sha256") != hashes["h6_resume_parity"]:
        raise AssertionError("H6 output validation does not bind the campaign resume-parity artifact")
    if values["h8_reporting_overhead"].get("checkpoint_comparison_sha256") != hashes["h8_checkpoint_comparison"]:
        raise AssertionError("H8 overhead report does not bind the campaign checkpoint comparison")

    for label in (
        "h1_inventory",
        "h3_inventory",
        "h4_inventory",
        "h5_inventory",
        "h6_continuous_inventory",
        "h6_resumed_inventory",
        "h7_t4_inventory",
        "h7_t8_inventory",
        "h7_t8_profile_inventory",
        "h8_inventory",
    ):
        require_status(values, label)
        if values[label].get("slurm", {}).get("job_account") != "aifac_f02_434":
            raise AssertionError(f"{label} did not execute under the personal account")
    h1_validation = validate_h1_reference_report(values["h1_reference"], expected_manifest_sha256=qualification_sha256)
    h2_validation = validate_h2_liger_report(
        values["h2_liger"], qualification=qualification, expected_manifest_sha256=qualification_sha256
    )
    for label in (
        "h3_full_ddp_ga",
        "h3_isolated_ddp",
        "h3_schedule_sharding",
        "h4_kernel_audit",
        "h5_output_validation",
        "h5_nccl_profile",
        "h6_continuous_output_validation",
        "h6_resumed_output_validation",
        "h6_resume_parity",
        "h7_t4_output_validation",
        "h7_t8_output_validation",
        "h7_topology_comparison",
        "h7_t8_profile_output_validation",
        "h7_t8_nccl_profile",
        "h8_coarse_output_validation",
        "h8_checkpoint_comparison",
        "h8_reporting_overhead",
    ):
        require_status(values, label)
    if values["h4_profile"].get("status") != "captured_pending_kernel_audit":
        raise AssertionError("H4 raw hardware profile status drift")
    if values["h4_profile_validation"].get("status") != (
        "required_categories_passed_pending_manual_kernel_source_review"
    ):
        raise AssertionError("H4 profile validation did not pass its automated phase")
    if values["h3_full_ddp_ga"].get("includes_zero_target_rank") is not True:
        raise AssertionError("H3 did not exercise an entirely zero-target DDP rank")
    if values["h3_isolated_ddp"].get("includes_zero_target_rank") is not True:
        raise AssertionError("H3 isolated-head proof omitted its zero-target DDP rank")
    if values["h3_full_ddp_ga"].get("optimizer_floating_dtypes") != ["torch.float32"]:
        raise AssertionError("H3 optimizer-state dtype drift")
    if values["h6_resume_parity"].get("atol") != 0 or values["h6_resume_parity"].get("rtol") != 0:
        raise AssertionError("H6 resume parity was not zero tolerance")
    if values["h6_resume_parity"].get("model", {}).get("bit_exact") is not True:
        raise AssertionError("H6 resumed model is not bit exact")
    if values["h8_checkpoint_comparison"].get("model", {}).get("bit_exact") is not True:
        raise AssertionError("H8 fine/coarse reporting runs are not bit exact")
    if values["h7_topology_comparison"].get("timing_decision", {}).get("repeat_required") is not False:
        raise AssertionError("H7 topology timing remains unstable")
    selected_topology = values["h7_topology_comparison"].get("timing_decision", {}).get("selected_topology")
    if selected_topology not in {"T4", "T8"}:
        raise AssertionError("H7 did not select a recognized topology")

    report = {
        "artifact": "qwen35_h9_hardware_qualification_closure",
        "schema_version": 1,
        "status": "qualified_with_topology_selection",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "campaign_manifest_sha256": sha256_file(args.campaign_manifest),
        "verified_artifact_hashes": hashes,
        "verified_gate_order": campaign["gate_order"],
        "h1_independent_semantic_validation": h1_validation,
        "h2_independent_numerical_validation": h2_validation,
        "selected_topology": selected_topology,
        "scientific_training_authorized_by_this_report": False,
        "final_evaluation_performed": False,
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
