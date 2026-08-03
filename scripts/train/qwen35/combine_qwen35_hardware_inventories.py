#!/usr/bin/env python3
"""Validate and bind one H0 hardware inventory per node in a multi-node job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_instruct.qwen35_qualification import load_qualification_manifest, sha256_file
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, action="append", required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--expected-gpus-per-node", type=int, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if len(args.inventory) != args.expected_nodes or args.expected_nodes <= 1 or args.expected_gpus_per_node <= 0:
        raise ValueError("multi-node inventory cardinality drift")
    reports = [json.loads(path.read_text()) for path in args.inventory]
    for path, report in zip(args.inventory, reports, strict=True):
        if (
            report.get("status") != "passed"
            or report.get("qualification_manifest_sha256") != qualification_sha256
            or report.get("slurm", {}).get("job_account") != qualification["scope"]["slurm_account"]
            or len(report.get("gpu_properties", [])) != args.expected_gpus_per_node
        ):
            raise AssertionError(f"invalid per-node hardware inventory: {path}")
    hostnames = [report["host"]["hostname"] for report in reports]
    if len(set(hostnames)) != args.expected_nodes:
        raise AssertionError(f"multi-node inventories do not cover distinct hosts: {hostnames}")
    job_ids = {report["slurm"]["job_id"] for report in reports}
    source_heads = {report["source"]["head"] for report in reports}
    runtimes = {json.dumps(report["runtime"], sort_keys=True) for report in reports}
    if len(job_ids) != 1 or len(source_heads) != 1 or len(runtimes) != 1:
        raise AssertionError("multi-node job, source, or runtime identity drift")
    gpu_uuids = []
    for report in reports:
        for row in report["nvidia_smi_query"]["stdout"].splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 10:
                raise ValueError(f"unexpected nvidia-smi row in combined inventory: {row!r}")
            gpu_uuids.append(fields[1])
    if len(set(gpu_uuids)) != args.expected_nodes * args.expected_gpus_per_node:
        raise AssertionError("multi-node hardware inventory contains repeated GPU UUIDs")
    report = {
        "artifact": "qwen35_multinode_hardware_inventory",
        "schema_version": 1,
        "status": "passed",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "slurm": {"job_id": next(iter(job_ids)), "job_account": qualification["scope"]["slurm_account"]},
        "source_head": next(iter(source_heads)),
        "expected_nodes": args.expected_nodes,
        "expected_gpus_per_node": args.expected_gpus_per_node,
        "hostnames": hostnames,
        "gpu_uuids": gpu_uuids,
        "inventory_sha256": {str(path.resolve()): sha256_file(path) for path in args.inventory},
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
