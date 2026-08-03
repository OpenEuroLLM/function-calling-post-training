#!/usr/bin/env python3
"""Extract NCCL collective evidence from a Qwen3.5 Chrome profiler trace."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from open_instruct.qwen35_qualification import load_qualification_manifest, sha256_file
from open_instruct.qwen35_training import write_json_atomic

NCCL_PATTERN = re.compile(r"nccl|all.?reduce|reduce.?scatter|all.?gather|broadcast", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if args.world_size <= 1 or args.num_nodes <= 0:
        raise ValueError("NCCL profile requires a distributed topology")
    trace = json.loads(args.trace.read_text())
    events = trace.get("traceEvents", [])
    matches = [event for event in events if NCCL_PATTERN.search(str(event.get("name", "")))]
    if not matches:
        raise AssertionError("profiler trace contains no NCCL/collective event")
    name_counts = collections.Counter(str(event.get("name")) for event in matches)
    duration_microseconds = sum(float(event.get("dur", 0)) for event in matches if event.get("ph") == "X")
    if duration_microseconds <= 0:
        raise AssertionError("NCCL/collective trace events contain no positive complete-event duration")
    report = {
        "artifact": "qwen35_nccl_profiler_evidence",
        "schema_version": 1,
        "status": "passed",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "trace_sha256": sha256_file(args.trace),
        "world_size": args.world_size,
        "num_nodes": args.num_nodes,
        "matched_event_count": len(matches),
        "complete_event_duration_microseconds_sum_with_overlap": duration_microseconds,
        "event_name_counts": dict(sorted(name_counts.items())),
        "duration_caveat": (
            "Chrome events can overlap and span CPU operators, CUDA runtime, and GPU kernels; the sum is evidence "
            "of observed collective activity, not a non-overlapping wall-time decomposition."
        ),
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
