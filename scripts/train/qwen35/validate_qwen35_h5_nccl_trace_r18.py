#!/usr/bin/env python3
"""Inventory and validate exact NCCL/collective events in the R18 H5 trace."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from pathlib import Path

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file
from open_instruct.qwen35_qualification_r18_h5 import (
    H5_FINAL_STEP,
    H5_SELECTED_CHUNK_SIZE,
    load_h5_contract,
    load_h5_harness_amendment,
    load_h5_harness_amendment_r2,
)
from open_instruct.qwen35_training import write_json_atomic

COLLECTIVE_PATTERN = re.compile(
    r"nccl|c10d|processgroup|all.?reduce|reduce.?scatter|all.?gather|broadcast|barrier", re.IGNORECASE
)
ALL_REDUCE_PATTERN = re.compile(r"all.?reduce", re.IGNORECASE)
ERROR_PATTERN = re.compile(r"timeout|abort|unhandled.?cuda|nccl.?error|collective.?mismatch", re.IGNORECASE)
MALFORMED_PROCESS_GROUP_DESCRIPTION = b'"Process Group Description": ,'
NULL_PROCESS_GROUP_DESCRIPTION = b'"Process Group Description": null,'
GENERIC_EMPTY_JSON_VALUE = b'": ,'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-contract", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--preregistration-closure", type=Path, required=True)
    parser.add_argument("--harness-amendment", type=Path, required=True)
    parser.add_argument("--harness-human-amendment", type=Path, required=True)
    parser.add_argument("--attempt01-failure-closure", type=Path, required=True)
    parser.add_argument("--harness-amendment-r2", type=Path, required=True)
    parser.add_argument("--harness-human-amendment-r2", type=Path, required=True)
    parser.add_argument("--attempt02-failure-closure", type=Path, required=True)
    parser.add_argument("--reload-type-diagnostic", type=Path, required=True)
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sanitized-trace-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def count_binary_pattern(path: Path, pattern: bytes, *, chunk_size: int = 1 << 20) -> int:
    """Count non-overlapping binary occurrences, including chunk-boundary matches."""

    if not pattern or chunk_size <= 0:
        raise ValueError("binary pattern and chunk size must be positive")
    count = 0
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            data = carry + chunk
            search_limit = max(0, len(data) - len(pattern) + 1)
            start = 0
            while True:
                index = data.find(pattern, start, search_limit + len(pattern) - 1)
                if index < 0 or index >= search_limit:
                    break
                count += 1
                start = index + len(pattern)
            carry = data[-(len(pattern) - 1) :] if len(pattern) > 1 else b""
    return count


def sanitize_profiler_trace(raw_path: Path, output_path: Path, *, chunk_size: int = 1 << 20) -> dict:
    """Publish a raw-bound derivative repairing only PyTorch's empty PG description."""

    if not raw_path.is_file() or raw_path.is_symlink() or raw_path.stat().st_size <= 0:
        raise FileNotFoundError(f"R18 H5 raw profiler trace is absent or invalid: {raw_path}")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    if raw_path.resolve() == output_path.resolve():
        raise ValueError("R18 H5 sanitized profiler trace may not overwrite the raw trace")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)

    replacements = 0
    buffer = b""
    try:
        with raw_path.open("rb") as source, temporary.open("xb") as target:
            while chunk := source.read(chunk_size):
                buffer += chunk
                while True:
                    index = buffer.find(MALFORMED_PROCESS_GROUP_DESCRIPTION)
                    if index >= 0:
                        target.write(buffer[:index])
                        target.write(NULL_PROCESS_GROUP_DESCRIPTION)
                        replacements += 1
                        buffer = buffer[index + len(MALFORMED_PROCESS_GROUP_DESCRIPTION) :]
                        continue
                    safe_bytes = len(buffer) - (len(MALFORMED_PROCESS_GROUP_DESCRIPTION) - 1)
                    if safe_bytes > 0:
                        target.write(buffer[:safe_bytes])
                        buffer = buffer[safe_bytes:]
                    break
            target.write(buffer)
            target.flush()
            os.fsync(target.fileno())

        expected_size = raw_path.stat().st_size + replacements * (
            len(NULL_PROCESS_GROUP_DESCRIPTION) - len(MALFORMED_PROCESS_GROUP_DESCRIPTION)
        )
        if temporary.stat().st_size != expected_size:
            raise AssertionError("R18 H5 profiler-trace sanitizer size accounting drift")
        if count_binary_pattern(temporary, MALFORMED_PROCESS_GROUP_DESCRIPTION, chunk_size=chunk_size) != 0:
            raise AssertionError("R18 H5 profiler-trace sanitizer retained the exact malformed field")
        if count_binary_pattern(temporary, GENERIC_EMPTY_JSON_VALUE, chunk_size=chunk_size) != 0:
            raise AssertionError("R18 H5 profiler-trace sanitizer found an unrecognized empty JSON value")
        os.link(temporary, output_path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    raw_sha256 = sha256_file(raw_path)
    sanitized_sha256 = sha256_file(output_path)
    if replacements == 0 and (
        raw_path.stat().st_size != output_path.stat().st_size or raw_sha256 != sanitized_sha256
    ):
        raise AssertionError("R18 H5 zero-replacement sanitizer output is not byte-identical")
    return {
        "from_ascii": MALFORMED_PROCESS_GROUP_DESCRIPTION.decode("ascii"),
        "generic_empty_json_values_remaining": 0,
        "raw_trace_bytes": raw_path.stat().st_size,
        "raw_trace_sha256": raw_sha256,
        "replacement_count": replacements,
        "replacement_size_delta_bytes_each": 4,
        "sanitized_trace_bytes": output_path.stat().st_size,
        "sanitized_trace_sha256": sanitized_sha256,
        "streaming_chunk_size_bytes": chunk_size,
        "to_ascii": NULL_PROCESS_GROUP_DESCRIPTION.decode("ascii"),
    }


def validate_trace(trace: dict, *, trace_path: Path) -> dict:
    events = trace.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise ValueError("R18 H5 profiler trace contains no traceEvents")
    aggregate: dict[str, dict] = {}
    rejected_error_names = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"R18 H5 trace event {index} is not an object")
        name = event.get("name")
        if not isinstance(name, str) or not name or not COLLECTIVE_PATTERN.search(name):
            continue
        if ERROR_PATTERN.search(name):
            rejected_error_names.append(name)
        duration = event.get("dur", 0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise ValueError(f"R18 H5 collective event {index} has invalid duration")
        if duration < 0:
            raise ValueError(f"R18 H5 collective event {index} has negative duration")
        row = aggregate.setdefault(
            name,
            {
                "categories": set(),
                "complete_event_count": 0,
                "complete_event_duration_microseconds_sum_with_overlap": 0.0,
                "count": 0,
                "phases": collections.Counter(),
            },
        )
        row["categories"].add(str(event.get("cat", "")))
        row["count"] += 1
        row["phases"][str(event.get("ph", ""))] += 1
        if event.get("ph") == "X" and duration > 0:
            row["complete_event_count"] += 1
            row["complete_event_duration_microseconds_sum_with_overlap"] += float(duration)
    if rejected_error_names:
        raise AssertionError(
            f"R18 H5 collective trace contains error-like event names: {sorted(set(rejected_error_names))}"
        )
    if not aggregate:
        raise AssertionError("R18 H5 profiler trace contains no NCCL/collective event")
    exact_names = sorted(aggregate)
    if not any("nccl" in name.lower() for name in exact_names):
        raise AssertionError("R18 H5 profiler trace contains no NCCL-identified event")
    if not any(ALL_REDUCE_PATTERN.search(name) for name in exact_names):
        raise AssertionError("R18 H5 profiler trace contains no all-reduce event")
    total_complete = sum(row["complete_event_count"] for row in aggregate.values())
    total_duration = sum(row["complete_event_duration_microseconds_sum_with_overlap"] for row in aggregate.values())
    if total_complete <= 0 or total_duration <= 0:
        raise AssertionError("R18 H5 collective events contain no positive-duration complete event")
    rows = []
    for name in exact_names:
        row = aggregate[name]
        rows.append(
            {
                "categories": sorted(row["categories"]),
                "complete_event_count": row["complete_event_count"],
                "complete_event_duration_microseconds_sum_with_overlap": row[
                    "complete_event_duration_microseconds_sum_with_overlap"
                ],
                "count": row["count"],
                "exact_name": name,
                "phases": dict(sorted(row["phases"].items())),
            }
        )
    return {
        "all_reduce_event_name_count": sum(bool(ALL_REDUCE_PATTERN.search(name)) for name in exact_names),
        "collective_complete_event_count": total_complete,
        "collective_complete_event_duration_microseconds_sum_with_overlap": total_duration,
        "distinct_collective_event_names": len(rows),
        "duration_caveat": (
            "Chrome events can overlap across CPU, CUDA runtime, and GPU kernels; summed durations prove observed "
            "activity but are not a non-overlapping wall-time or communication-overhead decomposition."
        ),
        "event_catalog": rows,
        "trace_event_count": len(events),
        "trace_sha256": sha256_file(trace_path),
    }


def main() -> None:
    args = parse_args()
    if args.report_output.exists():
        raise FileExistsError(args.report_output)
    contract, contract_sha256 = load_h5_contract(
        args.h5_contract,
        human_protocol_path=args.human_protocol,
        preregistration_closure_path=args.preregistration_closure,
    )
    _, amendment_sha256 = load_h5_harness_amendment(
        args.harness_amendment,
        human_amendment_path=args.harness_human_amendment,
        attempt01_failure_closure_path=args.attempt01_failure_closure,
    )
    _, amendment_r2_sha256 = load_h5_harness_amendment_r2(
        args.harness_amendment_r2,
        human_amendment_path=args.harness_human_amendment_r2,
        attempt02_failure_closure_path=args.attempt02_failure_closure,
        reload_type_diagnostic_path=args.reload_type_diagnostic,
    )
    profile = load_strict_json(args.hardware_profile)
    if profile.get("artifact") != "qwen35_cuda_hardware_profile":
        raise ValueError("R18 H5 raw hardware-profile artifact drift")
    if profile.get("status") != "captured_pending_kernel_audit":
        raise ValueError("R18 H5 raw hardware-profile status drift")
    if profile.get("completed_optimizer_steps") != H5_FINAL_STEP:
        raise ValueError("R18 H5 hardware-profile optimizer-step drift")
    if profile.get("candidate_chunk_size") != H5_SELECTED_CHUNK_SIZE:
        raise ValueError("R18 H5 hardware-profile selected-loss chunk drift")
    if profile.get("qualification_manifest_sha256") != contract["parent"]["r18_machine_manifest_sha256"]:
        raise ValueError("R18 H5 hardware-profile qualification-manifest drift")
    trace_sha256 = sha256_file(args.trace)
    if profile.get("trace_sha256") != trace_sha256 or profile.get("trace_bytes") != args.trace.stat().st_size:
        raise ValueError("R18 H5 hardware-profile trace identity drift")
    sanitization = sanitize_profiler_trace(args.trace, args.sanitized_trace_output)
    sanitized_trace = load_strict_json(args.sanitized_trace_output)
    inventory = validate_trace(sanitized_trace, trace_path=args.sanitized_trace_output)
    catalog_trace_sha256 = inventory.pop("trace_sha256")
    report = {
        "artifact": "qwen35_r18_h5_nccl_exact_event_catalog",
        "contract_sha256": contract_sha256,
        "catalog_trace_sha256": catalog_trace_sha256,
        "harness_amendment_sha256": amendment_sha256,
        "harness_amendment_r2_sha256": amendment_r2_sha256,
        "hardware_profile_sha256": sha256_file(args.hardware_profile),
        **inventory,
        "nodes": 1,
        "schema_version": 1,
        "status": "passed",
        "trace_sanitization": sanitization,
        "trace_sha256": trace_sha256,
        "world_size": 4,
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
