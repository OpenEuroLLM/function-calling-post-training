#!/usr/bin/env python3
"""Independently validate one R18 H4 profiler assay and inventory its exact accelerator events."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification import validate_memory_headroom
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h4 import (
    H4_ALLOCATOR_HISTORY_ENTRY_CAP,
    LEONARDO_A100_NAME,
    inventory_chrome_trace,
    load_h4_contract,
    load_strict_json,
    sha256_file,
    validate_memory_snapshot,
)
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--h4-contract", type=Path, required=True)
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--memory-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-chunk-size", type=int, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required nonempty H4 artifact is absent: {path}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h4, h4_sha256 = load_h4_contract(args.h4_contract)
    if qualification_sha256 != h4["parent"]["r18_machine_manifest_sha256"]:
        raise ValueError("H4 profile validator parent qualification drift")
    if args.candidate_chunk_size not in h4["candidate_chunk_sizes_in_execution_order"]:
        raise ValueError("H4 profile validator received an unknown candidate")
    for path in (args.hardware_profile, args.trace, args.memory_snapshot):
        _require_file(path)

    profile = load_strict_json(args.hardware_profile)
    expected_scalars = {
        "artifact": "qwen35_cuda_hardware_profile",
        "schema_version": 1,
        "status": "captured_pending_kernel_audit",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "h4_protocol_id": h4["protocol_id"],
        "h4_contract_sha256": h4_sha256,
        "assay": "profiler",
        "candidate_chunk_size": args.candidate_chunk_size,
        "completed_optimizer_steps": 4,
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": 3,
    }
    for key, expected in expected_scalars.items():
        if profile.get(key) != expected:
            raise ValueError(f"H4 hardware profile drift for {key}: {profile.get(key)!r} != {expected!r}")
    if profile.get("profiler_schedule") != h4["profiler_assay"]["profiler_schedule"]:
        raise ValueError("H4 profiler schedule drift")
    if profile.get("cuda_device_name") != LEONARDO_A100_NAME:
        raise ValueError(f"H4 unexpected CUDA device: {profile.get('cuda_device_name')!r}")
    if profile.get("cuda_device_capability") != [8, 0]:
        raise ValueError("H4 profiler did not execute on SM80")

    expected_event_keys = [str(step) for step in range(1, 5)]
    event_times = profile.get("cuda_event_step_milliseconds")
    if not isinstance(event_times, dict) or list(event_times) != expected_event_keys:
        raise ValueError("H4 profiler CUDA-event step set/order drift")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        for value in event_times.values()
    ):
        raise ValueError("H4 profiler contains a nonpositive or nonfinite CUDA-event duration")

    file_contract = (
        ("trace", args.trace, "trace_path", "trace_bytes", "trace_sha256"),
        (
            "memory snapshot",
            args.memory_snapshot,
            "memory_snapshot_path",
            "memory_snapshot_bytes",
            "memory_snapshot_sha256",
        ),
    )
    for label, path, path_key, bytes_key, sha_key in file_contract:
        if Path(profile.get(path_key, "")).resolve() != path.resolve():
            raise ValueError(f"H4 {label} path binding drift")
        if profile.get(bytes_key) != path.stat().st_size:
            raise ValueError(f"H4 {label} byte-size binding drift")
        if profile.get(sha_key) != sha256_file(path):
            raise ValueError(f"H4 {label} hash binding drift")

    memory = profile.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("H4 profile lacks memory accounting")
    recomputed_memory = validate_memory_headroom(
        peak_allocated_bytes=int(memory.get("peak_allocated_bytes", -1)),
        peak_reserved_bytes=int(memory.get("peak_reserved_bytes", -1)),
        total_device_bytes=int(memory.get("total_device_bytes", -1)),
        acceptance=h4["memory_acceptance"],
    )
    if memory != recomputed_memory:
        raise ValueError("H4 profile memory fields do not independently reproduce")
    per_step_memory = profile.get("per_step_memory", {})
    expected_memory_step_keys = [str(step) for step in range(1, 5)]
    allocated_by_step = per_step_memory.get("peak_allocated_bytes")
    reserved_by_step = per_step_memory.get("peak_reserved_bytes")
    if (
        not isinstance(allocated_by_step, dict)
        or not isinstance(reserved_by_step, dict)
        or list(allocated_by_step) != expected_memory_step_keys
        or list(reserved_by_step) != expected_memory_step_keys
        or per_step_memory.get("aggregation")
        != "maximum_across_all_four_steps_after_exact_metrics_window_resets"
    ):
        raise ValueError("H4 per-step memory evidence set/order/aggregation drift")
    for step in expected_memory_step_keys:
        allocated = allocated_by_step[step]
        reserved = reserved_by_step[step]
        if not isinstance(allocated, int) or not isinstance(reserved, int) or not 0 <= allocated <= reserved <= memory[
            "total_device_bytes"
        ]:
            raise ValueError(f"H4 invalid per-step CUDA memory accounting at step {step}")
    if max(allocated_by_step.values()) != memory["peak_allocated_bytes"]:
        raise ValueError("H4 aggregate peak-allocated bytes do not equal the per-step maximum")
    if max(reserved_by_step.values()) != memory["peak_reserved_bytes"]:
        raise ValueError("H4 aggregate peak-reserved bytes do not equal the per-step maximum")
    allocator = profile.get("allocator")
    if allocator != {"num_alloc_retries": 0, "num_ooms": 0}:
        raise AssertionError(f"H4 allocator failure evidence: {allocator!r}")
    device_memory = profile.get("device_memory_observations", {})
    if device_memory.get("total_bytes") != memory["total_device_bytes"]:
        raise ValueError("H4 device-memory total drift")
    for key in ("initial_free_bytes_after_model_load", "final_free_bytes"):
        value = device_memory.get(key)
        if not isinstance(value, int) or not 0 <= value <= memory["total_device_bytes"]:
            raise ValueError(f"H4 invalid device-memory observation {key}")
    history = profile.get("allocator_history", {})
    if history != {
        "clear_history": True,
        "context": "all",
        "enabled": "all",
        "maximum_entries": H4_ALLOCATOR_HISTORY_ENTRY_CAP,
        "stacks": "python",
    }:
        raise ValueError("H4 allocator-history capture contract drift")

    trace = load_strict_json(args.trace)
    trace_inventory = inventory_chrome_trace(trace)
    with args.memory_snapshot.open("rb") as handle:
        snapshot = pickle.load(handle)  # noqa: S301 - trusted, hash-bound qualification artifact
        if handle.read(1):
            raise ValueError("allocator snapshot pickle contains trailing bytes")
    snapshot_validation = validate_memory_snapshot(snapshot, history_entry_cap=history["maximum_entries"])

    return {
        "allocator": allocator,
        "artifact": "qwen35_r18_h4_profiler_assay_validation",
        "candidate_chunk_size": args.candidate_chunk_size,
        "cuda_event_step_milliseconds": event_times,
        "h4_contract_sha256": h4_sha256,
        "hardware_profile_sha256": sha256_file(args.hardware_profile),
        "memory": recomputed_memory,
        "memory_snapshot_bytes": args.memory_snapshot.stat().st_size,
        "memory_snapshot_sha256": sha256_file(args.memory_snapshot),
        "qualification_manifest_sha256": qualification_sha256,
        "schema_version": 1,
        "snapshot_validation": snapshot_validation,
        "status": "automated_profile_passed_pending_manual_kernel_mapping",
        "trace_bytes": args.trace.stat().st_size,
        "trace_inventory": trace_inventory,
        "trace_sha256": sha256_file(args.trace),
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
