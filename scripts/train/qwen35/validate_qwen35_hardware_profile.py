#!/usr/bin/env python3
"""Validate H4 memory evidence and inventory profiler operators/kernels for review."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from open_instruct.qwen35_qualification import load_qualification_manifest, sha256_file, validate_memory_headroom
from open_instruct.qwen35_training import write_json_atomic

CATEGORY_PATTERNS = {
    "matrix_multiplication": re.compile(r"gemm|cublas|cutlass|matmul|mm", re.IGNORECASE),
    "attention": re.compile(r"flash|fmha|scaled.?dot.?product|attention", re.IGNORECASE),
    "gated_delta_recurrence": re.compile(r"gated.?delta|delta.?rule|fla", re.IGNORECASE),
    "causal_convolution": re.compile(r"causal.?conv|conv1d", re.IGNORECASE),
    "fused_cross_entropy": re.compile(r"liger|cross.?entropy|xentropy|softmax", re.IGNORECASE),
    "adamw": re.compile(r"adam", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    profile = json.loads(args.hardware_profile.read_text())
    if profile.get("qualification_manifest_sha256") != qualification_sha256:
        raise ValueError("hardware profile uses a different qualification manifest")
    if profile.get("status") != "captured_pending_kernel_audit":
        raise ValueError("hardware profile has unexpected status")
    memory = profile.get("memory", {})
    recomputed_memory = validate_memory_headroom(
        peak_allocated_bytes=int(memory["peak_allocated_bytes"]),
        peak_reserved_bytes=int(memory["peak_reserved_bytes"]),
        total_device_bytes=int(memory["total_device_bytes"]),
        acceptance=qualification["memory_acceptance"],
    )
    if memory != recomputed_memory:
        raise ValueError("hardware profile memory fields do not independently reproduce")
    allocator = profile.get("allocator", {})
    if int(allocator.get("num_ooms", -1)) != 0:
        raise AssertionError("hardware profile recorded an OOM")
    if int(allocator.get("num_alloc_retries", -1)) > int(
        qualification["memory_acceptance"]["maximum_allocator_retries"]
    ):
        raise AssertionError("hardware profile recorded allocator retries above the frozen threshold")

    trace = json.loads(args.trace.read_text())
    events = trace.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise ValueError("Chrome trace has no traceEvents")
    names = [str(event.get("name", "")) for event in events if event.get("name")]
    counts = collections.Counter(names)
    category_matches = {
        category: sorted(name for name in counts if pattern.search(name))
        for category, pattern in CATEGORY_PATTERNS.items()
    }
    missing_categories = [category for category, matches in category_matches.items() if not matches]
    if missing_categories:
        raise AssertionError(f"profiler trace lacks required operator/kernel categories: {missing_categories}")

    compact_trace = json.dumps(trace, separators=(",", ":"))
    forbidden_shape_patterns = (r"\[1,32768,248320\]", r"\[32768,248320\]")
    forbidden_matches = [pattern for pattern in forbidden_shape_patterns if re.search(pattern, compact_trace)]
    if forbidden_matches:
        raise AssertionError("profiler trace contains a possible dense full-sequence vocabulary-logit shape")

    kernel_events = [
        event
        for event in events
        if str(event.get("cat", "")).lower() in {"kernel", "cuda_runtime", "gpu_memcpy", "gpu_memset"}
        or "kernel" in str(event.get("cat", "")).lower()
    ]
    kernel_counts = collections.Counter(str(event.get("name", "")) for event in kernel_events if event.get("name"))
    report = {
        "artifact": "qwen35_h4_hardware_profile_validation",
        "schema_version": 1,
        "status": "required_categories_passed_pending_manual_kernel_source_review",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "hardware_profile_sha256": sha256_file(args.hardware_profile),
        "trace_sha256": sha256_file(args.trace),
        "trace_event_count": len(events),
        "distinct_event_names": len(counts),
        "memory": recomputed_memory,
        "allocator": allocator,
        "required_category_matches": category_matches,
        "forbidden_dense_logit_shape_matches": forbidden_matches,
        "kernel_event_count": len(kernel_events),
        "distinct_kernel_names": len(kernel_counts),
        "kernel_name_counts": dict(sorted(kernel_counts.items())),
        "manual_review_required": (
            "Map every distinct GPU kernel name to pinned attention, FLA/GDN, causal-conv1d, Liger, torch, "
            "CUDA, NCCL, or optimizer source; reject an unknown or forbidden fallback before H4 passes."
        ),
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
