"""Fail-closed primitives for the preregistered R18 H4 real-32K gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID

H4_ARTIFACT = "qwen35_r18_h4_real_32k_memory_kernel_and_chunk_selection_contract"
H4_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h4-r1"
H4_CONTRACT_SHA256 = "adb6f08012d4893320e9ffc0f2d1753c48ceb744c51e98a3e6d10d7ff328fa33"
H4_PREREGISTRATION_CLOSURE_SHA256 = "fd1db81c568c473a417c2936f3492288a8d0188057fe548696a2211e26bb8080"
R18_MANIFEST_SHA256 = "679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0"
H3_FINAL_CLOSURE_SHA256 = "2e3929e758b947772658422a1aaa46c61f3be7ea19bcbe8564619e414fc08c9d"
H4_ALLOCATOR_HISTORY_ENTRY_CAP = 2_000_000
LEONARDO_A100_NAME = "NVIDIA A100-SXM-64GB"
LEONARDO_A100_MEMORY_MIB = "65536"
LEONARDO_A100_COMPUTE_CAPABILITY = "8.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_json(value: Any, *, context: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {context}")
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_json(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_json(child, context=f"{context}[{index}]")


def load_strict_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    require_finite_json(value, context=str(path))
    return value


def load_strict_jsonl(path: Path) -> list[Any]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        temporary = path.with_name(f"{path.name}:line:{line_number}")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r} in {temporary}")
                result[key] = value
            return result

        value = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        require_finite_json(value, context=str(temporary))
        records.append(value)
    if not records:
        raise ValueError(f"strict JSONL artifact is empty: {path}")
    return records


def load_h4_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != H4_CONTRACT_SHA256:
        raise ValueError(f"R18 H4 contract digest drift: {digest} != {H4_CONTRACT_SHA256}")
    value = load_strict_json(path)
    if value.get("schema_version") != 1 or value.get("artifact") != H4_ARTIFACT:
        raise ValueError("unsupported R18 H4 contract schema or artifact")
    if value.get("protocol_id") != H4_PROTOCOL_ID:
        raise ValueError("R18 H4 protocol identity drift")
    if value.get("status") != "preregistered_after_H3_before_H4_implementation_or_CUDA_execution":
        raise ValueError("R18 H4 contract has an unexpected preregistration status")
    if value.get("scientific_training_authorized") is not False:
        raise ValueError("R18 H4 contract may not authorize scientific training")
    if value.get("automatic_successor") is not False or value.get("allowed_successor_on_complete_pass") != "H5_only":
        raise ValueError("R18 H4 successor contract drift")
    if value.get("candidate_chunk_sizes_in_execution_order") != [128, 256, 512, 1024]:
        raise ValueError("R18 H4 candidate set or order drift")
    if value.get("parent", {}).get("r18_machine_manifest_sha256") != R18_MANIFEST_SHA256:
        raise ValueError("R18 H4 parent manifest drift")
    if value.get("parent", {}).get("h3_independent_validation_final_closure_sha256") != H3_FINAL_CLOSURE_SHA256:
        raise ValueError("R18 H4 H3 predecessor drift")
    if value.get("execution", {}).get("slurm_account") != "aifac_f02_434":
        raise ValueError("R18 H4 personal Slurm account drift")
    if value.get("execution", {}).get("liger_execution_allowed") is not False:
        raise ValueError("R18 H4 must forbid Liger execution")
    return value, digest


def _expected_boundaries(selected_rows: int, chunk_size: int) -> list[list[int]]:
    return [[start, min(start + chunk_size, selected_rows)] for start in range(0, selected_rows, chunk_size)]


def validate_forward_loss_audit(
    audit: dict[str, Any],
    *,
    expected_selected_rows: int,
    expected_global_target_count: int,
    expected_chunk_size: int,
    expected_vocabulary_size: int = 248_320,
    expected_hidden_size: int = 1_024,
) -> dict[str, Any]:
    """Recompute every selected-output audit field from independent inputs."""

    if not isinstance(audit, dict):
        raise TypeError("selected-output loss audit must be an object")
    require_finite_json(audit, context="forward_loss_audit")
    required = {
        "implementation_id",
        "checkpointed",
        "selected_rows",
        "chunk_size",
        "chunk_count",
        "chunk_boundaries",
        "maximum_chunk_rows",
        "vocabulary_size",
        "hidden_size",
        "maximum_logit_elements",
        "full_selected_logit_elements",
        "global_target_count",
        "zero_target",
        "returned_dense_logits",
    }
    if set(audit) != required:
        raise ValueError(
            f"selected-output audit field drift: missing={sorted(required - set(audit))}, "
            f"extra={sorted(set(audit) - required)}"
        )
    if expected_selected_rows < 0 or expected_global_target_count <= 0:
        raise ValueError("H4 real-pack selected rows must be nonnegative and the group divisor positive")
    if audit["implementation_id"] != IMPLEMENTATION_ID or audit["checkpointed"] is not True:
        raise ValueError("selected-output audit implementation or checkpoint status drift")
    if int(audit["selected_rows"]) != expected_selected_rows:
        raise ValueError("selected-output audit row count drift")
    if int(audit["chunk_size"]) != expected_chunk_size:
        raise ValueError("selected-output audit chunk-size drift")
    if int(audit["vocabulary_size"]) != expected_vocabulary_size:
        raise ValueError("selected-output audit vocabulary-size drift")
    if int(audit["hidden_size"]) != expected_hidden_size:
        raise ValueError("selected-output audit hidden-size drift")
    if int(audit["global_target_count"]) != expected_global_target_count:
        raise ValueError("selected-output audit global target divisor drift")
    expected_boundaries = _expected_boundaries(expected_selected_rows, expected_chunk_size)
    if audit["chunk_boundaries"] != expected_boundaries:
        raise ValueError("selected-output audit chunk boundaries drift")
    if int(audit["chunk_count"]) != len(expected_boundaries):
        raise ValueError("selected-output audit chunk count drift")
    maximum_rows = max((end - start for start, end in expected_boundaries), default=0)
    if int(audit["maximum_chunk_rows"]) != maximum_rows or maximum_rows > expected_chunk_size:
        raise ValueError("selected-output audit maximum chunk rows drift")
    if int(audit["maximum_logit_elements"]) != maximum_rows * expected_vocabulary_size:
        raise ValueError("selected-output audit transient-logit element count drift")
    if int(audit["full_selected_logit_elements"]) != expected_selected_rows * expected_vocabulary_size:
        raise ValueError("selected-output audit full-selected element count drift")
    if audit["zero_target"] is not (expected_selected_rows == 0):
        raise ValueError("selected-output audit zero-target flag drift")
    if audit["returned_dense_logits"] is not False:
        raise ValueError("selected-output audit reports returned dense logits")
    return {
        "chunk_count": len(expected_boundaries),
        "full_selected_logit_elements": expected_selected_rows * expected_vocabulary_size,
        "global_target_count": expected_global_target_count,
        "maximum_chunk_rows": maximum_rows,
        "maximum_logit_elements": maximum_rows * expected_vocabulary_size,
        "selected_rows": expected_selected_rows,
        "status": "passed",
    }


def _walk_json(value: Any, path: str = "root"):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]")


def inventory_chrome_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Inventory exact accelerator event names and reject dense 32K logits."""

    if not isinstance(trace, dict):
        raise TypeError("Chrome trace must be an object")
    events = trace.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise ValueError("Chrome trace has no traceEvents")
    forbidden_paths: list[str] = []
    string_pattern = re.compile(r"\[\s*(?:1\s*,\s*)?32768\s*,\s*248320\s*\]")
    for path, value in _walk_json(trace):
        if value in ([1, 32768, 248320], [32768, 248320]):
            forbidden_paths.append(path)
        elif isinstance(value, str) and string_pattern.search(value):
            forbidden_paths.append(path)
    if forbidden_paths:
        raise AssertionError(f"trace contains a dense full-sequence vocabulary shape at {forbidden_paths[:10]}")

    aggregate: dict[str, dict[str, Any]] = {}
    all_names: dict[str, dict[str, Any]] = {}
    category_counts: dict[str, int] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"trace event {index} is not an object")
        category = str(event.get("cat", ""))
        lowered = category.lower()
        category_counts[category] = category_counts.get(category, 0) + 1
        event_name = event.get("name")
        if isinstance(event_name, str) and event_name:
            all_row = all_names.setdefault(event_name, {"categories": set(), "count": 0})
            all_row["categories"].add(category)
            all_row["count"] += 1
        is_accelerator = "kernel" in lowered or lowered in {"gpu_memcpy", "gpu_memset"}
        if not is_accelerator:
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"accelerator trace event {index} has no exact name")
        duration = event.get("dur", 0)
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"accelerator trace event {index} has invalid duration")
        row = aggregate.setdefault(name, {"categories": set(), "count": 0, "duration_microseconds": 0.0})
        row["categories"].add(category)
        row["count"] += 1
        row["duration_microseconds"] += float(duration)
    if not aggregate:
        raise ValueError("Chrome trace contains no accelerator kernel/memcpy/memset events")
    rows = []
    for name in sorted(aggregate):
        row = aggregate[name]
        rows.append(
            {
                "categories": sorted(row["categories"]),
                "count": row["count"],
                "duration_microseconds": row["duration_microseconds"],
                "exact_name": name,
            }
        )
    return {
        "accelerator_event_count": sum(row["count"] for row in rows),
        "category_counts": dict(sorted(category_counts.items())),
        "distinct_accelerator_event_names": len(rows),
        "distinct_all_event_names": len(all_names),
        "forbidden_dense_shape_paths": [],
        "observed_accelerator_events": rows,
        "observed_all_event_names": [
            {"categories": sorted(all_names[name]["categories"]), "count": all_names[name]["count"], "exact_name": name}
            for name in sorted(all_names)
        ],
        "trace_event_count": len(events),
    }


def validate_memory_snapshot(snapshot: Any, *, history_entry_cap: int) -> dict[str, Any]:
    """Validate the pinned PyTorch allocator snapshot schema and history completeness."""

    if not isinstance(snapshot, dict):
        raise TypeError("CUDA allocator snapshot must deserialize to an object")
    if history_entry_cap <= 0:
        raise ValueError("allocator history-entry cap must be positive")
    segments = snapshot.get("segments")
    device_traces = snapshot.get("device_traces")
    if not isinstance(segments, list) or not isinstance(device_traces, list) or not device_traces:
        raise ValueError("allocator snapshot lacks segments or device_traces")
    action_counts: dict[str, int] = {}
    trace_lengths = []
    oom_events = []
    for device, trace in enumerate(device_traces):
        if not isinstance(trace, list):
            raise ValueError(f"allocator device trace {device} is not a list")
        trace_lengths.append(len(trace))
        if len(trace) >= history_entry_cap:
            raise AssertionError(
                f"allocator device trace {device} reached the {history_entry_cap}-entry cap; history may be truncated"
            )
        for index, event in enumerate(trace):
            if not isinstance(event, dict):
                raise ValueError(f"allocator trace event {device}:{index} is not an object")
            action = event.get("action")
            if not isinstance(action, str) or not action:
                raise ValueError(f"allocator trace event {device}:{index} lacks an action")
            action_counts[action] = action_counts.get(action, 0) + 1
            if action == "oom":
                oom_events.append({"device": device, "index": index, "event": event})
    if oom_events:
        raise AssertionError(f"allocator snapshot records OOM actions: {oom_events[:3]}")
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "device_trace_lengths": trace_lengths,
        "history_entry_cap": history_entry_cap,
        "history_entry_cap_reached": False,
        "oom_action_count": 0,
        "segment_count": len(segments),
        "top_level_keys": sorted(snapshot),
    }


def timing_statistics(values: list[float]) -> dict[str, float | int]:
    if len(values) != 10:
        raise ValueError(f"H4 timing selection requires exactly ten measured updates, found {len(values)}")
    if any(not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("H4 timing values must all be finite and positive")
    numeric = [float(value) for value in values]
    mean = statistics.fmean(numeric)
    sample_standard_deviation = statistics.stdev(numeric)
    return {
        "coefficient_of_variation": sample_standard_deviation / mean,
        "count": len(numeric),
        "maximum_seconds": max(numeric),
        "mean_seconds": mean,
        "median_seconds": statistics.median(numeric),
        "minimum_seconds": min(numeric),
        "sample_standard_deviation_seconds": sample_standard_deviation,
    }


def select_chunk_size(candidate_rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["candidate_chunk_sizes_in_execution_order"]
    observed = [int(row["chunk_size"]) for row in candidate_rows]
    if observed != expected or len(set(observed)) != len(expected):
        raise ValueError(f"H4 candidate order/set drift: {observed} != {expected}")
    threshold = float(contract["timing_selection"]["maximum_coefficient_of_variation"])
    for row in candidate_rows:
        if row.get("eligible") is not True:
            raise AssertionError(f"H4 candidate {row['chunk_size']} is not memory/kernel/update eligible")
        statistics_row = row.get("timing_statistics")
        recomputed = timing_statistics(row.get("measured_update_seconds", []))
        for key, value in recomputed.items():
            if isinstance(value, float):
                if not math.isclose(float(statistics_row.get(key, math.nan)), value, rel_tol=1e-15, abs_tol=0.0):
                    raise ValueError(f"H4 candidate {row['chunk_size']} timing statistic drift for {key}")
            elif statistics_row.get(key) != value:
                raise ValueError(f"H4 candidate {row['chunk_size']} timing statistic drift for {key}")
        if recomputed["coefficient_of_variation"] > threshold:
            raise AssertionError(f"H4 candidate {row['chunk_size']} timing CV exceeds the frozen threshold")
    fastest = min(float(row["timing_statistics"]["median_seconds"]) for row in candidate_rows)
    tie_fraction = float(contract["timing_selection"]["tie_fraction_inclusive"])
    tied = [
        int(row["chunk_size"])
        for row in candidate_rows
        if (float(row["timing_statistics"]["median_seconds"]) - fastest) / fastest <= tie_fraction
    ]
    if not tied:
        raise RuntimeError("H4 frozen timing rule produced an empty tied set")
    return {
        "fastest_median_seconds": fastest,
        "selected_chunk_size": min(tied),
        "tie_fraction_inclusive": tie_fraction,
        "tied_chunk_sizes": tied,
    }
