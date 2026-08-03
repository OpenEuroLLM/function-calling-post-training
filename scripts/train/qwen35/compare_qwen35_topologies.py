#!/usr/bin/env python3
"""Compare controlled T4 (4-GPU/GA2) and T8 (8-GPU/GA1) Qwen runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from open_instruct.qwen35_qualification import (
    load_qualification_manifest,
    scalar_comparison_metrics,
    select_topology,
    sha256_file,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--four-gpu-output", type=Path, required=True)
    parser.add_argument("--eight-gpu-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def probe_update_vector(report: dict) -> torch.Tensor:
    initial = report.get("initial_samples", {})
    final = report.get("final_samples", {})
    if initial.keys() != final.keys() or not initial:
        raise ValueError("parameter-probe samples are missing or structurally inconsistent")
    deltas = []
    for name in sorted(initial):
        if initial[name]["indices"] != final[name]["indices"]:
            raise ValueError(f"parameter-probe index drift in {name}")
        left = torch.tensor(initial[name]["values"], dtype=torch.float64)
        right = torch.tensor(final[name]["values"], dtype=torch.float64)
        deltas.append(right - left)
    return torch.cat(deltas)


def read_rank_cuda_times(root: Path, world_size: int, expected_steps: int) -> dict[int, list[float]]:
    values = {}
    for rank in range(world_size):
        path = root / f"qwen35_cuda_step_times_rank{rank:02d}.json"
        report = json.loads(path.read_text())
        if report.get("status") != "passed" or report.get("rank") != rank or report.get("world_size") != world_size:
            raise ValueError(f"invalid rank CUDA-event timing report {path}")
        timings = report.get("cuda_event_step_milliseconds", {})
        if sorted(int(step) for step in timings) != list(range(1, expected_steps + 1)):
            raise ValueError(f"rank CUDA-event timing step drift in {path}")
        values[rank] = [float(timings[str(step)]) for step in range(1, expected_steps + 1)]
    return values


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    topology = qualification["topology_acceptance"]
    numerical = qualification["numerical_acceptance"]
    expected_steps = int(topology["warmup_optimizer_updates"]) + int(topology["measured_optimizer_updates"])

    four_root = args.four_gpu_output.resolve()
    eight_root = args.eight_gpu_output.resolve()
    four_records = read_jsonl(four_root / "qwen35_exact_metrics.jsonl")
    eight_records = read_jsonl(eight_root / "qwen35_exact_metrics.jsonl")
    if len(four_records) != expected_steps or len(eight_records) != expected_steps:
        raise ValueError(f"topology runs must each contain {expected_steps} one-step reporting windows")

    step_comparisons = []
    for four, eight in zip(four_records, eight_records, strict=True):
        if four["step"] != eight["step"]:
            raise ValueError("topology step identity drift")
        if sorted(four["schedule_indices"]) != sorted(eight["schedule_indices"]):
            raise AssertionError(f"topology schedule-index drift at step {four['step']}")
        if sorted(four["pack_uids"]) != sorted(eight["pack_uids"]):
            raise AssertionError(f"topology pack-UID drift at step {four['step']}")
        if four["counts"] != eight["counts"]:
            raise AssertionError(f"topology accounting drift at step {four['step']}")
        if four["loss"]["global_assistant_target_divisor"] != eight["loss"]["global_assistant_target_divisor"]:
            raise AssertionError(f"topology target-divisor drift at step {four['step']}")
        if four["optimizer"]["applied_learning_rates"] != eight["optimizer"]["applied_learning_rates"]:
            raise AssertionError(f"topology applied-LR drift at step {four['step']}")
        loss_metrics = scalar_comparison_metrics(
            float(eight["loss"]["normalized_loss"]), float(four["loss"]["normalized_loss"])
        )
        validate_comparison_metrics(loss_metrics, numerical, kind="loss", context=f"topology step {four['step']}")
        step_comparisons.append(
            {
                "step": four["step"],
                "schedule_indices": sorted(four["schedule_indices"]),
                "loss_comparison": loss_metrics,
            }
        )

    four_probe_path = four_root / "qwen35_parameter_update_probe.json"
    eight_probe_path = eight_root / "qwen35_parameter_update_probe.json"
    four_probe = json.loads(four_probe_path.read_text())
    eight_probe = json.loads(eight_probe_path.read_text())
    if four_probe.get("initial_samples") != eight_probe.get("initial_samples"):
        raise AssertionError("topology runs did not start from identical sampled parameters")
    update_metrics = tensor_comparison_metrics(probe_update_vector(eight_probe), probe_update_vector(four_probe))
    validate_comparison_metrics(update_metrics, numerical, kind="update", context="topology sampled update")

    warmup = int(topology["warmup_optimizer_updates"])
    four_times = [float(record["elapsed_seconds"]) for record in four_records[warmup:]]
    eight_times = [float(record["elapsed_seconds"]) for record in eight_records[warmup:]]
    decision = select_topology(four_times, eight_times, topology)
    four_rank_times = read_rank_cuda_times(four_root, 4, expected_steps)
    eight_rank_times = read_rank_cuda_times(eight_root, 8, expected_steps)
    four_device_max_milliseconds = [
        max(four_rank_times[rank][step] for rank in four_rank_times) for step in range(expected_steps)
    ]
    eight_device_max_milliseconds = [
        max(eight_rank_times[rank][step] for rank in eight_rank_times) for step in range(expected_steps)
    ]

    report = {
        "artifact": "qwen35_four_vs_eight_gpu_topology_comparison",
        "schema_version": 1,
        "status": "repeat_required" if decision["repeat_required"] else "passed",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "four_gpu_output": str(four_root),
        "eight_gpu_output": str(eight_root),
        "four_gpu_metrics_sha256": sha256_file(four_root / "qwen35_exact_metrics.jsonl"),
        "eight_gpu_metrics_sha256": sha256_file(eight_root / "qwen35_exact_metrics.jsonl"),
        "four_gpu_probe_sha256": sha256_file(four_probe_path),
        "eight_gpu_probe_sha256": sha256_file(eight_probe_path),
        "warmup_optimizer_updates": warmup,
        "measured_optimizer_updates": len(four_times),
        "per_step_correctness": step_comparisons,
        "sampled_update_comparison": update_metrics,
        "timing_decision": decision,
        "four_gpu_seconds_per_update": four_times,
        "eight_gpu_seconds_per_update": eight_times,
        "four_gpu_seconds_per_measured_window": sum(four_times) * 4,
        "eight_gpu_seconds_per_measured_window": sum(eight_times) * 8,
        "four_gpu_per_rank_cuda_event_milliseconds": four_rank_times,
        "eight_gpu_per_rank_cuda_event_milliseconds": eight_rank_times,
        "four_gpu_max_rank_cuda_event_milliseconds": four_device_max_milliseconds,
        "eight_gpu_max_rank_cuda_event_milliseconds": eight_device_max_milliseconds,
        "decision_rule": "select T8 iff speedup >= 1.20 and both layouts pass correctness; otherwise T4",
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
