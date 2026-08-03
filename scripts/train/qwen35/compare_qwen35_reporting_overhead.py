#!/usr/bin/env python3
"""Compare high-frequency versus coarse exact-reporting synchronization overhead."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from open_instruct.qwen35_qualification import (
    coefficient_of_variation,
    load_qualification_manifest,
    scalar_comparison_metrics,
    sha256_file,
    validate_comparison_metrics,
)
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--fine-output", type=Path, required=True)
    parser.add_argument("--coarse-output", type=Path, required=True)
    parser.add_argument("--checkpoint-comparison", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def max_rank_step_times(root: Path, world_size: int, expected_steps: int) -> list[float]:
    rank_values = []
    for rank in range(world_size):
        path = root / f"qwen35_cuda_step_times_rank{rank:02d}.json"
        report = json.loads(path.read_text())
        if report.get("status") != "passed" or report.get("rank") != rank:
            raise ValueError(f"invalid CUDA-event timing report {path}")
        values = report.get("cuda_event_step_milliseconds", {})
        rank_values.append([float(values[str(step)]) for step in range(1, expected_steps + 1)])
    return [max(values[step] for values in rank_values) for step in range(expected_steps)]


def read_metrics(root: Path) -> list[dict]:
    records = [
        json.loads(line) for line in (root / "qwen35_exact_metrics.jsonl").read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"no exact metrics in {root}")
    return records


def compare_scientific_exposure(fine: list[dict], coarse: list[dict], numerical: dict) -> dict:
    fine_indices = [index for row in fine for index in row["schedule_indices"]]
    coarse_indices = [index for row in coarse for index in row["schedule_indices"]]
    fine_uids = [uid for row in fine for uid in row["pack_uids"]]
    coarse_uids = [uid for row in coarse for uid in row["pack_uids"]]
    fine_lrs = [rate for row in fine for rate in row["optimizer"]["applied_learning_rates"]]
    coarse_lrs = [rate for row in coarse for rate in row["optimizer"]["applied_learning_rates"]]
    if fine_indices != coarse_indices or fine_uids != coarse_uids or fine_lrs != coarse_lrs:
        raise AssertionError("fine/coarse reporting changed schedule, pack identity, or applied learning rates")
    fine_counts = {key: sum(int(row["counts"][key]) for row in fine) for key in fine[0]["counts"]}
    coarse_counts = {key: sum(int(row["counts"][key]) for row in coarse) for key in coarse[0]["counts"]}
    if fine_counts != coarse_counts:
        raise AssertionError("fine/coarse reporting changed exact aggregate accounting")
    fine_weighted_loss = (
        sum(float(row["loss"]["normalized_loss"]) * int(row["counts"]["assistant_targets"]) for row in fine)
        / fine_counts["assistant_targets"]
    )
    coarse_weighted_loss = (
        sum(float(row["loss"]["normalized_loss"]) * int(row["counts"]["assistant_targets"]) for row in coarse)
        / coarse_counts["assistant_targets"]
    )
    loss_metrics = scalar_comparison_metrics(coarse_weighted_loss, fine_weighted_loss)
    validate_comparison_metrics(loss_metrics, numerical, kind="loss", context="fine/coarse aggregate loss")
    return {
        "schedule_indices": fine_indices,
        "pack_uids": fine_uids,
        "aggregate_counts": fine_counts,
        "applied_learning_rates": fine_lrs,
        "aggregate_loss_comparison": loss_metrics,
    }


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    topology = qualification["topology_acceptance"]
    reporting = qualification["reporting_acceptance"]
    checkpoint_comparison = json.loads(args.checkpoint_comparison.read_text())
    if (
        checkpoint_comparison.get("status") != "passed"
        or checkpoint_comparison.get("atol") != 0
        or checkpoint_comparison.get("rtol") != 0
        or checkpoint_comparison.get("model", {}).get("bit_exact") is not True
        or checkpoint_comparison.get("optimizer", {}).get("bit_exact_tensors") is not True
    ):
        raise AssertionError("fine/coarse reporting checkpoint comparison is not bit exact")
    warmup = int(topology["warmup_optimizer_updates"])
    measured = int(topology["measured_optimizer_updates"])
    total = warmup + measured
    fine = max_rank_step_times(args.fine_output, 4, total)[warmup:]
    coarse = max_rank_step_times(args.coarse_output, 4, total)[warmup:]
    exposure = compare_scientific_exposure(
        read_metrics(args.fine_output), read_metrics(args.coarse_output), qualification["numerical_acceptance"]
    )
    fine_median = statistics.median(fine)
    coarse_median = statistics.median(coarse)
    overhead_fraction = fine_median / coarse_median - 1
    maximum = float(reporting["maximum_ordinary_instrumentation_overhead_fraction"])
    status = "passed" if overhead_fraction <= maximum else "failed_overhead"
    if coefficient_of_variation(fine) > float(topology["maximum_timing_coefficient_of_variation"]):
        status = "repeat_required"
    if coefficient_of_variation(coarse) > float(topology["maximum_timing_coefficient_of_variation"]):
        status = "repeat_required"
    report = {
        "artifact": "qwen35_exact_reporting_synchronization_overhead",
        "schema_version": 1,
        "status": status,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "fine_output": str(args.fine_output.resolve()),
        "coarse_output": str(args.coarse_output.resolve()),
        "checkpoint_comparison_sha256": sha256_file(args.checkpoint_comparison),
        "scientific_exposure_comparison": exposure,
        "fine_timing_hashes": {
            str(rank): sha256_file(args.fine_output / f"qwen35_cuda_step_times_rank{rank:02d}.json")
            for rank in range(4)
        },
        "coarse_timing_hashes": {
            str(rank): sha256_file(args.coarse_output / f"qwen35_cuda_step_times_rank{rank:02d}.json")
            for rank in range(4)
        },
        "warmup_steps_excluded": warmup,
        "measured_steps": measured,
        "fine_max_rank_cuda_event_milliseconds": fine,
        "coarse_max_rank_cuda_event_milliseconds": coarse,
        "fine_median_milliseconds": fine_median,
        "coarse_median_milliseconds": coarse_median,
        "fine_coefficient_of_variation": coefficient_of_variation(fine),
        "coarse_coefficient_of_variation": coefficient_of_variation(coarse),
        "additional_fine_sync_overhead_fraction": overhead_fraction,
        "maximum_accepted_overhead_fraction": maximum,
        "interpretation": (
            "This isolates the additional cost of synchronizing/reporting every update relative to the same exact "
            "reporter aggregated over all 13 updates; it is not a no-instrumentation baseline."
        ),
    }
    write_json_atomic(args.report_output, report)
    if status == "failed_overhead":
        raise AssertionError(f"fine-grained exact reporting overhead {overhead_fraction:.6f} exceeds {maximum:.6f}")
    print(json.dumps({"output": str(args.report_output), "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
