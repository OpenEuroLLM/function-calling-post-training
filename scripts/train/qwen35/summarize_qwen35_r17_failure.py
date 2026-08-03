#!/usr/bin/env python3
"""Independently summarize an evidence-valid R17 matched-reference failure."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_r17 import load_qualification_manifest, validate_h2_liger_report
from open_instruct.qwen35_training import write_json_atomic

TRAJECTORY_CONTEXT = re.compile(r"^(R17-T\d+) step (\d+) (?:parameter (\S+) )?(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--independent-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class MetricExtrema:
    """Collect finite extrema while retaining the context of each extremum."""

    def __init__(self) -> None:
        self.count = 0
        self.nonfinite_count = 0
        self.values: dict[str, tuple[float, str]] = {}

    def add(self, metric: dict[str, Any], context: str) -> None:
        self.count += 1
        self.nonfinite_count += int(metric.get("nonfinite_count", 0))
        for key in (
            "maximum_absolute_error",
            "relative_error",
            "relative_l2_error",
            "balanced_relative_l2_error",
        ):
            value = metric.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                current = self.values.get(f"maximum_{key}")
                if current is None or float(value) > current[0]:
                    self.values[f"maximum_{key}"] = (float(value), context)
        cosine = metric.get("cosine_similarity")
        if isinstance(cosine, (int, float)) and math.isfinite(float(cosine)):
            current = self.values.get("minimum_cosine_similarity")
            if current is None or float(cosine) < current[0]:
                self.values["minimum_cosine_similarity"] = (float(cosine), context)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_count": self.count,
            "nonfinite_count": self.nonfinite_count,
            **{key: {"value": value, "context": context} for key, (value, context) in sorted(self.values.items())},
        }


def _add(groups: dict[str, MetricExtrema], family: str, metric: dict[str, Any], context: str) -> None:
    groups.setdefault(family, MetricExtrema()).add(metric, context)


def _direct_extrema(report: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, MetricExtrema] = {}
    for section in ("historical_direct_cases", "confirmatory_direct_cases"):
        for case in report[section]:
            case_id = case["case_contract"]["case_id"]
            for field in (
                "loss_comparison",
                "selected_hidden_gradient_comparison",
                "output_head_gradient_comparison",
                "raw_first_adamw_update_comparison_diagnostic",
                "optimizer_exp_avg_comparison",
                "optimizer_exp_avg_sq_comparison",
                "post_step_parameter_comparison",
            ):
                _add(groups, f"primary.{field}", case[field], case_id)
            _add(groups, "primary.heldout.logit_comparison", case["heldout"]["logit_comparison"], case_id)
            _add(groups, "primary.heldout.loss_comparison", case["heldout"]["loss_comparison"], case_id)
            diagnostic = case["full_dense_diagnostic"]
            for field in (
                "loss_comparison",
                "selected_hidden_gradient_comparison",
                "output_head_gradient_comparison",
                "raw_first_adamw_update_comparison_diagnostic",
                "optimizer_exp_avg_comparison",
                "optimizer_exp_avg_sq_comparison",
                "post_step_parameter_comparison",
            ):
                _add(groups, f"diagnostic.{field}", diagnostic[field], case_id)
            _add(groups, "diagnostic.heldout.logit_comparison", diagnostic["heldout"]["logit_comparison"], case_id)
            _add(groups, "diagnostic.heldout.loss_comparison", diagnostic["heldout"]["loss_comparison"], case_id)
    return {family: value.as_dict() for family, value in sorted(groups.items())}


def _trajectory_extrema(report: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, MetricExtrema] = {}
    aggregate_fields = (
        "aggregate_preclip_gradient_comparison",
        "aggregate_clipped_gradient_comparison",
        "aggregate_raw_adamw_update_comparison_diagnostic",
        "aggregate_optimizer_exp_avg_comparison",
        "aggregate_optimizer_exp_avg_sq_comparison",
        "aggregate_cumulative_parameter_displacement_comparison",
        "aggregate_post_step_parameter_state_comparison",
    )
    named_fields = (
        "preclip_gradient_comparison",
        "clipped_gradient_comparison",
        "raw_adamw_update_comparison_diagnostic",
        "optimizer_exp_avg_comparison",
        "optimizer_exp_avg_sq_comparison",
        "cumulative_parameter_displacement_comparison",
        "post_step_parameter_state_comparison",
    )
    for trajectory in report["confirmatory_trajectories"]:
        trajectory_id = trajectory["trajectory_contract"]["trajectory_id"]
        for step in trajectory["steps"]:
            prefix = f"{trajectory_id} step {step['step']}"
            _add(groups, "primary.training_loss_comparison", step["training_loss_comparison"], prefix)
            for field in aggregate_fields:
                _add(groups, f"primary.{field}", step[field], prefix)
            for name, named in step["per_parameter_comparisons"].items():
                for field in named_fields:
                    _add(groups, f"primary.named.{field}", named[field], f"{prefix} parameter {name}")
            _add(groups, "primary.heldout.logit_comparison", step["heldout"]["logit_comparison"], prefix)
            _add(groups, "primary.heldout.loss_comparison", step["heldout"]["loss_comparison"], prefix)

            diagnostic = step["full_dense_diagnostic"]
            _add(groups, "diagnostic.training_loss_comparison", diagnostic["training_loss_comparison"], prefix)
            for field in aggregate_fields:
                _add(groups, f"diagnostic.{field}", diagnostic[field], prefix)
            for name, named in diagnostic["per_parameter_comparisons"].items():
                for field in named_fields:
                    _add(groups, f"diagnostic.named.{field}", named[field], f"{prefix} parameter {name}")
            _add(groups, "diagnostic.heldout.logit_comparison", diagnostic["heldout"]["logit_comparison"], prefix)
            _add(groups, "diagnostic.heldout.loss_comparison", diagnostic["heldout"]["loss_comparison"], prefix)
    return {family: value.as_dict() for family, value in sorted(groups.items())}


def _decision_anatomy(report: dict[str, Any]) -> dict[str, Any]:
    decision = report["decision"]
    failed = [row for row in decision["checks"] if not row["passed"]]
    failed_gating = [row for row in failed if row["gating"]]
    failed_diagnostic = [row for row in failed if not row["gating"]]
    by_trajectory: collections.Counter[str] = collections.Counter()
    by_family: collections.Counter[str] = collections.Counter()
    by_parameter: collections.Counter[str] = collections.Counter()
    by_kind: collections.Counter[str] = collections.Counter()
    by_reason: collections.Counter[str] = collections.Counter()
    steps_by_trajectory: dict[str, set[int]] = collections.defaultdict(set)
    direct = []
    for row in failed_gating:
        by_kind[row["kind"]] += 1
        message = row.get("message") or ""
        if message.startswith("failed: "):
            normalized_message = message.removeprefix("failed: ")
        elif ": " in message:
            normalized_message = message.split(": ", 1)[1]
        else:
            normalized_message = message
        for reason in normalized_message.split(", "):
            if reason:
                by_reason[reason] += 1
        match = TRAJECTORY_CONTEXT.fullmatch(row["context"])
        if match is None:
            direct.append(row["context"])
            continue
        trajectory_id, step, parameter, family = match.groups()
        by_trajectory[trajectory_id] += 1
        by_family[family] += 1
        steps_by_trajectory[trajectory_id].add(int(step))
        if parameter is not None:
            by_parameter[parameter] += 1
    return {
        "total_checks": decision["total_checks"],
        "gating_checks": decision["gating_checks"],
        "diagnostic_checks": decision["diagnostic_checks"],
        "failed_gating_checks": len(failed_gating),
        "failed_diagnostic_checks": len(failed_diagnostic),
        "failed_direct_gating_contexts": direct,
        "failed_gating_by_trajectory": dict(sorted(by_trajectory.items())),
        "failed_gating_by_family": dict(by_family.most_common()),
        "failed_gating_by_parameter": dict(by_parameter.most_common()),
        "failed_gating_by_decision_kind": dict(by_kind.most_common()),
        "failed_gating_by_reason": dict(by_reason.most_common()),
        "steps_with_at_least_one_gating_failure": {
            trajectory: {
                "count": len(steps),
                "first": min(steps),
                "last": max(steps),
            }
            for trajectory, steps in sorted(steps_by_trajectory.items())
        },
        "first_20_failed_gating_contexts": decision["failed_gating_checks"][:20],
        "last_20_failed_gating_contexts": decision["failed_gating_checks"][-20:],
    }


def summarize(
    *,
    qualification_path: Path,
    report_path: Path,
    independent_validation_path: Path,
) -> dict[str, Any]:
    qualification, manifest_sha256 = load_qualification_manifest(qualification_path)
    report = json.loads(report_path.read_text())
    independent = json.loads(independent_validation_path.read_text())
    validation = validate_h2_liger_report(
        report,
        qualification=qualification,
        expected_manifest_sha256=manifest_sha256,
        require_numerical_pass=False,
    )
    if independent.get("validation") != validation:
        raise ValueError("saved independent validation does not equal a fresh validation")
    if report.get("status") != "failed" or validation.get("numerical_status") != "failed":
        raise ValueError("R17 failure summarizer requires independently validated failed evidence")
    if report.get("mandatory_diagnostic_nonfinite_count") != 0:
        raise ValueError("R17 failure evidence contains a nonfinite mandatory diagnostic")
    zero = report["zero_target_matched_reference"]
    return {
        "artifact": "qwen35_r17_matched_reference_failure_anatomy",
        "schema_version": 1,
        "status": "evidence_validated_numerical_failure",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": manifest_sha256,
        "inputs": {
            "report_path": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "independent_validation_path": str(independent_validation_path.resolve()),
            "independent_validation_sha256": sha256_file(independent_validation_path),
        },
        "primary_comparison": report["primary_comparison"],
        "mandatory_diagnostic_comparison": report["mandatory_diagnostic_comparison"],
        "numerical_acceptance": report["numerical_acceptance"],
        "named_acceptance": {
            key: qualification["h2_acceptance"][key]
            for key in (
                "named_relative_metric",
                "named_relative_threshold",
                "named_minimum_cosine_similarity",
                "named_gradient_maximum_absolute_error",
            )
        },
        "independent_validation": validation,
        "decision_anatomy": _decision_anatomy(report),
        "zero_target_matched_reference": {
            "observed_loss": zero["observed_loss"],
            "reference_loss": zero["reference_loss"],
            "loss_maximum_absolute_error": zero["loss_comparison"]["maximum_absolute_error"],
            "hidden_gradient_difference_l2_norm": zero["hidden_gradient_comparison"]["difference_l2_norm"],
            "weight_gradient_difference_l2_norm": zero["output_weight_gradient_comparison"]["difference_l2_norm"],
            "graph_connected": zero["graph_connected"],
        },
        "direct_metric_extrema": _direct_extrema(report),
        "trajectory_metric_extrema": _trajectory_extrema(report),
        "failure_policy": qualification["h2_acceptance"]["liger_numerical_failure_policy"],
        "successor_gate_authorized": False,
        "scientific_training_authorized": False,
        "allowed_conclusion": (
            "R17 is an independently validated matched-reference numerical failure. Under the preregistered "
            "stop policy, Liger is abandoned for this campaign; H3 and scientific training remain blocked pending "
            "a separately preregistered correctness-first non-Liger successor."
        ),
    }


def main() -> None:
    args = parse_args()
    output = summarize(
        qualification_path=args.qualification_manifest,
        report_path=args.report,
        independent_validation_path=args.independent_validation,
    )
    write_json_atomic(args.output, output)
    print(json.dumps({"output": str(args.output), "status": output["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
