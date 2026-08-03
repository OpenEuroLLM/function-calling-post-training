#!/usr/bin/env python3
"""Diagnostic-only long-horizon localization of the immutable R16 H2 failure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import traceback
from pathlib import Path
from typing import Any

import torch
from scripts.train.qwen35 import diagnose_qwen35_h2_r15_postfailure as r15_diagnostic
from scripts.train.qwen35 import validate_qwen35_selective_loss as r14_assay

from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_r16 import load_qualification_manifest, validate_h2_liger_report

EXPECTED_R16_MANIFEST_SHA256 = "827da32eefdf20839fef364b1bed23afb37122e0c19a981e460324c9d5c1b4f8"
EXPECTED_R16_REPORT_SHA256 = "e823119d3b70b195134bcaa2d44b4a5a2c8e722467106fb1d5dbf3d7edbf9866"
CHECKPOINTS_128 = (1, 2, 4, 8, 16, 32, 64, 96, 128)
CHECKPOINTS_512 = (*CHECKPOINTS_128, 192, 256, 384, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--r16-failed-report", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _seed_identity(label: str) -> dict[str, str | int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return {"label": label, "sha256": digest, "seed": int(digest[:8], 16)}


def _long_contracts() -> list[dict[str, Any]]:
    result = []
    for index in range(3):
        base = f"qwen35-hardware-qualification-r16-postfailure-long-diagnostic-{index}"
        model = _seed_identity(base)
        batches = _seed_identity(f"{base}-batches")
        heldout = _seed_identity(f"{base}-heldout")
        result.append(
            {
                "trajectory_id": f"R16-PF-L{index}",
                "model_seed_label": model["label"],
                "model_seed_sha256": model["sha256"],
                "model_seed": model["seed"],
                "batch_seed_label": batches["label"],
                "batch_seed_sha256": batches["sha256"],
                "batch_seed_base": batches["seed"],
                "heldout_seed_label": heldout["label"],
                "heldout_seed_sha256": heldout["sha256"],
                "heldout_seed": heldout["seed"],
            }
        )
    return result


def _metric_bit_exact(metric: dict[str, Any]) -> bool:
    return metric["nonfinite_count"] == 0 and metric["difference_l2_norm"] == 0


def _run_bit_exact(run: dict[str, Any]) -> bool:
    if run.get("status") != "diagnostic_complete_no_gate":
        return False
    aggregate_fields = (
        "aggregate_preclip_gradient",
        "aggregate_clipped_gradient",
        "aggregate_raw_update",
        "aggregate_cumulative_displacement",
        "aggregate_post_step_parameter",
        "aggregate_adam_exp_avg",
        "aggregate_adam_exp_avg_sq",
    )
    named_fields = (
        "preclip_gradient",
        "clipped_gradient",
        "raw_update",
        "cumulative_displacement",
        "post_step_parameter",
        "adam_exp_avg",
        "adam_exp_avg_sq",
    )
    for step in run["steps"]:
        if step["loss"]["nonfinite_count"] or step["loss"]["maximum_absolute_error"] != 0:
            return False
        if any(not _metric_bit_exact(step[field]) for field in aggregate_fields):
            return False
        if not _metric_bit_exact(step["heldout"]["logits"]):
            return False
        if step["heldout"]["loss"]["nonfinite_count"] or step["heldout"]["loss"]["maximum_absolute_error"] != 0:
            return False
        if any(not _metric_bit_exact(row[field]) for row in step["per_parameter"].values() for field in named_fields):
            return False
    return True


def _ols_log_slope(steps: list[dict[str, Any]], field: str, start: int, end: int) -> float | None:
    values = []
    for row in steps:
        if start <= row["step"] <= end:
            value = row["aggregate_cumulative_displacement"][field]
            if value is None or not math.isfinite(value) or value <= 0:
                return None
            values.append((float(row["step"]), math.log10(value)))
    if len(values) < 2:
        return None
    mean_x = sum(x for x, _ in values) / len(values)
    mean_y = sum(y for _, y in values) / len(values)
    denominator = sum((x - mean_x) ** 2 for x, _ in values)
    return sum((x - mean_x) * (y - mean_y) for x, y in values) / denominator


def _checkpoint_summary(step: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any]:
    cumulative = step["aggregate_cumulative_displacement"]
    named_exceedances = []
    for name, row in step["per_parameter"].items():
        metric = row["cumulative_displacement"]
        if (
            metric["nonfinite_count"]
            or metric["relative_l2_error"] > acceptance["update_relative_l2_error"]
            or metric["cosine_similarity"] is None
            or metric["cosine_similarity"] < acceptance["update_minimum_cosine_similarity"]
        ):
            named_exceedances.append(
                {
                    "parameter": name,
                    "relative_l2_error": metric["relative_l2_error"],
                    "cosine_similarity": metric["cosine_similarity"],
                    "difference_l2_norm": metric["difference_l2_norm"],
                    "reference_l2_norm": metric["reference_l2_norm"],
                    "nonfinite_count": metric["nonfinite_count"],
                }
            )
    return {
        "step": step["step"],
        "aggregate_cumulative_displacement": cumulative,
        "aggregate_post_step_parameter": step["aggregate_post_step_parameter"],
        "aggregate_adam_exp_avg": step["aggregate_adam_exp_avg"],
        "aggregate_adam_exp_avg_sq": step["aggregate_adam_exp_avg_sq"],
        "heldout_logits": step["heldout"]["logits"],
        "heldout_loss": step["heldout"]["loss"],
        "named_cumulative_exceedance_count_under_historical_envelope": len(named_exceedances),
        "named_cumulative_exceedances_under_historical_envelope": named_exceedances,
    }


def _summary(run: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any]:
    if run.get("status") != "diagnostic_complete_no_gate":
        return {"status": run.get("status"), "variant": run.get("variant"), "trajectory_id": run.get("trajectory_id")}
    steps = run["steps"]
    checkpoints = CHECKPOINTS_512 if len(steps) == 512 else CHECKPOINTS_128
    cumulative = [step["aggregate_cumulative_displacement"] for step in steps]
    heldout = [step["heldout"]["logits"] for step in steps]
    post = [step["aggregate_post_step_parameter"] for step in steps]
    windows = [(1, 128)] if len(steps) == 128 else [(1, 128), (129, 256), (257, 512)]
    return {
        "status": run["status"],
        "variant": run["variant"],
        "trajectory_id": run["trajectory_contract"]["trajectory_id"],
        "steps": len(steps),
        "all_recorded_comparisons_bit_exact": _run_bit_exact(run),
        "maximum_cumulative_relative_l2": max(row["relative_l2_error"] for row in cumulative),
        "final_cumulative_relative_l2": cumulative[-1]["relative_l2_error"],
        "maximum_cumulative_difference_l2_norm": max(row["difference_l2_norm"] for row in cumulative),
        "final_cumulative_difference_l2_norm": cumulative[-1]["difference_l2_norm"],
        "final_cumulative_reference_l2_norm": cumulative[-1]["reference_l2_norm"],
        "maximum_post_step_parameter_relative_l2": max(row["relative_l2_error"] for row in post),
        "final_post_step_parameter_relative_l2": post[-1]["relative_l2_error"],
        "maximum_heldout_logit_relative_l2": max(row["relative_l2_error"] for row in heldout),
        "final_heldout_logit_relative_l2": heldout[-1]["relative_l2_error"],
        "maximum_heldout_logit_absolute_error": max(row["maximum_absolute_error"] for row in heldout),
        "final_heldout_logit_absolute_error": heldout[-1]["maximum_absolute_error"],
        "nonfinite_count": sum(
            step[field]["nonfinite_count"]
            for step in steps
            for field in (
                "aggregate_preclip_gradient",
                "aggregate_clipped_gradient",
                "aggregate_raw_update",
                "aggregate_cumulative_displacement",
                "aggregate_post_step_parameter",
                "aggregate_adam_exp_avg",
                "aggregate_adam_exp_avg_sq",
            )
        )
        + sum(step["heldout"]["logits"]["nonfinite_count"] for step in steps),
        "log10_ols_slopes_by_fixed_window": [
            {
                "start": start,
                "end": end,
                "difference_l2_norm_per_step": _ols_log_slope(steps, "difference_l2_norm", start, end),
                "relative_l2_error_per_step": _ols_log_slope(steps, "relative_l2_error", start, end),
            }
            for start, end in windows
        ],
        "fixed_checkpoints": [_checkpoint_summary(steps[step - 1], acceptance) for step in checkpoints],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R16 post-failure diagnostic requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if qualification_sha256 != EXPECTED_R16_MANIFEST_SHA256:
        raise RuntimeError("diagnostic received an unexpected R16 manifest")
    if sha256_file(args.r16_failed_report) != EXPECTED_R16_REPORT_SHA256:
        raise RuntimeError("diagnostic received an unexpected R16 failed report")
    parent = json.loads(args.r16_failed_report.read_text())
    parent_validation = validate_h2_liger_report(
        parent,
        qualification=qualification,
        expected_manifest_sha256=qualification_sha256,
        require_numerical_pass=False,
    )
    if parent_validation["numerical_status"] != "failed" or parent.get("successor_gate_authorized") is not False:
        raise RuntimeError("diagnostic parent is not the independently validated immutable R16 failure")

    h2 = qualification["h2_acceptance"]
    r16_contracts = h2["confirmatory_trajectories"]
    long_contracts = _long_contracts()
    plans = [
        {
            "variant": "bf16_dense_selected_vs_dense_full",
            "contracts": r16_contracts,
            "origin": "r16_outcome_known",
            "steps": 128,
        },
        {
            "variant": "fp32_liger_vs_dense_full",
            "contracts": r16_contracts,
            "origin": "r16_outcome_known",
            "steps": 128,
        },
        {
            "variant": "bf16_liger_vs_dense_full",
            "contracts": long_contracts,
            "origin": "outcome_unseen_long_diagnostic",
            "steps": 512,
        },
    ]
    source = r14_assay._verify_liger_source_pin(qualification["runtime_pins"]["liger_source_files_sha256"])
    runs = []
    for plan in plans:
        run_h2 = copy.deepcopy(h2)
        run_h2["trajectory_steps"] = plan["steps"]
        for index, contract in enumerate(plan["contracts"]):
            try:
                run = r15_diagnostic._run_pair(
                    variant=plan["variant"],
                    trajectory_contract=contract,
                    trajectory_index=index,
                    h2=run_h2,
                    optimizer_config=qualification["training_unit"],
                )
                run["contract_origin"] = plan["origin"]
            except Exception as error:
                run = {
                    "status": "diagnostic_failed_no_gate",
                    "variant": plan["variant"],
                    "trajectory_id": contract["trajectory_id"],
                    "trajectory_contract": contract,
                    "contract_origin": plan["origin"],
                    "planned_steps": plan["steps"],
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            runs.append(run)

    complete = sum(run["status"] == "diagnostic_complete_no_gate" for run in runs)
    overall = (
        "diagnostic_complete_no_gate"
        if complete == len(runs)
        else ("diagnostic_partial_no_gate" if complete else "diagnostic_failed_no_gate")
    )
    repo_root = Path(__file__).resolve().parents[3]
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "artifact": "qwen35_r16_h2_postfailure_long_horizon_diagnostic",
        "schema_version": 1,
        "status": overall,
        "successor_gate_authorized": False,
        "scientific_training_authorized": False,
        "allowed_conclusion": "Diagnostic localization only; R16 remains failed and H3 remains blocked.",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "r16_failed_report_sha256": EXPECTED_R16_REPORT_SHA256,
        "parent_failure_validation": parent_validation,
        "diagnostic_source_path": str(Path(__file__).resolve()),
        "diagnostic_source_sha256": sha256_file(Path(__file__)),
        "source_commit": source_commit,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "liger_kernel": source,
        "r16_trajectory_contracts": r16_contracts,
        "outcome_unseen_long_contracts": long_contracts,
        "fixed_checkpoints_128": list(CHECKPOINTS_128),
        "fixed_checkpoints_512": list(CHECKPOINTS_512),
        "planned_runs": 9,
        "complete_runs": complete,
        "runs": runs,
        "summaries": [_summary(run, qualification["numerical_acceptance"]) for run in runs],
    }
    r14_assay._write_strict_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": overall}, sort_keys=True))


if __name__ == "__main__":
    main()
