#!/usr/bin/env python3
"""Independently validate the immutable R16 dense-selection forensic R2 report.

This validator intentionally uses only the Python standard library and does
not import the CUDA producer or its metric helpers.  It validates provenance,
the parent replay/accounting relationship, serialized metric arithmetic,
repeatability, dtype/shape contracts, and the preregistered localization logic.
It has no gate authority and cannot change the failed R16 outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_REPORT_SHA256 = "69900b9720e1b74635d078583a355542a622fb655ab898c2c5a9e24fdf3d0657"
EXPECTED_PARENT_SHA256 = "b8e38873e64a6d1281b4510bb1213abe2f52b188f1c8e58d370fddf7ea9a99e7"
EXPECTED_MANIFEST_SHA256 = "827da32eefdf20839fef364b1bed23afb37122e0c19a981e460324c9d5c1b4f8"
EXPECTED_PRODUCER_SHA256 = "e8c20998f56bd0e38bda029b4f43df689ed79839b245bcbc1d89dc0cdf0d88b5"
EXPECTED_SOURCE_COMMIT = "e7a28fbd3de5b20b0219b980fa70d5eb97da2a9a"
EPSILON64 = 2.220446049250313e-16
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
CASES = (
    {"case_id": "F0", "trajectory_id": "R16-T0", "trajectory_index": 0, "replay_steps": 54, "assay_step": 55},
    {"case_id": "F1", "trajectory_id": "R16-T1", "trajectory_index": 1, "replay_steps": 64, "assay_step": 65},
    {"case_id": "F2", "trajectory_id": "R16-T2", "trajectory_index": 2, "replay_steps": 3, "assay_step": 4},
)
PATHS = ("full_ignore", "full_gather", "selected_gather")
COMPARISONS = (
    ("full_ignore", "full_gather"),
    ("full_gather", "selected_gather"),
    ("full_ignore", "selected_gather"),
)
TENSORS = (
    "selected_logits",
    "full_hidden_gradient",
    "selected_hidden_gradient",
    "output_weight_gradient",
    "selected_logit_gradient",
)
ARITHMETICS = (
    {
        "name": "production_bf16_autocast",
        "autocast": True,
        "projected_dtype": "torch.bfloat16",
    },
    {
        "name": "fp32_control",
        "autocast": False,
        "projected_dtype": "torch.float32",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--producer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float, *, rel_tol: float = 1e-10, abs_tol: float = 1e-14) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_from_label(label: str) -> tuple[str, int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return digest, int(digest[:8], 16)


def selected_target_count(sequence_length: int, modulus: int, offset: int) -> int:
    return sum((position + offset) % modulus == 0 for position in range(1, sequence_length))


def selected_positions(sequence_length: int, modulus: int, offset: int) -> list[int]:
    return [position - 1 for position in range(1, sequence_length) if (position + offset) % modulus == 0]


def expected_batch_accounting(*, step: int, trajectory_index: int, batch_seed_base: int) -> dict[str, int]:
    zero_index = step - 1
    modulus = (2, 3, 5, 7)[zero_index % 4]
    offset = (zero_index + trajectory_index) % modulus
    targets = selected_target_count(32, modulus, offset)
    extra = (zero_index * 3 + trajectory_index) % 13
    return {
        "seed": batch_seed_base + zero_index,
        "sequence_length": 32,
        "supervision_modulus": modulus,
        "supervision_offset": offset,
        "supervised_targets": targets,
        "divisor_extra": extra,
        "global_divisor": targets + extra,
    }


def validate_seed_contract(contract: dict[str, Any], expected_id: str) -> None:
    require(contract["trajectory_id"] == expected_id, f"{expected_id}: trajectory id drift")
    for prefix in ("model", "batch", "heldout"):
        label = contract[f"{prefix}_seed_label"]
        digest, seed = seed_from_label(label)
        seed_key = "batch_seed_base" if prefix == "batch" else f"{prefix}_seed"
        require(contract[f"{prefix}_seed_sha256"] == digest, f"{expected_id}: {prefix} seed digest drift")
        require(contract[seed_key] == seed, f"{expected_id}: {prefix} seed integer drift")


def validate_scalar(metric: dict[str, Any], context: str) -> None:
    require(
        set(metric) == {"observed", "reference", "maximum_absolute_error", "relative_error", "nonfinite_count"},
        f"{context}: scalar field set drift",
    )
    require(metric["nonfinite_count"] == 0, f"{context}: scalar nonfinite evidence")
    values = [metric[key] for key in ("observed", "reference", "maximum_absolute_error", "relative_error")]
    require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in values), f"{context}: invalid scalar")
    difference = abs(float(metric["observed"]) - float(metric["reference"]))
    require(close(metric["maximum_absolute_error"], difference, rel_tol=1e-12, abs_tol=1e-15), f"{context}: scalar absolute error drift")
    expected_relative = difference / max(abs(float(metric["reference"])), EPSILON64)
    require(close(metric["relative_error"], expected_relative, rel_tol=1e-12, abs_tol=1e-15), f"{context}: scalar relative error drift")


def validate_tensor_metric(metric: dict[str, Any], context: str) -> None:
    require(
        set(metric)
        == {
            "elements",
            "maximum_absolute_error",
            "relative_l2_error",
            "cosine_similarity",
            "observed_l2_norm",
            "reference_l2_norm",
            "difference_l2_norm",
            "nonfinite_count",
        },
        f"{context}: tensor metric field set drift",
    )
    require(isinstance(metric["elements"], int) and metric["elements"] > 0, f"{context}: invalid elements")
    require(metric["nonfinite_count"] == 0, f"{context}: tensor nonfinite evidence")
    names = (
        "maximum_absolute_error",
        "relative_l2_error",
        "cosine_similarity",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
    )
    require(all(isinstance(metric[name], (int, float)) and math.isfinite(metric[name]) for name in names), f"{context}: invalid tensor metric")
    maximum = float(metric["maximum_absolute_error"])
    difference = float(metric["difference_l2_norm"])
    observed = float(metric["observed_l2_norm"])
    reference = float(metric["reference_l2_norm"])
    require(min(maximum, difference, observed, reference) >= 0, f"{context}: negative norm")
    require(difference + 1e-14 >= maximum, f"{context}: L2 below Linf")
    require(difference <= math.sqrt(metric["elements"]) * maximum + 1e-12, f"{context}: L2 exceeds sqrt(n)*Linf")
    expected_relative = difference / max(reference, EPSILON64)
    require(close(metric["relative_l2_error"], expected_relative, rel_tol=1e-12, abs_tol=1e-14), f"{context}: relative L2 drift")
    if observed and reference:
        dot = (observed * observed + reference * reference - difference * difference) / 2
        expected_cosine = dot / (observed * reference)
        require(close(metric["cosine_similarity"], expected_cosine, rel_tol=5e-10, abs_tol=5e-12), f"{context}: cosine/norm-law drift")


def validate_tensor_identity(identity: dict[str, Any], *, shape: list[int], dtype: str, context: str) -> None:
    require(set(identity) == {"shape", "elements", "dtype", "nonfinite_count", "sha256"}, f"{context}: identity field set drift")
    require(identity["shape"] == shape, f"{context}: shape drift")
    require(identity["elements"] == math.prod(shape), f"{context}: element-count drift")
    require(identity["dtype"] == dtype, f"{context}: dtype drift")
    require(identity["nonfinite_count"] == 0, f"{context}: nonfinite tensor")
    require(isinstance(identity["sha256"], str) and SHA256_PATTERN.fullmatch(identity["sha256"]), f"{context}: invalid SHA-256")


def validate_repeatability(path: dict[str, Any], context: str) -> dict[str, bool]:
    repeats = path["repeats"]
    require(len(repeats) == 5, f"{context}: repeat cardinality drift")
    independent = {
        "loss_bit_exact": len({repeat["loss"] for repeat in repeats}) == 1,
        "tensor_hashes_bit_exact": {
            field: len({repeat["tensors"][field]["sha256"] for repeat in repeats}) == 1 for field in TENSORS
        },
    }
    independent["all_recorded_outputs_bit_exact"] = independent["loss_bit_exact"] and all(
        independent["tensor_hashes_bit_exact"].values()
    )
    require(path["repeatability"] == independent, f"{context}: producer repeatability summary drift")
    require(independent["all_recorded_outputs_bit_exact"], f"{context}: non-repeatable execution")
    return independent


def all_metrics_exact(metrics: dict[str, Any]) -> bool:
    return metrics["loss"]["maximum_absolute_error"] == 0 and all(
        metrics[field]["difference_l2_norm"] == 0 for field in TENSORS
    )


def classify_comparison(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "all_recorded_metrics_exact": all_metrics_exact(metrics),
        "loss_exact": metrics["loss"]["maximum_absolute_error"] == 0,
        **{f"{field}_exact": metrics[field]["difference_l2_norm"] == 0 for field in TENSORS},
    }


def localization_from_comparisons(comparisons: dict[tuple[str, str], dict[str, Any]]) -> str:
    ab = classify_comparison(comparisons[("full_ignore", "full_gather")])
    bc = classify_comparison(comparisons[("full_gather", "selected_gather")])
    if not ab["all_recorded_metrics_exact"]:
        return "ignore_mask_or_loss_reduction_shape_effect"
    if bc["selected_logits_exact"] and (
        not bc["full_hidden_gradient_exact"] or not bc["selected_hidden_gradient_exact"] or not bc["output_weight_gradient_exact"]
    ):
        if (
            not bc["full_hidden_gradient_exact"]
            and not bc["selected_hidden_gradient_exact"]
            and bc["output_weight_gradient_exact"]
            and bc["selected_logit_gradient_exact"]
        ):
            return "projection_shape_backward_hidden_input_gradient_only"
        return "projection_shape_backward_effect"
    if not bc["selected_logits_exact"]:
        return "projection_shape_forward_rounding_and_downstream_effect"
    if bc["all_recorded_metrics_exact"]:
        return "direct_objectives_exact"
    return "unclassified_projection_shape_effect"


def validate_parent_case(parent: dict[str, Any], expected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [
        run
        for run in parent["runs"]
        if run["variant"] == "bf16_dense_selected_vs_dense_full"
        and run["trajectory_contract"]["trajectory_id"] == expected["trajectory_id"]
    ]
    require(len(matches) == 1, f"{expected['case_id']}: parent trajectory cardinality drift")
    run = matches[0]
    assay = run["steps"][expected["assay_step"] - 1]
    prior = run["steps"][: expected["assay_step"] - 1]
    require(all(step["aggregate_preclip_gradient"]["difference_l2_norm"] == 0 for step in prior), f"{expected['case_id']}: earlier parent gradient divergence")
    require(assay["aggregate_preclip_gradient"]["difference_l2_norm"] > 0, f"{expected['case_id']}: parent assay has no divergence")
    if prior:
        require(prior[-1]["aggregate_cumulative_displacement"]["difference_l2_norm"] == 0, f"{expected['case_id']}: parent pre-assay state differs")
    return run, assay


def validate_case(case: dict[str, Any], expected: dict[str, Any], parent: dict[str, Any], counters: dict[str, int]) -> dict[str, Any]:
    context = expected["case_id"]
    for key, value in expected.items():
        require(case[key] == value, f"{context}: {key} drift")
    require(case["status"] == "forensic_case_complete_no_gate", f"{context}: incomplete case")
    run, parent_assay = validate_parent_case(parent, expected)
    contract = run["trajectory_contract"]
    validate_seed_contract(contract, expected["trajectory_id"])
    require(case["parent_evidence"]["trajectory_contract"] == contract, f"{context}: embedded parent contract drift")
    require(case["parent_evidence"]["parent_batch_accounting"] == parent_assay["batch_accounting"], f"{context}: embedded parent batch drift")
    require(
        case["parent_evidence"]["parent_first_gradient_difference_l2_norm"]
        == parent_assay["aggregate_preclip_gradient"]["difference_l2_norm"],
        f"{context}: embedded parent divergence drift",
    )
    require(case["parent_evidence"]["parent_pre_assay_complete_state_bit_exact"] is True, f"{context}: pre-assay state attestation drift")
    replay = case["replay"]
    require(replay["final_optimizer_step"] == expected["replay_steps"], f"{context}: replay final step drift")
    require(len(replay["steps"]) == expected["replay_steps"], f"{context}: replay length drift")
    for step_number, (step, parent_step) in enumerate(zip(replay["steps"], run["steps"]), start=1):
        step_context = f"{context}/replay-{step_number}"
        require(step["step"] == step_number, f"{step_context}: order drift")
        accounting = expected_batch_accounting(
            step=step_number,
            trajectory_index=expected["trajectory_index"],
            batch_seed_base=contract["batch_seed_base"],
        )
        require(step["batch_accounting"] == accounting == parent_step["batch_accounting"], f"{step_context}: accounting drift")
        require(step["autocast_contract"] == {"device_type": "cuda", "dtype": "torch.bfloat16", "enabled": True}, f"{step_context}: autocast drift")
        require(isinstance(step["loss"], (int, float)) and math.isfinite(step["loss"]), f"{step_context}: invalid loss")
        require(step["loss"] == parent_step["loss"]["reference"], f"{step_context}: replay loss differs from parent reference")
        require(isinstance(step["preclip_gradient_norm"], (int, float)) and math.isfinite(step["preclip_gradient_norm"]), f"{step_context}: invalid gradient norm")
        require(
            close(step["preclip_gradient_norm"], parent_step["aggregate_preclip_gradient"]["reference_l2_norm"], rel_tol=2e-6, abs_tol=2e-7),
            f"{step_context}: replay gradient norm differs from parent reference",
        )
        counters["replay_steps"] += 1
    assay_accounting = expected_batch_accounting(
        step=expected["assay_step"],
        trajectory_index=expected["trajectory_index"],
        batch_seed_base=contract["batch_seed_base"],
    )
    require(case["replayed_assay_batch_accounting"] == assay_accounting == parent_assay["batch_accounting"], f"{context}: assay accounting drift")
    expected_positions = selected_positions(32, assay_accounting["supervision_modulus"], assay_accounting["supervision_offset"])
    require(case["selected_positions"] == expected_positions, f"{context}: selected-position drift")
    require(len(expected_positions) == assay_accounting["supervised_targets"], f"{context}: selected-position count drift")
    require(SHA256_PATTERN.fullmatch(case["selected_targets_sha256"]), f"{context}: invalid selected-target hash")

    summaries = []
    require(len(case["arithmetic_results"]) == 2, f"{context}: arithmetic cardinality drift")
    for arithmetic, expected_arithmetic in zip(case["arithmetic_results"], ARITHMETICS):
        arithmetic_context = f"{context}/{expected_arithmetic['name']}"
        require(arithmetic["arithmetic"] == expected_arithmetic["name"], f"{arithmetic_context}: arithmetic order drift")
        require(arithmetic["autocast_enabled"] is expected_arithmetic["autocast"], f"{arithmetic_context}: autocast drift")
        require(
            arithmetic["arithmetic_contract"]
            == {
                "captured_hidden_dtype": "torch.float32",
                "weight_storage_dtype": "torch.float32",
                "expected_projected_logit_dtype": expected_arithmetic["projected_dtype"],
                "cross_entropy_dtype": "torch.float32",
            },
            f"{arithmetic_context}: arithmetic contract drift",
        )
        validate_tensor_identity(arithmetic["base_hidden"], shape=[1, 32, 64], dtype="torch.float32", context=f"{arithmetic_context}/base-hidden")
        validate_tensor_identity(arithmetic["base_weight"], shape=[256, 64], dtype="torch.float32", context=f"{arithmetic_context}/base-weight")
        require(set(arithmetic["paths"]) == set(PATHS), f"{arithmetic_context}: path set drift")
        first_repeats: dict[str, dict[str, Any]] = {}
        for path_name in PATHS:
            path = arithmetic["paths"][path_name]
            validate_repeatability(path, f"{arithmetic_context}/{path_name}")
            first_repeats[path_name] = path["repeats"][0]
            for repeat_number, repeat in enumerate(path["repeats"]):
                repeat_context = f"{arithmetic_context}/{path_name}/repeat-{repeat_number}"
                require(repeat["path"] == path_name and repeat["repeat"] == repeat_number, f"{repeat_context}: identity drift")
                require(isinstance(repeat["loss"], (int, float)) and math.isfinite(repeat["loss"]), f"{repeat_context}: invalid loss")
                require(repeat["loss_dtype"] == "torch.float32", f"{repeat_context}: loss dtype drift")
                require(repeat["leaf_hidden"] == arithmetic["base_hidden"], f"{repeat_context}: hidden leaf differs from base")
                require(repeat["leaf_weight"] == arithmetic["base_weight"], f"{repeat_context}: weight leaf differs from base")
                require(set(repeat["tensors"]) == set(TENSORS), f"{repeat_context}: tensor set drift")
                targets = assay_accounting["supervised_targets"]
                validate_tensor_identity(repeat["tensors"]["selected_logits"], shape=[1, targets, 256], dtype=expected_arithmetic["projected_dtype"], context=f"{repeat_context}/selected-logits")
                validate_tensor_identity(repeat["tensors"]["selected_logit_gradient"], shape=[1, targets, 256], dtype=expected_arithmetic["projected_dtype"], context=f"{repeat_context}/selected-logit-gradient")
                validate_tensor_identity(repeat["tensors"]["full_hidden_gradient"], shape=[1, 32, 64], dtype="torch.float32", context=f"{repeat_context}/full-hidden-gradient")
                validate_tensor_identity(repeat["tensors"]["selected_hidden_gradient"], shape=[1, targets, 64], dtype="torch.float32", context=f"{repeat_context}/selected-hidden-gradient")
                validate_tensor_identity(repeat["tensors"]["output_weight_gradient"], shape=[256, 64], dtype="torch.float32", context=f"{repeat_context}/output-weight-gradient")
                counters["path_repeats"] += 1
        require(len(arithmetic["comparisons"]) == 3, f"{arithmetic_context}: comparison cardinality drift")
        comparison_map: dict[tuple[str, str], dict[str, Any]] = {}
        for comparison, expected_pair in zip(arithmetic["comparisons"], COMPARISONS):
            pair = (comparison["left"], comparison["right"])
            require(pair == expected_pair, f"{arithmetic_context}: comparison order drift")
            require(set(comparison["metrics"]) == {"loss", *TENSORS}, f"{arithmetic_context}/{pair}: metric set drift")
            metrics = comparison["metrics"]
            validate_scalar(metrics["loss"], f"{arithmetic_context}/{pair}/loss")
            left = first_repeats[pair[0]]
            right = first_repeats[pair[1]]
            require(metrics["loss"]["observed"] == left["loss"], f"{arithmetic_context}/{pair}: observed loss path drift")
            require(metrics["loss"]["reference"] == right["loss"], f"{arithmetic_context}/{pair}: reference loss path drift")
            for field in TENSORS:
                validate_tensor_metric(metrics[field], f"{arithmetic_context}/{pair}/{field}")
                require(metrics[field]["elements"] == left["tensors"][field]["elements"] == right["tensors"][field]["elements"], f"{arithmetic_context}/{pair}/{field}: geometry drift")
                hashes_equal = left["tensors"][field]["sha256"] == right["tensors"][field]["sha256"]
                metric_exact = metrics[field]["difference_l2_norm"] == 0
                require(hashes_equal == metric_exact, f"{arithmetic_context}/{pair}/{field}: hash/metric exactness disagreement")
                counters["tensor_metrics"] += 1
            comparison_map[pair] = metrics
            counters["scalar_metrics"] += 1
        ab = classify_comparison(comparison_map[("full_ignore", "full_gather")])
        bc = classify_comparison(comparison_map[("full_gather", "selected_gather")])
        require(ab["all_recorded_metrics_exact"], f"{arithmetic_context}: A/B is not exact")
        localization = localization_from_comparisons(comparison_map)
        if expected_arithmetic["name"] == "production_bf16_autocast":
            require(localization == "projection_shape_backward_hidden_input_gradient_only", f"{arithmetic_context}: unexpected production localization")
        else:
            require(localization == "projection_shape_forward_rounding_and_downstream_effect", f"{arithmetic_context}: unexpected FP32 localization")
        summaries.append(
            {
                "arithmetic": expected_arithmetic["name"],
                "ab": ab,
                "bc": bc,
                "localization": localization,
                "bc_metrics": comparison_map[("full_gather", "selected_gather")],
            }
        )
    counters["cases"] += 1
    return {"case_id": context, "trajectory_id": expected["trajectory_id"], "arithmetic_summaries": summaries}


def main() -> None:
    args = parse_args()
    report_sha256 = sha256_file(args.report)
    parent_sha256 = sha256_file(args.parent_report)
    manifest_sha256 = sha256_file(args.qualification_manifest)
    producer_sha256 = sha256_file(args.producer_source)
    require(report_sha256 == EXPECTED_REPORT_SHA256, "unexpected forensic R2 report bytes")
    require(parent_sha256 == EXPECTED_PARENT_SHA256, "unexpected parent diagnostic bytes")
    require(manifest_sha256 == EXPECTED_MANIFEST_SHA256, "unexpected R16 manifest bytes")
    require(producer_sha256 == EXPECTED_PRODUCER_SHA256, "unexpected producer source bytes")
    report = json.loads(args.report.read_text())
    parent = json.loads(args.parent_report.read_text())
    manifest = json.loads(args.qualification_manifest.read_text())
    require(manifest["schema_version"] == 1 and manifest["protocol_id"] == "qwen35-hardware-qualification-r16", "manifest identity drift")
    require(parent["status"] == "diagnostic_complete_no_gate", "parent diagnostic status drift")
    require(parent["successor_gate_authorized"] is False and parent["scientific_training_authorized"] is False, "parent authority drift")
    require(report["artifact"] == "qwen35_r16_dense_selection_divergence_forensic", "artifact identity drift")
    require(report["schema_version"] == 1, "schema version drift")
    require(report["status"] == "forensic_complete_no_gate", "forensic did not complete")
    require(report["successor_gate_authorized"] is False and report["scientific_training_authorized"] is False, "forensic authority drift")
    require(report["allowed_conclusion"] == "Forensic localization only; R16 remains failed and H3 remains blocked.", "allowed conclusion drift")
    require(report["qualification_manifest_sha256"] == manifest_sha256, "embedded manifest digest drift")
    require(report["parent_diagnostic_sha256"] == parent_sha256, "embedded parent digest drift")
    require(report["source_commit"] == EXPECTED_SOURCE_COMMIT, "source commit drift")
    require(report["source_sha256"] == producer_sha256, "embedded producer digest drift")
    require(report["cuda_device"] == "NVIDIA A100-SXM-64GB", "CUDA device drift")
    require(report["repeat_count"] == 5, "repeat count drift")
    require(
        report["path_definitions"]
        == {
            "full_ignore": "full projection then shifted ignore-index cross entropy",
            "full_gather": "full projection then explicit supervised-position gather and cross entropy",
            "selected_gather": "supervised hidden-row gather then selected projection and cross entropy",
        },
        "path definitions drift",
    )
    require(len(report["cases"]) == len(CASES), "case cardinality drift")
    counters = {"cases": 0, "replay_steps": 0, "path_repeats": 0, "scalar_metrics": 0, "tensor_metrics": 0}
    summaries = [validate_case(case, expected, parent, counters) for case, expected in zip(report["cases"], CASES)]
    output = {
        "artifact": "qwen35_r16_dense_selection_forensic_r2_independent_validation",
        "schema_version": 1,
        "status": "evidence_validated_no_gate",
        "report_path": str(args.report.resolve()),
        "report_sha256": report_sha256,
        "parent_report_sha256": parent_sha256,
        "qualification_manifest_sha256": manifest_sha256,
        "producer_source_sha256": producer_sha256,
        "validation_scope": {
            "producer_status_not_trusted": True,
            "standard_library_only": True,
            "provenance_bytes_bound": True,
            "parent_first_divergence_and_replay_relationship_recomputed": True,
            "batch_seed_and_supervision_accounting_recomputed": True,
            "repeatability_recomputed_from_serialized_tensor_hashes": True,
            "metric_arithmetic_recomputed_from_serialized_primitives": True,
            "hash_metric_exactness_cross_checked": True,
            "preregistered_localization_recomputed": True,
            "raw_tensor_values_unavailable": True,
            "selected_target_values_unavailable": True,
        },
        "validated_counts": counters,
        "summaries": summaries,
        "allowed_conclusion": (
            "The serialized R2 evidence is internally consistent and localizes the production-BF16 full-versus-selected "
            "dense difference at all three first-divergence states to projection-shape-dependent backward rounding in "
            "the hidden-input gradient. Full ignore masking and full explicit gather are exact. This is forensic "
            "localization only: R16 remains failed, no successor gate is authorized, and H3/scientific training remain blocked."
        ),
        "successor_gate_authorized": False,
        "scientific_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": output["status"], "validated_counts": counters}, sort_keys=True))


if __name__ == "__main__":
    main()
