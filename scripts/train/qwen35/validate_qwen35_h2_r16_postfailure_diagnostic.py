#!/usr/bin/env python3
"""Independently validate and summarize the immutable R16 post-failure diagnostic.

This validator intentionally uses only the Python standard library and does
not import the diagnostic producer or its metric helpers.  It validates the
published evidence, but has no gate authority and cannot change the R16
failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

EXPECTED_REPORT_SHA256 = "b8e38873e64a6d1281b4510bb1213abe2f52b188f1c8e58d370fddf7ea9a99e7"
EXPECTED_MANIFEST_SHA256 = "827da32eefdf20839fef364b1bed23afb37122e0c19a981e460324c9d5c1b4f8"
EXPECTED_PARENT_SHA256 = "e823119d3b70b195134bcaa2d44b4a5a2c8e722467106fb1d5dbf3d7edbf9866"
EXPECTED_SOURCE_COMMIT = "76d1a394105acaec1f7c490c6b869824e8203c2d"
EXPECTED_PRODUCER_SHA256 = "ac222c7295b1c8d650375f023604ad77e13145d02b097705f9971761809c9ff6"
EPSILON64 = 2.220446049250313e-16
TOTAL_ELEMENTS = 57_568
TENSOR_FIELDS = (
    "preclip_gradient",
    "clipped_gradient",
    "raw_update",
    "cumulative_displacement",
    "post_step_parameter",
    "adam_exp_avg",
    "adam_exp_avg_sq",
)
AGGREGATE_FIELDS = tuple(f"aggregate_{field}" for field in TENSOR_FIELDS)
GEOMETRY = (
    ("model.embed_tokens.weight", (256, 64), 16_384),
    ("model.layers.0.self_attn.q_proj.weight", (128, 64), 8_192),
    ("model.layers.0.self_attn.k_proj.weight", (32, 64), 2_048),
    ("model.layers.0.self_attn.v_proj.weight", (32, 64), 2_048),
    ("model.layers.0.self_attn.o_proj.weight", (64, 64), 4_096),
    ("model.layers.0.self_attn.q_norm.weight", (16,), 16),
    ("model.layers.0.self_attn.k_norm.weight", (16,), 16),
    ("model.layers.0.mlp.gate_proj.weight", (128, 64), 8_192),
    ("model.layers.0.mlp.up_proj.weight", (128, 64), 8_192),
    ("model.layers.0.mlp.down_proj.weight", (64, 128), 8_192),
    ("model.layers.0.input_layernorm.weight", (64,), 64),
    ("model.layers.0.post_attention_layernorm.weight", (64,), 64),
    ("model.norm.weight", (64,), 64),
)
GEOMETRY_BY_NAME = {name: {"name": name, "shape": list(shape), "elements": elements} for name, shape, elements in GEOMETRY}
DTYPES = {
    "observed_parameters": ["torch.float32"],
    "reference_parameters": ["torch.float32"],
    "observed_gradients": ["torch.float32"],
    "reference_gradients": ["torch.float32"],
    "observed_optimizer": ["torch.float32"],
    "reference_optimizer": ["torch.float32"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
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


def validate_contract(contract: dict[str, Any], *, expected_id: str) -> None:
    require(contract["trajectory_id"] == expected_id, f"wrong trajectory id for {expected_id}")
    for prefix in ("model", "batch", "heldout"):
        label_key = f"{prefix}_seed_label"
        sha_key = f"{prefix}_seed_sha256"
        seed_key = "batch_seed_base" if prefix == "batch" else f"{prefix}_seed"
        digest, seed = seed_from_label(contract[label_key])
        require(contract[sha_key] == digest, f"{expected_id} {prefix} seed digest drift")
        require(contract[seed_key] == seed, f"{expected_id} {prefix} seed integer drift")


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


def validate_tensor(metric: dict[str, Any], context: str) -> None:
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
        f"{context}: tensor field set drift",
    )
    require(isinstance(metric["elements"], int) and metric["elements"] > 0, f"{context}: bad element count")
    require(metric["nonfinite_count"] == 0, f"{context}: nonfinite evidence")
    names = (
        "maximum_absolute_error",
        "relative_l2_error",
        "cosine_similarity",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
    )
    require(
        all(isinstance(metric[name], (int, float)) and math.isfinite(metric[name]) for name in names),
        f"{context}: missing or nonfinite derived metric",
    )
    n = metric["elements"]
    maximum = float(metric["maximum_absolute_error"])
    difference = float(metric["difference_l2_norm"])
    observed = float(metric["observed_l2_norm"])
    reference = float(metric["reference_l2_norm"])
    require(maximum >= 0 and difference >= 0 and observed >= 0 and reference >= 0, f"{context}: negative norm")
    require(difference + 1e-14 >= maximum, f"{context}: L2 below Linf")
    require(difference <= math.sqrt(n) * maximum + 1e-12, f"{context}: L2 exceeds sqrt(n)*Linf")
    expected_relative = difference / max(reference, EPSILON64)
    require(close(metric["relative_l2_error"], expected_relative, rel_tol=1e-12, abs_tol=1e-14), f"{context}: relative L2 drift")
    if observed and reference:
        dot = (observed * observed + reference * reference - difference * difference) / 2
        expected_cosine = dot / (observed * reference)
        require(close(metric["cosine_similarity"], expected_cosine, rel_tol=5e-10, abs_tol=5e-12), f"{context}: cosine/norm-law drift")


def validate_partition(aggregate: dict[str, Any], named: list[dict[str, Any]], context: str) -> None:
    require(aggregate["elements"] == TOTAL_ELEMENTS, f"{context}: aggregate geometry drift")
    require(sum(metric["elements"] for metric in named) == TOTAL_ELEMENTS, f"{context}: named geometry is not a partition")
    for key in ("observed_l2_norm", "reference_l2_norm", "difference_l2_norm"):
        expected_sq = sum(float(metric[key]) ** 2 for metric in named)
        require(close(float(aggregate[key]) ** 2, expected_sq, rel_tol=1e-9, abs_tol=1e-12), f"{context}: {key} energy mismatch")
    require(
        close(aggregate["maximum_absolute_error"], max(metric["maximum_absolute_error"] for metric in named), rel_tol=0, abs_tol=0),
        f"{context}: aggregate Linf mismatch",
    )
    require(aggregate["nonfinite_count"] == sum(metric["nonfinite_count"] for metric in named), f"{context}: nonfinite partition mismatch")


def selected_target_count(sequence_length: int, modulus: int, offset: int) -> int:
    return sum((position + offset) % modulus == 0 for position in range(1, sequence_length))


def first_step(steps: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float]) -> int | None:
    return next((step["step"] for step in steps if getter(step) != 0), None)


def temporal_rms(steps: list[dict[str, Any]], start: int, end: int) -> float:
    selected = steps[start - 1 : end]
    numerator = sum(step["aggregate_cumulative_displacement"]["difference_l2_norm"] ** 2 for step in selected)
    denominator = sum(step["aggregate_cumulative_displacement"]["reference_l2_norm"] ** 2 for step in selected)
    return math.sqrt(numerator / denominator) if denominator else 0.0


def balanced_checkpoint_exceedances(step: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate = step["aggregate_cumulative_displacement"]
    exceedances = []
    for name, row in step["per_parameter"].items():
        metric = row["cumulative_displacement"]
        floor = aggregate["reference_l2_norm"] * math.sqrt(metric["elements"] / TOTAL_ELEMENTS)
        denominator = max(metric["reference_l2_norm"], floor, EPSILON64)
        balanced = metric["difference_l2_norm"] / denominator
        if balanced > 0.01 or metric["cosine_similarity"] < 0.9999:
            exceedances.append(
                {
                    "parameter": name,
                    "balanced_relative_l2_error": balanced,
                    "ordinary_relative_l2_error": metric["relative_l2_error"],
                    "cosine_similarity": metric["cosine_similarity"],
                    "difference_l2_norm": metric["difference_l2_norm"],
                    "reference_l2_norm": metric["reference_l2_norm"],
                    "global_rms_allocation_floor_l2_norm": floor,
                }
            )
    return exceedances


def historical_envelope_failures(steps: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    identities: Counter[str] = Counter()
    for step in steps:
        for phase in ("preclip_gradient", "clipped_gradient"):
            aggregate = step[f"aggregate_{phase}"]
            if (
                aggregate["maximum_absolute_error"] > 0.01
                or aggregate["relative_l2_error"] > 0.01
                or aggregate["cosine_similarity"] < 0.9999
            ):
                counts["aggregate_gradient"] += 1
            for name, row in step["per_parameter"].items():
                metric = row[phase]
                floor = aggregate["reference_l2_norm"] * math.sqrt(metric["elements"] / TOTAL_ELEMENTS)
                balanced = metric["difference_l2_norm"] / max(metric["reference_l2_norm"], floor, EPSILON64)
                if metric["maximum_absolute_error"] > 0.01 or balanced > 0.01 or metric["cosine_similarity"] < 0.9999:
                    counts["named_gradient"] += 1
                    identities[f"{phase}:{name}"] += 1
        for phase in ("adam_exp_avg", "adam_exp_avg_sq", "cumulative_displacement", "post_step_parameter"):
            aggregate = step[f"aggregate_{phase}"]
            if aggregate["relative_l2_error"] > 0.01 or aggregate["cosine_similarity"] < 0.9999:
                counts[f"aggregate_{phase}"] += 1
            for name, row in step["per_parameter"].items():
                metric = row[phase]
                floor = aggregate["reference_l2_norm"] * math.sqrt(metric["elements"] / TOTAL_ELEMENTS)
                balanced = metric["difference_l2_norm"] / max(metric["reference_l2_norm"], floor, EPSILON64)
                if balanced > 0.01 or metric["cosine_similarity"] < 0.9999:
                    counts[f"named_{phase}"] += 1
                    identities[f"{phase}:{name}"] += 1
    return {"counts": dict(sorted(counts.items())), "identities": dict(sorted(identities.items()))}


def validate_run(run: dict[str, Any], expected: dict[str, Any], counters: Counter[str]) -> dict[str, Any]:
    variant = expected["variant"]
    trajectory_id = expected["trajectory_id"]
    context = f"{variant}/{trajectory_id}"
    require(run["status"] == "diagnostic_complete_no_gate", f"{context}: incomplete run")
    require(run["variant"] == variant, f"{context}: variant drift")
    require(run["trajectory_index"] == expected["trajectory_index"], f"{context}: trajectory index drift")
    validate_contract(run["trajectory_contract"], expected_id=trajectory_id)
    require(run["contract_origin"] == expected["origin"], f"{context}: contract-origin drift")
    require(run["parameter_geometry"] == list(GEOMETRY_BY_NAME.values()), f"{context}: parameter geometry drift")
    require(len(run["steps"]) == expected["steps"], f"{context}: step count drift")
    require(set(run["steps"][0]["per_parameter"]) == set(GEOMETRY_BY_NAME), f"{context}: named parameter set drift")
    autocast_enabled = variant != "fp32_liger_vs_dense_full"
    autocast_expected = {
        "device_type": "cuda",
        "enabled": autocast_enabled,
        "dtype": "torch.bfloat16" if autocast_enabled else None,
    }
    moduli = (2, 3, 5, 7)
    trajectory_index = expected["trajectory_index"]
    batch_seed_base = run["trajectory_contract"]["batch_seed_base"]
    for zero_index, step in enumerate(run["steps"]):
        step_context = f"{context}/step-{zero_index + 1}"
        require(step["step"] == zero_index + 1, f"{step_context}: step order drift")
        accounting = step["batch_accounting"]
        modulus = moduli[zero_index % len(moduli)]
        offset = (zero_index + trajectory_index) % modulus
        targets = selected_target_count(32, modulus, offset)
        extra = (zero_index * 3 + trajectory_index) % 13
        require(
            accounting
            == {
                "seed": batch_seed_base + zero_index,
                "sequence_length": 32,
                "supervision_modulus": modulus,
                "supervision_offset": offset,
                "supervised_targets": targets,
                "divisor_extra": extra,
                "global_divisor": targets + extra,
            },
            f"{step_context}: batch accounting drift",
        )
        require(step["autocast_contract"] == {"observed_training": autocast_expected, "reference_training": autocast_expected, "heldout": autocast_expected}, f"{step_context}: autocast drift")
        require(step["dtypes"] == DTYPES, f"{step_context}: dtype drift")
        validate_scalar(step["loss"], f"{step_context}/loss")
        validate_scalar(step["heldout"]["loss"], f"{step_context}/heldout-loss")
        validate_tensor(step["heldout"]["logits"], f"{step_context}/heldout-logits")
        require(step["heldout"]["supervised_targets"] == 7, f"{step_context}: heldout target-count drift")
        require(step["heldout"]["global_divisor"] == 12, f"{step_context}: heldout divisor drift")
        counters["scalar_metrics"] += 2
        counters["tensor_metrics"] += 1
        require(set(step["per_parameter"]) == set(GEOMETRY_BY_NAME), f"{step_context}: parameter-name drift")
        for name, geometry in GEOMETRY_BY_NAME.items():
            row = step["per_parameter"][name]
            require(row["elements"] == geometry["elements"], f"{step_context}/{name}: element-count drift")
            require(row["optimizer_step"] == {"observed": zero_index + 1, "reference": zero_index + 1}, f"{step_context}/{name}: optimizer counter drift")
        for aggregate_name, named_name in zip(AGGREGATE_FIELDS, TENSOR_FIELDS):
            aggregate = step[aggregate_name]
            validate_tensor(aggregate, f"{step_context}/{aggregate_name}")
            named = []
            for name in GEOMETRY_BY_NAME:
                metric = step["per_parameter"][name][named_name]
                validate_tensor(metric, f"{step_context}/{name}/{named_name}")
                require(metric["elements"] == GEOMETRY_BY_NAME[name]["elements"], f"{step_context}/{name}/{named_name}: metric geometry drift")
                named.append(metric)
            validate_partition(aggregate, named, f"{step_context}/{named_name}")
            counters["tensor_metrics"] += 1 + len(named)
            counters["partition_checks"] += 1
        counters["steps"] += 1
    steps = run["steps"]
    checkpoint_steps = (1, 2, 4, 8, 16, 32, 64, 96, 128)
    if len(steps) == 512:
        checkpoint_steps = (*checkpoint_steps, 192, 256, 384, 512)
    windows = [(1, len(steps))]
    if len(steps) == 512:
        windows = [(1, 512), (1, 128), (129, 256), (257, 512)]
    return {
        "variant": variant,
        "trajectory_id": trajectory_id,
        "steps": len(steps),
        "first_nonzero_difference_step": {
            "loss": first_step(steps, lambda step: step["loss"]["maximum_absolute_error"]),
            "aggregate_preclip_gradient": first_step(steps, lambda step: step["aggregate_preclip_gradient"]["difference_l2_norm"]),
            "aggregate_raw_update": first_step(steps, lambda step: step["aggregate_raw_update"]["difference_l2_norm"]),
            "aggregate_cumulative_displacement": first_step(steps, lambda step: step["aggregate_cumulative_displacement"]["difference_l2_norm"]),
            "aggregate_post_step_parameter": first_step(steps, lambda step: step["aggregate_post_step_parameter"]["difference_l2_norm"]),
            "heldout_logits": first_step(steps, lambda step: step["heldout"]["logits"]["difference_l2_norm"]),
        },
        "all_recorded_comparisons_bit_exact": all(
            step["loss"]["maximum_absolute_error"] == 0
            and step["heldout"]["loss"]["maximum_absolute_error"] == 0
            and step["heldout"]["logits"]["difference_l2_norm"] == 0
            and all(step[field]["difference_l2_norm"] == 0 for field in AGGREGATE_FIELDS)
            and all(step["per_parameter"][name][field]["difference_l2_norm"] == 0 for name in GEOMETRY_BY_NAME for field in TENSOR_FIELDS)
            for step in steps
        ),
        "temporal_rms_cumulative_relative_l2": [
            {"start": start, "end": end, "value": temporal_rms(steps, start, end)} for start, end in windows
        ],
        "maximum_cumulative_relative_l2": max(step["aggregate_cumulative_displacement"]["relative_l2_error"] for step in steps),
        "final_cumulative_relative_l2": steps[-1]["aggregate_cumulative_displacement"]["relative_l2_error"],
        "final_cumulative_difference_l2_norm": steps[-1]["aggregate_cumulative_displacement"]["difference_l2_norm"],
        "final_cumulative_reference_l2_norm": steps[-1]["aggregate_cumulative_displacement"]["reference_l2_norm"],
        "maximum_complete_state_relative_l2": max(step["aggregate_post_step_parameter"]["relative_l2_error"] for step in steps),
        "final_complete_state_relative_l2": steps[-1]["aggregate_post_step_parameter"]["relative_l2_error"],
        "maximum_heldout_logit_relative_l2": max(step["heldout"]["logits"]["relative_l2_error"] for step in steps),
        "final_heldout_logit_relative_l2": steps[-1]["heldout"]["logits"]["relative_l2_error"],
        "maximum_heldout_logit_absolute_error": max(step["heldout"]["logits"]["maximum_absolute_error"] for step in steps),
        "nonfinite_count": sum(
            step[field]["nonfinite_count"] for step in steps for field in AGGREGATE_FIELDS
        )
        + sum(step["heldout"]["logits"]["nonfinite_count"] for step in steps),
        "historical_r16_envelope_failures": historical_envelope_failures(steps),
        "balanced_cumulative_checkpoint_exceedances": [
            {
                "step": checkpoint,
                "count": len(exceedances),
                "exceedances": exceedances,
            }
            for checkpoint in checkpoint_steps
            for exceedances in [balanced_checkpoint_exceedances(steps[checkpoint - 1])]
        ],
    }


def validate_producer_summaries(report: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    require(len(report["summaries"]) == len(summaries), "producer summary cardinality drift")
    producer = {(row["variant"], row["trajectory_id"]): row for row in report["summaries"]}
    for summary in summaries:
        key = (summary["variant"], summary["trajectory_id"])
        require(key in producer, f"missing producer summary {key}")
        row = producer[key]
        require(row["steps"] == summary["steps"], f"{key}: producer step summary drift")
        require(row["all_recorded_comparisons_bit_exact"] == summary["all_recorded_comparisons_bit_exact"], f"{key}: producer bit-exact summary drift")
        mappings = {
            "maximum_cumulative_relative_l2": "maximum_cumulative_relative_l2",
            "final_cumulative_relative_l2": "final_cumulative_relative_l2",
            "final_cumulative_difference_l2_norm": "final_cumulative_difference_l2_norm",
            "final_cumulative_reference_l2_norm": "final_cumulative_reference_l2_norm",
            "maximum_post_step_parameter_relative_l2": "maximum_complete_state_relative_l2",
            "final_post_step_parameter_relative_l2": "final_complete_state_relative_l2",
            "maximum_heldout_logit_relative_l2": "maximum_heldout_logit_relative_l2",
            "final_heldout_logit_relative_l2": "final_heldout_logit_relative_l2",
            "maximum_heldout_logit_absolute_error": "maximum_heldout_logit_absolute_error",
        }
        for producer_key, independent_key in mappings.items():
            require(close(row[producer_key], summary[independent_key], rel_tol=1e-12, abs_tol=1e-15), f"{key}: producer {producer_key} drift")
        require(row["nonfinite_count"] == summary["nonfinite_count"], f"{key}: producer nonfinite summary drift")
        producer_checkpoints = {checkpoint["step"]: checkpoint for checkpoint in row["fixed_checkpoints"]}
        comparison = []
        for checkpoint in summary["balanced_cumulative_checkpoint_exceedances"]:
            require(checkpoint["step"] in producer_checkpoints, f"{key}: producer checkpoint missing")
            producer_count = producer_checkpoints[checkpoint["step"]][
                "named_cumulative_exceedance_count_under_historical_envelope"
            ]
            comparison.append(
                {
                    "step": checkpoint["step"],
                    "producer_reported_count": producer_count,
                    "independent_balanced_count": checkpoint["count"],
                    "counts_match": producer_count == checkpoint["count"],
                }
            )
        summary["producer_checkpoint_count_comparison"] = comparison


def main() -> None:
    args = parse_args()
    report_sha256 = sha256_file(args.report)
    require(report_sha256 == EXPECTED_REPORT_SHA256, "unexpected diagnostic report bytes")
    report = json.loads(args.report.read_text())
    require(report["artifact"] == "qwen35_r16_h2_postfailure_long_horizon_diagnostic", "artifact identity drift")
    require(report["schema_version"] == 1, "schema version drift")
    require(report["status"] == "diagnostic_complete_no_gate", "diagnostic did not complete")
    require(report["successor_gate_authorized"] is False, "diagnostic improperly authorizes a successor gate")
    require(report["scientific_training_authorized"] is False, "diagnostic improperly authorizes training")
    require(report["qualification_manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "manifest identity drift")
    require(report["r16_failed_report_sha256"] == EXPECTED_PARENT_SHA256, "parent identity drift")
    require(report["source_commit"] == EXPECTED_SOURCE_COMMIT, "source commit drift")
    require(report["diagnostic_source_sha256"] == EXPECTED_PRODUCER_SHA256, "producer source-byte drift")
    require(report["planned_runs"] == report["complete_runs"] == 9, "run cardinality drift")
    require(report["cuda_device"] == "NVIDIA A100-SXM-64GB", "GPU identity drift")
    r16_contracts = report["r16_trajectory_contracts"]
    long_contracts = report["outcome_unseen_long_contracts"]
    require([row["trajectory_id"] for row in r16_contracts] == ["R16-T0", "R16-T1", "R16-T2"], "R16 contract order drift")
    require([row["trajectory_id"] for row in long_contracts] == ["R16-PF-L0", "R16-PF-L1", "R16-PF-L2"], "long contract order drift")
    for contract in r16_contracts + long_contracts:
        validate_contract(contract, expected_id=contract["trajectory_id"])
    labels = [contract[key] for contract in r16_contracts + long_contracts for key in ("model_seed_label", "batch_seed_label", "heldout_seed_label")]
    require(len(labels) == len(set(labels)), "seed labels are not disjoint")
    plan = []
    for variant, contracts, origin, steps in (
        ("bf16_dense_selected_vs_dense_full", r16_contracts, "r16_outcome_known", 128),
        ("fp32_liger_vs_dense_full", r16_contracts, "r16_outcome_known", 128),
        ("bf16_liger_vs_dense_full", long_contracts, "outcome_unseen_long_diagnostic", 512),
    ):
        for index, contract in enumerate(contracts):
            plan.append(
                {
                    "variant": variant,
                    "trajectory_id": contract["trajectory_id"],
                    "trajectory_index": index,
                    "origin": origin,
                    "steps": steps,
                }
            )
    require(len(report["runs"]) == len(plan), "run list length drift")
    counters: Counter[str] = Counter()
    summaries = [validate_run(run, expected, counters) for run, expected in zip(report["runs"], plan)]
    validate_producer_summaries(report, summaries)
    checkpoint_count_mismatches = [
        {
            "variant": summary["variant"],
            "trajectory_id": summary["trajectory_id"],
            **comparison,
        }
        for summary in summaries
        for comparison in summary["producer_checkpoint_count_comparison"]
        if not comparison["counts_match"]
    ]
    output = {
        "artifact": "qwen35_r16_postfailure_diagnostic_independent_validation",
        "schema_version": 1,
        "status": "evidence_validated_with_producer_summary_defect_no_gate",
        "report_path": str(args.report.resolve()),
        "report_sha256": report_sha256,
        "validation_scope": {
            "producer_status_not_trusted": True,
            "standard_library_only": True,
            "metric_arithmetic_recomputed_from_serialized_primitives": True,
            "named_aggregate_energy_partitions_recomputed": True,
            "batch_seed_and_supervision_accounting_recomputed": True,
            "producer_summaries_recomputed": True,
            "raw_tensors_unavailable": True,
        },
        "validated_counts": dict(sorted(counters.items())),
        "producer_summary_defects": {
            "count": len(checkpoint_count_mismatches),
            "fixed_checkpoint_count_mismatches": checkpoint_count_mismatches,
            "cause": "The producer checkpoint summary applied ordinary named relative L2, while the R16 historical envelope uses the conditioning-resistant balanced named relative L2. Raw per-step evidence is unaffected and the independent counts are authoritative.",
        },
        "summaries": summaries,
        "allowed_conclusion": "The raw diagnostic evidence is internally consistent. The producer's fixed-checkpoint named-exceedance counts contain a non-gating summary-only metric-selection defect; use the independently recomputed balanced counts. R16 remains failed and this artifact has no gate or training authority.",
        "successor_gate_authorized": False,
        "scientific_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": output["status"], "validated_counts": output["validated_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
