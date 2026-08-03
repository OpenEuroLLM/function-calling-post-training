"""Fail-closed R17 matched-reference H2 contracts.

R17 resolves a hash-bound overlay over immutable R16.  It changes the primary
finite-precision reference from full-dense to dense-selected, retains the
full-dense comparison as mandatory non-gating evidence, and preserves every
R16 numerical acceptance value.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from open_instruct import qwen35_qualification as r15
from open_instruct import qwen35_qualification_r16 as r16

QUALIFICATION_PROTOCOL_ID = "qwen35-hardware-qualification-r17"
BASE_PROTOCOL_ID = "qwen35-hardware-qualification-r16"
BASE_MANIFEST_SHA256 = "827da32eefdf20839fef364b1bed23afb37122e0c19a981e460324c9d5c1b4f8"
CORRECTIVE_BASELINE_COMMIT = "7ec97195b2397bcb90b160d417b8aa08a36cb4f2"


def _seed_identity(label: str) -> dict[str, str | int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return {"seed_label": label, "seed_sha256": digest, "seed": int(digest[:8], 16)}


def _expected_direct_cases() -> list[dict[str, Any]]:
    geometries = (
        ("R17-D0", 73, [0, 2, 11, 36, 72], 37, 0.03125),
        ("R17-D1", 144, [1, 7, 31, 63, 95, 127, 142, 143], 113, 1.0),
        ("R17-D2", 257, [0, 64, 128, 192, 255, 256], 263, 8.0),
    )
    return [
        {
            "case_id": case_id,
            **_seed_identity(f"qwen35-hardware-qualification-r17-h2-direct-{index}"),
            "rows": rows,
            "supervision_kind": "explicit",
            "supervised_rows": supervised,
            "expected_supervised_count": len(supervised),
            "global_divisor": divisor,
            "hidden_scale": scale,
            "weight_standard_deviation": 0.02,
        }
        for index, (case_id, rows, supervised, divisor, scale) in enumerate(geometries)
    ]


def _expected_trajectories() -> list[dict[str, Any]]:
    trajectories = []
    for index in range(3):
        model_label = f"qwen35-hardware-qualification-r17-h2-trajectory-{index}"
        model = _seed_identity(model_label)
        batches = _seed_identity(f"{model_label}-batches")
        heldout = _seed_identity(f"{model_label}-heldout")
        trajectories.append(
            {
                "trajectory_id": f"R17-T{index}",
                "model_seed_label": model["seed_label"],
                "model_seed_sha256": model["seed_sha256"],
                "model_seed": model["seed"],
                "batch_seed_label": batches["seed_label"],
                "batch_seed_sha256": batches["seed_sha256"],
                "batch_seed_base": batches["seed"],
                "heldout_seed_label": heldout["seed_label"],
                "heldout_seed_sha256": heldout["seed_sha256"],
                "heldout_seed": heldout["seed"],
            }
        )
    return trajectories


def load_qualification_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    overlay = json.loads(raw)
    expected_top = {
        "schema_version",
        "protocol_id",
        "protocol_date",
        "status",
        "base_manifest",
        "transformations",
        "overrides",
    }
    if set(overlay) != expected_top:
        raise ValueError("R17 overlay top-level field set drift")
    if (
        overlay["schema_version"] != 1
        or overlay["protocol_id"] != QUALIFICATION_PROTOCOL_ID
        or overlay["protocol_date"] != "2026-07-19"
        or overlay["status"] != "ready_for_execution"
    ):
        raise ValueError("R17 overlay identity/status drift")
    expected_base = {
        "path": "qwen35_hardware_qualification_r16.json",
        "sha256": BASE_MANIFEST_SHA256,
        "protocol_id": BASE_PROTOCOL_ID,
    }
    if overlay["base_manifest"] != expected_base:
        raise ValueError("R17 base-manifest binding drift")
    base_path = path.parent / expected_base["path"]
    if r15.sha256_file(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError("R17 immutable R16 base-manifest bytes drift")
    base, base_digest = r16.load_qualification_manifest(base_path)
    if base_digest != BASE_MANIFEST_SHA256 or base["protocol_id"] != BASE_PROTOCOL_ID:
        raise ValueError("R17 base manifest did not independently validate as R16")

    expected_transformations = {
        "change_primary_reference_from_full_dense_to_dense_selected": True,
        "retain_full_dense_as_mandatory_nongating_diagnostic": True,
        "retain_all_r16_numerical_acceptance_values": True,
        "retain_r16_historical_direct_cases": True,
    }
    if overlay["transformations"] != expected_transformations:
        raise ValueError("R17 transformation contract drift")
    overrides = overlay["overrides"]
    if not isinstance(overrides, dict) or set(overrides) != {
        "protocol_id",
        "protocol_date",
        "source",
        "h2_acceptance",
    }:
        raise ValueError("R17 override scope drift")
    if overrides["protocol_id"] != QUALIFICATION_PROTOCOL_ID or overrides["protocol_date"] != "2026-07-19":
        raise ValueError("R17 override identity drift")
    if overrides["source"] != {"corrective_baseline_commit": CORRECTIVE_BASELINE_COMMIT}:
        raise ValueError("R17 source override drift")
    expected_h2_keys = {
        "protocol_revision",
        "confirmatory_direct_cases",
        "confirmatory_trajectories",
        "trajectory_steps",
        "primary_observed_path",
        "primary_reference_path",
        "mandatory_diagnostic_observed_path",
        "mandatory_diagnostic_reference_path",
        "full_dense_diagnostic_numerical_discrepancy_is_gating",
        "full_dense_diagnostic_integrity_and_finiteness_are_mandatory",
        "liger_numerical_failure_policy",
        "r16_failed_criteria_reclassified_as_pass",
    }
    if not isinstance(overrides["h2_acceptance"], dict) or set(overrides["h2_acceptance"]) != expected_h2_keys:
        raise ValueError("R17 H2 override field set drift")

    effective = r16._deep_merge(base, overrides)
    effective["manifest_derivation"] = {
        "kind": "sha256_bound_overlay",
        "base_manifest": copy.deepcopy(expected_base),
        "transformations": copy.deepcopy(expected_transformations),
    }
    if effective["protocol_id"] != QUALIFICATION_PROTOCOL_ID or effective["protocol_date"] != "2026-07-19":
        raise ValueError("R17 effective identity drift")
    if effective["scope"]["slurm_account"] != "aifac_f02_434":
        raise ValueError("R17 does not require the personal Slurm account")
    if effective["scope"]["automatic_scientific_training"] is not False:
        raise ValueError("R17 may not authorize automatic scientific training")
    if effective["scope"]["eligible_arm_ids"] != ["C00"]:
        raise ValueError("R17 scope drifted beyond C00")
    if effective["scope"]["forbidden_evaluations"] != ["BFCL", "tau2"]:
        raise ValueError("R17 forbidden-evaluation contract drift")
    if effective["source"]["corrective_baseline_commit"] != CORRECTIVE_BASELINE_COMMIT:
        raise ValueError("R17 corrective baseline drift")
    if effective["numerical_acceptance"] != base["numerical_acceptance"]:
        raise ValueError("R17 changed an R16 numerical threshold")
    if effective["runtime_pins"] != base["runtime_pins"]:
        raise ValueError("R17 runtime pin drift")
    if effective["model"] != base["model"] or effective["h1_acceptance"] != base["h1_acceptance"]:
        raise ValueError("R17 changed model or H1")
    if effective["h2_acceptance"]["historical_direct_cases"] != base["h2_acceptance"]["historical_direct_cases"]:
        raise ValueError("R17 historical direct-case lineage drift")

    h2 = effective["h2_acceptance"]
    if h2["confirmatory_direct_cases"] != _expected_direct_cases():
        raise ValueError("R17 confirmatory direct-case/seed drift")
    if h2["confirmatory_trajectories"] != _expected_trajectories():
        raise ValueError("R17 trajectory/seed drift")
    expected_scalars = {
        "protocol_revision": 4,
        "trajectory_steps": 512,
        "primary_observed_path": "liger_fused_selected_rows",
        "primary_reference_path": "pytorch_dense_selected_rows",
        "mandatory_diagnostic_observed_path": "pytorch_dense_selected_rows",
        "mandatory_diagnostic_reference_path": "pytorch_dense_full_rows_ignore_index",
        "full_dense_diagnostic_numerical_discrepancy_is_gating": False,
        "full_dense_diagnostic_integrity_and_finiteness_are_mandatory": True,
        "liger_numerical_failure_policy": "abandon_liger_no_outcome_fitted_threshold_rescue",
        "r16_failed_criteria_reclassified_as_pass": False,
    }
    for key, expected in expected_scalars.items():
        if h2.get(key) != expected:
            raise ValueError(f"R17 H2 scalar contract drift for {key}")
    for inherited_key in (
        "direct_hidden_size",
        "direct_vocab_size",
        "direct_heldout_rows",
        "trajectory_model_config",
        "trajectory_parameter_geometry",
        "trajectory_parameter_count",
        "trajectory_sequence_length",
        "trajectory_supervision_moduli",
        "trajectory_divisor_extra_modulus",
        "trajectory_divisor_extra_multiplier",
        "trajectory_heldout_supervision_modulus",
        "trajectory_heldout_divisor_extra",
        "raw_updates_are_diagnostic",
        "named_relative_metric",
        "named_relative_formula",
        "named_relative_threshold",
        "named_minimum_cosine_similarity",
        "named_gradient_maximum_absolute_error",
        "optimizer_moments_are_gating",
        "cumulative_parameter_displacement_is_gating",
        "named_post_step_parameter_state_is_gating",
    ):
        if h2[inherited_key] != base["h2_acceptance"][inherited_key]:
            raise ValueError(f"R17 inherited H2 contract drift for {inherited_key}")
    prior_labels = {
        case["seed_label"]
        for case in base["h2_acceptance"]["historical_direct_cases"]
        + base["h2_acceptance"]["confirmatory_direct_cases"]
        if "seed_label" in case
    }
    prior_labels.update(
        contract[key]
        for contract in base["h2_acceptance"]["confirmatory_trajectories"]
        for key in ("model_seed_label", "batch_seed_label", "heldout_seed_label")
    )
    new_labels = {case["seed_label"] for case in h2["confirmatory_direct_cases"]}
    new_labels.update(
        contract[key]
        for contract in h2["confirmatory_trajectories"]
        for key in ("model_seed_label", "batch_seed_label", "heldout_seed_label")
    )
    if len(new_labels) != 12 or new_labels & prior_labels:
        raise ValueError("R17 outcome-unseen seed labels are duplicate or overlap predecessors")
    return effective, hashlib.sha256(raw).hexdigest()


collect_h2_numerical_decisions = r16.collect_h2_numerical_decisions
balanced_tensor_comparison_metrics = r16.balanced_tensor_comparison_metrics
scalar_comparison_metrics = r16.scalar_comparison_metrics
tensor_comparison_metrics = r16.tensor_comparison_metrics


DIAGNOSTIC_AGGREGATE_FIELDS = {
    "preclip_gradient_comparison": "aggregate_preclip_gradient_comparison",
    "clipped_gradient_comparison": "aggregate_clipped_gradient_comparison",
    "raw_adamw_update_comparison_diagnostic": "aggregate_raw_adamw_update_comparison_diagnostic",
    "optimizer_exp_avg_comparison": "aggregate_optimizer_exp_avg_comparison",
    "optimizer_exp_avg_sq_comparison": "aggregate_optimizer_exp_avg_sq_comparison",
    "cumulative_parameter_displacement_comparison": "aggregate_cumulative_parameter_displacement_comparison",
    "post_step_parameter_state_comparison": "aggregate_post_step_parameter_state_comparison",
}


def _require_finite_scalar(value: Any, *, context: str) -> dict[str, Any]:
    metric = r16._require_scalar_metric(value, context=context)
    if metric["nonfinite_count"] != 0:
        raise ValueError(f"{context}: nonfinite diagnostic scalar")
    return metric


def _require_finite_tensor(value: Any, *, expected_elements: int, context: str) -> dict[str, Any]:
    metric = r16._require_tensor_metric(value, expected_elements=expected_elements, context=context)
    if metric["nonfinite_count"] != 0:
        raise ValueError(f"{context}: nonfinite diagnostic tensor")
    return metric


def _validate_direct_diagnostic(
    case: dict[str, Any], contract: dict[str, Any], h2: dict[str, Any], context: str
) -> int:
    diagnostic = case.get("full_dense_diagnostic")
    expected_fields = {
        "observed_loss",
        "reference_loss",
        "loss_comparison",
        "selected_hidden_gradient_comparison",
        "output_head_gradient_comparison",
        "raw_first_adamw_update_comparison_diagnostic",
        "optimizer_exp_avg_comparison",
        "optimizer_exp_avg_sq_comparison",
        "post_step_parameter_comparison",
        "heldout",
        "observed_path",
        "reference_path",
        "numerical_discrepancy_is_gating",
        "integrity_and_finiteness_are_mandatory",
        "autocast_contract",
        "optimizer_step_counters",
        "ignored_full_hidden_gradient_nonzero_count",
    }
    if not isinstance(diagnostic, dict) or set(diagnostic) != expected_fields:
        raise ValueError(f"{context}: direct diagnostic field coverage drift")
    if (
        diagnostic["observed_path"] != h2["mandatory_diagnostic_observed_path"]
        or diagnostic["reference_path"] != h2["mandatory_diagnostic_reference_path"]
        or diagnostic["numerical_discrepancy_is_gating"] is not False
        or diagnostic["integrity_and_finiteness_are_mandatory"] is not True
    ):
        raise ValueError(f"{context}: direct diagnostic role drift")
    if set(diagnostic["autocast_contract"]) != {"dense_selected", "dense_full"}:
        raise ValueError(f"{context}: direct diagnostic autocast coverage drift")
    for role, value in diagnostic["autocast_contract"].items():
        r16._require_autocast(value, context=f"{context} {role}")
    if diagnostic["optimizer_step_counters"] != {"dense_selected": [1], "dense_full": [1]}:
        raise ValueError(f"{context}: direct diagnostic optimizer counter drift")
    if diagnostic["ignored_full_hidden_gradient_nonzero_count"] != 0:
        raise ValueError(f"{context}: ignored full hidden rows received gradient")
    loss = _require_finite_scalar(diagnostic["loss_comparison"], context=f"{context} loss")
    if loss["observed"] != diagnostic["observed_loss"] or loss["reference"] != diagnostic["reference_loss"]:
        raise ValueError(f"{context}: direct diagnostic loss binding drift")
    hidden_elements = contract["expected_supervised_count"] * h2["direct_hidden_size"]
    head_elements = h2["direct_vocab_size"] * h2["direct_hidden_size"]
    _require_finite_tensor(
        diagnostic["selected_hidden_gradient_comparison"],
        expected_elements=hidden_elements,
        context=f"{context} selected-hidden gradient",
    )
    for field in (
        "output_head_gradient_comparison",
        "raw_first_adamw_update_comparison_diagnostic",
        "optimizer_exp_avg_comparison",
        "optimizer_exp_avg_sq_comparison",
        "post_step_parameter_comparison",
    ):
        _require_finite_tensor(diagnostic[field], expected_elements=head_elements, context=f"{context} {field}")
    heldout = diagnostic["heldout"]
    if not isinstance(heldout, dict) or set(heldout) != {
        "logit_comparison",
        "observed_loss",
        "reference_loss",
        "loss_comparison",
    }:
        raise ValueError(f"{context}: direct diagnostic heldout coverage drift")
    _require_finite_tensor(
        heldout["logit_comparison"],
        expected_elements=h2["direct_heldout_rows"] * h2["direct_vocab_size"],
        context=f"{context} heldout logits",
    )
    heldout_loss = _require_finite_scalar(heldout["loss_comparison"], context=f"{context} heldout loss")
    if heldout_loss["observed"] != heldout["observed_loss"] or heldout_loss["reference"] != heldout["reference_loss"]:
        raise ValueError(f"{context}: direct diagnostic heldout loss binding drift")
    return 9


def _validate_named_diagnostic_partition(
    *,
    diagnostic: dict[str, Any],
    names: list[str],
    elements_by_name: dict[str, int],
    parameter_count: int,
    context: str,
) -> int:
    per_parameter = diagnostic.get("per_parameter_comparisons")
    if not isinstance(per_parameter, dict) or set(per_parameter) != set(names):
        raise ValueError(f"{context}: diagnostic named coverage drift")
    partitions: dict[str, list[dict[str, Any]]] = {field: [] for field in DIAGNOSTIC_AGGREGATE_FIELDS}
    checks = 0
    for name in names:
        named = per_parameter[name]
        expected_elements = elements_by_name[name]
        if not isinstance(named, dict) or set(named) != {"elements", *DIAGNOSTIC_AGGREGATE_FIELDS}:
            raise ValueError(f"{context} {name}: diagnostic named fields drift")
        if named["elements"] != expected_elements:
            raise ValueError(f"{context} {name}: diagnostic named geometry drift")
        for named_field, aggregate_field in DIAGNOSTIC_AGGREGATE_FIELDS.items():
            aggregate = diagnostic[aggregate_field]
            r16.validate_balanced_metric_arithmetic(
                named[named_field],
                aggregate,
                expected_elements=expected_elements,
                aggregate_elements=parameter_count,
                context=f"{context} {name} {named_field}",
            )
            if named[named_field]["nonfinite_count"] != 0:
                raise ValueError(f"{context} {name} {named_field}: nonfinite diagnostic tensor")
            partitions[named_field].append(named[named_field])
            checks += 1
    for named_field, aggregate_field in DIAGNOSTIC_AGGREGATE_FIELDS.items():
        rows = partitions[named_field]
        aggregate = diagnostic[aggregate_field]
        if aggregate["nonfinite_count"] != 0:
            raise ValueError(f"{context} {named_field}: nonfinite diagnostic aggregate")
        for norm_key in ("observed_l2_norm", "reference_l2_norm", "difference_l2_norm"):
            partition_norm = math.sqrt(sum(float(row[norm_key]) ** 2 for row in rows))
            if not r16._energy_close(partition_norm, aggregate[norm_key]):
                raise ValueError(f"{context} {named_field}: named/aggregate {norm_key} energy drift")
        if not r16._energy_close(
            max(float(row["maximum_absolute_error"]) for row in rows), aggregate["maximum_absolute_error"]
        ):
            raise ValueError(f"{context} {named_field}: named/aggregate maximum-error drift")
    return checks


def _validate_step_diagnostic(
    *,
    step: dict[str, Any],
    step_number: int,
    names: list[str],
    elements_by_name: dict[str, int],
    parameter_count: int,
    h2: dict[str, Any],
    context: str,
) -> int:
    diagnostic = step.get("full_dense_diagnostic")
    expected_fields = {
        "observed_path",
        "reference_path",
        "numerical_discrepancy_is_gating",
        "integrity_and_finiteness_are_mandatory",
        "observed_loss",
        "reference_loss",
        "training_loss_comparison",
        *DIAGNOSTIC_AGGREGATE_FIELDS.values(),
        "per_parameter_comparisons",
        "preclip_gradient_norms",
        "optimizer_floating_state_dtypes",
        "optimizer_step_counters",
        "gradient_dtypes",
        "heldout",
    }
    if not isinstance(diagnostic, dict) or set(diagnostic) != expected_fields:
        raise ValueError(f"{context}: step diagnostic field coverage drift")
    if (
        diagnostic["observed_path"] != h2["mandatory_diagnostic_observed_path"]
        or diagnostic["reference_path"] != h2["mandatory_diagnostic_reference_path"]
        or diagnostic["numerical_discrepancy_is_gating"] is not False
        or diagnostic["integrity_and_finiteness_are_mandatory"] is not True
    ):
        raise ValueError(f"{context}: step diagnostic role drift")
    loss = _require_finite_scalar(diagnostic["training_loss_comparison"], context=f"{context} loss")
    if loss["observed"] != diagnostic["observed_loss"] or loss["reference"] != diagnostic["reference_loss"]:
        raise ValueError(f"{context}: step diagnostic loss binding drift")
    for field in DIAGNOSTIC_AGGREGATE_FIELDS.values():
        _require_finite_tensor(diagnostic[field], expected_elements=parameter_count, context=f"{context} {field}")
    if set(diagnostic["preclip_gradient_norms"]) != {"dense_selected", "dense_full"}:
        raise ValueError(f"{context}: step diagnostic preclip coverage drift")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0
        for value in diagnostic["preclip_gradient_norms"].values()
    ):
        raise ValueError(f"{context}: invalid diagnostic preclip norm")
    if diagnostic["optimizer_floating_state_dtypes"] != {
        "dense_selected": ["torch.float32"],
        "dense_full": ["torch.float32"],
    }:
        raise ValueError(f"{context}: diagnostic optimizer dtype drift")
    if diagnostic["optimizer_step_counters"] != {"dense_selected": [step_number], "dense_full": [step_number]}:
        raise ValueError(f"{context}: diagnostic optimizer counter drift")
    if diagnostic["gradient_dtypes"] != {"dense_selected": ["torch.float32"], "dense_full": ["torch.float32"]}:
        raise ValueError(f"{context}: diagnostic gradient dtype drift")
    named_checks = _validate_named_diagnostic_partition(
        diagnostic=diagnostic,
        names=names,
        elements_by_name=elements_by_name,
        parameter_count=parameter_count,
        context=context,
    )
    heldout = diagnostic["heldout"]
    if not isinstance(heldout, dict) or set(heldout) != {
        "supervised_targets",
        "global_divisor",
        "logit_comparison",
        "observed_loss",
        "reference_loss",
        "loss_comparison",
    }:
        raise ValueError(f"{context}: diagnostic heldout coverage drift")
    expected_heldout_targets = sum(
        position % h2["trajectory_heldout_supervision_modulus"] == 0
        for position in range(1, h2["trajectory_sequence_length"])
    )
    if (
        heldout["supervised_targets"] != expected_heldout_targets
        or heldout["global_divisor"] != expected_heldout_targets + h2["trajectory_heldout_divisor_extra"]
    ):
        raise ValueError(f"{context}: diagnostic heldout accounting drift")
    _require_finite_tensor(
        heldout["logit_comparison"],
        expected_elements=h2["trajectory_sequence_length"] * h2["trajectory_model_config"]["vocab_size"],
        context=f"{context} heldout logits",
    )
    heldout_loss = _require_finite_scalar(heldout["loss_comparison"], context=f"{context} heldout loss")
    if heldout_loss["observed"] != heldout["observed_loss"] or heldout_loss["reference"] != heldout["reference_loss"]:
        raise ValueError(f"{context}: diagnostic heldout loss binding drift")
    return named_checks


def validate_h2_liger_report(
    report: dict[str, Any],
    *,
    qualification: dict[str, Any],
    expected_manifest_sha256: str,
    require_numerical_pass: bool = True,
) -> dict[str, Any]:
    """Independently validate R17 primary and mandatory diagnostic evidence."""

    if report.get("artifact") != "qwen35_selective_liger_matched_reference_qualification_r17":
        raise ValueError("R17 report artifact identity drift")
    if report.get("schema_version") != 4 or report.get("qualification_protocol_id") != QUALIFICATION_PROTOCOL_ID:
        raise ValueError("R17 report schema/protocol drift")
    if report.get("qualification_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("R17 report manifest digest drift")
    if report.get("manifest_derivation") != qualification["manifest_derivation"]:
        raise ValueError("R17 report manifest derivation drift")
    h2 = qualification["h2_acceptance"]
    if report.get("primary_comparison") != {
        "observed_path": h2["primary_observed_path"],
        "reference_path": h2["primary_reference_path"],
        "numerical_discrepancy_is_gating": True,
    }:
        raise ValueError("R17 primary comparison identity drift")
    if report.get("mandatory_diagnostic_comparison") != {
        "observed_path": h2["mandatory_diagnostic_observed_path"],
        "reference_path": h2["mandatory_diagnostic_reference_path"],
        "numerical_discrepancy_is_gating": False,
        "integrity_and_finiteness_are_mandatory": True,
    }:
        raise ValueError("R17 diagnostic comparison identity drift")
    if report.get("mandatory_diagnostic_nonfinite_count") != 0:
        raise ValueError("R17 mandatory diagnostic contains nonfinite evidence")
    report_status = report.get("status")
    if report_status not in {"passed", "failed"}:
        raise ValueError("R17 report status drift")
    if require_numerical_pass and report_status != "passed":
        raise ValueError("R17 H2 report did not pass")

    primary_view = copy.deepcopy(report)
    primary_view["artifact"] = "qwen35_selective_liger_downstream_qualification_r16"
    primary_view["schema_version"] = 3
    primary_view["qualification_protocol_id"] = r16.QUALIFICATION_PROTOCOL_ID
    primary_view["status"] = primary_view["decision"]["status"]
    primary_view["successor_gate_authorized"] = primary_view["status"] == "passed"
    primary_validation = r16.validate_h2_liger_report(
        primary_view,
        qualification=qualification,
        expected_manifest_sha256=expected_manifest_sha256,
        require_numerical_pass=require_numerical_pass,
    )

    zero = report.get("zero_target_matched_reference")
    if not isinstance(zero, dict) or set(zero) != {
        "observed_path",
        "reference_path",
        "observed_loss",
        "reference_loss",
        "loss_comparison",
        "hidden_gradient_comparison",
        "output_weight_gradient_comparison",
        "autocast_contract",
        "graph_connected",
    }:
        raise ValueError("R17 matched zero-target evidence coverage drift")
    r16._require_autocast(zero["autocast_contract"], context="R17 matched zero target")
    zero_loss = _require_finite_scalar(zero["loss_comparison"], context="R17 matched zero-target loss")
    zero_hidden = _require_finite_tensor(
        zero["hidden_gradient_comparison"],
        expected_elements=h2["direct_hidden_size"],
        context="R17 matched zero-target hidden gradient",
    )
    zero_weight = _require_finite_tensor(
        zero["output_weight_gradient_comparison"],
        expected_elements=h2["direct_hidden_size"] * h2["direct_vocab_size"],
        context="R17 matched zero-target output-weight gradient",
    )
    if (
        zero["observed_loss"] != zero["reference_loss"]
        or zero_loss["maximum_absolute_error"] != 0
        or zero_hidden["difference_l2_norm"] != 0
        or zero_weight["difference_l2_norm"] != 0
        or set(zero["graph_connected"].values()) != {True}
    ):
        raise ValueError("R17 matched zero-target exact connected-zero contract drift")

    direct_checks = 0
    for section in ("historical_direct_cases", "confirmatory_direct_cases"):
        contracts = h2[section]
        for index, (case, contract) in enumerate(zip(report[section], contracts, strict=True)):
            direct_checks += _validate_direct_diagnostic(case, contract, h2, f"{section} {index}")

    geometry = h2["trajectory_parameter_geometry"]
    names = [row["name"] for row in geometry]
    elements_by_name = {row["name"]: row["elements"] for row in geometry}
    parameter_count = h2["trajectory_parameter_count"]
    diagnostic_steps = 0
    diagnostic_named_checks = 0
    for trajectory_index, trajectory in enumerate(report["confirmatory_trajectories"]):
        if trajectory.get("full_dense_forward_module_diagnostic") != trajectory.get("dense_forward_module"):
            raise ValueError(f"trajectory {trajectory_index}: full/selected dense forward module drift")
        if trajectory.get("full_dense_parameter_dtypes_diagnostic") != ["torch.float32"]:
            raise ValueError(f"trajectory {trajectory_index}: full-dense parameter dtype drift")
        for step_number, step in enumerate(trajectory["steps"], start=1):
            diagnostic_named_checks += _validate_step_diagnostic(
                step=step,
                step_number=step_number,
                names=names,
                elements_by_name=elements_by_name,
                parameter_count=parameter_count,
                h2=h2,
                context=f"{trajectory['trajectory_contract']['trajectory_id']} step {step_number}",
            )
            diagnostic_steps += 1
    expected_status = "passed" if report["decision"]["status"] == "passed" else "failed"
    if report_status != expected_status or report.get("successor_gate_authorized") is not (report_status == "passed"):
        raise ValueError("R17 overall status/primary decision disagreement")
    return {
        "status": "passed" if require_numerical_pass else "evidence_validated",
        "numerical_status": report_status,
        "primary_validation": primary_validation,
        "diagnostic_direct_metric_groups": direct_checks,
        "diagnostic_trajectory_steps": diagnostic_steps,
        "diagnostic_named_tensor_checks": diagnostic_named_checks,
        "matched_zero_target_sentinels": 1,
    }
