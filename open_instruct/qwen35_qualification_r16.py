"""Fail-closed R16 H2 contracts for Qwen3.5 selective-Liger qualification.

R15 remains implemented in :mod:`open_instruct.qwen35_qualification`.  This
module deliberately adds a new protocol instead of mutating or rescoring the
R15 contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from open_instruct import qwen35_qualification as r15

QUALIFICATION_PROTOCOL_ID = "qwen35-hardware-qualification-r16"
BASE_PROTOCOL_ID = "qwen35-hardware-qualification-r15"
BASE_MANIFEST_SHA256 = "bff52a9223d07cdf047bfe25dbcf7330d36176d753d38f66a330a5ff1780fc4f"
NAMED_RELATIVE_METRIC = "global_rms_energy_allocation_floor_v1"
NAMED_RELATIVE_FORMULA = (
    "difference_l2/max(reference_l2,aggregate_reference_l2*sqrt(named_elements/aggregate_elements),float64_epsilon)"
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _seed_identity(label: str) -> dict[str, str | int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return {"seed_label": label, "seed_sha256": digest, "seed": int(digest[:8], 16)}


def _remove_path(value: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    cursor: Any = value
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise ValueError(f"R16 removal path is missing: {dotted_path}")
        cursor = cursor[part]
    if not isinstance(cursor, dict) or parts[-1] not in cursor:
        raise ValueError(f"R16 removal path is missing: {dotted_path}")
    del cursor[parts[-1]]


def _expected_r16_direct_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "R16-D0",
            **_seed_identity("qwen35-hardware-qualification-r16-h2-direct-0"),
            "rows": 80,
            "supervision_kind": "explicit",
            "supervised_rows": [0, 3, 9, 18, 34, 55, 79],
            "expected_supervised_count": 7,
            "global_divisor": 41,
            "hidden_scale": 0.5,
            "weight_standard_deviation": 0.02,
        },
        {
            "case_id": "R16-D1",
            **_seed_identity("qwen35-hardware-qualification-r16-h2-direct-1"),
            "rows": 129,
            "supervision_kind": "explicit",
            "supervised_rows": [0, 1, 8, 21, 42, 64, 96, 127, 128],
            "expected_supervised_count": 9,
            "global_divisor": 97,
            "hidden_scale": 2.0,
            "weight_standard_deviation": 0.02,
        },
        {
            "case_id": "R16-D2",
            **_seed_identity("qwen35-hardware-qualification-r16-h2-direct-2"),
            "rows": 97,
            "supervision_kind": "explicit",
            "supervised_rows": [2, 16, 48, 95],
            "expected_supervised_count": 4,
            "global_divisor": 193,
            "hidden_scale": 0.125,
            "weight_standard_deviation": 0.02,
        },
    ]


def _expected_r16_trajectories() -> list[dict[str, Any]]:
    result = []
    for index in range(3):
        model_label = f"qwen35-hardware-qualification-r16-h2-trajectory-{index}"
        model = _seed_identity(model_label)
        batches = _seed_identity(f"{model_label}-batches")
        heldout = _seed_identity(f"{model_label}-heldout")
        result.append(
            {
                "trajectory_id": f"R16-T{index}",
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
    return result


def load_qualification_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Resolve and validate the hash-bound R16 overlay over immutable R15."""

    raw = path.read_bytes()
    overlay = json.loads(raw)
    if set(overlay) != {
        "schema_version",
        "protocol_id",
        "protocol_date",
        "status",
        "base_manifest",
        "transformations",
        "overrides",
    }:
        raise ValueError("R16 overlay top-level field set drift")
    if (
        overlay["schema_version"] != 1
        or overlay["protocol_id"] != QUALIFICATION_PROTOCOL_ID
        or overlay["protocol_date"] != "2026-07-19"
        or overlay["status"] != "ready_for_execution"
    ):
        raise ValueError("R16 overlay identity/status drift")
    expected_base = {
        "path": "qwen35_hardware_qualification_r15.json",
        "sha256": BASE_MANIFEST_SHA256,
        "protocol_id": BASE_PROTOCOL_ID,
    }
    if overlay["base_manifest"] != expected_base:
        raise ValueError("R16 base-manifest binding drift")
    base_path = path.parent / expected_base["path"]
    if r15.sha256_file(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError("R16 immutable R15 base-manifest bytes drift")
    base, base_digest = r15.load_qualification_manifest(base_path)
    if base_digest != BASE_MANIFEST_SHA256 or base["protocol_id"] != BASE_PROTOCOL_ID:
        raise ValueError("R16 base manifest did not independently validate as R15")

    expected_transformations = {
        "append_base_confirmatory_direct_cases_to_historical": True,
        "remove_effective_paths": [
            "h2_acceptance.raw_first_step_update_is_gating",
            "h2_acceptance.raw_update_gating_starts_at_step",
            "h2_acceptance.r14_failed_first_step_update_reclassified_as_pass",
        ],
    }
    if overlay["transformations"] != expected_transformations:
        raise ValueError("R16 manifest transformation contract drift")
    overrides = overlay["overrides"]
    if not isinstance(overrides, dict) or set(overrides) != {
        "protocol_id",
        "protocol_date",
        "source",
        "h2_acceptance",
    }:
        raise ValueError("R16 override scope drift")
    if overrides["protocol_id"] != QUALIFICATION_PROTOCOL_ID or overrides["protocol_date"] != "2026-07-19":
        raise ValueError("R16 override identity drift")
    if overrides["source"] != {"corrective_baseline_commit": "a47257b2a501f056120549dc4c75131a62c1f10c"}:
        raise ValueError("R16 source override scope drift")
    expected_h2_override_keys = {
        "protocol_revision",
        "confirmatory_direct_cases",
        "confirmatory_trajectories",
        "trajectory_steps",
        "raw_updates_are_diagnostic",
        "named_relative_metric",
        "named_relative_formula",
        "named_relative_threshold",
        "named_minimum_cosine_similarity",
        "named_gradient_maximum_absolute_error",
        "optimizer_moments_are_gating",
        "cumulative_parameter_displacement_is_gating",
        "named_post_step_parameter_state_is_gating",
        "long_horizon_is_outcome_unseen",
        "r15_failed_criteria_reclassified_as_pass",
    }
    if (
        not isinstance(overrides["h2_acceptance"], dict)
        or set(overrides["h2_acceptance"]) != expected_h2_override_keys
    ):
        raise ValueError("R16 H2 override field set drift")

    effective = copy.deepcopy(base)
    effective["h2_acceptance"]["historical_direct_cases"] = copy.deepcopy(
        base["h2_acceptance"]["historical_direct_cases"] + base["h2_acceptance"]["confirmatory_direct_cases"]
    )
    for dotted_path in expected_transformations["remove_effective_paths"]:
        _remove_path(effective, dotted_path)
    effective = _deep_merge(effective, overlay["overrides"])
    effective["manifest_derivation"] = {
        "kind": "sha256_bound_overlay",
        "base_manifest": copy.deepcopy(expected_base),
        "transformations": copy.deepcopy(expected_transformations),
    }

    if effective["protocol_id"] != QUALIFICATION_PROTOCOL_ID or effective["protocol_date"] != "2026-07-19":
        raise ValueError("R16 effective identity drift")
    if effective["scope"]["slurm_account"] != "aifac_f02_434":
        raise ValueError("R16 does not require the personal Slurm account")
    if effective["scope"]["automatic_scientific_training"] is not False:
        raise ValueError("R16 may not authorize automatic scientific training")
    if effective["scope"]["eligible_arm_ids"] != ["C00"]:
        raise ValueError("R16 qualification scope drifted beyond C00")
    if effective["scope"]["forbidden_evaluations"] != ["BFCL", "tau2"]:
        raise ValueError("R16 forbidden-evaluation contract drift")
    if effective["source"]["corrective_baseline_commit"] != "a47257b2a501f056120549dc4c75131a62c1f10c":
        raise ValueError("R16 corrective baseline drift")
    if [gate["gate_id"] for gate in effective["gates"]] != [f"H{i}" for i in range(10)]:
        raise ValueError("R16 gate order drift")
    if any(gate["mandatory"] is not True for gate in effective["gates"]):
        raise ValueError("R16 contains a non-mandatory qualification gate")

    h2 = effective["h2_acceptance"]
    if h2["historical_direct_cases"] != (
        base["h2_acceptance"]["historical_direct_cases"] + base["h2_acceptance"]["confirmatory_direct_cases"]
    ):
        raise ValueError("R16 historical direct-case lineage drift")
    if h2["confirmatory_direct_cases"] != _expected_r16_direct_cases():
        raise ValueError("R16 confirmatory direct-case/seed contract drift")
    if h2["confirmatory_trajectories"] != _expected_r16_trajectories():
        raise ValueError("R16 trajectory/seed contract drift")
    expected_r16_scalars = {
        "protocol_revision": 3,
        "trajectory_steps": 128,
        "raw_updates_are_diagnostic": True,
        "named_relative_metric": NAMED_RELATIVE_METRIC,
        "named_relative_formula": NAMED_RELATIVE_FORMULA,
        "named_relative_threshold": 0.01,
        "named_minimum_cosine_similarity": 0.9999,
        "named_gradient_maximum_absolute_error": 0.01,
        "optimizer_moments_are_gating": True,
        "cumulative_parameter_displacement_is_gating": True,
        "named_post_step_parameter_state_is_gating": True,
        "long_horizon_is_outcome_unseen": True,
        "r15_failed_criteria_reclassified_as_pass": False,
    }
    for key, expected in expected_r16_scalars.items():
        if h2.get(key) != expected:
            raise ValueError(f"R16 H2 scalar contract drift for {key}")
    forbidden_old = {
        "raw_first_step_update_is_gating",
        "raw_update_gating_starts_at_step",
        "r14_failed_first_step_update_reclassified_as_pass",
    }
    if forbidden_old & set(h2):
        raise ValueError("R16 retained an obsolete R15 raw-update decision field")
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
    ):
        if h2[inherited_key] != base["h2_acceptance"][inherited_key]:
            raise ValueError(f"R16 inherited H2 geometry drift for {inherited_key}")
    if h2["trajectory_parameter_count"] != sum(row["elements"] for row in h2["trajectory_parameter_geometry"]):
        raise ValueError("R16 parameter geometry does not partition the complete model")
    if effective["numerical_acceptance"] != base["numerical_acceptance"]:
        raise ValueError("R16 changed a pre-existing numerical threshold")
    if effective["runtime_pins"] != base["runtime_pins"]:
        raise ValueError("R16 runtime pin drift")
    if effective["model"] != base["model"] or effective["h1_acceptance"] != base["h1_acceptance"]:
        raise ValueError("R16 changed the model or H1 contract")
    return effective, hashlib.sha256(raw).hexdigest()


def tensor_comparison_metrics(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Strict-JSON-safe tensor metrics, including nonfinite failure evidence."""

    if observed.shape != reference.shape:
        raise ValueError(f"tensor shape mismatch: {tuple(observed.shape)} != {tuple(reference.shape)}")
    observed64 = observed.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if observed64.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    nonfinite_count = int((~(torch.isfinite(observed64) & torch.isfinite(reference64))).sum())
    if nonfinite_count:
        return {
            "elements": observed64.numel(),
            "maximum_absolute_error": None,
            "relative_l2_error": None,
            "cosine_similarity": None,
            "observed_l2_norm": None,
            "reference_l2_norm": None,
            "difference_l2_norm": None,
            "nonfinite_count": nonfinite_count,
        }
    return r15.tensor_comparison_metrics(observed64, reference64)


def scalar_comparison_metrics(observed: float, reference: float) -> dict[str, Any]:
    """Strict-JSON-safe scalar metrics, including nonfinite failure evidence."""

    if not (math.isfinite(observed) and math.isfinite(reference)):
        return {
            "observed": observed if math.isfinite(observed) else None,
            "reference": reference if math.isfinite(reference) else None,
            "maximum_absolute_error": None,
            "relative_error": None,
            "nonfinite_count": int(not math.isfinite(observed)) + int(not math.isfinite(reference)),
        }
    return r15.scalar_comparison_metrics(observed, reference)


def balanced_tensor_comparison_metrics(
    observed: torch.Tensor,
    reference: torch.Tensor,
    *,
    aggregate_reference_l2_norm: float | None,
    aggregate_elements: int,
) -> dict[str, float | int | None | str]:
    """Compare a named tensor with an energy-conserving global-RMS floor."""

    metrics = tensor_comparison_metrics(observed, reference)
    named_elements = int(metrics["elements"])
    if aggregate_elements <= 0 or not 0 < named_elements <= aggregate_elements:
        raise ValueError("invalid named/aggregate element geometry")
    if aggregate_reference_l2_norm is not None and (
        not math.isfinite(float(aggregate_reference_l2_norm)) or aggregate_reference_l2_norm < 0
    ):
        raise ValueError("invalid aggregate reference norm")
    if aggregate_reference_l2_norm is None or metrics["nonfinite_count"]:
        return {
            **metrics,
            "named_relative_metric": NAMED_RELATIVE_METRIC,
            "aggregate_elements": aggregate_elements,
            "aggregate_reference_l2_norm": aggregate_reference_l2_norm,
            "global_rms_allocation_floor_l2_norm": None,
            "balanced_denominator_l2_norm": None,
            "balanced_relative_l2_error": None,
        }
    floor = float(aggregate_reference_l2_norm) * math.sqrt(named_elements / aggregate_elements)
    denominator = max(float(metrics["reference_l2_norm"]), floor, torch.finfo(torch.float64).eps)
    balanced = float(metrics["difference_l2_norm"]) / denominator
    return {
        **metrics,
        "named_relative_metric": NAMED_RELATIVE_METRIC,
        "aggregate_elements": aggregate_elements,
        "aggregate_reference_l2_norm": float(aggregate_reference_l2_norm),
        "global_rms_allocation_floor_l2_norm": floor,
        "balanced_denominator_l2_norm": denominator,
        "balanced_relative_l2_error": balanced,
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def _energy_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-15)


def validate_balanced_metric_arithmetic(
    metrics: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    *,
    expected_elements: int,
    aggregate_elements: int,
    context: str,
) -> None:
    required = {
        "elements",
        "maximum_absolute_error",
        "relative_l2_error",
        "cosine_similarity",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
        "nonfinite_count",
        "named_relative_metric",
        "aggregate_elements",
        "aggregate_reference_l2_norm",
        "global_rms_allocation_floor_l2_norm",
        "balanced_denominator_l2_norm",
        "balanced_relative_l2_error",
    }
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ValueError(f"{context}: balanced metric field set drift")
    if metrics["elements"] != expected_elements or metrics["aggregate_elements"] != aggregate_elements:
        raise ValueError(f"{context}: balanced metric geometry drift")
    if metrics["named_relative_metric"] != NAMED_RELATIVE_METRIC:
        raise ValueError(f"{context}: named metric identity drift")
    raw_fields = {
        key: metrics[key]
        for key in (
            "elements",
            "maximum_absolute_error",
            "relative_l2_error",
            "cosine_similarity",
            "observed_l2_norm",
            "reference_l2_norm",
            "difference_l2_norm",
            "nonfinite_count",
        )
    }
    _require_tensor_metric(raw_fields, expected_elements=expected_elements, context=f"{context} raw metric")
    if aggregate_metrics.get("elements") != aggregate_elements:
        raise ValueError(f"{context}: aggregate metric geometry drift")
    if metrics["nonfinite_count"] or aggregate_metrics["nonfinite_count"]:
        if any(
            metrics[key] is not None
            for key in (
                "global_rms_allocation_floor_l2_norm",
                "balanced_denominator_l2_norm",
                "balanced_relative_l2_error",
            )
        ):
            raise ValueError(f"{context}: nonfinite balanced metric must have null derived scales")
        if metrics["aggregate_reference_l2_norm"] is not aggregate_metrics["reference_l2_norm"]:
            raise ValueError(f"{context}: nonfinite aggregate reference binding drift")
        return
    aggregate_norm = float(aggregate_metrics["reference_l2_norm"])
    if not _close(metrics["aggregate_reference_l2_norm"], aggregate_norm):
        raise ValueError(f"{context}: aggregate reference norm binding drift")
    floor = aggregate_norm * math.sqrt(expected_elements / aggregate_elements)
    denominator = max(float(metrics["reference_l2_norm"]), floor, torch.finfo(torch.float64).eps)
    balanced = float(metrics["difference_l2_norm"]) / denominator
    if not _close(metrics["global_rms_allocation_floor_l2_norm"], floor):
        raise ValueError(f"{context}: global-RMS floor arithmetic drift")
    if not _close(metrics["balanced_denominator_l2_norm"], denominator):
        raise ValueError(f"{context}: balanced denominator arithmetic drift")
    if not _close(metrics["balanced_relative_l2_error"], balanced):
        raise ValueError(f"{context}: balanced relative-L2 arithmetic drift")


def _balanced_metric_decision(
    metrics: dict[str, Any], h2: dict[str, Any], *, context: str, gating: bool, gradient: bool = False
) -> dict[str, Any]:
    failures = []
    if metrics.get("nonfinite_count") != 0:
        failures.append("nonfinite count")
    if metrics.get("balanced_relative_l2_error") is None or (
        metrics["balanced_relative_l2_error"] > h2["named_relative_threshold"]
    ):
        failures.append("balanced relative-L2 error")
    cosine = metrics.get("cosine_similarity")
    if cosine is None or cosine < h2["named_minimum_cosine_similarity"]:
        failures.append("cosine similarity")
    if gradient and (
        metrics.get("maximum_absolute_error") is None
        or metrics["maximum_absolute_error"] > h2["named_gradient_maximum_absolute_error"]
    ):
        failures.append("maximum absolute error")
    return {
        "context": context,
        "kind": "balanced_gradient" if gradient else "balanced_state",
        "gating": gating,
        "passed": not failures,
        "exception_type": "AssertionError" if failures else None,
        "message": f"failed: {', '.join(failures)}" if failures else None,
    }


def _ordinary_decision(
    metrics: dict[str, Any], acceptance: dict[str, Any], *, kind: str, context: str, gating: bool
) -> dict[str, Any]:
    try:
        r15.validate_comparison_metrics(metrics, acceptance, kind=kind, context=context)
    except Exception as error:
        return {
            "context": context,
            "kind": kind,
            "gating": gating,
            "passed": False,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return {
        "context": context,
        "kind": kind,
        "gating": gating,
        "passed": True,
        "exception_type": None,
        "message": None,
    }


def _logit_decision(metrics: dict[str, Any], acceptance: dict[str, Any], *, context: str) -> dict[str, Any]:
    failures = []
    if metrics.get("nonfinite_count") != 0:
        failures.append("nonfinite count")
    if metrics.get("maximum_absolute_error") is None or (
        metrics["maximum_absolute_error"] > acceptance["packed_logit_absolute_tolerance"]
    ):
        failures.append("maximum absolute error")
    if metrics.get("relative_l2_error") is None or (
        metrics["relative_l2_error"] > acceptance["packed_logit_relative_tolerance"]
    ):
        failures.append("relative-L2 error")
    if metrics.get("cosine_similarity") is None or (
        metrics["cosine_similarity"] < acceptance["heldout_logit_minimum_cosine_similarity"]
    ):
        failures.append("cosine similarity")
    return {
        "context": context,
        "kind": "heldout_logit",
        "gating": True,
        "passed": not failures,
        "exception_type": "AssertionError" if failures else None,
        "message": f"failed: {', '.join(failures)}" if failures else None,
    }


def collect_h2_numerical_decisions(report: dict[str, Any], qualification: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every R16 gating and diagnostic decision without early exit."""

    acceptance = qualification["numerical_acceptance"]
    h2 = qualification["h2_acceptance"]
    decisions: list[dict[str, Any]] = []
    for section in ("historical_direct_cases", "confirmatory_direct_cases"):
        for case in report[section]:
            case_id = case["case_contract"]["case_id"]
            decisions.extend(
                [
                    _ordinary_decision(
                        case["loss_comparison"], acceptance, kind="loss", context=f"{case_id} loss", gating=True
                    ),
                    _ordinary_decision(
                        case["selected_hidden_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{case_id} selected-hidden gradient",
                        gating=True,
                    ),
                    _ordinary_decision(
                        case["output_head_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{case_id} output-head gradient",
                        gating=True,
                    ),
                    _ordinary_decision(
                        case["raw_first_adamw_update_comparison_diagnostic"],
                        acceptance,
                        kind="update",
                        context=f"{case_id} raw first AdamW update",
                        gating=False,
                    ),
                    _ordinary_decision(
                        case["optimizer_exp_avg_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{case_id} Adam exp_avg",
                        gating=True,
                    ),
                    _ordinary_decision(
                        case["optimizer_exp_avg_sq_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{case_id} Adam exp_avg_sq",
                        gating=True,
                    ),
                    _ordinary_decision(
                        case["post_step_parameter_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{case_id} post-step parameter state",
                        gating=True,
                    ),
                    _logit_decision(
                        case["heldout"]["logit_comparison"], acceptance, context=f"{case_id} heldout logits"
                    ),
                    _ordinary_decision(
                        case["heldout"]["loss_comparison"],
                        acceptance,
                        kind="loss",
                        context=f"{case_id} heldout loss",
                        gating=True,
                    ),
                ]
            )

    balanced_fields = (
        ("preclip_gradient_comparison", True, True, "preclip gradient"),
        ("clipped_gradient_comparison", True, True, "clipped gradient"),
        ("raw_adamw_update_comparison_diagnostic", False, False, "raw AdamW update"),
        ("optimizer_exp_avg_comparison", True, False, "Adam exp_avg"),
        ("optimizer_exp_avg_sq_comparison", True, False, "Adam exp_avg_sq"),
        ("cumulative_parameter_displacement_comparison", True, False, "cumulative parameter displacement"),
        ("post_step_parameter_state_comparison", True, False, "post-step parameter state"),
    )
    for trajectory in report["confirmatory_trajectories"]:
        trajectory_id = trajectory["trajectory_contract"]["trajectory_id"]
        parameter_names = trajectory["parameter_names"]
        for step in trajectory["steps"]:
            prefix = f"{trajectory_id} step {step['step']}"
            decisions.extend(
                [
                    _ordinary_decision(
                        step["training_loss_comparison"],
                        acceptance,
                        kind="loss",
                        context=f"{prefix} training loss",
                        gating=True,
                    ),
                    _ordinary_decision(
                        step["aggregate_preclip_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{prefix} aggregate preclip gradient",
                        gating=True,
                    ),
                    _ordinary_decision(
                        step["aggregate_clipped_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{prefix} aggregate clipped gradient",
                        gating=True,
                    ),
                    _ordinary_decision(
                        step["aggregate_raw_adamw_update_comparison_diagnostic"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} aggregate raw AdamW update",
                        gating=False,
                    ),
                    _ordinary_decision(
                        step["aggregate_optimizer_exp_avg_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} aggregate Adam exp_avg",
                        gating=True,
                    ),
                    _ordinary_decision(
                        step["aggregate_optimizer_exp_avg_sq_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} aggregate Adam exp_avg_sq",
                        gating=True,
                    ),
                    _ordinary_decision(
                        step["aggregate_cumulative_parameter_displacement_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} aggregate cumulative parameter displacement",
                        gating=True,
                    ),
                    _ordinary_decision(
                        step["aggregate_post_step_parameter_state_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} aggregate post-step parameter state",
                        gating=True,
                    ),
                    _logit_decision(
                        step["heldout"]["logit_comparison"], acceptance, context=f"{prefix} heldout logits"
                    ),
                    _ordinary_decision(
                        step["heldout"]["loss_comparison"],
                        acceptance,
                        kind="loss",
                        context=f"{prefix} heldout loss",
                        gating=True,
                    ),
                ]
            )
            for name in parameter_names:
                named = step["per_parameter_comparisons"][name]
                for field, gating, gradient, label in balanced_fields:
                    decisions.append(
                        _balanced_metric_decision(
                            named[field],
                            h2,
                            context=f"{prefix} parameter {name} {label}",
                            gating=gating,
                            gradient=gradient,
                        )
                    )

    failed_gating = [row["context"] for row in decisions if row["gating"] and not row["passed"]]
    failed_diagnostic = [row["context"] for row in decisions if not row["gating"] and not row["passed"]]
    return {
        "checks": decisions,
        "total_checks": len(decisions),
        "gating_checks": sum(row["gating"] for row in decisions),
        "diagnostic_checks": sum(not row["gating"] for row in decisions),
        "failed_gating_checks": failed_gating,
        "failed_diagnostic_checks": failed_diagnostic,
        "status": "passed" if not failed_gating else "failed",
    }


def _require_tensor_metric(value: Any, *, expected_elements: int, context: str) -> dict[str, Any]:
    required = {
        "elements",
        "maximum_absolute_error",
        "relative_l2_error",
        "cosine_similarity",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
        "nonfinite_count",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{context}: tensor metric field set drift")
    if value["elements"] != expected_elements:
        raise ValueError(f"{context}: element-count drift")
    if not isinstance(value["nonfinite_count"], int) or not 0 <= value["nonfinite_count"] <= expected_elements:
        raise ValueError(f"{context}: invalid nonfinite count")
    numeric = (
        "maximum_absolute_error",
        "relative_l2_error",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
    )
    if value["nonfinite_count"]:
        if any(value[key] is not None for key in (*numeric, "cosine_similarity")):
            raise ValueError(f"{context}: nonfinite tensor metrics must use strict-JSON null numerics")
        return value
    if any(
        not isinstance(value[key], (int, float)) or not math.isfinite(float(value[key])) or value[key] < 0
        for key in numeric
    ):
        raise ValueError(f"{context}: invalid nonnegative tensor metric")
    if value["cosine_similarity"] is not None and (
        not isinstance(value["cosine_similarity"], (int, float))
        or not math.isfinite(float(value["cosine_similarity"]))
    ):
        raise ValueError(f"{context}: invalid cosine")
    reference_norm = float(value["reference_l2_norm"])
    observed_norm = float(value["observed_l2_norm"])
    difference_norm = float(value["difference_l2_norm"])
    denominator = max(reference_norm, torch.finfo(torch.float64).eps)
    if not _energy_close(value["relative_l2_error"], difference_norm / denominator):
        raise ValueError(f"{context}: relative-L2 arithmetic drift")
    if observed_norm == 0 and reference_norm == 0:
        expected_cosine: float | None = 1.0
    elif observed_norm == 0 or reference_norm == 0:
        expected_cosine = None
    else:
        dot = (observed_norm**2 + reference_norm**2 - difference_norm**2) / 2
        expected_cosine = max(-1.0, min(1.0, dot / (observed_norm * reference_norm)))
    if expected_cosine is None:
        if value["cosine_similarity"] is not None:
            raise ValueError(f"{context}: zero-norm cosine arithmetic drift")
    elif value["cosine_similarity"] is None or not _energy_close(value["cosine_similarity"], expected_cosine):
        raise ValueError(f"{context}: cosine/norm arithmetic drift")
    maximum = float(value["maximum_absolute_error"])
    tolerance = 1e-12 * max(1.0, difference_norm, maximum)
    if maximum > difference_norm + tolerance or difference_norm > math.sqrt(expected_elements) * maximum + tolerance:
        raise ValueError(f"{context}: maximum-error/norm inequality drift")
    return value


def _require_scalar_metric(value: Any, *, context: str) -> dict[str, Any]:
    required = {"observed", "reference", "maximum_absolute_error", "relative_error", "nonfinite_count"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{context}: scalar metric field set drift")
    if not isinstance(value["nonfinite_count"], int) or not 0 <= value["nonfinite_count"] <= 2:
        raise ValueError(f"{context}: invalid scalar nonfinite count")
    if value["nonfinite_count"]:
        if value["maximum_absolute_error"] is not None or value["relative_error"] is not None:
            raise ValueError(f"{context}: nonfinite scalar metrics must use null derived errors")
        finite_values = sum(
            item is not None and isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in (value["observed"], value["reference"])
        )
        if finite_values != 2 - value["nonfinite_count"]:
            raise ValueError(f"{context}: nonfinite scalar evidence cardinality drift")
        return value
    if any(
        not isinstance(value[key], (int, float)) or not math.isfinite(float(value[key]))
        for key in ("observed", "reference", "maximum_absolute_error", "relative_error")
    ):
        raise ValueError(f"{context}: scalar metric type drift")
    if value["maximum_absolute_error"] < 0 or value["relative_error"] < 0 or value["nonfinite_count"] < 0:
        raise ValueError(f"{context}: invalid scalar metric")
    observed = float(value["observed"])
    reference = float(value["reference"])
    absolute = abs(observed - reference)
    relative = absolute / max(abs(reference), torch.finfo(torch.float64).eps)
    if not _energy_close(value["maximum_absolute_error"], absolute) or not _energy_close(
        value["relative_error"], relative
    ):
        raise ValueError(f"{context}: scalar comparison arithmetic drift")
    return value


def _require_autocast(value: Any, *, context: str) -> None:
    if value != {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}:
        raise ValueError(f"{context}: BF16 autocast contract drift")


def _validate_source_binding(report: dict[str, Any], qualification: dict[str, Any]) -> None:
    source = report.get("liger_kernel")
    runtime = qualification["runtime_pins"]
    if not isinstance(source, dict):
        raise ValueError("R16 H2 Liger source evidence is missing")
    if source.get("commit") != runtime["liger_commit"] or source.get("version") != runtime["liger_version"]:
        raise ValueError("R16 H2 Liger version/commit drift")
    expected_files = runtime["liger_source_files_sha256"]
    files = source.get("implementation_files")
    if not isinstance(files, dict) or set(files) != set(expected_files):
        raise ValueError("R16 H2 executed Liger source-file set drift")
    for relative_path, expected_sha256 in expected_files.items():
        row = files[relative_path]
        if (
            not isinstance(row, dict)
            or row.get("sha256") != expected_sha256
            or "pinned-sources/liger-kernel" not in str(row.get("path", ""))
            or not str(row.get("path", "")).endswith(relative_path)
        ):
            raise ValueError(f"R16 H2 Liger source binding drift for {relative_path}")
    source_url = str(source.get("source_url", ""))
    archive_pinned = (
        source.get("archive_url_pinned") is True
        and runtime["liger_commit"] in source_url
        and "/archive/" in source_url
    )
    vcs_pinned = source.get("metadata_vcs_commit") == runtime["liger_commit"]
    if not (archive_pinned or vcs_pinned):
        raise ValueError("R16 H2 distribution metadata does not bind the Liger commit")


def _validate_direct_case(case: Any, contract: dict[str, Any], *, h2: dict[str, Any], context: str) -> None:
    if not isinstance(case, dict) or case.get("case_contract") != contract:
        raise ValueError(f"{context}: direct-case contract drift")
    rows = list(range(contract["rows"])) if contract["supervision_kind"] == "all" else contract["supervised_rows"]
    if case.get("supervised_rows_expanded") != rows or len(rows) != contract["expected_supervised_count"]:
        raise ValueError(f"{context}: supervised-row expansion drift")
    autocast = case.get("autocast_contract")
    if not isinstance(autocast, dict) or set(autocast) != {"selective", "dense_reference", "heldout"}:
        raise ValueError(f"{context}: autocast evidence coverage drift")
    for role, value in autocast.items():
        _require_autocast(value, context=f"{context} {role}")
    if case.get("dtypes") != {
        "hidden_input": "torch.bfloat16",
        "output_head_parameter": "torch.float32",
        "selective_hidden_gradient": "torch.bfloat16",
        "reference_hidden_gradient": "torch.bfloat16",
        "selective_output_head_gradient": "torch.float32",
        "reference_output_head_gradient": "torch.float32",
        "selective_optimizer_floating_state": ["torch.float32"],
        "reference_optimizer_floating_state": ["torch.float32"],
        "loss_accumulation": "torch.float32",
    }:
        raise ValueError(f"{context}: direct dtype contract drift")
    loss = _require_scalar_metric(case.get("loss_comparison"), context=f"{context} loss")
    if loss["observed"] != case.get("selective_loss") or loss["reference"] != case.get("reference_loss"):
        raise ValueError(f"{context}: direct loss binding drift")
    hidden_elements = contract["expected_supervised_count"] * h2["direct_hidden_size"]
    head_elements = h2["direct_vocab_size"] * h2["direct_hidden_size"]
    _require_tensor_metric(
        case.get("selected_hidden_gradient_comparison"),
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
        _require_tensor_metric(case.get(field), expected_elements=head_elements, context=f"{context} {field}")
    if case.get("optimizer_step_counters") != {"selective": [1], "dense_reference": [1]}:
        raise ValueError(f"{context}: direct optimizer step-counter drift")
    heldout = case.get("heldout")
    if not isinstance(heldout, dict) or heldout.get("rows") != h2["direct_heldout_rows"]:
        raise ValueError(f"{context}: direct heldout contract drift")
    _require_tensor_metric(
        heldout.get("logit_comparison"),
        expected_elements=h2["direct_heldout_rows"] * h2["direct_vocab_size"],
        context=f"{context} heldout logits",
    )
    heldout_loss = _require_scalar_metric(heldout.get("loss_comparison"), context=f"{context} heldout loss")
    if heldout_loss["observed"] != heldout.get("selective_loss") or heldout_loss["reference"] != heldout.get(
        "reference_loss"
    ):
        raise ValueError(f"{context}: heldout loss binding drift")


def validate_h2_liger_report(
    report: dict[str, Any],
    *,
    qualification: dict[str, Any],
    expected_manifest_sha256: str,
    require_numerical_pass: bool = True,
) -> dict[str, Any]:
    """Independently validate complete R16 H2 evidence and decision arithmetic."""

    if report.get("artifact") != "qwen35_selective_liger_downstream_qualification_r16":
        raise ValueError("R16 H2 report artifact identity drift")
    report_status = report.get("status")
    if report.get("schema_version") != 3 or report_status not in {"passed", "failed"}:
        raise ValueError("R16 H2 report did not publish a recognized schema-3 decision")
    if require_numerical_pass and report_status != "passed":
        raise ValueError("R16 H2 report did not publish a schema-3 pass")
    if report.get("qualification_protocol_id") != QUALIFICATION_PROTOCOL_ID:
        raise ValueError("R16 H2 report protocol drift")
    if report.get("qualification_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("R16 H2 report manifest identity drift")
    if report.get("manifest_derivation") != qualification["manifest_derivation"]:
        raise ValueError("R16 H2 base/overlay derivation drift")
    if report.get("numerical_acceptance") != qualification["numerical_acceptance"]:
        raise ValueError("R16 H2 numerical-acceptance drift")
    if report.get("h2_acceptance") != qualification["h2_acceptance"]:
        raise ValueError("R16 H2 acceptance contract drift")
    if report.get("precision_policy") != {
        "parameters": "torch.float32",
        "gradients": "dtype follows FP32 parameter storage; direct selected BF16 hidden-row leaf gradients are BF16",
        "adamw_moments": "torch.float32",
        "forward_backward_autocast": "torch.bfloat16",
        "loss_accumulation": "torch.float32",
    }:
        raise ValueError("R16 H2 precision-policy drift")
    if report.get("scientific_training_authorized") is not False:
        raise ValueError("R16 H2 improperly authorizes scientific training")
    if report.get("successor_gate_authorized") is not (report_status == "passed"):
        raise ValueError("R16 H2 successor-gate authorization/status disagreement")
    if report.get("torch_version") != qualification["runtime_pins"]["torch_version"]:
        raise ValueError("R16 H2 Torch runtime drift")
    if qualification["hardware_acceptance"]["gpu_name_contains"] not in str(report.get("cuda_device", "")):
        raise ValueError("R16 H2 GPU identity drift")
    _validate_source_binding(report, qualification)

    h2 = qualification["h2_acceptance"]
    if (
        report.get("direct_hidden_size") != h2["direct_hidden_size"]
        or report.get("direct_vocab_size") != h2["direct_vocab_size"]
    ):
        raise ValueError("R16 H2 direct geometry drift")
    historical = report.get("historical_direct_cases")
    confirmatory = report.get("confirmatory_direct_cases")
    if not isinstance(historical, list) or len(historical) != len(h2["historical_direct_cases"]):
        raise ValueError("R16 H2 historical direct-case cardinality drift")
    if not isinstance(confirmatory, list) or len(confirmatory) != len(h2["confirmatory_direct_cases"]):
        raise ValueError("R16 H2 confirmatory direct-case cardinality drift")
    for index, (case, contract) in enumerate(zip(historical, h2["historical_direct_cases"], strict=True)):
        _validate_direct_case(case, contract, h2=h2, context=f"historical direct case {index}")
    for index, (case, contract) in enumerate(zip(confirmatory, h2["confirmatory_direct_cases"], strict=True)):
        _validate_direct_case(case, contract, h2=h2, context=f"confirmatory direct case {index}")

    zero = report.get("zero_target_sentinel")
    if not isinstance(zero, dict):
        raise ValueError("R16 H2 zero-target sentinel missing")
    _require_autocast(zero.get("autocast_contract"), context="zero-target sentinel")
    if (
        zero.get("loss") != 0.0
        or zero.get("global_divisor") != 7
        or zero.get("hidden_input_dtype") != "torch.bfloat16"
        or zero.get("output_head_parameter_dtype") != "torch.float32"
        or zero.get("hidden_gradient_dtype") != "torch.bfloat16"
        or zero.get("output_head_gradient_dtype") != "torch.float32"
        or zero.get("hidden_gradient_connected") is not True
        or zero.get("weight_gradient_connected") is not True
        or zero.get("gradient_nonzero_count") != 0
    ):
        raise ValueError("R16 H2 zero-target finite connected-zero contract drift")

    trajectories = report.get("confirmatory_trajectories")
    expected_trajectories = h2["confirmatory_trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) != len(expected_trajectories):
        raise ValueError("R16 H2 trajectory cardinality drift")
    expected_geometry = h2["trajectory_parameter_geometry"]
    names = [row["name"] for row in expected_geometry]
    elements_by_name = {row["name"]: row["elements"] for row in expected_geometry}
    parameter_count = h2["trajectory_parameter_count"]
    heldout_supervised = sum(
        position % h2["trajectory_heldout_supervision_modulus"] == 0
        for position in range(1, h2["trajectory_sequence_length"])
    )
    expected_heldout = {
        "sequence_length": h2["trajectory_sequence_length"],
        "supervision_modulus": h2["trajectory_heldout_supervision_modulus"],
        "supervised_targets": heldout_supervised,
        "divisor_extra": h2["trajectory_heldout_divisor_extra"],
        "global_divisor": heldout_supervised + h2["trajectory_heldout_divisor_extra"],
    }
    aggregate_fields = {
        "preclip_gradient_comparison": "aggregate_preclip_gradient_comparison",
        "clipped_gradient_comparison": "aggregate_clipped_gradient_comparison",
        "raw_adamw_update_comparison_diagnostic": "aggregate_raw_adamw_update_comparison_diagnostic",
        "optimizer_exp_avg_comparison": "aggregate_optimizer_exp_avg_comparison",
        "optimizer_exp_avg_sq_comparison": "aggregate_optimizer_exp_avg_sq_comparison",
        "cumulative_parameter_displacement_comparison": ("aggregate_cumulative_parameter_displacement_comparison"),
        "post_step_parameter_state_comparison": "aggregate_post_step_parameter_state_comparison",
    }
    step_checks = 0
    named_checks = 0
    for trajectory_index, (trajectory, contract) in enumerate(zip(trajectories, expected_trajectories, strict=True)):
        context = f"trajectory {trajectory_index}"
        if not isinstance(trajectory, dict) or trajectory.get("trajectory_contract") != contract:
            raise ValueError(f"{context}: trajectory contract drift")
        if (
            trajectory.get("trajectory_index") != trajectory_index
            or trajectory.get("model_class") != "Qwen3_5ForCausalLM"
            or "liger_kernel" not in str(trajectory.get("patched_forward_module", ""))
            or "transformers" not in str(trajectory.get("dense_forward_module", ""))
            or trajectory.get("model_config") != h2["trajectory_model_config"]
            or trajectory.get("parameter_names") != names
            or trajectory.get("parameter_geometry") != expected_geometry
            or trajectory.get("parameter_count") != parameter_count
        ):
            raise ValueError(f"{context}: model/parameter geometry drift")
        if trajectory.get("parameter_dtypes") != {
            "selective": ["torch.float32"],
            "dense_reference": ["torch.float32"],
        }:
            raise ValueError(f"{context}: parameter dtype drift")
        if trajectory.get("heldout_contract") != {"seed": contract["heldout_seed"], **expected_heldout}:
            raise ValueError(f"{context}: heldout contract drift")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or len(steps) != h2["trajectory_steps"]:
            raise ValueError(f"{context}: step cardinality drift")
        for step_index, step in enumerate(steps):
            step_number = step_index + 1
            step_context = f"{contract['trajectory_id']} step {step_number}"
            modulus = h2["trajectory_supervision_moduli"][step_index % len(h2["trajectory_supervision_moduli"])]
            offset = (step_index + trajectory_index) % modulus
            supervised = sum(
                (position + offset) % modulus == 0 for position in range(1, h2["trajectory_sequence_length"])
            )
            divisor_extra = (step_index * h2["trajectory_divisor_extra_multiplier"] + trajectory_index) % h2[
                "trajectory_divisor_extra_modulus"
            ]
            expected_accounting = {
                "seed": contract["batch_seed_base"] + step_index,
                "sequence_length": h2["trajectory_sequence_length"],
                "supervision_modulus": modulus,
                "supervision_offset": offset,
                "supervised_targets": supervised,
                "divisor_extra": divisor_extra,
                "global_divisor": supervised + divisor_extra,
            }
            if not isinstance(step, dict) or step.get("step") != step_number:
                raise ValueError(f"{step_context}: step identity drift")
            if step.get("batch_accounting") != expected_accounting:
                raise ValueError(f"{step_context}: batch accounting drift")
            autocast = step.get("autocast_contract")
            if not isinstance(autocast, dict) or set(autocast) != {"training", "heldout"}:
                raise ValueError(f"{step_context}: autocast evidence coverage drift")
            for role, value in autocast.items():
                _require_autocast(value, context=f"{step_context} {role}")
            loss = _require_scalar_metric(step.get("training_loss_comparison"), context=f"{step_context} loss")
            if loss["observed"] != step.get("selective_loss") or loss["reference"] != step.get("reference_loss"):
                raise ValueError(f"{step_context}: training loss binding drift")
            for aggregate_field in aggregate_fields.values():
                _require_tensor_metric(
                    step.get(aggregate_field),
                    expected_elements=parameter_count,
                    context=f"{step_context} {aggregate_field}",
                )
            preclip_norms = step.get("preclip_gradient_norms")
            if not isinstance(preclip_norms, dict) or set(preclip_norms) != {"selective", "dense_reference"}:
                raise ValueError(f"{step_context}: preclip norm evidence drift")
            preclip_nonfinite = step["aggregate_preclip_gradient_comparison"]["nonfinite_count"]
            for value in preclip_norms.values():
                if value is None:
                    if not preclip_nonfinite:
                        raise ValueError(f"{step_context}: null preclip norm without nonfinite gradient evidence")
                elif not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                    raise ValueError(f"{step_context}: invalid preclip norm evidence")
            if step.get("raw_adamw_updates_are_gating") is not False:
                raise ValueError(f"{step_context}: raw AdamW update was improperly made gating")
            if step.get("optimizer_floating_state_dtypes") != {
                "selective": ["torch.float32"],
                "dense_reference": ["torch.float32"],
            }:
                raise ValueError(f"{step_context}: optimizer floating-state dtype drift")
            if step.get("optimizer_step_counters") != {"selective": [step_number], "dense_reference": [step_number]}:
                raise ValueError(f"{step_context}: optimizer step-counter drift")
            if step.get("gradient_dtypes") != {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]}:
                raise ValueError(f"{step_context}: gradient dtype drift")
            per_parameter = step.get("per_parameter_comparisons")
            if not isinstance(per_parameter, dict) or set(per_parameter) != set(names):
                raise ValueError(f"{step_context}: named-parameter coverage drift")
            observed_elements = 0
            partition_metrics: dict[str, list[dict[str, Any]]] = {field: [] for field in aggregate_fields}
            for name in names:
                named = per_parameter[name]
                expected_elements = elements_by_name[name]
                if not isinstance(named, dict) or named.get("elements") != expected_elements:
                    raise ValueError(f"{step_context} parameter {name}: geometry drift")
                if set(named) != {"elements", *aggregate_fields.keys()}:
                    raise ValueError(f"{step_context} parameter {name}: field coverage drift")
                observed_elements += expected_elements
                for named_field, aggregate_field in aggregate_fields.items():
                    validate_balanced_metric_arithmetic(
                        named[named_field],
                        step[aggregate_field],
                        expected_elements=expected_elements,
                        aggregate_elements=parameter_count,
                        context=f"{step_context} parameter {name} {named_field}",
                    )
                    partition_metrics[named_field].append(named[named_field])
                    named_checks += 1
            if observed_elements != parameter_count:
                raise ValueError(f"{step_context}: named tensors do not partition the model")
            for named_field, aggregate_field in aggregate_fields.items():
                rows = partition_metrics[named_field]
                aggregate_metric = step[aggregate_field]
                named_nonfinite = sum(int(row["nonfinite_count"]) for row in rows)
                if named_nonfinite != aggregate_metric["nonfinite_count"]:
                    raise ValueError(f"{step_context} {named_field}: named/aggregate nonfinite-count drift")
                if aggregate_metric["nonfinite_count"]:
                    continue
                for norm_key in ("observed_l2_norm", "reference_l2_norm", "difference_l2_norm"):
                    partition_norm = math.sqrt(sum(float(row[norm_key]) ** 2 for row in rows))
                    if not _energy_close(partition_norm, aggregate_metric[norm_key]):
                        raise ValueError(f"{step_context} {named_field}: named/aggregate {norm_key} energy drift")
                if not _energy_close(
                    max(float(row["maximum_absolute_error"]) for row in rows),
                    aggregate_metric["maximum_absolute_error"],
                ):
                    raise ValueError(f"{step_context} {named_field}: named/aggregate maximum-error drift")
            heldout = step.get("heldout")
            if (
                not isinstance(heldout, dict)
                or heldout.get("supervised_targets") != expected_heldout["supervised_targets"]
                or heldout.get("global_divisor") != expected_heldout["global_divisor"]
            ):
                raise ValueError(f"{step_context}: heldout accounting drift")
            _require_tensor_metric(
                heldout.get("logit_comparison"),
                expected_elements=h2["trajectory_sequence_length"] * h2["trajectory_model_config"]["vocab_size"],
                context=f"{step_context} heldout logits",
            )
            heldout_loss = _require_scalar_metric(
                heldout.get("loss_comparison"), context=f"{step_context} heldout loss"
            )
            if heldout_loss["observed"] != heldout.get("selective_loss") or heldout_loss["reference"] != heldout.get(
                "reference_loss"
            ):
                raise ValueError(f"{step_context}: heldout loss binding drift")
            step_checks += 1

    recomputed = collect_h2_numerical_decisions(report, qualification)
    decision = report.get("decision")
    if not isinstance(decision, dict) or decision != recomputed:
        raise ValueError("R16 H2 producer decision ledger is incomplete or inconsistent")
    contexts = [row.get("context") for row in decision["checks"]]
    if len(contexts) != len(set(contexts)) or any(not isinstance(value, str) or not value for value in contexts):
        raise ValueError("R16 H2 producer decision contexts are missing or duplicated")
    if decision["status"] != report_status:
        raise ValueError("R16 H2 decision/report status disagreement")
    if require_numerical_pass and decision["failed_gating_checks"]:
        first = next(row for row in decision["checks"] if row["gating"] and not row["passed"])
        raise AssertionError(f"{first['context']}: {first['message']}")
    if not require_numerical_pass and report_status == "failed" and not decision["failed_gating_checks"]:
        raise ValueError("R16 H2 failed status has no independently reproduced gating failure")

    validation: dict[str, Any] = {
        "status": "passed",
        "historical_direct_cases": len(historical),
        "confirmatory_direct_cases": len(confirmatory),
        "confirmatory_trajectories": len(trajectories),
        "trajectory_steps": step_checks,
        "named_tensor_checks": named_checks,
        "diagnostic_raw_update_checks": decision["diagnostic_checks"],
        "gating_checks": decision["gating_checks"],
        "zero_target_sentinels": 1,
    }
    if not require_numerical_pass:
        validation.update(
            {
                "status": "evidence_validated",
                "numerical_status": report_status,
                "failed_gating_checks": len(decision["failed_gating_checks"]),
                "failed_diagnostic_checks": len(decision["failed_diagnostic_checks"]),
            }
        )
    return validation
