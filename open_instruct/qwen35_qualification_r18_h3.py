"""Fail-closed contracts and independent validation for R18 H3.

The CUDA producer is intentionally separate from this module.  H3 evidence is
stored as JSON plus safetensors; this validator reconstructs the frozen case
contract and recomputes numerical comparisons from the saved tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID
from open_instruct.qwen35_qualification import (
    scalar_comparison_metrics,
    sha256_file,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest as load_r18_manifest

H3_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h3-r1"
H3_ARTIFACT = "qwen35_r18_h3_distributed_normalization_qualification"
H3_HARNESS_AMENDMENT_ARTIFACT = "qwen35_r18_h3_harness_amendment"
H3_HARNESS_AMENDMENT_SHA256 = "35618b046ec7120866125189b7a512c3ee38b4b81e075f67d1c880ccd71896a1"
H3_HARNESS_AMENDMENT_HUMAN_SHA256 = "a95bc1b725af200ce9ff82d099a07c96bdc251b3c725dfe6b5dfb306bd392430"
H3_HARNESS_AMENDMENT_PREREGISTRATION_SHA256 = "e6a9caf4a5eb4d6bb16e3d96ec6a5837983ff2d6549be28704488c729207cc78"
H3_ATTEMPT01_FAILURE_CLOSURE_SHA256 = "c0ff9961da3d8407d0f4b8b6f44f427bee960a6a0012f797d8c14c5c184cd990"
PATHS = ("central_graph_sum", "central_sequential_backward", "ddp_gradient_accumulation")
COMPARISON_PATHS = PATHS[1:]
STORED_FAMILIES = (
    "initial_parameter",
    "preclip_gradient",
    "clipped_gradient",
    "post_step_parameter",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
)
COMPARISON_FAMILIES = (
    "preclip_gradient",
    "clipped_gradient",
    "raw_adamw_update",
    "post_step_parameter",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
)
GRADIENT_FAMILIES = {"preclip_gradient", "clipped_gradient"}


def norm_summary_cross_backend_relative_bound(element_count: int) -> float:
    """Return an upward-rounded binary64 reduction-order bound for two L2 norms."""

    if type(element_count) is not int or element_count <= 0:
        raise ValueError("H3 norm-summary element count must be a positive plain integer")
    with localcontext() as context:
        context.prec = 80
        one = Decimal(1)
        two = Decimal(2)
        unit_roundoff = two**Decimal(-53)
        reduction_steps = Decimal(element_count - 1)
        denominator = one - reduction_steps * unit_roundoff
        if denominator <= 0:
            raise ValueError("H3 norm-summary element count exceeds the binary64 error-model domain")
        gamma = reduction_steps * unit_roundoff / denominator
        epsilon_sum = (one + unit_roundoff) * (one + gamma) - one
        if epsilon_sum >= one:
            raise ValueError("H3 norm-summary accumulation bound is not finite")
        upper = (one + unit_roundoff) * (one + epsilon_sum).sqrt() - one
        lower = one - (one - unit_roundoff) * (one - epsilon_sum).sqrt()
        single_backend_bound = max(upper, lower)
        exact_pair_bound = two * single_backend_bound / (one - single_backend_bound)
        result = float(exact_pair_bound)
        if Decimal.from_float(result) < exact_pair_bound:
            result = math.nextafter(result, math.inf)
        return result


def _positive_binary64_ulp_distance(left: float, right: float) -> int:
    left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(left_bits - right_bits)


def validate_norm_summary_cross_backend_consistency(
    producer_value: Any,
    recomputed_value: float,
    *,
    element_count: int,
    context: str,
) -> dict[str, Any]:
    """Check a redundant CUDA norm summary against authoritative CPU tensor evidence."""

    if type(producer_value) not in {int, float} or type(recomputed_value) is not float:
        raise ValueError(f"{context}: norm summary is not a plain numeric value")
    producer = float(producer_value)
    recomputed = float(recomputed_value)
    if not math.isfinite(producer) or not math.isfinite(recomputed):
        raise ValueError(f"{context}: norm summary is nonfinite")
    if producer < 0 or recomputed < 0:
        raise ValueError(f"{context}: norm summary is negative")
    bound = norm_summary_cross_backend_relative_bound(element_count)
    absolute_difference = abs(producer - recomputed)
    scale = max(abs(producer), abs(recomputed))
    if scale == 0:
        relative_difference = 0.0
    else:
        relative_difference = absolute_difference / scale
    if relative_difference > bound:
        raise ValueError(f"{context}: cross-backend norm-summary bound exceeded")
    return {
        "absolute_difference": absolute_difference,
        "binary64_ulp_distance": _positive_binary64_ulp_distance(producer, recomputed),
        "element_count": element_count,
        "producer_value": producer,
        "recomputed_value": recomputed,
        "relative_bound": bound,
        "relative_difference": relative_difference,
    }


def load_h3_harness_amendment(
    path: Path,
    *,
    human_amendment_path: Path,
    attempt01_failure_closure_path: Path,
    preregistration_closure_path: Path,
    h3_manifest_path: Path,
) -> tuple[dict[str, Any], str]:
    """Validate the sole preregistered correction after invalid H3 attempt 01."""

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != H3_HARNESS_AMENDMENT_SHA256:
        raise ValueError("H3 harness-amendment hash drift")
    amendment = json.loads(raw)
    _require_exact_keys(
        amendment,
        {
            "allowed_implementation_changes",
            "allowed_successor_on_complete_pass",
            "artifact",
            "automatic_successor",
            "forbidden_changes",
            "human_amendment",
            "parent",
            "required_cpu_tests",
            "retry",
            "schema_version",
            "scientific_training_authorized",
            "status",
            "trigger",
        },
        context="H3 harness amendment",
    )
    if (
        amendment["schema_version"] != 1
        or amendment["artifact"] != H3_HARNESS_AMENDMENT_ARTIFACT
        or amendment["status"] != "ready_for_corrected_implementation"
        or amendment["automatic_successor"] is not False
        or amendment["scientific_training_authorized"] is not False
        or amendment["allowed_successor_on_complete_pass"] != "H4_only"
    ):
        raise ValueError("H3 harness-amendment identity or authority drift")
    if amendment["human_amendment"] != {
        "path": "methodology/qwen35_hardware_qualification_r18_h3_harness_amendment_r2_20260719.md",
        "sha256": H3_HARNESS_AMENDMENT_HUMAN_SHA256,
    }:
        raise ValueError("H3 harness-amendment human binding drift")
    if sha256_file(human_amendment_path) != H3_HARNESS_AMENDMENT_HUMAN_SHA256:
        raise ValueError("H3 harness-amendment human file hash drift")
    if sha256_file(attempt01_failure_closure_path) != H3_ATTEMPT01_FAILURE_CLOSURE_SHA256:
        raise ValueError("H3 attempt-01 failure-closure hash drift")
    if sha256_file(h3_manifest_path) != amendment["parent"]["h3_machine_manifest_sha256"]:
        raise ValueError("H3 harness-amendment parent manifest drift")
    trigger = amendment["trigger"]
    if trigger != {
        "attempt01_failure_closure_sha256": H3_ATTEMPT01_FAILURE_CLOSURE_SHA256,
        "job_id": "49852000",
        "numerical_reports_produced": 0,
        "rank_failure_sha256": "114c03bac867d25a81595d25b4f58dc9e3310bd1d9e99b392c7d0fc3af64e661",
        "status": "invalid_harness_failure",
        "tensor_evidence_files_produced": 0,
    }:
        raise ValueError("H3 harness-amendment trigger drift")
    retry = amendment["retry"]
    if retry != {
        "accelerator": "NVIDIA_A100_SXM_64GB",
        "account": "aifac_f02_434",
        "fresh_output_root_required": True,
        "fresh_source_root_required": True,
        "maximum_corrected_attempts": 1,
        "nodes": 1,
        "ranks": 4,
    }:
        raise ValueError("H3 harness-amendment retry scope drift")

    closure_raw = preregistration_closure_path.read_bytes()
    if hashlib.sha256(closure_raw).hexdigest() != H3_HARNESS_AMENDMENT_PREREGISTRATION_SHA256:
        raise ValueError("H3 harness-amendment preregistration-closure hash drift")
    closure = json.loads(closure_raw)
    expected_closure = {
        "artifact": "qwen35_r18_h3_harness_amendment_r2_preregistration_closure",
        "closed_at_utc": "2026-07-19T20:03:54Z",
        "human_amendment": {
            "path": (
                "methodology/"
                "qwen35_hardware_qualification_r18_h3_harness_amendment_r2_20260719.md"
            ),
            "sha256": H3_HARNESS_AMENDMENT_HUMAN_SHA256,
        },
        "machine_amendment": {
            "git_blob_commit": "e691985079ebce1f1bc74d57f4c2e0263d7e12fd",
            "path": "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3_harness_amendment_r2.json",
            "sha256": H3_HARNESS_AMENDMENT_SHA256,
        },
        "parent_h3_manifest_sha256": "95aec699d2bab81c5eb3094d2048f997f137faa624dbb7128f92b32134b8abf4",
        "schema_version": 1,
        "statement": (
            "The H3 R2 harness amendment was frozen after invalid attempt 01 and before corrected implementation "
            "or corrected-retry CUDA output. It does not alter any scientific or numerical H3 field."
        ),
        "status": "closed_before_corrected_implementation_and_retry_execution",
        "trigger_attempt01_failure_closure_sha256": H3_ATTEMPT01_FAILURE_CLOSURE_SHA256,
    }
    if closure != expected_closure:
        raise ValueError("H3 harness-amendment preregistration-closure content drift")
    return amendment, digest


def _create_directory_exclusively(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def prepare_distributed_output_directory(
    output_dir: Path,
    *,
    rank: int,
    broadcast_object_list: Callable[..., None],
    create_directory: Callable[[Path], None] = _create_directory_exclusively,
) -> dict[str, Any]:
    """Make rank 0 the sole shared-directory owner and broadcast its decision."""

    outcome: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            create_directory(output_dir)
            outcome[0] = {"creator_rank": 0, "output_dir": str(output_dir), "status": "created"}
        except Exception as error:
            outcome[0] = {
                "creator_rank": 0,
                "exception_message": str(error),
                "exception_type": type(error).__name__,
                "output_dir": str(output_dir),
                "status": "failed",
            }
    broadcast_object_list(outcome, src=0)
    record = outcome[0]
    if not isinstance(record, dict):
        raise RuntimeError("H3 output-directory ownership broadcast was missing")
    if record.get("creator_rank") != 0 or record.get("output_dir") != str(output_dir):
        raise RuntimeError("H3 output-directory ownership broadcast drift")
    if record.get("status") == "created":
        _require_exact_keys(record, {"creator_rank", "output_dir", "status"}, context="H3 output directory")
        return record
    if record.get("status") != "failed":
        raise RuntimeError("H3 output-directory ownership status drift")
    _require_exact_keys(
        record,
        {"creator_rank", "exception_message", "exception_type", "output_dir", "status"},
        context="H3 output-directory failure",
    )
    message = f"rank-0 H3 output-directory creation failed: {record['exception_type']}: {record['exception_message']}"
    if record["exception_type"] == "FileExistsError":
        raise FileExistsError(message)
    raise RuntimeError(message)


def _sha256_label(label: str) -> tuple[str, int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return digest, int(digest[:8], 16)


def _require_exact_keys(value: Any, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{context}: key-set drift: {observed}")
    return value


def _validate_seed_record(record: Any, *, expected_label: str, context: str) -> None:
    _require_exact_keys(record, {"seed", "seed_label", "seed_sha256"}, context=context)
    if record["seed_label"] != expected_label:
        raise ValueError(f"{context}: seed label drift")
    digest, seed = _sha256_label(expected_label)
    if record["seed_sha256"] != digest or record["seed"] != seed:
        raise ValueError(f"{context}: seed derivation drift")


def load_h3_manifest(
    path: Path, *, r18_manifest_path: Path, human_protocol_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load and validate the immutable H3 supplement and its parent bindings."""

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    required_top = {
        "allowed_successor_on_pass",
        "artifact",
        "candidate_chunk_sizes_in_execution_order",
        "execution",
        "failure_policy",
        "human_protocol",
        "implementation_id",
        "model",
        "numerical_acceptance",
        "optimizer",
        "parent",
        "paths",
        "preregistration_baseline_commit",
        "protocol_date",
        "protocol_id",
        "required_comparisons",
        "required_validator_negative_controls",
        "scenarios",
        "schema_version",
        "status",
        "tensor_evidence",
    }
    _require_exact_keys(manifest, required_top, context="H3 manifest")
    if manifest["schema_version"] != 1 or manifest["protocol_id"] != H3_PROTOCOL_ID:
        raise ValueError("H3 manifest identity drift")
    if manifest["artifact"] != "qwen35_r18_h3_distributed_normalization_contract":
        raise ValueError("H3 manifest artifact drift")
    if manifest["status"] != "preregistered_before_implementation_and_cuda_output":
        raise ValueError("H3 manifest is not preregistered")
    if manifest["allowed_successor_on_pass"] != "H4_only" or manifest["failure_policy"] != (
        "stop_before_H4_no_threshold_rescue"
    ):
        raise ValueError("H3 stop-policy drift")
    if manifest["implementation_id"] != IMPLEMENTATION_ID:
        raise ValueError("H3 production implementation drift")
    if manifest["candidate_chunk_sizes_in_execution_order"] != [128, 256, 512, 1024]:
        raise ValueError("H3 candidate order drift")

    parent = manifest["parent"]
    if sha256_file(r18_manifest_path) != parent["r18_machine_manifest_sha256"]:
        raise ValueError("H3 parent R18 machine-manifest hash drift")
    if sha256_file(human_protocol_path) != manifest["human_protocol"]["sha256"]:
        raise ValueError("H3 human-protocol hash drift")
    r18, r18_digest = load_r18_manifest(r18_manifest_path)
    if r18_digest != parent["r18_machine_manifest_sha256"] or r18["protocol_id"] != parent["r18_protocol_id"]:
        raise ValueError("H3 resolved R18 parent drift")
    if r18["h2_acceptance"]["production_implementation_id"] != IMPLEMENTATION_ID:
        raise ValueError("H3 and R18 selected-loss implementations disagree")
    if manifest["candidate_chunk_sizes_in_execution_order"] != r18["h2_acceptance"]["candidate_chunk_sizes"]:
        raise ValueError("H3 and R18 candidate sets disagree")
    inherited = r18["numerical_acceptance"]
    numerical = manifest["numerical_acceptance"]
    for key in (
        "gradient_maximum_absolute_error",
        "gradient_minimum_cosine_similarity",
        "gradient_relative_l2_error",
        "loss_maximum_absolute_error",
        "loss_relative_error",
        "nonfinite_count",
        "update_minimum_cosine_similarity",
        "update_relative_l2_error",
    ):
        if numerical.get(key) != inherited.get(key):
            raise ValueError(f"H3 inherited numerical threshold drift for {key}")
    if numerical.get("preclip_norm_must_exceed") != 1.0 or numerical.get("postclip_norm_maximum") != 1.000001:
        raise ValueError("H3 active-clipping threshold drift")

    execution = manifest["execution"]
    if (
        execution.get("ranks") != 4
        or execution.get("backend") != "nccl"
        or execution.get("slurm_account") != "aifac_f02_434"
        or execution.get("liger_execution_allowed") is not False
        or execution.get("fresh_process_per_scenario_candidate") is not True
        or execution.get("automatic_successor") is not False
    ):
        raise ValueError("H3 execution contract drift")
    optimizer = manifest["optimizer"]
    if (
        optimizer.get("name") != "torch.optim.AdamW"
        or optimizer.get("fused") is not True
        or optimizer.get("foreach") is not False
        or optimizer.get("maximum_gradient_norm") != 1.0
        or optimizer.get("optimizer_steps") != 1
    ):
        raise ValueError("H3 optimizer contract drift")
    if manifest["paths"].get("primary_reference") != "central_graph_sum":
        raise ValueError("H3 primary-reference drift")
    if manifest["paths"]["ddp_gradient_accumulation"].get("backward_multiplier") != 4:
        raise ValueError("H3 DDP world-size multiplier drift")

    scenarios = manifest["scenarios"]
    if not isinstance(scenarios, list) or [item.get("scenario_id") for item in scenarios] != ["P4x2", "B4x4"]:
        raise ValueError("H3 scenario set/order drift")
    expected_shapes = {"P4x2": 2, "B4x4": 4}
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        counts = scenario["target_counts_by_slot_rank"]
        if len(counts) != expected_shapes[scenario_id] or any(len(row) != 4 for row in counts):
            raise ValueError(f"H3 {scenario_id} target allocation geometry drift")
        if scenario["accumulation_steps"] != len(counts):
            raise ValueError(f"H3 {scenario_id} accumulation-step drift")
        flat = [int(value) for row in counts for value in row]
        if any(value < 0 or value > 1025 for value in flat):
            raise ValueError(f"H3 {scenario_id} invalid target count")
        per_rank = [sum(row[rank] for row in counts) for rank in range(4)]
        if per_rank != scenario["per_rank_target_counts"] or sum(flat) != scenario["global_target_count"]:
            raise ValueError(f"H3 {scenario_id} target accounting drift")
        if per_rank[0] != 0 or any(value <= 0 for value in per_rank[1:]):
            raise ValueError(f"H3 {scenario_id} zero/positive rank allocation drift")
        _validate_seed_record(
            {
                "seed": scenario["model_seed"],
                "seed_label": scenario["model_seed_label"],
                "seed_sha256": scenario["model_seed_sha256"],
            },
            expected_label=f"qwen35-hardware-qualification-r18-h3-{'production' if scenario_id == 'P4x2' else 'boundary'}-model",
            context=f"H3 {scenario_id} model seed",
        )
        records = scenario["case_seed_records"]
        if len(records) != len(flat):
            raise ValueError(f"H3 {scenario_id} case-seed count drift")
        label_part = "production" if scenario_id == "P4x2" else "boundary"
        for index, record in enumerate(records):
            _validate_seed_record(
                record,
                expected_label=f"qwen35-hardware-qualification-r18-h3-{label_part}-microbatch-{index:02d}",
                context=f"H3 {scenario_id} case seed {index}",
            )
    boundary_flat = [value for row in scenarios[1]["target_counts_by_slot_rank"] for value in row]
    for chunk_size in manifest["candidate_chunk_sizes_in_execution_order"]:
        if not {chunk_size - 1, chunk_size, chunk_size + 1} <= set(boundary_flat):
            raise ValueError(f"H3 boundary scenario does not cover chunk {chunk_size}")
    return manifest, digest, r18


def scenario_by_id(manifest: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    matches = [item for item in manifest["scenarios"] if item["scenario_id"] == scenario_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate H3 scenario {scenario_id!r}")
    return matches[0]


def expected_case_records(manifest: dict[str, Any], scenario_id: str) -> list[dict[str, Any]]:
    """Return canonical slot-major/rank-minor case membership."""

    scenario = scenario_by_id(manifest, scenario_id)
    records = []
    index = 0
    for slot, row in enumerate(scenario["target_counts_by_slot_rank"]):
        for rank, target_count in enumerate(row):
            seed_record = scenario["case_seed_records"][index]
            records.append({"case_id": index, "slot": slot, "rank": rank, "target_count": target_count, **seed_record})
            index += 1
    return records


def tensor_key(path: str, family: str, parameter_name: str) -> str:
    return f"{path}::{family}::{parameter_name}"


def _expected_boundaries(target_count: int, chunk_size: int) -> list[list[int]]:
    return [[start, min(start + chunk_size, target_count)] for start in range(0, target_count, chunk_size)]


def _validate_audit(audit: Any, *, target_count: int, global_divisor: int, chunk_size: int, context: str) -> None:
    required = {
        "checkpointed",
        "chunk_boundaries",
        "chunk_count",
        "chunk_size",
        "full_selected_logit_elements",
        "global_target_count",
        "hidden_size",
        "implementation_id",
        "maximum_chunk_rows",
        "maximum_logit_elements",
        "returned_dense_logits",
        "selected_rows",
        "vocabulary_size",
        "zero_target",
    }
    _require_exact_keys(audit, required, context=context)
    expected_boundaries = _expected_boundaries(target_count, chunk_size)
    if audit["implementation_id"] != IMPLEMENTATION_ID or audit["checkpointed"] is not True:
        raise ValueError(f"{context}: implementation/checkpoint drift")
    if audit["chunk_size"] != chunk_size or audit["chunk_boundaries"] != expected_boundaries:
        raise ValueError(f"{context}: chunk boundary drift")
    if audit["chunk_count"] != len(expected_boundaries):
        raise ValueError(f"{context}: chunk count drift")
    if audit["selected_rows"] != target_count or audit["zero_target"] is not (target_count == 0):
        raise ValueError(f"{context}: selected-row or zero-target drift")
    expected_divisor = 0 if target_count == 0 else global_divisor
    if audit["global_target_count"] != expected_divisor:
        raise ValueError(f"{context}: global divisor drift")
    if audit["vocabulary_size"] != 256 or audit["hidden_size"] != 64:
        raise ValueError(f"{context}: projection geometry drift")
    maximum_rows = min(target_count, chunk_size) if target_count else 0
    if audit["maximum_chunk_rows"] != maximum_rows:
        raise ValueError(f"{context}: maximum chunk rows drift")
    if audit["maximum_logit_elements"] != maximum_rows * 256:
        raise ValueError(f"{context}: maximum logits accounting drift")
    if audit["full_selected_logit_elements"] != target_count * 256:
        raise ValueError(f"{context}: full selected logits accounting drift")
    if audit["returned_dense_logits"] is not False:
        raise ValueError(f"{context}: dense logits were returned")


def _metric_equal(observed: Any, expected: Any, *, context: str) -> None:
    if isinstance(expected, dict):
        _require_exact_keys(observed, set(expected), context=context)
        for key in expected:
            _metric_equal(observed[key], expected[key], context=f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{context}: list drift")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _metric_equal(left, right, context=f"{context}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(float(observed), expected, rel_tol=0, abs_tol=0):
            raise ValueError(f"{context}: numerical summary drift")
        return
    if observed != expected:
        raise ValueError(f"{context}: value drift")


def _aggregate(values: dict[str, torch.Tensor], parameter_names: list[str]) -> torch.Tensor:
    return torch.cat([values[name].detach().reshape(-1) for name in parameter_names])


def _family_values(
    tensors: dict[str, torch.Tensor], *, path: str, family: str, parameter_names: list[str]
) -> dict[str, torch.Tensor]:
    return {name: tensors[tensor_key(path, family, name)] for name in parameter_names}


def _recompute_comparison(
    tensors: dict[str, torch.Tensor],
    *,
    observed_path: str,
    family: str,
    parameter_names: list[str],
    numerical: dict[str, Any],
) -> dict[str, Any]:
    if family == "raw_adamw_update":
        reference = {
            name: tensors[tensor_key(PATHS[0], "post_step_parameter", name)]
            - tensors[tensor_key(PATHS[0], "initial_parameter", name)]
            for name in parameter_names
        }
        observed = {
            name: tensors[tensor_key(observed_path, "post_step_parameter", name)]
            - tensors[tensor_key(observed_path, "initial_parameter", name)]
            for name in parameter_names
        }
    else:
        reference = _family_values(tensors, path=PATHS[0], family=family, parameter_names=parameter_names)
        observed = _family_values(tensors, path=observed_path, family=family, parameter_names=parameter_names)
    kind = "gradient" if family in GRADIENT_FAMILIES else "update"
    named = {}
    for name in parameter_names:
        metrics = tensor_comparison_metrics(observed[name], reference[name])
        validate_comparison_metrics(metrics, numerical, kind=kind, context=f"{observed_path} {family} {name}")
        named[name] = metrics
    aggregate = tensor_comparison_metrics(
        _aggregate(observed, parameter_names), _aggregate(reference, parameter_names)
    )
    validate_comparison_metrics(aggregate, numerical, kind=kind, context=f"{observed_path} aggregate {family}")
    return {"aggregate": aggregate, "named": named}


def validate_h3_report(
    report: dict[str, Any],
    *,
    evidence_path: Path,
    h3_manifest: dict[str, Any],
    h3_manifest_sha256: str,
    r18_manifest_sha256: str,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Independently validate one scenario/candidate report and tensor file."""

    required_top = {
        "allowed_conclusion",
        "artifact",
        "audits",
        "case_records",
        "chunk_size",
        "clipping",
        "comparisons",
        "contract",
        "decision",
        "environment",
        "h3_manifest_sha256",
        "harness_amendment",
        "human_protocol_sha256",
        "losses",
        "optimizer",
        "protocol_id",
        "r18_manifest_sha256",
        "scaling",
        "scenario_id",
        "schema_version",
        "source_attestation",
        "status",
        "tensor_evidence",
    }
    _require_exact_keys(report, required_top, context="H3 report")
    if report["schema_version"] != 1 or report["artifact"] != H3_ARTIFACT:
        raise ValueError("H3 report identity drift")
    if report["protocol_id"] != H3_PROTOCOL_ID or report["h3_manifest_sha256"] != h3_manifest_sha256:
        raise ValueError("H3 report protocol/manifest drift")
    if report["r18_manifest_sha256"] != r18_manifest_sha256:
        raise ValueError("H3 report R18 manifest drift")
    if report["human_protocol_sha256"] != h3_manifest["human_protocol"]["sha256"]:
        raise ValueError("H3 report human-protocol drift")
    harness_amendment = report["harness_amendment"]
    _require_exact_keys(
        harness_amendment,
        {
            "attempt01_failure_closure_sha256",
            "cuda_device_binding",
            "human_protocol_sha256",
            "machine_manifest_sha256",
            "output_directory_initialization",
            "preregistration_closure_sha256",
            "status",
        },
        context="H3 harness amendment",
    )
    if harness_amendment != {
        "attempt01_failure_closure_sha256": H3_ATTEMPT01_FAILURE_CLOSURE_SHA256,
        "cuda_device_binding": "before_nccl_initialization_with_explicit_device_id",
        "human_protocol_sha256": H3_HARNESS_AMENDMENT_HUMAN_SHA256,
        "machine_manifest_sha256": H3_HARNESS_AMENDMENT_SHA256,
        "output_directory_initialization": {
            "creator_rank": 0,
            "output_dir": str(evidence_path.parent),
            "status": "created",
        },
        "preregistration_closure_sha256": H3_HARNESS_AMENDMENT_PREREGISTRATION_SHA256,
        "status": "ready_for_corrected_implementation",
    }:
        raise ValueError("H3 harness-amendment report binding drift")
    if report["status"] not in {"passed", "failed"}:
        raise ValueError("H3 report status drift")
    if require_pass and report["status"] != "passed":
        raise ValueError("H3 report is not a numerical pass")
    if report["chunk_size"] not in h3_manifest["candidate_chunk_sizes_in_execution_order"]:
        raise ValueError("H3 report candidate drift")
    scenario = scenario_by_id(h3_manifest, report["scenario_id"])
    expected_cases = expected_case_records(h3_manifest, report["scenario_id"])
    if report["contract"] != {
        "accumulation_steps": scenario["accumulation_steps"],
        "global_target_count": scenario["global_target_count"],
        "per_rank_target_counts": scenario["per_rank_target_counts"],
        "target_counts_by_slot_rank": scenario["target_counts_by_slot_rank"],
        "world_size": 4,
    }:
        raise ValueError("H3 report distributed contract drift")

    observed_cases = report["case_records"]
    if not isinstance(observed_cases, list) or len(observed_cases) != len(expected_cases):
        raise ValueError("H3 case record count drift")
    model = h3_manifest["model"]
    for expected, observed in zip(expected_cases, observed_cases, strict=True):
        required_case = set(expected) | {
            "input_ids_sha256",
            "labels_sha256",
            "logits_to_keep_sha256",
            "shift_labels_sha256",
        }
        _require_exact_keys(observed, required_case, context=f"H3 case {expected['case_id']}")
        for key, value in expected.items():
            if observed[key] != value:
                raise ValueError(f"H3 case {expected['case_id']} {key} drift")
        generator = torch.Generator(device="cpu").manual_seed(expected["seed"])
        input_ids = torch.randint(1, model["vocab_size"] - 1, (1, model["sequence_length"]), generator=generator)
        labels = torch.full_like(input_ids, -100)
        count = expected["target_count"]
        if count:
            labels[:, 1 : count + 1] = input_ids[:, 1 : count + 1]
            positions = torch.arange(count, dtype=torch.long)
            shifted = labels[:, 1 : count + 1].reshape(-1).contiguous()
        else:
            positions = torch.tensor([0], dtype=torch.long)
            shifted = torch.tensor([-100], dtype=torch.long)
        reconstructed = {
            "input_ids_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
            "logits_to_keep_sha256": hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
            "shift_labels_sha256": hashlib.sha256(shifted.numpy().tobytes()).hexdigest(),
        }
        for key, value in reconstructed.items():
            if observed[key] != value:
                raise ValueError(f"H3 case {expected['case_id']} {key} drift")

    audits = report["audits"]
    _require_exact_keys(audits, set(PATHS), context="H3 audits")
    expected_ids = list(range(len(expected_cases)))
    for path in PATHS:
        path_audits = audits[path]
        if not isinstance(path_audits, list) or [item.get("case_id") for item in path_audits] != expected_ids:
            raise ValueError(f"H3 {path} audit membership/order drift")
        for item, case in zip(path_audits, expected_cases, strict=True):
            _require_exact_keys(item, {"audit", "case_id"}, context=f"H3 {path} audit envelope")
            _validate_audit(
                item["audit"],
                target_count=case["target_count"],
                global_divisor=scenario["global_target_count"],
                chunk_size=report["chunk_size"],
                context=f"H3 {path} case {case['case_id']} audit",
            )

    environment = report["environment"]
    _require_exact_keys(
        environment,
        {
            "autocast",
            "backend",
            "cuda_version",
            "device_names",
            "liger_modules_by_rank",
            "torch_version",
            "world_size",
        },
        context="H3 environment",
    )
    device_names = environment["device_names"]
    liger_by_rank = environment["liger_modules_by_rank"]
    if (
        environment.get("world_size") != 4
        or environment.get("backend") != "nccl"
        or environment.get("autocast") != {"device_type": "cuda", "dtype": "torch.bfloat16", "enabled": True}
        or not isinstance(device_names, list)
        or len(device_names) != 4
        or any(not isinstance(value, str) or "A100" not in value for value in device_names)
        or not isinstance(liger_by_rank, list)
        or len(liger_by_rank) != 4
        or any(value != [] for value in liger_by_rank)
    ):
        raise ValueError("H3 environment, device, autocast, or Liger contract drift")
    source = report["source_attestation"]
    _require_exact_keys(
        source,
        {"git_commit", "git_worktree_clean", "implementation_id", "liger_modules_imported", "source_files_sha256"},
        context="H3 source attestation",
    )
    if source.get("git_worktree_clean") is not True or source.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("H3 source attestation drift")
    if source.get("liger_modules_imported") != []:
        raise ValueError("H3 source imported Liger")
    if (
        not isinstance(source["git_commit"], str)
        or len(source["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in source["git_commit"])
        or not isinstance(source["source_files_sha256"], dict)
        or not source["source_files_sha256"]
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source["source_files_sha256"].values()
        )
    ):
        raise ValueError("H3 source identity/hash structure drift")

    tensor_meta = report["tensor_evidence"]
    _require_exact_keys(
        tensor_meta,
        {"bytes", "families", "file_name", "format", "key_count", "parameter_geometry", "paths", "sha256"},
        context="H3 tensor evidence",
    )
    if tensor_meta.get("format") != "safetensors" or tensor_meta.get("file_name") != evidence_path.name:
        raise ValueError("H3 tensor-evidence format/path drift")
    if tensor_meta.get("families") != list(STORED_FAMILIES) or tensor_meta.get("paths") != list(PATHS):
        raise ValueError("H3 tensor-evidence path/family declaration drift")
    if not evidence_path.is_file():
        raise ValueError("H3 tensor evidence is missing")
    if tensor_meta.get("bytes") != evidence_path.stat().st_size or tensor_meta.get("sha256") != sha256_file(
        evidence_path
    ):
        raise ValueError("H3 tensor-evidence size/hash drift")
    tensors = load_file(evidence_path, device="cpu")
    geometry = tensor_meta.get("parameter_geometry")
    if not isinstance(geometry, list) or not geometry:
        raise ValueError("H3 parameter geometry is missing")
    parameter_names = []
    for item in geometry:
        _require_exact_keys(item, {"dtype", "elements", "name", "shape"}, context="H3 parameter geometry")
        name = item["name"]
        if name in parameter_names:
            raise ValueError("H3 duplicate parameter geometry")
        parameter_names.append(name)
        if item["dtype"] != "torch.float32" or math.prod(item["shape"]) != item["elements"]:
            raise ValueError(f"H3 invalid parameter geometry for {name}")
    expected_keys = {
        tensor_key(path, family, name) for path in PATHS for family in STORED_FAMILIES for name in parameter_names
    }
    if set(tensors) != expected_keys or tensor_meta.get("key_count") != len(expected_keys):
        raise ValueError("H3 tensor-evidence key-set drift")
    for item in geometry:
        for path in PATHS:
            for family in STORED_FAMILIES:
                value = tensors[tensor_key(path, family, item["name"])]
                if list(value.shape) != item["shape"] or value.dtype != torch.float32:
                    raise ValueError(f"H3 tensor shape/dtype drift for {path}/{family}/{item['name']}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"H3 nonfinite tensor for {path}/{family}/{item['name']}")
    for path in COMPARISON_PATHS:
        for name in parameter_names:
            if not torch.equal(
                tensors[tensor_key(path, "initial_parameter", name)],
                tensors[tensor_key(PATHS[0], "initial_parameter", name)],
            ):
                raise ValueError(f"H3 initial state differs for {path}/{name}")

    numerical = h3_manifest["numerical_acceptance"]
    recomputed = {}
    for path in COMPARISON_PATHS:
        recomputed[path] = {}
        for family in COMPARISON_FAMILIES:
            recomputed[path][family] = _recompute_comparison(
                tensors, observed_path=path, family=family, parameter_names=parameter_names, numerical=numerical
            )
    _metric_equal(report["comparisons"], recomputed, context="H3 producer comparison summaries")

    clipping = report["clipping"]
    _require_exact_keys(clipping, set(PATHS), context="H3 clipping")
    parameter_element_count = sum(item["elements"] for item in geometry)
    norm_summary_consistency = []
    for path in PATHS:
        pre = _aggregate(
            _family_values(tensors, path=path, family="preclip_gradient", parameter_names=parameter_names),
            parameter_names,
        )
        post = _aggregate(
            _family_values(tensors, path=path, family="clipped_gradient", parameter_names=parameter_names),
            parameter_names,
        )
        pre_norm = float(torch.linalg.vector_norm(pre.double()))
        post_norm = float(torch.linalg.vector_norm(post.double()))
        if not math.isfinite(pre_norm) or pre_norm <= numerical["preclip_norm_must_exceed"]:
            raise ValueError(f"H3 {path} did not exercise active clipping")
        if not math.isfinite(post_norm) or post_norm > numerical["postclip_norm_maximum"]:
            raise ValueError(f"H3 {path} post-clip norm exceeds limit")
        if torch.equal(pre, post):
            raise ValueError(f"H3 {path} clipping did not change gradients")
        _require_exact_keys(
            clipping[path],
            {"clip_grad_norm_return", "postclip_norm_from_evidence", "preclip_norm_from_evidence"},
            context=f"H3 {path} clipping summary",
        )
        for family, producer_value, recomputed_value in (
            ("postclip_norm_from_evidence", clipping[path]["postclip_norm_from_evidence"], post_norm),
            ("preclip_norm_from_evidence", clipping[path]["preclip_norm_from_evidence"], pre_norm),
        ):
            norm_summary_consistency.append(
                {
                    "family": family,
                    "path": path,
                    **validate_norm_summary_cross_backend_consistency(
                        producer_value,
                        recomputed_value,
                        element_count=parameter_element_count,
                        context=f"H3 {path} clipping summary.{family}",
                    ),
                }
            )
        clip_return = clipping[path]["clip_grad_norm_return"]
        if not isinstance(clip_return, (int, float)) or not math.isfinite(clip_return) or clip_return <= 1.0:
            raise ValueError(f"H3 {path} clip_grad_norm return does not prove active clipping")
        norm_metrics = scalar_comparison_metrics(float(clip_return), pre_norm)
        validate_comparison_metrics(norm_metrics, numerical, kind="loss", context=f"H3 {path} clip norm return")
        coefficient = torch.clamp(
            torch.tensor(1.0, dtype=torch.float32) / (torch.tensor(float(clip_return), dtype=torch.float32) + 1e-6),
            max=1.0,
        )
        expected_post = pre.float() * coefficient
        clip_metrics = tensor_comparison_metrics(post, expected_post)
        validate_comparison_metrics(clip_metrics, numerical, kind="gradient", context=f"H3 {path} clip transform")
    for path in COMPARISON_PATHS:
        metrics = scalar_comparison_metrics(
            clipping[path]["preclip_norm_from_evidence"], clipping[PATHS[0]]["preclip_norm_from_evidence"]
        )
        validate_comparison_metrics(metrics, numerical, kind="loss", context=f"H3 {path} preclip norm")

    losses = report["losses"]
    _require_exact_keys(losses, {"comparisons", "global_unscaled", "per_case_unscaled"}, context="H3 losses")
    if set(losses["global_unscaled"]) != set(PATHS) or set(losses["per_case_unscaled"]) != set(PATHS):
        raise ValueError("H3 loss path-set drift")
    for path in PATHS:
        values = losses["per_case_unscaled"][path]
        if (
            not isinstance(values, list)
            or len(values) != len(expected_cases)
            or any(not math.isfinite(v) for v in values)
        ):
            raise ValueError(f"H3 {path} per-case loss evidence drift")
        if not math.isfinite(losses["global_unscaled"][path]):
            raise ValueError(f"H3 {path} global loss is nonfinite")
        accumulated = float(torch.tensor(values, dtype=torch.float32).sum().item())
        accumulation_metrics = scalar_comparison_metrics(losses["global_unscaled"][path], accumulated)
        validate_comparison_metrics(
            accumulation_metrics, numerical, kind="loss", context=f"H3 {path} global/per-case loss accounting"
        )
    recomputed_losses = {}
    for path in COMPARISON_PATHS:
        metrics = scalar_comparison_metrics(losses["global_unscaled"][path], losses["global_unscaled"][PATHS[0]])
        validate_comparison_metrics(metrics, numerical, kind="loss", context=f"H3 {path} global loss")
        recomputed_losses[path] = metrics
    _metric_equal(losses["comparisons"], recomputed_losses, context="H3 loss comparisons")

    scaling = report["scaling"]
    _require_exact_keys(
        scaling, {"ddp_case_losses", "global_target_count", "world_size_multiplier"}, context="H3 scaling"
    )
    if (
        scaling.get("world_size_multiplier") != 4
        or scaling.get("global_target_count") != scenario["global_target_count"]
    ):
        raise ValueError("H3 DDP scaling contract drift")
    rows = scaling.get("ddp_case_losses")
    if not isinstance(rows, list) or [row.get("case_id") for row in rows] != list(range(len(expected_cases))):
        raise ValueError("H3 DDP scaling membership drift")
    for row, case in zip(rows, expected_cases, strict=True):
        _require_exact_keys(
            row,
            {
                "case_id",
                "global_target_count",
                "rank",
                "scaled_backward_loss",
                "slot",
                "synchronized_backward",
                "target_count",
                "unscaled_model_loss",
            },
            context=f"H3 DDP scaling case {case['case_id']}",
        )
        if (
            row.get("target_count") != case["target_count"]
            or row.get("rank") != case["rank"]
            or row.get("slot") != case["slot"]
        ):
            raise ValueError("H3 DDP scaling case identity drift")
        unscaled = row.get("unscaled_model_loss")
        scaled = row.get("scaled_backward_loss")
        if not isinstance(unscaled, (int, float)) or not isinstance(scaled, (int, float)):
            raise ValueError("H3 DDP scaling values are not numeric")
        if not math.isfinite(unscaled) or not math.isfinite(scaled) or scaled != unscaled * 4:
            raise ValueError("H3 DDP world-size scaling was not measured exactly")
        if row.get("global_target_count") != scenario["global_target_count"]:
            raise ValueError("H3 DDP case used a non-global divisor")
        if unscaled != losses["per_case_unscaled"]["ddp_gradient_accumulation"][case["case_id"]]:
            raise ValueError("H3 DDP scaling and per-case loss ledgers disagree")
        expected_sync = case["slot"] == scenario["accumulation_steps"] - 1
        if row.get("synchronized_backward") is not expected_sync:
            raise ValueError("H3 DDP no_sync/final-sync placement drift")

    optimizer = report["optimizer"]
    _require_exact_keys(
        optimizer,
        {
            "ddp_rank_post_step_state_sha256",
            "floating_state_dtypes",
            "foreach",
            "fused",
            "gradient_dtypes",
            "parameter_dtypes",
            "step_counters",
        },
        context="H3 optimizer",
    )
    if (
        optimizer.get("fused") is not True
        or optimizer.get("foreach") is not False
        or optimizer.get("floating_state_dtypes") != ["torch.float32"]
        or optimizer.get("parameter_dtypes") != ["torch.float32"]
        or optimizer.get("gradient_dtypes") != ["torch.float32"]
        or optimizer.get("step_counters") != {path: [1] for path in PATHS}
    ):
        raise ValueError("H3 optimizer path/dtype/counter drift")
    rank_hashes = optimizer.get("ddp_rank_post_step_state_sha256")
    if (
        not isinstance(rank_hashes, list)
        or len(rank_hashes) != 4
        or len(set(rank_hashes)) != 1
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in rank_hashes
        )
    ):
        raise ValueError("H3 DDP rank optimizer states are not bit-identical")

    decision = report["decision"]
    if decision != {
        "all_gating_comparisons_passed": report["status"] == "passed",
        "allowed_successor": "H4_only" if report["status"] == "passed" else None,
        "scientific_training_authorized": False,
    }:
        raise ValueError("H3 decision ledger drift")
    expected_conclusion = (
        "This scenario/candidate passed H3; only completion and independent validation of the full eight-run H3 set may authorize H4."
        if report["status"] == "passed"
        else "This scenario/candidate failed H3; H4 and scientific work remain blocked."
    )
    if report["allowed_conclusion"] != expected_conclusion:
        raise ValueError("H3 allowed-conclusion drift")
    return {
        "status": "passed",
        "scenario_id": report["scenario_id"],
        "chunk_size": report["chunk_size"],
        "cases": len(expected_cases),
        "parameter_tensors": len(parameter_names),
        "tensor_keys": len(expected_keys),
        "named_metric_groups_recomputed": len(COMPARISON_PATHS) * len(COMPARISON_FAMILIES) * len(parameter_names),
        "aggregate_metric_groups_recomputed": len(COMPARISON_PATHS) * len(COMPARISON_FAMILIES),
        "loss_metric_groups_recomputed": len(COMPARISON_PATHS),
        "active_clipping_paths": len(PATHS),
        "norm_summary_consistency": {
            "comparisons": len(norm_summary_consistency),
            "element_count": parameter_element_count,
            "maximum_absolute_difference": max(item["absolute_difference"] for item in norm_summary_consistency),
            "maximum_binary64_ulp_distance": max(
                item["binary64_ulp_distance"] for item in norm_summary_consistency
            ),
            "maximum_relative_difference": max(item["relative_difference"] for item in norm_summary_consistency),
            "records": norm_summary_consistency,
            "relative_bound": norm_summary_cross_backend_relative_bound(parameter_element_count),
        },
    }
