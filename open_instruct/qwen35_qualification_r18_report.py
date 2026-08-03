"""R18 comparison metrics and fail-closed H2 report validation.

This module deliberately does not execute either loss implementation.  The
CUDA producer records evidence; this module checks schema, accounting,
finite-value, exactness, and decision consistency independently of that
producer.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID, QUALIFIED_CHUNK_SIZES

EXACT_METRIC_FIELDS = {
    "shape",
    "elements",
    "observed_dtype",
    "reference_dtype",
    "observed_nonfinite_count",
    "reference_nonfinite_count",
    "value_equal",
    "bitwise_equal",
    "mismatched_values",
    "mismatched_bytes",
    "maximum_absolute_error",
}

DIAGNOSTIC_METRIC_FIELDS = {
    "elements",
    "maximum_absolute_error",
    "relative_l2_error",
    "cosine_similarity",
    "observed_l2_norm",
    "reference_l2_norm",
    "difference_l2_norm",
    "nonfinite_count",
}

DIRECT_FAMILIES = (
    "loss",
    "selected_hidden_gradient",
    "output_head_gradient",
    "raw_adamw_update",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
    "post_step_parameter",
    "heldout_logits",
    "heldout_loss",
)

TRAJECTORY_PARAMETER_FAMILIES = (
    "preclip_gradient",
    "clipped_gradient",
    "raw_adamw_update",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
    "cumulative_parameter_displacement",
    "post_step_parameter_state",
)


def _finite_count(value: torch.Tensor) -> int:
    return int(torch.count_nonzero(~torch.isfinite(value.detach())).item())


def _mismatch_statistics(observed: torch.Tensor, reference: torch.Tensor) -> tuple[int, int, float]:
    """Compute failure evidence in bounded slices; the pass path allocates none."""

    observed_flat = observed.detach().reshape(-1)
    reference_flat = reference.detach().reshape(-1)
    mismatched_values = 0
    mismatched_bytes = 0
    maximum_absolute_error = 0.0
    stride = 1 << 20
    for start in range(0, observed_flat.numel(), stride):
        stop = min(start + stride, observed_flat.numel())
        observed_chunk = observed_flat[start:stop]
        reference_chunk = reference_flat[start:stop]
        mismatched_values += int(torch.count_nonzero(observed_chunk != reference_chunk).item())
        observed_bytes = observed_chunk.contiguous().view(torch.uint8)
        reference_bytes = reference_chunk.contiguous().view(torch.uint8)
        mismatched_bytes += int(torch.count_nonzero(observed_bytes != reference_bytes).item())
        if torch.isfinite(observed_chunk).all() and torch.isfinite(reference_chunk).all():
            error = float((observed_chunk.double() - reference_chunk.double()).abs().max().item())
            maximum_absolute_error = max(maximum_absolute_error, error)
        else:
            maximum_absolute_error = math.inf
    return mismatched_values, mismatched_bytes, maximum_absolute_error


def exact_tensor_comparison_metrics(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Record value and raw-byte equality, including signed-zero distinctions."""

    if observed.shape != reference.shape:
        raise ValueError(f"exact comparison shape mismatch: {tuple(observed.shape)} != {tuple(reference.shape)}")
    if observed.dtype != reference.dtype:
        raise ValueError(f"exact comparison dtype mismatch: {observed.dtype} != {reference.dtype}")
    if observed.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    observed_detached = observed.detach().contiguous()
    reference_detached = reference.detach().contiguous()
    value_equal = bool(torch.equal(observed_detached, reference_detached))
    bitwise_equal = bool(
        torch.equal(observed_detached.reshape(-1).view(torch.uint8), reference_detached.reshape(-1).view(torch.uint8))
    )
    if value_equal and bitwise_equal:
        mismatched_values, mismatched_bytes, maximum_absolute_error = 0, 0, 0.0
    else:
        mismatched_values, mismatched_bytes, maximum_absolute_error = _mismatch_statistics(
            observed_detached, reference_detached
        )
    if not math.isfinite(maximum_absolute_error):
        maximum_absolute_error_value: float | None = None
    else:
        maximum_absolute_error_value = maximum_absolute_error
    return {
        "shape": list(observed.shape),
        "elements": int(observed.numel()),
        "observed_dtype": str(observed.dtype),
        "reference_dtype": str(reference.dtype),
        "observed_nonfinite_count": _finite_count(observed_detached),
        "reference_nonfinite_count": _finite_count(reference_detached),
        "value_equal": value_equal,
        "bitwise_equal": bitwise_equal,
        "mismatched_values": mismatched_values,
        "mismatched_bytes": mismatched_bytes,
        "maximum_absolute_error": maximum_absolute_error_value,
    }


def diagnostic_tensor_comparison_metrics(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """R17-style FP64 metrics computed in bounded chunks on the source device."""

    if observed.shape != reference.shape:
        raise ValueError(f"diagnostic comparison shape mismatch: {tuple(observed.shape)} != {tuple(reference.shape)}")
    if observed.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    observed_flat = observed.detach().reshape(-1)
    reference_flat = reference.detach().reshape(-1)
    nonfinite_count = 0
    maximum_absolute_error = 0.0
    observed_squares: list[float] = []
    reference_squares: list[float] = []
    difference_squares: list[float] = []
    dot_products: list[float] = []
    stride = 1 << 20
    for start in range(0, observed_flat.numel(), stride):
        stop = min(start + stride, observed_flat.numel())
        observed64 = observed_flat[start:stop].double()
        reference64 = reference_flat[start:stop].double()
        finite = torch.isfinite(observed64) & torch.isfinite(reference64)
        nonfinite_count += int(torch.count_nonzero(~finite).item())
        if not bool(finite.all()):
            continue
        difference = observed64 - reference64
        maximum_absolute_error = max(maximum_absolute_error, float(difference.abs().max().item()))
        observed_squares.append(float(torch.dot(observed64, observed64).item()))
        reference_squares.append(float(torch.dot(reference64, reference64).item()))
        difference_squares.append(float(torch.dot(difference, difference).item()))
        dot_products.append(float(torch.dot(observed64, reference64).item()))
    if nonfinite_count:
        return {
            "elements": int(observed.numel()),
            "maximum_absolute_error": None,
            "relative_l2_error": None,
            "cosine_similarity": None,
            "observed_l2_norm": None,
            "reference_l2_norm": None,
            "difference_l2_norm": None,
            "nonfinite_count": nonfinite_count,
        }
    observed_norm = math.sqrt(math.fsum(observed_squares))
    reference_norm = math.sqrt(math.fsum(reference_squares))
    difference_norm = math.sqrt(math.fsum(difference_squares))
    relative = difference_norm / max(reference_norm, torch.finfo(torch.float64).eps)
    if observed_norm == 0 and reference_norm == 0:
        cosine: float | None = 1.0
    elif observed_norm == 0 or reference_norm == 0:
        cosine = None
    else:
        cosine = math.fsum(dot_products) / (observed_norm * reference_norm)
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "elements": int(observed.numel()),
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": relative,
        "cosine_similarity": cosine,
        "observed_l2_norm": observed_norm,
        "reference_l2_norm": reference_norm,
        "difference_l2_norm": difference_norm,
        "nonfinite_count": 0,
    }


def _require_exact_metric(value: Any, *, expected_elements: int | None, context: str) -> None:
    if not isinstance(value, dict) or set(value) != EXACT_METRIC_FIELDS:
        raise ValueError(f"{context}: exact metric schema drift")
    elements = value["elements"]
    shape = value["shape"]
    if not isinstance(elements, int) or elements <= 0 or not isinstance(shape, list):
        raise ValueError(f"{context}: invalid exact metric geometry")
    if math.prod(shape) != elements or (expected_elements is not None and elements != expected_elements):
        raise ValueError(f"{context}: exact metric element accounting drift")
    if value["observed_dtype"] != value["reference_dtype"]:
        raise ValueError(f"{context}: exact metric dtype disagreement")
    if value["observed_nonfinite_count"] != 0 or value["reference_nonfinite_count"] != 0:
        raise ValueError(f"{context}: nonfinite primary evidence")
    if (
        value["value_equal"] is not True
        or value["bitwise_equal"] is not True
        or value["mismatched_values"] != 0
        or value["mismatched_bytes"] != 0
        or value["maximum_absolute_error"] != 0
    ):
        raise ValueError(f"{context}: primary evidence is not bit exact")


def _require_diagnostic_metric(value: Any, *, expected_elements: int | None, context: str) -> None:
    if not isinstance(value, dict) or set(value) != DIAGNOSTIC_METRIC_FIELDS:
        raise ValueError(f"{context}: diagnostic metric schema drift")
    elements = value["elements"]
    if (
        not isinstance(elements, int)
        or elements <= 0
        or (expected_elements is not None and elements != expected_elements)
    ):
        raise ValueError(f"{context}: diagnostic metric element accounting drift")
    if value["nonfinite_count"] != 0:
        raise ValueError(f"{context}: nonfinite diagnostic evidence")
    for key in (
        "maximum_absolute_error",
        "relative_l2_error",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
    ):
        number = value[key]
        if not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError(f"{context}: invalid finite diagnostic {key}")
    cosine = value["cosine_similarity"]
    if cosine is not None and (not math.isfinite(cosine) or not -1.0000001 <= cosine <= 1.0000001):
        raise ValueError(f"{context}: invalid diagnostic cosine")
    denominator = max(float(value["reference_l2_norm"]), torch.finfo(torch.float64).eps)
    expected_relative = float(value["difference_l2_norm"]) / denominator
    if not math.isclose(float(value["relative_l2_error"]), expected_relative, rel_tol=2e-15, abs_tol=0.0):
        raise ValueError(f"{context}: diagnostic relative-L2 arithmetic drift")


def _expected_boundaries(rows: int, chunk_size: int) -> list[list[int]]:
    return [[start, min(start + chunk_size, rows)] for start in range(0, rows, chunk_size)]


def _validate_path_comparison(
    comparison: Any, *, metric_kind: str, expected_elements: dict[str, int], context: str
) -> int:
    if not isinstance(comparison, dict) or set(comparison) != set(DIRECT_FAMILIES):
        raise ValueError(f"{context}: comparison family coverage drift")
    validator = _require_exact_metric if metric_kind == "exact" else _require_diagnostic_metric
    for family in DIRECT_FAMILIES:
        validator(comparison[family], expected_elements=expected_elements[family], context=f"{context} {family}")
    return len(DIRECT_FAMILIES)


def _validate_direct_case(
    result: Any, *, contract: dict[str, Any], chunk_size: int, hidden_size: int, vocab_size: int, context: str
) -> tuple[int, int]:
    required = {
        "case_contract",
        "chunk_size",
        "global_divisor",
        "observed_audit",
        "reference_audit",
        "execution_proof",
        "saved_tensor_proof",
        "autocast_contracts",
        "optimizer_step_counters",
        "primary",
        "diagnostic_a",
        "diagnostic_b",
        "status",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError(f"{context}: direct result schema drift")
    if result["case_contract"] != contract or result["chunk_size"] != chunk_size:
        raise ValueError(f"{context}: direct contract binding drift")
    rows = int(contract.get("selected_rows", contract.get("rows", 0)))
    expected_divisor = int(contract.get("global_divisor", rows + 37))
    if rows <= 0 or result["global_divisor"] != expected_divisor:
        raise ValueError(f"{context}: direct target/divisor accounting drift")
    boundaries = _expected_boundaries(rows, chunk_size)
    expected_audit = {
        "implementation_id": IMPLEMENTATION_ID,
        "selected_rows": rows,
        "chunk_size": chunk_size,
        "chunk_count": len(boundaries),
        "chunk_boundaries": boundaries,
        "maximum_chunk_rows": max(end - start for start, end in boundaries),
        "vocabulary_size": vocab_size,
        "hidden_size": hidden_size,
        "maximum_logit_elements": max(end - start for start, end in boundaries) * vocab_size,
        "full_selected_logit_elements": rows * vocab_size,
        "global_target_count": expected_divisor,
        "zero_target": False,
        "returned_dense_logits": False,
    }
    for checkpointed, key in ((True, "observed_audit"), (False, "reference_audit")):
        expected = {**expected_audit, "checkpointed": checkpointed}
        if result[key] != expected:
            raise ValueError(f"{context}: {key} drift")
    execution = result["execution_proof"]
    if execution != {
        "observed_after_forward": len(boundaries),
        "observed_after_backward": 2 * len(boundaries),
        "reference_after_forward": len(boundaries),
        "reference_after_backward": len(boundaries),
    }:
        raise ValueError(f"{context}: checkpoint recomputation count drift")
    saved = result["saved_tensor_proof"]
    if not isinstance(saved, dict) or set(saved) != {
        "checkpoint_saved_shapes",
        "ordinary_saved_shapes",
        "forbidden_logit_shapes",
        "checkpoint_saved_no_chunk_logits",
        "ordinary_saved_at_least_one_chunk_logit",
    }:
        raise ValueError(f"{context}: saved-tensor evidence schema drift")
    if (
        saved["checkpoint_saved_no_chunk_logits"] is not True
        or saved["ordinary_saved_at_least_one_chunk_logit"] is not True
    ):
        raise ValueError(f"{context}: checkpoint saved-tensor proof failed")
    forbidden = [[end - start, vocab_size] for start, end in boundaries]
    if saved["forbidden_logit_shapes"] != forbidden:
        raise ValueError(f"{context}: forbidden-logit shape accounting drift")
    if any(shape in forbidden for shape in saved["checkpoint_saved_shapes"]):
        raise ValueError(f"{context}: checkpoint retained a chunk logit")
    if not any(shape in forbidden for shape in saved["ordinary_saved_shapes"]):
        raise ValueError(f"{context}: ordinary control did not expose a saved chunk logit")
    expected_autocast = {
        role: {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}
        for role in ("observed", "reference", "unchunked", "full_ignore", "heldout")
    }
    if result["autocast_contracts"] != expected_autocast:
        raise ValueError(f"{context}: BF16 autocast contract drift")
    if result["optimizer_step_counters"] != {
        role: [1] for role in ("observed", "reference", "unchunked", "full_ignore")
    }:
        raise ValueError(f"{context}: AdamW step-counter drift")
    expected_elements = {
        "loss": 1,
        "selected_hidden_gradient": rows * hidden_size,
        "output_head_gradient": vocab_size * hidden_size,
        "raw_adamw_update": vocab_size * hidden_size,
        "optimizer_exp_avg": vocab_size * hidden_size,
        "optimizer_exp_avg_sq": vocab_size * hidden_size,
        "post_step_parameter": vocab_size * hidden_size,
        "heldout_logits": 17 * vocab_size,
        "heldout_loss": 1,
    }
    exact_checks = _validate_path_comparison(
        result["primary"], metric_kind="exact", expected_elements=expected_elements, context=f"{context} primary"
    )
    diagnostic_checks = 0
    for key in ("diagnostic_a", "diagnostic_b"):
        diagnostic_checks += _validate_path_comparison(
            result[key], metric_kind="diagnostic", expected_elements=expected_elements, context=f"{context} {key}"
        )
    if result["status"] != "passed":
        raise ValueError(f"{context}: direct case status is not passed")
    return exact_checks, diagnostic_checks


def _validate_zero_target(value: Any, *, chunk_size: int, h2: dict[str, Any], context: str) -> int:
    required = {
        "chunk_size",
        "loss",
        "loss_value",
        "hidden_gradient",
        "hidden_gradient_nonzero_count",
        "output_head_gradient",
        "output_head_gradient_nonzero_count",
        "execution_counter",
        "audit",
        "autocast_contract",
        "status",
    }
    if not isinstance(value, dict) or set(value) != required or value["chunk_size"] != chunk_size:
        raise ValueError(f"{context}: zero-target schema/identity drift")
    _require_exact_metric(value["loss"], expected_elements=1, context=f"{context} loss")
    _require_exact_metric(
        value["hidden_gradient"], expected_elements=h2["direct_hidden_size"], context=f"{context} hidden gradient"
    )
    _require_exact_metric(
        value["output_head_gradient"],
        expected_elements=h2["direct_hidden_size"] * h2["direct_vocab_size"],
        context=f"{context} head gradient",
    )
    if (
        value["loss_value"] != 0
        or value["hidden_gradient_nonzero_count"] != 0
        or value["output_head_gradient_nonzero_count"] != 0
    ):
        raise ValueError(f"{context}: zero-target loss or gradient is not exact zero")
    if value["execution_counter"] != {}:
        raise ValueError(f"{context}: zero-target path executed a projection")
    audit = value["audit"]
    if (
        audit.get("selected_rows") != 0
        or audit.get("chunk_count") != 0
        or audit.get("maximum_logit_elements") != 0
        or audit.get("zero_target") is not True
    ):
        raise ValueError(f"{context}: zero-target audit drift")
    if value["autocast_contract"] != {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}:
        raise ValueError(f"{context}: zero-target autocast drift")
    if value["status"] != "passed":
        raise ValueError(f"{context}: zero-target status failed")
    return 3


def _validate_trajectory_comparison(
    value: Any,
    *,
    metric_kind: str,
    names: list[str],
    elements: dict[str, int],
    parameter_count: int,
    vocab_size: int,
    sequence_length: int,
    context: str,
) -> int:
    required = {"loss", "preclip_global_norm", "aggregate", "named", "heldout_logits", "heldout_loss"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{context}: trajectory comparison schema drift")
    validator = _require_exact_metric if metric_kind == "exact" else _require_diagnostic_metric
    validator(value["loss"], expected_elements=1, context=f"{context} loss")
    validator(value["preclip_global_norm"], expected_elements=1, context=f"{context} preclip norm")
    validator(
        value["heldout_logits"], expected_elements=sequence_length * vocab_size, context=f"{context} heldout logits"
    )
    validator(value["heldout_loss"], expected_elements=1, context=f"{context} heldout loss")
    aggregate = value["aggregate"]
    named = value["named"]
    if not isinstance(aggregate, dict) or set(aggregate) != set(TRAJECTORY_PARAMETER_FAMILIES):
        raise ValueError(f"{context}: aggregate trajectory coverage drift")
    if not isinstance(named, dict) or set(named) != set(names):
        raise ValueError(f"{context}: named trajectory parameter coverage drift")
    checks = 4
    for family in TRAJECTORY_PARAMETER_FAMILIES:
        validator(aggregate[family], expected_elements=parameter_count, context=f"{context} aggregate {family}")
        checks += 1
    for name in names:
        if not isinstance(named[name], dict) or set(named[name]) != set(TRAJECTORY_PARAMETER_FAMILIES):
            raise ValueError(f"{context}: named family coverage drift for {name}")
        for family in TRAJECTORY_PARAMETER_FAMILIES:
            validator(named[name][family], expected_elements=elements[name], context=f"{context} {name} {family}")
            checks += 1
    return checks


def _validate_trajectory(
    value: Any, *, contract: dict[str, Any], chunk_size: int, h2: dict[str, Any], context: str
) -> tuple[int, int]:
    required = {
        "trajectory_contract",
        "chunk_size",
        "model_definition",
        "parameter_geometry",
        "parameter_count",
        "heldout_contract",
        "steps",
        "status",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{context}: trajectory schema drift")
    if value["trajectory_contract"] != contract or value["chunk_size"] != chunk_size:
        raise ValueError(f"{context}: trajectory contract binding drift")
    if value["model_definition"] != {
        **h2["trajectory_model"],
        "implementation": "embedding_tanh_linear_tanh_linear_tied_output_r1",
    }:
        raise ValueError(f"{context}: trajectory model-definition drift")
    geometry = value["parameter_geometry"]
    if not isinstance(geometry, list) or not geometry:
        raise ValueError(f"{context}: missing parameter geometry")
    names = [row.get("name") for row in geometry]
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{context}: invalid parameter names")
    elements = {row["name"]: row.get("elements") for row in geometry}
    if any(not isinstance(count, int) or count <= 0 for count in elements.values()):
        raise ValueError(f"{context}: invalid parameter element geometry")
    parameter_count = sum(elements.values())
    if value["parameter_count"] != parameter_count:
        raise ValueError(f"{context}: parameter partition drift")
    heldout = value["heldout_contract"]
    if heldout != {
        "seed": contract["heldout_seed"],
        "target_count": 257,
        "global_divisor": 294,
        "sequence_length": h2["trajectory_model"]["sequence_length"],
    }:
        raise ValueError(f"{context}: heldout contract drift")
    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) != h2["trajectory_steps"]:
        raise ValueError(f"{context}: trajectory step count drift")
    exact_checks = 0
    diagnostic_checks = 0
    target_cycle = h2["trajectory_target_count_cycle"]
    for index, step in enumerate(steps):
        step_context = f"{context} step {index + 1}"
        required_step = {
            "step",
            "batch_contract",
            "execution_proof",
            "autocast_contracts",
            "optimizer_step_counters",
            "primary",
            "diagnostic_a",
            "diagnostic_b",
            "status",
        }
        if not isinstance(step, dict) or set(step) != required_step or step["step"] != index + 1:
            raise ValueError(f"{step_context}: step schema/index drift")
        expected_targets = target_cycle[index % len(target_cycle)]
        batch = step["batch_contract"]
        if (
            not isinstance(batch, dict)
            or batch.get("seed") != (contract["batch_seed_base"] + index) % (2**32)
            or batch.get("target_count") != expected_targets
            or batch.get("global_divisor") != expected_targets + 37
            or set(batch)
            != {"seed", "target_count", "global_divisor", "input_ids_sha256", "positions_sha256", "targets_sha256"}
        ):
            raise ValueError(f"{step_context}: batch accounting/hash schema drift")
        if any(
            not isinstance(batch[key], str) or len(batch[key]) != 64
            for key in ("input_ids_sha256", "positions_sha256", "targets_sha256")
        ):
            raise ValueError(f"{step_context}: invalid batch digest")
        chunks = math.ceil(expected_targets / chunk_size)
        if step["execution_proof"] != {
            "observed_after_forward": chunks,
            "observed_after_backward": 2 * chunks,
            "reference_after_forward": chunks,
            "reference_after_backward": chunks,
        }:
            raise ValueError(f"{step_context}: trajectory recomputation count drift")
        expected_autocast = {
            role: {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}
            for role in ("observed", "reference", "unchunked", "full_ignore", "heldout")
        }
        if step["autocast_contracts"] != expected_autocast:
            raise ValueError(f"{step_context}: trajectory autocast drift")
        expected_steps = [index + 1]
        if step["optimizer_step_counters"] != {
            role: expected_steps for role in ("observed", "reference", "unchunked", "full_ignore")
        }:
            raise ValueError(f"{step_context}: trajectory optimizer counter drift")
        kwargs = {
            "names": names,
            "elements": elements,
            "parameter_count": parameter_count,
            "vocab_size": h2["trajectory_model"]["vocab_size"],
            "sequence_length": h2["trajectory_model"]["sequence_length"],
        }
        exact_checks += _validate_trajectory_comparison(
            step["primary"], metric_kind="exact", context=f"{step_context} primary", **kwargs
        )
        for key in ("diagnostic_a", "diagnostic_b"):
            diagnostic_checks += _validate_trajectory_comparison(
                step[key], metric_kind="diagnostic", context=f"{step_context} {key}", **kwargs
            )
        if step["status"] != "passed":
            raise ValueError(f"{step_context}: trajectory step status failed")
    if value["status"] != "passed":
        raise ValueError(f"{context}: trajectory status failed")
    return exact_checks, diagnostic_checks


def validate_h2_chunked_report(
    report: dict[str, Any], *, qualification: dict[str, Any], expected_manifest_sha256: str
) -> dict[str, Any]:
    """Validate complete successful R18 H2 evidence without executing its producer."""

    required_top = {
        "artifact",
        "schema_version",
        "qualification_protocol_id",
        "qualification_manifest_sha256",
        "manifest_derivation",
        "source_attestation",
        "environment",
        "primary_comparison",
        "diagnostic_comparisons",
        "candidate_results",
        "status",
        "successor_gate_authorized",
        "scientific_training_authorized",
        "allowed_conclusion",
    }
    if not isinstance(report, dict) or set(report) != required_top:
        raise ValueError("R18 H2 report top-level schema drift")
    if report["artifact"] != "qwen35_checkpointed_chunked_selected_loss_qualification_r18":
        raise ValueError("R18 H2 report artifact identity drift")
    if report["schema_version"] != 1 or report["qualification_protocol_id"] != qualification["protocol_id"]:
        raise ValueError("R18 H2 report schema/protocol drift")
    if report["qualification_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("R18 H2 report manifest digest drift")
    if report["manifest_derivation"] != qualification["manifest_derivation"]:
        raise ValueError("R18 H2 report derivation drift")
    source = report["source_attestation"]
    if not isinstance(source, dict) or set(source) != {
        "git_commit",
        "git_worktree_clean",
        "implementation_id",
        "source_files_sha256",
    }:
        raise ValueError("R18 H2 source-attestation schema drift")
    if source["git_worktree_clean"] is not True or source["implementation_id"] != IMPLEMENTATION_ID:
        raise ValueError("R18 H2 source identity/cleanliness drift")
    if not isinstance(source["git_commit"], str) or len(source["git_commit"]) != 40:
        raise ValueError("R18 H2 invalid source commit")
    if not isinstance(source["source_files_sha256"], dict) or not source["source_files_sha256"]:
        raise ValueError("R18 H2 missing source file hashes")
    if any(not isinstance(value, str) or len(value) != 64 for value in source["source_files_sha256"].values()):
        raise ValueError("R18 H2 invalid source file hash")
    environment = report["environment"]
    if (
        not isinstance(environment, dict)
        or environment.get("device_type") != "cuda"
        or "A100" not in environment.get("cuda_device", "")
        or environment.get("torch_version") != qualification["runtime_pins"]["torch_version"]
        or environment.get("torch_cuda_build") != qualification["runtime_pins"]["torch_cuda_build"]
        or environment.get("liger_imported") is not False
    ):
        raise ValueError("R18 H2 runtime/device attestation drift")
    h2 = qualification["h2_acceptance"]
    if report["primary_comparison"] != {
        "observed_path": h2["primary_observed_path"],
        "reference_path": h2["primary_reference_path"],
        "acceptance": h2["primary_acceptance"],
        "numerical_discrepancy_is_gating": True,
    }:
        raise ValueError("R18 H2 primary comparison identity drift")
    if report["diagnostic_comparisons"] != {
        "a": {
            "observed_path": h2["mandatory_diagnostic_a_observed_path"],
            "reference_path": h2["mandatory_diagnostic_a_reference_path"],
        },
        "b": {
            "observed_path": h2["mandatory_diagnostic_b_observed_path"],
            "reference_path": h2["mandatory_diagnostic_b_reference_path"],
        },
        "numerical_discrepancy_is_gating": False,
        "integrity_and_finiteness_are_mandatory": True,
    }:
        raise ValueError("R18 H2 diagnostic identity drift")
    candidates = report["candidate_results"]
    if not isinstance(candidates, list) or [row.get("chunk_size") for row in candidates] != list(
        QUALIFIED_CHUNK_SIZES
    ):
        raise ValueError("R18 H2 candidate coverage/order drift")
    exact_checks = 0
    diagnostic_checks = 0
    trajectory_steps = 0
    for candidate in candidates:
        required_candidate = {
            "chunk_size",
            "zero_target",
            "qwen_forward_integration",
            "direct_cases",
            "real_geometry_case",
            "trajectories",
            "status",
        }
        if not isinstance(candidate, dict) or set(candidate) != required_candidate:
            raise ValueError("R18 H2 candidate schema drift")
        chunk_size = candidate["chunk_size"]
        if candidate["status"] != "passed":
            raise ValueError(f"R18 H2 chunk {chunk_size} did not pass")
        exact_checks += _validate_zero_target(
            candidate["zero_target"], chunk_size=chunk_size, h2=h2, context=f"chunk {chunk_size} zero target"
        )
        integration = candidate["qwen_forward_integration"]
        if not isinstance(integration, dict) or set(integration) != {
            "chunk_size",
            "model_class",
            "attention_implementation",
            "forward_module",
            "loss",
            "named_parameter_gradients",
            "returned_logits_is_none",
            "audit",
            "status",
        }:
            raise ValueError(f"chunk {chunk_size}: Qwen integration schema drift")
        if (
            integration["chunk_size"] != chunk_size
            or integration["model_class"] != "Qwen3_5ForCausalLM"
            or integration["attention_implementation"] != "eager"
            or integration["forward_module"] != "open_instruct.qwen35_chunked_loss"
            or integration["returned_logits_is_none"] is not True
            or integration["audit"].get("chunk_size") != chunk_size
            or integration["status"] != "passed"
        ):
            raise ValueError(f"chunk {chunk_size}: Qwen production-forward integration failed")
        _require_exact_metric(integration["loss"], expected_elements=1, context=f"chunk {chunk_size} Qwen loss")
        gradients = integration["named_parameter_gradients"]
        if not isinstance(gradients, dict) or not gradients:
            raise ValueError(f"chunk {chunk_size}: missing Qwen gradient evidence")
        for name, metric in gradients.items():
            _require_exact_metric(metric, expected_elements=None, context=f"chunk {chunk_size} Qwen {name}")
            exact_checks += 1
        exact_checks += 1
        direct = candidate["direct_cases"]
        if not isinstance(direct, list) or len(direct) != len(h2["direct_cases"]):
            raise ValueError(f"chunk {chunk_size}: direct case coverage drift")
        for index, (result, contract) in enumerate(zip(direct, h2["direct_cases"], strict=True)):
            exact, diagnostic = _validate_direct_case(
                result,
                contract=contract,
                chunk_size=chunk_size,
                hidden_size=h2["direct_hidden_size"],
                vocab_size=h2["direct_vocab_size"],
                context=f"chunk {chunk_size} direct {index}",
            )
            exact_checks += exact
            diagnostic_checks += diagnostic
        real_contract = h2["real_geometry_case"]
        exact, diagnostic = _validate_direct_case(
            candidate["real_geometry_case"],
            contract=real_contract,
            chunk_size=chunk_size,
            hidden_size=real_contract["hidden_size"],
            vocab_size=real_contract["vocab_size"],
            context=f"chunk {chunk_size} real geometry",
        )
        exact_checks += exact
        diagnostic_checks += diagnostic
        trajectories = candidate["trajectories"]
        if not isinstance(trajectories, list) or len(trajectories) != len(h2["trajectories"]):
            raise ValueError(f"chunk {chunk_size}: trajectory coverage drift")
        for index, (trajectory, contract) in enumerate(zip(trajectories, h2["trajectories"], strict=True)):
            exact, diagnostic = _validate_trajectory(
                trajectory,
                contract=contract,
                chunk_size=chunk_size,
                h2=h2,
                context=f"chunk {chunk_size} trajectory {index}",
            )
            exact_checks += exact
            diagnostic_checks += diagnostic
            trajectory_steps += h2["trajectory_steps"]
    if (
        report["status"] != "passed"
        or report["successor_gate_authorized"] is not True
        or report["scientific_training_authorized"] is not False
        or report["allowed_conclusion"]
        != "R18 H2 passed; H3 may begin, while scientific training and evaluation remain unauthorized."
    ):
        raise ValueError("R18 H2 overall decision/scope drift")
    return {
        "status": "passed",
        "candidate_count": len(candidates),
        "exact_metric_groups": exact_checks,
        "diagnostic_metric_groups": diagnostic_checks,
        "trajectory_steps": trajectory_steps,
        "successor_gate_authorized": True,
        "scientific_training_authorized": False,
    }
