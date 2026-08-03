"""Fail-closed R18 non-Liger qualification manifest resolver."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from open_instruct import qwen35_qualification as r15
from open_instruct import qwen35_qualification_r17 as r17
from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID, QUALIFIED_CHUNK_SIZES

QUALIFICATION_PROTOCOL_ID = "qwen35-hardware-qualification-r18"
BASE_PROTOCOL_ID = "qwen35-hardware-qualification-r17"
BASE_MANIFEST_SHA256 = "67cbbea5ec1c1bd7b982c5f2c35346654a570d69832ac95a4fa93921a385f7d0"
CORRECTIVE_BASELINE_COMMIT = "8f15b85564a17943234b2ceb2bfe89a4482140bb"
HUMAN_PROTOCOL_SHA256 = "8ddcaa7fad78e65fa06500ef62f1028d5bd9d1fdc4acab8241811f9bd77382ef"


def _seed_identity(label: str) -> dict[str, str | int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return {"seed_label": label, "seed_sha256": digest, "seed": int(digest[:8], 16)}


def _expected_direct_cases() -> list[dict[str, Any]]:
    selected_counts = (1, 127, 128, 129, 256, 257, 512, 513, 1024, 1025)
    scales = (0.03125, 1.0, 8.0, 0.03125, 1.0, 8.0, 0.03125, 1.0, 8.0, 1.0)
    return [
        {
            "case_id": f"R18-D{index}",
            "selected_rows": selected_rows,
            **_seed_identity(f"qwen35-hardware-qualification-r18-h2-direct-{index}"),
            "hidden_scale": scales[index],
        }
        for index, selected_rows in enumerate(selected_counts)
    ]


def _expected_real_geometry_case() -> dict[str, Any]:
    return {
        "case_id": "R18-G0",
        "hidden_size": 1024,
        "vocab_size": 248320,
        "selected_rows": 1025,
        "global_divisor": 1062,
        "weight_standard_deviation": 0.02,
        "hidden_scale": 1.0,
        **_seed_identity("qwen35-hardware-qualification-r18-h2-real-geometry-0"),
    }


def _expected_trajectories() -> list[dict[str, Any]]:
    result = []
    for index in range(3):
        prefix = f"qwen35-hardware-qualification-r18-h2-trajectory-{index}"
        model = _seed_identity(prefix)
        batches = _seed_identity(f"{prefix}-batches")
        heldout = _seed_identity(f"{prefix}-heldout")
        result.append(
            {
                "trajectory_id": f"R18-T{index}",
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


def _expected_gates() -> list[dict[str, Any]]:
    names = (
        ("H0", "identity_account_runtime_hardware", []),
        ("H1", "official_text_gpu_parity", ["H0"]),
        ("H2", "checkpointed_chunked_selected_loss_bit_exact_parity", ["H1"]),
        ("H3", "full_model_ddp_gradient_accumulation_parity", ["H2"]),
        ("H4", "real_32k_memory_kernel_path_and_chunk_selection", ["H3"]),
        ("H5", "four_gpu_schedule_nccl_bounded_run", ["H4"]),
        ("H6", "continuous_resume_equality", ["H5"]),
        ("H7", "four_vs_eight_gpu_topology", ["H6"]),
        ("H8", "measured_reporting_profiler_reconciliation", ["H7"]),
        ("H9", "independent_closure_audit", ["H8"]),
    )
    return [
        {"gate_id": gate_id, "name": name, "depends_on": dependencies, "mandatory": True}
        for gate_id, name, dependencies in names
    ]


def _expected_runtime_pins(base: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(base["runtime_pins"])
    for key in ("liger_version", "liger_source_files_sha256", "liger_import_mode", "liger_commit"):
        runtime.pop(key)
    runtime["liger_execution_allowed"] = False
    return runtime


def _validate_h2(h2: dict[str, Any]) -> None:
    expected_keys = {
        "protocol_revision",
        "human_protocol_sha256",
        "production_implementation_id",
        "candidate_chunk_sizes",
        "primary_observed_path",
        "primary_reference_path",
        "primary_acceptance",
        "mandatory_diagnostic_a_observed_path",
        "mandatory_diagnostic_a_reference_path",
        "mandatory_diagnostic_b_observed_path",
        "mandatory_diagnostic_b_reference_path",
        "diagnostic_numerical_discrepancy_is_gating",
        "diagnostic_integrity_and_finiteness_are_mandatory",
        "checkpoint",
        "precision",
        "direct_hidden_size",
        "direct_vocab_size",
        "direct_weight_standard_deviation",
        "direct_global_divisor_extra",
        "direct_cases",
        "real_geometry_case",
        "trajectory_model",
        "trajectory_target_count_cycle",
        "trajectory_steps",
        "trajectories",
        "optimizer",
        "zero_target_graph_connected_finite_exact_zero",
        "saved_tensor_must_exclude_chunk_logits",
        "chunk_function_forward_and_recompute_count_required",
        "all_candidates_must_pass_h2_and_h3",
        "failure_policy",
    }
    if set(h2) != expected_keys:
        raise ValueError("R18 H2 field set drift")
    expected_scalars = {
        "protocol_revision": 5,
        "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
        "production_implementation_id": IMPLEMENTATION_ID,
        "candidate_chunk_sizes": list(QUALIFIED_CHUNK_SIZES),
        "primary_observed_path": "pytorch_checkpointed_chunked_selected_rows",
        "primary_reference_path": "pytorch_ordinary_chunked_selected_rows_same_order",
        "primary_acceptance": "bit_exact_all_recorded_quantities",
        "mandatory_diagnostic_a_observed_path": "pytorch_ordinary_chunked_selected_rows_same_order",
        "mandatory_diagnostic_a_reference_path": "pytorch_unchunked_selected_rows",
        "mandatory_diagnostic_b_observed_path": "pytorch_unchunked_selected_rows",
        "mandatory_diagnostic_b_reference_path": "pytorch_full_rows_ignore_index",
        "diagnostic_numerical_discrepancy_is_gating": False,
        "diagnostic_integrity_and_finiteness_are_mandatory": True,
        "direct_hidden_size": 256,
        "direct_vocab_size": 4096,
        "direct_weight_standard_deviation": 0.02,
        "direct_global_divisor_extra": 37,
        "trajectory_target_count_cycle": [1, 127, 128, 129, 256, 257, 512, 513, 1024, 1025],
        "trajectory_steps": 256,
        "zero_target_graph_connected_finite_exact_zero": True,
        "saved_tensor_must_exclude_chunk_logits": True,
        "chunk_function_forward_and_recompute_count_required": True,
        "all_candidates_must_pass_h2_and_h3": True,
        "failure_policy": "stop_no_threshold_rescue",
    }
    for key, expected in expected_scalars.items():
        if h2.get(key) != expected:
            raise ValueError(f"R18 H2 scalar drift for {key}")
    if h2["checkpoint"] != {
        "use_reentrant": False,
        "preserve_rng_state": True,
        "determinism_check": "default",
        "debug": False,
        "early_stop": True,
    }:
        raise ValueError("R18 checkpoint contract drift")
    if h2["precision"] != {
        "parameters": "torch.float32",
        "gradients": "torch.float32",
        "adamw_moments": "torch.float32",
        "projection_autocast": "torch.bfloat16",
        "cross_entropy_accumulation": "torch.float32",
    }:
        raise ValueError("R18 precision contract drift")
    if h2["direct_cases"] != _expected_direct_cases():
        raise ValueError("R18 direct case or seed drift")
    if h2["real_geometry_case"] != _expected_real_geometry_case():
        raise ValueError("R18 real-geometry case or seed drift")
    if h2["trajectories"] != _expected_trajectories():
        raise ValueError("R18 trajectory or seed drift")
    if h2["trajectory_model"] != {
        "vocab_size": 256,
        "hidden_size": 64,
        "sequence_length": 1056,
        "dropout": 0.0,
        "tied_input_output_embeddings": True,
        "hidden_transformations": 2,
    }:
        raise ValueError("R18 trajectory geometry drift")
    if h2["optimizer"] != {
        "name": "AdamW",
        "learning_rate": 2e-5,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "maximum_gradient_norm": 1.0,
    }:
        raise ValueError("R18 optimizer contract drift")


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
        raise ValueError("R18 overlay top-level field set drift")
    if (
        overlay["schema_version"] != 1
        or overlay["protocol_id"] != QUALIFICATION_PROTOCOL_ID
        or overlay["protocol_date"] != "2026-07-19"
        or overlay["status"] != "ready_for_implementation_validation"
    ):
        raise ValueError("R18 overlay identity/status drift")
    expected_base = {
        "path": "qwen35_hardware_qualification_r17.json",
        "sha256": BASE_MANIFEST_SHA256,
        "protocol_id": BASE_PROTOCOL_ID,
    }
    if overlay["base_manifest"] != expected_base:
        raise ValueError("R18 base-manifest binding drift")
    base_path = path.parent / expected_base["path"]
    if r15.sha256_file(base_path) != BASE_MANIFEST_SHA256:
        raise ValueError("R18 immutable R17 base-manifest bytes drift")
    base, base_digest = r17.load_qualification_manifest(base_path)
    if base_digest != BASE_MANIFEST_SHA256 or base["protocol_id"] != BASE_PROTOCOL_ID:
        raise ValueError("R18 base manifest did not independently validate as R17")

    expected_transformations = {
        "abandon_liger_without_rescoring_r17": True,
        "replace_h2_with_checkpointed_chunked_selected_rows": True,
        "require_bit_exact_same_chunk_reference": True,
        "retain_chunk_shape_differences_as_mandatory_nongating_diagnostics": True,
        "retain_h0_h1_scope_training_and_downstream_gates": True,
    }
    if overlay["transformations"] != expected_transformations:
        raise ValueError("R18 transformation contract drift")
    overrides = overlay["overrides"]
    expected_override_keys = {
        "protocol_id",
        "protocol_date",
        "source",
        "runtime_pins",
        "gates",
        "h2_acceptance",
        "memory_acceptance",
    }
    if not isinstance(overrides, dict) or set(overrides) != expected_override_keys:
        raise ValueError("R18 override scope drift")
    if overrides["protocol_id"] != QUALIFICATION_PROTOCOL_ID or overrides["protocol_date"] != "2026-07-19":
        raise ValueError("R18 override identity drift")
    expected_source = {
        "corrective_baseline_commit": CORRECTIVE_BASELINE_COMMIT,
        "branch": "codex/qwen35-causal-suite",
        "require_clean_worktree": True,
    }
    if overrides["source"] != expected_source:
        raise ValueError("R18 source baseline drift")
    if overrides["runtime_pins"] != _expected_runtime_pins(base):
        raise ValueError("R18 runtime pin drift")
    if overrides["gates"] != _expected_gates():
        raise ValueError("R18 gate sequence drift")
    _validate_h2(overrides["h2_acceptance"])
    expected_memory = {
        **base["memory_acceptance"],
        "chunk_candidates": list(QUALIFIED_CHUNK_SIZES),
        "minimum_measured_updates_per_candidate": 3,
        "selection_metric": "median_synchronized_steady_state_optimizer_update_seconds",
        "tie_fraction": 0.02,
        "tie_break": "smaller_chunk_for_hbm_headroom",
    }
    if overrides["memory_acceptance"] != expected_memory:
        raise ValueError("R18 memory/chunk-selection contract drift")

    effective = copy.deepcopy(base)
    for key in (
        "protocol_id",
        "protocol_date",
        "source",
        "runtime_pins",
        "gates",
        "h2_acceptance",
        "memory_acceptance",
    ):
        effective[key] = copy.deepcopy(overrides[key])
    effective["manifest_derivation"] = {
        "kind": "sha256_bound_replacement_overlay",
        "base_manifest": copy.deepcopy(expected_base),
        "transformations": copy.deepcopy(expected_transformations),
    }
    if effective["scope"] != base["scope"] or effective["training_unit"] != base["training_unit"]:
        raise ValueError("R18 changed scientific scope or training unit")
    if effective["model"] != base["model"] or effective["h1_acceptance"] != base["h1_acceptance"]:
        raise ValueError("R18 changed model or H1")
    if effective["numerical_acceptance"] != base["numerical_acceptance"]:
        raise ValueError("R18 changed inherited diagnostic numerical definitions")
    if effective["scope"]["slurm_account"] != "aifac_f02_434":
        raise ValueError("R18 does not require the personal Slurm account")
    if effective["scope"]["automatic_scientific_training"] is not False:
        raise ValueError("R18 may not authorize automatic scientific training")
    if effective["scope"]["eligible_arm_ids"] != ["C00"]:
        raise ValueError("R18 scope drifted beyond C00")
    if effective["scope"]["forbidden_evaluations"] != ["BFCL", "tau2"]:
        raise ValueError("R18 evaluation scope drift")

    prior_labels = {
        case["seed_label"]
        for case in base["h2_acceptance"].get("historical_direct_cases", [])
        + base["h2_acceptance"].get("confirmatory_direct_cases", [])
        if "seed_label" in case
    }
    prior_labels.update(
        row[key]
        for row in base["h2_acceptance"].get("confirmatory_trajectories", [])
        for key in ("model_seed_label", "batch_seed_label", "heldout_seed_label")
    )
    h2 = effective["h2_acceptance"]
    new_labels = {case["seed_label"] for case in h2["direct_cases"]}
    new_labels.add(h2["real_geometry_case"]["seed_label"])
    new_labels.update(
        row[key]
        for row in h2["trajectories"]
        for key in ("model_seed_label", "batch_seed_label", "heldout_seed_label")
    )
    if len(new_labels) != 20 or new_labels & prior_labels:
        raise ValueError("R18 seed labels are duplicate or overlap predecessors")
    return effective, hashlib.sha256(raw).hexdigest()
