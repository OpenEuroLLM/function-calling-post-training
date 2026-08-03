"""Shared fail-closed contracts for Qwen3.5 hardware qualification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import torch

QUALIFICATION_PROTOCOL_ID = "qwen35-hardware-qualification-r15"
H1_PACKED_LOGIT_ATOL = 0.05
H1_PACKED_LOGIT_RTOL = 0.01
H1_FROZEN_MODEL_CONTRACT = {
    "vocabulary_size": 248320,
    "text_hidden_size": 1024,
    "text_num_hidden_layers": 24,
    "text_layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6,
    "text_state_tensor_count": 321,
    "text_state_numel": 1_006_672_704,
}
EVIDENCE_SERIALIZATION_CONTRACT = {
    "format": "strict_json_rfc8259_no_nan_or_infinity",
    "sets": "recursively_canonicalized_to_content_sorted_arrays",
    "tuples": "recursively_canonicalized_to_arrays",
    "non_string_mapping_keys": "rejected",
    "unsupported_objects": "rejected",
    "write_mode": "atomic_replace_before_scientific_failure_raise",
}


def parse_glibc_versions(readelf_output: str) -> list[list[int]]:
    return [
        list(value)
        for value in sorted({tuple(map(int, match)) for match in re.findall(r"GLIBC_(\d+)\.(\d+)", readelf_output)})
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qualification_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported qualification-manifest schema")
    if manifest.get("protocol_id") != QUALIFICATION_PROTOCOL_ID:
        raise ValueError("unexpected qualification protocol")
    if manifest.get("status") != "ready_for_execution":
        raise ValueError("qualification manifest is not ready for execution")
    if manifest.get("scope", {}).get("slurm_account") != "aifac_f02_434":
        raise ValueError("qualification manifest does not require the personal Slurm account")
    gates = manifest.get("gates")
    if not isinstance(gates, list) or [gate.get("gate_id") for gate in gates] != [f"H{i}" for i in range(10)]:
        raise ValueError("qualification manifest must contain ordered gates H0 through H9")
    seen: set[str] = set()
    for gate in gates:
        gate_id = str(gate["gate_id"])
        if gate.get("mandatory") is not True:
            raise ValueError(f"qualification gate {gate_id} is not mandatory")
        dependencies = gate.get("depends_on")
        if not isinstance(dependencies, list) or any(dependency not in seen for dependency in dependencies):
            raise ValueError(f"qualification gate {gate_id} has invalid dependency order")
        seen.add(gate_id)
    numerical = manifest.get("numerical_acceptance", {})
    required_positive = (
        "packed_logit_absolute_tolerance",
        "packed_logit_relative_tolerance",
        "loss_maximum_absolute_error",
        "loss_relative_error",
        "gradient_maximum_absolute_error",
        "gradient_relative_l2_error",
        "update_relative_l2_error",
    )
    if any(not isinstance(numerical.get(key), (float, int)) or numerical[key] <= 0 for key in required_positive):
        raise ValueError("qualification numerical tolerances must be positive")
    if (
        numerical.get("packed_logit_absolute_tolerance") != H1_PACKED_LOGIT_ATOL
        or numerical.get("packed_logit_relative_tolerance") != H1_PACKED_LOGIT_RTOL
    ):
        raise ValueError("R15 packed-logit tolerances must remain identical to R9")
    if numerical.get("heldout_logit_minimum_cosine_similarity") != 0.9999:
        raise ValueError("R15 held-out-logit cosine threshold drift")
    for key in ("gradient_minimum_cosine_similarity", "update_minimum_cosine_similarity"):
        if not isinstance(numerical.get(key), (float, int)) or not 0 < numerical[key] <= 1:
            raise ValueError("qualification cosine threshold must be in (0, 1]")
    h1 = manifest.get("h1_acceptance", {})
    required_h1_true = (
        "conditional_text_state_tensor_bit_exact",
        "conditional_text_ordinary_logits_bit_exact",
        "conditional_text_dense_loss_bit_exact",
        "singleton_multi_pack_shape_diagnostic_uses_packed_logit_tolerances",
        "single_token_counterfactual_unchanged_segment_bit_exact",
        "full_document_counterfactual_unchanged_segment_bit_exact",
        "all_counterfactual_decoder_layers_bit_exact",
        "duplicate_document_reset_invariance_bit_exact",
        "packed_order_invariance_bit_exact",
        "corrupted_boundary_negative_control_must_show_cross_document_influence",
    )
    if any(h1.get(key) is not True for key in required_h1_true):
        raise ValueError("qualification H1 exact/kernel-matched acceptance is incomplete")
    if h1.get("ordinary_vs_variable_length_cross_kernel_diagnostic_is_gating") is not False:
        raise ValueError("ordinary-versus-variable-length H1 diagnostic must remain non-gating")
    if h1.get("singleton_multi_pack_shape_diagnostic_is_gating") is not False:
        raise ValueError("singleton-versus-multi pack-shape diagnostic must remain non-gating")
    if h1.get("r11_failed_singleton_multi_criterion_reclassified_as_pass") is not False:
        raise ValueError("R15 may not relabel the failed R11 criterion as a pass")
    if h1.get("tolerance_change_from_r9") is not False:
        raise ValueError("R15 may not relax the R9 numerical tolerances")
    model = manifest.get("model", {})
    for key, expected in H1_FROZEN_MODEL_CONTRACT.items():
        if model.get(key) != expected:
            raise ValueError(f"R15 frozen model contract drift for {key}")
    h2 = manifest.get("h2_acceptance", {})
    historical_direct_cases = [
        {
            "case_id": "R14-D0",
            "seed": 1,
            "rows": 64,
            "supervision_kind": "all",
            "supervised_rows": [],
            "expected_supervised_count": 64,
            "global_divisor": 64,
            "hidden_scale": 1.0,
            "weight_standard_deviation": 0.02,
        },
        {
            "case_id": "R14-D1",
            "seed": 2,
            "rows": 64,
            "supervision_kind": "explicit",
            "supervised_rows": [0, 7, 31, 63],
            "expected_supervised_count": 4,
            "global_divisor": 23,
            "hidden_scale": 1.0,
            "weight_standard_deviation": 0.02,
        },
        {
            "case_id": "R14-D2",
            "seed": 3,
            "rows": 64,
            "supervision_kind": "explicit",
            "supervised_rows": [63],
            "expected_supervised_count": 1,
            "global_divisor": 128,
            "hidden_scale": 1.0,
            "weight_standard_deviation": 0.02,
        },
    ]

    def seed_identity(label: str) -> dict[str, str | int]:
        digest = hashlib.sha256(label.encode()).hexdigest()
        return {"seed_label": label, "seed_sha256": digest, "seed": int(digest[:8], 16)}

    confirmatory_direct_cases = [
        {
            "case_id": "R15-D0",
            **seed_identity("qwen35-hardware-qualification-r15-h2-direct-0"),
            "rows": 96,
            "supervision_kind": "explicit",
            "supervised_rows": [0, 5, 17, 31, 47, 63, 79, 95],
            "expected_supervised_count": 8,
            "global_divisor": 37,
            "hidden_scale": 0.25,
            "weight_standard_deviation": 0.02,
        },
        {
            "case_id": "R15-D1",
            **seed_identity("qwen35-hardware-qualification-r15-h2-direct-1"),
            "rows": 128,
            "supervision_kind": "explicit",
            "supervised_rows": [1, 2, 7, 15, 32, 63, 64, 95, 126],
            "expected_supervised_count": 9,
            "global_divisor": 73,
            "hidden_scale": 1.0,
            "weight_standard_deviation": 0.02,
        },
        {
            "case_id": "R15-D2",
            **seed_identity("qwen35-hardware-qualification-r15-h2-direct-2"),
            "rows": 65,
            "supervision_kind": "explicit",
            "supervised_rows": [0, 1, 32, 63, 64],
            "expected_supervised_count": 5,
            "global_divisor": 131,
            "hidden_scale": 4.0,
            "weight_standard_deviation": 0.02,
        },
    ]
    trajectory_model_config = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "layer_types": ["full_attention"],
        "tie_word_embeddings": True,
        "attention_dropout": 0.0,
    }
    trajectory_parameter_geometry = [
        {"name": "model.embed_tokens.weight", "shape": [256, 64], "elements": 16_384},
        {"name": "model.layers.0.self_attn.q_proj.weight", "shape": [128, 64], "elements": 8_192},
        {"name": "model.layers.0.self_attn.k_proj.weight", "shape": [32, 64], "elements": 2_048},
        {"name": "model.layers.0.self_attn.v_proj.weight", "shape": [32, 64], "elements": 2_048},
        {"name": "model.layers.0.self_attn.o_proj.weight", "shape": [64, 64], "elements": 4_096},
        {"name": "model.layers.0.self_attn.q_norm.weight", "shape": [16], "elements": 16},
        {"name": "model.layers.0.self_attn.k_norm.weight", "shape": [16], "elements": 16},
        {"name": "model.layers.0.mlp.gate_proj.weight", "shape": [128, 64], "elements": 8_192},
        {"name": "model.layers.0.mlp.up_proj.weight", "shape": [128, 64], "elements": 8_192},
        {"name": "model.layers.0.mlp.down_proj.weight", "shape": [64, 128], "elements": 8_192},
        {"name": "model.layers.0.input_layernorm.weight", "shape": [64], "elements": 64},
        {"name": "model.layers.0.post_attention_layernorm.weight", "shape": [64], "elements": 64},
        {"name": "model.norm.weight", "shape": [64], "elements": 64},
    ]
    trajectories = []
    for index in range(3):
        model_label = f"qwen35-hardware-qualification-r15-h2-trajectory-{index}"
        batch_label = f"{model_label}-batches"
        heldout_label = f"{model_label}-heldout"
        model_identity = seed_identity(model_label)
        batch_identity = seed_identity(batch_label)
        heldout_identity = seed_identity(heldout_label)
        trajectories.append(
            {
                "trajectory_id": f"R15-T{index}",
                "model_seed_label": model_identity["seed_label"],
                "model_seed_sha256": model_identity["seed_sha256"],
                "model_seed": model_identity["seed"],
                "batch_seed_label": batch_identity["seed_label"],
                "batch_seed_sha256": batch_identity["seed_sha256"],
                "batch_seed_base": batch_identity["seed"],
                "heldout_seed_label": heldout_identity["seed_label"],
                "heldout_seed_sha256": heldout_identity["seed_sha256"],
                "heldout_seed": heldout_identity["seed"],
            }
        )
    expected_h2_scalars = {
        "protocol_revision": 2,
        "direct_hidden_size": 256,
        "direct_vocab_size": 4096,
        "direct_heldout_rows": 17,
        "trajectory_steps": 32,
        "trajectory_sequence_length": 32,
        "trajectory_supervision_moduli": [2, 3, 5, 7],
        "trajectory_divisor_extra_modulus": 13,
        "trajectory_divisor_extra_multiplier": 3,
        "trajectory_heldout_supervision_modulus": 4,
        "trajectory_heldout_divisor_extra": 5,
        "trajectory_parameter_count": 57_568,
        "raw_first_step_update_is_gating": False,
        "raw_update_gating_starts_at_step": 2,
        "post_step_parameter_state_is_gating": True,
        "heldout_function_is_gating": True,
        "direct_fused_and_dense_reference_use_bf16_autocast": True,
        "hidden_input_dtype": "torch.bfloat16",
        "output_head_parameter_dtype": "torch.float32",
        "loss_accumulation_dtype": "torch.float32",
        "gradient_dtype_by_parameter_storage": True,
        "adamw_moment_dtype": "torch.float32",
        "zero_target_graph_connected_finite_zero": True,
        "patched_qwen_forward_uses_bf16_autocast": True,
        "r14_failed_first_step_update_reclassified_as_pass": False,
        "independent_report_validation_required": True,
        "evidence_complete_failure_report_required": True,
    }
    observed_h2_scalars = {
        key: value
        for key, value in h2.items()
        if key
        not in {
            "historical_direct_cases",
            "confirmatory_direct_cases",
            "trajectory_model_config",
            "trajectory_parameter_geometry",
            "confirmatory_trajectories",
        }
    }
    if observed_h2_scalars != expected_h2_scalars:
        raise ValueError("R15 H2 scalar acceptance contract drift")
    if h2.get("historical_direct_cases") != historical_direct_cases:
        raise ValueError("R15 H2 historical direct-case contract drift")
    if h2.get("confirmatory_direct_cases") != confirmatory_direct_cases:
        raise ValueError("R15 H2 confirmatory direct-case/seed contract drift")
    if h2.get("trajectory_model_config") != trajectory_model_config:
        raise ValueError("R15 H2 trajectory model-config contract drift")
    if h2.get("trajectory_parameter_geometry") != trajectory_parameter_geometry:
        raise ValueError("R15 H2 trajectory parameter-geometry contract drift")
    if h2.get("confirmatory_trajectories") != trajectories:
        raise ValueError("R15 H2 trajectory holdout-seed contract drift")
    liger_source_files = manifest.get("runtime_pins", {}).get("liger_source_files_sha256")
    if not isinstance(liger_source_files, dict) or set(liger_source_files) != {
        "transformers/fused_linear_cross_entropy.py",
        "ops/fused_linear_cross_entropy.py",
        "ops/utils.py",
    }:
        raise ValueError("R15 Liger executed-source file set is incomplete")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in liger_source_files.values()
    ):
        raise ValueError("R15 Liger executed-source hash is missing or malformed")
    serialization = manifest.get("evidence_serialization", {})
    if serialization != EVIDENCE_SERIALIZATION_CONTRACT:
        raise ValueError("R15 evidence-serialization contract drift")
    memory = manifest.get("memory_acceptance", {})
    reserved = memory.get("maximum_peak_reserved_fraction")
    headroom = memory.get("minimum_headroom_fraction")
    if not isinstance(reserved, (float, int)) or not isinstance(headroom, (float, int)):
        raise ValueError("qualification memory fractions are missing")
    if not math.isclose(float(reserved) + float(headroom), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("qualification memory fractions do not complement each other")
    hardware = manifest.get("hardware_acceptance", {})
    if not hardware.get("gpu_name_contains") or int(hardware.get("minimum_device_memory_bytes", 0)) <= 0:
        raise ValueError("qualification hardware identity is incomplete")
    if hardware.get("compute_capability") != [8, 0]:
        raise ValueError("R15 qualification must remain pinned to NVIDIA Ampere compute capability 8.0")
    fixture = manifest.get("reference_fixture", {})
    if fixture.get("examples") != 128:
        raise ValueError("qualification reference fixture must contain 128 cases")
    for key in ("fixture_sha256", "cases_sha256", "lineage_manifest_sha256"):
        if not isinstance(fixture.get(key), str) or len(fixture[key]) != 64:
            raise ValueError(f"qualification reference fixture lacks {key}")
    return manifest, hashlib.sha256(raw).hexdigest()


def _require_h1_metric_geometry(metrics: Any, *, shape: list[int], context: str) -> None:
    if not isinstance(metrics, dict):
        raise ValueError(f"{context} metrics are missing")
    elements = math.prod(shape)
    if metrics.get("shape") != shape or metrics.get("elements") != elements or elements <= 0:
        raise ValueError(f"{context} tensor geometry drift")
    if metrics.get("nonfinite_count") != 0:
        raise ValueError(f"{context} contains non-finite values")
    if not isinstance(metrics.get("bit_exact"), bool) or not isinstance(metrics.get("allclose"), bool):
        raise ValueError(f"{context} lacks Boolean exactness decisions")
    mismatched = metrics.get("mismatched_elements")
    fraction = metrics.get("mismatched_fraction")
    if (
        not isinstance(mismatched, int)
        or not 0 <= mismatched <= elements
        or not isinstance(fraction, (float, int))
        or not math.isfinite(float(fraction))
        or not math.isclose(float(fraction), mismatched / elements, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise ValueError(f"{context} mismatch accounting drift")
    for key in (
        "maximum_absolute_error",
        "mean_absolute_error",
        "relative_l2_error",
        "observed_l2_norm",
        "reference_l2_norm",
        "difference_l2_norm",
    ):
        value = metrics.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{context} lacks finite non-negative {key}")
    cosine = metrics.get("cosine_similarity")
    if (
        not isinstance(cosine, (float, int))
        or not math.isfinite(float(cosine))
        or not -1.000000000001 <= cosine <= 1.000000000001
    ):
        raise ValueError(f"{context} lacks a finite cosine similarity")
    quantiles = metrics.get("absolute_error_quantiles")
    if not isinstance(quantiles, dict) or set(quantiles) != {"p50", "p90", "p99", "p99_9"}:
        raise ValueError(f"{context} absolute-error quantile coverage drift")
    quantile_values = [quantiles[key] for key in ("p50", "p90", "p99", "p99_9")]
    if any(
        not isinstance(value, (float, int)) or not math.isfinite(float(value)) or value < 0
        for value in quantile_values
    ):
        raise ValueError(f"{context} contains invalid absolute-error quantiles")
    if quantile_values != sorted(quantile_values) or quantile_values[-1] > metrics["maximum_absolute_error"]:
        raise ValueError(f"{context} absolute-error quantiles are inconsistent")
    top1 = metrics.get("top1_agreement")
    if not isinstance(top1, (float, int)) or not math.isfinite(float(top1)) or not 0 <= top1 <= 1:
        raise ValueError(f"{context} top-1 agreement is invalid")


def _require_exact_h1_metrics(metrics: Any, *, shape: list[int], context: str) -> None:
    _require_h1_metric_geometry(metrics, shape=shape, context=context)
    if (
        metrics.get("bit_exact") is not True
        or metrics.get("allclose") is not True
        or metrics.get("nonfinite_count") != 0
        or metrics.get("mismatched_elements") != 0
        or metrics.get("mismatched_fraction") != 0
        or metrics.get("maximum_absolute_error") != 0
        or metrics.get("difference_l2_norm") != 0
        or metrics.get("mean_absolute_error") != 0
        or metrics.get("relative_l2_error") != 0
        or any(value != 0 for value in metrics["absolute_error_quantiles"].values())
        or metrics.get("top1_agreement") != 1
        or metrics.get("atol") != 0
        or metrics.get("rtol") != 0
    ):
        raise ValueError(f"{context} is not finite and bit-exact")


def validate_h1_reference_report(report: dict[str, Any], *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Independently validate every mandatory R15 H1 claim in a saved report."""

    if report.get("qualification_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("H1 report qualification-manifest identity drift")
    parity = report.get("model_parity")
    if not isinstance(parity, dict) or parity.get("status") != "pass" or parity.get("failures") != []:
        raise ValueError("H1 model parity did not pass without failures")
    if parity.get("source_config_model_type") != "qwen3_5":
        raise ValueError("H1 did not begin from the official conditional Qwen3.5 checkpoint")
    if parity.get("production_model_class") != "Qwen3_5ForCausalLM":
        raise ValueError("H1 did not exercise the text-only production model class")
    if any(
        parity.get(key) != H1_FROZEN_MODEL_CONTRACT[key]
        for key in ("vocabulary_size", "text_hidden_size", "text_num_hidden_layers", "text_layer_types")
    ):
        raise ValueError("H1 frozen production-model geometry drift")
    if parity.get("standalone_reference_definition") != (
        "one document executed with the exact production packed metadata/kernel path"
    ):
        raise ValueError("H1 standalone-reference definition drift")
    if parity.get("atol") != H1_PACKED_LOGIT_ATOL or parity.get("rtol") != H1_PACKED_LOGIT_RTOL:
        raise ValueError("H1 top-level packed-logit tolerance drift")
    sequence_lengths = parity.get("sequence_lengths")
    if (
        not isinstance(sequence_lengths, list)
        or len(sequence_lengths) != 2
        or any(not isinstance(value, int) or value < 3 for value in sequence_lengths)
    ):
        raise ValueError("H1 two-document synthetic sequence coverage drift")

    conversion = parity.get("conditional_to_text_conversion")
    if not isinstance(conversion, dict) or conversion.get("checked") is not True or conversion.get("status") != "pass":
        raise ValueError("H1 conditional-to-text conversion parity did not pass")
    if conversion.get("atol") != 0 or conversion.get("rtol") != 0:
        raise ValueError("H1 conditional-to-text tolerance is not exact")
    loading_info = conversion.get("loading_info")
    if not isinstance(loading_info, dict) or any(
        loading_info.get(key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("H1 conditional-to-text checkpoint loading was incomplete")
    if conversion.get("loading_info_serialization") != EVIDENCE_SERIALIZATION_CONTRACT:
        raise ValueError("H1 loading-info strict-JSON normalization contract drift")

    state_mapping = conversion.get("state_mapping")
    if not isinstance(state_mapping, dict) or state_mapping.get("status") != "pass":
        raise ValueError("H1 conditional-to-text state mapping did not pass")
    if state_mapping.get("mismatched_target_keys") != []:
        raise ValueError("H1 conditional-to-text state mapping contains mismatches")
    rows = state_mapping.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != H1_FROZEN_MODEL_CONTRACT["text_state_tensor_count"]
        or state_mapping.get("target_tensor_count") != len(rows)
    ):
        raise ValueError("H1 state-mapping row coverage drift")
    target_keys = [row.get("target_key") for row in rows if isinstance(row, dict)]
    if len(target_keys) != len(rows) or len(set(target_keys)) != len(rows):
        raise ValueError("H1 state-mapping target keys are missing or duplicated")
    for row in rows:
        if (
            row.get("bit_exact") is not True
            or row.get("target_sha256") != row.get("source_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("target_sha256", "")))
            or row.get("shape") != row.get("source_shape")
            or row.get("dtype") != row.get("source_dtype")
            or not isinstance(row.get("numel"), int)
            or row["numel"] <= 0
            or not isinstance(row.get("shape"), list)
            or not row["shape"]
            or any(not isinstance(dimension, int) or dimension <= 0 for dimension in row["shape"])
            or math.prod(row["shape"]) != row["numel"]
        ):
            raise ValueError(f"H1 state tensor {row.get('target_key')!r} is not hash-identical")
    expected_rows_sha256 = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if state_mapping.get("rows_sha256") != expected_rows_sha256:
        raise ValueError("H1 state-mapping row digest drift")
    if (
        state_mapping.get("target_state_numel") != sum(row["numel"] for row in rows)
        or state_mapping.get("target_state_numel") != H1_FROZEN_MODEL_CONTRACT["text_state_numel"]
    ):
        raise ValueError("H1 state-mapping parameter accounting drift")

    ordinary_metrics = conversion.get("ordinary_logit_metrics")
    dense_losses = conversion.get("dense_next_token_losses")
    if not isinstance(ordinary_metrics, list) or len(ordinary_metrics) != 2:
        raise ValueError("H1 conditional/text ordinary-logit coverage drift")
    if not isinstance(dense_losses, list) or len(dense_losses) != 2:
        raise ValueError("H1 conditional/text dense-loss coverage drift")
    for index, metrics in enumerate(ordinary_metrics):
        _require_exact_h1_metrics(
            metrics,
            shape=[sequence_lengths[index], H1_FROZEN_MODEL_CONTRACT["vocabulary_size"]],
            context=f"H1 conditional/text ordinary logits {index}",
        )
    for index, loss in enumerate(dense_losses):
        if (
            not isinstance(loss, dict)
            or loss.get("finite") is not True
            or loss.get("bit_exact") is not True
            or loss.get("absolute_error") != 0
            or not all(math.isfinite(float(loss.get(key, math.nan))) for key in ("causal", "conditional"))
        ):
            raise ValueError(f"H1 conditional/text dense loss {index} is not finite and bit-exact")

    pack_shape = parity.get("singleton_vs_multi_pack_shape_diagnostic")
    if (
        not isinstance(pack_shape, dict)
        or pack_shape.get("gating") is not False
        or pack_shape.get("r11_failed_criterion_reclassified_as_pass") is not False
        or pack_shape.get("frozen_r9_tolerance_observation") not in {"within_tolerance", "exceeds_tolerance"}
        or not isinstance(pack_shape.get("reason"), str)
        or not pack_shape["reason"]
    ):
        raise ValueError("H1 singleton/multi pack-shape diagnostic contract drift")
    kernel_logits = pack_shape.get("logits")
    kernel_layers = pack_shape.get("layers")
    if not isinstance(kernel_logits, list) or len(kernel_logits) != 2:
        raise ValueError("H1 pack-shape logit diagnostic coverage drift")
    if not isinstance(kernel_layers, list) or len(kernel_layers) != 2:
        raise ValueError("H1 pack-shape layer-diagnostic coverage drift")
    for index, metrics in enumerate(kernel_logits):
        _require_h1_metric_geometry(
            metrics,
            shape=[sequence_lengths[index], H1_FROZEN_MODEL_CONTRACT["vocabulary_size"]],
            context=f"H1 pack-shape logits {index}",
        )
        if (
            metrics.get("atol") != H1_PACKED_LOGIT_ATOL
            or metrics.get("rtol") != H1_PACKED_LOGIT_RTOL
            or not isinstance(metrics.get("allclose"), bool)
        ):
            raise ValueError(f"H1 pack-shape logits {index} diagnostic is incomplete")
    layer_signatures = []
    for sequence_index, layer_rows in enumerate(kernel_layers):
        if not isinstance(layer_rows, list) or not layer_rows:
            raise ValueError(f"H1 pack-shape layer coverage is empty for sequence {sequence_index}")
        signature = [(row.get("layer_index"), row.get("layer_type")) for row in layer_rows]
        if [index for index, _ in signature] != list(range(len(signature))):
            raise ValueError("H1 pack-shape layer indices are not contiguous")
        if any(not isinstance(layer_type, str) or not layer_type for _, layer_type in signature):
            raise ValueError("H1 pack-shape layer type is missing")
        if any(
            row["metrics"].get("atol") != H1_PACKED_LOGIT_ATOL or row["metrics"].get("rtol") != H1_PACKED_LOGIT_RTOL
            for row in layer_rows
        ):
            raise ValueError("H1 pack-shape layer diagnostic contains invalid metrics")
        for row in layer_rows:
            _require_h1_metric_geometry(
                row["metrics"],
                shape=[sequence_lengths[sequence_index], H1_FROZEN_MODEL_CONTRACT["text_hidden_size"]],
                context=f"H1 pack-shape sequence {sequence_index} layer {row.get('layer_index')}",
            )
        layer_signatures.append(signature)
    if layer_signatures[0] != layer_signatures[1]:
        raise ValueError("H1 pack-shape layer coverage differs across sequences")
    expected_layer_signature = list(enumerate(H1_FROZEN_MODEL_CONTRACT["text_layer_types"]))
    if layer_signatures[0] != expected_layer_signature:
        raise ValueError("H1 decoder-layer type/order drift")

    cross_kernel = parity.get("cross_kernel_ordinary_vs_singleton_diagnostic")
    if not isinstance(cross_kernel, dict) or cross_kernel.get("gating") is not False:
        raise ValueError("H1 ordinary/variable-length cross-kernel diagnostic became gating")
    if not isinstance(cross_kernel.get("reason"), str) or not cross_kernel["reason"]:
        raise ValueError("H1 cross-kernel diagnostic lacks its confounding explanation")
    if not isinstance(cross_kernel.get("logits"), list) or len(cross_kernel["logits"]) != 2:
        raise ValueError("H1 cross-kernel logit diagnostic coverage drift")
    if not isinstance(cross_kernel.get("layers"), list) or len(cross_kernel["layers"]) != 2:
        raise ValueError("H1 cross-kernel layer diagnostic coverage drift")
    for sequence_index, metrics in enumerate(cross_kernel["logits"]):
        _require_h1_metric_geometry(
            metrics,
            shape=[sequence_lengths[sequence_index], H1_FROZEN_MODEL_CONTRACT["vocabulary_size"]],
            context=f"H1 cross-kernel logits {sequence_index}",
        )
        if metrics.get("atol") != H1_PACKED_LOGIT_ATOL or metrics.get("rtol") != H1_PACKED_LOGIT_RTOL:
            raise ValueError(f"H1 cross-kernel logits {sequence_index} diagnostic is incomplete")
    for sequence_index, layer_rows in enumerate(cross_kernel["layers"]):
        if (
            not isinstance(layer_rows, list)
            or [(row.get("layer_index"), row.get("layer_type")) for row in layer_rows]
            != layer_signatures[sequence_index]
            or any(
                row["metrics"].get("atol") != H1_PACKED_LOGIT_ATOL
                or row["metrics"].get("rtol") != H1_PACKED_LOGIT_RTOL
                for row in layer_rows
            )
        ):
            raise ValueError(f"H1 cross-kernel layer diagnostic {sequence_index} is incomplete")
        for row in layer_rows:
            _require_h1_metric_geometry(
                row["metrics"],
                shape=[sequence_lengths[sequence_index], H1_FROZEN_MODEL_CONTRACT["text_hidden_size"]],
                context=f"H1 cross-kernel sequence {sequence_index} layer {row.get('layer_index')}",
            )

    def validate_exact_family(family: Any, expected_lengths: dict[str, int], *, context: str) -> None:
        if not isinstance(family, dict) or set(family) != set(expected_lengths):
            raise ValueError(f"{context} coverage drift")
        for name, value in family.items():
            if not isinstance(value, dict):
                raise ValueError(f"{context} {name} is malformed")
            held_length = expected_lengths[name]
            _require_exact_h1_metrics(
                value.get("unchanged_segment_logits"),
                shape=[held_length, H1_FROZEN_MODEL_CONTRACT["vocabulary_size"]],
                context=f"{context} {name} logits",
            )
            layer_rows = value.get("unchanged_segment_layers")
            if (
                not isinstance(layer_rows, list)
                or [(row.get("layer_index"), row.get("layer_type")) for row in layer_rows] != layer_signatures[0]
            ):
                raise ValueError(f"{context} {name} decoder-layer coverage drift")
            for row in layer_rows:
                _require_exact_h1_metrics(
                    row.get("metrics"),
                    shape=[held_length, H1_FROZEN_MODEL_CONTRACT["text_hidden_size"]],
                    context=f"{context} {name} layer {row.get('layer_index')}",
                )

    single_counterfactuals = parity.get("single_token_counterfactual_no_cross_document_influence")
    expected_single = {
        "mutate_first_hold_second": sequence_lengths[1],
        "mutate_second_hold_first": sequence_lengths[0],
    }
    validate_exact_family(single_counterfactuals, expected_single, context="H1 single-token counterfactual")
    for name, value in single_counterfactuals.items():
        if not isinstance(value, dict):
            raise ValueError(f"H1 single-token counterfactual {name} is malformed")
        mutated_sequence_index = 0 if name == "mutate_first_hold_second" else 1
        mutation_position = value.get("mutation_position")
        if (
            not isinstance(mutation_position, int)
            or not 0 <= mutation_position < sequence_lengths[mutated_sequence_index]
        ):
            raise ValueError(f"H1 single-token counterfactual {name} mutation position is out of range")

    full_counterfactuals = parity.get("full_document_counterfactual_no_cross_document_influence")
    expected_full = {
        "mutate_every_first_token_hold_second": sequence_lengths[1],
        "mutate_every_second_token_hold_first": sequence_lengths[0],
    }
    validate_exact_family(full_counterfactuals, expected_full, context="H1 full-document counterfactual")
    if (
        full_counterfactuals["mutate_every_first_token_hold_second"].get("mutated_tokens") != sequence_lengths[0]
        or full_counterfactuals["mutate_every_second_token_hold_first"].get("mutated_tokens") != sequence_lengths[1]
    ):
        raise ValueError("H1 full-document counterfactual did not mutate every token")

    duplicate_invariance = parity.get("duplicate_document_reset_invariance")
    validate_exact_family(
        duplicate_invariance,
        {"sequence_0_first_vs_second": sequence_lengths[0], "sequence_1_first_vs_second": sequence_lengths[1]},
        context="H1 duplicate-document reset invariance",
    )
    order_invariance = parity.get("packed_order_invariance")
    validate_exact_family(
        order_invariance,
        {"first_document_moved_to_second": sequence_lengths[0], "second_document_moved_to_first": sequence_lengths[1]},
        context="H1 packed-order invariance",
    )

    negative = parity.get("corrupted_boundary_negative_control")
    if (
        not isinstance(negative, dict)
        or negative.get("expected_bit_exact") is not False
        or negative.get("sensitivity_passed") is not True
    ):
        raise ValueError("H1 corrupted-boundary negative control did not pass")
    negative_logits = negative.get("unchanged_segment_logits")
    negative_layers = negative.get("unchanged_segment_layers")
    _require_h1_metric_geometry(
        negative_logits,
        shape=[sequence_lengths[1], H1_FROZEN_MODEL_CONTRACT["vocabulary_size"]],
        context="H1 corrupted-boundary negative-control logits",
    )
    if (
        not isinstance(negative_logits, dict)
        or negative_logits.get("bit_exact") is not False
        or negative_logits.get("nonfinite_count") != 0
        or not isinstance(negative_layers, list)
        or [(row.get("layer_index"), row.get("layer_type")) for row in negative_layers] != layer_signatures[0]
        or not any(row.get("metrics", {}).get("bit_exact") is False for row in negative_layers)
    ):
        raise ValueError("H1 corrupted-boundary negative-control evidence is incomplete")
    for row in negative_layers:
        _require_h1_metric_geometry(
            row["metrics"],
            shape=[sequence_lengths[1], H1_FROZEN_MODEL_CONTRACT["text_hidden_size"]],
            context=f"H1 corrupted-boundary negative-control layer {row.get('layer_index')}",
        )
    return {
        "status": "passed",
        "state_tensors": len(rows),
        "decoder_layers": len(layer_signatures[0]),
        "exact_invariance_cases": (
            len(single_counterfactuals) + len(full_counterfactuals) + len(duplicate_invariance) + len(order_invariance)
        ),
        "negative_controls": 1,
    }


def _validate_h2_liger_report_r14(
    report: dict[str, Any], *, qualification: dict[str, Any], expected_manifest_sha256: str
) -> dict[str, Any]:
    """Historical R14 validator retained at the R15 source revision."""

    if report.get("artifact") != "qwen35_selective_liger_loss_qualification" or report.get("schema_version") != 1:
        raise ValueError("H2 report identity drift")
    if report.get("status") != "passed":
        raise ValueError("H2 report did not pass")
    if report.get("qualification_protocol_id") != QUALIFICATION_PROTOCOL_ID:
        raise ValueError("H2 report protocol drift")
    if report.get("qualification_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("H2 report qualification-manifest identity drift")
    if report.get("numerical_acceptance") != qualification["numerical_acceptance"]:
        raise ValueError("H2 report numerical-acceptance drift")
    if report.get("precision_policy") != {
        "parameters": "torch.float32",
        "gradients": "dtype follows FP32 parameter storage; selected BF16 hidden-row leaf gradients are BF16",
        "adamw_moments": "torch.float32",
        "forward_backward_autocast": "torch.bfloat16",
        "loss_accumulation": "torch.float32",
    }:
        raise ValueError("H2 report precision-policy drift")
    source = report.get("liger_kernel")
    if not isinstance(source, dict) or source.get("commit") != qualification["runtime_pins"]["liger_commit"]:
        raise ValueError("H2 report Liger source-pin drift")
    if source.get("version") != qualification["runtime_pins"]["liger_version"]:
        raise ValueError("H2 report Liger version drift")
    implementation_files = source.get("implementation_files")
    expected_source_files = qualification["runtime_pins"]["liger_source_files_sha256"]
    if not isinstance(implementation_files, dict) or set(implementation_files) != set(expected_source_files):
        raise ValueError("H2 report Liger executed-source file set drift")
    for relative_path, expected_sha256 in expected_source_files.items():
        row = implementation_files.get(relative_path)
        if (
            not isinstance(row, dict)
            or row.get("sha256") != expected_sha256
            or "pinned-sources/liger-kernel" not in str(row.get("path", ""))
            or not str(row.get("path", "")).endswith(relative_path)
        ):
            raise ValueError(f"H2 report Liger source binding drift for {relative_path}")
    source_url = str(source.get("source_url", ""))
    archive_pinned = (
        source.get("archive_url_pinned") is True
        and qualification["runtime_pins"]["liger_commit"] in source_url
        and "/archive/" in source_url
    )
    vcs_pinned = source.get("metadata_vcs_commit") == qualification["runtime_pins"]["liger_commit"]
    if not (archive_pinned or vcs_pinned):
        raise ValueError("H2 report Liger distribution metadata does not bind the pinned commit")

    acceptance = qualification["h2_acceptance"]
    hidden_size = acceptance["direct_hidden_size"]
    vocab_size = acceptance["direct_vocab_size"]
    if report.get("direct_hidden_size") != hidden_size or report.get("direct_vocab_size") != vocab_size:
        raise ValueError("H2 direct-assay geometry drift")

    expected_autocast = {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}

    def require_autocast(value: Any, *, context: str) -> None:
        if value != expected_autocast:
            raise ValueError(f"{context} did not attest the active production BF16 autocast context")

    def require_optimizer_dtypes(value: Any, *, context: str) -> None:
        if value != ["torch.float32"]:
            raise ValueError(f"{context} optimizer floating-state dtype drift")

    def require_metric(value: Any, *, kind: str, context: str, expected_elements: int | None = None) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{context} comparison evidence is missing")
        if expected_elements is not None and value.get("elements") != expected_elements:
            raise ValueError(f"{context} comparison geometry drift")
        validate_comparison_metrics(value, qualification["numerical_acceptance"], kind=kind, context=context)
        return value

    cases = report.get("cases")
    supervised_rows = acceptance["direct_case_supervised_rows"]
    global_divisors = acceptance["direct_case_global_divisors"]
    if not isinstance(cases, list) or len(cases) != len(supervised_rows):
        raise ValueError("H2 direct-case cardinality drift")
    for index, (case, expected_supervised, expected_divisor) in enumerate(
        zip(cases, supervised_rows, global_divisors, strict=True)
    ):
        if (
            not isinstance(case, dict)
            or case.get("rows") != 64
            or case.get("supervised_rows") != expected_supervised
            or case.get("global_divisor") != expected_divisor
        ):
            raise ValueError(f"H2 direct case {index} geometry/divisor drift")
        autocast = case.get("autocast_contract")
        if not isinstance(autocast, dict):
            raise ValueError(f"H2 direct case {index} lacks autocast evidence")
        require_autocast(autocast.get("fused"), context=f"H2 direct case {index} fused path")
        require_autocast(autocast.get("dense_reference"), context=f"H2 direct case {index} dense path")
        if autocast.get("hidden_input_dtype") != "torch.bfloat16":
            raise ValueError(f"H2 direct case {index} hidden-input dtype drift")
        if autocast.get("output_head_parameter_dtype") != "torch.float32":
            raise ValueError(f"H2 direct case {index} output-head dtype drift")
        if autocast.get("loss_accumulation_dtype") != "torch.float32":
            raise ValueError(f"H2 direct case {index} accumulation dtype drift")
        if case.get("gradient_dtypes") != {
            "fused_hidden_rows": "torch.bfloat16",
            "dense_reference_hidden_rows": "torch.bfloat16",
            "fused_output_head": "torch.float32",
            "dense_reference_output_head": "torch.float32",
        }:
            raise ValueError(f"H2 direct case {index} gradient dtype drift")
        optimizer_dtypes = case.get("optimizer_floating_state_dtypes")
        if not isinstance(optimizer_dtypes, dict):
            raise ValueError(f"H2 direct case {index} lacks optimizer-state evidence")
        require_optimizer_dtypes(optimizer_dtypes.get("fused"), context=f"H2 direct case {index} fused")
        require_optimizer_dtypes(
            optimizer_dtypes.get("dense_reference"), context=f"H2 direct case {index} dense reference"
        )
        if not all(math.isfinite(float(case.get(key, math.nan))) for key in ("loss_fused", "loss_reference")):
            raise ValueError(f"H2 direct case {index} has non-finite loss")
        loss_metrics = require_metric(case.get("loss_comparison"), kind="loss", context=f"H2 case {index} loss")
        if (
            loss_metrics.get("observed") != case["loss_fused"]
            or loss_metrics.get("reference") != case["loss_reference"]
        ):
            raise ValueError(f"H2 direct case {index} loss-metric binding drift")
        require_metric(
            case.get("row_gradient_comparison"),
            kind="gradient",
            context=f"H2 case {index} selected-row gradient",
            expected_elements=expected_supervised * hidden_size,
        )
        require_metric(
            case.get("weight_gradient_comparison"),
            kind="gradient",
            context=f"H2 case {index} output-weight gradient",
            expected_elements=vocab_size * hidden_size,
        )
        require_metric(
            case.get("adamw_update_comparison"),
            kind="update",
            context=f"H2 case {index} AdamW update",
            expected_elements=vocab_size * hidden_size,
        )

    zero = report.get("zero_target_sentinel")
    if not isinstance(zero, dict):
        raise ValueError("H2 zero-target sentinel evidence is missing")
    require_autocast(zero.get("autocast_contract"), context="H2 zero-target sentinel")
    if (
        zero.get("loss") != 0
        or zero.get("global_divisor") != 7
        or zero.get("hidden_input_dtype") != "torch.bfloat16"
        or zero.get("output_head_parameter_dtype") != "torch.float32"
        or zero.get("hidden_gradient_dtype") != "torch.bfloat16"
        or zero.get("output_head_gradient_dtype") != "torch.float32"
        or zero.get("hidden_gradient_connected") is not True
        or zero.get("weight_gradient_connected") is not True
        or zero.get("gradient_nonzero_count") != 0
    ):
        raise ValueError("H2 zero-target sentinel contract failed")

    patched = report.get("patched_qwen_forward")
    if not isinstance(patched, dict):
        raise ValueError("H2 patched-Qwen evidence is missing")
    require_autocast(patched.get("autocast_contract"), context="H2 patched-Qwen forward")
    if (
        patched.get("model_class") != "Qwen3_5ForCausalLM"
        or "liger_kernel" not in str(patched.get("patched_forward_module", ""))
        or patched.get("sequence_tokens") != 8
        or patched.get("supervised_targets") != 4
        or patched.get("global_divisor") != 13
        or patched.get("selected_positions") != [0, 2, 5, 6]
        or patched.get("parameter_dtypes") != {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]}
        or patched.get("gradient_dtypes") != {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]}
    ):
        raise ValueError("H2 patched-Qwen execution contract drift")
    patched_optimizer_dtypes = patched.get("optimizer_floating_state_dtypes")
    if not isinstance(patched_optimizer_dtypes, dict):
        raise ValueError("H2 patched-Qwen optimizer-state evidence is missing")
    require_optimizer_dtypes(patched_optimizer_dtypes.get("selective"), context="H2 patched Qwen selective")
    require_optimizer_dtypes(
        patched_optimizer_dtypes.get("dense_reference"), context="H2 patched Qwen dense reference"
    )
    if not all(math.isfinite(float(patched.get(key, math.nan))) for key in ("dense_loss", "selective_loss")):
        raise ValueError("H2 patched-Qwen loss is non-finite")
    patched_loss = require_metric(patched.get("loss_comparison"), kind="loss", context="H2 patched-Qwen loss")
    if (
        patched_loss.get("observed") != patched["selective_loss"]
        or patched_loss.get("reference") != patched["dense_loss"]
    ):
        raise ValueError("H2 patched-Qwen loss-metric binding drift")
    require_metric(
        patched.get("aggregate_gradient_comparison"), kind="gradient", context="H2 patched-Qwen aggregate gradient"
    )
    parameter_gradients = patched.get("parameter_gradient_comparisons")
    if (
        not isinstance(parameter_gradients, dict)
        or patched.get("checked_parameter_gradients") != len(parameter_gradients)
        or not parameter_gradients
    ):
        raise ValueError("H2 patched-Qwen parameter-gradient coverage drift")
    for name, metrics in parameter_gradients.items():
        require_metric(metrics, kind="gradient", context=f"H2 patched-Qwen parameter {name}")
    require_metric(
        patched.get("aggregate_adamw_update_comparison"), kind="update", context="H2 patched-Qwen AdamW update"
    )
    return {
        "status": "passed",
        "direct_cases": len(cases),
        "direct_supervised_rows": supervised_rows,
        "zero_target_sentinels": 1,
        "patched_qwen_parameter_gradients": len(parameter_gradients),
        "autocast_attestations": len(cases) * 2 + 2,
    }


def validate_h2_liger_report(
    report: dict[str, Any],
    *,
    qualification: dict[str, Any],
    expected_manifest_sha256: str,
    require_numerical_pass: bool = True,
) -> dict[str, Any]:
    """Independently validate R15 evidence and, by default, require an H2 pass."""

    if report.get("artifact") != "qwen35_selective_liger_downstream_qualification":
        raise ValueError("R15 H2 report artifact identity drift")
    report_status = report.get("status")
    if report.get("schema_version") != 2 or report_status not in {"passed", "failed"}:
        raise ValueError("R15 H2 report did not publish a recognized schema-2 decision")
    if require_numerical_pass and report_status != "passed":
        raise ValueError("R15 H2 report did not publish a schema-2 pass")
    if report.get("qualification_protocol_id") != QUALIFICATION_PROTOCOL_ID:
        raise ValueError("R15 H2 report protocol drift")
    if report.get("qualification_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("R15 H2 report qualification-manifest identity drift")
    if report.get("numerical_acceptance") != qualification["numerical_acceptance"]:
        raise ValueError("R15 H2 report numerical-acceptance drift")
    if report.get("h2_acceptance") != qualification["h2_acceptance"]:
        raise ValueError("R15 H2 report holdout/decision contract drift")
    if report.get("precision_policy") != {
        "parameters": "torch.float32",
        "gradients": "dtype follows FP32 parameter storage; direct selected BF16 hidden-row leaf gradients are BF16",
        "adamw_moments": "torch.float32",
        "forward_backward_autocast": "torch.bfloat16",
        "loss_accumulation": "torch.float32",
    }:
        raise ValueError("R15 H2 report precision-policy drift")
    if report.get("scientific_training_authorized") is not False:
        raise ValueError("R15 H2 improperly authorizes scientific training")
    if report.get("successor_gate_authorized") is not (report_status == "passed"):
        raise ValueError("R15 H2 successor-gate authorization disagrees with its numerical status")

    source = report.get("liger_kernel")
    if not isinstance(source, dict):
        raise ValueError("R15 H2 Liger source evidence is missing")
    runtime = qualification["runtime_pins"]
    if source.get("commit") != runtime["liger_commit"] or source.get("version") != runtime["liger_version"]:
        raise ValueError("R15 H2 Liger source version/commit drift")
    implementation_files = source.get("implementation_files")
    expected_source_files = runtime["liger_source_files_sha256"]
    if not isinstance(implementation_files, dict) or set(implementation_files) != set(expected_source_files):
        raise ValueError("R15 H2 executed Liger source-file set drift")
    for relative_path, expected_sha256 in expected_source_files.items():
        row = implementation_files.get(relative_path)
        if (
            not isinstance(row, dict)
            or row.get("sha256") != expected_sha256
            or "pinned-sources/liger-kernel" not in str(row.get("path", ""))
            or not str(row.get("path", "")).endswith(relative_path)
        ):
            raise ValueError(f"R15 H2 Liger source binding drift for {relative_path}")
    source_url = str(source.get("source_url", ""))
    archive_pinned = (
        source.get("archive_url_pinned") is True
        and runtime["liger_commit"] in source_url
        and "/archive/" in source_url
    )
    vcs_pinned = source.get("metadata_vcs_commit") == runtime["liger_commit"]
    if not (archive_pinned or vcs_pinned):
        raise ValueError("R15 H2 distribution metadata does not bind the Liger commit")

    acceptance = qualification["numerical_acceptance"]
    h2 = qualification["h2_acceptance"]
    hidden_size = h2["direct_hidden_size"]
    vocab_size = h2["direct_vocab_size"]
    if report.get("direct_hidden_size") != hidden_size or report.get("direct_vocab_size") != vocab_size:
        raise ValueError("R15 H2 direct geometry drift")
    expected_autocast = {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}

    def require_metric_geometry(value: Any, *, context: str, expected_elements: int | None = None) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{context} comparison evidence is missing")
        if expected_elements is not None and value.get("elements") != expected_elements:
            raise ValueError(f"{context} comparison element-count drift")
        if value.get("nonfinite_count") != 0:
            raise ValueError(f"{context} contains nonfinite values")
        for key in (
            "maximum_absolute_error",
            "relative_l2_error",
            "observed_l2_norm",
            "reference_l2_norm",
            "difference_l2_norm",
        ):
            metric = value.get(key)
            if not isinstance(metric, (float, int)) or not math.isfinite(float(metric)) or metric < 0:
                raise ValueError(f"{context} has invalid {key}")
        cosine = value.get("cosine_similarity")
        if (
            not isinstance(cosine, (float, int))
            or not math.isfinite(float(cosine))
            or not -1.000000000001 <= float(cosine) <= 1.000000000001
        ):
            raise ValueError(f"{context} has invalid cosine similarity")
        return value

    def require_metric(value: Any, *, kind: str, context: str, expected_elements: int | None = None) -> dict[str, Any]:
        value = require_metric_geometry(value, context=context, expected_elements=expected_elements)
        if require_numerical_pass:
            validate_comparison_metrics(value, acceptance, kind=kind, context=context)
        return value

    def require_loss(value: Any, *, context: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{context} scalar evidence is missing")
        for key in ("observed", "reference", "maximum_absolute_error", "relative_error"):
            metric = value.get(key)
            if not isinstance(metric, (float, int)) or not math.isfinite(float(metric)):
                raise ValueError(f"{context} has invalid {key}")
        if value.get("maximum_absolute_error") < 0 or value.get("relative_error") < 0:
            raise ValueError(f"{context} has a negative error metric")
        if value.get("nonfinite_count") != 0:
            raise ValueError(f"{context} contains nonfinite values")
        if require_numerical_pass:
            validate_comparison_metrics(value, acceptance, kind="loss", context=context)
        return value

    def require_logit(value: Any, *, context: str, expected_elements: int) -> dict[str, Any]:
        value = require_metric_geometry(value, context=context, expected_elements=expected_elements)
        if require_numerical_pass:
            if value["maximum_absolute_error"] > acceptance["packed_logit_absolute_tolerance"]:
                raise AssertionError(f"{context} maximum absolute error exceeds packed-logit tolerance")
            if value["relative_l2_error"] > acceptance["packed_logit_relative_tolerance"]:
                raise AssertionError(f"{context} relative-L2 error exceeds packed-logit tolerance")
            if value["cosine_similarity"] < acceptance["heldout_logit_minimum_cosine_similarity"]:
                raise AssertionError(f"{context} cosine similarity is below the held-out-logit threshold")
        return value

    direct_counts = {}
    for section, expected_contracts in (
        ("historical_direct_cases", h2["historical_direct_cases"]),
        ("confirmatory_direct_cases", h2["confirmatory_direct_cases"]),
    ):
        cases = report.get(section)
        if not isinstance(cases, list) or len(cases) != len(expected_contracts):
            raise ValueError(f"R15 H2 {section} cardinality drift")
        direct_counts[section] = len(cases)
        for index, (case, expected_contract) in enumerate(zip(cases, expected_contracts, strict=True)):
            context = f"R15 H2 {section} case {index}"
            if not isinstance(case, dict) or case.get("case_contract") != expected_contract:
                raise ValueError(f"{context} frozen contract drift")
            expected_rows = (
                list(range(expected_contract["rows"]))
                if expected_contract["supervision_kind"] == "all"
                else expected_contract["supervised_rows"]
            )
            if case.get("supervised_rows_expanded") != expected_rows:
                raise ValueError(f"{context} expanded supervised rows drift")
            autocast = case.get("autocast_contract")
            if not isinstance(autocast, dict) or any(
                autocast.get(key) != expected_autocast for key in ("selective", "dense_reference", "heldout")
            ):
                raise ValueError(f"{context} autocast contract drift")
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
                raise ValueError(f"{context} dtype/state contract drift")
            if not all(math.isfinite(float(case.get(key, math.nan))) for key in ("selective_loss", "reference_loss")):
                raise ValueError(f"{context} contains a nonfinite loss")
            loss = require_loss(case.get("loss_comparison"), context=f"{context} loss")
            if loss.get("observed") != case["selective_loss"] or loss.get("reference") != case["reference_loss"]:
                raise ValueError(f"{context} loss-metric binding drift")
            supervised_count = expected_contract["expected_supervised_count"]
            require_metric(
                case.get("selected_hidden_gradient_comparison"),
                kind="gradient",
                context=f"{context} selected-hidden gradient",
                expected_elements=supervised_count * hidden_size,
            )
            require_metric(
                case.get("output_head_gradient_comparison"),
                kind="gradient",
                context=f"{context} output-head gradient",
                expected_elements=vocab_size * hidden_size,
            )
            require_metric_geometry(
                case.get("raw_first_adamw_update_comparison_diagnostic"),
                context=f"{context} raw first AdamW update diagnostic",
                expected_elements=vocab_size * hidden_size,
            )
            require_metric(
                case.get("post_step_parameter_comparison"),
                kind="update",
                context=f"{context} post-step parameter state",
                expected_elements=vocab_size * hidden_size,
            )
            heldout = case.get("heldout")
            if not isinstance(heldout, dict) or heldout.get("rows") != h2["direct_heldout_rows"]:
                raise ValueError(f"{context} held-out geometry drift")
            require_logit(
                heldout.get("logit_comparison"),
                context=f"{context} held-out logits",
                expected_elements=h2["direct_heldout_rows"] * vocab_size,
            )
            heldout_loss = require_loss(heldout.get("loss_comparison"), context=f"{context} held-out loss")
            if heldout_loss.get("observed") != heldout.get("selective_loss") or heldout_loss.get(
                "reference"
            ) != heldout.get("reference_loss"):
                raise ValueError(f"{context} held-out loss-metric binding drift")

    zero = report.get("zero_target_sentinel")
    if not isinstance(zero, dict) or zero.get("autocast_contract") != expected_autocast:
        raise ValueError("R15 H2 zero-target sentinel evidence is missing")
    if (
        zero.get("loss") != 0
        or zero.get("global_divisor") != 7
        or zero.get("hidden_input_dtype") != "torch.bfloat16"
        or zero.get("output_head_parameter_dtype") != "torch.float32"
        or zero.get("hidden_gradient_dtype") != "torch.bfloat16"
        or zero.get("output_head_gradient_dtype") != "torch.float32"
        or zero.get("hidden_gradient_connected") is not True
        or zero.get("weight_gradient_connected") is not True
        or zero.get("gradient_nonzero_count") != 0
    ):
        raise ValueError("R15 H2 zero-target sentinel contract failed")

    trajectories = report.get("confirmatory_trajectories")
    expected_trajectories = h2["confirmatory_trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) != len(expected_trajectories):
        raise ValueError("R15 H2 confirmatory trajectory cardinality drift")
    parameter_gradient_checks = 0
    step_checks = 0
    diagnostic_first_updates = 0
    gated_updates = 0
    for trajectory_index, (trajectory, expected_contract) in enumerate(
        zip(trajectories, expected_trajectories, strict=True)
    ):
        context = f"R15 H2 trajectory {expected_contract['trajectory_id']}"
        if not isinstance(trajectory, dict) or trajectory.get("trajectory_contract") != expected_contract:
            raise ValueError(f"{context} frozen seed contract drift")
        if (
            trajectory.get("trajectory_index") != trajectory_index
            or trajectory.get("model_class") != "Qwen3_5ForCausalLM"
            or trajectory.get("dense_forward_module") != "transformers.models.qwen3_5.modeling_qwen3_5"
            or "liger_kernel" not in str(trajectory.get("patched_forward_module", ""))
            or trajectory.get("model_config") != h2["trajectory_model_config"]
            or trajectory.get("parameter_dtypes")
            != {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]}
        ):
            raise ValueError(f"{context} model/patch/dtype contract drift")
        expected_parameter_geometry = h2["trajectory_parameter_geometry"]
        if trajectory.get("parameter_geometry") != expected_parameter_geometry:
            raise ValueError(f"{context} exact parameter geometry drift")
        names = trajectory.get("parameter_names")
        expected_names = [row["name"] for row in expected_parameter_geometry]
        if names != expected_names:
            raise ValueError(f"{context} parameter-name coverage drift")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or len(steps) != h2["trajectory_steps"]:
            raise ValueError(f"{context} step cardinality drift")
        heldout_contract = trajectory.get("heldout_contract")
        heldout_supervised = sum(
            position % h2["trajectory_heldout_supervision_modulus"] == 0
            for position in range(1, h2["trajectory_sequence_length"])
        )
        expected_heldout = {
            "seed": expected_contract["heldout_seed"],
            "sequence_length": h2["trajectory_sequence_length"],
            "supervision_modulus": h2["trajectory_heldout_supervision_modulus"],
            "supervised_targets": heldout_supervised,
            "divisor_extra": h2["trajectory_heldout_divisor_extra"],
            "global_divisor": heldout_supervised + h2["trajectory_heldout_divisor_extra"],
        }
        if heldout_contract != expected_heldout:
            raise ValueError(f"{context} held-out contract drift")
        parameter_count = trajectory.get("parameter_count")
        if parameter_count != h2["trajectory_parameter_count"]:
            raise ValueError(f"{context} parameter count is invalid")
        expected_parameter_elements = {row["name"]: row["elements"] for row in expected_parameter_geometry}

        for step_index, step in enumerate(steps):
            step_number = step_index + 1
            step_context = f"{context} step {step_number}"
            if not isinstance(step, dict) or step.get("step") != step_number:
                raise ValueError(f"{step_context} identity/order drift")
            modulus = h2["trajectory_supervision_moduli"][step_index % len(h2["trajectory_supervision_moduli"])]
            offset = (step_index + trajectory_index) % modulus
            supervised = sum(
                (position + offset) % modulus == 0 for position in range(1, h2["trajectory_sequence_length"])
            )
            divisor_extra = (step_index * h2["trajectory_divisor_extra_multiplier"] + trajectory_index) % h2[
                "trajectory_divisor_extra_modulus"
            ]
            expected_accounting = {
                "seed": expected_contract["batch_seed_base"] + step_index,
                "sequence_length": h2["trajectory_sequence_length"],
                "supervision_modulus": modulus,
                "supervision_offset": offset,
                "supervised_targets": supervised,
                "divisor_extra": divisor_extra,
                "global_divisor": supervised + divisor_extra,
            }
            if step.get("batch_accounting") != expected_accounting:
                raise ValueError(f"{step_context} batch/divisor accounting drift")
            if step.get("autocast_contract") != {"training": expected_autocast, "heldout": expected_autocast}:
                raise ValueError(f"{step_context} autocast contract drift")
            if not all(math.isfinite(float(step.get(key, math.nan))) for key in ("selective_loss", "reference_loss")):
                raise ValueError(f"{step_context} contains a nonfinite loss")
            training_loss = require_loss(step.get("training_loss_comparison"), context=f"{step_context} training loss")
            if (
                training_loss.get("observed") != step["selective_loss"]
                or training_loss.get("reference") != step["reference_loss"]
            ):
                raise ValueError(f"{step_context} training loss-metric binding drift")
            require_metric(
                step.get("aggregate_preclip_gradient_comparison"),
                kind="gradient",
                context=f"{step_context} aggregate preclip gradient",
                expected_elements=parameter_count,
            )
            require_metric(
                step.get("aggregate_clipped_gradient_comparison"),
                kind="gradient",
                context=f"{step_context} aggregate clipped gradient",
                expected_elements=parameter_count,
            )
            per_parameter = step.get("per_parameter_gradient_comparisons")
            if not isinstance(per_parameter, dict) or set(per_parameter) != set(names):
                raise ValueError(f"{step_context} named-parameter gradient coverage drift")
            observed_parameter_count = 0
            for name in names:
                row = per_parameter[name]
                if not isinstance(row, dict) or row.get("elements") != expected_parameter_elements[name]:
                    raise ValueError(f"{step_context} parameter {name} geometry drift")
                observed_parameter_count += row["elements"]
                require_metric(
                    row.get("preclip_gradient_comparison"),
                    kind="gradient",
                    context=f"{step_context} parameter {name} preclip gradient",
                    expected_elements=row["elements"],
                )
                require_metric(
                    row.get("clipped_gradient_comparison"),
                    kind="gradient",
                    context=f"{step_context} parameter {name} clipped gradient",
                    expected_elements=row["elements"],
                )
                parameter_gradient_checks += 2
            if observed_parameter_count != parameter_count:
                raise ValueError(f"{step_context} named parameter elements do not sum to model parameter count")
            preclip_norms = step.get("preclip_gradient_norms")
            if (
                not isinstance(preclip_norms, dict)
                or set(preclip_norms) != {"selective", "dense_reference"}
                or any(
                    not isinstance(value, (float, int)) or not math.isfinite(float(value)) or value < 0
                    for value in preclip_norms.values()
                )
            ):
                raise ValueError(f"{step_context} preclip gradient-norm evidence is invalid")
            raw_update = require_metric_geometry(
                step.get("raw_adamw_update_comparison"),
                context=f"{step_context} raw AdamW update",
                expected_elements=parameter_count,
            )
            expected_update_gating = step_number >= h2["raw_update_gating_starts_at_step"]
            if step.get("raw_adamw_update_is_gating") is not expected_update_gating:
                raise ValueError(f"{step_context} raw-update gating classification drift")
            if expected_update_gating and require_numerical_pass:
                validate_comparison_metrics(raw_update, acceptance, kind="update", context=f"{step_context} update")
            if expected_update_gating:
                gated_updates += 1
            else:
                diagnostic_first_updates += 1
            require_metric(
                step.get("post_step_parameter_comparison"),
                kind="update",
                context=f"{step_context} post-step parameter state",
                expected_elements=parameter_count,
            )
            if step.get("optimizer_floating_state_dtypes") != {
                "selective": ["torch.float32"],
                "dense_reference": ["torch.float32"],
            }:
                raise ValueError(f"{step_context} AdamW floating-state dtype drift")
            if step.get("optimizer_step_counters") != {"selective": [step_number], "dense_reference": [step_number]}:
                raise ValueError(f"{step_context} AdamW step-counter drift")
            if step.get("gradient_dtypes") != {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]}:
                raise ValueError(f"{step_context} gradient dtype drift")
            heldout = step.get("heldout")
            if (
                not isinstance(heldout, dict)
                or heldout.get("supervised_targets") != expected_heldout["supervised_targets"]
                or heldout.get("global_divisor") != expected_heldout["global_divisor"]
            ):
                raise ValueError(f"{step_context} held-out accounting drift")
            require_logit(
                heldout.get("logit_comparison"),
                context=f"{step_context} held-out logits",
                expected_elements=h2["trajectory_sequence_length"] * h2["trajectory_model_config"]["vocab_size"],
            )
            heldout_loss = require_loss(heldout.get("loss_comparison"), context=f"{step_context} held-out loss")
            if heldout_loss.get("observed") != heldout.get("selective_loss") or heldout_loss.get(
                "reference"
            ) != heldout.get("reference_loss"):
                raise ValueError(f"{step_context} held-out loss-metric binding drift")
            step_checks += 1

    decision = report.get("decision")
    direct_diagnostic_checks = direct_counts["historical_direct_cases"] + direct_counts["confirmatory_direct_cases"]
    expected_total_checks = direct_diagnostic_checks * 7 + step_checks * 7 + parameter_gradient_checks
    expected_diagnostic_checks = direct_diagnostic_checks + diagnostic_first_updates
    expected_gating_checks = expected_total_checks - expected_diagnostic_checks
    checks = decision.get("checks") if isinstance(decision, dict) else None
    contexts = [check.get("context") for check in checks] if isinstance(checks, list) else []
    recomputed_failed_diagnostic = [
        check.get("context")
        for check in checks or []
        if check.get("gating") is False and check.get("passed") is not True
    ]
    recomputed_decision = collect_h2_numerical_decisions(report, qualification)
    if (
        not isinstance(decision, dict)
        or decision.get("status") != report_status
        or not isinstance(checks, list)
        or len(checks) != expected_total_checks
        or decision.get("total_checks") != expected_total_checks
        or decision.get("gating_checks") != expected_gating_checks
        or decision.get("diagnostic_checks") != expected_diagnostic_checks
        or len(contexts) != len(set(contexts))
        or any(
            not isinstance(check, dict)
            or not isinstance(check.get("context"), str)
            or not check.get("context")
            or not isinstance(check.get("kind"), str)
            or not isinstance(check.get("gating"), bool)
            or not isinstance(check.get("passed"), bool)
            for check in checks
        )
        or decision.get("failed_diagnostic_checks") != recomputed_failed_diagnostic
        or decision != recomputed_decision
    ):
        raise ValueError("R15 H2 producer decision ledger is incomplete or inconsistent")
    if require_numerical_pass and (
        decision.get("failed_gating_checks") != []
        or any(check["gating"] and check["passed"] is not True for check in checks)
    ):
        raise ValueError("R15 H2 producer decision ledger is not a numerical pass")
    if not require_numerical_pass and report_status == "failed" and not decision.get("failed_gating_checks"):
        raise ValueError("R15 H2 failed status has no independently reproduced gating failure")
    validation = {
        "status": "passed",
        "historical_direct_cases": direct_counts["historical_direct_cases"],
        "confirmatory_direct_cases": direct_counts["confirmatory_direct_cases"],
        "confirmatory_trajectories": len(trajectories),
        "trajectory_steps": step_checks,
        "parameter_gradient_checks": parameter_gradient_checks,
        "diagnostic_first_updates": diagnostic_first_updates,
        "gated_stateful_updates": gated_updates,
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


def tensor_comparison_metrics(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int | None]:
    if observed.shape != reference.shape:
        raise ValueError(f"tensor shape mismatch: {tuple(observed.shape)} != {tuple(reference.shape)}")
    observed64 = observed.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if observed64.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    finite = torch.isfinite(observed64) & torch.isfinite(reference64)
    nonfinite_count = int((~finite).sum())
    difference = observed64 - reference64
    maximum_absolute_error = float(difference.abs().max()) if nonfinite_count == 0 else math.inf
    observed_norm = float(torch.linalg.vector_norm(observed64))
    reference_norm = float(torch.linalg.vector_norm(reference64))
    difference_norm = float(torch.linalg.vector_norm(difference)) if nonfinite_count == 0 else math.inf
    denominator = max(reference_norm, torch.finfo(torch.float64).eps)
    relative_l2_error = difference_norm / denominator
    cosine_similarity: float | None
    if observed_norm == 0 and reference_norm == 0:
        cosine_similarity = 1.0
    elif observed_norm == 0 or reference_norm == 0 or nonfinite_count:
        cosine_similarity = None
    else:
        cosine_similarity = float(torch.dot(observed64, reference64) / (observed_norm * reference_norm))
    return {
        "elements": observed64.numel(),
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
        "observed_l2_norm": observed_norm,
        "reference_l2_norm": reference_norm,
        "difference_l2_norm": difference_norm,
        "nonfinite_count": nonfinite_count,
    }


def scalar_comparison_metrics(observed: float, reference: float) -> dict[str, float | int]:
    finite = math.isfinite(observed) and math.isfinite(reference)
    absolute_error = abs(observed - reference) if finite else math.inf
    relative_error = absolute_error / max(abs(reference), torch.finfo(torch.float64).eps)
    return {
        "observed": observed,
        "reference": reference,
        "maximum_absolute_error": absolute_error,
        "relative_error": relative_error,
        "nonfinite_count": int(not finite),
    }


def validate_comparison_metrics(
    metrics: dict[str, float | int | None], numerical_acceptance: dict[str, Any], *, kind: str, context: str
) -> None:
    if int(metrics["nonfinite_count"]) != int(numerical_acceptance["nonfinite_count"]):
        raise AssertionError(f"{context}: nonfinite values found")
    if kind == "loss":
        if float(metrics["maximum_absolute_error"]) > float(numerical_acceptance["loss_maximum_absolute_error"]):
            raise AssertionError(f"{context}: loss maximum-absolute error exceeds the frozen threshold")
        if float(metrics["relative_error"]) > float(numerical_acceptance["loss_relative_error"]):
            raise AssertionError(f"{context}: loss relative error exceeds the frozen threshold")
        return
    if kind not in {"gradient", "update"}:
        raise ValueError(f"unsupported comparison kind {kind!r}")
    prefix = "gradient" if kind == "gradient" else "update"
    maximum_key = f"{prefix}_maximum_absolute_error"
    if maximum_key in numerical_acceptance and float(metrics["maximum_absolute_error"]) > float(
        numerical_acceptance[maximum_key]
    ):
        raise AssertionError(f"{context}: {kind} maximum-absolute error exceeds the frozen threshold")
    if float(metrics["relative_l2_error"]) > float(numerical_acceptance[f"{prefix}_relative_l2_error"]):
        raise AssertionError(f"{context}: {kind} relative-L2 error exceeds the frozen threshold")
    cosine = metrics["cosine_similarity"]
    if cosine is None or float(cosine) < float(numerical_acceptance[f"{prefix}_minimum_cosine_similarity"]):
        raise AssertionError(f"{context}: {kind} cosine similarity is below the frozen threshold")


def _h2_metric_decision(
    metrics: dict[str, Any], acceptance: dict[str, Any], *, kind: str, context: str, gating: bool
) -> dict[str, Any]:
    try:
        validate_comparison_metrics(metrics, acceptance, kind=kind, context=context)
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


def _h2_logit_decision(metrics: dict[str, Any], acceptance: dict[str, Any], *, context: str) -> dict[str, Any]:
    failures = []
    if metrics.get("nonfinite_count") != acceptance["nonfinite_count"]:
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
    """Build the complete R15 numerical decision ledger without raising early."""

    acceptance = qualification["numerical_acceptance"]
    decisions = []
    for section in ("historical_direct_cases", "confirmatory_direct_cases"):
        for case in report[section]:
            case_id = case["case_contract"]["case_id"]
            decisions.extend(
                [
                    _h2_metric_decision(
                        case["loss_comparison"], acceptance, kind="loss", context=f"{case_id} loss", gating=True
                    ),
                    _h2_metric_decision(
                        case["selected_hidden_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{case_id} selected-hidden gradient",
                        gating=True,
                    ),
                    _h2_metric_decision(
                        case["output_head_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{case_id} output-head gradient",
                        gating=True,
                    ),
                    _h2_metric_decision(
                        case["raw_first_adamw_update_comparison_diagnostic"],
                        acceptance,
                        kind="update",
                        context=f"{case_id} raw first AdamW update",
                        gating=False,
                    ),
                    _h2_metric_decision(
                        case["post_step_parameter_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{case_id} post-step parameter state",
                        gating=True,
                    ),
                    _h2_logit_decision(
                        case["heldout"]["logit_comparison"], acceptance, context=f"{case_id} heldout logits"
                    ),
                    _h2_metric_decision(
                        case["heldout"]["loss_comparison"],
                        acceptance,
                        kind="loss",
                        context=f"{case_id} heldout loss",
                        gating=True,
                    ),
                ]
            )
    for trajectory in report["confirmatory_trajectories"]:
        trajectory_id = trajectory["trajectory_contract"]["trajectory_id"]
        for step in trajectory["steps"]:
            prefix = f"{trajectory_id} step {step['step']}"
            decisions.extend(
                [
                    _h2_metric_decision(
                        step["training_loss_comparison"],
                        acceptance,
                        kind="loss",
                        context=f"{prefix} training loss",
                        gating=True,
                    ),
                    _h2_metric_decision(
                        step["aggregate_preclip_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{prefix} aggregate preclip gradient",
                        gating=True,
                    ),
                    _h2_metric_decision(
                        step["aggregate_clipped_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{prefix} aggregate clipped gradient",
                        gating=True,
                    ),
                    _h2_metric_decision(
                        step["raw_adamw_update_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} raw AdamW update",
                        gating=step["raw_adamw_update_is_gating"],
                    ),
                    _h2_metric_decision(
                        step["post_step_parameter_comparison"],
                        acceptance,
                        kind="update",
                        context=f"{prefix} post-step parameter state",
                        gating=True,
                    ),
                    _h2_logit_decision(
                        step["heldout"]["logit_comparison"], acceptance, context=f"{prefix} heldout logits"
                    ),
                    _h2_metric_decision(
                        step["heldout"]["loss_comparison"],
                        acceptance,
                        kind="loss",
                        context=f"{prefix} heldout loss",
                        gating=True,
                    ),
                ]
            )
            for name in trajectory["parameter_names"]:
                gradients = step["per_parameter_gradient_comparisons"][name]
                decisions.append(
                    _h2_metric_decision(
                        gradients["preclip_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{prefix} parameter {name} preclip gradient",
                        gating=True,
                    )
                )
                decisions.append(
                    _h2_metric_decision(
                        gradients["clipped_gradient_comparison"],
                        acceptance,
                        kind="gradient",
                        context=f"{prefix} parameter {name} clipped gradient",
                        gating=True,
                    )
                )
    failed_gating = [decision["context"] for decision in decisions if decision["gating"] and not decision["passed"]]
    failed_diagnostic = [
        decision["context"] for decision in decisions if not decision["gating"] and not decision["passed"]
    ]
    return {
        "checks": decisions,
        "total_checks": len(decisions),
        "gating_checks": sum(decision["gating"] for decision in decisions),
        "diagnostic_checks": sum(not decision["gating"] for decision in decisions),
        "failed_gating_checks": failed_gating,
        "failed_diagnostic_checks": failed_diagnostic,
        "status": "passed" if not failed_gating else "failed",
    }


def validate_memory_headroom(
    *, peak_allocated_bytes: int, peak_reserved_bytes: int, total_device_bytes: int, acceptance: dict[str, Any]
) -> dict[str, float | int]:
    if not 0 <= peak_allocated_bytes <= peak_reserved_bytes <= total_device_bytes:
        raise ValueError("invalid CUDA memory accounting order")
    reserved_fraction = peak_reserved_bytes / total_device_bytes
    headroom_fraction = 1.0 - reserved_fraction
    epsilon = 1e-12
    if reserved_fraction > float(acceptance["maximum_peak_reserved_fraction"]) + epsilon:
        raise AssertionError("peak reserved CUDA memory exceeds the frozen fraction")
    if headroom_fraction + epsilon < float(acceptance["minimum_headroom_fraction"]):
        raise AssertionError("CUDA memory headroom is below the frozen fraction")
    return {
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "total_device_bytes": total_device_bytes,
        "peak_reserved_fraction": reserved_fraction,
        "headroom_bytes": total_device_bytes - peak_reserved_bytes,
        "headroom_fraction": headroom_fraction,
    }


def coefficient_of_variation(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timing values must be finite and positive")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def select_topology(four_gpu_seconds: list[float], eight_gpu_seconds: list[float], acceptance: dict[str, Any]) -> dict:
    if len(four_gpu_seconds) != len(eight_gpu_seconds):
        raise ValueError("topology timing windows have different cardinality")
    four_median = float(statistics.median(four_gpu_seconds))
    eight_median = float(statistics.median(eight_gpu_seconds))
    speedup = four_median / eight_median
    efficiency = speedup / 2
    selected = "T8" if speedup >= float(acceptance["eight_gpu_minimum_speedup"]) else "T4"
    return {
        "four_gpu_median_seconds": four_median,
        "eight_gpu_median_seconds": eight_median,
        "four_gpu_coefficient_of_variation": coefficient_of_variation(four_gpu_seconds),
        "eight_gpu_coefficient_of_variation": coefficient_of_variation(eight_gpu_seconds),
        "eight_gpu_speedup": speedup,
        "eight_gpu_scaling_efficiency": efficiency,
        "selected_topology": selected,
        "repeat_required": max(coefficient_of_variation(four_gpu_seconds), coefficient_of_variation(eight_gpu_seconds))
        > float(acceptance["maximum_timing_coefficient_of_variation"]),
    }
