from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch
from scripts.train.qwen35.compare_qwen35_reporting_overhead import compare_scientific_exposure
from scripts.train.qwen35.validate_qwen35_reference import (
    canonicalize_json_metadata,
    exact_invariance_passes,
    exact_segment_evidence,
    mutate_every_token,
    mutate_one_token,
    tensor_parity_metrics,
    write_strict_json_atomic,
)
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification import (
    coefficient_of_variation,
    collect_h2_numerical_decisions,
    load_qualification_manifest,
    parse_glibc_versions,
    scalar_comparison_metrics,
    select_topology,
    tensor_comparison_metrics,
    validate_comparison_metrics,
    validate_h2_liger_report,
    validate_memory_headroom,
)

MANIFEST = Path(__file__).parents[1] / "scripts/train/qwen35/qwen35_hardware_qualification_r15.json"
GUARD = Path(__file__).parents[1] / "scripts/train/qwen35/g2_job_guard.sh"


def _exact_h2_tensor_metrics(elements: int) -> dict:
    return {
        "elements": elements,
        "maximum_absolute_error": 0.0,
        "relative_l2_error": 0.0,
        "cosine_similarity": 1.0,
        "observed_l2_norm": 1.0,
        "reference_l2_norm": 1.0,
        "difference_l2_norm": 0.0,
        "nonfinite_count": 0,
    }


def _valid_h2_report(manifest: dict, manifest_sha256: str) -> dict:
    autocast = {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}
    loss = {
        "observed": 1.25,
        "reference": 1.25,
        "maximum_absolute_error": 0.0,
        "relative_error": 0.0,
        "nonfinite_count": 0,
    }
    hidden_size = manifest["h2_acceptance"]["direct_hidden_size"]
    vocab_size = manifest["h2_acceptance"]["direct_vocab_size"]
    h2 = manifest["h2_acceptance"]

    def direct_case(contract: dict) -> dict:
        supervised_rows = (
            list(range(contract["rows"])) if contract["supervision_kind"] == "all" else contract["supervised_rows"]
        )
        return {
            "case_contract": contract,
            "supervised_rows_expanded": supervised_rows,
            "autocast_contract": {"selective": autocast, "dense_reference": autocast, "heldout": autocast},
            "dtypes": {
                "hidden_input": "torch.bfloat16",
                "output_head_parameter": "torch.float32",
                "selective_hidden_gradient": "torch.bfloat16",
                "reference_hidden_gradient": "torch.bfloat16",
                "selective_output_head_gradient": "torch.float32",
                "reference_output_head_gradient": "torch.float32",
                "selective_optimizer_floating_state": ["torch.float32"],
                "reference_optimizer_floating_state": ["torch.float32"],
                "loss_accumulation": "torch.float32",
            },
            "selective_loss": 1.25,
            "reference_loss": 1.25,
            "loss_comparison": loss,
            "selected_hidden_gradient_comparison": _exact_h2_tensor_metrics(
                contract["expected_supervised_count"] * hidden_size
            ),
            "output_head_gradient_comparison": _exact_h2_tensor_metrics(vocab_size * hidden_size),
            "raw_first_adamw_update_comparison_diagnostic": _exact_h2_tensor_metrics(vocab_size * hidden_size),
            "post_step_parameter_comparison": _exact_h2_tensor_metrics(vocab_size * hidden_size),
            "heldout": {
                "rows": h2["direct_heldout_rows"],
                "logit_comparison": _exact_h2_tensor_metrics(h2["direct_heldout_rows"] * vocab_size),
                "selective_loss": 1.25,
                "reference_loss": 1.25,
                "loss_comparison": loss,
            },
        }

    parameter_geometry = copy.deepcopy(h2["trajectory_parameter_geometry"])
    parameter_names = [row["name"] for row in parameter_geometry]
    parameter_count = h2["trajectory_parameter_count"]
    heldout_supervised = sum(
        position % h2["trajectory_heldout_supervision_modulus"] == 0
        for position in range(1, h2["trajectory_sequence_length"])
    )
    trajectories = []
    for trajectory_index, contract in enumerate(h2["confirmatory_trajectories"]):
        steps = []
        for step_index in range(h2["trajectory_steps"]):
            modulus = h2["trajectory_supervision_moduli"][step_index % len(h2["trajectory_supervision_moduli"])]
            offset = (step_index + trajectory_index) % modulus
            supervised = sum(
                (position + offset) % modulus == 0 for position in range(1, h2["trajectory_sequence_length"])
            )
            divisor_extra = (step_index * h2["trajectory_divisor_extra_multiplier"] + trajectory_index) % h2[
                "trajectory_divisor_extra_modulus"
            ]
            steps.append(
                {
                    "step": step_index + 1,
                    "batch_accounting": {
                        "seed": contract["batch_seed_base"] + step_index,
                        "sequence_length": h2["trajectory_sequence_length"],
                        "supervision_modulus": modulus,
                        "supervision_offset": offset,
                        "supervised_targets": supervised,
                        "divisor_extra": divisor_extra,
                        "global_divisor": supervised + divisor_extra,
                    },
                    "autocast_contract": {"training": autocast, "heldout": autocast},
                    "selective_loss": 1.25,
                    "reference_loss": 1.25,
                    "training_loss_comparison": loss,
                    "aggregate_preclip_gradient_comparison": _exact_h2_tensor_metrics(parameter_count),
                    "aggregate_clipped_gradient_comparison": _exact_h2_tensor_metrics(parameter_count),
                    "per_parameter_gradient_comparisons": {
                        row["name"]: {
                            "elements": row["elements"],
                            "preclip_gradient_comparison": _exact_h2_tensor_metrics(row["elements"]),
                            "clipped_gradient_comparison": _exact_h2_tensor_metrics(row["elements"]),
                        }
                        for row in parameter_geometry
                    },
                    "preclip_gradient_norms": {"selective": 1.0, "dense_reference": 1.0},
                    "raw_adamw_update_comparison": _exact_h2_tensor_metrics(parameter_count),
                    "raw_adamw_update_is_gating": step_index + 1 >= h2["raw_update_gating_starts_at_step"],
                    "post_step_parameter_comparison": _exact_h2_tensor_metrics(parameter_count),
                    "optimizer_floating_state_dtypes": {
                        "selective": ["torch.float32"],
                        "dense_reference": ["torch.float32"],
                    },
                    "optimizer_step_counters": {"selective": [step_index + 1], "dense_reference": [step_index + 1]},
                    "gradient_dtypes": {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]},
                    "heldout": {
                        "supervised_targets": heldout_supervised,
                        "global_divisor": heldout_supervised + h2["trajectory_heldout_divisor_extra"],
                        "logit_comparison": _exact_h2_tensor_metrics(
                            h2["trajectory_sequence_length"] * h2["trajectory_model_config"]["vocab_size"]
                        ),
                        "selective_loss": 1.25,
                        "reference_loss": 1.25,
                        "loss_comparison": loss,
                    },
                }
            )
        trajectories.append(
            {
                "trajectory_contract": contract,
                "trajectory_index": trajectory_index,
                "model_class": "Qwen3_5ForCausalLM",
                "dense_forward_module": "transformers.models.qwen3_5.modeling_qwen3_5",
                "patched_forward_module": "liger_kernel.transformers.model.qwen3_5",
                "model_config": h2["trajectory_model_config"],
                "parameter_names": parameter_names,
                "parameter_geometry": parameter_geometry,
                "parameter_count": parameter_count,
                "parameter_dtypes": {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]},
                "heldout_contract": {
                    "seed": contract["heldout_seed"],
                    "sequence_length": h2["trajectory_sequence_length"],
                    "supervision_modulus": h2["trajectory_heldout_supervision_modulus"],
                    "supervised_targets": heldout_supervised,
                    "divisor_extra": h2["trajectory_heldout_divisor_extra"],
                    "global_divisor": heldout_supervised + h2["trajectory_heldout_divisor_extra"],
                },
                "steps": steps,
            }
        )
    report = {
        "artifact": "qwen35_selective_liger_downstream_qualification",
        "schema_version": 2,
        "status": "passed",
        "successor_gate_authorized": True,
        "scientific_training_authorized": False,
        "qualification_protocol_id": manifest["protocol_id"],
        "qualification_manifest_sha256": manifest_sha256,
        "torch_version": manifest["runtime_pins"]["torch_version"],
        "cuda_device": "NVIDIA A100-SXM-64GB",
        "direct_hidden_size": hidden_size,
        "direct_vocab_size": vocab_size,
        "liger_kernel": {
            "version": manifest["runtime_pins"]["liger_version"],
            "commit": manifest["runtime_pins"]["liger_commit"],
            "source_url": (
                f"https://github.com/linkedin/Liger-Kernel/archive/{manifest['runtime_pins']['liger_commit']}.tar.gz"
            ),
            "metadata_vcs_commit": None,
            "archive_url_pinned": True,
            "implementation_files": {
                relative_path: {
                    "path": f"/runtime/pinned-sources/liger-kernel/src/liger_kernel/{relative_path}",
                    "sha256": sha256,
                }
                for relative_path, sha256 in manifest["runtime_pins"]["liger_source_files_sha256"].items()
            },
        },
        "precision_policy": {
            "parameters": "torch.float32",
            "gradients": "dtype follows FP32 parameter storage; direct selected BF16 hidden-row leaf gradients are BF16",
            "adamw_moments": "torch.float32",
            "forward_backward_autocast": "torch.bfloat16",
            "loss_accumulation": "torch.float32",
        },
        "numerical_acceptance": manifest["numerical_acceptance"],
        "h2_acceptance": h2,
        "historical_direct_cases": [direct_case(contract) for contract in h2["historical_direct_cases"]],
        "confirmatory_direct_cases": [direct_case(contract) for contract in h2["confirmatory_direct_cases"]],
        "zero_target_sentinel": {
            "loss": 0.0,
            "global_divisor": 7,
            "autocast_contract": autocast,
            "hidden_input_dtype": "torch.bfloat16",
            "output_head_parameter_dtype": "torch.float32",
            "hidden_gradient_dtype": "torch.bfloat16",
            "output_head_gradient_dtype": "torch.float32",
            "hidden_gradient_connected": True,
            "weight_gradient_connected": True,
            "gradient_nonzero_count": 0,
        },
        "confirmatory_trajectories": trajectories,
    }
    report["decision"] = collect_h2_numerical_decisions(report, manifest)
    return report


def test_manifest_is_valid_and_hashed() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    assert manifest["protocol_id"] == "qwen35-hardware-qualification-r15"
    assert manifest["status"] == "ready_for_execution"
    assert len(digest) == 64
    assert [gate["gate_id"] for gate in manifest["gates"]] == [f"H{i}" for i in range(10)]
    assert manifest["hardware_acceptance"]["compute_capability"] == [8, 0]
    model = manifest["model"]
    assert model["vocabulary_size"] == 248320
    assert model["text_hidden_size"] == 1024
    assert model["text_num_hidden_layers"] == 24
    assert (
        model["text_layer_types"] == ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
    )
    assert model["text_state_tensor_count"] == 321
    assert model["text_state_numel"] == 1_006_672_704
    native = manifest["runtime_pins"]["native_extensions"]
    assert set(native) == {"causal-conv1d", "flash-attn"}
    assert all(value["maximum_glibc_version"] == [2, 28] for value in native.values())
    h1 = manifest["h1_acceptance"]
    assert h1["full_document_counterfactual_unchanged_segment_bit_exact"] is True
    assert h1["corrupted_boundary_negative_control_must_show_cross_document_influence"] is True
    assert h1["singleton_multi_pack_shape_diagnostic_is_gating"] is False
    assert h1["r11_failed_singleton_multi_criterion_reclassified_as_pass"] is False
    assert h1["ordinary_vs_variable_length_cross_kernel_diagnostic_is_gating"] is False
    assert h1["tolerance_change_from_r9"] is False
    h2 = manifest["h2_acceptance"]
    assert h2["direct_hidden_size"] == 256
    assert h2["direct_vocab_size"] == 4096
    assert h2["protocol_revision"] == 2
    assert [case["expected_supervised_count"] for case in h2["historical_direct_cases"]] == [64, 4, 1]
    assert [case["global_divisor"] for case in h2["historical_direct_cases"]] == [64, 23, 128]
    assert len(h2["confirmatory_direct_cases"]) == 3
    assert len(h2["confirmatory_trajectories"]) == 3
    assert h2["trajectory_steps"] == 32
    assert h2["raw_first_step_update_is_gating"] is False
    assert h2["raw_update_gating_starts_at_step"] == 2
    assert h2["direct_fused_and_dense_reference_use_bf16_autocast"] is True
    assert h2["r14_failed_first_step_update_reclassified_as_pass"] is False
    assert manifest["runtime_pins"]["liger_source_files_sha256"] == {
        "transformers/fused_linear_cross_entropy.py": (
            "063020937ac6caa19e92821966abd73011c00f9899c8d316f1c57106666640e8"
        ),
        "ops/fused_linear_cross_entropy.py": ("765747ba10bae599cd65c7c5f3dfebfd620e526bceb7696c8cef6b0cf42433ce"),
        "ops/utils.py": "f7457c3f412565e05ab610e19882a6ff9d3a361aecaf5a233b4a91979dff4e7d",
    }
    assert manifest["evidence_serialization"]["format"] == "strict_json_rfc8259_no_nan_or_infinity"
    fixture = manifest["reference_fixture"]
    assert (
        fixture["fixture_sha256"]
        == hashlib.sha256((Path(__file__).parents[1] / fixture["fixture_path"]).read_bytes()).hexdigest()
    )


def test_manifest_rejects_account_drift(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text())
    value["scope"]["slurm_account"] = "oellm_prod2026"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="personal Slurm"):
        load_qualification_manifest(path)


def test_manifest_rejects_confirmatory_seed_or_hash_drift(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text())
    value["h2_acceptance"]["confirmatory_trajectories"][0]["model_seed"] += 1
    path = tmp_path / "bad-seed.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="holdout-seed"):
        load_qualification_manifest(path)


def test_h2_frozen_trajectory_geometry_matches_pinned_qwen_implementation() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"]))
    observed = [
        {"name": name, "shape": list(parameter.shape), "elements": int(parameter.numel())}
        for name, parameter in model.named_parameters()
    ]
    assert observed == h2["trajectory_parameter_geometry"]
    assert sum(row["elements"] for row in observed) == h2["trajectory_parameter_count"] == 57_568


def test_h2_independent_validator_requires_complete_autocast_and_dtype_evidence() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    report = _valid_h2_report(manifest, digest)
    assert validate_h2_liger_report(report, qualification=manifest, expected_manifest_sha256=digest) == {
        "status": "passed",
        "historical_direct_cases": 3,
        "confirmatory_direct_cases": 3,
        "confirmatory_trajectories": 3,
        "trajectory_steps": 96,
        "parameter_gradient_checks": 2496,
        "diagnostic_first_updates": 3,
        "gated_stateful_updates": 93,
        "zero_target_sentinels": 1,
    }

    missing_autocast = copy.deepcopy(report)
    missing_autocast["confirmatory_direct_cases"][0]["autocast_contract"]["selective"]["enabled"] = False
    with pytest.raises(ValueError, match="autocast"):
        validate_h2_liger_report(missing_autocast, qualification=manifest, expected_manifest_sha256=digest)

    wrong_moments = copy.deepcopy(report)
    wrong_moments["confirmatory_trajectories"][0]["steps"][0]["optimizer_floating_state_dtypes"]["selective"] = [
        "torch.bfloat16"
    ]
    with pytest.raises(ValueError, match="floating-state"):
        validate_h2_liger_report(wrong_moments, qualification=manifest, expected_manifest_sha256=digest)

    incomplete_liger_source = copy.deepcopy(report)
    incomplete_liger_source["liger_kernel"]["implementation_files"].pop("ops/utils.py")
    with pytest.raises(ValueError, match="source-file set"):
        validate_h2_liger_report(incomplete_liger_source, qualification=manifest, expected_manifest_sha256=digest)

    fabricated_status_only = {
        "artifact": "qwen35_selective_liger_downstream_qualification",
        "schema_version": 2,
        "status": "passed",
        "qualification_protocol_id": manifest["protocol_id"],
        "qualification_manifest_sha256": digest,
    }
    with pytest.raises(ValueError):
        validate_h2_liger_report(fabricated_status_only, qualification=manifest, expected_manifest_sha256=digest)


def test_h2_first_raw_update_is_diagnostic_but_second_update_is_gating() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    first_only = _valid_h2_report(manifest, digest)
    first_update = first_only["confirmatory_trajectories"][0]["steps"][0]["raw_adamw_update_comparison"]
    first_update["relative_l2_error"] = 0.5
    first_update["cosine_similarity"] = 0.5
    first_only["decision"] = collect_h2_numerical_decisions(first_only, manifest)
    assert first_only["decision"]["status"] == "passed"
    assert first_only["decision"]["failed_diagnostic_checks"] == ["R15-T0 step 1 raw AdamW update"]
    validate_h2_liger_report(first_only, qualification=manifest, expected_manifest_sha256=digest)

    second = _valid_h2_report(manifest, digest)
    second_update = second["confirmatory_trajectories"][0]["steps"][1]["raw_adamw_update_comparison"]
    second_update["relative_l2_error"] = 0.5
    second_update["cosine_similarity"] = 0.5
    second["decision"] = collect_h2_numerical_decisions(second, manifest)
    with pytest.raises(AssertionError, match="relative-L2"):
        validate_h2_liger_report(second, qualification=manifest, expected_manifest_sha256=digest)


def test_h2_independently_validates_complete_failed_evidence_without_passing_it() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    report = _valid_h2_report(manifest, digest)
    failed_update = report["confirmatory_trajectories"][0]["steps"][1]["raw_adamw_update_comparison"]
    failed_update["relative_l2_error"] = 0.5
    failed_update["cosine_similarity"] = 0.5
    report["decision"] = collect_h2_numerical_decisions(report, manifest)
    report["status"] = "failed"
    report["successor_gate_authorized"] = False
    with pytest.raises(ValueError, match="did not publish a schema-2 pass"):
        validate_h2_liger_report(report, qualification=manifest, expected_manifest_sha256=digest)
    validation = validate_h2_liger_report(
        report, qualification=manifest, expected_manifest_sha256=digest, require_numerical_pass=False
    )
    assert validation["status"] == "evidence_validated"
    assert validation["numerical_status"] == "failed"
    assert validation["failed_gating_checks"] == 1


def test_h2_rejects_missing_holdout_step_or_fabricated_decision() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    missing_step = _valid_h2_report(manifest, digest)
    missing_step["confirmatory_trajectories"][1]["steps"].pop()
    with pytest.raises(ValueError, match="step cardinality"):
        validate_h2_liger_report(missing_step, qualification=manifest, expected_manifest_sha256=digest)

    bad_logits = _valid_h2_report(manifest, digest)
    metrics = bad_logits["confirmatory_trajectories"][2]["steps"][17]["heldout"]["logit_comparison"]
    metrics["maximum_absolute_error"] = 0.051
    with pytest.raises(AssertionError, match="packed-logit"):
        validate_h2_liger_report(bad_logits, qualification=manifest, expected_manifest_sha256=digest)

    fabricated = _valid_h2_report(manifest, digest)
    fabricated["decision"]["checks"] = []
    fabricated["decision"]["total_checks"] = 0
    fabricated["decision"]["gating_checks"] = 0
    fabricated["decision"]["diagnostic_checks"] = 0
    with pytest.raises(ValueError, match="decision ledger"):
        validate_h2_liger_report(fabricated, qualification=manifest, expected_manifest_sha256=digest)


def test_h2_rejects_omitted_parameter_even_when_reported_total_is_self_consistent() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    report = _valid_h2_report(manifest, digest)
    trajectory = report["confirmatory_trajectories"][0]
    omitted = trajectory["parameter_geometry"].pop()
    trajectory["parameter_names"].pop()
    trajectory["parameter_count"] -= omitted["elements"]
    for step in trajectory["steps"]:
        step["per_parameter_gradient_comparisons"].pop(omitted["name"])
        for field in (
            "aggregate_preclip_gradient_comparison",
            "aggregate_clipped_gradient_comparison",
            "raw_adamw_update_comparison",
            "post_step_parameter_comparison",
        ):
            step[field]["elements"] -= omitted["elements"]
    report["decision"] = collect_h2_numerical_decisions(report, manifest)
    with pytest.raises(ValueError, match="parameter geometry"):
        validate_h2_liger_report(report, qualification=manifest, expected_manifest_sha256=digest)


def test_h2_decision_ledger_is_stable_across_strict_json_key_sorting(tmp_path: Path) -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    report = _valid_h2_report(manifest, digest)
    path = tmp_path / "h2.json"
    write_strict_json_atomic(path, report)
    reloaded = json.loads(path.read_text())
    assert reloaded["decision"] == collect_h2_numerical_decisions(reloaded, manifest)
    validate_h2_liger_report(reloaded, qualification=manifest, expected_manifest_sha256=digest)


def test_tensor_metrics_exact_and_perturbed() -> None:
    reference = torch.tensor([1.0, -2.0, 3.0])
    exact = tensor_comparison_metrics(reference, reference)
    assert exact["maximum_absolute_error"] == 0
    assert exact["relative_l2_error"] == 0
    assert exact["cosine_similarity"] == 1
    perturbed = tensor_comparison_metrics(reference + 0.1, reference)
    assert perturbed["maximum_absolute_error"] == pytest.approx(0.1)
    assert 0 < float(perturbed["relative_l2_error"]) < 1


def test_h1_tensor_parity_metrics_preserve_exact_and_tolerance_failures() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    exact = tensor_parity_metrics(reference.clone(), reference, atol=0, rtol=0)
    assert exact["bit_exact"] is True
    assert exact["allclose"] is True
    assert exact["mismatched_elements"] == 0
    perturbed = tensor_parity_metrics(reference + 0.1, reference, atol=0.05, rtol=0)
    assert perturbed["bit_exact"] is False
    assert perturbed["allclose"] is False
    assert perturbed["mismatched_elements"] == reference.numel()
    assert perturbed["top1_agreement"] == 1.0
    nonfinite = tensor_parity_metrics(torch.tensor([float("nan")]), torch.zeros(1), atol=0, rtol=0)
    assert nonfinite["nonfinite_count"] == 1
    assert nonfinite["maximum_absolute_error"] is None
    json.dumps(nonfinite, allow_nan=False)


def test_h1_counterfactual_mutation_changes_exactly_one_in_range_token() -> None:
    original = [1, 2, 3, 4, 5]
    mutated, position = mutate_one_token(original, vocabulary_size=16)
    assert len(mutated) == len(original)
    assert 0 <= mutated[position] < 16
    assert mutated[position] != original[position]
    assert sum(left != right for left, right in zip(original, mutated, strict=True)) == 1


def test_h1_full_document_mutation_changes_every_token_without_length_drift() -> None:
    original = [1, 2, 3, 4, 5]
    mutated = mutate_every_token(original, vocabulary_size=16)
    assert len(mutated) == len(original)
    assert all(0 <= token < 16 for token in mutated)
    assert all(left != right for left, right in zip(original, mutated, strict=True))


def test_h1_segment_evidence_compares_explicit_candidate_and_reference_ranges() -> None:
    reference = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    candidate = torch.cat([reference[3:], reference[:3]])
    reference_layers = {0: reference[:, :2], 1: reference[:, 2:]}
    candidate_layers = {0: candidate[:, :2], 1: candidate[:, 2:]}
    evidence = exact_segment_evidence(
        candidate,
        reference,
        candidate_layers,
        reference_layers,
        candidate_start=3,
        reference_start=0,
        length=3,
        layer_types=["linear_attention", "full_attention"],
    )
    assert exact_invariance_passes(evidence)
    candidate[3, 0] += 1
    evidence = exact_segment_evidence(
        candidate,
        reference,
        candidate_layers,
        reference_layers,
        candidate_start=3,
        reference_start=0,
        length=3,
        layer_types=["linear_attention", "full_attention"],
    )
    assert not exact_invariance_passes(evidence)


def test_h1_external_metadata_is_canonicalized_to_strict_json_without_content_loss() -> None:
    raw = {"missing_keys": {"model.z", "model.a"}, "nested": ({"beta", "alpha"}, (3, 2, 1)), "empty": set()}
    normalized = canonicalize_json_metadata(raw)
    assert normalized == {
        "empty": [],
        "missing_keys": ["model.a", "model.z"],
        "nested": [["alpha", "beta"], [3, 2, 1]],
    }
    assert canonicalize_json_metadata(raw) == normalized
    json.dumps(normalized, allow_nan=False, sort_keys=True)


def test_h1_external_metadata_rejects_ambiguous_or_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="non-string mapping key"):
        canonicalize_json_metadata({1: "ambiguous"})
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_json_metadata({"value": float("inf")})
    with pytest.raises(TypeError, match="unsupported object"):
        canonicalize_json_metadata({"value": object()})


def test_h1_strict_json_writer_rejects_nonstandard_numbers_before_publication(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_strict_json_atomic(output, {"value": float("nan")})
    assert not output.exists()
    write_strict_json_atomic(output, {"status": "passed", "values": [1, 2, 3]})
    assert json.loads(output.read_text()) == {"status": "passed", "values": [1, 2, 3]}


def test_zero_tensor_cosine_is_well_defined() -> None:
    metrics = tensor_comparison_metrics(torch.zeros(4), torch.zeros(4))
    assert metrics["cosine_similarity"] == 1


def test_glibc_version_parser_is_numeric_unique_and_ordered() -> None:
    output = "GLIBC_2.2.5 GLIBC_2.32 GLIBC_2.28 GLIBC_2.9 GLIBC_2.32"
    assert parse_glibc_versions(output) == [[2, 2], [2, 9], [2, 28], [2, 32]]


def test_nonfinite_metrics_fail() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    metrics = tensor_comparison_metrics(torch.tensor([float("nan")]), torch.tensor([0.0]))
    with pytest.raises(AssertionError, match="nonfinite"):
        validate_comparison_metrics(metrics, manifest["numerical_acceptance"], kind="gradient", context="bad")


def test_scalar_and_tensor_threshold_validation() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    acceptance = manifest["numerical_acceptance"]
    validate_comparison_metrics(scalar_comparison_metrics(1.0001, 1.0), acceptance, kind="loss", context="loss")
    validate_comparison_metrics(
        tensor_comparison_metrics(torch.tensor([1.0001]), torch.tensor([1.0])),
        acceptance,
        kind="gradient",
        context="gradient",
    )
    with pytest.raises(AssertionError):
        validate_comparison_metrics(scalar_comparison_metrics(2.0, 1.0), acceptance, kind="loss", context="bad loss")


def test_memory_headroom_boundary() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    acceptance = manifest["memory_acceptance"]
    result = validate_memory_headroom(
        peak_allocated_bytes=80, peak_reserved_bytes=90, total_device_bytes=100, acceptance=acceptance
    )
    assert result["headroom_fraction"] == pytest.approx(0.1)
    with pytest.raises(AssertionError):
        validate_memory_headroom(
            peak_allocated_bytes=80, peak_reserved_bytes=91, total_device_bytes=100, acceptance=acceptance
        )


def test_topology_selection_and_variability() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    stable_four = [10.0, 10.1, 9.9]
    stable_eight = [7.0, 7.1, 6.9]
    result = select_topology(stable_four, stable_eight, manifest["topology_acceptance"])
    assert result["selected_topology"] == "T8"
    assert result["repeat_required"] is False
    slow_eight = select_topology(stable_four, [9.0, 9.1, 8.9], manifest["topology_acceptance"])
    assert slow_eight["selected_topology"] == "T4"
    assert coefficient_of_variation(stable_four) < 0.05


def test_topology_uses_conventional_even_sample_median() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    result = select_topology([1.0, 2.0, 100.0, 101.0], [1.0, 1.0, 1.0, 1.0], manifest["topology_acceptance"])
    assert result["four_gpu_median_seconds"] == pytest.approx(51.0)


def test_reporting_exposure_comparison_is_target_weighted_and_fail_closed() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    counts_one = {"fixed_tokens": 10, "assistant_targets": 2}
    counts_two = {"fixed_tokens": 10, "assistant_targets": 3}
    fine = [
        {
            "schedule_indices": [0],
            "pack_uids": ["p0"],
            "counts": counts_one,
            "loss": {"normalized_loss": 1.0},
            "optimizer": {"applied_learning_rates": [2e-5]},
        },
        {
            "schedule_indices": [1],
            "pack_uids": ["p1"],
            "counts": counts_two,
            "loss": {"normalized_loss": 2.0},
            "optimizer": {"applied_learning_rates": [1e-5]},
        },
    ]
    coarse = [
        {
            "schedule_indices": [0, 1],
            "pack_uids": ["p0", "p1"],
            "counts": {"fixed_tokens": 20, "assistant_targets": 5},
            "loss": {"normalized_loss": 1.6},
            "optimizer": {"applied_learning_rates": [2e-5, 1e-5]},
        }
    ]
    report = compare_scientific_exposure(fine, coarse, manifest["numerical_acceptance"])
    assert report["aggregate_loss_comparison"]["maximum_absolute_error"] == pytest.approx(0)
    coarse[0]["pack_uids"] = ["p1", "p0"]
    with pytest.raises(AssertionError, match="schedule, pack identity"):
        compare_scientific_exposure(fine, coarse, manifest["numerical_acceptance"])


def test_shell_guard_accepts_only_the_personal_canonical_root() -> None:
    accepted = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; g2_require_personal_path test "$2"',
            "guard-test",
            str(GUARD),
            "/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3/qualification/output",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; g2_require_personal_path test "$2"',
            "guard-test",
            str(GUARD),
            "/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3-escape/output",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "outside personal" in rejected.stderr


def test_shell_guard_pins_repository_imports_and_rejects_python_overlays() -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; '
                'export QWEN35_REPO="$2" PYTHONPATH=/contaminating/path '
                "PYTHONHOME=/contaminating/home PYTHONNOUSERSITE=0; "
                "g2_pin_python_import_environment; "
                'printf "%s\\n%s\\n%s\\n" "$PYTHONPATH" "$PYTHONNOUSERSITE" "${PYTHONHOME-unset}"'
            ),
            "guard-test",
            str(GUARD),
            "/frozen/open-instruct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["/frozen/open-instruct", "1", "unset"]
