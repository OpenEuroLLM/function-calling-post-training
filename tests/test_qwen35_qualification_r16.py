from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from open_instruct.qwen35_qualification_r16 import (
    BASE_MANIFEST_SHA256,
    NAMED_RELATIVE_METRIC,
    balanced_tensor_comparison_metrics,
    collect_h2_numerical_decisions,
    load_qualification_manifest,
    tensor_comparison_metrics,
    validate_h2_liger_report,
)

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r16.json"


def _scalar() -> dict:
    return {
        "observed": 1.25,
        "reference": 1.25,
        "maximum_absolute_error": 0.0,
        "relative_error": 0.0,
        "nonfinite_count": 0,
    }


def _tensor(elements: int, *, norm: float = 1.0) -> dict:
    return {
        "elements": elements,
        "maximum_absolute_error": 0.0,
        "relative_l2_error": 0.0,
        "cosine_similarity": 1.0,
        "observed_l2_norm": norm,
        "reference_l2_norm": norm,
        "difference_l2_norm": 0.0,
        "nonfinite_count": 0,
    }


def _balanced(elements: int, aggregate_elements: int, *, aggregate_norm: float = 1.0) -> dict:
    named_norm = aggregate_norm * math.sqrt(elements / aggregate_elements)
    floor = named_norm
    return {
        **_tensor(elements, norm=named_norm),
        "named_relative_metric": NAMED_RELATIVE_METRIC,
        "aggregate_elements": aggregate_elements,
        "aggregate_reference_l2_norm": aggregate_norm,
        "global_rms_allocation_floor_l2_norm": floor,
        "balanced_denominator_l2_norm": floor,
        "balanced_relative_l2_error": 0.0,
    }


def _valid_report(manifest: dict, digest: str) -> dict:
    h2 = manifest["h2_acceptance"]
    autocast = {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}
    hidden_size = h2["direct_hidden_size"]
    vocab_size = h2["direct_vocab_size"]
    head_elements = hidden_size * vocab_size

    def direct(contract: dict) -> dict:
        rows = list(range(contract["rows"])) if contract["supervision_kind"] == "all" else contract["supervised_rows"]
        return {
            "case_contract": contract,
            "supervised_rows_expanded": rows,
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
            "optimizer_step_counters": {"selective": [1], "dense_reference": [1]},
            "selective_loss": 1.25,
            "reference_loss": 1.25,
            "loss_comparison": _scalar(),
            "selected_hidden_gradient_comparison": _tensor(contract["expected_supervised_count"] * hidden_size),
            "output_head_gradient_comparison": _tensor(head_elements),
            "raw_first_adamw_update_comparison_diagnostic": _tensor(head_elements),
            "optimizer_exp_avg_comparison": _tensor(head_elements),
            "optimizer_exp_avg_sq_comparison": _tensor(head_elements),
            "post_step_parameter_comparison": _tensor(head_elements),
            "heldout": {
                "rows": h2["direct_heldout_rows"],
                "logit_comparison": _tensor(h2["direct_heldout_rows"] * vocab_size),
                "selective_loss": 1.25,
                "reference_loss": 1.25,
                "loss_comparison": _scalar(),
            },
        }

    geometry = h2["trajectory_parameter_geometry"]
    names = [row["name"] for row in geometry]
    parameter_count = h2["trajectory_parameter_count"]
    heldout_supervised = sum(
        position % h2["trajectory_heldout_supervision_modulus"] == 0
        for position in range(1, h2["trajectory_sequence_length"])
    )
    balanced_fields = (
        "preclip_gradient_comparison",
        "clipped_gradient_comparison",
        "raw_adamw_update_comparison_diagnostic",
        "optimizer_exp_avg_comparison",
        "optimizer_exp_avg_sq_comparison",
        "cumulative_parameter_displacement_comparison",
        "post_step_parameter_state_comparison",
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
            per_parameter = {
                row["name"]: {
                    "elements": row["elements"],
                    **{field: _balanced(row["elements"], parameter_count) for field in balanced_fields},
                }
                for row in geometry
            }
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
                    "training_loss_comparison": _scalar(),
                    "aggregate_preclip_gradient_comparison": _tensor(parameter_count),
                    "aggregate_clipped_gradient_comparison": _tensor(parameter_count),
                    "aggregate_raw_adamw_update_comparison_diagnostic": _tensor(parameter_count),
                    "aggregate_optimizer_exp_avg_comparison": _tensor(parameter_count),
                    "aggregate_optimizer_exp_avg_sq_comparison": _tensor(parameter_count),
                    "aggregate_cumulative_parameter_displacement_comparison": _tensor(parameter_count),
                    "aggregate_post_step_parameter_state_comparison": _tensor(parameter_count),
                    "per_parameter_comparisons": per_parameter,
                    "preclip_gradient_norms": {"selective": 1.0, "dense_reference": 1.0},
                    "raw_adamw_updates_are_gating": False,
                    "optimizer_floating_state_dtypes": {
                        "selective": ["torch.float32"],
                        "dense_reference": ["torch.float32"],
                    },
                    "optimizer_step_counters": {"selective": [step_index + 1], "dense_reference": [step_index + 1]},
                    "gradient_dtypes": {"selective": ["torch.float32"], "dense_reference": ["torch.float32"]},
                    "heldout": {
                        "supervised_targets": heldout_supervised,
                        "global_divisor": heldout_supervised + h2["trajectory_heldout_divisor_extra"],
                        "logit_comparison": _tensor(
                            h2["trajectory_sequence_length"] * h2["trajectory_model_config"]["vocab_size"]
                        ),
                        "selective_loss": 1.25,
                        "reference_loss": 1.25,
                        "loss_comparison": _scalar(),
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
                "parameter_names": names,
                "parameter_geometry": geometry,
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
        "artifact": "qwen35_selective_liger_downstream_qualification_r16",
        "schema_version": 3,
        "status": "passed",
        "successor_gate_authorized": True,
        "scientific_training_authorized": False,
        "qualification_protocol_id": manifest["protocol_id"],
        "qualification_manifest_sha256": digest,
        "manifest_derivation": manifest["manifest_derivation"],
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
        "historical_direct_cases": [direct(contract) for contract in h2["historical_direct_cases"]],
        "confirmatory_direct_cases": [direct(contract) for contract in h2["confirmatory_direct_cases"]],
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


@pytest.fixture(scope="module")
def valid_bundle() -> tuple[dict, str, dict]:
    manifest, digest = load_qualification_manifest(MANIFEST)
    return manifest, digest, _valid_report(manifest, digest)


def test_r16_manifest_is_hash_bound_and_outcome_unseen() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    assert len(digest) == 64
    assert manifest["protocol_id"] == "qwen35-hardware-qualification-r16"
    assert manifest["manifest_derivation"]["base_manifest"]["sha256"] == BASE_MANIFEST_SHA256
    h2 = manifest["h2_acceptance"]
    assert h2["protocol_revision"] == 3
    assert len(h2["historical_direct_cases"]) == 6
    assert [case["case_id"] for case in h2["confirmatory_direct_cases"]] == ["R16-D0", "R16-D1", "R16-D2"]
    assert h2["trajectory_steps"] == 128
    assert h2["raw_updates_are_diagnostic"] is True
    assert h2["r15_failed_criteria_reclassified_as_pass"] is False
    for contract in h2["confirmatory_trajectories"]:
        for role in ("model", "batch", "heldout"):
            label_key = f"{role}_seed_label" if role != "batch" else "batch_seed_label"
            sha_key = f"{role}_seed_sha256" if role != "batch" else "batch_seed_sha256"
            seed_key = f"{role}_seed" if role != "batch" else "batch_seed_base"
            expected = hashlib.sha256(contract[label_key].encode()).hexdigest()
            assert contract[sha_key] == expected
            assert contract[seed_key] == int(expected[:8], 16)


def test_r16_overlay_rejects_out_of_scope_override(tmp_path: Path) -> None:
    overlay = json.loads(MANIFEST.read_text())
    overlay["overrides"]["training_unit"] = {"learning_rate": 1.0}
    (tmp_path / "qwen35_hardware_qualification_r15.json").write_bytes(
        (MANIFEST.parent / "qwen35_hardware_qualification_r15.json").read_bytes()
    )
    path = tmp_path / MANIFEST.name
    path.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match="override scope drift"):
        load_qualification_manifest(path)


def test_balanced_metric_uses_global_rms_floor_without_hiding_absolute_error() -> None:
    reference = torch.full((16,), 1e-6)
    observed = reference + 1e-7
    metric = balanced_tensor_comparison_metrics(
        observed, reference, aggregate_reference_l2_norm=1.0, aggregate_elements=57_568
    )
    assert metric["relative_l2_error"] == pytest.approx(0.1)
    assert metric["global_rms_allocation_floor_l2_norm"] == pytest.approx(math.sqrt(16 / 57_568))
    assert metric["balanced_denominator_l2_norm"] == metric["global_rms_allocation_floor_l2_norm"]
    assert metric["balanced_relative_l2_error"] < 1e-4
    assert metric["maximum_absolute_error"] == pytest.approx(1e-7)


def test_r16_nonfinite_metrics_are_strict_json_and_fail_closed() -> None:
    metric = tensor_comparison_metrics(torch.tensor([float("nan"), 1.0]), torch.tensor([0.0, 1.0]))
    assert metric == {
        "elements": 2,
        "maximum_absolute_error": None,
        "relative_l2_error": None,
        "cosine_similarity": None,
        "observed_l2_norm": None,
        "reference_l2_norm": None,
        "difference_l2_norm": None,
        "nonfinite_count": 1,
    }
    json.dumps(metric, allow_nan=False)


def test_r16_independent_validator_accepts_complete_simulated_report(valid_bundle: tuple[dict, str, dict]) -> None:
    manifest, digest, report = valid_bundle
    validation = validate_h2_liger_report(report, qualification=manifest, expected_manifest_sha256=digest)
    assert validation == {
        "status": "passed",
        "historical_direct_cases": 6,
        "confirmatory_direct_cases": 3,
        "confirmatory_trajectories": 3,
        "trajectory_steps": 384,
        "named_tensor_checks": 34_944,
        "diagnostic_raw_update_checks": 5_385,
        "gating_checks": 33_480,
        "zero_target_sentinels": 1,
    }


def test_r16_validator_rejects_balanced_arithmetic_fabrication(valid_bundle: tuple[dict, str, dict]) -> None:
    manifest, digest, report = valid_bundle
    fabricated = copy.deepcopy(report)
    metric = fabricated["confirmatory_trajectories"][0]["steps"][0]["per_parameter_comparisons"][
        "model.layers.0.self_attn.q_norm.weight"
    ]["preclip_gradient_comparison"]
    metric["balanced_relative_l2_error"] = 0.0 + 1e-6
    fabricated["decision"] = collect_h2_numerical_decisions(fabricated, manifest)
    with pytest.raises(ValueError, match="balanced relative-L2 arithmetic drift"):
        validate_h2_liger_report(fabricated, qualification=manifest, expected_manifest_sha256=digest)


def test_r16_validator_rejects_named_aggregate_energy_fabrication(valid_bundle: tuple[dict, str, dict]) -> None:
    manifest, digest, report = valid_bundle
    fabricated = copy.deepcopy(report)
    metric = fabricated["confirmatory_trajectories"][0]["steps"][0]["per_parameter_comparisons"][
        "model.layers.0.self_attn.q_norm.weight"
    ]["optimizer_exp_avg_comparison"]
    metric["observed_l2_norm"] *= 2
    fabricated["decision"] = collect_h2_numerical_decisions(fabricated, manifest)
    with pytest.raises(ValueError, match="named/aggregate observed_l2_norm energy drift"):
        validate_h2_liger_report(fabricated, qualification=manifest, expected_manifest_sha256=digest)


def test_r16_independently_validates_but_does_not_pass_nonfinite_failure(valid_bundle: tuple[dict, str, dict]) -> None:
    manifest, digest, report = valid_bundle
    failed = copy.deepcopy(report)
    metric = failed["confirmatory_direct_cases"][0]["selected_hidden_gradient_comparison"]
    metric.update(
        {
            "maximum_absolute_error": None,
            "relative_l2_error": None,
            "cosine_similarity": None,
            "observed_l2_norm": None,
            "reference_l2_norm": None,
            "difference_l2_norm": None,
            "nonfinite_count": 1,
        }
    )
    failed["decision"] = collect_h2_numerical_decisions(failed, manifest)
    failed["status"] = "failed"
    failed["successor_gate_authorized"] = False
    json.dumps(failed, allow_nan=False)
    with pytest.raises(ValueError, match="did not publish a schema-3 pass"):
        validate_h2_liger_report(failed, qualification=manifest, expected_manifest_sha256=digest)
    validation = validate_h2_liger_report(
        failed, qualification=manifest, expected_manifest_sha256=digest, require_numerical_pass=False
    )
    assert validation["status"] == "evidence_validated"
    assert validation["numerical_status"] == "failed"
    assert validation["failed_gating_checks"] == 1
