from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from open_instruct.qwen35_qualification_loader import load_qualification_manifest as dispatch_manifest
from open_instruct.qwen35_qualification_r16 import NAMED_RELATIVE_METRIC
from open_instruct.qwen35_qualification_r17 import (
    BASE_MANIFEST_SHA256,
    DIAGNOSTIC_AGGREGATE_FIELDS,
    _validate_direct_diagnostic,
    _validate_step_diagnostic,
    load_qualification_manifest,
)

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r17.json"


def scalar() -> dict:
    return {
        "observed": 1.0,
        "reference": 1.0,
        "maximum_absolute_error": 0.0,
        "relative_error": 0.0,
        "nonfinite_count": 0,
    }


def tensor(elements: int) -> dict:
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


def balanced_tensor(elements: int, aggregate_elements: int) -> dict:
    named_norm = math.sqrt(elements / aggregate_elements)
    return {
        **tensor(elements),
        "observed_l2_norm": named_norm,
        "reference_l2_norm": named_norm,
        "named_relative_metric": NAMED_RELATIVE_METRIC,
        "aggregate_elements": aggregate_elements,
        "aggregate_reference_l2_norm": 1.0,
        "global_rms_allocation_floor_l2_norm": named_norm,
        "balanced_denominator_l2_norm": named_norm,
        "balanced_relative_l2_error": 0.0,
    }


def direct_case(contract: dict, h2: dict) -> dict:
    hidden_elements = contract["expected_supervised_count"] * h2["direct_hidden_size"]
    head_elements = h2["direct_hidden_size"] * h2["direct_vocab_size"]
    autocast = {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}
    return {
        "full_dense_diagnostic": {
            "observed_loss": 1.0,
            "reference_loss": 1.0,
            "loss_comparison": scalar(),
            "selected_hidden_gradient_comparison": tensor(hidden_elements),
            "output_head_gradient_comparison": tensor(head_elements),
            "raw_first_adamw_update_comparison_diagnostic": tensor(head_elements),
            "optimizer_exp_avg_comparison": tensor(head_elements),
            "optimizer_exp_avg_sq_comparison": tensor(head_elements),
            "post_step_parameter_comparison": tensor(head_elements),
            "heldout": {
                "logit_comparison": tensor(h2["direct_heldout_rows"] * h2["direct_vocab_size"]),
                "observed_loss": 1.0,
                "reference_loss": 1.0,
                "loss_comparison": scalar(),
            },
            "observed_path": "pytorch_dense_selected_rows",
            "reference_path": "pytorch_dense_full_rows_ignore_index",
            "numerical_discrepancy_is_gating": False,
            "integrity_and_finiteness_are_mandatory": True,
            "autocast_contract": {"dense_selected": autocast, "dense_full": autocast},
            "optimizer_step_counters": {"dense_selected": [1], "dense_full": [1]},
            "ignored_full_hidden_gradient_nonzero_count": 0,
        }
    }


def trajectory_step(h2: dict, *, step_number: int = 1) -> dict:
    geometry = h2["trajectory_parameter_geometry"]
    parameter_count = h2["trajectory_parameter_count"]
    heldout_targets = sum(
        position % h2["trajectory_heldout_supervision_modulus"] == 0
        for position in range(1, h2["trajectory_sequence_length"])
    )
    per_parameter = {
        row["name"]: {
            "elements": row["elements"],
            **{
                field: balanced_tensor(row["elements"], parameter_count)
                for field in DIAGNOSTIC_AGGREGATE_FIELDS
            },
        }
        for row in geometry
    }
    return {
        "full_dense_diagnostic": {
            "observed_path": h2["mandatory_diagnostic_observed_path"],
            "reference_path": h2["mandatory_diagnostic_reference_path"],
            "numerical_discrepancy_is_gating": False,
            "integrity_and_finiteness_are_mandatory": True,
            "observed_loss": 1.0,
            "reference_loss": 1.0,
            "training_loss_comparison": scalar(),
            **{field: tensor(parameter_count) for field in DIAGNOSTIC_AGGREGATE_FIELDS.values()},
            "per_parameter_comparisons": per_parameter,
            "preclip_gradient_norms": {"dense_selected": 1.0, "dense_full": 1.0},
            "optimizer_floating_state_dtypes": {
                "dense_selected": ["torch.float32"],
                "dense_full": ["torch.float32"],
            },
            "optimizer_step_counters": {
                "dense_selected": [step_number],
                "dense_full": [step_number],
            },
            "gradient_dtypes": {
                "dense_selected": ["torch.float32"],
                "dense_full": ["torch.float32"],
            },
            "heldout": {
                "supervised_targets": heldout_targets,
                "global_divisor": heldout_targets + h2["trajectory_heldout_divisor_extra"],
                "logit_comparison": tensor(
                    h2["trajectory_sequence_length"] * h2["trajectory_model_config"]["vocab_size"]
                ),
                "observed_loss": 1.0,
                "reference_loss": 1.0,
                "loss_comparison": scalar(),
            },
        }
    }


def test_r17_manifest_is_hash_bound_outcome_unseen_and_threshold_preserving() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    assert len(digest) == 64
    assert manifest["protocol_id"] == "qwen35-hardware-qualification-r17"
    assert manifest["manifest_derivation"]["base_manifest"]["sha256"] == BASE_MANIFEST_SHA256
    h2 = manifest["h2_acceptance"]
    assert h2["protocol_revision"] == 4
    assert h2["trajectory_steps"] == 512
    assert h2["primary_reference_path"] == "pytorch_dense_selected_rows"
    assert h2["full_dense_diagnostic_numerical_discrepancy_is_gating"] is False
    assert h2["liger_numerical_failure_policy"] == "abandon_liger_no_outcome_fitted_threshold_rescue"
    assert h2["r16_failed_criteria_reclassified_as_pass"] is False
    assert [case["case_id"] for case in h2["confirmatory_direct_cases"]] == ["R17-D0", "R17-D1", "R17-D2"]
    labels = []
    for case in h2["confirmatory_direct_cases"]:
        expected = hashlib.sha256(case["seed_label"].encode()).hexdigest()
        assert case["seed_sha256"] == expected
        assert case["seed"] == int(expected[:8], 16)
        labels.append(case["seed_label"])
    for contract in h2["confirmatory_trajectories"]:
        for prefix, seed_key in (("model", "model_seed"), ("batch", "batch_seed_base"), ("heldout", "heldout_seed")):
            label = contract[f"{prefix}_seed_label"]
            expected = hashlib.sha256(label.encode()).hexdigest()
            assert contract[f"{prefix}_seed_sha256"] == expected
            assert contract[seed_key] == int(expected[:8], 16)
            labels.append(label)
    assert len(labels) == len(set(labels)) == 12


def test_qualification_dispatcher_preserves_r16_and_resolves_r17() -> None:
    r16, _ = dispatch_manifest(MANIFEST.parent / "qwen35_hardware_qualification_r16.json")
    r17, _ = dispatch_manifest(MANIFEST)
    assert r16["protocol_id"] == "qwen35-hardware-qualification-r16"
    assert r17["protocol_id"] == "qwen35-hardware-qualification-r17"
    assert r16["numerical_acceptance"] == r17["numerical_acceptance"]


def test_r17_overlay_rejects_out_of_scope_override(tmp_path: Path) -> None:
    overlay = json.loads(MANIFEST.read_text())
    overlay["overrides"]["training_unit"] = {"learning_rate": 1.0}
    for name in ("qwen35_hardware_qualification_r15.json", "qwen35_hardware_qualification_r16.json"):
        (tmp_path / name).write_bytes((MANIFEST.parent / name).read_bytes())
    path = tmp_path / MANIFEST.name
    path.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match="override scope drift"):
        load_qualification_manifest(path)


def test_r17_direct_diagnostic_validator_accepts_complete_finite_evidence() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    contract = h2["confirmatory_direct_cases"][0]
    assert _validate_direct_diagnostic(direct_case(contract, h2), contract, h2, "valid") == 9


def test_r17_direct_diagnostic_validator_rejects_gating_or_nonfinite_fabrication() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    contract = h2["confirmatory_direct_cases"][0]
    gating = direct_case(contract, h2)
    gating["full_dense_diagnostic"]["numerical_discrepancy_is_gating"] = True
    with pytest.raises(ValueError, match="role drift"):
        _validate_direct_diagnostic(gating, contract, h2, "gating")
    nonfinite = direct_case(contract, h2)
    metric = nonfinite["full_dense_diagnostic"]["selected_hidden_gradient_comparison"]
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
    with pytest.raises(ValueError, match="nonfinite diagnostic tensor"):
        _validate_direct_diagnostic(nonfinite, contract, h2, "nonfinite")


def test_r17_direct_diagnostic_validator_rejects_ignored_row_gradient() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    contract = h2["confirmatory_direct_cases"][0]
    fabricated = copy.deepcopy(direct_case(contract, h2))
    fabricated["full_dense_diagnostic"]["ignored_full_hidden_gradient_nonzero_count"] = 1
    with pytest.raises(ValueError, match="ignored full hidden rows received gradient"):
        _validate_direct_diagnostic(fabricated, contract, h2, "fabricated")


def test_r17_step_diagnostic_validator_checks_every_named_tensor_partition() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    geometry = h2["trajectory_parameter_geometry"]
    names = [row["name"] for row in geometry]
    checks = _validate_step_diagnostic(
        step=trajectory_step(h2),
        step_number=1,
        names=names,
        elements_by_name={row["name"]: row["elements"] for row in geometry},
        parameter_count=h2["trajectory_parameter_count"],
        h2=h2,
        context="valid step",
    )
    assert checks == len(geometry) * len(DIAGNOSTIC_AGGREGATE_FIELDS)


def test_r17_step_diagnostic_validator_rejects_derived_metric_and_partition_tampering() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    geometry = h2["trajectory_parameter_geometry"]
    names = [row["name"] for row in geometry]
    kwargs = {
        "step_number": 1,
        "names": names,
        "elements_by_name": {row["name"]: row["elements"] for row in geometry},
        "parameter_count": h2["trajectory_parameter_count"],
        "h2": h2,
        "context": "tampered step",
    }

    arithmetic = trajectory_step(h2)
    first = arithmetic["full_dense_diagnostic"]["per_parameter_comparisons"][names[0]]
    first["preclip_gradient_comparison"]["balanced_relative_l2_error"] = 0.5
    with pytest.raises(ValueError, match="balanced relative-L2 arithmetic drift"):
        _validate_step_diagnostic(step=arithmetic, **kwargs)

    partition = trajectory_step(h2)
    first = partition["full_dense_diagnostic"]["per_parameter_comparisons"][names[0]]
    first["preclip_gradient_comparison"]["observed_l2_norm"] *= 2
    with pytest.raises(ValueError, match="named/aggregate observed_l2_norm energy drift"):
        _validate_step_diagnostic(step=partition, **kwargs)
