from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID, QUALIFIED_CHUNK_SIZES
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_report import (
    DIRECT_FAMILIES,
    TRAJECTORY_PARAMETER_FAMILIES,
    diagnostic_tensor_comparison_metrics,
    exact_tensor_comparison_metrics,
    validate_h2_chunked_report,
)

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18.json"
AUTocast = {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"}
ROLES = ("observed", "reference", "unchunked", "full_ignore")


def exact(elements: int, shape: list[int] | None = None) -> dict:
    return {
        "shape": shape or [elements],
        "elements": elements,
        "observed_dtype": "torch.float32",
        "reference_dtype": "torch.float32",
        "observed_nonfinite_count": 0,
        "reference_nonfinite_count": 0,
        "value_equal": True,
        "bitwise_equal": True,
        "mismatched_values": 0,
        "mismatched_bytes": 0,
        "maximum_absolute_error": 0.0,
    }


def diagnostic(elements: int) -> dict:
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


def direct_comparison(rows: int, hidden: int, vocabulary: int, metric) -> dict:
    sizes = {
        "loss": 1,
        "selected_hidden_gradient": rows * hidden,
        "output_head_gradient": vocabulary * hidden,
        "raw_adamw_update": vocabulary * hidden,
        "optimizer_exp_avg": vocabulary * hidden,
        "optimizer_exp_avg_sq": vocabulary * hidden,
        "post_step_parameter": vocabulary * hidden,
        "heldout_logits": 17 * vocabulary,
        "heldout_loss": 1,
    }
    assert set(sizes) == set(DIRECT_FAMILIES)
    return {name: metric(elements) for name, elements in sizes.items()}


def direct_case(contract: dict, chunk_size: int, hidden: int, vocabulary: int) -> dict:
    rows = contract["selected_rows"]
    divisor = contract.get("global_divisor", rows + 37)
    boundaries = [[start, min(start + chunk_size, rows)] for start in range(0, rows, chunk_size)]
    maximum = max(end - start for start, end in boundaries)
    audit = {
        "implementation_id": IMPLEMENTATION_ID,
        "selected_rows": rows,
        "chunk_size": chunk_size,
        "chunk_count": len(boundaries),
        "chunk_boundaries": boundaries,
        "maximum_chunk_rows": maximum,
        "vocabulary_size": vocabulary,
        "hidden_size": hidden,
        "maximum_logit_elements": maximum * vocabulary,
        "full_selected_logit_elements": rows * vocabulary,
        "global_target_count": divisor,
        "zero_target": False,
        "returned_dense_logits": False,
    }
    forbidden = [[end - start, vocabulary] for start, end in boundaries]
    return {
        "case_contract": contract,
        "chunk_size": chunk_size,
        "global_divisor": divisor,
        "observed_audit": {**audit, "checkpointed": True},
        "reference_audit": {**audit, "checkpointed": False},
        "execution_proof": {
            "observed_after_forward": len(boundaries),
            "observed_after_backward": 2 * len(boundaries),
            "reference_after_forward": len(boundaries),
            "reference_after_backward": len(boundaries),
        },
        "saved_tensor_proof": {
            "checkpoint_saved_shapes": [[rows, hidden]],
            "ordinary_saved_shapes": [forbidden[0]],
            "forbidden_logit_shapes": forbidden,
            "checkpoint_saved_no_chunk_logits": True,
            "ordinary_saved_at_least_one_chunk_logit": True,
        },
        "autocast_contracts": {**{role: AUTocast for role in ROLES}, "heldout": AUTocast},
        "optimizer_step_counters": {role: [1] for role in ROLES},
        "primary": direct_comparison(rows, hidden, vocabulary, exact),
        "diagnostic_a": direct_comparison(rows, hidden, vocabulary, diagnostic),
        "diagnostic_b": direct_comparison(rows, hidden, vocabulary, diagnostic),
        "status": "passed",
    }


def trajectory_comparison(parameter_count: int, vocabulary: int, sequence_length: int, metric) -> dict:
    return {
        "loss": metric(1),
        "preclip_global_norm": metric(1),
        "aggregate": {family: metric(parameter_count) for family in TRAJECTORY_PARAMETER_FAMILIES},
        "named": {"parameter": {family: metric(parameter_count) for family in TRAJECTORY_PARAMETER_FAMILIES}},
        "heldout_logits": metric(sequence_length * vocabulary),
        "heldout_loss": metric(1),
    }


def trajectory(contract: dict, chunk_size: int, h2: dict) -> dict:
    model = h2["trajectory_model"]
    steps = []
    for index in range(h2["trajectory_steps"]):
        targets = h2["trajectory_target_count_cycle"][index % len(h2["trajectory_target_count_cycle"])]
        chunks = math.ceil(targets / chunk_size)
        comparison_args = (1, model["vocab_size"], model["sequence_length"])
        steps.append(
            {
                "step": index + 1,
                "batch_contract": {
                    "seed": (contract["batch_seed_base"] + index) % (2**32),
                    "target_count": targets,
                    "global_divisor": targets + 37,
                    "input_ids_sha256": "1" * 64,
                    "positions_sha256": "2" * 64,
                    "targets_sha256": "3" * 64,
                },
                "execution_proof": {
                    "observed_after_forward": chunks,
                    "observed_after_backward": 2 * chunks,
                    "reference_after_forward": chunks,
                    "reference_after_backward": chunks,
                },
                "autocast_contracts": {**{role: AUTocast for role in ROLES}, "heldout": AUTocast},
                "optimizer_step_counters": {role: [index + 1] for role in ROLES},
                "primary": trajectory_comparison(*comparison_args, exact),
                "diagnostic_a": trajectory_comparison(*comparison_args, diagnostic),
                "diagnostic_b": trajectory_comparison(*comparison_args, diagnostic),
                "status": "passed",
            }
        )
    return {
        "trajectory_contract": contract,
        "chunk_size": chunk_size,
        "model_definition": {**model, "implementation": "embedding_tanh_linear_tanh_linear_tied_output_r1"},
        "parameter_geometry": [{"name": "parameter", "shape": [1], "elements": 1}],
        "parameter_count": 1,
        "heldout_contract": {
            "seed": contract["heldout_seed"],
            "target_count": 257,
            "global_divisor": 294,
            "sequence_length": model["sequence_length"],
        },
        "steps": steps,
        "status": "passed",
    }


def complete_report() -> tuple[dict, dict, str]:
    qualification, digest = load_qualification_manifest(MANIFEST)
    qualification = copy.deepcopy(qualification)
    qualification["h2_acceptance"]["trajectory_steps"] = 1
    h2 = qualification["h2_acceptance"]
    candidates = []
    for chunk_size in QUALIFIED_CHUNK_SIZES:
        direct = [
            direct_case(contract, chunk_size, h2["direct_hidden_size"], h2["direct_vocab_size"])
            for contract in h2["direct_cases"]
        ]
        real_contract = h2["real_geometry_case"]
        candidates.append(
            {
                "chunk_size": chunk_size,
                "zero_target": {
                    "chunk_size": chunk_size,
                    "loss": exact(1),
                    "loss_value": 0.0,
                    "hidden_gradient": exact(h2["direct_hidden_size"]),
                    "hidden_gradient_nonzero_count": 0,
                    "output_head_gradient": exact(h2["direct_hidden_size"] * h2["direct_vocab_size"]),
                    "output_head_gradient_nonzero_count": 0,
                    "execution_counter": {},
                    "audit": {"selected_rows": 0, "chunk_count": 0, "maximum_logit_elements": 0, "zero_target": True},
                    "autocast_contract": AUTocast,
                    "status": "passed",
                },
                "qwen_forward_integration": {
                    "chunk_size": chunk_size,
                    "model_class": "Qwen3_5ForCausalLM",
                    "attention_implementation": "eager",
                    "forward_module": "open_instruct.qwen35_chunked_loss",
                    "loss": exact(1),
                    "named_parameter_gradients": {"parameter": exact(1)},
                    "returned_logits_is_none": True,
                    "audit": {"chunk_size": chunk_size},
                    "status": "passed",
                },
                "direct_cases": direct,
                "real_geometry_case": direct_case(
                    real_contract, chunk_size, real_contract["hidden_size"], real_contract["vocab_size"]
                ),
                "trajectories": [trajectory(contract, chunk_size, h2) for contract in h2["trajectories"]],
                "status": "passed",
            }
        )
    report = {
        "artifact": "qwen35_checkpointed_chunked_selected_loss_qualification_r18",
        "schema_version": 1,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": digest,
        "manifest_derivation": qualification["manifest_derivation"],
        "source_attestation": {
            "git_commit": "a" * 40,
            "git_worktree_clean": True,
            "implementation_id": IMPLEMENTATION_ID,
            "source_files_sha256": {"producer.py": "b" * 64},
        },
        "environment": {
            "device_type": "cuda",
            "cuda_device": "NVIDIA A100-SXM-64GB",
            "torch_version": qualification["runtime_pins"]["torch_version"],
            "torch_cuda_build": qualification["runtime_pins"]["torch_cuda_build"],
            "liger_imported": False,
        },
        "primary_comparison": {
            "observed_path": h2["primary_observed_path"],
            "reference_path": h2["primary_reference_path"],
            "acceptance": h2["primary_acceptance"],
            "numerical_discrepancy_is_gating": True,
        },
        "diagnostic_comparisons": {
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
        },
        "candidate_results": candidates,
        "status": "passed",
        "successor_gate_authorized": True,
        "scientific_training_authorized": False,
        "allowed_conclusion": "R18 H2 passed; H3 may begin, while scientific training and evaluation remain unauthorized.",
    }
    return report, qualification, digest


def test_exact_metric_distinguishes_signed_zero_bit_patterns() -> None:
    positive = torch.tensor([0.0], dtype=torch.float32)
    negative = torch.tensor([-0.0], dtype=torch.float32)
    metric = exact_tensor_comparison_metrics(positive, negative)
    assert metric["value_equal"] is True
    assert metric["bitwise_equal"] is False
    assert metric["mismatched_values"] == 0
    assert metric["mismatched_bytes"] > 0


def test_exact_metric_supports_scalar_losses() -> None:
    metric = exact_tensor_comparison_metrics(torch.tensor(1.25), torch.tensor(1.25))
    assert metric["shape"] == []
    assert metric["elements"] == 1
    assert metric["bitwise_equal"] is True


def test_diagnostic_metric_matches_direct_float64_definition() -> None:
    observed = torch.tensor([1.0, 2.0, -4.0], dtype=torch.float32)
    reference = torch.tensor([1.5, 1.0, -3.0], dtype=torch.float32)
    metric = diagnostic_tensor_comparison_metrics(observed, reference)
    difference = observed.double() - reference.double()
    assert metric["maximum_absolute_error"] == float(difference.abs().max())
    assert metric["observed_l2_norm"] == float(torch.linalg.vector_norm(observed.double()))
    assert metric["reference_l2_norm"] == float(torch.linalg.vector_norm(reference.double()))
    assert metric["difference_l2_norm"] == float(torch.linalg.vector_norm(difference))


def test_independent_r18_report_validator_accepts_complete_finite_exact_evidence() -> None:
    report, qualification, digest = complete_report()
    validation = validate_h2_chunked_report(report, qualification=qualification, expected_manifest_sha256=digest)
    assert validation["status"] == "passed"
    assert validation["candidate_count"] == 4
    assert validation["trajectory_steps"] == 12
    assert validation["exact_metric_groups"] > 0
    assert validation["diagnostic_metric_groups"] > validation["exact_metric_groups"]


def test_independent_r18_report_validator_accepts_sorted_json_object_round_trip() -> None:
    report, qualification, digest = complete_report()
    reloaded = json.loads(json.dumps(report, allow_nan=False, sort_keys=True))
    validation = validate_h2_chunked_report(reloaded, qualification=qualification, expected_manifest_sha256=digest)
    assert validation["status"] == "passed"


@pytest.mark.parametrize("mutation", ["missing", "extra", "substituted"])
def test_independent_r18_report_validator_rejects_named_parameter_coverage_drift(mutation: str) -> None:
    report, qualification, digest = complete_report()
    named = report["candidate_results"][0]["trajectories"][0]["steps"][0]["primary"]["named"]
    first_name = next(iter(named))
    first_value = copy.deepcopy(named[first_name])
    if mutation == "missing":
        named.pop(first_name)
    elif mutation == "extra":
        named["unexpected.parameter"] = first_value
    else:
        named.pop(first_name)
        named["substituted.parameter"] = first_value
    with pytest.raises(ValueError, match="named trajectory parameter coverage drift"):
        validate_h2_chunked_report(report, qualification=qualification, expected_manifest_sha256=digest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["candidate_results"][0]["direct_cases"][0]["primary"]["loss"].update(
                {"bitwise_equal": False, "mismatched_bytes": 1}
            ),
            "not bit exact",
        ),
        (
            lambda report: report["candidate_results"][1]["real_geometry_case"]["diagnostic_a"][
                "output_head_gradient"
            ].update(
                {
                    "nonfinite_count": 1,
                    "maximum_absolute_error": None,
                    "relative_l2_error": None,
                    "cosine_similarity": None,
                    "observed_l2_norm": None,
                    "reference_l2_norm": None,
                    "difference_l2_norm": None,
                }
            ),
            "nonfinite diagnostic",
        ),
        (
            lambda report: report["candidate_results"][2]["direct_cases"][3]["execution_proof"].update(
                {"observed_after_backward": 1}
            ),
            "recomputation count drift",
        ),
        (lambda report: report["candidate_results"][3]["zero_target"].update({"loss_value": 1.0}), "not exact zero"),
        (
            lambda report: report["candidate_results"][0]["trajectories"][0]["steps"][0]["batch_contract"].update(
                {"target_count": 2}
            ),
            "batch accounting/hash schema drift",
        ),
    ],
)
def test_independent_r18_report_validator_rejects_single_field_corruption(mutation, message) -> None:
    report, qualification, digest = complete_report()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        validate_h2_chunked_report(report, qualification=qualification, expected_manifest_sha256=digest)


def test_fabricated_report_fixture_is_strict_json_serializable() -> None:
    report, _, _ = complete_report()
    serialized = json.dumps(report, allow_nan=False)
    assert "NaN" not in serialized and "Infinity" not in serialized
