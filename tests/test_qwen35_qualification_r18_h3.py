from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from scripts.train.qwen35.validate_qwen35_h3_reports_r18 import (
    H2_PREDECESSOR_SHA256,
    VALIDATOR_AMENDMENT_HUMAN_SHA256,
    VALIDATOR_AMENDMENT_PREREGISTRATION_SHA256,
    VALIDATOR_AMENDMENT_SHA256,
    VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_SHA256,
    VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION_SHA256,
    VALIDATOR_CONSISTENCY_AMENDMENT_SHA256,
    VALIDATOR_FAILURE_FORENSIC_CLOSURE_SHA256,
    VALIDATOR_V2_FAILURE_CLOSURE_SHA256,
    validate_h2_predecessor,
    validate_h2_predecessor_value,
    validate_validator_amendment,
    validate_validator_consistency_amendment,
)

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID
from open_instruct.qwen35_qualification import scalar_comparison_metrics, sha256_file
from open_instruct.qwen35_qualification_r18_h3 import (
    COMPARISON_FAMILIES,
    COMPARISON_PATHS,
    H3_ARTIFACT,
    H3_ATTEMPT01_FAILURE_CLOSURE_SHA256,
    H3_HARNESS_AMENDMENT_HUMAN_SHA256,
    H3_HARNESS_AMENDMENT_PREREGISTRATION_SHA256,
    H3_HARNESS_AMENDMENT_SHA256,
    H3_PROTOCOL_ID,
    PATHS,
    STORED_FAMILIES,
    _recompute_comparison,
    expected_case_records,
    load_h3_harness_amendment,
    load_h3_manifest,
    norm_summary_cross_backend_relative_bound,
    prepare_distributed_output_directory,
    tensor_key,
    validate_h3_report,
    validate_norm_summary_cross_backend_consistency,
)

ROOT = Path(__file__).parents[1]
H3_MANIFEST = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3.json"
H3_HARNESS_AMENDMENT = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3_harness_amendment_r2.json"
R18_MANIFEST = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18.json"
HUMAN_PROTOCOL = Path(
    os.environ.get(
        "QWEN35_H3_HUMAN_PROTOCOL",
        ROOT.parents[1] / "methodology/qwen35_hardware_qualification_r18_h3_protocol_r1_20260719.md",
    )
)
H3_HARNESS_AMENDMENT_HUMAN = Path(
    os.environ.get(
        "QWEN35_H3_HARNESS_AMENDMENT_HUMAN_PROTOCOL",
        ROOT.parents[1] / "methodology/qwen35_hardware_qualification_r18_h3_harness_amendment_r2_20260719.md",
    )
)
H3_HARNESS_AMENDMENT_PREREGISTRATION = Path(
    os.environ.get(
        "QWEN35_H3_HARNESS_AMENDMENT_PREREG_CLOSURE",
        ROOT.parents[1] / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h3_harness_amendment_r2_preregistration_closure.json",
    )
)
H3_ATTEMPT01_FAILURE_CLOSURE = Path(
    os.environ.get(
        "QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE",
        ROOT.parents[1] / "artifacts/qwen35_hardware_qualification_20260718/r18_h3_gpu_attempt01_failure_closure.json",
    )
)
H3_VALIDATOR_AMENDMENT = ROOT / "scripts/train/qwen35/qwen35_h3_validator_amendment_r18_v1.json"
H3_VALIDATOR_AMENDMENT_HUMAN = Path(
    os.environ.get(
        "QWEN35_H3_VALIDATOR_AMENDMENT_HUMAN_PROTOCOL",
        ROOT.parents[1] / "methodology/qwen35_hardware_qualification_r18_h3_validator_amendment_v1_20260719.md",
    )
)
H3_VALIDATOR_AMENDMENT_PREREGISTRATION = Path(
    os.environ.get(
        "QWEN35_H3_VALIDATOR_AMENDMENT_PREREG_CLOSURE",
        ROOT.parents[1]
        / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h3_validator_amendment_v1_preregistration_closure.json",
    )
)
H3_VALIDATOR_CONSISTENCY_AMENDMENT = (
    ROOT / "scripts/train/qwen35/qwen35_h3_validator_consistency_amendment_r18_v3.json"
)
H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN = Path(
    os.environ.get(
        "QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_PROTOCOL",
        ROOT.parents[1]
        / "methodology/qwen35_hardware_qualification_r18_h3_validator_consistency_amendment_v3_20260719.md",
    )
)
H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION = Path(
    os.environ.get(
        "QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREG_CLOSURE",
        ROOT.parents[1]
        / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h3_validator_consistency_amendment_v3_preregistration_closure.json",
    )
)
H3_VALIDATOR_V2_FAILURE_CLOSURE = Path(
    os.environ.get(
        "QWEN35_H3_VALIDATOR_V2_FAILURE_CLOSURE",
        ROOT.parents[1]
        / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h3v2_validator_attempt02_failure_closure_20260719.json",
    )
)
H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE = Path(
    os.environ.get(
        "QWEN35_H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE",
        ROOT.parents[1]
        / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h3_independent_failure_forensic_closure_20260719.json",
    )
)
H2_INDEPENDENT_VALIDATION = Path(
    os.environ.get(
        "QWEN35_H2_INDEPENDENT_VALIDATION",
        ROOT.parents[1]
        / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18v1_00fd47e_validation/h2_independent_validation_amended_v1.json",
    )
)


def _loaded() -> tuple[dict, str]:
    manifest, digest, _ = load_h3_manifest(
        H3_MANIFEST, r18_manifest_path=R18_MANIFEST, human_protocol_path=HUMAN_PROTOCOL
    )
    return manifest, digest


def _loaded_amendment() -> tuple[dict, str]:
    return load_h3_harness_amendment(
        H3_HARNESS_AMENDMENT,
        human_amendment_path=H3_HARNESS_AMENDMENT_HUMAN,
        attempt01_failure_closure_path=H3_ATTEMPT01_FAILURE_CLOSURE,
        preregistration_closure_path=H3_HARNESS_AMENDMENT_PREREGISTRATION,
        h3_manifest_path=H3_MANIFEST,
    )


class _SequentialBroadcast:
    def __init__(self) -> None:
        self.payload = None
        self.sources = []

    def __call__(self, values, *, src: int) -> None:
        self.sources.append(src)
        if values[0] is not None:
            self.payload = copy.deepcopy(values[0])
        else:
            values[0] = copy.deepcopy(self.payload)


def _gloo_output_directory_worker(
    rank: int, world_size: int, rendezvous_path: str, output_dir: str, results_dir: str
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        try:
            record = prepare_distributed_output_directory(
                Path(output_dir), rank=rank, broadcast_object_list=torch.distributed.broadcast_object_list
            )
            result = {"record": record, "status": "passed"}
        except Exception as error:
            result = {"exception_message": str(error), "exception_type": type(error).__name__, "status": "failed"}
        (Path(results_dir) / f"rank_{rank:02d}.json").write_text(json.dumps(result, sort_keys=True))
    finally:
        torch.distributed.destroy_process_group()


def _audit(count: int, *, chunk_size: int, divisor: int) -> dict:
    boundaries = [[start, min(start + chunk_size, count)] for start in range(0, count, chunk_size)]
    maximum = min(count, chunk_size) if count else 0
    return {
        "checkpointed": True,
        "chunk_boundaries": boundaries,
        "chunk_count": len(boundaries),
        "chunk_size": chunk_size,
        "full_selected_logit_elements": count * 256,
        "global_target_count": divisor if count else 0,
        "hidden_size": 64,
        "implementation_id": IMPLEMENTATION_ID,
        "maximum_chunk_rows": maximum,
        "maximum_logit_elements": maximum * 256,
        "returned_dense_logits": False,
        "selected_rows": count,
        "vocabulary_size": 256,
        "zero_target": count == 0,
    }


def _case_with_hashes(record: dict, model: dict) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(record["seed"])
    input_ids = torch.randint(1, model["vocab_size"] - 1, (1, model["sequence_length"]), generator=generator)
    labels = torch.full_like(input_ids, -100)
    count = record["target_count"]
    if count:
        labels[:, 1 : count + 1] = input_ids[:, 1 : count + 1]
        positions = torch.arange(count, dtype=torch.long)
        shifted = labels[:, 1 : count + 1].reshape(-1).contiguous()
    else:
        positions = torch.tensor([0], dtype=torch.long)
        shifted = torch.tensor([-100], dtype=torch.long)
    return {
        **record,
        "input_ids_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "logits_to_keep_sha256": hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
        "shift_labels_sha256": hashlib.sha256(shifted.numpy().tobytes()).hexdigest(),
    }


def _write_evidence(path: Path, *, mutation=None) -> dict[str, torch.Tensor]:
    base = {
        "initial_parameter": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "preclip_gradient": torch.tensor([2.0, 0.0], dtype=torch.float32),
        "clipped_gradient": torch.tensor([0.999999, 0.0], dtype=torch.float32),
        "post_step_parameter": torch.tensor([0.999, 1.999], dtype=torch.float32),
        "optimizer_exp_avg": torch.tensor([0.2, 0.0], dtype=torch.float32),
        "optimizer_exp_avg_sq": torch.tensor([0.004, 0.0], dtype=torch.float32),
    }
    tensors = {
        tensor_key(path_name, family, "weight"): value.clone() for path_name in PATHS for family, value in base.items()
    }
    if mutation is not None:
        mutation(tensors)
    save_file(tensors, str(path), metadata={"protocol_id": H3_PROTOCOL_ID})
    return tensors


def _valid_report(tmp_path: Path) -> tuple[dict, Path, dict, str]:
    manifest, digest = _loaded()
    scenario = manifest["scenarios"][0]
    records = expected_case_records(manifest, "P4x2")
    case_records = [_case_with_hashes(record, manifest["model"]) for record in records]
    evidence_path = tmp_path / "h3_evidence.safetensors"
    tensors = _write_evidence(evidence_path)
    numerical = manifest["numerical_acceptance"]
    comparisons = {
        path: {
            family: _recompute_comparison(
                tensors, observed_path=path, family=family, parameter_names=["weight"], numerical=numerical
            )
            for family in COMPARISON_FAMILIES
        }
        for path in COMPARISON_PATHS
    }
    per_case = [0.0 if record["target_count"] == 0 else 0.5 for record in records]
    global_loss = sum(per_case)
    report = {
        "allowed_conclusion": (
            "This scenario/candidate passed H3; only completion and independent validation of the full eight-run H3 set may authorize H4."
        ),
        "artifact": H3_ARTIFACT,
        "audits": {
            path: [
                {
                    "audit": _audit(record["target_count"], chunk_size=128, divisor=scenario["global_target_count"]),
                    "case_id": record["case_id"],
                }
                for record in records
            ]
            for path in PATHS
        },
        "case_records": case_records,
        "chunk_size": 128,
        "clipping": {
            path: {
                "clip_grad_norm_return": 2.0,
                "postclip_norm_from_evidence": float(
                    torch.linalg.vector_norm(tensors[tensor_key(path, "clipped_gradient", "weight")].double())
                ),
                "preclip_norm_from_evidence": float(
                    torch.linalg.vector_norm(tensors[tensor_key(path, "preclip_gradient", "weight")].double())
                ),
            }
            for path in PATHS
        },
        "comparisons": comparisons,
        "contract": {
            "accumulation_steps": scenario["accumulation_steps"],
            "global_target_count": scenario["global_target_count"],
            "per_rank_target_counts": scenario["per_rank_target_counts"],
            "target_counts_by_slot_rank": scenario["target_counts_by_slot_rank"],
            "world_size": 4,
        },
        "decision": {
            "all_gating_comparisons_passed": True,
            "allowed_successor": "H4_only",
            "scientific_training_authorized": False,
        },
        "environment": {
            "autocast": {"device_type": "cuda", "dtype": "torch.bfloat16", "enabled": True},
            "backend": "nccl",
            "cuda_version": "12.9",
            "device_names": ["NVIDIA A100-SXM-64GB"] * 4,
            "liger_modules_by_rank": [[], [], [], []],
            "torch_version": "2.9.1+cu129",
            "world_size": 4,
        },
        "h3_manifest_sha256": digest,
        "harness_amendment": {
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
        },
        "human_protocol_sha256": manifest["human_protocol"]["sha256"],
        "losses": {
            "comparisons": {path: scalar_comparison_metrics(global_loss, global_loss) for path in COMPARISON_PATHS},
            "global_unscaled": {path: global_loss for path in PATHS},
            "per_case_unscaled": {path: list(per_case) for path in PATHS},
        },
        "optimizer": {
            "ddp_rank_post_step_state_sha256": ["a" * 64] * 4,
            "floating_state_dtypes": ["torch.float32"],
            "foreach": False,
            "fused": True,
            "gradient_dtypes": ["torch.float32"],
            "parameter_dtypes": ["torch.float32"],
            "step_counters": {path: [1] for path in PATHS},
        },
        "protocol_id": H3_PROTOCOL_ID,
        "r18_manifest_sha256": sha256_file(R18_MANIFEST),
        "scaling": {
            "ddp_case_losses": [
                {
                    "case_id": record["case_id"],
                    "global_target_count": scenario["global_target_count"],
                    "rank": record["rank"],
                    "scaled_backward_loss": per_case[record["case_id"]] * 4,
                    "slot": record["slot"],
                    "synchronized_backward": record["slot"] == scenario["accumulation_steps"] - 1,
                    "target_count": record["target_count"],
                    "unscaled_model_loss": per_case[record["case_id"]],
                }
                for record in records
            ],
            "global_target_count": scenario["global_target_count"],
            "world_size_multiplier": 4,
        },
        "scenario_id": "P4x2",
        "schema_version": 1,
        "source_attestation": {
            "git_commit": "f" * 40,
            "git_worktree_clean": True,
            "implementation_id": IMPLEMENTATION_ID,
            "liger_modules_imported": [],
            "source_files_sha256": {"producer.py": "e" * 64},
        },
        "status": "passed",
        "tensor_evidence": {
            "bytes": evidence_path.stat().st_size,
            "families": list(STORED_FAMILIES),
            "file_name": evidence_path.name,
            "format": "safetensors",
            "key_count": len(tensors),
            "parameter_geometry": [{"dtype": "torch.float32", "elements": 2, "name": "weight", "shape": [2]}],
            "paths": list(PATHS),
            "sha256": sha256_file(evidence_path),
        },
    }
    return report, evidence_path, manifest, digest


def _validate(report: dict, evidence: Path, manifest: dict, digest: str) -> dict:
    return validate_h3_report(
        report,
        evidence_path=evidence,
        h3_manifest=manifest,
        h3_manifest_sha256=digest,
        r18_manifest_sha256=sha256_file(R18_MANIFEST),
    )


def test_h3_manifest_is_frozen_and_complete() -> None:
    manifest, digest = _loaded()
    assert digest == "95aec699d2bab81c5eb3094d2048f997f137faa624dbb7128f92b32134b8abf4"
    assert [len(expected_case_records(manifest, scenario)) for scenario in ("P4x2", "B4x4")] == [8, 16]
    boundary = expected_case_records(manifest, "B4x4")
    counts = {record["target_count"] for record in boundary}
    for chunk_size in (128, 256, 512, 1024):
        assert {chunk_size - 1, chunk_size, chunk_size + 1} <= counts


def test_h3_manifest_rejects_loosened_threshold(tmp_path: Path) -> None:
    value = json.loads(H3_MANIFEST.read_text())
    value["numerical_acceptance"]["gradient_relative_l2_error"] = 1.0
    path = tmp_path / H3_MANIFEST.name
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="threshold drift"):
        load_h3_manifest(path, r18_manifest_path=R18_MANIFEST, human_protocol_path=HUMAN_PROTOCOL)


def test_h3_harness_amendment_is_exactly_bound() -> None:
    amendment, digest = _loaded_amendment()
    assert digest == H3_HARNESS_AMENDMENT_SHA256
    assert amendment["trigger"]["job_id"] == "49852000"
    assert amendment["trigger"]["numerical_reports_produced"] == 0
    assert amendment["retry"]["maximum_corrected_attempts"] == 1
    assert amendment["scientific_training_authorized"] is False


def test_h3_harness_amendment_rejects_machine_manifest_drift(tmp_path: Path) -> None:
    value = json.loads(H3_HARNESS_AMENDMENT.read_text())
    value["retry"]["maximum_corrected_attempts"] = 2
    changed = tmp_path / H3_HARNESS_AMENDMENT.name
    changed.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(ValueError, match="harness-amendment hash drift"):
        load_h3_harness_amendment(
            changed,
            human_amendment_path=H3_HARNESS_AMENDMENT_HUMAN,
            attempt01_failure_closure_path=H3_ATTEMPT01_FAILURE_CLOSURE,
            preregistration_closure_path=H3_HARNESS_AMENDMENT_PREREGISTRATION,
            h3_manifest_path=H3_MANIFEST,
        )


def test_h3_independent_validator_amendment_is_exactly_bound() -> None:
    amendment, digest = validate_validator_amendment(
        H3_VALIDATOR_AMENDMENT,
        human_path=H3_VALIDATOR_AMENDMENT_HUMAN,
        preregistration_path=H3_VALIDATOR_AMENDMENT_PREREGISTRATION,
    )
    assert digest == VALIDATOR_AMENDMENT_SHA256
    assert sha256_file(H3_VALIDATOR_AMENDMENT_HUMAN) == VALIDATOR_AMENDMENT_HUMAN_SHA256
    assert (
        sha256_file(H3_VALIDATOR_AMENDMENT_PREREGISTRATION)
        == VALIDATOR_AMENDMENT_PREREGISTRATION_SHA256
    )
    assert amendment["trigger"]["h3_tensor_evidence_opened_before_failure"] is False
    assert amendment["scientific_training_authorized"] is False


def test_h3_validator_consistency_amendment_is_exactly_bound() -> None:
    amendment, digest = validate_validator_consistency_amendment(
        H3_VALIDATOR_CONSISTENCY_AMENDMENT,
        human_path=H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN,
        preregistration_path=H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION,
        validator_failure_path=H3_VALIDATOR_V2_FAILURE_CLOSURE,
        forensic_closure_path=H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE,
    )
    assert digest == VALIDATOR_CONSISTENCY_AMENDMENT_SHA256
    assert sha256_file(H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN) == (
        VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_SHA256
    )
    assert sha256_file(H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION) == (
        VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION_SHA256
    )
    assert sha256_file(H3_VALIDATOR_V2_FAILURE_CLOSURE) == VALIDATOR_V2_FAILURE_CLOSURE_SHA256
    assert sha256_file(H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE) == (
        VALIDATOR_FAILURE_FORENSIC_CLOSURE_SHA256
    )
    assert amendment["bound"]["element_count"] == 103672
    assert amendment["required_summary_comparisons"] == 48
    assert amendment["scientific_training_authorized"] is False


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("machine", "machine-manifest hash drift"),
        ("human", "human file hash drift"),
        ("preregistration", "preregistration-closure hash drift"),
        ("failure", "V2 failure-closure hash drift"),
        ("forensic", "failure-forensic closure hash drift"),
    ],
)
def test_h3_validator_consistency_amendment_rejects_bound_file_drift(
    tmp_path: Path, target: str, message: str
) -> None:
    paths = {
        "machine": H3_VALIDATOR_CONSISTENCY_AMENDMENT,
        "human": H3_VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN,
        "preregistration": H3_VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION,
        "failure": H3_VALIDATOR_V2_FAILURE_CLOSURE,
        "forensic": H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE,
    }
    replacements = dict(paths)
    changed = tmp_path / paths[target].name
    changed.write_bytes(paths[target].read_bytes() + b"\n")
    replacements[target] = changed
    with pytest.raises(ValueError, match=message):
        validate_validator_consistency_amendment(
            replacements["machine"],
            human_path=replacements["human"],
            preregistration_path=replacements["preregistration"],
            validator_failure_path=replacements["failure"],
            forensic_closure_path=replacements["forensic"],
        )


def test_h3_norm_summary_bound_is_high_precision_and_upward_rounded() -> None:
    bound = norm_summary_cross_backend_relative_bound(103672)
    assert bound == 1.1510126185730685e-11
    assert Decimal.from_float(bound) > Decimal(
        "1.1510126185730684230549368235693935884182021079188570533453842551364243845533590e-11"
    )


def test_h3_norm_summary_consistency_accepts_equal_and_strictly_in_bound_values() -> None:
    bound = norm_summary_cross_backend_relative_bound(103672)
    equal = validate_norm_summary_cross_backend_consistency(
        1.0, 1.0, element_count=103672, context="equal"
    )
    inside = validate_norm_summary_cross_backend_consistency(
        1.0 + bound / 2, 1.0, element_count=103672, context="inside"
    )
    assert equal["relative_difference"] == 0.0
    assert inside["relative_difference"] < inside["relative_bound"]


def test_h3_norm_summary_consistency_rejects_value_above_derived_bound() -> None:
    bound = norm_summary_cross_backend_relative_bound(103672)
    with pytest.raises(ValueError, match="bound exceeded"):
        validate_norm_summary_cross_backend_consistency(
            1.0 + 2 * bound, 1.0, element_count=103672, context="outside"
        )


def test_h3_norm_summary_consistency_rejects_zero_versus_nonzero() -> None:
    with pytest.raises(ValueError, match="bound exceeded"):
        validate_norm_summary_cross_backend_consistency(
            0.0, math.nextafter(0.0, math.inf), element_count=103672, context="zero-versus-nonzero"
        )


@pytest.mark.parametrize("value", [True, -1.0, float("nan"), float("inf")])
def test_h3_norm_summary_consistency_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="numeric|negative|nonfinite"):
        validate_norm_summary_cross_backend_consistency(
            value, 1.0, element_count=103672, context="invalid"
        )


@pytest.mark.parametrize("element_count", [True, 0, -1, 1.5])
def test_h3_norm_summary_bound_rejects_invalid_element_count(element_count) -> None:
    with pytest.raises(ValueError, match="positive plain integer"):
        norm_summary_cross_backend_relative_bound(element_count)


def test_h3_validator_accepts_exact_authoritative_h2_predecessor() -> None:
    value = validate_h2_predecessor(H2_INDEPENDENT_VALIDATION)
    assert sha256_file(H2_INDEPENDENT_VALIDATION) == H2_PREDECESSOR_SHA256
    assert value["successor_gate_authorized"] == "H3_only"
    assert "allowed_successor" not in value


def test_h3_validator_rejects_h2_predecessor_hash_drift(tmp_path: Path) -> None:
    changed = tmp_path / H2_INDEPENDENT_VALIDATION.name
    changed.write_bytes(H2_INDEPENDENT_VALIDATION.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash drift"):
        validate_h2_predecessor(changed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(status="failed"), "status does not authorize"),
        (lambda value: value.pop("successor_gate_authorized"), "successor authorization"),
        (lambda value: value.update(successor_gate_authorized="H4_only"), "successor authorization"),
        (lambda value: value.update(allowed_successor="H3_only"), "stale allowed_successor"),
        (lambda value: value.update(scientific_training_authorized=True), "scientific-training authority"),
        (lambda value: value.update(qualification_manifest_sha256="0" * 64), "manifest binding"),
        (lambda value: value["validation"].update(status="failed"), "nested validation did not pass"),
        (
            lambda value: value["validation"].update(successor_gate_authorized=False),
            "nested successor authorization",
        ),
        (
            lambda value: value["validation"].update(scientific_training_authorized=True),
            "nested scientific-training authority",
        ),
        (lambda value: value["validation"].update(candidate_count=3), "candidate count"),
        (lambda value: value["validation"].update(trajectory_steps=3071), "trajectory-step count"),
        (lambda value: value.update(schema_version=True), "schema version"),
    ],
)
def test_h3_validator_rejects_adversarial_h2_predecessor_fields(mutate, message: str) -> None:
    value = json.loads(H2_INDEPENDENT_VALIDATION.read_text())
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_h2_predecessor_value(value)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("machine", "machine-manifest hash drift"),
        ("human", "human file hash drift"),
        ("preregistration", "preregistration-closure hash drift"),
    ],
)
def test_h3_validator_amendment_rejects_bound_file_drift(tmp_path: Path, target: str, message: str) -> None:
    paths = {
        "machine": H3_VALIDATOR_AMENDMENT,
        "human": H3_VALIDATOR_AMENDMENT_HUMAN,
        "preregistration": H3_VALIDATOR_AMENDMENT_PREREGISTRATION,
    }
    changed = tmp_path / paths[target].name
    changed.write_bytes(paths[target].read_bytes() + b"\n")
    paths[target] = changed
    with pytest.raises(ValueError, match=message):
        validate_validator_amendment(
            paths["machine"], human_path=paths["human"], preregistration_path=paths["preregistration"]
        )


@pytest.mark.parametrize(
    ("path_name", "message"),
    [
        ("human", "human file hash drift"),
        ("failure", "failure-closure hash drift"),
        ("preregistration", "preregistration-closure hash drift"),
    ],
)
def test_h3_harness_amendment_rejects_bound_file_drift(tmp_path: Path, path_name: str, message: str) -> None:
    paths = {
        "human": H3_HARNESS_AMENDMENT_HUMAN,
        "failure": H3_ATTEMPT01_FAILURE_CLOSURE,
        "preregistration": H3_HARNESS_AMENDMENT_PREREGISTRATION,
    }
    replacements = dict(paths)
    changed = tmp_path / paths[path_name].name
    changed.write_bytes(paths[path_name].read_bytes() + b"\n")
    replacements[path_name] = changed
    with pytest.raises(ValueError, match=message):
        load_h3_harness_amendment(
            H3_HARNESS_AMENDMENT,
            human_amendment_path=replacements["human"],
            attempt01_failure_closure_path=replacements["failure"],
            preregistration_closure_path=replacements["preregistration"],
            h3_manifest_path=H3_MANIFEST,
        )


def test_h3_output_directory_rank_zero_is_sole_owner(tmp_path: Path) -> None:
    output = tmp_path / "shared"
    broadcast = _SequentialBroadcast()
    rank0 = prepare_distributed_output_directory(output, rank=0, broadcast_object_list=broadcast)

    def forbidden_nonowner_create(_: Path) -> None:
        raise AssertionError("non-owner attempted directory creation")

    rank1 = prepare_distributed_output_directory(
        output, rank=1, broadcast_object_list=broadcast, create_directory=forbidden_nonowner_create
    )
    assert rank0 == rank1 == {"creator_rank": 0, "output_dir": str(output), "status": "created"}
    assert output.is_dir()
    assert broadcast.sources == [0, 0]


def test_h3_output_directory_preexisting_path_fails_every_rank(tmp_path: Path) -> None:
    output = tmp_path / "shared"
    output.mkdir()
    broadcast = _SequentialBroadcast()
    for rank in (0, 1):
        with pytest.raises(FileExistsError, match="rank-0 H3 output-directory creation failed"):
            prepare_distributed_output_directory(output, rank=rank, broadcast_object_list=broadcast)


def test_h3_output_directory_injected_creation_error_fails_every_rank(tmp_path: Path) -> None:
    output = tmp_path / "shared"
    broadcast = _SequentialBroadcast()

    def injected(_: Path) -> None:
        raise PermissionError("injected rank-zero failure")

    with pytest.raises(RuntimeError, match="PermissionError: injected rank-zero failure"):
        prepare_distributed_output_directory(
            output, rank=0, broadcast_object_list=broadcast, create_directory=injected
        )
    with pytest.raises(RuntimeError, match="PermissionError: injected rank-zero failure"):
        prepare_distributed_output_directory(output, rank=3, broadcast_object_list=broadcast)


@pytest.mark.parametrize("preexisting", [False, True])
def test_h3_output_directory_four_process_gloo_integration(tmp_path: Path, preexisting: bool) -> None:
    assert torch.distributed.is_gloo_available()
    output = tmp_path / "shared"
    if preexisting:
        output.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    torch.multiprocessing.spawn(
        _gloo_output_directory_worker,
        args=(4, str(tmp_path / "rendezvous"), str(output), str(results)),
        nprocs=4,
        join=True,
    )
    observed = [json.loads((results / f"rank_{rank:02d}.json").read_text()) for rank in range(4)]
    if preexisting:
        assert all(item["status"] == "failed" and item["exception_type"] == "FileExistsError" for item in observed)
        assert len({item["exception_message"] for item in observed}) == 1
    else:
        assert all(item["status"] == "passed" for item in observed)
        assert len({json.dumps(item["record"], sort_keys=True) for item in observed}) == 1
        assert output.is_dir()


def test_h3_cuda_device_is_explicitly_bound_before_nccl_initialization() -> None:
    producer = (ROOT / "scripts/train/qwen35/validate_qwen35_ddp_ga_r18_h3.py").read_text()
    local_rank = producer.index('local_rank = int(os.environ["LOCAL_RANK"])')
    set_device = producer.index("torch.cuda.set_device(local_rank)")
    init_process_group = producer.index('torch.distributed.init_process_group("nccl", device_id=device)')
    first_collective = producer.index("torch.distributed.get_rank()")
    assert local_rank < set_device < init_process_group < first_collective


def test_h3_independent_validator_accepts_complete_tensor_evidence(tmp_path: Path) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)
    result = _validate(report, evidence, manifest, digest)
    assert result["status"] == "passed"
    assert result["active_clipping_paths"] == 3
    assert result["cases"] == 8
    assert result["norm_summary_consistency"]["comparisons"] == 6
    assert result["norm_summary_consistency"]["element_count"] == 2
    assert result["norm_summary_consistency"]["maximum_relative_difference"] == 0.0
    assert len(result["norm_summary_consistency"]["records"]) == 6


def test_h3_validator_accepts_cross_backend_norm_summary_difference_within_derived_bound(tmp_path: Path) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)
    bound = norm_summary_cross_backend_relative_bound(2)
    original = report["clipping"]["central_graph_sum"]["postclip_norm_from_evidence"]
    inside = math.nextafter(original, math.inf)
    relative_difference = abs(inside - original) / max(abs(inside), abs(original))
    assert 0 < relative_difference <= bound
    report["clipping"]["central_graph_sum"]["postclip_norm_from_evidence"] = inside
    result = _validate(report, evidence, manifest, digest)
    assert result["status"] == "passed"
    assert result["norm_summary_consistency"]["maximum_relative_difference"] > 0


def test_h3_validator_rejects_cross_backend_norm_summary_difference_above_derived_bound(tmp_path: Path) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)
    bound = norm_summary_cross_backend_relative_bound(2)
    original = report["clipping"]["central_graph_sum"]["postclip_norm_from_evidence"]
    outside = original * (1 + 1e-12)
    relative_difference = abs(outside - original) / max(abs(outside), abs(original))
    assert relative_difference > bound
    report["clipping"]["central_graph_sum"]["postclip_norm_from_evidence"] = outside
    with pytest.raises(ValueError, match="bound exceeded"):
        _validate(report, evidence, manifest, digest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report["contract"].update(global_target_count=1), "distributed contract drift"),
        (lambda report: report["case_records"].pop(), "case record count drift"),
        (lambda report: report["case_records"].__setitem__(1, copy.deepcopy(report["case_records"][0])), "case 1"),
        (
            lambda report: report["audits"]["central_graph_sum"][1]["audit"].update(
                chunk_boundaries=[[0, 126], [126, 127]]
            ),
            "chunk boundary drift",
        ),
        (
            lambda report: report["audits"]["ddp_gradient_accumulation"][1]["audit"].update(global_target_count=127),
            "global divisor drift",
        ),
        (
            lambda report: report["audits"]["central_graph_sum"][0]["audit"].update(selected_rows=1),
            "selected-row or zero-target drift",
        ),
        (
            lambda report: report["audits"]["central_graph_sum"][1]["audit"].update(returned_dense_logits=True),
            "dense logits",
        ),
        (lambda report: report.update(chunk_size=64), "candidate drift"),
        (
            lambda report: report["harness_amendment"].update(machine_manifest_sha256="0" * 64),
            "harness-amendment report binding drift",
        ),
        (lambda report: report["optimizer"].update(fused=False), "optimizer path"),
        (lambda report: report.update(status="failed"), "not a numerical pass"),
        (lambda report: report["source_attestation"].update(liger_modules_imported=["liger_kernel"]), "Liger"),
        (lambda report: report["environment"].update(liger_modules_by_rank=[["liger_kernel"], [], [], []]), "Liger"),
        (
            lambda report: report["scaling"]["ddp_case_losses"][1].update(scaled_backward_loss=0.5),
            "world-size scaling",
        ),
        (
            lambda report: report["scaling"]["ddp_case_losses"][1].update(scaled_backward_loss=4.0),
            "world-size scaling",
        ),
    ],
)
def test_h3_validator_rejects_adversarial_json_mutations(tmp_path: Path, mutate, message: str) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)
    mutate(report)
    with pytest.raises(ValueError, match=message):
        _validate(report, evidence, manifest, digest)


def _replace_evidence(report: dict, evidence: Path, mutation) -> None:
    tensors = dict(load_file(evidence))
    mutation(tensors)
    save_file(tensors, str(evidence))
    report["tensor_evidence"]["bytes"] = evidence.stat().st_size
    report["tensor_evidence"]["sha256"] = sha256_file(evidence)
    report["tensor_evidence"]["key_count"] = len(tensors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda tensors: tensors.__setitem__(
                tensor_key("central_graph_sum", "preclip_gradient", "weight"),
                torch.tensor([float("nan"), 0.0], dtype=torch.float32),
            ),
            "nonfinite tensor",
        ),
        (lambda tensors: tensors.pop(tensor_key("central_graph_sum", "preclip_gradient", "weight")), "key-set"),
        (
            lambda tensors: tensors.__setitem__(
                tensor_key("central_graph_sum", "preclip_gradient", "weight"), torch.ones(3, dtype=torch.float32)
            ),
            "shape/dtype drift",
        ),
        (
            lambda tensors: tensors.__setitem__(
                tensor_key("central_graph_sum", "preclip_gradient", "weight"), torch.ones(2, dtype=torch.float16)
            ),
            "shape/dtype drift",
        ),
    ],
)
def test_h3_validator_rejects_adversarial_tensor_mutations(tmp_path: Path, mutation, message: str) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)
    _replace_evidence(report, evidence, mutation)
    with pytest.raises((ValueError, AssertionError), match=message):
        _validate(report, evidence, manifest, digest)


def test_h3_validator_rejects_inactive_clipping_even_with_consistent_summaries(tmp_path: Path) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)

    def mutation(tensors):
        for path in PATHS:
            tensors[tensor_key(path, "preclip_gradient", "weight")] = torch.tensor([0.5, 0.0])
            tensors[tensor_key(path, "clipped_gradient", "weight")] = torch.tensor([0.499, 0.0])

    _replace_evidence(report, evidence, mutation)
    tensors = dict(load_file(evidence))
    report["comparisons"] = {
        path: {
            family: _recompute_comparison(
                tensors,
                observed_path=path,
                family=family,
                parameter_names=["weight"],
                numerical=manifest["numerical_acceptance"],
            )
            for family in COMPARISON_FAMILIES
        }
        for path in COMPARISON_PATHS
    }
    for path in PATHS:
        report["clipping"][path]["preclip_norm_from_evidence"] = 0.5
        report["clipping"][path]["postclip_norm_from_evidence"] = 0.499
    with pytest.raises(ValueError, match="active clipping"):
        _validate(report, evidence, manifest, digest)


def test_h3_validator_rejects_evidence_hash_drift(tmp_path: Path) -> None:
    report, evidence, manifest, digest = _valid_report(tmp_path)
    report["tensor_evidence"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="size/hash drift"):
        _validate(report, evidence, manifest, digest)


def test_h3_wrappers_are_personal_bounded_non_liger_and_bytecode_isolated() -> None:
    producer = (ROOT / "scripts/train/qwen35/leonardo_h3_r18.sbatch").read_text()
    validator = (ROOT / "scripts/train/qwen35/leonardo_validate_h3_r18.sbatch").read_text()
    guard = (ROOT / "scripts/train/qwen35/g2_job_guard.sh").read_text()
    assert "#SBATCH --account=aifac_f02_434" in producer
    assert "#SBATCH --gres=gpu:4" in producer
    assert "for scenario in P4x2 B4x4" in producer
    assert "for chunk_size in 128 256 512 1024" in producer
    assert "QWEN35_H3_HARNESS_AMENDMENT" in producer
    assert "QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE" in producer
    assert "--harness-amendment-preregistration-closure" in producer
    assert "leonardo_h4" not in producer.lower()
    assert "sbatch" not in "\n".join(line for line in producer.splitlines() if not line.startswith("#SBATCH"))
    assert "liger_kernel" not in producer
    assert "#SBATCH --partition=lrd_all_serial" in validator
    assert "#SBATCH --mem=24G" in validator
    assert "QWEN35_H3_HARNESS_AMENDMENT" in validator
    assert "QWEN35_H3_ATTEMPT01_FAILURE_CLOSURE" in validator
    assert "QWEN35_H3_VALIDATOR_AMENDMENT" in validator
    assert "--validator-amendment-preregistration-closure" in validator
    assert "QWEN35_H3_VALIDATOR_CONSISTENCY_AMENDMENT" in validator
    assert "QWEN35_H3_VALIDATOR_V2_FAILURE_CLOSURE" in validator
    assert "QWEN35_H3_VALIDATOR_FAILURE_FORENSIC_CLOSURE" in validator
    assert "--validator-consistency-amendment-preregistration-closure" in validator
    assert "--validator-failure-forensic-closure" in validator
    assert 'export PYTHONPYCACHEPREFIX="$QWEN35_OUTPUT_DIR/pycache"' in validator
    cache_position = guard.index('export PYTHONPYCACHEPREFIX="$G2_JOB_TMP/pycache"')
    first_venv_import = guard.index('"$QWEN35_VENV/bin/python" -c')
    assert cache_position < first_venv_import
