from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from scripts.train.qwen35.validate_qwen35_h6_r18_r2 import validate_strict_load_audits

from open_instruct.qwen35_qualification_r18_h4 import sha256_file
from open_instruct.qwen35_qualification_r18_h6_r2 import (
    H6_R2_CONTRACT_SHA256,
    H6_R2_EXPECTED_TARGETS_BY_UPDATE,
    load_h6_r2_contract,
    validate_h6_r2_source_delta,
)

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parents[1]
CONTRACT = REPOSITORY / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h6_r2.json"
HUMAN_PROTOCOL = WORKSPACE / "methodology/qwen35_hardware_qualification_r18_h6_resume_repair_protocol_r2_20260720.md"
H5_FINAL = WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h5_final_closure_20260720.json"
R1_FAILURE = (
    WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h6_gpu_attempt01_failure_closure_20260720.json"
)
R1_COMPARISON = (
    WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h6_gpu_attempt01_49887474/run/"
    "h6_checkpoint_comparison.json"
)
PREREGISTRATION = (
    WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h6_r2_preregistration_closure_20260720.json"
)


def _load_contract(path: Path = CONTRACT):
    return load_h6_r2_contract(
        path,
        human_protocol_path=HUMAN_PROTOCOL,
        h5_final_closure_path=H5_FINAL,
        r1_failure_closure_path=R1_FAILURE,
        r1_failed_comparison_path=R1_COMPARISON,
        preregistration_closure_path=PREREGISTRATION,
    )


def test_h6_r2_contract_and_retained_failure_bindings_load_exactly() -> None:
    contract, digest = _load_contract()
    assert digest == H6_R2_CONTRACT_SHA256
    assert contract["common_prefix_design"]["resume_checkpoint"] == "continuous/checkpoint-5"
    assert contract["exposure"]["per_update_assistant_targets"] == list(H6_R2_EXPECTED_TARGETS_BY_UPDATE)
    assert contract["scientific_training_authorized"] is False
    assert contract["successor_on_complete_independent_pass"] == "H7_only"


def test_h6_r2_contract_rejects_semantically_plausible_mutation(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text())
    value["comparison"]["atol"] = 1e-8
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="contract digest drift"):
        _load_contract(mutated)


def _audit(checkpoint: Path, rank: int) -> dict:
    model_path = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    size = model_path.stat().st_size
    return {
        "all_ranks_passed": True,
        "artifact": "qwen35_strict_trainer_checkpoint_load_audit",
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_identity_sha256": "a" * 64,
        "config_architectures": ["Qwen3_5ForCausalLM"],
        "config_model_type": "qwen3_5_text",
        "config_sha256": sha256_file(config_path),
        "copied_source_tensor_count": 320,
        "copied_unique_elements": 752_393_024,
        "exact_post_copy_values": True,
        "layout": "single_model_safetensors",
        "mapping_rows_sha256": "b" * 64,
        "metadata_preflight_completed_before_copy": True,
        "missing_source_keys": [],
        "optimizer_execution_authorized": True,
        "parameter_objects_preserved": True,
        "rank": rank,
        "safetensor_header_manifest_sha256": "c" * 64,
        "schema_version": 1,
        "source_dtype": "F32",
        "source_rows_sha256": "d" * 64,
        "source_tensor_count": 320,
        "status": "passed",
        "storage_pointers_preserved": True,
        "target_dtype": "torch.float32",
        "target_state_key_count": 321,
        "tied_input_output_embeddings_after": True,
        "tied_input_output_embeddings_before": True,
        "tied_source_key": "model.language_model.embed_tokens.weight",
        "tied_target_keys": ["lm_head.weight", "model.embed_tokens.weight"],
        "unexpected_source_keys": [],
        "unique_target_storage_count": 320,
        "upstream_trainer_strict_false_used": False,
        "weight_files": {
            "model.safetensors": {"header_sha256": "e" * 64, "sha256": sha256_file(model_path), "size": size}
        },
    }


def _write_audit_fixture(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "continuous" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    save_file({"tiny": torch.zeros(1)}, checkpoint / "model.safetensors")
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text"})
    )
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    for rank in range(4):
        (resumed / f"qwen35_resume_model_load_audit_rank_{rank}.json").write_text(
            json.dumps(_audit(checkpoint, rank), sort_keys=True) + "\n"
        )
    return resumed, checkpoint


def test_four_strict_load_audits_are_reconciled_except_rank(tmp_path: Path) -> None:
    resumed, checkpoint = _write_audit_fixture(tmp_path)
    report = validate_strict_load_audits(resumed, checkpoint)
    assert report["status"] == "passed"
    assert report["rank_count"] == 4
    assert set(report["audit_sha256"]) == {"0", "1", "2", "3"}


@pytest.mark.parametrize("mutation", ["missing", "semantic", "cross_rank", "failure_artifact"])
def test_strict_load_audit_validator_rejects_incomplete_or_drifting_evidence(tmp_path: Path, mutation: str) -> None:
    resumed, checkpoint = _write_audit_fixture(tmp_path)
    rank_two = resumed / "qwen35_resume_model_load_audit_rank_2.json"
    if mutation == "missing":
        rank_two.unlink()
    elif mutation == "semantic":
        value = json.loads(rank_two.read_text())
        value["upstream_trainer_strict_false_used"] = True
        rank_two.write_text(json.dumps(value))
    elif mutation == "cross_rank":
        value = json.loads(rank_two.read_text())
        value["mapping_rows_sha256"] = "f" * 64
        rank_two.write_text(json.dumps(value))
    else:
        (resumed / "qwen35_resume_model_load_failure_rank_1.json").write_text("{}")
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_strict_load_audits(resumed, checkpoint)


def test_r2_wrapper_is_common_prefix_bounded_personal_and_non_scientific() -> None:
    wrapper = (REPOSITORY / "scripts/train/qwen35/leonardo_h6_r18_r2.sbatch").read_text()
    assert "#SBATCH --account=aifac_f02_434" in wrapper
    assert "#SBATCH --gres=gpu:4" in wrapper
    assert "#SBATCH --time=01:00:00" in wrapper
    assert wrapper.count('"$QWEN35_VENV/bin/torchrun"') == 2
    assert "resume_checkpoint=$continuous_dir/checkpoint-5" in wrapper
    assert '--resume_from_checkpoint "$resume_checkpoint"' in wrapper
    assert "--save_steps 5" in wrapper
    assert "--save_total_limit 2" in wrapper
    assert "--expected_arm_id C00" in wrapper
    assert "--selected_loss_chunk_size 512" in wrapper
    assert "--use_liger_fused_linear_cross_entropy false" in wrapper
    assert "--atol 0" in wrapper and "--rtol 0" in wrapper
    assert "R18_H6_R2_PRODUCER_PASSED_PENDING_INDEPENDENT_CLOSURE" in wrapper
    assert "sbatch " not in wrapper
    assert "C01" not in wrapper and "BFCL" not in wrapper and "tau2" not in wrapper


def test_h6_r2_source_delta_is_clean_and_inside_preregistered_allowlist() -> None:
    if subprocess.check_output(["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True).strip():
        pytest.skip("source-delta gate is exercised after the implementation commit")
    head = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    report = validate_h6_r2_source_delta(REPOSITORY, expected_head=head)
    assert report["status"] == "passed"
    assert set(report["observed_changed_paths"]) <= set(report["allowed_paths"])


def test_rank_projection_rejects_every_field_other_than_rank(tmp_path: Path) -> None:
    resumed, checkpoint = _write_audit_fixture(tmp_path)
    baseline = json.loads((resumed / "qwen35_resume_model_load_audit_rank_3.json").read_text())
    for key in ("checkpoint_identity_sha256", "source_rows_sha256", "weight_files"):
        value = copy.deepcopy(baseline)
        value[key] = "0" * 64 if key != "weight_files" else {}
        (resumed / "qwen35_resume_model_load_audit_rank_3.json").write_text(json.dumps(value))
        with pytest.raises(ValueError):
            validate_strict_load_audits(resumed, checkpoint)
        (resumed / "qwen35_resume_model_load_audit_rank_3.json").write_text(json.dumps(baseline))
