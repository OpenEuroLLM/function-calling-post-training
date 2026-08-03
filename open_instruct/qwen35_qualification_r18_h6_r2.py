"""Fail-closed identities for the preregistered R18 H6 r2 gate."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file

H6_R2_ARTIFACT = "qwen35_r18_h6_common_prefix_strict_trainer_resume_contract"
H6_R2_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h6-r2"
H6_R2_CONTRACT_SHA256 = "7ef1590617f453b6d70afdec185ca7f70574d14e5a791b5ace96103e1d55383d"
H6_R2_HUMAN_PROTOCOL_SHA256 = "35ed7330ba4fc111d4c667f0aaf28b8a57e340d2d0dc83c0e5cb2a12c0fa7c9f"
H6_R2_PREREGISTRATION_CLOSURE_SHA256 = "ad5df3dc8c669ee04ce9cda81ca4e3ccca2693682eba090e621b31ca929e6c3b"
H6_R2_PREREGISTRATION_COMMIT = "a59e200a3f644d3a474003d1e40d39e1c6fb77f0"
H6_R2_BASE_SOURCE_COMMIT = "77b8fb7d7e92b4e28db6aa54e304065e5ede36e6"
H6_R2_BASE_SOURCE_TREE = "cb3a347305e3bc820dcaf43e703a8b436710bf67"
H6_R2_H5_FINAL_CLOSURE_SHA256 = "0d38dfdb814ed0facca07ca92fd28421b8fc66b405e755b14cd8b13708580e4a"
H6_R2_R1_FAILURE_CLOSURE_SHA256 = "bf6dd4e1be1e2bf3ac3c3ca184061af8cdc17f4015f73a3abf8d80c959690e17"
H6_R2_R1_FAILED_COMPARISON_SHA256 = "0784f1a6b94c63775564c0d3166c855e26f2c417356a309055bb6369ad4cce01"
H6_R2_EXPECTED_TARGETS_BY_UPDATE = (54_778, 78_125, 50_730, 69_201, 56_143, 65_752, 45_663, 63_992, 57_005, 61_240)


def load_h6_r2_contract(
    path: Path,
    *,
    human_protocol_path: Path,
    h5_final_closure_path: Path,
    r1_failure_closure_path: Path,
    r1_failed_comparison_path: Path,
    preregistration_closure_path: Path,
) -> tuple[dict[str, Any], str]:
    """Load r2 only when its prospective and failed-predecessor bindings match."""

    digest = sha256_file(path)
    if digest != H6_R2_CONTRACT_SHA256:
        raise ValueError(f"R18 H6 r2 contract digest drift: {digest} != {H6_R2_CONTRACT_SHA256}")
    bound_paths = {
        "human protocol": (human_protocol_path, H6_R2_HUMAN_PROTOCOL_SHA256),
        "H5 final closure": (h5_final_closure_path, H6_R2_H5_FINAL_CLOSURE_SHA256),
        "r1 failure closure": (r1_failure_closure_path, H6_R2_R1_FAILURE_CLOSURE_SHA256),
        "r1 failed comparison": (r1_failed_comparison_path, H6_R2_R1_FAILED_COMPARISON_SHA256),
        "r2 preregistration closure": (preregistration_closure_path, H6_R2_PREREGISTRATION_CLOSURE_SHA256),
    }
    for label, (bound_path, expected) in bound_paths.items():
        if sha256_file(bound_path) != expected:
            raise ValueError(f"R18 H6 r2 {label} digest drift")

    value = load_strict_json(path)
    h5 = load_strict_json(h5_final_closure_path)
    failure = load_strict_json(r1_failure_closure_path)
    preregistration = load_strict_json(preregistration_closure_path)
    if value.get("schema_version") != 1 or value.get("artifact") != H6_R2_ARTIFACT:
        raise ValueError("unsupported R18 H6 r2 contract schema or artifact")
    if value.get("protocol", {}).get("id") != H6_R2_PROTOCOL_ID:
        raise ValueError("R18 H6 r2 protocol identity drift")
    if value.get("protocol") != {
        "date": "2026-07-20",
        "human_path": "methodology/qwen35_hardware_qualification_r18_h6_resume_repair_protocol_r2_20260720.md",
        "human_sha256": H6_R2_HUMAN_PROTOCOL_SHA256,
        "id": H6_R2_PROTOCOL_ID,
    }:
        raise ValueError("R18 H6 r2 human-protocol binding drift")
    if value.get("base_source") != {"commit": H6_R2_BASE_SOURCE_COMMIT, "tree": H6_R2_BASE_SOURCE_TREE}:
        raise ValueError("R18 H6 r2 base-source binding drift")
    if value.get("status") != ("preregistered_after_retained_H6_r1_failure_before_r2_implementation_or_GPU_execution"):
        raise ValueError("R18 H6 r2 preregistration status drift")
    if (
        value.get("scientific_training_authorized") is not False
        or value.get("automatic_successor") is not False
        or value.get("successor_on_complete_independent_pass") != "H7_only"
    ):
        raise ValueError("R18 H6 r2 authority boundary drift")
    if value.get("comparison") != {
        "atol": 0,
        "checkpoint_file_bytes_required_equal": False,
        "deterministic_metric_projection_required_equal": True,
        "discrete_and_structured_state_required_equal": True,
        "model_optimizer_scheduler_rng_tensors_required_bit_equal": True,
        "per_rank_strict_load_audits_required": 4,
        "rtol": 0,
        "schedule_cursor_and_pack_sequence_required_equal": True,
    }:
        raise ValueError("R18 H6 r2 comparison contract drift")
    if value.get("common_prefix_design") != {
        "continuous_checkpoint_step": 5,
        "continuous_updates": [1, 10],
        "reconstructed_resume_prefix_source": "continuous_metrics_steps_1_through_5",
        "resume_checkpoint": "continuous/checkpoint-5",
        "resumed_updates": [6, 10],
    }:
        raise ValueError("R18 H6 r2 common-prefix design drift")
    expected_execution = {
        "cuda_event_step_timing": False,
        "gradient_accumulation_steps": 2,
        "hardware_profile": False,
        "maximum_scheduler_steps": 10,
        "nodes": 1,
        "per_device_train_batch_size": 1,
        "ranks": 4,
        "save_steps": 5,
        "save_total_limit": 2,
        "selected_loss_chunk_size": 512,
        "sequence_length": 32768,
        "slurm_account": "aifac_f02_434",
    }
    if value.get("execution") != expected_execution:
        raise ValueError("R18 H6 r2 execution contract drift")
    exposure = value.get("exposure", {})
    if exposure != {
        "assistant_targets": 602_629,
        "fixed_positions": 2_621_440,
        "optimizer_updates": 10,
        "per_update_assistant_targets": list(H6_R2_EXPECTED_TARGETS_BY_UPDATE),
        "scheduled_packs": 80,
        "synthetic_packs": 0,
    }:
        raise ValueError("R18 H6 r2 exposure contract drift")
    strict_restore = value.get("strict_trainer_model_restore", {})
    if strict_restore != {
        "allow_pickle_model_weights": False,
        "allowed_layouts": ["single_model_safetensors", "indexed_safetensors_shards"],
        "allowlisted_duplicate_source_mapping": {
            "model.language_model.embed_tokens.weight": ["lm_head.weight", "model.embed_tokens.weight"]
        },
        "complete_metadata_preflight_before_copy": True,
        "missing_or_unexpected_keys_allowed": False,
        "preserve_live_parameter_objects": True,
        "require_exact_post_copy_values": True,
        "require_fp32_source_and_target": True,
        "require_tied_embeddings_before_and_after": True,
        "upstream_trainer_strict_false_fallback_allowed": False,
    }:
        raise ValueError("R18 H6 r2 strict-restore contract drift")
    immutable = value.get("immutable_predecessors", {})
    if (
        immutable.get("h5_final_closure_sha256") != H6_R2_H5_FINAL_CLOSURE_SHA256
        or immutable.get("h6_r1_attempt01_failure_closure_sha256") != H6_R2_R1_FAILURE_CLOSURE_SHA256
        or immutable.get("h6_r1_failed_comparison_sha256") != H6_R2_R1_FAILED_COMPARISON_SHA256
    ):
        raise ValueError("R18 H6 r2 immutable-predecessor binding drift")
    if (
        h5.get("status") != "passed_H5_independently_closed_H6_only_authorized"
        or h5.get("scientific_training_authorized") is not False
        or failure.get("status") != "failed_retained_H6_not_passed_no_successor_authorized"
        or failure.get("authority", {}).get("h6_passed") is not False
        or failure.get("authority", {}).get("scientific_training_authorized") is not False
    ):
        raise ValueError("R18 H6 r2 predecessor disposition drift")
    if (
        preregistration.get("status") != "preregistered_H6_r2_implementation_only_authorized"
        or preregistration.get("authority", {}).get("h6_r2_implementation_authorized") is not True
        or preregistration.get("authority", {}).get("h6_r2_submission_authorized") is not False
        or preregistration.get("authority", {}).get("scientific_training_authorized") is not False
        or preregistration.get("preregistration_commit", {}).get("commit") != H6_R2_PREREGISTRATION_COMMIT
        or preregistration.get("preregistration_commit", {}).get("parent") != H6_R2_BASE_SOURCE_COMMIT
        or preregistration.get("contract", {}).get("sha256") != H6_R2_CONTRACT_SHA256
        or preregistration.get("human_protocol", {}).get("sha256") != H6_R2_HUMAN_PROTOCOL_SHA256
    ):
        raise ValueError("R18 H6 r2 preregistration-closure drift")
    return value, digest


def validate_h6_r2_source_delta(repository: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    """Require a clean preregistration descendant inside the r2 source allowlist."""

    repository = repository.resolve()

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()

    head = git("rev-parse", "HEAD")
    if expected_head is not None and head != expected_head:
        raise ValueError(f"R18 H6 r2 implementation HEAD drift: {head} != {expected_head}")
    if git("status", "--porcelain"):
        raise ValueError("R18 H6 r2 implementation repository is not clean")
    if git("merge-base", "--is-ancestor", H6_R2_PREREGISTRATION_COMMIT, head) != "":
        raise AssertionError("unexpected git merge-base output")
    contract_path = repository / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h6_r2.json"
    if sha256_file(contract_path) != H6_R2_CONTRACT_SHA256:
        raise ValueError("R18 H6 r2 implementation changed its preregistered contract")
    contract = load_strict_json(contract_path)
    allowed = set(contract["source"]["allowed_implementation_paths"])
    observed = set(filter(None, git("diff", "--name-only", f"{H6_R2_PREREGISTRATION_COMMIT}..{head}").splitlines()))
    if not observed:
        raise ValueError("R18 H6 r2 implementation source delta is empty")
    if not observed <= allowed:
        raise ValueError(f"R18 H6 r2 implementation changed forbidden paths: {sorted(observed - allowed)}")
    return {
        "allowed_paths": sorted(allowed),
        "head": head,
        "observed_changed_paths": sorted(observed),
        "preregistration_commit": H6_R2_PREREGISTRATION_COMMIT,
        "status": "passed",
    }
