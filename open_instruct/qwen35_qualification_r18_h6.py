"""Fail-closed identities and helpers for the preregistered R18 H6 gate."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file

H6_ARTIFACT = "qwen35_r18_h6_continuous_resume_equality_contract"
H6_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h6-r1"
H6_CONTRACT_SHA256 = "376d2a210c35a791b79743f5354808d767303ae422f907ca0fbf0c42b7361e10"
H6_HUMAN_PROTOCOL_SHA256 = "2add03918c475fe67066acb9d7b1acad4fb50600f68cd968987cd76e75a87081"
H6_H5_FINAL_CLOSURE_SHA256 = "0d38dfdb814ed0facca07ca92fd28421b8fc66b405e755b14cd8b13708580e4a"
H6_PREREGISTRATION_CLOSURE_SHA256 = "47ff866893804a86617b700e13cbbe11d2c3d6015df7581f2fb784e94944f335"
H6_PREREGISTRATION_COMMIT = "9d10f49745754f07cf9e09d8cefcb604b3cacd78"
H6_BASE_SOURCE_COMMIT = "260a804037dd45d68fdb57f8d595ec4781b88a9c"
H6_BASE_SOURCE_TREE = "913524f0e921bd00801e9b440a0d030a0f0b058c"
H6_SCHEDULE_FILE_SHA256 = "4e8d4a9fb1a1fd0c92abff0321e71355e066c5723f626bed847fd799704e0c2c"
H6_SCHEDULE_SHA256 = "6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281"
H6_SCHEDULE_ENTRIES_SHA256 = "163db66ee13a7f07659be59f90b91d2a21de7ed9067d841cc2feabf49dd239cd"
H6_H5_CHECKPOINT_MANIFEST_SHA256 = "76aee3cd63d1a0bf6291f65cf0b065c102f6710d497de9ae3782e75bcf67cfa9"
H6_H5_CHECKPOINT_RELOAD_SHA256 = "8d3b16e0f993e3a44d4e42f003ca0cf0997fc8c415d1cf65d53e86be6ce386fb"
H6_H5_INDEPENDENT_CLOSURE_SHA256 = "7abe52065816c24694f0d99786cc31acd2cc8826a5fb725024182117ccafb75c"
H6_H5_METRICS_SHA256 = "693fcc630fff09fea6a9d42fdbd2843f8f4b42f1827d621b1474e601ae8fb848"
H6_H5_MODEL_SHA256 = "11b8000cca8d0c0049b3607728814e5268ad1dab15cda17403527b98814e026f"
H6_MODEL_MANIFEST_SHA256 = "017de528d444839a04929fc68b45242169675a917f8249baa295b9b969618e39"
H6_QUALIFICATION_MANIFEST_SHA256 = "679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0"
H6_RUNTIME_REPORT_SHA256 = "728e6a5d618db6ea7f321fe8013133e5590bc89616ec646d3dba99b7848266cf"
H6_EXPECTED_TARGETS_BY_UPDATE = (54_778, 78_125, 50_730, 69_201, 56_143, 65_752, 45_663, 63_992, 57_005, 61_240)


def load_h6_contract(
    path: Path,
    *,
    human_protocol_path: Path,
    h5_final_closure_path: Path,
    preregistration_closure_path: Path,
) -> tuple[dict[str, Any], str]:
    """Load H6 only when every prospective identity and authority boundary matches."""

    digest = sha256_file(path)
    if digest != H6_CONTRACT_SHA256:
        raise ValueError(f"R18 H6 contract digest drift: {digest} != {H6_CONTRACT_SHA256}")
    if sha256_file(human_protocol_path) != H6_HUMAN_PROTOCOL_SHA256:
        raise ValueError("R18 H6 human-protocol digest drift")
    if sha256_file(h5_final_closure_path) != H6_H5_FINAL_CLOSURE_SHA256:
        raise ValueError("R18 H6 H5-final-closure digest drift")
    if sha256_file(preregistration_closure_path) != H6_PREREGISTRATION_CLOSURE_SHA256:
        raise ValueError("R18 H6 preregistration-closure digest drift")

    value = load_strict_json(path)
    h5 = load_strict_json(h5_final_closure_path)
    closure = load_strict_json(preregistration_closure_path)
    if value.get("schema_version") != 1 or value.get("artifact") != H6_ARTIFACT:
        raise ValueError("unsupported R18 H6 contract schema or artifact")
    if value.get("protocol_id") != H6_PROTOCOL_ID:
        raise ValueError("R18 H6 protocol identity drift")
    if value.get("status") != "preregistered_after_independent_H5_pass_before_H6_implementation_or_GPU_execution":
        raise ValueError("R18 H6 preregistration status drift")
    if (
        value.get("scientific_training_authorized") is not False
        or value.get("automatic_successor") is not False
        or value.get("successor_on_complete_independent_pass") != "H7_only"
    ):
        raise ValueError("R18 H6 authority boundary drift")
    if value.get("human_protocol") != {
        "path": "methodology/qwen35_hardware_qualification_r18_h6_protocol_r1_20260720.md",
        "sha256": H6_HUMAN_PROTOCOL_SHA256,
    }:
        raise ValueError("R18 H6 human-protocol binding drift")
    if value.get("base_source") != {"commit": H6_BASE_SOURCE_COMMIT, "tree": H6_BASE_SOURCE_TREE}:
        raise ValueError("R18 H6 base-source binding drift")
    if value.get("comparison") != {
        "atol": 0,
        "checkpoint_file_bytes_required_equal": False,
        "deterministic_metric_projection_required_equal": True,
        "discrete_and_structured_state_required_equal": True,
        "model_optimizer_scheduler_rng_tensors_required_bit_equal": True,
        "rtol": 0,
        "schedule_cursor_and_pack_sequence_required_equal": True,
    }:
        raise ValueError("R18 H6 comparison contract drift")
    expected_execution = {
        "cuda_event_step_timing": False,
        "gradient_accumulation_steps": 2,
        "hardware_profile": False,
        "maximum_scheduler_steps": 10,
        "nodes": 1,
        "per_device_train_batch_size": 1,
        "ranks": 4,
        "save_steps": 10,
        "selected_loss_chunk_size": 512,
        "sequence_length": 32768,
        "slurm_account": "aifac_f02_434",
    }
    if value.get("execution") != expected_execution:
        raise ValueError("R18 H6 execution contract drift")
    expected_schedule = {
        "embedded_schedule_sha256": H6_SCHEDULE_SHA256,
        "entries_sha256": H6_SCHEDULE_ENTRIES_SHA256,
        "file_sha256": H6_SCHEDULE_FILE_SHA256,
        "optimizer_updates": 10,
        "scheduled_packs": 80,
        "synthetic_packs": 0,
    }
    if value.get("schedule") != expected_schedule:
        raise ValueError("R18 H6 schedule contract drift")
    exposure = value.get("exposure", {})
    if exposure.get("per_update_assistant_targets") != list(H6_EXPECTED_TARGETS_BY_UPDATE):
        raise ValueError("R18 H6 per-update assistant-target contract drift")
    if exposure != {
        "assistant_targets": 602_629,
        "attention_length_squared": 11_096_774_268,
        "fixed_positions": 2_621_440,
        "optimizer_updates": 10,
        "padding_positions": 819,
        "per_update_assistant_targets": list(H6_EXPECTED_TARGETS_BY_UPDATE),
        "real_positions": 2_620_621,
        "scheduled_packs": 80,
        "synthetic_packs": 0,
    }:
        raise ValueError("R18 H6 exposure contract drift")
    if value.get("continuous_path") != {
        "expected_final_global_step": 10,
        "expected_initial_global_step": 0,
        "optimizer_updates_executed": 10,
        "resume_checkpoint": None,
        "schedule_index_interval": [0, 79],
    }:
        raise ValueError("R18 H6 continuous-path contract drift")
    if value.get("resume_path") != {
        "expected_final_global_step": 10,
        "expected_initial_global_step": 5,
        "optimizer_updates_executed": 5,
        "resume_checkpoint": "accepted_H5_checkpoint_5",
        "schedule_index_interval": [40, 79],
    }:
        raise ValueError("R18 H6 resumed-path contract drift")

    predecessor = value.get("h5_predecessor", {})
    if (
        predecessor.get("final_closure_sha256") != H6_H5_FINAL_CLOSURE_SHA256
        or predecessor.get("checkpoint_file_manifest_sha256") != H6_H5_CHECKPOINT_MANIFEST_SHA256
        or predecessor.get("checkpoint_reload_validation_sha256") != H6_H5_CHECKPOINT_RELOAD_SHA256
        or predecessor.get("independent_closure_sha256") != H6_H5_INDEPENDENT_CLOSURE_SHA256
        or predecessor.get("exact_metrics_jsonl_sha256") != H6_H5_METRICS_SHA256
        or predecessor.get("checkpoint_model_safetensors_sha256") != H6_H5_MODEL_SHA256
        or predecessor.get("status") != "passed_H5_independently_closed_H6_only_authorized"
    ):
        raise ValueError("R18 H6 H5 predecessor binding drift")
    if (
        h5.get("status") != "passed_H5_independently_closed_H6_only_authorized"
        or h5.get("successor_authorized") != "H6_only"
        or h5.get("scientific_training_authorized") is not False
        or h5.get("source_commit") != H6_BASE_SOURCE_COMMIT
        or h5.get("source_tree") != H6_BASE_SOURCE_TREE
    ):
        raise ValueError("R18 H6 H5 final-closure disposition drift")
    if (
        closure.get("status") != "preregistered_H6_implementation_and_CPU_staging_only"
        or closure.get("scientific_training_authorized") is not False
        or closure.get("preregistration_source", {}).get("commit") != H6_PREREGISTRATION_COMMIT
        or closure.get("preregistration_source", {}).get("parent") != H6_BASE_SOURCE_COMMIT
        or closure.get("machine_contract", {}).get("sha256") != H6_CONTRACT_SHA256
        or closure.get("human_protocol", {}).get("sha256") != H6_HUMAN_PROTOCOL_SHA256
        or closure.get("h5_predecessor", {}).get("final_closure_sha256") != H6_H5_FINAL_CLOSURE_SHA256
    ):
        raise ValueError("R18 H6 preregistration closure drift")
    return value, digest


def validate_h6_source_delta(repository: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    """Require a clean descendant whose delta is contained in the frozen H6 allowlist."""

    repository = repository.resolve()

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()

    head = git("rev-parse", "HEAD")
    if expected_head is not None and head != expected_head:
        raise ValueError(f"R18 H6 implementation HEAD drift: {head} != {expected_head}")
    if git("status", "--porcelain"):
        raise ValueError("R18 H6 implementation repository is not clean")
    if git("merge-base", "--is-ancestor", H6_PREREGISTRATION_COMMIT, head) != "":
        raise AssertionError("unexpected git merge-base output")
    contract_path = repository / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h6.json"
    if sha256_file(contract_path) != H6_CONTRACT_SHA256:
        raise ValueError("R18 H6 implementation changed its preregistered contract")
    contract = load_strict_json(contract_path)
    allowed = set(contract["source"]["allowed_h6_implementation_paths"])
    observed = set(filter(None, git("diff", "--name-only", f"{H6_PREREGISTRATION_COMMIT}..{head}").splitlines()))
    if not observed:
        raise ValueError("R18 H6 implementation source delta is empty")
    if not observed <= allowed:
        raise ValueError(f"R18 H6 implementation changed forbidden paths: {sorted(observed - allowed)}")
    return {
        "allowed_paths": sorted(allowed),
        "head": head,
        "observed_changed_paths": sorted(observed),
        "preregistration_commit": H6_PREREGISTRATION_COMMIT,
        "status": "passed",
    }
