"""Fail-closed identities and helpers for the preregistered R18 H5 gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file

H5_ARTIFACT = "qwen35_r18_h5_four_gpu_production_path_contract"
H5_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h5-r1"
H5_CONTRACT_SHA256 = "74476804f5bb72f2212394f7d1646b31d1f3e2df8ca0f65cda261997f2efebb1"
H5_HUMAN_PROTOCOL_SHA256 = "fd35b116ccd7a4f806f577bd16a752d1e3149166751ce9101f79739e98d9468d"
H5_PREREGISTRATION_CLOSURE_SHA256 = "1b7fc59e698f38192a8a4019a6ff54baa4fab2672376375a87f7864343101148"
H5_PREREGISTRATION_COMMIT = "311d72c425425302d39671c3f737a299bc4aa9b7"
H5_HARNESS_AMENDMENT_ARTIFACT = "qwen35_r18_h5_distributed_lifecycle_and_trace_integrity_amendment"
H5_HARNESS_AMENDMENT_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h5-harness-amendment-r1"
H5_HARNESS_AMENDMENT_SHA256 = "d5f1521240ab5f9054f75ee50cf8659711edfdc022a176ab99c902572807c7f5"
H5_HARNESS_HUMAN_AMENDMENT_SHA256 = "315b6bbdd594972becd4cf0d6515c46db25fec5ec3bd45a76fc72f524d681a5b"
H5_ATTEMPT01_FAILURE_CLOSURE_SHA256 = "5ef82ccdc46325c97040d85019d57bd5a63f2f3029a39c3e55a03fc089cc1d80"
H5_ATTEMPT01_FAILED_IMPLEMENTATION_COMMIT = "cb46873dc6929042b4a4b20b18729a17420ac476"
H5_HARNESS_AMENDMENT_PREREGISTRATION_COMMIT = "c6b5ef656a7af36a4010d51e331bd3f9d7faf49c"
H5_HARNESS_AMENDMENT_R2_ARTIFACT = "qwen35_r18_h5_metrics_barrier_and_reload_serialization_amendment"
H5_HARNESS_AMENDMENT_R2_PROTOCOL_ID = "qwen35-hardware-qualification-r18-h5-harness-amendment-r2"
H5_HARNESS_AMENDMENT_R2_SHA256 = "0ffb0610d6bbef4bc42c3b0c6ecee9327214fb77a4d219440af069fbde09a057"
H5_HARNESS_HUMAN_AMENDMENT_R2_SHA256 = "3e976e767b2c5b4897f804a18b54c42da8427a6585ad05319d7a3ab27568bc5b"
H5_ATTEMPT02_FAILURE_CLOSURE_SHA256 = "256a2867af66ba8b162d3ee5f9b22b41e7405ecab6cb4e669b4dbb26da6414e9"
H5_ATTEMPT02_RELOAD_TYPE_DIAGNOSTIC_SHA256 = "891c28dc4484780349ea05bf388b70c5e7ecbd2c64dc633154ccbe0f9895ccf1"
H5_ATTEMPT02_FAILED_IMPLEMENTATION_COMMIT = "f58eb54b6c69e02b87f6976ab2972ec6b2578511"
H5_ATTEMPT02_FAILED_IMPLEMENTATION_TREE = "32f53a9610dca246ef281fc5b24ad13eeab8ee03"
H5_HARNESS_AMENDMENT_R2_PREREGISTRATION_COMMIT = "e20e4324cd837171b5a7d55626e81ecb54245d53"
H5_SCHEDULE_FILE_SHA256 = "4e8d4a9fb1a1fd0c92abff0321e71355e066c5723f626bed847fd799704e0c2c"
H5_SCHEDULE_SHA256 = "6fb941968793bc8d3b11c178279de0be16ffde97837b16401e9d1c457a046281"
H5_SCHEDULE_ENTRIES_SHA256 = "163db66ee13a7f07659be59f90b91d2a21de7ed9067d841cc2feabf49dd239cd"
H5_FIRST_FIVE_ENTRIES_SHA256 = "feb6beb0730fc78288832ba49eac17f312c1cfdb55e4d7bb1d81bd189ba453d7"
H5_EXPECTED_TARGETS_BY_UPDATE = (54_778, 78_125, 50_730, 69_201, 56_143)
H5_SELECTED_CHUNK_SIZE = 512
H5_WORLD_SIZE = 4
H5_GRADIENT_ACCUMULATION = 2
H5_FINAL_STEP = 5
H5_SCHEDULER_HORIZON = 10
R18_MANIFEST_SHA256 = "679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def load_h5_contract(
    path: Path, *, human_protocol_path: Path, preregistration_closure_path: Path
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != H5_CONTRACT_SHA256:
        raise ValueError(f"R18 H5 contract digest drift: {digest} != {H5_CONTRACT_SHA256}")
    if sha256_file(human_protocol_path) != H5_HUMAN_PROTOCOL_SHA256:
        raise ValueError("R18 H5 human-protocol digest drift")
    if sha256_file(preregistration_closure_path) != H5_PREREGISTRATION_CLOSURE_SHA256:
        raise ValueError("R18 H5 preregistration-closure digest drift")
    value = load_strict_json(path)
    closure = load_strict_json(preregistration_closure_path)
    if value.get("schema_version") != 1 or value.get("artifact") != H5_ARTIFACT:
        raise ValueError("unsupported R18 H5 contract schema or artifact")
    if value.get("protocol_id") != H5_PROTOCOL_ID:
        raise ValueError("R18 H5 protocol identity drift")
    if value.get("status") != "preregistered_after_H4_and_schedule_freeze_before_H5_implementation_or_GPU_execution":
        raise ValueError("R18 H5 contract preregistration status drift")
    if value.get("scientific_training_authorized") is not False:
        raise ValueError("R18 H5 contract may not authorize scientific training")
    if value.get("automatic_successor") is not False or value.get("allowed_successor_on_complete_pass") != "H6_only":
        raise ValueError("R18 H5 successor boundary drift")
    if value.get("human_protocol") != {
        "path": "methodology/qwen35_hardware_qualification_r18_h5_protocol_r1_20260720.md",
        "sha256": H5_HUMAN_PROTOCOL_SHA256,
    }:
        raise ValueError("R18 H5 human-protocol contract drift")
    if closure.get("status") != "preregistered_H5_implementation_and_CPU_staging_only":
        raise ValueError("R18 H5 preregistration closure status drift")
    if closure.get("scientific_training_authorized") is not False:
        raise ValueError("R18 H5 preregistration closure authorizes scientific training")
    if closure.get("preregistration_source", {}).get("commit") != H5_PREREGISTRATION_COMMIT:
        raise ValueError("R18 H5 preregistration source identity drift")
    if closure.get("machine_contract", {}).get("sha256") != H5_CONTRACT_SHA256:
        raise ValueError("R18 H5 preregistration closure machine-contract drift")
    if closure.get("human_protocol", {}).get("sha256") != H5_HUMAN_PROTOCOL_SHA256:
        raise ValueError("R18 H5 preregistration closure human-protocol drift")

    execution = value.get("execution", {})
    expected_execution = {
        "gradient_accumulation_steps": H5_GRADIENT_ACCUMULATION,
        "maximum_scheduler_steps": H5_SCHEDULER_HORIZON,
        "nodes": 1,
        "ranks": H5_WORLD_SIZE,
        "selected_loss_chunk_size": H5_SELECTED_CHUNK_SIZE,
        "sequence_length": 32768,
        "slurm_account": "aifac_f02_434",
        "stop_after_steps": H5_FINAL_STEP,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"R18 H5 execution {key} drift")
    if execution.get("liger_execution_allowed") is not False:
        raise ValueError("R18 H5 must forbid Liger execution")
    if value.get("parent", {}).get("r18_machine_manifest_sha256") != R18_MANIFEST_SHA256:
        raise ValueError("R18 H5 R18-manifest predecessor drift")
    schedule = value.get("schedule", {})
    schedule_expected = {
        "embedded_schedule_sha256": H5_SCHEDULE_SHA256,
        "entries_sha256": H5_SCHEDULE_ENTRIES_SHA256,
        "file_sha256": H5_SCHEDULE_FILE_SHA256,
        "optimizer_updates": H5_SCHEDULER_HORIZON,
        "scheduled_packs": 80,
        "synthetic_packs": 0,
    }
    for key, expected in schedule_expected.items():
        if schedule.get(key) != expected:
            raise ValueError(f"R18 H5 schedule {key} drift")
    exposure = value.get("five_update_exposure", {})
    if exposure.get("entries_sha256") != H5_FIRST_FIVE_ENTRIES_SHA256:
        raise ValueError("R18 H5 five-update entry digest drift")
    if exposure.get("per_update_assistant_targets") != list(H5_EXPECTED_TARGETS_BY_UPDATE):
        raise ValueError("R18 H5 per-update target divisors drift")
    if exposure.get("schedule_index_interval") != [0, 39] or exposure.get("scheduled_packs") != 40:
        raise ValueError("R18 H5 exposure interval drift")
    return value, digest


def load_h5_harness_amendment(
    path: Path, *, human_amendment_path: Path, attempt01_failure_closure_path: Path
) -> tuple[dict[str, Any], str]:
    """Load the prospective H5 attempt-1 harness amendment by exact identity."""

    digest = sha256_file(path)
    if digest != H5_HARNESS_AMENDMENT_SHA256:
        raise ValueError(f"R18 H5 harness-amendment digest drift: {digest} != {H5_HARNESS_AMENDMENT_SHA256}")
    if sha256_file(human_amendment_path) != H5_HARNESS_HUMAN_AMENDMENT_SHA256:
        raise ValueError("R18 H5 human harness-amendment digest drift")
    if sha256_file(attempt01_failure_closure_path) != H5_ATTEMPT01_FAILURE_CLOSURE_SHA256:
        raise ValueError("R18 H5 attempt-01 failure-closure digest drift")
    value = load_strict_json(path)
    failure = load_strict_json(attempt01_failure_closure_path)
    if value.get("schema_version") != 1 or value.get("artifact") != H5_HARNESS_AMENDMENT_ARTIFACT:
        raise ValueError("unsupported R18 H5 harness-amendment schema or artifact")
    if value.get("protocol_id") != H5_HARNESS_AMENDMENT_PROTOCOL_ID:
        raise ValueError("R18 H5 harness-amendment protocol drift")
    if value.get("status") != (
        "preregistered_after_attempt01_failure_before_repair_source_or_replacement_gpu_execution"
    ):
        raise ValueError("R18 H5 harness-amendment preregistration status drift")
    if value.get("base_h5_contract_sha256") != H5_CONTRACT_SHA256:
        raise ValueError("R18 H5 harness-amendment parent-contract drift")
    if value.get("human_amendment", {}).get("sha256") != H5_HARNESS_HUMAN_AMENDMENT_SHA256:
        raise ValueError("R18 H5 harness-amendment human-document binding drift")
    failed = value.get("failed_attempt", {})
    if (
        failed.get("failure_closure_sha256") != H5_ATTEMPT01_FAILURE_CLOSURE_SHA256
        or failed.get("source_commit") != H5_ATTEMPT01_FAILED_IMPLEMENTATION_COMMIT
        or failed.get("job_id") != "49878043"
    ):
        raise ValueError("R18 H5 harness-amendment failed-attempt binding drift")
    if (
        failure.get("status") != "failed_retained_no_H6_or_scientific_successor"
        or failure.get("authoritative_disposition") != "failed_nccl_or_rank_completion"
        or failure.get("no_retroactive_pass") is not True
        or failure.get("scientific_training_authorized") is not False
    ):
        raise ValueError("R18 H5 attempt-01 failure-closure disposition drift")
    if (
        value.get("scientific_training_authorized") is not False
        or value.get("automatic_successor") is not False
        or value.get("successor_on_complete_independent_pass") != "H6_only"
        or value.get("full_h5_rerun_required") is not True
        or value.get("model_loss_data_schedule_optimizer_precision_or_threshold_change_allowed") is not False
    ):
        raise ValueError("R18 H5 harness-amendment authority boundary drift")
    sanitizer = value.get("frozen_corrections", {}).get("sanitizer", {})
    if sanitizer != {
        "from_ascii": '"Process Group Description": ,',
        "replacement_size_delta_bytes": 4,
        "streaming_boundary_safe": True,
        "to_ascii": '"Process Group Description": null,',
    }:
        raise ValueError("R18 H5 harness-amendment sanitizer contract drift")
    return value, digest


def load_h5_harness_amendment_r2(
    path: Path, *, human_amendment_path: Path, attempt02_failure_closure_path: Path, reload_type_diagnostic_path: Path
) -> tuple[dict[str, Any], str]:
    """Load the prospective H5 attempt-2 harness amendment by exact identity."""

    digest = sha256_file(path)
    if digest != H5_HARNESS_AMENDMENT_R2_SHA256:
        raise ValueError(f"R18 H5 harness-amendment-R2 digest drift: {digest} != {H5_HARNESS_AMENDMENT_R2_SHA256}")
    if sha256_file(human_amendment_path) != H5_HARNESS_HUMAN_AMENDMENT_R2_SHA256:
        raise ValueError("R18 H5 human harness-amendment-R2 digest drift")
    if sha256_file(attempt02_failure_closure_path) != H5_ATTEMPT02_FAILURE_CLOSURE_SHA256:
        raise ValueError("R18 H5 attempt-02 failure-closure digest drift")
    if sha256_file(reload_type_diagnostic_path) != H5_ATTEMPT02_RELOAD_TYPE_DIAGNOSTIC_SHA256:
        raise ValueError("R18 H5 attempt-02 reload-type diagnostic digest drift")
    value = load_strict_json(path)
    failure = load_strict_json(attempt02_failure_closure_path)
    diagnostic = load_strict_json(reload_type_diagnostic_path)
    if value.get("schema_version") != 1 or value.get("artifact") != H5_HARNESS_AMENDMENT_R2_ARTIFACT:
        raise ValueError("unsupported R18 H5 harness-amendment-R2 schema or artifact")
    if value.get("protocol_id") != H5_HARNESS_AMENDMENT_R2_PROTOCOL_ID:
        raise ValueError("R18 H5 harness-amendment-R2 protocol drift")
    if value.get("status") != (
        "preregistered_after_attempt02_failure_before_r2_repair_source_or_replacement_gpu_execution"
    ):
        raise ValueError("R18 H5 harness-amendment-R2 preregistration status drift")
    if (
        value.get("base_h5_contract_sha256") != H5_CONTRACT_SHA256
        or value.get("parent_harness_amendment_sha256") != H5_HARNESS_AMENDMENT_SHA256
    ):
        raise ValueError("R18 H5 harness-amendment-R2 parent binding drift")
    if value.get("human_amendment", {}).get("sha256") != H5_HARNESS_HUMAN_AMENDMENT_R2_SHA256:
        raise ValueError("R18 H5 harness-amendment-R2 human-document binding drift")
    failed = value.get("failed_attempt", {})
    if failed != {
        "failure_closure_path": (
            "artifacts/qwen35_hardware_qualification_20260718/r18_h5_attempt02_failure_closure_20260720.json"
        ),
        "failure_closure_sha256": H5_ATTEMPT02_FAILURE_CLOSURE_SHA256,
        "job_id": "49880933",
        "source_commit": H5_ATTEMPT02_FAILED_IMPLEMENTATION_COMMIT,
        "source_tree": H5_ATTEMPT02_FAILED_IMPLEMENTATION_TREE,
    }:
        raise ValueError("R18 H5 harness-amendment-R2 failed-attempt binding drift")
    if (
        failure.get("artifact") != "qwen35_r18_h5_attempt02_failure_closure"
        or failure.get("status") != "failed_retained_no_H6_or_scientific_successor"
        or failure.get("authoritative_disposition") != "failed_nccl_or_rank_completion"
        or failure.get("attempt", {}).get("job_id") != "49880933"
        or failure.get("attempt", {}).get("source_commit") != H5_ATTEMPT02_FAILED_IMPLEMENTATION_COMMIT
        or failure.get("no_retroactive_pass") is not True
        or failure.get("scientific_training_authorized") is not False
    ):
        raise ValueError("R18 H5 attempt-02 failure-closure disposition drift")
    diagnostic_binding = value.get("reload_type_diagnostic", {})
    expected_set_paths = ["$.mismatched_keys", "$.missing_keys", "$.unexpected_keys"]
    if diagnostic_binding != {
        "job_id": "49881614",
        "report_path": (
            "artifacts/qwen35_hardware_qualification_20260718/"
            "r18_h5_attempt02_reload_type_diagnostic_49881614/loading_info_type_report.json"
        ),
        "report_sha256": H5_ATTEMPT02_RELOAD_TYPE_DIAGNOSTIC_SHA256,
        "set_paths": expected_set_paths,
        "status": "confirmed_set_serialization_source",
    }:
        raise ValueError("R18 H5 harness-amendment-R2 reload-diagnostic binding drift")
    if (
        diagnostic.get("artifact") != "qwen35_r18_h5_attempt02_checkpoint_reload_loading_info_type_diagnostic"
        or diagnostic.get("source_commit") != H5_ATTEMPT02_FAILED_IMPLEMENTATION_COMMIT
        or diagnostic.get("status") != "confirmed_set_serialization_source"
        or diagnostic.get("set_paths") != expected_set_paths
    ):
        raise ValueError("R18 H5 attempt-02 reload-type diagnostic content drift")
    if value.get("frozen_corrections") != {
        "checkpoint_loading_info": {
            "accepted_container_types": ["frozenset", "list", "set", "tuple"],
            "duplicate_entries_rejected": True,
            "element_type": "str",
            "nonempty_remains_strict_reload_failure": True,
            "normalized_json_type": "sorted_array",
        },
        "exact_metrics_window_barrier": {
            "existing_helper": "distributed_barrier_on_local_cuda_device",
            "number_or_placement_change_allowed": False,
            "source_location_at_failed_commit": "scripts/train/qwen35/train_qwen35_sft.py:381",
        },
        "trainer_barrier_ast_policy": "every torch.distributed.barrier call has device_ids keyword",
    }:
        raise ValueError("R18 H5 harness-amendment-R2 frozen-correction drift")
    if (
        value.get("scientific_training_authorized") is not False
        or value.get("automatic_successor") is not False
        or value.get("successor_on_complete_independent_pass") != "H6_only"
        or value.get("full_h5_rerun_required") is not True
        or value.get("model_loss_data_schedule_optimizer_precision_or_threshold_change_allowed") is not False
    ):
        raise ValueError("R18 H5 harness-amendment-R2 authority boundary drift")
    return value, digest


def validate_h5_source_delta(
    repository: Path,
    *,
    expected_head: str | None = None,
    harness_amendment_path: Path | None = None,
    harness_human_amendment_path: Path | None = None,
    attempt01_failure_closure_path: Path | None = None,
    harness_amendment_r2_path: Path | None = None,
    harness_human_amendment_r2_path: Path | None = None,
    attempt02_failure_closure_path: Path | None = None,
    reload_type_diagnostic_path: Path | None = None,
) -> dict[str, Any]:
    repository = repository.resolve()

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()

    head = git("rev-parse", "HEAD")
    if expected_head is not None and head != expected_head:
        raise ValueError(f"R18 H5 implementation HEAD drift: {head} != {expected_head}")
    if git("status", "--porcelain"):
        raise ValueError("R18 H5 implementation repository is not clean")
    amendment_r1_paths = (harness_amendment_path, harness_human_amendment_path, attempt01_failure_closure_path)
    amendment_r2_paths = (
        harness_amendment_r2_path,
        harness_human_amendment_r2_path,
        attempt02_failure_closure_path,
        reload_type_diagnostic_path,
    )
    use_amendment = any(path is not None for path in amendment_r1_paths)
    use_amendment_r2 = any(path is not None for path in amendment_r2_paths)
    if use_amendment and not all(path is not None for path in amendment_r1_paths):
        raise ValueError("R18 H5 source-delta amendment paths must be supplied together")
    if use_amendment_r2 and not all(path is not None for path in amendment_r2_paths):
        raise ValueError("R18 H5 source-delta amendment-R2 paths must be supplied together")
    if use_amendment_r2 and not use_amendment:
        raise ValueError("R18 H5 source-delta amendment-R2 requires the complete amendment-R1 binding")
    baseline = H5_PREREGISTRATION_COMMIT
    amendment_sha256 = None
    amendment_r2_sha256 = None
    if use_amendment_r2:
        _, amendment_sha256 = load_h5_harness_amendment(
            harness_amendment_path,
            human_amendment_path=harness_human_amendment_path,
            attempt01_failure_closure_path=attempt01_failure_closure_path,
        )
        amendment_r2, amendment_r2_sha256 = load_h5_harness_amendment_r2(
            harness_amendment_r2_path,
            human_amendment_path=harness_human_amendment_r2_path,
            attempt02_failure_closure_path=attempt02_failure_closure_path,
            reload_type_diagnostic_path=reload_type_diagnostic_path,
        )
        baseline = H5_HARNESS_AMENDMENT_R2_PREREGISTRATION_COMMIT
        allowed = set(amendment_r2["source"]["allowed_paths_from_amendment_preregistration_commit"])
    elif use_amendment:
        amendment, amendment_sha256 = load_h5_harness_amendment(
            harness_amendment_path,
            human_amendment_path=harness_human_amendment_path,
            attempt01_failure_closure_path=attempt01_failure_closure_path,
        )
        baseline = H5_HARNESS_AMENDMENT_PREREGISTRATION_COMMIT
        allowed = set(amendment["source"]["allowed_paths_from_amendment_preregistration_commit"])
    else:
        contract_path = repository / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h5.json"
        contract = load_strict_json(contract_path)
        allowed = set(contract["source"]["allowed_h5_implementation_paths"])
    if git("merge-base", "--is-ancestor", baseline, head) != "":
        raise AssertionError("unexpected git merge-base output")
    contract_path = repository / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h5.json"
    observed = set(filter(None, git("diff", "--name-only", f"{baseline}..{head}").splitlines()))
    if not observed:
        raise ValueError("R18 H5 implementation source delta is empty")
    if not observed <= allowed:
        raise ValueError(f"R18 H5 implementation changed forbidden paths: {sorted(observed - allowed)}")
    unexpected_contract_change = "scripts/train/qwen35/qwen35_hardware_qualification_r18_h5.json" in observed
    if unexpected_contract_change or sha256_file(contract_path) != H5_CONTRACT_SHA256:
        raise ValueError("R18 H5 implementation changed its preregistered contract")
    return {
        "allowed_paths": sorted(allowed),
        "head": head,
        "harness_amendment_sha256": amendment_sha256,
        "harness_amendment_r2_sha256": amendment_r2_sha256,
        "observed_changed_paths": sorted(observed),
        "preregistration_commit": baseline,
        "status": "passed",
    }
