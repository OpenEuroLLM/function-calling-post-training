#!/usr/bin/env python3
"""Produce or independently recompute the preregistered R18 H6 r2 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from scripts.train.qwen35.compare_qwen35_checkpoints import compare_checkpoints
from scripts.train.qwen35.validate_qwen35_h6_r18 import (
    FLOP_FORMULA_SHA256,
    PROCESS_FAILURE_PATTERN,
    SELECTED_LOSS_IMPLEMENTATION,
    expected_learning_rates,
    require_file,
    validate_metrics,
    verify_code_manifest,
)

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, require_finite_json, sha256_file
from open_instruct.qwen35_qualification_r18_h5 import canonical_json_bytes
from open_instruct.qwen35_qualification_r18_h6 import (
    H6_MODEL_MANIFEST_SHA256,
    H6_QUALIFICATION_MANIFEST_SHA256,
    H6_RUNTIME_REPORT_SHA256,
    H6_SCHEDULE_ENTRIES_SHA256,
    H6_SCHEDULE_FILE_SHA256,
    H6_SCHEDULE_SHA256,
)
from open_instruct.qwen35_qualification_r18_h6_r2 import (
    H6_R2_EXPECTED_TARGETS_BY_UPDATE,
    load_h6_r2_contract,
    validate_h6_r2_source_delta,
)
from open_instruct.qwen35_training import write_json_atomic

STRICT_RESUME_FAILURE_PATTERN = re.compile(
    PROCESS_FAILURE_PATTERN.pattern
    + r"|Some weights of .* were not initialized|missing_keys|unexpected_keys|strict Qwen3\.5 checkpoint restoration failed",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("producer", "independent"), required=True)
    parser.add_argument("--h6-r2-contract", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--h5-final-closure", type=Path, required=True)
    parser.add_argument("--r1-failure-closure", type=Path, required=True)
    parser.add_argument("--r1-failed-comparison", type=Path, required=True)
    parser.add_argument("--preregistration-closure", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--source-code-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--numpy-data", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--hardware-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-comparison", type=Path, required=True)
    parser.add_argument("--producer-validation", type=Path)
    parser.add_argument("--slurm-record", type=Path)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def validate_run_r2(
    root: Path, *, initial_step: int, resume_checkpoint: Path | None, expected_numpy_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Validate one common-prefix run without inheriting r1's H5-prefix assumptions."""

    run_manifest_path = root / "qwen35_run_manifest.json"
    update_probe_path = root / "qwen35_parameter_update_probe.json"
    for path in (run_manifest_path, update_probe_path):
        require_file(path)
    run = load_strict_json(run_manifest_path)
    update = load_strict_json(update_probe_path)
    if (
        run.get("model_class") != "Qwen3_5ForCausalLM"
        or run.get("model_config_type") != "qwen3_5_text"
        or run.get("model_revision") != "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
        or run.get("model_parameter_count") != 752_393_024
        or run.get("trainable_parameter_count") != 752_393_024
        or run.get("model_text_vocab_size") != 248_320
        or run.get("vision_tower_loaded") is not False
        or run.get("world_size") != 4
        or run.get("gradient_accumulation_steps") != 2
        or run.get("sequence_length") != 32_768
        or run.get("effective_tokens_per_optimizer_step") != 262_144
        or run.get("drop_last") is not False
        or run.get("numpy_contract_version") != "open-instruct-qwen35-numpy-v2"
        or run.get("conditional_checkpoint_conversion") != "strict_direct_to_Qwen3_5ForCausalLM"
        or run.get("frozen_data_validation", {}).get("arm_id") != "C00"
        or run.get("frozen_data_validation", {}).get("suite_id") != "v3-semantic-causal-suite-r1-core-frozen"
        or run.get("frozen_data_validation", {}).get("renderer") != "qwen35_native_tools"
        or run.get("numpy_manifest", {}).get("arm_id") != "C00"
        or run.get("numpy_manifest", {}).get("enable_thinking") is not False
        or run.get("numpy_manifest", {}).get("max_seq_length") != 32_768
        or run.get("schedule_validation", {}).get("schedule_sha256") != H6_SCHEDULE_SHA256
        or run.get("schedule_validation", {}).get("entries_sha256") != H6_SCHEDULE_ENTRIES_SHA256
        or run.get("schedule_validation", {}).get("scheduled_pack_count") != 80
        or run.get("schedule_validation", {}).get("synthetic_all_masked_pack_count") != 0
    ):
        raise ValueError(f"H6 r2 run identity drift in {root}")
    if run.get("numpy_manifest") != expected_numpy_manifest:
        raise ValueError(f"H6 r2 embedded NumPy-manifest logical drift in {root}")
    if run.get("precision_policy") != {
        "adamw_moments": "FP32",
        "forward_backward_autocast": "BF16",
        "gradients": "FP32",
        "parameters": "FP32",
    } or run.get("precision_validation") != {
        "parameter_dtype": "torch.float32",
        "trainable_parameter_tensors": 320,
        "trainable_parameters": 752_393_024,
    }:
        raise ValueError(f"H6 r2 precision-policy drift in {root}")
    hardware = run.get("hardware_qualification", {})
    if (
        hardware.get("manifest_sha256") != H6_QUALIFICATION_MANIFEST_SHA256
        or hardware.get("protocol_id") != "qwen35-hardware-qualification-r18"
        or hardware.get("require_no_dense_logits") is not True
        or hardware.get("hardware_profile") is not False
        or hardware.get("cuda_event_step_timing") is not False
    ):
        raise ValueError(f"H6 r2 hardware-qualification manifest drift in {root}")
    selective = run.get("selective_output_projection", {})
    if (
        selective.get("enabled") is not True
        or selective.get("implementation") != SELECTED_LOSS_IMPLEMENTATION
        or selective.get("chunk_size") != 512
        or selective.get("liger_status") != "abandoned_after_r17"
    ):
        raise ValueError(f"H6 r2 selected-output implementation drift in {root}")
    formula = run.get("flop_formula", {})
    if formula.get("formula_sha256") != FLOP_FORMULA_SHA256:
        raise ValueError(f"H6 r2 FLOP-formula drift in {root}")
    args = run.get("training_arguments", {})
    expected_arguments = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "cuda_event_step_timing": False,
        "data_seed": 3407,
        "dataloader_drop_last": False,
        "dataloader_num_workers": 0,
        "expected_final_global_step": 10,
        "expected_initial_global_step": initial_step,
        "full_determinism": False,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "hardware_profile": False,
        "ignore_data_skip": False,
        "learning_rate": 2e-5,
        "max_grad_norm": 1.0,
        "max_steps": 10,
        "optim": "adamw_torch_fused",
        "per_device_train_batch_size": 1,
        "require_forward_loss_audit": True,
        "require_no_dense_logits": True,
        "save_steps": 5,
        "save_total_limit": 2,
        "seed": 3407,
        "stop_after_steps": None,
        "train_sampling_strategy": "sequential",
        "use_liger_kernel": False,
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,
    }
    for key, expected in expected_arguments.items():
        if args.get(key) != expected:
            raise ValueError(f"H6 r2 training argument drift for {key} in {root}")
    observed_resume = args.get("resume_from_checkpoint")
    if resume_checkpoint is None:
        if observed_resume is not None:
            raise ValueError("H6 r2 continuous path unexpectedly resumed a checkpoint")
    elif not isinstance(observed_resume, str) or Path(observed_resume).resolve() != resume_checkpoint.resolve():
        raise ValueError("H6 r2 resumed path checkpoint identity drift")
    if (
        update.get("status") != "passed"
        or update.get("observed_initial_global_step") != initial_step
        or update.get("final_global_step") != 10
        or update.get("optimizer_steps_observed") != 10 - initial_step
    ):
        raise ValueError(f"H6 r2 parameter-update probe step drift in {root}")
    expected_steps = list(range(initial_step + 1, 11))
    for label in ("finite_losses", "finite_gradient_norms"):
        rows = update.get(label)
        if (
            not isinstance(rows, list)
            or [row.get("step") for row in rows] != expected_steps
            or any(
                isinstance(row.get("value"), bool)
                or not isinstance(row.get("value"), (int, float))
                or not math.isfinite(row["value"])
                for row in rows
            )
        ):
            raise ValueError(f"H6 r2 update-probe {label} drift in {root}")
    comparison = update.get("parameter_comparison", {})
    if (
        comparison.get("changed_sampled_values", 0) <= 0
        or comparison.get("max_absolute_delta", 0) <= 0
        or comparison.get("initial_values_sha256") == comparison.get("final_values_sha256")
    ):
        raise ValueError(f"H6 r2 update-probe parameter-change evidence drift in {root}")
    expected_checkpoint_names = {"checkpoint-5", "checkpoint-10"} if initial_step == 0 else {"checkpoint-10"}
    checkpoint_names = {path.name for path in root.glob("checkpoint-*") if path.is_dir()}
    if checkpoint_names != expected_checkpoint_names:
        raise ValueError(
            f"H6 r2 checkpoint-set drift in {root}: {sorted(checkpoint_names)} != {sorted(expected_checkpoint_names)}"
        )
    return {
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "update_probe_sha256": sha256_file(update_probe_path),
        "checkpoint_names": sorted(checkpoint_names),
    }


def validate_strict_load_audits(resumed_root: Path, checkpoint: Path) -> dict[str, Any]:
    """Require four identical rank-local strict-load audits for the common checkpoint."""

    failures = sorted(resumed_root.glob("qwen35_resume_model_load_failure_rank_*.json"))
    if failures:
        raise ValueError(f"H6 r2 has strict-load failure artifacts: {[path.name for path in failures]}")
    paths = [resumed_root / f"qwen35_resume_model_load_audit_rank_{rank}.json" for rank in range(4)]
    audits = []
    for rank, path in enumerate(paths):
        require_file(path)
        audit = load_strict_json(path)
        if (
            audit.get("artifact") != "qwen35_strict_trainer_checkpoint_load_audit"
            or audit.get("schema_version") != 1
            or audit.get("status") != "passed"
            or audit.get("rank") != rank
            or audit.get("all_ranks_passed") is not True
            or audit.get("optimizer_execution_authorized") is not True
            or Path(audit.get("checkpoint_dir", "")).resolve() != checkpoint.resolve()
            or audit.get("layout") != "single_model_safetensors"
            or audit.get("source_tensor_count") != 320
            or audit.get("target_state_key_count") != 321
            or audit.get("unique_target_storage_count") != 320
            or audit.get("copied_source_tensor_count") != 320
            or audit.get("copied_unique_elements") != 752_393_024
            or audit.get("source_dtype") != "F32"
            or audit.get("target_dtype") != "torch.float32"
            or audit.get("tied_input_output_embeddings_before") is not True
            or audit.get("tied_input_output_embeddings_after") is not True
            or audit.get("parameter_objects_preserved") is not True
            or audit.get("storage_pointers_preserved") is not True
            or audit.get("missing_source_keys") != []
            or audit.get("unexpected_source_keys") != []
            or audit.get("upstream_trainer_strict_false_used") is not False
            or audit.get("metadata_preflight_completed_before_copy") is not True
            or audit.get("exact_post_copy_values") is not True
        ):
            raise ValueError(f"H6 r2 strict-load audit semantic drift for rank {rank}")
        weight = audit.get("weight_files", {}).get("model.safetensors", {})
        if (
            weight.get("sha256") != sha256_file(checkpoint / "model.safetensors")
            or weight.get("size") != (checkpoint / "model.safetensors").stat().st_size
            or audit.get("config_sha256") != sha256_file(checkpoint / "config.json")
        ):
            raise ValueError(f"H6 r2 strict-load audit checkpoint identity drift for rank {rank}")
        require_finite_json(audit, context=f"H6.r2.strict_load.rank{rank}")
        audits.append(audit)
    projections = []
    for audit in audits:
        projection = dict(audit)
        del projection["rank"]
        projections.append(projection)
    if any(projection != projections[0] for projection in projections[1:]):
        raise ValueError("H6 r2 rank-local strict-load audits differ beyond rank identity")
    return {
        "audit_sha256": {str(rank): sha256_file(path) for rank, path in enumerate(paths)},
        "common_projection_sha256": hashlib.sha256(canonical_json_bytes(projections[0])).hexdigest(),
        "checkpoint_identity_sha256": audits[0]["checkpoint_identity_sha256"],
        "rank_count": 4,
        "status": "passed",
    }


def load_context(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any]]:
    contract, contract_sha = load_h6_r2_contract(
        args.h6_r2_contract,
        human_protocol_path=args.human_protocol,
        h5_final_closure_path=args.h5_final_closure,
        r1_failure_closure_path=args.r1_failure_closure,
        r1_failed_comparison_path=args.r1_failed_comparison,
        preregistration_closure_path=args.preregistration_closure,
    )
    for path in (
        args.source_code_manifest,
        args.schedule,
        args.runtime_report,
        args.model_manifest,
        args.hardware_inventory,
    ):
        require_file(path)
    if sha256_file(args.schedule) != H6_SCHEDULE_FILE_SHA256:
        raise ValueError("H6 r2 schedule file digest drift")
    schedule = load_strict_json(args.schedule)
    entries = schedule.get("entries")
    if (
        schedule.get("schedule_sha256") != H6_SCHEDULE_SHA256
        or schedule.get("entries_sha256") != H6_SCHEDULE_ENTRIES_SHA256
        or not isinstance(entries, list)
        or len(entries) != 80
        or [entry.get("schedule_index") for entry in entries] != list(range(80))
        or len({entry.get("pack_uid") for entry in entries}) != 80
        or len({entry.get("pack_index") for entry in entries}) != 80
        or any(entry.get("synthetic") is not False for entry in entries)
        or [sum(entry["assistant_targets"] for entry in entries[index : index + 8]) for index in range(0, 80, 8)]
        != list(H6_R2_EXPECTED_TARGETS_BY_UPDATE)
    ):
        raise ValueError("H6 r2 schedule semantic identity drift")
    numpy_manifest = args.numpy_data / "manifest.json"
    require_file(numpy_manifest)
    if sha256_file(numpy_manifest) != contract["model_and_data"]["numpy_manifest_sha256"]:
        raise ValueError("H6 r2 C00 NumPy manifest drift")
    if sha256_file(args.runtime_report) != H6_RUNTIME_REPORT_SHA256:
        raise ValueError("H6 r2 pinned runtime-report digest drift")
    if load_strict_json(args.runtime_report).get("status") != "passed":
        raise ValueError("H6 r2 runtime report did not pass")
    if sha256_file(args.model_manifest) != H6_MODEL_MANIFEST_SHA256:
        raise ValueError("H6 r2 pinned model-manifest digest drift")
    inventory = load_strict_json(args.hardware_inventory)
    gpu_properties = inventory.get("gpu_properties")
    if (
        inventory.get("status") != "passed"
        or inventory.get("qualification_manifest_sha256") != H6_QUALIFICATION_MANIFEST_SHA256
        or inventory.get("source", {}).get("head") != args.expected_source_head
        or not isinstance(gpu_properties, list)
        or len(gpu_properties) != 4
        or any(
            row.get("name") != "NVIDIA A100-SXM-64GB"
            or row.get("total_memory_bytes") != 68_099_571_712
            or [row.get("major"), row.get("minor")] != [8, 0]
            for row in gpu_properties
        )
    ):
        raise ValueError("H6 r2 hardware-inventory identity drift")
    identities = inventory.get("identity_files", {})
    for path in (args.source_code_manifest, args.model_manifest, args.numpy_data / "manifest.json", args.schedule):
        row = identities.get(str(path.resolve()), {})
        if row != {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}:
            raise ValueError(f"H6 r2 captured identity-file drift: {path}")
    return contract, contract_sha, schedule


def producer(args: argparse.Namespace) -> dict[str, Any]:
    _, contract_sha, schedule = load_context(args)
    source_delta = validate_h6_r2_source_delta(args.source_repository, expected_head=args.expected_source_head)
    verify_code_manifest(args.source_repository, args.source_code_manifest)
    continuous = args.output_dir / "continuous"
    resumed = args.output_dir / "resumed"
    resume_checkpoint = continuous / "checkpoint-5"
    expected_numpy_manifest = load_strict_json(args.numpy_data / "manifest.json")
    run_evidence = {
        "continuous": validate_run_r2(
            continuous, initial_step=0, resume_checkpoint=None, expected_numpy_manifest=expected_numpy_manifest
        ),
        "resumed": validate_run_r2(
            resumed,
            initial_step=5,
            resume_checkpoint=resume_checkpoint,
            expected_numpy_manifest=expected_numpy_manifest,
        ),
    }
    strict_load_audits = validate_strict_load_audits(resumed, resume_checkpoint)
    continuous_metrics = validate_metrics(
        continuous / "qwen35_exact_metrics.jsonl",
        continuous / "qwen35_exact_metrics_summary.json",
        first_step=1,
        last_step=10,
        schedule_entries=schedule["entries"],
    )
    resumed_metrics = validate_metrics(
        resumed / "qwen35_exact_metrics.jsonl",
        resumed / "qwen35_exact_metrics_summary.json",
        first_step=6,
        last_step=10,
        schedule_entries=schedule["entries"],
    )
    reconstructed = continuous_metrics["projections"][:5] + resumed_metrics["projections"]
    if continuous_metrics["projections"] != reconstructed:
        for index, (left, right) in enumerate(
            zip(continuous_metrics["projections"], reconstructed, strict=True), start=1
        ):
            if left != right:
                raise AssertionError(f"H6 r2 deterministic metric projection first differs at step {index}")
        raise AssertionError("H6 r2 deterministic metric projection differs")
    for step, record in enumerate(continuous_metrics["records"], start=1):
        expected_lr = expected_learning_rates()[step - 1]
        if record.get("optimizer", {}).get("learning_rate") != expected_lr:
            raise ValueError(f"H6 r2 learning-rate drift at step {step}")

    require_file(args.checkpoint_comparison)
    observed_comparison = load_strict_json(args.checkpoint_comparison)
    recomputed_comparison = compare_checkpoints(
        continuous / "checkpoint-10", resumed / "checkpoint-10", atol=0.0, rtol=0.0
    )
    if observed_comparison != recomputed_comparison:
        raise ValueError("H6 r2 checkpoint-comparison report differs from recomputation")
    if (
        recomputed_comparison.get("status") != "passed"
        or recomputed_comparison.get("model", {}).get("bit_exact") is not True
        or recomputed_comparison.get("optimizer", {}).get("bit_exact_tensors") is not True
        or recomputed_comparison.get("scheduler", {}).get("bit_exact_tensors") is not True
        or recomputed_comparison.get("trainer_state", {}).get("global_step") != 10
        or len(recomputed_comparison.get("rng", {})) != 4
        or any(not row.get("bit_exact_tensors") for row in recomputed_comparison.get("rng", {}).values())
    ):
        raise ValueError("H6 r2 checkpoint comparison did not prove complete bit equality")
    projection_sha = hashlib.sha256(canonical_json_bytes(continuous_metrics["projections"])).hexdigest()
    return {
        "artifact": "qwen35_r18_h6_r2_producer_validation",
        "checkpoint_comparison_sha256": sha256_file(args.checkpoint_comparison),
        "common_prefix_checkpoint": str(resume_checkpoint.resolve()),
        "contract_sha256": contract_sha,
        "deterministic_metric_projection_sha256": projection_sha,
        "hardware_inventory_sha256": sha256_file(args.hardware_inventory),
        "immutable_predecessor_sha256": {
            "h5_final_closure": sha256_file(args.h5_final_closure),
            "r1_failed_comparison": sha256_file(args.r1_failed_comparison),
            "r1_failure_closure": sha256_file(args.r1_failure_closure),
        },
        "metric_evidence": {
            "continuous": {
                key: value for key, value in continuous_metrics.items() if key not in {"records", "projections"}
            },
            "resumed_suffix": {
                key: value for key, value in resumed_metrics.items() if key not in {"records", "projections"}
            },
        },
        "run_evidence": run_evidence,
        "schema_version": 1,
        "scientific_training_authorized": False,
        "source_delta": source_delta,
        "status": "producer_passed_pending_slurm_and_independent_closure",
        "strict_load_audits": strict_load_audits,
        "successor_authorized": None,
    }


def independent(args: argparse.Namespace) -> dict[str, Any]:
    if args.producer_validation is None or args.slurm_record is None:
        raise ValueError("H6 r2 independent mode requires producer validation and Slurm record")
    recomputed = producer(args)
    producer_report = load_strict_json(args.producer_validation)
    if producer_report != recomputed:
        raise ValueError("H6 r2 producer report differs from independent recomputation")
    slurm = load_strict_json(args.slurm_record)
    required = {
        "account": "aifac_f02_434",
        "alloc_gpus": 4,
        "exit_code": "0:0",
        "nodes": 1,
        "requeued": False,
        "state": "COMPLETED",
    }
    for key, expected in required.items():
        if slurm.get(key) != expected:
            raise ValueError(f"H6 r2 Slurm completion drift for {key}")
    elapsed_seconds = slurm.get("elapsed_seconds")
    if (
        slurm.get("partition") != "boost_usr_prod"
        or slurm.get("qos") != "normal"
        or isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not 0 < elapsed_seconds <= 3_600
    ):
        raise ValueError("H6 r2 Slurm resource-ceiling or partition/QOS drift")
    personal_root = Path("/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3")
    for label in ("stdout", "stderr"):
        path = Path(slurm[f"{label}_path"])
        if not path.resolve().is_relative_to(personal_root):
            raise ValueError(f"H6 r2 Slurm {label} escaped the personal work root")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"H6 r2 Slurm {label} is absent or symlinked")
        if label == "stdout" and path.stat().st_size <= 0:
            raise FileNotFoundError("H6 r2 Slurm stdout is empty")
        if slurm.get(f"{label}_sha256") != sha256_file(path):
            raise ValueError(f"H6 r2 Slurm {label} digest drift")
        contents = path.read_text(errors="replace")
        match = STRICT_RESUME_FAILURE_PATTERN.search(contents)
        if match:
            raise AssertionError(f"H6 r2 Slurm {label} contains a failure marker: {match.group(0)!r}")
        if label == "stdout" and "R18_H6_R2_PRODUCER_PASSED_PENDING_INDEPENDENT_CLOSURE" not in contents:
            raise ValueError("H6 r2 Slurm stdout lacks the terminal producer marker")
    exit_report = load_strict_json(args.output_dir / "g2_job_exit.json")
    if exit_report != {"exit_code": 0, "slurm_job_id": str(slurm["job_id"])}:
        raise ValueError("H6 r2 wrapper exit record drift")
    return {
        "artifact": "qwen35_r18_h6_r2_independent_closure",
        "contract_sha256": recomputed["contract_sha256"],
        "producer_validation_sha256": sha256_file(args.producer_validation),
        "schema_version": 1,
        "scientific_training_authorized": False,
        "slurm_record_sha256": sha256_file(args.slurm_record),
        "status": "passed_H7_only_authorized",
        "successor_authorized": "H7_only",
    }


def write_failure(path: Path, *, mode: str, error: BaseException) -> None:
    if path.exists() or path.is_symlink():
        return
    write_json_atomic(
        path,
        {
            "artifact": "qwen35_r18_h6_r2_validation_failure",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "mode": mode,
            "schema_version": 1,
            "scientific_training_authorized": False,
            "status": "failed",
        },
    )


def main() -> int:
    args = parse_args()
    if args.report_output.exists() or args.report_output.is_symlink():
        raise FileExistsError(args.report_output)
    try:
        report = producer(args) if args.mode == "producer" else independent(args)
        require_finite_json(report, context=f"H6.r2.{args.mode}.report")
        write_json_atomic(args.report_output, report)
    except BaseException as error:
        write_failure(args.report_output, mode=args.mode, error=error)
        raise
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
