#!/usr/bin/env python3
"""Produce or independently recompute the preregistered R18 H6 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.train.qwen35.compare_qwen35_checkpoints import compare_checkpoints

from open_instruct.qwen35_qualification_r18_h4 import (
    load_strict_json,
    load_strict_jsonl,
    require_finite_json,
    sha256_file,
)
from open_instruct.qwen35_qualification_r18_h5 import canonical_json_bytes
from open_instruct.qwen35_qualification_r18_h6 import (
    H6_EXPECTED_TARGETS_BY_UPDATE,
    H6_H5_CHECKPOINT_MANIFEST_SHA256,
    H6_H5_CHECKPOINT_RELOAD_SHA256,
    H6_H5_INDEPENDENT_CLOSURE_SHA256,
    H6_H5_METRICS_SHA256,
    H6_H5_MODEL_SHA256,
    H6_MODEL_MANIFEST_SHA256,
    H6_QUALIFICATION_MANIFEST_SHA256,
    H6_RUNTIME_REPORT_SHA256,
    H6_SCHEDULE_ENTRIES_SHA256,
    H6_SCHEDULE_FILE_SHA256,
    H6_SCHEDULE_SHA256,
    load_h6_contract,
    validate_h6_source_delta,
)
from open_instruct.qwen35_training import write_json_atomic

PROCESS_FAILURE_PATTERN = re.compile(
    r"NCCL\s+(?:WARN|ERROR)|collective.*(?:timeout|abort)|nccl.*(?:timeout|abort|unhandled)"
    r"|destroy_process_group\(\) was not called"
    r"|barrier\(\): using the device under current context"
    r"|Guessing device ID based on global rank"
    r"|Object of type set is not JSON serializable",
    re.IGNORECASE,
)
SELECTED_LOSS_IMPLEMENTATION = "pytorch_nonreentrant_checkpointed_chunked_selected_rows_r1"
FLOP_FORMULA_SHA256 = "4ff82fa48f2e1501330b1d076a969421fb85f14e89602c2e8cb80c374a9b90c1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("producer", "independent"), required=True)
    parser.add_argument("--h6-contract", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--h5-final-closure", type=Path, required=True)
    parser.add_argument("--preregistration-closure", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--source-code-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--numpy-data", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--hardware-inventory", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-comparison", type=Path, required=True)
    parser.add_argument("--producer-validation", type=Path)
    parser.add_argument("--slurm-record", type=Path)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required nonempty regular H6 file is absent: {path}")


def verify_code_manifest(repository: Path, manifest: Path) -> None:
    require_file(manifest)
    subprocess.run(
        ["sha256sum", "--check", "--strict", str(manifest.resolve())],
        cwd=repository.resolve(),
        check=True,
        stdout=subprocess.DEVNULL,
    )


def validate_checkpoint_file_manifest(checkpoint: Path, report_path: Path) -> dict[str, Any]:
    """Rebind every accepted H5 checkpoint file before and after resumption."""

    require_file(report_path)
    if sha256_file(report_path) != H6_H5_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("H6 accepted H5 checkpoint-manifest digest drift")
    report = load_strict_json(report_path)
    rows = report.get("files")
    if (
        report.get("artifact") != "qwen35_r18_h5_checkpoint_file_manifest"
        or report.get("status") != "passed"
        or not isinstance(rows, list)
        or len(rows) != 14
    ):
        raise ValueError("H6 accepted H5 checkpoint-manifest structure drift")
    observed_names = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("H6 checkpoint manifest file row must be a mapping")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).name != relative
            or Path(relative).is_absolute()
        ):
            raise ValueError("H6 checkpoint manifest contains an unsafe relative path")
        path = checkpoint / relative
        require_file(path)
        if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"H6 accepted H5 checkpoint file drift: {relative}")
        observed_names.append(relative)
    actual_names = sorted(path.name for path in checkpoint.iterdir())
    if sorted(observed_names) != actual_names or len(observed_names) != len(set(observed_names)):
        raise ValueError("H6 accepted H5 checkpoint file set drift")
    return {"file_count": len(rows), "manifest_sha256": sha256_file(report_path), "status": "passed"}


def deterministic_metric_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Retain training semantics while excluding observational time/memory fields."""

    analytic = record.get("analytic_flops", {})
    return {
        "analytic_flops": {
            "components": analytic.get("components"),
            "formula_sha256": analytic.get("formula_sha256"),
            "formula_version": analytic.get("formula_version"),
            "isolated_causal_attention_pairs": analytic.get("isolated_causal_attention_pairs"),
        },
        "artifact": record.get("artifact"),
        "counts": record.get("counts"),
        "gradient_dtype_audits": record.get("gradient_dtype_audits"),
        "loss": record.get("loss"),
        "optimizer": record.get("optimizer"),
        "optimizer_updates": record.get("optimizer_updates"),
        "pack_uids": record.get("pack_uids"),
        "schedule_indices": record.get("schedule_indices"),
        "schedule_sha256": record.get("schedule_sha256"),
        "schema_version": record.get("schema_version"),
        "selected_output_audits": record.get("selected_output_audits"),
        "step": record.get("step"),
        "synchronized_timing": record.get("synchronized_timing"),
        "window_start_step": record.get("window_start_step"),
        "world_size": record.get("world_size"),
    }


def validate_metrics(
    metrics_path: Path,
    summary_path: Path,
    *,
    first_step: int,
    last_step: int,
    schedule_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    require_file(metrics_path)
    require_file(summary_path)
    records = load_strict_jsonl(metrics_path)
    summary = load_strict_json(summary_path)
    expected_steps = list(range(first_step, last_step + 1))
    if [record.get("step") for record in records] != expected_steps:
        raise ValueError(f"H6 metrics step sequence drift for {metrics_path}")
    all_indices: list[int] = []
    all_uids: list[str] = []
    for record in records:
        step = int(record["step"])
        require_finite_json(record, context=f"H6.metrics.step{step}")
        expected_indices = list(range((step - 1) * 8, step * 8))
        expected_entries = [schedule_entries[index] for index in expected_indices]
        if record.get("schedule_indices") != expected_indices:
            raise ValueError(f"H6 schedule-index exposure drift at step {step}")
        uids = record.get("pack_uids")
        expected_uids = [entry["pack_uid"] for entry in expected_entries]
        if uids != expected_uids:
            raise ValueError(f"H6 pack-UID exposure drift at step {step}")
        counts = record.get("counts", {})
        expected_target_count = H6_EXPECTED_TARGETS_BY_UPDATE[step - 1]
        expected_counts = {
            "assistant_targets": sum(entry["assistant_targets"] for entry in expected_entries),
            "attention_length_squared": sum(
                entry["attention_length_squared"] for entry in expected_entries
            ),
            "documents": sum(entry["document_count"] for entry in expected_entries),
            "fixed_tokens": 262_144,
            "packs": 8,
            "padding_tokens": sum(entry["padding_tokens"] for entry in expected_entries),
            "real_tokens": sum(entry["real_tokens"] for entry in expected_entries),
            "synthetic_packs": sum(bool(entry["synthetic"]) for entry in expected_entries),
        }
        if expected_counts["assistant_targets"] != expected_target_count or counts != expected_counts:
            raise ValueError(f"H6 exposure count drift at step {step}")
        if (
            record.get("schedule_sha256") != H6_SCHEDULE_SHA256
            or record.get("world_size") != 4
            or record.get("optimizer_updates") != 1
            or record.get("synchronized_timing") is not True
            or record.get("loss", {}).get("global_assistant_target_divisor") != expected_target_count
        ):
            raise ValueError(f"H6 metric identity/divisor drift at step {step}")
        loss = record.get("loss", {}).get("normalized_loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"H6 normalized loss is invalid at step {step}")
        audits = record.get("selected_output_audits")
        if not isinstance(audits, list) or len(audits) != 8:
            raise ValueError(f"H6 selected-output audit count drift at step {step}")
        for audit, index, uid, entry in zip(audits, expected_indices, uids, expected_entries, strict=True):
            if audit.get("schedule_index") != index or audit.get("pack_uid") != uid:
                raise ValueError(f"H6 selected-output audit identity drift at step {step}")
            detail = audit.get("audit", {})
            validation = audit.get("validation", {})
            selected_rows = entry["assistant_targets"]
            if (
                audit.get("loss_dtype") != "torch.float32"
                or detail.get("returned_dense_logits") is not False
                or detail.get("checkpointed") is not True
                or detail.get("implementation_id") != SELECTED_LOSS_IMPLEMENTATION
                or detail.get("chunk_size") != 512
                or detail.get("maximum_chunk_rows", 513) > 512
                or detail.get("hidden_size") != 1024
                or detail.get("vocabulary_size") != 248_320
                or detail.get("selected_rows") != selected_rows
                or detail.get("global_target_count") != expected_target_count
                or detail.get("zero_target") != (selected_rows == 0)
                or validation.get("status") != "passed"
                or validation.get("selected_rows") != selected_rows
                or validation.get("global_target_count") != expected_target_count
            ):
                raise ValueError(f"H6 selected-output audit semantic drift at step {step}")
        gradients = record.get("gradient_dtype_audits")
        if (
            not isinstance(gradients, list)
            or len(gradients) != 1
            or gradients[0].get("missing_gradient_count") != 0
            or gradients[0].get("gradient_dtype_counts")
            != {"torch.float32": gradients[0].get("gradient_tensor_count")}
        ):
            raise ValueError(f"H6 gradient dtype/connectivity drift at step {step}")
        optimizer = record.get("optimizer", {})
        expected_learning_rate = expected_learning_rates()[step - 1]
        if optimizer != {
            "applied_learning_rates": [expected_learning_rate],
            "first_applied_learning_rate": expected_learning_rate,
            "last_applied_learning_rate": expected_learning_rate,
            "learning_rate": expected_learning_rate,
        }:
            raise ValueError(f"H6 applied learning-rate drift at step {step}")
        analytic = record.get("analytic_flops", {})
        if (
            analytic.get("formula_sha256") != FLOP_FORMULA_SHA256
            or analytic.get("formula_version") != "qwen35-hybrid-causal-selected-output-v2"
            or analytic.get("isolated_causal_attention_pairs")
            != (expected_counts["attention_length_squared"] + expected_counts["fixed_tokens"]) // 2
            or not isinstance(analytic.get("components"), dict)
            or analytic["components"].get("total", 0) <= 0
        ):
            raise ValueError(f"H6 analytic-FLOP identity drift at step {step}")
        all_indices.extend(expected_indices)
        all_uids.extend(uids)
    if len(all_indices) != len(set(all_indices)) or len(all_uids) != len(set(all_uids)):
        raise ValueError("H6 metrics duplicate a schedule index or pack UID")
    if (
        summary.get("first_step") != first_step
        or summary.get("last_step") != last_step
        or summary.get("optimizer_steps") != len(expected_steps)
        or summary.get("schedule_sha256") != H6_SCHEDULE_SHA256
        or summary.get("world_size") != 4
        or summary.get("loaded_liger_modules") != []
    ):
        raise ValueError("H6 exact-metrics summary identity drift")
    projections = [deterministic_metric_projection(record) for record in records]
    return {
        "metrics_sha256": sha256_file(metrics_path),
        "projection_sha256": hashlib.sha256(canonical_json_bytes(projections)).hexdigest(),
        "projections": projections,
        "records": records,
        "summary_sha256": sha256_file(summary_path),
    }


def expected_learning_rates() -> list[float]:
    warmup_steps = math.ceil(10 * 0.03)
    values = []
    for step in range(10):
        if step < warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, 10 - warmup_steps)
            factor = max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
        values.append(2e-5 * factor)
    return values


def validate_run(
    root: Path,
    *,
    initial_step: int,
    resume_checkpoint: Path | None,
    expected_numpy_manifest: dict[str, Any],
) -> dict[str, Any]:
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
        or run.get("sequence_length") != 32768
        or run.get("effective_tokens_per_optimizer_step") != 262_144
        or run.get("drop_last") is not False
        or run.get("numpy_contract_version") != "open-instruct-qwen35-numpy-v2"
        or run.get("conditional_checkpoint_conversion") != "strict_direct_to_Qwen3_5ForCausalLM"
        or run.get("frozen_data_validation", {}).get("arm_id") != "C00"
        or run.get("frozen_data_validation", {}).get("suite_id")
        != "v3-semantic-causal-suite-r1-core-frozen"
        or run.get("frozen_data_validation", {}).get("renderer") != "qwen35_native_tools"
        or run.get("numpy_manifest", {}).get("arm_id") != "C00"
        or run.get("numpy_manifest", {}).get("enable_thinking") is not False
        or run.get("numpy_manifest", {}).get("max_seq_length") != 32768
        or run.get("schedule_validation", {}).get("schedule_sha256") != H6_SCHEDULE_SHA256
        or run.get("schedule_validation", {}).get("entries_sha256") != H6_SCHEDULE_ENTRIES_SHA256
        or run.get("schedule_validation", {}).get("scheduled_pack_count") != 80
        or run.get("schedule_validation", {}).get("synthetic_all_masked_pack_count") != 0
    ):
        raise ValueError(f"H6 run identity drift in {root}")
    if run.get("numpy_manifest") != expected_numpy_manifest:
        raise ValueError(f"H6 embedded NumPy-manifest logical drift in {root}")
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
        raise ValueError(f"H6 precision-policy drift in {root}")
    hardware = run.get("hardware_qualification", {})
    if (
        hardware.get("manifest_sha256") != H6_QUALIFICATION_MANIFEST_SHA256
        or hardware.get("protocol_id") != "qwen35-hardware-qualification-r18"
        or hardware.get("require_no_dense_logits") is not True
        or hardware.get("hardware_profile") is not False
        or hardware.get("cuda_event_step_timing") is not False
    ):
        raise ValueError(f"H6 hardware-qualification manifest drift in {root}")
    selective = run.get("selective_output_projection", {})
    if (
        selective.get("enabled") is not True
        or selective.get("chunk_size") != 512
        or selective.get("liger_status") != "abandoned_after_r17"
    ):
        raise ValueError(f"H6 selected-output implementation drift in {root}")
    args = run.get("training_arguments", {})
    expected = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "cuda_event_step_timing": False,
        "data_seed": 3407,
        "expected_final_global_step": 10,
        "expected_initial_global_step": initial_step,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "hardware_profile": False,
        "ignore_data_skip": False,
        "learning_rate": 2e-5,
        "max_grad_norm": 1.0,
        "max_steps": 10,
        "dataloader_drop_last": False,
        "dataloader_num_workers": 0,
        "full_determinism": False,
        "optim": "adamw_torch_fused",
        "per_device_train_batch_size": 1,
        "require_forward_loss_audit": True,
        "require_no_dense_logits": True,
        "save_steps": 10,
        "seed": 3407,
        "stop_after_steps": None,
        "train_sampling_strategy": "sequential",
        "use_liger_kernel": False,
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,
    }
    for key, expected_value in expected.items():
        if args.get(key) != expected_value:
            raise ValueError(f"H6 training argument drift for {key} in {root}")
    observed_resume = args.get("resume_from_checkpoint")
    if resume_checkpoint is None:
        if observed_resume is not None:
            raise ValueError("H6 continuous path unexpectedly resumed a checkpoint")
    elif not isinstance(observed_resume, str) or Path(observed_resume).resolve() != resume_checkpoint.resolve():
        raise ValueError("H6 resumed path checkpoint identity drift")
    if (
        update.get("status") != "passed"
        or update.get("observed_initial_global_step") != initial_step
        or update.get("final_global_step") != 10
        or update.get("optimizer_steps_observed") != 10 - initial_step
    ):
        raise ValueError(f"H6 parameter-update probe step drift in {root}")
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
            raise ValueError(f"H6 update-probe {label} drift in {root}")
    parameter_comparison = update.get("parameter_comparison", {})
    if (
        parameter_comparison.get("changed_sampled_values", 0) <= 0
        or parameter_comparison.get("max_absolute_delta", 0) <= 0
        or parameter_comparison.get("initial_values_sha256")
        == parameter_comparison.get("final_values_sha256")
    ):
        raise ValueError(f"H6 update-probe parameter-change evidence drift in {root}")
    checkpoint = root / "checkpoint-10"
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise FileNotFoundError(f"H6 checkpoint-10 is absent in {root}")
    return {
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "update_probe_sha256": sha256_file(update_probe_path),
    }


def load_context(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any]]:
    contract, contract_sha = load_h6_contract(
        args.h6_contract,
        human_protocol_path=args.human_protocol,
        h5_final_closure_path=args.h5_final_closure,
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
        raise ValueError("H6 schedule file digest drift")
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
        or sum(entry["assistant_targets"] for entry in entries) != 602_629
        or sum(entry["attention_length_squared"] for entry in entries) != 11_096_774_268
        or sum(entry["padding_tokens"] for entry in entries) != 819
        or sum(entry["real_tokens"] for entry in entries) != 2_620_621
    ):
        raise ValueError("H6 schedule semantic identity drift")
    numpy_manifest = args.numpy_data / "manifest.json"
    require_file(numpy_manifest)
    if sha256_file(numpy_manifest) != contract["model_and_data"]["numpy_manifest_sha256"]:
        raise ValueError("H6 C00 NumPy manifest drift")
    if sha256_file(args.runtime_report) != H6_RUNTIME_REPORT_SHA256:
        raise ValueError("H6 pinned runtime-report digest drift")
    if load_strict_json(args.runtime_report).get("status") != "passed":
        raise ValueError("H6 runtime report did not pass")
    if sha256_file(args.model_manifest) != H6_MODEL_MANIFEST_SHA256:
        raise ValueError("H6 pinned model-manifest digest drift")
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
        raise ValueError("H6 hardware-inventory identity drift")
    identities = inventory.get("identity_files", {})
    identity_paths = (
        args.source_code_manifest,
        args.model_manifest,
        args.numpy_data / "manifest.json",
        args.schedule,
    )
    for path in identity_paths:
        row = identities.get(str(path.resolve()), {})
        if row != {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}:
            raise ValueError(f"H6 captured identity-file drift: {path}")
    return contract, contract_sha, schedule


def producer(args: argparse.Namespace) -> dict[str, Any]:
    _, contract_sha, schedule = load_context(args)
    source_delta = validate_h6_source_delta(args.source_repository, expected_head=args.expected_source_head)
    verify_code_manifest(args.source_repository, args.source_code_manifest)

    h5_paths = {
        "checkpoint_manifest": args.h5_run / "h5_checkpoint_5_file_manifest.json",
        "checkpoint_reload": args.h5_run / "h5_checkpoint_5_reload_validation.json",
        "independent_closure": args.h5_run / "h5_independent_closure.json",
        "metrics": args.h5_run / "training/qwen35_exact_metrics.jsonl",
        "metrics_summary": args.h5_run / "training/qwen35_exact_metrics_summary.json",
        "model": args.h5_run / "training/checkpoint-5/model.safetensors",
    }
    for path in h5_paths.values():
        require_file(path)
    expected_h5_hashes = {
        "checkpoint_manifest": H6_H5_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_reload": H6_H5_CHECKPOINT_RELOAD_SHA256,
        "independent_closure": H6_H5_INDEPENDENT_CLOSURE_SHA256,
        "metrics": H6_H5_METRICS_SHA256,
        "model": H6_H5_MODEL_SHA256,
    }
    for label, expected in expected_h5_hashes.items():
        if sha256_file(h5_paths[label]) != expected:
            raise ValueError(f"H6 accepted H5 {label} identity drift")
    if load_strict_json(h5_paths["independent_closure"]).get("status") != "passed_H6_only_authorized":
        raise ValueError("H6 accepted H5 independent-closure status drift")

    continuous = args.output_dir / "continuous"
    resumed = args.output_dir / "resumed"
    h5_checkpoint = args.h5_run / "training/checkpoint-5"
    checkpoint_manifest_validation = validate_checkpoint_file_manifest(
        h5_checkpoint, h5_paths["checkpoint_manifest"]
    )
    expected_numpy_manifest = load_strict_json(args.numpy_data / "manifest.json")
    run_evidence = {
        "continuous": validate_run(
            continuous,
            initial_step=0,
            resume_checkpoint=None,
            expected_numpy_manifest=expected_numpy_manifest,
        ),
        "resumed": validate_run(
            resumed,
            initial_step=5,
            resume_checkpoint=h5_checkpoint,
            expected_numpy_manifest=expected_numpy_manifest,
        ),
    }
    continuous_metrics = validate_metrics(
        continuous / "qwen35_exact_metrics.jsonl",
        continuous / "qwen35_exact_metrics_summary.json",
        first_step=1,
        last_step=10,
        schedule_entries=schedule["entries"],
    )
    h5_metrics = validate_metrics(
        h5_paths["metrics"],
        h5_paths["metrics_summary"],
        first_step=1,
        last_step=5,
        schedule_entries=schedule["entries"],
    )
    resumed_metrics = validate_metrics(
        resumed / "qwen35_exact_metrics.jsonl",
        resumed / "qwen35_exact_metrics_summary.json",
        first_step=6,
        last_step=10,
        schedule_entries=schedule["entries"],
    )
    reconstructed = h5_metrics["projections"] + resumed_metrics["projections"]
    if continuous_metrics["projections"] != reconstructed:
        for index, (left, right) in enumerate(
            zip(continuous_metrics["projections"], reconstructed, strict=True), start=1
        ):
            if left != right:
                raise AssertionError(f"H6 deterministic metric projection first differs at step {index}")
        raise AssertionError("H6 deterministic metric projection differs")

    require_file(args.checkpoint_comparison)
    observed_comparison = load_strict_json(args.checkpoint_comparison)
    recomputed_comparison = compare_checkpoints(
        continuous / "checkpoint-10", resumed / "checkpoint-10", atol=0.0, rtol=0.0
    )
    if observed_comparison != recomputed_comparison:
        raise ValueError("H6 checkpoint-comparison report differs from recomputation")
    if (
        recomputed_comparison.get("status") != "passed"
        or recomputed_comparison.get("model", {}).get("bit_exact") is not True
        or recomputed_comparison.get("optimizer", {}).get("bit_exact_tensors") is not True
        or recomputed_comparison.get("scheduler", {}).get("bit_exact_tensors") is not True
        or recomputed_comparison.get("trainer_state", {}).get("global_step") != 10
        or len(recomputed_comparison.get("rng", {})) != 4
        or any(not row.get("bit_exact_tensors") for row in recomputed_comparison.get("rng", {}).values())
    ):
        raise ValueError("H6 checkpoint comparison did not prove complete bit equality")
    projection_sha = hashlib.sha256(canonical_json_bytes(continuous_metrics["projections"])).hexdigest()
    return {
        "artifact": "qwen35_r18_h6_producer_validation",
        "checkpoint_comparison_sha256": sha256_file(args.checkpoint_comparison),
        "contract_sha256": contract_sha,
        "deterministic_metric_projection_sha256": projection_sha,
        "hardware_inventory_sha256": sha256_file(args.hardware_inventory),
        "h5_evidence_sha256": {label: sha256_file(path) for label, path in h5_paths.items()},
        "h5_checkpoint_manifest_validation": checkpoint_manifest_validation,
        "metric_evidence": {
            "continuous": {
                key: value
                for key, value in continuous_metrics.items()
                if key not in {"records", "projections"}
            },
            "h5_prefix": {key: value for key, value in h5_metrics.items() if key not in {"records", "projections"}},
            "resumed_suffix": {
                key: value
                for key, value in resumed_metrics.items()
                if key not in {"records", "projections"}
            },
        },
        "run_evidence": run_evidence,
        "schema_version": 1,
        "scientific_training_authorized": False,
        "source_delta": source_delta,
        "status": "producer_passed_pending_slurm_and_independent_closure",
        "successor_authorized": None,
    }


def independent(args: argparse.Namespace) -> dict[str, Any]:
    if args.producer_validation is None or args.slurm_record is None:
        raise ValueError("H6 independent mode requires producer validation and Slurm record")
    recomputed = producer(args)
    producer_report = load_strict_json(args.producer_validation)
    if producer_report != recomputed:
        raise ValueError("H6 producer report differs from independent recomputation")
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
            raise ValueError(f"H6 Slurm completion drift for {key}")
    elapsed_seconds = slurm.get("elapsed_seconds")
    if (
        slurm.get("partition") != "boost_usr_prod"
        or slurm.get("qos") != "normal"
        or isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not 0 < elapsed_seconds <= 2_700
    ):
        raise ValueError("H6 Slurm resource-ceiling or partition/QOS drift")
    personal_root = Path("/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3")
    for label in ("stdout", "stderr"):
        path = Path(slurm[f"{label}_path"])
        if not path.resolve().is_relative_to(personal_root):
            raise ValueError(f"H6 Slurm {label} escaped the personal work root")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"H6 Slurm {label} is absent or symlinked")
        if label == "stdout" and path.stat().st_size <= 0:
            raise FileNotFoundError("H6 Slurm stdout is empty")
        if slurm.get(f"{label}_sha256") != sha256_file(path):
            raise ValueError(f"H6 Slurm {label} digest drift")
        contents = path.read_text(errors="replace")
        match = PROCESS_FAILURE_PATTERN.search(contents)
        if match:
            raise AssertionError(f"H6 Slurm {label} contains a process failure marker: {match.group(0)!r}")
        if label == "stdout" and "R18_H6_PRODUCER_PASSED_PENDING_SLURM_AND_INDEPENDENT_CLOSURE" not in contents:
            raise ValueError("H6 Slurm stdout lacks the terminal producer marker")
    exit_report = load_strict_json(args.output_dir / "g2_job_exit.json")
    if exit_report != {"exit_code": 0, "slurm_job_id": str(slurm["job_id"])}:
        raise ValueError("H6 wrapper exit record drift")
    return {
        "artifact": "qwen35_r18_h6_independent_closure",
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
            "artifact": "qwen35_r18_h6_validation_failure",
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
        require_finite_json(report, context=f"H6.{args.mode}.report")
        write_json_atomic(args.report_output, report)
    except BaseException as error:
        write_failure(args.report_output, mode=args.mode, error=error)
        raise
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
