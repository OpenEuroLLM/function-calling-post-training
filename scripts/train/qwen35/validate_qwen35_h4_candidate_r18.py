#!/usr/bin/env python3
"""Validate both assays for one R18 H4 chunk candidate, before manual kernel review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h4 import (
    LEONARDO_A100_COMPUTE_CAPABILITY,
    LEONARDO_A100_NAME,
    load_h4_contract,
    load_strict_json,
    load_strict_jsonl,
    require_finite_json,
    sha256_file,
    timing_statistics,
    validate_forward_loss_audit,
)
from open_instruct.qwen35_reporting import summarize_reporting_records
from open_instruct.qwen35_training import write_json_atomic

CUDA_TIMING_SCOPE = "rank-local default-stream events around Trainer optimizer step; synchronized once at train end"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--h4-contract", type=Path, required=True)
    parser.add_argument("--candidate-chunk-size", type=int, required=True)
    parser.add_argument("--profile-output-dir", type=Path, required=True)
    parser.add_argument("--timing-output-dir", type=Path, required=True)
    parser.add_argument("--four-update-schedule", type=Path, required=True)
    parser.add_argument("--thirteen-update-schedule", type=Path, required=True)
    parser.add_argument("--profile-validation", type=Path, required=True)
    parser.add_argument("--job-identity", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required nonempty H4 artifact is absent: {path}")


def _validate_cuda_timing_artifact(
    value: dict[str, Any],
    *,
    candidate_chunk_size: int,
    qualification: dict[str, Any],
    qualification_sha256: str,
    h4: dict[str, Any],
    h4_sha256: str,
) -> dict[str, float]:
    expected_fields = {
        "artifact",
        "assay",
        "candidate_chunk_size",
        "completed_optimizer_steps",
        "cuda_event_step_milliseconds",
        "h4_contract_sha256",
        "h4_protocol_id",
        "qualification_manifest_sha256",
        "qualification_protocol_id",
        "rank",
        "schema_version",
        "status",
        "timing_scope",
        "world_size",
    }
    if set(value) != expected_fields:
        raise ValueError("H4 timing CUDA-event artifact field drift")
    if (
        value.get("artifact") != "qwen35_per_rank_cuda_event_step_timing"
        or value.get("schema_version") != 1
        or value.get("status") != "passed"
        or value.get("assay") != "timing"
        or value.get("candidate_chunk_size") != candidate_chunk_size
        or value.get("qualification_protocol_id") != qualification["protocol_id"]
        or value.get("qualification_manifest_sha256") != qualification_sha256
        or value.get("h4_protocol_id") != h4["protocol_id"]
        or value.get("h4_contract_sha256") != h4_sha256
        or value.get("rank") != 0
        or value.get("world_size") != 1
        or value.get("completed_optimizer_steps") != 13
        or value.get("timing_scope") != CUDA_TIMING_SCOPE
    ):
        raise ValueError("H4 timing CUDA-event artifact drift")
    cuda_times = value.get("cuda_event_step_milliseconds")
    expected_keys = {str(step) for step in range(1, 14)}
    if not isinstance(cuda_times, dict) or set(cuda_times) != expected_keys:
        raise ValueError("H4 timing CUDA-event step set drift")
    for step in range(1, 14):
        duration = cuda_times[str(step)]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("H4 timing CUDA-event duration is not numeric")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("H4 timing CUDA-event duration is nonpositive or nonfinite")
    return {str(step): float(cuda_times[str(step)]) for step in range(1, 14)}


def _expected_learning_rates(max_steps: int, *, base_learning_rate: float, warmup_ratio: float) -> list[float]:
    warmup_steps = math.ceil(max_steps * warmup_ratio)
    values = []
    for scheduler_step_before_update in range(max_steps):
        if scheduler_step_before_update < warmup_steps:
            factor = scheduler_step_before_update / max(1, warmup_steps)
        else:
            progress = (scheduler_step_before_update - warmup_steps) / max(1, max_steps - warmup_steps)
            factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        values.append(base_learning_rate * factor)
    return values


def _validate_schedule(path: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_file(path)
    if sha256_file(path) != contract["file_sha256"]:
        raise ValueError("H4 schedule file hash drift")
    schedule = load_strict_json(path)
    if schedule.get("schedule_sha256") != contract["embedded_schedule_sha256"]:
        raise ValueError("H4 embedded schedule identity drift")
    if schedule.get("entries_sha256") != contract["entries_sha256"]:
        raise ValueError("H4 schedule entry digest drift")
    if schedule.get("optimizer_updates") != contract["optimizer_updates"]:
        raise ValueError("H4 schedule optimizer-step count drift")
    entries = schedule.get("entries")
    if not isinstance(entries, list) or len(entries) != contract["scheduled_packs"]:
        raise ValueError("H4 schedule pack count drift")
    if [row.get("schedule_index") for row in entries] != list(range(len(entries))):
        raise ValueError("H4 schedule indices are not contiguous")
    if len({row.get("pack_uid") for row in entries}) != len(entries):
        raise ValueError("H4 schedule repeats a pack UID")
    if any(row.get("synthetic") is not False for row in entries):
        raise ValueError("H4 schedule contains a synthetic pack")
    totals = schedule.get("totals", {})
    expected_totals = {
        "assistant_targets": contract["assistant_targets"],
        "attention_length_squared": contract["attention_length_squared"],
        "fixed_tokens": contract["fixed_positions"],
        "padding_tokens": contract["padding_positions"],
        "real_tokens": contract["real_positions"],
    }
    if totals != expected_totals:
        raise ValueError(f"H4 schedule totals drift: {totals!r} != {expected_totals!r}")
    return schedule, entries


def _validate_checkpoint(root: Path, final_step: int, world_size: int) -> dict[str, Any]:
    checkpoint = root / f"checkpoint-{final_step}"
    required = {
        "checkpoint_model": checkpoint / "model.safetensors",
        "checkpoint_optimizer": checkpoint / "optimizer.pt",
        "checkpoint_scheduler": checkpoint / "scheduler.pt",
        "checkpoint_trainer_state": checkpoint / "trainer_state.json",
        "root_config": root / "config.json",
        "root_model": root / "model.safetensors",
        "root_trainer_state": root / "trainer_state.json",
    }
    rng_paths = (
        [checkpoint / "rng_state.pth"]
        if world_size == 1
        else [checkpoint / f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    for path in [*required.values(), *rng_paths]:
        _require_file(path)
    root_state = load_strict_json(required["root_trainer_state"])
    checkpoint_state = load_strict_json(required["checkpoint_trainer_state"])
    for name, state in (("root", root_state), ("checkpoint", checkpoint_state)):
        if state.get("global_step") != final_step or state.get("max_steps") != final_step:
            raise ValueError(f"H4 {name} trainer state step horizon drift")

    model_dtype_counts: dict[str, int] = {}
    model_tensor_count = 0
    with safe_open(required["checkpoint_model"], framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 - safe_open exposes keys(), not ordinary iteration
            dtype = handle.get_slice(key).get_dtype()
            model_dtype_counts[dtype] = model_dtype_counts.get(dtype, 0) + 1
            model_tensor_count += 1
    if model_dtype_counts != {"F32": model_tensor_count} or model_tensor_count <= 0:
        raise ValueError(f"H4 checkpoint parameter dtype drift: {model_dtype_counts}")

    optimizer_state = torch.load(required["checkpoint_optimizer"], map_location="cpu", weights_only=True)
    moment_tensors = 0
    moment_dtype_counts: dict[str, int] = {}
    step_values = set()
    for parameter_state in optimizer_state.get("state", {}).values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = parameter_state.get(name)
            if value is None:
                continue
            moment_tensors += 1
            key = str(value.dtype)
            moment_dtype_counts[key] = moment_dtype_counts.get(key, 0) + 1
        step = parameter_state.get("step")
        if torch.is_tensor(step):
            step_values.add(float(step.item()))
    if moment_tensors <= 0 or moment_dtype_counts != {"torch.float32": moment_tensors}:
        raise ValueError(f"H4 AdamW moment dtype drift: {moment_dtype_counts}")
    if step_values != {float(final_step)}:
        raise ValueError(f"H4 AdamW optimizer step counters drift: {sorted(step_values)}")
    scheduler_state = torch.load(required["checkpoint_scheduler"], map_location="cpu", weights_only=True)
    if scheduler_state.get("last_epoch") != final_step:
        raise ValueError("H4 checkpoint scheduler step counter drift")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(required["checkpoint_model"]),
        "checkpoint_optimizer_sha256": sha256_file(required["checkpoint_optimizer"]),
        "checkpoint_scheduler_sha256": sha256_file(required["checkpoint_scheduler"]),
        "checkpoint_trainer_state_sha256": sha256_file(required["checkpoint_trainer_state"]),
        "model_dtype_counts": model_dtype_counts,
        "moment_dtype_counts": moment_dtype_counts,
        "optimizer_step_values": sorted(step_values),
        "rng_state_sha256": [sha256_file(path) for path in rng_paths],
        "root_config_sha256": sha256_file(required["root_config"]),
        "root_model_sha256": sha256_file(required["root_model"]),
        "root_trainer_state_sha256": sha256_file(required["root_trainer_state"]),
    }


def _validate_metrics(
    *,
    root: Path,
    schedule: dict[str, Any],
    entries: list[dict[str, Any]],
    final_step: int,
    chunk_size: int,
    expected_assay: str,
) -> dict[str, Any]:
    metrics_path = root / "qwen35_exact_metrics.jsonl"
    summary_path = root / "qwen35_exact_metrics_summary.json"
    _require_file(metrics_path)
    _require_file(summary_path)
    records = load_strict_jsonl(metrics_path)
    summary = load_strict_json(summary_path)
    if len(records) != final_step:
        raise ValueError("H4 exact metrics must contain one synchronized record per optimizer update")
    if [record.get("step") for record in records] != list(range(1, final_step + 1)):
        raise ValueError("H4 exact metrics step sequence drift")
    expected_learning_rates = _expected_learning_rates(final_step, base_learning_rate=2e-5, warmup_ratio=0.03)
    all_audits = []
    all_gradient_audits = []
    for step_index, record in enumerate(records):
        if record.get("schedule_sha256") != schedule["schedule_sha256"]:
            raise ValueError("H4 exact metrics schedule identity drift")
        if record.get("synchronized_timing") is not True or record.get("optimizer_updates") != 1:
            raise ValueError("H4 exact metrics timing/window contract drift")
        elapsed = record.get("elapsed_seconds")
        if (
            not isinstance(elapsed, (float, int))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("H4 exact metrics contains an invalid synchronized duration")
        group = entries[step_index * 8 : (step_index + 1) * 8]
        expected_indices = [row["schedule_index"] for row in group]
        expected_uids = [row["pack_uid"] for row in group]
        if record.get("schedule_indices") != expected_indices or record.get("pack_uids") != expected_uids:
            raise ValueError("H4 exact metrics exposure differs from the frozen schedule")
        expected_counts = {
            "assistant_targets": sum(row["assistant_targets"] for row in group),
            "attention_length_squared": sum(row["attention_length_squared"] for row in group),
            "documents": sum(row["document_count"] for row in group),
            "fixed_tokens": 8 * 32768,
            "packs": 8,
            "padding_tokens": sum(row["padding_tokens"] for row in group),
            "real_tokens": sum(row["real_tokens"] for row in group),
            "synthetic_packs": 0,
        }
        if record.get("counts") != expected_counts:
            raise ValueError("H4 exact metrics count accounting drift")
        divisor = expected_counts["assistant_targets"]
        if record.get("loss", {}).get("global_assistant_target_divisor") != divisor:
            raise ValueError("H4 exact metrics global loss divisor drift")
        loss = record.get("loss", {}).get("normalized_loss")
        if not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
            raise ValueError("H4 exact metrics normalized loss is invalid")
        applied = record.get("optimizer", {}).get("applied_learning_rates")
        expected_lr = expected_learning_rates[step_index]
        if (
            not isinstance(applied, list)
            or len(applied) != 1
            or not math.isclose(float(applied[0]), expected_lr, rel_tol=1e-14, abs_tol=0.0)
        ):
            raise ValueError(
                f"H4 applied learning-rate drift at step {step_index + 1}: {applied!r} != {expected_lr!r}"
            )
        audits = record.get("selected_output_audits")
        if not isinstance(audits, list) or len(audits) != 8:
            raise ValueError("H4 exact metrics omitted an every-forward selected-output audit")
        for audit_row, schedule_row in zip(audits, group, strict=True):
            if (
                audit_row.get("schedule_index") != schedule_row["schedule_index"]
                or audit_row.get("pack_uid") != schedule_row["pack_uid"]
                or audit_row.get("loss_dtype") != "torch.float32"
            ):
                raise ValueError("H4 selected-output audit identity drift")
            recomputed = validate_forward_loss_audit(
                audit_row.get("audit"),
                expected_selected_rows=schedule_row["assistant_targets"],
                expected_global_target_count=divisor,
                expected_chunk_size=chunk_size,
            )
            if audit_row.get("validation") != recomputed:
                raise ValueError("H4 producer selected-output audit validation drift")
        all_audits.extend(audits)
        gradient_audits = record.get("gradient_dtype_audits")
        if not isinstance(gradient_audits, list) or len(gradient_audits) != 1:
            raise ValueError("H4 exact metrics omitted an optimizer-step gradient dtype audit")
        gradient_audit = gradient_audits[0]
        if (
            gradient_audit.get("missing_gradient_count") != 0
            or gradient_audit.get("gradient_tensor_count", 0) <= 0
            or gradient_audit.get("gradient_numel", 0) <= 0
            or gradient_audit.get("gradient_dtype_counts")
            != {"torch.float32": gradient_audit.get("gradient_tensor_count")}
        ):
            raise ValueError("H4 gradient dtype/connectivity audit drift")
        all_gradient_audits.extend(gradient_audits)

    independently_summarized = summarize_reporting_records(records)
    for key, value in independently_summarized.items():
        if summary.get(key) != value:
            raise ValueError(f"H4 exact metrics summary drift for {key}")
    canonical_audits = json.dumps(all_audits, sort_keys=True, separators=(",", ":")).encode()
    if summary.get("selected_output_audit_count") != len(entries):
        raise ValueError("H4 exact metrics summary selected-output audit count drift")
    if summary.get("selected_output_audits_sha256") != hashlib.sha256(canonical_audits).hexdigest():
        raise ValueError("H4 exact metrics summary selected-output audit hash drift")
    canonical_gradient_audits = json.dumps(all_gradient_audits, sort_keys=True, separators=(",", ":")).encode()
    if summary.get("gradient_dtype_audit_count") != final_step:
        raise ValueError("H4 exact metrics summary gradient dtype audit count drift")
    if summary.get("gradient_dtype_audits_sha256") != hashlib.sha256(canonical_gradient_audits).hexdigest():
        raise ValueError("H4 exact metrics summary gradient dtype audit hash drift")
    if summary.get("loaded_liger_modules") != []:
        raise AssertionError("H4 process imported a forbidden Liger module")
    return {
        "all_selected_output_audits_sha256": hashlib.sha256(canonical_audits).hexdigest(),
        "assay": expected_assay,
        "exact_metrics_sha256": sha256_file(metrics_path),
        "exact_metrics_summary_sha256": sha256_file(summary_path),
        "gradient_dtype_audits_sha256": hashlib.sha256(canonical_gradient_audits).hexdigest(),
        "records": records,
        "selected_output_audit_count": len(all_audits),
    }


def _validate_assay(
    *,
    root: Path,
    schedule: dict[str, Any],
    entries: list[dict[str, Any]],
    final_step: int,
    chunk_size: int,
    assay: str,
    qualification_sha256: str,
    h4_sha256: str,
    h4_contract_path: Path,
) -> dict[str, Any]:
    required_json = {
        "conversion_ledger": root / "qwen35_text_conversion_ledger.json",
        "run_manifest": root / "qwen35_run_manifest.json",
        "train_results": root / "train_results.json",
        "update_probe": root / "qwen35_parameter_update_probe.json",
    }
    for path in required_json.values():
        _require_file(path)
    values = {name: load_strict_json(path) for name, path in required_json.items()}
    for name, value in values.items():
        require_finite_json(value, context=f"{assay}.{name}")
    run = values["run_manifest"]
    if run.get("model_class") != "Qwen3_5ForCausalLM" or run.get("model_config_type") != "qwen3_5_text":
        raise ValueError("H4 assay model class/config drift")
    if run.get("vision_tower_loaded") is not False or run.get("model_text_vocab_size") != 248320:
        raise ValueError("H4 assay text-only model geometry drift")
    if run.get("world_size") != 1 or run.get("gradient_accumulation_steps") != 8:
        raise ValueError("H4 assay distributed/batch geometry drift")
    if run.get("sequence_length") != 32768 or run.get("effective_tokens_per_optimizer_step") != 262144:
        raise ValueError("H4 assay fixed-token geometry drift")
    if run.get("frozen_data_validation", {}).get("arm_id") != "C00":
        raise ValueError("H4 assay did not consume C00")
    if run.get("schedule_validation", {}).get("schedule_sha256") != schedule["schedule_sha256"]:
        raise ValueError("H4 run-manifest schedule drift")
    hardware = run.get("hardware_qualification", {})
    if (
        hardware.get("manifest_sha256") != qualification_sha256
        or hardware.get("require_no_dense_logits") is not True
        or hardware.get("hardware_profile") is not (assay == "profiler")
        or hardware.get("cuda_event_step_timing") is not (assay == "timing")
    ):
        raise ValueError("H4 run-manifest R18 identity or dense-logit guard drift")
    h4_identity = run.get("h4_qualification", {})
    if h4_identity != {
        "assay": assay,
        "candidate_chunk_size": chunk_size,
        "contract_path": str(h4_contract_path.resolve()),
        "contract_sha256": h4_sha256,
        "loaded_liger_modules_at_manifest": [],
        "preregistration_closure_sha256": "fd1db81c568c473a417c2936f3492288a8d0188057fe548696a2211e26bb8080",
        "protocol_id": "qwen35-hardware-qualification-r18-h4-r1",
        "require_forward_loss_audit": True,
    }:
        raise ValueError("H4 run-manifest H4 identity drift")
    selective = run.get("selective_output_projection", {})
    if (
        selective.get("enabled") is not True
        or selective.get("implementation") != IMPLEMENTATION_ID
        or selective.get("chunk_size") != chunk_size
        or selective.get("liger_status") != "abandoned_after_r17"
    ):
        raise ValueError("H4 selected-output implementation drift")
    args = run.get("training_arguments", {})
    expected_training_args = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "data_seed": 3407,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 2e-5,
        "max_grad_norm": 1.0,
        "max_steps": final_step,
        "optim": "adamw_torch_fused",
        "per_device_train_batch_size": 1,
        "seed": 3407,
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,
    }
    for key, expected in expected_training_args.items():
        if args.get(key) != expected:
            raise ValueError(f"H4 training argument drift for {key}: {args.get(key)!r} != {expected!r}")
    scheduler = args.get("lr_scheduler_type")
    if scheduler not in ("cosine", {"name": "cosine"}):
        raise ValueError(f"H4 scheduler serialization drift: {scheduler!r}")

    conversion = values["conversion_ledger"]
    if (
        conversion.get("target_class") != "Qwen3_5ForCausalLM"
        or conversion.get("target_config_model_type") != "qwen3_5_text"
        or conversion.get("tied_input_output_embeddings") is not True
        or conversion.get("tensor_hashes_enabled") is not True
        or not conversion.get("rows")
        or any(not row.get("tensor_sha256") for row in conversion["rows"])
    ):
        raise ValueError("H4 text conversion ledger is incomplete")
    update = values["update_probe"]
    if (
        update.get("status") != "passed"
        or update.get("observed_initial_global_step") != 0
        or update.get("final_global_step") != final_step
        or update.get("optimizer_steps_observed") != final_step
        or update.get("parameter_comparison", {}).get("changed_sampled_values", 0) <= 0
        or update.get("parameter_comparison", {}).get("max_absolute_delta", 0) <= 0
    ):
        raise ValueError("H4 parameter update probe drift")
    for key in ("finite_losses", "finite_gradient_norms"):
        rows = update.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"H4 update probe lacks {key}")
        if any(not math.isfinite(float(row.get("value", math.nan))) for row in rows):
            raise ValueError(f"H4 update probe contains nonfinite {key}")

    metrics = _validate_metrics(
        root=root,
        schedule=schedule,
        entries=entries,
        final_step=final_step,
        chunk_size=chunk_size,
        expected_assay=assay,
    )
    checkpoint = _validate_checkpoint(root, final_step, world_size=1)
    return {
        "artifact_sha256": {name: sha256_file(path) for name, path in required_json.items()},
        "assay": assay,
        "checkpoint": checkpoint,
        "conversion_ledger_rows_sha256": conversion.get("rows_sha256"),
        "metrics": metrics,
        "root": str(root.resolve()),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h4, h4_sha256 = load_h4_contract(args.h4_contract)
    if qualification_sha256 != h4["parent"]["r18_machine_manifest_sha256"]:
        raise ValueError("H4 candidate validator R18/H4 identity drift")
    if args.candidate_chunk_size not in h4["candidate_chunk_sizes_in_execution_order"]:
        raise ValueError("H4 candidate validator received an unknown chunk size")
    job_identity = load_strict_json(args.job_identity)
    if (
        job_identity.get("artifact") != "qwen35_r18_h4_candidate_job_identity"
        or job_identity.get("status") != "passed_identity_capture_only"
        or job_identity.get("candidate_chunk_size") != args.candidate_chunk_size
        or job_identity.get("h4_contract_sha256") != h4_sha256
        or job_identity.get("qualification_manifest_sha256") != qualification_sha256
        or job_identity.get("scientific_training_authorized") is not False
        or job_identity.get("slurm", {}).get("account") != "aifac_f02_434"
        or job_identity.get("git", {}).get("commit") != args.expected_code_commit
        or job_identity.get("git", {}).get("status_porcelain") != ""
        or job_identity.get("source_python_bytecode_files") != []
    ):
        raise ValueError("H4 candidate job identity drift")
    if len(job_identity.get("gpu_inventory", [])) != 1:
        raise ValueError("H4 candidate job identity did not bind exactly one GPU")
    gpu = job_identity["gpu_inventory"][0]
    if gpu.get("name") != LEONARDO_A100_NAME or gpu.get("compute_cap") != LEONARDO_A100_COMPUTE_CAPABILITY:
        raise ValueError("H4 candidate job identity GPU drift")
    identity_files = job_identity.get("file_identities", {})
    expected_identity_hashes = {
        "four_update_schedule": h4["four_update_schedule"]["file_sha256"],
        "h4_contract": h4_sha256,
        "numpy_manifest": h4["data"]["numpy_manifest_sha256"],
        "qualification_manifest": qualification_sha256,
        "thirteen_update_schedule": h4["thirteen_update_schedule"]["file_sha256"],
    }
    for label, expected in expected_identity_hashes.items():
        if identity_files.get(label, {}).get("sha256") != expected:
            raise ValueError(f"H4 job identity file hash drift for {label}")
    four, four_entries = _validate_schedule(args.four_update_schedule, h4["four_update_schedule"])
    thirteen, thirteen_entries = _validate_schedule(args.thirteen_update_schedule, h4["thirteen_update_schedule"])
    if four_entries != thirteen_entries[:32]:
        raise ValueError("H4 four-update schedule is not an exact prefix of the thirteen-update schedule")
    profile = _validate_assay(
        root=args.profile_output_dir,
        schedule=four,
        entries=four_entries,
        final_step=4,
        chunk_size=args.candidate_chunk_size,
        assay="profiler",
        qualification_sha256=qualification_sha256,
        h4_sha256=h4_sha256,
        h4_contract_path=args.h4_contract,
    )
    timing = _validate_assay(
        root=args.timing_output_dir,
        schedule=thirteen,
        entries=thirteen_entries,
        final_step=13,
        chunk_size=args.candidate_chunk_size,
        assay="timing",
        qualification_sha256=qualification_sha256,
        h4_sha256=h4_sha256,
        h4_contract_path=args.h4_contract,
    )
    if profile["conversion_ledger_rows_sha256"] != timing["conversion_ledger_rows_sha256"]:
        raise ValueError("H4 profiler and timing assays did not start from the same converted checkpoint tensors")
    if [row["schedule_indices"] for row in profile["metrics"]["records"]] != [
        row["schedule_indices"] for row in timing["metrics"]["records"][:4]
    ]:
        raise ValueError("H4 profiler exposure is not the exact timing-assay prefix")

    profile_validation = load_strict_json(args.profile_validation)
    if (
        profile_validation.get("status") != "automated_profile_passed_pending_manual_kernel_mapping"
        or profile_validation.get("candidate_chunk_size") != args.candidate_chunk_size
        or profile_validation.get("qualification_manifest_sha256") != qualification_sha256
        or profile_validation.get("h4_contract_sha256") != h4_sha256
    ):
        raise ValueError("H4 profiler validation identity or status drift")
    hardware_profile_path = args.profile_output_dir / "qwen35_cuda_hardware_profile.json"
    _require_file(hardware_profile_path)
    if profile_validation.get("hardware_profile_sha256") != sha256_file(hardware_profile_path):
        raise ValueError("H4 profiler validation does not bind the candidate's raw hardware profile")

    cuda_timing_path = args.timing_output_dir / "qwen35_cuda_step_times_rank00.json"
    _require_file(cuda_timing_path)
    cuda_timing = load_strict_json(cuda_timing_path)
    cuda_times = _validate_cuda_timing_artifact(
        cuda_timing,
        candidate_chunk_size=args.candidate_chunk_size,
        qualification=qualification,
        qualification_sha256=qualification_sha256,
        h4=h4,
        h4_sha256=h4_sha256,
    )

    synchronized = [float(row["elapsed_seconds"]) for row in timing["metrics"]["records"]]
    measured = synchronized[3:13]
    timing_stats = timing_statistics(measured)
    cv_threshold = float(h4["timing_selection"]["maximum_coefficient_of_variation"])
    return {
        "artifact": "qwen35_r18_h4_candidate_automated_validation",
        "candidate_chunk_size": args.candidate_chunk_size,
        "cuda_event_step_milliseconds": cuda_times,
        "eligible_pending_manual_kernel_mapping": True,
        "h4_contract_sha256": h4_sha256,
        "job_identity_sha256": sha256_file(args.job_identity),
        "source_commit": args.expected_code_commit,
        "measured_synchronized_update_seconds": measured,
        "profile_assay": {key: value for key, value in profile.items() if key != "metrics"},
        "profile_validation_sha256": sha256_file(args.profile_validation),
        "qualification_manifest_sha256": qualification_sha256,
        "schema_version": 1,
        "slurm_account": job_identity["slurm"]["account"],
        "slurm_job_id": job_identity["slurm"]["job_id"],
        "status": "automated_candidate_passed_pending_manual_kernel_mapping",
        "synchronized_update_seconds_all": synchronized,
        "timing_assay": {key: value for key, value in timing.items() if key != "metrics"},
        "timing_coefficient_of_variation_exceeds_threshold": timing_stats["coefficient_of_variation"] > cv_threshold,
        "timing_cv_threshold": cv_threshold,
        "timing_statistics": timing_stats,
        "timing_cuda_event_sha256": sha256_file(cuda_timing_path),
    }


def main() -> int:
    args = parse_args()
    if args.report_output.exists():
        raise FileExistsError(args.report_output)
    report = validate(args)
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
