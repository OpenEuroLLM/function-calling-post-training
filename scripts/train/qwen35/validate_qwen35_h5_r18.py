#!/usr/bin/env python3
"""Capture and independently validate the preregistered R18 H5 production-path evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID
from open_instruct.qwen35_qualification import validate_memory_headroom
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h4 import (
    LEONARDO_A100_COMPUTE_CAPABILITY,
    LEONARDO_A100_MEMORY_MIB,
    LEONARDO_A100_NAME,
    load_strict_json,
    load_strict_jsonl,
    require_finite_json,
    sha256_file,
    validate_forward_loss_audit,
)
from open_instruct.qwen35_qualification_r18_h5 import (
    H5_EXPECTED_TARGETS_BY_UPDATE,
    H5_FINAL_STEP,
    H5_FIRST_FIVE_ENTRIES_SHA256,
    H5_SCHEDULE_ENTRIES_SHA256,
    H5_SCHEDULE_FILE_SHA256,
    H5_SCHEDULE_SHA256,
    H5_SCHEDULER_HORIZON,
    H5_SELECTED_CHUNK_SIZE,
    H5_WORLD_SIZE,
    canonical_json_bytes,
    load_h5_contract,
    load_h5_harness_amendment,
    load_h5_harness_amendment_r2,
    validate_h5_source_delta,
)
from open_instruct.qwen35_reporting import summarize_reporting_records
from open_instruct.qwen35_training import write_json_atomic

EXPECTED_PREDECESSORS = {
    "h4_final_closure": "d363c5868ebe440b0540431b536cb0baff061c64e633a725a79fa6277538945d",
    "h4_set_validation": "5650915894222a3ba50b8dc3b2049db8983c90cf43842e7993b42b8eaa018cda",
    "schedule_closure": "10afebd0c2ed11820ea61b4d4263173d331a0622bd120710578c293d11e7c463",
    "schedule_validation": "836acb3d91782b649e6d54b9a7a8a1db2a3833954e38a667c2f8881305c21e80",
    "schedule_staging": "cfecaac6f6df22bfab6230b286b6e07d27d0d6c6533a67d15f33c4e8a09573d1",
    "schedule_source_code_manifest": "e154006acc996b90402e6c6d67f1f5dc421b50e9f8e4608435b58ce82fba8257",
}
EXPECTED_EXPOSURE_TOTALS = {
    "assistant_targets": 308_977,
    "attention_length_squared": 5_904_515_208,
    "documents": 470,
    "fixed_tokens": 1_310_720,
    "padding_tokens": 812,
    "real_tokens": 1_309_908,
    "synthetic_packs": 0,
}
CUDA_TIMING_SCOPE = "rank-local default-stream events around Trainer optimizer step; synchronized once at train end"
NCCL_FAILURE_PATTERN = re.compile(
    r"NCCL\s+(?:WARN|ERROR)|collective.*(?:timeout|abort)|nccl.*(?:timeout|abort|unhandled)"
    r"|destroy_process_group\(\) was not called"
    r"|barrier\(\): using the device under current context"
    r"|Guessing device ID based on global rank"
    r"|Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "producer", "independent"), required=True)
    parser.add_argument("--h5-contract", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--preregistration-closure", type=Path, required=True)
    parser.add_argument("--harness-amendment", type=Path, required=True)
    parser.add_argument("--harness-human-amendment", type=Path, required=True)
    parser.add_argument("--attempt01-failure-closure", type=Path, required=True)
    parser.add_argument("--harness-amendment-r2", type=Path, required=True)
    parser.add_argument("--harness-human-amendment-r2", type=Path, required=True)
    parser.add_argument("--attempt02-failure-closure", type=Path, required=True)
    parser.add_argument("--reload-type-diagnostic", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--h4-set-validation", type=Path, required=True)
    parser.add_argument("--h4-final-closure", type=Path, required=True)
    parser.add_argument("--schedule-validation", type=Path, required=True)
    parser.add_argument("--schedule-closure", type=Path, required=True)
    parser.add_argument("--schedule-staging", type=Path, required=True)
    parser.add_argument("--schedule-source-code-manifest", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--source-code-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--numpy-data", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-validation", type=Path)
    parser.add_argument("--slurm-record", type=Path)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required nonempty regular H5 artifact is absent: {path}")


def _require_regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular H5 file is absent or symlinked: {path}")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _nvidia_inventory() -> list[dict[str, str]]:
    fields = ["index", "name", "uuid", "memory.total", "driver_version", "compute_cap"]
    raw = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in raw.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(fields):
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(dict(zip(fields, values, strict=True)))
    if len(rows) != H5_WORLD_SIZE:
        raise ValueError(f"H5 requires four visible GPUs, observed {len(rows)}")
    expected = {
        "compute_cap": LEONARDO_A100_COMPUTE_CAPABILITY,
        "memory.total": LEONARDO_A100_MEMORY_MIB,
        "name": LEONARDO_A100_NAME,
    }
    for row in rows:
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError(f"unexpected H5 GPU identity: {row}")
    if len({row["uuid"] for row in rows}) != H5_WORLD_SIZE:
        raise ValueError("H5 GPU UUIDs are not unique")
    return rows


def _load_context(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any], str, str, str]:
    contract, contract_sha = load_h5_contract(
        args.h5_contract,
        human_protocol_path=args.human_protocol,
        preregistration_closure_path=args.preregistration_closure,
    )
    _, amendment_sha = load_h5_harness_amendment(
        args.harness_amendment,
        human_amendment_path=args.harness_human_amendment,
        attempt01_failure_closure_path=args.attempt01_failure_closure,
    )
    _, amendment_r2_sha = load_h5_harness_amendment_r2(
        args.harness_amendment_r2,
        human_amendment_path=args.harness_human_amendment_r2,
        attempt02_failure_closure_path=args.attempt02_failure_closure,
        reload_type_diagnostic_path=args.reload_type_diagnostic,
    )
    qualification, qualification_sha = load_qualification_manifest(args.qualification_manifest)
    if qualification_sha != contract["parent"]["r18_machine_manifest_sha256"]:
        raise ValueError("H5 qualification-manifest predecessor drift")
    paths = {
        "h4_set_validation": args.h4_set_validation,
        "h4_final_closure": args.h4_final_closure,
        "schedule_validation": args.schedule_validation,
        "schedule_closure": args.schedule_closure,
        "schedule_staging": args.schedule_staging,
        "schedule_source_code_manifest": args.schedule_source_code_manifest,
    }
    for label, path in paths.items():
        _require_file(path)
        observed = sha256_file(path)
        if observed != EXPECTED_PREDECESSORS[label]:
            raise ValueError(f"H5 predecessor {label} digest drift: {observed}")
    expected_status = {
        "h4_set_validation": "passed_H5_only_authorized",
        "h4_final_closure": "passed_H5_only_authorized_independently_reconciled",
        "schedule_validation": "passed_schedule_freeze_only_H5_GPU_remains_blocked",
        "schedule_closure": "passed_schedule_materialization_only_H5_GPU_remains_blocked",
        "schedule_staging": "passed_CPU_staging_only_schedule_materialization_authorized",
    }
    for label, status in expected_status.items():
        if load_strict_json(paths[label]).get("status") != status:
            raise ValueError(f"H5 predecessor {label} status drift")
    return contract, contract_sha, qualification, qualification_sha, amendment_sha, amendment_r2_sha


def _validate_schedule(
    args: argparse.Namespace, contract: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_file(args.schedule)
    if sha256_file(args.schedule) != H5_SCHEDULE_FILE_SHA256:
        raise ValueError("H5 schedule file digest drift")
    schedule = load_strict_json(args.schedule)
    if schedule.get("schedule_sha256") != H5_SCHEDULE_SHA256:
        raise ValueError("H5 embedded schedule digest drift")
    if schedule.get("entries_sha256") != H5_SCHEDULE_ENTRIES_SHA256:
        raise ValueError("H5 schedule entries digest drift")
    entries = schedule.get("entries")
    if not isinstance(entries, list) or len(entries) != 80:
        raise ValueError("H5 schedule must contain exactly 80 entries")
    if [row.get("schedule_index") for row in entries] != list(range(80)):
        raise ValueError("H5 schedule indices are not contiguous")
    if len({row.get("pack_uid") for row in entries}) != 80 or len({row.get("pack_index") for row in entries}) != 80:
        raise ValueError("H5 schedule repeats a pack UID or pack index")
    if any(row.get("synthetic") is not False for row in entries):
        raise ValueError("H5 schedule contains a synthetic pack")
    prefix = entries[:40]
    if hashlib.sha256(canonical_json_bytes(prefix)).hexdigest() != H5_FIRST_FIVE_ENTRIES_SHA256:
        raise ValueError("H5 five-update prefix digest drift")
    totals = {
        "assistant_targets": sum(row["assistant_targets"] for row in prefix),
        "attention_length_squared": sum(row["attention_length_squared"] for row in prefix),
        "documents": sum(row["document_count"] for row in prefix),
        "fixed_tokens": 40 * 32768,
        "padding_tokens": sum(row["padding_tokens"] for row in prefix),
        "real_tokens": sum(row["real_tokens"] for row in prefix),
        "synthetic_packs": sum(bool(row["synthetic"]) for row in prefix),
    }
    if totals != EXPECTED_EXPOSURE_TOTALS or totals != {
        "assistant_targets": contract["five_update_exposure"]["assistant_targets"],
        "attention_length_squared": contract["five_update_exposure"]["attention_length_squared"],
        "documents": contract["five_update_exposure"]["documents"],
        "fixed_tokens": contract["five_update_exposure"]["fixed_positions"],
        "padding_tokens": contract["five_update_exposure"]["padding_positions"],
        "real_tokens": contract["five_update_exposure"]["real_positions"],
        "synthetic_packs": 0,
    }:
        raise ValueError(f"H5 five-update schedule accounting drift: {totals}")
    return schedule, prefix


def _verify_code_manifest(repository: Path, manifest: Path) -> None:
    _require_file(manifest)
    subprocess.run(["sha256sum", "--check", "--strict", str(manifest.resolve())], cwd=repository.resolve(), check=True)


def capture(args: argparse.Namespace) -> dict[str, Any]:
    contract, contract_sha, _, qualification_sha, amendment_sha, amendment_r2_sha = _load_context(args)
    schedule, _ = _validate_schedule(args, contract)
    source_delta = validate_h5_source_delta(
        args.source_repository,
        expected_head=args.expected_source_head,
        harness_amendment_path=args.harness_amendment,
        harness_human_amendment_path=args.harness_human_amendment,
        attempt01_failure_closure_path=args.attempt01_failure_closure,
        harness_amendment_r2_path=args.harness_amendment_r2,
        harness_human_amendment_r2_path=args.harness_human_amendment_r2,
        attempt02_failure_closure_path=args.attempt02_failure_closure,
        reload_type_diagnostic_path=args.reload_type_diagnostic,
    )
    _verify_code_manifest(args.source_repository, args.source_code_manifest)
    numpy_manifest = args.numpy_data / "manifest.json"
    for path in (numpy_manifest, args.runtime_report, args.model_manifest):
        _require_file(path)
    if sha256_file(numpy_manifest) != contract["data"]["numpy_manifest_sha256"]:
        raise ValueError("H5 C00 NumPy manifest digest drift")
    runtime = load_strict_json(args.runtime_report)
    if runtime.get("status") != "passed":
        raise ValueError("H5 pinned runtime import report did not pass")
    loaded_liger = sorted(name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel."))
    if loaded_liger:
        raise RuntimeError(f"H5 identity capture imported forbidden Liger modules: {loaded_liger}")
    if os.environ.get("SLURM_JOB_ACCOUNT") != "aifac_f02_434" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("H5 identity capture requires a personal-account Slurm allocation")
    files = {
        "h5_contract": args.h5_contract,
        "h5_human_protocol": args.human_protocol,
        "h5_preregistration_closure": args.preregistration_closure,
        "h5_harness_amendment": args.harness_amendment,
        "h5_harness_human_amendment": args.harness_human_amendment,
        "h5_attempt01_failure_closure": args.attempt01_failure_closure,
        "h5_harness_amendment_r2": args.harness_amendment_r2,
        "h5_harness_human_amendment_r2": args.harness_human_amendment_r2,
        "h5_attempt02_failure_closure": args.attempt02_failure_closure,
        "h5_reload_type_diagnostic": args.reload_type_diagnostic,
        "h4_set_validation": args.h4_set_validation,
        "h4_final_closure": args.h4_final_closure,
        "schedule_validation": args.schedule_validation,
        "schedule_closure": args.schedule_closure,
        "schedule_staging": args.schedule_staging,
        "schedule_source_code_manifest": args.schedule_source_code_manifest,
        "source_code_manifest": args.source_code_manifest,
        "qualification_manifest": args.qualification_manifest,
        "numpy_manifest": numpy_manifest,
        "runtime_report": args.runtime_report,
        "model_manifest": args.model_manifest,
        "schedule": args.schedule,
    }
    return {
        "artifact": "qwen35_r18_h5_immutable_input_and_hardware_inventory",
        "contract_sha256": contract_sha,
        "harness_amendment_sha256": amendment_sha,
        "harness_amendment_r2_sha256": amendment_r2_sha,
        "environment": {
            "pythonpath": os.environ.get("PYTHONPATH"),
            "python_bytecode_cache": os.environ.get("PYTHONPYCACHEPREFIX"),
        },
        "file_identities": {
            label: {"bytes": path.stat().st_size, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in sorted(files.items())
        },
        "git": {
            "branch": _git(args.source_repository, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": args.expected_source_head,
            "status_porcelain": "",
            "tree": _git(args.source_repository, "rev-parse", "HEAD^{tree}"),
        },
        "gpu_inventory": _nvidia_inventory(),
        "loaded_liger_modules": loaded_liger,
        "qualification_manifest_sha256": qualification_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "schema_version": 1,
        "scientific_training_authorized": False,
        "slurm": {
            "account": os.environ["SLURM_JOB_ACCOUNT"],
            "job_id": os.environ["SLURM_JOB_ID"],
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "source_delta": source_delta,
        "source_python_bytecode_files": sorted(
            str(path.relative_to(args.source_repository)) for path in args.source_repository.rglob("*.pyc")
        ),
        "status": "passed_identity_capture_only",
    }


def _expected_learning_rates() -> list[float]:
    warmup_steps = math.ceil(H5_SCHEDULER_HORIZON * 0.03)
    values = []
    for step in range(H5_FINAL_STEP):
        if step < warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, H5_SCHEDULER_HORIZON - warmup_steps)
            factor = max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
        values.append(2e-5 * factor)
    return values


def _validate_metrics(root: Path, schedule: dict[str, Any], prefix: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_path = root / "qwen35_exact_metrics.jsonl"
    summary_path = root / "qwen35_exact_metrics_summary.json"
    for path in (metrics_path, summary_path):
        _require_file(path)
    records = load_strict_jsonl(metrics_path)
    summary = load_strict_json(summary_path)
    if len(records) != H5_FINAL_STEP or [row.get("step") for row in records] != list(range(1, 6)):
        raise ValueError("H5 exact metrics step set drift")
    all_audits: list[dict[str, Any]] = []
    all_gradient_audits: list[dict[str, Any]] = []
    expected_lrs = _expected_learning_rates()
    for index, record in enumerate(records):
        require_finite_json(record, context=f"H5.metrics.step{index + 1}")
        group = prefix[index * 8 : (index + 1) * 8]
        expected_counts = {
            "assistant_targets": sum(row["assistant_targets"] for row in group),
            "attention_length_squared": sum(row["attention_length_squared"] for row in group),
            "documents": sum(row["document_count"] for row in group),
            "fixed_tokens": 262_144,
            "packs": 8,
            "padding_tokens": sum(row["padding_tokens"] for row in group),
            "real_tokens": sum(row["real_tokens"] for row in group),
            "synthetic_packs": 0,
        }
        if expected_counts["assistant_targets"] != H5_EXPECTED_TARGETS_BY_UPDATE[index]:
            raise ValueError("H5 frozen per-update divisor and schedule disagree")
        if record.get("schedule_sha256") != schedule["schedule_sha256"]:
            raise ValueError("H5 metric schedule identity drift")
        if record.get("schedule_indices") != [row["schedule_index"] for row in group]:
            raise ValueError("H5 metric schedule-index exposure drift")
        if record.get("pack_uids") != [row["pack_uid"] for row in group] or record.get("counts") != expected_counts:
            raise ValueError("H5 metric pack or count accounting drift")
        if record.get("synchronized_timing") is not True or record.get("optimizer_updates") != 1:
            raise ValueError("H5 metric synchronization window drift")
        if record.get("loss", {}).get("global_assistant_target_divisor") != expected_counts["assistant_targets"]:
            raise ValueError("H5 global assistant-target divisor drift")
        loss = record.get("loss", {}).get("normalized_loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
            raise ValueError("H5 normalized loss is invalid")
        applied = record.get("optimizer", {}).get("applied_learning_rates")
        if (
            not isinstance(applied, list)
            or len(applied) != 1
            or not math.isclose(float(applied[0]), expected_lrs[index], rel_tol=1e-14, abs_tol=0.0)
        ):
            raise ValueError(f"H5 applied learning-rate drift at step {index + 1}")
        audits = record.get("selected_output_audits")
        if not isinstance(audits, list) or len(audits) != 8:
            raise ValueError("H5 omitted an every-forward selected-output audit")
        for audit_row, schedule_row in zip(audits, group, strict=True):
            if (
                audit_row.get("schedule_index") != schedule_row["schedule_index"]
                or audit_row.get("pack_uid") != schedule_row["pack_uid"]
            ):
                raise ValueError("H5 selected-output audit identity drift")
            if audit_row.get("loss_dtype") != "torch.float32":
                raise ValueError("H5 selected-output loss dtype drift")
            recomputed = validate_forward_loss_audit(
                audit_row.get("audit"),
                expected_selected_rows=schedule_row["assistant_targets"],
                expected_global_target_count=expected_counts["assistant_targets"],
                expected_chunk_size=H5_SELECTED_CHUNK_SIZE,
            )
            if audit_row.get("validation") != recomputed:
                raise ValueError("H5 producer forward-audit validation drift")
        all_audits.extend(audits)
        gradients = record.get("gradient_dtype_audits")
        if not isinstance(gradients, list) or len(gradients) != 1:
            raise ValueError("H5 omitted a gradient dtype audit")
        gradient = gradients[0]
        if gradient.get("missing_gradient_count") != 0 or gradient.get("gradient_tensor_count", 0) <= 0:
            raise ValueError("H5 gradient connectivity audit drift")
        if gradient.get("gradient_dtype_counts") != {"torch.float32": gradient["gradient_tensor_count"]}:
            raise ValueError("H5 gradient dtype audit drift")
        all_gradient_audits.extend(gradients)
    independent_summary = summarize_reporting_records(records)
    for key, value in independent_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"H5 exact-metrics summary drift for {key}")
    audit_sha = hashlib.sha256(canonical_json_bytes(all_audits)).hexdigest()
    gradient_sha = hashlib.sha256(canonical_json_bytes(all_gradient_audits)).hexdigest()
    if summary.get("selected_output_audit_count") != 40 or summary.get("selected_output_audits_sha256") != audit_sha:
        raise ValueError("H5 selected-output audit aggregate drift")
    if summary.get("gradient_dtype_audit_count") != 5 or summary.get("gradient_dtype_audits_sha256") != gradient_sha:
        raise ValueError("H5 gradient audit aggregate drift")
    if summary.get("loaded_liger_modules") != []:
        raise AssertionError("H5 trainer imported a forbidden Liger module")
    return {
        "exact_metrics_sha256": sha256_file(metrics_path),
        "exact_metrics_summary_sha256": sha256_file(summary_path),
        "records": records,
        "selected_output_audits_sha256": audit_sha,
        "gradient_dtype_audits_sha256": gradient_sha,
    }


def _validate_preflights(
    root: Path,
    contract_sha: str,
    qualification: dict[str, Any],
    qualification_sha: str,
    amendment_sha: str,
    amendment_r2_sha: str,
) -> dict[str, str]:
    paths = {
        "schedule_sharding": root / "h5_accelerate_schedule_sharding_preflight.json",
        "ddp_normalization": root / "h5_non_liger_ddp_normalization_preflight.json",
    }
    values = {}
    for label, path in paths.items():
        _require_file(path)
        values[label] = load_strict_json(path)
        require_finite_json(values[label], context=f"H5.preflight.{label}")
        if (
            values[label].get("status") != "passed"
            or values[label].get("contract_sha256") != contract_sha
            or values[label].get("harness_amendment_sha256") != amendment_sha
            or values[label].get("harness_amendment_r2_sha256") != amendment_r2_sha
        ):
            raise ValueError(f"H5 {label} preflight identity/status drift")
        if values[label].get("loaded_liger_modules") != [] or values[label].get("world_size") != H5_WORLD_SIZE:
            raise ValueError(f"H5 {label} preflight world-size/Liger drift")
    sharding = values["schedule_sharding"]
    if sharding.get("even_batches") is not False or sharding.get("global_indices_exactly_once") != list(range(40)):
        raise ValueError("H5 Accelerate sharding proof drift")
    if sharding.get("rank_indices") != [list(range(rank, 40, 4)) for rank in range(4)]:
        raise ValueError("H5 Accelerate rank-wise stride drift")
    ddp = values["ddp_normalization"]
    if (
        ddp.get("chunk_size") != H5_SELECTED_CHUNK_SIZE
        or ddp.get("implementation_id") != IMPLEMENTATION_ID
        or ddp.get("includes_zero_target_rank") is not True
        or ddp.get("per_rank_assistant_targets") != [0, 127, 513, 1025]
        or ddp.get("qualification_manifest_sha256") != qualification_sha
        or ddp.get("numerical_acceptance") != qualification["numerical_acceptance"]
    ):
        raise ValueError("H5 non-Liger DDP normalization proof drift")
    if (
        len({row["gradient_sha256"] for row in ddp.get("rank_tensor_hashes", [])}) != 1
        or len({row["parameter_sha256"] for row in ddp.get("rank_tensor_hashes", [])}) != 1
    ):
        raise ValueError("H5 DDP preflight rank tensors are not identical")
    return {label: sha256_file(path) for label, path in paths.items()}


def _validate_run(
    root: Path,
    contract: dict[str, Any],
    qualification_sha: str,
    schedule: dict[str, Any],
    prefix: list[dict[str, Any]],
) -> dict[str, Any]:
    training = root / "training"
    required = {
        "conversion": training / "qwen35_text_conversion_ledger.json",
        "run_manifest": training / "qwen35_run_manifest.json",
        "train_results": training / "train_results.json",
        "update_probe": training / "qwen35_parameter_update_probe.json",
    }
    for path in required.values():
        _require_file(path)
    values = {label: load_strict_json(path) for label, path in required.items()}
    for label, value in values.items():
        require_finite_json(value, context=f"H5.run.{label}")
    run = values["run_manifest"]
    if (
        run.get("model_class") != "Qwen3_5ForCausalLM"
        or run.get("model_config_type") != "qwen3_5_text"
        or run.get("vision_tower_loaded") is not False
    ):
        raise ValueError("H5 text-only Qwen model identity drift")
    if (
        run.get("world_size") != 4
        or run.get("gradient_accumulation_steps") != 2
        or run.get("per_device_train_batch_size") != 1
    ):
        raise ValueError("H5 distributed batch geometry drift")
    if run.get("sequence_length") != 32768 or run.get("effective_tokens_per_optimizer_step") != 262144:
        raise ValueError("H5 fixed-token geometry drift")
    if run.get("frozen_data_validation", {}).get("arm_id") != "C00":
        raise ValueError("H5 did not consume C00")
    if run.get("schedule_validation", {}).get("schedule_sha256") != schedule["schedule_sha256"]:
        raise ValueError("H5 run schedule identity drift")
    hardware = run.get("hardware_qualification", {})
    if hardware != {
        "protocol_id": "qwen35-hardware-qualification-r18",
        "manifest_path": str(Path(run.get("hardware_qualification", {}).get("manifest_path", "")).resolve()),
        "manifest_sha256": qualification_sha,
        "hardware_profile": True,
        "cuda_event_step_timing": True,
        "require_no_dense_logits": True,
    }:
        raise ValueError("H5 R18 runtime/hardware flag identity drift")
    if run.get("h4_qualification") is not None:
        raise ValueError("H5 must not masquerade as an H4 assay")
    selective = run.get("selective_output_projection", {})
    if (
        selective.get("enabled") is not True
        or selective.get("implementation") != IMPLEMENTATION_ID
        or selective.get("chunk_size") != 512
        or selective.get("liger_status") != "abandoned_after_r17"
    ):
        raise ValueError("H5 selected-row implementation drift")
    train_args = run.get("training_arguments", {})
    expected = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "data_seed": 3407,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "learning_rate": 2e-5,
        "max_grad_norm": 1.0,
        "max_steps": 10,
        "optim": "adamw_torch_fused",
        "per_device_train_batch_size": 1,
        "seed": 3407,
        "stop_after_steps": 5,
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,
    }
    for key, expected_value in expected.items():
        if train_args.get(key) != expected_value:
            raise ValueError(f"H5 training argument drift for {key}")
    scheduler = train_args.get("lr_scheduler_type")
    if scheduler not in ("cosine", {"name": "cosine"}):
        raise ValueError("H5 scheduler implementation drift")
    conversion = values["conversion"]
    if (
        conversion.get("target_class") != "Qwen3_5ForCausalLM"
        or conversion.get("target_config_model_type") != "qwen3_5_text"
        or conversion.get("tied_input_output_embeddings") is not True
        or conversion.get("tensor_hashes_enabled") is not True
    ):
        raise ValueError("H5 conversion ledger drift")
    if not conversion.get("rows") or any(not row.get("tensor_sha256") for row in conversion["rows"]):
        raise ValueError("H5 conversion ledger lacks tensor hashes")
    update = values["update_probe"]
    if (
        update.get("status") != "passed"
        or update.get("observed_initial_global_step") != 0
        or update.get("final_global_step") != 5
        or update.get("optimizer_steps_observed") != 5
    ):
        raise ValueError("H5 update probe step drift")
    if (
        update.get("parameter_comparison", {}).get("changed_sampled_values", 0) <= 0
        or update.get("parameter_comparison", {}).get("max_absolute_delta", 0) <= 0
    ):
        raise ValueError("H5 update probe found no parameter displacement")
    metrics = _validate_metrics(training, schedule, prefix)
    return {
        "artifact_sha256": {label: sha256_file(path) for label, path in required.items()},
        "conversion_rows_sha256": conversion.get("rows_sha256"),
        "metrics": {key: value for key, value in metrics.items() if key != "records"},
    }


def _validate_hardware(
    root: Path,
    contract_sha: str,
    qualification: dict[str, Any],
    qualification_sha: str,
    amendment_sha: str,
    amendment_r2_sha: str,
) -> dict[str, Any]:
    training = root / "training"
    profile_path = training / "qwen35_cuda_hardware_profile.json"
    trace_path = training / "qwen35_cuda_profiler_trace.json"
    sanitized_trace_path = training / "qwen35_cuda_profiler_trace_sanitized.json"
    snapshot_path = training / "qwen35_cuda_memory_snapshot.pickle"
    nccl_path = root / "h5_nccl_exact_event_catalog.json"
    for path in (profile_path, trace_path, sanitized_trace_path, snapshot_path, nccl_path):
        _require_file(path)
    profile = load_strict_json(profile_path)
    expected_profile = {
        "artifact": "qwen35_cuda_hardware_profile",
        "status": "captured_pending_kernel_audit",
        "qualification_manifest_sha256": qualification_sha,
        "h4_protocol_id": None,
        "h4_contract_sha256": None,
        "candidate_chunk_size": 512,
        "completed_optimizer_steps": 5,
        "warmup_optimizer_steps": 1,
        "measured_optimizer_steps": 4,
        "cuda_device_name": LEONARDO_A100_NAME,
        "cuda_device_capability": [8, 0],
    }
    for key, expected in expected_profile.items():
        if profile.get(key) != expected:
            raise ValueError(f"H5 hardware profile drift for {key}")
    if profile.get("profiler_schedule") != {"wait": 0, "warmup": 1, "active": 4, "repeat": 1, "skip_first": 0}:
        raise ValueError("H5 profiler schedule drift")
    if profile.get("trace_sha256") != sha256_file(trace_path) or profile.get("memory_snapshot_sha256") != sha256_file(
        snapshot_path
    ):
        raise ValueError("H5 profiler raw-artifact binding drift")
    memory = profile.get("memory", {})
    recomputed = validate_memory_headroom(
        peak_allocated_bytes=int(memory.get("peak_allocated_bytes", -1)),
        peak_reserved_bytes=int(memory.get("peak_reserved_bytes", -1)),
        total_device_bytes=int(memory.get("total_device_bytes", -1)),
        acceptance=qualification["memory_acceptance"],
    )
    if memory != recomputed or profile.get("allocator") != {"num_alloc_retries": 0, "num_ooms": 0}:
        raise ValueError("H5 memory or allocator evidence drift")
    expected_step_keys = {str(step) for step in range(1, 6)}
    event_times = profile.get("cuda_event_step_milliseconds")
    if (
        not isinstance(event_times, dict)
        or set(event_times) != expected_step_keys
        or any(not math.isfinite(float(value)) or float(value) <= 0 for value in event_times.values())
    ):
        raise ValueError("H5 rank-zero profiler CUDA-event timing drift")
    per_step = profile.get("per_step_memory", {})
    allocated = per_step.get("peak_allocated_bytes")
    reserved = per_step.get("peak_reserved_bytes")
    if not isinstance(allocated, dict) or not isinstance(reserved, dict):
        raise ValueError("H5 per-step memory evidence is absent")
    if set(allocated) != expected_step_keys or set(reserved) != expected_step_keys:
        raise ValueError("H5 per-step memory step set drift")
    if (
        max(allocated.values()) != memory["peak_allocated_bytes"]
        or max(reserved.values()) != memory["peak_reserved_bytes"]
    ):
        raise ValueError("H5 aggregate memory does not equal the five-step maximum")
    legacy_aggregation_label = "maximum_across_all_four_steps_after_exact_metrics_window_resets"
    if per_step.get("aggregation") != legacy_aggregation_label:
        raise ValueError("H5 raw profiler aggregation-label ABI drift")
    nccl = load_strict_json(nccl_path)
    if (
        nccl.get("status") != "passed"
        or nccl.get("contract_sha256") != contract_sha
        or nccl.get("harness_amendment_sha256") != amendment_sha
        or nccl.get("harness_amendment_r2_sha256") != amendment_r2_sha
    ):
        raise ValueError("H5 NCCL catalog status/contract drift")
    sanitization = nccl.get("trace_sanitization", {})
    if (
        nccl.get("world_size") != 4
        or nccl.get("trace_sha256") != sha256_file(trace_path)
        or nccl.get("catalog_trace_sha256") != sha256_file(sanitized_trace_path)
        or sanitization.get("raw_trace_sha256") != sha256_file(trace_path)
        or sanitization.get("sanitized_trace_sha256") != sha256_file(sanitized_trace_path)
        or sanitization.get("raw_trace_bytes") != trace_path.stat().st_size
        or sanitization.get("sanitized_trace_bytes") != sanitized_trace_path.stat().st_size
        or sanitization.get("replacement_size_delta_bytes_each") != 4
        or sanitization.get("generic_empty_json_values_remaining") != 0
        or nccl.get("collective_complete_event_count", 0) <= 0
    ):
        raise ValueError("H5 NCCL event evidence drift")
    timings = []
    for rank in range(4):
        path = training / f"qwen35_cuda_step_times_rank{rank:02d}.json"
        _require_file(path)
        value = load_strict_json(path)
        if (
            value.get("rank") != rank
            or value.get("world_size") != 4
            or value.get("candidate_chunk_size") != 512
            or value.get("completed_optimizer_steps") != 5
            or value.get("h4_protocol_id") is not None
            or value.get("timing_scope") != CUDA_TIMING_SCOPE
        ):
            raise ValueError(f"H5 CUDA event timing drift on rank {rank}")
        durations = value.get("cuda_event_step_milliseconds")
        if (
            not isinstance(durations, dict)
            or set(durations) != {str(step) for step in range(1, 6)}
            or any(not math.isfinite(float(item)) or float(item) <= 0 for item in durations.values())
        ):
            raise ValueError(f"H5 invalid CUDA event duration on rank {rank}")
        timings.append({"rank": rank, "sha256": sha256_file(path)})
    debug_logs = sorted(root.glob("nccl_debug.*.log"))
    if len(debug_logs) < H5_WORLD_SIZE:
        raise ValueError(f"H5 retained only {len(debug_logs)} NCCL debug logs; expected at least four")
    debug_rows = []
    for path in debug_logs:
        _require_file(path)
        contents = path.read_text(errors="replace")
        if "NCCL INFO" not in contents:
            raise ValueError(f"H5 NCCL debug log lacks INFO evidence: {path.name}")
        match = NCCL_FAILURE_PATTERN.search(contents)
        if match:
            raise AssertionError(f"H5 NCCL debug failure marker in {path.name}: {match.group(0)!r}")
        debug_rows.append({"bytes": path.stat().st_size, "name": path.name, "sha256": sha256_file(path)})
    return {
        "cuda_event_timings": timings,
        "nccl_debug_logs": debug_rows,
        "hardware_profile_sha256": sha256_file(profile_path),
        "memory_aggregation_semantics": {
            "authoritative_step_set": [1, 2, 3, 4, 5],
            "independent_interpretation": "maximum_across_all_five_H5_optimizer_steps",
            "raw_legacy_label": legacy_aggregation_label,
        },
        "memory": recomputed,
        "memory_snapshot_sha256": sha256_file(snapshot_path),
        "nccl_catalog_sha256": sha256_file(nccl_path),
        "sanitized_trace_sha256": sha256_file(sanitized_trace_path),
        "trace_sha256": sha256_file(trace_path),
    }


def _validate_checkpoint_evidence(
    root: Path, contract_sha: str, amendment_sha: str, amendment_r2_sha: str
) -> dict[str, str]:
    training = root / "training"
    manifest_path = root / "h5_checkpoint_5_file_manifest.json"
    reload_path = root / "h5_checkpoint_5_reload_validation.json"
    for path in (manifest_path, reload_path):
        _require_file(path)
    root_files = {
        "config": training / "config.json",
        "model": training / "model.safetensors",
        "trainer_state": training / "trainer_state.json",
    }
    for path in root_files.values():
        _require_file(path)
    root_state = load_strict_json(root_files["trainer_state"])
    if root_state.get("global_step") != 5 or root_state.get("max_steps") != 10:
        raise ValueError("H5 root Trainer state drift")
    manifest = load_strict_json(manifest_path)
    reload = load_strict_json(reload_path)
    if (
        manifest.get("status") != "passed"
        or manifest.get("contract_sha256") != contract_sha
        or manifest.get("harness_amendment_sha256") != amendment_sha
        or manifest.get("harness_amendment_r2_sha256") != amendment_r2_sha
        or not manifest.get("files")
    ):
        raise ValueError("H5 checkpoint file manifest drift")
    checkpoint = training / "checkpoint-5"
    for row in manifest["files"]:
        path = checkpoint / row["path"]
        _require_file(path)
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"H5 checkpoint file drift: {row['path']}")
    if (
        reload.get("status") != "passed"
        or reload.get("contract_sha256") != contract_sha
        or reload.get("harness_amendment_sha256") != amendment_sha
        or reload.get("harness_amendment_r2_sha256") != amendment_r2_sha
        or reload.get("checkpoint_file_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("H5 checkpoint reload report drift")
    if (
        reload.get("trainer_global_step") != 5
        or reload.get("trainer_max_steps") != 10
        or reload.get("scheduler_last_epoch") != 5
    ):
        raise ValueError("H5 checkpoint semantic state drift")
    if any(reload.get("loading_info", {}).values()) or reload.get("loaded_liger_modules") != []:
        raise ValueError("H5 checkpoint strict reload or Liger drift")
    if (
        reload.get("tiny_cuda_forward", {}).get("finite") is not True
        or reload.get("model", {}).get("tied_storage") is not True
    ):
        raise ValueError("H5 checkpoint tiny-forward/tied-embedding drift")
    return {
        "file_manifest_sha256": sha256_file(manifest_path),
        "reload_validation_sha256": sha256_file(reload_path),
        "root_file_sha256": {label: sha256_file(path) for label, path in root_files.items()},
    }


def _validate_identity(
    args: argparse.Namespace, contract_sha: str, qualification_sha: str, amendment_r2_sha: str
) -> dict[str, Any]:
    path = args.output_dir / "h5_immutable_input_and_hardware_inventory.json"
    _require_file(path)
    identity = load_strict_json(path)
    if (
        identity.get("artifact") != "qwen35_r18_h5_immutable_input_and_hardware_inventory"
        or identity.get("status") != "passed_identity_capture_only"
    ):
        raise ValueError("H5 immutable input inventory schema/status drift")
    if (
        identity.get("contract_sha256") != contract_sha
        or identity.get("qualification_manifest_sha256") != qualification_sha
        or identity.get("harness_amendment_sha256") != sha256_file(args.harness_amendment)
        or identity.get("harness_amendment_r2_sha256") != amendment_r2_sha
    ):
        raise ValueError("H5 immutable input inventory parent drift")
    if (
        identity.get("git", {}).get("commit") != args.expected_source_head
        or identity.get("git", {}).get("status_porcelain") != ""
    ):
        raise ValueError("H5 immutable input source drift")
    if identity.get("slurm", {}).get("account") != "aifac_f02_434" or identity.get("loaded_liger_modules") != []:
        raise ValueError("H5 immutable input account/Liger drift")
    if identity.get("source_python_bytecode_files") != []:
        raise ValueError("H5 staged source contains bytecode")
    if len(identity.get("gpu_inventory", [])) != 4 or len({row.get("uuid") for row in identity["gpu_inventory"]}) != 4:
        raise ValueError("H5 immutable input GPU inventory drift")
    expected_hashes = {
        **EXPECTED_PREDECESSORS,
        "h5_contract": contract_sha,
        "h5_human_protocol": sha256_file(args.human_protocol),
        "h5_preregistration_closure": sha256_file(args.preregistration_closure),
        "h5_harness_amendment": sha256_file(args.harness_amendment),
        "h5_harness_human_amendment": sha256_file(args.harness_human_amendment),
        "h5_attempt01_failure_closure": sha256_file(args.attempt01_failure_closure),
        "h5_harness_amendment_r2": sha256_file(args.harness_amendment_r2),
        "h5_harness_human_amendment_r2": sha256_file(args.harness_human_amendment_r2),
        "h5_attempt02_failure_closure": sha256_file(args.attempt02_failure_closure),
        "h5_reload_type_diagnostic": sha256_file(args.reload_type_diagnostic),
        "model_manifest": sha256_file(args.model_manifest),
        "numpy_manifest": sha256_file(args.numpy_data / "manifest.json"),
        "qualification_manifest": qualification_sha,
        "runtime_report": sha256_file(args.runtime_report),
        "schedule": H5_SCHEDULE_FILE_SHA256,
        "source_code_manifest": sha256_file(args.source_code_manifest),
    }
    for label, expected in expected_hashes.items():
        if identity.get("file_identities", {}).get(label, {}).get("sha256") != expected:
            raise ValueError(f"H5 immutable input file identity drift for {label}")
    return identity


def producer(args: argparse.Namespace) -> dict[str, Any]:
    contract, contract_sha, qualification, qualification_sha, amendment_sha, amendment_r2_sha = _load_context(args)
    schedule, prefix = _validate_schedule(args, contract)
    source_delta = validate_h5_source_delta(
        args.source_repository,
        expected_head=args.expected_source_head,
        harness_amendment_path=args.harness_amendment,
        harness_human_amendment_path=args.harness_human_amendment,
        attempt01_failure_closure_path=args.attempt01_failure_closure,
        harness_amendment_r2_path=args.harness_amendment_r2,
        harness_human_amendment_r2_path=args.harness_human_amendment_r2,
        attempt02_failure_closure_path=args.attempt02_failure_closure,
        reload_type_diagnostic_path=args.reload_type_diagnostic,
    )
    _verify_code_manifest(args.source_repository, args.source_code_manifest)
    identity = _validate_identity(args, contract_sha, qualification_sha, amendment_r2_sha)
    preflights = _validate_preflights(
        args.output_dir, contract_sha, qualification, qualification_sha, amendment_sha, amendment_r2_sha
    )
    run = _validate_run(args.output_dir, contract, qualification_sha, schedule, prefix)
    hardware = _validate_hardware(
        args.output_dir, contract_sha, qualification, qualification_sha, amendment_sha, amendment_r2_sha
    )
    checkpoint = _validate_checkpoint_evidence(args.output_dir, contract_sha, amendment_sha, amendment_r2_sha)
    return {
        "artifact": "qwen35_r18_h5_producer_validation",
        "checkpoint": checkpoint,
        "contract_sha256": contract_sha,
        "harness_amendment_sha256": amendment_sha,
        "harness_amendment_r2_sha256": amendment_r2_sha,
        "hardware": hardware,
        "identity_sha256": sha256_file(args.output_dir / "h5_immutable_input_and_hardware_inventory.json"),
        "preflights": preflights,
        "qualification_manifest_sha256": qualification_sha,
        "run": run,
        "schedule": {
            "file_sha256": H5_SCHEDULE_FILE_SHA256,
            "five_update_entries_sha256": H5_FIRST_FIVE_ENTRIES_SHA256,
            "observed_indices": list(range(40)),
            "per_update_assistant_targets": list(H5_EXPECTED_TARGETS_BY_UPDATE),
        },
        "schema_version": 1,
        "scientific_training_authorized": False,
        "slurm_job_id": identity["slurm"]["job_id"],
        "source_delta": source_delta,
        "status": "producer_passed_pending_slurm_and_independent_closure",
        "successor_authorized": None,
    }


def independent(args: argparse.Namespace) -> dict[str, Any]:
    if args.producer_validation is None or args.slurm_record is None:
        raise ValueError("independent mode requires --producer-validation and --slurm-record")
    recomputed = producer(args)
    producer_report = load_strict_json(args.producer_validation)
    if producer_report != recomputed:
        raise ValueError("H5 producer report differs from independent recomputation")
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
            raise ValueError(f"H5 Slurm completion drift for {key}")
    if str(slurm.get("job_id")) != str(recomputed["slurm_job_id"]):
        raise ValueError("H5 Slurm job identity drift")
    for label in ("stdout", "stderr"):
        path_value = slurm.get(f"{label}_path")
        if not isinstance(path_value, str):
            raise ValueError(f"H5 Slurm record lacks {label} path")
        path = Path(path_value)
        _require_regular_file(path)
        if slurm.get(f"{label}_sha256") != sha256_file(path):
            raise ValueError(f"H5 Slurm {label} digest drift")
        contents = path.read_text(errors="replace")
        match = NCCL_FAILURE_PATTERN.search(contents)
        if match:
            raise AssertionError(f"H5 Slurm {label} contains an NCCL/process failure marker: {match.group(0)!r}")
        if label == "stdout" and "R18_H5_PRODUCER_PASSED_PENDING_SLURM_AND_INDEPENDENT_CLOSURE" not in contents:
            raise ValueError("H5 Slurm stdout lacks the terminal producer success marker")
    exit_path = args.output_dir / "g2_job_exit.json"
    _require_file(exit_path)
    exit_report = load_strict_json(exit_path)
    if exit_report != {"exit_code": 0, "slurm_job_id": str(slurm["job_id"])}:
        raise ValueError("H5 wrapper exit record drift")
    return {
        "artifact": "qwen35_r18_h5_independent_closure",
        "contract_sha256": recomputed["contract_sha256"],
        "harness_amendment_sha256": recomputed["harness_amendment_sha256"],
        "harness_amendment_r2_sha256": recomputed["harness_amendment_r2_sha256"],
        "producer_validation_sha256": sha256_file(args.producer_validation),
        "schema_version": 1,
        "scientific_training_authorized": False,
        "slurm_record_sha256": sha256_file(args.slurm_record),
        "status": "passed_H6_only_authorized",
        "successor_authorized": "H6_only",
    }


def main() -> int:
    args = parse_args()
    if args.report_output.exists():
        raise FileExistsError(args.report_output)
    if args.mode == "capture":
        report = capture(args)
    elif args.mode == "producer":
        report = producer(args)
    else:
        report = independent(args)
    require_finite_json(report, context=f"H5.{args.mode}.report")
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
