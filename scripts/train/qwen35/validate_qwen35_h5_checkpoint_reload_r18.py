#!/usr/bin/env python3
"""Validate, hash, strictly reload, and execute the R18 H5 step-five checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file
from open_instruct.qwen35_qualification_r18_h5 import (
    H5_FINAL_STEP,
    H5_SCHEDULER_HORIZON,
    load_h5_contract,
    load_h5_harness_amendment,
    load_h5_harness_amendment_r2,
)
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--file-manifest-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing, empty, or symlinked H5 checkpoint file: {path}")


def normalize_loading_info(loading_info: object) -> dict[str, list[str]]:
    """Normalize pinned Transformers loading-info containers without weakening strict reload."""

    expected_keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if not isinstance(loading_info, dict) or set(loading_info) != set(expected_keys):
        raise TypeError("R18 H5 loading_info must be a mapping with exactly the four pinned keys")
    normalized: dict[str, list[str]] = {}
    accepted_types = (list, tuple, set, frozenset)
    for key in expected_keys:
        values = loading_info[key]
        if type(values) not in accepted_types:
            raise TypeError(f"R18 H5 loading_info[{key!r}] has an unsupported container type: {type(values).__name__}")
        rows = list(values)
        if any(type(row) is not str for row in rows):
            raise TypeError(f"R18 H5 loading_info[{key!r}] contains a non-string entry")
        if len(rows) != len(set(rows)):
            raise ValueError(f"R18 H5 loading_info[{key!r}] contains duplicate entries")
        normalized[key] = sorted(rows)
    return normalized


def main() -> None:
    args = parse_args()
    _, contract_sha256 = load_h5_contract(
        args.h5_contract,
        human_protocol_path=args.human_protocol,
        preregistration_closure_path=args.preregistration_closure,
    )
    _, amendment_sha256 = load_h5_harness_amendment(
        args.harness_amendment,
        human_amendment_path=args.harness_human_amendment,
        attempt01_failure_closure_path=args.attempt01_failure_closure,
    )
    _, amendment_r2_sha256 = load_h5_harness_amendment_r2(
        args.harness_amendment_r2,
        human_amendment_path=args.harness_human_amendment_r2,
        attempt02_failure_closure_path=args.attempt02_failure_closure,
        reload_type_diagnostic_path=args.reload_type_diagnostic,
    )
    checkpoint = args.checkpoint.resolve()
    if checkpoint.name != f"checkpoint-{H5_FINAL_STEP}":
        raise ValueError("R18 H5 reload assay requires checkpoint-5")
    if args.report_output.exists() or args.file_manifest_output.exists():
        raise FileExistsError("R18 H5 checkpoint validation output already exists")
    required = {
        "config": checkpoint / "config.json",
        "model": checkpoint / "model.safetensors",
        "optimizer": checkpoint / "optimizer.pt",
        "scheduler": checkpoint / "scheduler.pt",
        "tokenizer": checkpoint / "tokenizer.json",
        "tokenizer_config": checkpoint / "tokenizer_config.json",
        "trainer_state": checkpoint / "trainer_state.json",
    }
    rng_files = [checkpoint / f"rng_state_{rank}.pth" for rank in range(4)]
    for path in [*required.values(), *rng_files]:
        require_file(path)
    state = load_strict_json(required["trainer_state"])
    if state.get("global_step") != H5_FINAL_STEP or state.get("max_steps") != H5_SCHEDULER_HORIZON:
        raise ValueError("R18 H5 checkpoint Trainer step or scheduler horizon drift")

    model_dtype_counts: dict[str, int] = {}
    model_tensor_count = 0
    with safe_open(required["model"], framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 - safe_open exposes keys(), not ordinary iteration
            dtype = handle.get_slice(key).get_dtype()
            model_dtype_counts[dtype] = model_dtype_counts.get(dtype, 0) + 1
            model_tensor_count += 1
    if model_tensor_count <= 0 or model_dtype_counts != {"F32": model_tensor_count}:
        raise ValueError(f"R18 H5 checkpoint parameter dtype drift: {model_dtype_counts}")

    optimizer_state = torch.load(required["optimizer"], map_location="cpu", weights_only=True)
    optimizer_state_rows = optimizer_state.get("state", {})
    if not isinstance(optimizer_state_rows, dict) or not optimizer_state_rows:
        raise ValueError("R18 H5 checkpoint has no initialized AdamW state")
    moment_dtype_counts: dict[str, int] = {}
    moment_tensors = 0
    optimizer_step_values = set()
    state_rows_with_both_moments = 0
    for parameter_state in optimizer_state_rows.values():
        present = 0
        for name in ("exp_avg", "exp_avg_sq"):
            value = parameter_state.get(name)
            if value is None:
                continue
            present += 1
            moment_tensors += 1
            dtype = str(value.dtype)
            moment_dtype_counts[dtype] = moment_dtype_counts.get(dtype, 0) + 1
        if present == 2:
            state_rows_with_both_moments += 1
        step = parameter_state.get("step")
        if torch.is_tensor(step):
            optimizer_step_values.add(float(step.item()))
    if state_rows_with_both_moments != len(optimizer_state_rows):
        raise ValueError("R18 H5 checkpoint AdamW state row is missing a moment")
    if moment_dtype_counts != {"torch.float32": moment_tensors} or moment_tensors <= 0:
        raise ValueError(f"R18 H5 checkpoint AdamW moment dtype drift: {moment_dtype_counts}")
    if optimizer_step_values != {float(H5_FINAL_STEP)}:
        raise ValueError(f"R18 H5 optimizer step counter drift: {sorted(optimizer_step_values)}")
    scheduler_state = torch.load(required["scheduler"], map_location="cpu", weights_only=True)
    if scheduler_state.get("last_epoch") != H5_FINAL_STEP:
        raise ValueError("R18 H5 checkpoint scheduler counter drift")

    rows = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_dir():
            continue
        require_file(path)
        rows.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(checkpoint).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    file_manifest = {
        "artifact": "qwen35_r18_h5_checkpoint_file_manifest",
        "checkpoint": str(checkpoint),
        "contract_sha256": contract_sha256,
        "files": rows,
        "harness_amendment_r2_sha256": amendment_r2_sha256,
        "harness_amendment_sha256": amendment_sha256,
        "schema_version": 1,
        "status": "passed",
    }
    write_json_atomic(args.file_manifest_output, file_manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("R18 H5 checkpoint reload assay requires CUDA")
    loaded_liger_before = sorted(
        name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
    )
    if loaded_liger_before:
        raise RuntimeError(f"R18 H5 checkpoint reload imported forbidden Liger: {loaded_liger_before}")
    from transformers import Qwen3_5ForCausalLM  # noqa: PLC0415 - keep pure normalization tests runtime-independent

    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="flash_attention_2",
        output_loading_info=True,
    )
    normalized_loading_info = normalize_loading_info(loading_info)
    if any(normalized_loading_info.values()):
        raise ValueError(f"R18 H5 strict checkpoint reload key drift: {normalized_loading_info}")
    parameter_dtypes: dict[str, int] = {}
    parameter_count = 0
    parameter_numel = 0
    for parameter in model.parameters():
        parameter_count += 1
        parameter_numel += parameter.numel()
        dtype = str(parameter.dtype)
        parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + 1
    if parameter_dtypes != {"torch.float32": parameter_count}:
        raise ValueError(f"R18 H5 reloaded parameter dtype drift: {parameter_dtypes}")
    input_embedding = model.get_input_embeddings().weight
    output_embedding = model.get_output_embeddings().weight
    tied = input_embedding.data_ptr() == output_embedding.data_ptr()
    if not tied or model.config.tie_word_embeddings is not True:
        raise ValueError("R18 H5 reloaded checkpoint does not retain tied embeddings")
    if model.__class__.__name__ != "Qwen3_5ForCausalLM" or model.config.model_type != "qwen3_5_text":
        raise ValueError("R18 H5 reloaded checkpoint class or model type drift")
    if hasattr(model, "visual") or hasattr(model, "vision_tower"):
        raise ValueError("R18 H5 reloaded checkpoint unexpectedly contains a vision tower")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1, return_dict=True
        )
    logits = output.logits
    if logits is None or list(logits.shape) != [1, 1, model.config.vocab_size] or not torch.isfinite(logits).all():
        raise AssertionError("R18 H5 tiny checkpoint-reload forward is missing, malformed, or non-finite")
    tiny = {
        "dtype": str(logits.dtype),
        "finite": True,
        "maximum": float(logits.float().max()),
        "minimum": float(logits.float().min()),
        "shape": list(logits.shape),
    }
    if not math.isfinite(tiny["maximum"]) or not math.isfinite(tiny["minimum"]):
        raise AssertionError("R18 H5 tiny checkpoint-reload forward extrema are non-finite")
    loaded_liger_after = sorted(
        name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
    )
    if loaded_liger_after:
        raise RuntimeError(f"R18 H5 checkpoint reload imported forbidden Liger: {loaded_liger_after}")

    report = {
        "adamw": {
            "moment_dtype_counts": moment_dtype_counts,
            "moment_tensors": moment_tensors,
            "optimizer_state_rows": len(optimizer_state_rows),
            "state_rows_with_both_moments": state_rows_with_both_moments,
            "step_values": sorted(optimizer_step_values),
        },
        "artifact": "qwen35_r18_h5_checkpoint_reload_validation",
        "checkpoint": str(checkpoint),
        "checkpoint_file_manifest_sha256": sha256_file(args.file_manifest_output),
        "contract_sha256": contract_sha256,
        "cuda_device": torch.cuda.get_device_name(device),
        "loaded_liger_modules": loaded_liger_after,
        "loading_info": normalized_loading_info,
        "harness_amendment_r2_sha256": amendment_r2_sha256,
        "harness_amendment_sha256": amendment_sha256,
        "model": {
            "class": model.__class__.__name__,
            "model_type": model.config.model_type,
            "parameter_count": parameter_count,
            "parameter_dtypes": parameter_dtypes,
            "parameter_numel": parameter_numel,
            "safetensors_dtype_counts": model_dtype_counts,
            "safetensors_tensor_count": model_tensor_count,
            "tie_word_embeddings": model.config.tie_word_embeddings,
            "tied_storage": tied,
        },
        "scheduler_last_epoch": scheduler_state["last_epoch"],
        "schema_version": 1,
        "status": "passed",
        "tiny_cuda_forward": tiny,
        "trainer_global_step": state["global_step"],
        "trainer_max_steps": state["max_steps"],
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
