#!/usr/bin/env python3
"""Semantically compare continuous and stop/resume Qwen3.5 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--resumed-checkpoint", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    """Load trusted checkpoint structures without enabling arbitrary pickle."""

    # HF Trainer RNG checkpoints contain NumPy's MT19937 uint32 array. Torch's
    # weights-only unpickler rejects that array unless these exact constructors
    # and dtype classes are allowlisted. Keep unrestricted weights_only=False
    # prohibited even though these checkpoints are produced by our own jobs.
    numpy_safe_globals = [
        np._core.multiarray._reconstruct,  # noqa: SLF001 - exact NumPy pickle constructor
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(numpy_safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def safetensor_files(root: Path) -> list[Path]:
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model safetensors found in {root}")
    return files


def safetensor_index(root: Path) -> dict[str, Path]:
    result = {}
    for path in safetensor_files(root):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                if key in result:
                    raise ValueError(f"duplicate safetensor key {key!r} in {root}")
                result[key] = path
    return result


def tensor_bit_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare dense CPU representations, including signed zero and NaN payload bits."""

    if left.shape != right.shape or left.dtype != right.dtype or left.layout != torch.strided:
        return False
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def compare_model(left: Path, right: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    left_index = safetensor_index(left)
    right_index = safetensor_index(right)
    if left_index.keys() != right_index.keys():
        raise ValueError("continuous and resumed model tensor-key sets differ")
    maximum_absolute_error = 0.0
    nonidentical_tensors = 0
    dtype_counts: dict[str, int] = {}
    for key in sorted(left_index):
        with safe_open(left_index[key], framework="pt", device="cpu") as left_handle:
            left_tensor = left_handle.get_tensor(key)
        with safe_open(right_index[key], framework="pt", device="cpu") as right_handle:
            right_tensor = right_handle.get_tensor(key)
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise ValueError(f"model tensor shape/dtype drift for {key}")
        dtype_counts[str(left_tensor.dtype)] = dtype_counts.get(str(left_tensor.dtype), 0) + 1
        identical = tensor_bit_equal(left_tensor, right_tensor)
        nonidentical_tensors += not identical
        if left_tensor.is_floating_point():
            error = (
                float((left_tensor.float() - right_tensor.float()).abs().max())
                if left_tensor.numel()
                else 0.0
            )
            maximum_absolute_error = max(maximum_absolute_error, error)
            try:
                torch.testing.assert_close(left_tensor, right_tensor, atol=atol, rtol=rtol)
            except AssertionError as error:
                raise AssertionError(f"model tensor differs at {key}: {error}") from error
        elif not identical:
            raise AssertionError(f"non-floating model tensor differs: {key}")
    return {
        "tensor_count": len(left_index),
        "dtype_counts": dtype_counts,
        "nonidentical_tensors": nonidentical_tensors,
        "maximum_absolute_error": maximum_absolute_error,
        "bit_exact": nonidentical_tensors == 0,
    }


def compare_nested(left: Any, right: Any, *, path: str, atol: float, rtol: float, counters: dict[str, int]) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f"tensor/non-tensor mismatch at {path}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"tensor shape/dtype mismatch at {path}")
        counters["tensors"] += 1
        identical = tensor_bit_equal(left, right)
        if left.is_floating_point():
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=lambda msg: f"{path}: {msg}")
        elif not identical:
            raise AssertionError(f"non-floating tensor differs at {path}")
        counters["nonidentical_tensors"] += not identical
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ValueError(f"array/non-array mismatch at {path}")
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise AssertionError(f"NumPy RNG array differs at {path}")
        counters["numpy_arrays"] += 1
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"mapping-key mismatch at {path}")
        for key in left:
            compare_nested(left[key], right[key], path=f"{path}.{key}", atol=atol, rtol=rtol, counters=counters)
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"sequence mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            compare_nested(left_item, right_item, path=f"{path}[{index}]", atol=atol, rtol=rtol, counters=counters)
        return
    if type(left) is not type(right):
        raise ValueError(f"scalar type mismatch at {path}: {type(left).__name__} != {type(right).__name__}")
    if left != right:
        raise AssertionError(f"value differs at {path}: {left!r} != {right!r}")
    counters["scalars"] += 1


def compare_torch_artifact(left: Path, right: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(f"missing checkpoint artifact {left} or {right}")
    left_value = safe_torch_load(left)
    right_value = safe_torch_load(right)
    left_sha256 = sha256_file(left)
    right_sha256 = sha256_file(right)
    counters = {"tensors": 0, "nonidentical_tensors": 0, "numpy_arrays": 0, "scalars": 0}
    compare_nested(left_value, right_value, path=left.name, atol=atol, rtol=rtol, counters=counters)
    return {
        **counters,
        "bit_exact_tensors": counters["nonidentical_tensors"] == 0,
        "left_file_sha256": left_sha256,
        "right_file_sha256": right_sha256,
        "byte_identical_file": left_sha256 == right_sha256,
    }


EXPECTED_TRAINER_STATE_KEYS = {
    "best_global_step",
    "best_metric",
    "best_model_checkpoint",
    "epoch",
    "eval_steps",
    "global_step",
    "is_hyper_param_search",
    "is_local_process_zero",
    "is_world_process_zero",
    "log_history",
    "logging_steps",
    "max_steps",
    "num_input_tokens_seen",
    "num_train_epochs",
    "save_steps",
    "stateful_callbacks",
    "total_flos",
    "train_batch_size",
    "trial_name",
    "trial_params",
}
PER_STEP_LOG_KEYS = {"epoch", "grad_norm", "learning_rate", "loss", "step"}
OBSERVATIONAL_FINAL_LOG_KEYS = {
    "epoch",
    "step",
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}


def deterministic_log_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact optimizer-step logs while rejecting unknown log shapes."""

    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("trainer log_history must be a list")
    result = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ValueError(f"trainer log_history[{index}] must be a mapping")
        keys = set(row)
        if keys == PER_STEP_LOG_KEYS:
            result.append(row)
        elif keys == OBSERVATIONAL_FINAL_LOG_KEYS:
            continue
        else:
            raise ValueError(f"unexpected trainer log_history[{index}] key set: {sorted(keys)}")
    expected_steps = list(range(1, int(state["global_step"]) + 1))
    if [row["step"] for row in result] != expected_steps:
        raise ValueError("trainer deterministic log rows do not cover each optimizer step exactly once")
    return result


def compare_trainer_state(left: Path, right: Path, *, strict_h6: bool = False) -> dict[str, Any]:
    left_state = load_strict_json(left)
    right_state = load_strict_json(right)
    if strict_h6 and (
        set(left_state) != EXPECTED_TRAINER_STATE_KEYS or set(right_state) != EXPECTED_TRAINER_STATE_KEYS
    ):
        raise ValueError("trainer state top-level key set drift")
    stable_keys = (
        "best_global_step",
        "best_metric",
        "best_model_checkpoint",
        "epoch",
        "eval_steps",
        "global_step",
        "is_hyper_param_search",
        "is_local_process_zero",
        "is_world_process_zero",
        "logging_steps",
        "max_steps",
        "num_input_tokens_seen",
        "num_train_epochs",
        "save_steps",
        "stateful_callbacks",
        "total_flos",
        "train_batch_size",
        "trial_name",
        "trial_params",
    )
    differences = {
        key: [left_state.get(key), right_state.get(key)]
        for key in stable_keys
        if left_state.get(key) != right_state.get(key)
    }
    if differences:
        raise AssertionError(f"trainer state semantic fields differ: {differences}")
    if strict_h6:
        left_history = deterministic_log_history(left_state)
        right_history = deterministic_log_history(right_state)
        if left_history != right_history:
            raise AssertionError("trainer deterministic per-step log history differs")
    else:
        left_history = left_state.get("log_history", [])
        right_history = right_state.get("log_history", [])
    return {
        "stable_keys": list(stable_keys),
        "global_step": left_state.get("global_step"),
        "deterministic_log_history_length": len(left_history),
        "raw_log_history_lengths": [len(left_state["log_history"]), len(right_state["log_history"])],
        "semantic_state_exact": not differences,
        "byte_identical_file": sha256_file(left) == sha256_file(right),
    }


def compare_checkpoints(left: Path, right: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    """Compare all checkpointed semantic state and return a complete report."""

    if atol < 0 or rtol < 0:
        raise ValueError("comparison tolerances must be nonnegative")
    left = left.resolve()
    right = right.resolve()
    model_report = compare_model(left, right, atol=atol, rtol=rtol)
    optimizer_report = compare_torch_artifact(
        left / "optimizer.pt", right / "optimizer.pt", atol=atol, rtol=rtol
    )
    scheduler_report = compare_torch_artifact(
        left / "scheduler.pt", right / "scheduler.pt", atol=atol, rtol=rtol
    )
    left_rng = sorted(left.glob("rng_state*.pth"))
    right_rng = sorted(right.glob("rng_state*.pth"))
    if [path.name for path in left_rng] != [path.name for path in right_rng] or not left_rng:
        raise ValueError("continuous and resumed RNG-state file sets differ")
    rng_report = {
        left_path.name: compare_torch_artifact(left_path, right / left_path.name, atol=0.0, rtol=0.0)
        for left_path in left_rng
    }
    trainer_report = compare_trainer_state(
        left / "trainer_state.json", right / "trainer_state.json", strict_h6=True
    )
    if (atol == 0 and rtol == 0) and (
        not model_report["bit_exact"]
        or not optimizer_report["bit_exact_tensors"]
        or not scheduler_report["bit_exact_tensors"]
        or any(not row["bit_exact_tensors"] for row in rng_report.values())
    ):
        raise AssertionError("zero-tolerance qualification was not bit-exact")
    return {
        "artifact": "qwen35_continuous_resume_checkpoint_comparison",
        "atol": atol,
        "continuous_checkpoint": str(left),
        "model": model_report,
        "optimizer": optimizer_report,
        "resumed_checkpoint": str(right),
        "rng": rng_report,
        "rtol": rtol,
        "scheduler": scheduler_report,
        "schema_version": 1,
        "status": "passed",
        "trainer_state": trainer_report,
    }


def write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    left = args.continuous_checkpoint.resolve()
    right = args.resumed_checkpoint.resolve()
    try:
        report = compare_checkpoints(left, right, atol=args.atol, rtol=args.rtol)
    except BaseException as error:
        failure = {
            "artifact": "qwen35_continuous_resume_checkpoint_comparison_failure",
            "atol": args.atol,
            "continuous_checkpoint": str(left),
            "error_message": str(error),
            "error_type": type(error).__name__,
            "resumed_checkpoint": str(right),
            "rtol": args.rtol,
            "schema_version": 1,
            "status": "failed",
        }
        write_report_atomic(args.report_output, failure)
        raise
    write_report_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
