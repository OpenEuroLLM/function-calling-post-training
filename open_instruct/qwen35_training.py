"""Correctness primitives for the Qwen3.5 text-only selective-loss trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

TEXT_CONVERSION_SCHEMA_VERSION = 1


def conditional_source_key_for_text_target(target_key: str) -> str:
    """Map a text-only CausalLM state key to the multimodal checkpoint key."""

    if target_key.startswith("model."):
        return "model.language_model." + target_key.removeprefix("model.")
    if target_key == "lm_head.weight":
        # The official conditional checkpoint ties the LM head to the text
        # embedding and therefore serializes only this physical source tensor.
        return "model.language_model.embed_tokens.weight"
    raise ValueError(f"unexpected Qwen3.5 text-only state key: {target_key}")


def validate_text_loading_info(loading_info: dict[str, Any]) -> None:
    problems = {}
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading_info.get(key, [])
        if values:
            problems[key] = sorted(values) if not isinstance(values, list) else values
    if problems:
        raise RuntimeError(f"Qwen3.5 text-only checkpoint conversion was incomplete: {problems}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one CPU tensor without creating a second tensor-sized byte string."""

    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(byte_view))
    return digest.hexdigest()


def build_text_conversion_ledger(
    model: torch.nn.Module, *, source_model: str, source_revision: str, hash_tensors: bool
) -> dict[str, Any]:
    state = model.state_dict()
    rows = []
    for target_key in sorted(state):
        tensor = state[target_key]
        rows.append(
            {
                "source_key": conditional_source_key_for_text_target(target_key),
                "target_key": target_key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "tensor_sha256": tensor_sha256(tensor) if hash_tensors else None,
            }
        )
    parameters = dict(model.named_parameters())
    if "model.embed_tokens.weight" not in parameters:
        raise RuntimeError("text-only model has no input embedding parameter")
    if "lm_head.weight" not in parameters:
        # Tied parameters are de-duplicated by named_parameters; inspect modules directly.
        lm_head = getattr(model, "lm_head", None)
        target_weight = getattr(lm_head, "weight", None)
    else:
        target_weight = parameters["lm_head.weight"]
    embed_weight = parameters["model.embed_tokens.weight"]
    if target_weight is None or target_weight.data_ptr() != embed_weight.data_ptr():
        raise RuntimeError("Qwen3.5 text-only input and output embeddings are not tied")

    rows_sha256 = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "artifact": "qwen35_conditional_to_text_conversion_ledger",
        "schema_version": TEXT_CONVERSION_SCHEMA_VERSION,
        "source_model": source_model,
        "source_revision": source_revision,
        "target_class": type(model).__name__,
        "target_config_model_type": model.config.model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensor_count": len(rows),
        "tied_input_output_embeddings": True,
        "tensor_hashes_enabled": hash_tensors,
        "rows_sha256": rows_sha256,
        "rows": rows,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_fp32_trainable_parameters(model: torch.nn.Module) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Qwen3.5 text-only model has no trainable parameters")
    bad = [(name, str(parameter.dtype)) for name, parameter in trainable if parameter.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"trainable parameters must remain FP32; first violations: {bad[:10]}")
    return {
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "parameter_dtype": "torch.float32",
    }


def validate_fp32_optimizer_state(optimizer: torch.optim.Optimizer, *, require_initialized: bool) -> dict[str, Any]:
    parameter_dtypes = set()
    tensor_state_dtypes = set()
    initialized_parameters = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            parameter_dtypes.add(parameter.dtype)
            state = optimizer.state.get(parameter, {})
            if state:
                initialized_parameters += 1
            for key, value in state.items():
                if not torch.is_tensor(value) or key == "step":
                    continue
                tensor_state_dtypes.add(value.dtype)
                if value.dtype != torch.float32:
                    raise RuntimeError(f"optimizer state {key!r} uses {value.dtype}, expected torch.float32")
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(f"optimizer contains non-FP32 trainable parameters: {parameter_dtypes}")
    if require_initialized and initialized_parameters == 0:
        raise RuntimeError("optimizer state was not initialized after an optimizer step")
    return {
        "parameter_dtypes": sorted(str(value) for value in parameter_dtypes),
        "optimizer_tensor_state_dtypes": sorted(str(value) for value in tensor_state_dtypes),
        "initialized_parameter_states": initialized_parameters,
    }


def select_supervised_predecessor_rows(
    hidden_states: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predecessor rows, target IDs, and sequence positions exactly."""

    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected hidden_states [B,S,H] and labels [B,S]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden-state and label batch/sequence shapes differ")
    if hidden_states.shape[0] != 1:
        raise ValueError("the pre-packed Qwen3.5 path requires per-device batch one")
    shifted = labels[..., 1:]
    valid = shifted.ne(-100)
    positions = torch.nonzero(valid[0], as_tuple=False).flatten()
    rows = hidden_states[:, :-1, :][valid]
    targets = shifted[valid]
    return rows, targets, positions


def reference_selective_linear_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_target_count: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 full-logit reference restricted to algebraically live rows."""

    rows, targets, _ = select_supervised_predecessor_rows(hidden_states, labels)
    if targets.numel() == 0:
        return (hidden_states.sum() + lm_head_weight.sum()) * 0.0
    logits = torch.nn.functional.linear(rows, lm_head_weight).float()
    loss_sum = torch.nn.functional.cross_entropy(logits, targets.to(logits.device), reduction="sum")
    divisor: int | torch.Tensor = targets.numel() if global_target_count is None else global_target_count
    if torch.is_tensor(divisor):
        divisor = divisor.to(loss_sum.device)
    if int(divisor) <= 0:
        raise ValueError("global_target_count must be positive when supervised targets exist")
    return loss_sum / divisor
