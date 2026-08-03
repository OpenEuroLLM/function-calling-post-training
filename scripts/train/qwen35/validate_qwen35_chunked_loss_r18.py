#!/usr/bin/env python3
"""Produce complete CUDA H2 evidence for the preregistered R18 objective."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_chunked_loss import (
    IMPLEMENTATION_ID,
    checkpointed_chunked_selective_linear_cross_entropy,
    install_qwen35_checkpointed_chunked_loss,
    ordinary_chunked_selective_linear_cross_entropy,
)
from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_report import (
    DIRECT_FAMILIES,
    TRAJECTORY_PARAMETER_FAMILIES,
    diagnostic_tensor_comparison_metrics,
    exact_tensor_comparison_metrics,
)

ROLE_ORDER = ("observed", "reference", "unchunked", "full_ignore")
HELDOUT_ROWS = 17
TRAJECTORY_IMPLEMENTATION = "embedding_tanh_linear_tanh_linear_tied_output_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    return parser.parse_args()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_attestation() -> dict[str, Any]:
    root = _source_root()
    files = (
        "open_instruct/qwen35_chunked_loss.py",
        "open_instruct/qwen35_qualification_r18.py",
        "open_instruct/qwen35_qualification_r18_report.py",
        "scripts/train/qwen35/validate_qwen35_chunked_loss_r18.py",
        "scripts/train/qwen35/validate_qwen35_h2_report_r18.py",
    )
    return {
        "git_commit": _git_output(root, "rev-parse", "HEAD"),
        "git_worktree_clean": _git_output(root, "status", "--porcelain") == "",
        "implementation_id": IMPLEMENTATION_ID,
        "source_files_sha256": {relative: sha256_file(root / relative) for relative in files},
    }


def _autocast_contract() -> dict[str, Any]:
    return {
        "device_type": "cuda",
        "enabled": bool(torch.is_autocast_enabled("cuda")),
        "dtype": str(torch.get_autocast_dtype("cuda")),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _optimizer(parameters, config: dict[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["epsilon"],
        weight_decay=config["weight_decay"],
        foreach=False,
        fused=False,
    )


def _step_counters(optimizer: torch.optim.Optimizer) -> list[int]:
    counters = {
        int(state["step"].item()) for state in optimizer.state.values() if isinstance(state, dict) and "step" in state
    }
    if not counters:
        raise RuntimeError("AdamW produced no step counter")
    return sorted(counters)


def _all_exact(value: Any) -> bool:
    if isinstance(value, dict):
        if {"bitwise_equal", "observed_nonfinite_count", "reference_nonfinite_count"} <= set(value):
            return (
                value["bitwise_equal"] is True
                and value["observed_nonfinite_count"] == 0
                and value["reference_nonfinite_count"] == 0
            )
        return all(_all_exact(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_exact(child) for child in value)
    return True


def _all_diagnostic_finite(value: Any) -> bool:
    if isinstance(value, dict):
        if "nonfinite_count" in value and "relative_l2_error" in value:
            return value["nonfinite_count"] == 0
        return all(_all_diagnostic_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_diagnostic_finite(child) for child in value)
    return True


@dataclass
class _DirectBranch:
    loss: torch.Tensor
    hidden_gradient: torch.Tensor
    weight_gradient: torch.Tensor
    raw_update: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    parameter: torch.Tensor
    heldout_logits: torch.Tensor
    heldout_loss: torch.Tensor
    step_counters: list[int]
    autocast_contract: dict[str, Any]
    counter_after_forward: int | None
    counter_after_backward: int | None
    audit: dict[str, Any] | None
    saved_shapes: list[list[int]]


def _run_direct_branch(
    *,
    role: str,
    initial_hidden: torch.Tensor,
    initial_weight: torch.Tensor,
    targets: torch.Tensor,
    full_targets: torch.Tensor,
    global_divisor: int,
    chunk_size: int,
    heldout_hidden: torch.Tensor,
    heldout_targets: torch.Tensor,
    optimizer_config: dict[str, Any],
) -> _DirectBranch:
    hidden = initial_hidden.detach().clone().requires_grad_(True)
    weight = initial_weight.detach().clone().requires_grad_(True)
    execution_counter: dict[str, int] = {}
    saved_shapes: list[list[int]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(list(tensor.shape))
        return tensor

    with (
        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        autocast = _autocast_contract()
        if role == "observed":
            result = checkpointed_chunked_selective_linear_cross_entropy(
                hidden,
                weight,
                targets,
                global_target_count=global_divisor,
                chunk_size=chunk_size,
                execution_counter=execution_counter,
                return_audit=True,
            )
            loss, audit = result
        elif role == "reference":
            result = ordinary_chunked_selective_linear_cross_entropy(
                hidden,
                weight,
                targets,
                global_target_count=global_divisor,
                chunk_size=chunk_size,
                execution_counter=execution_counter,
                return_audit=True,
            )
            loss, audit = result
        elif role == "unchunked":
            logits = F.linear(hidden, weight)
            loss = F.cross_entropy(logits.float(), targets, reduction="sum") / global_divisor
            audit = None
        elif role == "full_ignore":
            logits = F.linear(hidden, weight)
            loss = F.cross_entropy(logits.float(), full_targets, ignore_index=-100, reduction="sum") / global_divisor
            audit = None
        else:
            raise ValueError(f"unsupported direct branch {role!r}")
    counter_after_forward = execution_counter.get("chunk_function_calls") if execution_counter else None
    loss.backward()
    counter_after_backward = execution_counter.get("chunk_function_calls") if execution_counter else None
    if hidden.grad is None or weight.grad is None:
        raise RuntimeError(f"{role} direct branch disconnected a required leaf")
    hidden_gradient = hidden.grad.detach().clone()
    weight_gradient = weight.grad.detach().clone()
    before = weight.detach().clone()
    optimizer = _optimizer([weight], optimizer_config)
    optimizer.step()
    state = optimizer.state[weight]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        heldout_logits = F.linear(heldout_hidden, weight)
    heldout_loss = F.cross_entropy(heldout_logits.float(), heldout_targets, reduction="sum") / (
        heldout_targets.numel() + 5
    )
    return _DirectBranch(
        loss=loss.detach().clone(),
        hidden_gradient=hidden_gradient,
        weight_gradient=weight_gradient,
        raw_update=weight.detach().clone() - before,
        exp_avg=state["exp_avg"].detach().clone(),
        exp_avg_sq=state["exp_avg_sq"].detach().clone(),
        parameter=weight.detach().clone(),
        heldout_logits=heldout_logits.detach().clone(),
        heldout_loss=heldout_loss.detach().clone(),
        step_counters=_step_counters(optimizer),
        autocast_contract=autocast,
        counter_after_forward=counter_after_forward,
        counter_after_backward=counter_after_backward,
        audit=audit.to_dict() if audit is not None else None,
        saved_shapes=saved_shapes,
    )


def _direct_comparison(
    observed: _DirectBranch, reference: _DirectBranch, metric: Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]
) -> dict[str, Any]:
    tensors = {
        "loss": (observed.loss, reference.loss),
        "selected_hidden_gradient": (observed.hidden_gradient, reference.hidden_gradient),
        "output_head_gradient": (observed.weight_gradient, reference.weight_gradient),
        "raw_adamw_update": (observed.raw_update, reference.raw_update),
        "optimizer_exp_avg": (observed.exp_avg, reference.exp_avg),
        "optimizer_exp_avg_sq": (observed.exp_avg_sq, reference.exp_avg_sq),
        "post_step_parameter": (observed.parameter, reference.parameter),
        "heldout_logits": (observed.heldout_logits, reference.heldout_logits),
        "heldout_loss": (observed.heldout_loss, reference.heldout_loss),
    }
    if set(tensors) != set(DIRECT_FAMILIES):
        raise RuntimeError("internal direct comparison family drift")
    return {name: metric(*pair) for name, pair in tensors.items()}


def _direct_case(
    *,
    contract: dict[str, Any],
    hidden_size: int,
    vocab_size: int,
    weight_standard_deviation: float,
    chunk_size: int,
    optimizer_config: dict[str, Any],
) -> dict[str, Any]:
    rows = int(contract["selected_rows"])
    global_divisor = int(contract.get("global_divisor", rows + 37))
    generator = torch.Generator(device="cuda").manual_seed(contract["seed"])
    initial_hidden = (
        torch.randn(rows, hidden_size, generator=generator, device="cuda", dtype=torch.float32)
        * contract["hidden_scale"]
    )
    initial_weight = (
        torch.randn(vocab_size, hidden_size, generator=generator, device="cuda", dtype=torch.float32)
        * weight_standard_deviation
    )
    targets = torch.randint(0, vocab_size, (rows,), generator=generator, device="cuda", dtype=torch.long)
    full_targets = targets.detach().clone()
    heldout_hidden = torch.randn(HELDOUT_ROWS, hidden_size, generator=generator, device="cuda", dtype=torch.float32)
    heldout_targets = torch.randint(
        0, vocab_size, (HELDOUT_ROWS,), generator=generator, device="cuda", dtype=torch.long
    )
    branches = {
        role: _run_direct_branch(
            role=role,
            initial_hidden=initial_hidden,
            initial_weight=initial_weight,
            targets=targets,
            full_targets=full_targets,
            global_divisor=global_divisor,
            chunk_size=chunk_size,
            heldout_hidden=heldout_hidden,
            heldout_targets=heldout_targets,
            optimizer_config=optimizer_config,
        )
        for role in ROLE_ORDER
    }
    primary = _direct_comparison(branches["observed"], branches["reference"], exact_tensor_comparison_metrics)
    diagnostic_a = _direct_comparison(
        branches["reference"], branches["unchunked"], diagnostic_tensor_comparison_metrics
    )
    diagnostic_b = _direct_comparison(
        branches["unchunked"], branches["full_ignore"], diagnostic_tensor_comparison_metrics
    )
    boundaries = branches["observed"].audit["chunk_boundaries"]
    forbidden = [[end - start, vocab_size] for start, end in boundaries]
    result = {
        "case_contract": contract,
        "chunk_size": chunk_size,
        "global_divisor": global_divisor,
        "observed_audit": branches["observed"].audit,
        "reference_audit": branches["reference"].audit,
        "execution_proof": {
            "observed_after_forward": branches["observed"].counter_after_forward,
            "observed_after_backward": branches["observed"].counter_after_backward,
            "reference_after_forward": branches["reference"].counter_after_forward,
            "reference_after_backward": branches["reference"].counter_after_backward,
        },
        "saved_tensor_proof": {
            "checkpoint_saved_shapes": branches["observed"].saved_shapes,
            "ordinary_saved_shapes": branches["reference"].saved_shapes,
            "forbidden_logit_shapes": forbidden,
            "checkpoint_saved_no_chunk_logits": not any(
                shape in forbidden for shape in branches["observed"].saved_shapes
            ),
            "ordinary_saved_at_least_one_chunk_logit": any(
                shape in forbidden for shape in branches["reference"].saved_shapes
            ),
        },
        "autocast_contracts": {
            **{role: branches[role].autocast_contract for role in ROLE_ORDER},
            "heldout": {"device_type": "cuda", "enabled": True, "dtype": "torch.bfloat16"},
        },
        "optimizer_step_counters": {role: branches[role].step_counters for role in ROLE_ORDER},
        "primary": primary,
        "diagnostic_a": diagnostic_a,
        "diagnostic_b": diagnostic_b,
    }
    result["status"] = (
        "passed"
        if _all_exact(primary)
        and _all_diagnostic_finite(diagnostic_a)
        and _all_diagnostic_finite(diagnostic_b)
        and result["saved_tensor_proof"]["checkpoint_saved_no_chunk_logits"]
        and result["saved_tensor_proof"]["ordinary_saved_at_least_one_chunk_logit"]
        else "failed"
    )
    return result


def _zero_target(chunk_size: int, h2: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(2026071901 + chunk_size)
    hidden_base = torch.randn(1, h2["direct_hidden_size"], generator=generator, device="cuda", dtype=torch.float32)
    weight_base = torch.randn(
        h2["direct_vocab_size"], h2["direct_hidden_size"], generator=generator, device="cuda", dtype=torch.float32
    )
    observed_hidden = hidden_base.detach().clone().requires_grad_(True)
    reference_hidden = hidden_base.detach().clone().requires_grad_(True)
    observed_weight = weight_base.detach().clone().requires_grad_(True)
    reference_weight = weight_base.detach().clone().requires_grad_(True)
    targets = torch.tensor([-100], device="cuda", dtype=torch.long)
    counter: dict[str, int] = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        autocast = _autocast_contract()
        result = checkpointed_chunked_selective_linear_cross_entropy(
            observed_hidden,
            observed_weight,
            targets,
            global_target_count=None,
            chunk_size=chunk_size,
            execution_counter=counter,
            return_audit=True,
        )
        observed_loss, audit = result
        reference_loss = (reference_hidden.sum() + reference_weight.sum()) * 0.0
    observed_loss.backward()
    reference_loss.backward()
    if any(value.grad is None for value in (observed_hidden, reference_hidden, observed_weight, reference_weight)):
        raise RuntimeError("R18 zero-target branch disconnected a leaf")
    metrics = {
        "chunk_size": chunk_size,
        "loss": exact_tensor_comparison_metrics(observed_loss, reference_loss),
        "loss_value": float(observed_loss.detach().item()),
        "hidden_gradient": exact_tensor_comparison_metrics(observed_hidden.grad, reference_hidden.grad),
        "hidden_gradient_nonzero_count": int(torch.count_nonzero(observed_hidden.grad).item()),
        "output_head_gradient": exact_tensor_comparison_metrics(observed_weight.grad, reference_weight.grad),
        "output_head_gradient_nonzero_count": int(torch.count_nonzero(observed_weight.grad).item()),
        "execution_counter": counter,
        "audit": audit.to_dict(),
        "autocast_contract": autocast,
    }
    metrics["status"] = (
        "passed"
        if metrics["loss_value"] == 0.0
        and metrics["hidden_gradient_nonzero_count"] == 0
        and metrics["output_head_gradient_nonzero_count"] == 0
        and counter == {}
        and _all_exact(metrics)
        else "failed"
    )
    return metrics


def _tiny_qwen_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["full_attention"],
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return config


def _qwen_forward_integration(chunk_size: int) -> dict[str, Any]:
    torch.manual_seed(2026071902)
    reference = Qwen3_5ForCausalLM(_tiny_qwen_config()).cuda().train()
    observed = Qwen3_5ForCausalLM(_tiny_qwen_config()).cuda().train()
    observed.load_state_dict(reference.state_dict(), strict=True)
    install_qwen35_checkpointed_chunked_loss(observed, chunk_size=chunk_size)
    input_ids = torch.tensor([[1, 7, 2, 11, 3, 5, 13, 17]], device="cuda", dtype=torch.long)
    labels = input_ids.clone()
    labels[:, [0, 2, 4, 5]] = -100
    shifted = labels[:, 1:]
    positions = torch.nonzero(shifted[0].ne(-100), as_tuple=False).flatten()
    targets = shifted[0, positions].contiguous()
    divisor = int(targets.numel()) + 9
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        observed_output = observed(
            input_ids=input_ids,
            labels=labels,
            logits_to_keep=positions,
            shift_labels=targets,
            num_items_in_batch=divisor,
            use_cache=False,
        )
        reference_hidden = reference.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
        reference_rows = reference_hidden[:, positions, :].reshape(-1, reference_hidden.shape[-1])
        reference_loss = ordinary_chunked_selective_linear_cross_entropy(
            reference_rows, reference.lm_head.weight, targets, global_target_count=divisor, chunk_size=chunk_size
        )
    observed_output.loss.backward()
    reference_loss.backward()
    observed_parameters = dict(observed.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    if list(observed_parameters) != list(reference_parameters):
        raise RuntimeError("R18 Qwen integration parameter order drift")
    gradients = {}
    for name in observed_parameters:
        observed_gradient = observed_parameters[name].grad
        reference_gradient = reference_parameters[name].grad
        if observed_gradient is None or reference_gradient is None:
            raise RuntimeError(f"R18 Qwen integration disconnected {name}")
        gradients[name] = exact_tensor_comparison_metrics(observed_gradient, reference_gradient)
    result = {
        "chunk_size": chunk_size,
        "model_class": type(observed).__name__,
        "attention_implementation": observed.config._attn_implementation,
        "forward_module": observed.forward.__module__,
        "loss": exact_tensor_comparison_metrics(observed_output.loss, reference_loss),
        "named_parameter_gradients": gradients,
        "returned_logits_is_none": observed_output.logits is None,
        "audit": observed._qwen35_last_loss_audit,
    }
    result["status"] = "passed" if _all_exact(result) and result["returned_logits_is_none"] is True else "failed"
    return result


class _TrajectoryNetwork(torch.nn.Module):
    def __init__(self, *, vocabulary_size: int, hidden_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.transformations = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(2)])

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        value = self.embedding(input_ids)
        for layer in self.transformations:
            value = torch.tanh(layer(value))
        return value


def _trajectory_batch(*, seed: int, target_count: int, model: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    sequence_length = model["sequence_length"]
    input_ids = torch.randint(
        0, model["vocab_size"], (1, sequence_length), generator=generator, device="cuda", dtype=torch.long
    )
    positions = torch.randperm(sequence_length - 1, generator=generator, device="cuda")[:target_count]
    positions = torch.sort(positions).values.contiguous()
    targets = torch.randint(
        0, model["vocab_size"], (target_count,), generator=generator, device="cuda", dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "positions": positions,
        "targets": targets,
        "target_count": target_count,
        "global_divisor": target_count + 37,
        "seed": seed,
    }


def _trajectory_loss(
    *, role: str, model: _TrajectoryNetwork, batch: dict[str, Any], chunk_size: int, counter: dict[str, int]
) -> torch.Tensor:
    hidden = model.hidden(batch["input_ids"])[0]
    selected = hidden[batch["positions"]]
    if role == "observed":
        return checkpointed_chunked_selective_linear_cross_entropy(
            selected,
            model.embedding.weight,
            batch["targets"],
            global_target_count=batch["global_divisor"],
            chunk_size=chunk_size,
            execution_counter=counter,
        )
    if role == "reference":
        return ordinary_chunked_selective_linear_cross_entropy(
            selected,
            model.embedding.weight,
            batch["targets"],
            global_target_count=batch["global_divisor"],
            chunk_size=chunk_size,
            execution_counter=counter,
        )
    if role == "unchunked":
        logits = F.linear(selected, model.embedding.weight)
        return F.cross_entropy(logits.float(), batch["targets"], reduction="sum") / batch["global_divisor"]
    if role == "full_ignore":
        full_targets = torch.full((hidden.shape[0],), -100, device=hidden.device, dtype=torch.long)
        full_targets[batch["positions"]] = batch["targets"]
        logits = F.linear(hidden, model.embedding.weight)
        return (
            F.cross_entropy(logits.float(), full_targets, ignore_index=-100, reduction="sum") / batch["global_divisor"]
        )
    raise ValueError(f"unknown trajectory role {role!r}")


def _named_state(
    *,
    parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    initial: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    preclip: dict[str, torch.Tensor],
    clipped: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "preclip_gradient": preclip,
        "clipped_gradient": clipped,
        "raw_adamw_update": {name: parameters[name].detach().clone() - before[name] for name in parameters},
        "optimizer_exp_avg": {
            name: optimizer.state[parameters[name]]["exp_avg"].detach().clone() for name in parameters
        },
        "optimizer_exp_avg_sq": {
            name: optimizer.state[parameters[name]]["exp_avg_sq"].detach().clone() for name in parameters
        },
        "cumulative_parameter_displacement": {
            name: parameters[name].detach().clone() - initial[name] for name in parameters
        },
        "post_step_parameter_state": {name: parameters[name].detach().clone() for name in parameters},
    }


def _flatten_named(values: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1) for name in names])


def _trajectory_comparison(
    *,
    observed_loss: torch.Tensor,
    reference_loss: torch.Tensor,
    observed_norm: torch.Tensor,
    reference_norm: torch.Tensor,
    observed_state: dict[str, dict[str, torch.Tensor]],
    reference_state: dict[str, dict[str, torch.Tensor]],
    names: list[str],
    observed_heldout_logits: torch.Tensor,
    reference_heldout_logits: torch.Tensor,
    observed_heldout_loss: torch.Tensor,
    reference_heldout_loss: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {
        family: metric(_flatten_named(observed_state[family], names), _flatten_named(reference_state[family], names))
        for family in TRAJECTORY_PARAMETER_FAMILIES
    }
    named = {
        name: {
            family: metric(observed_state[family][name], reference_state[family][name])
            for family in TRAJECTORY_PARAMETER_FAMILIES
        }
        for name in names
    }
    return {
        "loss": metric(observed_loss, reference_loss),
        "preclip_global_norm": metric(observed_norm.reshape(()), reference_norm.reshape(())),
        "aggregate": aggregate,
        "named": named,
        "heldout_logits": metric(observed_heldout_logits, reference_heldout_logits),
        "heldout_loss": metric(observed_heldout_loss, reference_heldout_loss),
    }


def _heldout(model: _TrajectoryNetwork, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.hidden(batch["input_ids"])[0]
    logits = F.linear(hidden, model.embedding.weight)
    loss = (
        F.cross_entropy(logits[batch["positions"]].float(), batch["targets"], reduction="sum")
        / batch["global_divisor"]
    )
    return logits, loss


def _trajectory(
    *, contract: dict[str, Any], chunk_size: int, h2: dict[str, Any], optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    model_definition = h2["trajectory_model"]
    torch.manual_seed(contract["model_seed"])
    base = (
        _TrajectoryNetwork(vocabulary_size=model_definition["vocab_size"], hidden_size=model_definition["hidden_size"])
        .cuda()
        .train()
    )
    models = {role: copy.deepcopy(base) for role in ROLE_ORDER}
    parameters = {role: dict(model.named_parameters()) for role, model in models.items()}
    names = list(parameters["observed"])
    if any(list(parameters[role]) != names for role in ROLE_ORDER):
        raise RuntimeError("R18 trajectory parameter-name order drift")
    if any(
        not torch.equal(parameters["observed"][name], parameters[role][name])
        for role in ROLE_ORDER[1:]
        for name in names
    ):
        raise RuntimeError("R18 trajectory initial parameters are not bit exact")
    if any(
        sorted({str(parameter.dtype) for parameter in model.parameters()}) != ["torch.float32"]
        for model in models.values()
    ):
        raise RuntimeError("R18 trajectory parameters are not exclusively FP32")
    geometry = [
        {
            "name": name,
            "shape": list(parameters["observed"][name].shape),
            "elements": int(parameters["observed"][name].numel()),
        }
        for name in names
    ]
    initial = {role: {name: parameters[role][name].detach().clone() for name in names} for role in ROLE_ORDER}
    optimizers = {role: _optimizer(models[role].parameters(), optimizer_config) for role in ROLE_ORDER}
    heldout = _trajectory_batch(seed=contract["heldout_seed"], target_count=257, model=model_definition)
    heldout["global_divisor"] = 294
    steps = []
    target_cycle = h2["trajectory_target_count_cycle"]
    for step_index in range(h2["trajectory_steps"]):
        seed = (contract["batch_seed_base"] + step_index) % (2**32)
        batch = _trajectory_batch(
            seed=seed, target_count=target_cycle[step_index % len(target_cycle)], model=model_definition
        )
        counters = {role: {} for role in ROLE_ORDER}
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        losses = {}
        autocast_contracts = {}
        for role in ROLE_ORDER:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                autocast_contracts[role] = _autocast_contract()
                losses[role] = _trajectory_loss(
                    role=role, model=models[role], batch=batch, chunk_size=chunk_size, counter=counters[role]
                )
        after_forward = {role: counters[role].get("chunk_function_calls") for role in ("observed", "reference")}
        for role in ROLE_ORDER:
            losses[role].backward()
        after_backward = {role: counters[role].get("chunk_function_calls") for role in ("observed", "reference")}
        if any(parameters[role][name].grad is None for role in ROLE_ORDER for name in names):
            raise RuntimeError("R18 trajectory disconnected a named parameter")
        before = {role: {name: parameters[role][name].detach().clone() for name in names} for role in ROLE_ORDER}
        preclip = {role: {name: parameters[role][name].grad.detach().clone() for name in names} for role in ROLE_ORDER}
        preclip_norms = {
            role: torch.nn.utils.clip_grad_norm_(
                models[role].parameters(), optimizer_config["maximum_gradient_norm"], foreach=False
            )
            .detach()
            .clone()
            for role in ROLE_ORDER
        }
        clipped = {role: {name: parameters[role][name].grad.detach().clone() for name in names} for role in ROLE_ORDER}
        for optimizer in optimizers.values():
            optimizer.step()
        states = {
            role: _named_state(
                parameters=parameters[role],
                optimizer=optimizers[role],
                initial=initial[role],
                before=before[role],
                preclip=preclip[role],
                clipped=clipped[role],
            )
            for role in ROLE_ORDER
        }
        heldout_values = {}
        with torch.no_grad():
            for role in ROLE_ORDER:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    heldout_autocast = _autocast_contract()
                    heldout_values[role] = _heldout(models[role], heldout)
        primary = _trajectory_comparison(
            observed_loss=losses["observed"].detach(),
            reference_loss=losses["reference"].detach(),
            observed_norm=preclip_norms["observed"],
            reference_norm=preclip_norms["reference"],
            observed_state=states["observed"],
            reference_state=states["reference"],
            names=names,
            observed_heldout_logits=heldout_values["observed"][0],
            reference_heldout_logits=heldout_values["reference"][0],
            observed_heldout_loss=heldout_values["observed"][1],
            reference_heldout_loss=heldout_values["reference"][1],
            metric=exact_tensor_comparison_metrics,
        )
        diagnostic_a = _trajectory_comparison(
            observed_loss=losses["reference"].detach(),
            reference_loss=losses["unchunked"].detach(),
            observed_norm=preclip_norms["reference"],
            reference_norm=preclip_norms["unchunked"],
            observed_state=states["reference"],
            reference_state=states["unchunked"],
            names=names,
            observed_heldout_logits=heldout_values["reference"][0],
            reference_heldout_logits=heldout_values["unchunked"][0],
            observed_heldout_loss=heldout_values["reference"][1],
            reference_heldout_loss=heldout_values["unchunked"][1],
            metric=diagnostic_tensor_comparison_metrics,
        )
        diagnostic_b = _trajectory_comparison(
            observed_loss=losses["unchunked"].detach(),
            reference_loss=losses["full_ignore"].detach(),
            observed_norm=preclip_norms["unchunked"],
            reference_norm=preclip_norms["full_ignore"],
            observed_state=states["unchunked"],
            reference_state=states["full_ignore"],
            names=names,
            observed_heldout_logits=heldout_values["unchunked"][0],
            reference_heldout_logits=heldout_values["full_ignore"][0],
            observed_heldout_loss=heldout_values["unchunked"][1],
            reference_heldout_loss=heldout_values["full_ignore"][1],
            metric=diagnostic_tensor_comparison_metrics,
        )
        step = {
            "step": step_index + 1,
            "batch_contract": {
                "seed": seed,
                "target_count": batch["target_count"],
                "global_divisor": batch["global_divisor"],
                "input_ids_sha256": _tensor_sha256(batch["input_ids"]),
                "positions_sha256": _tensor_sha256(batch["positions"]),
                "targets_sha256": _tensor_sha256(batch["targets"]),
            },
            "execution_proof": {
                "observed_after_forward": after_forward["observed"],
                "observed_after_backward": after_backward["observed"],
                "reference_after_forward": after_forward["reference"],
                "reference_after_backward": after_backward["reference"],
            },
            "autocast_contracts": {**autocast_contracts, "heldout": heldout_autocast},
            "optimizer_step_counters": {role: _step_counters(optimizers[role]) for role in ROLE_ORDER},
            "primary": primary,
            "diagnostic_a": diagnostic_a,
            "diagnostic_b": diagnostic_b,
        }
        step["status"] = (
            "passed"
            if _all_exact(primary) and _all_diagnostic_finite(diagnostic_a) and _all_diagnostic_finite(diagnostic_b)
            else "failed"
        )
        steps.append(step)
    return {
        "trajectory_contract": contract,
        "chunk_size": chunk_size,
        "model_definition": {**model_definition, "implementation": TRAJECTORY_IMPLEMENTATION},
        "parameter_geometry": geometry,
        "parameter_count": sum(row["elements"] for row in geometry),
        "heldout_contract": {
            "seed": contract["heldout_seed"],
            "target_count": 257,
            "global_divisor": 294,
            "sequence_length": model_definition["sequence_length"],
        },
        "steps": steps,
        "status": "passed" if all(step["status"] == "passed" for step in steps) else "failed",
    }


def _candidate(chunk_size: int, h2: dict[str, Any]) -> dict[str, Any]:
    optimizer_config = h2["optimizer"]
    direct = [
        _direct_case(
            contract=contract,
            hidden_size=h2["direct_hidden_size"],
            vocab_size=h2["direct_vocab_size"],
            weight_standard_deviation=h2["direct_weight_standard_deviation"],
            chunk_size=chunk_size,
            optimizer_config=optimizer_config,
        )
        for contract in h2["direct_cases"]
    ]
    real_contract = h2["real_geometry_case"]
    real = _direct_case(
        contract=real_contract,
        hidden_size=real_contract["hidden_size"],
        vocab_size=real_contract["vocab_size"],
        weight_standard_deviation=real_contract["weight_standard_deviation"],
        chunk_size=chunk_size,
        optimizer_config=optimizer_config,
    )
    trajectories = [
        _trajectory(contract=contract, chunk_size=chunk_size, h2=h2, optimizer_config=optimizer_config)
        for contract in h2["trajectories"]
    ]
    result = {
        "chunk_size": chunk_size,
        "zero_target": _zero_target(chunk_size, h2),
        "qwen_forward_integration": _qwen_forward_integration(chunk_size),
        "direct_cases": direct,
        "real_geometry_case": real,
        "trajectories": trajectories,
    }
    result["status"] = (
        "passed"
        if result["zero_target"]["status"] == "passed"
        and result["qwen_forward_integration"]["status"] == "passed"
        and all(case["status"] == "passed" for case in direct)
        and real["status"] == "passed"
        and all(trajectory["status"] == "passed" for trajectory in trajectories)
        else "failed"
    )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R18 H2 qualification requires CUDA")
    if "A100" not in torch.cuda.get_device_name():
        raise RuntimeError(f"R18 H2 requires Leonardo A100 hardware, found {torch.cuda.get_device_name()!r}")
    qualification, manifest_sha256 = load_qualification_manifest(args.qualification_manifest)
    h2 = qualification["h2_acceptance"]
    source = _source_attestation()
    if not source["git_worktree_clean"]:
        raise RuntimeError("R18 qualification requires a clean immutable source worktree")
    liger_imported = any(name == "liger_kernel" or name.startswith("liger_kernel.") for name in sys.modules)
    if liger_imported:
        raise RuntimeError("R18 qualification process imported Liger despite the frozen non-Liger contract")
    report_base = {
        "artifact": "qwen35_checkpointed_chunked_selected_loss_qualification_r18",
        "schema_version": 1,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": manifest_sha256,
        "manifest_derivation": qualification["manifest_derivation"],
        "source_attestation": source,
        "environment": {
            "device_type": "cuda",
            "cuda_device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_capability": list(torch.cuda.get_device_capability()),
            "liger_imported": liger_imported,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "primary_comparison": {
            "observed_path": h2["primary_observed_path"],
            "reference_path": h2["primary_reference_path"],
            "acceptance": h2["primary_acceptance"],
            "numerical_discrepancy_is_gating": True,
        },
        "diagnostic_comparisons": {
            "a": {
                "observed_path": h2["mandatory_diagnostic_a_observed_path"],
                "reference_path": h2["mandatory_diagnostic_a_reference_path"],
            },
            "b": {
                "observed_path": h2["mandatory_diagnostic_b_observed_path"],
                "reference_path": h2["mandatory_diagnostic_b_reference_path"],
            },
            "numerical_discrepancy_is_gating": False,
            "integrity_and_finiteness_are_mandatory": True,
        },
    }
    candidates: list[dict[str, Any]] = []
    try:
        for chunk_size in h2["candidate_chunk_sizes"]:
            candidates.append(_candidate(chunk_size, h2))
            partial = {
                **report_base,
                "candidate_results": candidates,
                "status": "running",
                "successor_gate_authorized": False,
                "scientific_training_authorized": False,
                "allowed_conclusion": "R18 H2 is incomplete; no successor gate or scientific work is authorized.",
            }
            _write_json_atomic(args.report_output, partial)
        passed = all(candidate["status"] == "passed" for candidate in candidates)
        report = {
            **report_base,
            "candidate_results": candidates,
            "status": "passed" if passed else "failed",
            "successor_gate_authorized": passed,
            "scientific_training_authorized": False,
            "allowed_conclusion": (
                "R18 H2 passed; H3 may begin, while scientific training and evaluation remain unauthorized."
                if passed
                else "R18 H2 failed; H3 and all scientific work remain blocked without threshold rescue."
            ),
        }
        _write_json_atomic(args.report_output, report)
        print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
        if not passed:
            raise SystemExit(1)
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        failure = {
            **report_base,
            "candidate_results": candidates,
            "status": "error",
            "successor_gate_authorized": False,
            "scientific_training_authorized": False,
            "allowed_conclusion": "R18 H2 execution was incomplete; H3 and all scientific work remain blocked.",
            "execution_error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json_atomic(args.report_output, failure)
        raise


if __name__ == "__main__":
    main()
