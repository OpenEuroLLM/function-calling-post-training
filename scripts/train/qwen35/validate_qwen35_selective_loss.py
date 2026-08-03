#!/usr/bin/env python3
"""CUDA qualification for selective Liger fused-linear cross entropy."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import traceback
from importlib import metadata
from pathlib import Path

import torch
from liger_kernel.ops import fused_linear_cross_entropy as liger_fused_linear_cross_entropy_ops
from liger_kernel.ops import utils as liger_ops_utils
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification import (
    load_qualification_manifest,
    scalar_comparison_metrics,
    sha256_file,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)

PINNED_LIGER_COMMIT = "72a4ed47a5c593b58045a0af14d3f774a037bd92"


def _write_strict_json_atomic(path: Path, value: dict) -> None:
    """Publish finite RFC-8259 JSON without exposing a partial report."""

    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete.{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, path)


def _active_bf16_autocast_contract() -> dict[str, str | bool]:
    """Fail unless execution is inside the production CUDA autocast contract."""

    enabled = torch.is_autocast_enabled("cuda")
    dtype = torch.get_autocast_dtype("cuda")
    if not enabled or dtype != torch.bfloat16:
        raise RuntimeError(f"production BF16 autocast is not active: enabled={enabled}, dtype={dtype}")
    return {
        "device_type": "cuda",
        "enabled": True,
        "dtype": str(dtype),
    }


def _floating_optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> list[str]:
    dtypes = {
        str(value.dtype)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if not dtypes:
        raise RuntimeError("AdamW produced no floating optimizer state")
    return sorted(dtypes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    return parser.parse_args()


def _run_case(
    *,
    seed: int,
    rows: int,
    supervised_rows: list[int],
    hidden_size: int,
    vocab_size: int,
    global_divisor_extra: int,
    numerical_acceptance: dict,
    optimizer_config: dict,
) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(rows, hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, generator=generator, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.randint(0, vocab_size, (len(supervised_rows),), generator=generator, device="cuda")
    selected = hidden[supervised_rows].detach().clone().requires_grad_(True)
    fused_weight = weight.detach().clone().requires_grad_(True)
    reference_rows = selected.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    divisor = len(supervised_rows) + global_divisor_extra

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fused_autocast = _active_bf16_autocast_contract()
        fused_sum = LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(
            fused_weight, selected, targets
        )
    fused_loss = fused_sum / divisor
    fused_loss.backward()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_autocast = _active_bf16_autocast_contract()
        reference_logits = torch.nn.functional.linear(reference_rows, reference_weight)
    reference_loss = torch.nn.functional.cross_entropy(reference_logits.float(), targets, reduction="sum") / divisor
    reference_loss.backward()

    loss_metrics = scalar_comparison_metrics(float(fused_loss), float(reference_loss))
    row_gradient_metrics = tensor_comparison_metrics(selected.grad, reference_rows.grad)
    weight_gradient_metrics = tensor_comparison_metrics(fused_weight.grad, reference_weight.grad)
    validate_comparison_metrics(loss_metrics, numerical_acceptance, kind="loss", context="direct fused loss")
    validate_comparison_metrics(
        row_gradient_metrics, numerical_acceptance, kind="gradient", context="direct selected-row gradient"
    )
    validate_comparison_metrics(
        weight_gradient_metrics, numerical_acceptance, kind="gradient", context="direct output-weight gradient"
    )

    fused_optimizer = torch.optim.AdamW(
        [fused_weight],
        lr=optimizer_config["learning_rate"],
        betas=(optimizer_config["adam_beta1"], optimizer_config["adam_beta2"]),
        eps=optimizer_config["adam_epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    reference_optimizer = torch.optim.AdamW(
        [reference_weight],
        lr=optimizer_config["learning_rate"],
        betas=(optimizer_config["adam_beta1"], optimizer_config["adam_beta2"]),
        eps=optimizer_config["adam_epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    fused_optimizer.step()
    reference_optimizer.step()
    fused_optimizer_dtypes = _floating_optimizer_state_dtypes(fused_optimizer)
    reference_optimizer_dtypes = _floating_optimizer_state_dtypes(reference_optimizer)
    if fused_optimizer_dtypes != ["torch.float32"] or reference_optimizer_dtypes != ["torch.float32"]:
        raise AssertionError(
            "direct AdamW optimizer state is not exclusively FP32: "
            f"fused={fused_optimizer_dtypes}, reference={reference_optimizer_dtypes}"
        )
    fused_update = fused_weight.detach() - weight
    reference_update = reference_weight.detach() - weight
    update_metrics = tensor_comparison_metrics(fused_update, reference_update)
    validate_comparison_metrics(update_metrics, numerical_acceptance, kind="update", context="direct AdamW update")
    return {
        "rows": rows,
        "supervised_rows": len(supervised_rows),
        "global_divisor": divisor,
        "autocast_contract": {
            "fused": fused_autocast,
            "dense_reference": reference_autocast,
            "hidden_input_dtype": str(selected.dtype),
            "output_head_parameter_dtype": str(fused_weight.dtype),
            "loss_accumulation_dtype": "torch.float32",
        },
        "gradient_dtypes": {
            "fused_hidden_rows": str(selected.grad.dtype),
            "dense_reference_hidden_rows": str(reference_rows.grad.dtype),
            "fused_output_head": str(fused_weight.grad.dtype),
            "dense_reference_output_head": str(reference_weight.grad.dtype),
        },
        "optimizer_floating_state_dtypes": {
            "fused": fused_optimizer_dtypes,
            "dense_reference": reference_optimizer_dtypes,
        },
        "loss_fused": float(fused_loss),
        "loss_reference": float(reference_loss),
        "loss_comparison": loss_metrics,
        "row_gradient_comparison": row_gradient_metrics,
        "weight_gradient_comparison": weight_gradient_metrics,
        "adamw_update_comparison": update_metrics,
    }


def _run_zero_target_sentinel(hidden_size: int, vocab_size: int) -> dict:
    hidden = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32) * 0.02).requires_grad_(True)
    target = torch.tensor([-100], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        autocast_contract = _active_bf16_autocast_contract()
        loss = (
            LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(weight, hidden, target) / 7
        )
    loss.backward()
    if not torch.isfinite(loss) or float(loss) != 0:
        raise AssertionError(f"all-ignored sentinel loss is not finite zero: {float(loss)}")
    if hidden.grad is None or weight.grad is None:
        raise AssertionError("all-ignored sentinel disconnected the hidden or output-head graph")
    if torch.count_nonzero(hidden.grad) or torch.count_nonzero(weight.grad):
        raise AssertionError("all-ignored sentinel produced a nonzero gradient")
    return {
        "loss": float(loss),
        "global_divisor": 7,
        "autocast_contract": autocast_contract,
        "hidden_input_dtype": str(hidden.dtype),
        "output_head_parameter_dtype": str(weight.dtype),
        "hidden_gradient_dtype": str(hidden.grad.dtype),
        "output_head_gradient_dtype": str(weight.grad.dtype),
        "hidden_gradient_connected": hidden.grad is not None,
        "weight_gradient_connected": weight.grad is not None,
        "gradient_nonzero_count": int(torch.count_nonzero(hidden.grad)) + int(torch.count_nonzero(weight.grad)),
    }


def _verify_liger_source_pin(expected_source_files: dict[str, str]) -> dict:
    distribution = metadata.distribution("liger-kernel")
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("Liger Kernel has no direct_url.json; its audited source cannot be proven")
    direct_url = json.loads(direct_url_text)
    source_url = str(direct_url.get("url", ""))
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    archive_pinned = PINNED_LIGER_COMMIT in source_url and "/archive/" in source_url
    if installed_commit != PINNED_LIGER_COMMIT and not archive_pinned:
        raise RuntimeError(f"Liger Kernel source commit mismatch: {installed_commit!r} != {PINNED_LIGER_COMMIT!r}")
    source_objects = {
        "transformers/fused_linear_cross_entropy.py": LigerFusedLinearCrossEntropyLoss,
        "ops/fused_linear_cross_entropy.py": liger_fused_linear_cross_entropy_ops,
        "ops/utils.py": liger_ops_utils,
    }
    if set(expected_source_files) != set(source_objects):
        raise RuntimeError("qualification manifest does not bind the complete executed Liger source-file set")
    implementation_files = {}
    for relative_path, source_object in source_objects.items():
        implementation_file = Path(inspect.getfile(source_object)).resolve()
        implementation_sha256 = sha256_file(implementation_file)
        if implementation_sha256 != expected_source_files[relative_path]:
            raise RuntimeError(
                f"Liger source hash mismatch for {relative_path}: "
                f"{implementation_sha256!r} != {expected_source_files[relative_path]!r}"
            )
        if "pinned-sources/liger-kernel" not in str(implementation_file) or not str(implementation_file).endswith(
            relative_path
        ):
            raise RuntimeError(
                f"Liger source {relative_path} does not resolve through the qualified pinned checkout: "
                f"{implementation_file}"
            )
        implementation_files[relative_path] = {
            "path": str(implementation_file),
            "sha256": implementation_sha256,
        }
    return {
        "version": distribution.version,
        "source_url": source_url,
        "commit": PINNED_LIGER_COMMIT,
        "metadata_vcs_commit": installed_commit,
        "archive_url_pinned": archive_pinned,
        "implementation_files": implementation_files,
    }


def _run_patched_qwen_forward_case(numerical_acceptance: dict, optimizer_config: dict) -> dict:
    """Compare the exact production patch interface with dense Qwen loss."""

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
    )
    torch.manual_seed(20260718)
    dense = Qwen3_5ForCausalLM(config).cuda().train()
    selective = Qwen3_5ForCausalLM(config).cuda().train()
    selective.load_state_dict(dense.state_dict(), strict=True)
    dense_parameter_dtypes = sorted({str(parameter.dtype) for parameter in dense.parameters()})
    selective_parameter_dtypes = sorted({str(parameter.dtype) for parameter in selective.parameters()})
    if dense_parameter_dtypes != ["torch.float32"] or selective_parameter_dtypes != ["torch.float32"]:
        raise AssertionError(
            "patched-Qwen parameters are not exclusively FP32: "
            f"dense={dense_parameter_dtypes}, selective={selective_parameter_dtypes}"
        )
    apply_liger_kernel_to_qwen3_5(
        rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=False, swiglu=False, model=selective
    )
    if "liger_kernel" not in selective.forward.__module__:
        raise RuntimeError("Liger did not patch the Qwen3.5 CausalLM forward")

    input_ids = torch.tensor([[1, 7, 2, 11, 3, 5, 13, 17]], dtype=torch.long, device="cuda")
    labels = input_ids.clone()
    labels[:, [0, 2, 4, 5]] = -100
    shifted = labels[:, 1:]
    selected_positions = torch.nonzero(shifted[0].ne(-100), as_tuple=False).flatten()
    selected_targets = shifted[0, selected_positions].contiguous()
    global_divisor = int(selected_targets.numel()) + 9

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        autocast_contract = _active_bf16_autocast_contract()
        dense_loss = dense(input_ids=input_ids, labels=labels, num_items_in_batch=global_divisor, use_cache=False).loss
        selective_loss = selective(
            input_ids=input_ids,
            labels=labels,
            logits_to_keep=selected_positions,
            shift_labels=selected_targets,
            num_items_in_batch=global_divisor,
            use_cache=False,
        ).loss
    dense_loss.backward()
    selective_loss.backward()

    loss_metrics = scalar_comparison_metrics(float(selective_loss), float(dense_loss))
    validate_comparison_metrics(loss_metrics, numerical_acceptance, kind="loss", context="patched Qwen loss")
    gradient_metrics: dict[str, dict] = {}
    dense_gradient_vector = []
    selective_gradient_vector = []
    for (dense_name, dense_parameter), (selective_name, selective_parameter) in zip(
        dense.named_parameters(), selective.named_parameters(), strict=True
    ):
        if dense_name != selective_name:
            raise RuntimeError(f"Qwen parameter-order drift: {dense_name!r} != {selective_name!r}")
        if dense_parameter.grad is None or selective_parameter.grad is None:
            raise RuntimeError(f"Qwen selective parity found a disconnected parameter: {dense_name}")
        metrics = tensor_comparison_metrics(selective_parameter.grad, dense_parameter.grad)
        validate_comparison_metrics(
            metrics, numerical_acceptance, kind="gradient", context=f"patched Qwen parameter {dense_name}"
        )
        gradient_metrics[dense_name] = metrics
        dense_gradient_vector.append(dense_parameter.grad.detach().reshape(-1))
        selective_gradient_vector.append(selective_parameter.grad.detach().reshape(-1))
    aggregate_gradient_metrics = tensor_comparison_metrics(
        torch.cat(selective_gradient_vector), torch.cat(dense_gradient_vector)
    )
    dense_gradient_dtypes = sorted({str(parameter.grad.dtype) for parameter in dense.parameters()})
    selective_gradient_dtypes = sorted({str(parameter.grad.dtype) for parameter in selective.parameters()})
    if dense_gradient_dtypes != ["torch.float32"] or selective_gradient_dtypes != ["torch.float32"]:
        raise AssertionError(
            "patched-Qwen gradients are not exclusively FP32: "
            f"dense={dense_gradient_dtypes}, selective={selective_gradient_dtypes}"
        )
    validate_comparison_metrics(
        aggregate_gradient_metrics, numerical_acceptance, kind="gradient", context="patched Qwen aggregate gradient"
    )

    dense_initial = {name: parameter.detach().clone() for name, parameter in dense.named_parameters()}
    selective_initial = {name: parameter.detach().clone() for name, parameter in selective.named_parameters()}
    dense_optimizer = torch.optim.AdamW(
        dense.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=(optimizer_config["adam_beta1"], optimizer_config["adam_beta2"]),
        eps=optimizer_config["adam_epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    selective_optimizer = torch.optim.AdamW(
        selective.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=(optimizer_config["adam_beta1"], optimizer_config["adam_beta2"]),
        eps=optimizer_config["adam_epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    torch.nn.utils.clip_grad_norm_(dense.parameters(), optimizer_config["max_gradient_norm"])
    torch.nn.utils.clip_grad_norm_(selective.parameters(), optimizer_config["max_gradient_norm"])
    dense_optimizer.step()
    selective_optimizer.step()
    dense_optimizer_dtypes = _floating_optimizer_state_dtypes(dense_optimizer)
    selective_optimizer_dtypes = _floating_optimizer_state_dtypes(selective_optimizer)
    if dense_optimizer_dtypes != ["torch.float32"] or selective_optimizer_dtypes != ["torch.float32"]:
        raise AssertionError(
            "patched-Qwen AdamW optimizer state is not exclusively FP32: "
            f"dense={dense_optimizer_dtypes}, selective={selective_optimizer_dtypes}"
        )
    dense_updates = []
    selective_updates = []
    for name, parameter in dense.named_parameters():
        dense_updates.append((parameter.detach() - dense_initial[name]).reshape(-1))
    for name, parameter in selective.named_parameters():
        selective_updates.append((parameter.detach() - selective_initial[name]).reshape(-1))
    update_metrics = tensor_comparison_metrics(torch.cat(selective_updates), torch.cat(dense_updates))
    validate_comparison_metrics(update_metrics, numerical_acceptance, kind="update", context="patched Qwen update")
    return {
        "model_class": type(selective).__name__,
        "patched_forward_module": selective.forward.__module__,
        "sequence_tokens": input_ids.numel(),
        "supervised_targets": selected_targets.numel(),
        "global_divisor": global_divisor,
        "autocast_contract": autocast_contract,
        "parameter_dtypes": {
            "selective": selective_parameter_dtypes,
            "dense_reference": dense_parameter_dtypes,
        },
        "gradient_dtypes": {
            "selective": selective_gradient_dtypes,
            "dense_reference": dense_gradient_dtypes,
        },
        "optimizer_floating_state_dtypes": {
            "selective": selective_optimizer_dtypes,
            "dense_reference": dense_optimizer_dtypes,
        },
        "selected_positions": selected_positions.tolist(),
        "dense_loss": float(dense_loss),
        "selective_loss": float(selective_loss),
        "loss_comparison": loss_metrics,
        "aggregate_gradient_comparison": aggregate_gradient_metrics,
        "parameter_gradient_comparisons": gradient_metrics,
        "aggregate_adamw_update_comparison": update_metrics,
        "checked_parameter_gradients": len(gradient_metrics),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("selective Liger qualification requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    report_base = {
        "artifact": "qwen35_selective_liger_loss_qualification",
        "schema_version": 1,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "direct_hidden_size": args.hidden_size,
        "direct_vocab_size": args.vocab_size,
        "precision_policy": {
            "parameters": "torch.float32",
            "gradients": "dtype follows FP32 parameter storage; selected BF16 hidden-row leaf gradients are BF16",
            "adamw_moments": "torch.float32",
            "forward_backward_autocast": "torch.bfloat16",
            "loss_accumulation": "torch.float32",
        },
    }
    numerical_acceptance = qualification["numerical_acceptance"]
    optimizer_config = qualification["training_unit"]
    try:
        liger_source = _verify_liger_source_pin(qualification["runtime_pins"]["liger_source_files_sha256"])
        cases = [
            _run_case(
                seed=1,
                rows=64,
                supervised_rows=list(range(64)),
                hidden_size=args.hidden_size,
                vocab_size=args.vocab_size,
                global_divisor_extra=0,
                numerical_acceptance=numerical_acceptance,
                optimizer_config=optimizer_config,
            ),
            _run_case(
                seed=2,
                rows=64,
                supervised_rows=[0, 7, 31, 63],
                hidden_size=args.hidden_size,
                vocab_size=args.vocab_size,
                global_divisor_extra=19,
                numerical_acceptance=numerical_acceptance,
                optimizer_config=optimizer_config,
            ),
            _run_case(
                seed=3,
                rows=64,
                supervised_rows=[63],
                hidden_size=args.hidden_size,
                vocab_size=args.vocab_size,
                global_divisor_extra=127,
                numerical_acceptance=numerical_acceptance,
                optimizer_config=optimizer_config,
            ),
        ]
        report = {
            **report_base,
            "status": "passed",
            "liger_kernel": liger_source,
            "numerical_acceptance": numerical_acceptance,
            "cases": cases,
            "zero_target_sentinel": _run_zero_target_sentinel(args.hidden_size, args.vocab_size),
            "patched_qwen_forward": _run_patched_qwen_forward_case(numerical_acceptance, optimizer_config),
        }
    except Exception as error:
        failure_report = {
            **report_base,
            "status": "failed",
            "failure": {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "allowed_conclusion": "H2 did not pass; inspect the recorded exception and do not run successor gates.",
        }
        _write_strict_json_atomic(args.report_output, failure_report)
        raise
    _write_strict_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
