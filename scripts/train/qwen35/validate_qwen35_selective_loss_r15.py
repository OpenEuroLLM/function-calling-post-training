#!/usr/bin/env python3
"""R15 CUDA qualification for selective Liger downstream training parity."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from scripts.train.qwen35 import validate_qwen35_selective_loss as r14_assay
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification import (
    collect_h2_numerical_decisions,
    load_qualification_manifest,
    scalar_comparison_metrics,
    tensor_comparison_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    return parser.parse_args()


def _optimizer(parameters, config: dict[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
    )


def _supervised_rows(case: dict[str, Any]) -> list[int]:
    if case["supervision_kind"] == "all":
        rows = list(range(case["rows"]))
    elif case["supervision_kind"] == "explicit":
        rows = list(case["supervised_rows"])
    else:
        raise ValueError(f"unsupported direct supervision kind: {case['supervision_kind']!r}")
    if len(rows) != case["expected_supervised_count"]:
        raise ValueError(f"direct case {case['case_id']} supervised-count drift")
    if len(set(rows)) != len(rows) or rows != sorted(rows) or any(not 0 <= row < case["rows"] for row in rows):
        raise ValueError(f"direct case {case['case_id']} has invalid supervised rows")
    return rows


def _direct_case(
    *, case: dict[str, Any], hidden_size: int, vocab_size: int, heldout_rows: int, optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    supervised_rows = _supervised_rows(case)
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    hidden = (
        torch.randn(case["rows"], hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16)
        * case["hidden_scale"]
    )
    initial_weight = (
        torch.randn(vocab_size, hidden_size, generator=generator, device="cuda", dtype=torch.float32)
        * case["weight_standard_deviation"]
    )
    targets = torch.randint(0, vocab_size, (len(supervised_rows),), generator=generator, device="cuda")
    selective_rows = hidden[supervised_rows].detach().clone().requires_grad_(True)
    reference_rows = selective_rows.detach().clone().requires_grad_(True)
    selective_weight = initial_weight.detach().clone().requires_grad_(True)
    reference_weight = initial_weight.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        selective_autocast = r14_assay._active_bf16_autocast_contract()
        selective_loss = (
            LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(
                selective_weight, selective_rows, targets
            )
            / case["global_divisor"]
        )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_autocast = r14_assay._active_bf16_autocast_contract()
        reference_logits = F.linear(reference_rows, reference_weight)
    reference_loss = F.cross_entropy(reference_logits.float(), targets, reduction="sum") / case["global_divisor"]
    selective_loss.backward()
    reference_loss.backward()

    selective_optimizer = _optimizer([selective_weight], optimizer_config)
    reference_optimizer = _optimizer([reference_weight], optimizer_config)
    selective_optimizer.step()
    reference_optimizer.step()
    selective_optimizer_dtypes = r14_assay._floating_optimizer_state_dtypes(selective_optimizer)
    reference_optimizer_dtypes = r14_assay._floating_optimizer_state_dtypes(reference_optimizer)
    selective_update = selective_weight.detach() - initial_weight
    reference_update = reference_weight.detach() - initial_weight

    heldout_hidden = torch.randn(heldout_rows, hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16)
    heldout_targets = torch.randint(0, vocab_size, (heldout_rows,), generator=generator, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        heldout_autocast = r14_assay._active_bf16_autocast_contract()
        selective_heldout_logits = F.linear(heldout_hidden, selective_weight)
        reference_heldout_logits = F.linear(heldout_hidden, reference_weight)
    selective_heldout_loss = F.cross_entropy(selective_heldout_logits.float(), heldout_targets)
    reference_heldout_loss = F.cross_entropy(reference_heldout_logits.float(), heldout_targets)

    return {
        "case_contract": case,
        "supervised_rows_expanded": supervised_rows,
        "autocast_contract": {
            "selective": selective_autocast,
            "dense_reference": reference_autocast,
            "heldout": heldout_autocast,
        },
        "dtypes": {
            "hidden_input": str(selective_rows.dtype),
            "output_head_parameter": str(selective_weight.dtype),
            "selective_hidden_gradient": str(selective_rows.grad.dtype),
            "reference_hidden_gradient": str(reference_rows.grad.dtype),
            "selective_output_head_gradient": str(selective_weight.grad.dtype),
            "reference_output_head_gradient": str(reference_weight.grad.dtype),
            "selective_optimizer_floating_state": selective_optimizer_dtypes,
            "reference_optimizer_floating_state": reference_optimizer_dtypes,
            "loss_accumulation": "torch.float32",
        },
        "selective_loss": float(selective_loss.detach().item()),
        "reference_loss": float(reference_loss.detach().item()),
        "loss_comparison": scalar_comparison_metrics(
            float(selective_loss.detach().item()), float(reference_loss.detach().item())
        ),
        "selected_hidden_gradient_comparison": tensor_comparison_metrics(selective_rows.grad, reference_rows.grad),
        "output_head_gradient_comparison": tensor_comparison_metrics(selective_weight.grad, reference_weight.grad),
        "raw_first_adamw_update_comparison_diagnostic": tensor_comparison_metrics(selective_update, reference_update),
        "post_step_parameter_comparison": tensor_comparison_metrics(
            selective_weight.detach(), reference_weight.detach()
        ),
        "heldout": {
            "rows": heldout_rows,
            "logit_comparison": tensor_comparison_metrics(selective_heldout_logits, reference_heldout_logits),
            "selective_loss": float(selective_heldout_loss.detach().item()),
            "reference_loss": float(reference_heldout_loss.detach().item()),
            "loss_comparison": scalar_comparison_metrics(
                float(selective_heldout_loss.detach().item()), float(reference_heldout_loss.detach().item())
            ),
        },
    }


def _trajectory_batch(*, seed: int, step_index: int, trajectory_index: int, h2: dict[str, Any]) -> dict[str, Any]:
    sequence_length = h2["trajectory_sequence_length"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        1, h2["trajectory_model_config"]["vocab_size"], (1, sequence_length), generator=generator
    )
    input_ids = input_ids.cuda()
    labels = input_ids.clone()
    modulus = h2["trajectory_supervision_moduli"][step_index % len(h2["trajectory_supervision_moduli"])]
    offset = (step_index + trajectory_index) % modulus
    positions = torch.arange(sequence_length, device="cuda")
    labels[:, ((positions + offset) % modulus).ne(0)] = -100
    shifted = labels[:, 1:]
    selected_positions = torch.nonzero(shifted[0].ne(-100), as_tuple=False).flatten().contiguous()
    selected_targets = shifted[0, selected_positions].contiguous()
    divisor_extra = (step_index * h2["trajectory_divisor_extra_multiplier"] + trajectory_index) % h2[
        "trajectory_divisor_extra_modulus"
    ]
    global_divisor = int(selected_targets.numel()) + divisor_extra
    if not selected_targets.numel() or global_divisor <= 0:
        raise RuntimeError("trajectory batch unexpectedly has no supervised target")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "selected_positions": selected_positions,
        "selected_targets": selected_targets,
        "accounting": {
            "seed": seed,
            "sequence_length": sequence_length,
            "supervision_modulus": modulus,
            "supervision_offset": offset,
            "supervised_targets": int(selected_targets.numel()),
            "divisor_extra": divisor_extra,
            "global_divisor": global_divisor,
        },
    }


def _heldout_batch(*, seed: int, h2: dict[str, Any]) -> dict[str, Any]:
    sequence_length = h2["trajectory_sequence_length"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        1, h2["trajectory_model_config"]["vocab_size"], (1, sequence_length), generator=generator
    )
    input_ids = input_ids.cuda()
    labels = input_ids.clone()
    positions = torch.arange(sequence_length, device="cuda")
    labels[:, positions.remainder(h2["trajectory_heldout_supervision_modulus"]).ne(0)] = -100
    supervised_targets = int(labels[:, 1:].ne(-100).sum().item())
    return {
        "input_ids": input_ids,
        "labels": labels,
        "global_divisor": supervised_targets + h2["trajectory_heldout_divisor_extra"],
        "supervised_targets": supervised_targets,
    }


def _flatten_named(values: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([values[name].detach().reshape(-1) for name in names])


def _optimizer_step_counters(optimizer: torch.optim.Optimizer) -> list[int]:
    counters = []
    for state in optimizer.state.values():
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            counters.append(int(step.item()))
        elif step is not None:
            counters.append(int(step))
    if not counters:
        raise RuntimeError("AdamW produced no step counters")
    return sorted(set(counters))


def _trajectory(
    *, contract: dict[str, Any], trajectory_index: int, h2: dict[str, Any], optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    torch.manual_seed(contract["model_seed"])
    dense = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    selective = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    selective.load_state_dict(dense.state_dict(), strict=True)
    apply_liger_kernel_to_qwen3_5(
        rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=False, swiglu=False, model=selective
    )
    if "liger_kernel" not in selective.forward.__module__:
        raise RuntimeError("Liger did not patch the confirmatory Qwen3.5 forward")
    names = [name for name, _ in dense.named_parameters()]
    if names != [name for name, _ in selective.named_parameters()]:
        raise RuntimeError("dense/selective parameter-name order drift")
    parameter_geometry = [
        {"name": name, "shape": list(parameter.shape), "elements": int(parameter.numel())}
        for name, parameter in dense.named_parameters()
    ]
    if parameter_geometry != h2["trajectory_parameter_geometry"]:
        raise RuntimeError("trajectory parameter geometry drifted from the frozen contract")
    if sorted({str(parameter.dtype) for parameter in dense.parameters()}) != ["torch.float32"]:
        raise RuntimeError("dense trajectory parameters are not exclusively FP32")
    if sorted({str(parameter.dtype) for parameter in selective.parameters()}) != ["torch.float32"]:
        raise RuntimeError("selective trajectory parameters are not exclusively FP32")
    dense_optimizer = _optimizer(dense.parameters(), optimizer_config)
    selective_optimizer = _optimizer(selective.parameters(), optimizer_config)
    heldout = _heldout_batch(seed=contract["heldout_seed"], h2=h2)
    steps = []

    for step_index in range(h2["trajectory_steps"]):
        batch = _trajectory_batch(
            seed=contract["batch_seed_base"] + step_index,
            step_index=step_index,
            trajectory_index=trajectory_index,
            h2=h2,
        )
        dense_optimizer.zero_grad(set_to_none=True)
        selective_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            autocast_contract = r14_assay._active_bf16_autocast_contract()
            dense_loss = dense(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                num_items_in_batch=batch["accounting"]["global_divisor"],
                use_cache=False,
            ).loss
            selective_loss = selective(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                logits_to_keep=batch["selected_positions"],
                shift_labels=batch["selected_targets"],
                num_items_in_batch=batch["accounting"]["global_divisor"],
                use_cache=False,
            ).loss
        dense_loss.backward()
        selective_loss.backward()

        dense_parameters = dict(dense.named_parameters())
        selective_parameters = dict(selective.named_parameters())
        if any(dense_parameters[name].grad is None or selective_parameters[name].grad is None for name in names):
            raise RuntimeError("trajectory found a disconnected named parameter")
        dense_preclip = {name: dense_parameters[name].grad.detach().clone() for name in names}
        selective_preclip = {name: selective_parameters[name].grad.detach().clone() for name in names}
        dense_before = {name: dense_parameters[name].detach().clone() for name in names}
        selective_before = {name: selective_parameters[name].detach().clone() for name in names}
        dense_preclip_vector = _flatten_named(dense_preclip, names)
        selective_preclip_vector = _flatten_named(selective_preclip, names)

        dense_preclip_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), optimizer_config["max_gradient_norm"])
        selective_preclip_norm = torch.nn.utils.clip_grad_norm_(
            selective.parameters(), optimizer_config["max_gradient_norm"]
        )
        dense_clipped = {name: dense_parameters[name].grad.detach().clone() for name in names}
        selective_clipped = {name: selective_parameters[name].grad.detach().clone() for name in names}
        dense_clipped_vector = _flatten_named(dense_clipped, names)
        selective_clipped_vector = _flatten_named(selective_clipped, names)
        dense_optimizer.step()
        selective_optimizer.step()

        dense_after = {name: dense_parameters[name].detach().clone() for name in names}
        selective_after = {name: selective_parameters[name].detach().clone() for name in names}
        dense_updates = {name: dense_after[name] - dense_before[name] for name in names}
        selective_updates = {name: selective_after[name] - selective_before[name] for name in names}
        dense_update_vector = _flatten_named(dense_updates, names)
        selective_update_vector = _flatten_named(selective_updates, names)
        dense_parameter_vector = _flatten_named(dense_after, names)
        selective_parameter_vector = _flatten_named(selective_after, names)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            heldout_autocast = r14_assay._active_bf16_autocast_contract()
            dense_logits = dense(input_ids=heldout["input_ids"], use_cache=False).logits
            selective_logits = selective(input_ids=heldout["input_ids"], use_cache=False).logits
        dense_heldout_loss = (
            F.cross_entropy(
                dense_logits[:, :-1].float().reshape(-1, dense_logits.shape[-1]),
                heldout["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / heldout["global_divisor"]
        )
        selective_heldout_loss = (
            F.cross_entropy(
                selective_logits[:, :-1].float().reshape(-1, selective_logits.shape[-1]),
                heldout["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / heldout["global_divisor"]
        )

        per_parameter = {}
        for name in names:
            per_parameter[name] = {
                "elements": int(dense_parameters[name].numel()),
                "preclip_gradient_comparison": tensor_comparison_metrics(selective_preclip[name], dense_preclip[name]),
                "clipped_gradient_comparison": tensor_comparison_metrics(selective_clipped[name], dense_clipped[name]),
            }
        steps.append(
            {
                "step": step_index + 1,
                "batch_accounting": batch["accounting"],
                "autocast_contract": {"training": autocast_contract, "heldout": heldout_autocast},
                "selective_loss": float(selective_loss.detach().item()),
                "reference_loss": float(dense_loss.detach().item()),
                "training_loss_comparison": scalar_comparison_metrics(
                    float(selective_loss.detach().item()), float(dense_loss.detach().item())
                ),
                "aggregate_preclip_gradient_comparison": tensor_comparison_metrics(
                    selective_preclip_vector, dense_preclip_vector
                ),
                "aggregate_clipped_gradient_comparison": tensor_comparison_metrics(
                    selective_clipped_vector, dense_clipped_vector
                ),
                "per_parameter_gradient_comparisons": per_parameter,
                "preclip_gradient_norms": {
                    "selective": float(selective_preclip_norm.detach().item()),
                    "dense_reference": float(dense_preclip_norm.detach().item()),
                },
                "raw_adamw_update_comparison": tensor_comparison_metrics(selective_update_vector, dense_update_vector),
                "raw_adamw_update_is_gating": step_index + 1 >= h2["raw_update_gating_starts_at_step"],
                "post_step_parameter_comparison": tensor_comparison_metrics(
                    selective_parameter_vector, dense_parameter_vector
                ),
                "optimizer_floating_state_dtypes": {
                    "selective": r14_assay._floating_optimizer_state_dtypes(selective_optimizer),
                    "dense_reference": r14_assay._floating_optimizer_state_dtypes(dense_optimizer),
                },
                "optimizer_step_counters": {
                    "selective": _optimizer_step_counters(selective_optimizer),
                    "dense_reference": _optimizer_step_counters(dense_optimizer),
                },
                "gradient_dtypes": {
                    "selective": sorted({str(parameter.grad.dtype) for parameter in selective.parameters()}),
                    "dense_reference": sorted({str(parameter.grad.dtype) for parameter in dense.parameters()}),
                },
                "heldout": {
                    "supervised_targets": heldout["supervised_targets"],
                    "global_divisor": heldout["global_divisor"],
                    "logit_comparison": tensor_comparison_metrics(selective_logits, dense_logits),
                    "selective_loss": float(selective_heldout_loss.detach().item()),
                    "reference_loss": float(dense_heldout_loss.detach().item()),
                    "loss_comparison": scalar_comparison_metrics(
                        float(selective_heldout_loss.detach().item()), float(dense_heldout_loss.detach().item())
                    ),
                },
            }
        )

    return {
        "trajectory_contract": contract,
        "trajectory_index": trajectory_index,
        "model_class": type(selective).__name__,
        "dense_forward_module": dense.forward.__module__,
        "patched_forward_module": selective.forward.__module__,
        "model_config": h2["trajectory_model_config"],
        "parameter_names": names,
        "parameter_geometry": parameter_geometry,
        "parameter_count": int(sum(parameter.numel() for parameter in dense.parameters())),
        "parameter_dtypes": {
            "selective": sorted({str(parameter.dtype) for parameter in selective.parameters()}),
            "dense_reference": sorted({str(parameter.dtype) for parameter in dense.parameters()}),
        },
        "heldout_contract": {
            "seed": contract["heldout_seed"],
            "sequence_length": h2["trajectory_sequence_length"],
            "supervision_modulus": h2["trajectory_heldout_supervision_modulus"],
            "supervised_targets": heldout["supervised_targets"],
            "divisor_extra": h2["trajectory_heldout_divisor_extra"],
            "global_divisor": heldout["global_divisor"],
        },
        "steps": steps,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R15 selective-Liger qualification requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h2 = qualification["h2_acceptance"]
    optimizer_config = qualification["training_unit"]
    report_base = {
        "artifact": "qwen35_selective_liger_downstream_qualification",
        "schema_version": 2,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "direct_hidden_size": h2["direct_hidden_size"],
        "direct_vocab_size": h2["direct_vocab_size"],
        "precision_policy": {
            "parameters": "torch.float32",
            "gradients": "dtype follows FP32 parameter storage; direct selected BF16 hidden-row leaf gradients are BF16",
            "adamw_moments": "torch.float32",
            "forward_backward_autocast": "torch.bfloat16",
            "loss_accumulation": "torch.float32",
        },
        "numerical_acceptance": qualification["numerical_acceptance"],
        "h2_acceptance": h2,
        "scientific_training_authorized": False,
    }
    historical_cases = []
    confirmatory_cases = []
    trajectories = []
    source = None
    zero = None
    try:
        source = r14_assay._verify_liger_source_pin(qualification["runtime_pins"]["liger_source_files_sha256"])
        for case in h2["historical_direct_cases"]:
            historical_cases.append(
                _direct_case(
                    case=case,
                    hidden_size=h2["direct_hidden_size"],
                    vocab_size=h2["direct_vocab_size"],
                    heldout_rows=h2["direct_heldout_rows"],
                    optimizer_config=optimizer_config,
                )
            )
        for case in h2["confirmatory_direct_cases"]:
            confirmatory_cases.append(
                _direct_case(
                    case=case,
                    hidden_size=h2["direct_hidden_size"],
                    vocab_size=h2["direct_vocab_size"],
                    heldout_rows=h2["direct_heldout_rows"],
                    optimizer_config=optimizer_config,
                )
            )
        zero = r14_assay._run_zero_target_sentinel(h2["direct_hidden_size"], h2["direct_vocab_size"])
        for trajectory_index, contract in enumerate(h2["confirmatory_trajectories"]):
            trajectories.append(
                _trajectory(
                    contract=contract, trajectory_index=trajectory_index, h2=h2, optimizer_config=optimizer_config
                )
            )
        report = {
            **report_base,
            "liger_kernel": source,
            "historical_direct_cases": historical_cases,
            "confirmatory_direct_cases": confirmatory_cases,
            "zero_target_sentinel": zero,
            "confirmatory_trajectories": trajectories,
        }
        decision = collect_h2_numerical_decisions(report, qualification)
        report["decision"] = decision
        report["status"] = decision["status"]
        report["successor_gate_authorized"] = report["status"] == "passed"
        if report["status"] != "passed":
            report["allowed_conclusion"] = "R15 H2 did not pass; H3 and all later gates remain blocked."
        r14_assay._write_strict_json_atomic(args.report_output, report)
        if report["status"] != "passed":
            raise AssertionError(f"R15 H2 failed {len(decision['failed_gating_checks'])} gating checks")
    except Exception as error:
        if not args.report_output.exists():
            failure_report = {
                **report_base,
                "status": "failed",
                "successor_gate_authorized": False,
                "liger_kernel": source,
                "historical_direct_cases": historical_cases,
                "confirmatory_direct_cases": confirmatory_cases,
                "zero_target_sentinel": zero,
                "confirmatory_trajectories": trajectories,
                "failure": {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "allowed_conclusion": "R15 H2 did not complete; H3 and all later gates remain blocked.",
            }
            r14_assay._write_strict_json_atomic(args.report_output, failure_report)
        raise
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
