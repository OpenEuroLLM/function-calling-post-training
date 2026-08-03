#!/usr/bin/env python3
"""R16 CUDA qualification for selective Liger downstream training parity."""

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
from scripts.train.qwen35 import validate_qwen35_selective_loss_r15 as r15_assay
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification_r16 import (
    balanced_tensor_comparison_metrics,
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


def _json_finite(value: torch.Tensor | float) -> float | None:
    result = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    return result if torch.isfinite(torch.tensor(result, dtype=torch.float64)).item() else None


def _optimizer_moments(
    optimizer: torch.optim.Optimizer, parameters: dict[str, torch.nn.Parameter], names: list[str]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    exp_avg: dict[str, torch.Tensor] = {}
    exp_avg_sq: dict[str, torch.Tensor] = {}
    for name in names:
        state = optimizer.state[parameters[name]]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"unexpected AdamW state fields for {name}: {sorted(state)}")
        exp_avg[name] = state["exp_avg"].detach().clone()
        exp_avg_sq[name] = state["exp_avg_sq"].detach().clone()
    return exp_avg, exp_avg_sq


def _direct_case(
    *, case: dict[str, Any], hidden_size: int, vocab_size: int, heldout_rows: int, optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    supervised_rows = r15_assay._supervised_rows(case)
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

    selective_optimizer = r15_assay._optimizer([selective_weight], optimizer_config)
    reference_optimizer = r15_assay._optimizer([reference_weight], optimizer_config)
    selective_optimizer.step()
    reference_optimizer.step()
    selective_update = selective_weight.detach() - initial_weight
    reference_update = reference_weight.detach() - initial_weight
    selective_state = selective_optimizer.state[selective_weight]
    reference_state = reference_optimizer.state[reference_weight]

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
            "selective_optimizer_floating_state": r14_assay._floating_optimizer_state_dtypes(selective_optimizer),
            "reference_optimizer_floating_state": r14_assay._floating_optimizer_state_dtypes(reference_optimizer),
            "loss_accumulation": "torch.float32",
        },
        "optimizer_step_counters": {
            "selective": r15_assay._optimizer_step_counters(selective_optimizer),
            "dense_reference": r15_assay._optimizer_step_counters(reference_optimizer),
        },
        "selective_loss": _json_finite(selective_loss),
        "reference_loss": _json_finite(reference_loss),
        "loss_comparison": scalar_comparison_metrics(
            float(selective_loss.detach().item()), float(reference_loss.detach().item())
        ),
        "selected_hidden_gradient_comparison": tensor_comparison_metrics(selective_rows.grad, reference_rows.grad),
        "output_head_gradient_comparison": tensor_comparison_metrics(selective_weight.grad, reference_weight.grad),
        "raw_first_adamw_update_comparison_diagnostic": tensor_comparison_metrics(selective_update, reference_update),
        "optimizer_exp_avg_comparison": tensor_comparison_metrics(
            selective_state["exp_avg"], reference_state["exp_avg"]
        ),
        "optimizer_exp_avg_sq_comparison": tensor_comparison_metrics(
            selective_state["exp_avg_sq"], reference_state["exp_avg_sq"]
        ),
        "post_step_parameter_comparison": tensor_comparison_metrics(
            selective_weight.detach(), reference_weight.detach()
        ),
        "heldout": {
            "rows": heldout_rows,
            "logit_comparison": tensor_comparison_metrics(selective_heldout_logits, reference_heldout_logits),
            "selective_loss": _json_finite(selective_heldout_loss),
            "reference_loss": _json_finite(reference_heldout_loss),
            "loss_comparison": scalar_comparison_metrics(
                float(selective_heldout_loss.detach().item()), float(reference_heldout_loss.detach().item())
            ),
        },
    }


def _balanced_named(
    observed: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    names: list[str],
    aggregate_metrics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        name: balanced_tensor_comparison_metrics(
            observed[name],
            reference[name],
            aggregate_reference_l2_norm=aggregate_metrics["reference_l2_norm"],
            aggregate_elements=int(aggregate_metrics["elements"]),
        )
        for name in names
    }


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
        raise RuntimeError("Liger did not patch the R16 Qwen3.5 forward")
    names = [name for name, _ in dense.named_parameters()]
    if names != [name for name, _ in selective.named_parameters()]:
        raise RuntimeError("dense/selective parameter-name order drift")
    geometry = [
        {"name": name, "shape": list(parameter.shape), "elements": int(parameter.numel())}
        for name, parameter in dense.named_parameters()
    ]
    if geometry != h2["trajectory_parameter_geometry"]:
        raise RuntimeError("R16 trajectory parameter geometry drift")
    if sorted({str(parameter.dtype) for parameter in dense.parameters()}) != ["torch.float32"]:
        raise RuntimeError("dense R16 trajectory parameters are not exclusively FP32")
    if sorted({str(parameter.dtype) for parameter in selective.parameters()}) != ["torch.float32"]:
        raise RuntimeError("selective R16 trajectory parameters are not exclusively FP32")
    dense_parameters = dict(dense.named_parameters())
    selective_parameters = dict(selective.named_parameters())
    if any(not torch.equal(dense_parameters[name], selective_parameters[name]) for name in names):
        raise RuntimeError("R16 dense/selective initial parameter state is not bit exact")
    dense_initial = {name: dense_parameters[name].detach().clone() for name in names}
    selective_initial = {name: selective_parameters[name].detach().clone() for name in names}
    dense_optimizer = r15_assay._optimizer(dense.parameters(), optimizer_config)
    selective_optimizer = r15_assay._optimizer(selective.parameters(), optimizer_config)
    heldout = r15_assay._heldout_batch(seed=contract["heldout_seed"], h2=h2)
    steps = []

    for step_index in range(h2["trajectory_steps"]):
        batch = r15_assay._trajectory_batch(
            seed=contract["batch_seed_base"] + step_index,
            step_index=step_index,
            trajectory_index=trajectory_index,
            h2=h2,
        )
        dense_optimizer.zero_grad(set_to_none=True)
        selective_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            training_autocast = r14_assay._active_bf16_autocast_contract()
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
        if any(dense_parameters[name].grad is None or selective_parameters[name].grad is None for name in names):
            raise RuntimeError("R16 trajectory found a disconnected named parameter")
        dense_preclip = {name: dense_parameters[name].grad.detach().clone() for name in names}
        selective_preclip = {name: selective_parameters[name].grad.detach().clone() for name in names}
        dense_before = {name: dense_parameters[name].detach().clone() for name in names}
        selective_before = {name: selective_parameters[name].detach().clone() for name in names}
        dense_preclip_vector = r15_assay._flatten_named(dense_preclip, names)
        selective_preclip_vector = r15_assay._flatten_named(selective_preclip, names)

        dense_preclip_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), optimizer_config["max_gradient_norm"])
        selective_preclip_norm = torch.nn.utils.clip_grad_norm_(
            selective.parameters(), optimizer_config["max_gradient_norm"]
        )
        dense_clipped = {name: dense_parameters[name].grad.detach().clone() for name in names}
        selective_clipped = {name: selective_parameters[name].grad.detach().clone() for name in names}
        dense_clipped_vector = r15_assay._flatten_named(dense_clipped, names)
        selective_clipped_vector = r15_assay._flatten_named(selective_clipped, names)

        dense_optimizer.step()
        selective_optimizer.step()
        dense_after = {name: dense_parameters[name].detach().clone() for name in names}
        selective_after = {name: selective_parameters[name].detach().clone() for name in names}
        dense_updates = {name: dense_after[name] - dense_before[name] for name in names}
        selective_updates = {name: selective_after[name] - selective_before[name] for name in names}
        dense_cumulative = {name: dense_after[name] - dense_initial[name] for name in names}
        selective_cumulative = {name: selective_after[name] - selective_initial[name] for name in names}
        dense_exp_avg, dense_exp_avg_sq = _optimizer_moments(dense_optimizer, dense_parameters, names)
        selective_exp_avg, selective_exp_avg_sq = _optimizer_moments(selective_optimizer, selective_parameters, names)

        tensor_families = {
            "preclip_gradient": (selective_preclip, dense_preclip, selective_preclip_vector, dense_preclip_vector),
            "clipped_gradient": (selective_clipped, dense_clipped, selective_clipped_vector, dense_clipped_vector),
            "raw_adamw_update": (
                selective_updates,
                dense_updates,
                r15_assay._flatten_named(selective_updates, names),
                r15_assay._flatten_named(dense_updates, names),
            ),
            "optimizer_exp_avg": (
                selective_exp_avg,
                dense_exp_avg,
                r15_assay._flatten_named(selective_exp_avg, names),
                r15_assay._flatten_named(dense_exp_avg, names),
            ),
            "optimizer_exp_avg_sq": (
                selective_exp_avg_sq,
                dense_exp_avg_sq,
                r15_assay._flatten_named(selective_exp_avg_sq, names),
                r15_assay._flatten_named(dense_exp_avg_sq, names),
            ),
            "cumulative_parameter_displacement": (
                selective_cumulative,
                dense_cumulative,
                r15_assay._flatten_named(selective_cumulative, names),
                r15_assay._flatten_named(dense_cumulative, names),
            ),
            "post_step_parameter_state": (
                selective_after,
                dense_after,
                r15_assay._flatten_named(selective_after, names),
                r15_assay._flatten_named(dense_after, names),
            ),
        }
        aggregate = {
            family: tensor_comparison_metrics(selective_vector, reference_vector)
            for family, (_, _, selective_vector, reference_vector) in tensor_families.items()
        }
        named_metrics = {
            family: _balanced_named(selective_named, reference_named, names, aggregate[family])
            for family, (selective_named, reference_named, _, _) in tensor_families.items()
        }

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

        per_parameter = {
            name: {
                "elements": int(dense_parameters[name].numel()),
                "preclip_gradient_comparison": named_metrics["preclip_gradient"][name],
                "clipped_gradient_comparison": named_metrics["clipped_gradient"][name],
                "raw_adamw_update_comparison_diagnostic": named_metrics["raw_adamw_update"][name],
                "optimizer_exp_avg_comparison": named_metrics["optimizer_exp_avg"][name],
                "optimizer_exp_avg_sq_comparison": named_metrics["optimizer_exp_avg_sq"][name],
                "cumulative_parameter_displacement_comparison": named_metrics["cumulative_parameter_displacement"][
                    name
                ],
                "post_step_parameter_state_comparison": named_metrics["post_step_parameter_state"][name],
            }
            for name in names
        }
        steps.append(
            {
                "step": step_index + 1,
                "batch_accounting": batch["accounting"],
                "autocast_contract": {"training": training_autocast, "heldout": heldout_autocast},
                "selective_loss": _json_finite(selective_loss),
                "reference_loss": _json_finite(dense_loss),
                "training_loss_comparison": scalar_comparison_metrics(
                    float(selective_loss.detach().item()), float(dense_loss.detach().item())
                ),
                "aggregate_preclip_gradient_comparison": aggregate["preclip_gradient"],
                "aggregate_clipped_gradient_comparison": aggregate["clipped_gradient"],
                "aggregate_raw_adamw_update_comparison_diagnostic": aggregate["raw_adamw_update"],
                "aggregate_optimizer_exp_avg_comparison": aggregate["optimizer_exp_avg"],
                "aggregate_optimizer_exp_avg_sq_comparison": aggregate["optimizer_exp_avg_sq"],
                "aggregate_cumulative_parameter_displacement_comparison": aggregate[
                    "cumulative_parameter_displacement"
                ],
                "aggregate_post_step_parameter_state_comparison": aggregate["post_step_parameter_state"],
                "per_parameter_comparisons": per_parameter,
                "preclip_gradient_norms": {
                    "selective": _json_finite(selective_preclip_norm),
                    "dense_reference": _json_finite(dense_preclip_norm),
                },
                "raw_adamw_updates_are_gating": False,
                "optimizer_floating_state_dtypes": {
                    "selective": r14_assay._floating_optimizer_state_dtypes(selective_optimizer),
                    "dense_reference": r14_assay._floating_optimizer_state_dtypes(dense_optimizer),
                },
                "optimizer_step_counters": {
                    "selective": r15_assay._optimizer_step_counters(selective_optimizer),
                    "dense_reference": r15_assay._optimizer_step_counters(dense_optimizer),
                },
                "gradient_dtypes": {
                    "selective": sorted({str(parameter.grad.dtype) for parameter in selective.parameters()}),
                    "dense_reference": sorted({str(parameter.grad.dtype) for parameter in dense.parameters()}),
                },
                "heldout": {
                    "supervised_targets": heldout["supervised_targets"],
                    "global_divisor": heldout["global_divisor"],
                    "logit_comparison": tensor_comparison_metrics(selective_logits, dense_logits),
                    "selective_loss": _json_finite(selective_heldout_loss),
                    "reference_loss": _json_finite(dense_heldout_loss),
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
        "parameter_geometry": geometry,
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
        raise RuntimeError("R16 selective-Liger qualification requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h2 = qualification["h2_acceptance"]
    optimizer_config = qualification["training_unit"]
    report_base = {
        "artifact": "qwen35_selective_liger_downstream_qualification_r16",
        "schema_version": 3,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "manifest_derivation": qualification["manifest_derivation"],
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
    historical_cases: list[dict[str, Any]] = []
    confirmatory_cases: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
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
            report["allowed_conclusion"] = "R16 H2 did not pass; H3 and all later gates remain blocked."
        r14_assay._write_strict_json_atomic(args.report_output, report)
        if report["status"] != "passed":
            raise AssertionError(f"R16 H2 failed {len(decision['failed_gating_checks'])} gating checks")
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
                "allowed_conclusion": "R16 H2 did not complete; H3 and all later gates remain blocked.",
            }
            r14_assay._write_strict_json_atomic(args.report_output, failure_report)
        raise
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
