#!/usr/bin/env python3
"""R17 CUDA H2 qualification with a computationally matched reference."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from scripts.train.qwen35 import validate_qwen35_selective_loss as r14_assay
from scripts.train.qwen35 import validate_qwen35_selective_loss_r15 as r15_assay
from scripts.train.qwen35 import validate_qwen35_selective_loss_r16 as r16_assay
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification_r17 import (
    balanced_tensor_comparison_metrics,
    collect_h2_numerical_decisions,
    load_qualification_manifest,
    scalar_comparison_metrics,
    tensor_comparison_metrics,
)

FAMILIES = (
    "preclip_gradient",
    "clipped_gradient",
    "raw_adamw_update",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
    "cumulative_parameter_displacement",
    "post_step_parameter_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    return parser.parse_args()


def _json_finite(value: torch.Tensor | float) -> float | None:
    result = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    return result if math.isfinite(result) else None


def _balanced_named(
    observed: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], names: list[str], aggregate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        name: balanced_tensor_comparison_metrics(
            observed[name],
            reference[name],
            aggregate_reference_l2_norm=aggregate["reference_l2_norm"],
            aggregate_elements=int(aggregate["elements"]),
        )
        for name in names
    }


def _direct_comparison(
    *,
    observed_loss: torch.Tensor,
    reference_loss: torch.Tensor,
    observed_hidden_gradient: torch.Tensor,
    reference_hidden_gradient: torch.Tensor,
    observed_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    initial_weight: torch.Tensor,
    observed_optimizer: torch.optim.Optimizer,
    reference_optimizer: torch.optim.Optimizer,
    observed_heldout_logits: torch.Tensor,
    reference_heldout_logits: torch.Tensor,
    heldout_targets: torch.Tensor,
) -> dict[str, Any]:
    observed_state = observed_optimizer.state[observed_weight]
    reference_state = reference_optimizer.state[reference_weight]
    observed_heldout_loss = F.cross_entropy(observed_heldout_logits.float(), heldout_targets)
    reference_heldout_loss = F.cross_entropy(reference_heldout_logits.float(), heldout_targets)
    return {
        "observed_loss": _json_finite(observed_loss),
        "reference_loss": _json_finite(reference_loss),
        "loss_comparison": scalar_comparison_metrics(
            float(observed_loss.detach().item()), float(reference_loss.detach().item())
        ),
        "selected_hidden_gradient_comparison": tensor_comparison_metrics(
            observed_hidden_gradient, reference_hidden_gradient
        ),
        "output_head_gradient_comparison": tensor_comparison_metrics(observed_weight.grad, reference_weight.grad),
        "raw_first_adamw_update_comparison_diagnostic": tensor_comparison_metrics(
            observed_weight.detach() - initial_weight, reference_weight.detach() - initial_weight
        ),
        "optimizer_exp_avg_comparison": tensor_comparison_metrics(
            observed_state["exp_avg"], reference_state["exp_avg"]
        ),
        "optimizer_exp_avg_sq_comparison": tensor_comparison_metrics(
            observed_state["exp_avg_sq"], reference_state["exp_avg_sq"]
        ),
        "post_step_parameter_comparison": tensor_comparison_metrics(
            observed_weight.detach(), reference_weight.detach()
        ),
        "heldout": {
            "logit_comparison": tensor_comparison_metrics(observed_heldout_logits, reference_heldout_logits),
            "observed_loss": _json_finite(observed_heldout_loss),
            "reference_loss": _json_finite(reference_heldout_loss),
            "loss_comparison": scalar_comparison_metrics(
                float(observed_heldout_loss.detach().item()), float(reference_heldout_loss.detach().item())
            ),
        },
    }


def _direct_case(
    *, case: dict[str, Any], hidden_size: int, vocab_size: int, heldout_rows: int, optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    supervised_rows = r15_assay._supervised_rows(case)
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    base_hidden = (
        torch.randn(case["rows"], hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16)
        * case["hidden_scale"]
    )
    initial_weight = (
        torch.randn(vocab_size, hidden_size, generator=generator, device="cuda", dtype=torch.float32)
        * case["weight_standard_deviation"]
    )
    targets = torch.randint(0, vocab_size, (len(supervised_rows),), generator=generator, device="cuda")
    full_targets = torch.full((case["rows"],), -100, device="cuda", dtype=torch.long)
    full_targets[supervised_rows] = targets

    liger_hidden = base_hidden[supervised_rows].detach().clone().requires_grad_(True)
    selected_hidden = base_hidden[supervised_rows].detach().clone().requires_grad_(True)
    full_hidden = base_hidden.detach().clone().requires_grad_(True)
    liger_weight = initial_weight.detach().clone().requires_grad_(True)
    selected_weight = initial_weight.detach().clone().requires_grad_(True)
    full_weight = initial_weight.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        liger_autocast = r14_assay._active_bf16_autocast_contract()
        liger_loss = (
            LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(
                liger_weight, liger_hidden, targets
            )
            / case["global_divisor"]
        )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        selected_autocast = r14_assay._active_bf16_autocast_contract()
        selected_logits = F.linear(selected_hidden, selected_weight)
    selected_loss = F.cross_entropy(selected_logits.float(), targets, reduction="sum") / case["global_divisor"]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_autocast = r14_assay._active_bf16_autocast_contract()
        full_logits = F.linear(full_hidden, full_weight)
    full_loss = (
        F.cross_entropy(full_logits.float(), full_targets, ignore_index=-100, reduction="sum") / case["global_divisor"]
    )

    liger_loss.backward()
    selected_loss.backward()
    full_loss.backward()
    if any(
        value.grad is None
        for value in (liger_hidden, selected_hidden, full_hidden, liger_weight, selected_weight, full_weight)
    ):
        raise RuntimeError(f"{case['case_id']} produced a disconnected direct-case leaf")

    optimizers = {
        "liger": r15_assay._optimizer([liger_weight], optimizer_config),
        "selected": r15_assay._optimizer([selected_weight], optimizer_config),
        "full": r15_assay._optimizer([full_weight], optimizer_config),
    }
    weights = {"liger": liger_weight, "selected": selected_weight, "full": full_weight}
    for name in ("liger", "selected", "full"):
        optimizers[name].step()

    heldout_hidden = torch.randn(heldout_rows, hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16)
    heldout_targets = torch.randint(0, vocab_size, (heldout_rows,), generator=generator, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        heldout_autocast = r14_assay._active_bf16_autocast_contract()
        heldout_logits = {name: F.linear(heldout_hidden, weight) for name, weight in weights.items()}

    primary = _direct_comparison(
        observed_loss=liger_loss,
        reference_loss=selected_loss,
        observed_hidden_gradient=liger_hidden.grad,
        reference_hidden_gradient=selected_hidden.grad,
        observed_weight=liger_weight,
        reference_weight=selected_weight,
        initial_weight=initial_weight,
        observed_optimizer=optimizers["liger"],
        reference_optimizer=optimizers["selected"],
        observed_heldout_logits=heldout_logits["liger"],
        reference_heldout_logits=heldout_logits["selected"],
        heldout_targets=heldout_targets,
    )
    diagnostic = _direct_comparison(
        observed_loss=selected_loss,
        reference_loss=full_loss,
        observed_hidden_gradient=selected_hidden.grad,
        reference_hidden_gradient=full_hidden.grad[supervised_rows],
        observed_weight=selected_weight,
        reference_weight=full_weight,
        initial_weight=initial_weight,
        observed_optimizer=optimizers["selected"],
        reference_optimizer=optimizers["full"],
        observed_heldout_logits=heldout_logits["selected"],
        reference_heldout_logits=heldout_logits["full"],
        heldout_targets=heldout_targets,
    )
    ignored_rows = sorted(set(range(case["rows"])) - set(supervised_rows))
    diagnostic.update(
        {
            "observed_path": "pytorch_dense_selected_rows",
            "reference_path": "pytorch_dense_full_rows_ignore_index",
            "numerical_discrepancy_is_gating": False,
            "integrity_and_finiteness_are_mandatory": True,
            "autocast_contract": {"dense_selected": selected_autocast, "dense_full": full_autocast},
            "optimizer_step_counters": {
                "dense_selected": r15_assay._optimizer_step_counters(optimizers["selected"]),
                "dense_full": r15_assay._optimizer_step_counters(optimizers["full"]),
            },
            "ignored_full_hidden_gradient_nonzero_count": (
                int(torch.count_nonzero(full_hidden.grad[ignored_rows])) if ignored_rows else 0
            ),
        }
    )

    return {
        "case_contract": case,
        "supervised_rows_expanded": supervised_rows,
        "autocast_contract": {
            "selective": liger_autocast,
            "dense_reference": selected_autocast,
            "heldout": heldout_autocast,
        },
        "dtypes": {
            "hidden_input": str(liger_hidden.dtype),
            "output_head_parameter": str(liger_weight.dtype),
            "selective_hidden_gradient": str(liger_hidden.grad.dtype),
            "reference_hidden_gradient": str(selected_hidden.grad.dtype),
            "selective_output_head_gradient": str(liger_weight.grad.dtype),
            "reference_output_head_gradient": str(selected_weight.grad.dtype),
            "selective_optimizer_floating_state": r14_assay._floating_optimizer_state_dtypes(optimizers["liger"]),
            "reference_optimizer_floating_state": r14_assay._floating_optimizer_state_dtypes(optimizers["selected"]),
            "loss_accumulation": "torch.float32",
        },
        "optimizer_step_counters": {
            "selective": r15_assay._optimizer_step_counters(optimizers["liger"]),
            "dense_reference": r15_assay._optimizer_step_counters(optimizers["selected"]),
        },
        "selective_loss": primary["observed_loss"],
        "reference_loss": primary["reference_loss"],
        "loss_comparison": primary["loss_comparison"],
        "selected_hidden_gradient_comparison": primary["selected_hidden_gradient_comparison"],
        "output_head_gradient_comparison": primary["output_head_gradient_comparison"],
        "raw_first_adamw_update_comparison_diagnostic": primary["raw_first_adamw_update_comparison_diagnostic"],
        "optimizer_exp_avg_comparison": primary["optimizer_exp_avg_comparison"],
        "optimizer_exp_avg_sq_comparison": primary["optimizer_exp_avg_sq_comparison"],
        "post_step_parameter_comparison": primary["post_step_parameter_comparison"],
        "heldout": {
            "rows": heldout_rows,
            "logit_comparison": primary["heldout"]["logit_comparison"],
            "selective_loss": primary["heldout"]["observed_loss"],
            "reference_loss": primary["heldout"]["reference_loss"],
            "loss_comparison": primary["heldout"]["loss_comparison"],
        },
        "full_dense_diagnostic": diagnostic,
    }


def _dense_selected_loss(model: Qwen3_5ForCausalLM, batch: dict[str, Any]) -> torch.Tensor:
    hidden = model.model(input_ids=batch["input_ids"], use_cache=False, return_dict=True).last_hidden_state
    selected_hidden = hidden[:, batch["selected_positions"], :]
    selected_logits = F.linear(selected_hidden, model.lm_head.weight)
    return (
        F.cross_entropy(
            selected_logits.float().reshape(-1, selected_logits.shape[-1]),
            batch["selected_targets"].reshape(-1),
            reduction="sum",
        )
        / batch["accounting"]["global_divisor"]
    )


def _pair_family_metrics(
    *, observed: dict[str, dict[str, torch.Tensor]], reference: dict[str, dict[str, torch.Tensor]], names: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    aggregate = {
        family: tensor_comparison_metrics(
            r15_assay._flatten_named(observed[family], names), r15_assay._flatten_named(reference[family], names)
        )
        for family in FAMILIES
    }
    named = {
        family: _balanced_named(observed[family], reference[family], names, aggregate[family]) for family in FAMILIES
    }
    return aggregate, named


def _per_parameter(
    *, names: list[str], parameters: dict[str, torch.nn.Parameter], named: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "elements": int(parameters[name].numel()),
            "preclip_gradient_comparison": named["preclip_gradient"][name],
            "clipped_gradient_comparison": named["clipped_gradient"][name],
            "raw_adamw_update_comparison_diagnostic": named["raw_adamw_update"][name],
            "optimizer_exp_avg_comparison": named["optimizer_exp_avg"][name],
            "optimizer_exp_avg_sq_comparison": named["optimizer_exp_avg_sq"][name],
            "cumulative_parameter_displacement_comparison": named["cumulative_parameter_displacement"][name],
            "post_step_parameter_state_comparison": named["post_step_parameter_state"][name],
        }
        for name in names
    }


def _aggregate_fields(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "aggregate_preclip_gradient_comparison": aggregate["preclip_gradient"],
        "aggregate_clipped_gradient_comparison": aggregate["clipped_gradient"],
        "aggregate_raw_adamw_update_comparison_diagnostic": aggregate["raw_adamw_update"],
        "aggregate_optimizer_exp_avg_comparison": aggregate["optimizer_exp_avg"],
        "aggregate_optimizer_exp_avg_sq_comparison": aggregate["optimizer_exp_avg_sq"],
        "aggregate_cumulative_parameter_displacement_comparison": aggregate["cumulative_parameter_displacement"],
        "aggregate_post_step_parameter_state_comparison": aggregate["post_step_parameter_state"],
    }


def _trajectory(
    *, contract: dict[str, Any], trajectory_index: int, h2: dict[str, Any], optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    torch.manual_seed(contract["model_seed"])
    dense_full = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    dense_selected = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    liger = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    common_state = dense_full.state_dict()
    dense_selected.load_state_dict(common_state, strict=True)
    liger.load_state_dict(common_state, strict=True)
    apply_liger_kernel_to_qwen3_5(
        rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=False, swiglu=False, model=liger
    )
    if "liger_kernel" not in liger.forward.__module__:
        raise RuntimeError("Liger did not patch the R17 Qwen3.5 forward")
    models = {"liger": liger, "dense_selected": dense_selected, "dense_full": dense_full}
    names = [name for name, _ in dense_full.named_parameters()]
    if any([name for name, _ in model.named_parameters()] != names for model in models.values()):
        raise RuntimeError("R17 model parameter-name order drift")
    parameters = {role: dict(model.named_parameters()) for role, model in models.items()}
    geometry = [
        {
            "name": name,
            "shape": list(parameters["dense_full"][name].shape),
            "elements": int(parameters["dense_full"][name].numel()),
        }
        for name in names
    ]
    if geometry != h2["trajectory_parameter_geometry"]:
        raise RuntimeError("R17 trajectory parameter geometry drift")
    if any(
        sorted({str(parameter.dtype) for parameter in model.parameters()}) != ["torch.float32"]
        for model in models.values()
    ):
        raise RuntimeError("R17 trajectory parameters are not exclusively FP32")
    if any(
        not torch.equal(parameters["dense_full"][name], parameters[role][name])
        for role in ("liger", "dense_selected")
        for name in names
    ):
        raise RuntimeError("R17 three-way initial parameter state is not bit exact")
    initial = {role: {name: parameters[role][name].detach().clone() for name in names} for role in models}
    optimizers = {role: r15_assay._optimizer(model.parameters(), optimizer_config) for role, model in models.items()}
    heldout = r15_assay._heldout_batch(seed=contract["heldout_seed"], h2=h2)
    steps = []

    for step_index in range(h2["trajectory_steps"]):
        batch = r15_assay._trajectory_batch(
            seed=contract["batch_seed_base"] + step_index,
            step_index=step_index,
            trajectory_index=trajectory_index,
            h2=h2,
        )
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            training_autocast = r14_assay._active_bf16_autocast_contract()
            full_loss = dense_full(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                num_items_in_batch=batch["accounting"]["global_divisor"],
                use_cache=False,
            ).loss
            selected_loss = _dense_selected_loss(dense_selected, batch)
            liger_loss = liger(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                logits_to_keep=batch["selected_positions"],
                shift_labels=batch["selected_targets"],
                num_items_in_batch=batch["accounting"]["global_divisor"],
                use_cache=False,
            ).loss
        losses = {"liger": liger_loss, "dense_selected": selected_loss, "dense_full": full_loss}
        for loss in losses.values():
            loss.backward()
        if any(parameters[role][name].grad is None for role in models for name in names):
            raise RuntimeError("R17 trajectory found a disconnected named parameter")

        before = {role: {name: parameters[role][name].detach().clone() for name in names} for role in models}
        preclip = {role: {name: parameters[role][name].grad.detach().clone() for name in names} for role in models}
        preclip_norms = {
            role: torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
            for role, model in models.items()
        }
        clipped = {role: {name: parameters[role][name].grad.detach().clone() for name in names} for role in models}
        for optimizer in optimizers.values():
            optimizer.step()
        after = {role: {name: parameters[role][name].detach().clone() for name in names} for role in models}
        updates = {role: {name: after[role][name] - before[role][name] for name in names} for role in models}
        cumulative = {role: {name: after[role][name] - initial[role][name] for name in names} for role in models}
        moments = {role: r16_assay._optimizer_moments(optimizers[role], parameters[role], names) for role in models}
        families = {
            role: {
                "preclip_gradient": preclip[role],
                "clipped_gradient": clipped[role],
                "raw_adamw_update": updates[role],
                "optimizer_exp_avg": moments[role][0],
                "optimizer_exp_avg_sq": moments[role][1],
                "cumulative_parameter_displacement": cumulative[role],
                "post_step_parameter_state": after[role],
            }
            for role in models
        }
        primary_aggregate, primary_named = _pair_family_metrics(
            observed=families["liger"], reference=families["dense_selected"], names=names
        )
        diagnostic_aggregate, diagnostic_named = _pair_family_metrics(
            observed=families["dense_selected"], reference=families["dense_full"], names=names
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            heldout_autocast = r14_assay._active_bf16_autocast_contract()
            heldout_logits = {
                role: model(input_ids=heldout["input_ids"], use_cache=False).logits for role, model in models.items()
            }
        heldout_losses = {
            role: F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                heldout["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / heldout["global_divisor"]
            for role, logits in heldout_logits.items()
        }

        diagnostic = {
            "observed_path": "pytorch_dense_selected_rows",
            "reference_path": "pytorch_dense_full_rows_ignore_index",
            "numerical_discrepancy_is_gating": False,
            "integrity_and_finiteness_are_mandatory": True,
            "observed_loss": _json_finite(selected_loss),
            "reference_loss": _json_finite(full_loss),
            "training_loss_comparison": scalar_comparison_metrics(
                float(selected_loss.detach().item()), float(full_loss.detach().item())
            ),
            **_aggregate_fields(diagnostic_aggregate),
            "per_parameter_comparisons": _per_parameter(
                names=names, parameters=parameters["dense_selected"], named=diagnostic_named
            ),
            "preclip_gradient_norms": {
                "dense_selected": _json_finite(preclip_norms["dense_selected"]),
                "dense_full": _json_finite(preclip_norms["dense_full"]),
            },
            "optimizer_floating_state_dtypes": {
                "dense_selected": r14_assay._floating_optimizer_state_dtypes(optimizers["dense_selected"]),
                "dense_full": r14_assay._floating_optimizer_state_dtypes(optimizers["dense_full"]),
            },
            "optimizer_step_counters": {
                "dense_selected": r15_assay._optimizer_step_counters(optimizers["dense_selected"]),
                "dense_full": r15_assay._optimizer_step_counters(optimizers["dense_full"]),
            },
            "gradient_dtypes": {
                "dense_selected": sorted({str(parameter.grad.dtype) for parameter in dense_selected.parameters()}),
                "dense_full": sorted({str(parameter.grad.dtype) for parameter in dense_full.parameters()}),
            },
            "heldout": {
                "supervised_targets": heldout["supervised_targets"],
                "global_divisor": heldout["global_divisor"],
                "logit_comparison": tensor_comparison_metrics(
                    heldout_logits["dense_selected"], heldout_logits["dense_full"]
                ),
                "observed_loss": _json_finite(heldout_losses["dense_selected"]),
                "reference_loss": _json_finite(heldout_losses["dense_full"]),
                "loss_comparison": scalar_comparison_metrics(
                    float(heldout_losses["dense_selected"].detach().item()),
                    float(heldout_losses["dense_full"].detach().item()),
                ),
            },
        }
        steps.append(
            {
                "step": step_index + 1,
                "batch_accounting": batch["accounting"],
                "autocast_contract": {"training": training_autocast, "heldout": heldout_autocast},
                "selective_loss": _json_finite(liger_loss),
                "reference_loss": _json_finite(selected_loss),
                "training_loss_comparison": scalar_comparison_metrics(
                    float(liger_loss.detach().item()), float(selected_loss.detach().item())
                ),
                **_aggregate_fields(primary_aggregate),
                "per_parameter_comparisons": _per_parameter(
                    names=names, parameters=parameters["liger"], named=primary_named
                ),
                "preclip_gradient_norms": {
                    "selective": _json_finite(preclip_norms["liger"]),
                    "dense_reference": _json_finite(preclip_norms["dense_selected"]),
                },
                "raw_adamw_updates_are_gating": False,
                "optimizer_floating_state_dtypes": {
                    "selective": r14_assay._floating_optimizer_state_dtypes(optimizers["liger"]),
                    "dense_reference": r14_assay._floating_optimizer_state_dtypes(optimizers["dense_selected"]),
                },
                "optimizer_step_counters": {
                    "selective": r15_assay._optimizer_step_counters(optimizers["liger"]),
                    "dense_reference": r15_assay._optimizer_step_counters(optimizers["dense_selected"]),
                },
                "gradient_dtypes": {
                    "selective": sorted({str(parameter.grad.dtype) for parameter in liger.parameters()}),
                    "dense_reference": sorted(
                        {str(parameter.grad.dtype) for parameter in dense_selected.parameters()}
                    ),
                },
                "heldout": {
                    "supervised_targets": heldout["supervised_targets"],
                    "global_divisor": heldout["global_divisor"],
                    "logit_comparison": tensor_comparison_metrics(
                        heldout_logits["liger"], heldout_logits["dense_selected"]
                    ),
                    "selective_loss": _json_finite(heldout_losses["liger"]),
                    "reference_loss": _json_finite(heldout_losses["dense_selected"]),
                    "loss_comparison": scalar_comparison_metrics(
                        float(heldout_losses["liger"].detach().item()),
                        float(heldout_losses["dense_selected"].detach().item()),
                    ),
                },
                "full_dense_diagnostic": diagnostic,
            }
        )

    return {
        "trajectory_contract": contract,
        "trajectory_index": trajectory_index,
        "model_class": type(liger).__name__,
        "dense_forward_module": dense_selected.forward.__module__,
        "patched_forward_module": liger.forward.__module__,
        "full_dense_forward_module_diagnostic": dense_full.forward.__module__,
        "model_config": h2["trajectory_model_config"],
        "parameter_names": names,
        "parameter_geometry": geometry,
        "parameter_count": int(sum(parameter.numel() for parameter in dense_full.parameters())),
        "parameter_dtypes": {
            "selective": sorted({str(parameter.dtype) for parameter in liger.parameters()}),
            "dense_reference": sorted({str(parameter.dtype) for parameter in dense_selected.parameters()}),
        },
        "full_dense_parameter_dtypes_diagnostic": sorted(
            {str(parameter.dtype) for parameter in dense_full.parameters()}
        ),
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


def _zero_target_sentinel(hidden_size: int, vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(170017)
    base_hidden = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16)
    base_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32) * 0.02
    liger_hidden = base_hidden.detach().clone().requires_grad_(True)
    reference_hidden = base_hidden.detach().clone().requires_grad_(True)
    liger_weight = base_weight.detach().clone().requires_grad_(True)
    reference_weight = base_weight.detach().clone().requires_grad_(True)
    target = torch.tensor([-100], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        autocast_contract = r14_assay._active_bf16_autocast_contract()
        liger_loss = (
            LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)(
                liger_weight, liger_hidden, target
            )
            / 7
        )
        reference_loss = (reference_hidden.float().sum() + reference_weight.sum()) * 0.0 / 7
    liger_loss.backward()
    reference_loss.backward()
    if not torch.isfinite(liger_loss) or not torch.isfinite(reference_loss):
        raise AssertionError("R17 zero-target sentinel produced a nonfinite loss")
    if float(liger_loss.detach().item()) != 0.0 or float(reference_loss.detach().item()) != 0.0:
        raise AssertionError("R17 zero-target sentinel loss is not exact zero")
    if any(value.grad is None for value in (liger_hidden, reference_hidden, liger_weight, reference_weight)):
        raise AssertionError("R17 zero-target sentinel disconnected a leaf")
    if any(
        torch.count_nonzero(value.grad) for value in (liger_hidden, reference_hidden, liger_weight, reference_weight)
    ):
        raise AssertionError("R17 zero-target sentinel produced a nonzero gradient")
    old_schema = {
        "loss": float(liger_loss.detach().item()),
        "global_divisor": 7,
        "autocast_contract": autocast_contract,
        "hidden_input_dtype": str(liger_hidden.dtype),
        "output_head_parameter_dtype": str(liger_weight.dtype),
        "hidden_gradient_dtype": str(liger_hidden.grad.dtype),
        "output_head_gradient_dtype": str(liger_weight.grad.dtype),
        "hidden_gradient_connected": liger_hidden.grad is not None,
        "weight_gradient_connected": liger_weight.grad is not None,
        "gradient_nonzero_count": int(torch.count_nonzero(liger_hidden.grad))
        + int(torch.count_nonzero(liger_weight.grad)),
    }
    matched = {
        "observed_path": "liger_fused_selected_rows",
        "reference_path": "pytorch_dense_selected_rows_connected_zero",
        "observed_loss": float(liger_loss.detach().item()),
        "reference_loss": float(reference_loss.detach().item()),
        "loss_comparison": scalar_comparison_metrics(
            float(liger_loss.detach().item()), float(reference_loss.detach().item())
        ),
        "hidden_gradient_comparison": tensor_comparison_metrics(liger_hidden.grad, reference_hidden.grad),
        "output_weight_gradient_comparison": tensor_comparison_metrics(liger_weight.grad, reference_weight.grad),
        "autocast_contract": autocast_contract,
        "graph_connected": {
            "observed_hidden": liger_hidden.grad is not None,
            "observed_weight": liger_weight.grad is not None,
            "reference_hidden": reference_hidden.grad is not None,
            "reference_weight": reference_weight.grad is not None,
        },
    }
    return old_schema, matched


def _diagnostic_nonfinite_count(value: Any, *, in_diagnostic: bool = False) -> int:
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            child_active = in_diagnostic or key == "full_dense_diagnostic"
            if child_active and key == "nonfinite_count" and isinstance(child, int):
                total += child
            else:
                total += _diagnostic_nonfinite_count(child, in_diagnostic=child_active)
        return total
    if isinstance(value, list):
        return sum(_diagnostic_nonfinite_count(child, in_diagnostic=in_diagnostic) for child in value)
    if in_diagnostic and value is None:
        return 1
    if in_diagnostic and isinstance(value, float) and not math.isfinite(value):
        return 1
    return 0


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R17 selective-Liger qualification requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h2 = qualification["h2_acceptance"]
    optimizer_config = qualification["training_unit"]
    report_base = {
        "artifact": "qwen35_selective_liger_matched_reference_qualification_r17",
        "schema_version": 4,
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "manifest_derivation": qualification["manifest_derivation"],
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "direct_hidden_size": h2["direct_hidden_size"],
        "direct_vocab_size": h2["direct_vocab_size"],
        "primary_comparison": {
            "observed_path": h2["primary_observed_path"],
            "reference_path": h2["primary_reference_path"],
            "numerical_discrepancy_is_gating": True,
        },
        "mandatory_diagnostic_comparison": {
            "observed_path": h2["mandatory_diagnostic_observed_path"],
            "reference_path": h2["mandatory_diagnostic_reference_path"],
            "numerical_discrepancy_is_gating": False,
            "integrity_and_finiteness_are_mandatory": True,
        },
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
    matched_zero = None
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
        zero, matched_zero = _zero_target_sentinel(h2["direct_hidden_size"], h2["direct_vocab_size"])
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
            "zero_target_matched_reference": matched_zero,
            "confirmatory_trajectories": trajectories,
        }
        decision = collect_h2_numerical_decisions(report, qualification)
        diagnostic_nonfinite = _diagnostic_nonfinite_count(report)
        report["decision"] = decision
        report["mandatory_diagnostic_nonfinite_count"] = diagnostic_nonfinite
        report["status"] = "passed" if decision["status"] == "passed" and diagnostic_nonfinite == 0 else "failed"
        report["successor_gate_authorized"] = report["status"] == "passed"
        report["allowed_conclusion"] = (
            "R17 H2 passed the matched-reference gate; H3 may begin, but scientific training remains unauthorized."
            if report["status"] == "passed"
            else "R17 H2 failed; H3 remains blocked and the campaign must abandon Liger under the frozen policy."
        )
        r14_assay._write_strict_json_atomic(args.report_output, report)
        if report["status"] != "passed":
            raise AssertionError(
                f"R17 H2 failed {len(decision['failed_gating_checks'])} primary gating checks and "
                f"found {diagnostic_nonfinite} diagnostic nonfinite values"
            )
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
                "zero_target_matched_reference": matched_zero,
                "confirmatory_trajectories": trajectories,
                "failure": {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "allowed_conclusion": "R17 H2 did not complete; H3 and all later gates remain blocked.",
            }
            r14_assay._write_strict_json_atomic(args.report_output, failure_report)
        raise
    print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
