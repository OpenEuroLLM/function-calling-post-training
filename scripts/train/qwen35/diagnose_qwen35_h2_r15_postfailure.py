#!/usr/bin/env python3
"""Diagnostic-only causal localization of the immutable R15 H2 failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
from scripts.train.qwen35 import validate_qwen35_selective_loss as r14_assay
from scripts.train.qwen35 import validate_qwen35_selective_loss_r15 as r15_assay
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification import (
    load_qualification_manifest,
    scalar_comparison_metrics,
    sha256_file,
    tensor_comparison_metrics,
)

EXPECTED_R15_MANIFEST_SHA256 = "bff52a9223d07cdf047bfe25dbcf7330d36176d753d38f66a330a5ff1780fc4f"
EXPECTED_R15_REPORT_SHA256 = "b452847bc4c4993418330c41b95be3996a0b97efee0542866b095b658a11e2a4"
SMALL_GRADIENT_BINS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, float("inf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--r15-failed-report", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _seed_identity(label: str) -> dict[str, str | int]:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return {"label": label, "sha256": digest, "seed": int(digest[:8], 16)}


def _diagnostic_contracts() -> list[dict[str, Any]]:
    contracts = []
    for index in range(3):
        base = f"qwen35-hardware-qualification-r15-postfailure-diagnostic-{index}"
        model = _seed_identity(base)
        batches = _seed_identity(f"{base}-batches")
        heldout = _seed_identity(f"{base}-heldout")
        contracts.append(
            {
                "trajectory_id": f"R15-PF-D{index}",
                "model_seed_label": model["label"],
                "model_seed_sha256": model["sha256"],
                "model_seed": model["seed"],
                "batch_seed_label": batches["label"],
                "batch_seed_sha256": batches["sha256"],
                "batch_seed_base": batches["seed"],
                "heldout_seed_label": heldout["label"],
                "heldout_seed_sha256": heldout["sha256"],
                "heldout_seed": heldout["seed"],
            }
        )
    return contracts


def _autocast(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _autocast_contract(enabled: bool) -> dict[str, Any]:
    if enabled:
        return r14_assay._active_bf16_autocast_contract()
    return {"device_type": "cuda", "enabled": False, "dtype": None}


def _flatten(values: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([values[name].detach().reshape(-1) for name in names])


def _state_tensor(optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter, key: str) -> torch.Tensor:
    value = optimizer.state[parameter].get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"optimizer state {key!r} is missing")
    return value.detach().clone()


def _step_counter(optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter) -> int:
    value = optimizer.state[parameter].get("step")
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if value is None:
        raise RuntimeError("optimizer step counter is missing")
    return int(value)


def _sign_and_bin_diagnostic(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    observed64 = observed.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    absolute = reference64.abs()
    rows = []
    for lower, upper in zip(SMALL_GRADIENT_BINS[:-1], SMALL_GRADIENT_BINS[1:], strict=True):
        mask = absolute.ge(lower) & absolute.lt(upper)
        difference = observed64[mask] - reference64[mask]
        rows.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": None if upper == float("inf") else upper,
                "elements": int(mask.sum()),
                "reference_gradient_energy": float(torch.square(reference64[mask]).sum()),
                "difference_energy": float(torch.square(difference).sum()),
            }
        )
    return {
        "opposite_nonzero_signs": int(((observed64 * reference64) < 0).sum()),
        "observed_zero_reference_nonzero": int(((observed64 == 0) & (reference64 != 0)).sum()),
        "reference_zero_observed_nonzero": int(((reference64 == 0) & (observed64 != 0)).sum()),
        "reference_absolute_gradient_bins": rows,
    }


def _ordinary_loss(
    model: Qwen3_5ForCausalLM, batch: dict[str, Any], *, autocast_enabled: bool
) -> tuple[torch.Tensor, dict[str, Any]]:
    with _autocast(autocast_enabled):
        contract = _autocast_contract(autocast_enabled)
        loss = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            num_items_in_batch=batch["accounting"]["global_divisor"],
            use_cache=False,
        ).loss
    return loss, contract


def _comparison_loss(
    model: Qwen3_5ForCausalLM, batch: dict[str, Any], *, variant: str, autocast_enabled: bool
) -> tuple[torch.Tensor, dict[str, Any]]:
    with _autocast(autocast_enabled):
        contract = _autocast_contract(autocast_enabled)
        if variant in {"bf16_liger_vs_dense_full", "fp32_liger_vs_dense_full"}:
            loss = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                logits_to_keep=batch["selected_positions"],
                shift_labels=batch["selected_targets"],
                num_items_in_batch=batch["accounting"]["global_divisor"],
                use_cache=False,
            ).loss
        elif variant == "bf16_dense_selected_vs_dense_full":
            hidden = model.model(input_ids=batch["input_ids"], use_cache=False, return_dict=True).last_hidden_state
            selected_hidden = hidden[:, batch["selected_positions"], :]
            selected_logits = model.lm_head(selected_hidden)
            loss = (
                F.cross_entropy(
                    selected_logits.float().reshape(-1, selected_logits.shape[-1]),
                    batch["selected_targets"].reshape(-1),
                    reduction="sum",
                )
                / batch["accounting"]["global_divisor"]
            )
        else:
            raise ValueError(f"unknown diagnostic variant: {variant}")
    return loss, contract


def _run_pair(
    *,
    variant: str,
    trajectory_contract: dict[str, Any],
    trajectory_index: int,
    h2: dict[str, Any],
    optimizer_config: dict[str, Any],
) -> dict[str, Any]:
    autocast_enabled = variant != "fp32_liger_vs_dense_full"
    torch.manual_seed(trajectory_contract["model_seed"])
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    observed = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    observed.load_state_dict(reference.state_dict(), strict=True)
    if "liger_vs" in variant:
        apply_liger_kernel_to_qwen3_5(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            model=observed,
        )
    names = [name for name, _ in reference.named_parameters()]
    geometry = [
        {"name": name, "shape": list(parameter.shape), "elements": int(parameter.numel())}
        for name, parameter in reference.named_parameters()
    ]
    if geometry != h2["trajectory_parameter_geometry"]:
        raise RuntimeError("diagnostic model geometry drift")
    if names != [name for name, _ in observed.named_parameters()]:
        raise RuntimeError("diagnostic branch parameter order drift")
    reference_initial = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    observed_initial = {name: parameter.detach().clone() for name, parameter in observed.named_parameters()}
    if any(not torch.equal(reference_initial[name], observed_initial[name]) for name in names):
        raise RuntimeError("diagnostic branches do not have a bit-exact common initialization")
    reference_optimizer = r15_assay._optimizer(reference.parameters(), optimizer_config)
    observed_optimizer = r15_assay._optimizer(observed.parameters(), optimizer_config)
    heldout = r15_assay._heldout_batch(seed=trajectory_contract["heldout_seed"], h2=h2)
    steps = []

    for step_index in range(h2["trajectory_steps"]):
        batch = r15_assay._trajectory_batch(
            seed=trajectory_contract["batch_seed_base"] + step_index,
            step_index=step_index,
            trajectory_index=trajectory_index,
            h2=h2,
        )
        reference_optimizer.zero_grad(set_to_none=True)
        observed_optimizer.zero_grad(set_to_none=True)
        reference_loss, reference_autocast = _ordinary_loss(reference, batch, autocast_enabled=autocast_enabled)
        observed_loss, observed_autocast = _comparison_loss(
            observed, batch, variant=variant, autocast_enabled=autocast_enabled
        )
        reference_loss.backward()
        observed_loss.backward()

        reference_parameters = dict(reference.named_parameters())
        observed_parameters = dict(observed.named_parameters())
        if any(reference_parameters[name].grad is None or observed_parameters[name].grad is None for name in names):
            raise RuntimeError("diagnostic encountered a disconnected parameter")
        reference_preclip = {name: reference_parameters[name].grad.detach().clone() for name in names}
        observed_preclip = {name: observed_parameters[name].grad.detach().clone() for name in names}
        reference_before = {name: reference_parameters[name].detach().clone() for name in names}
        observed_before = {name: observed_parameters[name].detach().clone() for name in names}
        reference_preclip_vector = _flatten(reference_preclip, names)
        observed_preclip_vector = _flatten(observed_preclip, names)
        reference_preclip_norm = torch.nn.utils.clip_grad_norm_(
            reference.parameters(), optimizer_config["max_gradient_norm"]
        )
        observed_preclip_norm = torch.nn.utils.clip_grad_norm_(
            observed.parameters(), optimizer_config["max_gradient_norm"]
        )
        reference_clipped = {name: reference_parameters[name].grad.detach().clone() for name in names}
        observed_clipped = {name: observed_parameters[name].grad.detach().clone() for name in names}
        reference_clipped_vector = _flatten(reference_clipped, names)
        observed_clipped_vector = _flatten(observed_clipped, names)
        reference_optimizer.step()
        observed_optimizer.step()

        reference_after = {name: reference_parameters[name].detach().clone() for name in names}
        observed_after = {name: observed_parameters[name].detach().clone() for name in names}
        reference_updates = {name: reference_after[name] - reference_before[name] for name in names}
        observed_updates = {name: observed_after[name] - observed_before[name] for name in names}
        reference_cumulative = {name: reference_after[name] - reference_initial[name] for name in names}
        observed_cumulative = {name: observed_after[name] - observed_initial[name] for name in names}
        reference_exp_avg = {
            name: _state_tensor(reference_optimizer, reference_parameters[name], "exp_avg") for name in names
        }
        observed_exp_avg = {
            name: _state_tensor(observed_optimizer, observed_parameters[name], "exp_avg") for name in names
        }
        reference_exp_avg_sq = {
            name: _state_tensor(reference_optimizer, reference_parameters[name], "exp_avg_sq") for name in names
        }
        observed_exp_avg_sq = {
            name: _state_tensor(observed_optimizer, observed_parameters[name], "exp_avg_sq") for name in names
        }

        with torch.no_grad(), _autocast(autocast_enabled):
            heldout_autocast = _autocast_contract(autocast_enabled)
            reference_logits = reference(input_ids=heldout["input_ids"], use_cache=False).logits
            observed_logits = observed(input_ids=heldout["input_ids"], use_cache=False).logits
        reference_heldout_loss = (
            F.cross_entropy(
                reference_logits[:, :-1].float().reshape(-1, reference_logits.shape[-1]),
                heldout["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / heldout["global_divisor"]
        )
        observed_heldout_loss = (
            F.cross_entropy(
                observed_logits[:, :-1].float().reshape(-1, observed_logits.shape[-1]),
                heldout["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / heldout["global_divisor"]
        )

        aggregate_reference_gradient_energy = float(torch.square(reference_preclip_vector.double()).sum())
        aggregate_difference = observed_preclip_vector.double() - reference_preclip_vector.double()
        aggregate_difference_energy = float(torch.square(aggregate_difference).sum())
        aggregate_reference_norm = float(torch.linalg.vector_norm(reference_preclip_vector.double()))
        per_parameter = {}
        for name in names:
            reference_gradient = reference_preclip[name].double()
            gradient_difference = observed_preclip[name].double() - reference_gradient
            parameter_row = {
                "elements": int(reference_parameters[name].numel()),
                "preclip_gradient": tensor_comparison_metrics(observed_preclip[name], reference_preclip[name]),
                "clipped_gradient": tensor_comparison_metrics(observed_clipped[name], reference_clipped[name]),
                "raw_update": tensor_comparison_metrics(observed_updates[name], reference_updates[name]),
                "cumulative_displacement": tensor_comparison_metrics(
                    observed_cumulative[name], reference_cumulative[name]
                ),
                "post_step_parameter": tensor_comparison_metrics(observed_after[name], reference_after[name]),
                "adam_exp_avg": tensor_comparison_metrics(observed_exp_avg[name], reference_exp_avg[name]),
                "adam_exp_avg_sq": tensor_comparison_metrics(observed_exp_avg_sq[name], reference_exp_avg_sq[name]),
                "reference_gradient_energy_fraction": (
                    float(torch.square(reference_gradient).sum()) / aggregate_reference_gradient_energy
                    if aggregate_reference_gradient_energy
                    else 0.0
                ),
                "difference_energy_fraction": (
                    float(torch.square(gradient_difference).sum()) / aggregate_difference_energy
                    if aggregate_difference_energy
                    else 0.0
                ),
                "difference_norm_over_aggregate_reference_gradient_norm": (
                    float(torch.linalg.vector_norm(gradient_difference)) / aggregate_reference_norm
                    if aggregate_reference_norm
                    else 0.0
                ),
                "optimizer_step": {
                    "observed": _step_counter(observed_optimizer, observed_parameters[name]),
                    "reference": _step_counter(reference_optimizer, reference_parameters[name]),
                },
            }
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                parameter_row["sign_and_magnitude"] = _sign_and_bin_diagnostic(
                    observed_preclip[name], reference_preclip[name]
                )
            per_parameter[name] = parameter_row

        steps.append(
            {
                "step": step_index + 1,
                "batch_accounting": batch["accounting"],
                "autocast_contract": {
                    "observed_training": observed_autocast,
                    "reference_training": reference_autocast,
                    "heldout": heldout_autocast,
                },
                "loss": scalar_comparison_metrics(
                    float(observed_loss.detach().item()), float(reference_loss.detach().item())
                ),
                "aggregate_preclip_gradient": tensor_comparison_metrics(
                    observed_preclip_vector, reference_preclip_vector
                ),
                "aggregate_clipped_gradient": tensor_comparison_metrics(
                    observed_clipped_vector, reference_clipped_vector
                ),
                "preclip_gradient_norms": {
                    "observed": float(observed_preclip_norm.detach().item()),
                    "reference": float(reference_preclip_norm.detach().item()),
                },
                "aggregate_raw_update": tensor_comparison_metrics(
                    _flatten(observed_updates, names), _flatten(reference_updates, names)
                ),
                "aggregate_cumulative_displacement": tensor_comparison_metrics(
                    _flatten(observed_cumulative, names), _flatten(reference_cumulative, names)
                ),
                "aggregate_post_step_parameter": tensor_comparison_metrics(
                    _flatten(observed_after, names), _flatten(reference_after, names)
                ),
                "aggregate_adam_exp_avg": tensor_comparison_metrics(
                    _flatten(observed_exp_avg, names), _flatten(reference_exp_avg, names)
                ),
                "aggregate_adam_exp_avg_sq": tensor_comparison_metrics(
                    _flatten(observed_exp_avg_sq, names), _flatten(reference_exp_avg_sq, names)
                ),
                "per_parameter": per_parameter,
                "heldout": {
                    "logits": tensor_comparison_metrics(observed_logits, reference_logits),
                    "loss": scalar_comparison_metrics(
                        float(observed_heldout_loss.detach().item()), float(reference_heldout_loss.detach().item())
                    ),
                    "supervised_targets": heldout["supervised_targets"],
                    "global_divisor": heldout["global_divisor"],
                },
                "dtypes": {
                    "observed_parameters": sorted({str(parameter.dtype) for parameter in observed.parameters()}),
                    "reference_parameters": sorted({str(parameter.dtype) for parameter in reference.parameters()}),
                    "observed_gradients": sorted({str(parameter.grad.dtype) for parameter in observed.parameters()}),
                    "reference_gradients": sorted({str(parameter.grad.dtype) for parameter in reference.parameters()}),
                    "observed_optimizer": r14_assay._floating_optimizer_state_dtypes(observed_optimizer),
                    "reference_optimizer": r14_assay._floating_optimizer_state_dtypes(reference_optimizer),
                },
            }
        )

    return {
        "status": "diagnostic_complete_no_gate",
        "variant": variant,
        "trajectory_contract": trajectory_contract,
        "trajectory_index": trajectory_index,
        "observed_forward_module": observed.forward.__module__,
        "reference_forward_module": reference.forward.__module__,
        "parameter_geometry": geometry,
        "steps": steps,
    }


def _variant_summary(run: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any]:
    if run.get("status") != "diagnostic_complete_no_gate":
        return {"status": run.get("status"), "variant": run.get("variant"), "trajectory_id": run.get("trajectory_id")}
    steps = run["steps"]
    named_gradient_exceedances = []
    for step in steps:
        for name, row in step["per_parameter"].items():
            for phase in ("preclip_gradient", "clipped_gradient"):
                metric = row[phase]
                if (
                    metric["maximum_absolute_error"] > acceptance["gradient_maximum_absolute_error"]
                    or metric["relative_l2_error"] > acceptance["gradient_relative_l2_error"]
                    or metric["cosine_similarity"] < acceptance["gradient_minimum_cosine_similarity"]
                ):
                    named_gradient_exceedances.append(
                        {"step": step["step"], "parameter": name, "phase": phase, "metrics": metric}
                    )
    return {
        "status": run["status"],
        "variant": run["variant"],
        "trajectory_id": run["trajectory_contract"]["trajectory_id"],
        "named_gradient_exceedances_under_r15_thresholds": named_gradient_exceedances,
        "maximum_aggregate_gradient_relative_l2": max(
            step["aggregate_preclip_gradient"]["relative_l2_error"] for step in steps
        ),
        "maximum_aggregate_raw_update_relative_l2": max(
            step["aggregate_raw_update"]["relative_l2_error"] for step in steps
        ),
        "maximum_post_step_parameter_relative_l2": max(
            step["aggregate_post_step_parameter"]["relative_l2_error"] for step in steps
        ),
        "maximum_adam_exp_avg_relative_l2": max(step["aggregate_adam_exp_avg"]["relative_l2_error"] for step in steps),
        "maximum_adam_exp_avg_sq_relative_l2": max(
            step["aggregate_adam_exp_avg_sq"]["relative_l2_error"] for step in steps
        ),
        "maximum_heldout_logit_relative_l2": max(step["heldout"]["logits"]["relative_l2_error"] for step in steps),
        "maximum_heldout_logit_absolute_error": max(
            step["heldout"]["logits"]["maximum_absolute_error"] for step in steps
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R15 post-failure diagnostic requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if qualification_sha256 != EXPECTED_R15_MANIFEST_SHA256:
        raise RuntimeError("diagnostic received an unexpected R15 manifest")
    if sha256_file(args.r15_failed_report) != EXPECTED_R15_REPORT_SHA256:
        raise RuntimeError("diagnostic received an unexpected R15 failed report")
    r15_report = json.loads(args.r15_failed_report.read_text())
    if r15_report.get("status") != "failed" or r15_report.get("successor_gate_authorized") is not False:
        raise RuntimeError("diagnostic parent report is not the immutable failed R15 result")
    h2 = qualification["h2_acceptance"]
    r15_contracts = h2["confirmatory_trajectories"]
    diagnostic_contracts = _diagnostic_contracts()
    source = r14_assay._verify_liger_source_pin(qualification["runtime_pins"]["liger_source_files_sha256"])
    variants = [
        {
            "variant": "bf16_liger_vs_dense_full",
            "contracts": r15_contracts + diagnostic_contracts,
            "contract_origin": ["r15"] * 3 + ["outcome_unseen_diagnostic"] * 3,
        },
        {"variant": "bf16_dense_selected_vs_dense_full", "contracts": r15_contracts, "contract_origin": ["r15"] * 3},
        {"variant": "fp32_liger_vs_dense_full", "contracts": r15_contracts, "contract_origin": ["r15"] * 3},
    ]
    runs = []
    for variant in variants:
        for index, (contract, origin) in enumerate(zip(variant["contracts"], variant["contract_origin"], strict=True)):
            try:
                run = _run_pair(
                    variant=variant["variant"],
                    trajectory_contract=contract,
                    trajectory_index=index % 3,
                    h2=h2,
                    optimizer_config=qualification["training_unit"],
                )
                run["contract_origin"] = origin
            except Exception as error:
                run = {
                    "status": "diagnostic_failed_no_gate",
                    "variant": variant["variant"],
                    "trajectory_id": contract["trajectory_id"],
                    "trajectory_contract": contract,
                    "contract_origin": origin,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            runs.append(run)
    complete = sum(run["status"] == "diagnostic_complete_no_gate" for run in runs)
    overall = (
        "diagnostic_complete_no_gate"
        if complete == len(runs)
        else ("diagnostic_partial_no_gate" if complete else "diagnostic_failed_no_gate")
    )
    repo_root = Path(__file__).resolve().parents[3]
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "artifact": "qwen35_r15_h2_postfailure_diagnostic",
        "schema_version": 1,
        "status": overall,
        "successor_gate_authorized": False,
        "scientific_training_authorized": False,
        "allowed_conclusion": "Diagnostic localization only; R15 remains failed and H3 remains blocked.",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "r15_failed_report_sha256": EXPECTED_R15_REPORT_SHA256,
        "diagnostic_source_path": str(Path(__file__).resolve()),
        "diagnostic_source_sha256": sha256_file(Path(__file__)),
        "source_commit": source_commit,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "liger_kernel": source,
        "small_gradient_bins": [None if value == float("inf") else value for value in SMALL_GRADIENT_BINS],
        "r15_trajectory_contracts": r15_contracts,
        "outcome_unseen_diagnostic_contracts": diagnostic_contracts,
        "planned_runs": 12,
        "complete_runs": complete,
        "runs": runs,
        "summaries": [_variant_summary(run, qualification["numerical_acceptance"]) for run in runs],
    }
    r14_assay._write_strict_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": overall}, sort_keys=True))


if __name__ == "__main__":
    main()
