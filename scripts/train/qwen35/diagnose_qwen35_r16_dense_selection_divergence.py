#!/usr/bin/env python3
"""Forensic-only localization of R16 BF16 dense-selection divergence."""

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
from scripts.train.qwen35 import diagnose_qwen35_h2_r15_postfailure as r15_diagnostic
from scripts.train.qwen35 import validate_qwen35_selective_loss as r14_assay
from scripts.train.qwen35 import validate_qwen35_selective_loss_r15 as r15_assay
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from open_instruct.qwen35_qualification import scalar_comparison_metrics, sha256_file, tensor_comparison_metrics
from open_instruct.qwen35_qualification_r16 import load_qualification_manifest

EXPECTED_R16_MANIFEST_SHA256 = "827da32eefdf20839fef364b1bed23afb37122e0c19a981e460324c9d5c1b4f8"
EXPECTED_PARENT_DIAGNOSTIC_SHA256 = "b8e38873e64a6d1281b4510bb1213abe2f52b188f1c8e58d370fddf7ea9a99e7"
REPEATS = 5
CASES = (
    {"case_id": "F0", "trajectory_id": "R16-T0", "trajectory_index": 0, "replay_steps": 54, "assay_step": 55},
    {"case_id": "F1", "trajectory_id": "R16-T1", "trajectory_index": 1, "replay_steps": 64, "assay_step": 65},
    {"case_id": "F2", "trajectory_id": "R16-T2", "trajectory_index": 2, "replay_steps": 3, "assay_step": 4},
)
PATHS = ("full_ignore", "full_gather", "selected_gather")
COMPARISONS = (("full_ignore", "full_gather"), ("full_gather", "selected_gather"), ("full_ignore", "selected_gather"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--parent-diagnostic", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _autocast(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "elements": tensor.numel(),
        "dtype": str(tensor.dtype),
        "nonfinite_count": int((~torch.isfinite(tensor)).sum().item()),
        "sha256": _tensor_sha256(tensor),
    }


def _execute_path(
    *,
    path: str,
    base_hidden: torch.Tensor,
    base_weight: torch.Tensor,
    labels: torch.Tensor,
    selected_positions: torch.Tensor,
    selected_targets: torch.Tensor,
    global_divisor: int,
    autocast_enabled: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | float]]:
    hidden = base_hidden.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    with _autocast(autocast_enabled):
        if path == "full_ignore":
            full_logits = F.linear(hidden, weight)
            full_logits.retain_grad()
            selected_logits = full_logits[:, selected_positions, :]
            loss = (
                F.cross_entropy(
                    full_logits[:, :-1, :].float().reshape(-1, full_logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                / global_divisor
            )
        elif path == "full_gather":
            full_logits = F.linear(hidden, weight)
            full_logits.retain_grad()
            selected_logits = full_logits[:, selected_positions, :]
            loss = (
                F.cross_entropy(
                    selected_logits.float().reshape(-1, selected_logits.shape[-1]),
                    selected_targets.reshape(-1),
                    reduction="sum",
                )
                / global_divisor
            )
        elif path == "selected_gather":
            full_logits = None
            selected_hidden = hidden[:, selected_positions, :]
            selected_logits = F.linear(selected_hidden, weight)
            selected_logits.retain_grad()
            loss = (
                F.cross_entropy(
                    selected_logits.float().reshape(-1, selected_logits.shape[-1]),
                    selected_targets.reshape(-1),
                    reduction="sum",
                )
                / global_divisor
            )
        else:
            raise ValueError(f"unknown path {path!r}")
    loss.backward()
    if hidden.grad is None or weight.grad is None:
        raise RuntimeError(f"{path} produced a disconnected leaf")
    if full_logits is not None:
        if full_logits.grad is None:
            raise RuntimeError(f"{path} produced no full-logit gradient")
        selected_logit_gradient = full_logits.grad[:, selected_positions, :]
    else:
        if selected_logits.grad is None:
            raise RuntimeError(f"{path} produced no selected-logit gradient")
        selected_logit_gradient = selected_logits.grad
    tensors = {
        "selected_logits": selected_logits.detach().clone(),
        "full_hidden_gradient": hidden.grad.detach().clone(),
        "selected_hidden_gradient": hidden.grad[:, selected_positions, :].detach().clone(),
        "output_weight_gradient": weight.grad.detach().clone(),
        "selected_logit_gradient": selected_logit_gradient.detach().clone(),
    }
    record = {
        "path": path,
        "loss": float(loss.detach().item()),
        "loss_dtype": str(loss.dtype),
        "leaf_hidden": _tensor_identity(hidden),
        "leaf_weight": _tensor_identity(weight),
        "tensors": {name: _tensor_identity(tensor) for name, tensor in tensors.items()},
    }
    expected_logit_dtype = torch.bfloat16 if autocast_enabled else torch.float32
    if selected_logits.dtype != expected_logit_dtype:
        raise RuntimeError(
            f"{path} projected logits have dtype {selected_logits.dtype}, expected {expected_logit_dtype}"
        )
    if loss.dtype != torch.float32:
        raise RuntimeError(f"{path} loss has dtype {loss.dtype}, expected torch.float32")
    return record, {"loss": float(loss.detach().item()), **tensors}


def _repeat_path(**kwargs) -> tuple[dict[str, Any], dict[str, torch.Tensor | float]]:
    records = []
    first_values = None
    for repeat in range(REPEATS):
        record, values = _execute_path(**kwargs)
        record["repeat"] = repeat
        records.append(record)
        if first_values is None:
            first_values = values
    assert first_values is not None
    fields = ("selected_logits", "full_hidden_gradient", "selected_hidden_gradient", "output_weight_gradient", "selected_logit_gradient")
    repeatability = {
        "loss_bit_exact": len({record["loss"] for record in records}) == 1,
        "tensor_hashes_bit_exact": {
            field: len({record["tensors"][field]["sha256"] for record in records}) == 1 for field in fields
        },
    }
    repeatability["all_recorded_outputs_bit_exact"] = repeatability["loss_bit_exact"] and all(
        repeatability["tensor_hashes_bit_exact"].values()
    )
    return {"repeats": records, "repeatability": repeatability}, first_values


def _compare(left: dict[str, torch.Tensor | float], right: dict[str, torch.Tensor | float]) -> dict[str, Any]:
    return {
        "loss": scalar_comparison_metrics(float(left["loss"]), float(right["loss"])),
        **{
            field: tensor_comparison_metrics(right[field], left[field])
            for field in (
                "selected_logits",
                "full_hidden_gradient",
                "selected_hidden_gradient",
                "output_weight_gradient",
                "selected_logit_gradient",
            )
        },
    }


def _validate_parent_case(parent: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    matching = [
        run
        for run in parent["runs"]
        if run["variant"] == "bf16_dense_selected_vs_dense_full"
        and run["trajectory_contract"]["trajectory_id"] == case["trajectory_id"]
    ]
    if len(matching) != 1:
        raise RuntimeError(f"parent diagnostic does not contain exactly one {case['trajectory_id']} dense control")
    run = matching[0]
    assay = run["steps"][case["assay_step"] - 1]
    prior = run["steps"][: case["assay_step"] - 1]
    if any(step["aggregate_preclip_gradient"]["difference_l2_norm"] != 0 for step in prior):
        raise RuntimeError(f"{case['case_id']} is not the first parent gradient divergence")
    if assay["aggregate_preclip_gradient"]["difference_l2_norm"] == 0:
        raise RuntimeError(f"{case['case_id']} parent assay step has no gradient divergence")
    if case["assay_step"] > 1 and prior[-1]["aggregate_cumulative_displacement"]["difference_l2_norm"] != 0:
        raise RuntimeError(f"{case['case_id']} branches did not have a common pre-assay parameter state")
    return {
        "trajectory_contract": run["trajectory_contract"],
        "parent_batch_accounting": assay["batch_accounting"],
        "parent_first_gradient_difference_l2_norm": assay["aggregate_preclip_gradient"]["difference_l2_norm"],
        "parent_pre_assay_complete_state_bit_exact": True,
    }


def _replay_reference(
    *, case: dict[str, Any], contract: dict[str, Any], h2: dict[str, Any], optimizer_config: dict[str, Any]
) -> tuple[Qwen3_5ForCausalLM, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(contract["model_seed"])
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**h2["trajectory_model_config"])).cuda().train()
    optimizer = r15_assay._optimizer(model.parameters(), optimizer_config)
    replay_ledger = []
    for step_index in range(case["replay_steps"]):
        batch = r15_assay._trajectory_batch(
            seed=contract["batch_seed_base"] + step_index,
            step_index=step_index,
            trajectory_index=case["trajectory_index"],
            h2=h2,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, autocast_contract = r15_diagnostic._ordinary_loss(model, batch, autocast_enabled=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
        optimizer.step()
        replay_ledger.append(
            {
                "step": step_index + 1,
                "batch_accounting": batch["accounting"],
                "loss": float(loss.detach().item()),
                "preclip_gradient_norm": float(gradient_norm.detach().item()),
                "autocast_contract": autocast_contract,
            }
        )
    assay_index = case["assay_step"] - 1
    assay_batch = r15_assay._trajectory_batch(
        seed=contract["batch_seed_base"] + assay_index,
        step_index=assay_index,
        trajectory_index=case["trajectory_index"],
        h2=h2,
    )
    return model, assay_batch, {"steps": replay_ledger, "final_optimizer_step": case["replay_steps"]}


def _run_case(
    *, case: dict[str, Any], parent: dict[str, Any], h2: dict[str, Any], optimizer_config: dict[str, Any]
) -> dict[str, Any]:
    parent_evidence = _validate_parent_case(parent, case)
    contract = parent_evidence["trajectory_contract"]
    model, batch, replay = _replay_reference(
        case=case, contract=contract, h2=h2, optimizer_config=optimizer_config
    )
    if batch["accounting"] != parent_evidence["parent_batch_accounting"]:
        raise RuntimeError(f"{case['case_id']} replayed batch accounting differs from parent evidence")
    with torch.no_grad(), _autocast(True):
        hidden = model.model(input_ids=batch["input_ids"], use_cache=False, return_dict=True).last_hidden_state
    base_weight = model.lm_head.weight.detach()
    if not hidden.is_floating_point() or int((~torch.isfinite(hidden)).sum().item()):
        raise RuntimeError(f"{case['case_id']} captured an invalid model hidden state")
    if base_weight.dtype != torch.float32 or int((~torch.isfinite(base_weight)).sum().item()):
        raise RuntimeError(f"{case['case_id']} output-weight storage is not finite FP32")
    arithmetic_results = []
    for arithmetic, autocast_enabled in (("production_bf16_autocast", True), ("fp32_control", False)):
        arithmetic_hidden = hidden if autocast_enabled else hidden.float()
        arithmetic_weight = base_weight if autocast_enabled else base_weight.float()
        paths = {}
        values = {}
        for path in PATHS:
            paths[path], values[path] = _repeat_path(
                path=path,
                base_hidden=arithmetic_hidden,
                base_weight=arithmetic_weight,
                labels=batch["labels"],
                selected_positions=batch["selected_positions"],
                selected_targets=batch["selected_targets"],
                global_divisor=batch["accounting"]["global_divisor"],
                autocast_enabled=autocast_enabled,
            )
        arithmetic_results.append(
            {
                "arithmetic": arithmetic,
                "autocast_enabled": autocast_enabled,
                "arithmetic_contract": {
                    "captured_hidden_dtype": str(arithmetic_hidden.dtype),
                    "weight_storage_dtype": str(arithmetic_weight.dtype),
                    "expected_projected_logit_dtype": (
                        "torch.bfloat16" if autocast_enabled else "torch.float32"
                    ),
                    "cross_entropy_dtype": "torch.float32",
                },
                "base_hidden": _tensor_identity(arithmetic_hidden),
                "base_weight": _tensor_identity(arithmetic_weight),
                "paths": paths,
                "comparisons": [
                    {
                        "left": left,
                        "right": right,
                        "metrics": _compare(values[left], values[right]),
                    }
                    for left, right in COMPARISONS
                ],
            }
        )
    return {
        **case,
        "status": "forensic_case_complete_no_gate",
        "parent_evidence": parent_evidence,
        "replay": replay,
        "replayed_assay_batch_accounting": batch["accounting"],
        "selected_positions": batch["selected_positions"].tolist(),
        "selected_targets_sha256": _tensor_sha256(batch["selected_targets"]),
        "arithmetic_results": arithmetic_results,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("dense-selection forensic diagnostic requires CUDA")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if qualification_sha256 != EXPECTED_R16_MANIFEST_SHA256:
        raise RuntimeError("unexpected R16 qualification manifest")
    if sha256_file(args.parent_diagnostic) != EXPECTED_PARENT_DIAGNOSTIC_SHA256:
        raise RuntimeError("unexpected parent diagnostic bytes")
    parent = json.loads(args.parent_diagnostic.read_text())
    if parent["status"] != "diagnostic_complete_no_gate" or parent["successor_gate_authorized"] is not False:
        raise RuntimeError("parent diagnostic status/authority drift")
    h2 = qualification["h2_acceptance"]
    results = []
    for case in CASES:
        try:
            results.append(
                _run_case(
                    case=case,
                    parent=parent,
                    h2=h2,
                    optimizer_config=qualification["training_unit"],
                )
            )
        except Exception as error:
            results.append(
                {
                    **case,
                    "status": "forensic_case_failed_no_gate",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    complete = sum(result["status"] == "forensic_case_complete_no_gate" for result in results)
    status = "forensic_complete_no_gate" if complete == len(results) else ("forensic_partial_no_gate" if complete else "forensic_failed_no_gate")
    repo_root = Path(__file__).resolve().parents[3]
    source_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "artifact": "qwen35_r16_dense_selection_divergence_forensic",
        "schema_version": 1,
        "status": status,
        "successor_gate_authorized": False,
        "scientific_training_authorized": False,
        "allowed_conclusion": "Forensic localization only; R16 remains failed and H3 remains blocked.",
        "qualification_manifest_sha256": qualification_sha256,
        "parent_diagnostic_sha256": EXPECTED_PARENT_DIAGNOSTIC_SHA256,
        "source_commit": source_commit,
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__)),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "repeat_count": REPEATS,
        "path_definitions": {
            "full_ignore": "full projection then shifted ignore-index cross entropy",
            "full_gather": "full projection then explicit supervised-position gather and cross entropy",
            "selected_gather": "supervised hidden-row gather then selected projection and cross entropy",
        },
        "cases": results,
    }
    r14_assay._write_strict_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
