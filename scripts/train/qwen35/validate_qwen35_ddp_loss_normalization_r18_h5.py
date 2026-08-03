#!/usr/bin/env python3
"""Four-rank non-Liger R18 H5 global-target normalization preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

from open_instruct.qwen35_chunked_loss import (
    IMPLEMENTATION_ID,
    checkpointed_chunked_selective_linear_cross_entropy,
    ordinary_chunked_selective_linear_cross_entropy,
)
from open_instruct.qwen35_qualification import (
    scalar_comparison_metrics,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h5 import (
    H5_SELECTED_CHUNK_SIZE,
    H5_WORLD_SIZE,
    load_h5_contract,
    load_h5_harness_amendment,
    load_h5_harness_amendment_r2,
)
from open_instruct.qwen35_training import write_json_atomic

TARGET_COUNTS = (0, 127, 513, 1025)


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
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1024)
    return parser.parse_args()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def rank_case(rank: int, hidden_size: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    count = TARGET_COUNTS[rank]
    generator = torch.Generator(device=device).manual_seed(10_000 + rank)
    if count == 0:
        rows = torch.randn(1, hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
        targets = torch.full((1,), -100, device=device, dtype=torch.long)
    else:
        rows = torch.randn(count, hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
        targets = torch.randint(0, vocab_size, (count,), generator=generator, device=device)
    return rows, targets


class TinyCheckpointedHead(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(1234)
        self.weight = torch.nn.Parameter(
            torch.randn(vocab_size, hidden_size, generator=generator, device=device, dtype=torch.float32) * 0.02
        )
        self.last_audit: dict | None = None

    def forward(self, rows: torch.Tensor, targets: torch.Tensor, global_divisor: torch.Tensor, world_size: int):
        loss, audit = checkpointed_chunked_selective_linear_cross_entropy(
            rows,
            self.weight,
            targets,
            global_target_count=global_divisor,
            chunk_size=H5_SELECTED_CHUNK_SIZE,
            return_audit=True,
        )
        self.last_audit = audit.to_dict()
        return loss * world_size


def optimizer(parameter: torch.nn.Parameter) -> torch.optim.AdamW:
    return torch.optim.AdamW([parameter], lr=2e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=False)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R18 H5 DDP normalization preflight requires CUDA")
    if args.hidden_size <= 0 or args.vocab_size <= 1:
        raise ValueError("invalid tiny-head dimensions")
    contract, contract_sha256 = load_h5_contract(
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
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    if qualification_sha256 != contract["parent"]["r18_machine_manifest_sha256"]:
        raise ValueError("R18 H5 DDP preflight qualification-manifest drift")
    if any(name == "liger_kernel" or name.startswith("liger_kernel.") for name in sys.modules):
        raise RuntimeError("R18 H5 DDP preflight imported forbidden Liger before initialization")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", device_id=device)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != H5_WORLD_SIZE or len(TARGET_COUNTS) != H5_WORLD_SIZE:
        raise RuntimeError(f"R18 H5 DDP preflight requires exactly {H5_WORLD_SIZE} ranks")
    module = TinyCheckpointedHead(args.hidden_size, args.vocab_size, device)
    initial_weight = module.weight.detach().clone()
    distributed_optimizer = optimizer(module.weight)
    ddp = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])

    rows, targets = rank_case(rank, args.hidden_size, args.vocab_size, device)
    local_count = TARGET_COUNTS[rank]
    global_divisor = torch.tensor(local_count, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(global_divisor, op=torch.distributed.ReduceOp.SUM)
    if int(global_divisor.item()) != sum(TARGET_COUNTS):
        raise AssertionError("R18 H5 DDP preflight global target divisor drift")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        local_scaled_loss = ddp(rows, targets, global_divisor, world_size)
    local_global_contribution = local_scaled_loss.detach().float() / world_size
    torch.distributed.all_reduce(local_global_contribution, op=torch.distributed.ReduceOp.SUM)
    local_scaled_loss.backward()
    if module.weight.grad is None:
        raise AssertionError("R18 H5 DDP preflight produced no DDP gradient")
    observed_preclip_gradient = module.weight.grad.detach().clone()

    reference_weight = initial_weight.detach().clone().requires_grad_(True)
    reference_loss = torch.zeros((), dtype=torch.float32, device=device)
    reference_audits = []
    for case_rank in range(world_size):
        case_rows, case_targets = rank_case(case_rank, args.hidden_size, args.vocab_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            case_loss, case_audit = ordinary_chunked_selective_linear_cross_entropy(
                case_rows,
                reference_weight,
                case_targets,
                global_target_count=global_divisor,
                chunk_size=H5_SELECTED_CHUNK_SIZE,
                return_audit=True,
            )
        reference_loss = reference_loss + case_loss
        reference_audits.append(case_audit.to_dict())
    reference_loss.backward()
    if reference_weight.grad is None:
        raise AssertionError("R18 H5 DDP preflight produced no central-reference gradient")
    reference_preclip_gradient = reference_weight.grad.detach().clone()

    acceptance = qualification["numerical_acceptance"]
    loss_metrics = scalar_comparison_metrics(float(local_global_contribution), float(reference_loss.detach()))
    validate_comparison_metrics(loss_metrics, acceptance, kind="loss", context="R18 H5 globally reduced loss")
    gradient_metrics = tensor_comparison_metrics(observed_preclip_gradient, reference_preclip_gradient)
    validate_comparison_metrics(
        gradient_metrics, acceptance, kind="gradient", context="R18 H5 DDP globally normalized gradient"
    )

    observed_norm = torch.nn.utils.clip_grad_norm_([module.weight], max_norm=1.0)
    reference_norm = torch.nn.utils.clip_grad_norm_([reference_weight], max_norm=1.0)
    norm_metrics = scalar_comparison_metrics(float(observed_norm), float(reference_norm))
    validate_comparison_metrics(norm_metrics, acceptance, kind="loss", context="R18 H5 preclip gradient norm")
    observed_coefficient = min(1.0, 1.0 / (float(observed_norm) + 1e-6))
    reference_coefficient = min(1.0, 1.0 / (float(reference_norm) + 1e-6))
    if not math.isclose(observed_coefficient, reference_coefficient, rel_tol=1e-3, abs_tol=1e-6):
        raise AssertionError("R18 H5 clipping coefficient drift")

    reference_optimizer = optimizer(reference_weight)
    distributed_optimizer.step()
    reference_optimizer.step()
    update_metrics = tensor_comparison_metrics(module.weight.detach(), reference_weight.detach())
    validate_comparison_metrics(update_metrics, acceptance, kind="update", context="R18 H5 AdamW parameter update")
    observed_state = distributed_optimizer.state[module.weight]
    reference_state = reference_optimizer.state[reference_weight]
    state_metrics = {}
    for state_name in ("exp_avg", "exp_avg_sq"):
        if observed_state[state_name].dtype != torch.float32 or reference_state[state_name].dtype != torch.float32:
            raise AssertionError(f"R18 H5 {state_name} is not FP32")
        metrics = tensor_comparison_metrics(observed_state[state_name], reference_state[state_name])
        validate_comparison_metrics(metrics, acceptance, kind="update", context=f"R18 H5 AdamW {state_name}")
        state_metrics[state_name] = metrics
    if int(observed_state["step"].item()) != 1 or int(reference_state["step"].item()) != 1:
        raise AssertionError("R18 H5 AdamW step counter drift")
    if module.weight.dtype != torch.float32 or module.weight.grad.dtype != torch.float32:
        raise AssertionError("R18 H5 tiny DDP parameter or gradient is not FP32")

    gradient_hash = tensor_sha256(observed_preclip_gradient)
    update_hash = tensor_sha256(module.weight.detach())
    gathered_hashes: list[dict | None] = [None] * world_size
    torch.distributed.all_gather_object(
        gathered_hashes, {"gradient_sha256": gradient_hash, "parameter_sha256": update_hash, "rank": rank}
    )
    if len({row["gradient_sha256"] for row in gathered_hashes if row}) != 1:
        raise AssertionError("R18 H5 reduced gradients are not identical across ranks")
    if len({row["parameter_sha256"] for row in gathered_hashes if row}) != 1:
        raise AssertionError("R18 H5 updated parameters are not identical across ranks")
    loaded_liger_modules = sorted(
        name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
    )
    if loaded_liger_modules:
        raise RuntimeError(f"R18 H5 DDP preflight imported forbidden Liger modules: {loaded_liger_modules}")

    finite = all(
        math.isfinite(value)
        for value in (
            float(local_global_contribution),
            float(reference_loss.detach()),
            float(observed_norm),
            float(reference_norm),
        )
    ) and bool(torch.isfinite(module.weight).all() and torch.isfinite(observed_preclip_gradient).all())
    if not finite:
        raise AssertionError("R18 H5 DDP preflight produced non-finite state")
    if rank == 0:
        if args.report_output.exists():
            raise FileExistsError(args.report_output)
        report = {
            "adamw_moment_comparisons": state_metrics,
            "adamw_step": 1,
            "artifact": "qwen35_r18_h5_non_liger_ddp_normalization_preflight",
            "central_reference": "sum_of_per_rank_ordinary_chunked_numerators_divided_by_global_targets",
            "chunk_size": H5_SELECTED_CHUNK_SIZE,
            "clip_coefficient": {"observed": observed_coefficient, "reference": reference_coefficient},
            "contract_sha256": contract_sha256,
            "cuda_device": torch.cuda.get_device_name(),
            "central_reference_audits": reference_audits,
            "ddp_gradient_comparison": gradient_metrics,
            "gradient_norm_comparison": norm_metrics,
            "harness_amendment_sha256": amendment_sha256,
            "harness_amendment_r2_sha256": amendment_r2_sha256,
            "includes_zero_target_rank": TARGET_COUNTS[0] == 0,
            "implementation_id": IMPLEMENTATION_ID,
            "loaded_liger_modules": loaded_liger_modules,
            "loss_comparison": loss_metrics,
            "numerical_acceptance": acceptance,
            "parameter_update_comparison": update_metrics,
            "per_rank_assistant_targets": list(TARGET_COUNTS),
            "rank0_checkpointed_observed_audit": module.last_audit,
            "qualification_manifest_sha256": qualification_sha256,
            "rank_tensor_hashes": gathered_hashes,
            "schema_version": 1,
            "status": "passed",
            "world_size": world_size,
        }
        write_json_atomic(args.report_output, report)
        print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))
    torch.distributed.barrier(device_ids=[local_rank])


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
