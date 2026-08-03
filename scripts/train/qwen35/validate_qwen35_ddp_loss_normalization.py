#!/usr/bin/env python3
"""Adversarial DDP proof for unequal Qwen3.5 assistant-target counts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from open_instruct.qwen35_qualification import (
    load_qualification_manifest,
    tensor_comparison_metrics,
    validate_comparison_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1024)
    return parser.parse_args()


def rank_case(rank: int, hidden_size: int, vocab_size: int, device: torch.device):
    supervised = rank
    generator = torch.Generator(device=device).manual_seed(10_000 + rank)
    if supervised:
        hidden = torch.randn(supervised, hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
        targets = torch.randint(0, vocab_size, (supervised,), generator=generator, device=device)
    else:
        hidden = torch.randn(1, hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
        targets = torch.full((1,), -100, device=device, dtype=torch.long)
    return hidden, targets, supervised


class TinySelectiveHead(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, device: torch.device) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(1234)
        self.weight = torch.nn.Parameter(
            torch.randn(vocab_size, hidden_size, generator=generator, device=device, dtype=torch.float32) * 0.02
        )
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum", accum_dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, targets: torch.Tensor, global_divisor: torch.Tensor, world_size: int):
        # Mirrors Trainer: local sum/global target count, then multiply by
        # world size so DDP's gradient average becomes the global target mean.
        return self.loss(self.weight, hidden, targets) / global_divisor * world_size


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP normalization qualification requires CUDA")
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        raise RuntimeError("launch this validator with torchrun --nproc_per_node at least 2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)

    module = TinySelectiveHead(args.hidden_size, args.vocab_size, device)
    initial_weight = module.weight.detach().clone()
    ddp = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
    hidden, targets, local_count = rank_case(rank, args.hidden_size, args.vocab_size, device)
    global_divisor = torch.tensor(local_count, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(global_divisor, op=torch.distributed.ReduceOp.SUM)
    if int(global_divisor) <= 0:
        raise RuntimeError("adversarial DDP case unexpectedly has no supervision")
    loss = ddp(hidden, targets, global_divisor, world_size)
    loss.backward()
    observed_gradient = module.weight.grad.detach().clone()

    reference_weight = initial_weight.detach().clone().requires_grad_(True)
    reference_loss = torch.zeros((), dtype=torch.float32, device=device)
    per_rank_counts = []
    for case_rank in range(world_size):
        case_hidden, case_targets, case_count = rank_case(case_rank, args.hidden_size, args.vocab_size, device)
        per_rank_counts.append(case_count)
        if case_count:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = torch.nn.functional.linear(case_hidden, reference_weight)
            reference_loss = reference_loss + torch.nn.functional.cross_entropy(
                logits.float(), case_targets, reduction="sum"
            )
    reference_loss = reference_loss / int(global_divisor)
    reference_loss.backward()
    gradient_metrics = tensor_comparison_metrics(observed_gradient, reference_weight.grad)
    validate_comparison_metrics(
        gradient_metrics,
        qualification["numerical_acceptance"],
        kind="gradient",
        context="isolated-head DDP globally normalized gradient",
    )
    finite = torch.tensor(int(torch.isfinite(loss) and torch.isfinite(observed_gradient).all()), device=device)
    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    if not int(finite):
        raise AssertionError("at least one DDP rank produced a non-finite loss or gradient")

    if rank == 0:
        report = {
            "artifact": "qwen35_ddp_global_target_normalization_qualification",
            "schema_version": 1,
            "status": "passed",
            "qualification_protocol_id": qualification["protocol_id"],
            "qualification_manifest_sha256": qualification_sha256,
            "world_size": world_size,
            "per_rank_assistant_targets": per_rank_counts,
            "global_assistant_target_divisor": int(global_divisor),
            "includes_zero_target_rank": per_rank_counts[0] == 0,
            "ddp_gradient_comparison": gradient_metrics,
            "numerical_acceptance": qualification["numerical_acceptance"],
            "cuda_device": torch.cuda.get_device_name(),
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report_output.with_name(f".{args.report_output.name}.incomplete.{os.getpid()}")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report_output)
        print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
