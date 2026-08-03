#!/usr/bin/env python3
"""Prove Accelerate consumes a sequential schedule without tail duplication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader, SequentialSampler

from open_instruct.qwen35_qualification import load_qualification_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--schedule-length", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    if accelerator.even_batches:
        raise RuntimeError("Accelerate did not retain even_batches=False")
    world_size = accelerator.num_processes
    if world_size < 2:
        raise RuntimeError("launch this probe with torchrun --nproc_per_node at least 2")
    if args.schedule_length <= 0 or args.schedule_length % world_size:
        raise ValueError("schedule length must be positive and divisible by world size")
    dataset = torch.arange(args.schedule_length, dtype=torch.int64)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1, drop_last=False)
    dataloader = accelerator.prepare(dataloader)
    observed = [int(batch.item()) for batch in dataloader]
    expected = list(range(accelerator.process_index, args.schedule_length, world_size))
    if observed != expected:
        raise AssertionError(f"rank {accelerator.process_index} observed {observed}, expected {expected}")
    gathered: list[list[int] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, observed)
    flattened = sorted(index for rank_indices in gathered for index in (rank_indices or []))
    if flattened != list(range(args.schedule_length)):
        raise AssertionError("distributed sequential schedule was duplicated, omitted, or reordered")
    if accelerator.is_main_process:
        report = {
            "artifact": "qwen35_accelerate_sequential_sharding_qualification",
            "schema_version": 1,
            "status": "passed",
            "qualification_protocol_id": qualification["protocol_id"],
            "qualification_manifest_sha256": qualification_sha256,
            "accelerate_version": __import__("accelerate").__version__,
            "world_size": world_size,
            "schedule_length": args.schedule_length,
            "even_batches": accelerator.even_batches,
            "rank_indices": gathered,
            "global_indices_exactly_once": flattened,
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report_output.with_name(f".{args.report_output.name}.incomplete.{os.getpid()}")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report_output)
        print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
