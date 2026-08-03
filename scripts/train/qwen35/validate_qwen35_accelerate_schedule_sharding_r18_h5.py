#!/usr/bin/env python3
"""Prove four-rank Accelerate sharding of the exact R18 H5 schedule prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader, SequentialSampler

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file
from open_instruct.qwen35_qualification_r18_h5 import (
    H5_FIRST_FIVE_ENTRIES_SHA256,
    H5_SCHEDULE_FILE_SHA256,
    H5_SCHEDULE_SHA256,
    H5_WORLD_SIZE,
    canonical_json_bytes,
    load_h5_contract,
    load_h5_harness_amendment,
    load_h5_harness_amendment_r2,
)
from open_instruct.qwen35_training import write_json_atomic


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
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if sha256_file(args.schedule) != H5_SCHEDULE_FILE_SHA256:
        raise ValueError("R18 H5 sharding preflight schedule file digest drift")
    schedule = load_strict_json(args.schedule)
    if schedule.get("schedule_sha256") != H5_SCHEDULE_SHA256:
        raise ValueError("R18 H5 sharding preflight embedded schedule digest drift")
    entries = schedule.get("entries")
    if not isinstance(entries, list) or len(entries) != 80:
        raise ValueError("R18 H5 sharding preflight schedule entry-count drift")
    prefix = entries[:40]
    if hashlib.sha256(canonical_json_bytes(prefix)).hexdigest() != H5_FIRST_FIVE_ENTRIES_SHA256:
        raise ValueError("R18 H5 sharding preflight five-update prefix drift")
    if [entry.get("schedule_index") for entry in prefix] != list(range(40)):
        raise ValueError("R18 H5 sharding preflight schedule indices are not contiguous")

    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(even_batches=False))
    if accelerator.even_batches:
        raise RuntimeError("Accelerate did not retain even_batches=False")
    if accelerator.num_processes != H5_WORLD_SIZE:
        raise RuntimeError(f"R18 H5 sharding preflight requires {H5_WORLD_SIZE} ranks")
    dataset = torch.arange(40, dtype=torch.int64)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1, drop_last=False)
    prepared = accelerator.prepare(dataloader)
    observed = [int(batch.item()) for batch in prepared]
    expected = list(range(accelerator.process_index, 40, H5_WORLD_SIZE))
    if observed != expected:
        raise AssertionError(f"rank {accelerator.process_index} observed {observed}, expected {expected}")
    gathered: list[list[int] | None] = [None] * H5_WORLD_SIZE
    torch.distributed.all_gather_object(gathered, observed)
    if gathered != [list(range(rank, 40, H5_WORLD_SIZE)) for rank in range(H5_WORLD_SIZE)]:
        raise AssertionError("R18 H5 rank-wise schedule stride drift")
    flattened = sorted(index for rank_indices in gathered for index in (rank_indices or []))
    if flattened != list(range(40)):
        raise AssertionError("R18 H5 distributed schedule duplicated or omitted an index")
    for update in range(5):
        assigned = sorted(
            index
            for rank_indices in gathered
            for index in (rank_indices or [])
            if 8 * update <= index < 8 * (update + 1)
        )
        if assigned != list(range(8 * update, 8 * (update + 1))):
            raise AssertionError(f"R18 H5 optimizer update {update + 1} assignment drift")
    loaded_liger_modules = sorted(
        name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
    )
    if loaded_liger_modules:
        raise RuntimeError(f"R18 H5 sharding preflight imported forbidden Liger modules: {loaded_liger_modules}")

    if accelerator.is_main_process:
        if args.report_output.exists():
            raise FileExistsError(args.report_output)
        report = {
            "accelerate_version": __import__("accelerate").__version__,
            "artifact": "qwen35_r18_h5_accelerate_schedule_sharding_preflight",
            "contract_sha256": contract_sha256,
            "even_batches": accelerator.even_batches,
            "first_five_entries_sha256": H5_FIRST_FIVE_ENTRIES_SHA256,
            "global_indices_exactly_once": flattened,
            "harness_amendment_sha256": amendment_sha256,
            "harness_amendment_r2_sha256": amendment_r2_sha256,
            "loaded_liger_modules": loaded_liger_modules,
            "optimizer_update_global_indices": [list(range(8 * update, 8 * (update + 1))) for update in range(5)],
            "prefix_pack_indices": [entry["pack_index"] for entry in prefix],
            "prefix_pack_uids": [entry["pack_uid"] for entry in prefix],
            "rank_indices": gathered,
            "schedule_file_sha256": H5_SCHEDULE_FILE_SHA256,
            "schedule_sha256": H5_SCHEDULE_SHA256,
            "schema_version": 1,
            "status": "passed",
            "world_size": accelerator.num_processes,
        }
        write_json_atomic(args.report_output, report)
        print(json.dumps({"output": str(args.report_output), "status": "passed"}, sort_keys=True))
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
