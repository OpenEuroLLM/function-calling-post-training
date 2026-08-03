#!/usr/bin/env python3
"""Build and validate a deterministic no-repeat Qwen3.5 pack schedule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from open_instruct.qwen35_data import Qwen35NumpyPackedDataset
from open_instruct.qwen35_schedule import build_schedule_manifest, validate_schedule_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--schedule-seed", type=int, required=True)
    parser.add_argument("--global-packs-per-update", type=int, required=True)
    parser.add_argument("--real-pack-limit", type=int)
    parser.add_argument("--target-assistant-tokens", type=int)
    parser.add_argument("--assistant-relative-tolerance", type=float, default=0.001)
    parser.add_argument("--verify-data-hashes", action="store_true")
    parser.add_argument("--allow-synthetic-final-group-padding", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace existing schedule without --overwrite: {args.output}")
    dataset = Qwen35NumpyPackedDataset(
        args.numpy_data_dir,
        sequence_length=args.sequence_length,
        drop_last=False,
        verify_hashes=args.verify_data_hashes,
    )
    schedule = build_schedule_manifest(
        dataset,
        seed=args.schedule_seed,
        global_packs_per_update=args.global_packs_per_update,
        real_pack_limit=args.real_pack_limit,
        target_assistant_tokens=args.target_assistant_tokens,
        assistant_relative_tolerance=args.assistant_relative_tolerance,
        allow_synthetic_final_group_padding=args.allow_synthetic_final_group_padding,
    )
    validation = validate_schedule_manifest(
        schedule,
        dataset,
        expected_seed=args.schedule_seed,
        expected_global_packs_per_update=args.global_packs_per_update,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
