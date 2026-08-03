#!/usr/bin/env python3
"""Build the frozen 2-, 10-, and 13-update C00 qualification schedules in one verified data pass."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from open_instruct.qwen35_data import Qwen35NumpyPackedDataset
from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_schedule import build_schedule_manifest, validate_schedule_manifest
from open_instruct.qwen35_training import write_json_atomic

UPDATE_COUNTS = (2, 10, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--numpy-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing pre-existing qualification schedule directory: {output}")
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        unit = qualification["training_unit"]
        group = int(unit["global_packs_per_optimizer_update"])
        seed = int(unit["schedule_seed"])
        dataset = Qwen35NumpyPackedDataset(
            args.numpy_data_dir, sequence_length=int(unit["sequence_length"]), drop_last=False, verify_hashes=True
        )
        if dataset.manifest.get("arm_id") != "C00":
            raise ValueError("hardware qualification schedules may be built from C00 only")
        rows = []
        for updates in UPDATE_COUNTS:
            packs = updates * group
            schedule = build_schedule_manifest(
                dataset,
                seed=seed,
                global_packs_per_update=group,
                real_pack_limit=packs,
                allow_synthetic_final_group_padding=False,
            )
            validation = validate_schedule_manifest(
                schedule, dataset, expected_seed=seed, expected_global_packs_per_update=group
            )
            if validation["optimizer_updates"] != updates:
                raise AssertionError("qualification schedule update-count drift")
            name = f"qwen35_c00_seed{seed}_{updates:03d}steps_{packs:03d}packs.json"
            path = temporary / name
            write_json_atomic(path, schedule)
            rows.append(
                {
                    "optimizer_updates": updates,
                    "scheduled_packs": packs,
                    "path": name,
                    "file_sha256": sha256_file(path),
                    "schedule_sha256": schedule["schedule_sha256"],
                    "totals": schedule["totals"],
                    "validation": validation,
                }
            )
        summary = {
            "artifact": "qwen35_hardware_qualification_schedule_set",
            "schema_version": 1,
            "status": "passed",
            "qualification_manifest_sha256": qualification_sha256,
            "numpy_data_dir": str(args.numpy_data_dir.resolve()),
            "numpy_manifest_sha256": sha256_file(args.numpy_data_dir / "manifest.json"),
            "dataset_accounting": dataset.accounting(),
            "schedules": rows,
        }
        write_json_atomic(temporary / "qualification_schedule_set.json", summary)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
