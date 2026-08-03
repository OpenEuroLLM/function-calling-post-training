#!/usr/bin/env python3
"""Split or reassemble a large artifact with per-shard and whole-file hashes.

This exists for audited transfer through bandwidth-limited SSH paths.  The
published artifact is created as a sibling ``.incomplete`` file and atomically
renamed only after every ordered shard and the reconstructed whole-file digest
match the deterministic manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, BinaryIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--chunk-size-mib", type=int, default=64)
    split.add_argument("--overwrite", action="store_true")

    reassemble = subparsers.add_parser("reassemble")
    reassemble.add_argument("--manifest", type=Path, required=True)
    reassemble.add_argument("--shard-dir", type=Path, required=True)
    reassemble.add_argument("--output", type=Path, required=True)
    reassemble.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_exact(source: BinaryIO, destination: BinaryIO, *, length: int, digest: Any) -> int:
    copied = 0
    while copied < length:
        block = source.read(min(8 * 1024 * 1024, length - copied))
        if not block:
            break
        destination.write(block)
        digest.update(block)
        copied += len(block)
    return copied


def split_file(input_path: Path, output_dir: Path, chunk_size: int, overwrite: bool) -> None:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    working = output_dir.parent / f".{output_dir.name}.incomplete"
    if working.exists():
        if not overwrite:
            raise FileExistsError(working)
        shutil.rmtree(working)
    working.mkdir(parents=True)

    whole_digest = hashlib.sha256()
    shards: list[dict[str, object]] = []
    total_size = input_path.stat().st_size
    with input_path.open("rb") as source:
        index = 0
        while source.tell() < total_size:
            shard_name = f"{input_path.name}.part-{index:05d}"
            shard_path = working / shard_name
            shard_digest = hashlib.sha256()
            requested = min(chunk_size, total_size - source.tell())
            with shard_path.open("wb") as destination:
                copied = 0
                while copied < requested:
                    block = source.read(min(8 * 1024 * 1024, requested - copied))
                    if not block:
                        raise RuntimeError("source ended before its recorded size")
                    destination.write(block)
                    shard_digest.update(block)
                    whole_digest.update(block)
                    copied += len(block)
                destination.flush()
                os.fsync(destination.fileno())
            shards.append(
                {"index": index, "path": shard_name, "size_bytes": copied, "sha256": shard_digest.hexdigest()}
            )
            index += 1
    manifest = {
        "artifact": "verified_sharded_transfer",
        "schema_version": 1,
        "source_name": input_path.name,
        "size_bytes": total_size,
        "sha256": whole_digest.hexdigest(),
        "chunk_size_bytes": chunk_size,
        "shards": shards,
    }
    (working / "transfer_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(working, output_dir)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "shards": len(shards), "sha256": whole_digest.hexdigest()}, sort_keys=True
        )
    )


def reassemble_file(manifest_path: Path, shard_dir: Path, output: Path, overwrite: bool) -> None:
    manifest_path = manifest_path.resolve()
    shard_dir = shard_dir.resolve()
    output = output.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact") != "verified_sharded_transfer":
        raise ValueError("unexpected transfer manifest artifact")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("transfer manifest contains no shards")
    if [row.get("index") for row in shards] != list(range(len(shards))):
        raise ValueError("transfer shard indices are not contiguous and ordered")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    whole_digest = hashlib.sha256()
    total_size = 0
    try:
        with temporary.open("wb") as destination:
            for row in shards:
                shard_path = shard_dir / row["path"]
                if not shard_path.is_file():
                    raise FileNotFoundError(shard_path)
                expected_size = int(row["size_bytes"])
                if shard_path.stat().st_size != expected_size:
                    raise ValueError(f"shard size drift: {shard_path}")
                shard_digest = hashlib.sha256()
                with shard_path.open("rb") as source:
                    copied = copy_exact(source, destination, length=expected_size, digest=shard_digest)
                    if copied != expected_size or source.read(1):
                        raise ValueError(f"shard length drift: {shard_path}")
                if shard_digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"shard SHA-256 drift: {shard_path}")
                # The bytes were already written; update the whole digest from
                # the verified shard in one sequential pass.
                with shard_path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        whole_digest.update(block)
                total_size += copied
            destination.flush()
            os.fsync(destination.fileno())
        if total_size != manifest["size_bytes"]:
            raise ValueError("reassembled file-size drift")
        if whole_digest.hexdigest() != manifest["sha256"]:
            raise ValueError("reassembled whole-file SHA-256 drift")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {"output": str(output), "size_bytes": total_size, "sha256": whole_digest.hexdigest()}, sort_keys=True
        )
    )


def main() -> int:
    args = parse_args()
    if args.command == "split":
        split_file(args.input, args.output_dir, args.chunk_size_mib * 1024**2, args.overwrite)
    else:
        reassemble_file(args.manifest, args.shard_dir, args.output, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
