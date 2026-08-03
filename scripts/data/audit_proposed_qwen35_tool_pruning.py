#!/usr/bin/env python3
"""Audit the proposed five-row referenced-tool pruning amendment without packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer

from open_instruct import qwen35_data
from open_instruct.qwen35_data import (
    ChatTemplateTokenizer,
    plan_referenced_tool_pruning,
    prune_qwen35_tools_to_fit,
    tokenize_qwen35_example,
)

EXPECTED = {
    277626: ("cd3392225f3998340230426b48a40bc3a5918157fee30f06d5aa575e74ca45e4", 54927, 99),
    294589: ("591028f79894dd50873a4c51a9fbe60d5bc35b910adf571418a1774d25bf47fd", 55043, 187),
    328871: ("a0d39fcac3cb83b4987c3fc8b1638ab17d39c7043cd83a218833bd3b002a87a8", 55139, 259),
    356557: ("7558c6b2233fc982bfb38af45651084c203b046a0ab124f84fdf0a68b960036f", 55085, 200),
    413482: ("2556104fb4601af091cd5c7a1a08ed7464aac2266dbe27a026ed2eaf68f0f159", 54778, 25),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl-zst", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def token_sha256(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u4").tobytes()).hexdigest()


def mask_sha256(values: list[bool]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.bool_).tobytes()).hexdigest()


def iter_jsonl_zst(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    pending = b""
    with pa.input_stream(str(path), compression="zstd") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("canonical JSONL row is not an object")
                    yield value
    if pending:
        value = json.loads(pending)
        if not isinstance(value, dict):
            raise ValueError("canonical JSONL row is not an object")
        yield value


def main() -> int:
    args = parse_args()
    if args.max_seq_length <= 0:
        raise ValueError("max sequence length must be positive")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    tokenizer_path = args.tokenizer.resolve()
    implementation_path = Path(qwen35_data.__file__).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, revision=args.tokenizer_revision, trust_remote_code=True, local_files_only=True
    )
    if tokenizer is None or not isinstance(tokenizer.chat_template, str):
        raise ValueError("the pinned tokenizer lacks a string native chat template")
    native_tokenizer = cast(ChatTemplateTokenizer, tokenizer)
    found: dict[int, dict[str, Any]] = {}
    for row_number, record in enumerate(iter_jsonl_zst(args.canonical_jsonl_zst.resolve())):
        if row_number in EXPECTED:
            found[row_number] = record
        if row_number >= max(EXPECTED):
            break
    if set(found) != set(EXPECTED):
        raise ValueError(f"missing proposed-amendment rows: {sorted(set(EXPECTED) - set(found))}")

    reports: list[dict[str, Any]] = []
    for row_number in sorted(found):
        record = found[row_number]
        uid, expected_total, expected_loss = EXPECTED[row_number]
        if record.get("sample_uid") != uid:
            raise ValueError(f"sample UID drift at canonical row {row_number}")
        messages = record.get("messages")
        tools = record.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise ValueError(f"canonical row {row_number} lacks messages/tools")
        original = tokenize_qwen35_example(native_tokenizer, messages, tools)
        truncated = tokenize_qwen35_example(native_tokenizer, messages, tools, max_seq_length=args.max_seq_length)
        if len(original.input_ids) != expected_total or original.trainable_tokens != expected_loss:
            raise ValueError(f"original token-count drift at row {row_number}")
        if truncated.trainable_tokens != 0:
            raise ValueError(f"row {row_number} no longer triggers the amendment")
        plan = plan_referenced_tool_pruning(messages, tools)
        pruned = prune_qwen35_tools_to_fit(native_tokenizer, messages, tools, max_seq_length=args.max_seq_length)
        after = pruned.tokenized
        if len(after.input_ids) > args.max_seq_length or after.trainable_tokens <= 0:
            raise RuntimeError(f"proposed pruning failed at row {row_number}")
        reports.append(
            {
                "global_row_number": row_number,
                "source_key": record.get("source_key"),
                "sample_uid": uid,
                "messages_canonical_sha256": canonical_sha256(messages),
                "tools_canonical_sha256": canonical_sha256(tools),
                "referenced_tool_names": list(plan.referenced_tool_names),
                "before": {
                    "tools": len(tools),
                    "total_tokens": len(original.input_ids),
                    "assistant_loss_tokens": original.trainable_tokens,
                    "truncated_assistant_loss_tokens": truncated.trainable_tokens,
                    "token_ids_sha256": token_sha256(original.input_ids),
                    "labels_mask_sha256": mask_sha256(original.labels_mask),
                },
                "after": {
                    "tools": len(pruned.retained_tools),
                    "total_tokens": len(after.input_ids),
                    "assistant_loss_tokens": after.trainable_tokens,
                    "token_ids_sha256": token_sha256(after.input_ids),
                    "labels_mask_sha256": mask_sha256(after.labels_mask),
                    "retained_tools_canonical_sha256": canonical_sha256(pruned.retained_tools),
                },
                "removed_tools": [
                    {
                        "original_index": row.original_index,
                        "function_name": row.function_name,
                        "canonical_sha256": row.canonical_sha256,
                    }
                    for row in pruned.removed_tools
                ],
            }
        )
    report = {
        "artifact": "proposed_qwen35_referenced_tool_pruning_audit",
        "schema_version": 1,
        "status": "proposal_only_not_applied_to_frozen_arms",
        "max_seq_length": args.max_seq_length,
        "inputs": {
            "canonical_jsonl_zst": {
                "path": str(args.canonical_jsonl_zst.resolve()),
                "sha256": sha256_file(args.canonical_jsonl_zst.resolve()),
            },
            "tokenizer": str(tokenizer_path),
            "tokenizer_revision": args.tokenizer_revision,
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "implementation": {"path": str(implementation_path), "sha256": sha256_file(implementation_path)},
        },
        "rows": reports,
        "validation": {
            "expected_rows": len(EXPECTED),
            "observed_rows": len(reports),
            "all_fit": all(row["after"]["total_tokens"] <= args.max_seq_length for row in reports),
            "all_positive_loss": all(row["after"]["assistant_loss_tokens"] > 0 for row in reports),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(f"Wrote proposed pruning audit to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
