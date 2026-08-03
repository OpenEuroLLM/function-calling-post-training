#!/usr/bin/env python3
"""Validate the accepted 32K renderer amendment on a synthetic native edge.

This is deliberately separate from the frozen 128-example lineage fixture.
It constructs 372 deterministic function definitions and makes only the final,
unreferenced definition large enough to push every assistant target beyond the
ordinary 32K right-truncation boundary.  The accepted rule must remove exactly
that one definition, preserve the referenced definition and all messages, and
reproduce the pinned native-render token and mask digests below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from transformers import Qwen2TokenizerFast

from open_instruct.qwen35_data import (
    REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
    plan_referenced_tool_pruning,
    sha256_text,
    tokenize_qwen35_example,
    tokenize_qwen35_example_with_referenced_tool_pruning,
)

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_TOKENIZER_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
EXPECTED_CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
EXPECTED = {
    "messages_canonical_sha256": "1b50a2bde5fff53268b5bd82e8f5b49fda6acd983929804f86ec74b2cbf6bcd0",
    "tools_canonical_sha256": "75446d25a892644b8fa1b46ba74f6a78fab6feddb72af5942a3901f6c7d2b024",
    "retained_tools_canonical_sha256": "4eb4b410855b7e5fb083fb9f5ef87f85fb0e684fcbc937664835613146e91a31",
    "removed_tool_canonical_sha256": "8a7e4efb3a70a2f5e7a8a059860b123c06e0b112bb874b3f014717e63d663168",
    "full_total_tokens": 45215,
    "full_assistant_loss_tokens": 30,
    "full_token_ids_sha256": "b43b8663880774fea0b6f0bbc7d024149a667ecaa036f6b8aa12b1da270e14c1",
    "full_labels_mask_sha256": "b841e82b8e483deeb97199e8dd2b8785221d6b6741c4db60a6841f5ecda0467e",
    "ordinary_total_tokens": 32768,
    "ordinary_assistant_loss_tokens": 0,
    "ordinary_token_ids_sha256": "595b83975055631eaedba5586372bbbf93c6d112a501256092b5a4d2da51f497",
    "ordinary_labels_mask_sha256": "c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479",
    "amended_total_tokens": 19168,
    "amended_assistant_loss_tokens": 30,
    "amended_token_ids_sha256": "05d2af293a1d6add4c897c0bf285efc79649aff1b2d383447b87512959896457",
    "amended_labels_mask_sha256": "aaf0fb5e52d0612e39ef9d0af746a50dec34c589fac45600bf7258ef5edfc709",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def token_sha256(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u4").tobytes()).hexdigest()


def mask_sha256(values: list[bool]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.bool_).tobytes()).hexdigest()


def build_synthetic_edge_case() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools: list[dict[str, Any]] = []
    for index in range(372):
        function = {
            "name": f"tool_{index:03d}",
            "description": "Synthetic edge tool.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
        }
        if index == 371:
            function["description"] = "padding " * 26000
        tools.append({"type": "function", "function": function})
    messages = [
        {"role": "user", "content": "Call tool zero."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_edge",
                    "type": "function",
                    "function": {"name": "tool_000", "arguments": '{"value":"ok"}'},
                }
            ],
        },
    ]
    return messages, tools


def assert_expected(observed: dict[str, Any]) -> None:
    drift = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED.items()
        if observed.get(key) != expected
    }
    if drift:
        raise AssertionError(f"synthetic native-render regression drift: {drift}")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.tokenizer_name_or_path, revision=args.tokenizer_revision, cache_dir=args.cache_dir
    )
    template_sha256 = sha256_text(tokenizer.chat_template or "")
    if template_sha256 != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise AssertionError("pinned Qwen3.5 native chat-template digest drift")

    messages, tools = build_synthetic_edge_case()
    messages_before = canonical_sha256(messages)
    tools_before = canonical_sha256(tools)
    full = tokenize_qwen35_example(tokenizer, messages, tools)
    ordinary = tokenize_qwen35_example(tokenizer, messages, tools, max_seq_length=32768)
    amended = tokenize_qwen35_example_with_referenced_tool_pruning(tokenizer, messages, tools, max_seq_length=32768)
    if amended.pruning is None:
        raise AssertionError("the synthetic edge did not trigger the amendment")
    pruning = amended.pruning
    plan = plan_referenced_tool_pruning(messages, tools)
    if plan.referenced_tool_names != ("tool_000",):
        raise AssertionError("synthetic referenced-tool set drift")
    if len(pruning.removed_tools) != 1:
        raise AssertionError("the amendment must remove exactly one synthetic definition")
    removed = pruning.removed_tools[0]
    if (removed.original_index, removed.function_name) != (371, "tool_371"):
        raise AssertionError("the amendment removed the wrong synthetic definition")
    if canonical_sha256(messages) != messages_before or canonical_sha256(tools) != tools_before:
        raise AssertionError("renderer amendment mutated the synthetic source objects")
    if pruning.retained_tools[0] != tools[0] or len(pruning.retained_tools) != 371:
        raise AssertionError("referenced definition or retained-tool ordering drift")

    observed = {
        "messages_canonical_sha256": messages_before,
        "tools_canonical_sha256": tools_before,
        "retained_tools_canonical_sha256": canonical_sha256(pruning.retained_tools),
        "removed_tool_canonical_sha256": removed.canonical_sha256,
        "full_total_tokens": len(full.input_ids),
        "full_assistant_loss_tokens": full.trainable_tokens,
        "full_token_ids_sha256": token_sha256(full.input_ids),
        "full_labels_mask_sha256": mask_sha256(full.labels_mask),
        "ordinary_total_tokens": len(ordinary.input_ids),
        "ordinary_assistant_loss_tokens": ordinary.trainable_tokens,
        "ordinary_token_ids_sha256": token_sha256(ordinary.input_ids),
        "ordinary_labels_mask_sha256": mask_sha256(ordinary.labels_mask),
        "amended_total_tokens": len(amended.tokenized.input_ids),
        "amended_assistant_loss_tokens": amended.tokenized.trainable_tokens,
        "amended_token_ids_sha256": token_sha256(amended.tokenized.input_ids),
        "amended_labels_mask_sha256": mask_sha256(amended.tokenized.labels_mask),
    }
    assert_expected(observed)
    report = {
        "artifact": "qwen35_referenced_tool_pruning_synthetic_native_edge",
        "schema_version": 1,
        "status": "passed",
        "amendment_id": REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
        "tokenizer": {
            "name_or_path": args.tokenizer_name_or_path,
            "revision": args.tokenizer_revision,
            "chat_template_sha256": template_sha256,
        },
        "construction": {
            "tool_definitions": len(tools),
            "referenced_tool_names": list(plan.referenced_tool_names),
            "large_unreferenced_tool_index": 371,
            "removed_tool_indices": [row.original_index for row in pruning.removed_tools],
            "messages_preserved": True,
            "referenced_tool_preserved": True,
            "retained_tool_order_preserved": True,
        },
        "observed": observed,
        "expected": EXPECTED,
        "checks": {
            "ordinary_trigger_positive_untruncated_loss": full.trainable_tokens > 0,
            "ordinary_trigger_zero_32k_loss": ordinary.trainable_tokens == 0,
            "amended_fits_32k": len(amended.tokenized.input_ids) <= 32768,
            "amended_positive_loss": amended.tokenized.trainable_tokens > 0,
            "all_pinned_digests_match": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
