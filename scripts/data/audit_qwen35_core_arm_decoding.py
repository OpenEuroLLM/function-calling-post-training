#!/usr/bin/env python3
"""Decode and independently re-render a deterministic audit of C00-C11.

This is the human-inspection complement to the exhaustive binary verifier. It
selects structural, semantic, source, amendment, and per-arm operation cases;
checks whether each frozen operation retained, edited, or removed the expected
document; decodes the stored NumPy tokens and loss spans; and re-renders every
retained audit case from the canonical record to require exact token-ID and
mask equality.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import Qwen2TokenizerFast

from open_instruct.qwen35_data import (
    REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
    prune_qwen35_tools_to_fit,
    right_truncate_qwen35_example,
    sha256_text,
    tokenize_qwen35_example,
)

ARM_ORDER = tuple(f"C{index:02d}" for index in range(12))
EXPECTED_CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--canonical-jsonl-zst", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--core-operations", type=Path, required=True)
    parser.add_argument("--causal-sample-features", type=Path, required=True)
    parser.add_argument("--renderer-amendment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-chars", type=int, default=1600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with pa.input_stream(str(path), compression="zstd") as compressed:
        stream = io.BufferedReader(compressed)
        while raw_line := stream.readline():
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"unterminated JSONL row in {path}")
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row in {path}")
            yield value


def load_operations(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    operations: dict[str, dict[int, dict[str, Any]]] = {arm_id: {} for arm_id in ARM_ORDER}
    for operation in iter_jsonl_zst(path):
        arm_id = operation.get("arm_id")
        row = operation.get("global_row_number")
        if arm_id not in operations or not isinstance(row, int):
            raise ValueError("invalid operation identity")
        if row in operations[arm_id]:
            raise ValueError(f"duplicate operation for {arm_id} row {row}")
        operations[arm_id][row] = operation
    return operations


def add_case(cases: dict[int, dict[str, Any]], row: int, *, tag: str, inspected_arm: str = "C00") -> None:
    case = cases.setdefault(row, {"tags": [], "inspected_arms": set()})
    if tag not in case["tags"]:
        case["tags"].append(tag)
    case["inspected_arms"].add("C00")
    case["inspected_arms"].add(inspected_arm)


def first_matching_row(table: pa.Table, condition: np.ndarray) -> int:
    rows = table["global_row_number"].to_numpy(zero_copy_only=False)
    matches = np.flatnonzero(condition)
    if len(matches) == 0:
        raise ValueError("a required decoded-audit selection stratum is empty")
    return int(rows[int(matches[0])])


def select_cases(
    feature_path: Path, operations: Mapping[str, Mapping[int, Mapping[str, Any]]], amendment_rows: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    columns = [
        "global_row_number",
        "source_key",
        "has_high_confidence_ams",
        "has_low_confidence_ams",
        "is_c11_pure_candidate",
        "num_tool_calls",
        "num_no_call_traces",
        "num_single_call_traces",
        "num_sequential_traces",
        "num_parallel_traces",
        "num_hybrid_traces",
        "is_multi_turn",
        "qwen_truncated",
        "qwen_total_tokens_untruncated",
        "qwen_assistant_loss_tokens",
    ]
    table = pq.read_table(feature_path, columns=columns)
    rows = table["global_row_number"].to_numpy(zero_copy_only=False)
    if not np.array_equal(rows, np.arange(len(rows), dtype=rows.dtype)):
        raise ValueError("feature rows are not exactly 0..N-1")
    positive_loss = table["qwen_assistant_loss_tokens"].to_numpy(zero_copy_only=False) > 0
    cases: dict[int, dict[str, Any]] = {}
    selections: dict[str, int] = {}

    source_values = np.asarray(table["source_key"].to_pylist(), dtype=object)
    for source in sorted(set(source_values.tolist())):
        row = first_matching_row(table, (source_values == source) & positive_loss)
        selections[f"source:{source}"] = row
        add_case(cases, row, tag=f"source:{source}")

    bool_or_count_criteria = {
        "semantic:high_confidence_ams": table["has_high_confidence_ams"].to_numpy(zero_copy_only=False),
        "semantic:low_confidence_ams": table["has_low_confidence_ams"].to_numpy(zero_copy_only=False),
        "semantic:justified_no_call_candidate": table["is_c11_pure_candidate"].to_numpy(zero_copy_only=False),
        "structure:no_call_only": (table["num_no_call_traces"].to_numpy(zero_copy_only=False) > 0)
        & (table["num_tool_calls"].to_numpy(zero_copy_only=False) == 0),
        "structure:single_call": table["num_single_call_traces"].to_numpy(zero_copy_only=False) > 0,
        "structure:sequential": table["num_sequential_traces"].to_numpy(zero_copy_only=False) > 0,
        "structure:parallel": table["num_parallel_traces"].to_numpy(zero_copy_only=False) > 0,
        "structure:hybrid": table["num_hybrid_traces"].to_numpy(zero_copy_only=False) > 0,
        "structure:multi_turn": table["is_multi_turn"].to_numpy(zero_copy_only=False),
        "structure:truncated": table["qwen_truncated"].to_numpy(zero_copy_only=False),
    }
    for tag, condition in bool_or_count_criteria.items():
        row = first_matching_row(table, np.asarray(condition) & positive_loss)
        selections[tag] = row
        add_case(cases, row, tag=tag)

    lengths = table["qwen_total_tokens_untruncated"].to_numpy(zero_copy_only=False)
    longest_row = int(rows[int(np.argmax(lengths))])
    selections["structure:longest_untruncated"] = longest_row
    add_case(cases, longest_row, tag="structure:longest_untruncated")

    for row in amendment_rows:
        selections[f"renderer_amendment:{row}"] = int(row)
        add_case(cases, int(row), tag="renderer_amendment")

    for arm_id in ARM_ORDER[1:]:
        if not operations[arm_id]:
            raise ValueError(f"{arm_id} has no operation to audit")
        by_action: dict[str, int] = {}
        for row, operation in sorted(operations[arm_id].items()):
            by_action.setdefault(str(operation["action"]), row)
        for action, row in sorted(by_action.items()):
            tag = f"operation:{arm_id}:{action}"
            selections[tag] = row
            add_case(cases, row, tag=tag, inspected_arm=arm_id)

    return cases, selections


def load_selected_canonical(path: Path, selected_rows: set[int]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    maximum = max(selected_rows)
    for row, record in enumerate(iter_jsonl_zst(path)):
        if row in selected_rows:
            selected[row] = record
        if row >= maximum:
            break
    if set(selected) != selected_rows:
        raise ValueError(f"canonical selection incomplete: {sorted(selected_rows - set(selected))}")
    return selected


def load_documents(
    build_root: Path, requested: Mapping[str, set[int]]
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, dict[str, Any]]]:
    documents: dict[str, dict[int, dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for arm_id, rows in sorted(requested.items()):
        manifest = read_json(build_root / arm_id / "manifest.json")
        manifests[arm_id] = manifest
        found: dict[int, dict[str, Any]] = {}
        maximum = max(rows)
        with gzip.open(build_root / arm_id / manifest["documents_index"], "rt") as handle:
            for line in handle:
                document = json.loads(line)
                row = int(document["global_row_number"])
                if row in rows:
                    found[row] = document
                if row > maximum:
                    break
        documents[arm_id] = found
    return documents, manifests


def apply_edit(messages: Sequence[Mapping[str, Any]], operation: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cursor = 0
    removed = 0
    for span in operation["merged_spans"]:
        start, stop = (int(value) for value in span)
        if start < cursor or stop <= start or stop > len(messages):
            raise ValueError("invalid decoded-audit edit span")
        output.extend(dict(message) for message in messages[cursor:start])
        cursor = stop
        removed += stop - start
    output.extend(dict(message) for message in messages[cursor:])
    if removed != int(operation["removed_messages"]):
        raise ValueError("decoded-audit removed-message drift")
    return output


def stored_arrays(
    build_root: Path, arm_id: str, manifest: Mapping[str, Any], document: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    ids: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for segment in document["segments"]:
        part = manifest["parts"][int(segment["part"])]
        start, end = int(segment["start"]), int(segment["end"])
        num_tokens = int(part["num_tokens"])
        token_array = np.memmap(
            build_root / arm_id / part["token_ids"], mode="r", dtype=np.dtype(part["token_dtype"]), shape=(num_tokens,)
        )
        mask_array = np.memmap(
            build_root / arm_id / part["labels_mask"], mode="r", dtype=np.bool_, shape=(num_tokens,)
        )
        ids.append(np.asarray(token_array[start:end]).copy())
        masks.append(np.asarray(mask_array[start:end]).copy())
    return np.concatenate(ids), np.concatenate(masks)


def contiguous_true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate((np.asarray([False]), mask, np.asarray([False])))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in transitions.reshape(-1, 2)]


def preview(value: str, limit: int) -> dict[str, Any]:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if len(value) <= limit:
        rendered = value
        truncated = False
    else:
        left = limit // 2
        right = limit - left
        rendered = value[:left] + "\n…<PREVIEW_TRUNCATED>…\n" + value[-right:]
        truncated = True
    return {"characters": len(value), "sha256": digest, "truncated": truncated, "text": rendered}


def expected_render(
    tokenizer: Qwen2TokenizerFast,
    record: Mapping[str, Any],
    operation: Mapping[str, Any] | None,
    *,
    max_seq_length: int,
) -> tuple[list[int], list[bool], bool, list[dict[str, Any]]]:
    messages = record["messages"]
    if operation is not None and operation["action"] == "drop_real_turn_spans":
        messages = apply_edit(messages, operation)
    tools = record["tools"]
    full = tokenize_qwen35_example(tokenizer, messages, tools, enable_thinking=False)
    ordinary = right_truncate_qwen35_example(full, max_seq_length)
    if ordinary.trainable_tokens > 0:
        return ordinary.input_ids, ordinary.labels_mask, False, messages
    pruned = prune_qwen35_tools_to_fit(tokenizer, messages, tools, max_seq_length=max_seq_length)
    return pruned.tokenized.input_ids, pruned.tokenized.labels_mask, True, messages


def count_calls(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(message.get("tool_calls") or []) for message in messages)


def main() -> int:
    args = parse_args()
    if args.preview_chars < 200:
        raise ValueError("preview-chars must be at least 200")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    build_root = args.build_root.resolve()
    root_manifest = read_json(build_root / "build_manifest.json")
    if root_manifest.get("full_build") is not True or set(root_manifest.get("arms", {})) != set(ARM_ORDER):
        raise ValueError("decoded audit requires a full C00-C11 build")
    if root_manifest.get("build_configuration", {}).get("selected_arms") != list(ARM_ORDER):
        raise ValueError("decoded audit requires the frozen arm order")

    canonical_path = args.canonical_jsonl_zst.resolve()
    index_path = args.sample_index.resolve()
    operations_path = args.core_operations.resolve()
    feature_path = args.causal_sample_features.resolve()
    amendment_path = args.renderer_amendment_manifest.resolve()
    passed_hashes = {
        "canonical_jsonl_zst_sha256": sha256_file(canonical_path),
        "sample_index_sha256": sha256_file(index_path),
        "core_operations_sha256": sha256_file(operations_path),
    }
    for key, observed in passed_hashes.items():
        if root_manifest["inputs"].get(key) != observed:
            raise ValueError(f"decoded-audit input pin drift for {key}")

    amendment = read_json(amendment_path)
    if amendment.get("status") != "accepted_pre_outcome":
        raise ValueError("renderer amendment is not accepted_pre_outcome")
    if amendment.get("amendment_id") != REFERENCED_TOOL_PRUNING_AMENDMENT_ID:
        raise ValueError("renderer amendment ID drift")
    amendment_rows = [int(row) for row in amendment["expected_global_rows"]]
    operations = load_operations(operations_path)
    cases, selections = select_cases(feature_path, operations, amendment_rows)

    index = pq.read_table(index_path, columns=["global_row_number", "source_key", "sample_uid"])
    if len(index) != int(root_manifest["input_rows"]):
        raise ValueError("sample-index/build row-count drift")
    index_rows = index["global_row_number"].to_numpy(zero_copy_only=False)
    if not np.array_equal(index_rows, np.arange(len(index_rows), dtype=index_rows.dtype)):
        raise ValueError("sample-index rows are not exactly 0..N-1")
    source_by_row = index["source_key"].to_pylist()
    uid_by_row = index["sample_uid"].to_pylist()

    selected_rows = set(cases)
    canonical = load_selected_canonical(canonical_path, selected_rows)
    requested: dict[str, set[int]] = defaultdict(set)
    for row, case in cases.items():
        for arm_id in case["inspected_arms"]:
            requested[arm_id].add(row)
    documents, manifests = load_documents(build_root, requested)

    tokenizer = Qwen2TokenizerFast.from_pretrained(build_root / "tokenizer")
    if sha256_text(tokenizer.chat_template or "") != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("decoded-audit tokenizer template drift")

    audited_cases: list[dict[str, Any]] = []
    for row in sorted(cases):
        record = canonical[row]
        if record.get("sample_uid") != uid_by_row[row] or record.get("source_key") != source_by_row[row]:
            raise ValueError(f"canonical/index identity drift at row {row}")
        arm_results: dict[str, Any] = {}
        for arm_id in sorted(cases[row]["inspected_arms"]):
            operation = operations[arm_id].get(row)
            document = documents[arm_id].get(row)
            expected_presence = operation is None or operation["action"] != "drop_sample"
            if (document is not None) != expected_presence:
                raise ValueError(f"{arm_id} row {row} presence does not match frozen operation")
            result: dict[str, Any] = {
                "expected_presence": expected_presence,
                "observed_presence": document is not None,
                "operation": operation,
            }
            if document is not None:
                manifest = manifests[arm_id]
                token_ids, mask = stored_arrays(build_root, arm_id, manifest, document)
                expected_ids, expected_mask, expected_amended, rendered_messages = expected_render(
                    tokenizer, record, operation, max_seq_length=int(manifest["max_seq_length"])
                )
                if not np.array_equal(token_ids, np.asarray(expected_ids, dtype=token_ids.dtype)):
                    raise ValueError(f"{arm_id} row {row} stored token IDs differ from independent re-render")
                if not np.array_equal(mask, np.asarray(expected_mask, dtype=np.bool_)):
                    raise ValueError(f"{arm_id} row {row} stored mask differs from independent re-render")
                if bool(document["renderer_amended"]) != expected_amended:
                    raise ValueError(f"{arm_id} row {row} amendment flag differs from independent re-render")
                decoded = tokenizer.decode(
                    token_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                spans = contiguous_true_spans(mask)
                decoded_loss_spans = [
                    tokenizer.decode(
                        token_ids[start:stop].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    for start, stop in spans
                ]
                result.update(
                    {
                        "stored_equals_independent_rerender": True,
                        "renderer_amended": expected_amended,
                        "num_tokens": int(len(token_ids)),
                        "num_trainable_tokens": int(np.count_nonzero(mask)),
                        "loss_token_spans": [list(span) for span in spans],
                        "rendered_message_roles": [message["role"] for message in rendered_messages],
                        "rendered_tool_calls": count_calls(rendered_messages),
                        "native_marker_counts": {
                            "assistant_headers": decoded.count("<|im_start|>assistant"),
                            "tool_calls": decoded.count("<tool_call>"),
                            "tool_responses": decoded.count("<tool_response>"),
                        },
                        "decoded_document": preview(decoded, args.preview_chars),
                        "decoded_trainable_spans": [preview(text, args.preview_chars) for text in decoded_loss_spans],
                    }
                )
            arm_results[arm_id] = result
        audited_cases.append(
            {
                "global_row_number": row,
                "sample_uid": uid_by_row[row],
                "source_key": source_by_row[row],
                "tags": cases[row]["tags"],
                "canonical_structure": {
                    "message_roles": [message["role"] for message in record["messages"]],
                    "num_messages": len(record["messages"]),
                    "num_tools": len(record["tools"]),
                    "num_tool_calls": count_calls(record["messages"]),
                },
                "arms": arm_results,
            }
        )

    report = {
        "artifact": "qwen35_core_C00_C11_decoded_audit",
        "schema_version": 1,
        "status": "passed",
        "purpose": "deterministic human-readable decoding plus exact canonical re-render checks",
        "build_root": str(build_root),
        "build_manifest_sha256": sha256_file(build_root / "build_manifest.json"),
        "inputs": {
            **passed_hashes,
            "causal_sample_features_sha256": sha256_file(feature_path),
            "renderer_amendment_manifest_sha256": sha256_file(amendment_path),
        },
        "selection_rules": selections,
        "unique_cases": len(audited_cases),
        "all_retained_cases_exactly_rerendered": True,
        "all_frozen_operation_presence_checks_passed": True,
        "cases": audited_cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "status": "passed", "unique_cases": len(audited_cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
