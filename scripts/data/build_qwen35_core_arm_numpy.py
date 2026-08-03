#!/usr/bin/env python3
"""Build all frozen C00-C11 Qwen3.5 NumPy arms in one ordered render pass.

The builder verifies the canonical contract and frozen operation ledger, renders
each canonical row once, rerenders only surviving C02/C10 message-span edits,
applies the accepted five-row renderer amendment explicitly, and writes one
trainer-consumable NumPy contract per arm. No model training or evaluation is
performed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import sys
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from scripts.data.convert_sft_data_for_qwen35 import (
    PartWriter,
    choose_token_dtype,
    deterministic_gzip_text,
    load_renderer_amendment,
    sha256_file,
    write_json,
)
from transformers import Qwen2TokenizerFast

from open_instruct.qwen35_data import (
    NUMPY_CONTRACT_VERSION,
    REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
    TokenizedQwen35Example,
    compute_qwen35_token_feature_row,
    prune_qwen35_tools_to_fit,
    right_truncate_qwen35_example,
    sha256_text,
    tokenize_qwen35_example,
)

ARM_ORDER = tuple(f"C{index:02d}" for index in range(12))
EDIT_ARMS = frozenset({"C02", "C10"})
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_TOKENIZER_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PINNED_TRANSFORMERS_COMMIT = "d7d894cf917562d62c61497588ab64e4ae2c699d"
EXPECTED_CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
_WORKER_TOKENIZER: Qwen2TokenizerFast | None = None

QWEN_TOKEN_FEATURE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("global_row_number", pa.int64(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("sample_uid", pa.string(), nullable=False),
        pa.field("canonical_record_sha256", pa.string(), nullable=False),
        pa.field("max_seq_length", pa.int32(), nullable=False),
        pa.field("qwen_total_tokens_untruncated", pa.int64(), nullable=False),
        pa.field("qwen_total_tokens", pa.int32(), nullable=False),
        pa.field("qwen_truncated", pa.bool_(), nullable=False),
        pa.field("qwen_assistant_loss_tokens_untruncated", pa.int64(), nullable=False),
        pa.field("qwen_assistant_loss_tokens", pa.int32(), nullable=False),
        pa.field("qwen_fc_assistant_loss_tokens_untruncated", pa.int64(), nullable=False),
        pa.field("qwen_fc_assistant_loss_tokens", pa.int32(), nullable=False),
        pa.field("qwen_first_token_trainable", pa.bool_(), nullable=False),
        pa.field("qwen_last_token_trainable", pa.bool_(), nullable=False),
        pa.field("assistant_message_indices", pa.list_(pa.int32()), nullable=False),
        pa.field("assistant_loss_tokens_by_message_untruncated", pa.list_(pa.int64()), nullable=False),
        pa.field("assistant_loss_tokens_by_message", pa.list_(pa.int32()), nullable=False),
    ],
    metadata={b"artifact": b"qwen35_causal_token_features_after_renderer_amendment", b"schema_version": b"1"},
)


@dataclass(frozen=True)
class RenderedDocument:
    input_ids: list[int]
    labels_mask: list[bool]
    ordinary_num_tokens_before_truncation: int
    ordinary_num_tokens_after_truncation: int
    messages_sha256: str
    amendment: dict[str, Any] | None

    @property
    def trainable_tokens(self) -> int:
        return sum(self.labels_mask)


@dataclass(frozen=True)
class RenderJobResult:
    global_row_number: int
    source_key: str
    sample_uid: str
    canonical_record_sha256: str
    base: RenderedDocument
    edits: dict[str, RenderedDocument]
    base_feature: dict[str, Any]


@dataclass
class ArmStatistics:
    input_records: int = 0
    targeted_records: int = 0
    dropped_records: int = 0
    edited_records: int = 0
    kept_unedited_records: int = 0
    output_documents: int = 0
    operation_removed_messages: int = 0
    surviving_edit_removed_messages: int = 0
    total_tokens: int = 0
    trainable_tokens: int = 0
    ordinary_truncated_records: int = 0
    referenced_tool_pruned_records: int = 0
    referenced_tool_definitions_removed: int = 0
    per_source_documents: Counter[str] = field(default_factory=Counter)
    per_source_tokens: Counter[str] = field(default_factory=Counter)
    per_source_trainable_tokens: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        counter_keys = ("per_source_documents", "per_source_tokens", "per_source_trainable_tokens")
        # dataclasses.asdict() reconstructs Counter from an iterator of pairs,
        # which turns those pairs into tuple-valued keys. Copy scalar fields
        # directly and normalize the counters explicitly instead.
        values = {key: value for key, value in vars(self).items() if key not in counter_keys}
        for key in counter_keys:
            values[key] = dict(sorted(getattr(self, key).items()))
        values["trainable_token_fraction"] = self.trainable_tokens / self.total_tokens
        return values


class FeatureWriter:
    def __init__(self, path: Path, batch_size: int) -> None:
        self.rows = 0
        self.batch_size = batch_size
        self.buffer: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(
            path, QWEN_TOKEN_FEATURE_SCHEMA, compression="zstd", use_dictionary=True, write_statistics=True
        )

    def append(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        self.rows += 1
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(
                pa.Table.from_pylist(self.buffer, schema=QWEN_TOKEN_FEATURE_SCHEMA), row_group_size=self.batch_size
            )
            self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


class ArmSink:
    def __init__(self, root: Path, arm_id: str, token_dtype: np.dtype, max_tokens_per_part: int) -> None:
        self.arm_id = arm_id
        self.output_dir = root / arm_id
        self.output_dir.mkdir()
        self.writer = PartWriter(self.output_dir, token_dtype, max_tokens_per_part)
        self.document_handle, self.document_raw_handle = deterministic_gzip_text(
            self.output_dir / "documents.jsonl.gz"
        )
        self.amendment_handle, self.amendment_raw_handle = deterministic_gzip_text(
            self.output_dir / "renderer_amendments.jsonl.gz"
        )
        self.statistics = ArmStatistics()
        self.amended_uids: set[str] = set()

    def drop(self, operation: Mapping[str, Any]) -> None:
        self.statistics.targeted_records += 1
        self.statistics.dropped_records += 1
        self.statistics.operation_removed_messages += int(operation.get("removed_messages", 0))

    def write_document(
        self,
        rendered: RenderedDocument,
        *,
        global_row_number: int,
        source_key: str,
        sample_uid: str,
        canonical_record_sha256: str,
        operation: Mapping[str, Any] | None,
    ) -> None:
        if rendered.trainable_tokens <= 0:
            raise RuntimeError(f"{self.arm_id} row {global_row_number} has no trainable token")
        action = "keep" if operation is None else str(operation["action"])
        if operation is None:
            self.statistics.kept_unedited_records += 1
        else:
            self.statistics.targeted_records += 1
            self.statistics.edited_records += 1
            removed_messages = int(operation["removed_messages"])
            self.statistics.operation_removed_messages += removed_messages
            self.statistics.surviving_edit_removed_messages += removed_messages
        segments = self.writer.write_document(rendered.input_ids, rendered.labels_mask)
        self.statistics.output_documents += 1
        self.statistics.total_tokens += len(rendered.input_ids)
        self.statistics.trainable_tokens += rendered.trainable_tokens
        self.statistics.ordinary_truncated_records += (
            rendered.ordinary_num_tokens_before_truncation > rendered.ordinary_num_tokens_after_truncation
        )
        self.statistics.per_source_documents[source_key] += 1
        self.statistics.per_source_tokens[source_key] += len(rendered.input_ids)
        self.statistics.per_source_trainable_tokens[source_key] += rendered.trainable_tokens
        self.document_handle.write(
            json.dumps(
                {
                    "global_row_number": global_row_number,
                    "source_key": source_key,
                    "sample_uid": sample_uid,
                    "canonical_record_sha256": canonical_record_sha256,
                    "operation_id": operation.get("operation_id") if operation else None,
                    "operation_action": action,
                    "rendered_messages_sha256": rendered.messages_sha256,
                    "num_tokens": len(rendered.input_ids),
                    "num_trainable_tokens": rendered.trainable_tokens,
                    "renderer_amended": rendered.amendment is not None,
                    "segments": segments,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        if rendered.amendment is not None:
            if sample_uid in self.amended_uids:
                raise RuntimeError(f"duplicate amendment row in {self.arm_id}: {sample_uid}")
            self.amended_uids.add(sample_uid)
            self.statistics.referenced_tool_pruned_records += 1
            self.statistics.referenced_tool_definitions_removed += len(rendered.amendment["removed_tools"])
            self.amendment_handle.write(
                json.dumps(
                    {
                        **rendered.amendment,
                        "arm_id": self.arm_id,
                        "global_row_number": global_row_number,
                        "source_key": source_key,
                        "sample_uid": sample_uid,
                        "operation_id": operation.get("operation_id") if operation else None,
                        "operation_action": action,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def close(self) -> list[dict[str, Any]]:
        self.document_handle.close()
        self.document_raw_handle.close()
        self.amendment_handle.close()
        self.amendment_raw_handle.close()
        return self.writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl-zst", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--contract-manifest", type=Path, required=True)
    parser.add_argument("--core-operations", type=Path, required=True)
    parser.add_argument("--core-membership-manifest", type=Path, required=True)
    parser.add_argument("--frozen-design-manifest", type=Path, required=True)
    parser.add_argument("--renderer-amendment-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-part-size-gib", type=float, default=4.0)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=ARM_ORDER, default=list(ARM_ORDER))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with pa.input_stream(str(path), compression="zstd") as compressed:
        stream = io.BufferedReader(compressed)
        while raw_line := stream.readline():
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"JSONL row in {path} has no trailing newline")
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row in {path} is not an object")
            yield row


def iter_index_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    required = {
        "global_row_number",
        "source_key",
        "sample_uid",
        "canonical_record_sha256",
        "logical_uncompressed_offset",
        "logical_uncompressed_length",
    }
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"sample index lacks columns: {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=65536, columns=sorted(required)):
        yield from batch.to_pylist()


def verify_runtime() -> dict[str, str]:
    direct_url_text = metadata.distribution("transformers").read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("Transformers has no direct_url.json; source pin is unprovable")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    source_url = direct_url.get("url", "")
    archive_pinned = (
        isinstance(source_url, str) and PINNED_TRANSFORMERS_COMMIT in source_url and "/archive/" in source_url
    )
    if commit != PINNED_TRANSFORMERS_COMMIT and not archive_pinned:
        raise RuntimeError("Transformers installation does not prove the pinned source commit")
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "transformers": transformers.__version__,
        "transformers_commit": PINNED_TRANSFORMERS_COMMIT,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }


def initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
    if sha256_text(_WORKER_TOKENIZER.chat_template or "") != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("worker tokenizer chat-template drift")


def apply_message_edit(messages: Sequence[Mapping[str, Any]], operation: Mapping[str, Any]) -> list[dict[str, Any]]:
    if operation.get("action") != "drop_real_turn_spans":
        raise ValueError("message editing requires drop_real_turn_spans")
    spans = operation.get("merged_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("edit operation has no merged spans")
    output: list[dict[str, Any]] = []
    cursor = 0
    removed = 0
    for raw_span in spans:
        if not isinstance(raw_span, list) or len(raw_span) != 2:
            raise ValueError("edit span must be [start, stop]")
        start, stop = (int(value) for value in raw_span)
        if start < cursor or stop <= start or stop > len(messages):
            raise ValueError(f"invalid or overlapping edit span {(start, stop)}")
        output.extend(dict(message) for message in messages[cursor:start])
        cursor = stop
        removed += stop - start
    output.extend(dict(message) for message in messages[cursor:])
    if removed != operation.get("removed_messages"):
        raise ValueError("edit removed-message accounting drift")
    if not any(message.get("role") == "user" for message in output):
        raise ValueError("surviving edit has no user message")
    if not any(message.get("role") == "assistant" for message in output):
        raise ValueError("surviving edit has no assistant message")
    return output


def render_document(
    record: Mapping[str, Any], messages: Sequence[Mapping[str, Any]], *, max_seq_length: int
) -> tuple[RenderedDocument, TokenizedQwen35Example]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    tools = record.get("tools")
    if not isinstance(tools, list):
        raise ValueError("canonical tools must be a list")
    full = tokenize_qwen35_example(_WORKER_TOKENIZER, messages, tools, enable_thinking=False)
    ordinary = right_truncate_qwen35_example(full, max_seq_length)
    if full.trainable_tokens <= 0:
        raise ValueError(f"sample {record.get('sample_uid')} has no untruncated assistant loss")
    amendment: dict[str, Any] | None = None
    feature_input = full
    output = ordinary
    if ordinary.trainable_tokens == 0:
        pruned = prune_qwen35_tools_to_fit(_WORKER_TOKENIZER, messages, tools, max_seq_length=max_seq_length)
        output = pruned.tokenized
        feature_input = pruned.tokenized
        amendment = {
            "amendment_id": REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
            "ordinary_num_tokens_before_truncation": len(full.input_ids),
            "ordinary_num_tokens_after_truncation": len(ordinary.input_ids),
            "ordinary_trainable_tokens_after_truncation": ordinary.trainable_tokens,
            "amended_num_tokens": len(output.input_ids),
            "amended_trainable_tokens": output.trainable_tokens,
            "tools_before": len(tools),
            "tools_after": len(pruned.retained_tools),
            "removed_tools": [asdict(value) for value in pruned.removed_tools],
        }
    if output.trainable_tokens <= 0 or len(output.input_ids) > max_seq_length:
        raise RuntimeError("final amended render is invalid")
    return (
        RenderedDocument(
            input_ids=output.input_ids,
            labels_mask=output.labels_mask,
            ordinary_num_tokens_before_truncation=len(full.input_ids),
            ordinary_num_tokens_after_truncation=len(ordinary.input_ids),
            messages_sha256=canonical_sha256(messages),
            amendment=amendment,
        ),
        feature_input,
    )


def render_batch(
    jobs: list[tuple[int, dict[str, Any], str, dict[str, dict[str, Any]]]], max_seq_length: int
) -> list[RenderJobResult]:
    results: list[RenderJobResult] = []
    for global_row_number, record, record_sha256, edit_operations in jobs:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError("canonical messages must be a list")
        base, feature_input = render_document(record, messages, max_seq_length=max_seq_length)
        feature = compute_qwen35_token_feature_row(
            record,
            feature_input,
            global_row_number=global_row_number,
            canonical_record_sha256=record_sha256,
            max_seq_length=max_seq_length,
        )
        edits: dict[str, RenderedDocument] = {}
        for arm_id, operation in sorted(edit_operations.items()):
            edited_messages = apply_message_edit(messages, operation)
            edits[arm_id], _ = render_document(record, edited_messages, max_seq_length=max_seq_length)
        results.append(
            RenderJobResult(
                global_row_number=global_row_number,
                source_key=str(record["source_key"]),
                sample_uid=str(record["sample_uid"]),
                canonical_record_sha256=record_sha256,
                base=base,
                edits=edits,
                base_feature=feature,
            )
        )
    return results


def load_operations(path: Path, *, expected_sha256: str, expected_count: int) -> dict[int, dict[str, dict[str, Any]]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("core operation ledger SHA-256 drift")
    result: dict[int, dict[str, dict[str, Any]]] = {}
    count = 0
    previous_key: tuple[int, int] | None = None
    for operation in iter_jsonl_zst(path):
        arm_id = operation.get("arm_id")
        row_number = operation.get("global_row_number")
        action = operation.get("action")
        if arm_id not in ARM_ORDER or not isinstance(row_number, int):
            raise ValueError("invalid operation arm or global row")
        if action not in {"drop_sample", "drop_real_turn_spans"}:
            raise ValueError(f"invalid operation action {action!r}")
        # The frozen unified ledger is grouped by arm and ordered by canonical
        # row within each arm.  Preserve and verify that exact serialization
        # contract even though the in-memory lookup below is row-major.
        key = (ARM_ORDER.index(arm_id), row_number)
        if previous_key is not None and key <= previous_key:
            raise ValueError("core operation ledger is not in strict arm/canonical-row order")
        previous_key = key
        by_arm = result.setdefault(row_number, {})
        if arm_id in by_arm:
            raise ValueError(f"duplicate operation for row {row_number}, arm {arm_id}")
        by_arm[arm_id] = operation
        count += 1
    if count != expected_count:
        raise ValueError(f"core operation count {count} != expected {expected_count}")
    return result


def prepare_output_root(output_root: Path, overwrite: bool) -> Path:
    output_root = output_root.resolve()
    working = output_root.parent / f".{output_root.name}.incomplete"
    for path in (output_root, working):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"output exists: {path}")
            shutil.rmtree(path)
    working.mkdir(parents=True)
    return working


def tokenizer_artifacts(tokenizer_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(tokenizer_dir.iterdir())
        if path.is_file()
    ]


def main() -> int:
    global _WORKER_TOKENIZER
    args = parse_args()
    selected_arms = tuple(arm for arm in ARM_ORDER if arm in set(args.arms))
    if not selected_arms:
        raise ValueError("at least one arm is required")
    if args.max_seq_length != 32768:
        raise ValueError("the accepted core renderer amendment is restricted to 32768")
    if args.max_part_size_gib <= 0 or args.feature_batch_size <= 0:
        raise ValueError("part size and feature batch size must be positive")
    if args.workers <= 0 or args.worker_batch_size <= 0:
        raise ValueError("workers and worker batch size must be positive")
    if args.num_examples < 0 or args.progress_every < 0:
        raise ValueError("num_examples and progress_every cannot be negative")

    runtime = verify_runtime()
    canonical_path = args.canonical_jsonl_zst.resolve()
    index_path = args.sample_index.resolve()
    contract_path = args.contract_manifest.resolve()
    operations_path = args.core_operations.resolve()
    membership_manifest_path = args.core_membership_manifest.resolve()
    design_path = args.frozen_design_manifest.resolve()
    amendment_path = args.renderer_amendment_manifest.resolve()
    contract = json.loads(contract_path.read_text())
    membership_manifest = json.loads(membership_manifest_path.read_text())
    design = json.loads(design_path.read_text())
    amendment, amendment_sha256 = load_renderer_amendment(amendment_path)
    if amendment is None or amendment_sha256 is None:
        raise RuntimeError("accepted renderer amendment was not loaded")
    if design.get("suite_id") != membership_manifest.get("suite_id"):
        raise ValueError("frozen design and core membership suite IDs differ")
    design_pin = membership_manifest["inputs"]["frozen_design_manifest"]
    if sha256_file(design_path) != design_pin["sha256"]:
        raise ValueError("frozen design SHA-256 drift")
    canonical_pin = contract["artifacts"]["canonical_samples"]
    index_pin = contract["artifacts"]["sample_index"]
    if sha256_file(canonical_path) != canonical_pin["sha256"]:
        raise ValueError("canonical compressed SHA-256 drift")
    if sha256_file(index_path) != index_pin["sha256"]:
        raise ValueError("sample-index SHA-256 drift")
    expected_rows = int(canonical_pin["rows"])
    operations_pin = membership_manifest["unified_operations"]
    operations = load_operations(
        operations_path, expected_sha256=operations_pin["sha256"], expected_count=int(operations_pin["operations"])
    )
    summary_path = membership_manifest_path.parent / membership_manifest["summary"]["path"]
    if sha256_file(summary_path) != membership_manifest["summary"]["sha256"]:
        raise ValueError("core summary SHA-256 drift")
    expected_summary = json.loads(summary_path.read_text())

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.tokenizer_name_or_path, revision=args.tokenizer_revision, cache_dir=args.cache_dir
    )
    if sha256_text(tokenizer.chat_template or "") != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("pinned Qwen3.5 chat-template SHA-256 drift")
    token_dtype = choose_token_dtype(tokenizer)
    max_tokens_per_part = int(args.max_part_size_gib * 1024**3) // token_dtype.itemsize

    working = prepare_output_root(args.output_root, args.overwrite)
    tokenizer_dir = working / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer_files = tokenizer_artifacts(tokenizer_dir)
    sinks = {arm_id: ArmSink(working, arm_id, token_dtype, max_tokens_per_part) for arm_id in selected_arms}
    for sink in sinks.values():
        shutil.copytree(tokenizer_dir, sink.output_dir / "tokenizer")
    feature_path = working / "qwen35_token_features_after_amendment.parquet"
    feature_writer = FeatureWriter(feature_path, args.feature_batch_size)

    index_rows = iter_index_rows(index_path)
    logical_sha256 = hashlib.sha256()
    logical_offset = 0
    read_rows = 0
    processed_rows = 0
    source_counts: Counter[str] = Counter()
    ordinary_zero_loss_rows = 0
    amended_base_uids: set[str] = set()

    executor: ProcessPoolExecutor | None = None
    pending: deque[Future[list[RenderJobResult]]] = deque()
    jobs: list[tuple[int, dict[str, Any], str, dict[str, dict[str, Any]]]] = []
    if args.workers == 1:
        _WORKER_TOKENIZER = tokenizer
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(str(tokenizer_dir),),
        )

    def consume(results: list[RenderJobResult]) -> None:
        nonlocal processed_rows, ordinary_zero_loss_rows
        for result in results:
            row_operations = operations.get(result.global_row_number, {})
            feature_writer.append(result.base_feature)
            if result.base_feature["qwen_assistant_loss_tokens"] == 0:
                raise RuntimeError("amended base feature still has zero assistant loss")
            if result.base.amendment is not None:
                ordinary_zero_loss_rows += 1
                amended_base_uids.add(result.sample_uid)
            for arm_id, sink in sinks.items():
                sink.statistics.input_records += 1
                operation = row_operations.get(arm_id)
                if operation is not None:
                    if operation.get("sample_uid") != result.sample_uid:
                        raise ValueError("operation/sample UID drift")
                    if operation.get("source_key") != result.source_key:
                        raise ValueError("operation/source drift")
                if operation is not None and operation["action"] == "drop_sample":
                    sink.drop(operation)
                    continue
                rendered = (
                    result.edits[arm_id]
                    if operation is not None and operation["action"] == "drop_real_turn_spans"
                    else result.base
                )
                sink.write_document(
                    rendered,
                    global_row_number=result.global_row_number,
                    source_key=result.source_key,
                    sample_uid=result.sample_uid,
                    canonical_record_sha256=result.canonical_record_sha256,
                    operation=operation,
                )
            processed_rows += 1
            if args.progress_every and processed_rows % args.progress_every == 0:
                print(f"processed_rows={processed_rows:,}", flush=True)

    def submit(batch: list[tuple[int, dict[str, Any], str, dict[str, dict[str, Any]]]]) -> None:
        if not batch:
            return
        if executor is None:
            consume(render_batch(batch, args.max_seq_length))
            return
        pending.append(executor.submit(render_batch, batch, args.max_seq_length))
        if len(pending) >= args.workers * 2:
            consume(pending.popleft().result())

    try:
        with pa.input_stream(str(canonical_path), compression="zstd") as compressed:
            stream = io.BufferedReader(compressed)
            while True:
                if args.num_examples and read_rows >= args.num_examples:
                    break
                raw_line = stream.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    raise ValueError(f"canonical row {read_rows} has no trailing newline")
                record = json.loads(raw_line)
                index_row = next(index_rows)
                if index_row["global_row_number"] != read_rows:
                    raise ValueError(f"global row drift at {read_rows}")
                if index_row["logical_uncompressed_offset"] != logical_offset:
                    raise ValueError(f"logical offset drift at {read_rows}")
                if index_row["logical_uncompressed_length"] != len(raw_line):
                    raise ValueError(f"logical length drift at {read_rows}")
                if record.get("sample_uid") != index_row["sample_uid"]:
                    raise ValueError(f"sample UID drift at {read_rows}")
                if record.get("source_key") != index_row["source_key"]:
                    raise ValueError(f"source drift at {read_rows}")
                if raw_line[:-1] != canonical_bytes(record):
                    raise ValueError(f"canonical JSON normalization drift at {read_rows}")
                record_sha256 = hashlib.sha256(raw_line[:-1]).hexdigest()
                if record_sha256 != index_row["canonical_record_sha256"]:
                    raise ValueError(f"canonical record SHA-256 drift at {read_rows}")
                row_operations = operations.get(read_rows, {})
                edits = {
                    arm_id: operation
                    for arm_id, operation in row_operations.items()
                    if arm_id in selected_arms and operation["action"] == "drop_real_turn_spans"
                }
                if set(edits) - EDIT_ARMS:
                    raise ValueError(f"unexpected edited arm(s): {sorted(set(edits) - EDIT_ARMS)}")
                jobs.append((read_rows, record, record_sha256, edits))
                logical_sha256.update(raw_line)
                logical_offset += len(raw_line)
                source_counts[str(record["source_key"])] += 1
                read_rows += 1
                if len(jobs) == args.worker_batch_size:
                    submit(jobs)
                    jobs = []
            submit(jobs)
            while pending:
                consume(pending.popleft().result())
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if processed_rows != read_rows or feature_writer.rows != read_rows:
        raise RuntimeError("read/render/feature row counts differ")
    full_build = args.num_examples == 0
    if full_build and read_rows != expected_rows:
        raise ValueError(f"read {read_rows} rows, expected {expected_rows}")
    if full_build:
        try:
            extra_index = next(index_rows)
        except StopIteration:
            extra_index = None
        if extra_index is not None:
            raise ValueError("sample index contains rows after canonical JSONL")
        if logical_sha256.hexdigest() != canonical_pin["logical_jsonl_sha256"]:
            raise ValueError("canonical logical JSONL SHA-256 drift")

    feature_writer.close()
    expected_amended_uids = set(amendment["expected_sample_uids"])
    if full_build and amended_base_uids != expected_amended_uids:
        raise ValueError(
            "full-population amendment UIDs differ from the accepted manifest: "
            f"missing={sorted(expected_amended_uids - amended_base_uids)}, "
            f"unexpected={sorted(amended_base_uids - expected_amended_uids)}"
        )

    script_sha256 = sha256_file(Path(__file__).resolve())
    arm_manifests: dict[str, dict[str, Any]] = {}
    for arm_id, sink in sinks.items():
        parts = sink.close()
        if full_build:
            expected = expected_summary[arm_id]
            actual = sink.statistics
            checks = {
                "input_samples": actual.input_records,
                "targeted_samples": actual.targeted_records,
                "drop_samples": actual.dropped_records,
                "edited_samples": actual.edited_records,
                "output_samples": actual.output_documents,
                "removed_messages": actual.surviving_edit_removed_messages,
            }
            if checks != expected:
                raise ValueError(f"{arm_id} final operation accounting drift: {checks} != {expected}")
            if sink.amended_uids != expected_amended_uids:
                raise ValueError(f"{arm_id} amendment UID set drift")
        statistics_path = sink.output_dir / "statistics.json"
        write_json(statistics_path, sink.statistics.to_json())
        manifest = {
            "contract_version": NUMPY_CONTRACT_VERSION,
            "suite_id": membership_manifest["suite_id"],
            "arm_id": arm_id,
            "renderer": "qwen35_native_tools",
            "packing_semantics": "atomic_documents_best_fit_decreasing_no_cross_pack_or_part_splits",
            "enable_thinking": False,
            "max_seq_length": args.max_seq_length,
            "tokenizer": {
                "name_or_path": args.tokenizer_name_or_path,
                "revision": args.tokenizer_revision,
                "class": type(tokenizer).__name__,
                "vocab_size": tokenizer.vocab_size,
                "length": len(tokenizer),
                "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
                "directory": "tokenizer",
                "files": tokenizer_files,
            },
            "inputs": {
                "canonical_jsonl_zst_sha256": canonical_pin["sha256"],
                "canonical_logical_jsonl_sha256": canonical_pin["logical_jsonl_sha256"],
                "sample_index_sha256": index_pin["sha256"],
                "core_operations_sha256": operations_pin["sha256"],
                "core_operations_logical_jsonl_sha256": operations_pin["logical_jsonl_sha256"],
                "core_membership_manifest_sha256": sha256_file(membership_manifest_path),
                "frozen_design_manifest_sha256": sha256_file(design_path),
            },
            "renderer_amendment": {
                "amendment_id": amendment["amendment_id"],
                "manifest_sha256": amendment_sha256,
                "observed_sample_uids": sorted(sink.amended_uids),
                "ledger": "renderer_amendments.jsonl.gz",
                "ledger_sha256": sha256_file(sink.output_dir / "renderer_amendments.jsonl.gz"),
            },
            "documents_index": "documents.jsonl.gz",
            "documents_index_sha256": sha256_file(sink.output_dir / "documents.jsonl.gz"),
            "parts": parts,
            "statistics": "statistics.json",
            "statistics_sha256": sha256_file(statistics_path),
            "implementation_sha256": script_sha256,
            "runtime": runtime,
        }
        write_json(sink.output_dir / "manifest.json", manifest)
        arm_manifests[arm_id] = {
            "path": f"{arm_id}/manifest.json",
            "sha256": sha256_file(sink.output_dir / "manifest.json"),
            "output_documents": sink.statistics.output_documents,
            "total_tokens": sink.statistics.total_tokens,
            "trainable_tokens": sink.statistics.trainable_tokens,
        }

    root_manifest = {
        "artifact": "qwen35_frozen_core_C00_C11_numpy_build",
        "schema_version": 1,
        "suite_id": membership_manifest["suite_id"],
        "contract_version": NUMPY_CONTRACT_VERSION,
        "full_build": full_build,
        "input_rows": read_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "build_configuration": {
            "selected_arms": list(selected_arms),
            "workers": args.workers,
            "worker_batch_size": args.worker_batch_size,
            "feature_batch_size": args.feature_batch_size,
            "progress_every": args.progress_every,
            "max_part_size_gib": args.max_part_size_gib,
            "max_tokens_per_part": max_tokens_per_part,
            "token_dtype": token_dtype.name,
        },
        "arms": arm_manifests,
        "token_features_after_amendment": {
            "path": feature_path.name,
            "rows": feature_writer.rows,
            "sha256": sha256_file(feature_path),
            "size_bytes": feature_path.stat().st_size,
        },
        "renderer_amendment": {
            "amendment_id": amendment["amendment_id"],
            "manifest_sha256": amendment_sha256,
            "ordinary_zero_loss_rows": ordinary_zero_loss_rows,
            "observed_base_sample_uids": sorted(amended_base_uids),
            "remaining_zero_loss_rows": 0,
        },
        "inputs": {
            "canonical_jsonl_zst_sha256": canonical_pin["sha256"],
            "sample_index_sha256": index_pin["sha256"],
            "core_operations_sha256": operations_pin["sha256"],
            "core_membership_manifest_sha256": sha256_file(membership_manifest_path),
            "frozen_design_manifest_sha256": sha256_file(design_path),
        },
        "tokenizer": {
            "revision": args.tokenizer_revision,
            "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
            "files": tokenizer_files,
        },
        "implementation_sha256": script_sha256,
        "runtime": runtime,
    }
    write_json(working / "build_manifest.json", root_manifest)
    output_root = args.output_root.resolve()
    os.replace(working, output_root)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "input_rows": read_rows,
                "arms": list(selected_arms),
                "ordinary_zero_loss_rows": ordinary_zero_loss_rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
