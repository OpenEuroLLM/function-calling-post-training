#!/usr/bin/env python3
"""Convert canonical OpenAI-style FC JSONL into the packed NumPy contract.

This is intentionally separate from ``convert_sft_data_for_olmocore.py``.  It
uses Qwen3.5's native tool template and writes the same three core artifact
types used by the OLMo pipeline: raw token memmaps, Boolean label masks, and
gzip-compressed document boundaries.

Example:

    python scripts/data/convert_sft_data_for_qwen35.py \
        --input-jsonl tests/fixtures/qwen35_fc/fixture.jsonl \
        --output-dir /tmp/qwen35-fixture-numpy \
        --tokenizer-name-or-path Qwen/Qwen3.5-0.8B-Base \
        --tokenizer-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
        --max-seq-length 32768
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
from transformers import Qwen2TokenizerFast

from open_instruct.qwen35_data import (
    NUMPY_CONTRACT_VERSION,
    REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
    sha256_text,
    tokenize_qwen35_example,
    tokenize_qwen35_example_with_referenced_tool_pruning,
)

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_TOKENIZER_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


@dataclass
class ConversionStatistics:
    input_records: int = 0
    output_documents: int = 0
    skipped_zero_label_records: int = 0
    total_tokens: int = 0
    trainable_tokens: int = 0
    truncated_records: int = 0
    parsed_argument_strings: int = 0
    referenced_tool_pruned_records: int = 0
    referenced_tool_definitions_removed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument(
        "--max-part-size-gib",
        type=float,
        default=1.0,
        help="Maximum raw token file size; documents crossing a part are represented by one boundary per fragment.",
    )
    parser.add_argument(
        "--assistant-loss-manifest",
        type=Path,
        help="Optional JSONL keyed by sample_uid with a Boolean assistant_loss_mask list.",
    )
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--zero-label-policy", choices=("error", "skip"), default="error")
    parser.add_argument(
        "--renderer-amendment-manifest",
        type=Path,
        help=(
            "Accepted immutable amendment manifest. Supplying it explicitly "
            "enables referenced-tool pruning only for positive-untruncated/"
            "zero-truncated assistant-loss records."
        ),
    )
    parser.add_argument(
        "--enforce-amendment-expected-uids",
        action="store_true",
        help="Require observed amended UIDs to equal the accepted manifest's expected_sample_uids.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip_text(path: Path) -> tuple[TextIO, BinaryIO]:
    raw_handle = path.open("wb")
    gzip_handle = gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0)
    return io.TextIOWrapper(gzip_handle, encoding="utf-8", newline=""), raw_handle


def open_jsonl(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(paths: Sequence[Path]) -> Iterator[tuple[Path, int, bytes, dict[str, Any]]]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with open_jsonl(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
                yield path, line_number, line.encode("utf-8"), row


def sample_uid(row: Mapping[str, Any], *, path: Path, line_number: int) -> str:
    explicit = row.get("sample_uid")
    if isinstance(explicit, str) and explicit:
        return explicit
    dataset = row.get("dataset")
    sample_id = row.get("sample_id", row.get("id"))
    if isinstance(dataset, str) and dataset and sample_id is not None:
        return f"{dataset}:{sample_id}"
    raise ValueError(f"row at {path}:{line_number} needs sample_uid or the canonical (dataset, sample_id) identity")


def load_loss_manifest(path: Path | None) -> dict[str, list[bool]]:
    if path is None:
        return {}
    result: dict[str, list[bool]] = {}
    for source_path, line_number, _, row in iter_jsonl([path]):
        uid = row.get("sample_uid")
        mask = row.get("assistant_loss_mask")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"loss manifest row at {source_path}:{line_number} has no sample_uid")
        if uid in result:
            raise ValueError(f"duplicate loss manifest sample_uid {uid!r}")
        if not isinstance(mask, list) or not all(isinstance(value, bool) for value in mask):
            raise ValueError(f"loss manifest mask for {uid!r} must be a Boolean list")
        result[uid] = mask
    return result


def count_string_arguments(messages: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            count += isinstance(function.get("arguments"), str)
    return count


def choose_token_dtype(tokenizer: Qwen2TokenizerFast) -> np.dtype:
    vocabulary_bound = max(len(tokenizer), int(tokenizer.vocab_size)) - 1
    for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
        if vocabulary_bound <= np.iinfo(candidate).max:
            return np.dtype(candidate)
    raise ValueError(f"tokenizer vocabulary bound {vocabulary_bound} exceeds uint64")


class PartWriter:
    """Stream raw memmaps and boundary metadata without a dataset-sized list."""

    def __init__(self, output_dir: Path, token_dtype: np.dtype, max_tokens_per_part: int) -> None:
        if max_tokens_per_part <= 0:
            raise ValueError("max_tokens_per_part must be positive")
        self.output_dir = output_dir
        self.token_dtype = token_dtype
        self.max_tokens_per_part = max_tokens_per_part
        self.parts: list[dict[str, Any]] = []
        self.part_index = -1
        self.part_tokens = 0
        self._token_handle: BinaryIO | None = None
        self._mask_handle: BinaryIO | None = None
        self._boundary_handle: TextIO | None = None
        self._boundary_raw_handle: BinaryIO | None = None

    def _start_part(self) -> None:
        self.part_index += 1
        self.part_tokens = 0
        stem = f"part_{self.part_index:04d}"
        self._token_handle = (self.output_dir / f"token_ids_{stem}.npy.tmp").open("wb")
        self._mask_handle = (self.output_dir / f"labels_mask_{stem}.npy.tmp").open("wb")
        self._boundary_handle, self._boundary_raw_handle = deterministic_gzip_text(
            self.output_dir / f"token_ids_{stem}.csv.gz.tmp"
        )

    def _finish_part(self) -> None:
        if (
            self._token_handle is None
            or self._mask_handle is None
            or self._boundary_handle is None
            or self._boundary_raw_handle is None
        ):
            return
        token_handle = self._token_handle
        mask_handle = self._mask_handle
        boundary_handle = self._boundary_handle
        boundary_raw_handle = self._boundary_raw_handle
        token_tmp = Path(token_handle.name)
        mask_tmp = Path(mask_handle.name)
        boundary_tmp = Path(boundary_raw_handle.name)
        token_handle.flush()
        os.fsync(token_handle.fileno())
        token_handle.close()
        mask_handle.flush()
        os.fsync(mask_handle.fileno())
        mask_handle.close()
        boundary_handle.close()
        boundary_raw_handle.close()

        token_path = token_tmp.with_suffix("")
        mask_path = mask_tmp.with_suffix("")
        boundary_path = boundary_tmp.with_suffix("")
        os.replace(token_tmp, token_path)
        os.replace(mask_tmp, mask_path)
        os.replace(boundary_tmp, boundary_path)
        self.parts.append(
            {
                "token_ids": token_path.name,
                "labels_mask": mask_path.name,
                "boundaries": boundary_path.name,
                "token_dtype": self.token_dtype.name,
                "num_tokens": self.part_tokens,
                "token_ids_sha256": sha256_file(token_path),
                "labels_mask_sha256": sha256_file(mask_path),
                "boundaries_sha256": sha256_file(boundary_path),
            }
        )
        self._token_handle = None
        self._mask_handle = None
        self._boundary_handle = None
        self._boundary_raw_handle = None

    def write_document(self, input_ids: Sequence[int], labels_mask: Sequence[bool]) -> list[dict[str, int]]:
        if len(input_ids) != len(labels_mask) or not input_ids:
            raise ValueError("document token IDs and label mask must be non-empty and equal length")
        if len(input_ids) > self.max_tokens_per_part:
            raise ValueError(
                f"document has {len(input_ids)} tokens but part capacity is {self.max_tokens_per_part}; "
                "increase --max-part-size-gib so conversations are never split across parts"
            )
        if self._token_handle is None:
            self._start_part()
        elif self.part_tokens + len(input_ids) > self.max_tokens_per_part:
            self._finish_part()
            self._start_part()

        start = self.part_tokens
        end = start + len(input_ids)
        token_handle = self._token_handle
        mask_handle = self._mask_handle
        boundary_handle = self._boundary_handle
        if token_handle is None or mask_handle is None or boundary_handle is None:
            raise RuntimeError("part writer failed to initialize all output handles")
        np.asarray(input_ids, dtype=self.token_dtype).tofile(token_handle)
        np.asarray(labels_mask, dtype=np.bool_).tofile(mask_handle)
        boundary_handle.write(f"{start},{end}\n")
        self.part_tokens = end
        if self.part_tokens == self.max_tokens_per_part:
            self._finish_part()
        return [{"part": self.part_index, "start": start, "end": end}]

    def close(self) -> list[dict[str, Any]]:
        self._finish_part()
        return self.parts


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    working_dir = output_dir.parent / f".{output_dir.name}.incomplete"
    if working_dir.exists():
        if not overwrite:
            raise FileExistsError(f"incomplete output exists: {working_dir}; inspect or pass --overwrite")
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True)
    return working_dir


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_renderer_amendment(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if manifest.get("amendment_id") != REFERENCED_TOOL_PRUNING_AMENDMENT_ID:
        raise ValueError("renderer amendment ID does not match the implemented rule")
    if manifest.get("status") != "accepted_pre_outcome":
        raise ValueError("renderer amendment manifest is not accepted_pre_outcome")
    if manifest.get("max_seq_length") != 32768:
        raise ValueError("renderer amendment is defined only for max_seq_length=32768")
    expected = manifest.get("expected_sample_uids")
    if not isinstance(expected, list) or not all(isinstance(value, str) and len(value) == 64 for value in expected):
        raise ValueError("renderer amendment expected_sample_uids must be SHA-256 UIDs")
    if len(expected) != len(set(expected)):
        raise ValueError("renderer amendment expected_sample_uids contains duplicates")
    return manifest, sha256_file(resolved)


def main() -> None:
    args = parse_args()
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive")
    if args.max_part_size_gib <= 0:
        raise ValueError("--max-part-size-gib must be positive")
    if args.num_examples < 0:
        raise ValueError("--num-examples cannot be negative")
    if args.renderer_amendment_manifest and args.max_seq_length != 32768:
        raise ValueError("the accepted renderer amendment is restricted to 32,768 tokens")
    if args.enforce_amendment_expected_uids and not args.renderer_amendment_manifest:
        raise ValueError("--enforce-amendment-expected-uids requires --renderer-amendment-manifest")
    if args.renderer_amendment_manifest and args.zero_label_policy != "error":
        raise ValueError("the accepted renderer amendment forbids skipping zero-label records")

    amendment_manifest, amendment_manifest_sha256 = load_renderer_amendment(args.renderer_amendment_manifest)

    working_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.tokenizer_name_or_path, revision=args.tokenizer_revision)
    if not tokenizer.chat_template:
        raise ValueError("the pinned tokenizer has no native chat template")
    tokenizer_dir = working_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    token_dtype = choose_token_dtype(tokenizer)
    max_part_bytes = int(args.max_part_size_gib * 1024**3)
    max_tokens_per_part = max_part_bytes // token_dtype.itemsize
    writer = PartWriter(working_dir, token_dtype, max_tokens_per_part)
    loss_manifest = load_loss_manifest(args.assistant_loss_manifest)
    used_loss_manifest_uids: set[str] = set()
    seen_uids: set[str] = set()
    logical_input_hashes = {str(path.resolve()): hashlib.sha256() for path in args.input_jsonl}
    per_dataset = Counter()
    statistics = ConversionStatistics()
    document_index_handle, document_index_raw_handle = deterministic_gzip_text(working_dir / "documents.jsonl.gz")
    amendment_handle, amendment_raw_handle = deterministic_gzip_text(working_dir / "renderer_amendments.jsonl.gz")
    amended_uids: set[str] = set()

    try:
        for path, line_number, raw_line, row in iter_jsonl(args.input_jsonl):
            if args.num_examples and statistics.input_records >= args.num_examples:
                break
            statistics.input_records += 1
            logical_input_hashes[str(path.resolve())].update(raw_line)
            uid = sample_uid(row, path=path, line_number=line_number)
            if uid in seen_uids:
                raise ValueError(f"duplicate canonical sample UID {uid!r}")
            seen_uids.add(uid)
            messages = row.get("messages")
            tools = row.get("tools")
            if not isinstance(messages, list) or not isinstance(tools, list):
                raise ValueError(f"canonical row {uid!r} must contain list-valued messages and tools")
            assistant_loss_mask = loss_manifest.get(uid)
            if assistant_loss_mask is not None:
                used_loss_manifest_uids.add(uid)
            statistics.parsed_argument_strings += count_string_arguments(messages)
            if amendment_manifest is None:
                tokenized = tokenize_qwen35_example(
                    tokenizer,
                    messages,
                    tools,
                    max_seq_length=args.max_seq_length,
                    assistant_loss_mask=assistant_loss_mask,
                    enable_thinking=False,
                )
                amended = None
                ordinary_before = tokenized.num_tokens_before_truncation
                ordinary_after = len(tokenized.input_ids)
            else:
                render = tokenize_qwen35_example_with_referenced_tool_pruning(
                    tokenizer,
                    messages,
                    tools,
                    max_seq_length=args.max_seq_length,
                    assistant_loss_mask=assistant_loss_mask,
                )
                tokenized = render.tokenized
                amended = render.pruning
                ordinary_before = render.ordinary_num_tokens_before_truncation
                ordinary_after = render.ordinary_num_tokens_after_truncation
            statistics.truncated_records += ordinary_before > ordinary_after
            if amended is not None:
                if uid in amended_uids:
                    raise RuntimeError(f"duplicate renderer amendment ledger row for {uid}")
                amended_uids.add(uid)
                statistics.referenced_tool_pruned_records += 1
                statistics.referenced_tool_definitions_removed += len(amended.removed_tools)
                amendment_handle.write(
                    json.dumps(
                        {
                            "amendment_id": REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
                            "sample_uid": uid,
                            "dataset": row.get("dataset"),
                            "sample_id": row.get("sample_id", row.get("id")),
                            "ordinary_num_tokens_before_truncation": ordinary_before,
                            "ordinary_num_tokens_after_truncation": ordinary_after,
                            "ordinary_trainable_tokens_after_truncation": 0,
                            "amended_num_tokens": len(tokenized.input_ids),
                            "amended_trainable_tokens": tokenized.trainable_tokens,
                            "tools_before": len(tools),
                            "tools_after": len(amended.retained_tools),
                            "removed_tools": [asdict(value) for value in amended.removed_tools],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            if tokenized.trainable_tokens == 0:
                if args.zero_label_policy == "error":
                    raise ValueError(f"canonical row {uid!r} has no trainable tokens after truncation")
                statistics.skipped_zero_label_records += 1
                continue

            segments = writer.write_document(tokenized.input_ids, tokenized.labels_mask)
            dataset = row.get("dataset", "unknown")
            per_dataset[str(dataset)] += 1
            statistics.output_documents += 1
            statistics.total_tokens += len(tokenized.input_ids)
            statistics.trainable_tokens += tokenized.trainable_tokens
            document_index_handle.write(
                json.dumps(
                    {
                        "sample_uid": uid,
                        "dataset": dataset,
                        "sample_id": row.get("sample_id", row.get("id")),
                        "num_tokens": len(tokenized.input_ids),
                        "num_trainable_tokens": tokenized.trainable_tokens,
                        "renderer_amended": amended is not None,
                        "segments": segments,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    finally:
        document_index_handle.close()
        document_index_raw_handle.close()
        amendment_handle.close()
        amendment_raw_handle.close()

    if amendment_manifest is not None and args.enforce_amendment_expected_uids:
        expected_uids = set(amendment_manifest["expected_sample_uids"])
        if amended_uids != expected_uids:
            raise ValueError(
                "observed renderer-amendment UIDs differ from the accepted manifest: "
                f"missing={sorted(expected_uids - amended_uids)}, "
                f"unexpected={sorted(amended_uids - expected_uids)}"
            )

    unused_loss_uids = set(loss_manifest) - used_loss_manifest_uids
    if unused_loss_uids:
        preview = sorted(unused_loss_uids)[:10]
        raise ValueError(f"assistant-loss manifest has {len(unused_loss_uids)} unused UIDs; first: {preview}")
    parts = writer.close()
    if not parts:
        raise ValueError("conversion produced no output parts")

    write_json(
        working_dir / "statistics.json",
        {
            **asdict(statistics),
            "trainable_token_fraction": statistics.trainable_tokens / statistics.total_tokens,
            "per_dataset_documents": dict(sorted(per_dataset.items())),
        },
    )
    manifest = {
        "contract_version": NUMPY_CONTRACT_VERSION,
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
            "chat_template_sha256": sha256_text(tokenizer.chat_template),
            "directory": "tokenizer",
        },
        "input_jsonl": [
            {
                "path": str(path.resolve()),
                "logical_lines_sha256": logical_input_hashes[str(path.resolve())].hexdigest(),
            }
            for path in args.input_jsonl
        ],
        "assistant_loss_manifest": str(args.assistant_loss_manifest.resolve())
        if args.assistant_loss_manifest
        else None,
        "renderer_amendment": {
            "enabled": amendment_manifest is not None,
            "amendment_id": amendment_manifest.get("amendment_id") if amendment_manifest else None,
            "manifest_path": str(args.renderer_amendment_manifest.resolve())
            if args.renderer_amendment_manifest
            else None,
            "manifest_sha256": amendment_manifest_sha256,
            "observed_sample_uids": sorted(amended_uids),
            "ledger": "renderer_amendments.jsonl.gz",
            "ledger_sha256": sha256_file(working_dir / "renderer_amendments.jsonl.gz"),
        },
        "documents_index": "documents.jsonl.gz",
        "documents_index_sha256": sha256_file(working_dir / "documents.jsonl.gz"),
        "parts": parts,
        "statistics": "statistics.json",
    }
    write_json(working_dir / "manifest.json", manifest)

    output_dir = args.output_dir.resolve()
    os.replace(working_dir, output_dir)
    print(
        f"Wrote {statistics.output_documents:,} documents, {statistics.total_tokens:,} tokens "
        f"({statistics.trainable_tokens:,} trainable) to {output_dir}"
    )


if __name__ == "__main__":
    main()
