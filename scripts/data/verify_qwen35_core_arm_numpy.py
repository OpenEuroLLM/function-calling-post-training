#!/usr/bin/env python3
"""Independently verify and compare two full C00-C11 Qwen NumPy builds.

The verifier does not use the production builder.  It rehashes both artifact
trees, reconstructs every arm's expected membership from the frozen operation
ledger and canonical sample index, checks every document boundary and loss
count against the raw masks, invokes the trainer's deterministic packer, and
recomputes final matching tolerances.  It fails closed on unaccounted files,
zero-loss documents, membership drift, packing loss, or any byte difference.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import struct
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from open_instruct.qwen35_data import (
    NUMPY_CONTRACT_VERSION,
    REFERENCED_TOOL_PRUNING_AMENDMENT_ID,
    Qwen35NumpyPackedDataset,
)

ARM_ORDER = tuple(f"C{index:02d}" for index in range(12))
MATCHED_ARMS = ("C05", "C06", "C07")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--comparison-build-root", type=Path, required=True)
    parser.add_argument("--canonical-jsonl-zst", type=Path, required=True)
    parser.add_argument("--contract-manifest", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--core-operations", type=Path, required=True)
    parser.add_argument("--core-membership-manifest", type=Path, required=True)
    parser.add_argument("--renderer-amendment-manifest", type=Path, required=True)
    parser.add_argument("--frozen-design-manifest", type=Path, required=True)
    parser.add_argument("--pre-amendment-token-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
                raise ValueError(f"JSONL row in {path} has no trailing newline")
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row in {path} is not an object")
            yield value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def tree_hashes(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    digest = hashlib.sha256()
    for relative, facts in files.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(facts["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(facts["sha256"].encode("ascii"))
        digest.update(b"\n")
    return files, {
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files.values()),
        "tree_sha256": digest.hexdigest(),
    }


def require_file_hash(file_hashes: Mapping[str, Mapping[str, Any]], relative_path: str, expected_sha256: str) -> None:
    facts = file_hashes.get(relative_path)
    if facts is None:
        raise FileNotFoundError(f"manifest references missing file {relative_path}")
    if facts["sha256"] != expected_sha256:
        raise ValueError(f"SHA-256 drift for {relative_path}")


def load_sample_index(path: Path) -> dict[str, list[Any]]:
    columns = ["global_row_number", "source_key", "sample_uid"]
    table = pq.read_table(path, columns=columns)
    rows = table["global_row_number"].to_numpy(zero_copy_only=False)
    if not np.array_equal(rows, np.arange(len(rows), dtype=rows.dtype)):
        raise ValueError("sample index global rows are not exactly 0..N-1")
    return {
        "global_row_number": rows.tolist(),
        "source_key": table["source_key"].to_pylist(),
        "sample_uid": table["sample_uid"].to_pylist(),
    }


def load_operations(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {arm_id: {} for arm_id in ARM_ORDER}
    previous_key: tuple[int, int] | None = None
    for operation in iter_jsonl_zst(path):
        arm_id = operation.get("arm_id")
        row_number = operation.get("global_row_number")
        if arm_id not in result or not isinstance(row_number, int):
            raise ValueError("invalid arm/global row in operation ledger")
        key = (ARM_ORDER.index(arm_id), row_number)
        if previous_key is not None and key <= previous_key:
            raise ValueError("operation ledger order drift")
        previous_key = key
        if row_number in result[arm_id]:
            raise ValueError(f"duplicate {arm_id} operation for row {row_number}")
        result[arm_id][row_number] = operation
    return result


def load_boundaries(arm_root: Path, manifest: Mapping[str, Any]) -> tuple[list[tuple[int, int, int]], list[np.memmap]]:
    boundaries: list[tuple[int, int, int]] = []
    masks: list[np.memmap] = []
    for part_index, part in enumerate(manifest["parts"]):
        num_tokens = int(part["num_tokens"])
        mask = np.memmap(arm_root / part["labels_mask"], mode="r", dtype=np.bool_, shape=(num_tokens,))
        masks.append(mask)
        previous_end = 0
        with gzip.open(arm_root / part["boundaries"], "rt") as handle:
            for line_number, line in enumerate(handle, start=1):
                values = line.rstrip("\n").split(",")
                if len(values) != 2:
                    raise ValueError(f"invalid boundary at part {part_index}:{line_number}")
                start, end = (int(value) for value in values)
                if start != previous_end or end <= start or end > num_tokens:
                    raise ValueError(f"boundary coverage drift at part {part_index}:{line_number}")
                boundaries.append((part_index, start, end))
                previous_end = end
        if previous_end != num_tokens:
            raise ValueError(f"part {part_index} boundaries do not cover every token")
    return boundaries, masks


def pack_digest(dataset: Qwen35NumpyPackedDataset) -> str:
    digest = hashlib.sha256()
    for pack in dataset.packs:
        digest.update(struct.pack("<I", len(pack)))
        for piece in pack:
            digest.update(struct.pack("<IQQ", piece.part_index, piece.start, piece.end))
    return digest.hexdigest()


def verify_documents(
    arm_id: str,
    arm_root: Path,
    manifest: Mapping[str, Any],
    statistics: Mapping[str, Any],
    index: Mapping[str, list[Any]],
    arm_operations: Mapping[int, Mapping[str, Any]],
    expected_amended_uids: set[str],
) -> dict[str, Any]:
    boundaries, masks = load_boundaries(arm_root, manifest)
    drop_rows = {
        row_number for row_number, operation in arm_operations.items() if operation["action"] == "drop_sample"
    }
    expected_output = len(index["global_row_number"]) - len(drop_rows)
    if len(boundaries) != expected_output:
        raise ValueError(f"{arm_id} boundary/document count drift")

    document_count = 0
    next_expected_row = 0
    total_tokens = 0
    total_trainable = 0
    first_token_trainable = 0
    amended_uids: set[str] = set()
    per_source_documents: Counter[str] = Counter()
    per_source_tokens: Counter[str] = Counter()
    per_source_trainable: Counter[str] = Counter()
    documents_path = arm_root / manifest["documents_index"]
    with gzip.open(documents_path, "rt") as handle:
        for document_count, line in enumerate(handle, start=1):
            document = json.loads(line)
            while next_expected_row in drop_rows:
                next_expected_row += 1
            row_number = document.get("global_row_number")
            if row_number != next_expected_row:
                raise ValueError(
                    f"{arm_id} document {document_count - 1} row {row_number} != expected {next_expected_row}"
                )
            source_key = index["source_key"][row_number]
            sample_uid = index["sample_uid"][row_number]
            if document.get("source_key") != source_key or document.get("sample_uid") != sample_uid:
                raise ValueError(f"{arm_id} identity drift at global row {row_number}")
            part_index, start, end = boundaries[document_count - 1]
            if document.get("segments") != [{"part": part_index, "start": start, "end": end}]:
                raise ValueError(f"{arm_id} segment drift at global row {row_number}")
            mask = masks[part_index][start:end]
            actual_trainable = int(np.count_nonzero(mask))
            if actual_trainable <= 0:
                raise ValueError(f"{arm_id} emitted zero-loss document at row {row_number}")
            if document.get("num_tokens") != end - start:
                raise ValueError(f"{arm_id} token-count drift at row {row_number}")
            if document.get("num_trainable_tokens") != actual_trainable:
                raise ValueError(f"{arm_id} loss-count drift at row {row_number}")
            operation = arm_operations.get(row_number)
            if operation is None:
                expected_action = "keep"
                expected_operation_id = None
            elif operation["action"] == "drop_sample":
                raise ValueError(f"{arm_id} retained a frozen drop row {row_number}")
            else:
                expected_action = "drop_real_turn_spans"
                expected_operation_id = operation["operation_id"]
            if document.get("operation_action") != expected_action:
                raise ValueError(f"{arm_id} action drift at row {row_number}")
            if document.get("operation_id") != expected_operation_id:
                raise ValueError(f"{arm_id} operation-ID drift at row {row_number}")
            if document.get("renderer_amended"):
                amended_uids.add(sample_uid)
            total_tokens += end - start
            total_trainable += actual_trainable
            first_token_trainable += bool(mask[0])
            per_source_documents[source_key] += 1
            per_source_tokens[source_key] += end - start
            per_source_trainable[source_key] += actual_trainable
            next_expected_row += 1
    while next_expected_row in drop_rows:
        next_expected_row += 1
    if document_count != expected_output or next_expected_row != len(index["global_row_number"]):
        raise ValueError(f"{arm_id} document ledger ended early")
    if amended_uids != expected_amended_uids:
        raise ValueError(f"{arm_id} amended document UID set drift")

    amendment_rows: list[dict[str, Any]] = []
    with gzip.open(arm_root / manifest["renderer_amendment"]["ledger"], "rt") as handle:
        for line in handle:
            amendment_rows.append(json.loads(line))
    if {row["sample_uid"] for row in amendment_rows} != expected_amended_uids:
        raise ValueError(f"{arm_id} amendment ledger UID set drift")
    for row in amendment_rows:
        if row.get("amendment_id") != REFERENCED_TOOL_PRUNING_AMENDMENT_ID:
            raise ValueError(f"{arm_id} amendment ID drift")
        if not row.get("removed_tools") or row.get("amended_trainable_tokens", 0) <= 0:
            raise ValueError(f"{arm_id} incomplete amendment ledger row")

    checks = {
        "output_documents": document_count,
        "total_tokens": total_tokens,
        "trainable_tokens": total_trainable,
        "per_source_documents": dict(sorted(per_source_documents.items())),
        "per_source_tokens": dict(sorted(per_source_tokens.items())),
        "per_source_trainable_tokens": dict(sorted(per_source_trainable.items())),
    }
    for key, value in checks.items():
        if statistics.get(key) != value:
            raise ValueError(f"{arm_id} statistics drift for {key}")
    return {
        **checks,
        "first_token_trainable": first_token_trainable,
        "drop_rows": len(drop_rows),
        "edit_rows": sum(operation["action"] == "drop_real_turn_spans" for operation in arm_operations.values()),
        "targeted_rows": len(arm_operations),
        "edit_removed_messages": sum(
            int(operation["removed_messages"])
            for operation in arm_operations.values()
            if operation["action"] == "drop_real_turn_spans"
        ),
        "amended_rows": len(amendment_rows),
    }


def verify_feature_projection(
    build_root: Path,
    root_manifest: Mapping[str, Any],
    pre_amendment_path: Path,
    index: Mapping[str, list[Any]],
    expected_rows: set[int],
) -> dict[str, Any]:
    feature_path = build_root / root_manifest["token_features_after_amendment"]["path"]
    columns = [
        "global_row_number",
        "sample_uid",
        "qwen_total_tokens",
        "qwen_assistant_loss_tokens",
        "qwen_fc_assistant_loss_tokens",
    ]
    before = pq.read_table(pre_amendment_path, columns=columns)
    after = pq.read_table(feature_path, columns=columns)
    if len(before) != len(after) or len(after) != len(index["global_row_number"]):
        raise ValueError("pre/post-amendment projection row-count drift")
    for identity in ("global_row_number", "sample_uid"):
        if not before[identity].equals(after[identity]):
            raise ValueError(f"pre/post-amendment {identity} drift")
    changed = np.zeros(len(after), dtype=np.bool_)
    deltas: dict[str, int] = {}
    for column in columns[2:]:
        before_values = before[column].to_numpy(zero_copy_only=False).astype(np.int64)
        after_values = after[column].to_numpy(zero_copy_only=False).astype(np.int64)
        changed |= before_values != after_values
        deltas[column] = int(after_values.sum() - before_values.sum())
        if column == "qwen_assistant_loss_tokens" and np.any(after_values <= 0):
            raise ValueError("post-amendment projection retains zero-loss rows")
    observed_rows = set(np.flatnonzero(changed).tolist())
    if observed_rows != expected_rows:
        raise ValueError(
            "projection changes differ from amendment rows: "
            f"missing={sorted(expected_rows - observed_rows)}, "
            f"unexpected={sorted(observed_rows - expected_rows)}"
        )
    return {
        "rows": len(after),
        "changed_global_rows": sorted(observed_rows),
        "aggregate_deltas_after_minus_before": deltas,
        "remaining_zero_loss_rows": 0,
    }


def relative_error(observed: int, target: int) -> float:
    if target == 0:
        return 0.0 if observed == 0 else float("inf")
    return abs(observed - target) / abs(target)


def verify_matching(arm_reports: Mapping[str, Mapping[str, Any]], design: Mapping[str, Any]) -> dict[str, Any]:
    balance = design["balance"]
    total_tolerance = float(balance["per_source_total_token_relative_tolerance"])
    trainable_tolerance = float(balance["global_assistant_loss_token_relative_tolerance"])
    per_source_trainable_tolerance = float(balance["per_source_trainable_token_relative_tolerance"])
    effective_tolerance = float(design["training_controls"]["effective_assistant_loss_tolerance"])
    raw = arm_reports["C00"]
    target = arm_reports["C01"]
    target_total_drop = raw["total_tokens"] - target["total_tokens"]
    target_trainable_drop = raw["trainable_tokens"] - target["trainable_tokens"]
    target_effective_drop = (
        raw["packing_accounting"]["effective_trainable_tokens"]
        - target["packing_accounting"]["effective_trainable_tokens"]
    )
    reports: dict[str, Any] = {}
    for arm_id in MATCHED_ARMS:
        arm = arm_reports[arm_id]
        total_drop = raw["total_tokens"] - arm["total_tokens"]
        trainable_drop = raw["trainable_tokens"] - arm["trainable_tokens"]
        effective_drop = (
            raw["packing_accounting"]["effective_trainable_tokens"]
            - arm["packing_accounting"]["effective_trainable_tokens"]
        )
        per_source_total: dict[str, float] = {}
        per_source_trainable: dict[str, float] = {}
        for source_key in raw["per_source_documents"]:
            target_source_total = raw["per_source_tokens"][source_key] - target["per_source_tokens"][source_key]
            arm_source_total = raw["per_source_tokens"][source_key] - arm["per_source_tokens"][source_key]
            target_source_trainable = (
                raw["per_source_trainable_tokens"][source_key] - target["per_source_trainable_tokens"][source_key]
            )
            arm_source_trainable = (
                raw["per_source_trainable_tokens"][source_key] - arm["per_source_trainable_tokens"][source_key]
            )
            per_source_total[source_key] = relative_error(arm_source_total, target_source_total)
            per_source_trainable[source_key] = relative_error(arm_source_trainable, target_source_trainable)
        result = {
            "sample_count_matches_C01": arm["output_documents"] == target["output_documents"],
            "total_token_depletion": total_drop,
            "target_total_token_depletion": target_total_drop,
            "total_token_relative_error": relative_error(total_drop, target_total_drop),
            "trainable_token_depletion": trainable_drop,
            "target_trainable_token_depletion": target_trainable_drop,
            "trainable_token_relative_error": relative_error(trainable_drop, target_trainable_drop),
            "effective_trainable_token_depletion": effective_drop,
            "target_effective_trainable_token_depletion": target_effective_drop,
            "effective_trainable_token_relative_error": relative_error(effective_drop, target_effective_drop),
            "per_source_total_token_relative_error": per_source_total,
            "per_source_trainable_token_relative_error": per_source_trainable,
        }
        if not result["sample_count_matches_C01"]:
            raise ValueError(f"{arm_id} sample count no longer matches C01")
        if result["total_token_relative_error"] > total_tolerance:
            raise ValueError(f"{arm_id} total-token matching tolerance failed")
        if result["trainable_token_relative_error"] > trainable_tolerance:
            raise ValueError(f"{arm_id} trainable-token matching tolerance failed")
        if result["effective_trainable_token_relative_error"] > effective_tolerance:
            raise ValueError(f"{arm_id} effective-loss matching tolerance failed")
        if max(per_source_total.values(), default=0.0) > total_tolerance:
            raise ValueError(f"{arm_id} per-source total-token tolerance failed")
        if max(per_source_trainable.values(), default=0.0) > per_source_trainable_tolerance:
            raise ValueError(f"{arm_id} per-source trainable-token tolerance failed")
        reports[arm_id] = result
    return {
        "tolerances": {
            "per_source_total_token_relative": total_tolerance,
            "per_source_trainable_token_relative": per_source_trainable_tolerance,
            "global_trainable_token_relative": trainable_tolerance,
            "effective_trainable_token_relative": effective_tolerance,
        },
        "arms": reports,
    }


def verify_build(
    build_root: Path,
    file_hashes: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, list[Any]],
    operations: Mapping[str, Mapping[int, Mapping[str, Any]]],
    operations_sha256: str,
    amendment: Mapping[str, Any],
    design: Mapping[str, Any],
    expected_input_pins: Mapping[str, str],
    membership_summary: Mapping[str, Mapping[str, Any]],
    pre_amendment_features: Path,
    sequence_length: int,
) -> dict[str, Any]:
    root_manifest = read_json(build_root / "build_manifest.json")
    if root_manifest.get("full_build") is not True:
        raise ValueError("production verification requires full_build=true")
    if root_manifest.get("contract_version") != NUMPY_CONTRACT_VERSION:
        raise ValueError("root NumPy contract version drift")
    if root_manifest.get("input_rows") != len(index["global_row_number"]):
        raise ValueError("root input-row count drift")
    if set(root_manifest.get("arms", {})) != set(ARM_ORDER):
        raise ValueError("root manifest does not contain exactly C00-C11")
    if root_manifest["inputs"]["core_operations_sha256"] != operations_sha256:
        raise ValueError("root operation-ledger pin drift")
    for key, expected_sha256 in expected_input_pins.items():
        if root_manifest["inputs"].get(key) != expected_sha256:
            raise ValueError(f"root input pin drift for {key}")
    configuration = root_manifest.get("build_configuration", {})
    if configuration.get("selected_arms") != list(ARM_ORDER):
        raise ValueError("root build did not select exactly C00-C11 in frozen order")
    for key in ("workers", "worker_batch_size", "feature_batch_size", "max_tokens_per_part"):
        if not isinstance(configuration.get(key), int) or configuration[key] <= 0:
            raise ValueError(f"root build configuration lacks positive {key}")
    if configuration.get("max_part_size_gib") != 4.0 or configuration.get("token_dtype") != "uint32":
        raise ValueError("root production part-size or token-dtype configuration drift")

    expected_amended_uids = set(amendment["expected_sample_uids"])
    expected_amended_rows = set(amendment["expected_global_rows"])
    if root_manifest["renderer_amendment"]["amendment_id"] != REFERENCED_TOOL_PRUNING_AMENDMENT_ID:
        raise ValueError("root renderer-amendment ID drift")
    if set(root_manifest["renderer_amendment"]["observed_base_sample_uids"]) != expected_amended_uids:
        raise ValueError("root amended UID set drift")
    if root_manifest["renderer_amendment"]["remaining_zero_loss_rows"] != 0:
        raise ValueError("root manifest reports unresolved zero-loss rows")
    for row_number in expected_amended_rows:
        if any(row_number in operations[arm_id] for arm_id in ARM_ORDER):
            raise ValueError("renderer-amended row is unexpectedly targeted by a core arm")

    accounted_files = {"build_manifest.json", root_manifest["token_features_after_amendment"]["path"]}
    require_file_hash(
        file_hashes,
        root_manifest["token_features_after_amendment"]["path"],
        root_manifest["token_features_after_amendment"]["sha256"],
    )
    for tokenizer_file in root_manifest["tokenizer"]["files"]:
        relative = f"tokenizer/{tokenizer_file['path']}"
        accounted_files.add(relative)
        require_file_hash(file_hashes, relative, tokenizer_file["sha256"])

    arm_reports: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        root_arm = root_manifest["arms"][arm_id]
        manifest_relative = root_arm["path"]
        accounted_files.add(manifest_relative)
        require_file_hash(file_hashes, manifest_relative, root_arm["sha256"])
        arm_root = build_root / arm_id
        manifest = read_json(build_root / manifest_relative)
        if manifest.get("arm_id") != arm_id or manifest.get("max_seq_length") != sequence_length:
            raise ValueError(f"{arm_id} manifest identity/sequence-length drift")
        if manifest.get("contract_version") != NUMPY_CONTRACT_VERSION:
            raise ValueError(f"{arm_id} contract version drift")
        for key, expected_sha256 in expected_input_pins.items():
            if key in manifest["inputs"] and manifest["inputs"].get(key) != expected_sha256:
                raise ValueError(f"{arm_id} input pin drift for {key}")
        if set(manifest["renderer_amendment"]["observed_sample_uids"]) != expected_amended_uids:
            raise ValueError(f"{arm_id} manifest amendment UID drift")

        small_files = (
            (manifest["statistics"], manifest["statistics_sha256"]),
            (manifest["documents_index"], manifest["documents_index_sha256"]),
            (manifest["renderer_amendment"]["ledger"], manifest["renderer_amendment"]["ledger_sha256"]),
        )
        for relative_in_arm, expected_sha256 in small_files:
            relative = f"{arm_id}/{relative_in_arm}"
            accounted_files.add(relative)
            require_file_hash(file_hashes, relative, expected_sha256)
        for tokenizer_file in manifest["tokenizer"]["files"]:
            relative = f"{arm_id}/tokenizer/{tokenizer_file['path']}"
            accounted_files.add(relative)
            require_file_hash(file_hashes, relative, tokenizer_file["sha256"])
        for part in manifest["parts"]:
            for path_key, sha_key in (
                ("token_ids", "token_ids_sha256"),
                ("labels_mask", "labels_mask_sha256"),
                ("boundaries", "boundaries_sha256"),
            ):
                relative = f"{arm_id}/{part[path_key]}"
                accounted_files.add(relative)
                require_file_hash(file_hashes, relative, part[sha_key])

        statistics = read_json(arm_root / manifest["statistics"])
        document_report = verify_documents(
            arm_id, arm_root, manifest, statistics, index, operations[arm_id], expected_amended_uids
        )
        expected_membership = membership_summary[arm_id]
        observed_membership = {
            "input_samples": len(index["global_row_number"]),
            "targeted_samples": document_report["targeted_rows"],
            "drop_samples": document_report["drop_rows"],
            "edited_samples": document_report["edit_rows"],
            "output_samples": document_report["output_documents"],
            "removed_messages": document_report["edit_removed_messages"],
        }
        if observed_membership != expected_membership:
            raise ValueError(f"{arm_id} frozen membership-summary accounting drift")
        dataset = Qwen35NumpyPackedDataset(
            arm_root, sequence_length=sequence_length, drop_last=False, verify_hashes=False
        )
        packing = dataset.accounting()
        if packing["raw_tokens"] != document_report["total_tokens"]:
            raise ValueError(f"{arm_id} loader raw-token accounting drift")
        if packing["packed_real_tokens"] != packing["raw_tokens"] or packing["dropped_tokens"] != 0:
            raise ValueError(f"{arm_id} loader dropped data with drop_last=false")
        if packing["packed_trainable_tokens_before_boundary_mask"] != document_report["trainable_tokens"]:
            raise ValueError(f"{arm_id} loader trainable-token accounting drift")
        if packing["boundary_masked_trainable_tokens"] != document_report["first_token_trainable"]:
            raise ValueError(f"{arm_id} boundary-mask accounting drift")
        packed_documents = sum(len(pack) for pack in dataset.packs)
        if packed_documents != document_report["output_documents"]:
            raise ValueError(f"{arm_id} pack membership lost documents")
        arm_reports[arm_id] = {
            **document_report,
            "num_packs": len(dataset),
            "pack_membership_sha256": pack_digest(dataset),
            "packing_accounting": packing,
        }

    unaccounted = set(file_hashes) - accounted_files
    missing = accounted_files - set(file_hashes)
    if unaccounted or missing:
        raise ValueError(
            f"artifact file accounting drift: unaccounted={sorted(unaccounted)}, missing={sorted(missing)}"
        )
    feature_report = verify_feature_projection(
        build_root, root_manifest, pre_amendment_features, index, expected_amended_rows
    )
    matching_report = verify_matching(arm_reports, design)
    return {
        "root_manifest_sha256": file_hashes["build_manifest.json"]["sha256"],
        "feature_projection": feature_report,
        "arms": arm_reports,
        "matching": matching_report,
        "all_files_accounted": True,
    }


def main() -> int:
    args = parse_args()
    if args.sequence_length != 32768:
        raise ValueError("the accepted renderer amendment is restricted to 32768")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    build_root = args.build_root.resolve()
    comparison_root = args.comparison_build_root.resolve()
    canonical_path = args.canonical_jsonl_zst.resolve()
    contract_path = args.contract_manifest.resolve()
    index_path = args.sample_index.resolve()
    operations_path = args.core_operations.resolve()
    membership_path = args.core_membership_manifest.resolve()
    amendment_path = args.renderer_amendment_manifest.resolve()
    design_path = args.frozen_design_manifest.resolve()
    pre_amendment_path = args.pre_amendment_token_features.resolve()
    for path in (
        build_root,
        comparison_root,
        canonical_path,
        contract_path,
        index_path,
        operations_path,
        membership_path,
        amendment_path,
        design_path,
        pre_amendment_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    run1_hashes, run1_tree = tree_hashes(build_root)
    run2_hashes, run2_tree = tree_hashes(comparison_root)
    if run1_hashes != run2_hashes:
        differing = sorted(
            relative
            for relative in set(run1_hashes) | set(run2_hashes)
            if run1_hashes.get(relative) != run2_hashes.get(relative)
        )
        raise ValueError(f"two builds are not byte-identical; differing files: {differing}")

    index = load_sample_index(index_path)
    operations = load_operations(operations_path)
    contract = read_json(contract_path)
    membership = read_json(membership_path)
    amendment = read_json(amendment_path)
    design = read_json(design_path)
    if membership.get("suite_id") != design.get("suite_id"):
        raise ValueError("membership and frozen-design suite IDs differ")
    canonical_pin = contract["artifacts"]["canonical_samples"]
    index_pin = contract["artifacts"]["sample_index"]
    if sha256_file(canonical_path) != canonical_pin["sha256"]:
        raise ValueError("canonical compressed SHA-256 drift")
    if sha256_file(index_path) != index_pin["sha256"]:
        raise ValueError("sample-index SHA-256 drift")
    operations_pin = membership["unified_operations"]
    if sha256_file(operations_path) != operations_pin["sha256"]:
        raise ValueError("membership operation-ledger SHA-256 drift")
    if operations_pin["operations"] != sum(len(by_row) for by_row in operations.values()):
        raise ValueError("membership operation count drift")
    if membership["inputs"]["frozen_design_manifest"]["sha256"] != sha256_file(design_path):
        raise ValueError("membership frozen-design SHA-256 drift")
    summary_path = membership_path.parent / membership["summary"]["path"]
    if sha256_file(summary_path) != membership["summary"]["sha256"]:
        raise ValueError("membership summary SHA-256 drift")
    membership_summary = read_json(summary_path)
    if amendment.get("status") != "accepted_pre_outcome":
        raise ValueError("renderer amendment is not accepted_pre_outcome")
    if amendment.get("amendment_id") != REFERENCED_TOOL_PRUNING_AMENDMENT_ID:
        raise ValueError("renderer amendment ID drift")
    verification = verify_build(
        build_root,
        run1_hashes,
        index,
        operations,
        sha256_file(operations_path),
        amendment,
        design,
        {
            "canonical_jsonl_zst_sha256": canonical_pin["sha256"],
            "sample_index_sha256": index_pin["sha256"],
            "core_operations_sha256": operations_pin["sha256"],
            "core_membership_manifest_sha256": sha256_file(membership_path),
            "frozen_design_manifest_sha256": sha256_file(design_path),
        },
        membership_summary,
        pre_amendment_path,
        args.sequence_length,
    )
    report = {
        "artifact": "qwen35_core_C00_C11_numpy_independent_verification",
        "schema_version": 1,
        "status": "passed",
        "sequence_length": args.sequence_length,
        "contract_version": NUMPY_CONTRACT_VERSION,
        "two_builds_byte_identical": True,
        "tree": run1_tree,
        "comparison_tree": run2_tree,
        "inputs": {
            "canonical_jsonl_zst_sha256": sha256_file(canonical_path),
            "contract_manifest_sha256": sha256_file(contract_path),
            "sample_index_sha256": sha256_file(index_path),
            "core_operations_sha256": sha256_file(operations_path),
            "core_membership_manifest_sha256": sha256_file(membership_path),
            "core_membership_summary_sha256": sha256_file(summary_path),
            "renderer_amendment_manifest_sha256": sha256_file(amendment_path),
            "frozen_design_manifest_sha256": sha256_file(design_path),
            "pre_amendment_token_features_sha256": sha256_file(pre_amendment_path),
        },
        "verification": verification,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
