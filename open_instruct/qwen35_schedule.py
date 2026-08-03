"""Deterministic, no-repeat exposure schedules for packed Qwen3.5 SFT."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from open_instruct.qwen35_data import Qwen35NumpyPackedDataset, Qwen35PackMetadata

SCHEDULE_SCHEMA_VERSION = 1
SCHEDULE_ARTIFACT = "qwen35_hashed_no_repeat_pack_schedule"
SYNTHETIC_PACK_UID_PREFIX = "synthetic-all-masked"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def priority_digest(seed: int, pack_uid: str) -> str:
    """Arm-independent random priority for paired schedule seed ``seed``."""

    material = f"qwen35-schedule-v{SCHEDULE_SCHEMA_VERSION}\0{seed}\0{pack_uid}".encode()
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class Qwen35ScheduleEntry:
    schedule_index: int
    pack_index: int | None
    pack_uid: str
    priority_sha256: str | None
    synthetic: bool
    real_tokens: int
    assistant_targets: int
    padding_tokens: int
    attention_length_squared: int
    document_count: int


def _entry_from_pack(schedule_index: int, metadata: Qwen35PackMetadata, seed: int) -> Qwen35ScheduleEntry:
    return Qwen35ScheduleEntry(
        schedule_index=schedule_index,
        pack_index=metadata.pack_index,
        pack_uid=metadata.pack_uid,
        priority_sha256=priority_digest(seed, metadata.pack_uid),
        synthetic=False,
        real_tokens=metadata.real_tokens,
        assistant_targets=metadata.assistant_targets,
        padding_tokens=metadata.padding_tokens,
        attention_length_squared=metadata.attention_length_squared,
        document_count=len(metadata.document_uids),
    )


def _synthetic_entry(schedule_index: int, sequence_length: int) -> Qwen35ScheduleEntry:
    return Qwen35ScheduleEntry(
        schedule_index=schedule_index,
        pack_index=None,
        pack_uid=f"{SYNTHETIC_PACK_UID_PREFIX}-{schedule_index:08d}",
        priority_sha256=None,
        synthetic=True,
        real_tokens=0,
        assistant_targets=0,
        padding_tokens=sequence_length,
        attention_length_squared=sequence_length**2,
        document_count=0,
    )


def _best_single_swap(
    selected: set[int],
    unselected: set[int],
    metadata: list[Qwen35PackMetadata],
    priorities: dict[int, str],
    desired_delta: int,
) -> tuple[int, int, int] | None:
    """Find the deterministic one-for-one swap closest to ``desired_delta``."""

    unselected_sorted = sorted(
        unselected, key=lambda index: (metadata[index].assistant_targets, priorities[index], index)
    )
    unselected_counts = [metadata[index].assistant_targets for index in unselected_sorted]
    best: tuple[tuple[int, str, str, int, int], int, int, int] | None = None
    for remove_index in sorted(selected, key=lambda index: (priorities[index], index)):
        wanted_add_count = metadata[remove_index].assistant_targets + desired_delta
        insertion = bisect.bisect_left(unselected_counts, wanted_add_count)
        for position in (insertion - 1, insertion, insertion + 1):
            if position < 0 or position >= len(unselected_sorted):
                continue
            add_index = unselected_sorted[position]
            delta = metadata[add_index].assistant_targets - metadata[remove_index].assistant_targets
            score = (
                abs(desired_delta - delta),
                priorities[remove_index],
                priorities[add_index],
                remove_index,
                add_index,
            )
            if best is None or score < best[0]:
                best = (score, remove_index, add_index, delta)
    if best is None:
        return None
    return best[1], best[2], best[3]


def select_pack_indices(
    metadata: list[Qwen35PackMetadata],
    *,
    seed: int,
    real_pack_limit: int | None,
    target_assistant_tokens: int | None,
    assistant_relative_tolerance: float,
    maximum_swap_iterations: int = 128,
) -> tuple[list[int], dict[str, Any]]:
    """Select a deterministic exact-count subset and optionally target supervision mass."""

    if not metadata:
        raise ValueError("cannot schedule an empty packed dataset")
    if real_pack_limit is None:
        real_pack_limit = len(metadata)
    if real_pack_limit <= 0 or real_pack_limit > len(metadata):
        raise ValueError(f"real_pack_limit must be in [1, {len(metadata)}]")
    if assistant_relative_tolerance < 0 or not math.isfinite(assistant_relative_tolerance):
        raise ValueError("assistant_relative_tolerance must be finite and nonnegative")
    if any(row.pack_index != index for index, row in enumerate(metadata)):
        raise ValueError("pack metadata must be ordered contiguously by pack_index")

    priorities = {row.pack_index: priority_digest(seed, row.pack_uid) for row in metadata}
    ordered = sorted(range(len(metadata)), key=lambda index: (priorities[index], metadata[index].pack_uid, index))
    selected = set(ordered[:real_pack_limit])
    unselected = set(ordered[real_pack_limit:])
    initial_total = sum(metadata[index].assistant_targets for index in selected)
    total = initial_total
    swaps: list[dict[str, int]] = []

    if target_assistant_tokens is not None:
        if target_assistant_tokens <= 0:
            raise ValueError("target_assistant_tokens must be positive")
        all_counts = sorted(row.assistant_targets for row in metadata)
        minimum = sum(all_counts[:real_pack_limit])
        maximum = sum(all_counts[-real_pack_limit:])
        if not minimum <= target_assistant_tokens <= maximum:
            raise ValueError(
                f"assistant target {target_assistant_tokens} is infeasible for {real_pack_limit} packs; "
                f"feasible bounds are [{minimum}, {maximum}]"
            )

        allowed_error = math.ceil(target_assistant_tokens * assistant_relative_tolerance)
        for _ in range(maximum_swap_iterations):
            error = target_assistant_tokens - total
            if abs(error) <= allowed_error:
                break
            candidate = _best_single_swap(selected, unselected, metadata, priorities, error)
            if candidate is None:
                break
            remove_index, add_index, delta = candidate
            if abs(error - delta) >= abs(error):
                break
            selected.remove(remove_index)
            selected.add(add_index)
            unselected.remove(add_index)
            unselected.add(remove_index)
            total += delta
            swaps.append(
                {"remove_pack_index": remove_index, "add_pack_index": add_index, "assistant_target_delta": delta}
            )
        final_error = total - target_assistant_tokens
        if abs(final_error) > allowed_error:
            raise ValueError(
                "deterministic exact-count balancing did not meet assistant-target tolerance: "
                f"target={target_assistant_tokens}, actual={total}, error={final_error}, "
                f"allowed={allowed_error}, swaps={len(swaps)}"
            )
    else:
        allowed_error = None
        final_error = None

    final_order = sorted(selected, key=lambda index: (priorities[index], metadata[index].pack_uid, index))
    return final_order, {
        "available_real_packs": len(metadata),
        "selected_real_packs": len(final_order),
        "initial_assistant_targets": initial_total,
        "final_assistant_targets": total,
        "target_assistant_tokens": target_assistant_tokens,
        "assistant_relative_tolerance": assistant_relative_tolerance,
        "allowed_absolute_assistant_error": allowed_error,
        "final_assistant_error": final_error,
        "balancing_swaps": swaps,
    }


def build_schedule_manifest(
    dataset: Qwen35NumpyPackedDataset,
    *,
    seed: int,
    global_packs_per_update: int,
    real_pack_limit: int | None = None,
    target_assistant_tokens: int | None = None,
    assistant_relative_tolerance: float = 0.001,
    allow_synthetic_final_group_padding: bool = False,
) -> dict[str, Any]:
    if global_packs_per_update <= 0:
        raise ValueError("global_packs_per_update must be positive")
    metadata = [dataset.pack_metadata(index) for index in range(len(dataset))]
    selected, selection = select_pack_indices(
        metadata,
        seed=seed,
        real_pack_limit=real_pack_limit,
        target_assistant_tokens=target_assistant_tokens,
        assistant_relative_tolerance=assistant_relative_tolerance,
    )
    entries = [_entry_from_pack(index, metadata[pack_index], seed) for index, pack_index in enumerate(selected)]
    synthetic_count = (-len(entries)) % global_packs_per_update
    if synthetic_count and not allow_synthetic_final_group_padding:
        raise ValueError(
            f"{len(entries)} real packs do not fill groups of {global_packs_per_update}; "
            "choose a divisible --real-pack-limit or explicitly allow synthetic final-group padding"
        )
    for _ in range(synthetic_count):
        entries.append(_synthetic_entry(len(entries), dataset.sequence_length))
    entry_dicts = [asdict(entry) for entry in entries]
    dataset_manifest_path = dataset.data_dir / "manifest.json"
    body: dict[str, Any] = {
        "artifact": SCHEDULE_ARTIFACT,
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "suite_id": dataset.manifest.get("suite_id"),
        "arm_id": dataset.manifest.get("arm_id"),
        "numpy_contract_version": dataset.manifest.get("contract_version"),
        "numpy_manifest_sha256": sha256_file(dataset_manifest_path),
        "documents_index_sha256": dataset.manifest.get("documents_index_sha256"),
        "sequence_length": dataset.sequence_length,
        "schedule_seed": seed,
        "priority_function": "sha256(qwen35-schedule-v1\\0seed\\0pack_uid)",
        "global_packs_per_update": global_packs_per_update,
        "real_pack_count": len(selected),
        "synthetic_all_masked_pack_count": synthetic_count,
        "final_group_policy": "synthetic_all_masked" if synthetic_count else "real_packs_only",
        "scheduled_pack_count": len(entries),
        "optimizer_updates": len(entries) // global_packs_per_update,
        "selection": selection,
        "totals": {
            "fixed_tokens": len(entries) * dataset.sequence_length,
            "real_tokens": sum(entry.real_tokens for entry in entries),
            "assistant_targets": sum(entry.assistant_targets for entry in entries),
            "padding_tokens": sum(entry.padding_tokens for entry in entries),
            "attention_length_squared": sum(entry.attention_length_squared for entry in entries),
        },
        "entries": entry_dicts,
    }
    body["entries_sha256"] = hashlib.sha256(canonical_json_bytes(entry_dicts)).hexdigest()
    body["schedule_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def validate_schedule_manifest(
    manifest: dict[str, Any],
    dataset: Qwen35NumpyPackedDataset,
    *,
    expected_seed: int | None = None,
    expected_global_packs_per_update: int | None = None,
) -> dict[str, Any]:
    if manifest.get("artifact") != SCHEDULE_ARTIFACT or manifest.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ValueError("unsupported Qwen3.5 schedule artifact")
    claimed_hash = manifest.get("schedule_sha256")
    body = dict(manifest)
    body.pop("schedule_sha256", None)
    actual_hash = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if claimed_hash != actual_hash:
        raise ValueError("schedule_sha256 mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("schedule has no entries")
    if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != manifest.get("entries_sha256"):
        raise ValueError("entries_sha256 mismatch")
    if manifest.get("numpy_manifest_sha256") != sha256_file(dataset.data_dir / "manifest.json"):
        raise ValueError("schedule was built for a different NumPy manifest")
    if manifest.get("sequence_length") != dataset.sequence_length:
        raise ValueError("schedule and dataset sequence lengths differ")
    if manifest.get("suite_id") != dataset.manifest.get("suite_id"):
        raise ValueError("schedule suite_id drift")
    if manifest.get("arm_id") != dataset.manifest.get("arm_id"):
        raise ValueError("schedule arm_id drift")
    if expected_seed is not None and manifest.get("schedule_seed") != expected_seed:
        raise ValueError("schedule seed does not equal the paired training seed")
    policy = manifest.get("final_group_policy")
    if policy not in {"real_packs_only", "synthetic_all_masked"}:
        raise ValueError("schedule has an unsupported final-group policy")

    group = int(manifest.get("global_packs_per_update", 0))
    if group <= 0 or len(entries) % group:
        raise ValueError("schedule does not contain complete global optimizer groups")
    if expected_global_packs_per_update is not None and group != expected_global_packs_per_update:
        raise ValueError("schedule global group is incompatible with world size and gradient accumulation")
    zero_target_groups = [
        index // group
        for index in range(0, len(entries), group)
        if sum(int(entry["assistant_targets"]) for entry in entries[index : index + group]) <= 0
    ]
    if zero_target_groups:
        raise ValueError(
            f"every optimizer group must contain assistant supervision; zero-target groups={zero_target_groups[:10]}"
        )

    seen_pack_indices: set[int] = set()
    seen_pack_uids: set[str] = set()
    seen_synthetic = False
    totals = {
        key: 0
        for key in ("fixed_tokens", "real_tokens", "assistant_targets", "padding_tokens", "attention_length_squared")
    }
    for schedule_index, raw_entry in enumerate(entries):
        if raw_entry.get("schedule_index") != schedule_index:
            raise ValueError("schedule indices are not contiguous")
        synthetic = bool(raw_entry.get("synthetic"))
        if synthetic:
            seen_synthetic = True
            if raw_entry.get("pack_index") is not None:
                raise ValueError("synthetic entry unexpectedly references a real pack")
            expected = _synthetic_entry(schedule_index, dataset.sequence_length)
        else:
            if seen_synthetic:
                raise ValueError("real pack appears after synthetic final-group padding")
            pack_index = raw_entry.get("pack_index")
            if not isinstance(pack_index, int) or not 0 <= pack_index < len(dataset):
                raise ValueError("invalid real pack index")
            if pack_index in seen_pack_indices:
                raise ValueError("real pack index repeats in schedule")
            metadata = dataset.pack_metadata(pack_index)
            if metadata.pack_uid in seen_pack_uids:
                raise ValueError("real pack UID repeats in schedule")
            seen_pack_indices.add(pack_index)
            seen_pack_uids.add(metadata.pack_uid)
            expected = _entry_from_pack(schedule_index, metadata, int(manifest["schedule_seed"]))
        if raw_entry != asdict(expected):
            raise ValueError(f"schedule entry {schedule_index} does not match dataset accounting")
        totals["fixed_tokens"] += dataset.sequence_length
        for key in ("real_tokens", "assistant_targets", "padding_tokens", "attention_length_squared"):
            totals[key] += int(raw_entry[key])

    if totals != manifest.get("totals"):
        raise ValueError("schedule totals do not equal recomputed entry totals")
    if len(seen_pack_indices) != manifest.get("real_pack_count"):
        raise ValueError("schedule real-pack count drift")
    if len(entries) - len(seen_pack_indices) != manifest.get("synthetic_all_masked_pack_count"):
        raise ValueError("schedule synthetic-pack count drift")
    if (len(entries) > len(seen_pack_indices)) != (policy == "synthetic_all_masked"):
        raise ValueError("schedule final-group policy does not match its entries")
    if len(entries) // group != manifest.get("optimizer_updates"):
        raise ValueError("schedule optimizer-update count drift")
    return {
        "schedule_sha256": claimed_hash,
        "entries_sha256": manifest["entries_sha256"],
        "real_pack_count": len(seen_pack_indices),
        "synthetic_all_masked_pack_count": len(entries) - len(seen_pack_indices),
        "scheduled_pack_count": len(entries),
        "optimizer_updates": len(entries) // group,
        "totals": totals,
    }


class ScheduledQwen35Dataset(Dataset):
    """Expose a validated schedule as a sequential map-style dataset."""

    def __init__(self, dataset: Qwen35NumpyPackedDataset, schedule: dict[str, Any]) -> None:
        self.dataset = dataset
        self.schedule = schedule
        self.entries = schedule["entries"]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        if entry["synthetic"]:
            item: dict[str, Any] = {"_qwen35_synthetic": True}
        else:
            item = dict(self.dataset[int(entry["pack_index"])])
        item.update(
            {
                "_qwen35_schedule_index": int(entry["schedule_index"]),
                "_qwen35_pack_index": -1 if entry["pack_index"] is None else int(entry["pack_index"]),
                "_qwen35_pack_uid": str(entry["pack_uid"]),
                "_qwen35_synthetic": bool(entry["synthetic"]),
                "_qwen35_real_tokens": int(entry["real_tokens"]),
                "_qwen35_assistant_targets": int(entry["assistant_targets"]),
                "_qwen35_padding_tokens": int(entry["padding_tokens"]),
                "_qwen35_attention_length_squared": int(entry["attention_length_squared"]),
                "_qwen35_document_count": int(entry["document_count"]),
            }
        )
        return item
