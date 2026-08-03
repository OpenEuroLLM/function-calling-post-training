import gzip
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from open_instruct.qwen35_data import NUMPY_CONTRACT_VERSION, Qwen35NumpyPackedDataset, Qwen35PackedCollator
from open_instruct.qwen35_schedule import (
    ScheduledQwen35Dataset,
    build_schedule_manifest,
    canonical_json_bytes,
    select_pack_indices,
    validate_schedule_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_numpy_fixture(
    directory: Path,
    *,
    document_lengths: list[int],
    trainable_counts: list[int] | None = None,
    sequence_length: int = 8,
) -> Qwen35NumpyPackedDataset:
    if trainable_counts is None:
        trainable_counts = [max(0, length - 1) for length in document_lengths]
    assert len(document_lengths) == len(trainable_counts)
    token_rows = []
    mask_rows = []
    boundaries = []
    documents = []
    offset = 0
    for index, (length, trainable_count) in enumerate(zip(document_lengths, trainable_counts, strict=True)):
        assert 0 <= trainable_count <= length - 1
        token_rows.extend(range(100 + 10 * index, 100 + 10 * index + length))
        mask_rows.extend([False] * (length - trainable_count) + [True] * trainable_count)
        boundaries.append((offset, offset + length))
        documents.append(
            {"sample_uid": f"sample-{index:04d}", "segments": [{"part": 0, "start": offset, "end": offset + length}]}
        )
        offset += length

    token_path = directory / "token_ids_part_0000.npy"
    mask_path = directory / "labels_mask_part_0000.npy"
    boundary_path = directory / "token_ids_part_0000.csv.gz"
    document_path = directory / "documents.jsonl.gz"
    np.asarray(token_rows, dtype=np.uint16).tofile(token_path)
    np.asarray(mask_rows, dtype=np.bool_).tofile(mask_path)
    with gzip.open(boundary_path, "wt") as handle:
        for start, end in boundaries:
            handle.write(f"{start},{end}\n")
    with gzip.open(document_path, "wt") as handle:
        for row in documents:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "contract_version": NUMPY_CONTRACT_VERSION,
        "suite_id": "test-suite",
        "arm_id": "C00",
        "documents_index": document_path.name,
        "documents_index_sha256": _sha256(document_path),
        "parts": [
            {
                "token_ids": token_path.name,
                "labels_mask": mask_path.name,
                "boundaries": boundary_path.name,
                "token_dtype": "uint16",
                "num_tokens": len(token_rows),
                "token_ids_sha256": _sha256(token_path),
                "labels_mask_sha256": _sha256(mask_path),
                "boundaries_sha256": _sha256(boundary_path),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return Qwen35NumpyPackedDataset(directory, sequence_length=sequence_length, drop_last=False, verify_hashes=True)


def _reseal(schedule: dict) -> None:
    schedule["entries_sha256"] = hashlib.sha256(canonical_json_bytes(schedule["entries"])).hexdigest()
    body = dict(schedule)
    body.pop("schedule_sha256", None)
    schedule["schedule_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def test_document_index_drives_stable_pack_identity_and_exact_accounting(tmp_path):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[3, 2, 3], trainable_counts=[2, 1, 1], sequence_length=8)

    metadata = dataset.pack_metadata(0)

    assert metadata.document_uids == ("sample-0000", "sample-0001", "sample-0002")
    assert metadata.document_lengths == (3, 2, 3)
    assert metadata.real_tokens == 8
    assert metadata.assistant_targets == 4
    assert metadata.padding_tokens == 0
    assert metadata.attention_length_squared == 3**2 + 2**2 + 3**2
    assert len(metadata.pack_uid) == 64


@pytest.mark.parametrize("fault", ["missing", "duplicate_uid", "multi_segment"])
def test_document_index_mismatch_is_rejected(tmp_path, fault):
    _write_numpy_fixture(tmp_path, document_lengths=[3, 3], sequence_length=8)
    document_path = tmp_path / "documents.jsonl.gz"
    with gzip.open(document_path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate_uid":
        rows[1]["sample_uid"] = rows[0]["sample_uid"]
    else:
        rows[0]["segments"].append(dict(rows[0]["segments"][0]))
    with gzip.open(document_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="document index|duplicate sample_uid|exactly one atomic segment"):
        Qwen35NumpyPackedDataset(tmp_path, sequence_length=8, drop_last=False)


def test_document_index_hash_is_verified_directly(tmp_path):
    _write_numpy_fixture(tmp_path, document_lengths=[3, 3], sequence_length=8)
    document_path = tmp_path / "documents.jsonl.gz"
    with document_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        Qwen35NumpyPackedDataset(tmp_path, sequence_length=8, drop_last=False, verify_hashes=True)


def test_collator_selects_exact_predecessor_rows_and_zero_loss_sentinel(tmp_path):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[5], trainable_counts=[2], sequence_length=8)
    feature = dict(dataset[0])
    feature.update({"_qwen35_schedule_index": 7, "_qwen35_pack_uid": "pack", "_qwen35_synthetic": False})
    collator = Qwen35PackedCollator(pad_token_id=0, sequence_length=8)

    batch = collator([feature])

    # Source mask is [F,F,F,T,T], so labels 3 and 4 are predicted by hidden rows 2 and 3.
    assert batch["logits_to_keep"].tolist() == [2, 3]
    assert batch["shift_labels"].tolist() == feature["input_ids"][3:5].tolist()
    assert batch["_qwen35_assistant_targets"] == 2
    assert batch["_qwen35_schedule_index"] == 7

    synthetic = collator([{"_qwen35_synthetic": True, "_qwen35_schedule_index": 8}])
    assert synthetic["labels"].eq(-100).all()
    assert synthetic["logits_to_keep"].tolist() == [0]
    assert synthetic["shift_labels"].tolist() == [-100]
    assert synthetic["_qwen35_assistant_targets"] == 0
    assert synthetic["cu_seq_lens_q"].tolist() == [0, 8]


def test_schedule_is_deterministic_no_repeat_and_complete_global_groups(tmp_path):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[5] * 10, sequence_length=8)

    first = build_schedule_manifest(
        dataset, seed=3407, global_packs_per_update=8, allow_synthetic_final_group_padding=True
    )
    second = build_schedule_manifest(
        dataset, seed=3407, global_packs_per_update=8, allow_synthetic_final_group_padding=True
    )
    other_seed = build_schedule_manifest(
        dataset, seed=3408, global_packs_per_update=8, allow_synthetic_final_group_padding=True
    )
    validation = validate_schedule_manifest(first, dataset, expected_seed=3407, expected_global_packs_per_update=8)

    assert first == second
    assert first["entries"][:10] != other_seed["entries"][:10]
    assert validation["real_pack_count"] == 10
    assert validation["synthetic_all_masked_pack_count"] == 6
    assert validation["scheduled_pack_count"] == 16
    assert validation["optimizer_updates"] == 2
    real_indices = [row["pack_index"] for row in first["entries"] if not row["synthetic"]]
    assert len(real_indices) == len(set(real_indices)) == 10
    assert all(row["synthetic"] for row in first["entries"][-6:])


def test_schedule_refuses_implicit_synthetic_compute(tmp_path):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[5] * 5, sequence_length=8)

    with pytest.raises(ValueError, match="explicitly allow synthetic"):
        build_schedule_manifest(dataset, seed=3407, global_packs_per_update=4)


@pytest.mark.parametrize("fault", ["hash", "duplicate", "entry_drift", "real_after_synthetic", "wrong_group"])
def test_schedule_validator_fails_closed_on_adversarial_drift(tmp_path, fault):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[5] * 5, sequence_length=8)
    schedule = deepcopy(
        build_schedule_manifest(
            dataset, seed=3407, global_packs_per_update=4, allow_synthetic_final_group_padding=True
        )
    )
    if fault == "hash":
        schedule["schedule_sha256"] = "0" * 64
    elif fault == "duplicate":
        schedule["entries"][1] = dict(schedule["entries"][0], schedule_index=1)
        _reseal(schedule)
    elif fault == "entry_drift":
        schedule["entries"][0]["assistant_targets"] += 1
        schedule["totals"]["assistant_targets"] += 1
        _reseal(schedule)
    elif fault == "real_after_synthetic":
        schedule["entries"][4], schedule["entries"][5] = schedule["entries"][5], schedule["entries"][4]
        for index, entry in enumerate(schedule["entries"]):
            entry["schedule_index"] = index
            if entry["synthetic"]:
                entry["pack_uid"] = f"synthetic-all-masked-{index:08d}"
        _reseal(schedule)
    else:
        schedule["global_packs_per_update"] = 3
        _reseal(schedule)

    with pytest.raises(ValueError):
        validate_schedule_manifest(schedule, dataset, expected_seed=3407, expected_global_packs_per_update=4)


def test_scheduled_dataset_maps_real_and_synthetic_entries_exactly(tmp_path):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[5] * 3, sequence_length=8)
    schedule = build_schedule_manifest(
        dataset, seed=3407, global_packs_per_update=4, allow_synthetic_final_group_padding=True
    )
    scheduled = ScheduledQwen35Dataset(dataset, schedule)

    real = scheduled[0]
    synthetic = scheduled[3]

    assert real["_qwen35_schedule_index"] == 0
    assert real["_qwen35_pack_uid"] == schedule["entries"][0]["pack_uid"]
    assert real["_qwen35_assistant_targets"] == schedule["entries"][0]["assistant_targets"]
    assert synthetic == {
        "_qwen35_schedule_index": 3,
        "_qwen35_pack_index": -1,
        "_qwen35_pack_uid": "synthetic-all-masked-00000003",
        "_qwen35_synthetic": True,
        "_qwen35_real_tokens": 0,
        "_qwen35_assistant_targets": 0,
        "_qwen35_padding_tokens": 8,
        "_qwen35_attention_length_squared": 64,
        "_qwen35_document_count": 0,
    }


def test_assistant_target_balancing_is_exact_when_feasible_and_rejects_impossible_target(tmp_path):
    dataset = _write_numpy_fixture(
        tmp_path, document_lengths=[7] * 6, trainable_counts=[1, 2, 3, 4, 5, 6], sequence_length=8
    )
    metadata = [dataset.pack_metadata(index) for index in range(len(dataset))]

    selected, report = select_pack_indices(
        metadata, seed=3407, real_pack_limit=2, target_assistant_tokens=7, assistant_relative_tolerance=0
    )

    assert len(selected) == 2
    assert sum(metadata[index].assistant_targets for index in selected) == 7
    assert report["final_assistant_error"] == 0
    with pytest.raises(ValueError, match="infeasible"):
        select_pack_indices(
            metadata, seed=3407, real_pack_limit=2, target_assistant_tokens=20, assistant_relative_tolerance=0
        )


def test_schedule_rejects_an_optimizer_group_with_no_supervision(tmp_path):
    dataset = _write_numpy_fixture(tmp_path, document_lengths=[5, 5], trainable_counts=[0, 0], sequence_length=8)
    schedule = build_schedule_manifest(dataset, seed=3407, global_packs_per_update=2)

    with pytest.raises(ValueError, match="assistant supervision"):
        validate_schedule_manifest(schedule, dataset)


def test_pack_uid_changes_if_document_identity_changes(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _write_numpy_fixture(first_dir, document_lengths=[4], sequence_length=8)
    second = _write_numpy_fixture(second_dir, document_lengths=[4], sequence_length=8)
    document_path = second_dir / "documents.jsonl.gz"
    with gzip.open(document_path, "rt") as handle:
        row = json.loads(handle.readline())
    row["sample_uid"] = "different-sample"
    with gzip.open(document_path, "wt") as handle:
        handle.write(json.dumps(row) + "\n")
    second = Qwen35NumpyPackedDataset(second_dir, sequence_length=8, drop_last=False)

    assert first.pack_metadata(0).pack_uid != second.pack_metadata(0).pack_uid


def test_qualification_schedule_builder_accepts_recursive_r18_overlay():
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts/data/build_qwen35_qualification_schedules.py"
    specification = importlib.util.spec_from_file_location("qwen35_qualification_schedule_builder", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    manifest = repository / "scripts/train/qwen35/qwen35_hardware_qualification_r18.json"
    qualification, digest = module.load_qualification_manifest(manifest)

    assert module.load_qualification_manifest.__module__ == "open_instruct.qwen35_qualification_loader"
    assert qualification["protocol_id"] == "qwen35-hardware-qualification-r18"
    assert qualification["runtime_pins"]["liger_execution_allowed"] is False
    assert len(digest) == 64
