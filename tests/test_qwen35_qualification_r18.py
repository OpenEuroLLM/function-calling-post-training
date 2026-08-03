from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID, QUALIFIED_CHUNK_SIZES
from open_instruct.qwen35_qualification_loader import load_qualification_manifest as dispatch_manifest
from open_instruct.qwen35_qualification_r18 import (
    BASE_MANIFEST_SHA256,
    CORRECTIVE_BASELINE_COMMIT,
    HUMAN_PROTOCOL_SHA256,
    load_qualification_manifest,
)

ROOT = Path(__file__).parents[1]
MANIFEST_DIR = ROOT / "scripts/train/qwen35"
MANIFEST = MANIFEST_DIR / "qwen35_hardware_qualification_r18.json"
BASE_NAMES = (
    "qwen35_hardware_qualification_r15.json",
    "qwen35_hardware_qualification_r16.json",
    "qwen35_hardware_qualification_r17.json",
)


def _copy_manifest_chain(tmp_path: Path, overlay: dict | None = None) -> Path:
    for name in BASE_NAMES:
        (tmp_path / name).write_bytes((MANIFEST_DIR / name).read_bytes())
    destination = tmp_path / MANIFEST.name
    if overlay is None:
        destination.write_bytes(MANIFEST.read_bytes())
    else:
        destination.write_text(json.dumps(overlay))
    return destination


def test_r18_manifest_is_hash_bound_and_preserves_scientific_scope() -> None:
    manifest, digest = load_qualification_manifest(MANIFEST)
    assert digest == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert manifest["protocol_id"] == "qwen35-hardware-qualification-r18"
    assert manifest["manifest_derivation"]["base_manifest"]["sha256"] == BASE_MANIFEST_SHA256
    assert manifest["source"] == {
        "corrective_baseline_commit": CORRECTIVE_BASELINE_COMMIT,
        "branch": "codex/qwen35-causal-suite",
        "require_clean_worktree": True,
    }
    assert manifest["scope"]["slurm_account"] == "aifac_f02_434"
    assert manifest["scope"]["eligible_arm_ids"] == ["C00"]
    assert manifest["scope"]["automatic_scientific_training"] is False
    assert manifest["scope"]["forbidden_evaluations"] == ["BFCL", "tau2"]
    assert manifest["runtime_pins"]["liger_execution_allowed"] is False
    assert not any(key.startswith("liger_") for key in manifest["runtime_pins"] if key != "liger_execution_allowed")


def test_r18_h2_contract_and_seeds_are_mechanically_reproducible() -> None:
    manifest, _ = load_qualification_manifest(MANIFEST)
    h2 = manifest["h2_acceptance"]
    assert h2["protocol_revision"] == 5
    assert h2["human_protocol_sha256"] == HUMAN_PROTOCOL_SHA256
    assert h2["production_implementation_id"] == IMPLEMENTATION_ID
    assert h2["candidate_chunk_sizes"] == list(QUALIFIED_CHUNK_SIZES)
    assert h2["primary_acceptance"] == "bit_exact_all_recorded_quantities"
    assert h2["diagnostic_numerical_discrepancy_is_gating"] is False
    assert h2["diagnostic_integrity_and_finiteness_are_mandatory"] is True
    assert h2["failure_policy"] == "stop_no_threshold_rescue"

    identities: list[tuple[str, str, int]] = []
    identities.extend((case["seed_label"], case["seed_sha256"], case["seed"]) for case in h2["direct_cases"])
    real = h2["real_geometry_case"]
    identities.append((real["seed_label"], real["seed_sha256"], real["seed"]))
    for row in h2["trajectories"]:
        identities.extend(
            (row[f"{prefix}_seed_label"], row[f"{prefix}_seed_sha256"], row[seed_key])
            for prefix, seed_key in (
                ("model", "model_seed"),
                ("batch", "batch_seed_base"),
                ("heldout", "heldout_seed"),
            )
        )
    assert len(identities) == len({label for label, _, _ in identities}) == 20
    for label, digest, seed in identities:
        expected = hashlib.sha256(label.encode()).hexdigest()
        assert digest == expected
        assert seed == int(expected[:8], 16)


def test_dispatcher_resolves_r16_r17_and_r18_without_reinterpreting_them() -> None:
    values = [
        dispatch_manifest(MANIFEST_DIR / f"qwen35_hardware_qualification_r{revision}.json")[0]
        for revision in (16, 17, 18)
    ]
    assert [value["protocol_id"] for value in values] == [
        "qwen35-hardware-qualification-r16",
        "qwen35-hardware-qualification-r17",
        "qwen35-hardware-qualification-r18",
    ]
    assert values[0]["scope"] == values[1]["scope"] == values[2]["scope"]
    assert values[0]["training_unit"] == values[1]["training_unit"] == values[2]["training_unit"]


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("status",), "qualified", "identity/status drift"),
        (("transformations", "require_bit_exact_same_chunk_reference"), False, "transformation contract drift"),
        (("overrides", "source", "corrective_baseline_commit"), "0" * 40, "source baseline drift"),
        (("overrides", "runtime_pins", "liger_execution_allowed"), True, "runtime pin drift"),
        (("overrides", "gates", 2, "name"), "weaker_h2", "gate sequence drift"),
        (("overrides", "h2_acceptance", "candidate_chunk_sizes"), [128, 256], "scalar drift"),
        (("overrides", "h2_acceptance", "primary_acceptance"), "close_enough", "scalar drift"),
        (("overrides", "h2_acceptance", "diagnostic_numerical_discrepancy_is_gating"), True, "scalar drift"),
        (("overrides", "memory_acceptance", "tie_fraction"), 0.03, "memory/chunk-selection contract drift"),
    ],
)
def test_r18_overlay_rejects_adversarial_contract_mutations(
    tmp_path: Path, path: tuple, replacement: object, message: str
) -> None:
    overlay = json.loads(MANIFEST.read_text())
    node = overlay
    for component in path[:-1]:
        node = node[component]
    node[path[-1]] = replacement
    with pytest.raises(ValueError, match=message):
        load_qualification_manifest(_copy_manifest_chain(tmp_path, overlay))


def test_r18_rejects_byte_drift_in_immutable_r17_base(tmp_path: Path) -> None:
    path = _copy_manifest_chain(tmp_path)
    base_path = tmp_path / "qwen35_hardware_qualification_r17.json"
    base = json.loads(base_path.read_text())
    base["status"] = "tampered"
    base_path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="immutable R17 base-manifest bytes drift"):
        load_qualification_manifest(path)


def test_r18_loader_does_not_mutate_validated_r17_value() -> None:
    r17, _ = dispatch_manifest(MANIFEST_DIR / "qwen35_hardware_qualification_r17.json")
    before = copy.deepcopy(r17)
    r18, _ = load_qualification_manifest(MANIFEST)
    assert r17 == before
    assert r18["h2_acceptance"] != r17["h2_acceptance"]
