"""Fail-closed Qwen3.5 text-model restoration for Trainer checkpoints.

Transformers serializes ``Qwen3_5ForCausalLM`` weights in the canonical
conditional-model namespace.  ``Trainer`` does not apply the inverse
conversion when it resumes.  This module performs that conversion explicitly
while retaining the live parameter objects used by the trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors import safe_open

from open_instruct.qwen35_training import conditional_source_key_for_text_target

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
UNSAFE_MODEL_WEIGHT_NAMES = ("pytorch_model.bin", "pytorch_model.bin.index.json", "model.bin", "model.pt")
EXPECTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_MODEL_TYPE = "qwen3_5_text"
TIED_SOURCE_KEY = "model.language_model.embed_tokens.weight"
TIED_TARGET_KEYS = frozenset({"model.embed_tokens.weight", "lm_head.weight"})


class Qwen35CheckpointResumeError(RuntimeError):
    """Raised when a Trainer checkpoint cannot be restored exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise Qwen35CheckpointResumeError(f"JSON contains duplicate member {key!r}")
        value[key] = member
    return value


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_json_members)
    except Qwen35CheckpointResumeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen35CheckpointResumeError(f"cannot read strict JSON {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen35CheckpointResumeError(f"{path.name} must contain a JSON object")
    return value


def _require_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise Qwen35CheckpointResumeError(f"checkpoint file may not be a symlink: {path.name}")
    if not path.is_file():
        raise Qwen35CheckpointResumeError(f"required checkpoint file is absent: {path.name}")


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _safetensors_header_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise Qwen35CheckpointResumeError(f"truncated safe-tensor prefix: {path.name}")
        header_length = int.from_bytes(prefix, byteorder="little", signed=False)
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise Qwen35CheckpointResumeError(f"invalid safe-tensor header length in {path.name}")
        header = handle.read(header_length)
    if len(header) != header_length:
        raise Qwen35CheckpointResumeError(f"truncated safe-tensor header: {path.name}")
    return hashlib.sha256(prefix + header).hexdigest()


def _validate_shard_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen35CheckpointResumeError("safe-tensor index contains a non-string shard name")
    if "\\" in value:
        raise Qwen35CheckpointResumeError(f"safe-tensor shard path uses a backslash: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise Qwen35CheckpointResumeError(f"safe-tensor shard path is not a plain filename: {value!r}")
    if not value.endswith(".safetensors"):
        raise Qwen35CheckpointResumeError(f"indexed model shard is not a safe-tensor file: {value!r}")
    return value


def _discover_layout(checkpoint_dir: Path) -> tuple[str, dict[str, Path], dict[str, str] | None]:
    single = checkpoint_dir / SAFE_WEIGHTS_NAME
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    if single.exists() and index_path.exists():
        raise Qwen35CheckpointResumeError("checkpoint has simultaneous single-file and indexed weight layouts")
    for name in UNSAFE_MODEL_WEIGHT_NAMES:
        if (checkpoint_dir / name).exists():
            raise Qwen35CheckpointResumeError(f"unsafe pickle model weights are forbidden: {name}")

    if single.exists():
        _require_regular_file(single)
        safe_files = {path.name for path in checkpoint_dir.glob("*.safetensors")}
        if safe_files != {SAFE_WEIGHTS_NAME}:
            raise Qwen35CheckpointResumeError(
                f"single-file layout has unexpected safe-tensor files: {sorted(safe_files)}"
            )
        return "single_model_safetensors", {SAFE_WEIGHTS_NAME: single}, None

    if not index_path.exists():
        raise Qwen35CheckpointResumeError("checkpoint has no supported safe-tensor model-weight layout")
    _require_regular_file(index_path)
    index = _load_json_strict(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise Qwen35CheckpointResumeError("safe-tensor index has no non-empty weight_map")
    normalized_map: dict[str, str] = {}
    for key, shard_value in weight_map.items():
        if not isinstance(key, str) or not key:
            raise Qwen35CheckpointResumeError("safe-tensor index contains an invalid tensor key")
        normalized_map[key] = _validate_shard_name(shard_value)
    shard_names = set(normalized_map.values())
    safe_files = {path.name for path in checkpoint_dir.glob("*.safetensors")}
    if safe_files != shard_names:
        raise Qwen35CheckpointResumeError(
            "indexed safe-tensor shard set disagrees with directory contents: "
            f"index={sorted(shard_names)}, files={sorted(safe_files)}"
        )
    shards = {}
    for name in sorted(shard_names):
        path = checkpoint_dir / name
        _require_regular_file(path)
        shards[name] = path
    return "indexed_safetensors_shards", shards, normalized_map


def _validate_model_and_config(model: torch.nn.Module, checkpoint_dir: Path) -> tuple[Path, dict[str, Any]]:
    if type(model).__name__ != EXPECTED_MODEL_CLASS:
        raise Qwen35CheckpointResumeError(
            f"strict loader requires {EXPECTED_MODEL_CLASS}, found {type(model).__name__}"
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != EXPECTED_MODEL_TYPE:
        raise Qwen35CheckpointResumeError(
            f"live model_type must be {EXPECTED_MODEL_TYPE!r}, found {getattr(config, 'model_type', None)!r}"
        )
    checkpoint_dir = checkpoint_dir.expanduser()
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise Qwen35CheckpointResumeError("resume checkpoint must be a real, non-symlinked directory")
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    config_path = checkpoint_dir / "config.json"
    _require_regular_file(config_path)
    checkpoint_config = _load_json_strict(config_path)
    if checkpoint_config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise Qwen35CheckpointResumeError("checkpoint config model_type is not qwen3_5_text")
    if checkpoint_config.get("architectures") != [EXPECTED_MODEL_CLASS]:
        raise Qwen35CheckpointResumeError("checkpoint config architectures do not name Qwen3_5ForCausalLM exactly")
    return config_path, checkpoint_config


def _tied_weights(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    input_module = getattr(getattr(model, "model", None), "embed_tokens", None)
    output_module = getattr(model, "lm_head", None)
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if not torch.is_tensor(input_weight) or not torch.is_tensor(output_weight):
        raise Qwen35CheckpointResumeError("live model has no input/output embedding tensors")
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise Qwen35CheckpointResumeError("live input/output embeddings are not tied")
    return input_weight, output_weight


def _preflight_sources(
    *,
    shards: dict[str, Path],
    index_map: dict[str, str] | None,
    source_to_targets: dict[str, list[str]],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    source_metadata: dict[str, dict[str, Any]] = {}
    observed_by_shard: dict[str, list[str]] = {}
    for shard_name, shard_path in shards.items():
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                shard_keys = sorted(handle.keys())
                observed_by_shard[shard_name] = shard_keys
                for source_key in shard_keys:
                    if source_key in source_metadata:
                        raise Qwen35CheckpointResumeError(
                            f"safe-tensor key occurs in more than one shard: {source_key}"
                        )
                    tensor_slice = handle.get_slice(source_key)
                    source_metadata[source_key] = {
                        "shard": shard_name,
                        "shape": list(tensor_slice.get_shape()),
                        "dtype": tensor_slice.get_dtype(),
                    }
        except Qwen35CheckpointResumeError:
            raise
        except Exception as error:
            raise Qwen35CheckpointResumeError(f"cannot preflight safe-tensor shard {shard_name}: {error}") from error

    if index_map is not None:
        if set(index_map) != set(source_metadata):
            raise Qwen35CheckpointResumeError("safe-tensor index key set disagrees with shard headers")
        for key, expected_shard in index_map.items():
            if source_metadata[key]["shard"] != expected_shard:
                raise Qwen35CheckpointResumeError(f"safe-tensor index assigns {key!r} to the wrong shard")

    required_sources = set(source_to_targets)
    observed_sources = set(source_metadata)
    if observed_sources != required_sources:
        missing = sorted(required_sources - observed_sources)
        unexpected = sorted(observed_sources - required_sources)
        raise Qwen35CheckpointResumeError(
            f"checkpoint source-key set mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    for source_key, target_keys in source_to_targets.items():
        metadata = source_metadata[source_key]
        if metadata["dtype"] != "F32":
            raise Qwen35CheckpointResumeError(
                f"checkpoint tensor {source_key} has dtype {metadata['dtype']}, expected F32"
            )
        for target_key in target_keys:
            target = target_state[target_key]
            if list(target.shape) != metadata["shape"]:
                raise Qwen35CheckpointResumeError(
                    f"shape mismatch for {target_key}: checkpoint={metadata['shape']}, target={list(target.shape)}"
                )
            if target.dtype != torch.float32:
                raise Qwen35CheckpointResumeError(
                    f"target tensor {target_key} has dtype {target.dtype}, expected torch.float32"
                )
    return source_metadata, observed_by_shard


def load_qwen35_text_checkpoint_for_trainer(
    model: torch.nn.Module, checkpoint_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Restore one Qwen3.5 Trainer checkpoint with exact, audited semantics.

    All checkpoint metadata is validated before the first live tensor copy.
    The model object and every parameter object remain unchanged.
    """

    config_path, checkpoint_config = _validate_model_and_config(model, Path(checkpoint_dir))
    resolved_dir = config_path.parent
    layout, shards, index_map = _discover_layout(resolved_dir)
    input_weight, output_weight = _tied_weights(model)

    target_state = model.state_dict(keep_vars=True)
    if not target_state:
        raise Qwen35CheckpointResumeError("live model state dictionary is empty")
    if set(TIED_TARGET_KEYS) - set(target_state):
        raise Qwen35CheckpointResumeError("live model state dictionary lacks tied embedding aliases")
    object_identity = {key: id(value) for key, value in target_state.items()}
    storage_identity = {key: value.data_ptr() for key, value in target_state.items()}

    source_to_targets: dict[str, list[str]] = {}
    mapping_rows = []
    for target_key in sorted(target_state):
        try:
            source_key = conditional_source_key_for_text_target(target_key)
        except ValueError as error:
            raise Qwen35CheckpointResumeError(str(error)) from error
        source_to_targets.setdefault(source_key, []).append(target_key)
        mapping_rows.append({"source_key": source_key, "target_key": target_key})
    duplicate_mappings = {
        source: frozenset(targets) for source, targets in source_to_targets.items() if len(targets) > 1
    }
    if duplicate_mappings != {TIED_SOURCE_KEY: TIED_TARGET_KEYS}:
        raise Qwen35CheckpointResumeError(f"unexpected many-target source mapping: {dict(duplicate_mappings)}")

    source_metadata, observed_by_shard = _preflight_sources(
        shards=shards, index_map=index_map, source_to_targets=source_to_targets, target_state=target_state
    )

    file_identities = {}
    for name, path in shards.items():
        file_identities[name] = {
            "sha256": _sha256_file(path),
            "header_sha256": _safetensors_header_sha256(path),
            "stat": _stat_identity(path),
            "source_key_count": len(observed_by_shard[name]),
        }
    config_sha256 = _sha256_file(config_path)

    copied_sources = 0
    copied_elements = 0
    with torch.no_grad():
        for shard_name, shard_path in shards.items():
            shard_sources = sorted(
                source for source, metadata in source_metadata.items() if metadata["shard"] == shard_name
            )
            try:
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for source_key in shard_sources:
                        source = handle.get_tensor(source_key)
                        targets = source_to_targets[source_key]
                        representative = target_state[targets[0]]
                        representative.copy_(source, non_blocking=False)
                        expected = source.to(device=representative.device)
                        if not torch.equal(representative.detach(), expected):
                            raise Qwen35CheckpointResumeError(
                                f"post-copy exact-value comparison failed for {source_key}"
                            )
                        for target_key in targets[1:]:
                            if not torch.equal(target_state[target_key].detach(), representative.detach()):
                                raise Qwen35CheckpointResumeError(
                                    f"tied target value mismatch after loading {target_key}"
                                )
                        copied_sources += 1
                        copied_elements += source.numel()
            except Qwen35CheckpointResumeError:
                raise
            except Exception as error:
                raise Qwen35CheckpointResumeError(f"cannot copy safe-tensor shard {shard_name}: {error}") from error

    post_state = model.state_dict(keep_vars=True)
    if set(post_state) != set(target_state):
        raise Qwen35CheckpointResumeError("live model state-key set changed during restoration")
    for key, value in post_state.items():
        if id(value) != object_identity[key] or value.data_ptr() != storage_identity[key]:
            raise Qwen35CheckpointResumeError(f"live tensor identity changed during restoration: {key}")
    post_input_weight, post_output_weight = _tied_weights(model)
    if id(post_input_weight) != id(input_weight) or id(post_output_weight) != id(output_weight):
        raise Qwen35CheckpointResumeError("tied parameter object identity changed during restoration")
    for name, path in shards.items():
        if _stat_identity(path) != file_identities[name]["stat"]:
            raise Qwen35CheckpointResumeError(f"safe-tensor shard changed while loading: {name}")

    source_rows = [
        {
            "source_key": source,
            "targets": sorted(source_to_targets[source]),
            "shard": source_metadata[source]["shard"],
            "shape": source_metadata[source]["shape"],
            "dtype": source_metadata[source]["dtype"],
        }
        for source in sorted(source_metadata)
    ]
    checkpoint_identity = {
        "config_sha256": config_sha256,
        "weight_files": {
            name: {"sha256": value["sha256"], "header_sha256": value["header_sha256"], "size": value["stat"]["size"]}
            for name, value in sorted(file_identities.items())
        },
    }
    return {
        "artifact": "qwen35_strict_trainer_checkpoint_load_audit",
        "schema_version": 1,
        "status": "passed",
        "checkpoint_dir": str(resolved_dir),
        "checkpoint_identity_sha256": _canonical_sha256(checkpoint_identity),
        "config_sha256": config_sha256,
        "config_model_type": checkpoint_config["model_type"],
        "config_architectures": checkpoint_config["architectures"],
        "layout": layout,
        "weight_files": checkpoint_identity["weight_files"],
        "safetensor_header_manifest_sha256": _canonical_sha256(
            {name: value["header_sha256"] for name, value in sorted(file_identities.items())}
        ),
        "source_tensor_count": len(source_metadata),
        "target_state_key_count": len(target_state),
        "unique_target_storage_count": len(set(storage_identity.values())),
        "copied_source_tensor_count": copied_sources,
        "copied_unique_elements": copied_elements,
        "source_dtype": "F32",
        "target_dtype": "torch.float32",
        "mapping_rows_sha256": _canonical_sha256(mapping_rows),
        "source_rows_sha256": _canonical_sha256(source_rows),
        "tied_source_key": TIED_SOURCE_KEY,
        "tied_target_keys": sorted(TIED_TARGET_KEYS),
        "tied_input_output_embeddings_before": True,
        "tied_input_output_embeddings_after": True,
        "parameter_objects_preserved": True,
        "storage_pointers_preserved": True,
        "missing_source_keys": [],
        "unexpected_source_keys": [],
        "upstream_trainer_strict_false_used": False,
        "metadata_preflight_completed_before_copy": True,
        "exact_post_copy_values": True,
    }
