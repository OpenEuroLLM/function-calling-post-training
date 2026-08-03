from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from open_instruct.qwen35_checkpoint_resume import Qwen35CheckpointResumeError, load_qwen35_text_checkpoint_for_trainer
from open_instruct.qwen35_training import conditional_source_key_for_text_target


class TinyQwenBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.norm = nn.LayerNorm(4, bias=False)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, *, tied: bool = True, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5_text")
        self.model = TinyQwenBody().to(dtype=dtype)
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=dtype)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight


def _write_config(path: Path, *, model_type: str = "qwen3_5_text") -> None:
    path.write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "model_type": model_type}) + "\n")


def _source_state(model: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    next_value = 1.0
    for target_key, target in sorted(model.state_dict().items()):
        source_key = conditional_source_key_for_text_target(target_key)
        if source_key in result:
            continue
        values = torch.arange(target.numel(), dtype=torch.float32).reshape(target.shape)
        result[source_key] = (values + next_value).contiguous()
        next_value += target.numel()
    return result


def _write_single_checkpoint(path: Path, source: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    _write_config(path / "config.json")
    save_file(source, path / "model.safetensors")


def _write_indexed_checkpoint(path: Path, source: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    _write_config(path / "config.json")
    keys = sorted(source)
    midpoint = len(keys) // 2
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    first_keys = keys[:midpoint]
    second_keys = keys[midpoint:]
    save_file({key: source[key] for key in first_keys}, path / shard_names[0])
    save_file({key: source[key] for key in second_keys}, path / shard_names[1])
    weight_map = {key: shard_names[0] for key in first_keys} | {key: shard_names[1] for key in second_keys}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}) + "\n")


def _expected_target(source: dict[str, torch.Tensor], target_key: str) -> torch.Tensor:
    return source[conditional_source_key_for_text_target(target_key)]


@pytest.mark.parametrize("indexed", [False, True])
def test_strict_restore_is_exact_and_preserves_tied_parameter_objects(tmp_path: Path, indexed: bool) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    checkpoint = tmp_path / "checkpoint-5"
    (_write_indexed_checkpoint if indexed else _write_single_checkpoint)(checkpoint, source)
    ids_before = {key: id(value) for key, value in model.state_dict(keep_vars=True).items()}
    pointers_before = {key: value.data_ptr() for key, value in model.state_dict(keep_vars=True).items()}

    report = load_qwen35_text_checkpoint_for_trainer(model, checkpoint)

    assert report["status"] == "passed"
    assert report["source_tensor_count"] == 3
    assert report["target_state_key_count"] == 4
    assert report["missing_source_keys"] == []
    assert report["unexpected_source_keys"] == []
    assert report["upstream_trainer_strict_false_used"] is False
    for key, value in model.state_dict(keep_vars=True).items():
        assert torch.equal(value.detach(), _expected_target(source, key))
        assert id(value) == ids_before[key]
        assert value.data_ptr() == pointers_before[key]
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()


@pytest.mark.parametrize("mode", ["missing", "extra", "namespace", "shape", "dtype"])
def test_metadata_mismatch_fails_before_any_parameter_changes(tmp_path: Path, mode: str) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    first_key = sorted(source)[0]
    if mode == "missing":
        del source[first_key]
    elif mode == "extra":
        source["model.language_model.unexpected.weight"] = torch.zeros(1)
    elif mode == "namespace":
        source["model.embed_tokens.weight"] = source.pop("model.language_model.embed_tokens.weight")
    elif mode == "shape":
        source[first_key] = torch.zeros(source[first_key].numel() + 1)
    elif mode == "dtype":
        source[first_key] = source[first_key].to(torch.float16)
    checkpoint = tmp_path / "checkpoint-5"
    _write_single_checkpoint(checkpoint, source)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    with pytest.raises(Qwen35CheckpointResumeError):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_untied_or_non_fp32_target_is_rejected_before_copy(tmp_path: Path) -> None:
    reference = Qwen3_5ForCausalLM()
    checkpoint = tmp_path / "checkpoint-5"
    _write_single_checkpoint(checkpoint, _source_state(reference))
    for model in (Qwen3_5ForCausalLM(tied=False), Qwen3_5ForCausalLM(dtype=torch.float16)):
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with pytest.raises(Qwen35CheckpointResumeError):
            load_qwen35_text_checkpoint_for_trainer(model, checkpoint)
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key])


def test_pickle_model_weights_are_rejected_even_with_safe_weights(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    checkpoint = tmp_path / "checkpoint-5"
    _write_single_checkpoint(checkpoint, _source_state(model))
    (checkpoint / "pytorch_model.bin").write_bytes(b"not a safe model format")
    with pytest.raises(Qwen35CheckpointResumeError, match="pickle"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_symlinked_weight_file_is_rejected(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    real_weights = tmp_path / "real.safetensors"
    save_file(source, real_weights)
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    _write_config(checkpoint / "config.json")
    (checkpoint / "model.safetensors").symlink_to(real_weights)
    with pytest.raises(Qwen35CheckpointResumeError, match="symlink"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


@pytest.mark.parametrize("bad_path", ["../outside.safetensors", "/tmp/outside.safetensors", "a\\b.safetensors"])
def test_unsafe_index_shard_path_is_rejected(tmp_path: Path, bad_path: str) -> None:
    model = Qwen3_5ForCausalLM()
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    _write_config(checkpoint / "config.json")
    key = sorted(_source_state(model))[0]
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: bad_path}}))
    with pytest.raises(Qwen35CheckpointResumeError, match="shard path"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_index_assignment_drift_is_rejected(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    checkpoint = tmp_path / "checkpoint-5"
    _write_indexed_checkpoint(checkpoint, source)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first_key, last_key = sorted(index["weight_map"])[0], sorted(index["weight_map"])[-1]
    index["weight_map"][first_key], index["weight_map"][last_key] = (
        index["weight_map"][last_key],
        index["weight_map"][first_key],
    )
    index_path.write_text(json.dumps(index))
    with pytest.raises(Qwen35CheckpointResumeError, match="wrong shard"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_duplicate_key_across_shards_is_rejected(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    checkpoint = tmp_path / "checkpoint-5"
    _write_indexed_checkpoint(checkpoint, source)
    duplicate_key = sorted(source)[0]
    second = checkpoint / "model-00002-of-00002.safetensors"
    second_source = {key: value for key, value in source.items() if "norm" in key}
    second_source[duplicate_key] = source[duplicate_key]
    save_file(second_source, second)
    with pytest.raises(Qwen35CheckpointResumeError, match="more than one shard"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_simultaneous_single_and_indexed_layouts_are_rejected(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    checkpoint = tmp_path / "checkpoint-5"
    _write_single_checkpoint(checkpoint, source)
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(Qwen35CheckpointResumeError, match="simultaneous"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_unexpected_safe_tensor_file_is_rejected(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    source = _source_state(model)
    checkpoint = tmp_path / "checkpoint-5"
    _write_single_checkpoint(checkpoint, source)
    save_file({"x": torch.zeros(1)}, checkpoint / "extra.safetensors")
    with pytest.raises(Qwen35CheckpointResumeError, match="unexpected safe-tensor"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_duplicate_index_json_member_is_rejected(tmp_path: Path) -> None:
    model = Qwen3_5ForCausalLM()
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    _write_config(checkpoint / "config.json")
    index_text = '{"weight_map": {}, "weight_map": {}}\n'
    (checkpoint / "model.safetensors.index.json").write_text(index_text)
    with pytest.raises(Qwen35CheckpointResumeError, match="duplicate member"):
        load_qwen35_text_checkpoint_for_trainer(model, checkpoint)


def test_trainer_override_has_no_upstream_checkpoint_fallback() -> None:
    path = Path("scripts/train/qwen35/train_qwen35_sft.py")
    tree = ast.parse(path.read_text())
    trainer_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen35ExactTrainer"
    )
    method = next(
        node
        for node in trainer_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_load_from_checkpoint"
    )
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "load_qwen35_text_checkpoint_for_trainer" for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "_load_from_checkpoint" for call in calls
    )
