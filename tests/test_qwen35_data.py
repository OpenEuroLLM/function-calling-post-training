import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from scripts.data.convert_sft_data_for_qwen35 import PartWriter
from scripts.data.validate_qwen35_tool_pruning_edge import EXPECTED as SYNTHETIC_EDGE_EXPECTED
from scripts.data.validate_qwen35_tool_pruning_edge import build_synthetic_edge_case, canonical_sha256

from open_instruct import qwen35_data
from open_instruct.qwen35_data import (
    NUMPY_CONTRACT_VERSION,
    ChatTemplateTokenizer,
    Qwen35NumpyPackedDataset,
    Qwen35PackedCollator,
    TokenizedQwen35Example,
    apply_referenced_tool_pruning,
    compute_qwen35_token_feature_row,
    instrument_qwen35_chat_template,
    normalize_qwen35_example,
    plan_referenced_tool_pruning,
    prune_qwen35_tools_to_fit,
    tokenize_qwen35_example_with_referenced_tool_pruning,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "qwen35_fc"


def test_normalize_qwen35_example_parses_arguments_without_losing_ids():
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {}}}}]
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "weather", "content": "sunny"},
        {"role": "assistant", "content": "Sunny."},
    ]

    normalized_messages, normalized_tools = normalize_qwen35_example(messages, tools)

    assert normalized_messages[1]["tool_calls"][0]["id"] == "call-1"
    assert normalized_messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert normalized_messages[2]["tool_call_id"] == "call-1"
    assert normalized_tools == tools
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'


def test_normalize_qwen35_example_preserves_duplicate_tool_definitions():
    tools = [
        {"type": "function", "function": {"name": "f", "description": "first"}},
        {"type": "function", "function": {"name": "f", "description": "second"}},
    ]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ]

    _, normalized_tools = normalize_qwen35_example(messages, tools)

    assert normalized_tools == tools


def _tool(name: str, description: str = "") -> dict:
    return {"type": "function", "function": {"name": name, "description": description}}


def test_referenced_tool_pruning_plan_removes_only_unreferenced_tools_from_end():
    tools = [_tool("a"), _tool("b"), _tool("c"), _tool("d")]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "b", "arguments": {}}}]},
    ]

    plan = plan_referenced_tool_pruning(messages, tools)
    retained, removed = apply_referenced_tool_pruning(tools, plan, removal_count=2)

    assert plan.referenced_tool_names == ("b",)
    assert plan.removable_original_indices == (3, 2, 0)
    assert [tool["function"]["name"] for tool in retained] == ["a", "b"]
    assert [(row.original_index, row.function_name) for row in removed] == [(3, "d"), (2, "c")]
    assert all(len(row.canonical_sha256) == 64 for row in removed)
    assert tools == [_tool("a"), _tool("b"), _tool("c"), _tool("d")]


def test_referenced_tool_pruning_rejects_ambiguous_referenced_name():
    tools = [_tool("f", "first"), _tool("f", "second")]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}]},
    ]
    with pytest.raises(ValueError, match="exactly one definition"):
        plan_referenced_tool_pruning(messages, tools)


def test_explicit_tool_pruning_removes_one_definition_at_a_time_until_fit(monkeypatch):
    tools = [_tool("a"), _tool("b"), _tool("c"), _tool("d")]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "b", "arguments": {}}}]},
    ]

    def fake_tokenize(tokenizer, input_messages, input_tools, **kwargs):
        del tokenizer, input_messages
        length = 20 + 10 * len(input_tools)
        mask = [False] * (length - 1) + [True]
        max_length = kwargs.get("max_seq_length")
        if max_length is not None:
            mask = mask[:max_length]
            length = min(length, max_length)
        return TokenizedQwen35Example(
            input_ids=list(range(length)),
            labels_mask=mask,
            assistant_spans=[(length - 1, length)],
            normalized_messages=[],
            normalized_tools=[dict(tool) for tool in input_tools],
            num_tokens_before_truncation=20 + 10 * len(input_tools),
        )

    monkeypatch.setattr(qwen35_data, "tokenize_qwen35_example", fake_tokenize)
    result = prune_qwen35_tools_to_fit(cast(ChatTemplateTokenizer, object()), messages, tools, max_seq_length=30)

    assert [tool["function"]["name"] for tool in result.retained_tools] == ["b"]
    assert [row.original_index for row in result.removed_tools] == [3, 2, 0]
    assert len(result.tokenized.input_ids) == 30
    assert result.tokenized.trainable_tokens == 1


def test_accepted_renderer_wrapper_changes_only_zero_loss_trigger(monkeypatch):
    tools = [_tool("a"), _tool("b"), _tool("c"), _tool("d")]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "b", "arguments": {}}}]},
    ]

    def fake_tokenize(tokenizer, input_messages, input_tools, **kwargs):
        del tokenizer, input_messages
        length = 20 + 10 * len(input_tools)
        mask = [False] * (length - 1) + [True]
        max_length = kwargs.get("max_seq_length")
        if max_length is not None:
            mask = mask[:max_length]
            length = min(length, max_length)
        return TokenizedQwen35Example(
            input_ids=list(range(length)),
            labels_mask=mask,
            assistant_spans=[(length - 1, length)],
            normalized_messages=[],
            normalized_tools=[dict(tool) for tool in input_tools],
            num_tokens_before_truncation=20 + 10 * len(input_tools),
        )

    monkeypatch.setattr(qwen35_data, "tokenize_qwen35_example", fake_tokenize)
    amended = tokenize_qwen35_example_with_referenced_tool_pruning(
        cast(ChatTemplateTokenizer, object()), messages, tools, max_seq_length=30
    )
    ordinary = tokenize_qwen35_example_with_referenced_tool_pruning(
        cast(ChatTemplateTokenizer, object()), messages, [_tool("b")], max_seq_length=30
    )

    assert amended.amended is True
    assert amended.ordinary_num_tokens_before_truncation == 60
    assert amended.ordinary_num_tokens_after_truncation == 30
    assert amended.ordinary_trainable_tokens_after_truncation == 0
    assert amended.pruning is not None
    assert [row.original_index for row in amended.pruning.removed_tools] == [3, 2, 0]
    assert ordinary.amended is False
    assert ordinary.tokenized.trainable_tokens == 1


@pytest.mark.parametrize("arguments", ["not-json", "[]", "1", None])
def test_normalize_qwen35_example_rejects_non_object_arguments(arguments):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": arguments}}]},
    ]
    with pytest.raises(ValueError, match="arguments"):
        normalize_qwen35_example(messages, tools)


def test_template_instrumentation_changes_no_literal_output_statements():
    template = (
        "prefix\n"
        '{%- elif message.role == "assistant" %}\n'
        "        {{- message.content }}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        '    {%- elif message.role == "tool" %}\n'
        "suffix\n"
    )
    instrumented = instrument_qwen35_chat_template(template)
    assert (
        instrumented.replace("        {%- generation %}\n", "").replace("        {%- endgeneration %}\n", "")
        == template
    )


def test_token_feature_projection_preserves_per_message_and_fc_loss():
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]},
        {"role": "assistant", "content": "two"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]},
        {"role": "assistant", "content": "four"},
        {"role": "assistant", "content": "five"},
    ]
    tokenized = TokenizedQwen35Example(
        input_ids=list(range(20)),
        labels_mask=[
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        assistant_spans=[(1, 5), (5, 8), (8, 11), (11, 15), (15, 20)],
        normalized_messages=messages,
        normalized_tools=[],
        num_tokens_before_truncation=20,
    )
    record = {"source_key": "source", "sample_uid": "a" * 64, "messages": messages}

    row = compute_qwen35_token_feature_row(
        record, tokenized, global_row_number=7, canonical_record_sha256="b" * 64, max_seq_length=13
    )

    assert row["qwen_total_tokens_untruncated"] == 20
    assert row["qwen_total_tokens"] == 13
    assert row["qwen_truncated"] is True
    assert row["assistant_loss_tokens_by_message_untruncated"] == [2, 1, 2, 3, 4]
    assert row["assistant_loss_tokens_by_message"] == [2, 1, 2, 2, 0]
    assert row["qwen_assistant_loss_tokens_untruncated"] == 12
    assert row["qwen_assistant_loss_tokens"] == 7
    assert row["qwen_fc_assistant_loss_tokens_untruncated"] == 4
    assert row["qwen_fc_assistant_loss_tokens"] == 4
    assert row["qwen_first_token_trainable"] is False
    assert row["qwen_last_token_trainable"] is True


def test_token_feature_projection_requires_untruncated_renderer_result():
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    tokenized = TokenizedQwen35Example(
        input_ids=[1, 2],
        labels_mask=[False, True],
        assistant_spans=[(1, 2)],
        normalized_messages=messages,
        normalized_tools=[],
        num_tokens_before_truncation=3,
    )
    with pytest.raises(ValueError, match="untruncated"):
        compute_qwen35_token_feature_row(
            {"source_key": "source", "sample_uid": "a" * 64, "messages": messages},
            tokenized,
            global_row_number=0,
            canonical_record_sha256="b" * 64,
            max_seq_length=2,
        )


def test_token_feature_projection_records_zero_loss_after_right_truncation():
    messages = [{"role": "user", "content": "long context"}, {"role": "assistant", "content": "target"}]
    tokenized = TokenizedQwen35Example(
        input_ids=list(range(8)),
        labels_mask=[False, False, False, False, False, False, True, True],
        assistant_spans=[(5, 8)],
        normalized_messages=messages,
        normalized_tools=[],
        num_tokens_before_truncation=8,
    )
    row = compute_qwen35_token_feature_row(
        {"source_key": "source", "sample_uid": "a" * 64, "messages": messages},
        tokenized,
        global_row_number=0,
        canonical_record_sha256="b" * 64,
        max_seq_length=5,
    )
    assert row["qwen_truncated"] is True
    assert row["qwen_assistant_loss_tokens_untruncated"] == 2
    assert row["qwen_assistant_loss_tokens"] == 0
    assert row["assistant_loss_tokens_by_message"] == [0]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_numpy_packing_resets_boundaries_and_masks_cross_document_label(tmp_path):
    token_ids = np.asarray([10, 11, 12, 20, 21, 30, 31, 32], dtype=np.uint16)
    labels_mask = np.asarray([False, True, True, True, True, True, False, True], dtype=np.bool_)
    token_path = tmp_path / "token_ids_part_0000.npy"
    mask_path = tmp_path / "labels_mask_part_0000.npy"
    boundaries_path = tmp_path / "token_ids_part_0000.csv.gz"
    token_ids.tofile(token_path)
    labels_mask.tofile(mask_path)
    with gzip.open(boundaries_path, "wt") as handle:
        handle.write("0,3\n3,5\n5,8\n")
    manifest = {
        "contract_version": NUMPY_CONTRACT_VERSION,
        "tokenizer": {"directory": "tokenizer"},
        "parts": [
            {
                "token_ids": token_path.name,
                "labels_mask": mask_path.name,
                "boundaries": boundaries_path.name,
                "token_dtype": "uint16",
                "num_tokens": len(token_ids),
                "token_ids_sha256": _sha256(token_path),
                "labels_mask_sha256": _sha256(mask_path),
                "boundaries_sha256": _sha256(boundaries_path),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    dataset = Qwen35NumpyPackedDataset(tmp_path, sequence_length=8, drop_last=True, verify_hashes=True)
    item = dataset[0]

    assert item["input_ids"].tolist() == token_ids.tolist()
    assert item["labels"].tolist() == [-100, 11, 12, -100, 21, -100, -100, 32]
    assert item["position_ids"].tolist() == [0, 1, 2, 0, 1, 0, 1, 2]
    assert item["seq_idx"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert item["cu_seq_lens_q"].tolist() == [0, 3, 5, 8]
    assert dataset.accounting() == {
        "raw_tokens": 8,
        "packed_real_tokens": 8,
        "dropped_tokens": 0,
        "packed_trainable_tokens_before_boundary_mask": 6,
        "boundary_masked_trainable_tokens": 2,
        "effective_trainable_tokens": 4,
        "fixed_sequence_tokens": 8,
        "padding_tokens": 0,
    }

    batch = Qwen35PackedCollator(pad_token_id=0, sequence_length=8, pad_to_sequence_length=False)([item])
    assert cast(torch.Tensor, batch["input_ids"]).shape == (1, 8)
    assert cast(torch.Tensor, batch["labels"]).shape == (1, 8)
    assert batch["max_length_q"] == 3


def test_numpy_reader_rejects_boundary_gaps(tmp_path):
    np.asarray([1, 2, 3], dtype=np.uint8).tofile(tmp_path / "token_ids_part_0000.npy")
    np.asarray([True, True, True], dtype=np.bool_).tofile(tmp_path / "labels_mask_part_0000.npy")
    with gzip.open(tmp_path / "token_ids_part_0000.csv.gz", "wt") as handle:
        handle.write("0,2\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "contract_version": NUMPY_CONTRACT_VERSION,
                "parts": [
                    {
                        "token_ids": "token_ids_part_0000.npy",
                        "labels_mask": "labels_mask_part_0000.npy",
                        "boundaries": "token_ids_part_0000.csv.gz",
                        "token_dtype": "uint8",
                        "num_tokens": 3,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="do not cover every token"):
        Qwen35NumpyPackedDataset(tmp_path, sequence_length=3)


def test_numpy_reader_rejects_internal_boundary_gaps(tmp_path):
    np.asarray([1, 2, 3, 4], dtype=np.uint8).tofile(tmp_path / "token_ids_part_0000.npy")
    np.asarray([False, True, False, True], dtype=np.bool_).tofile(tmp_path / "labels_mask_part_0000.npy")
    with gzip.open(tmp_path / "token_ids_part_0000.csv.gz", "wt") as handle:
        handle.write("0,1\n2,4\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "contract_version": NUMPY_CONTRACT_VERSION,
                "parts": [
                    {
                        "token_ids": "token_ids_part_0000.npy",
                        "labels_mask": "labels_mask_part_0000.npy",
                        "boundaries": "token_ids_part_0000.csv.gz",
                        "token_dtype": "uint8",
                        "num_tokens": 4,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="invalid or overlapping boundary"):
        Qwen35NumpyPackedDataset(tmp_path, sequence_length=4)


def test_atomic_best_fit_packing_never_splits_conversations(tmp_path):
    token_ids = np.asarray([10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25], dtype=np.uint16)
    labels_mask = np.asarray([False, True, True, True, True, True] * 2, dtype=np.bool_)
    token_path = tmp_path / "token_ids_part_0000.npy"
    mask_path = tmp_path / "labels_mask_part_0000.npy"
    boundaries_path = tmp_path / "token_ids_part_0000.csv.gz"
    token_ids.tofile(token_path)
    labels_mask.tofile(mask_path)
    with gzip.open(boundaries_path, "wt") as handle:
        handle.write("0,6\n6,12\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "contract_version": NUMPY_CONTRACT_VERSION,
                "parts": [
                    {
                        "token_ids": token_path.name,
                        "labels_mask": mask_path.name,
                        "boundaries": boundaries_path.name,
                        "token_dtype": "uint16",
                        "num_tokens": len(token_ids),
                    }
                ],
            }
        )
    )

    dataset = Qwen35NumpyPackedDataset(tmp_path, sequence_length=8, drop_last=False)

    assert len(dataset) == 2
    assert [item["input_ids"].tolist() for item in dataset] == [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]
    assert [item["position_ids"].tolist() for item in dataset] == [list(range(6)), list(range(6))]
    assert dataset.accounting() == {
        "raw_tokens": 12,
        "packed_real_tokens": 12,
        "dropped_tokens": 0,
        "packed_trainable_tokens_before_boundary_mask": 10,
        "boundary_masked_trainable_tokens": 0,
        "effective_trainable_tokens": 10,
        "fixed_sequence_tokens": 16,
        "padding_tokens": 4,
    }


def test_part_writer_starts_a_new_part_instead_of_splitting_a_document(tmp_path):
    writer = PartWriter(tmp_path, np.dtype(np.uint16), max_tokens_per_part=5)

    first = writer.write_document([1, 2, 3], [False, True, True])
    second = writer.write_document([4, 5, 6], [False, True, True])
    parts = writer.close()

    assert first == [{"part": 0, "start": 0, "end": 3}]
    assert second == [{"part": 1, "start": 0, "end": 3}]
    assert [part["num_tokens"] for part in parts] == [3, 3]
    for part in parts:
        with gzip.open(tmp_path / part["boundaries"], "rt") as handle:
            assert handle.read() == "0,3\n"


def test_part_writer_rejects_document_larger_than_part_capacity(tmp_path):
    writer = PartWriter(tmp_path, np.dtype(np.uint16), max_tokens_per_part=2)
    with pytest.raises(ValueError, match="never split across parts"):
        writer.write_document([1, 2, 3], [False, True, True])


def test_collator_represents_padding_as_an_isolated_non_loss_document():
    feature = {
        "input_ids": torch.tensor([10, 11, 20]),
        "labels": torch.tensor([-100, 11, -100]),
        "position_ids": torch.tensor([0, 1, 0]),
        "seq_idx": torch.tensor([0, 0, 1], dtype=torch.int32),
        "cu_seq_lens_q": torch.tensor([0, 2, 3], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 3], dtype=torch.int32),
        "max_length_q": torch.tensor(2, dtype=torch.int32),
        "max_length_k": torch.tensor(2, dtype=torch.int32),
    }
    batch = Qwen35PackedCollator(pad_token_id=0, sequence_length=6, pad_to_sequence_length=True)([feature])
    assert cast(torch.Tensor, batch["input_ids"]).tolist() == [[10, 11, 20, 0, 0, 0]]
    assert cast(torch.Tensor, batch["labels"]).tolist() == [[-100, 11, -100, -100, -100, -100]]
    assert cast(torch.Tensor, batch["position_ids"]).tolist() == [[0, 1, 0, 0, 1, 2]]
    assert cast(torch.Tensor, batch["seq_idx"]).tolist() == [[0, 0, 1, 2, 2, 2]]
    assert cast(torch.Tensor, batch["cu_seq_lens_q"]).tolist() == [0, 2, 3, 6]
    assert batch["max_length_q"] == 3


def test_frozen_qwen35_fixture_integrity_and_lineage():
    fixture_path = FIXTURE_DIR / "fixture.jsonl"
    cases_path = FIXTURE_DIR / "cases.jsonl"
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    fixture_rows = [json.loads(line) for line in fixture_path.read_text().splitlines()]
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]

    assert len(fixture_rows) == len(cases) == manifest["fixture_size"] == 128
    assert manifest["frozen"] is True
    assert manifest["coverage_shortfall"] == {}
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == manifest["fixture_jsonl_sha256"]
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == manifest["cases_jsonl_sha256"]
    assert Counter(case["source_key"] for case in cases) == manifest["source_counts"]
    assert [case["fixture_index"] for case in cases] == list(range(128))
    assert len({case["sample_uid"] for case in cases}) == 128

    for row, case in zip(fixture_rows, cases):
        assert set(row) == {"messages", "tools", "dataset", "sample_id"}
        assert row["messages"] and isinstance(row["tools"], list)
        assert case["lineage"]["dolci_roundtrip"] == "exact_parsed_object_match"
        canonical_bytes = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
        assert hashlib.sha256(canonical_bytes).hexdigest() == case["canonical_record_sha256"]


def test_synthetic_native_tool_pruning_edge_construction_is_frozen_separately():
    messages, tools = build_synthetic_edge_case()

    assert len(tools) == 372
    assert tools[0]["function"]["name"] == "tool_000"
    assert tools[-1]["function"]["name"] == "tool_371"
    assert len(tools[-1]["function"]["description"]) == len("padding ") * 26000
    assert messages[1]["tool_calls"][0]["function"]["name"] == "tool_000"
    assert canonical_sha256(messages) == SYNTHETIC_EDGE_EXPECTED["messages_canonical_sha256"]
    assert canonical_sha256(tools) == SYNTHETIC_EDGE_EXPECTED["tools_canonical_sha256"]
