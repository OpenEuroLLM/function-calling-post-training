"""Qwen3.5-native function-calling tokenization and packed NumPy loading.

The canonical examples accepted here retain OpenAI-style ``messages`` and
``tools``.  They are rendered with the tokenizer's own chat template; this
module deliberately does not consume the lossy Dolci representation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch.utils.data import Dataset

NUMPY_CONTRACT_VERSION = "open-instruct-qwen35-numpy-v2"
REFERENCED_TOOL_PRUNING_AMENDMENT_ID = "v3-semantic-causal-qwen35-32k-referenced-tool-pruning-r1"
ASSISTANT_ROLE = "assistant"
_ASSISTANT_BRANCH = '{%- elif message.role == "assistant" %}\n'
_ASSISTANT_BRANCH_END = "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
_ASSISTANT_HEADER = "<|im_start|>assistant\n"
_EMPTY_THINKING_PREFIX = "<think>\n\n</think>\n\n"


class ChatTemplateTokenizer(Protocol):
    """Small tokenizer protocol used by the renderer and its unit tests."""

    chat_template: str | None

    def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TokenizedQwen35Example:
    input_ids: list[int]
    labels_mask: list[bool]
    assistant_spans: list[tuple[int, int]]
    normalized_messages: list[dict[str, Any]]
    normalized_tools: list[dict[str, Any]]
    num_tokens_before_truncation: int

    @property
    def trainable_tokens(self) -> int:
        return sum(self.labels_mask)


@dataclass(frozen=True)
class ReferencedToolPruningPlan:
    """Outcome-blind plan for the accepted 32K tool-prefix amendment."""

    referenced_tool_names: tuple[str, ...]
    removable_original_indices: tuple[int, ...]


@dataclass(frozen=True)
class RemovedToolDefinition:
    original_index: int
    function_name: str
    canonical_sha256: str


@dataclass(frozen=True)
class PrunedQwen35Example:
    tokenized: TokenizedQwen35Example
    retained_tools: list[dict[str, Any]]
    removed_tools: tuple[RemovedToolDefinition, ...]


@dataclass(frozen=True)
class AmendedQwen35Render:
    """One native render plus the optional accepted zero-loss amendment."""

    tokenized: TokenizedQwen35Example
    pruning: PrunedQwen35Example | None
    ordinary_num_tokens_before_truncation: int
    ordinary_num_tokens_after_truncation: int
    ordinary_trainable_tokens_after_truncation: int

    @property
    def amended(self) -> bool:
        return self.pruning is not None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def instrument_qwen35_chat_template(chat_template: str) -> str:
    """Add Jinja generation blocks without changing rendered text.

    Qwen3.5's native template currently has no ``generation`` blocks, so
    Transformers cannot return assistant masks directly.  We insert a block
    around the existing assistant branch in memory.  Callers must retain the
    render-identity assertion in :func:`tokenize_qwen35_example`.
    """

    if chat_template.count(_ASSISTANT_BRANCH) != 1:
        raise ValueError("Qwen3.5 template must contain exactly one assistant branch anchor")
    if chat_template.count(_ASSISTANT_BRANCH_END) != 1:
        raise ValueError("Qwen3.5 template must contain exactly one assistant branch-end anchor")
    instrumented = chat_template.replace(_ASSISTANT_BRANCH, _ASSISTANT_BRANCH + "        {%- generation %}\n", 1)
    return instrumented.replace(
        _ASSISTANT_BRANCH_END,
        "        {{- '<|im_end|>\\n' }}\n        {%- endgeneration %}\n    {%- elif message.role == \"tool\" %}",
        1,
    )


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object, got {type(value).__name__}")
    return dict(value)


def normalize_qwen35_example(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate canonical FC data and make only renderer-required copies.

    OpenAI tool-call arguments are commonly serialized JSON strings.  The
    Qwen3.5 template iterates the argument mapping, so those strings are parsed
    into dictionaries in the rendering copy.  IDs, message boundaries, roles,
    tool results, and tool schemas are retained.
    """

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise ValueError("tools must be a sequence")

    normalized_tools: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    for tool_index, raw_tool in enumerate(tools):
        if not isinstance(raw_tool, Mapping):
            raise ValueError(f"tools[{tool_index}] must be an object")
        tool = dict(raw_tool)
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, Mapping):
            raise ValueError(f"tools[{tool_index}] must be an OpenAI function tool")
        function = dict(function)
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tools[{tool_index}].function.name must be a non-empty string")
        # Preserve duplicate source definitions verbatim. Deduplicating or
        # renaming them would change the canonical C00 population, while the
        # native Qwen template can render the ordered list exactly as supplied.
        tool_names.add(name)
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError(f"tools[{tool_index}].function.parameters must be an object")
        tool["function"] = function
        normalized_tools.append(tool)

    normalized_messages: list[dict[str, Any]] = []
    assistant_call_ids: set[str] = set()
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"messages[{message_index}] must be an object")
        message = dict(raw_message)
        role = message.get("role")
        if role not in {"system", "user", ASSISTANT_ROLE, "tool"}:
            raise ValueError(f"messages[{message_index}].role is invalid: {role!r}")
        if role == "system" and message_index != 0:
            raise ValueError("system messages are only valid at index 0 in the Qwen3.5 template")

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if role != ASSISTANT_ROLE:
                raise ValueError(f"messages[{message_index}].tool_calls requires assistant role")
            if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
                raise ValueError(f"messages[{message_index}].tool_calls must be a sequence")
            normalized_calls: list[dict[str, Any]] = []
            for call_index, raw_call in enumerate(tool_calls):
                if not isinstance(raw_call, Mapping):
                    raise ValueError(f"messages[{message_index}].tool_calls[{call_index}] must be an object")
                call = dict(raw_call)
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise ValueError(f"messages[{message_index}].tool_calls[{call_index}].function must be an object")
                function = dict(function)
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"messages[{message_index}].tool_calls[{call_index}].function.name is invalid")
                if tool_names and name not in tool_names:
                    raise ValueError(f"tool call references undefined tool {name!r}")
                function["arguments"] = _json_object(
                    function.get("arguments", {}),
                    context=f"messages[{message_index}].tool_calls[{call_index}].function.arguments",
                )
                call_id = call.get("id")
                if call_id is not None:
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError(f"messages[{message_index}].tool_calls[{call_index}].id is invalid")
                    if call_id in assistant_call_ids:
                        raise ValueError(f"duplicate tool call id {call_id!r}")
                    assistant_call_ids.add(call_id)
                call["function"] = function
                normalized_calls.append(call)
            message["tool_calls"] = normalized_calls

        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id is not None and (not isinstance(call_id, str) or not call_id):
                raise ValueError(f"messages[{message_index}].tool_call_id is invalid")
            if call_id is not None and assistant_call_ids and call_id not in assistant_call_ids:
                raise ValueError(f"tool result references unknown prior call id {call_id!r}")

        normalized_messages.append(message)

    if not any(message["role"] == "user" for message in normalized_messages):
        raise ValueError("Qwen3.5 chat template requires at least one user query")
    return normalized_messages, normalized_tools


def plan_referenced_tool_pruning(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
) -> ReferencedToolPruningPlan:
    """Plan the accepted end-pruning of unreferenced tool definitions.

    Every referenced name must resolve to exactly one source definition. The
    removable order is descending original index, so applying any prefix of the
    plan preserves the relative order and exact content of retained tools.
    """
    normalized_messages, normalized_tools = normalize_qwen35_example(messages, tools)
    referenced_in_order: list[str] = []
    referenced: set[str] = set()
    for message in normalized_messages:
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            if name not in referenced:
                referenced.add(name)
                referenced_in_order.append(name)
    definition_counts = Counter(tool["function"]["name"] for tool in normalized_tools)
    ambiguous = sorted(name for name in referenced if definition_counts[name] != 1)
    if ambiguous:
        raise ValueError(f"referenced tool names must resolve to exactly one definition: {ambiguous}")
    removable = tuple(
        index
        for index in range(len(normalized_tools) - 1, -1, -1)
        if normalized_tools[index]["function"]["name"] not in referenced
    )
    return ReferencedToolPruningPlan(
        referenced_tool_names=tuple(referenced_in_order), removable_original_indices=removable
    )


def apply_referenced_tool_pruning(
    tools: Sequence[Mapping[str, Any]], plan: ReferencedToolPruningPlan, *, removal_count: int
) -> tuple[list[dict[str, Any]], tuple[RemovedToolDefinition, ...]]:
    """Apply a prefix of a validated pruning plan and emit its exact ledger."""
    if not 0 <= removal_count <= len(plan.removable_original_indices):
        raise ValueError("removal_count is outside the pruning plan")
    copied_tools = [dict(tool) for tool in tools]
    selected_indices = plan.removable_original_indices[:removal_count]
    selected = set(selected_indices)
    retained = [tool for index, tool in enumerate(copied_tools) if index not in selected]
    removed = tuple(
        RemovedToolDefinition(
            original_index=index,
            function_name=str(copied_tools[index]["function"]["name"]),
            canonical_sha256=_canonical_json_sha256(copied_tools[index]),
        )
        for index in selected_indices
    )
    return retained, removed


def prune_qwen35_tools_to_fit(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    max_seq_length: int,
    assistant_loss_mask: Sequence[bool] | None = None,
) -> PrunedQwen35Example:
    """Implement, but do not implicitly enable, the accepted 32K amendment.

    This function is deliberately separate from :func:`tokenize_qwen35_example`.
    Callers must explicitly invoke it and record the returned ledger. It rejects
    ordinary examples and applies only to the positive-untruncated/zero-after-
    truncation condition described by the pre-outcome amendment.
    """
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    original = tokenize_qwen35_example(tokenizer, messages, tools, assistant_loss_mask=assistant_loss_mask)
    truncated = tokenize_qwen35_example(
        tokenizer, messages, tools, max_seq_length=max_seq_length, assistant_loss_mask=assistant_loss_mask
    )
    if original.trainable_tokens <= 0 or truncated.trainable_tokens != 0:
        raise ValueError(
            "referenced-tool pruning is restricted to positive-untruncated/zero-truncated assistant-loss examples"
        )
    plan = plan_referenced_tool_pruning(messages, tools)
    for removal_count in range(1, len(plan.removable_original_indices) + 1):
        retained, removed = apply_referenced_tool_pruning(tools, plan, removal_count=removal_count)
        tokenized = tokenize_qwen35_example(tokenizer, messages, retained, assistant_loss_mask=assistant_loss_mask)
        if len(tokenized.input_ids) > max_seq_length:
            continue
        if tokenized.trainable_tokens <= 0:
            raise RuntimeError("fitted pruned render has no trainable assistant token")
        retained_names = Counter(tool["function"]["name"] for tool in retained)
        if any(retained_names[name] != 1 for name in plan.referenced_tool_names):
            raise RuntimeError("pruned render did not retain exactly one referenced tool")
        return PrunedQwen35Example(tokenized=tokenized, retained_tools=retained, removed_tools=removed)
    raise ValueError(f"removing every unreferenced tool did not fit the native render within {max_seq_length} tokens")


def tokenize_qwen35_example_with_referenced_tool_pruning(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    max_seq_length: int,
    assistant_loss_mask: Sequence[bool] | None = None,
) -> AmendedQwen35Render:
    """Apply the amendment only when ordinary right truncation loses all supervision."""

    untruncated = tokenize_qwen35_example(
        tokenizer, messages, tools, assistant_loss_mask=assistant_loss_mask, enable_thinking=False
    )
    ordinary = right_truncate_qwen35_example(untruncated, max_seq_length)
    if ordinary.trainable_tokens > 0:
        return AmendedQwen35Render(
            tokenized=ordinary,
            pruning=None,
            ordinary_num_tokens_before_truncation=ordinary.num_tokens_before_truncation,
            ordinary_num_tokens_after_truncation=len(ordinary.input_ids),
            ordinary_trainable_tokens_after_truncation=ordinary.trainable_tokens,
        )

    pruned = prune_qwen35_tools_to_fit(
        tokenizer, messages, tools, max_seq_length=max_seq_length, assistant_loss_mask=assistant_loss_mask
    )
    return AmendedQwen35Render(
        tokenized=pruned.tokenized,
        pruning=pruned,
        ordinary_num_tokens_before_truncation=ordinary.num_tokens_before_truncation,
        ordinary_num_tokens_after_truncation=len(ordinary.input_ids),
        ordinary_trainable_tokens_after_truncation=ordinary.trainable_tokens,
    )


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"tokenizer mapping has no input_ids key: {sorted(value)}")
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized conversation")
        value = value[0]
    return [int(item) for item in value]


def _contiguous_spans(mask: Sequence[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate([*mask, False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _token_ids(tokenizer: ChatTemplateTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return _as_int_list(encoded["input_ids"])


def _split_adjacent_assistant_spans(
    input_ids: Sequence[int], raw_spans: Sequence[tuple[int, int]], header_ids: Sequence[int], expected_count: int
) -> list[tuple[int, int]]:
    """Split generation spans merged across consecutive assistant messages."""

    spans: list[tuple[int, int]] = []
    for raw_start, raw_end in raw_spans:
        segment = input_ids[raw_start:raw_end]
        header_offsets = [
            offset
            for offset in range(0, len(segment) - len(header_ids) + 1)
            if segment[offset : offset + len(header_ids)] == list(header_ids)
        ]
        if not header_offsets or header_offsets[0] != 0:
            raise RuntimeError("assistant generation span does not start with the native assistant header")
        boundaries = [raw_start + offset for offset in header_offsets] + [raw_end]
        spans.extend(zip(boundaries[:-1], boundaries[1:]))
    if len(spans) != expected_count:
        raise RuntimeError(
            f"instrumented template identified {len(spans)} assistant headers, expected {expected_count}"
        )
    return spans


def tokenize_qwen35_example(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    max_seq_length: int | None = None,
    assistant_loss_mask: Sequence[bool] | None = None,
    enable_thinking: bool = False,
) -> TokenizedQwen35Example:
    """Render one canonical example and derive an exact assistant-token mask."""

    if enable_thinking:
        raise ValueError("this text-only FC data path is preregistered with enable_thinking=False")
    if not tokenizer.chat_template:
        raise ValueError("tokenizer does not provide a native chat template")
    normalized_messages, normalized_tools = normalize_qwen35_example(messages, tools)
    assistant_count = sum(message["role"] == ASSISTANT_ROLE for message in normalized_messages)
    if assistant_loss_mask is None:
        assistant_loss_mask = [True] * assistant_count
    if len(assistant_loss_mask) != assistant_count:
        raise ValueError(f"assistant_loss_mask has {len(assistant_loss_mask)} entries, expected {assistant_count}")
    if not all(isinstance(value, (bool, np.bool_)) for value in assistant_loss_mask):
        raise ValueError("assistant_loss_mask values must be booleans")

    template_kwargs = {
        "tools": normalized_tools,
        "enable_thinking": False,
        "add_generation_prompt": False,
        "tokenize": True,
    }
    native_ids = _as_int_list(tokenizer.apply_chat_template(normalized_messages, **template_kwargs))
    instrumented_template = instrument_qwen35_chat_template(tokenizer.chat_template)
    instrumented = tokenizer.apply_chat_template(
        normalized_messages,
        **template_kwargs,
        chat_template=instrumented_template,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    instrumented_ids = _as_int_list(instrumented["input_ids"])
    if instrumented_ids != native_ids:
        raise RuntimeError("assistant-mask instrumentation changed Qwen3.5 native rendered token IDs")
    raw_mask = instrumented.get("assistant_masks")
    if raw_mask is None:
        raw_mask = instrumented.get("assistant_mask")
    if raw_mask is None:
        raise RuntimeError("Transformers did not return assistant masks for the instrumented template")
    labels_mask = [bool(value) for value in _as_int_list(raw_mask)]
    if len(labels_mask) != len(native_ids):
        raise RuntimeError("assistant mask and input IDs have different lengths")

    header_ids = _token_ids(tokenizer, _ASSISTANT_HEADER)
    header_and_empty_thinking_ids = _token_ids(tokenizer, _ASSISTANT_HEADER + _EMPTY_THINKING_PREFIX)
    spans = _split_adjacent_assistant_spans(native_ids, _contiguous_spans(labels_mask), header_ids, assistant_count)
    for enabled, (start, end) in zip(assistant_loss_mask, spans):
        if not enabled:
            labels_mask[start:end] = [False] * (end - start)
            continue
        segment = native_ids[start:end]
        prefix_length = 0
        if segment[: len(header_and_empty_thinking_ids)] == header_and_empty_thinking_ids:
            prefix_length = len(header_and_empty_thinking_ids)
        elif segment[: len(header_ids)] == header_ids:
            prefix_length = len(header_ids)
        else:
            raise RuntimeError("assistant span does not start with the Qwen3.5 assistant header")
        labels_mask[start : start + prefix_length] = [False] * prefix_length

    num_tokens_before_truncation = len(native_ids)
    if max_seq_length is not None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        native_ids = native_ids[:max_seq_length]
        labels_mask = labels_mask[:max_seq_length]
        spans = [(start, min(end, max_seq_length)) for start, end in spans if start < max_seq_length]

    return TokenizedQwen35Example(
        input_ids=native_ids,
        labels_mask=labels_mask,
        assistant_spans=spans,
        normalized_messages=normalized_messages,
        normalized_tools=normalized_tools,
        num_tokens_before_truncation=num_tokens_before_truncation,
    )


def right_truncate_qwen35_example(tokenized: TokenizedQwen35Example, max_seq_length: int) -> TokenizedQwen35Example:
    """Right-truncate an already rendered native example without rerendering."""

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    if len(tokenized.input_ids) != len(tokenized.labels_mask):
        raise ValueError("assistant mask and input IDs have different lengths")
    if tokenized.num_tokens_before_truncation != len(tokenized.input_ids):
        raise ValueError("right_truncate_qwen35_example requires an untruncated input")
    return TokenizedQwen35Example(
        input_ids=tokenized.input_ids[:max_seq_length],
        labels_mask=tokenized.labels_mask[:max_seq_length],
        assistant_spans=[
            (start, min(end, max_seq_length)) for start, end in tokenized.assistant_spans if start < max_seq_length
        ],
        normalized_messages=tokenized.normalized_messages,
        normalized_tools=tokenized.normalized_tools,
        num_tokens_before_truncation=tokenized.num_tokens_before_truncation,
    )


def compute_qwen35_token_feature_row(
    record: Mapping[str, Any],
    tokenized: TokenizedQwen35Example,
    *,
    global_row_number: int,
    canonical_record_sha256: str,
    max_seq_length: int,
) -> dict[str, Any]:
    """Project one untruncated native render into causal matching features.

    This pure projection lives in the Python-3.12 Qwen runtime. The resulting
    typed Parquet artifact is joined to Python-3.14 fcanalysis features later;
    neither runtime imports the other.
    """

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    messages = record.get("messages")
    source_key = record.get("source_key")
    sample_uid = record.get("sample_uid")
    if not isinstance(messages, list) or not messages:
        raise ValueError("canonical record must contain non-empty messages")
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("canonical record has no source_key")
    if not isinstance(sample_uid, str) or len(sample_uid) != 64:
        raise ValueError("canonical record has invalid sample_uid")
    if len(canonical_record_sha256) != 64:
        raise ValueError("invalid canonical_record_sha256")
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and cast(dict[str, Any], message).get("role") == ASSISTANT_ROLE
    ]
    if len(tokenized.assistant_spans) != len(assistant_indices):
        raise ValueError(
            f"native renderer returned {len(tokenized.assistant_spans)} assistant spans for "
            f"{len(assistant_indices)} assistant messages"
        )
    if len(tokenized.input_ids) != len(tokenized.labels_mask):
        raise ValueError("native token IDs and label mask lengths differ")
    if tokenized.num_tokens_before_truncation != len(tokenized.input_ids):
        raise ValueError("token features require an untruncated renderer result")

    truncated_length = min(len(tokenized.input_ids), max_seq_length)
    untruncated_by_message: list[int] = []
    truncated_by_message: list[int] = []
    previous_end = 0
    for start, end in tokenized.assistant_spans:
        if start < previous_end or end <= start or end > len(tokenized.labels_mask):
            raise ValueError(f"invalid or overlapping assistant span {(start, end)}")
        untruncated_by_message.append(sum(bool(value) for value in tokenized.labels_mask[start:end]))
        clipped_start = min(start, truncated_length)
        clipped_end = min(end, truncated_length)
        truncated_by_message.append(sum(bool(value) for value in tokenized.labels_mask[clipped_start:clipped_end]))
        previous_end = end

    fc_positions = [
        ordinal for ordinal, message_index in enumerate(assistant_indices) if messages[message_index].get("tool_calls")
    ]
    truncated_mask = tokenized.labels_mask[:truncated_length]
    return {
        "schema_version": 1,
        "global_row_number": global_row_number,
        "source_key": source_key,
        "sample_uid": sample_uid,
        "canonical_record_sha256": canonical_record_sha256,
        "max_seq_length": max_seq_length,
        "qwen_total_tokens_untruncated": len(tokenized.input_ids),
        "qwen_total_tokens": truncated_length,
        "qwen_truncated": len(tokenized.input_ids) > max_seq_length,
        "qwen_assistant_loss_tokens_untruncated": sum(bool(value) for value in tokenized.labels_mask),
        "qwen_assistant_loss_tokens": sum(bool(value) for value in truncated_mask),
        "qwen_fc_assistant_loss_tokens_untruncated": sum(
            untruncated_by_message[position] for position in fc_positions
        ),
        "qwen_fc_assistant_loss_tokens": sum(truncated_by_message[position] for position in fc_positions),
        "qwen_first_token_trainable": bool(truncated_mask[0]) if truncated_mask else False,
        "qwen_last_token_trainable": bool(truncated_mask[-1]) if truncated_mask else False,
        "assistant_message_indices": assistant_indices,
        "assistant_loss_tokens_by_message_untruncated": untruncated_by_message,
        "assistant_loss_tokens_by_message": truncated_by_message,
    }


@dataclass(frozen=True)
class NumpyPart:
    token_path: Path
    labels_mask_path: Path
    boundaries_path: Path
    token_dtype: np.dtype
    num_tokens: int


@dataclass(frozen=True)
class DocumentSlice:
    part_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Qwen35PackMetadata:
    """Stable identity and exact accounting for one physical pack."""

    pack_index: int
    pack_uid: str
    document_uids: tuple[str, ...]
    document_lengths: tuple[int, ...]
    real_tokens: int
    assistant_targets: int
    padding_tokens: int
    attention_length_squared: int


class Qwen35NumpyPackedDataset(Dataset):
    """Map-style, deterministic fixed-sequence loader for the NumPy contract.

    Packs preserve the CSV document boundaries.  Position IDs and ``seq_idx``
    restart at every boundary, and the first label in every document is masked
    to prevent a cross-document next-token target.
    """

    def __init__(
        self, data_dir: str | Path, sequence_length: int, *, drop_last: bool = True, verify_hashes: bool = False
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.drop_last = drop_last
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing Qwen3.5 NumPy manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("contract_version") != NUMPY_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported contract {self.manifest.get('contract_version')!r}; expected {NUMPY_CONTRACT_VERSION!r}"
            )
        self.parts = self._load_parts(verify_hashes=verify_hashes)
        self._token_maps = [
            np.memmap(part.token_path, mode="r", dtype=part.token_dtype, shape=(part.num_tokens,))
            for part in self.parts
        ]
        self._mask_maps = [
            np.memmap(part.labels_mask_path, mode="r", dtype=np.bool_, shape=(part.num_tokens,)) for part in self.parts
        ]
        documents = self._load_document_slices()
        self.packs = self._build_packs(documents)
        self._document_uids = self._load_document_uids(documents, verify_hashes=verify_hashes)
        self._accounting = self._compute_accounting()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_parts(self, *, verify_hashes: bool) -> list[NumpyPart]:
        parts: list[NumpyPart] = []
        for index, raw_part in enumerate(self.manifest.get("parts", [])):
            token_path = self.data_dir / raw_part["token_ids"]
            mask_path = self.data_dir / raw_part["labels_mask"]
            boundaries_path = self.data_dir / raw_part["boundaries"]
            for path in (token_path, mask_path, boundaries_path):
                if not path.is_file():
                    raise FileNotFoundError(f"manifest part {index} references missing file {path}")
            dtype = np.dtype(raw_part["token_dtype"])
            num_tokens = int(raw_part["num_tokens"])
            if token_path.stat().st_size != num_tokens * dtype.itemsize:
                raise ValueError(f"token file size mismatch: {token_path}")
            if mask_path.stat().st_size != num_tokens * np.dtype(np.bool_).itemsize:
                raise ValueError(f"label-mask file size mismatch: {mask_path}")
            if verify_hashes:
                for key, path in (
                    ("token_ids_sha256", token_path),
                    ("labels_mask_sha256", mask_path),
                    ("boundaries_sha256", boundaries_path),
                ):
                    if self._sha256_file(path) != raw_part[key]:
                        raise ValueError(f"SHA-256 mismatch for {path}")
            parts.append(NumpyPart(token_path, mask_path, boundaries_path, dtype, num_tokens))
        if not parts:
            raise ValueError("manifest has no NumPy parts")
        return parts

    def _load_document_slices(self) -> list[DocumentSlice]:
        documents: list[DocumentSlice] = []
        for part_index, part in enumerate(self.parts):
            previous_end = 0
            with gzip.open(part.boundaries_path, "rt", newline="") as handle:
                for row_index, row in enumerate(csv.reader(handle)):
                    if len(row) != 2:
                        raise ValueError(f"invalid boundary row {row_index} in {part.boundaries_path}")
                    start, end = (int(value) for value in row)
                    if start != previous_end or end <= start or end > part.num_tokens:
                        raise ValueError(f"invalid or overlapping boundary {(start, end)} in {part.boundaries_path}")
                    documents.append(DocumentSlice(part_index, start, end))
                    previous_end = end
            if previous_end != part.num_tokens:
                raise ValueError(f"boundaries do not cover every token in {part.boundaries_path}")
        return documents

    def _load_document_uids(
        self, documents: Sequence[DocumentSlice], *, verify_hashes: bool
    ) -> dict[DocumentSlice, str]:
        """Load stable sample UIDs and prove that the document index matches boundaries."""

        index_name = self.manifest.get("documents_index")
        if not index_name:
            return {document: f"part-{document.part_index}:{document.start}:{document.end}" for document in documents}
        index_path = self.data_dir / str(index_name)
        if not index_path.is_file():
            raise FileNotFoundError(f"manifest references missing document index: {index_path}")
        if verify_hashes:
            expected_hash = self.manifest.get("documents_index_sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError("manifest has no valid documents_index_sha256")
            if self._sha256_file(index_path) != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {index_path}")

        rows: dict[DocumentSlice, str] = {}
        opener = gzip.open if index_path.suffix == ".gz" else Path.open
        with opener(index_path, "rt") as handle:
            for row_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                segments = row.get("segments")
                if not isinstance(segments, list) or len(segments) != 1:
                    raise ValueError(f"document-index row {row_number} must contain exactly one atomic segment")
                segment = segments[0]
                document = DocumentSlice(int(segment["part"]), int(segment["start"]), int(segment["end"]))
                sample_uid = row.get("sample_uid")
                if not isinstance(sample_uid, str) or not sample_uid:
                    raise ValueError(f"document-index row {row_number} has no stable sample_uid")
                if document in rows:
                    raise ValueError(f"duplicate document segment in index: {document}")
                rows[document] = sample_uid

        expected = set(documents)
        observed = set(rows)
        if expected != observed:
            raise ValueError(
                "document index and boundary metadata disagree: "
                f"missing={len(expected - observed)}, unexpected={len(observed - expected)}"
            )
        if len(set(rows.values())) != len(rows):
            raise ValueError("document index contains duplicate sample_uid values")
        return rows

    def _build_packs(self, documents: Sequence[DocumentSlice]) -> list[tuple[DocumentSlice, ...]]:
        """Best-fit-decreasing packing without splitting any conversation.

        A conversation has already been right-truncated by the converter to at
        most ``sequence_length``. Splitting it again at a pack boundary would
        discard its prefix context, reset positions mid-conversation, and make
        accounting depend on preceding document lengths. Integer-capacity
        buckets make deterministic best-fit selection efficient at 32K.
        """

        oversized = [document.length for document in documents if document.length > self.sequence_length]
        if oversized:
            raise ValueError(
                f"{len(oversized)} documents exceed sequence_length={self.sequence_length}; "
                f"largest={max(oversized)}; truncate during conversion"
            )

        ordered = sorted(enumerate(documents), key=lambda item: (-item[1].length, item[0]))
        mutable_packs: list[list[tuple[int, DocumentSlice]]] = []
        remaining_by_capacity: list[list[int]] = [[] for _ in range(self.sequence_length + 1)]
        available_capacities = 0

        for ordinal, document in ordered:
            eligible = available_capacities >> document.length
            if eligible:
                lowest = eligible & -eligible
                capacity = document.length + lowest.bit_length() - 1
                pack_index = remaining_by_capacity[capacity].pop()
                if not remaining_by_capacity[capacity]:
                    available_capacities &= ~(1 << capacity)
                mutable_packs[pack_index].append((ordinal, document))
                remaining = capacity - document.length
            else:
                pack_index = len(mutable_packs)
                mutable_packs.append([(ordinal, document)])
                remaining = self.sequence_length - document.length

            if remaining:
                remaining_by_capacity[remaining].append(pack_index)
                available_capacities |= 1 << remaining

        indexed_packs = [
            (min(ordinal for ordinal, _ in pack), tuple(document for _, document in sorted(pack)))
            for pack in mutable_packs
        ]
        indexed_packs.sort(key=lambda item: item[0])
        packs = [pack for _, pack in indexed_packs]
        if self.drop_last:
            packs = [pack for pack in packs if sum(document.length for document in pack) == self.sequence_length]
        if not packs:
            raise ValueError("dataset produced no packs; use drop_last=False for a small fixture")
        return packs

    def __len__(self) -> int:
        return len(self.packs)

    def _compute_accounting(self) -> dict[str, int]:
        packed_real_tokens = 0
        packed_trainable_before_boundary_mask = 0
        boundary_masked_trainable_tokens = 0
        pack_metadata: list[Qwen35PackMetadata] = []
        for pack_index, pack in enumerate(self.packs):
            pack_real_tokens = 0
            pack_assistant_targets = 0
            for piece in pack:
                piece_mask = self._mask_maps[piece.part_index][piece.start : piece.end]
                packed_real_tokens += piece.length
                pack_real_tokens += piece.length
                packed_trainable_before_boundary_mask += int(np.count_nonzero(piece_mask))
                boundary_masked_trainable_tokens += bool(piece_mask[0])
                pack_assistant_targets += int(np.count_nonzero(piece_mask)) - int(bool(piece_mask[0]))
            padding_tokens = self.sequence_length - pack_real_tokens
            document_uids = tuple(self._document_uids[piece] for piece in pack)
            identity = json.dumps(
                {
                    "contract_version": NUMPY_CONTRACT_VERSION,
                    "sequence_length": self.sequence_length,
                    "document_uids": document_uids,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            pack_metadata.append(
                Qwen35PackMetadata(
                    pack_index=pack_index,
                    pack_uid=hashlib.sha256(identity).hexdigest(),
                    document_uids=document_uids,
                    document_lengths=tuple(piece.length for piece in pack),
                    real_tokens=pack_real_tokens,
                    assistant_targets=pack_assistant_targets,
                    padding_tokens=padding_tokens,
                    attention_length_squared=(
                        sum(piece.length**2 for piece in pack) + (padding_tokens**2 if padding_tokens else 0)
                    ),
                )
            )
        self._pack_metadata = tuple(pack_metadata)
        return {
            "raw_tokens": sum(part.num_tokens for part in self.parts),
            "packed_real_tokens": packed_real_tokens,
            "dropped_tokens": sum(part.num_tokens for part in self.parts) - packed_real_tokens,
            "packed_trainable_tokens_before_boundary_mask": packed_trainable_before_boundary_mask,
            "boundary_masked_trainable_tokens": boundary_masked_trainable_tokens,
            "effective_trainable_tokens": (packed_trainable_before_boundary_mask - boundary_masked_trainable_tokens),
            "fixed_sequence_tokens": len(self.packs) * self.sequence_length,
            "padding_tokens": len(self.packs) * self.sequence_length - packed_real_tokens,
        }

    def accounting(self) -> dict[str, int]:
        return dict(self._accounting)

    def pack_metadata(self, index: int) -> Qwen35PackMetadata:
        return self._pack_metadata[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pieces = self.packs[index]
        token_arrays: list[np.ndarray] = []
        mask_arrays: list[np.ndarray] = []
        position_arrays: list[np.ndarray] = []
        seq_arrays: list[np.ndarray] = []
        cumulative_lengths = [0]
        for sequence_index, piece in enumerate(pieces):
            token_arrays.append(np.asarray(self._token_maps[piece.part_index][piece.start : piece.end]))
            labels_mask = np.asarray(self._mask_maps[piece.part_index][piece.start : piece.end], dtype=np.bool_).copy()
            labels_mask[0] = False
            mask_arrays.append(labels_mask)
            position_arrays.append(np.arange(piece.length, dtype=np.int64))
            seq_arrays.append(np.full(piece.length, sequence_index, dtype=np.int32))
            cumulative_lengths.append(cumulative_lengths[-1] + piece.length)

        input_ids = np.concatenate(token_arrays).astype(np.int64, copy=False)
        labels_mask = np.concatenate(mask_arrays)
        labels = input_ids.copy()
        labels[~labels_mask] = -100
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "position_ids": torch.from_numpy(np.concatenate(position_arrays)),
            "seq_idx": torch.from_numpy(np.concatenate(seq_arrays)),
            "cu_seq_lens_q": torch.tensor(cumulative_lengths, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(cumulative_lengths, dtype=torch.int32),
            "max_length_q": torch.tensor(max(piece.length for piece in pieces), dtype=torch.int32),
            "max_length_k": torch.tensor(max(piece.length for piece in pieces), dtype=torch.int32),
        }


@dataclass
class Qwen35PackedCollator:
    """Collate one pack and select only supervised predecessor hidden rows.

    ``logits_to_keep`` contains positions ``t`` whose next-token label
    ``labels[t + 1]`` is supervised. ``shift_labels`` contains those targets in
    the same order. An all-masked synthetic pack receives one ignored sentinel
    row so every DDP rank retains the same parameter graph while contributing
    exactly zero target loss.
    """

    pad_token_id: int
    sequence_length: int
    pad_to_sequence_length: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("pre-packed Qwen3.5 training requires per_device_train_batch_size=1")
        feature = features[0]
        synthetic = bool(feature.get("_qwen35_synthetic", False))
        if synthetic:
            input_ids = torch.full((self.sequence_length,), self.pad_token_id, dtype=torch.int64)
            labels = torch.full((self.sequence_length,), -100, dtype=torch.int64)
            position_ids = torch.arange(self.sequence_length, dtype=torch.int64)
            seq_idx = torch.zeros(self.sequence_length, dtype=torch.int32)
            cu_seq_lens_q = torch.tensor([0, self.sequence_length], dtype=torch.int32)
            cu_seq_lens_k = torch.tensor([0, self.sequence_length], dtype=torch.int32)
            max_length_q = self.sequence_length
            max_length_k = self.sequence_length
            length = self.sequence_length
            padding = 0
        else:
            input_ids = cast(torch.Tensor, feature["input_ids"])
            labels = cast(torch.Tensor, feature["labels"])
            position_ids = cast(torch.Tensor, feature["position_ids"])
            seq_idx = cast(torch.Tensor, feature["seq_idx"])
            cu_seq_lens_q = cast(torch.Tensor, feature["cu_seq_lens_q"])
            cu_seq_lens_k = cast(torch.Tensor, feature["cu_seq_lens_k"])
            max_length_q = int(feature["max_length_q"])
            max_length_k = int(feature["max_length_k"])
            length = int(input_ids.numel())
            padding = self.sequence_length - length if self.pad_to_sequence_length else 0
        if length > self.sequence_length:
            raise ValueError(f"pack length {length} exceeds configured sequence length {self.sequence_length}")

        def pad(value: torch.Tensor, fill: int) -> torch.Tensor:
            if not padding:
                return value
            return torch.nn.functional.pad(value, (0, padding), value=fill)

        if padding:
            padding_sequence_index = len(cu_seq_lens_q) - 1
            seq_idx = torch.nn.functional.pad(seq_idx, (0, padding), value=padding_sequence_index)
            position_ids = torch.cat([position_ids, torch.arange(padding, dtype=position_ids.dtype)])
            final_boundary = torch.tensor([self.sequence_length], dtype=cu_seq_lens_q.dtype)
            cu_seq_lens_q = torch.cat([cu_seq_lens_q, final_boundary])
            cu_seq_lens_k = torch.cat([cu_seq_lens_k, final_boundary])
            max_length_q = max(max_length_q, padding)
            max_length_k = max(max_length_k, padding)

        padded_labels = pad(labels, -100).unsqueeze(0)
        shifted = padded_labels[..., 1:]
        valid = shifted.ne(-100)
        selected_positions = torch.nonzero(valid[0], as_tuple=False).flatten().to(torch.int64)
        selected_shift_labels = shifted[valid].contiguous()
        target_count = int(selected_shift_labels.numel())
        if target_count == 0:
            selected_positions = torch.zeros(1, dtype=torch.int64)
            selected_shift_labels = torch.full((1,), -100, dtype=padded_labels.dtype)

        batch: dict[str, Any] = {
            "input_ids": pad(input_ids, self.pad_token_id).unsqueeze(0),
            "labels": padded_labels,
            "position_ids": position_ids.unsqueeze(0),
            "seq_idx": seq_idx.unsqueeze(0),
            "cu_seq_lens_q": cu_seq_lens_q,
            "cu_seq_lens_k": cu_seq_lens_k,
            "max_length_q": max_length_q,
            "max_length_k": max_length_k,
            "logits_to_keep": selected_positions,
            "shift_labels": selected_shift_labels,
            "_qwen35_assistant_targets": target_count,
            "_qwen35_synthetic": synthetic,
        }
        for key, value in feature.items():
            if key.startswith("_qwen35_") and key not in batch:
                batch[key] = value
        return batch
