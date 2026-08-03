#!/usr/bin/env python3
"""Validate native rendering/masking and optional packed-logit parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, Qwen2TokenizerFast, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5

from open_instruct.qwen35_data import tokenize_qwen35_example
from open_instruct.qwen35_qualification import EVIDENCE_SERIALIZATION_CONTRACT
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_training import (
    conditional_source_key_for_text_target,
    tensor_sha256,
    validate_text_loading_info,
    write_json_atomic,
)

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-jsonl", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--model-parity",
        action="store_true",
        help="Load the official conditional model and compare independent versus packed logits on CUDA.",
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--qualification-manifest", type=Path)
    parser.add_argument("--parity-atol", type=float, default=0.05)
    parser.add_argument("--parity-rtol", type=float, default=0.01)
    parser.add_argument(
        "--require-conditional-text-conversion-parity",
        action="store_true",
        help="Require model-name-or-path to be a conditional checkpoint and compare its text logits to CausalLM.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_runtime_versions() -> dict[str, str]:
    versions = {"transformers": transformers.__version__, "torch": torch.__version__}
    for package in ("flash-attn", "causal-conv1d", "flash-linear-attention", "fla-core"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def canonicalize_json_metadata(value: Any, *, context: str = "metadata") -> Any:
    """Losslessly map supported Python containers to deterministic strict JSON values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in {context}")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"non-string mapping key in {context}")
        return {key: canonicalize_json_metadata(value[key], context=f"{context}.{key}") for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        normalized = [canonicalize_json_metadata(child, context=f"{context}{{item}}") for child in value]
        encoded = [json.dumps(child, allow_nan=False, sort_keys=True, separators=(",", ":")) for child in normalized]
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"distinct set elements collide after JSON normalization in {context}")
        return [child for _, child in sorted(zip(encoded, normalized, strict=True))]
    if isinstance(value, (list, tuple)):
        return [canonicalize_json_metadata(child, context=f"{context}[{index}]") for index, child in enumerate(value)]
    raise TypeError(f"unsupported {type(value).__name__} object in {context}")


def write_strict_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Reject NaN/Infinity before using the shared atomic JSON writer."""

    json.dumps(value, allow_nan=False)
    write_json_atomic(path, value)


def validate_fixture(
    fixture_path: Path, tokenizer: Qwen2TokenizerFast, *, max_seq_length: int, max_examples: int
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    token_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    cases: list[dict[str, Any]] = []
    with fixture_path.open() as handle:
        for fixture_index, line in enumerate(handle):
            if max_examples and fixture_index >= max_examples:
                break
            row = json.loads(line)
            tokenized = tokenize_qwen35_example(
                tokenizer, row["messages"], row["tools"], max_seq_length=max_seq_length, enable_thinking=False
            )
            assistant_count = sum(message["role"] == "assistant" for message in row["messages"])
            if not tokenized.assistant_spans or len(tokenized.assistant_spans) > assistant_count:
                raise AssertionError(f"fixture row {fixture_index} assistant-span mismatch")
            if tokenized.trainable_tokens == 0:
                raise AssertionError(f"fixture row {fixture_index} has no trainable tokens")
            if len(tokenized.input_ids) != len(tokenized.labels_mask):
                raise AssertionError(f"fixture row {fixture_index} token/mask length mismatch")
            token_bytes = torch.tensor(tokenized.input_ids, dtype=torch.int64).numpy().tobytes()
            mask_bytes = torch.tensor(tokenized.labels_mask, dtype=torch.bool).numpy().tobytes()
            token_digest.update(token_bytes)
            mask_digest.update(mask_bytes)
            counts["examples"] += 1
            counts["tokens"] += len(tokenized.input_ids)
            counts["trainable_tokens"] += tokenized.trainable_tokens
            counts["assistant_messages"] += assistant_count
            counts["assistant_spans_after_truncation"] += len(tokenized.assistant_spans)
            counts["truncated_examples"] += tokenized.num_tokens_before_truncation > len(tokenized.input_ids)
            counts["tool_calls"] += sum(len(message.get("tool_calls") or []) for message in row["messages"])
            cases.append(
                {
                    "fixture_index": fixture_index,
                    "dataset": row.get("dataset"),
                    "sample_id": row.get("sample_id"),
                    "num_tokens": len(tokenized.input_ids),
                    "num_tokens_before_truncation": tokenized.num_tokens_before_truncation,
                    "num_trainable_tokens": tokenized.trainable_tokens,
                    "assistant_spans": tokenized.assistant_spans,
                }
            )
    if not counts["examples"]:
        raise ValueError("fixture validation selected no examples")
    return {
        "counts": dict(counts),
        "concatenated_token_ids_sha256": token_digest.hexdigest(),
        "concatenated_labels_mask_sha256": mask_digest.hexdigest(),
        "cases": cases,
    }


def render_synthetic_ids(tokenizer: Qwen2TokenizerFast, prompt: str, response: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tools=[],
        enable_thinking=False,
        add_generation_prompt=False,
        tokenize=True,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token_id) for token_id in rendered]


def tensor_parity_metrics(
    observed: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    """Return finite, scale-aware, all-close, and decision metrics without raising."""

    if observed.shape != reference.shape:
        raise ValueError(f"comparison shape mismatch: {tuple(observed.shape)} != {tuple(reference.shape)}")
    observed = observed.detach().to(device="cpu", dtype=torch.float32)
    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    if observed.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    finite = torch.isfinite(observed) & torch.isfinite(reference)
    nonfinite_count = int((~finite).sum())
    difference = observed - reference
    absolute = difference.abs()
    reference_norm = torch.linalg.vector_norm(reference.double()) if nonfinite_count == 0 else None
    observed_norm = torch.linalg.vector_norm(observed.double()) if nonfinite_count == 0 else None
    difference_norm = torch.linalg.vector_norm(difference.double()) if nonfinite_count == 0 else None
    cosine = None
    if nonfinite_count == 0:
        assert reference_norm is not None and observed_norm is not None and difference_norm is not None
        if float(reference_norm) == 0 and float(observed_norm) == 0:
            cosine = 1.0
        elif float(reference_norm) != 0 and float(observed_norm) != 0:
            cosine = float(
                torch.dot(observed.double().reshape(-1), reference.double().reshape(-1))
                / (observed_norm * reference_norm)
            )
    close = torch.isclose(observed, reference, atol=atol, rtol=rtol, equal_nan=False)
    quantile_levels = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32)
    quantiles = torch.quantile(absolute.reshape(-1), quantile_levels) if nonfinite_count == 0 else None
    result: dict[str, Any] = {
        "shape": list(observed.shape),
        "elements": observed.numel(),
        "nonfinite_count": nonfinite_count,
        "bit_exact": bool(torch.equal(observed, reference)),
        "allclose": bool(close.all()),
        "mismatched_elements": int((~close).sum()),
        "mismatched_fraction": float((~close).double().mean()),
        "maximum_absolute_error": float(absolute.max()) if nonfinite_count == 0 else None,
        "mean_absolute_error": float(absolute.mean()) if nonfinite_count == 0 else None,
        "absolute_error_quantiles": {
            "p50": float(quantiles[0]) if quantiles is not None else None,
            "p90": float(quantiles[1]) if quantiles is not None else None,
            "p99": float(quantiles[2]) if quantiles is not None else None,
            "p99_9": float(quantiles[3]) if quantiles is not None else None,
        },
        "relative_l2_error": (
            float(difference_norm) / max(float(reference_norm), torch.finfo(torch.float64).eps)
            if difference_norm is not None and reference_norm is not None
            else None
        ),
        "cosine_similarity": cosine,
        "observed_l2_norm": float(observed_norm) if observed_norm is not None else None,
        "reference_l2_norm": float(reference_norm) if reference_norm is not None else None,
        "difference_l2_norm": float(difference_norm) if difference_norm is not None else None,
        "atol": atol,
        "rtol": rtol,
    }
    if observed.ndim == 2 and observed.shape[-1] > 1:
        result["top1_agreement"] = (
            float((observed.argmax(-1) == reference.argmax(-1)).double().mean()) if nonfinite_count == 0 else None
        )
    return result


def compare_text_state_to_conditional(model: torch.nn.Module, conditional: torch.nn.Module) -> dict[str, Any]:
    """Prove every text state tensor is the exact mapped conditional tensor."""

    target_state = model.state_dict()
    source_state = conditional.state_dict()
    rows = []
    mismatches = []
    for target_key in sorted(target_state):
        source_key = conditional_source_key_for_text_target(target_key)
        if source_key not in source_state:
            raise RuntimeError(f"mapped conditional tensor is absent: {source_key}")
        target = target_state[target_key]
        source = source_state[source_key]
        equal = target.shape == source.shape and target.dtype == source.dtype and torch.equal(target, source)
        row = {
            "target_key": target_key,
            "source_key": source_key,
            "shape": list(target.shape),
            "source_shape": list(source.shape),
            "dtype": str(target.dtype),
            "source_dtype": str(source.dtype),
            "numel": target.numel(),
            "target_sha256": tensor_sha256(target),
            "source_sha256": tensor_sha256(source),
            "bit_exact": equal,
        }
        rows.append(row)
        if not equal or row["target_sha256"] != row["source_sha256"]:
            mismatches.append(target_key)
    return {
        "status": "pass" if not mismatches else "fail",
        "target_tensor_count": len(rows),
        "target_state_numel": sum(value.numel() for value in target_state.values()),
        "mismatched_target_keys": mismatches,
        "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "rows": rows,
    }


@contextmanager
def capture_text_layer_outputs(model: Qwen3_5ForCausalLM):
    outputs: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):

        def capture(_module, _inputs, output, *, index=layer_index):
            value = output[0] if isinstance(output, tuple) else output
            outputs[index] = value[0].detach().float().cpu()

        handles.append(layer.register_forward_hook(capture))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def packed_metadata(lengths: list[int], *, device: torch.device) -> dict[str, Any]:
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    seq_idx = torch.cat(
        [torch.full((length,), index, dtype=torch.int32, device=device) for index, length in enumerate(lengths)]
    ).unsqueeze(0)
    return {
        "seq_idx": seq_idx,
        "cu_seq_lens_q": boundaries,
        "cu_seq_lens_k": boundaries,
        "max_length_q": max(lengths),
        "max_length_k": max(lengths),
    }


def run_text_forward(
    model: Qwen3_5ForCausalLM, sequences: list[list[int]], *, packed_metadata_lengths: list[int] | None
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    device = next(model.parameters()).device
    lengths = [len(sequence) for sequence in sequences]
    input_ids = torch.tensor(sum(sequences, []), dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.cat([torch.arange(length, device=device) for length in lengths]).unsqueeze(0)
    if packed_metadata_lengths is not None and sum(packed_metadata_lengths) != sum(lengths):
        raise ValueError("packed metadata and flattened token lengths differ")
    kwargs = packed_metadata(packed_metadata_lengths, device=device) if packed_metadata_lengths is not None else {}
    with capture_text_layer_outputs(model) as layer_outputs, torch.inference_mode():
        logits = (
            model(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kwargs).logits[0].float().cpu()
        )
    return logits, layer_outputs


def mutate_one_token(sequence: list[int], *, vocabulary_size: int) -> tuple[list[int], int]:
    if len(sequence) < 3:
        raise ValueError("counterfactual sequence must contain at least three tokens")
    position = len(sequence) // 2
    mutated = list(sequence)
    mutated[position] = (mutated[position] + 997) % vocabulary_size
    if mutated[position] == sequence[position]:
        mutated[position] = (mutated[position] + 1) % vocabulary_size
    return mutated, position


def mutate_every_token(sequence: list[int], *, vocabulary_size: int) -> list[int]:
    mutated = [int((token + 997 + index) % vocabulary_size) for index, token in enumerate(sequence)]
    for index, token in enumerate(sequence):
        if mutated[index] == token:
            mutated[index] = (mutated[index] + 1) % vocabulary_size
    if len(mutated) != len(sequence) or any(left == right for left, right in zip(mutated, sequence, strict=True)):
        raise AssertionError("full-document mutation did not change every token exactly once in place")
    return mutated


def exact_segment_evidence(
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    candidate_layers: dict[int, torch.Tensor],
    reference_layers: dict[int, torch.Tensor],
    *,
    candidate_start: int,
    reference_start: int,
    length: int,
    layer_types: list[str],
) -> dict[str, Any]:
    candidate_slice = slice(candidate_start, candidate_start + length)
    reference_slice = slice(reference_start, reference_start + length)
    return {
        "unchanged_segment_logits": tensor_parity_metrics(
            candidate_logits[candidate_slice], reference_logits[reference_slice], atol=0, rtol=0
        ),
        "unchanged_segment_layers": [
            {
                "layer_index": index,
                "layer_type": layer_types[index],
                "metrics": tensor_parity_metrics(
                    candidate_layers[index][candidate_slice], reference_layers[index][reference_slice], atol=0, rtol=0
                ),
            }
            for index in range(len(layer_types))
        ],
    }


def exact_invariance_passes(value: dict[str, Any]) -> bool:
    return value["unchanged_segment_logits"]["bit_exact"] and all(
        row["metrics"]["bit_exact"] for row in value["unchanged_segment_layers"]
    )


def validate_model_parity(
    tokenizer: Qwen2TokenizerFast,
    model_name_or_path: str,
    revision: str,
    *,
    atol: float,
    rtol: float,
    require_conditional_text_conversion_parity: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("--model-parity requires a CUDA GPU")
    source_config = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
    conditional = None
    loading_info = None
    if source_config.model_type == "qwen3_5":
        conditional = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name_or_path, revision=revision, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            config=source_config.text_config,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            output_loading_info=True,
        )
        validate_text_loading_info(loading_info)
        loading_info = canonicalize_json_metadata(loading_info, context="conditional_to_text_loading_info")
    elif source_config.model_type == "qwen3_5_text":
        if require_conditional_text_conversion_parity:
            raise RuntimeError("conversion parity requires the original qwen3_5 conditional checkpoint")
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path, revision=revision, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    else:
        raise RuntimeError(f"unsupported Qwen3.5 reference config type: {source_config.model_type!r}")
    state_mapping = compare_text_state_to_conditional(model, conditional) if conditional is not None else None
    model = model.cuda()
    model.eval()
    model.config.use_cache = False
    if conditional is not None:
        conditional = conditional.cuda()
        conditional.eval()
        conditional.config.use_cache = False
        conditional.config.text_config.use_cache = False
    if modeling_qwen3_5.chunk_gated_delta_rule is None:
        raise RuntimeError("packed parity requires flash-linear-attention")
    linear_layers = [module for module in model.modules() if type(module).__name__ == "Qwen3_5GatedDeltaNet"]
    if not linear_layers or any(module.causal_conv1d_fn is None for module in linear_layers):
        raise RuntimeError("packed parity requires causal-conv1d seq_idx support")

    sequences = [
        render_synthetic_ids(tokenizer, "Return the word alpha.", "alpha"),
        render_synthetic_ids(tokenizer, "Return the word beta twice.", "beta beta"),
    ]
    lengths = [len(sequence) for sequence in sequences]
    ordinary_logits: list[torch.Tensor] = []
    ordinary_layers: list[dict[int, torch.Tensor]] = []
    singleton_logits: list[torch.Tensor] = []
    singleton_layers: list[dict[int, torch.Tensor]] = []
    conditional_text_metrics = []
    conditional_text_losses = []
    for sequence in sequences:
        ordinary, ordinary_capture = run_text_forward(model, [sequence], packed_metadata_lengths=None)
        singleton, singleton_capture = run_text_forward(model, [sequence], packed_metadata_lengths=[len(sequence)])
        ordinary_logits.append(ordinary)
        ordinary_layers.append(ordinary_capture)
        singleton_logits.append(singleton)
        singleton_layers.append(singleton_capture)
        if conditional is not None:
            input_ids = torch.tensor(sequence, dtype=torch.long, device="cuda").unsqueeze(0)
            position_ids = torch.arange(len(sequence), device="cuda").unsqueeze(0)
            with torch.inference_mode():
                conditional_logits = (
                    conditional(input_ids=input_ids, position_ids=position_ids, use_cache=False)
                    .logits[0]
                    .float()
                    .cpu()
                )
            metrics = tensor_parity_metrics(ordinary, conditional_logits, atol=0, rtol=0)
            conditional_text_metrics.append(metrics)
            causal_loss = torch.nn.functional.cross_entropy(ordinary[:-1], input_ids[0, 1:].cpu(), reduction="mean")
            conditional_loss = torch.nn.functional.cross_entropy(
                conditional_logits[:-1], input_ids[0, 1:].cpu(), reduction="mean"
            )
            losses_are_finite = math.isfinite(float(causal_loss)) and math.isfinite(float(conditional_loss))
            conditional_text_losses.append(
                {
                    "causal": float(causal_loss) if losses_are_finite else None,
                    "conditional": float(conditional_loss) if losses_are_finite else None,
                    "finite": losses_are_finite,
                    "bit_exact": losses_are_finite and bool(torch.equal(causal_loss, conditional_loss)),
                    "absolute_error": float((causal_loss - conditional_loss).abs()) if losses_are_finite else None,
                }
            )

    packed_logits, packed_layers = run_text_forward(model, sequences, packed_metadata_lengths=lengths)
    packed_segments = list(packed_logits.split(lengths))
    pack_shape_logits = [
        tensor_parity_metrics(packed, singleton, atol=atol, rtol=rtol)
        for packed, singleton in zip(packed_segments, singleton_logits, strict=True)
    ]
    cross_kernel_logits = [
        tensor_parity_metrics(singleton, ordinary, atol=atol, rtol=rtol)
        for singleton, ordinary in zip(singleton_logits, ordinary_logits, strict=True)
    ]
    layer_types = list(model.config.layer_types)
    pack_shape_layers = []
    cross_kernel_layers = []
    for sequence_index, length in enumerate(lengths):
        segment_start = sum(lengths[:sequence_index])
        segment_stop = segment_start + length
        kernel_rows = []
        cross_rows = []
        for layer_index, layer_type in enumerate(layer_types):
            kernel_rows.append(
                {
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    "metrics": tensor_parity_metrics(
                        packed_layers[layer_index][segment_start:segment_stop],
                        singleton_layers[sequence_index][layer_index],
                        atol=atol,
                        rtol=rtol,
                    ),
                }
            )
            cross_rows.append(
                {
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    "metrics": tensor_parity_metrics(
                        singleton_layers[sequence_index][layer_index],
                        ordinary_layers[sequence_index][layer_index],
                        atol=atol,
                        rtol=rtol,
                    ),
                }
            )
        pack_shape_layers.append(kernel_rows)
        cross_kernel_layers.append(cross_rows)

    mutated_first, first_mutation_position = mutate_one_token(sequences[0], vocabulary_size=model.config.vocab_size)
    mutated_second, second_mutation_position = mutate_one_token(sequences[1], vocabulary_size=model.config.vocab_size)
    mutate_first_logits, mutate_first_layers = run_text_forward(
        model, [mutated_first, sequences[1]], packed_metadata_lengths=lengths
    )
    mutate_second_logits, mutate_second_layers = run_text_forward(
        model, [sequences[0], mutated_second], packed_metadata_lengths=lengths
    )
    first_length = lengths[0]
    single_token_counterfactuals = {
        "mutate_first_hold_second": {
            "mutation_position": first_mutation_position,
            **exact_segment_evidence(
                mutate_first_logits,
                packed_logits,
                mutate_first_layers,
                packed_layers,
                candidate_start=first_length,
                reference_start=first_length,
                length=lengths[1],
                layer_types=layer_types,
            ),
        },
        "mutate_second_hold_first": {
            "mutation_position": second_mutation_position,
            **exact_segment_evidence(
                mutate_second_logits,
                packed_logits,
                mutate_second_layers,
                packed_layers,
                candidate_start=0,
                reference_start=0,
                length=lengths[0],
                layer_types=layer_types,
            ),
        },
    }

    fully_mutated = [mutate_every_token(sequence, vocabulary_size=model.config.vocab_size) for sequence in sequences]
    full_first_logits, full_first_layers = run_text_forward(
        model, [fully_mutated[0], sequences[1]], packed_metadata_lengths=lengths
    )
    full_second_logits, full_second_layers = run_text_forward(
        model, [sequences[0], fully_mutated[1]], packed_metadata_lengths=lengths
    )
    full_document_counterfactuals = {
        "mutate_every_first_token_hold_second": {
            "mutated_tokens": lengths[0],
            **exact_segment_evidence(
                full_first_logits,
                packed_logits,
                full_first_layers,
                packed_layers,
                candidate_start=first_length,
                reference_start=first_length,
                length=lengths[1],
                layer_types=layer_types,
            ),
        },
        "mutate_every_second_token_hold_first": {
            "mutated_tokens": lengths[1],
            **exact_segment_evidence(
                full_second_logits,
                packed_logits,
                full_second_layers,
                packed_layers,
                candidate_start=0,
                reference_start=0,
                length=lengths[0],
                layer_types=layer_types,
            ),
        },
    }

    duplicate_invariance = {}
    for sequence_index, sequence in enumerate(sequences):
        length = lengths[sequence_index]
        duplicate_logits, duplicate_layers = run_text_forward(
            model, [sequence, sequence], packed_metadata_lengths=[length, length]
        )
        duplicate_invariance[f"sequence_{sequence_index}_first_vs_second"] = exact_segment_evidence(
            duplicate_logits,
            duplicate_logits,
            duplicate_layers,
            duplicate_layers,
            candidate_start=length,
            reference_start=0,
            length=length,
            layer_types=layer_types,
        )

    swapped_logits, swapped_layers = run_text_forward(
        model, [sequences[1], sequences[0]], packed_metadata_lengths=[lengths[1], lengths[0]]
    )
    order_invariance = {
        "first_document_moved_to_second": exact_segment_evidence(
            swapped_logits,
            packed_logits,
            swapped_layers,
            packed_layers,
            candidate_start=lengths[1],
            reference_start=0,
            length=lengths[0],
            layer_types=layer_types,
        ),
        "second_document_moved_to_first": exact_segment_evidence(
            swapped_logits,
            packed_logits,
            swapped_layers,
            packed_layers,
            candidate_start=0,
            reference_start=first_length,
            length=lengths[1],
            layer_types=layer_types,
        ),
    }

    corrupted_logits, corrupted_layers = run_text_forward(model, sequences, packed_metadata_lengths=[sum(lengths)])
    corrupted_mutation_logits, corrupted_mutation_layers = run_text_forward(
        model, [fully_mutated[0], sequences[1]], packed_metadata_lengths=[sum(lengths)]
    )
    corrupted_boundary_negative_control = exact_segment_evidence(
        corrupted_mutation_logits,
        corrupted_logits,
        corrupted_mutation_layers,
        corrupted_layers,
        candidate_start=first_length,
        reference_start=first_length,
        length=lengths[1],
        layer_types=layer_types,
    )
    corrupted_boundary_negative_control["expected_bit_exact"] = False
    corrupted_boundary_negative_control["sensitivity_passed"] = not corrupted_boundary_negative_control[
        "unchanged_segment_logits"
    ]["bit_exact"] and any(
        not row["metrics"]["bit_exact"] for row in corrupted_boundary_negative_control["unchanged_segment_layers"]
    )

    failures = []
    if state_mapping is not None and state_mapping["status"] != "pass":
        failures.append("conditional-to-text state mapping is not bit-exact")
    if any(not value["bit_exact"] or value["nonfinite_count"] for value in conditional_text_metrics):
        failures.append("conditional and text ordinary logits are not bit-exact and finite")
    if any(not value["bit_exact"] for value in conditional_text_losses):
        failures.append("conditional and text dense next-token losses are not bit-exact")
    for family_name, family in (
        ("single-token counterfactual", single_token_counterfactuals),
        ("full-document counterfactual", full_document_counterfactuals),
        ("duplicate-document reset", duplicate_invariance),
        ("packed-order invariance", order_invariance),
    ):
        for name, value in family.items():
            if not exact_invariance_passes(value):
                failures.append(f"{family_name} {name} is not bit-exact")
    if corrupted_boundary_negative_control["sensitivity_passed"] is not True:
        failures.append("corrupted-boundary negative control did not reveal cross-document influence")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "model_name_or_path": model_name_or_path,
        "model_revision": revision,
        "dtype": "bfloat16",
        "source_config_model_type": source_config.model_type,
        "production_model_class": type(model).__name__,
        "vocabulary_size": model.config.vocab_size,
        "text_hidden_size": model.config.hidden_size,
        "text_num_hidden_layers": model.config.num_hidden_layers,
        "text_layer_types": layer_types,
        "model_config_commit": getattr(source_config, "_commit_hash", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cuda_device": torch.cuda.get_device_name(0),
        "attention_implementation": model.config._attn_implementation,
        "sequence_lengths": lengths,
        "atol": atol,
        "rtol": rtol,
        "standalone_reference_definition": "one document executed with the exact production packed metadata/kernel path",
        "singleton_vs_multi_pack_shape_diagnostic": {
            "gating": False,
            "reason": "This contrast changes the packed launch problem shape and therefore cannot identify cross-document content dependence by itself.",
            "frozen_r9_tolerance_observation": (
                "within_tolerance" if all(value["allclose"] for value in pack_shape_logits) else "exceeds_tolerance"
            ),
            "r11_failed_criterion_reclassified_as_pass": False,
            "logits": pack_shape_logits,
            "layers": pack_shape_layers,
        },
        "cross_kernel_ordinary_vs_singleton_diagnostic": {
            "gating": False,
            "reason": "This contrast changes ordinary versus variable-length Flash/FLA/conv kernel modes and cannot identify boundary leakage by itself.",
            "logits": cross_kernel_logits,
            "layers": cross_kernel_layers,
        },
        "single_token_counterfactual_no_cross_document_influence": single_token_counterfactuals,
        "full_document_counterfactual_no_cross_document_influence": full_document_counterfactuals,
        "duplicate_document_reset_invariance": duplicate_invariance,
        "packed_order_invariance": order_invariance,
        "corrupted_boundary_negative_control": corrupted_boundary_negative_control,
        "conditional_to_text_conversion": {
            "checked": conditional is not None,
            "status": (
                "pass"
                if conditional is not None
                and state_mapping is not None
                and state_mapping["status"] == "pass"
                and all(value["bit_exact"] for value in conditional_text_metrics)
                and all(value["bit_exact"] for value in conditional_text_losses)
                else "not_applicable_text_checkpoint"
                if conditional is None
                else "fail"
            ),
            "loading_info": loading_info,
            "loading_info_serialization": EVIDENCE_SERIALIZATION_CONTRACT,
            "state_mapping": state_mapping,
            "ordinary_logit_metrics": conditional_text_metrics,
            "dense_next_token_losses": conditional_text_losses,
            "atol": 0,
            "rtol": 0,
        },
    }


def main() -> None:
    args = parse_args()
    qualification_sha256 = None
    if args.qualification_manifest is not None:
        qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
        args.parity_atol = float(qualification["numerical_acceptance"]["packed_logit_absolute_tolerance"])
        args.parity_rtol = float(qualification["numerical_acceptance"]["packed_logit_relative_tolerance"])
        fixture_contract = qualification["reference_fixture"]
        if sha256_file(args.fixture_jsonl) != fixture_contract["fixture_sha256"]:
            raise ValueError("qualification fixture SHA-256 drift")
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.tokenizer_name_or_path, revision=args.tokenizer_revision)
    fixture_report = validate_fixture(
        args.fixture_jsonl, tokenizer, max_seq_length=args.max_seq_length, max_examples=args.max_examples
    )
    report = {
        "artifact": "qwen35_native_reference_validation",
        "fixture_jsonl": str(args.fixture_jsonl.resolve()),
        "fixture_jsonl_sha256": sha256_file(args.fixture_jsonl),
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "tokenizer_revision": args.tokenizer_revision,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "max_seq_length": args.max_seq_length,
        "runtime_versions": get_runtime_versions(),
        "qualification_manifest_sha256": qualification_sha256,
        "fixture_validation": fixture_report,
        "model_parity": None,
    }
    if args.model_parity:
        try:
            report["model_parity"] = validate_model_parity(
                tokenizer,
                args.model_name_or_path,
                args.model_revision,
                atol=args.parity_atol,
                rtol=args.parity_rtol,
                require_conditional_text_conversion_parity=args.require_conditional_text_conversion_parity,
            )
            write_strict_json_atomic(args.report_output, report)
        except Exception as error:
            report["model_parity"] = {
                "status": "error",
                "failures": [f"{type(error).__name__}: {error}"],
                "model_name_or_path": args.model_name_or_path,
                "model_revision": args.model_revision,
                "atol": args.parity_atol,
                "rtol": args.parity_rtol,
            }
            write_strict_json_atomic(args.report_output, report)
            raise
    else:
        write_strict_json_atomic(args.report_output, report)
    print(
        json.dumps(
            {
                "fixture_counts": fixture_report["counts"],
                "model_parity_status": (
                    report["model_parity"]["status"] if report["model_parity"] is not None else "not_requested"
                ),
                "model_parity_failures": (
                    report["model_parity"]["failures"] if report["model_parity"] is not None else []
                ),
                "report_output": str(args.report_output.resolve()),
            },
            indent=2,
        )
    )
    if report["model_parity"] is not None and report["model_parity"]["status"] != "pass":
        raise AssertionError(f"Qwen3.5 model parity failed: {report['model_parity']['failures']}")


if __name__ == "__main__":
    main()
