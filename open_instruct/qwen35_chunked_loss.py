"""Pure-PyTorch checkpointed chunked selected-row loss for Qwen3.5.

The finite-precision objective is deliberately explicit: supervised hidden
rows remain in causal sequence order, contiguous chunks are projected in
left-to-right order, each chunk's cross entropy is accumulated in FP32, and
the final numerator is divided by the global assistant-target count supplied
by Transformers Trainer.

Non-reentrant activation checkpointing is a memory transformation only.  The
qualification reference executes the same chunks without checkpointing and
requires bit equality.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

QUALIFIED_CHUNK_SIZES = (128, 256, 512, 1024)
IMPLEMENTATION_ID = "pytorch_nonreentrant_checkpointed_chunked_selected_rows_r1"


@dataclass(frozen=True)
class ChunkedLossAudit:
    implementation_id: str
    checkpointed: bool
    selected_rows: int
    chunk_size: int
    chunk_count: int
    chunk_boundaries: tuple[tuple[int, int], ...]
    maximum_chunk_rows: int
    vocabulary_size: int
    hidden_size: int
    maximum_logit_elements: int
    full_selected_logit_elements: int
    global_target_count: int
    zero_target: bool
    returned_dense_logits: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ValidatedInputs:
    rows: torch.Tensor
    targets: torch.Tensor
    divisor: torch.Tensor
    divisor_value: int
    vocabulary_size: int
    hidden_size: int
    zero_target: bool


def _scalar_positive_integer(value: int | torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, int]:
    divisor = torch.as_tensor(value, device=device).detach().reshape(())
    if divisor.dtype.is_floating_point:
        scalar = float(divisor.item())
        if not math.isfinite(scalar) or not scalar.is_integer():
            raise ValueError("global_target_count must be a finite integer scalar")
        integer = int(scalar)
    else:
        integer = int(divisor.item())
    if integer <= 0:
        raise ValueError("global_target_count must be positive when supervised targets exist")
    return divisor, integer


def _validate_inputs(
    rows: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    global_target_count: int | torch.Tensor | None,
    chunk_size: int,
) -> _ValidatedInputs:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if rows.ndim != 2 or lm_head_weight.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected rows [N,H], output weight [V,H], and targets [N]")
    if rows.shape[0] != targets.numel():
        raise ValueError("selected row and target counts differ")
    if rows.shape[1] != lm_head_weight.shape[1]:
        raise ValueError("selected rows and output weight use different hidden sizes")
    if rows.device != lm_head_weight.device or rows.device != targets.device:
        raise ValueError("selected rows, output weight, and targets must share one device")
    if not rows.is_floating_point() or not lm_head_weight.is_floating_point():
        raise ValueError("selected rows and output weight must be floating point")
    if targets.dtype != torch.long:
        raise ValueError("selected targets must use torch.long")
    if lm_head_weight.dtype != torch.float32:
        raise ValueError("the qualified output-head parameter must remain FP32")

    live = targets.ne(-100)
    live_count = int(live.sum().item())
    if live_count == 0:
        if targets.numel() != 1 or rows.shape[0] != 1:
            raise ValueError("zero-target input must use exactly one ignored graph-connected sentinel row")
        divisor = torch.ones((), device=rows.device, dtype=torch.int64)
        return _ValidatedInputs(
            rows=rows,
            targets=targets,
            divisor=divisor,
            divisor_value=0,
            vocabulary_size=int(lm_head_weight.shape[0]),
            hidden_size=int(lm_head_weight.shape[1]),
            zero_target=True,
        )
    if live_count != targets.numel():
        raise ValueError("ordinary selected targets may not mix live IDs with ignore-index sentinels")
    minimum_target = int(targets.min().item())
    maximum_target = int(targets.max().item())
    vocabulary_size = int(lm_head_weight.shape[0])
    if minimum_target < 0 or maximum_target >= vocabulary_size:
        raise ValueError("selected target ID is outside the output vocabulary")
    divisor_value: int | torch.Tensor = live_count if global_target_count is None else global_target_count
    divisor, integer = _scalar_positive_integer(divisor_value, device=rows.device)
    return _ValidatedInputs(
        rows=rows,
        targets=targets,
        divisor=divisor,
        divisor_value=integer,
        vocabulary_size=vocabulary_size,
        hidden_size=int(lm_head_weight.shape[1]),
        zero_target=False,
    )


def _chunk_boundaries(selected_rows: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(start + chunk_size, selected_rows)) for start in range(0, selected_rows, chunk_size))


def _connected_zero(rows: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
    # Both operands remain in the graph, including on a DDP rank with no
    # assistant target in an entire optimizer update.
    return (rows.sum() + lm_head_weight.sum()) * 0.0


def _fp32_chunk_loss(rows: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(rows, weight)
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def _audit(
    validated: _ValidatedInputs, *, chunk_size: int, boundaries: tuple[tuple[int, int], ...], checkpointed: bool
) -> ChunkedLossAudit:
    selected_rows = 0 if validated.zero_target else int(validated.targets.numel())
    maximum_chunk_rows = max((end - start for start, end in boundaries), default=0)
    return ChunkedLossAudit(
        implementation_id=IMPLEMENTATION_ID,
        checkpointed=checkpointed,
        selected_rows=selected_rows,
        chunk_size=chunk_size,
        chunk_count=len(boundaries),
        chunk_boundaries=boundaries,
        maximum_chunk_rows=maximum_chunk_rows,
        vocabulary_size=validated.vocabulary_size,
        hidden_size=validated.hidden_size,
        maximum_logit_elements=maximum_chunk_rows * validated.vocabulary_size,
        full_selected_logit_elements=selected_rows * validated.vocabulary_size,
        global_target_count=validated.divisor_value,
        zero_target=validated.zero_target,
        returned_dense_logits=False,
    )


def checkpointed_chunked_selective_linear_cross_entropy(
    rows: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    global_target_count: int | torch.Tensor | None,
    chunk_size: int,
    execution_counter: dict[str, int] | None = None,
    return_audit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ChunkedLossAudit]:
    """Execute the production objective with non-reentrant recomputation."""

    validated = _validate_inputs(
        rows, lm_head_weight, targets, global_target_count=global_target_count, chunk_size=chunk_size
    )
    if validated.zero_target:
        loss = _connected_zero(rows, lm_head_weight)
        audit = _audit(validated, chunk_size=chunk_size, boundaries=(), checkpointed=True)
        return (loss, audit) if return_audit else loss

    boundaries = _chunk_boundaries(int(targets.numel()), chunk_size)

    def run_chunk(chunk_rows: torch.Tensor, weight: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        if execution_counter is not None:
            execution_counter["chunk_function_calls"] = execution_counter.get("chunk_function_calls", 0) + 1
        return _fp32_chunk_loss(chunk_rows, weight, chunk_targets)

    loss_sum: torch.Tensor | None = None
    for start, end in boundaries:
        chunk_loss = checkpoint(
            run_chunk,
            rows[start:end],
            lm_head_weight,
            targets[start:end],
            use_reentrant=False,
            preserve_rng_state=True,
            determinism_check="default",
            debug=False,
            early_stop=True,
        )
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
    if loss_sum is None:
        raise RuntimeError("live selected targets produced no chunks")
    loss = loss_sum / validated.divisor.to(dtype=loss_sum.dtype)
    audit = _audit(validated, chunk_size=chunk_size, boundaries=boundaries, checkpointed=True)
    return (loss, audit) if return_audit else loss


def ordinary_chunked_selective_linear_cross_entropy(
    rows: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    global_target_count: int | torch.Tensor | None,
    chunk_size: int,
    execution_counter: dict[str, int] | None = None,
    return_audit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ChunkedLossAudit]:
    """Independent ordinary-autograd reference for the same chunk order."""

    validated = _validate_inputs(
        rows, lm_head_weight, targets, global_target_count=global_target_count, chunk_size=chunk_size
    )
    if validated.zero_target:
        loss = _connected_zero(rows, lm_head_weight)
        audit = _audit(validated, chunk_size=chunk_size, boundaries=(), checkpointed=False)
        return (loss, audit) if return_audit else loss

    boundaries = _chunk_boundaries(int(targets.numel()), chunk_size)
    loss_sum: torch.Tensor | None = None
    for start, end in boundaries:
        if execution_counter is not None:
            execution_counter["chunk_function_calls"] = execution_counter.get("chunk_function_calls", 0) + 1
        logits = F.linear(rows[start:end], lm_head_weight)
        chunk_loss = F.cross_entropy(logits.float(), targets[start:end], reduction="sum")
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
    if loss_sum is None:
        raise RuntimeError("live selected targets produced no reference chunks")
    loss = loss_sum / validated.divisor.to(dtype=loss_sum.dtype)
    audit = _audit(validated, chunk_size=chunk_size, boundaries=boundaries, checkpointed=False)
    return (loss, audit) if return_audit else loss


def qwen35_checkpointed_chunked_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    shift_labels: torch.LongTensor | None = None,
    num_items_in_batch: int | torch.Tensor | None = None,
    skip_logits: bool | None = None,
    **kwargs: Any,
):
    """Qwen3.5 CausalLM forward using the qualified loss only when labeled."""

    # Keep the arithmetic primitive importable in minimal Torch-only test and
    # qualification environments; Transformers is required only by this model
    # adapter.
    from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: PLC0415

    return_dict = kwargs.pop("return_dict", None)
    if return_dict is None:
        return_dict = self.config.use_return_dict
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]
    has_loss_input = labels is not None or shift_labels is not None
    if skip_logits is None:
        skip_logits = has_loss_input

    logits = None
    loss = None
    if has_loss_input and skip_logits:
        if shift_labels is None:
            raise ValueError("qualified selected-row loss requires explicit shift_labels from the packed collator")
        if not torch.is_tensor(logits_to_keep) or logits_to_keep.ndim != 1:
            raise ValueError("qualified selected-row loss requires a one-dimensional logits_to_keep tensor")
        if num_items_in_batch is None and shift_labels.ne(-100).any():
            raise ValueError("qualified selected-row loss requires the global assistant-target divisor")
        rows = kept_hidden_states.reshape(-1, kept_hidden_states.shape[-1])
        result = checkpointed_chunked_selective_linear_cross_entropy(
            rows,
            self.lm_head.weight,
            shift_labels.reshape(-1),
            global_target_count=num_items_in_batch,
            chunk_size=int(self._qwen35_selected_loss_chunk_size),
            return_audit=True,
        )
        loss, audit = result
        self._qwen35_last_loss_audit = audit.to_dict()
    else:
        logits = self.lm_head(kept_hidden_states)
        if has_loss_input:
            if shift_labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    shift_labels=shift_labels,
                    vocab_size=self.config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                )
            else:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def install_qwen35_checkpointed_chunked_loss(model: torch.nn.Module, *, chunk_size: int) -> None:
    """Install the R18 forward without changing parameters or submodules."""

    if chunk_size not in QUALIFIED_CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {QUALIFIED_CHUNK_SIZES}")
    if getattr(model, "_qwen35_selected_loss_implementation_id", None) is not None:
        raise RuntimeError("a Qwen3.5 selected-loss implementation is already installed")
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError("expected a Qwen3.5 CausalLM with model and lm_head modules")
    model._qwen35_selected_loss_implementation_id = IMPLEMENTATION_ID
    model._qwen35_selected_loss_chunk_size = int(chunk_size)
    model._qwen35_last_loss_audit = None
    model.forward = MethodType(qwen35_checkpointed_chunked_forward, model)
    if model.forward.__func__ is not qwen35_checkpointed_chunked_forward:
        raise RuntimeError("failed to install the Qwen3.5 checkpointed chunked forward")
