from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from open_instruct.qwen35_chunked_loss import (
    IMPLEMENTATION_ID,
    QUALIFIED_CHUNK_SIZES,
    checkpointed_chunked_selective_linear_cross_entropy,
    install_qwen35_checkpointed_chunked_loss,
    ordinary_chunked_selective_linear_cross_entropy,
)


def _leaves(rows: int, *, vocabulary: int = 37, hidden: int = 11, seed: int = 1701):
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randn(rows, hidden, generator=generator, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(vocabulary, hidden, generator=generator, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, vocabulary, (rows,), generator=generator, dtype=torch.long)
    return selected, weight, targets


@pytest.mark.parametrize("chunk_size", QUALIFIED_CHUNK_SIZES)
@pytest.mark.parametrize("rows", [1, 127, 128, 129, 256, 257, 512, 513, 1024, 1025])
def test_checkpointed_and_ordinary_same_chunk_objectives_are_bit_exact(chunk_size, rows):
    observed_rows, observed_weight, targets = _leaves(rows, seed=10_000 + rows)
    reference_rows = observed_rows.detach().clone().requires_grad_(True)
    reference_weight = observed_weight.detach().clone().requires_grad_(True)
    observed_counter: dict[str, int] = {}
    reference_counter: dict[str, int] = {}
    divisor = rows + 37

    observed, observed_audit = checkpointed_chunked_selective_linear_cross_entropy(
        observed_rows,
        observed_weight,
        targets,
        global_target_count=divisor,
        chunk_size=chunk_size,
        execution_counter=observed_counter,
        return_audit=True,
    )
    reference, reference_audit = ordinary_chunked_selective_linear_cross_entropy(
        reference_rows,
        reference_weight,
        targets,
        global_target_count=divisor,
        chunk_size=chunk_size,
        execution_counter=reference_counter,
        return_audit=True,
    )
    assert torch.equal(observed, reference)
    assert observed_audit.chunk_boundaries == reference_audit.chunk_boundaries
    assert observed_audit.maximum_logit_elements <= chunk_size * observed_audit.vocabulary_size
    assert observed_counter["chunk_function_calls"] == observed_audit.chunk_count
    assert reference_counter["chunk_function_calls"] == reference_audit.chunk_count

    observed.backward()
    reference.backward()

    assert observed_counter["chunk_function_calls"] == 2 * observed_audit.chunk_count
    assert reference_counter["chunk_function_calls"] == reference_audit.chunk_count
    assert torch.equal(observed_rows.grad, reference_rows.grad)
    assert torch.equal(observed_weight.grad, reference_weight.grad)


def test_checkpointed_forward_does_not_save_chunk_logits_for_backward():
    rows, weight, targets = _leaves(257, vocabulary=41, hidden=13)
    checkpoint_saved_shapes = []

    def pack(tensor):
        checkpoint_saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = checkpointed_chunked_selective_linear_cross_entropy(
            rows, weight, targets, global_target_count=targets.numel(), chunk_size=128
        )
    assert (128, 41) not in checkpoint_saved_shapes
    assert (1, 41) not in checkpoint_saved_shapes
    loss.backward()

    reference_rows = rows.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    ordinary_saved_shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: ordinary_saved_shapes.append(tuple(tensor.shape)) or tensor, lambda tensor: tensor
    ):
        ordinary_chunked_selective_linear_cross_entropy(
            reference_rows, reference_weight, targets, global_target_count=targets.numel(), chunk_size=128
        )
    assert (128, 41) in ordinary_saved_shapes


def test_chunked_objective_matches_an_independent_float64_logsumexp_definition():
    rows, weight, targets = _leaves(19, vocabulary=23, hidden=7)
    divisor = 31
    observed = ordinary_chunked_selective_linear_cross_entropy(
        rows, weight, targets, global_target_count=divisor, chunk_size=5
    )
    logits = rows.double() @ weight.double().T
    independent_terms = [torch.logsumexp(logits[index], dim=0) - logits[index, targets[index]] for index in range(19)]
    independent = sum(independent_terms) / divisor

    assert torch.allclose(observed.double(), independent, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_zero_target_sentinel_is_exact_connected_zero_without_projection(checkpointed):
    rows, weight, _ = _leaves(1)
    targets = torch.tensor([-100], dtype=torch.long)
    counter: dict[str, int] = {}
    function = (
        checkpointed_chunked_selective_linear_cross_entropy
        if checkpointed
        else ordinary_chunked_selective_linear_cross_entropy
    )
    loss, audit = function(
        rows, weight, targets, global_target_count=None, chunk_size=128, execution_counter=counter, return_audit=True
    )
    loss.backward()

    assert loss.item() == 0.0
    assert rows.grad is not None and torch.count_nonzero(rows.grad) == 0
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0
    assert counter == {}
    assert audit.zero_target is True
    assert audit.selected_rows == 0
    assert audit.chunk_count == 0
    assert audit.maximum_logit_elements == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r, w, t: (r, w, t, 0), "positive integer"),
        (lambda r, w, t: (r[:-1], w, t, 128), "counts differ"),
        (lambda r, w, t: (r, w.bfloat16(), t, 128), "must remain FP32"),
        (lambda r, w, t: (r, w, t.to(torch.int32), 128), "torch.long"),
        (lambda r, w, t: (r, w, torch.tensor([0, -100, 1]), 128), "may not mix"),
        (lambda r, w, t: (r, w, torch.tensor([0, 1, w.shape[0]]), 128), "outside"),
    ],
)
def test_invalid_objective_inputs_fail_closed(mutation, message):
    rows, weight, targets = _leaves(3)
    rows, weight, targets, chunk_size = mutation(rows, weight, targets)
    with pytest.raises(ValueError, match=message):
        checkpointed_chunked_selective_linear_cross_entropy(
            rows, weight, targets, global_target_count=3, chunk_size=chunk_size
        )


def test_nonpositive_or_fractional_global_divisor_fails_closed():
    rows, weight, targets = _leaves(3)
    for divisor in (0, -1, float("nan"), 3.5):
        with pytest.raises(ValueError, match="global_target_count"):
            ordinary_chunked_selective_linear_cross_entropy(
                rows, weight, targets, global_target_count=divisor, chunk_size=128
            )


class _Backbone(torch.nn.Module):
    def __init__(self, vocabulary: int, hidden: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary, hidden)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None)


class _CausalModel(torch.nn.Module):
    def __init__(self, vocabulary=31, hidden=9):
        super().__init__()
        self.model = _Backbone(vocabulary, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocabulary, bias=False)
        self.config = SimpleNamespace(use_return_dict=True, vocab_size=vocabulary)


def test_installed_qwen_forward_returns_only_loss_and_audits_selected_rows():
    pytest.importorskip("transformers")
    model = _CausalModel()
    install_qwen35_checkpointed_chunked_loss(model, chunk_size=128)
    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor([0, 2, 3], dtype=torch.long)
    targets = torch.tensor([2, 4, 5], dtype=torch.long)

    output = model(
        input_ids=inputs,
        labels=torch.tensor([[-100, 2, -100, 4, 5]], dtype=torch.long),
        logits_to_keep=positions,
        shift_labels=targets,
        num_items_in_batch=11,
    )
    output.loss.backward()

    assert output.logits is None
    assert math.isfinite(float(output.loss))
    assert model.forward.__module__ == "open_instruct.qwen35_chunked_loss"
    assert model._qwen35_selected_loss_implementation_id == IMPLEMENTATION_ID
    assert model._qwen35_last_loss_audit["selected_rows"] == 3
    assert model._qwen35_last_loss_audit["global_target_count"] == 11
    assert model._qwen35_last_loss_audit["returned_dense_logits"] is False


def test_installed_forward_is_bit_exact_to_same_chunk_reference_through_all_parameters():
    pytest.importorskip("transformers")
    torch.manual_seed(419)
    observed = _CausalModel(vocabulary=31, hidden=9)
    reference = _CausalModel(vocabulary=31, hidden=9)
    reference.load_state_dict(observed.state_dict(), strict=True)
    install_qwen35_checkpointed_chunked_loss(observed, chunk_size=128)
    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    positions = torch.tensor([0, 2, 3], dtype=torch.long)
    targets = torch.tensor([2, 4, 5], dtype=torch.long)

    output = observed(
        input_ids=inputs,
        labels=torch.tensor([[-100, 2, -100, 4, 5]], dtype=torch.long),
        logits_to_keep=positions,
        shift_labels=targets,
        num_items_in_batch=11,
    )
    hidden = reference.model(input_ids=inputs).last_hidden_state[:, positions, :].reshape(-1, 9)
    reference_loss = ordinary_chunked_selective_linear_cross_entropy(
        hidden, reference.lm_head.weight, targets, global_target_count=11, chunk_size=128
    )
    output.loss.backward()
    reference_loss.backward()

    assert torch.equal(output.loss, reference_loss)
    observed_parameters = dict(observed.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert list(observed_parameters) == list(reference_parameters)
    for name in observed_parameters:
        assert observed_parameters[name].grad is not None
        assert torch.equal(observed_parameters[name].grad, reference_parameters[name].grad), name


def test_installed_forward_zero_target_is_graph_connected_and_projection_free():
    pytest.importorskip("transformers")
    model = _CausalModel()
    install_qwen35_checkpointed_chunked_loss(model, chunk_size=128)
    output = model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        labels=torch.full((1, 3), -100, dtype=torch.long),
        logits_to_keep=torch.tensor([0], dtype=torch.long),
        shift_labels=torch.tensor([-100], dtype=torch.long),
        num_items_in_batch=None,
    )
    output.loss.backward()

    assert output.loss.item() == 0.0
    assert output.logits is None
    assert model._qwen35_last_loss_audit["zero_target"] is True
    assert model._qwen35_last_loss_audit["maximum_logit_elements"] == 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_installer_rejects_unqualified_or_duplicate_installation():
    model = _CausalModel()
    with pytest.raises(ValueError, match="one of"):
        install_qwen35_checkpointed_chunked_loss(model, chunk_size=64)
    install_qwen35_checkpointed_chunked_loss(model, chunk_size=128)
    with pytest.raises(RuntimeError, match="already installed"):
        install_qwen35_checkpointed_chunked_loss(model, chunk_size=128)
