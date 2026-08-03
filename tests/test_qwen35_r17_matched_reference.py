import pytest
import torch

pytest.importorskip("liger_kernel", reason="R17 matched-reference assay requires the pinned qualification runtime")

from scripts.train.qwen35.validate_qwen35_selective_loss_r17 import _dense_selected_loss, _diagnostic_nonfinite_count
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def tiny_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(1701)
    return Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["full_attention"],
            tie_word_embeddings=True,
            use_cache=False,
        )
    ).train()


def test_dense_selected_reference_uses_shifted_supervised_prediction_positions():
    model = tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, [1, 4, 7]] = input_ids[0, [1, 4, 7]]
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "selected_positions": torch.tensor([0, 3, 6], dtype=torch.long),
        "selected_targets": labels[:, 1:][labels[:, 1:] != -100],
        "accounting": {"global_divisor": 11},
    }
    selected_loss = _dense_selected_loss(model, batch)
    full_loss = model(
        input_ids=input_ids, labels=labels, num_items_in_batch=batch["accounting"]["global_divisor"], use_cache=False
    ).loss
    assert torch.isfinite(selected_loss)
    assert torch.isfinite(full_loss)
    assert torch.allclose(selected_loss, full_loss, atol=1e-6, rtol=1e-6)


def test_diagnostic_nonfinite_counter_is_scoped_and_fail_closed():
    report = {
        "primary": {"nonfinite_count": 7, "value": float("inf")},
        "full_dense_diagnostic": {
            "metric": {"nonfinite_count": 2},
            "observed_loss": None,
            "nested": {"reference_loss": float("inf")},
        },
    }
    assert _diagnostic_nonfinite_count(report) == 4
