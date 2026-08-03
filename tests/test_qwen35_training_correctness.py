import json
from copy import deepcopy

import pytest
import torch

from open_instruct.qwen35_training import (
    build_text_conversion_ledger,
    conditional_source_key_for_text_target,
    reference_selective_linear_cross_entropy,
    select_supervised_predecessor_rows,
    tensor_sha256,
    validate_fp32_optimizer_state,
    validate_fp32_trainable_parameters,
    validate_text_loading_info,
    write_json_atomic,
)


def _dense_masked_reference(hidden_states, weight, labels, divisor):
    logits = torch.nn.functional.linear(hidden_states[:, :-1, :], weight).float()
    return (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="sum"
        )
        / divisor
    )


def test_selective_rows_are_exact_shifted_predecessors():
    hidden = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    labels = torch.tensor([[-100, 2, -100, 4, 5, -100]])

    rows, targets, positions = select_supervised_predecessor_rows(hidden, labels)

    assert positions.tolist() == [0, 2, 3]
    assert targets.tolist() == [2, 4, 5]
    assert torch.equal(rows, hidden[0, [0, 2, 3]])


@pytest.mark.parametrize(
    "labels", [[[-100, 1, 2, 3, 4, 5]], [[-100, -100, 2, -100, 4, -100]], [[-100, -100, -100, -100, -100, 5]]]
)
def test_selective_loss_and_gradients_equal_dense_masked_reference(labels):
    generator = torch.Generator().manual_seed(1701)
    hidden = torch.randn(1, 6, 7, generator=generator, requires_grad=True)
    weight = torch.randn(11, 7, generator=generator, requires_grad=True)
    labels = torch.tensor(labels)
    divisor = int(labels[:, 1:].ne(-100).sum()) + 7

    selective = reference_selective_linear_cross_entropy(hidden, weight, labels, global_target_count=divisor)
    selective_gradients = torch.autograd.grad(selective, (hidden, weight), retain_graph=True)
    dense = _dense_masked_reference(hidden, weight, labels, divisor)
    dense_gradients = torch.autograd.grad(dense, (hidden, weight))

    assert selective == dense
    # The subset and dense GEMMs can accumulate in a different order on CPU,
    # so require numerical equality at a substantially tighter tolerance than
    # BF16 qualification rather than requiring byte identity.
    assert torch.allclose(selective_gradients[0], dense_gradients[0], rtol=1e-6, atol=1e-7)
    assert torch.allclose(selective_gradients[1], dense_gradients[1], rtol=1e-6, atol=1e-7)


def test_unequal_rank_target_normalization_sums_to_one_global_mean():
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(13, 5, generator=generator, requires_grad=True)
    left_hidden = torch.randn(1, 4, 5, generator=generator, requires_grad=True)
    right_hidden = torch.randn(1, 4, 5, generator=generator, requires_grad=True)
    left_labels = torch.tensor([[-100, 1, -100, 2]])
    right_labels = torch.tensor([[-100, 3, 4, 5]])
    global_count = 5

    distributed_sum = reference_selective_linear_cross_entropy(
        left_hidden, weight, left_labels, global_target_count=global_count
    ) + reference_selective_linear_cross_entropy(right_hidden, weight, right_labels, global_target_count=global_count)
    concatenated_sum = _dense_masked_reference(
        left_hidden, weight, left_labels, global_count
    ) + _dense_masked_reference(right_hidden, weight, right_labels, global_count)

    assert distributed_sum == concatenated_sum


def test_zero_target_loss_is_zero_but_keeps_hidden_and_head_in_graph():
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    weight = torch.randn(7, 3, requires_grad=True)
    labels = torch.full((1, 4), -100)

    loss = reference_selective_linear_cross_entropy(hidden, weight, labels)
    loss.backward()

    assert loss.item() == 0
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0


@pytest.mark.parametrize(
    ("hidden_shape", "label_shape", "message"),
    [((2, 4, 3), (2, 4), "batch one"), ((1, 4, 3), (1, 5), "shapes differ")],
)
def test_selective_row_helper_rejects_unsupported_shapes(hidden_shape, label_shape, message):
    with pytest.raises(ValueError, match=message):
        select_supervised_predecessor_rows(torch.zeros(hidden_shape), torch.zeros(label_shape, dtype=torch.long))


def test_global_divisor_must_be_positive_for_live_targets():
    with pytest.raises(ValueError, match="positive"):
        reference_selective_linear_cross_entropy(
            torch.zeros(1, 2, 3), torch.zeros(5, 3), torch.tensor([[-100, 1]]), global_target_count=0
        )


def test_fp32_parameter_and_adamw_state_guards_accept_only_full_precision():
    model = torch.nn.Linear(4, 3)
    parameter_report = validate_fp32_trainable_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    state_report = validate_fp32_optimizer_state(optimizer, require_initialized=True)

    assert parameter_report["parameter_dtype"] == "torch.float32"
    assert state_report["optimizer_tensor_state_dtypes"] == ["torch.float32"]
    assert state_report["initialized_parameter_states"] == 2

    half_model = deepcopy(model).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="must remain FP32"):
        validate_fp32_trainable_parameters(half_model)


def test_optimizer_state_guard_rejects_uninitialized_and_non_fp32_state():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="not initialized"):
        validate_fp32_optimizer_state(optimizer, require_initialized=True)
    parameter = next(model.parameters())
    optimizer.state[parameter] = {"exp_avg": torch.zeros_like(parameter, dtype=torch.bfloat16)}
    with pytest.raises(RuntimeError, match="expected torch.float32"):
        validate_fp32_optimizer_state(optimizer, require_initialized=False)


def test_loading_info_validator_and_key_mapper_fail_closed():
    validate_text_loading_info({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_text_loading_info({"missing_keys": ["model.bad"]})
    assert conditional_source_key_for_text_target("model.layers.0.weight") == "model.language_model.layers.0.weight"
    assert conditional_source_key_for_text_target("lm_head.weight") == "model.language_model.embed_tokens.weight"
    with pytest.raises(ValueError, match="unexpected"):
        conditional_source_key_for_text_target("visual.weight")


def test_tensor_hash_and_atomic_json_are_stable(tmp_path):
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    assert tensor_sha256(tensor) == tensor_sha256(tensor.clone())
    assert tensor_sha256(tensor) != tensor_sha256(tensor + 1)
    output = tmp_path / "nested" / "ledger.json"
    write_json_atomic(output, {"b": 2, "a": 1})
    assert json.loads(output.read_text()) == {"a": 1, "b": 2}


def test_tiny_conditional_checkpoint_converts_losslessly_to_text_causal_lm(tmp_path):
    transformers = pytest.importorskip("transformers")
    required = [
        "Qwen3_5Config",
        "Qwen3_5TextConfig",
        "Qwen3_5VisionConfig",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
    ]
    if any(not hasattr(transformers, name) for name in required):
        pytest.skip("installed Transformers has no Qwen3.5 text and conditional classes")
    text_config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True,
    )
    vision_config = transformers.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    full_config = transformers.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=True,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    torch.manual_seed(7)
    conditional = transformers.Qwen3_5ForConditionalGeneration(full_config).eval()
    checkpoint = tmp_path / "conditional"
    conditional.save_pretrained(checkpoint)

    causal, loading_info = transformers.Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=text_config, output_loading_info=True
    )
    causal.eval()
    validate_text_loading_info(loading_info)

    conditional_state = conditional.state_dict()
    for target_key, target_tensor in causal.state_dict().items():
        source_key = conditional_source_key_for_text_target(target_key)
        assert torch.equal(target_tensor, conditional_state[source_key]), target_key
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = conditional(input_ids=input_ids).logits
        actual = causal(input_ids=input_ids).logits
    assert torch.equal(actual, expected)
    ledger = build_text_conversion_ledger(
        causal, source_model="synthetic/qwen35", source_revision="revision", hash_tensors=True
    )
    assert ledger["target_class"] == "Qwen3_5ForCausalLM"
    assert ledger["target_config_model_type"] == "qwen3_5_text"
    assert ledger["tied_input_output_embeddings"] is True
    assert ledger["state_tensor_count"] == len(causal.state_dict())
    assert all(row["tensor_sha256"] for row in ledger["rows"])
