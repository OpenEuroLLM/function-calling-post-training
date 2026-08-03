import copy

import pytest

from scripts.train.qwen35.validate_qwen35_r16_dense_selection_forensic import (
    expected_batch_accounting,
    localization_from_comparisons,
    selected_positions,
    validate_repeatability,
    validate_scalar,
    validate_tensor_metric,
)


def tensor_metric(*, elements=4, maximum=0.0, difference=0.0, observed=2.0, reference=2.0, cosine=1.0):
    return {
        "elements": elements,
        "maximum_absolute_error": maximum,
        "relative_l2_error": difference / reference,
        "cosine_similarity": cosine,
        "observed_l2_norm": observed,
        "reference_l2_norm": reference,
        "difference_l2_norm": difference,
        "nonfinite_count": 0,
    }


def scalar_metric():
    return {
        "observed": 1.0,
        "reference": 1.0,
        "maximum_absolute_error": 0.0,
        "relative_error": 0.0,
        "nonfinite_count": 0,
    }


def metric_set():
    return {
        "loss": scalar_metric(),
        "selected_logits": tensor_metric(),
        "full_hidden_gradient": tensor_metric(),
        "selected_hidden_gradient": tensor_metric(),
        "output_weight_gradient": tensor_metric(),
        "selected_logit_gradient": tensor_metric(),
    }


def test_validator_recomputes_assay_accounting_and_selected_positions():
    accounting = expected_batch_accounting(step=55, trajectory_index=0, batch_seed_base=202_143_702)
    assert accounting == {
        "seed": 202_143_756,
        "sequence_length": 32,
        "supervision_modulus": 5,
        "supervision_offset": 4,
        "supervised_targets": 7,
        "divisor_extra": 6,
        "global_divisor": 13,
    }
    assert selected_positions(32, 5, 4) == [0, 5, 10, 15, 20, 25, 30]


def test_validator_rejects_scalar_metric_tampering():
    metric = scalar_metric()
    validate_scalar(metric, "valid")
    fabricated = copy.deepcopy(metric)
    fabricated["maximum_absolute_error"] = 1.0
    with pytest.raises(ValueError, match="scalar absolute error drift"):
        validate_scalar(fabricated, "fabricated")


def test_validator_rejects_tensor_norm_law_tampering():
    metric = tensor_metric()
    validate_tensor_metric(metric, "valid")
    fabricated = copy.deepcopy(metric)
    fabricated["relative_l2_error"] = 0.1
    with pytest.raises(ValueError, match="relative L2 drift"):
        validate_tensor_metric(fabricated, "fabricated")


def test_validator_rejects_repeatability_summary_tampering():
    hashes = {
        field: {"sha256": "a" * 64}
        for field in (
            "selected_logits",
            "full_hidden_gradient",
            "selected_hidden_gradient",
            "output_weight_gradient",
            "selected_logit_gradient",
        )
    }
    path = {
        "repeats": [{"loss": 1.0, "tensors": copy.deepcopy(hashes)} for _ in range(5)],
        "repeatability": {
            "loss_bit_exact": True,
            "tensor_hashes_bit_exact": {field: True for field in hashes},
            "all_recorded_outputs_bit_exact": False,
        },
    }
    with pytest.raises(ValueError, match="repeatability summary drift"):
        validate_repeatability(path, "fabricated")


def test_preregistered_localization_distinguishes_backward_and_forward_effects():
    ab = metric_set()
    backward = metric_set()
    backward["full_hidden_gradient"] = tensor_metric(maximum=0.1, difference=0.1, observed=2.0, reference=2.0, cosine=0.99875)
    backward["selected_hidden_gradient"] = copy.deepcopy(backward["full_hidden_gradient"])
    comparisons = {
        ("full_ignore", "full_gather"): ab,
        ("full_gather", "selected_gather"): backward,
    }
    assert localization_from_comparisons(comparisons) == "projection_shape_backward_hidden_input_gradient_only"
    forward = copy.deepcopy(backward)
    forward["selected_logits"] = copy.deepcopy(backward["full_hidden_gradient"])
    comparisons[("full_gather", "selected_gather")] = forward
    assert localization_from_comparisons(comparisons) == "projection_shape_forward_rounding_and_downstream_effect"
