import copy

import pytest

from scripts.train.qwen35.validate_qwen35_h2_r16_postfailure_diagnostic import (
    seed_from_label,
    selected_target_count,
    validate_partition,
    validate_scalar,
    validate_tensor,
)


def tensor_metric(*, elements=4, norm=2.0):
    return {
        "elements": elements,
        "maximum_absolute_error": 0.0,
        "relative_l2_error": 0.0,
        "cosine_similarity": 1.0,
        "observed_l2_norm": norm,
        "reference_l2_norm": norm,
        "difference_l2_norm": 0.0,
        "nonfinite_count": 0,
    }


def test_validator_rejects_derived_tensor_arithmetic_tampering():
    metric = tensor_metric()
    validate_tensor(metric, "exact")
    fabricated = copy.deepcopy(metric)
    fabricated["relative_l2_error"] = 0.1
    with pytest.raises(ValueError, match="relative L2 drift"):
        validate_tensor(fabricated, "fabricated")


def test_validator_rejects_scalar_arithmetic_tampering():
    metric = {
        "observed": 2.0,
        "reference": 1.0,
        "maximum_absolute_error": 1.0,
        "relative_error": 1.0,
        "nonfinite_count": 0,
    }
    validate_scalar(metric, "valid")
    fabricated = copy.deepcopy(metric)
    fabricated["maximum_absolute_error"] = 0.0
    with pytest.raises(ValueError, match="absolute error drift"):
        validate_scalar(fabricated, "fabricated")


def test_validator_rejects_named_aggregate_energy_tampering():
    named = [tensor_metric(elements=1, norm=1.0) for _ in range(4)]
    aggregate = tensor_metric(elements=57_568, norm=2.0)
    validate_partition(aggregate, named[:-1] + [dict(named[-1], elements=57_565)], "valid-partition")
    fabricated = copy.deepcopy(aggregate)
    fabricated["observed_l2_norm"] = 3.0
    with pytest.raises(ValueError, match="observed_l2_norm energy mismatch"):
        validate_partition(fabricated, named[:-1] + [dict(named[-1], elements=57_565)], "fabricated")


def test_seed_and_supervision_recomputation():
    digest, seed = seed_from_label("qwen35-hardware-qualification-r16-postfailure-long-diagnostic-0")
    assert digest == "3ef6784c3d2bcb8eaea04f329dd59fde4849af42b2e72b2b28e2bac77327921a"
    assert seed == 1_056_340_044
    assert selected_target_count(32, 2, 0) == 15
    assert selected_target_count(32, 3, 1) == 10
