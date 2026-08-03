import copy

import pytest

from scripts.train.qwen35.diagnose_qwen35_r16_dense_selection_divergence import CASES, _validate_parent_case


def parent_fixture(*, first_gradient_step=4, prior_state_difference=0.0):
    steps = []
    for step in range(1, 129):
        steps.append(
            {
                "step": step,
                "batch_accounting": {"seed": step},
                "aggregate_preclip_gradient": {
                    "difference_l2_norm": 1e-6 if step == first_gradient_step else 0.0
                },
                "aggregate_cumulative_displacement": {
                    "difference_l2_norm": prior_state_difference if step == first_gradient_step - 1 else 0.0
                },
            }
        )
    return {
        "runs": [
            {
                "variant": "bf16_dense_selected_vs_dense_full",
                "trajectory_contract": {"trajectory_id": "R16-T2"},
                "steps": steps,
            }
        ]
    }


def test_forensic_cases_freeze_the_observed_first_gradient_steps():
    assert [(case["trajectory_id"], case["replay_steps"], case["assay_step"]) for case in CASES] == [
        ("R16-T0", 54, 55),
        ("R16-T1", 64, 65),
        ("R16-T2", 3, 4),
    ]


def test_parent_validation_accepts_exact_first_divergence():
    result = _validate_parent_case(parent_fixture(), CASES[2])
    assert result["parent_first_gradient_difference_l2_norm"] == 1e-6
    assert result["parent_pre_assay_complete_state_bit_exact"] is True


def test_parent_validation_rejects_earlier_divergence_or_state_drift():
    earlier = parent_fixture()
    earlier["runs"][0]["steps"][1]["aggregate_preclip_gradient"]["difference_l2_norm"] = 1e-9
    with pytest.raises(RuntimeError, match="not the first"):
        _validate_parent_case(earlier, CASES[2])

    drifted = parent_fixture(prior_state_difference=1e-9)
    with pytest.raises(RuntimeError, match="common pre-assay parameter state"):
        _validate_parent_case(drifted, CASES[2])

    missing = copy.deepcopy(parent_fixture())
    missing["runs"][0]["trajectory_contract"]["trajectory_id"] = "wrong"
    with pytest.raises(RuntimeError, match="exactly one"):
        _validate_parent_case(missing, CASES[2])
