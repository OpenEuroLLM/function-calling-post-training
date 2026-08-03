from types import SimpleNamespace

import pytest

from open_instruct.qwen35_reporting import (
    NOMINAL_A100_DENSE_BF16_FLOPS_PER_SECOND,
    Qwen35FlopFormula,
    Qwen35WindowCounts,
    append_jsonl,
    build_reporting_record,
    summarize_reporting_records,
)


def _qwen35_08b_config():
    return SimpleNamespace(
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=248_320,
        layer_types=["linear_attention"] * 18 + ["full_attention"] * 6,
    )


def _record(*, step=1, schedule_index=0, elapsed=2.0):
    formula = Qwen35FlopFormula.from_config(_qwen35_08b_config())
    counts = Qwen35WindowCounts(
        fixed_tokens=32_768,
        real_tokens=30_000,
        assistant_targets=6_000,
        padding_tokens=2_768,
        attention_length_squared=16_000_000,
        documents=9,
        packs=1,
        synthetic_packs=0,
    )
    return build_reporting_record(
        formula=formula,
        step=step,
        world_size=2,
        elapsed_seconds=elapsed,
        counts=counts,
        schedule_sha256="a" * 64,
        pack_uids=[f"pack-{schedule_index}"],
        schedule_indices=[schedule_index],
        learning_rate=2e-5,
        normalized_loss=1.5,
        global_target_divisor=6_000,
        peak_allocated_bytes=100,
        peak_reserved_bytes=120,
        synchronized=True,
    )


def test_qwen35_08b_formula_matches_independent_frozen_derivation():
    formula = Qwen35FlopFormula.from_config(_qwen35_08b_config())

    assert formula.num_gdn_layers == 18
    assert formula.num_full_attention_layers == 6
    assert formula.decoder_linear_training_flops_per_fixed_token == 2_985_689_088
    assert formula.gdn_training_flops_per_fixed_token == 99_090_432
    assert len(formula.formula_sha256) == 64

    fixed_tokens = 2_704_080_896
    assistant_targets = 569_590_984
    attention_length_squared = 13_645_681_167_814
    components = formula.window_flops(
        fixed_tokens=fixed_tokens,
        assistant_targets=assistant_targets,
        attention_length_squared=attention_length_squared,
    )
    assert components["decoder_linear_and_mlp"] / fixed_tokens == 2_985_689_088
    assert components["gdn_recurrence_approximation"] / fixed_tokens == 99_090_432
    assert components["document_isolated_causal_full_attention"] / fixed_tokens == pytest.approx(
        372_129_454.080094, abs=1e-6
    )
    assert components["selected_output_projection"] / fixed_tokens == pytest.approx(321_370_740.106154, abs=1e-6)
    assert components["total"] / fixed_tokens == pytest.approx(3_778_279_714.186248, abs=1e-6)


def test_formula_rejects_an_unrepresented_gdn_head_geometry():
    config = _qwen35_08b_config()
    config.linear_num_value_heads = config.linear_num_key_heads - 1

    with pytest.raises(ValueError, match="equal GDN key/value"):
        Qwen35FlopFormula.from_config(config)


def test_reporting_keeps_fixed_real_and_assistant_rates_distinct():
    record = _record()

    assert record["rates"]["fixed_tokens_per_second_global"] == 16_384
    assert record["rates"]["fixed_tokens_per_second_per_gpu"] == 8_192
    assert record["rates"]["real_tokens_per_second_global"] == 15_000
    assert record["rates"]["assistant_targets_per_second_global"] == 3_000
    assert record["loss"]["global_assistant_target_divisor"] == 6_000
    assert record["optimizer"]["applied_learning_rates"] == [2e-5]
    assert record["analytic_flops"]["isolated_causal_attention_pairs"] == (16_000_000 + 32_768) // 2
    expected_mfu = record["analytic_flops"]["components"]["total"] / (
        2 * NOMINAL_A100_DENSE_BF16_FLOPS_PER_SECOND * 2.0
    )
    assert record["analytic_flops"]["analytic_model_mfu"] == expected_mfu


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda kwargs: kwargs.update(step=0), "step"),
        (lambda kwargs: kwargs.update(elapsed_seconds=0), "elapsed"),
        (lambda kwargs: kwargs["counts"].__setattr__("padding_tokens", 0), "do not sum"),
        (lambda kwargs: kwargs.update(global_target_divisor=1), "divisor"),
        (lambda kwargs: kwargs.update(pack_uids=[]), "identity"),
        (lambda kwargs: kwargs.update(peak_allocated_bytes=121), "cannot exceed"),
    ],
)
def test_reporting_record_rejects_internally_inconsistent_metrics(mutation, message):
    kwargs = {
        "formula": Qwen35FlopFormula.from_config(_qwen35_08b_config()),
        "step": 1,
        "world_size": 1,
        "elapsed_seconds": 1.0,
        "counts": Qwen35WindowCounts(fixed_tokens=8, real_tokens=7, assistant_targets=2, padding_tokens=1, packs=1),
        "schedule_sha256": "a" * 64,
        "pack_uids": ["p"],
        "schedule_indices": [0],
        "learning_rate": 1e-5,
        "normalized_loss": 1.0,
        "global_target_divisor": 2,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 120,
        "synchronized": True,
    }
    mutation(kwargs)
    with pytest.raises(ValueError, match=message):
        build_reporting_record(**kwargs)


def test_jsonl_and_summary_preserve_exact_totals(tmp_path):
    first = _record(step=1, schedule_index=0, elapsed=2.0)
    second = _record(step=2, schedule_index=1, elapsed=3.0)
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, first)
    append_jsonl(path, second)

    summary = summarize_reporting_records([first, second])

    assert len(path.read_text().splitlines()) == 2
    assert summary["reporting_windows"] == 2
    assert summary["optimizer_steps"] == 2
    assert summary["elapsed_seconds"] == 5.0
    assert summary["counts"]["fixed_tokens"] == 65_536
    assert summary["counts"]["assistant_targets"] == 12_000
    assert summary["aggregate_rates"]["fixed_tokens_per_second_global"] == 65_536 / 5
    assert summary["analytic_flops"]["total"] == sum(
        row["analytic_flops"]["components"]["total"] for row in (first, second)
    )


@pytest.mark.parametrize("fault", ["schedule", "formula", "step", "repeat", "world"])
def test_summary_rejects_mixed_or_repeated_streams(fault):
    first = _record(step=1, schedule_index=0)
    second = _record(step=2, schedule_index=1)
    if fault == "schedule":
        second["schedule_sha256"] = "b" * 64
    elif fault == "formula":
        second["analytic_flops"]["formula_sha256"] = "b" * 64
    elif fault == "step":
        second["step"] = 3
    elif fault == "repeat":
        second["schedule_indices"] = [0]
    else:
        second["world_size"] = 4

    with pytest.raises(ValueError):
        summarize_reporting_records([first, second])
