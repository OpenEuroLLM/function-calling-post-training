from scripts.train.qwen35.summarize_qwen35_r17_failure import TRAJECTORY_CONTEXT, MetricExtrema


def test_r17_failure_context_parser_separates_trajectory_parameter_and_family() -> None:
    match = TRAJECTORY_CONTEXT.fullmatch(
        "R17-T2 step 509 parameter model.layers.0.self_attn.q_norm.weight preclip gradient"
    )

    aggregate = TRAJECTORY_CONTEXT.fullmatch("R17-T0 step 1 aggregate cumulative parameter displacement")
    assert aggregate is not None
    assert aggregate.groups() == ("R17-T0", "1", None, "aggregate cumulative parameter displacement")
    assert match is not None
    assert match.groups() == (
        "R17-T2",
        "509",
        "model.layers.0.self_attn.q_norm.weight",
        "preclip gradient",
    )


def test_metric_extrema_retains_values_and_contexts() -> None:
    extrema = MetricExtrema()
    extrema.add(
        {
            "maximum_absolute_error": 0.25,
            "relative_l2_error": 0.5,
            "cosine_similarity": 0.99,
            "nonfinite_count": 0,
        },
        "first",
    )
    extrema.add(
        {
            "maximum_absolute_error": 0.125,
            "relative_l2_error": 0.75,
            "cosine_similarity": 0.98,
            "nonfinite_count": 0,
        },
        "second",
    )
    value = extrema.as_dict()
    assert value["metric_count"] == 2
    assert value["nonfinite_count"] == 0
    assert value["maximum_maximum_absolute_error"] == {"value": 0.25, "context": "first"}
    assert value["maximum_relative_l2_error"] == {"value": 0.75, "context": "second"}
    assert value["minimum_cosine_similarity"] == {"value": 0.98, "context": "second"}
