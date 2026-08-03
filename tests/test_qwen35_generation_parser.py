import pytest
from scripts.train.qwen35.validate_qwen35_generation_parser import (
    generation_cases,
    parse_qwen35_tool_output,
    validate_fixed_parser_corpus,
)


def test_fixed_parser_corpus_passes_and_rejects_every_invalid_case():
    report = validate_fixed_parser_corpus()

    assert report == {"valid_cases": 3, "invalid_cases": 5, "invalid_cases_rejected": 5}


def test_parser_preserves_plain_content_around_parallel_calls():
    output = (
        "preface\n"
        '<tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>\n'
        '<tool_call>{"name":"lookup","arguments":{"x":2}}</tool_call>\n'
        "epilogue"
    )

    parsed = parse_qwen35_tool_output(output, allowed_tool_names={"lookup"})

    assert parsed["content"] == "preface\n\n\nepilogue"
    assert parsed["tool_calls"] == [
        {"name": "lookup", "arguments": {"x": 1}},
        {"name": "lookup", "arguments": {"x": 2}},
    ]


def test_generation_corpus_covers_every_preregistered_semantic_shape():
    cases = generation_cases()

    assert [case["case_id"] for case in cases] == [
        "explicit_single_call",
        "explicit_parallel_calls",
        "sequential_second_call",
        "multi_turn_followup_call",
        "justified_no_call",
    ]
    assert [case["expected_call_names"] for case in cases] == [
        ["get_weather"],
        ["get_weather", "get_weather"],
        ["get_weather"],
        ["get_weather"],
        [],
    ]
    assert any(message["role"] == "tool" for message in cases[2]["messages"])
    assert sum(message["role"] == "user" for message in cases[3]["messages"]) == 2


@pytest.mark.parametrize(
    "output",
    [
        "</tool_call>",
        "<tool_call>{}</tool_call>",
        '<tool_call>{"name":"lookup","arguments":[]}</tool_call>',
        '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
        '<tool_call>{"name":"lookup","arguments":{}}',
        '<tool_call><tool_call>{"name":"lookup","arguments":{}}</tool_call></tool_call>',
    ],
)
def test_parser_fails_closed_on_malformed_or_unavailable_calls(output):
    with pytest.raises(ValueError):
        parse_qwen35_tool_output(output, allowed_tool_names={"lookup"})
