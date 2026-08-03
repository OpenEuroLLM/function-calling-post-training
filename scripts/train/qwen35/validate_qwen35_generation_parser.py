#!/usr/bin/env python3
"""Validate strict Qwen tool parsing and single-versus-batched generation.

This is a post-resume G2 compatibility gate, not a benchmark.  It first tests
the parser on fixed valid and invalid native-tool strings, then greedily
generates a small fixed prompt set both one at a time and as one left-padded
batch.  Generated token IDs must be exactly invariant, and the decoded outputs
must satisfy the preregistered call/no-call expectations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import Qwen2TokenizerFast, Qwen3_5ForCausalLM

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_qwen35_tool_output(text: str, *, allowed_tool_names: set[str]) -> dict[str, Any]:
    """Parse complete native ``<tool_call>`` blocks without repairing output."""

    if not isinstance(text, str):
        raise TypeError("model output must be a string")
    calls: list[dict[str, Any]] = []
    text_fragments: list[str] = []
    cursor = 0
    while True:
        opening = text.find(TOOL_CALL_OPEN, cursor)
        stray_closing = text.find(TOOL_CALL_CLOSE, cursor)
        if opening < 0:
            if stray_closing >= 0:
                raise ValueError("native output contains a closing tool-call tag without an opening tag")
            text_fragments.append(text[cursor:])
            break
        if 0 <= stray_closing < opening:
            raise ValueError("native output contains a closing tool-call tag before the next opening tag")
        text_fragments.append(text[cursor:opening])
        body_start = opening + len(TOOL_CALL_OPEN)
        closing = text.find(TOOL_CALL_CLOSE, body_start)
        if closing < 0:
            raise ValueError("native output contains an unclosed tool-call block")
        body = text[body_start:closing].strip()
        if TOOL_CALL_OPEN in body or TOOL_CALL_CLOSE in body:
            raise ValueError("native output contains nested tool-call tags")
        try:
            call = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool-call body is not one complete JSON value: {exc}") from exc
        if not isinstance(call, Mapping):
            raise ValueError("tool-call JSON must be an object")
        if set(call) != {"name", "arguments"}:
            raise ValueError("tool-call JSON must contain exactly name and arguments")
        name = call["name"]
        arguments = call["arguments"]
        if not isinstance(name, str) or not name:
            raise ValueError("tool-call name must be a non-empty string")
        if name not in allowed_tool_names:
            raise ValueError(f"tool-call references unavailable tool {name!r}")
        if not isinstance(arguments, Mapping):
            raise ValueError("tool-call arguments must be a JSON object")
        calls.append({"name": name, "arguments": dict(arguments)})
        cursor = closing + len(TOOL_CALL_CLOSE)

    return {"content": "".join(text_fragments).strip(), "tool_calls": calls}


def validate_fixed_parser_corpus() -> dict[str, Any]:
    valid = [
        ("The answer is hello.", set(), 0),
        ('<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>', {"get_weather"}, 1),
        (
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>\n'
            '<tool_call>{"name":"get_weather","arguments":{"city":"London"}}</tool_call>',
            {"get_weather"},
            2,
        ),
    ]
    invalid = [
        ("</tool_call>", {"get_weather"}),
        ("<tool_call>{}</tool_call>", {"get_weather"}),
        ('<tool_call>{"name":"unknown","arguments":{}}</tool_call>', {"get_weather"}),
        ('<tool_call>{"name":"get_weather","arguments":[]}</tool_call>', {"get_weather"}),
        ('<tool_call>{"name":"get_weather","arguments":{}}', {"get_weather"}),
    ]
    for output, allowed, expected_calls in valid:
        parsed = parse_qwen35_tool_output(output, allowed_tool_names=allowed)
        if len(parsed["tool_calls"]) != expected_calls:
            raise AssertionError("fixed valid parser case has unexpected call count")
    rejected = 0
    for output, allowed in invalid:
        try:
            parse_qwen35_tool_output(output, allowed_tool_names=allowed)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("fixed invalid parser case was accepted")
    return {"valid_cases": len(valid), "invalid_cases": len(invalid), "invalid_cases_rejected": rejected}


def function_tool(
    name: str, description: str, properties: Mapping[str, Any], required: Sequence[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": dict(properties), "required": list(required)},
        },
    }


def generation_cases() -> list[dict[str, Any]]:
    weather = function_tool(
        "get_weather",
        "Get the current weather in a city.",
        {"city": {"type": "string", "description": "City name"}},
        ["city"],
    )
    user_lookup = function_tool(
        "get_user",
        "Get a user record by numeric identifier.",
        {"user_id": {"type": "integer", "description": "User identifier"}},
        ["user_id"],
    )
    return [
        {
            "case_id": "explicit_single_call",
            "messages": [{"role": "user", "content": "Use the weather tool to get the weather in Paris."}],
            "tools": [weather],
            "expected_call_names": ["get_weather"],
        },
        {
            "case_id": "explicit_parallel_calls",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the weather tool for both Paris and London. "
                        "Emit both independent calls together in this response."
                    ),
                }
            ],
            "tools": [weather],
            "expected_call_names": ["get_weather", "get_weather"],
        },
        {
            "case_id": "sequential_second_call",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Get user 7, then use the city in that record to get the weather. Use the tools in sequence."
                    ),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_user_7",
                            "type": "function",
                            "function": {"name": "get_user", "arguments": {"user_id": 7}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_user_7", "content": '{"name":"Alice","city":"Paris"}'},
            ],
            "tools": [user_lookup, weather],
            "expected_call_names": ["get_weather"],
        },
        {
            "case_id": "multi_turn_followup_call",
            "messages": [
                {"role": "user", "content": "Use the weather tool for Paris."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather_paris",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather_paris",
                    "content": '{"city":"Paris","condition":"sunny"}',
                },
                {"role": "assistant", "content": "Paris is sunny."},
                {"role": "user", "content": "Now use the same tool for London."},
            ],
            "tools": [weather],
            "expected_call_names": ["get_weather"],
        },
        {
            "case_id": "justified_no_call",
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "tools": [weather],
            "expected_call_names": [],
        },
    ]


def render_prompts(tokenizer: Qwen2TokenizerFast, cases: Sequence[Mapping[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for case in cases:
        rendered = tokenizer.apply_chat_template(
            list(case["messages"]),
            tools=list(case["tools"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise TypeError("native generation prompt did not render to text")
        prompts.append(rendered)
    return prompts


def trim_generated(ids: Sequence[int], *, pad_token_id: int) -> list[int]:
    output = [int(token_id) for token_id in ids]
    while output and output[-1] == pad_token_id:
        output.pop()
    return output


def greedy_generate(
    model: Qwen3_5ForCausalLM, tokenizer: Qwen2TokenizerFast, prompts: Sequence[str], *, max_new_tokens: int
) -> list[list[int]]:
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        trim_generated(row[prompt_width:].detach().cpu().tolist(), pad_token_id=tokenizer.pad_token_id)
        for row in generated
    ]


def validate_generation(checkpoint: Path, *, max_new_tokens: int) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("generation validation requires CUDA")
    tokenizer = Qwen2TokenizerFast.from_pretrained(checkpoint)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        raise ValueError("checkpoint tokenizer must define a pad token ID")
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    model.eval()
    cases = generation_cases()
    prompts = render_prompts(tokenizer, cases)
    independent = [greedy_generate(model, tokenizer, [prompt], max_new_tokens=max_new_tokens)[0] for prompt in prompts]
    batched = greedy_generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    if independent != batched:
        differing = [case["case_id"] for case, left, right in zip(cases, independent, batched) if left != right]
        raise AssertionError(f"single-versus-batched greedy generation token drift: {differing}")

    case_reports: list[dict[str, Any]] = []
    concatenated_ids = hashlib.sha256()
    for case, token_ids in zip(cases, independent, strict=True):
        concatenated_ids.update(torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes())
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        allowed = {tool["function"]["name"] for tool in case["tools"]}
        parsed = parse_qwen35_tool_output(decoded, allowed_tool_names=allowed)
        call_names = [call["name"] for call in parsed["tool_calls"]]
        if call_names != case["expected_call_names"]:
            raise AssertionError(
                f"{case['case_id']} produced call names {call_names!r} instead of {case['expected_call_names']!r}"
            )
        case_reports.append(
            {
                "case_id": case["case_id"],
                "generated_token_count": len(token_ids),
                "generated_token_ids_sha256": hashlib.sha256(
                    torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes()
                ).hexdigest(),
                "decoded_output": decoded,
                "parsed": parsed,
                "single_batch_token_exact": True,
            }
        )
    return {
        "status": "passed",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_tokenizer_config_sha256": sha256_file(checkpoint / "tokenizer_config.json"),
        "cuda_device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attention_implementation": model.config._attn_implementation,
        "generation_eos_token_id": model.generation_config.eos_token_id,
        "max_new_tokens": max_new_tokens,
        "single_batch_token_exact_for_all_cases": True,
        "concatenated_generated_token_ids_sha256": concatenated_ids.hexdigest(),
        "cases": case_reports,
    }


def runtime_versions() -> dict[str, str]:
    versions = {"transformers": transformers.__version__, "torch": torch.__version__}
    for package in ("flash-attn", "causal-conv1d", "flash-linear-attention", "fla-core"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    output = args.report_output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {
        "artifact": "qwen35_generation_parser_batch_invariance",
        "schema_version": 1,
        "status": "passed",
        "parser_corpus": validate_fixed_parser_corpus(),
        "runtime_versions": runtime_versions(),
        "generation": validate_generation(checkpoint, max_new_tokens=args.max_new_tokens),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
