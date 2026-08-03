"""Convert Nemotron-Agentic-v1 tool_calling subset to Dolci format.

Converts from OpenAI-style messages to the format expected by OLMo 3's
apply_chat_template (matching allenai/Dolci-Instruct-SFT-Tool-Use).

Transformations:
  1. Create system message with FC instructions + tools in `functions` field
  2. Convert assistant `tool_calls` list → `function_calls` string (Python call syntax)
  3. Change role=tool → role=environment
  4. Stringify dict content in tool responses
  5. Add function_calls=None and functions=None on all messages
  6. Drop reasoning_content, id on tool_calls, tool_call_id on tool responses

Usage:
    python scripts/data/convert_nemotron_to_dolci_format.py --output_path nemotron_fc.parquet
"""

import argparse
import json

from huggingface_hub import hf_hub_download

SYSTEM_CONTENT = (
    "You are a helpful function-calling AI assistant. You are provided with function "
    "signatures within <functions></functions> XML tags. You may call one or more functions "
    "to assist with the user query. Output any function calls within "
    "<function_calls></function_calls> XML tags. Don't make assumptions about what values "
    "to plug into functions."
)


def format_argument_value(value):
    """Format a single argument value as Python literal."""
    if isinstance(value, str):
        # Use json.dumps to get proper escaping with double quotes
        return json.dumps(value)
    elif isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        return json.dumps(value)
    elif isinstance(value, dict):
        return json.dumps(value)
    elif value is None:
        return "None"
    else:
        return json.dumps(value)


def format_function_call(name, arguments):
    """Convert a tool call to Python-style function call string.

    Example: get_weather(location="Paris", days=5)
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError(f"arguments is {type(arguments).__name__}, expected dict")

    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={format_argument_value(value)}")
    return f"{name}({', '.join(parts)})"


def convert_sample(sample):
    """Convert one Nemotron sample to Dolci format."""
    tools = sample["tools"]
    functions_str = json.dumps(tools)

    messages = []

    # System message with tools
    messages.append({
        "role": "system",
        "content": SYSTEM_CONTENT,
        "function_calls": None,
        "functions": functions_str,
    })

    for msg in sample["messages"]:
        role = msg["role"]

        # Skip original system message (empty string in tool_calling)
        if role == "system":
            continue

        if role == "user":
            messages.append({
                "role": "user",
                "content": msg["content"],
                "function_calls": None,
                "functions": None,
            })

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Convert tool_calls list to function_calls string
                call_strings = []
                for call in tool_calls:
                    func = call["function"]
                    call_strings.append(format_function_call(func["name"], func["arguments"]))
                function_calls_str = "\n".join(call_strings)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_calls": function_calls_str,
                    "functions": None,
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "function_calls": None,
                    "functions": None,
                })

        elif role == "tool":
            content = msg["content"]
            if not isinstance(content, str):
                content = json.dumps(content)

            # Merge consecutive tool responses into a single environment message
            # (Dolci format: parallel call responses are newline-separated in one turn)
            if messages and messages[-1]["role"] == "environment":
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({
                    "role": "environment",
                    "content": content,
                    "function_calls": None,
                    "functions": None,
                })

    return {"messages": messages, "id": sample["uuid"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Output path (.jsonl)")
    args = parser.parse_args()

    print("Downloading tool_calling.jsonl...")
    path = hf_hub_download("nvidia/Nemotron-Agentic-v1", "data/tool_calling.jsonl", repo_type="dataset")

    print("Converting...")
    converted = []
    skipped_list_args = 0
    with open(path) as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            try:
                converted.append(convert_sample(sample))
            except ValueError as e:
                if "expected dict" in str(e):
                    skipped_list_args += 1
                else:
                    raise
            # Any other exception will propagate and crash — intentional.

    print(f"Converted: {len(converted)}")
    print(f"Skipped (list arguments): {skipped_list_args}")

    # Save as JSONL. Arrow/datasets can't handle the mixed None/string types
    # in nested structs during schema inference or even with explicit schemas.
    # JSONL sidesteps this entirely, and convert_sft_data_for_olmocore.py
    # supports .jsonl files via DatasetConfig.
    print(f"Saving to {args.output_path}...")
    with open(args.output_path, "w") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(converted)} rows.")


if __name__ == "__main__":
    main()
