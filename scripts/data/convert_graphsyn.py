"""Convert ToolMind GraphSyn dataset to Dolci format.

Converts from OpenAI-style messages to the format expected by OLMo 3's
apply_chat_template (matching allenai/Dolci-Instruct-SFT-Tool-Use).

Transformations:
  1. Create system message with FC instructions + tools in `functions` field
  2. Convert assistant `tool_calls` list → `function_calls` string (Python call syntax)
  3. Change role=tool → role=environment
  4. Stringify dict content in tool responses
  5. Add function_calls=None and functions=None on all messages
  6. Normalize non-standard JSON Schema types in tool definitions
  7. Strip <think>...</think> reasoning blocks from assistant messages
  8. Truncate dangling tool-call tails (assistant tool call with no tool response)

Usage:
    # Download from HuggingFace:
    python scripts/data/convert_graphsyn_to_dolci_format.py --output_path graphsyn_fc.jsonl

    # From local file:
    python scripts/data/convert_graphsyn_to_dolci_format.py \
        --input_path /path/to/graphsyn.jsonl --output_path graphsyn_fc.jsonl
"""

import argparse
import json
import re
import uuid

# Same system prompt used in Dolci and Nemotron FC data.
SYSTEM_CONTENT = (
    "You are a helpful function-calling AI assistant. You are provided with function "
    "signatures within <functions></functions> XML tags. You may call one or more functions "
    "to assist with the user query. Output any function calls within "
    "<function_calls></function_calls> XML tags. Don't make assumptions about what values "
    "to plug into functions."
)

# GraphSyn uses non-standard JSON Schema types. Map them to standard ones.
# (from bfcl/analysis_code/loaders/toolmind.py)
_TYPE_FIXES = {
    "dict": "object",
    "str": "string",
    "str, optional": "string",
    "int": "integer",
    "int, optional": "integer",
    "float": "number",
    "float, optional": "number",
    "bool": "boolean",
    "bool, optional": "boolean",
    "list": "array",
    "List": "array",
    "list[str]": "array",
    "list[int]": "array",
    "list[float]": "array",
    "list[dict]": "array",
    "List[str]": "array",
    "List[int]": "array",
    "List[float]": "array",
    "List[dict]": "array",
    "array[str]": "array",
    "tuple": "array",
    "None": "null",
}


def _fix_schema_types(obj):
    """Recursively fix non-standard JSON Schema types in tool definitions."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str) and v in _TYPE_FIXES:
                result[k] = _TYPE_FIXES[v]
            else:
                result[k] = _fix_schema_types(v)
        return result
    elif isinstance(obj, list):
        return [_fix_schema_types(item) for item in obj]
    return obj


def _normalize_tool(tool):
    """Normalize a tool definition to OpenAI format.

    Handles double-wrapped tools, missing type:"object" wrapper on parameters,
    'arguments' key used instead of 'parameters', and non-standard types.
    """
    # Unwrap outer {type: "function", function: ...} if present
    func = tool.get("function", tool)

    # Unwrap double-wrapping
    if "function" in func and isinstance(func.get("function"), dict):
        inner = func["function"]
        if isinstance(inner, dict) and "name" in inner:
            func = inner

    name = func.get("name", "")
    description = func.get("description", "")
    params = func.get("parameters") or func.get("arguments") or {}

    # Wrap flat property maps in type: "object"
    if isinstance(params, dict) and "properties" not in params and params:
        looks_like_properties = all(
            isinstance(v, dict) and ("type" in v or "description" in v)
            for v in params.values()
            if isinstance(v, dict)
        )
        if looks_like_properties and any(isinstance(v, dict) for v in params.values()):
            params = {"type": "object", "properties": params}

    params = _fix_schema_types(params)

    # Move 'required' from func level to params if needed
    required = func.get("required")
    if required is not None and isinstance(params, dict) and "required" not in params:
        if isinstance(required, list):
            params["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


def _strip_thinking(content):
    """Strip <think>...</think> blocks from assistant content.

    Returns the content after the think block, stripped of leading whitespace.
    Returns None if the entire content is a think block with nothing after.
    """
    if not content or "<think>" not in content:
        return content
    # Think blocks are always at the start of content
    stripped = re.sub(r"^<think>.*?</think>\s*", "", content, count=1, flags=re.DOTALL)
    if not stripped:
        # Unclosed think tag or nothing after think block
        stripped = re.sub(r"^<think>.*", "", content, count=1, flags=re.DOTALL)
    return stripped.strip() or None


def _format_argument_value(value):
    """Format a single argument value as Python literal."""
    if isinstance(value, str):
        return json.dumps(value)
    elif isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "None"
    else:
        return json.dumps(value)


def _format_function_call(name, arguments):
    """Convert a tool call to Python-style function call string.

    Example: get_weather(location="Paris", days=5)
    """
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError(f"arguments is {type(arguments).__name__}, expected dict")

    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={_format_argument_value(value)}")
    return f"{name}({', '.join(parts)})"


def _truncate_dangling_tail(messages):
    """Remove dangling tool-call tail from the end of a conversation.

    If the conversation ends with an assistant tool call that has no tool
    response following, find the preceding user message and truncate from
    there onwards.

    Returns (messages, was_truncated). messages is None if truncation
    would leave fewer than 3 messages (system + user + assistant).
    """
    if not messages:
        return messages, False

    last = messages[-1]
    if last["role"] != "assistant" or not last.get("function_calls"):
        return messages, False

    # Walk backwards to find the last user message before the dangling call
    truncate_idx = None
    for i in range(len(messages) - 2, -1, -1):
        if messages[i]["role"] == "user":
            truncate_idx = i
            break

    if truncate_idx is None:
        return None, True

    truncated = messages[:truncate_idx]

    # Need at least system + user + assistant (one complete exchange)
    if len(truncated) < 3:
        return None, True

    return truncated, True


def convert_sample(sample):
    """Convert one GraphSyn sample to Dolci format.

    Returns list of messages in Dolci format, or raises on error.
    """
    raw_tools = sample.get("tools") or []
    tools = [_normalize_tool(t) for t in raw_tools]
    functions_str = json.dumps(tools)

    messages = []

    # System message with FC instructions + tools
    messages.append({
        "role": "system",
        "content": SYSTEM_CONTENT,
        "function_calls": None,
        "functions": functions_str,
    })

    for msg in sample["conversations"]:
        role = msg["role"]

        # Skip any existing system messages
        if role == "system":
            continue

        if role == "user":
            messages.append({
                "role": "user",
                "content": msg.get("content") or "",
                "function_calls": None,
                "functions": None,
            })

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                call_strings = []
                for call in tool_calls:
                    func = call.get("function", call)
                    name = func.get("name", "")
                    arguments = func.get("arguments", {})
                    call_strings.append(_format_function_call(name, arguments))
                function_calls_str = "\n".join(call_strings)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_calls": function_calls_str,
                    "functions": None,
                })
            else:
                content = _strip_thinking(msg.get("content") or "")
                if content is None:
                    # Entire message was a think block with no actual response;
                    # treat as empty text to preserve conversation structure.
                    content = ""
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "function_calls": None,
                    "functions": None,
                })

        elif role == "tool":
            content = msg.get("content") or ""
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

    return messages


def main():
    parser = argparse.ArgumentParser(description="Convert ToolMind GraphSyn to Dolci format")
    parser.add_argument(
        "--input_path", type=str, default=None,
        help="Path to local graphsyn.jsonl. Downloads from HuggingFace if not provided.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output path (.jsonl)")
    args = parser.parse_args()

    if args.input_path is None:
        from huggingface_hub import hf_hub_download

        print("Downloading graphsyn.jsonl from HuggingFace...")
        input_path = hf_hub_download(
            repo_id="Nanbeige/ToolMind",
            filename="graph_syn_datasets/graphsyn.jsonl",
            repo_type="dataset",
        )
    else:
        input_path = args.input_path

    print(f"Loading from: {input_path}")

    converted = []
    total = 0
    truncated = 0
    skipped_after_truncation = 0
    skipped_error = 0

    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1

            sample = json.loads(line)
            try:
                messages = convert_sample(sample)
            except (ValueError, KeyError, TypeError) as e:
                skipped_error += 1
                if skipped_error <= 5:
                    print(f"  Warning: skipped sample {i}: {e}")
                continue

            messages, was_truncated = _truncate_dangling_tail(messages)

            if messages is None:
                skipped_after_truncation += 1
                continue

            if was_truncated:
                truncated += 1

            converted.append({
                "messages": messages,
                "id": str(uuid.uuid4()),
            })

    print(f"\n=== Conversion Stats ===")
    print(f"Total input samples:           {total}")
    print(f"Converted:                     {len(converted)}")
    print(f"  - of which truncated tail:   {truncated}")
    print(f"Skipped (empty after trunc):   {skipped_after_truncation}")
    print(f"Skipped (conversion error):    {skipped_error}")

    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, "w") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(converted)} rows.")


if __name__ == "__main__":
    main()
