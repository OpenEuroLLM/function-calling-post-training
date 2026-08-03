"""Convert OpenAI-format JSONL to Dolci format for OLMo 3 training.

Takes JSONL produced by the fcanalysis curated pipeline (OpenAI chat format
with tools) and converts to the Dolci message format expected by the OLMo 3
chat template and tokenization pipeline (convert_sft_data_for_olmocore.py).

Transformations:
  1. Strip original system message; create new one with FC instructions + tools
  2. Convert assistant tool_calls list -> function_calls string (Python syntax)
  3. Change role=tool -> role=environment; merge consecutive tool responses
  4. Ensure every message has exactly 4 keys: role, content, function_calls, functions
  5. Extract tool["function"] from OpenAI tool wrapper for functions field

Input format (per line):
  {"messages": [...], "tools": [...], "dataset": "...", "sample_id": "..."}

Output format (per line):
  {"messages": [...], "id": "..."}

Parallelism: by default uses all CPUs via multiprocessing.Pool.imap (preserves
output order). Pass --num-workers 1 for the original sequential code path,
useful for debugging or for environments where multiprocessing is undesirable.
The per-sample conversion logic (convert_sample) is unchanged from the
single-process version, so the parallel and sequential paths produce
byte-identical output.

Usage:
    # Defaults: use all available CPUs.
    python scripts/data/convert_openai_to_dolci_format.py \\
        --input /path/to/openai_format.jsonl \\
        --output /path/to/dolci_format.jsonl

    # Force sequential (matches original script behavior exactly).
    python scripts/data/convert_openai_to_dolci_format.py \\
        --input ... --output ... --num-workers 1
"""

import argparse
import json
import multiprocessing
import os
import sys
from collections.abc import Iterator


SYSTEM_PERSONA = "You are a helpful function-calling AI assistant."
SYSTEM_INSTRUCTIONS = (
    "You are provided with function "
    "signatures within <functions></functions> XML tags. You may call one or more functions "
    "to assist with the user query. Output any function calls within "
    "<function_calls></function_calls> XML tags. Don't make assumptions about what values "
    "to plug into functions."
)
SYSTEM_CONTENT = f"{SYSTEM_PERSONA} {SYSTEM_INSTRUCTIONS}"


def build_system_content(source_system: str) -> str:
    """Build the system message content for a Dolci-format sample.

    Three cases:
    1. Source system is empty/whitespace → return canonical SYSTEM_CONTENT.
       Matches existing Phase 2 behavior for v1/GS/TxT360 (which had no source system).
    2. Source system already contains FC format markers ("<functions>" or
       "<function_calls>") → return source as-is. The source author already
       included the FC instructions; we trust it (Dolci's canonical or extended
       prompts; future curated datasets).
    3. Source system has content but no FC format markers → preserve source as
       persona/task context and append SYSTEM_INSTRUCTIONS. This handles
       v2.search (single persona), v2.interactive_agent (836 personas),
       nemotron_terminal (task descriptions), and any other dataset whose source
       supplies persona without FC instructions.
    """
    stripped = (source_system or "").strip()
    if not stripped:
        return SYSTEM_CONTENT
    if "<functions>" in stripped or "<function_calls>" in stripped:
        # Source already specifies FC format; trust it (don't duplicate instructions).
        return stripped
    return f"{stripped}\n\n{SYSTEM_INSTRUCTIONS}"


def format_argument_value(value):
    """Format a single argument value as a Python literal."""
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


def format_function_call(name, arguments):
    """Convert a tool call to Python-style function call string.

    Example: get_weather(location="Paris", days=5)
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError(
            f"arguments for {name} is {type(arguments).__name__}, expected dict"
        )

    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={format_argument_value(value)}")
    return f"{name}({', '.join(parts)})"


def convert_sample(sample):
    """Convert one OpenAI-format sample to Dolci format.

    Returns dict with 'messages' (Dolci format) and 'id'.
    """
    tools = sample.get("tools") or []

    # Extract function specs from OpenAI tool wrapper:
    # [{"type": "function", "function": {"name": ..., ...}}] -> [{"name": ..., ...}]
    functions = []
    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            functions.append(tool["function"])
        else:
            functions.append(tool)
    functions_str = json.dumps(functions) if functions else None

    # Extract source system message (if any) to preserve persona/task context.
    # Falls back to canonical SYSTEM_CONTENT when source has none.
    source_system_content = ""
    for msg in sample["messages"]:
        if msg.get("role") == "system":
            source_system_content = msg.get("content") or ""
            break

    messages = []

    # System message with persona (source's if present, generic otherwise) +
    # FC format instructions + tool definitions.
    messages.append({
        "role": "system",
        "content": build_system_content(source_system_content),
        "function_calls": None,
        "functions": functions_str,
    })

    for msg in sample["messages"]:
        role = msg["role"]

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
            content = msg.get("content")

            if tool_calls:
                call_strings = []
                for call in tool_calls:
                    func = call["function"]
                    call_strings.append(
                        format_function_call(func["name"], func["arguments"])
                    )
                function_calls_str = "\n".join(call_strings)

                messages.append({
                    "role": "assistant",
                    "content": content if content else None,
                    "function_calls": function_calls_str,
                    "functions": None,
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": content if content else "",
                    "function_calls": None,
                    "functions": None,
                })

        elif role == "tool":
            tool_content = msg.get("content") or ""
            if not isinstance(tool_content, str):
                tool_content = json.dumps(tool_content)

            if messages and messages[-1]["role"] == "environment":
                messages[-1]["content"] += "\n" + tool_content
            else:
                messages.append({
                    "role": "environment",
                    "content": tool_content,
                    "function_calls": None,
                    "functions": None,
                })

    sample_id = sample.get("sample_id", sample.get("id", "unknown"))
    return {"messages": messages, "id": str(sample_id)}


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _convert_chunk(chunk):
    """Worker: convert a list of (line_index, raw_line) tuples.

    Returns (converted_lines, skipped_records) where:
      - converted_lines is a list of JSON strings (no trailing newline)
      - skipped_records is a list of (line_index, error_message) tuples

    The conversion logic mirrors the sequential path exactly: empty lines are
    silently skipped, parse/conversion errors are caught and recorded, all
    other errors propagate so we fail fast on logic bugs.
    """
    converted = []
    skipped = []
    for i, line in chunk:
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            result = convert_sample(sample)
            converted.append(json.dumps(result))
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            skipped.append((i, str(e)))
    return converted, skipped


def _chunked_lines(file_handle, chunk_size: int) -> Iterator[list[tuple[int, str]]]:
    """Stream the input file in chunks of (line_index, raw_line) tuples.

    Memory bounded by chunk_size * line_size (one chunk in flight in the
    main process at a time; multiprocessing.Pool buffers a small number of
    chunks across workers).
    """
    chunk: list[tuple[int, str]] = []
    for i, line in enumerate(file_handle):
        chunk.append((i, line))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _convert_sequential(input_path: str, output_path: str):
    """Single-process conversion. Identical logic to the original script."""
    converted = 0
    skipped = 0
    errors: list[str] = []

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                result = convert_sample(sample)
                fout.write(json.dumps(result) + "\n")
                converted += 1
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                skipped += 1
                if len(errors) < 10:
                    errors.append(f"  line {i}: {e}")

    return converted, skipped, errors


def _convert_parallel(
    input_path: str,
    output_path: str,
    num_workers: int,
    chunk_size: int,
):
    """Multi-process conversion using Pool.imap (preserves output order).

    Workers process chunks of lines independently. The main process iterates
    Pool.imap (which yields results in input order) and writes to output.
    Output order, sample-by-sample, is identical to the sequential path,
    which is the property we rely on for byte-identical results across
    different worker counts.
    """
    converted = 0
    skipped = 0
    errors: list[str] = []

    with open(input_path) as fin, open(output_path, "w") as fout:
        with multiprocessing.Pool(num_workers) as pool:
            chunks = _chunked_lines(fin, chunk_size)
            for conv_lines, skip_records in pool.imap(_convert_chunk, chunks):
                for line in conv_lines:
                    fout.write(line + "\n")
                converted += len(conv_lines)
                skipped += len(skip_records)
                for line_idx, err in skip_records:
                    if len(errors) < 10:
                        errors.append(f"  line {line_idx}: {err}")

    return converted, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenAI-format JSONL to Dolci format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL file (OpenAI format from fcanalysis pipeline)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file (Dolci format for OLMo 3 training)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of worker processes. 0 (default) = os.cpu_count(); "
             "1 = single-process (sequential, identical to the original "
             "script's behavior); >1 = explicit worker count.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Lines per chunk dispatched to a worker. Smaller chunks improve "
             "load balancing but raise IPC overhead; larger chunks reduce "
             "overhead but risk worker idleness on small files. Default: 1000.",
    )
    args = parser.parse_args()

    if args.num_workers == 0:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = args.num_workers

    if num_workers == 1:
        converted, skipped, errors = _convert_sequential(args.input, args.output)
        mode = "sequential"
    else:
        converted, skipped, errors = _convert_parallel(
            args.input, args.output, num_workers, args.chunk_size
        )
        mode = f"parallel ({num_workers} workers, chunk_size={args.chunk_size})"

    print(f"Converted: {converted}  [{mode}]")
    if skipped:
        print(f"Skipped: {skipped}")
        for err in errors:
            print(err, file=sys.stderr)

    if skipped > converted * 0.01:
        print(
            f"WARNING: {skipped}/{converted + skipped} samples skipped "
            f"({skipped / (converted + skipped) * 100:.1f}%)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
