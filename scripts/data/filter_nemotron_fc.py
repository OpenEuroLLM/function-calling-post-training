"""Filter nemotron_fc.jsonl to remove entries that cause tool-refusal behavior.

Removes two categories of problematic entries identified by BFCL analysis
(see /mnt/nfs/ytahtah/bfcl/analysis/IRRELEVANCE_VS_RELEVANCE_INVESTIGATION.md):

1. No-tools entries (~3,637): system message has empty tool list ('[]').
   These teach the model to fabricate answers when no tools exist.

2. Refuse-then-call (RTC) entries (~32,677): first assistant turn has no
   function_calls, but a later assistant turn does. These teach the model
   to refuse on the first turn and only call after user pushback —
   catastrophic for single-turn evaluation. All RTC entries are removed
   regardless of whether the initial refusal is "justified" (e.g., asking
   for missing parameters), because the sheer volume (10.3% of dataset)
   teaches a strong "don't call first" signal. Dolci FC already provides
   parameter-gathering training signal at much lower prevalence (0.47%).

Keeps:
- Normal FC entries (~247K): assistant calls tools on the first opportunity.
- Never-call entries (~33K): tools are provided but the assistant correctly
  identifies that they don't match the query (genuine irrelevance detection)
  or that required parameters are missing.

Usage:
    python scripts/data/filter_nemotron_fc.py \
        --input_path nemotron_fc.jsonl \
        --output_path filtered_nemotron_fc.jsonl
"""

import argparse
import json


def has_function_calls(msg):
    """Check if a message has non-empty function_calls."""
    fc = msg.get("function_calls")
    return bool(fc and fc.strip())


def classify_entry(row):
    """Classify a nemotron_fc entry into one of: no_tools, rtc, keep.

    Returns:
        str: "no_tools", "rtc", or "keep"
    """
    msgs = row["messages"]

    # --- Filter 1: No-tools entries ---
    # System message should be first; check its functions field.
    sys_msg = msgs[0] if msgs and msgs[0]["role"] == "system" else None
    if sys_msg:
        funcs = sys_msg.get("functions")
        if not funcs or funcs in ("null", "[]", "None", ""):
            return "no_tools"

    # --- Filter 2: Refuse-then-call (RTC) ---
    # First assistant turn has no FC, but a later assistant turn does.
    asst_turns = [m for m in msgs if m["role"] == "assistant"]

    if len(asst_turns) >= 2:
        first_has_fc = has_function_calls(asst_turns[0])
        if not first_has_fc:
            later_has_fc = any(has_function_calls(m) for m in asst_turns[1:])
            if later_has_fc:
                return "rtc"

    return "keep"


def main():
    parser = argparse.ArgumentParser(
        description="Filter nemotron_fc.jsonl to remove tool-refusal-inducing entries."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Input nemotron_fc.jsonl path")
    parser.add_argument("--output_path", type=str, required=True, help="Output filtered JSONL path")
    parser.add_argument(
        "--removed_path",
        type=str,
        default=None,
        help="Optional: save removed entries to this JSONL path for inspection",
    )
    args = parser.parse_args()

    print(f"Loading {args.input_path}...")
    entries = []
    with open(args.input_path) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  Loaded {len(entries)} entries")

    # Classify
    kept = []
    removed_no_tools = []
    removed_rtc = []

    for row in entries:
        label = classify_entry(row)
        if label == "no_tools":
            removed_no_tools.append(row)
        elif label == "rtc":
            removed_rtc.append(row)
        else:
            kept.append(row)

    total_removed = len(removed_no_tools) + len(removed_rtc)

    # Print stats
    print()
    print("=== Filtering Results ===")
    print(f"Total input:          {len(entries)}")
    print(f"Removed (no-tools):   {len(removed_no_tools)}")
    print(f"Removed (RTC):        {len(removed_rtc)}")
    print(f"Total removed:        {total_removed}")
    print(f"Kept:                 {len(kept)}")
    print()

    # Breakdown of kept entries
    kept_with_fc = sum(1 for row in kept if any(has_function_calls(m) for m in row["messages"]))
    kept_never_call = len(kept) - kept_with_fc
    print(f"Kept breakdown:")
    print(f"  Normal FC (has tool calls):     {kept_with_fc}")
    print(f"  Never-call (genuine no-call):   {kept_never_call}")
    print()

    assert len(kept) + total_removed == len(entries), (
        f"Count mismatch: {len(kept)} + {total_removed} != {len(entries)}"
    )

    # Save kept entries
    print(f"Saving {len(kept)} kept entries to {args.output_path}...")
    with open(args.output_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print("  Done.")

    # Optionally save removed entries
    if args.removed_path:
        print(f"Saving {total_removed} removed entries to {args.removed_path}...")
        with open(args.removed_path, "w") as f:
            # NOTE: mutates original dicts in-place. Fine for this script,
            # but use {**row, "_removal_reason": ...} if reusing as a library.
            for row in removed_no_tools:
                row["_removal_reason"] = "no_tools"
                f.write(json.dumps(row) + "\n")
            for row in removed_rtc:
                row["_removal_reason"] = "rtc"
                f.write(json.dumps(row) + "\n")
        print("  Done.")

    print()
    print("Summary:")
    print(f"  {args.input_path}: {len(entries)} entries")
    print(f"  {args.output_path}: {len(kept)} entries ({len(kept)/len(entries)*100:.1f}%)")
    if args.removed_path:
        print(f"  {args.removed_path}: {total_removed} entries ({total_removed/len(entries)*100:.1f}%)")


if __name__ == "__main__":
    main()
