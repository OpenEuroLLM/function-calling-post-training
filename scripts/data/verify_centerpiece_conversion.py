"""Rigorous verifier for OpenAI → Dolci-format JSONL conversion.

Independently re-encodes the conversion specification (see
`scripts/data/convert_openai_to_dolci_format.py`) and verifies that every Dolci
output sample is consistent with its OAI input. The verifier deliberately does
NOT import `convert_sample` or any logic from the converter — using the same
function to verify itself would be tautological (it would catch hand edits and
non-determinism, but not bugs in the converter logic itself).

Instead, the spec is restated here and applied to every sample, exhaustively,
in parallel across all CPU cores.

Each (OAI, Dolci) pair is checked against the following invariants. A failure
of any check is recorded with the sample_id, aggregated across the file, and
the first 5 example sample_ids per failing check are surfaced in the report.

  A. Dolci top-level shape
     A1  exactly two top-level keys: {messages, id}
     A2  id is a string

  B. ID parity
     B1  per-line: str(OAI.sample_id) == Dolci.id

  C. Message-list shape
     C1  messages is a list
     C2  messages is non-empty
     C3  every entry is a dict
     C4  every message has exactly {role, content, function_calls, functions}
     C5  every role ∈ {system, user, assistant, environment}
     C6  the first message has role=system
     C7  no message after the first has role=system

  D. System message (msgs[0]) content
     D1  content equals build_expected_system_content(OAI's first system content):
           empty source             →  canonical SYSTEM_CONTENT
           contains <functions> or <function_calls> in source  →  source as-is
           else                     →  source + "\\n\\n" + SYSTEM_INSTRUCTIONS
     D2  functions is None ↔ OAI had no tools
     D3  functions (when present) parses as a JSON list
     D4  functions list length == OAI.tools length
     D5  each function dict == unwrap_tool(OAI.tools[i])
           where unwrap_tool(t) = t["function"] if t is {"type": "function", "function": ...},
                                  else t
     D6  function_calls is None
     D7  content contains "<functions>" or "<function_calls>" (the FC instruction markers)

  E. Role-sequence preservation
     Build the expected non-system role sequence from OAI by:
       - filtering messages to roles in {user, assistant, tool}
       - replacing each maximal run of tool with a single "environment"
     Then assert that Dolci.messages[1:] roles match this expected sequence
     exactly (E1). If E1 fails, downstream positional checks are skipped for
     that sample (E2 catches the case where the Dolci-side length still
     diverges after role check passed, which would indicate a verifier bug).

  F. User messages
     F1  Dolci.content == (OAI.content or "")       — None → ""
     F2  function_calls is None
     F3  functions is None

  G. Assistant messages
     With OAI.tool_calls present (non-empty list):
       G1  Dolci.content == OAI.content if truthy else None
       G2  Dolci.function_calls is a non-empty string
       G3  function_calls has exactly len(tool_calls) newline-separated lines
       G4  line k starts with "{tool_calls[k].function.name}("
       G5  line k ends with ")"
       G6  every key in tool_calls[k].function.arguments appears as "{key}=" in line k
     Without OAI.tool_calls (or empty list):
       G7  Dolci.content == OAI.content if truthy else ""
       G8  Dolci.function_calls is None
     Always:
       G9  Dolci.functions is None

  H. Environment messages
     For each maximal run of consecutive OAI tool messages, the Dolci environment
     message at the corresponding position must have:
       H1  content == "\\n".join(stringified contents of each tool in the run),
           where stringified = the string itself if str, json.dumps(...) if a
           non-str non-None value, "" if None or missing
       H2  function_calls is None
       H3  functions is None

  I. Composition (across files, post per-sample)
     I1  centerpiece_random.ids ⊊ centerpiece_raw.ids
     I2  centerpiece_turndrop.ids ⊊ centerpiece_raw.ids
     I3  centerpiece_AMS.ids == union of (*_ams component file ids)
     I4  centerpiece_turndrop.ids == union of (*_turndrop component file ids)

The verifier exits 0 only when every check on every sample of every selected
file passes AND the composition checks pass.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OAI_DIR = Path("/mnt/nfs/ytahtah/fcanalysis/phase2_data")
DOLCI_DIR = Path("/mnt/nfs/ytahtah/phase2_dolci_format")

CENTERPIECE_FILES = [
    "centerpiece_raw",
    "centerpiece_AMS",
    "centerpiece_random",
    "centerpiece_turndrop",
    "centerpiece_AMS_v2",
    "centerpiece_turndrop_v2",
    "centerpiece_random_v2",
]

AMS_SOURCES = [
    "dolci.jsonl",
    "graphsyn.jsonl",
    "v1_dedup_ams.jsonl",
    "v2_interactive_agent_ams.jsonl",
    "txt360_ams.jsonl",
]

TURNDROP_SOURCES = [
    "dolci.jsonl",
    "graphsyn.jsonl",
    "v1_dedup_turndrop.jsonl",
    "v2_interactive_agent_turndrop.jsonl",
    "txt360_turndrop.jsonl",
]

AMS_V2_SOURCES = [
    "dolci.jsonl",
    "graphsyn.jsonl",
    "v1_dedup_ams_v2.jsonl",
    "v2_interactive_agent_ams_v2.jsonl",
    "txt360_ams_v2.jsonl",
]

TURNDROP_V2_SOURCES = [
    "dolci.jsonl",
    "graphsyn.jsonl",
    "v1_dedup_turndrop_v2.jsonl",
    "v2_interactive_agent_turndrop_v2.jsonl",
    "txt360_turndrop_v2.jsonl",
]

# Spec constants — independently restated, NOT imported from the converter.
# If these texts drift from the converter, the D1 / D7 checks will surface it.
SYSTEM_PERSONA = "You are a helpful function-calling AI assistant."
SYSTEM_INSTRUCTIONS = (
    "You are provided with function "
    "signatures within <functions></functions> XML tags. You may call one or more functions "
    "to assist with the user query. Output any function calls within "
    "<function_calls></function_calls> XML tags. Don't make assumptions about what values "
    "to plug into functions."
)
SYSTEM_CONTENT_CANONICAL = f"{SYSTEM_PERSONA} {SYSTEM_INSTRUCTIONS}"

REQUIRED_DOLCI_TOP_KEYS = frozenset({"messages", "id"})
REQUIRED_DOLCI_MESSAGE_KEYS = frozenset({"role", "content", "function_calls", "functions"})
VALID_DOLCI_ROLES = frozenset({"system", "user", "assistant", "environment"})
OAI_REAL_ROLES = frozenset({"user", "assistant", "tool"})

EXAMPLES_PER_CHECK = 5


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    failure_counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    n_samples: int = 0

    def fail(self, name: str, sid: str, detail: str = "") -> None:
        self.failure_counts[name] = self.failure_counts.get(name, 0) + 1
        bucket = self.examples.setdefault(name, [])
        if len(bucket) < EXAMPLES_PER_CHECK:
            bucket.append(f"{sid}: {detail}" if detail else sid)

    def merge(self, other: "CheckResult") -> None:
        for k, v in other.failure_counts.items():
            self.failure_counts[k] = self.failure_counts.get(k, 0) + v
        for k, exs in other.examples.items():
            bucket = self.examples.setdefault(k, [])
            remaining = EXAMPLES_PER_CHECK - len(bucket)
            if remaining > 0:
                bucket.extend(exs[:remaining])
        self.n_samples += other.n_samples


# ---------------------------------------------------------------------------
# Spec helpers (re-stated, independent of the converter implementation)
# ---------------------------------------------------------------------------


def build_expected_system_content(source: str) -> str:
    """Spec rule for the Dolci system message body."""
    stripped = (source or "").strip()
    if not stripped:
        return SYSTEM_CONTENT_CANONICAL
    if "<functions>" in stripped or "<function_calls>" in stripped:
        return stripped
    return f"{stripped}\n\n{SYSTEM_INSTRUCTIONS}"


def unwrap_tool(t: object) -> object:
    """Spec rule for OAI tool extraction: peel the {type, function} wrapper if present."""
    if isinstance(t, dict) and "function" in t:
        return t["function"]
    return t


def stringify_tool_content(c: object) -> str:
    """Spec rule for how a single OAI tool message's content is stringified
    before being merged into a Dolci environment.content."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return json.dumps(c)


# ---------------------------------------------------------------------------
# Per-sample checks
# ---------------------------------------------------------------------------


def check_pair(oai: dict, dolci: dict, sid: str, r: CheckResult) -> None:
    """Run every check on one (OAI, Dolci) pair, recording failures into r."""

    # ===== A. Top-level shape =====
    dolci_top = set(dolci.keys())
    if dolci_top != REQUIRED_DOLCI_TOP_KEYS:
        r.fail("A1_top_keys", sid, f"got {sorted(dolci_top)}")
    if not isinstance(dolci.get("id"), str):
        r.fail("A2_id_is_string", sid, type(dolci.get("id")).__name__)

    # ===== B. ID parity =====
    oai_sid = oai.get("sample_id")
    if str(oai_sid) != dolci.get("id"):
        r.fail("B1_id_parity", sid, f"OAI={oai_sid!r} Dolci={dolci.get('id')!r}")

    # ===== C. Message-list shape =====
    msgs = dolci.get("messages")
    if not isinstance(msgs, list):
        r.fail("C1_messages_is_list", sid, type(msgs).__name__)
        return
    if len(msgs) == 0:
        r.fail("C2_messages_nonempty", sid)
        return

    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            r.fail("C3_message_is_dict", sid, f"msg[{i}]: {type(m).__name__}")
            return
        if set(m.keys()) != REQUIRED_DOLCI_MESSAGE_KEYS:
            r.fail("C4_message_keys", sid, f"msg[{i}] keys={sorted(m.keys())}")
        if m.get("role") not in VALID_DOLCI_ROLES:
            r.fail("C5_message_role", sid, f"msg[{i}] role={m.get('role')!r}")

    if msgs[0].get("role") != "system":
        r.fail("C6_first_msg_system", sid, f"got role={msgs[0].get('role')!r}")
        return

    for i, m in enumerate(msgs[1:], start=1):
        if m.get("role") == "system":
            r.fail("C7_only_one_system", sid, f"msg[{i}] is also system")

    # ===== D. System message =====
    sys_msg = msgs[0]

    # D1: content matches the spec rule applied to OAI's first system content
    oai_sys_content = ""
    for m in oai.get("messages", []) or []:
        if isinstance(m, dict) and m.get("role") == "system":
            oai_sys_content = m.get("content") or ""
            break
    expected_sys = build_expected_system_content(oai_sys_content)
    if sys_msg.get("content") != expected_sys:
        r.fail("D1_system_content", sid)

    # D2-D5: functions field
    oai_tools = oai.get("tools") or []
    dolci_functions = sys_msg.get("functions")
    if oai_tools:
        if dolci_functions is None:
            r.fail("D2_system_functions_present", sid, "OAI had tools, Dolci system.functions is None")
        else:
            try:
                fns = json.loads(dolci_functions)
            except (json.JSONDecodeError, TypeError) as e:
                r.fail("D3_system_functions_json", sid, f"{type(e).__name__}: {e}")
                fns = None
            if fns is not None:
                if not isinstance(fns, list):
                    r.fail("D3_system_functions_json", sid, f"not a list: {type(fns).__name__}")
                elif len(fns) != len(oai_tools):
                    r.fail("D4_system_functions_count", sid, f"OAI={len(oai_tools)} Dolci={len(fns)}")
                else:
                    for i, (oai_t, dolci_fn) in enumerate(zip(oai_tools, fns)):
                        if unwrap_tool(oai_t) != dolci_fn:
                            r.fail("D5_function_unwrap", sid, f"tool[{i}] mismatch")
    else:
        if dolci_functions is not None:
            r.fail("D2_system_functions_present", sid, "OAI had no tools, Dolci system.functions is set")

    # D6: function_calls is always None on system
    if sys_msg.get("function_calls") is not None:
        r.fail("D6_system_function_calls", sid)

    # D7: sanity — FC instructions are present somewhere in content
    sc = sys_msg.get("content") or ""
    if "<functions>" not in sc and "<function_calls>" not in sc:
        r.fail("D7_system_has_fc_markers", sid)

    # ===== E. Role-sequence preservation =====
    oai_real = [m for m in (oai.get("messages") or []) if isinstance(m, dict) and m.get("role") in OAI_REAL_ROLES]

    expected_roles: list[str] = []
    last_was_tool = False
    for m in oai_real:
        role = m["role"]
        if role == "tool":
            if not last_was_tool:
                expected_roles.append("environment")
            last_was_tool = True
        else:
            expected_roles.append(role)
            last_was_tool = False

    actual_roles = [m.get("role") for m in msgs[1:]]
    if expected_roles != actual_roles:
        r.fail(
            "E1_role_sequence",
            sid,
            f"expected[:8]={expected_roles[:8]} actual[:8]={actual_roles[:8]} "
            f"(lens exp={len(expected_roles)} act={len(actual_roles)})",
        )
        return  # downstream positional checks would be meaningless

    # ===== F/G/H. Per-message content =====
    dolci_non_sys = msgs[1:]
    j = 0  # cursor into Dolci non-system messages
    i = 0  # cursor into oai_real
    while i < len(oai_real):
        oai_m = oai_real[i]
        role = oai_m["role"]

        if role == "tool":
            # Collect this maximal run of consecutive tools.
            run_start = i
            while i < len(oai_real) and oai_real[i]["role"] == "tool":
                i += 1
            group = oai_real[run_start:i]

            dolci_m = dolci_non_sys[j]
            # H1: merged content
            parts = [stringify_tool_content(tm.get("content")) for tm in group]
            expected_content = "\n".join(parts)
            if dolci_m.get("content") != expected_content:
                r.fail("H1_env_content", sid, f"at j={j} (run_len={len(group)})")
            if dolci_m.get("function_calls") is not None:
                r.fail("H2_env_function_calls_none", sid, f"at j={j}")
            if dolci_m.get("functions") is not None:
                r.fail("H3_env_functions_none", sid, f"at j={j}")
            j += 1
            continue

        # Single user or assistant message
        dolci_m = dolci_non_sys[j]

        if role == "user":
            expected = oai_m.get("content") or ""
            if dolci_m.get("content") != expected:
                r.fail("F1_user_content", sid, f"at j={j}")
            if dolci_m.get("function_calls") is not None:
                r.fail("F2_user_function_calls_none", sid, f"at j={j}")
            if dolci_m.get("functions") is not None:
                r.fail("F3_user_functions_none", sid, f"at j={j}")

        else:  # assistant
            tool_calls = oai_m.get("tool_calls")
            content = oai_m.get("content")

            if tool_calls:
                expected_content = content if content else None
                if dolci_m.get("content") != expected_content:
                    r.fail("G1_asst_content_with_calls", sid, f"at j={j}")

                fc = dolci_m.get("function_calls")
                if fc is None or not isinstance(fc, str) or fc == "":
                    r.fail("G2_function_calls_present", sid, f"at j={j}")
                else:
                    fc_lines = fc.split("\n")
                    if len(fc_lines) != len(tool_calls):
                        r.fail(
                            "G3_function_calls_count",
                            sid,
                            f"at j={j}: fc_lines={len(fc_lines)} tool_calls={len(tool_calls)}",
                        )
                    for k, call in enumerate(tool_calls):
                        if k >= len(fc_lines):
                            break
                        func = call.get("function") if isinstance(call, dict) else None
                        if not isinstance(func, dict):
                            continue
                        name = func.get("name")
                        args = func.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = None
                        fc_line = fc_lines[k]
                        if isinstance(name, str) and name:
                            if not fc_line.startswith(f"{name}("):
                                r.fail("G4_function_name_preserved", sid, f"at j={j}, k={k}: name={name!r}")
                            if not fc_line.endswith(")"):
                                r.fail("G5_function_call_closed", sid, f"at j={j}, k={k}")
                        if isinstance(args, dict):
                            for arg_key in args.keys():
                                if f"{arg_key}=" not in fc_line:
                                    r.fail("G6_arg_key_preserved", sid, f"at j={j}, k={k}: arg={arg_key!r}")
            else:
                expected_content = content if content else ""
                if dolci_m.get("content") != expected_content:
                    r.fail("G7_asst_content_no_calls", sid, f"at j={j}")
                if dolci_m.get("function_calls") is not None:
                    r.fail("G8_no_function_calls", sid, f"at j={j}")

            if dolci_m.get("functions") is not None:
                r.fail("G9_asst_functions_none", sid, f"at j={j}")

        j += 1
        i += 1

    # Sanity: we consumed all Dolci non-system messages
    if j != len(dolci_non_sys):
        r.fail("E2_dolci_msg_count", sid, f"j={j} vs len={len(dolci_non_sys)}")


# ---------------------------------------------------------------------------
# Streaming + multiprocessing
# ---------------------------------------------------------------------------


def _process_chunk(pairs: list[tuple[str, str]]) -> CheckResult:
    """Worker: validate a chunk of (oai_line, dolci_line) raw strings."""
    result = CheckResult()
    for oai_line, dolci_line in pairs:
        result.n_samples += 1
        sid = "?"
        try:
            oai = json.loads(oai_line)
        except json.JSONDecodeError as e:
            result.fail("PARSE_OAI", sid, str(e))
            continue
        try:
            dolci = json.loads(dolci_line)
        except json.JSONDecodeError as e:
            result.fail("PARSE_DOLCI", sid, str(e))
            continue
        sid = str(dolci.get("id", oai.get("sample_id", "?")))
        try:
            check_pair(oai, dolci, sid, result)
        except Exception as e:
            result.fail("EXCEPTION", sid, f"{type(e).__name__}: {e}")
    return result


def _chunked_pairs(
    oai_path: Path, dolci_path: Path, chunk_size: int
) -> Iterator[list[tuple[str, str]]]:
    """Yield chunks of zipped (oai_line, dolci_line) raw string pairs.

    Both files are streamed; we assume line-order correspondence (the converter
    uses Pool.imap which preserves input order, and we verified line-count
    parity up front — so line N in OAI matches line N in Dolci).
    """
    with open(oai_path) as foai, open(dolci_path) as fdolci:
        chunk: list[tuple[str, str]] = []
        for oai_line, dolci_line in zip(foai, fdolci):
            chunk.append((oai_line, dolci_line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def verify_file(name: str, num_workers: int, chunk_size: int, limit: int | None) -> CheckResult:
    """Verify all (or up to `limit`) samples of one centerpiece file."""
    oai_path = OAI_DIR / f"{name}.jsonl"
    dolci_path = DOLCI_DIR / f"{name}.jsonl"

    result = CheckResult()

    # Pre-check: line-count parity. If broken, we still continue with min().
    n_oai = count_lines(oai_path)
    n_dolci = count_lines(dolci_path)
    if n_oai != n_dolci:
        result.fail("LINE_COUNT_PARITY", name, f"OAI={n_oai:,} Dolci={n_dolci:,}")

    # Per-pair checks (parallel)
    chunks_iter = _chunked_pairs(oai_path, dolci_path, chunk_size)
    if limit is not None:
        # Truncate the stream to roughly `limit` samples (rounded up to chunk).
        def _capped(it):
            seen = 0
            for c in it:
                if seen >= limit:
                    return
                yield c
                seen += len(c)

        chunks_iter = _capped(chunks_iter)

    if num_workers == 1:
        for chunk in chunks_iter:
            result.merge(_process_chunk(chunk))
    else:
        with multiprocessing.Pool(num_workers) as pool:
            for partial in pool.imap_unordered(_process_chunk, chunks_iter, chunksize=1):
                result.merge(partial)

    return result


# ---------------------------------------------------------------------------
# Composition checks (sample-id-set level)
# ---------------------------------------------------------------------------


def file_id_set(path: Path, key: str) -> set[str]:
    ids: set[str] = set()
    with open(path, "rb") as f:
        for line in f:
            obj = json.loads(line)
            ids.add(str(obj[key]))
    return ids


def composition_checks(selected: list[str]) -> bool:
    """Check that filtered centerpieces are correctly composed from their sources.

    Uses the Dolci files' ids as the centerpiece truth (we already verified ID
    parity per-sample above), and the OAI source component files for the union
    side. Stringifies IDs on both sides for an apples-to-apples comparison.
    """
    print("\n" + "=" * 72)
    print("Composition checks")
    print("=" * 72)

    needed = set(selected)
    cp_ids: dict[str, set[str]] = {}
    for name in CENTERPIECE_FILES:
        if name not in needed:
            continue
        path = DOLCI_DIR / f"{name}.jsonl"
        if not path.exists():
            print(f"  ⚠  {name}.jsonl missing; skipping")
            continue
        ids = file_id_set(path, "id")
        cp_ids[name] = ids
        print(f"  {name}: {len(ids):,} unique ids")

    all_ok = True

    def _strict_subset(small: str, big: str, label: str) -> None:
        nonlocal all_ok
        if small not in cp_ids or big not in cp_ids:
            return
        diff = cp_ids[small] - cp_ids[big]
        ok = (len(diff) == 0 and len(cp_ids[small]) < len(cp_ids[big]))
        all_ok &= ok
        marker = "✓" if ok else "✗"
        detail = ""
        if not ok:
            if len(diff) > 0:
                detail = f" — {len(diff):,} ids in {small} not in {big}"
            elif len(cp_ids[small]) >= len(cp_ids[big]):
                detail = f" — {small} ({len(cp_ids[small]):,}) not strictly smaller than {big} ({len(cp_ids[big]):,})"
        print(f"  {marker} {label}{detail}")

    def _equals_union(cp: str, sources: list[str], label: str) -> None:
        nonlocal all_ok
        if cp not in cp_ids:
            return
        union: set[str] = set()
        for src in sources:
            src_path = OAI_DIR / src
            if not src_path.exists():
                print(f"  ⚠  source {src} missing; skipping composition check for {cp}")
                return
            sids = file_id_set(src_path, "sample_id")
            union |= sids
            print(f"    + {src}: {len(sids):,} ids")
        only_cp = cp_ids[cp] - union
        only_src = union - cp_ids[cp]
        ok = (len(only_cp) == 0 and len(only_src) == 0)
        all_ok &= ok
        marker = "✓" if ok else "✗"
        detail = ""
        if not ok:
            detail = f" — only_in_centerpiece={len(only_cp)}, only_in_sources={len(only_src)}"
        print(f"  {marker} {label}{detail}")

    print("\n[I1] centerpiece_random ⊊ centerpiece_raw")
    _strict_subset("centerpiece_random", "centerpiece_raw", "random ⊊ raw")

    print("\n[I2] centerpiece_turndrop ⊊ centerpiece_raw")
    _strict_subset("centerpiece_turndrop", "centerpiece_raw", "turndrop ⊊ raw")

    print("\n[I3] centerpiece_AMS == union(AMS_SOURCES)")
    _equals_union("centerpiece_AMS", AMS_SOURCES, "centerpiece_AMS == union(AMS_SOURCES)")

    print("\n[I4] centerpiece_turndrop == union(TURNDROP_SOURCES)")
    _equals_union("centerpiece_turndrop", TURNDROP_SOURCES, "centerpiece_turndrop == union(TURNDROP_SOURCES)")

    print("\n[I5] centerpiece_random_v2 ⊊ centerpiece_raw")
    _strict_subset("centerpiece_random_v2", "centerpiece_raw", "random_v2 ⊊ raw")

    print("\n[I6] centerpiece_turndrop_v2 ⊊ centerpiece_raw")
    _strict_subset("centerpiece_turndrop_v2", "centerpiece_raw", "turndrop_v2 ⊊ raw")

    print("\n[I7] centerpiece_AMS_v2 == union(AMS_V2_SOURCES)")
    _equals_union("centerpiece_AMS_v2", AMS_V2_SOURCES, "centerpiece_AMS_v2 == union(AMS_V2_SOURCES)")

    print("\n[I8] centerpiece_turndrop_v2 == union(TURNDROP_V2_SOURCES)")
    _equals_union("centerpiece_turndrop_v2", TURNDROP_V2_SOURCES, "centerpiece_turndrop_v2 == union(TURNDROP_V2_SOURCES)")

    return all_ok


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_file_report(name: str, r: CheckResult, elapsed_s: float) -> bool:
    """Return True iff this file is clean."""
    print(f"\n--- {name} ({r.n_samples:,} samples in {elapsed_s:.1f}s) ---")
    if not r.failure_counts:
        print("  ✓ All per-sample checks passed")
        return True
    for check_name in sorted(r.failure_counts.keys()):
        count = r.failure_counts[check_name]
        rate = 100.0 * count / r.n_samples if r.n_samples else 0.0
        print(f"  ✗ {check_name}: {count:,} ({rate:.3f}%)")
        for ex in r.examples.get(check_name, [])[:3]:
            print(f"      e.g.: {ex}")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for the per-sample pass. 0 = os.cpu_count(). "
        "1 = single-process (useful for profiling / deterministic test runs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Pairs per worker chunk. Larger = less IPC overhead; smaller = better load balancing.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated subset of centerpiece names to verify. "
            f"Default: all four — {','.join(CENTERPIECE_FILES)}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap per-file sample count (rounded up to chunk size). Use for smoke tests.",
    )
    parser.add_argument(
        "--skip-per-sample",
        action="store_true",
        help="Skip per-sample checks (just run composition).",
    )
    parser.add_argument(
        "--skip-composition",
        action="store_true",
        help="Skip composition checks (just run per-sample).",
    )
    args = parser.parse_args()

    if args.workers == 0:
        args.workers = os.cpu_count() or 1

    if args.only:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
        for s in selected:
            if s not in CENTERPIECE_FILES:
                raise SystemExit(f"Unknown centerpiece: {s} (valid: {CENTERPIECE_FILES})")
    else:
        selected = list(CENTERPIECE_FILES)

    print("=" * 72)
    print("Rigorous OAI → Dolci conversion verifier")
    print(f"  Workers   : {args.workers}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Files     : {selected}")
    if args.limit:
        print(f"  Limit     : {args.limit:,} samples per file")
    print("=" * 72)

    all_ok = True

    if not args.skip_per_sample:
        for name in selected:
            print(f"\nVerifying {name}...")
            t0 = time.monotonic()
            r = verify_file(name, args.workers, args.chunk_size, args.limit)
            elapsed = time.monotonic() - t0
            all_ok &= print_file_report(name, r, elapsed)

    if not args.skip_composition:
        all_ok &= composition_checks(selected)

    print("\n" + "=" * 72)
    if all_ok:
        print("ALL CHECKS PASSED ✓")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED ✗ — see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
