"""Token-length distribution analyzer for Dolci-format JSONL files.

Uses the production tokenizer's built-in chat template directly
(`tokenizer.apply_chat_template`) so the counts reported here are exactly
what the training pipeline will see. No hand-coded template — we accept the
runtime cost in exchange for zero risk of renderer drift.

Pipeline per worker:
  1. Read a chunk of raw JSONL bytes (sent by the main process)
  2. Parse with msgspec (~3× faster than stdlib json)
  3. Apply the OLMo chat template per sample (`tokenize=False`) — this is
     the slow step; Jinja is per-sample Python
  4. Batch-tokenize the rendered strings with the HF fast (Rust) backend
  5. Return integer token-counts

Throughput on this 144-core box: ~3K samples/s aggregate at 64 workers, so a
single centerpiece (~890K samples) finishes in ~5 minutes. Running all four
takes ~20 minutes.

Usage:
    # All four centerpieces (default):
    python scripts/data/analyze_token_distribution.py

    # Just one file:
    python scripts/data/analyze_token_distribution.py \\
        --files /mnt/nfs/ytahtah/phase2_dolci_format/centerpiece_raw.jsonl

    # Smoke test:
    python scripts/data/analyze_token_distribution.py --limit 5000

    # Persist aggregated stats:
    python scripts/data/analyze_token_distribution.py \\
        --save-json notes/token_distribution.json

Recommended: run inside a tmux session so the process survives terminal
disconnects.

    tmux new -s tokens
    source .venv/bin/activate
    python scripts/data/analyze_token_distribution.py \\
        --files /mnt/nfs/ytahtah/phase2_dolci_format/centerpiece_raw.jsonl \\
        --workers 96 --chunk-size 256 \\
        --save-json notes/token_distribution_centerpiece_raw.json
    # Detach: Ctrl-b then d. Reattach: tmux attach -t tokens.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import msgspec
import numpy as np
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOLCI_DIR = Path("/mnt/nfs/ytahtah/phase2_dolci_format")
DEFAULT_FILES = [
    DOLCI_DIR / "centerpiece_raw.jsonl",
    DOLCI_DIR / "centerpiece_AMS.jsonl",
    DOLCI_DIR / "centerpiece_random.jsonl",
    DOLCI_DIR / "centerpiece_turndrop.jsonl",
]

TOKENIZER_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
MAX_SEQ_LEN = 32768  # Training sequence length boundary for "truncation" stat

HISTOGRAM_EDGES = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, float("inf")]

_DECODE = msgspec.json.Decoder().decode


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


_WORKER_TOK = None


def _init_worker() -> None:
    """Load the tokenizer (with its built-in chat template) once per worker."""
    global _WORKER_TOK
    _WORKER_TOK = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _process_chunk(raw_lines: list[bytes]) -> list[int]:
    """Parse JSONL bytes → render via chat template → batch-tokenize → return lengths."""
    tok = _WORKER_TOK
    rendered: list[str] = []
    for raw in raw_lines:
        sample = _DECODE(raw)
        rendered.append(
            tok.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
        )
    encoded = tok(rendered, add_special_tokens=False, return_attention_mask=False)
    return [len(ids) for ids in encoded["input_ids"]]


def _stream_chunks(path: Path, chunk_size: int, limit: int | None) -> Iterator[list[bytes]]:
    """Stream raw JSONL line bytes in fixed-size chunks."""
    chunk: list[bytes] = []
    n = 0
    with open(path, "rb") as f:
        for line in f:
            chunk.append(line)
            n += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if limit is not None and n >= limit:
                break
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Statistics + reporting
# ---------------------------------------------------------------------------


def _compute_stats(counts: np.ndarray, elapsed_s: float, path: Path) -> dict:
    n = int(counts.size)
    hist, _ = np.histogram(counts, bins=HISTOGRAM_EDGES)
    bucket_pairs: list[tuple[int, int | None, int]] = []
    for i in range(len(HISTOGRAM_EDGES) - 1):
        lo = int(HISTOGRAM_EDGES[i])
        hi_raw = HISTOGRAM_EDGES[i + 1]
        hi: int | None = int(hi_raw) if hi_raw != float("inf") else None
        bucket_pairs.append((lo, hi, int(hist[i])))

    return {
        "path": str(path),
        "n_samples": n,
        "elapsed_s": round(elapsed_s, 2),
        "samples_per_s": round(n / elapsed_s, 1) if elapsed_s > 0 else 0.0,
        "total_tokens": int(counts.sum()),
        "min": int(counts.min()),
        "mean": float(round(counts.mean(), 1)),
        "std": float(round(counts.std(), 1)),
        "p25": float(np.percentile(counts, 25)),
        "p50": float(np.percentile(counts, 50)),
        "p75": float(np.percentile(counts, 75)),
        "p90": float(np.percentile(counts, 90)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "p99_9": float(np.percentile(counts, 99.9)),
        "max": int(counts.max()),
        "n_over_max_seq": int(np.sum(counts > MAX_SEQ_LEN)),
        "n_under_512": int(np.sum(counts < 512)),
        "histogram": bucket_pairs,
    }


def print_report(stats: dict) -> None:
    name = Path(stats["path"]).stem
    n = stats["n_samples"]
    print(f"\n=== {name} ===")
    print(f"  Samples: {n:,}  |  {stats['elapsed_s']:.1f}s  ({stats['samples_per_s']:,.0f} samples/s)")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print()
    print(f"  Min:    {stats['min']:>8,}")
    print(f"  Mean:   {stats['mean']:>8,.0f}   ± {stats['std']:,.0f}")
    print(f"  Median: {stats['p50']:>8,.0f}")
    print(f"  P25:    {stats['p25']:>8,.0f}")
    print(f"  P75:    {stats['p75']:>8,.0f}")
    print(f"  P90:    {stats['p90']:>8,.0f}")
    print(f"  P95:    {stats['p95']:>8,.0f}")
    print(f"  P99:    {stats['p99']:>8,.0f}")
    print(f"  P99.9:  {stats['p99_9']:>8,.0f}")
    print(f"  Max:    {stats['max']:>8,}")
    print()
    if stats["n_over_max_seq"]:
        pct = 100.0 * stats["n_over_max_seq"] / n
        print(f"  Samples > {MAX_SEQ_LEN:,} (truncate / pack): {stats['n_over_max_seq']:,} ({pct:.3f}%)")
    else:
        print(f"  Samples > {MAX_SEQ_LEN:,}: 0 (no truncation needed)")
    pct = 100.0 * stats["n_under_512"] / n
    print(f"  Samples < 512 (very short / padding overhead): {stats['n_under_512']:,} ({pct:.2f}%)")
    print()
    print("  Length histogram:")
    for lo, hi, count in stats["histogram"]:
        if count == 0:
            continue
        pct = 100.0 * count / n
        label = f">{lo:>6,}" if hi is None else f"{lo:>6,}–{hi:>6,}"
        bar = "█" * int(round(pct))
        print(f"    {label}: {count:>8,} ({pct:5.2f}%) {bar}")


def analyze_file(path: Path, workers: int, chunk_size: int, limit: int | None) -> dict:
    counts: list[int] = []
    t0 = time.monotonic()

    if workers == 1:
        _init_worker()
        for chunk in _stream_chunks(path, chunk_size, limit):
            counts.extend(_process_chunk(chunk))
    else:
        with multiprocessing.Pool(workers, initializer=_init_worker) as pool:
            for partial in pool.imap_unordered(
                _process_chunk, _stream_chunks(path, chunk_size, limit), chunksize=1
            ):
                counts.extend(partial)

    elapsed = time.monotonic() - t0
    arr = np.asarray(counts, dtype=np.int64)
    return _compute_stats(arr, elapsed, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--files", nargs="+", type=Path, default=DEFAULT_FILES,
        help="Dolci JSONL files to analyze.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes. 0 = os.cpu_count(). 1 = single-process.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help="Raw lines per worker batch. Smaller = better load balance for "
        "files with high length variance; larger = less IPC overhead.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap per-file sample count (smoke test).",
    )
    parser.add_argument(
        "--save-json", type=Path, default=None,
        help="Write aggregated stats to this JSON path.",
    )
    args = parser.parse_args()

    if args.workers == 0:
        args.workers = os.cpu_count() or 1

    print("=" * 72)
    print("Token-length distribution analysis")
    print(f"  Tokenizer  : {TOKENIZER_NAME}  (built-in chat_template.jinja)")
    print(f"  Workers    : {args.workers}")
    print(f"  Chunk size : {args.chunk_size}")
    if args.limit:
        print(f"  Limit      : {args.limit:,} samples per file")
    print(f"  Files      : {[str(p) for p in args.files]}")
    print("=" * 72)

    existing = [p for p in args.files if p.exists()]
    missing = [p for p in args.files if not p.exists()]
    for p in missing:
        print(f"⚠ {p} not found, skipping", file=sys.stderr)

    if not existing:
        print("No input files exist — nothing to do.", file=sys.stderr)
        sys.exit(2)

    all_stats: list[dict] = []
    for path in existing:
        print(f"\nAnalyzing {path.name}...", flush=True)
        stats = analyze_file(path, args.workers, args.chunk_size, args.limit)
        print_report(stats)
        all_stats.append(stats)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(
                {
                    "tokenizer": TOKENIZER_NAME,
                    "max_seq_len": MAX_SEQ_LEN,
                    "histogram_edges": [
                        e if e != float("inf") else None for e in HISTOGRAM_EDGES
                    ],
                    "files": all_stats,
                },
                f, indent=2,
            )
        print(f"\nWrote aggregated stats to {args.save_json}")


if __name__ == "__main__":
    main()
