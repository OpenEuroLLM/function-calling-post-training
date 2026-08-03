"""Build a step-matched A1' subsample of (Dolci + GraphSyn) for Phase 2.

A1' addresses the step-count confound that A1 inherits from being a forward
add-one against P1: A1 trains for ~12% more steps than P1 because adding
GraphSyn (23,427 samples) to Dolci (203,035) makes the mix 226,462 samples
in 1 epoch. A1 vs P1 cannot cleanly attribute its delta to "GraphSyn signal"
vs "extra training steps." A1' fixes this by sub-sampling the merged
(Dolci + GraphSyn) pool to exactly 203,035 samples — the same step count
as P1 — using a random shuffle with a fixed seed. The resulting mix
preserves the original proportion (~10.3% GraphSyn) in expectation.

A1' vs P1 is then volume-matched, step-matched, and recipe-identical;
the only varying axis is composition. A1 vs A1' decomposes A1's forward
delta into "GraphSyn-specific signal" vs "additional-step contribution."

Reproducibility: this script's `random.Random(seed).sample(range(N), k)` is
deterministic for a fixed seed. Default seed=42 matches the convention used
elsewhere in Phase 2 (Q2 random sampling, K-replication seeds 42/7/13).

Usage:
    .venv/bin/python scripts/data/build_a1prime_step_matched.py \\
        --dolci    /mnt/nfs/ytahtah/phase2_dolci_format/dolci.jsonl \\
        --graphsyn /mnt/nfs/ytahtah/phase2_dolci_format/graphsyn.jsonl \\
        --output   /mnt/nfs/ytahtah/phase2_dolci_format/a1prime_step_matched.jsonl
"""

import argparse
import random
import time
from pathlib import Path


DEFAULT_TARGET_SIZE = 203_035  # matches P1 (curated Dolci) sample count exactly
DEFAULT_SEED = 42


def build_a1prime(
    dolci_path: Path,
    graphsyn_path: Path,
    output_path: Path,
    target_size: int,
    seed: int,
) -> None:
    """Merge Dolci + GraphSyn JSONL line-pools, shuffle with seed, take first N."""
    print(f"Reading {dolci_path}...", flush=True)
    t0 = time.monotonic()
    with open(dolci_path, "rb") as f:
        dolci_lines = f.readlines()
    print(f"  {len(dolci_lines):,} samples ({time.monotonic() - t0:.1f}s)")

    print(f"Reading {graphsyn_path}...", flush=True)
    t0 = time.monotonic()
    with open(graphsyn_path, "rb") as f:
        graphsyn_lines = f.readlines()
    print(f"  {len(graphsyn_lines):,} samples ({time.monotonic() - t0:.1f}s)")

    all_lines = dolci_lines + graphsyn_lines
    n = len(all_lines)
    print(f"Combined pool: {n:,} samples (Dolci first, then GraphSyn)")

    if not 0 <= target_size <= n:
        raise ValueError(
            f"target_size ({target_size}) must be in [0, {n}] (combined pool has {n} samples)"
        )

    print(
        f"Sampling {target_size:,} samples with random.Random(seed={seed})...",
        flush=True,
    )
    t0 = time.monotonic()
    rng = random.Random(seed)
    selected_indices = rng.sample(range(n), target_size)
    print(f"  selection done ({time.monotonic() - t0:.1f}s)")

    expected_dolci = sum(1 for i in selected_indices if i < len(dolci_lines))
    expected_graphsyn = target_size - expected_dolci
    proportion_graphsyn = expected_graphsyn / target_size
    print(
        f"  expected composition: {expected_dolci:,} Dolci + "
        f"{expected_graphsyn:,} GraphSyn ({proportion_graphsyn:.2%} GraphSyn)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...", flush=True)
    t0 = time.monotonic()
    with open(output_path, "wb") as f:
        for i in selected_indices:
            f.write(all_lines[i])
    elapsed = time.monotonic() - t0
    print(f"  {target_size:,} samples written ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dolci",
        type=Path,
        required=True,
        help="Path to Dolci-format dolci.jsonl (203,035 samples expected)",
    )
    parser.add_argument(
        "--graphsyn",
        type=Path,
        required=True,
        help="Path to Dolci-format graphsyn.jsonl (23,427 samples expected)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for the step-matched merged subsample",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=(
            f"Number of samples to retain (default: {DEFAULT_TARGET_SIZE}, "
            "matches P1's curated Dolci count)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the shuffle (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if not args.dolci.exists():
        raise FileNotFoundError(f"Dolci file not found: {args.dolci}")
    if not args.graphsyn.exists():
        raise FileNotFoundError(f"GraphSyn file not found: {args.graphsyn}")

    build_a1prime(
        dolci_path=args.dolci,
        graphsyn_path=args.graphsyn,
        output_path=args.output,
        target_size=args.target_size,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
