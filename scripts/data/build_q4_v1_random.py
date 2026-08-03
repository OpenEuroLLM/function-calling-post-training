"""Build the v1_dedup random subsample for Phase 2 experiment Q4.

Q4 is the v1 analog of Q2 for TxT360: a random-removal control for the AMS
filtering test on v1.

  A2 = Dolci + v1_dedup (200,310)               total 403,345
  A2' = Dolci + v1_dedup_ams (182,892)          total 385,927  [AMS-targeted removal]
  Q4 = Dolci + v1_dedup_random (182,892)        total 385,927  [random removal at same volume]

A2' vs Q4 is volume-matched, step-matched, recipe-identical; only the
identity of the 17,418 removed v1 samples differs (AMS-targeted vs random).
This is the v1 analog of A3 vs Q2 for TxT360 and supports Claim 4
("AMS-targeted filtering > random") with multi-dataset evidence.

Reproducibility: this script's `random.Random(seed).sample(range(N), k)` is
deterministic for a fixed seed. Default seed=42 matches the convention used
elsewhere in Phase 2 (Q2 random sampling, K-replication seeds 42/7/13).

Composition expectation under random sampling: of the 200,310 v1_dedup
samples, 17,418 are AMS-flagged. A random subsample of 182,892 keeps in
expectation ~15,907 AMS samples (and ~166,985 non-AMS), vs A2' which keeps
0 AMS and 182,892 non-AMS. The AMS contrast between A2' and Q4 is therefore
~15.9K samples.

Usage:
    .venv/bin/python scripts/data/build_q4_v1_random.py \\
        --input /mnt/nfs/ytahtah/phase2_dolci_format/v1_dedup.jsonl \\
        --output /mnt/nfs/ytahtah/phase2_dolci_format/v1_dedup_random.jsonl

For K-replications later (paired with K-replicated A2'), pass --seeds:
    .venv/bin/python scripts/data/build_q4_v1_random.py \\
        --input ... --output-dir ... --seeds 42 7 13
"""

import argparse
import random
import time
from pathlib import Path


DEFAULT_TARGET_SIZE = 182_892  # matches v1_dedup_ams sample count exactly
DEFAULT_SEED = 42


def build_random_subsample(
    input_path: Path,
    output_path: Path,
    target_size: int,
    seed: int,
) -> None:
    """Read input JSONL, random-sample target_size lines with seed, write output."""
    print(f"Reading {input_path}...", flush=True)
    t0 = time.monotonic()
    with open(input_path, "rb") as f:
        lines = f.readlines()
    n = len(lines)
    print(f"  {n:,} samples ({time.monotonic() - t0:.1f}s)")

    if not 0 <= target_size <= n:
        raise ValueError(
            f"target_size ({target_size}) must be in [0, {n}] (input has {n} samples)"
        )

    print(
        f"Sampling {target_size:,} samples with random.Random(seed={seed})...",
        flush=True,
    )
    t0 = time.monotonic()
    rng = random.Random(seed)
    indices = rng.sample(range(n), target_size)
    print(f"  selection done ({time.monotonic() - t0:.1f}s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...", flush=True)
    t0 = time.monotonic()
    with open(output_path, "wb") as f:
        for i in indices:
            f.write(lines[i])
    elapsed = time.monotonic() - t0
    print(f"  {target_size:,} samples written ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Dolci-format v1_dedup.jsonl (200,310 samples expected)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path (single-seed mode). Required unless --seeds and "
            "--output-dir are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (multi-seed mode). When --seeds has multiple "
            "values, files are named v1_dedup_random_seed{seed}.jsonl in this dir."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[DEFAULT_SEED],
        help=f"One or more replication seeds (default: [{DEFAULT_SEED}])",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=(
            f"Number of samples to retain (default: {DEFAULT_TARGET_SIZE}, "
            "matches v1_dedup_ams)"
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if len(args.seeds) == 1 and args.output is not None:
        # Single-seed mode with explicit output path
        build_random_subsample(
            input_path=args.input,
            output_path=args.output,
            target_size=args.target_size,
            seed=args.seeds[0],
        )
    elif args.output_dir is not None:
        # Multi-seed mode (or single-seed with --output-dir)
        for seed in args.seeds:
            output_path = args.output_dir / f"v1_dedup_random_seed{seed}.jsonl"
            build_random_subsample(
                input_path=args.input,
                output_path=output_path,
                target_size=args.target_size,
                seed=seed,
            )
    else:
        parser.error(
            "Provide either --output (single-seed mode) or --output-dir "
            "(multi-seed mode with --seeds)."
        )

    print("Done.")


if __name__ == "__main__":
    main()
