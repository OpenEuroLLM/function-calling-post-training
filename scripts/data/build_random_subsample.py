"""Random-subsample a JSONL file to a target line count with a fixed seed.

Generic Q4-pattern helper. Used to build random-removal controls that match
the volume of AMS-filtered (or other filter-targeted) variants. The
input/output JSONLs are typically in Dolci format already; this script
just shuffles indices and copies the selected lines.

Reproducibility: `random.Random(seed).sample(range(N), k)` is deterministic
for fixed seed. Default seed=42 matches the Phase 2 convention (Q2 random
sampling, Q4, K-replication seeds 42/7/13).

Usage:
    .venv/bin/python scripts/data/build_random_subsample.py \\
        --input /mnt/nfs/ytahtah/phase2_dolci_format/v2_interactive_agent.jsonl \\
        --output /mnt/nfs/ytahtah/phase2_dolci_format/v2_interactive_agent_random.jsonl \\
        --target-size 255481

    # multi-seed (for K-replication):
    .venv/bin/python scripts/data/build_random_subsample.py \\
        --input ... --output-dir ... --output-name-template "X_random_seed{seed}.jsonl" \\
        --seeds 42 7 13 --target-size 255481

History: replaces the dataset-specific build_q4_v1_random.py for the
Wave 0 random subsamples (v2_interactive_agent_random, nemotron_terminal_random)
and any future random-subsample needs.
"""

import argparse
import random
import time
from pathlib import Path


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
        help="Path to source JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path (single-seed mode). Required unless --seeds has "
            "multiple values, in which case use --output-dir + --output-name-template."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (multi-seed mode). Used with --output-name-template; "
            "the template's {seed} placeholder is substituted for each seed."
        ),
    )
    parser.add_argument(
        "--output-name-template",
        type=str,
        default=None,
        help=(
            'Filename template for multi-seed mode, e.g. "X_random_seed{seed}.jsonl". '
            "Must contain '{seed}'."
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
        required=True,
        help="Number of samples to retain.",
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
    elif args.output_dir is not None and args.output_name_template is not None:
        if "{seed}" not in args.output_name_template:
            parser.error(
                "--output-name-template must contain '{seed}' placeholder."
            )
        for seed in args.seeds:
            output_path = args.output_dir / args.output_name_template.format(seed=seed)
            build_random_subsample(
                input_path=args.input,
                output_path=output_path,
                target_size=args.target_size,
                seed=seed,
            )
    else:
        parser.error(
            "Provide either --output (single-seed mode), or both --output-dir + "
            "--output-name-template (multi-seed mode)."
        )

    print("Done.")


if __name__ == "__main__":
    main()
