"""Filter Dolci-Instruct-SFT to remove Tool-Use samples, producing IT-only data.

Loads both datasets, collects Tool-Use IDs, filters them out from the full SFT
dataset, and saves the result as parquet for use with convert_sft_data_for_olmocore.py.

Usage:
    python scripts/data/filter_dolci_it_only.py --output_path /path/to/dolci_it_only.parquet
"""

import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Filter Dolci-Instruct-SFT to IT-only (no FC samples).")
    parser.add_argument("--output_path", type=str, required=True, help="Output parquet file path.")
    args = parser.parse_args()

    print("Loading Dolci-Instruct-SFT-Tool-Use (to get IDs to exclude)...")
    ds_tu = load_dataset("allenai/Dolci-Instruct-SFT-Tool-Use", split="train")
    tu_ids = set(ds_tu["id"])
    print(f"  Tool-Use IDs to exclude: {len(tu_ids)}")

    print("Loading Dolci-Instruct-SFT (full dataset)...")
    ds_sft = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    print(f"  Full SFT samples: {len(ds_sft)}")

    print("Filtering...")
    ds_it = ds_sft.filter(lambda x: x["id"] not in tu_ids, num_proc=8)
    print(f"  IT-only samples: {len(ds_it)}")
    print(f"  Removed: {len(ds_sft) - len(ds_it)}")

    assert len(ds_it) == len(ds_sft) - len(tu_ids), (
        f"Expected {len(ds_sft) - len(tu_ids)} samples, got {len(ds_it)}"
    )

    print(f"Saving to {args.output_path}...")
    ds_it.to_parquet(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
