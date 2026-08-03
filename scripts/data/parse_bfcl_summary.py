"""Parse BFCL summary.txt files and compute Full BFCL Overall.

Reads 5-run mean values from the base eval summary.txt and 1-run Web Search
mean from the __websearch summary.txt, then computes:

    Full BFCL Overall = 0.10*NL + 0.10*Live + 0.10*Irrel + 0.30*MT + 0.40*(Mem+WS)/2

Also prints the recomputed Base Overall (with WS=0 in the agentic term) so we
can sanity-check it against the BFCL CLI's reported "Overall Acc".

Usage:
    python parse_bfcl_summary.py [exp ...]

If no arguments are given, parses the 7 Wave 0 experiments by default.
"""

import argparse
import re
from pathlib import Path

EVAL_DIR = Path("/mnt/nfs/ytahtah/bfcl/eval_results")

WAVE0_EXPS = [
    "phase2-a5-search-fc-sft",
    "phase2-a5-ia-fc-sft",
    "phase2-a5-ia-ams-fc-sft",
    "phase2-a5-ia-random-fc-sft",
    "phase2-a7-fc-sft",
    "phase2-a7-ams-fc-sft",
    "phase2-a7-random-fc-sft",
]

METRIC_LINE = re.compile(r"^([A-Z][^:]+):\s*$")
MEAN_LINE = re.compile(r"^\s+Mean:\s+([\d.]+)%?\s+Std:\s+([\d.]+)%?\s*$")
RUN_LINE = re.compile(r"^\s+Run\s+(\d+):\s+([\d.]+)%?\s*$")


def parse_summary(path: Path) -> dict[str, dict]:
    """Parse a summary.txt into metric_name -> {mean, std, runs: [..]}."""
    out: dict[str, dict] = {}
    current: str | None = None
    with open(path) as f:
        for line in f:
            m = METRIC_LINE.match(line.rstrip())
            if m:
                current = m.group(1).strip()
                out[current] = {"mean": None, "std": None, "runs": []}
                continue
            if current is None:
                continue
            m = MEAN_LINE.match(line)
            if m:
                out[current]["mean"] = float(m.group(1))
                out[current]["std"] = float(m.group(2))
                continue
            m = RUN_LINE.match(line)
            if m:
                out[current]["runs"].append(float(m.group(2)))
    # Drop empty entries (header sections without numbers)
    return {k: v for k, v in out.items() if v["mean"] is not None}


def get_mean(d: dict, key: str) -> float:
    return d[key]["mean"]


def get_std(d: dict, key: str) -> float:
    return d[key]["std"]


def get_runs(d: dict, key: str) -> list[float]:
    return d[key]["runs"]


def compute_overall_from_components(nl: float, live: float, irrel: float, mt: float, mem: float, ws: float) -> float:
    """Apply the BFCL v4 Full Overall formula on a single set of component values.

    Setting ws=0 reproduces the BFCL CLI's reported "Overall Acc" (base eval).
    """
    return 0.10 * nl + 0.10 * live + 0.10 * irrel + 0.30 * mt + 0.40 * (mem + ws) / 2


def compute_overall(base: dict, ws_value: float = 0.0) -> float:
    """Apply the formula on mean values."""
    return compute_overall_from_components(
        nl=get_mean(base, "Non-Live AST Acc"),
        live=get_mean(base, "Live Acc"),
        irrel=get_mean(base, "Irrelevance Detection"),
        mt=get_mean(base, "Multi Turn Acc"),
        mem=get_mean(base, "Memory Acc"),
        ws=ws_value,
    )


def per_run_full(base: dict, ws_value: float) -> list[float]:
    """Compute Full BFCL Overall per run (same WS applied to each run since WS is 1-run).

    Returns list of 5 Full Overall values (assuming 5 base runs). Captures variance
    from base components but NOT from WS (since WS has no multi-run data).
    """
    nl_runs = get_runs(base, "Non-Live AST Acc")
    live_runs = get_runs(base, "Live Acc")
    irrel_runs = get_runs(base, "Irrelevance Detection")
    mt_runs = get_runs(base, "Multi Turn Acc")
    mem_runs = get_runs(base, "Memory Acc")
    n = min(len(nl_runs), len(live_runs), len(irrel_runs), len(mt_runs), len(mem_runs))
    return [
        compute_overall_from_components(
            nl_runs[i], live_runs[i], irrel_runs[i], mt_runs[i], mem_runs[i], ws_value
        )
        for i in range(n)
    ]


def std_of(values: list[float]) -> float:
    """Population std (matches BFCL CLI's std convention)."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / n) ** 0.5


def report_one(exp: str) -> None:
    base_path = EVAL_DIR / f"eval_{exp}_fc" / "summary.txt"
    ws_path = EVAL_DIR / f"eval_{exp}_fc__websearch" / "summary.txt"
    base = parse_summary(base_path)
    ws = parse_summary(ws_path)
    ws_acc = get_mean(ws, "Web Search Acc")
    cli_overall_mean = get_mean(base, "Overall Acc")
    cli_overall_std = get_std(base, "Overall Acc")
    recomputed_base = compute_overall(base, ws_value=0.0)
    full = compute_overall(base, ws_value=ws_acc)
    full_runs = per_run_full(base, ws_acc)
    full_std = std_of(full_runs)
    base_runs = per_run_full(base, 0.0)
    base_std = std_of(base_runs)
    print(f"\n=== {exp} ===")
    print(f"  CLI Overall (base, WS=0)             : {cli_overall_mean:.4f}  ± {cli_overall_std:.4f}")
    print(f"  Recomputed Base Overall (WS=0)       : {recomputed_base:.4f}  ± {base_std:.4f}  (per-run)")
    print(f"  Δ (recomputed - CLI)                 : {recomputed_base - cli_overall_mean:+.4f}")
    print(f"  Web Search Acc (1-run)               : {ws_acc:.4f}  (no std — 1 run)")
    print(f"  Full BFCL Overall (WS included)      : {full:.4f}  ± {full_std:.4f}  (std excludes WS variance)")
    print()
    print(f"  Non-Live AST Acc      : {get_mean(base, 'Non-Live AST Acc'):.2f} ± {get_std(base, 'Non-Live AST Acc'):.2f}")
    print(f"    Simple              : {get_mean(base, 'Non-Live Simple AST'):.2f}")
    print(f"    Multiple            : {get_mean(base, 'Non-Live Multiple AST'):.2f}")
    print(f"    Parallel            : {get_mean(base, 'Non-Live Parallel AST'):.2f}")
    print(f"    Parallel Multiple   : {get_mean(base, 'Non-Live Parallel Multiple AST'):.2f}")
    print(f"  Live Acc              : {get_mean(base, 'Live Acc'):.2f} ± {get_std(base, 'Live Acc'):.2f}")
    print(f"    Simple              : {get_mean(base, 'Live Simple AST'):.2f}")
    print(f"    Multiple            : {get_mean(base, 'Live Multiple AST'):.2f}")
    print(f"    Parallel            : {get_mean(base, 'Live Parallel AST'):.2f}")
    print(f"    Parallel Multiple   : {get_mean(base, 'Live Parallel Multiple AST'):.2f}")
    print(f"  Multi Turn Acc        : {get_mean(base, 'Multi Turn Acc'):.2f} ± {get_std(base, 'Multi Turn Acc'):.2f}")
    print(f"    Base                : {get_mean(base, 'Multi Turn Base'):.2f}")
    print(f"    Miss Func           : {get_mean(base, 'Multi Turn Miss Func'):.2f}")
    print(f"    Miss Param          : {get_mean(base, 'Multi Turn Miss Param'):.2f}")
    print(f"    Long Context        : {get_mean(base, 'Multi Turn Long Context'):.2f}")
    print(f"  Memory Acc            : {get_mean(base, 'Memory Acc'):.2f} ± {get_std(base, 'Memory Acc'):.2f}")
    print(f"    KV                  : {get_mean(base, 'Memory KV'):.2f}")
    print(f"    Vector              : {get_mean(base, 'Memory Vector'):.2f}")
    print(f"    Recursive Summ.     : {get_mean(base, 'Memory Recursive Summarization'):.2f}")
    print(f"  Relevance Detection   : {get_mean(base, 'Relevance Detection'):.2f} ± {get_std(base, 'Relevance Detection'):.2f}")
    print(f"  Irrelevance Detection : {get_mean(base, 'Irrelevance Detection'):.2f} ± {get_std(base, 'Irrelevance Detection'):.2f}")
    print(f"  Latency Mean (s)      : {get_mean(base, 'Latency Mean (s)'):.2f}")
    print(f"  Latency 95th (s)      : {get_mean(base, 'Latency 95th Percentile (s)'):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*", help="Experiment names (default: Wave 0 set)")
    args = ap.parse_args()
    exps = args.exps if args.exps else WAVE0_EXPS
    for exp in exps:
        report_one(exp)


if __name__ == "__main__":
    main()
