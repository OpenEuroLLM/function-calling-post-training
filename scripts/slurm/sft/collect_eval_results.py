"""Collect and display evaluation results across experiments and runs.

Usage:
    python collect_eval_results.py [/path/to/eval-results]
"""

import glob
import json
import os
import sys


BASE_DIR = os.environ.get(
    "EVAL_RESULTS_DIR",
    "/leonardo_work/OELLM_prod2026/ytahtah0/eval-results",
)

EXPERIMENTS = {
    "A (IT+FC repro)": "exp_a_instruct_sft_v2",
    "B (Dolci FC)": "exp_b_dolci_fc_v2",
    "C (Nemotron FC)": "exp_c_nemotron_fc_v2",
    "D (IT only)": "exp_d_it_only_v2",
    "E (Mixed FC)": "exp_e_mixed_fc_v2",
    "F (IT+Nemotron)": "exp_f_instruct_plus_nemotron_v2",
}

# Benchmarks: display_name -> (file glob pattern, aggregation, expected_count)
#   'single' = one metrics file, use primary_score directly
#   'macro'  = multiple subject files, macro-average their primary_score
#   expected_count = how many subject files a complete run should have
BENCHMARKS = {
    "PopQA": ("popqa", "single", 1),
    "GSM8K": ("gsm8k", "single", 1),
    "IFEval": ("ifeval", "single", 1),
    "GPQA": ("gpqa", "single", 1),
    "MMLU": ("mmlu_*:cot", "macro", 57),
    "BBH": ("bbh_*", "macro", 27),
    "AGI Eval": ("agi_eval_*", "macro", 8),
    "MATH": ("minerva_math_*", "macro", 7),
    "HumanEval+": ("codex_humanevalplus", "single", 1),
    "MBPP+": ("mbppplus", "single", 1),
}


def get_score(directory, pattern, agg, expected):
    """Extract primary_score from metrics files matching pattern.

    For macro-averaged benchmarks, only returns a score if all expected
    subject files are present (avoids misleading partial averages).
    """
    files = sorted(glob.glob(os.path.join(directory, f"*-{pattern}-metrics.json")))
    if not files:
        return None, 0
    scores = []
    for f in files:
        try:
            d = json.load(open(f))
            s = d["metrics"].get("primary_score")
            if s is not None:
                scores.append(s)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    if not scores:
        return None, 0
    if agg == "macro" and len(scores) < expected:
        # Incomplete — don't report a misleading partial average
        return None, len(scores)
    if agg == "single":
        return scores[0], len(scores)
    else:  # macro average
        return sum(scores) / len(scores), len(scores)


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR

    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Collect all results
    results = {}  # results[bench][exp_label] = (mean, num_runs, [scores])
    for bench_name, (pattern, agg, expected) in BENCHMARKS.items():
        results[bench_name] = {}
        for label, exp_dir in EXPERIMENTS.items():
            run_scores = []
            for run in [1, 2, 3]:
                d = os.path.join(base_dir, exp_dir, f"run_{run}")
                s, _ = get_score(d, pattern, agg, expected)
                if s is not None:
                    run_scores.append(s)
            if run_scores:
                mean = sum(run_scores) / len(run_scores)
                results[bench_name][label] = (mean, len(run_scores), run_scores)

    # Print main table
    exp_labels = list(EXPERIMENTS.keys())
    col_w = 17
    header = f"{'Benchmark':<12}"
    for label in exp_labels:
        header += f" | {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for bench_name in BENCHMARKS:
        row = f"{bench_name:<12}"
        for label in exp_labels:
            if label in results[bench_name]:
                mean, n, _ = results[bench_name][label]
                row += f" | {mean * 100:12.1f}({n}r)"
            else:
                row += f" | {'—':>{col_w}}"
        print(row)

    # Print per-run detail
    print("\n")
    print("=" * 80)
    print("Per-run breakdown (primary_score x 100)")
    print("=" * 80)
    for label, exp_dir in EXPERIMENTS.items():
        print(f"\n--- {label} ({exp_dir}) ---")
        row_header = f"{'Benchmark':<12}"
        for run in [1, 2, 3]:
            row_header += f" {'Run ' + str(run):>10}"
        row_header += f" {'Mean':>10} {'Std':>8}"
        print(row_header)

        for bench_name, (pattern, agg, expected) in BENCHMARKS.items():
            row = f"{bench_name:<12}"
            scores = []
            for run in [1, 2, 3]:
                d = os.path.join(base_dir, exp_dir, f"run_{run}")
                s, n_files = get_score(d, pattern, agg, expected)
                if s is not None:
                    row += f" {s * 100:10.2f}"
                    scores.append(s)
                elif n_files > 0:
                    row += f" {n_files}/{expected}".rjust(10)
                else:
                    row += f" {'—':>10}"
            if scores:
                mean = sum(scores) / len(scores)
                row += f" {mean * 100:10.2f}"
                if len(scores) > 1:
                    variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
                    std = variance ** 0.5
                    row += f" {std * 100:8.2f}"
                else:
                    row += f" {'—':>8}"
            print(row)

    # Completion status
    print("\n")
    print("=" * 80)
    print("Completion status (prediction files per run)")
    print("=" * 80)
    for label, exp_dir in EXPERIMENTS.items():
        counts = []
        for run in [1, 2, 3]:
            d = os.path.join(base_dir, exp_dir, f"run_{run}")
            total = len(glob.glob(os.path.join(d, "*predictions*")))
            counts.append(str(total))
        print(f"  {label:<20} runs: {'/'.join(counts)}")


if __name__ == "__main__":
    main()
