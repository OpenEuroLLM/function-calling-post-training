#!/usr/bin/env python3
"""Independent, read-only fact-check of every BFCL number in notes/experiments.md.

Re-derives all metrics from the per-run BFCL CSVs (primary ground truth), the
5-run summary.txt files, and the markdown tables in experiments.md. Reports every
discrepancy > 0.01 (i.e. values that round to different 2-decimal numbers).

Does NOT import parse_bfcl_summary.py. Does NOT modify any file.
"""

import csv
import os
import re
import statistics
from collections import defaultdict

DOC = "/mnt/nfs/ytahtah/open-instruct/notes/experiments.md"
EVAL_ROOT = "/mnt/nfs/ytahtah/bfcl/eval_results"
TOL = 0.01  # flag if |delta| > 0.01

# Index ID -> eval name
EXPERIMENTS = [
    ("P1", "phase2-p1-fc-sft"),
    ("A1", "phase2-a1-fc-sft"),
    ("A1'", "phase2-a1prime-fc-sft"),
    ("A2", "phase2-a2-fc-sft"),
    ("A2'", "phase2-a2prime-fc-sft"),
    ("Q4", "phase2-q4-fc-sft"),
    ("Q1", "phase2-q1-fc-sft"),
    ("A3", "phase2-a3-fc-sft"),
    ("Q2", "phase2-q2-fc-sft"),
    ("A5_search", "phase2-a5-search-fc-sft"),
    ("A5_ia", "phase2-a5-ia-fc-sft"),
    ("A5_ia_ams", "phase2-a5-ia-ams-fc-sft"),
    ("A5_ia_random", "phase2-a5-ia-random-fc-sft"),
    ("A7", "phase2-a7-fc-sft"),
    ("A7_ams", "phase2-a7-ams-fc-sft"),
    ("A7_random", "phase2-a7-random-fc-sft"),
    ("centerpiece_raw", "phase2-centerpiece-raw-fc-sft"),
    ("centerpiece_AMS", "phase2-centerpiece-AMS-fc-sft"),
    ("centerpiece_random", "phase2-centerpiece-random-fc-sft"),
    ("centerpiece_turndrop", "phase2-centerpiece-turndrop-fc-sft"),
]

# Live sample weights for v3 Live weighting
LIVE_W = {
    "simple": 257, "multiple": 1052, "parallel": 15,
    "parallel_multiple": 23, "irrel": 884, "relevance": 16,
}

discrepancies = []  # (exp, metric, doc, recomputed, delta, comparison)
structural = []


def add_disc(exp, metric, doc, rec, comparison):
    if doc is None or rec is None:
        return
    delta = doc - rec
    if abs(delta) > TOL:
        discrepancies.append((exp, metric, doc, rec, delta, comparison))


def pct(s):
    """Parse '66.00%' or '66.00' -> 66.00 float; '' / 'N/A' -> None."""
    if s is None:
        return None
    s = s.strip().replace("%", "")
    if s in ("", "N/A", "-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pstd(vals):
    return statistics.pstdev(vals)


# ---------------------------------------------------------------------------
# 1. Read per-run CSVs -> per-run raw values
# ---------------------------------------------------------------------------
def read_csv_row(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def load_runs(eval_name, websearch=False):
    """Return dict run_idx -> dict of raw metrics parsed from that run's CSVs."""
    suffix = "_fc__websearch" if websearch else "_fc"
    base = os.path.join(EVAL_ROOT, f"eval_{eval_name}{suffix}")
    if not os.path.isdir(base):
        return None, base
    runs = {}
    for i in range(1, 6):
        score = os.path.join(base, f"run_{i}", "score")
        d = {}
        nl = read_csv_row(os.path.join(score, "data_non_live.csv"))
        lv = read_csv_row(os.path.join(score, "data_live.csv"))
        mt = read_csv_row(os.path.join(score, "data_multi_turn.csv"))
        ov = read_csv_row(os.path.join(score, "data_overall.csv"))
        ag = read_csv_row(os.path.join(score, "data_agentic.csv"))
        if nl:
            d["nl_simple"] = pct(nl.get("Simple AST"))
            d["nl_multiple"] = pct(nl.get("Multiple AST"))
            d["nl_parallel"] = pct(nl.get("Parallel AST"))
            d["nl_parallel_multiple"] = pct(nl.get("Parallel Multiple AST"))
            d["nl_irrel"] = pct(nl.get("Irrelevance Detection"))
            d["nl_ast_summary"] = pct(nl.get("AST Summary"))
        if lv:
            d["lv_simple"] = pct(lv.get("Python Simple AST"))
            d["lv_multiple"] = pct(lv.get("Python Multiple AST"))
            d["lv_parallel"] = pct(lv.get("Python Parallel AST"))
            d["lv_parallel_multiple"] = pct(lv.get("Python Parallel Multiple AST"))
            d["lv_irrel"] = pct(lv.get("Irrelevance Detection"))
            d["lv_relevance"] = pct(lv.get("Relevance Detection"))
            d["lv_acc"] = pct(lv.get("Live Overall Acc"))
        if mt:
            d["mt_base"] = pct(mt.get("Base"))
            d["mt_miss_func"] = pct(mt.get("Miss Func"))
            d["mt_miss_param"] = pct(mt.get("Miss Param"))
            d["mt_long_context"] = pct(mt.get("Long Context"))
            d["mt_acc"] = pct(mt.get("Multi Turn Overall Acc"))
        if ov:
            d["overall_acc"] = pct(ov.get("Overall Acc"))
            d["lat_mean"] = pct(ov.get("Latency Mean (s)"))
            d["lat_p95"] = pct(ov.get("Latency 95th Percentile (s)"))
            d["ov_relevance"] = pct(ov.get("Relevance Detection"))
            d["ov_irrel"] = pct(ov.get("Irrelevance Detection"))
        if ag:
            d["mem_kv"] = pct(ag.get("Memory KV"))
            d["mem_vector"] = pct(ag.get("Memory Vector"))
            d["mem_rec_sum"] = pct(ag.get("Memory Recursive Summarization"))
            d["mem_summary"] = pct(ag.get("Memory Summary"))
            d["ws_base"] = pct(ag.get("Web Search Base"))
            d["ws_no_snippet"] = pct(ag.get("Web Search No Snippet"))
            d["ws_summary"] = pct(ag.get("Web Search Summary"))
        runs[i] = d
    return runs, base


# ---------------------------------------------------------------------------
# 2. Parse summary.txt -> {label: (mean, std, [runs])}
# ---------------------------------------------------------------------------
def parse_summary(path):
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    blocks = {}
    # Each block: "Label:\n  Mean: X%  Std: Y%\n  Run 1: .. etc"
    pattern = re.compile(
        r"^(?P<label>[A-Za-z0-9 ()/'\-]+):\n"
        r"\s*Mean:\s*(?P<mean>[-\d.]+)%?\s+Std:\s*(?P<std>[-\d.]+)%?",
        re.MULTILINE,
    )
    for m in pattern.finditer(txt):
        label = m.group("label").strip()
        blocks[label] = {"mean": float(m.group("mean")), "std": float(m.group("std"))}
        # capture run values that follow
        start = m.end()
        runs = []
        for rm in re.finditer(r"Run \d+:\s*([-\d.]+)%?", txt[start:start + 400]):
            runs.append(float(rm.group(1)))
            if len(runs) == 5:
                break
        blocks[label]["runs"] = runs
    return blocks


# ---------------------------------------------------------------------------
# 3. Derived metrics per run
# ---------------------------------------------------------------------------
def nl_ast_v4(d):
    vals = [d["nl_simple"], d["nl_multiple"], d["nl_parallel"], d["nl_parallel_multiple"]]
    return sum(vals) / 4


def live_ast_weighted(d):
    """v3/v4 Live AST: sample-weighted over the 4 AST categories only (this is
    what BFCL reports as 'Live Acc' / AST Summary in v4)."""
    w = LIVE_W
    num = (d["lv_simple"] * w["simple"] + d["lv_multiple"] * w["multiple"]
           + d["lv_parallel"] * w["parallel"]
           + d["lv_parallel_multiple"] * w["parallel_multiple"])
    den = w["simple"] + w["multiple"] + w["parallel"] + w["parallel_multiple"]
    return num / den


def live_v3(d):
    """v3 Live: sample-weighted over 6 terms (4 AST + live irrel + relevance)."""
    w = LIVE_W
    num = (d["lv_simple"] * w["simple"] + d["lv_multiple"] * w["multiple"]
           + d["lv_parallel"] * w["parallel"]
           + d["lv_parallel_multiple"] * w["parallel_multiple"]
           + d["lv_irrel"] * w["irrel"] + d["lv_relevance"] * w["relevance"])
    den = sum(w.values())
    return num / den


def nl_v3(d):
    """v3 Non-Live (no EXEC): unweighted mean of 4 AST + non-live irrel."""
    vals = [d["nl_simple"], d["nl_multiple"], d["nl_parallel"],
            d["nl_parallel_multiple"], d["nl_irrel"]]
    return sum(vals) / 5


def mt_acc(d):
    vals = [d["mt_base"], d["mt_miss_func"], d["mt_miss_param"], d["mt_long_context"]]
    return sum(vals) / 4


def memory_acc(d):
    vals = [d["mem_kv"], d["mem_vector"], d["mem_rec_sum"]]
    return sum(vals) / 3


def total_irrel(d):
    return (d["nl_irrel"] + d["lv_irrel"]) / 2


def base_v4(d):
    return (0.10 * nl_ast_v4(d) + 0.10 * live_ast_weighted(d)
            + 0.10 * total_irrel(d) + 0.30 * mt_acc(d) + 0.20 * memory_acc(d))


def v3_overall(d):
    return (nl_v3(d) + live_v3(d) + mt_acc(d)) / 3


# ---------------------------------------------------------------------------
# 4. Parse experiments.md
# ---------------------------------------------------------------------------
def parse_doc():
    lines = open(DOC).read().splitlines()
    # Index table
    index = {}
    in_index = False
    for ln in lines:
        if ln.startswith("| ID |"):
            in_index = True
            continue
        if in_index:
            if not ln.startswith("|"):
                break
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 8 or cells[0] in ("---", "ID"):
                continue
            idv = cells[0]
            index[idv] = {
                "v3": pct(cells[5]), "base": pct(cells[6]), "full": pct(cells[7]),
            }
    # Detailed sections: split on '### '
    sections = {}
    cur = None
    buf = []
    for ln in lines:
        m = re.match(r"^### (\S+) ", ln)
        if m:
            if cur is not None:
                sections[cur] = buf
            cur = m.group(1)
            buf = []
        elif cur is not None:
            buf.append(ln)
    if cur is not None:
        sections[cur] = buf
    return index, sections


def parse_detail_table(sec_lines):
    """Parse the BFCL results table rows: label -> (mean, std)."""
    rows = {}
    for ln in sec_lines:
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        label = cells[0]
        # strip markdown bold and nbsp
        label = label.replace("**", "").replace("&nbsp;", "").strip()
        if label in ("Category", "---") or label.startswith("--"):
            continue
        mean = pct(cells[1]) if len(cells) > 1 else None
        std = pct(cells[2]) if len(cells) > 2 else None
        rows[label] = (mean, std)
    return rows


# ---------------------------------------------------------------------------
# Mapping: summary.txt label -> (per-run-derivation-key, doc detail label)
# ---------------------------------------------------------------------------
# For per-run -> summary check, we map summary label to a function of run dict.
SUMMARY_TO_RUNKEY = {
    "Non-Live AST Acc": nl_ast_v4,
    "Non-Live Simple AST": lambda d: d["nl_simple"],
    "Non-Live Multiple AST": lambda d: d["nl_multiple"],
    "Non-Live Parallel AST": lambda d: d["nl_parallel"],
    "Non-Live Parallel Multiple AST": lambda d: d["nl_parallel_multiple"],
    "Live Acc": lambda d: d["lv_acc"],
    "Live Simple AST": lambda d: d["lv_simple"],
    "Live Multiple AST": lambda d: d["lv_multiple"],
    "Live Parallel AST": lambda d: d["lv_parallel"],
    "Live Parallel Multiple AST": lambda d: d["lv_parallel_multiple"],
    "Multi Turn Acc": lambda d: d["mt_acc"],
    "Multi Turn Base": lambda d: d["mt_base"],
    "Multi Turn Miss Func": lambda d: d["mt_miss_func"],
    "Multi Turn Miss Param": lambda d: d["mt_miss_param"],
    "Multi Turn Long Context": lambda d: d["mt_long_context"],
    "Memory Acc": memory_acc,
    "Memory KV": lambda d: d["mem_kv"],
    "Memory Vector": lambda d: d["mem_vector"],
    "Memory Recursive Summarization": lambda d: d["mem_rec_sum"],
    "Relevance Detection": lambda d: d["ov_relevance"],
    "Irrelevance Detection": lambda d: d["ov_irrel"],
    "Overall Acc": lambda d: d["overall_acc"],
    "Latency Mean (s)": lambda d: d["lat_mean"],
    "Latency 95th Percentile (s)": lambda d: d["lat_p95"],
}

# doc detail label -> summary.txt label (for transcription check)
DOC_TO_SUMMARY = {
    "Non-Live AST": "Non-Live AST Acc",
    "Simple": None,  # ambiguous (appears twice); handled positionally below
    "Multiple": None,
    "Parallel": None,
    "Parallel Multiple": None,
    "Live Acc": "Live Acc",
    "Multi-Turn": "Multi Turn Acc",
    "Base": "Multi Turn Base",
    "Miss Func": "Multi Turn Miss Func",
    "Miss Param": "Multi Turn Miss Param",
    "Long Context": "Multi Turn Long Context",
    "Memory": "Memory Acc",
    "KV": "Memory KV",
    "Vector": "Memory Vector",
    "Recursive Summarization": "Memory Recursive Summarization",
    "Relevance Detection": "Relevance Detection",
    "Irrelevance Detection": "Irrelevance Detection",
    "Irrelevance Detection (total)": "Irrelevance Detection",
    "Latency Mean (s)": "Latency Mean (s)",
    "Latency P95 (s)": "Latency 95th Percentile (s)",
}


def get_ordered_detail(sec_lines):
    """Return list of (label, mean, std) in document order, labels de-duplicated
    with section context so Simple/Multiple under Non-Live vs Live can be told apart."""
    ordered = []
    context = None
    for ln in sec_lines:
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        raw_label = cells[0]
        indented = "&nbsp;" in raw_label
        label = raw_label.replace("**", "").replace("&nbsp;", "").strip()
        if label in ("Category",) or label.startswith("--"):
            continue
        mean = pct(cells[1]) if len(cells) > 1 else None
        std = pct(cells[2]) if len(cells) > 2 else None
        if not indented:
            context = label
        ordered.append((context, label, indented, mean, std))
    return ordered


def main():
    index, sections = parse_doc()

    # which checks passed
    passed = {
        "perrun_summary": [], "summary_doc": [], "derived": [],
        "index_detail": [], "ws": [], "colsem": [],
    }

    sanity_done = 0

    for exp_id, eval_name in EXPERIMENTS:
        runs, base_dir = load_runs(eval_name, websearch=False)
        if runs is None:
            structural.append(f"MISSING base eval dir: {base_dir}")
            continue
        summ = parse_summary(os.path.join(base_dir, "summary.txt"))
        if summ is None:
            structural.append(f"MISSING summary.txt: {base_dir}/summary.txt")
        ws_runs, ws_dir = load_runs(eval_name, websearch=True)
        ws_summ = None
        if ws_runs is None:
            structural.append(f"MISSING websearch eval dir: {ws_dir}")
        else:
            ws_summ = parse_summary(os.path.join(ws_dir, "summary.txt"))

        # ---- column-semantic sanity (>=2 experiments) ----
        if sanity_done < 3 and summ:
            # summary Irrelevance == mean over runs of mean(nl_irrel, lv_irrel)
            per_run_total_irrel = [total_irrel(runs[i]) for i in range(1, 6)]
            rec = statistics.mean(per_run_total_irrel)
            if "Irrelevance Detection" in summ:
                d = summ["Irrelevance Detection"]["mean"]
                if abs(d - rec) > TOL:
                    structural.append(
                        f"COLSEM {exp_id}: summary Irrelevance {d} != mean(nl,lv) {rec:.4f}")
                else:
                    passed["colsem"].append(f"{exp_id}:irrel")
            # summary Relevance == live relevance
            per_run_rel = [runs[i]["lv_relevance"] for i in range(1, 6)]
            rec_rel = statistics.mean(per_run_rel)
            if "Relevance Detection" in summ:
                d = summ["Relevance Detection"]["mean"]
                if abs(d - rec_rel) > TOL:
                    structural.append(
                        f"COLSEM {exp_id}: summary Relevance {d} != live relevance {rec_rel:.4f}")
                else:
                    passed["colsem"].append(f"{exp_id}:rel")
            sanity_done += 1

        # ---- 1. per-run -> summary.txt ----
        ok = True
        if summ:
            for slabel, fn in SUMMARY_TO_RUNKEY.items():
                if slabel not in summ:
                    continue
                try:
                    vals = [fn(runs[i]) for i in range(1, 6)]
                except (KeyError, TypeError):
                    continue
                if any(v is None for v in vals):
                    continue
                rec_mean = statistics.mean(vals)
                rec_std = pstd(vals)
                d_mean = summ[slabel]["mean"]
                d_std = summ[slabel]["std"]
                if abs(d_mean - rec_mean) > TOL:
                    discrepancies.append((exp_id, f"summary[{slabel}].mean", d_mean,
                                          rec_mean, d_mean - rec_mean, "per-run->summary"))
                    ok = False
                if abs(d_std - rec_std) > TOL:
                    discrepancies.append((exp_id, f"summary[{slabel}].std", d_std,
                                          rec_std, d_std - rec_std, "per-run->summary"))
                    ok = False
                # cross-check: sample stdev (wrong divisor) detection
                if len(set(vals)) > 1:
                    samp = statistics.stdev(vals)
                    if abs(d_std - samp) <= TOL and abs(d_std - rec_std) > TOL:
                        structural.append(
                            f"STD-DIVISOR {exp_id} {slabel}: summary std {d_std} matches "
                            f"SAMPLE stdev {samp:.4f} not population {rec_std:.4f}")
        if ok:
            passed["perrun_summary"].append(exp_id)

        # websearch per-run -> summary
        if ws_runs and ws_summ:
            for slabel, key in [("Web Search Acc", "ws_summary"),
                                ("Web Search Base", "ws_base"),
                                ("Web Search No Snippet", "ws_no_snippet")]:
                if slabel not in ws_summ:
                    continue
                vals = [ws_runs[i].get(key) for i in range(1, 6)]
                if any(v is None for v in vals):
                    continue
                rec_mean = statistics.mean(vals)
                rec_std = pstd(vals)
                if abs(ws_summ[slabel]["mean"] - rec_mean) > TOL:
                    discrepancies.append((exp_id, f"WSsummary[{slabel}].mean",
                                          ws_summ[slabel]["mean"], rec_mean,
                                          ws_summ[slabel]["mean"] - rec_mean,
                                          "per-run->summary(WS)"))
                if abs(ws_summ[slabel]["std"] - rec_std) > TOL:
                    discrepancies.append((exp_id, f"WSsummary[{slabel}].std",
                                          ws_summ[slabel]["std"], rec_std,
                                          ws_summ[slabel]["std"] - rec_std,
                                          "per-run->summary(WS)"))

        # ---- 2. summary.txt -> doc detailed table ----
        sec = sections.get(exp_id)
        if sec is None:
            structural.append(f"MISSING doc section for {exp_id}")
            continue
        ordered = get_ordered_detail(sec)
        # Build doc detail lookup keyed by (context,label,indented)
        ok2 = True
        for context, label, indented, mean, std in ordered:
            # Resolve summary label
            slabel = None
            if not indented:
                slabel = DOC_TO_SUMMARY.get(label)
            else:
                # sub-rows: resolve by context
                if context == "Non-Live AST":
                    slabel = {"Simple": "Non-Live Simple AST",
                              "Multiple": "Non-Live Multiple AST",
                              "Parallel": "Non-Live Parallel AST",
                              "Parallel Multiple": "Non-Live Parallel Multiple AST"}.get(label)
                elif context == "Live Acc":
                    slabel = {"Simple": "Live Simple AST",
                              "Multiple": "Live Multiple AST",
                              "Parallel": "Live Parallel AST",
                              "Parallel Multiple": "Live Parallel Multiple AST"}.get(label)
                elif context in ("Multi-Turn",):
                    slabel = DOC_TO_SUMMARY.get(label)
                elif context == "Memory":
                    slabel = DOC_TO_SUMMARY.get(label)
                elif context and context.startswith("Web Search"):
                    slabel = None  # handled in WS section
            if slabel and summ and slabel in summ:
                dm = summ[slabel]["mean"]
                ds = summ[slabel]["std"]
                if mean is not None and abs(mean - dm) > TOL:
                    discrepancies.append((exp_id, f"detail[{context}/{label}].mean",
                                          mean, dm, mean - dm, "summary->doc"))
                    ok2 = False
                if std is not None and abs(std - ds) > TOL:
                    discrepancies.append((exp_id, f"detail[{context}/{label}].std",
                                          std, ds, std - ds, "summary->doc"))
                    ok2 = False
        if ok2:
            passed["summary_doc"].append(exp_id)

        # ---- 5. Web Search doc rows vs WS summary ----
        okws = True
        doc_rows = {}
        for context, label, indented, mean, std in ordered:
            doc_rows[(context, label, indented)] = (mean, std)
        if ws_summ:
            # main Web Search row (mean of base+no_snippet)
            ws_acc = ws_summ.get("Web Search Acc")
            # Find the doc's Web Search top row
            for (context, label, indented), (mean, std) in doc_rows.items():
                if not indented and (label == "Web Search"
                                     or label.startswith("Web Search (")):
                    if ws_acc and mean is not None:
                        if abs(mean - ws_acc["mean"]) > TOL:
                            discrepancies.append((exp_id, "WS.mean", mean,
                                                  ws_acc["mean"], mean - ws_acc["mean"], "WS"))
                            okws = False
                        if std is not None and abs(std - ws_acc["std"]) > TOL:
                            discrepancies.append((exp_id, "WS.std", std,
                                                  ws_acc["std"], std - ws_acc["std"], "WS"))
                            okws = False
                if indented and label == "Web Search Base":
                    wsb = ws_summ.get("Web Search Base")
                    if wsb and mean is not None and abs(mean - wsb["mean"]) > TOL:
                        discrepancies.append((exp_id, "WS Base.mean", mean, wsb["mean"],
                                              mean - wsb["mean"], "WS"))
                        okws = False
                    if wsb and std is not None and abs(std - wsb["std"]) > TOL:
                        discrepancies.append((exp_id, "WS Base.std", std, wsb["std"],
                                              std - wsb["std"], "WS"))
                        okws = False
                if indented and label == "Web Search No Snippet":
                    wsn = ws_summ.get("Web Search No Snippet")
                    if wsn and mean is not None and abs(mean - wsn["mean"]) > TOL:
                        discrepancies.append((exp_id, "WS NoSnip.mean", mean, wsn["mean"],
                                              mean - wsn["mean"], "WS"))
                        okws = False
        if okws:
            passed["ws"].append(exp_id)

        # ---- 3. Derived metrics from per-run CSVs ----
        v3_vals = [v3_overall(runs[i]) for i in range(1, 6)]
        base_vals = [base_v4(runs[i]) for i in range(1, 6)]
        v3_mean, v3_std = statistics.mean(v3_vals), pstd(v3_vals)
        base_mean, base_std = statistics.mean(base_vals), pstd(base_vals)
        ws_mean = ws_summ["Web Search Acc"]["mean"] if (ws_summ and "Web Search Acc" in ws_summ) else 0.0
        full_mean = base_mean + 0.20 * ws_mean

        # Cross-check: recomputed base_v4 should match summary Overall Acc mean
        if summ and "Overall Acc" in summ:
            if abs(summ["Overall Acc"]["mean"] - base_mean) > TOL:
                structural.append(
                    f"DERIVED-CHECK {exp_id}: recomputed Base v4 {base_mean:.4f} != "
                    f"summary Overall Acc {summ['Overall Acc']['mean']}")

        # Compare derived vs DETAIL table headline rows
        okd = True
        for context, label, indented, mean, std in ordered:
            if indented:
                continue
            l = label
            if l.startswith("v3 Overall"):
                if mean is not None and abs(mean - v3_mean) > TOL:
                    discrepancies.append((exp_id, "v3 Overall.mean(detail)", mean,
                                          v3_mean, mean - v3_mean, "derived"))
                    okd = False
                if std is not None and abs(std - v3_std) > TOL:
                    discrepancies.append((exp_id, "v3 Overall.std(detail)", std,
                                          v3_std, std - v3_std, "derived"))
                    okd = False
            elif l.startswith("Overall (base"):
                if mean is not None and abs(mean - base_mean) > TOL:
                    discrepancies.append((exp_id, "Base v4.mean(detail)", mean,
                                          base_mean, mean - base_mean, "derived"))
                    okd = False
                if std is not None and abs(std - base_std) > TOL:
                    discrepancies.append((exp_id, "Base v4.std(detail)", std,
                                          base_std, std - base_std, "derived"))
                    okd = False
            elif l.startswith("Full Overall"):
                if mean is not None and abs(mean - full_mean) > TOL:
                    discrepancies.append((exp_id, "Full v4.mean(detail)", mean,
                                          full_mean, mean - full_mean, "derived"))
                    okd = False
        if okd:
            passed["derived"].append(exp_id)

        # ---- 4. Index <-> detail consistency + derived vs index ----
        oki = True
        idx = index.get(exp_id)
        if idx is None:
            structural.append(f"MISSING index row for {exp_id}")
        else:
            # detail headline values
            det = {}
            for context, label, indented, mean, std in ordered:
                if indented:
                    continue
                if label.startswith("v3 Overall"):
                    det["v3"] = mean
                elif label.startswith("Overall (base"):
                    det["base"] = mean
                elif label.startswith("Full Overall"):
                    det["full"] = mean
            for k, derived_val in [("v3", v3_mean), ("base", base_mean), ("full", full_mean)]:
                # index vs detail
                if idx.get(k) is not None and det.get(k) is not None:
                    if abs(idx[k] - det[k]) > TOL:
                        discrepancies.append((exp_id, f"Index.{k} vs detail.{k}",
                                              idx[k], det[k], idx[k] - det[k],
                                              "Index<->detail"))
                        oki = False
                # index vs derived
                if idx.get(k) is not None and abs(idx[k] - derived_val) > TOL:
                    discrepancies.append((exp_id, f"Index.{k} vs derived",
                                          idx[k], derived_val, idx[k] - derived_val,
                                          "derived(Index)"))
                    oki = False
        if oki:
            passed["index_detail"].append(exp_id)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("BFCL NUMBER VERIFICATION REPORT")
    print("=" * 100)
    n = len(discrepancies)
    print(f"\nVERDICT: {'PASS' if n == 0 and not any('STD-DIVISOR' in s or 'DERIVED-CHECK' in s or 'COLSEM' in s for s in structural) else 'FAIL'}  "
          f"({n} value discrepancies > {TOL})\n")

    if discrepancies:
        print("-" * 100)
        print(f"{'EXP':<20} {'METRIC':<34} {'DOC':>9} {'RECOMP':>9} {'DELTA':>9}  COMPARISON")
        print("-" * 100)
        for exp, metric, doc, rec, delta, cmp in discrepancies:
            print(f"{exp:<20} {metric:<34} {doc:>9.2f} {rec:>9.4f} {delta:>9.4f}  {cmp}")

    print("\n" + "-" * 100)
    print("STRUCTURAL / SANITY NOTES")
    print("-" * 100)
    if structural:
        for s in structural:
            print(" - " + s)
    else:
        print(" (none)")

    print("\n" + "-" * 100)
    print("CHECKS PASSED (experiment IDs)")
    print("-" * 100)
    for k, v in passed.items():
        print(f" {k:<18}: {len(v)} -> {sorted(set(v))}")


if __name__ == "__main__":
    main()
