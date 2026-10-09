#!/usr/bin/env python3
"""ours/report_fluid.py: PDF report comparing B1 with the fluid-dual policy over the system
scale, for an underloaded and an overloaded operating point.

    python ours/report_fluid.py --out ours/experiments/<date>_report_fluid_dual_scale64 \\
        --under-b1 <folder> --under-fd <folder> --over-b1 <folder> --over-fd <folder>

Each folder is a scaling experiment written by sweep.py and valued by objective.py
(summary.csv, objective.csv, specs/fluid_n*.yaml for the fluid-dual folders). The report
states the strategy of each policy for admission, routing, batching and eviction, then shows
the revenue per second against the system scale n and the main statistics, for both loads.
Compiled with pdflatex. House style: no em dashes or en dashes.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POL = {"B1": ("B1 (llm-d default)", "#4C72B0"), "FD": ("fluid-dual", "#DD8452")}
TYPES = ["type1", "type2", "type3"]
TYPE_WORDS = {"type1": "type1 (short chat)", "type2": "type2 (RAG)", "type3": "type3 (long generation)", "all": "all types"}


def esc(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$").replace("{", r"\{").replace("}", r"\}"))


def load_exp(folder, fluid=False):
    s = pd.read_csv(os.path.join(folder, "summary.csv"))
    o = pd.read_csv(os.path.join(folder, "objective.csv"))
    m = json.load(open(os.path.join(folder, "manifest.json")))
    fl = None
    if fluid:
        rows = []
        for p in glob.glob(os.path.join(folder, "specs", "fluid_n*.yaml")):
            meta = yaml.safe_load(open(p))["meta"]
            tps = yaml.safe_load(open(p))["types"]
            rows.append({"n": int(re.search(r"fluid_n(\d+)", p).group(1)), "value": meta["value"], "bound": meta["bound"],
                         "gap": meta["gap"], "status": meta["status"], "eta0": meta["eta0"],
                         "kkt": meta.get("kkt_residual"),
                         **{f"adm_{t['name']}": t["admitted"] / t["lambda"] if t["lambda"] else float("nan") for t in tps}})
        fl = pd.DataFrame(rows).sort_values("n")
    return {"summary": s, "objective": o, "manifest": m, "fluid": fl, "folder": folder}


def series(d, col, t="all", src="summary"):
    df = d[src]
    g = df[df["type"] == t].sort_values("n")
    ci = g[col + "_ci95"].fillna(0).values if col + "_ci95" in g else np.zeros(len(g))
    return g.n.values, g[col + "_mean"].values, ci


def style(ax, ylabel, title=None, logy=False):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    xs = sorted({float(v) for line in ax.get_lines() for v in line.get_xdata()})
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in xs]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("system scale n (instances)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, color="#e5e5e5", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_line(ax, d, key, col, t="all", src="summary", scale=1.0, label=True):
    x, y, e = series(d, col, t, src)
    name, color = POL[key]
    ax.plot(x, y * scale, marker="o", ms=4, lw=1.6, color=color, label=name if label else None)
    ax.fill_between(x, (y - e) * scale, (y + e) * scale, color=color, alpha=0.2, linewidth=0)


def revenue_figure(data, out):
    """Top row: revenue value per second against n. Bottom row: the same divided by n."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=150)
    for c, load in enumerate(("under", "over")):
        b1, fd = data[load]["B1"], data[load]["FD"]
        fl = fd["fluid"]
        for r, per in enumerate((False, True)):
            ax = axes[r, c]
            for key, d in (("B1", b1), ("FD", fd)):
                x, y, e = series(d, "value_per_s", "all", "objective")
                k = x if per else np.ones_like(x)
                name, color = POL[key]
                ax.plot(x, y / k, marker="o", ms=4, lw=1.6, color=color, label=name)
                ax.fill_between(x, (y - e) / k, (y + e) / k, color=color, alpha=0.2, linewidth=0)
            ax.plot(fl.n, fl.value / (fl.n if per else 1), ls="--", color="black", lw=1.2, label="fluid optimum V*")
            style(ax, "revenue value per second per instance" if per else "revenue value per second",
                  data[load]["title"] if r == 0 else None)
            if c == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def grid_figure(data, metrics, out):
    """rows = metrics, columns = loads, lines = policies (type all)."""
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 3.1 * len(metrics)), dpi=150, squeeze=False)
    for r, (col, label, scale, logy) in enumerate(metrics):
        for c, load in enumerate(("under", "over")):
            ax = axes[r, c]
            for key in ("B1", "FD"):
                plot_line(ax, data[load][key], key, col, scale=scale)
            style(ax, label, data[load]["title"] if r == 0 else None, logy)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def type_figure(data, col, label, scale, out, src="summary"):
    """rows = types, columns = loads."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.6), dpi=150, squeeze=False)
    for r, t in enumerate(TYPES):
        for c, load in enumerate(("under", "over")):
            ax = axes[r, c]
            for key in ("B1", "FD"):
                plot_line(ax, data[load][key], key, col, t=t, src=src, scale=scale)
            style(ax, f"{label}, {t}", data[load]["title"] if r == 0 else None)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fmt(v, ci=None, digits=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{v:,.{digits}f}"
    if ci is not None and not (isinstance(ci, float) and math.isnan(ci)):
        s += f" $\\pm$ {ci:,.{digits}f}"
    return s


def results_table(data, load):
    b1, fd = data[load]["B1"], data[load]["FD"]
    rows = []
    head = (r"n & policy & revenue/s & E2E mean (s) & TTFT mean (s) & TTFT p99 (s) & rejected \% & unfinished \% & preempt./arrival \\")
    for n in sorted(set(b1["summary"].n)):
        for key, d in (("B1", b1), ("FD", fd)):
            s = d["summary"][(d["summary"].n == n) & (d["summary"]["type"] == "all")]
            o = d["objective"][(d["objective"].n == n) & (d["objective"]["type"] == "all")]
            if s.empty or o.empty:
                continue
            s, o = s.iloc[0], o.iloc[0]
            rows.append(" & ".join([
                str(int(n)), esc(POL[key][0]),
                fmt(o.value_per_s_mean, o.value_per_s_ci95),
                fmt(s.e2e_mean_ms_mean / 1000, s.e2e_mean_ms_ci95 / 1000, 2),
                fmt(s.ttft_mean_ms_mean / 1000, s.ttft_mean_ms_ci95 / 1000, 3),
                fmt(s.ttft_p99_ms_mean / 1000, None, 2),
                fmt(100 * s.rejected_frac_mean, None, 2),
                fmt(100 * s.unfinished_frac_mean, None, 2),
                fmt(s.preempt_per_arrival_mean, None, 3),
            ]) + r" \\")
    return ("\\begin{center}\\scriptsize\\begin{tabular}{rlrrrrrrr}\\toprule\n" + head + "\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def type_table(data, load):
    rows = []
    head = r"n & type & B1 revenue/s & FD revenue/s & B1 rejected \% & FD rejected \% & fluid admits \% & B1 E2E (s) & FD E2E (s) \\"
    b1, fd = data[load]["B1"], data[load]["FD"]
    for n in sorted(set(b1["summary"].n)):
        for t in TYPES:
            g = lambda d, src, col: d[src][(d[src].n == n) & (d[src]["type"] == t)][col].iloc[0]  # noqa: E731
            try:
                fl = fd["fluid"][fd["fluid"].n == n][f"adm_{t}"].iloc[0]
            except (IndexError, KeyError):
                fl = float("nan")
            rows.append(" & ".join([str(int(n)), t,
                                    fmt(g(b1, "objective", "value_per_s_mean")), fmt(g(fd, "objective", "value_per_s_mean")),
                                    fmt(100 * g(b1, "summary", "rejected_frac_mean"), None, 2),
                                    fmt(100 * g(fd, "summary", "rejected_frac_mean"), None, 2),
                                    fmt(100 * fl, None, 1),
                                    fmt(g(b1, "summary", "e2e_mean_ms_mean") / 1000, None, 2),
                                    fmt(g(fd, "summary", "e2e_mean_ms_mean") / 1000, None, 2)]) + r" \\")
    return ("\\begin{center}\\scriptsize\\begin{tabular}{rlrrrrrrr}\\toprule\n" + head + "\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def fluid_table(data):
    rows = []
    for load in ("under", "over"):
        fl = data[load]["FD"]["fluid"]
        for _, r in fl.iterrows():
            rows.append(" & ".join([esc(data[load]["short"]), str(int(r.n)), fmt(r.value, None, 2), fmt(r.bound, None, 2),
                                    f"{100 * r.gap:.3f}" if r.gap is not None and not math.isnan(r.gap) else "--",
                                    esc(r.status), fmt(r.eta0, None, 2)]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{lrrrrlr}\\toprule\n"
            r"load & n & fluid value V* & upper bound & gap \% & solver status & $\eta = V^*/J$ \\" + "\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


STRATEGY = r"""
\begin{center}\small
\begin{tabular}{p{2.2cm}p{6.3cm}p{6.3cm}}\toprule
Decision & B1 (llm-d production default) & Fluid-dual (model draft, Definition 1) \\\midrule
Admission & Every arrival is admitted. & At the arrival of a type-$i$ job compute, for every instance $j$, the index $I_{ij}(t) = r_i - c^\star_{ij} - \gamma^M_j(t)\,a^M_{ij} - \gamma^B_j(t)\,a^B_{ij}$. Accept iff $\max_j I_{ij}(t) \ge 0$; otherwise reject (the job is lost, earns nothing and costs nothing). \\
Routing & Immediately at arrival, to the instance with the highest weighted score of three llm-d scorers (precise prefix cache 2, queue depth 1, KV utilisation 1) computed on instance snapshots refreshed every 50 ms. & Immediately at acceptance, to an instance with the largest index $I_{ij}(t)$, ties to the smallest instance index. Prices use live state: $\gamma^M_j(t) = [\gamma^{M\star}_j + \eta (K_j(t) - K^\star_j)/M_j]^+$ with $K_j(t)$ the KV memory in use, and $\gamma^B_j(t) = [\gamma^{B\star}_j + \eta (S_j(t) - S^\star_j)/b_j]^+$ with $S_j(t)$ the prompt tokens of the stage-0 jobs assigned to $j$ (waiting, mid-prefill, evicted, or in the API delay after routing). \\
Batching & vLLM continuous batching at every step: every running job continues (one decode token, or the next prefill chunk); then waiting jobs join in arrival order while the per-step token budget (8,192), the request limit (1,024) and free KV allow; chunked prefill; head-of-line stop; no admission in a step that evicted. & Identical to B1 (Definition 1, part 2, is vLLM's rule: every prefilled job, then stage-0 jobs in arrival order while the token budget and memory hold). \\
Eviction & When the running jobs need more KV than is free, the most recently started job is evicted (recompute: its KV is freed and it restarts from its prompt on the same instance). & Identical to B1 (Definition 1, part 3: evict prefilled jobs, the most recently prefilled first). \\
Pool & None: every admitted job is routed at once (push). & None: the pool stays empty under this policy (push). \\\bottomrule
\end{tabular}
\end{center}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--under-b1", required=True)
    ap.add_argument("--under-fd", required=True)
    ap.add_argument("--over-b1", required=True)
    ap.add_argument("--over-fd", required=True)
    ap.add_argument("--rcap", type=float, default=12.0, help="B1 capacity point per instance, req/s")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}
    for load, b1p, fdp in (("under", a.under_b1, a.under_fd), ("over", a.over_b1, a.over_fd)):
        b1, fd = load_exp(b1p), load_exp(fdp, fluid=True)
        rpi = b1["manifest"]["rate_per_instance"]
        word = "underloaded" if load == "under" else "overloaded"
        data[load] = {"B1": b1, "FD": fd, "rpi": rpi,
                      "title": f"{word}: {rpi:g} req/s per instance ({rpi / a.rcap:.2f} R_cap)",
                      "short": f"{word} ({rpi:g})"}
    figs = {
        "revenue": os.path.join(a.out, "revenue_vs_scale.png"),
        "latency": os.path.join(a.out, "latency_vs_scale.png"),
        "congestion": os.path.join(a.out, "admission_congestion_vs_scale.png"),
        "revtype": os.path.join(a.out, "revenue_by_type.png"),
        "rejtype": os.path.join(a.out, "rejection_by_type.png"),
    }
    revenue_figure(data, figs["revenue"])
    grid_figure(data, [("e2e_mean_ms", "E2E latency, mean (s)", 1e-3, False),
                       ("ttft_mean_ms", "TTFT, mean (s)", 1e-3, True),
                       ("ttft_p99_ms", "TTFT, p99 (s)", 1e-3, True)], figs["latency"])
    grid_figure(data, [("rejected_frac", "rejected at admission (percent of arrivals)", 100, False),
                       ("unfinished_frac", "unfinished at horizon (percent of arrivals)", 100, False),
                       ("preempt_per_arrival", "evictions per arrival", 1, False),
                       ("output_tok_per_s", "output tokens per second (completed)", 1, False)], figs["congestion"])
    type_figure(data, "value_per_s", "revenue/s", 1, figs["revtype"], src="objective")
    type_figure(data, "rejected_frac", "rejected (percent)", 100, figs["rejtype"])

    m = data["under"]["B1"]["manifest"]
    seeds = m["seeds"]
    cut = m.get("seeds_from_n") or ""
    obj = yaml.safe_load(open(os.path.join(HERE, "specs", "objective.yaml")))["types"]
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{B1 versus the fluid-dual policy over the system scale}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
"""]
    tex.append(r"\section{Setup}")
    tex.append("\\begin{itemize}\n"
               f"\\item System: n identical instances, n = {', '.join(str(x) for x in m['instances'])}; each instance is one H100 GPU serving Qwen3-14B (tensor parallelism 1) with {m['blocks']:,} KV blocks of 16 tokens ({m['blocks']*16:,} tokens), per-step token budget 8,192, request limit 1,024. Step times from BLIS's trained-physics model.\n"
               "\\item Workload: Poisson arrivals, three job types in fixed shares 50 / 35 / 15 percent: type1 prompt 256 tokens, mean output 64; type2 prompt 2,048, mean output 128; type3 prompt 512, mean output 1,024 (output lengths rounded exponential).\n"
               f"\\item Loads: total arrival rate = n times the rate per instance; underloaded {data['under']['rpi']:g} req/s per instance ({data['under']['rpi']/a.rcap:.2f} times B1's capacity point of {a.rcap:g} req/s per instance), overloaded {data['over']['rpi']:g} req/s per instance ({data['over']['rpi']/a.rcap:.2f} times).\n"
               f"\\item Runs: fixed horizon {m['horizon_s']:g} s, statistics over requests arriving in the window [{m['warmup_s']:g}, {m['horizon_s']-m['tail_s']:g}) s; {len(seeds)} seeds per point" + (f" ({esc(cut.split(':')[1])} seeds for n $\\ge$ {esc(cut.split(':')[0])})" if cut else "") + "; both policies use the same seeds (common random numbers). Bands are 95 percent Student t intervals over seeds.\n"
               f"\\item Revenue (model draft, eq. 10): per second of the window, the rewards $(\\pi_{{in}} P + \\pi_{{out}} o)/1000$ of the jobs completing in the window minus $h$ times the time-integral of the number of jobs in the system; $\\pi_{{in}} = {obj['type1']['pi_in']:g}$, $\\pi_{{out}} = {obj['type1']['pi_out']:g}$, $h = {obj['type1']['h']:g}$ per second for every type. Rejected jobs earn and cost nothing.\n"
               "\\item Latencies (E2E, TTFT) are over completed jobs that arrived in the window.\n"
               "\\end{itemize}")
    tex.append(r"\section{Strategies of the two policies}")
    tex.append(STRATEGY)
    tex.append("Fluid-dual inputs, computed before each run for the given n and arrival rate: the global optimum $x^\\star$ of the fluid problem (28), solved with Gurobi (spatial branch and bound, target gap 0.01 percent, time limit 600 s, started from the replicated solution of the next smaller n); the multipliers $\\nu^\\star, \\gamma^{M\\star}, \\gamma^{B\\star}$ from the KKT conditions (31) at $x^\\star$; the marginal congestion costs $c^\\star_{ij}$ (30); the fluid levels $K^\\star_j$ and $S^\\star_j$ (29); $\\eta = V^\\star/J$. The round-time constants of eq. (2) are fitted to BLIS's step-time function (H100: $\\tau_0$ = 12.1 ms, $\\tau_p$ = 4.4 $\\mu$s per prompt token, $\\tau_{kv}$ = 98 ns per cached token, $R^2$ above 0.9999). The job-type constants (22) use BLIS's actual output-length distribution.")
    tex.append(fluid_table(data))
    tex.append(r"\section{Revenue against the system scale}")
    tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{revenue_vs_scale.png}\\caption{Top: revenue value per second against the system scale n, left underloaded, right overloaded. Bottom: the same divided by n (revenue per second per instance), which makes small differences visible. Dashed: the fluid optimum $V^\\star$ of problem (28). Bands: 95 percent intervals over seeds.}\\end{figure}")
    tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{revenue_by_type.png}\\caption{Revenue value per second by job type.}\\end{figure}")
    tex.append(r"\section{Latency}")
    tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{latency_vs_scale.png}\\caption{Mean end-to-end latency, mean TTFT and p99 TTFT (log scale) against n, all job types.}\\end{figure}")
    tex.append(r"\section{Admission and congestion}")
    tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{admission_congestion_vs_scale.png}\\caption{Share of arrivals rejected at admission, share still unfinished at the horizon, evictions per arrival, and output tokens per second of completed jobs, against n.}\\end{figure}")
    tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{rejection_by_type.png}\\caption{Share of arrivals rejected by job type.}\\end{figure}")
    tex.append(r"\section{Result tables}")
    for load in ("under", "over"):
        tex.append(f"\\subsection{{{esc(data[load]['title'])}}}")
        tex.append(results_table(data, load))
        tex.append(type_table(data, load))
    tex.append(r"\section{Artifacts}")
    tex.append("\\begin{itemize}" + "".join(
        f"\\item {esc(data[l]['short'])}, {esc(POL[k][0])}: \\texttt{{{esc(os.path.relpath(data[l][k]['folder'], os.path.dirname(HERE)).replace(os.sep, '/'))}}}"
        for l in ("under", "over") for k in ("B1", "FD")) + "\\end{itemize}")
    tex.append(f"Fork commit: \\texttt{{{esc(m.get('commit', '')[:10])}}}. Generated by ours/report\\_fluid.py.")
    tex.append(r"\end{document}")
    texp = os.path.join(a.out, "report.tex")
    open(texp, "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out,
                           capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed; report.tex kept")
    print("wrote", os.path.join(a.out, "report.pdf"))


if __name__ == "__main__":
    main()
