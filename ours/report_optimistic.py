#!/usr/bin/env python3
"""ours/report_optimistic.py: the optimistic-dual test of the model draft (Section "An Optimistic
Benchmark and a Policy Test"): fluid-dual (Definition 1 on problem (28)), optimistic fluid-dual
(the same index rule on the optimistic program, eq. opt-feedback) and B2 (B1 + GAIE-legacy
admission), against the fluid optimum V* of (28) and the optimistic benchmark U_opt
(eq. opt-program), for n = 2 .. 64 at 11 and 14.4 req/s per instance.

    python ours/report_optimistic.py --out ours/experiments/<date>_report_optimistic

Every statistic is computed on the seeds the three policies share at that (load, n), from each
run's own window block (eq. 10 objective, time-average estimator). Paired differences use the
same seed (common arrivals and output lengths). House style: no em or en dashes.
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import objective  # noqa: E402
from report_robust import EXP, SPEC, esc, xstyle  # noqa: E402

LOADS = [("11", "underloaded, 11 req/s per instance (0.82 of fluid capacity)"),
         ("14.4", "overloaded, 14.4 req/s per instance (1.07 of fluid capacity)")]
POL = {  # key: (label, colour, marker, folder template)
    "FD": ("fluid-dual", "#C0504D", "o", "2026-10-11_scaling_scale1024_fluid_dual_kv6500_rpi{r}"),
    "OFD": ("optimistic fluid-dual", "#8064A2", "D", "2026-10-16_scaling_scale64_fluid_dual_optimistic_kv6500_rpi{r}"),
    "B2": ("B2 (B1 + GAIE admission)", "#2E75B6", "s", "2026-10-10_scaling_scale64_B2_kv6500_rpi{r}"),
}
TYPES = [("type1", "chat"), ("type2", "RAG"), ("type3", "long gen.")]
NMAX = 64


def runs(folder, params):
    """one row per run: n, seed, value, reward, cost, mean in system, and per-type admission."""
    path = os.path.join(EXP, folder)
    m = json.load(open(os.path.join(path, "manifest.json")))
    out = []
    for p in sorted(glob.glob(os.path.join(path, "runs", "*_s*.json.gz"))):
        tag = analyze.parse_tag(os.path.basename(p), m)
        if not tag or tag[0] > NMAX:
            continue
        n, _, seed = tag
        win = json.load(gzip.open(p, "rt"))["window"]
        rows = {r["type"]: r for r in objective.window_value(win, params)}
        a = rows["all"]
        row = {"n": n, "seed": seed, "value": a["value_per_s"], "reward": a["reward_per_s"], "cost": a["cost_per_s"],
               "wip": a["mean_in_system"]}
        arr = sum(win["types"].get(t, {}).get("arrived", 0) for t, _ in TYPES)
        row["preempt"] = sum(win["types"].get(t, {}).get("sum_preemptions", 0) for t, _ in TYPES) / arr if arr else np.nan
        row["wasted"] = sum(win["types"].get(t, {}).get("sum_wasted_tokens", 0) for t, _ in TYPES) / arr if arr else np.nan
        for t, _ in TYPES:
            s = win["types"].get(t, {})
            row[f"acc_{t}"] = 100 * (1 - s.get("rejected", 0) / s["arrived"]) if s.get("arrived") else np.nan
        out.append(row)
    return pd.DataFrame(out)


def benchmarks(r):
    """V* (problem (28), aggregate) from the fluid-dual folder, U_opt and its dual bound from the optimistic one."""
    out = {}
    for p in glob.glob(os.path.join(EXP, POL["FD"][3].format(r=r), "specs", "fluid_n*.yaml")):
        mt = yaml.safe_load(open(p))["meta"]
        if mt["instances"] <= NMAX:
            out.setdefault(mt["instances"], {})["V"] = mt["value"]
    for p in glob.glob(os.path.join(EXP, POL["OFD"][3].format(r=r), "specs", "fluid_n*.yaml")):
        d = yaml.safe_load(open(p))
        mt = d["meta"]
        b = out.setdefault(mt["instances"], {})
        b.update(U=mt["value"], U_bound=mt["bound"], U_status=mt["status"], V_check=mt["reference_value"],
                 kkt=mt["kkt_residual"], eta0=mt["eta0"],
                 gM=d["instances"][0]["gammaM"], cU={t["name"]: d["instances"][0]["types"][t["name"]]["c"] for t in d["types"]})
    return out


def ci(x):
    x = np.asarray(x, float)
    return analyze.t_crit(len(x)) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def collect():
    params = yaml.safe_load(open(os.path.join(SPEC, "objective.yaml")))
    rows, paired, missing = [], [], []
    for r, _ in LOADS:
        bm = benchmarks(r)
        data = {}
        for k, (_, _, _, tmpl) in POL.items():
            f = tmpl.format(r=r)
            data[k] = runs(f, params) if os.path.isdir(os.path.join(EXP, f, "runs")) else pd.DataFrame()
            if data[k].empty:
                missing.append(f"{POL[k][0]} at {r} req/s")
        if any(d.empty for d in data.values()):
            continue
        for n in sorted(set.intersection(*(set(d.n) for d in data.values()))):
            if n not in bm or "U" not in bm[n] or "V" not in bm[n]:
                continue
            seeds = sorted(set.intersection(*(set(d[d.n == n].seed) for d in data.values())))
            U, V = bm[n]["U"], bm[n]["V"]
            row = {"load": r, "n": int(n), "seeds": len(seeds), "U": U, "U_bound": bm[n]["U_bound"], "V": V}
            sub = {k: d[(d.n == n) & d.seed.isin(seeds)].set_index("seed").sort_index() for k, d in data.items()}
            for k, s in sub.items():
                row[k] = s.value.mean()
                row[f"{k}_ci"] = ci(s.value)
                row[f"{k}_gapU"] = 100 * (U - s.value.mean()) / U
                row[f"{k}_gapV"] = 100 * (V - s.value.mean()) / V
                for col in ("reward", "cost", "wip", "preempt", "wasted", *[f"acc_{t}" for t, _ in TYPES]):
                    row[f"{k}_{col}"] = s[col].mean()
            dlt = 100 * (sub["OFD"].value - sub["FD"].value) / U
            row["OFD_minus_FD"], row["OFD_minus_FD_ci"] = dlt.mean(), ci(dlt)
            dlt2 = 100 * (sub["OFD"].value - sub["B2"].value) / U
            row["OFD_minus_B2"], row["OFD_minus_B2_ci"] = dlt2.mean(), ci(dlt2)
            rows.append(row)
            paired.append(pd.DataFrame({"load": r, "n": int(n), "seed": seeds, "OFD_minus_FD_pct_of_U": dlt.values}))
    return pd.DataFrame(rows), (pd.concat(paired) if paired else pd.DataFrame()), missing


def fig_revenue(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    for ax, (r, label) in zip(axes, LOADS):
        g = df[df.load == r].sort_values("n")
        if g.empty:
            ax.set_axis_off()
            continue
        for k, (name, col, mk, _) in POL.items():
            y, e = g[k] / g.n, g[f"{k}_ci"].fillna(0) / g.n
            ax.plot(g.n, y, marker=mk, ms=5, lw=1.8, color=col, label=name)
            ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.18, lw=0)
        ax.plot(g.n, g.U / g.n, ls=":", lw=1.6, color="black", label="optimistic benchmark $U_{opt}/n$")
        ax.plot(g.n, g.V / g.n, ls="--", lw=1.1, color="grey", label="fluid optimum $V^*/n$ (problem 28)")
        xstyle(ax, list(g.n))
        ax.set_ylabel("revenue per second per instance")
        ax.set_title(label, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_gaps(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    for c, (r, label) in enumerate(LOADS):
        g = df[df.load == r].sort_values("n")
        for row, (suffix, bench) in enumerate((("gapU", "$U_{opt}$"), ("gapV", "$V^*$"))):
            ax = axes[row, c]
            if g.empty:
                ax.set_axis_off()
                continue
            for k, (name, col, mk, _) in POL.items():
                y = g[f"{k}_{suffix}"]
                e = 100 * g[f"{k}_ci"].fillna(0) / (g.U if suffix == "gapU" else g.V)
                ax.plot(g.n, y, marker=mk, ms=5, lw=1.8, color=col, label=name)
                ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.18, lw=0)
            ax.axhline(0, color="grey", lw=0.8)
            xstyle(ax, list(g.n))
            ax.set_ylabel(f"gap to {bench}, percent of {bench}")
            ax.set_title(f"{label.split(',')[0]}: relative gap to {bench}", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_paired(df, out):
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    for (r, label), col in zip(LOADS, ("#4C72B0", "#DD8452")):
        g = df[df.load == r].sort_values("n")
        if g.empty:
            continue
        ax.errorbar(g.n, g.OFD_minus_FD, yerr=g.OFD_minus_FD_ci.fillna(0), marker="o", ms=5, lw=1.6, capsize=3,
                    color=col, label=label.split(",")[0])
    ax.axhline(0, color="black", lw=0.9)
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("optimistic fluid-dual minus fluid-dual,\npercent of $U_{opt}$ (same seed)")
    ax.set_title("Paired difference on common seeds; bars 95 percent", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_diag(df, out):
    specs = [("reward", "gross revenue per s per instance", True), ("cost", "holding cost per s per instance", True),
             ("wip", "mean jobs in system per instance", True), ("wasted", "recomputed (wasted) tokens per arrival", False),
             ("acc_type3", "long-generation jobs accepted, percent", False), ("acc_type2", "RAG jobs accepted, percent", False)]
    fig, axes = plt.subplots(len(specs), 2, figsize=(13, 3.0 * len(specs)), dpi=140)
    for c, (r, label) in enumerate(LOADS):
        g = df[df.load == r].sort_values("n")
        for row, (col_, ylab, per_inst) in enumerate(specs):
            ax = axes[row, c]
            if g.empty:
                ax.set_axis_off()
                continue
            for k, (name, colr, mk, _) in POL.items():
                y = g[f"{k}_{col_}"] / (g.n if per_inst else 1)
                ax.plot(g.n, y, marker=mk, ms=4, lw=1.6, color=colr, label=name)
            xstyle(ax, list(g.n))
            ax.set_ylabel(ylab, fontsize=8)
            if row == 0:
                ax.set_title(label, fontsize=10)
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def f(v, d=2):
    return "--" if v is None or v != v else f"{v:,.{d}f}"


def table_bench(df):
    rows = []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            rows.append(" & ".join([r, str(x.n), f(x.V, 2), f(x.U, 2), f(x.U_bound, 2), f(100 * (x.U - x.V) / x.V, 3)]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{lrrrrr}\\toprule\n"
            r"req/s per inst. & $n$ & $V^\star$ (28) & $U_{opt}$ & dual bound of $U_{opt}$ & $U_{opt}$ above $V^\star$, \% \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def table_main(df):
    rows = []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            rows.append(" & ".join([r, str(x.n), str(x.seeds), f(x.U, 1), f(x.FD, 1), f(x.OFD, 1), f(x.B2, 1),
                                    f(x.FD_gapU), f(x.OFD_gapU), f(x.B2_gapU), f(x.OFD_minus_FD, 3), f(x.OFD_minus_FD_ci, 3)]) + r" \\")
    return ("\\begin{center}\\scriptsize\\begin{tabular}{lrrrrrrrrrrr}\\toprule\n"
            r"req/s & $n$ & seeds & $U_{opt}$ & FD & OFD & B2 & FD gap \% & OFD gap \% & B2 gap \% & OFD $-$ FD, \% of $U_{opt}$ & $\pm$ 95\% \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def table_diag(df):
    rows = []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            cells = [r, str(x.n)]
            for k in ("FD", "OFD", "B2"):
                cells += [f(x[f"{k}_reward"] / x.n, 3), f(x[f"{k}_cost"] / x.n, 3), f(x[f"{k}_wip"] / x.n, 1),
                          f(x[f"{k}_acc_type3"], 1), f(x[f"{k}_wasted"], 1)]
            rows.append(" & ".join(cells) + r" \\")
    head = r"req/s & $n$ & " + " & ".join(f"{k} rev. & {k} hold. & {k} WIP & {k} t3 acc. & {k} wasted" for k in ("FD", "OFD", "B2"))
    return ("\\begin{center}\\tiny\\begin{tabular}{lr" + "rrrrr" * 3 + "}\\toprule\n" + head + r" \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, paired, missing = collect()
    if df.empty:
        sys.exit("no complete (load, n) yet: " + ", ".join(missing))
    df.to_csv(os.path.join(a.out, "optimistic_test.csv"), index=False)
    paired.to_csv(os.path.join(a.out, "optimistic_test_paired_by_seed.csv"), index=False)
    fig_revenue(df, os.path.join(a.out, "opt_revenue.png"))
    fig_gaps(df, os.path.join(a.out, "opt_gaps.png"))
    fig_paired(df, os.path.join(a.out, "opt_paired.png"))
    fig_diag(df, os.path.join(a.out, "opt_diagnostics.png"))
    bm = {r: benchmarks(r) for r, _ in LOADS}
    prices = []
    for r, _ in LOADS:
        b = bm[r].get(8, {})
        if "gM" in b:
            prices.append(f"{r} req/s: $\\gamma^{{M,U}} = {b['gM']:.3f}$; $c^U$ = " + ", ".join(f"{v:.4f}" for v in b["cU"].values()))
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{The optimistic-dual test: fluid-dual, optimistic fluid-dual and B2 against $V^\star$ and $U_{opt}$}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{What was run}
Baseline instance (H100, 6,500 KV blocks, Qwen3-14B, type shares 50/35/15, $\pi_{in} = 1$, $\pi_{out} = 4$, $h = 0.03$), 11 and 14.4 req/s per instance, $n = 2, 4, 8, 16, 32, 64$, horizon 1,200 s, statistics over arrivals in [500, 1,100) s, revenue = eq. (10) per second of the window. Seeds 1 to 20 for $n \le 16$ and 1 to 10 for $n = 32, 64$, identical for the three policies (common arrivals and output lengths); every number below uses only the seeds the three share.
\begin{itemize}
\item \textbf{Fluid-dual (FD)}: Definition 1 on problem (28), solved by GPU type; $\eta = V^\star/J$.
\item \textbf{Optimistic fluid-dual (OFD)}: the same index rule built on the optimistic program (eq. opt-program): marginal charges $c^U_{ij}$ (opt-marginal), minimum-norm KKT multipliers, centers $K^U_j$, $Z^U_j$ (opt-targets) and the reference gain $\eta = V^\star/J$, as the draft prescribes. The program is concave; by symmetry of identical instances it is solved by GPU type with a rotated second-order cone, and Gurobi's barrier method proves it optimal with a dual bound.
\item \textbf{B2}: B1 (llm-d weighted routing) plus GAIE-legacy saturation admission with every type sheddable.
\end{itemize}
Benchmarks: $V^\star$, the optimum of problem (28); $U_{opt}$, the optimistic benchmark, which by the draft's lemma bounds the stochastic optimum from above under persistent bounded output. BLIS draws each job's output length once (a preempted job is recomputed with the same length) and clamps it at 2,048 / 2,048 / 4,096 tokens, so the finite coefficients are BLIS's own capped law. BLIS uses chunked prefill and a fitted affine step time, so by the draft's Section 4.4 these results are an implementation transfer test, not a certified benchmark claim.

Fluid quantities at $n = 8$ (identical per instance for every $n$): """ + "; ".join(prices) + r""". The optimistic and reference programs admit the same rates here ($x^U = x^\star$), so the centers coincide ($K^U_j = K^\star_j$, $Z^U_j = S^\star_j$); the policies differ only in the marginal charges and the multipliers.
"""]
    if missing:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(missing)) + ".\n")
    tex.append(r"\section{Benchmarks}" + table_bench(df))
    tex.append(r"""\section{Revenue and gaps}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{opt_revenue.png}\caption{Revenue per second per instance against $n$; dotted $U_{opt}/n$, dashed $V^\star/n$; bands 95 percent over seeds.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{opt_gaps.png}\caption{Relative gap $100\,(B - V)/B$ against $n$ for $B = U_{opt}$ (top) and $B = V^\star$ (bottom).}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=0.8\textwidth]{opt_paired.png}\caption{Is the optimistic policy better? Optimistic fluid-dual minus fluid-dual on the same seed, percent of $U_{opt}$.}\end{figure}
""" + table_main(df))
    tex.append(r"""\section{Diagnostics}
Gross revenue, holding cost and mean jobs in system per instance; acceptance of long-generation jobs; recomputed tokens per arrival (replay work after preemption).
\begin{figure}[H]\centering\includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{opt_diagnostics.png}\caption{Diagnostics against $n$ for the three policies.}\end{figure}
""" + table_diag(df))
    tex.append(r"""\section{Artifacts}
optimistic\_test.csv (one row per load and $n$), optimistic\_test\_paired\_by\_seed.csv (paired differences by seed). Folders: ours/experiments/2026-10-16\_scaling\_scale64\_fluid\_dual\_optimistic\_kv6500\_rpi\{11, 14.4\} (fluid files carry $U_{opt}$, its dual bound, $V^\star$, $x^U$, $c^U$, multipliers and centers), 2026-10-11\_scaling\_scale1024\_fluid\_dual\_kv6500\_rpi\{11, 14.4\}, 2026-10-10\_scaling\_scale64\_B2\_kv6500\_rpi\{11, 14.4\}. Generated by ours/report\_optimistic.py; solver ours/fluid/optimistic.py.
\end{document}""")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| missing:", missing)
    print(df[["load", "n", "seeds", "U", "V", "FD_gapU", "OFD_gapU", "B2_gapU", "OFD_minus_FD", "OFD_minus_FD_ci"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
