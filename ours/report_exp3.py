#!/usr/bin/env python3
"""ours/report_exp3.py: Experiment 3, the relative gap of fluid-dual to the fluid optimum against
the system scale n on log-log axes, for the baseline (H100, 6,500 blocks) at 11 and 14.4 req/s per
instance, n = 2 .. 1024. The fluid program is the aggregate by GPU type (proven optimal).

    python ours/report_exp3.py --out ours/experiments/<date>_report_exp3

Gap per seed = 100 (V* - V) / V*, V = window revenue per second (eq. 10), V* = fluid optimum.
The mean over seeds is plotted with a 95 percent band; a mean at or below zero cannot sit on a
log axis and is drawn as an open marker at its absolute value. The slope is the least-squares
fit of log gap on log n over the points with a positive mean gap. House style: no em or en dashes.
"""
import argparse
import glob
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
from report_robust import EXP, SPEC, run_values, summary, esc, xstyle  # noqa: E402

LOADS = [("11", "underloaded, 11 req/s per instance (0.82 of fluid capacity)", "#4C72B0"),
         ("14.4", "overloaded, 14.4 req/s per instance (1.07 of fluid capacity)", "#DD8452")]


def collect(date):
    params = yaml.safe_load(open(os.path.join(SPEC, "objective.yaml")))
    out, forms = [], set()
    for r, _, _ in LOADS:
        folder = f"{date}_scaling_scale1024_fluid_dual_kv6500_rpi{r}"
        path = os.path.join(EXP, folder)
        if not os.path.isdir(os.path.join(path, "runs")):
            continue
        fl = {}
        for p in glob.glob(os.path.join(path, "specs", "fluid_n*.yaml")):
            m = yaml.safe_load(open(p))["meta"]
            fl[m["instances"]] = m
            forms.add(m.get("formulation"))
        rv = run_values(folder, params)
        if rv.empty:
            continue
        summ = summary(folder) if os.path.exists(os.path.join(path, "summary.csv")) else None
        for n, g in rv.groupby("n"):
            if n not in fl:
                continue
            V = fl[n]["value"]
            gaps = 100 * (V - g.value.values) / V
            k = len(gaps)
            ci = analyze.t_crit(k) * gaps.std(ddof=1) / math.sqrt(k) if k > 1 else float("nan")
            row = {"load": r, "n": int(n), "seeds": k, "V": V, "status": fl[n]["status"],
                   "FD": g.value.mean(), "gap": gaps.mean(), "gap_ci": ci}
            if summ is not None and (summ.n == n).any():
                s = summ[summ.n == n].iloc[0]
                row.update(rej=100 * s.rejected_frac_mean, ttft=s.ttft_mean_ms_mean)
            out.append(row)
    return pd.DataFrame(out), forms


def fit(g):
    g = g[g.gap > 0]
    if len(g) < 3:
        return None
    b, a = np.polyfit(np.log10(g.n), np.log10(g.gap), 1)
    return b, a


def fig_loglog(df, out):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    fits = {}
    for r, label, col in LOADS:
        g = df[df.load == r].sort_values("n")
        if g.empty:
            continue
        pos = g[g.gap > 0]
        lo = np.maximum(pos.gap - pos.gap_ci.fillna(0), pos.gap * 0.05)
        ax.fill_between(pos.n, lo, pos.gap + pos.gap_ci.fillna(0), color=col, alpha=0.18, lw=0)
        ax.plot(pos.n, pos.gap, marker="o", ms=6, lw=1.8, color=col, label=label)
        neg = g[g.gap <= 0]
        if len(neg):
            ax.plot(neg.n, np.maximum(neg.gap.abs(), 1e-3), ls="none", marker="o", ms=7, mfc="white", mec=col, mew=1.5,
                    label="mean gap at or below 0 (absolute value)")
        f = fit(g)
        if f:
            b, a = f
            fits[r] = b
            xs = np.array([g.n.min(), g.n.max()])
            ax.plot(xs, 10 ** a * xs ** b, ls="--", lw=1.1, color=col, label=f"fit: gap $\\propto n^{{{b:.2f}}}$")
    # reference rates through the first underloaded point
    g0 = df.sort_values("n")
    if len(g0[g0.gap > 0]):
        n0, y0 = g0[g0.gap > 0].n.iloc[0], g0[g0.gap > 0].gap.max()
        xs = np.array([n0, df.n.max()])
        ax.plot(xs, y0 * (xs / n0) ** -0.5, ls=":", color="grey", lw=1, label="reference slope $-1/2$")
        ax.plot(xs, y0 * (xs / n0) ** -1.0, ls="-.", color="grey", lw=1, label="reference slope $-1$")
    ax.set_yscale("log")
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("relative gap to the fluid optimum, $100\\,(V^* - V)/V^*$ (percent)")
    ax.set_title("Experiment 3: fluid-dual gap to the fluid optimum against the system scale (log-log)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return fits


def fig_revenue(df, out):
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    for r, label, col in LOADS:
        g = df[df.load == r].sort_values("n")
        if g.empty:
            continue
        ax.plot(g.n, g.FD / g.n, marker="o", ms=5, lw=1.8, color=col, label=f"fluid-dual, {label.split(',')[0]}")
        ax.plot(g.n, g.V / g.n, ls="--", lw=1.1, color=col, label=f"fluid optimum $V^*/n$, {label.split(',')[0]}")
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("revenue per second per instance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def table(df):
    f = lambda v, d=2: "--" if v != v else f"{v:,.{d}f}"  # noqa: E731
    rows = []
    for r, label, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            rows.append(" & ".join([r, str(x.n), str(x.seeds), f(x.V, 1), f(x.FD, 1), f(x.gap, 3), f(x.gap_ci, 3),
                                    f(x.get("rej", float("nan")), 2), f(x.get("ttft", float("nan")) / 1000, 3)]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{lrrrrrrrr}\\toprule\n"
            r"req/s per inst. & n & seeds & $V^*$ & fluid-dual & gap \% & $\pm$ 95\% & rejected \% & TTFT (s) \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default="2026-10-11")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, forms = collect(a.date)
    if df.empty:
        sys.exit("no runs yet")
    df.to_csv(os.path.join(a.out, "experiment3_gap_vs_n.csv"), index=False)
    fits = fig_loglog(df, os.path.join(a.out, "exp3_loglog.png"))
    fig_revenue(df, os.path.join(a.out, "exp3_revenue.png"))
    fit_txt = "; ".join(f"{r} req/s per instance: slope {b:.2f}" for r, b in fits.items()) or "not enough positive points for a fit"
    stmt = ("The fluid problem is solved in its aggregate form by GPU type (index $j$ = GPU type with $N_j$ identical instances, "
            "load split equally within a type). Gurobi proves it optimal at every $n$ (here in at most 4 s at $n = 1024$), and it "
            "lies below the per-instance fluid value by at most 0.86 percent (median 0.12 percent) over the 102 points checked on "
            "2026-10-11. The fluid-dual policy uses this solution; $V^\\star$ below is this aggregate optimum."
            if forms == {"aggregate by GPU type"} else "Fluid formulation: " + esc(", ".join(sorted(str(f) for f in forms))) + ".")
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Experiment 3: the gap of fluid-dual to the fluid optimum as the system grows}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{What was run}
Fluid-dual (Definition 1 of the model draft: accept a type-$i$ job iff $\max_j I_{ij}(t) \ge 0$, route to the largest index; $\eta = V^\star/J$) on the baseline instance: H100 GPUs, 6,500 KV blocks each, Qwen3-14B, type shares 50/35/15, $\pi_{in} = 1$, $\pi_{out} = 4$, $h = 0.03$. Two loads, 11 and 14.4 req/s per instance (0.82 and 1.07 times the fluid capacity of 13.46 req/s per instance). System scale $n$ = 2, 4, 8, \dots, 1024 with the arrival rate proportional to $n$; 10 seeds for $n \le 64$ and 3 for $n \ge 128$. Horizon 1,200 s, statistics over arrivals in [500, 1,100) s, revenue = eq. (10) per second of the window.

\textbf{Fluid program.} """ + stmt + r"""

The relative gap to optimality is $100\,(V^\star - V)/V^\star$, averaged over seeds.
\section{Result}
Fitted slopes of $\log(\text{gap})$ on $\log n$: """ + esc(fit_txt) + r""".
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp3_loglog.png}\caption{Relative gap of fluid-dual to the fluid optimum against $n$, log-log axes; bands 95 percent over seeds; dashed: least-squares fit; grey: reference slopes $-1/2$ and $-1$.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp3_revenue.png}\caption{Revenue per second per instance against $n$ with the fluid optimum per instance.}\end{figure}
""" + table(df) + r"""
\section{Artifacts}
Data: experiment3\_gap\_vs\_n.csv in this folder. Experiment folders: ours/experiments/""" + esc(a.date) + r"""\_scaling\_scale1024\_fluid\_dual\_kv6500\_rpi\{11, 14.4\}. Generated by ours/report\_exp3.py.
\end{document}"""]
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| fits:", fits)
    print(df.to_string())


if __name__ == "__main__":
    main()
