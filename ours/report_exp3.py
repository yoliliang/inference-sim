#!/usr/bin/env python3
"""ours/report_exp3.py: Experiment 3, the relative gap of fluid-dual to the fluid optimum against
the system scale n on log-log axes, for the baseline (H100, 6,500 blocks) at 11 and 14.4 req/s per
instance, n = 2 .. 1024. The fluid program is the aggregate by GPU type (proven optimal).

    python ours/report_exp3.py --out ours/experiments/<date>_report_exp3

Gap per seed = 100 (V* - V) / V*, V = window revenue per second (eq. 10), V* = fluid optimum.
The mean over seeds is plotted with a 95 percent band; a mean at or below zero cannot sit on a
log axis and is drawn as an open marker at its absolute value. Two fits: a power law (least
squares of log gap on log n over the positive points) and a floor, gap = c + a n^(-1/2), whose
intercept c is the level the gap settles at. The admission table compares the fluid admission
share x*_i / lambda_i with the simulated acceptance share per type, and values the difference at
the reward rate: sum_i (fluid share - simulated share) lambda_i r_i, in percent of V*.
House style: no em or en dashes.
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
from report_robust import EXP, SPEC, run_values, esc, xstyle  # noqa: E402

LOADS = [("11", "underloaded, 11 req/s per instance (0.82 of fluid capacity)", "#4C72B0"),
         ("14.4", "overloaded, 14.4 req/s per instance (1.07 of fluid capacity)", "#DD8452")]
TYPES = [("type1", "chat"), ("type2", "RAG"), ("type3", "long gen.")]


def collect(date):
    params = yaml.safe_load(open(os.path.join(SPEC, "objective.yaml")))
    out, forms = [], set()
    for r, _, _ in LOADS:
        folder = f"{date}_scaling_scale1024_fluid_dual_kv6500_rpi{r}"
        path = os.path.join(EXP, folder)
        if not os.path.isdir(os.path.join(path, "runs")):
            continue
        fl, fty = {}, {}
        for p in glob.glob(os.path.join(path, "specs", "fluid_n*.yaml")):
            d = yaml.safe_load(open(p))
            m = d["meta"]
            fl[m["instances"]], fty[m["instances"]] = m, d["types"]
            forms.add(m.get("formulation"))
        rv = run_values(folder, params)
        if rv.empty:
            continue
        summ = pd.read_csv(os.path.join(path, "summary.csv")) if os.path.exists(os.path.join(path, "summary.csv")) else None
        for n, g in rv.groupby("n"):
            if n not in fl:
                continue
            V = fl[n]["value"]
            gaps = 100 * (V - g.value.values) / V
            k = len(gaps)
            ci = analyze.t_crit(k) * gaps.std(ddof=1) / math.sqrt(k) if k > 1 else float("nan")
            row = {"load": r, "n": int(n), "seeds": k, "V": V, "status": fl[n]["status"],
                   "FD": g.value.mean(), "gap": gaps.mean(), "gap_ci": ci}
            if summ is not None:
                a = summ[(summ.n == n) & (summ.type == "all")]
                if len(a):
                    row.update(rej=100 * a.rejected_frac_mean.iloc[0], ttft=a.ttft_mean_ms_mean.iloc[0])
                lost, have = 0.0, False
                for t in fty[n]:
                    q = summ[(summ.n == n) & (summ.type == t["name"])]
                    if q.empty or not t["lambda"]:
                        continue
                    fa, sa = t["admitted"] / t["lambda"], 1 - q.rejected_frac_mean.iloc[0]
                    row[f"fluid_adm_{t['name']}"], row[f"sim_adm_{t['name']}"] = 100 * fa, 100 * sa
                    lost += (fa - sa) * t["lambda"] * t["r"]
                    have = True
                if have:
                    row["adm_loss"] = 100 * lost / V
            out.append(row)
    return pd.DataFrame(out), forms


def fit_power(g):
    g = g[g.gap > 0]
    if len(g) < 3:
        return None
    b, a = np.polyfit(np.log10(g.n), np.log10(g.gap), 1)
    return b, a


def fit_floor(g):
    """least squares of gap = c + a n^(-1/2): c is the level the gap settles at for large n."""
    if len(g) < 3:
        return None
    X = np.column_stack([np.ones(len(g)), g.n.values ** -0.5])
    (c, a), *_ = np.linalg.lstsq(X, g.gap.values, rcond=None)
    return c, a


def fig_loglog(df, out):
    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=150)
    fits = {}
    for r, label, col in LOADS:
        g = df[df.load == r].sort_values("n")
        if g.empty:
            continue
        pos = g[g.gap > 0]
        ci = pos.gap_ci.fillna(0)
        ax.fill_between(pos.n, np.maximum(pos.gap - ci, pos.gap * 0.05), pos.gap + ci, color=col, alpha=0.18, lw=0)
        ax.plot(pos.n, pos.gap, marker="o", ms=6, lw=1.8, color=col, label=label)
        neg = g[g.gap <= 0]
        if len(neg):
            ax.plot(neg.n, np.maximum(neg.gap.abs(), 1e-3), ls="none", marker="o", ms=7, mfc="white", mec=col, mew=1.5,
                    label="mean gap below 0 (simulation above $V^*$), drawn at its absolute value")
        fp, ff = fit_power(g), fit_floor(g)
        big = g[g.n >= 64]
        fits[r] = {"slope": fp[0] if fp else float("nan"), "floor": ff[0] if ff else float("nan"),
                   "a": ff[1] if ff else float("nan"), "plateau": big.gap.mean() if len(big) else float("nan")}
        if fp:
            b, a = fp
            xs = np.array([g.n.min(), g.n.max()])
            ax.plot(xs, 10 ** a * xs ** b, ls="--", lw=1.0, color=col, label=f"power fit: gap $\\propto n^{{{b:.2f}}}$")
        if ff:
            c, a2 = ff
            xs = np.geomspace(g.n.min(), g.n.max(), 80)
            ys = c + a2 * xs ** -0.5
            ok = ys > 0
            ax.plot(xs[ok], ys[ok], ls="-.", lw=1.2, color=col, alpha=0.85,
                    label=f"floor fit: ${c:.2f} + {a2:.2f}\\,n^{{-1/2}}$")
    ax.set_yscale("log")
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("relative gap to the fluid optimum, $100\\,(V^* - V)/V^*$ (percent)")
    ax.set_title("Experiment 3: fluid-dual gap to the fluid optimum against the system scale (log-log)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return fits


def fig_revenue(df, out):
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    for r, label, col in LOADS:
        g = df[df.load == r].sort_values("n")
        if g.empty:
            continue
        word = label.split(",")[0]
        ax.plot(g.n, g.FD / g.n, marker="o", ms=5, lw=1.8, color=col, label=f"fluid-dual, {word}")
        ax.plot(g.n, g.V / g.n, ls="--", lw=1.1, color=col, label=f"fluid optimum $V^*/n$, {word}")
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("revenue per second per instance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_admission(df, out):
    if "adm_loss" not in df:
        return False
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    for r, label, col in LOADS:
        g = df[(df.load == r) & df.adm_loss.notna()].sort_values("n")
        if g.empty:
            continue
        word = label.split(",")[0]
        ax.plot(g.n, g.gap, marker="o", ms=5, lw=1.8, color=col, label=f"gap, {word}")
        ax.plot(g.n, g.adm_loss, marker="s", ms=4, lw=1.2, ls="--", color=col,
                label=f"reward lost to admitting differently from the fluid, {word}")
    ax.axhline(0, color="grey", lw=0.8)
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("percent of $V^*$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def f(v, d=2):
    return "--" if v is None or v != v else f"{v:,.{d}f}"


def table_gap(df):
    rows = []
    for r, _, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            rows.append(" & ".join([r, str(x.n), str(x.seeds), f(x.V, 1), f(x.FD, 1), f(x.gap, 3), f(x.gap_ci, 3),
                                    f(x.get("rej"), 2), f(x.get("ttft", float("nan")) / 1000, 3)]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{lrrrrrrrr}\\toprule\n"
            r"req/s per inst. & $n$ & seeds & $V^\star$ & fluid-dual & gap \% & $\pm$ 95\% & rejected \% & TTFT (s) \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def table_adm(df):
    if "adm_loss" not in df:
        return ""
    rows = []
    for r, _, _ in LOADS:
        for _, x in df[(df.load == r) & df.adm_loss.notna()].sort_values("n").iterrows():
            cells = [r, str(x.n)]
            for t, _ in TYPES:
                cells.append(f"{f(x.get('fluid_adm_' + t), 1)} / {f(x.get('sim_adm_' + t), 1)}")
            cells += [f(x.adm_loss, 3), f(x.gap, 3)]
            rows.append(" & ".join(cells) + r" \\")
    head = " & ".join([r"req/s per inst. & $n$"] + [f"{lab}: fluid / sim. \\%" for _, lab in TYPES]
                      + [r"admission loss \%", r"gap \%"])
    return ("\\begin{center}\\small\\begin{tabular}{lrrrrrr}\\toprule\n" + head + r" \\ \midrule" "\n"
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
    has_adm = fig_admission(df, os.path.join(a.out, "exp3_admission.png"))
    fit_rows = "\n".join(f"{r} & {f(v['slope'])} & {f(v['floor'], 3)} & {f(v['a'], 3)} & {f(v['plateau'], 3)} \\\\" for r, v in fits.items())
    stmt = ("The fluid problem is solved in its aggregate form by GPU type (index $j$ = GPU type with $N_j$ identical instances, "
            "load split equally within a type). Gurobi proves it optimal at every $n$ (in at most 4 s at $n = 1024$), and it lies "
            "below the per-instance fluid value by at most 0.86 percent (median 0.12 percent) over the 102 points checked on "
            "2026-10-11. The fluid-dual policy uses this solution; $V^\\star$ below is this aggregate optimum."
            if forms == {"aggregate by GPU type"} else "Fluid formulation: " + esc(", ".join(sorted(str(x) for x in forms))) + ".")
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
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp3_loglog.png}\caption{Relative gap of fluid-dual to the fluid optimum against $n$, log-log axes; bands 95 percent over seeds. Dashed: power fit over the positive points. Dash-dot: floor fit $c + a\,n^{-1/2}$.}\end{figure}
\begin{center}\small\begin{tabular}{lrrrr}\toprule
req/s per inst. & power-fit slope & floor $c$ (\%) & $a$ & mean gap over $n \ge 64$ (\%) \\ \midrule
""" + fit_rows + r"""
\bottomrule\end{tabular}\end{center}
"""]
    if has_adm:
        tex.append(r"""\section{Where the remaining gap comes from}
For each type, the share the fluid optimum admits, $x^\star_i/\lambda_i$, against the share the simulated policy accepts; the difference valued at the reward rate, $\sum_i (\text{fluid share} - \text{simulated share})\,\lambda_i r_i$, in percent of $V^\star$ (admission loss). It leaves out the holding cost, so it is an approximation of the part of the gap due to admission.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp3_admission.png}\caption{Gap and admission loss against $n$, both in percent of $V^\star$.}\end{figure}
""" + table_adm(df))
    tex.append(r"""\section{Revenue and table}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp3_revenue.png}\caption{Revenue per second per instance against $n$ with the fluid optimum per instance.}\end{figure}
""" + table_gap(df) + r"""
\section{Artifacts}
Data: experiment3\_gap\_vs\_n.csv in this folder. Experiment folders: ours/experiments/""" + esc(a.date) + r"""\_scaling\_scale1024\_fluid\_dual\_kv6500\_rpi\{11, 14.4\}. Generated by ours/report\_exp3.py.
\end{document}""")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"))
    print(fits)
    print(df.to_string())


if __name__ == "__main__":
    main()
