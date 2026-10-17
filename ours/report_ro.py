#!/usr/bin/env python3
"""ours/report_ro.py: RO-dual (the index policy built from the Refined Optimistic program, draft Section
"Refined Optimistic Fluid Program and Its Test Policy") against fluid-dual (Definition 1 on problem
(28)), in the setting of the optimistic test: n = 2 .. 64 at 11 and 14.4 req/s per instance.

    python ours/report_ro.py --out ours/experiments/<date>_report_ro_dual

Benchmarks: V* (problem 28) and the certified U_RO (eq. ro-program, with Gurobi's dual bound); the
processing-only optimistic value U_opt is listed for reference. All statistics use the seeds both arms
share; paired differences use the same seed. Also reported, per the draft's protocol: completion
revenue, holding cost, admission by type, replay work, time-average WIP, WIP growth (unfinished jobs at
the horizon) and terminal unfinished jobs. House style: no em or en dashes.
"""
import argparse
import glob
import gzip
import json
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
from report_optimistic import runs, ci, f, TYPES  # noqa: E402
from report_robust import EXP, SPEC, esc, xstyle  # noqa: E402

LOADS = [("11", "underloaded, 11 req/s per instance (0.82 of fluid capacity)"),
         ("14.4", "overloaded, 14.4 req/s per instance (1.07 of fluid capacity)")]
POL = {"FD": ("fluid-dual (problem 28)", "#C0504D", "o", "2026-10-11_scaling_scale1024_fluid_dual_kv6500_rpi{r}"),
       "RO": ("RO-dual (Refined Optimistic)", "#17BECF", "P", "2026-10-17_scaling_scale64_fluid_dual_ro_kv6500_rpi{r}")}
NMAX = 64


def terminal(folder):
    """per (n, seed): unfinished jobs at the horizon (still queued + still running)."""
    path = os.path.join(EXP, folder)
    m = json.load(open(os.path.join(path, "manifest.json")))
    out = {}
    for p in glob.glob(os.path.join(path, "runs", "*_s*.json.gz")):
        tag = analyze.parse_tag(os.path.basename(p), m)
        if tag and tag[0] <= NMAX:
            d = json.load(gzip.open(p, "rt"))
            out[(tag[0], tag[2])] = d["still_queued"] + d["still_running"]
    return out


def bench(r):
    out = {}
    for p in glob.glob(os.path.join(EXP, POL["FD"][3].format(r=r), "specs", "fluid_n*.yaml")):
        mt = yaml.safe_load(open(p))["meta"]
        if mt["instances"] <= NMAX:
            out.setdefault(mt["instances"], {})["V"] = mt["value"]
    for p in glob.glob(os.path.join(EXP, POL["RO"][3].format(r=r), "specs", "fluid_n*.yaml")):
        d = yaml.safe_load(open(p))
        mt = d["meta"]
        out.setdefault(mt["instances"], {}).update(
            U=mt["value"], U_bound=mt["bound"], U_status=mt["status"], U_opt=mt.get("optimistic_processing_only_value"),
            gM=d["instances"][0]["gammaM"], cRO={t["name"]: d["instances"][0]["types"][t["name"]]["c"] for t in d["types"]},
            K=d["instances"][0]["kv_target"], Z=d["instances"][0]["stage0_target"])
    return out


def collect():
    params = yaml.safe_load(open(os.path.join(SPEC, "objective.yaml")))
    rows, missing = [], []
    for r, _ in LOADS:
        bm = bench(r)
        data, term = {}, {}
        for k, (_, _, _, tmpl) in POL.items():
            fo = tmpl.format(r=r)
            if os.path.isdir(os.path.join(EXP, fo, "runs")):
                data[k], term[k] = runs(fo, params), terminal(fo)
            if k not in data or data[k].empty:
                missing.append(f"{POL[k][0]} at {r} req/s")
        if len(data) < 2 or any(d.empty for d in data.values()):
            continue
        for n in sorted(set(data["FD"].n) & set(data["RO"].n)):
            if n not in bm or "U" not in bm[n]:
                continue
            seeds = sorted(set(data["FD"][data["FD"].n == n].seed) & set(data["RO"][data["RO"].n == n].seed))
            U, V = bm[n]["U"], bm[n]["V"]
            row = {"load": r, "n": int(n), "seeds": len(seeds), "U": U, "U_bound": bm[n]["U_bound"], "U_opt": bm[n]["U_opt"], "V": V}
            sub = {k: d[(d.n == n) & d.seed.isin(seeds)].set_index("seed").sort_index() for k, d in data.items()}
            for k, s in sub.items():
                row[k], row[f"{k}_ci"] = s.value.mean(), ci(s.value)
                row[f"{k}_gapU"] = 100 * (U - s.value.mean()) / U
                row[f"{k}_gapV"] = 100 * (V - s.value.mean()) / V
                for col in ("reward", "cost", "wip", "preempt", "wasted", *[f"acc_{t}" for t, _ in TYPES]):
                    row[f"{k}_{col}"] = s[col].mean()
                row[f"{k}_terminal"] = float(np.mean([term[k][(n, sd)] for sd in seeds]))
            dlt = 100 * (sub["RO"].value - sub["FD"].value) / U
            row["RO_minus_FD"], row["RO_minus_FD_ci"] = dlt.mean(), ci(dlt)
            rows.append(row)
    return pd.DataFrame(rows), missing


def figs(df, out_rev, out_pair):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    for c, (r, label) in enumerate(LOADS):
        g = df[df.load == r].sort_values("n")
        ax = axes[0, c]
        if g.empty:
            ax.set_axis_off(); axes[1, c].set_axis_off()
            continue
        for k, (name, col, mk, _) in POL.items():
            y, e = g[k] / g.n, g[f"{k}_ci"].fillna(0) / g.n
            ax.plot(g.n, y, marker=mk, ms=5, lw=1.7, color=col, label=name)
            ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.18, lw=0)
        ax.plot(g.n, g.U / g.n, ls=":", lw=1.6, color="black", label="$U_{RO}/n$ (certified)")
        ax.plot(g.n, g.V / g.n, ls="--", lw=1.1, color="grey", label="$V^*/n$ (problem 28)")
        xstyle(ax, list(g.n)); ax.set_ylabel("revenue per s per instance"); ax.set_title(label, fontsize=10); ax.legend(frameon=False, fontsize=8)
        ax = axes[1, c]
        for k, (name, col, mk, _) in POL.items():
            ax.plot(g.n, g[f"{k}_gapU"], marker=mk, ms=5, lw=1.7, color=col, label=name)
        ax.axhline(0, color="grey", lw=0.8)
        xstyle(ax, list(g.n)); ax.set_ylabel("gap to $U_{RO}$, percent"); ax.set_title(label.split(",")[0] + ": relative gap to $U_{RO}$", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_rev); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    for (r, label), col in zip(LOADS, ("#4C72B0", "#DD8452")):
        g = df[df.load == r].sort_values("n")
        if g.empty:
            continue
        ax.errorbar(g.n, g.RO_minus_FD, yerr=g.RO_minus_FD_ci.fillna(0), marker="o", ms=5, lw=1.6, capsize=3, color=col, label=label.split(",")[0])
    ax.axhline(0, color="black", lw=0.9)
    xstyle(ax, sorted(set(df.n)))
    ax.set_ylabel("RO-dual minus fluid-dual,\npercent of $U_{RO}$ (same seed)")
    ax.set_title("Paired difference on common seeds; bars 95 percent", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_pair); plt.close(fig)


def tables(df):
    b, m, d = [], [], []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            b.append(" & ".join([r, str(x.n), f(x.V, 2), f(x.U, 2), f(x.U_bound, 2), f(x.U_opt, 2)]) + r" \\")
            m.append(" & ".join([r, str(x.n), str(x.seeds), f(x.FD, 1), f(x.RO, 1), f(x.FD_gapU), f(x.RO_gapU), f(x.FD_gapV), f(x.RO_gapV),
                                 f"{f(x.RO_minus_FD, 3)} $\\pm$ {f(x.RO_minus_FD_ci, 3)}"]) + r" \\")
            for k in ("FD", "RO"):
                d.append(" & ".join([r, str(x.n), k, f(x[f"{k}_reward"] / x.n, 3), f(x[f"{k}_cost"] / x.n, 3), f(x[f"{k}_wip"] / x.n, 1),
                                     "/".join(f(x[f"{k}_acc_{t}"], 1) for t, _ in TYPES), f(x[f"{k}_wasted"], 1),
                                     f(x[f"{k}_terminal"] / x.n, 1)]) + r" \\")
    tb = ("\\begin{center}\\small\\begin{tabular}{lrrrrr}\\toprule\n"
          r"req/s & $n$ & $V^\star$ (28) & $U_{RO}$ & dual bound of $U_{RO}$ & $U_{opt}$ (processing only, reference) \\ \midrule" "\n" + "\n".join(b) + "\n\\bottomrule\\end{tabular}\\end{center}")
    tm = ("\\begin{center}\\small\\begin{tabular}{lrrrrrrrrr}\\toprule\n"
          r"req/s & $n$ & seeds & FD & RO-dual & FD gap to $U_{RO}$ \% & RO gap \% & FD gap to $V^\star$ \% & RO gap \% & RO $-$ FD, \% of $U_{RO}$ \\ \midrule" "\n" + "\n".join(m) + "\n\\bottomrule\\end{tabular}\\end{center}")
    td = ("{\\small\\begin{longtable}{lrlrrrrrr}\\toprule\n"
          r"req/s & $n$ & arm & revenue / s / inst. & holding / s / inst. & WIP / inst. & accepted \% (types 1/2/3) & replay tok. / arrival & unfinished at horizon / inst. \\ \midrule\endhead" "\n" + "\n".join(d) + "\n\\bottomrule\\end{longtable}}")
    return tb, tm, td


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, missing = collect()
    if df.empty:
        sys.exit("nothing to report yet: " + ", ".join(missing))
    df.to_csv(os.path.join(a.out, "ro_dual_test.csv"), index=False)
    figs(df, os.path.join(a.out, "ro_revenue_gaps.png"), os.path.join(a.out, "ro_paired.png"))
    tb, tm, td = tables(df)
    inputs = []
    for r, _ in LOADS:
        x = bench(r).get(8, {})
        if "gM" in x:
            inputs.append(f"{r} req/s: $\\gamma^{{M,\\rm RO}} = {x['gM']:.4f}$, $c^{{\\rm RO}}$ = " + ", ".join(f"{v:.4f}" for v in x["cRO"].values())
                          + f", $K^{{\\rm RO}} = {x['K']:.0f}$, $Z^{{\\rm RO}} = {x['Z']:.1f}$")
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float,longtable}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{RO-dual against fluid-dual}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{What was run}
Two arms in the setting of the optimistic test: baseline instance (H100, 6,500 KV blocks, Qwen3-14B, type shares 50/35/15, $\pi_{in} = 1$, $\pi_{out} = 4$, $h = 0.03$), 11 and 14.4 req/s per instance, $n = 2, 4, 8, 16, 32, 64$, horizon 1,200 s, statistics over arrivals in [500, 1,100) s, revenue = eq. (10) per second of the window. Seeds 1 to 20 for $n \le 16$ and 1 to 10 for $n = 32, 64$, identical across arms (same arrivals and once-sampled output lengths); statistics use only the seeds both arms share.
\begin{itemize}
\item \textbf{Fluid-dual}: Definition 1 on problem (28), solved by GPU type; gain $\eta = V^\star/J$.
\item \textbf{RO-dual}: the same index policy built from the Refined Optimistic program (eq. ro-program), with holding charge $H^{\rm RO}_j = \tau_0 A_j + \tau_0 (G_j^2 + F_j)/(1 - \rho_j)$, $F_j = \sum_i h_i x_{ij}^2/(2\lambda_i)$ ($\lambda_i$ external); marginal charge (ro-marginal) with both waiting terms; minimum-norm KKT multipliers; centers (ro-centers) at $x^{\rm RO}$; the same gain $\eta = V^\star/J$ (the baseline's gain, as the protocol prescribes). Solved by GPU type (the equal split of identical instances is optimal by concavity) as a second-order cone program; Gurobi's barrier certifies optimality at every point with a primal-dual gap of about $10^{-9}$.
\end{itemize}
Batching and eviction are BLIS's vLLM rules in both arms. BLIS draws each output length once (persistent across eviction) and clamps it, but uses chunked prefill and a fitted step time, so this is an implementation transfer test in the sense of the draft's protocol. RO-dual inputs at $n = 8$ (identical per instance for every $n$): """ + "; ".join(inputs) + r""".
"""]
    if missing:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(missing)) + ".\n")
    tex.append(r"\section{Benchmarks}" + tb)
    tex.append(r"""\section{Results}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{ro_revenue_gaps.png}\caption{Top: revenue per second per instance with $U_{RO}/n$ and $V^\star/n$; bands 95 percent over seeds. Bottom: relative gap to $U_{RO}$.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=0.8\textwidth]{ro_paired.png}\caption{RO-dual minus fluid-dual on the same seed, percent of $U_{RO}$.}\end{figure}
""" + tm)
    tex.append(r"\section{Diagnostics}" + td)
    tex.append(r"""\section{Artifacts}
ro\_dual\_test.csv in this folder. Folders ours/experiments/2026-10-17\_scaling\_scale64\_fluid\_dual\_ro\_kv6500\_rpi\{11, 14.4\} (fluid files carry $U_{RO}$, its dual bound, $V^\star$, $U_{opt}$, $x^{\rm RO}$, $c^{\rm RO}$, multipliers, centers, gain and solver settings) and 2026-10-11\_scaling\_scale1024\_fluid\_dual\_kv6500\_rpi\{11, 14.4\}. Generated by ours/report\_ro.py; solver ours/fluid/optimistic.py (refined) via solve.py --refined.
\end{document}""")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        rr = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if rr.returncode != 0:
        print(rr.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| missing:", missing)
    print(df[["load", "n", "seeds", "V", "U", "FD_gapU", "RO_gapU", "RO_minus_FD", "RO_minus_FD_ci"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
