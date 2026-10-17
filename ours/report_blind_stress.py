#!/usr/bin/env python3
"""ours/report_blind_stress.py: stress test of fluid-dual blind (problem (28) with every holding cost
zero) against fluid-dual, in settings where the congestion cost changes the fluid rates.

    python ours/report_blind_stress.py --out ours/experiments/<date>_report_blind_stress

Settings: S3 h = 0.5 with 6,500 KV blocks (holding cost dominates), S1 h = 0.2 with 6,500 blocks,
S2 h = 0.1 with 15,909 blocks (memory not binding); two loads each (0.817 and 1.070 of the fluid
capacity); n = 2 .. 64. Each setting's runs are valued with that setting's objective file. All
statistics use the seeds both policies share; paired differences use the same seed. The baseline
(h = 0.03, 6,500 blocks) from 2026-10-16 is included for reference. House style: no em or en dashes.
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

TYPES = [("type1", "chat"), ("type2", "RAG"), ("type3", "long gen.")]
SETTINGS = [  # key, label, objective file, loads (rate per instance), folder template {pol}, {r}
    ("base", "baseline: $h = 0.03$, 6,500 blocks", "objective.yaml", ("11", "14.4"),
     {"FD": "2026-10-11_scaling_scale1024_fluid_dual_kv6500_rpi{r}", "BL": "2026-10-16_scaling_scale64_fluid_dual_blind_kv6500_rpi{r}"}),
    ("S3", "S3: $h = 0.5$, 6,500 blocks (holding cost dominates)", "objective_h0.5.yaml", ("11", "14.4"),
     {"FD": "2026-10-16_scaling_blindstress_h0.5_kv6500_fluid_dual_rpi{r}", "BL": "2026-10-16_scaling_blindstress_h0.5_kv6500_fluid_dual_blind_rpi{r}"}),
    ("S1", "S1: $h = 0.2$, 6,500 blocks", "objective_h0.2.yaml", ("11", "14.4"),
     {"FD": "2026-10-16_scaling_blindstress_h0.2_kv6500_fluid_dual_rpi{r}", "BL": "2026-10-16_scaling_blindstress_h0.2_kv6500_fluid_dual_blind_rpi{r}"}),
    ("S2", "S2: $h = 0.1$, 15,909 blocks (memory not binding)", "objective_h0.1.yaml", ("15.793", "20.6745"),
     {"FD": "2026-10-16_scaling_blindstress_h0.1_kv15909_fluid_dual_rpi{r}", "BL": "2026-10-16_scaling_blindstress_h0.1_kv15909_fluid_dual_blind_rpi{r}"}),
]
POL = {"FD": ("fluid-dual", "#C0504D", "o"), "BL": ("fluid-dual blind (h = 0 in the fluid problem)", "#E69F00", "v")}
NMAX = 64


def runs(folder, params):
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
        for t, _ in TYPES:
            s = win["types"].get(t, {})
            row[f"acc_{t}"] = 100 * (1 - s.get("rejected", 0) / s["arrived"]) if s.get("arrived") else np.nan
        out.append(row)
    return pd.DataFrame(out)


def fluid(folder):
    """per n: V* of the true problem, the fluid admission shares, and (blind folders) the blind LP value."""
    out = {}
    for p in glob.glob(os.path.join(EXP, folder, "specs", "fluid_n*.yaml")):
        d = yaml.safe_load(open(p))
        mt = d["meta"]
        out[mt["instances"]] = {"V": mt.get("reference_value", mt["value"]), "blind_value": mt.get("blind_value"),
                                **{f"fl_{t['name']}": 100 * t["admitted"] / t["lambda"] for t in d["types"]}}
    return out


def ci(x):
    x = np.asarray(x, float)
    return analyze.t_crit(len(x)) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def collect():
    rows, missing = [], []
    for key, label, objf, loads, tmpl in SETTINGS:
        params = yaml.safe_load(open(os.path.join(SPEC, objf)))
        for li, r in enumerate(loads):
            data, fl = {}, {}
            for k in POL:
                f = tmpl[k].format(r=r)
                if os.path.isdir(os.path.join(EXP, f, "runs")):
                    d = runs(f, params)
                    if not d.empty:
                        data[k], fl[k] = d, fluid(f)
            if len(data) < 2:
                missing.append(f"{key} {r} req/s")
                continue
            for n in sorted(set(data["FD"].n) & set(data["BL"].n)):
                seeds = sorted(set(data["FD"][data["FD"].n == n].seed) & set(data["BL"][data["BL"].n == n].seed))
                if not seeds or n not in fl["FD"]:
                    continue
                V = fl["FD"][n]["V"]
                row = {"setting": key, "label": label, "load": "under" if li == 0 else "over", "rate": r, "n": int(n),
                       "seeds": len(seeds), "V": V, "blind_lp": (fl["BL"].get(n) or {}).get("blind_value")}
                for t, _ in TYPES:
                    row[f"FD_fl_{t}"] = fl["FD"][n][f"fl_{t}"]
                    row[f"BL_fl_{t}"] = (fl["BL"].get(n) or {}).get(f"fl_{t}")
                sub = {k: data[k][(data[k].n == n) & data[k].seed.isin(seeds)].set_index("seed").sort_index() for k in POL}
                for k, s in sub.items():
                    row[k], row[f"{k}_ci"] = s.value.mean(), ci(s.value)
                    row[f"{k}_gapV"] = 100 * (V - s.value.mean()) / abs(V)
                    for col in ("reward", "cost", "wip", "preempt", *[f"acc_{t}" for t, _ in TYPES]):
                        row[f"{k}_{col}"] = s[col].mean()
                dlt = 100 * (sub["BL"].value - sub["FD"].value) / abs(V)
                row["BL_minus_FD"], row["BL_minus_FD_ci"] = dlt.mean(), ci(dlt)
                rows.append(row)
    return pd.DataFrame(rows), missing


def grid(df, out, what):
    keys = [s for s in SETTINGS if s[0] in set(df.setting)]
    fig, axes = plt.subplots(len(keys), 2, figsize=(13, 3.6 * len(keys)), dpi=140, squeeze=False)
    for row, (key, label, _, loads, _) in enumerate(keys):
        for c, load in enumerate(("under", "over")):
            ax = axes[row, c]
            g = df[(df.setting == key) & (df.load == load)].sort_values("n")
            if g.empty:
                ax.set_axis_off()
                continue
            if what == "revenue":
                for k, (name, col, mk) in POL.items():
                    y, e = g[k] / g.n, g[f"{k}_ci"].fillna(0) / g.n
                    ax.plot(g.n, y, marker=mk, ms=5, lw=1.7, color=col, label=name)
                    ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.18, lw=0)
                ax.plot(g.n, g.V / g.n, ls="--", lw=1.1, color="black", label="fluid optimum $V^*/n$")
                ax.set_ylabel("revenue per s per instance")
            elif what == "paired":
                ax.errorbar(g.n, g.BL_minus_FD, yerr=g.BL_minus_FD_ci.fillna(0), marker="v", ms=5, lw=1.5, capsize=3,
                            color=POL["BL"][1], label="blind minus fluid-dual (same seed)")
                ax.axhline(0, color="black", lw=0.9)
                ax.set_ylabel("percent of $|V^*|$")
            else:  # mechanism: long-job acceptance and holding cost
                for k, (name, col, mk) in POL.items():
                    ax.plot(g.n, g[f"{k}_acc_type3"], marker=mk, ms=5, lw=1.7, color=col, label=f"{name}: long jobs accepted %")
                ax.plot(g.n, g.FD_fl_type3, ls="--", lw=1.0, color=POL["FD"][1], label="fluid share, problem (28)")
                ax.plot(g.n, g.BL_fl_type3, ls=":", lw=1.2, color=POL["BL"][1], label="fluid share, blind LP")
                ax.set_ylabel("long-generation jobs accepted, percent")
                ax2 = ax.twinx()
                for k, (name, col, mk) in POL.items():
                    ax2.plot(g.n, g[f"{k}_cost"] / g.n, ls="-.", lw=1.0, color=col, alpha=0.6)
                ax2.set_ylabel("holding cost per s per instance (dash-dot)", fontsize=8)
            xstyle(ax, list(g.n))
            ax.set_title(f"{label}, {load}loaded ({g.rate.iloc[0]} req/s per instance)", fontsize=9)
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def f(v, d=2):
    return "--" if v is None or v != v else f"{v:,.{d}f}"


def table(df):
    rows = []
    for key, label, _, _, _ in SETTINGS:
        for load in ("under", "over"):
            for _, x in df[(df.setting == key) & (df.load == load)].sort_values("n").iterrows():
                rows.append(" & ".join([key, load, str(x.n), str(x.seeds), f(x.V, 1), f(x.FD, 1), f(x.BL, 1), f(x.FD_gapV), f(x.BL_gapV),
                                        f"{f(x.BL_minus_FD)} $\\pm$ {f(x.BL_minus_FD_ci)}",
                                        f"{f(x.FD_acc_type3, 1)} / {f(x.BL_acc_type3, 1)}",
                                        f"{f(x.FD_cost / x.n, 2)} / {f(x.BL_cost / x.n, 2)}"]) + r" \\")
    return ("{\\scriptsize\\begin{longtable}{llrrrrrrrrrr}\\toprule\n"
            r"set. & load & $n$ & seeds & $V^\star$ & FD & blind & FD gap \% & blind gap \% & blind $-$ FD, \% of $|V^\star|$ & long acc. \% FD / blind & hold. / s / inst. FD / blind \\ \midrule\endhead" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}")


def table_fluid(df):
    rows = []
    for key, label, objf, loads, _ in SETTINGS:
        for load in ("under", "over"):
            g = df[(df.setting == key) & (df.load == load)]
            if g.empty:
                continue
            x = g.sort_values("n").iloc[-1]
            rows.append(" & ".join([key, load, x.rate, f(x.V / x.n, 3), f((x.blind_lp or float("nan")) / x.n, 3),
                                    "/".join(f(x[f"FD_fl_{t}"], 1) for t, _ in TYPES), "/".join(f(x[f"BL_fl_{t}"], 1) for t, _ in TYPES)]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{llrrrrr}\\toprule\n"
            r"setting & load & req/s per inst. & $V^\star / n$ & blind LP $/ n$ (revenue only) & fluid shares (28) \% & fluid shares blind \% \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, missing = collect()
    if df.empty:
        sys.exit("nothing to report yet: " + ", ".join(missing))
    df.to_csv(os.path.join(a.out, "blind_stress.csv"), index=False)
    grid(df, os.path.join(a.out, "bs_revenue.png"), "revenue")
    grid(df, os.path.join(a.out, "bs_paired.png"), "paired")
    grid(df, os.path.join(a.out, "bs_mechanism.png"), "mechanism")
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float,longtable}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Stress test: fluid-dual blind against fluid-dual}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{Question and design}
At the baseline ($h = 0.03$, 6,500 KV blocks) fluid-dual blind, the index policy built from problem (28) with every holding cost set to zero, behaves like fluid-dual. The reason found at the fluid level: there the congestion cost is too small to change the fluid rates (type 3's marginal congestion cost, about 0.97, is far below its reward, 4.53), so both programs admit the same rates, share their centers and give the same index signs and rankings. This test uses settings where the congestion cost does change the fluid rates.
\begin{itemize}
\item \textbf{S3}: $h = 0.5$, 6,500 blocks. At the blind policy's rates the holding cost is 125 (underloaded) and 143 (overloaded) percent of gross revenue in the fluid model; problem (28) turns away long jobs and keeps it near 47 percent. Holding cost dominates.
\item \textbf{S1}: $h = 0.2$, 6,500 blocks. Holding cost 50 / 57 percent of revenue at the blind rates, 35 / 29 percent in (28).
\item \textbf{S2}: $h = 0.1$, 15,909 blocks (the H100's physical maximum), so memory does not bind and congestion rises steeply with utilisation.
\end{itemize}
Both policies use $\eta = V^\star/J$ from each setting's own problem (28); the blind policy's prices, marginal costs and centers come from the revenue-only LP. Each setting's runs are valued with that setting's holding cost. Loads at 0.817 and 1.070 of the setting's fluid capacity (11 and 14.4 req/s per instance at 6,500 blocks; 15.793 and 20.6745 at 15,909 blocks). $n = 2$ to 64; seeds 1 to 20 for $n \le 16$ and 1 to 10 for $n = 32, 64$, common to both policies. Horizon 1,200 s, window [500, 1,100) s.
"""]
    if missing:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(missing)) + ".\n")
    tex.append(r"\section{Fluid solutions}" + table_fluid(df))
    tex.append(r"""\section{Results}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{bs_revenue.png}\caption{Revenue per second per instance against $n$, with $V^\star/n$ (dashed); bands 95 percent over seeds.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{bs_paired.png}\caption{Blind minus fluid-dual on the same seed, percent of $|V^\star|$; bars 95 percent.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{bs_mechanism.png}\caption{Mechanism: share of long-generation jobs accepted (solid) against the fluid shares of (28) (dashed) and of the blind LP (dotted); holding cost per second per instance on the right axis (dash-dot).}\end{figure}
""" + table(df))
    tex.append(r"""\section{Artifacts}
blind\_stress.csv in this folder. Folders ours/experiments/2026-10-16\_scaling\_blindstress\_\{h0.5\_kv6500, h0.2\_kv6500, h0.1\_kv15909\}\_\{fluid\_dual, fluid\_dual\_blind\}\_rpi*; baseline folders as in the 2026-10-16 variants report; objective files ours/specs/objective\_h\{0.1, 0.2, 0.5\}.yaml. Generated by ours/report\_blind\_stress.py.
\end{document}""")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| missing:", missing)
    print(df[["setting", "load", "n", "seeds", "V", "FD_gapV", "BL_gapV", "BL_minus_FD", "BL_minus_FD_ci", "FD_acc_type3", "BL_acc_type3"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
