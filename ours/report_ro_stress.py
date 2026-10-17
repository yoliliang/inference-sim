#!/usr/bin/env python3
"""ours/report_ro_stress.py: RO-dual against fluid-dual in settings where problem (28) and the Refined
Optimistic program choose different fluid rates (stress settings and the 2026-10-17 enrichment), with
the baseline for reference.

    python ours/report_ro_stress.py --out ours/experiments/<date>_report_ro_stress

Each (setting, load) pairs a fluid-dual folder and an RO-dual folder with the same seeds; runs are
valued with that setting's objective file. Benchmarks: V* (problem 28) and the certified U_RO (its
dual bound recorded). Statistics use the seeds both arms share; paired differences use the same seed.
n = 1 rows appear when those runs exist. House style: no em or en dashes.
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

D16, D17 = "2026-10-16_scaling_", "2026-10-17_scaling_"
FD_BS = D16 + "blindstress_{s}_fluid_dual_rpi{r}"
RO_RS = D17 + "rostress_{s}_fluid_dual_ro_rpi{r}"
FD_EN = D17 + "enrich_{s}_fluid_dual_rpi{r}"
RO_EN = D17 + "enrich_{s}_fluid_dual_ro_rpi{r}"
# key, label, objective file, [(load label, rate, fluid-dual folder, RO-dual folder)]
SETTINGS = [
    ("base", "baseline: $h = 0.03$, 6,500 blocks", "objective.yaml", [
        ("0.82 cap", "11", D16.replace("2026-10-16", "2026-10-11") + "scale1024_fluid_dual_kv6500_rpi11", D17 + "scale64_fluid_dual_ro_kv6500_rpi11"),
        ("1.07 cap", "14.4", D16.replace("2026-10-16", "2026-10-11") + "scale1024_fluid_dual_kv6500_rpi14.4", D17 + "scale64_fluid_dual_ro_kv6500_rpi14.4")]),
    ("h0.15", "$h = 0.15$, 6,500 blocks", "objective_h0.15.yaml", [
        ("0.82 cap", "11", FD_EN.format(s="h0.15_kv6500", r="11"), RO_EN.format(s="h0.15_kv6500", r="11")),
        ("1.00 cap", "13.4609", FD_EN.format(s="h0.15_kv6500", r="13.4609"), RO_EN.format(s="h0.15_kv6500", r="13.4609")),
        ("1.07 cap", "14.4", FD_EN.format(s="h0.15_kv6500", r="14.4"), RO_EN.format(s="h0.15_kv6500", r="14.4"))]),
    ("S1", "S1: $h = 0.2$, 6,500 blocks", "objective_h0.2.yaml", [
        ("0.82 cap", "11", FD_BS.format(s="h0.2_kv6500", r="11"), RO_RS.format(s="h0.2_kv6500", r="11")),
        ("1.00 cap", "13.4609", FD_EN.format(s="h0.2_kv6500", r="13.4609"), RO_EN.format(s="h0.2_kv6500", r="13.4609")),
        ("1.07 cap", "14.4", FD_BS.format(s="h0.2_kv6500", r="14.4"), RO_RS.format(s="h0.2_kv6500", r="14.4"))]),
    ("S3", "S3: $h = 0.5$, 6,500 blocks", "objective_h0.5.yaml", [
        ("0.82 cap", "11", FD_BS.format(s="h0.5_kv6500", r="11"), RO_RS.format(s="h0.5_kv6500", r="11")),
        ("1.07 cap", "14.4", FD_BS.format(s="h0.5_kv6500", r="14.4"), RO_RS.format(s="h0.5_kv6500", r="14.4"))]),
    ("S2", "S2: $h = 0.1$, 15,909 blocks", "objective_h0.1.yaml", [
        ("0.82 cap", "15.793", FD_BS.format(s="h0.1_kv15909", r="15.793"), RO_RS.format(s="h0.1_kv15909", r="15.793")),
        ("1.00 cap", "19.3262", FD_EN.format(s="h0.1_kv15909", r="19.3262"), RO_EN.format(s="h0.1_kv15909", r="19.3262")),
        ("1.07 cap", "20.6745", FD_BS.format(s="h0.1_kv15909", r="20.6745"), RO_RS.format(s="h0.1_kv15909", r="20.6745"))]),
    ("h0.2b", "$h = 0.2$, 15,909 blocks", "objective_h0.2.yaml", [
        ("0.82 cap", "15.793", FD_EN.format(s="h0.2_kv15909", r="15.793"), RO_EN.format(s="h0.2_kv15909", r="15.793")),
        ("1.07 cap", "20.6745", FD_EN.format(s="h0.2_kv15909", r="20.6745"), RO_EN.format(s="h0.2_kv15909", r="20.6745"))]),
]
POL = {"FD": ("fluid-dual (problem 28)", "#C0504D", "o"), "RO": ("RO-dual", "#17BECF", "P")}
NMAX = 64


def fluidinfo(folder):
    out = {}
    for p in glob.glob(os.path.join(EXP, folder, "specs", "fluid_n*.yaml")):
        d = yaml.safe_load(open(p))
        mt = d["meta"]
        out[mt["instances"]] = {"value": mt["value"], "bound": mt.get("bound"),
                                **{f"fl_{t['name']}": 100 * t["admitted"] / t["lambda"] for t in d["types"]}}
    return out


def terminal(folder):
    path = os.path.join(EXP, folder)
    m = json.load(open(os.path.join(path, "manifest.json")))
    out = {}
    for p in glob.glob(os.path.join(path, "runs", "*_s*.json.gz")):
        tag = analyze.parse_tag(os.path.basename(p), m)
        if tag and tag[0] <= NMAX:
            d = json.load(gzip.open(p, "rt"))
            out[(tag[0], tag[2])] = d["still_queued"] + d["still_running"]
    return out


def collect():
    rows, missing = [], []
    for key, label, objf, loads in SETTINGS:
        params = yaml.safe_load(open(os.path.join(SPEC, objf)))
        for lname, rate, ffd, fro in loads:
            ok = all(os.path.isdir(os.path.join(EXP, fo, "runs")) for fo in (ffd, fro))
            data = {k: runs(fo, params) for k, fo in (("FD", ffd), ("RO", fro))} if ok else {}
            if not ok or any(d.empty for d in data.values()):
                missing.append(f"{key} {rate}")
                continue
            info = {"FD": fluidinfo(ffd), "RO": fluidinfo(fro)}
            term = {"FD": terminal(ffd), "RO": terminal(fro)}
            for n in sorted(set(data["FD"].n) & set(data["RO"].n)):
                if n not in info["FD"] or n not in info["RO"]:
                    continue
                seeds = sorted(set(data["FD"][data["FD"].n == n].seed) & set(data["RO"][data["RO"].n == n].seed))
                if not seeds:
                    continue
                V, U = info["FD"][n]["value"], info["RO"][n]["value"]
                row = {"setting": key, "label": label, "load": lname, "rate": rate, "n": int(n), "seeds": len(seeds),
                       "V": V, "U": U, "U_bound": info["RO"][n]["bound"]}
                for t, _ in TYPES:
                    row[f"FD_fl_{t}"], row[f"RO_fl_{t}"] = info["FD"][n][f"fl_{t}"], info["RO"][n][f"fl_{t}"]
                sub = {k: d[(d.n == n) & d.seed.isin(seeds)].set_index("seed").sort_index() for k, d in data.items()}
                for k, s in sub.items():
                    row[k], row[f"{k}_ci"] = s.value.mean(), ci(s.value)
                    row[f"{k}_gapU"] = 100 * (U - s.value.mean()) / abs(U)
                    row[f"{k}_gapV"] = 100 * (V - s.value.mean()) / abs(V)
                    for col in ("reward", "cost", "wip", "preempt", "wasted", *[f"acc_{t}" for t, _ in TYPES]):
                        row[f"{k}_{col}"] = s[col].mean()
                    row[f"{k}_terminal"] = float(np.mean([term[k].get((n, sd), np.nan) for sd in seeds]))
                dlt = 100 * (sub["RO"].value - sub["FD"].value) / abs(U)
                row["RO_minus_FD"], row["RO_minus_FD_ci"] = dlt.mean(), ci(dlt)
                rows.append(row)
    return pd.DataFrame(rows), missing


def figure(df, out, what):
    keys = [s for s in SETTINGS if s[0] in set(df.setting)]
    ncol = max(len(s[3]) for s in keys)
    fig, axes = plt.subplots(len(keys), ncol, figsize=(5.2 * ncol, 3.5 * len(keys)), dpi=130, squeeze=False)
    for row, (key, label, _, loads) in enumerate(keys):
        for c in range(ncol):
            ax = axes[row, c]
            if c >= len(loads):
                ax.set_axis_off()
                continue
            lname, rate = loads[c][0], loads[c][1]
            g = df[(df.setting == key) & (df.rate == rate)].sort_values("n")
            if g.empty:
                ax.set_axis_off()
                continue
            if what == "revenue":
                for k, (name, col, mk) in POL.items():
                    y, e = g[k] / g.n, g[f"{k}_ci"].fillna(0) / g.n
                    ax.plot(g.n, y, marker=mk, ms=4, lw=1.6, color=col, label=name)
                    ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.18, lw=0)
                ax.plot(g.n, g.U / g.n, ls=":", lw=1.4, color="black", label="$U_{RO}/n$")
                ax.plot(g.n, g.V / g.n, ls="--", lw=1.0, color="grey", label="$V^*/n$")
                ax.set_ylabel("revenue per s per instance", fontsize=8)
            elif what == "paired":
                ax.errorbar(g.n, g.RO_minus_FD, yerr=g.RO_minus_FD_ci.fillna(0), marker="P", ms=5, lw=1.4, capsize=3,
                            color=POL["RO"][1], label="RO-dual minus fluid-dual")
                ax.axhline(0, color="black", lw=0.9)
                ax.set_ylabel("percent of $|U_{RO}|$", fontsize=8)
            else:
                for k, (name, col, mk) in POL.items():
                    ax.plot(g.n, g[f"{k}_acc_type3"], marker=mk, ms=4, lw=1.6, color=col, label=f"{name}: accepted")
                    ax.plot(g.n, g[f"{k}_fl_type3"], ls="--", lw=1.0, color=col, label=f"fluid share, {'(28)' if k == 'FD' else 'RO'}")
                ax.set_ylabel("long-generation jobs, percent", fontsize=8)
            xstyle(ax, list(g.n))
            ax.set_title(f"{label}, {lname} ({rate} req/s/inst.)", fontsize=8)
            ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summary_table(df):
    rows = []
    for key, label, _, loads in SETTINGS:
        for lname, rate, _, _ in loads:
            g = df[(df.setting == key) & (df.rate == rate)]
            if g.empty:
                continue
            for _, x in g.sort_values("n").iterrows():
                if x.n not in (min(g.n), 8, max(g.n)):
                    continue
                rows.append(" & ".join([key, lname, str(x.n), f"{f(x.FD_fl_type3, 1)} / {f(x.RO_fl_type3, 1)}",
                                        f"{f(x.FD_acc_type3, 1)} / {f(x.RO_acc_type3, 1)}", f(x.FD_gapU), f(x.RO_gapU),
                                        f"{f(x.RO_minus_FD)} $\\pm$ {f(x.RO_minus_FD_ci)}"]) + r" \\")
    return ("{\\small\\begin{longtable}{llrrrrrr}\\toprule\n"
            r"setting & load & $n$ & type-3 fluid share \% (28) / RO & type-3 accepted \% FD / RO & FD gap to $U_{RO}$ \% & RO gap \% & RO $-$ FD, \% of $|U_{RO}|$ \\ \midrule\endhead" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}")


def full_table(df):
    rows = []
    for key, label, _, loads in SETTINGS:
        for lname, rate, _, _ in loads:
            for _, x in df[(df.setting == key) & (df.rate == rate)].sort_values("n").iterrows():
                rows.append(" & ".join([key, rate, str(x.n), str(x.seeds), f(x.V, 1), f(x.U, 1), f(x.FD, 1), f(x.RO, 1),
                                        f(x.FD_gapV), f(x.RO_gapV), f(x.FD_cost / x.n, 2), f(x.RO_cost / x.n, 2),
                                        f(x.FD_terminal / x.n, 1), f(x.RO_terminal / x.n, 1)]) + r" \\")
    return ("{\\scriptsize\\begin{longtable}{llrrrrrrrrrrrr}\\toprule\n"
            r"set. & req/s & $n$ & seeds & $V^\star$ & $U_{RO}$ & FD & RO & FD gap $V^\star$ \% & RO gap $V^\star$ \% & FD hold./s/inst. & RO hold. & FD unfin./inst. & RO unfin. \\ \midrule\endhead" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, missing = collect()
    if df.empty:
        sys.exit("nothing to report yet: " + ", ".join(missing))
    df.to_csv(os.path.join(a.out, "ro_stress.csv"), index=False)
    figure(df, os.path.join(a.out, "rs_revenue.png"), "revenue")
    figure(df, os.path.join(a.out, "rs_paired.png"), "paired")
    figure(df, os.path.join(a.out, "rs_mechanism.png"), "mechanism")
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.6cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float,longtable}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{RO-dual against fluid-dual where the two fluid programs differ}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{Question and design}
At the baseline ($h = 0.03$, 6,500 KV blocks) problem (28) and the Refined Optimistic program choose the same fluid rates, so their index policies coincide. Here both arms run where the two programs choose different rates: larger holding costs ($h = 0.15, 0.2, 0.5$) and, with 15,909 KV blocks (the H100's physical maximum), settings where memory does not bind. Loads at 0.817, 1.000 and 1.070 of each setting's fluid capacity (6,500 blocks: 11, 13.4609, 14.4 req/s per instance; 15,909 blocks: 15.793, 19.3262, 20.6745). Both arms use the gain $\eta = V^\star/J$ of that setting's problem (28); RO-dual takes its rates, marginal charges (ro-marginal), minimum-norm multipliers and centers (ro-centers) from the Refined Optimistic program, certified optimal by Gurobi's barrier with a dual bound. Runs are valued with that setting's holding cost. $n = 2$ to 64 (and $n = 1$ where run), seeds 1 to 20 for $n \le 16$ and 1 to 10 for $n = 32, 64$, common to both arms. BLIS uses chunked prefill and a fitted step time: an implementation transfer test in the draft's sense.
"""]
    if missing:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(missing)) + ".\n")
    tex.append(r"""\section{Summary}
Rows at the smallest $n$, at $n = 8$ and at the largest $n$ of each setting and load.
""" + summary_table(df))
    tex.append(r"""\section{Figures}
\begin{figure}[H]\centering\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{rs_paired.png}\caption{RO-dual minus fluid-dual on the same seed, percent of $|U_{RO}|$; bars 95 percent; positive means RO-dual is better.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{rs_revenue.png}\caption{Revenue per second per instance, with $U_{RO}/n$ (dotted) and $V^\star/n$ (dashed); bands 95 percent.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth,height=0.88\textheight,keepaspectratio]{rs_mechanism.png}\caption{Long-generation jobs accepted (solid) against each program's fluid share (dashed).}\end{figure}
\section{All points}
""" + full_table(df))
    tex.append(r"""\section{Artifacts}
ro\_stress.csv in this folder. Folders under ours/experiments: 2026-10-16\_scaling\_blindstress\_*\_fluid\_dual\_* and 2026-10-17\_scaling\_rostress\_* (S1, S2, S3), 2026-10-17\_scaling\_enrich\_* (capacity loads, $h = 0.15$, $h = 0.2$ with 15,909 blocks), baseline folders of the RO-dual test. Generated by ours/report\_ro\_stress.py.
\end{document}""")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        rr = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if rr.returncode != 0:
        print(rr.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| missing:", missing)
    print(df[["setting", "rate", "n", "seeds", "FD_fl_type3", "RO_fl_type3", "FD_gapU", "RO_gapU", "RO_minus_FD", "RO_minus_FD_ci"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
