#!/usr/bin/env python3
"""ours/report_optimistic.py: fluid-dual and its variants against B2 and the two fluid benchmarks,
for n = 2 .. 64 at 11 and 14.4 req/s per instance.

Policies: fluid-dual (Definition 1 on problem (28)); optimistic fluid-dual (the same index rule on
the optimistic program, draft eq. opt-feedback); fluid-dual with buffer (beta^M_j = eta B / M_j
added to the memory price, B = 4,096 tokens); fluid-dual blind (problem (28) with every holding
cost set to zero); B2 (B1 + GAIE-legacy admission). Benchmarks: V*, the optimum of (28); U_opt,
the optimistic benchmark (eq. opt-program), a valid upper bound under persistent bounded output.

    python ours/report_optimistic.py --out ours/experiments/<date>_report_fluid_dual_variants

Every statistic is computed on the seeds all available policies share at that (load, n), from each
run's own window block (eq. 10 objective, time-average estimator). Paired differences against
fluid-dual use the same seed (common arrivals and output lengths). A policy without runs is left
out. House style: no em or en dashes.
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
    "BUF": ("fluid-dual with buffer (B = 4,096)", "#4E9A06", "^", "2026-10-16_scaling_scale64_fluid_dual_buffer4096_kv6500_rpi{r}"),
    "BL": ("fluid-dual blind (h = 0)", "#E69F00", "v", "2026-10-16_scaling_scale64_fluid_dual_blind_kv6500_rpi{r}"),
    "B2": ("B2 (B1 + GAIE admission)", "#2E75B6", "s", "2026-10-10_scaling_scale64_B2_kv6500_rpi{r}"),
}
SHORT = {"FD": "FD", "OFD": "OFD", "BUF": "FD+buffer", "BL": "FD blind", "B2": "B2"}
TYPES = [("type1", "chat"), ("type2", "RAG"), ("type3", "long gen.")]
NMAX = 64


def runs(folder, params):
    """one row per run: n, seed, value, reward, cost, mean in system, preemption and per-type admission."""
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
    """V* (problem (28), aggregate) from the fluid-dual folder; U_opt and its dual bound from the optimistic one."""
    out = {}
    for p in glob.glob(os.path.join(EXP, POL["FD"][3].format(r=r), "specs", "fluid_n*.yaml")):
        mt = yaml.safe_load(open(p))["meta"]
        if mt["instances"] <= NMAX:
            out.setdefault(mt["instances"], {})["V"] = mt["value"]
    for p in glob.glob(os.path.join(EXP, POL["OFD"][3].format(r=r), "specs", "fluid_n*.yaml")):
        d = yaml.safe_load(open(p))
        mt = d["meta"]
        b = out.setdefault(mt["instances"], {})
        b.update(U=mt["value"], U_bound=mt["bound"], U_status=mt["status"],
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
            d = runs(f, params) if os.path.isdir(os.path.join(EXP, f, "runs")) else pd.DataFrame()
            if d.empty:
                missing.append(f"{POL[k][0]} at {r} req/s")
            else:
                data[k] = d
        if "FD" not in data:
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
                if k != "FD":
                    dlt = 100 * (s.value - sub["FD"].value) / U
                    row[f"{k}_minus_FD"], row[f"{k}_minus_FD_ci"] = dlt.mean(), ci(dlt)
                    paired.append(pd.DataFrame({"load": r, "n": int(n), "policy": k, "seed": seeds, "minus_FD_pct_of_U": dlt.values}))
            rows.append(row)
    return pd.DataFrame(rows), (pd.concat(paired) if paired else pd.DataFrame()), missing


def present(df):
    return [k for k in POL if k in df.columns]


def fig_revenue(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
    for ax, (r, label) in zip(axes, LOADS):
        g = df[df.load == r].sort_values("n")
        if g.empty:
            ax.set_axis_off()
            continue
        for k in present(df):
            if g[k].isna().all():
                continue
            name, col, mk, _ = POL[k]
            y, e = g[k] / g.n, g[f"{k}_ci"].fillna(0) / g.n
            ax.plot(g.n, y, marker=mk, ms=5, lw=1.6, color=col, label=name)
            ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.15, lw=0)
        ax.plot(g.n, g.U / g.n, ls=":", lw=1.6, color="black", label="optimistic benchmark $U_{opt}/n$")
        ax.plot(g.n, g.V / g.n, ls="--", lw=1.1, color="grey", label="fluid optimum $V^*/n$ (problem 28)")
        xstyle(ax, list(g.n))
        ax.set_ylabel("revenue per second per instance")
        ax.set_title(label, fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
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
            for k in present(df):
                if g[k].isna().all():
                    continue
                name, col, mk, _ = POL[k]
                y = g[f"{k}_{suffix}"]
                e = 100 * g[f"{k}_ci"].fillna(0) / (g.U if suffix == "gapU" else g.V)
                ax.plot(g.n, y, marker=mk, ms=5, lw=1.6, color=col, label=name)
                ax.fill_between(g.n, y - e, y + e, color=col, alpha=0.15, lw=0)
            ax.axhline(0, color="grey", lw=0.8)
            xstyle(ax, list(g.n))
            ax.set_ylabel(f"gap to {bench}, percent of {bench}")
            ax.set_title(f"{label.split(',')[0]}: relative gap to {bench}", fontsize=10)
            ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_paired(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    others = [k for k in present(df) if k != "FD"]
    for ax, (r, label) in zip(axes, LOADS):
        g = df[df.load == r].sort_values("n")
        if g.empty:
            ax.set_axis_off()
            continue
        for q, k in enumerate(others):
            if f"{k}_minus_FD" not in g or g[f"{k}_minus_FD"].isna().all():
                continue
            x = g.n * (1 + 0.04 * (q - (len(others) - 1) / 2))  # small horizontal offsets so bars do not overlap
            ax.errorbar(x, g[f"{k}_minus_FD"], yerr=g[f"{k}_minus_FD_ci"].fillna(0), marker=POL[k][2], ms=5, lw=1.4,
                        capsize=3, color=POL[k][1], label=POL[k][0])
        ax.axhline(0, color="black", lw=0.9)
        xstyle(ax, list(g.n))
        ax.set_ylabel("policy minus fluid-dual, percent of $U_{opt}$\n(same seed)")
        ax.set_title(label.split(",")[0] + ": paired difference against fluid-dual; bars 95 percent", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_diag(df, out):
    specs = [("reward", "gross revenue per s per instance", True), ("cost", "holding cost per s per instance", True),
             ("wip", "mean jobs in system per instance", True), ("preempt", "preemptions per arrival", False),
             ("wasted", "recomputed (wasted) tokens per arrival", False), ("acc_type3", "long-generation jobs accepted, percent", False)]
    fig, axes = plt.subplots(len(specs), 2, figsize=(13, 3.0 * len(specs)), dpi=140)
    for c, (r, label) in enumerate(LOADS):
        g = df[df.load == r].sort_values("n")
        for row, (col_, ylab, per_inst) in enumerate(specs):
            ax = axes[row, c]
            if g.empty:
                ax.set_axis_off()
                continue
            for k in present(df):
                if g[k].isna().all():
                    continue
                name, colr, mk, _ = POL[k]
                y = g[f"{k}_{col_}"] / (g.n if per_inst else 1)
                ax.plot(g.n, y, marker=mk, ms=4, lw=1.4, color=colr, label=name)
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


def table_gaps(df):
    ks = present(df)
    rows = []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            rows.append(" & ".join([r, str(x.n), str(x.seeds), f(x.U, 1)] + [f(x.get(f"{k}_gapU")) for k in ks]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{lrrr" + "r" * len(ks) + "}\\toprule\n"
            r"req/s & $n$ & seeds & $U_{opt}$ & " + " & ".join(f"{SHORT[k]} gap \\%" for k in ks) + r" \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def table_paired(df):
    ks = [k for k in present(df) if k != "FD"]
    rows = []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            rows.append(" & ".join([r, str(x.n)] + [f"{f(x.get(f'{k}_minus_FD'), 3)} $\\pm$ {f(x.get(f'{k}_minus_FD_ci'), 3)}" for k in ks]) + r" \\")
    return ("\\begin{center}\\small\\begin{tabular}{lr" + "r" * len(ks) + "}\\toprule\n"
            r"req/s & $n$ & " + " & ".join(f"{SHORT[k]} $-$ FD" for k in ks) + r" \\ \midrule" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def table_diag(df):
    rows = []
    for r, _ in LOADS:
        for _, x in df[df.load == r].sort_values("n").iterrows():
            for k in present(df):
                if x.get(k) != x.get(k):
                    continue
                rows.append(" & ".join([r, str(x.n), SHORT[k], f(x[f"{k}_reward"] / x.n, 3), f(x[f"{k}_cost"] / x.n, 3),
                                        f(x[f"{k}_wip"] / x.n, 1), f(x[f"{k}_acc_type3"], 2), f(x[f"{k}_preempt"], 4),
                                        f(x[f"{k}_wasted"], 1)]) + r" \\")
    return ("{\\small\\begin{longtable}{lrlrrrrrr}\\toprule\n"
            r"req/s & $n$ & policy & revenue / s / inst. & holding / s / inst. & WIP / inst. & type 3 acc. \% & preempt. / arrival & wasted tok. / arrival \\ \midrule\endhead" "\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, paired, missing = collect()
    if df.empty:
        sys.exit("no complete (load, n) yet: " + ", ".join(missing))
    df.to_csv(os.path.join(a.out, "fluid_dual_variants.csv"), index=False)
    paired.to_csv(os.path.join(a.out, "fluid_dual_variants_paired_by_seed.csv"), index=False)
    fig_revenue(df, os.path.join(a.out, "var_revenue.png"))
    fig_gaps(df, os.path.join(a.out, "var_gaps.png"))
    fig_paired(df, os.path.join(a.out, "var_paired.png"))
    fig_diag(df, os.path.join(a.out, "var_diagnostics.png"))
    bm = {r: benchmarks(r) for r, _ in LOADS}
    prices = []
    for r, _ in LOADS:
        b = bm[r].get(8, {})
        if "gM" in b:
            prices.append(f"{r} req/s: $\\gamma^{{M,U}} = {b['gM']:.3f}$; $c^U$ = " + ", ".join(f"{v:.4f}" for v in b["cU"].values()))
    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float,longtable}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Fluid-dual and its variants against B2, $V^\star$ and $U_{opt}$}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{What was run}
Baseline instance (H100, 6,500 KV blocks, Qwen3-14B, type shares 50/35/15, $\pi_{in} = 1$, $\pi_{out} = 4$, $h = 0.03$), 11 and 14.4 req/s per instance, $n = 2, 4, 8, 16, 32, 64$, horizon 1,200 s, statistics over arrivals in [500, 1,100) s, revenue = eq. (10) per second of the window. Seeds 1 to 20 for $n \le 16$ and 1 to 10 for $n = 32, 64$, identical for every policy (common arrivals and output lengths); every number below uses only the seeds all policies share. The environment is the same for every policy; only the policy changes.
\begin{itemize}
\item \textbf{Fluid-dual (FD)}: Definition 1 on problem (28), solved by GPU type; $\eta = V^\star/J$.
\item \textbf{Optimistic fluid-dual (OFD)}: the same index rule built on the optimistic program (eq. opt-program): marginal charges $c^U_{ij}$ (opt-marginal), minimum-norm KKT multipliers, centers $K^U_j$, $Z^U_j$ (opt-targets); the reference gain $\eta = V^\star/J$.
\item \textbf{Fluid-dual with buffer (FD+buffer)}: fluid-dual with a constant buffer in the memory price,
\[\gamma^M_j(t) = \Big[\gamma^{M\star}_j + \beta^M_j + \eta\,\frac{K_j(t) - K^\star_j}{M_j}\Big]^+,\qquad \beta^M_j = \eta\,\frac{B}{M_j},\qquad \beta^B_j = 0,\]
which moves the memory target down by $B$ tokens. $B = 4{,}096$ tokens (3.9 percent of $M_j = 104{,}000$), about one standard deviation of one instance's KV occupancy under fluid-dual (4,293 tokens, measured at $n = 8$, 14.4 req/s per instance, seed 1); $\beta^M_j = 0.73$ at 11 and $0.91$ at 14.4 req/s per instance.
\item \textbf{Fluid-dual blind (FD blind)}: the same index policy built from problem (28) with every holding cost set to zero (the revenue-only LP), so the marginal congestion costs are zero; $\eta = V^\star/J$ of the true problem. Evaluated with the true objective.
\item \textbf{B2}: B1 (llm-d weighted routing) plus GAIE-legacy saturation admission with every type sheddable.
\end{itemize}
Benchmarks: $V^\star$, the optimum of problem (28); $U_{opt}$, the optimistic benchmark, which by the draft's lemma bounds the stochastic optimum from above under persistent bounded output. BLIS draws each job's output length once (a preempted job is recomputed with the same length) and clamps it at 2,048 / 2,048 / 4,096 tokens; it uses chunked prefill and a fitted affine step time, so by the draft's Section 4.4 these results are an implementation transfer test, not a certified benchmark claim.

In this baseline the optimistic program, the blind LP and problem (28) admit the same rates (the binding memory constraint decides admission), so OFD, FD blind and FD share their centers $K_j$, $Z_j$ and differ only in the marginal charges and the multipliers. Optimistic quantities at $n = 8$ (identical per instance for every $n$): """ + "; ".join(prices) + r""".
"""]
    if missing:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(missing)) + ".\n")
    tex.append(r"\section{Benchmarks}" + table_bench(df))
    tex.append(r"""\section{Revenue and gaps}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{var_revenue.png}\caption{Revenue per second per instance against $n$; dotted $U_{opt}/n$, dashed $V^\star/n$; bands 95 percent over seeds.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{var_gaps.png}\caption{Relative gap $100\,(B - V)/B$ against $n$ for $B = U_{opt}$ (top) and $B = V^\star$ (bottom).}\end{figure}
""" + table_gaps(df))
    tex.append(r"""\section{Is a variant better than fluid-dual?}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{var_paired.png}\caption{Each policy minus fluid-dual on the same seed, percent of $U_{opt}$; positive means better than fluid-dual.}\end{figure}
""" + table_paired(df))
    tex.append(r"""\section{Diagnostics}
Gross revenue, holding cost and mean jobs in system per instance; acceptance of long-generation jobs; preemptions and recomputed tokens per arrival (replay work after preemption).
\begin{figure}[H]\centering\includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{var_diagnostics.png}\caption{Diagnostics against $n$.}\end{figure}
""" + table_diag(df))
    tex.append(r"""\section{Artifacts}
fluid\_dual\_variants.csv (one row per load and $n$), fluid\_dual\_variants\_paired\_by\_seed.csv (paired differences by seed). Folders under ours/experiments: 2026-10-11\_scaling\_scale1024\_fluid\_dual\_kv6500\_rpi\{11, 14.4\} (FD), 2026-10-16\_scaling\_scale64\_fluid\_dual\_optimistic\_kv6500\_rpi\{11, 14.4\} (OFD; fluid files carry $U_{opt}$, its dual bound, $V^\star$, $x^U$, $c^U$, multipliers and centers), 2026-10-16\_scaling\_scale64\_fluid\_dual\_buffer4096\_kv6500\_rpi\{11, 14.4\} (FD+buffer), 2026-10-16\_scaling\_scale64\_fluid\_dual\_blind\_kv6500\_rpi\{11, 14.4\} (FD blind), 2026-10-10\_scaling\_scale64\_B2\_kv6500\_rpi\{11, 14.4\} (B2). Generated by ours/report\_optimistic.py.
\end{document}""")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| missing:", missing)
    cols = ["load", "n", "seeds"] + [f"{k}_gapU" for k in present(df)] + [f"{k}_minus_FD" for k in present(df) if k != "FD"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
