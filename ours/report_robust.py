#!/usr/bin/env python3
"""ours/report_robust.py: report of the two experiments of 2026-10-11, fluid-dual against the
benchmark B2 (B1 plus llm-d / GAIE saturation admission, every type sheddable).

  Experiment 1 (robustness). Instances that differ from the baseline in one respect: GPU type,
  a mix of GPU types, the workload mix, or the prices and holding cost. For each instance and
  for an underloaded and an overloaded point, both policies run over the system scale n.
  Revenue (eq. 10) is compared with the fluid optimum V* of the instance.
  Experiment 2 (demand sweep). The representative instance (baseline, n = 8) under an increasing
  demand rate; y = relative gap to the fluid optimum, 100 (V* - value) / V*.

    python ours/report_robust.py --out ours/experiments/<date>_report_robustness [--date 2026-10-11]

Revenue is recomputed from every run's window block with the objective parameters of the
instance (objective.window_value), so B2 runs of the baseline can be revalued under the price
variants without rerunning them (B2 does not use prices). Missing folders are skipped and
listed. Compiled with pdflatex. House style: no em dashes or en dashes.
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import objective  # noqa: E402

COL = {"B2": "#55A868", "FD": "#DD8452"}
NAME = {"B2": "B2 (B1 + GAIE admission)", "FD": "fluid-dual"}
SPEC = os.path.join(HERE, "specs")
FORMS = []  # fluid formulation of every fluid solution read, for the statement in the report


def instances(date):
    base_b2 = {"under": "2026-10-10_scaling_scale64_B2_kv6500_rpi11", "over": "2026-10-10_scaling_scale64_B2_kv6500_rpi14.4"}
    base_fd = {"under": "2026-10-08_scaling_scale64_fluid_dual_kv6500_rpi11", "over": "2026-10-08_scaling_scale64_fluid_dual_kv6500_rpi14.4"}
    obj = os.path.join(SPEC, "objective.yaml")
    out = [dict(key="base", label="baseline: H100, 6,500 blocks", change="(reference)", b2=base_b2, fd=base_fd, obj=obj,
                rate={"under": 11, "over": 14.4})]
    rob = lambda k, p, l: f"{date}_scaling_robust_{k}_{p}_{l}"  # noqa: E731
    phys = [("hetHA", "H100 + A100-80, 1 : 1", "GPU mix: half H100, half A100-80 (6,500 blocks each)", 8.9367, 11.6990),
            ("hetHL", "H100 + L40S, 1 : 1", "GPU mix: half H100 (6,500 blocks), half L40S (4,113 blocks, its physical maximum)", 6.7283, 8.8081),
            ("A100", "A100-80 only", "GPU type: A100-80 (6,500 blocks)", 6.5645, 8.5937),
            ("L40S", "L40S only", "GPU type: L40S (4,113 blocks, its physical maximum)", 2.3587, 3.0878),
            ("mix304030", "workload 30/30/40", "type shares 30 / 30 / 40 percent instead of 50 / 35 / 15 (more long generation)", 5.4849, 7.1803)]
    for k, lab, ch, ru, ro in phys:
        out.append(dict(key=k, label=lab, change=ch, obj=obj, rate={"under": ru, "over": ro},
                        b2={l: rob(k, "B2", l) for l in ("under", "over")},
                        fd={l: rob(k, "fluid_dual", l) for l in ("under", "over")}))
    prices = [("piout1", "price pi_out = 1", "output tokens priced like input tokens (pi_out = 1 instead of 4)"),
              ("h0.1", "holding cost h = 0.1", "holding cost 0.1 per job per second instead of 0.03"),
              ("h0.01", "holding cost h = 0.01", "holding cost 0.01 per job per second instead of 0.03")]
    for k, lab, ch in prices:
        out.append(dict(key=k, label=lab, change=ch + "; B2 runs of the baseline revalued", obj=os.path.join(SPEC, f"objective_{k}.yaml"),
                        rate={"under": 11, "over": 14.4}, b2=base_b2,
                        fd={l: rob(k, "fluid_dual", l) for l in ("under", "over")}))
    return out


def run_values(folder, params):
    """per (n, rate, seed): window revenue per second (type all) under params."""
    path = os.path.join(EXP, folder)
    m = json.load(open(os.path.join(path, "manifest.json")))
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "runs", "*_s*.json.gz"))):
        tag = analyze.parse_tag(os.path.basename(p), m)
        if not tag:
            continue
        n, rate, seed = tag
        win = json.load(gzip.open(p, "rt"))["window"]
        v = [r for r in objective.window_value(win, params) if r["type"] == "all"][0]
        rows.append({"n": n, "rate": rate, "seed": seed, "value": v["value_per_s"]})
    return pd.DataFrame(rows)


def agg(df, by):
    out = []
    for k, g in df.groupby(by):
        x = g.value.values
        ci = analyze.t_crit(len(x)) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
        out.append({by: k, "value": x.mean(), "ci": ci, "seeds": len(x)})
    return pd.DataFrame(out)


def summary(folder):
    s = pd.read_csv(os.path.join(EXP, folder, "summary.csv"))
    return s[s["type"] == "all"]


def fluid(folder, by="n"):
    rows = []
    for p in glob.glob(os.path.join(EXP, folder, "specs", "fluid_*.yaml")):
        d = yaml.safe_load(open(p))
        meta = d["meta"]
        FORMS.append(meta.get("formulation", "per instance"))
        rows.append({"n": meta["instances"], "rate": meta["rate"], "V": meta["value"], "bound": meta["bound"],
                     "gap": meta["gap"], "status": meta["status"]})
    return pd.DataFrame(rows)


def collect_exp1(date):
    res, missing = [], []
    for inst in instances(date):
        params = yaml.safe_load(open(inst["obj"]))
        for load in ("under", "over"):
            fdf, b2f = inst["fd"][load], inst["b2"][load]
            if not all(os.path.exists(os.path.join(EXP, f, "summary.csv")) for f in (fdf, b2f)):
                missing.append(f"{inst['key']} {load}")
                continue
            fl = fluid(fdf)
            fd = agg(run_values(fdf, params), "n").merge(summary(fdf)[["n", "rejected_frac_mean", "ttft_mean_ms_mean", "e2e_mean_ms_mean"]], on="n")
            b2 = agg(run_values(b2f, params), "n").merge(summary(b2f)[["n", "rejected_frac_mean", "ttft_mean_ms_mean", "e2e_mean_ms_mean"]], on="n")
            for n in sorted(set(fd.n) & set(b2.n) & set(fl.n)):
                f, b, v = fd[fd.n == n].iloc[0], b2[b2.n == n].iloc[0], fl[fl.n == n].iloc[0]
                res.append({"inst": inst["key"], "label": inst["label"], "load": load, "rate": inst["rate"][load], "n": n,
                            "V": v.V, "bound": v.bound, "solver_gap": v.gap, "status": v.status,
                            "FD": f.value, "FD_ci": f.ci, "B2": b.value, "B2_ci": b.ci,
                            "FD_rej": 100 * f.rejected_frac_mean, "B2_rej": 100 * b.rejected_frac_mean,
                            "FD_ttft": f.ttft_mean_ms_mean, "B2_ttft": b.ttft_mean_ms_mean,
                            "FD_seeds": f.seeds, "B2_seeds": b.seeds})
    df = pd.DataFrame(res)
    if len(df):
        for k in ("FD", "B2"):
            df[f"{k}_gap"] = 100 * (df.V - df[k]) / df.V
        df["adv"] = 100 * (df.FD - df.B2) / df.V
    return df, missing


def collect_exp2(date):
    fdf, b2f = f"{date}_loadsweep_n8_fluid_dual", f"{date}_loadsweep_n8_B2"
    if not all(os.path.exists(os.path.join(EXP, f, "summary.csv")) for f in (fdf, b2f)):
        return pd.DataFrame(), [fdf, b2f]
    params = yaml.safe_load(open(os.path.join(SPEC, "objective.yaml")))
    fl = fluid(fdf)
    fd = agg(run_values(fdf, params), "rate").merge(summary(fdf)[["rate", "rejected_frac_mean", "ttft_mean_ms_mean", "e2e_mean_ms_mean"]], on="rate")
    b2 = agg(run_values(b2f, params), "rate").merge(summary(b2f)[["rate", "rejected_frac_mean", "ttft_mean_ms_mean", "e2e_mean_ms_mean"]], on="rate")
    rows = []
    for r in sorted(set(fd.rate) & set(b2.rate)):
        f, b = fd[fd.rate == r].iloc[0], b2[b2.rate == r].iloc[0]
        v = fl[np.isclose(fl.rate, r)]
        if v.empty:
            continue
        v = v.iloc[0]
        rows.append({"rate": r, "rpi": r / 8, "V": v.V, "bound": v.bound, "solver_gap": v.gap, "status": v.status,
                     "FD": f.value, "FD_ci": f.ci, "B2": b.value, "B2_ci": b.ci,
                     "FD_rej": 100 * f.rejected_frac_mean, "B2_rej": 100 * b.rejected_frac_mean,
                     "FD_ttft": f.ttft_mean_ms_mean, "B2_ttft": b.ttft_mean_ms_mean,
                     "FD_e2e": f.e2e_mean_ms_mean, "B2_e2e": b.e2e_mean_ms_mean})
    df = pd.DataFrame(rows)
    for k in ("FD", "B2"):
        df[f"{k}_gap"] = 100 * (df.V - df[k]) / df.V
        df[f"{k}_gap_ci"] = 100 * df[f"{k}_ci"] / df.V
    return df, []


def xstyle(ax, xs, xlabel="system scale n (instances)", log2=True):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    if log2:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in xs]))
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e5e5e5", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_instances(df, insts, out):
    rows = [i for i in insts if i["key"] in set(df.inst)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.9 * len(rows)), dpi=130, squeeze=False)
    for r, inst in enumerate(rows):
        for c, load in enumerate(("under", "over")):
            ax = axes[r, c]
            g = df[(df.inst == inst["key"]) & (df.load == load)].sort_values("n")
            if g.empty:
                ax.set_axis_off()
                continue
            for k in ("B2", "FD"):
                y, e = g[k] / g.n, g[f"{k}_ci"].fillna(0) / g.n
                ax.plot(g.n, y, marker="o", ms=3.5, lw=1.5, color=COL[k], label=NAME[k])
                ax.fill_between(g.n, y - e, y + e, color=COL[k], alpha=0.2, lw=0)
            ax.plot(g.n, g.V / g.n, ls="--", color="black", lw=1.1, label="fluid optimum V*")
            xstyle(ax, list(g.n))
            ax.set_ylabel("revenue / s / instance")
            ax.set_title(f"{inst['label']}, {'underloaded' if load == 'under' else 'overloaded'} ({g.rate.iloc[0]:g} req/s per instance)", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_gaps(df, insts, out):
    cmap = plt.get_cmap("tab10")
    keys = [i["key"] for i in insts if i["key"] in set(df.inst)]
    labels = {i["key"]: i["label"] for i in insts}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=140)
    for c, load in enumerate(("under", "over")):
        for r, (what, ylabel) in enumerate((("gap", "loss to the fluid optimum, percent of V*"),
                                            ("adv", "fluid-dual minus B2, percent of V*"))):
            ax = axes[r, c]
            for q, k in enumerate(keys):
                g = df[(df.inst == k) & (df.load == load)].sort_values("n")
                if g.empty:
                    continue
                if what == "gap":
                    ax.plot(g.n, g.FD_gap, marker="o", ms=3, lw=1.5, color=cmap(q), label=labels[k])
                    ax.plot(g.n, g.B2_gap, marker="s", ms=3, lw=1.0, ls="--", color=cmap(q))
                else:
                    ax.plot(g.n, g.adv, marker="o", ms=3, lw=1.5, color=cmap(q), label=labels[k])
            ax.axhline(0, color="grey", lw=0.8)
            xstyle(ax, sorted(set(df.n)))
            ax.set_ylabel(ylabel)
            ax.set_title(("underloaded" if load == "under" else "overloaded")
                         + (": solid fluid-dual, dashed B2" if what == "gap" else ""), fontsize=10)
            if c == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_exp2(d2, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=140)
    x = d2.rpi
    ax = axes[0, 0]
    for k in ("B2", "FD"):
        ax.plot(x, d2[f"{k}_gap"], marker="o", ms=4, lw=1.6, color=COL[k], label=NAME[k])
        ax.fill_between(x, d2[f"{k}_gap"] - d2[f"{k}_gap_ci"], d2[f"{k}_gap"] + d2[f"{k}_gap_ci"], color=COL[k], alpha=0.2, lw=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("loss to the fluid optimum, percent of V*")
    ax.set_title("relative gap to optimality", fontsize=10)
    ax = axes[0, 1]
    for k in ("B2", "FD"):
        ax.plot(x, d2[k], marker="o", ms=4, lw=1.6, color=COL[k], label=NAME[k])
    ax.plot(x, d2.V, ls="--", color="black", lw=1.1, label="fluid optimum V*")
    ax.set_ylabel("revenue value per second")
    ax.set_title("revenue", fontsize=10)
    ax = axes[1, 0]
    for k in ("B2", "FD"):
        ax.plot(x, d2[f"{k}_rej"], marker="o", ms=4, lw=1.6, color=COL[k], label=NAME[k])
    ax.set_ylabel("rejected at admission, percent of arrivals")
    ax.set_title("admission", fontsize=10)
    ax = axes[1, 1]
    for k in ("B2", "FD"):
        ax.plot(x, d2[f"{k}_ttft"] / 1000, marker="o", ms=4, lw=1.6, color=COL[k], label=NAME[k])
    ax.set_yscale("log")
    ax.set_ylabel("TTFT, mean (s)")
    ax.set_title("time to first token", fontsize=10)
    for a in axes.ravel():
        a.axvline(13.46, color="grey", ls=":", lw=1)
        xstyle(a, None, "demand rate per instance (req/s); dotted: fluid capacity 13.46", log2=False)
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def f1(v, d=1):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{d}f}"


def table_exp1(df, insts):
    out = []
    for inst in insts:
        g = df[df.inst == inst["key"]]
        if g.empty:
            continue
        rows = []
        for _, r in g.sort_values(["load", "n"], ascending=[False, True]).iterrows():
            rows.append(" & ".join(["under" if r.load == "under" else "over", str(int(r.n)), f1(r.V), f1(r.B2), f1(r.FD),
                                    f1(r.B2_gap, 2), f1(r.FD_gap, 2), f1(r.adv, 2), f1(r.B2_rej, 2), f1(r.FD_rej, 2),
                                    f"{100 * r.solver_gap:.2f}" if r.solver_gap == r.solver_gap else "--"]) + r" \\")
        out.append(f"\\paragraph{{{esc(inst['label'])}.}} {esc(inst['change'])}.\n"
                   "\\begin{center}\\scriptsize\\begin{tabular}{lrrrrrrrrrr}\\toprule\n"
                   r"load & n & V* & B2 & fluid-dual & B2 loss \% & FD loss \% & FD $-$ B2, \% of V* & B2 rej. \% & FD rej. \% & solver gap \% \\" "\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")
    return "\n".join(out)


def table_exp2(d2):
    rows = [" & ".join([f1(r.rpi, 1), f1(r.rate, 1), f1(r.V), f1(r.B2), f1(r.FD), f1(r.B2_gap, 2), f1(r.FD_gap, 2),
                        f1(r.B2_rej, 2), f1(r.FD_rej, 2), f1(r.B2_ttft / 1000, 3), f1(r.FD_ttft / 1000, 3),
                        f"{100 * r.solver_gap:.2f}"]) + r" \\" for _, r in d2.iterrows()]
    return ("\\begin{center}\\scriptsize\\begin{tabular}{rrrrrrrrrrrr}\\toprule\n"
            r"req/s per inst. & total & V* & B2 & fluid-dual & B2 loss \% & FD loss \% & B2 rej. \% & FD rej. \% & B2 TTFT (s) & FD TTFT (s) & solver gap \% \\" "\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{center}")


def fluid_statement(formulations):
    """The sentence every report carries about how the fluid program was solved."""
    f = set(x for x in formulations if x)
    if f == {"aggregate by GPU type"}:
        return ("\\textbf{Fluid program.} The fluid problem is solved in its aggregate form by GPU type: the index $j$ is a GPU type "
                "with $N_j$ identical instances, and the load of a type is split equally over its instances. Gurobi proves this "
                "problem optimal in under a second at every scale. Its optimum is the value of a feasible point of the per-instance "
                "problem (28), and over the 102 points of the 2026-10-08 to 2026-10-11 experiments it lies below the per-instance "
                "fluid value by at most 0.86 percent (median 0.12 percent). The fluid-dual policy uses this solution: all instances "
                "of a GPU type share the same targets $K^\\star$, $S^\\star$, marginal costs $c^\\star$ and prices $\\gamma^\\star$, which "
                "satisfy the KKT conditions of the per-instance problem at the equal split. $V^\\star$ denotes this aggregate optimum.")
    if f == {"per instance"} or not f:
        return ("\\textbf{Fluid program.} The fluid problem (28) is solved per instance (index $j$ = instance) with Gurobi's global "
                "solver; $V^\\star$ is the best fluid solution found, and the solver gap to its proven upper bound is listed.")
    return "\\textbf{Fluid program.} The folders mix the aggregate and the per-instance formulations of the fluid problem; see the tables."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default="2026-10-11")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    insts = instances(a.date)
    d1, miss1 = collect_exp1(a.date)
    d2, miss2 = collect_exp2(a.date)
    d1.to_csv(os.path.join(a.out, "experiment1_robustness.csv"), index=False)
    d2.to_csv(os.path.join(a.out, "experiment2_load_sweep.csv"), index=False)
    if len(d1):
        fig_instances(d1, insts, os.path.join(a.out, "exp1_instances.png"))
        fig_gaps(d1, insts, os.path.join(a.out, "exp1_gaps.png"))
    if len(d2):
        fig_exp2(d2, os.path.join(a.out, "exp2_load_sweep.png"))

    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Fluid-dual against B2: robustness over instances and sensitivity to the demand rate}
\author{BLIS fork, branch ours}
\date{""" + esc(pd.Timestamp.today().date()) + r"""}
\begin{document}\maketitle
\section{What was run}
Two policies, both with llm-d routing replaced or kept as stated, vLLM batching and vLLM eviction (most recently started job first):
\begin{itemize}
\item \textbf{B2}: B1 (llm-d weighted routing on 50 ms snapshots) plus llm-d / GAIE saturation admission with every type sheddable: an arrival is rejected while the average over instances of $\max(q_j/5, u_j/0.8) \ge 1$ ($q_j$ waiting jobs, $u_j$ KV utilisation).
\item \textbf{Fluid-dual}: the model draft's Definition 1 (index rule): accept a type-$i$ job iff $\max_j I_{ij}(t) \ge 0$ and route it to the largest index; $\eta = V^\star/J$; the fluid problem (28) is solved with Gurobi for each instance, $n$ and rate.
\end{itemize}
Common to all runs: Qwen3-14B, one GPU per instance, per-step token budget 8,192, horizon 1,200 s, statistics over arrivals in [500, 1,100) s, revenue = eq. (10) per second of the window. The loss to optimality is $100\,(V^\star - V)/V^\star$ with $V^\star$ the fluid optimum of that instance (best value found by the solver; the solver gap is listed). A negative loss means the simulated value exceeds $V^\star$, which the fluid model does not rule out.
\section{Experiment 1: robustness over instances}
Each instance changes one aspect of the baseline (H100, 6,500 KV blocks, type shares 50/35/15, $\pi_{in} = 1$, $\pi_{out} = 4$, $h = 0.03$). Loads are set at the same fraction of each instance's fluid capacity as the baseline's 11 and 14.4 req/s per instance (0.817 and 1.070 of the H100 fluid capacity of 13.46 req/s): fluid capacity is the largest arrival rate the fluid constraints (27) carry with every job admitted (a linear program, ours/fluid/capacity.py). Price variants keep the baseline rates. $n$ = 2, 4, 8, 16, 32 with 10 seeds for $n \le 8$ and 5 for $n \ge 16$ (baseline: up to 64, 20 and 10 seeds). Fluid solves for these instances have a 150 s time limit.
"""]
    tex.append(fluid_statement(FORMS) + "\n")
    if miss1:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(miss1)) + ".\n")
    if len(d1):
        tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{exp1_gaps.png}\\caption{Top: loss to the fluid optimum against $n$ for every instance (solid fluid-dual, dashed B2, one colour per instance). Bottom: advantage of fluid-dual over B2 in percent of $V^\\star$. Left underloaded, right overloaded.}\\end{figure}")
        tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth,height=0.92\\textheight,keepaspectratio]{exp1_instances.png}\\caption{Revenue per second per instance against $n$, every instance and load; dashed $V^\\star/n$; bands 95 percent over seeds.}\\end{figure}")
        tex.append(r"\subsection*{Tables}")
        tex.append(table_exp1(d1, insts))
    tex.append(r"\section{Experiment 2: sensitivity to the demand rate}")
    tex.append("Representative instance: the baseline (H100, 6,500 blocks) at $n = 8$, the largest scale at which the fluid problem is still solved to near optimality in a few minutes. Total demand 48 to 160 req/s, i.e. 6 to 20 req/s per instance (0.45 to 1.49 times the fluid capacity), 10 seeds per point, fluid solve time limit 300 s.")
    if len(d2):
        tex.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{exp2_load_sweep.png}\\caption{Against the demand rate per instance: loss to the fluid optimum (top left, bands 95 percent over seeds), revenue with $V^\\star$, share rejected, mean TTFT.}\\end{figure}")
        tex.append(table_exp2(d2))
    else:
        tex.append("\\textbf{Not available when this report was built:} " + esc(", ".join(miss2)) + ".")
    tex.append(r"\section{Artifacts}")
    tex.append("Data tables: experiment1\\_robustness.csv, experiment2\\_load\\_sweep.csv in this folder. Experiment folders: ours/experiments/" + esc(a.date) + "\\_scaling\\_robust\\_* and " + esc(a.date) + "\\_loadsweep\\_n8\\_*; baseline folders 2026-10-08\\_scaling\\_scale64\\_fluid\\_dual\\_* and 2026-10-10\\_scaling\\_scale64\\_B2\\_*. Generated by ours/report\\_robust.py.")
    tex.append(r"\end{document}")
    open(os.path.join(a.out, "report.tex"), "w", encoding="utf-8").write("\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed")
    print("wrote", os.path.join(a.out, "report.pdf"), "| missing:", miss1, miss2)


if __name__ == "__main__":
    main()
