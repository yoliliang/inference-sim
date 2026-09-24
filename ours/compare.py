#!/usr/bin/env python3
"""ours/compare.py <name> --base <folder> --cand <folder> [--cand <folder> ...] [--label ...]
                          [--headline e2e_mean_ms]

Paired comparison of one or more candidate experiments against a baseline experiment that
was run with the same grid (n, rate) and the same seeds (common random numbers). Nothing
is simulated here: the inputs are the summary_runs.csv, objective_runs.csv and the per-path
state csv files that sweep.py, analyze.py and objective.py already wrote. Output folder:
ours/experiments/<date>_compare_<name>/

    compare_runs.csv          per (n, rate, seed, type): candidate minus baseline, every metric
    compare.csv               per (n, rate, type): mean difference, 95 percent band (Student t
                              over seeds), baseline mean, candidate mean, relative change
    objective_vs_scale.png    revenue objective against the grid: levels for every policy, and
                              the paired gain per unit of scale by request type
    paper_metrics.png         the metrics the policy papers report (Kim et al. 2024, Fig. 9):
                              mean E2E latency, average TPOT, preemptions per arrival, KV
                              memory usage; rows = metrics, columns = request types, lines =
                              policies, x = the grid
    headline_improvement.png  relative change of the paper's headline metric (default mean
                              E2E latency) by request type at every grid point, with bands
    manifest.json             the folders compared, their fork commits and profiles
    README.md                 question and the generated result tables

Metric definitions added here (not in summary_runs.csv):
    tpot_ms      average time per output token over the window, per type: (mean E2E minus mean
                 TTFT) divided by (mean output length minus 1), the ratio of totals
    kv_util_pct  KV cache in use, percent of capacity, averaged over the window and over the
                 instances of one path (from the state csv; defined for type all only)

Pairing is on (n, rate, seed, type). Seeds present in only one folder are dropped and
reported. House style: no em dashes or en dashes.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import parse_tag, t_crit  # noqa: E402

KEYS = ["n", "rate", "seed", "type"]
# (column, label, unit scale, unit label, better direction)
METRICS = [
    ("value_per_s", "objective value per second, whole system", 1, "", "+"),
    ("output_tok_per_s", "output tokens per second, whole system", 1, "", "+"),
    ("e2e_mean_ms", "mean end-to-end latency", 1e-3, "s", "-"),
    ("tpot_ms", "average TPOT (time per output token)", 1, "ms", "-"),
    ("ttft_mean_ms", "mean TTFT", 1e-3, "s", "-"),
    ("ttft_p99_ms", "TTFT p99", 1e-3, "s", "-"),
    ("delay_mean_ms", "mean queue wait", 1e-3, "s", "-"),
    ("preempt_per_arrival", "preemptions per arrival", 1, "", "-"),
    ("preempt_per_s", "preemptions per second, whole system", 1, "", "-"),
    ("wasted_tok_per_arrival", "wasted tokens per arrival", 1, "", "-"),
    ("kv_util_pct", "KV memory usage, percent of capacity", 1, "percent", ""),
    ("batch_mean", "mean running batch size per instance", 1, "", ""),
    ("unfinished_frac", "unfinished at horizon, fraction of arrivals", 100, "percent", "-"),
]
SPEC = {m[0]: m for m in METRICS}
TYPES = ["all", "type1", "type2", "type3"]
TYPE_COLORS = {"all": "black", "type1": "tab:blue", "type2": "tab:orange", "type3": "tab:green"}
TYPE_WORDS = {"all": "all types", "type1": "type1 (short chat)", "type2": "type2 (RAG)", "type3": "type3 (long generation)"}
POLICY_STYLES = [("C0", "-"), ("C3", "--"), ("C2", "-."), ("C4", ":")]
BASE_LABEL = "B1 (vLLM default)"


def commit_of(manifest):
    """sweep.py records the fork commit under one of these keys depending on its version."""
    for k in ("fork_commit", "commit", "git_commit"):
        if manifest.get(k):
            return manifest[k]
    return "unknown"


def kv_util_by_path(exp, manifest):
    """Mean KV utilisation (percent) over the window and the instances, per (n, rate, seed)."""
    warm = float(manifest.get("warmup_s") or 0)
    horizon = float(manifest.get("horizon_s") or 0)
    tail = float(manifest.get("tail_s") or 0)
    end = horizon - tail if horizon else None
    rows = []
    for f in glob.glob(os.path.join(exp, "runs", "*state.csv")):
        base = os.path.basename(f).replace("_state.csv", "").replace(".json.gz", "").replace(".json", "")
        parsed = parse_tag(base + ".json.gz", manifest)
        if not parsed:
            continue
        n, rate, seed = parsed
        d = pd.read_csv(f, usecols=["clock_us", "kv_utilization", "batch_size"])
        t = d.clock_us / 1e6
        w = d[(t >= warm) & ((t < end) if end else True)]
        if len(w):
            rows.append({"n": n, "rate": rate, "seed": seed, "type": "all", "kv_util_pct": 100 * w.kv_utilization.mean(),
                         "batch_mean": w.batch_size.mean()})
    return pd.DataFrame(rows) if rows else None


def load_runs(exp):
    s = pd.read_csv(os.path.join(exp, "summary_runs.csv"))
    m = json.load(open(os.path.join(exp, "manifest.json")))
    op = os.path.join(exp, "objective_runs.csv")
    if os.path.exists(op):
        o = pd.read_csv(op)[KEYS + ["value_per_s", "reward_per_s", "cost_per_s"]]
        s = s.merge(o, on=KEYS, how="left")
    # average TPOT: decode time per generated token, ratio of window totals
    mean_o = s.output_tok_per_s * s.window_s / s.completed.replace(0, np.nan)
    s["tpot_ms"] = (s.e2e_mean_ms - s.ttft_mean_ms) / (mean_o - 1).where(mean_o > 1)
    kv = kv_util_by_path(exp, m)
    if kv is not None:
        s = s.merge(kv, on=KEYS, how="left")
    else:
        s["kv_util_pct"] = np.nan
        s["batch_mean"] = np.nan
    return s, m


def paired(base, cand):
    b = base.set_index(KEYS)
    c = cand.set_index(KEYS)
    common = b.index.intersection(c.index)
    lost_b, lost_c = len(b.index.difference(common)), len(c.index.difference(common))
    cols = [m for m in SPEC if m in b.columns and m in c.columns]
    d = (c.loc[common, cols] - b.loc[common, cols]).reset_index()
    return d, cols, lost_b, lost_c


def band_table(diff, base, cand, cols):
    rows = []
    for (n, rate, t), g in diff.groupby(["n", "rate", "type"]):
        row = {"n": n, "rate": rate, "type": t, "n_seeds": len(g)}
        bm = base[(base.n == n) & (base.rate == rate) & (base.type == t)]
        cm = cand[(cand.n == n) & (cand.rate == rate) & (cand.type == t)]
        for c in cols:
            g_c = g[c].dropna()
            k = len(g_c)
            mean = g_c.mean() if k else np.nan
            band = t_crit(k) * g_c.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
            b0 = bm[c].mean()
            row[f"{c}_base"] = b0
            row[f"{c}_cand"] = cm[c].mean()
            row[f"{c}_diff"] = mean
            row[f"{c}_band"] = band
            # relative to the size of the baseline mean, so a gain on a negative objective reads positive
            row[f"{c}_rel"] = mean / abs(b0) if not pd.isna(b0) and b0 != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def x_axis(df):
    """Scaling experiments vary n at fixed rate per instance; rate experiments vary rate."""
    if df.n.nunique() > 1:
        return "n", "system scale n"
    return "rate", "total arrival rate, req/s"


def _style_x(ax, xcol, xlabel):
    ax.set_xlabel(xlabel)
    if xcol == "n":
        ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)


def _level_series(df, xcol, col, t="all"):
    g = df[df.type == t].groupby(xcol)[col]
    x = g.mean().index.values
    m = g.mean().values
    k = g.count().values
    sd = g.std(ddof=1).fillna(0).values
    band = np.array([t_crit(max(int(kk), 2)) for kk in k]) * sd / np.sqrt(np.maximum(k, 1))
    return x, m, band


def objective_plot(base_s, cands, labels, tables, out_png, title):
    """Left: objective per second, whole system, every policy. Right: paired gain per unit of
    scale by request type (candidate minus baseline, divided by n)."""
    xcol, xlabel = x_axis(base_s)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for i, (df, lab) in enumerate([(base_s, BASE_LABEL)] + list(zip(cands, labels))):
        col, ls = POLICY_STYLES[i % len(POLICY_STYLES)]
        x, m, b = _level_series(df, xcol, "value_per_s")
        ax.plot(x, m, ls, marker="o", ms=3.5, color=col, label=lab)
        ax.fill_between(x, m - b, m + b, color=col, alpha=0.2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("objective value per second, whole system")
    ax.set_title("Revenue objective, levels")
    _style_x(ax, xcol, xlabel)
    ax.legend(fontsize=8)
    ax = axes[1]
    for i, (tab, lab) in enumerate(zip(tables, labels)):
        for t in TYPES:
            g = tab[tab.type == t].sort_values(xcol)
            if g.empty:
                continue
            scale = g["n"].values if xcol == "n" else 1.0
            m, b = g["value_per_s_diff"].values / scale, g["value_per_s_band"].values / scale
            ls = "-" if len(tables) == 1 else POLICY_STYLES[i % len(POLICY_STYLES)][1]
            ax.plot(g[xcol], m, ls, marker="o", ms=3.5, color=TYPE_COLORS[t],
                    label=TYPE_WORDS[t] if len(tables) == 1 else f"{lab}, {TYPE_WORDS[t]}")
            ax.fill_between(g[xcol], m - b, m + b, color=TYPE_COLORS[t], alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("gain in objective value per second" + (" per unit of scale" if xcol == "n" else ""))
    ax.set_title("Candidate minus baseline, by request type")
    _style_x(ax, xcol, xlabel)
    ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


PAPER_ROWS = [  # rows of the Kim et al. Figure 9 grid, in their order
    ("e2e_mean_ms", "latency: mean E2E, s", 1e-3),
    ("tpot_ms", "avg TPOT, ms", 1),
    ("preempt_per_arrival", "preemptions per arrival", 1),
    ("kv_util_pct", "memory usage: KV in use, percent", 1),
    ("batch_mean", "avg batch size per instance", 1),
]


def paper_metrics_plot(base_s, cands, labels, out_png, title):
    """Rows: the four metrics of Kim et al. (2024) Figure 9. Columns: all types, then each type.
    Lines: the policies. Memory usage is a property of the instance, so it is drawn once."""
    xcol, xlabel = x_axis(base_s)
    nrow, ncol = len(PAPER_ROWS), len(TYPES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow))
    for r, (col, ylabel, scale) in enumerate(PAPER_ROWS):
        for c, t in enumerate(TYPES):
            ax = axes[r, c]
            if col in ("kv_util_pct", "batch_mean") and t != "all":
                ax.axis("off")
                if c == 1:
                    ax.text(0.0, 0.5, "per instance, not per request type", fontsize=9, transform=ax.transAxes)
                continue
            for i, (df, lab) in enumerate([(base_s, BASE_LABEL)] + list(zip(cands, labels))):
                if col not in df.columns or df[df.type == t][col].isna().all():
                    continue
                pc, ls = POLICY_STYLES[i % len(POLICY_STYLES)]
                x, m, b = _level_series(df, xcol, col, t)
                ax.plot(x, m * scale, ls, marker="o", ms=3, color=pc, label=lab)
                ax.fill_between(x, (m - b) * scale, (m + b) * scale, color=pc, alpha=0.2)
            if r == 0:
                ax.set_title(TYPE_WORDS[t], fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel)
            if col == "kv_util_pct":
                ax.set_ylim(0, 105)
            _style_x(ax, xcol, xlabel)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def headline_plot(tables, labels, headline, out_png, title):
    """Grouped bars: relative change of the headline metric, candidate against baseline, in
    percent, one group per grid point, one bar per request type, error bars = 95 percent band."""
    spec = SPEC[headline]
    xcol, xlabel = x_axis(tables[0])
    fig, axes = plt.subplots(1, len(tables), figsize=(7.5 * len(tables), 4.6), squeeze=False)
    for ax, tab, lab in zip(axes[0], tables, labels):
        xs = sorted(tab[xcol].unique())
        width = 0.8 / len(TYPES)
        for j, t in enumerate(TYPES):
            g = tab[tab.type == t].set_index(xcol).reindex(xs)
            rel = 100 * g[f"{headline}_rel"].values
            band = 100 * g[f"{headline}_band"].values / g[f"{headline}_base"].abs().values
            pos = np.arange(len(xs)) + (j - (len(TYPES) - 1) / 2) * width
            ax.bar(pos, rel, width, yerr=band, capsize=2, color=TYPE_COLORS[t], alpha=0.85, label=TYPE_WORDS[t])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel(xlabel)
        better = {"+": "higher is better", "-": "lower is better", "": ""}[spec[4]]
        ax.set_ylabel(f"{spec[1].split(' (')[0]}, change against B1, percent")
        ax.set_title(f"{lab} against {BASE_LABEL}" + (f"; {better}" if better else ""), fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(title.split(":")[0] + ": headline metric of the paper", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fmt(v, scale=1, unit=""):
    if pd.isna(v):
        return ""
    v = v * scale
    s = f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return s + (f" {unit}" if unit else "")


def readme_table(tab, label, cols, t="all"):
    """Markdown table of the paired differences per grid point for one request type."""
    xcol, _ = x_axis(tab)
    a = tab[tab.type == t].sort_values(xcol)
    lines = [f"### {label} minus B1, {TYPE_WORDS[t]}, mean over paired seeds (band = 95 percent half-width)", "",
             "| " + xcol + " | " + " | ".join(SPEC[c][1].split(",")[0].split(":")[0] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for _, r in a.iterrows():
        cells = []
        for c in cols:
            spec = SPEC[c]
            if pd.isna(r[c + "_diff"]):
                cells.append("")
                continue
            cells.append(f"{fmt(r[c + '_diff'], spec[2], spec[3])} +/- {fmt(r[c + '_band'], spec[2])}"
                         + (f" ({r[c + '_rel'] * 100:+.1f} percent)" if not pd.isna(r[c + '_rel']) else ""))
        lines.append(f"| {r[xcol]:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--label", action="append", default=None,
                    help="one per --cand; default: the candidate's profile name")
    ap.add_argument("--headline", default="e2e_mean_ms", choices=list(SPEC),
                    help="the metric the paper claims to improve (default mean E2E latency)")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    a = ap.parse_args()

    out = os.path.join(HERE, "experiments", f"{a.date}_compare_{a.name}")
    os.makedirs(out, exist_ok=True)
    base_s, base_m = load_runs(a.base)
    cands, cand_ms, labels, tables, all_runs = [], [], [], [], []
    for i, c in enumerate(a.cand):
        s, m = load_runs(c)
        lab = (a.label[i] if a.label and i < len(a.label) else m.get("profile", os.path.basename(c)))
        d, cols, lb, lc = paired(base_s, s)
        if lb or lc:
            print(f"warning: {lb} baseline rows and {lc} candidate rows had no partner and were dropped ({lab})")
        d.insert(0, "candidate", lab)
        all_runs.append(d)
        tab = band_table(d, base_s, s, cols)
        tab.insert(0, "candidate", lab)
        tables.append(tab)
        cands.append(s)
        cand_ms.append(m)
        labels.append(lab)
    runs = pd.concat(all_runs, ignore_index=True)
    runs.to_csv(os.path.join(out, "compare_runs.csv"), index=False)
    pd.concat(tables, ignore_index=True).to_csv(os.path.join(out, "compare.csv"), index=False)

    cols = [c for c in SPEC if c in runs.columns]
    name = os.path.basename(out)
    objective_plot(base_s, cands, labels, tables, os.path.join(out, "objective_vs_scale.png"),
                   f"{name}: revenue objective, B1 against " + ", ".join(labels))
    paper_metrics_plot(base_s, cands, labels, os.path.join(out, "paper_metrics.png"),
                       f"{name}: the metrics of Kim et al. (2024) Fig. 9, by request type")
    headline_plot(tables, labels, a.headline, os.path.join(out, "headline_improvement.png"),
                  f"{name}: headline metric of the paper, relative change with 95 percent bands")

    json.dump({
        "experiment": "compare",
        "baseline": {"folder": os.path.abspath(a.base), "profile": base_m.get("profile"), "fork_commit": commit_of(base_m)},
        "candidates": [{"folder": os.path.abspath(c), "label": l, "profile": m.get("profile"), "fork_commit": commit_of(m)}
                       for c, l, m in zip(a.cand, labels, cand_ms)],
        "pairing": KEYS, "metrics": cols, "headline": a.headline,
        "grid": {"x": x_axis(base_s)[0], "instances": sorted(base_s.n.unique().tolist()),
                 "rates": sorted(base_s.rate.unique().tolist()), "seeds": sorted(base_s.seed.unique().tolist())},
        "window": {k: base_m.get(k) for k in ("horizon_s", "warmup_s", "tail_s")},
    }, open(os.path.join(out, "manifest.json"), "w"), indent=2)

    md_path = os.path.join(out, "README.md")
    if not os.path.exists(md_path):
        q = (f"# compare_{a.name} ({a.date})\n\n"
             f"Paired comparison of {', '.join(labels)} against the B1 baseline on the same grid and the same seeds "
             f"(common random numbers). Baseline folder: {os.path.basename(a.base)}. Candidate folders: "
             + ", ".join(os.path.basename(c) for c in a.cand) + ". Every number is candidate minus baseline on one "
             "sample path, then averaged over seeds; the band is the 95 percent half-width from the Student t "
             "distribution over the paired differences.\n\n## Result\n\n")
        q += "\n\n".join(readme_table(t, l, cols, ty) for t, l in zip(tables, labels) for ty in TYPES)
        open(md_path, "w", encoding="utf-8").write(q + "\n")
    print("compare written to", out)
    print(readme_table(tables[0], labels[0], cols))


if __name__ == "__main__":
    main()
