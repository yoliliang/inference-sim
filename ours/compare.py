#!/usr/bin/env python3
"""ours/compare.py <name> --base <folder> --cand <folder> [--cand <folder> ...] [--label ...]

Paired comparison of one or more candidate experiments against a baseline experiment that
was run with the same grid (n, rate) and the same seeds (common random numbers). Nothing
is simulated here: the inputs are the summary_runs.csv and objective_runs.csv files that
sweep.py, analyze.py and objective.py already wrote. Output folder:
ours/experiments/<date>_compare_<name>/

    compare_runs.csv   per (n, rate, seed, type): candidate minus baseline for every metric
    compare.csv        per (n, rate, type): mean difference, 95 percent band (Student t over
                       seeds), baseline mean, candidate mean, relative change
    compare_levels.png the key statistics against n (or rate) for baseline and candidates,
                       mean with shaded band
    compare_diffs.png  the paired differences against n (or rate), mean with shaded band,
                       zero line
    manifest.json      the folders compared, their fork commits and profiles
    README.md          question and a generated summary table; edit the interpretation

Pairing is on (n, rate, seed, type). Seeds present in only one folder are dropped and
reported. House style: no em dashes or en dashes.
"""
import argparse
import datetime as dt
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
from analyze import t_crit  # noqa: E402

KEYS = ["n", "rate", "seed", "type"]
METRICS = [  # (column, label, unit scale, unit label)
    ("value_per_s", "objective value per second, whole system", 1, ""),
    ("output_tok_per_s", "output tokens per second, whole system", 1, ""),
    ("preempt_per_s", "evictions per second, whole system", 1, ""),
    ("wasted_tok_per_arrival", "wasted tokens per arrival", 1, ""),
    ("delay_mean_ms", "mean queue wait", 1e-3, "s"),
    ("ttft_p99_ms", "time to first token p99", 1e-3, "s"),
    ("e2e_mean_ms", "mean end-to-end time", 1e-3, "s"),
    ("unfinished_frac", "unfinished at horizon, fraction of arrivals", 100, "percent"),
]
TYPE_COLORS = {"all": "black", "type1": "tab:blue", "type2": "tab:orange", "type3": "tab:green"}


def commit_of(manifest):
    """sweep.py records the fork commit under one of these keys depending on its version."""
    for k in ("fork_commit", "commit", "git_commit"):
        if manifest.get(k):
            return manifest[k]
    return "unknown"


def load_runs(exp):
    s = pd.read_csv(os.path.join(exp, "summary_runs.csv"))
    op = os.path.join(exp, "objective_runs.csv")
    if os.path.exists(op):
        o = pd.read_csv(op)[KEYS + ["value_per_s", "reward_per_s", "cost_per_s"]]
        s = s.merge(o, on=KEYS, how="left")
    m = json.load(open(os.path.join(exp, "manifest.json")))
    return s, m


def paired(base, cand):
    b = base.set_index(KEYS)
    c = cand.set_index(KEYS)
    common = b.index.intersection(c.index)
    lost_b, lost_c = len(b.index.difference(common)), len(c.index.difference(common))
    cols = [m for m, *_ in METRICS if m in b.columns and m in c.columns]
    d = (c.loc[common, cols] - b.loc[common, cols]).reset_index()
    return d, cols, lost_b, lost_c


def band_table(diff, base, cand, cols):
    rows = []
    for (n, rate, t), g in diff.groupby(["n", "rate", "type"]):
        k = len(g)
        row = {"n": n, "rate": rate, "type": t, "n_seeds": k}
        bm = base[(base.n == n) & (base.rate == rate) & (base.type == t)]
        cm = cand[(cand.n == n) & (cand.rate == rate) & (cand.type == t)]
        for c in cols:
            mean = g[c].mean()
            band = t_crit(k) * g[c].std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
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


def levels_plot(base_s, cands, labels, out_png, title, cols):
    xcol, xlabel = x_axis(base_s)
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))
    axes = axes.flatten()
    styles = ["-", "--", "-.", ":"]
    for ax, c in zip(axes, cols):
        spec = next(m for m in METRICS if m[0] == c)
        scale, unit = spec[2], spec[3]
        for i, (df, lab) in enumerate([(base_s, "B1 baseline")] + list(zip(cands, labels))):
            g = df[df.type == "all"].groupby(xcol)[c]
            x = g.mean().index.values
            m = g.mean().values * scale
            k = g.count().values
            band = np.array([t_crit(kk) for kk in k]) * g.std(ddof=1).fillna(0).values / np.sqrt(k) * scale
            ax.plot(x, m, styles[i % 4], marker="o", ms=3, color=f"C{i}", label=lab)
            ax.fill_between(x, m - band, m + band, color=f"C{i}", alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(spec[1] + (f", {unit}" if unit else ""))
        if xcol == "n":
            ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def diffs_plot(tables, labels, out_png, title, cols):
    xcol, xlabel = x_axis(tables[0])
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))
    axes = axes.flatten()
    for ax, c in zip(axes, cols):
        spec = next(m for m in METRICS if m[0] == c)
        scale, unit = spec[2], spec[3]
        for i, (tab, lab) in enumerate(zip(tables, labels)):
            for t, col in TYPE_COLORS.items():
                g = tab[tab.type == t].sort_values(xcol)
                if g.empty:
                    continue
                m, b = g[f"{c}_diff"].values * scale, g[f"{c}_band"].values * scale
                ls = "-" if len(tables) == 1 else ["-", "--", "-.", ":"][i % 4]
                ax.plot(g[xcol], m, ls, marker="o", ms=3, color=col,
                        label=(t if len(tables) == 1 else f"{lab} {t}"))
                ax.fill_between(g[xcol], m - b, m + b, color=col, alpha=0.15)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("candidate minus baseline: " + spec[1] + (f", {unit}" if unit else ""))
        if xcol == "n":
            ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fmt(v, scale=1, unit=""):
    if pd.isna(v):
        return ""
    v = v * scale
    s = f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return s + (f" {unit}" if unit else "")


def readme_table(tab, label, cols):
    """Markdown table of the all-types paired differences per grid point."""
    xcol, _ = x_axis(tab)
    a = tab[tab.type == "all"].sort_values(xcol)
    lines = [f"### {label} minus B1, all types, mean over paired seeds (band = 95 percent half-width)", "",
             "| " + xcol + " | " + " | ".join(next(m for m in METRICS if m[0] == c)[1].split(",")[0] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for _, r in a.iterrows():
        cells = []
        for c in cols:
            spec = next(m for m in METRICS if m[0] == c)
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

    cols = [c for c in [m[0] for m in METRICS] if c in runs.columns]
    title = f"{os.path.basename(out)}: {base_m.get('profile', 'baseline')} against " + ", ".join(labels)
    levels_plot(base_s, cands, labels, os.path.join(out, "compare_levels.png"), title + " (levels)", cols)
    diffs_plot(tables, labels, os.path.join(out, "compare_diffs.png"), title + " (paired differences)", cols)

    json.dump({
        "experiment": "compare",
        "baseline": {"folder": os.path.abspath(a.base), "profile": base_m.get("profile"), "fork_commit": commit_of(base_m)},
        "candidates": [{"folder": os.path.abspath(c), "label": l, "profile": m.get("profile"), "fork_commit": commit_of(m)}
                       for c, l, m in zip(a.cand, labels, cand_ms)],
        "pairing": KEYS, "metrics": cols,
    }, open(os.path.join(out, "manifest.json"), "w"), indent=2)

    md_path = os.path.join(out, "README.md")
    if not os.path.exists(md_path):
        q = (f"# compare_{a.name} ({a.date})\n\n"
             f"Paired comparison of {', '.join(labels)} against the B1 baseline on the same grid and the same seeds "
             f"(common random numbers). Baseline folder: {os.path.basename(a.base)}. Candidate folders: "
             + ", ".join(os.path.basename(c) for c in a.cand) + ". Every number is candidate minus baseline on one "
             "sample path, then averaged over seeds; the band is the 95 percent half-width from the Student t "
             "distribution over the paired differences.\n\n## Result\n\n")
        q += "\n\n".join(readme_table(t, l, cols) for t, l in zip(tables, labels))
        q += "\n\n## Interpretation\n\n(to write)\n"
        open(md_path, "w", encoding="utf-8").write(q)
    print("compare written to", out)
    print(readme_table(tables[0], labels[0], cols))


if __name__ == "__main__":
    main()
