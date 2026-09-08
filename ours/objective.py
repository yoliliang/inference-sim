#!/usr/bin/env python3
"""ours/objective.py <experiment folder> [--params ours/specs/objective.yaml]

Revenue management shell over the per-request records of every run in an
experiment folder. For each run and the steady-state window [warmup_s, horizon_s - tail_s):

    value = sum over completed  (pi_in_k P + pi_out_k o - h_k (e - a))
          - sum over unfinished  h_k (horizon - a)          lower bound on their cost
          + 0 for rejected

reported as value per second of window, decomposed into reward and congestion cost,
per type and in total, then averaged across seeds with a 95 percent t CI.

Outputs in the experiment folder:
    objective_runs.csv   per (rate, seed, type)
    objective.csv        per (rate, type): across-seed mean and CI
    objective.png        value per second, reward and congestion cost against rate
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402

TYPES = analyze.TYPES
COLORS = analyze.TYPE_COLORS


def run_value(df, params, warm, end, horizon):
    w = df[(df.arrived_at >= warm) & (df.arrived_at < end)]
    rows = []
    for t in TYPES:
        p = params["types"][t]
        g = w[w.tenant_id == t]
        done = g[g.status == "completed"]
        unf = g[g.status == "unfinished"]
        reward = (p["pi_in"] * done.num_prefill_tokens.sum() + p["pi_out"] * done.num_decode_tokens.sum()) / 1000.0
        cost_done = p["h"] * (done.e2e_ms / 1000.0).sum()
        cost_unf = p["h"] * (horizon - unf.arrived_at).sum() if len(unf) else 0.0
        rows.append({
            "type": t, "arrived": len(g), "completed": len(done),
            "rejected": int((g.status == "rejected").sum()), "unfinished": len(unf),
            "reward_per_s": reward / (end - warm),
            "cost_per_s": (cost_done + cost_unf) / (end - warm),
            "cost_unfinished_per_s": cost_unf / (end - warm),
        })
    tot = {"type": "all"}
    for k in ("arrived", "completed", "rejected", "unfinished", "reward_per_s", "cost_per_s",
              "cost_unfinished_per_s"):
        tot[k] = sum(r[k] for r in rows)
    rows.append(tot)
    for r in rows:
        r["value_per_s"] = r["reward_per_s"] - r["cost_per_s"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--params", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "specs", "objective.yaml"))
    a = ap.parse_args()
    params = yaml.safe_load(open(a.params))
    manifest = analyze.json.load(open(os.path.join(a.exp, "manifest.json")))
    horizon = float(manifest["horizon_s"]) if manifest.get("horizon_s") else None

    rows = []
    paths = sorted(glob.glob(os.path.join(a.exp, "runs", "rate*_s*.json")) +
                   glob.glob(os.path.join(a.exp, "runs", "rate*_s*.json.gz")))
    for path in paths:
        mt = analyze.RUN_RE.search(os.path.basename(path))
        if not mt:
            continue
        rate, seed = float(mt.group(1)), int(mt.group(2))
        m, df, _ = analyze.load_run(path)
        warm, end = analyze.window_bounds(manifest, m)
        for r in run_value(df, params, warm, end, horizon or m["vllm_estimated_duration_s"]):
            r.update({"rate": rate, "seed": seed})
            rows.append(r)
    if not rows:
        sys.exit("no runs found")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.exp, "objective_runs.csv"), index=False)

    out = []
    for (rate, t), g in runs.groupby(["rate", "type"]):
        row = {"rate": rate, "type": t, "n_seeds": len(g)}
        for col in ("value_per_s", "reward_per_s", "cost_per_s", "cost_unfinished_per_s"):
            x = g[col]
            row[col + "_mean"] = x.mean()
            row[col + "_ci95"] = analyze.t_crit(len(x)) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) >= 2 else np.nan
        out.append(row)
    summary = pd.DataFrame(out).sort_values(["type", "rate"])
    summary.to_csv(os.path.join(a.exp, "objective.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
    for ax, col, label in zip(axes, ("value_per_s", "reward_per_s", "cost_per_s"),
                              ("value per second (reward minus congestion cost)",
                               "reward per second", "congestion cost per second")):
        for t in TYPES + ["all"]:
            s = summary[summary["type"] == t].sort_values("rate")
            ax.errorbar(s.rate, s[col + "_mean"], yerr=s[col + "_ci95"], marker="o", ms=4,
                        lw=1.5, capsize=3, color=COLORS[t], label=t)
        ax.set_xlabel("total arrival rate, req/s")
        ax.set_ylabel(label)
        ax.grid(True, color="#e5e5e5", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{os.path.basename(a.exp)}    objective with {os.path.basename(a.params)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.exp, "objective.png"))

    show = ["rate", "type", "n_seeds", "value_per_s_mean", "value_per_s_ci95", "reward_per_s_mean",
            "cost_per_s_mean", "cost_unfinished_per_s_mean"]
    with pd.option_context("display.width", 200, "display.float_format", "{:,.1f}".format):
        print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
