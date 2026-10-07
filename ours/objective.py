#!/usr/bin/env python3
"""ours/objective.py <experiment folder> [--params ours/specs/objective.yaml]

Revenue of every run in an experiment folder, as defined in the model draft, eq. (10): the
long-run average net reward per unit time,

    value_per_s = [ sum over jobs completing in the window of (pi_in P + pi_out o) / 1000
                    - sum over types of h * integral over the window of N_type(t) dt ] / L

over the steady-state window [warmup_s, horizon_s - tail_s) of calendar time, L its length.
o counts output tokens as BLIS does (the prefill pass produces the first one), N_type(t) is
the number of accepted type-k jobs in the system at time t (pool and instances, from
arrival to the last token), rejected jobs earn and cost nothing. This is the time-average
estimator of (10) after the warm-up; it needs no censoring because jobs still in the system
at the horizon are counted only up to the window end.

The earlier arrival-cohort definition is kept for comparison as value_cohort_per_s: rewards
and sojourn costs of the jobs that ARRIVE in the window, unfinished ones charged up to the
horizon (a lower bound). In steady state the two agree in expectation (Little's law); under
overload the cohort value is biased by that censoring. Metrics files written before
2026-10-07 have no time-average sums; for them value_per_s falls back to the cohort value and
the column revenue_definition says so.

Outputs in the experiment folder:
    objective_runs.csv   per (rate, seed, type)
    objective.csv        per (rate, type): across-seed mean and CI
    headline.csv         revenue, end-to-end latency and TTFT per (n, rate, type): the first table to read
    objective.png        value per second, reward and congestion cost against rate
"""
import argparse
import glob
import shutil
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


SUM_COLS = ("arrived", "completed", "rejected", "unfinished", "completed_in_window", "mean_in_system",
            "reward_per_s", "cost_per_s", "reward_cohort_per_s", "cost_cohort_per_s", "cost_unfinished_per_s")


def _finish(rows, definition):
    tot = {"type": "all"}
    for k in SUM_COLS:
        tot[k] = sum(r[k] for r in rows)
    rows.append(tot)
    for r in rows:
        r["value_cohort_per_s"] = r["reward_cohort_per_s"] - r["cost_cohort_per_s"]
        if definition == "time-average":
            r["value_per_s"] = r["reward_per_s"] - r["cost_per_s"]
        else:  # old metrics file: no time-average sums
            r["reward_per_s"], r["cost_per_s"] = r["reward_cohort_per_s"], r["cost_cohort_per_s"]
            r["value_per_s"] = r["value_cohort_per_s"]
        r["revenue_definition"] = definition
    return rows


def window_value(win, params):
    """Objective from the window block's per-type sums (no per-request data needed)."""
    L = win["end_s"] - win["start_s"]
    has_ta = all("ta_time_in_system_s" in s for s in win["types"].values())
    rows = []
    for t in TYPES:
        p = params["types"][t]
        s = win["types"].get(t, {})
        reward_c = (p["pi_in"] * s.get("sum_input_tokens", 0) + p["pi_out"] * s.get("sum_output_tokens", 0)) / 1000.0
        cost_done = p["h"] * s.get("sum_sojourn_s", 0.0)
        cost_unf = p["h"] * s.get("sum_censored_s", 0.0)
        reward_ta = (p["pi_in"] * s.get("ta_sum_input_tokens", 0) + p["pi_out"] * s.get("ta_sum_output_tokens", 0)) / 1000.0
        rows.append({
            "type": t, "arrived": s.get("arrived", 0), "completed": s.get("completed", 0),
            "rejected": s.get("rejected", 0), "unfinished": s.get("unfinished", 0),
            "completed_in_window": s.get("ta_completed", 0),
            "mean_in_system": s.get("ta_time_in_system_s", 0.0) / L,
            "reward_per_s": reward_ta / L, "cost_per_s": p["h"] * s.get("ta_time_in_system_s", 0.0) / L,
            "reward_cohort_per_s": reward_c / L, "cost_cohort_per_s": (cost_done + cost_unf) / L,
            "cost_unfinished_per_s": cost_unf / L,
        })
    return _finish(rows, "time-average" if has_ta else "cohort")


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
        # time-average estimator of (10) from all accepted jobs of this type, whatever their arrival
        acc = df[(df.tenant_id == t) & (df.status != "rejected")]
        depart = np.where(acc.status == "completed", acc.arrived_at + acc.e2e_ms / 1000.0, horizon)
        stay = np.clip(np.minimum(depart, end) - np.maximum(acc.arrived_at, warm), 0, None).sum()
        inwin = acc[(acc.status == "completed") & (depart >= warm) & (depart < end)]
        reward_ta = (p["pi_in"] * inwin.num_prefill_tokens.sum() + p["pi_out"] * inwin.num_decode_tokens.sum()) / 1000.0
        rows.append({
            "type": t, "arrived": len(g), "completed": len(done),
            "rejected": int((g.status == "rejected").sum()), "unfinished": len(unf),
            "completed_in_window": len(inwin), "mean_in_system": stay / (end - warm),
            "reward_per_s": reward_ta / (end - warm), "cost_per_s": p["h"] * stay / (end - warm),
            "reward_cohort_per_s": reward / (end - warm),
            "cost_cohort_per_s": (cost_done + cost_unf) / (end - warm),
            "cost_unfinished_per_s": cost_unf / (end - warm),
        })
    return _finish(rows, "time-average")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--params", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "specs", "objective.yaml"))
    a = ap.parse_args()
    params = yaml.safe_load(open(a.params))
    os.makedirs(os.path.join(a.exp, "specs"), exist_ok=True)
    shutil.copyfile(a.params, os.path.join(a.exp, "specs", "objective.yaml"))  # the prices this folder was valued with
    manifest = analyze.json.load(open(os.path.join(a.exp, "manifest.json")))
    horizon = float(manifest["horizon_s"]) if manifest.get("horizon_s") else None

    rows = []
    paths = sorted(glob.glob(os.path.join(a.exp, "runs", "*_s*.json")) +
                   glob.glob(os.path.join(a.exp, "runs", "*_s*.json.gz")))
    for path in paths:
        parsed = analyze.parse_tag(os.path.basename(path), manifest)
        if not parsed:
            continue
        n, rate, seed = parsed
        m = analyze.json.load(analyze._open_json(path))
        if "window" in m:
            vals = window_value(m["window"], params)
        else:
            _, df, _ = analyze.load_run(path)
            warm, end = analyze.window_bounds(manifest, m)
            vals = run_value(df, params, warm, end, horizon or m["vllm_estimated_duration_s"])
        for r in vals:
            r.update({"n": n, "rate": rate, "seed": seed})
            rows.append(r)
    if not rows:
        sys.exit("no runs found")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.exp, "objective_runs.csv"), index=False)

    out = []
    for (n, rate, t), g in runs.groupby(["n", "rate", "type"]):
        row = {"n": n, "rate": rate, "type": t, "n_seeds": len(g)}
        row["revenue_definition"] = ",".join(sorted(set(g.revenue_definition)))
        for col in ("value_per_s", "reward_per_s", "cost_per_s", "value_cohort_per_s", "cost_unfinished_per_s",
                    "mean_in_system"):
            x = g[col]
            row[col + "_mean"] = x.mean()
            row[col + "_ci95"] = analyze.t_crit(len(x)) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) >= 2 else np.nan
        out.append(row)
    summary = pd.DataFrame(out).sort_values(["type", "n", "rate"])
    summary.to_csv(os.path.join(a.exp, "objective.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
    for ax, col, label in zip(axes, ("value_per_s", "reward_per_s", "cost_per_s"),
                              ("value per second (reward minus congestion cost)",
                               "reward per second", "congestion cost per second")):
        for t in TYPES + ["all"]:
            s = summary[summary["type"] == t].sort_values("rate")
            x, y, e = s.rate.values, s[col + "_mean"].values, s[col + "_ci95"].fillna(0).values
            ax.plot(x, y, marker="o", ms=4, lw=1.5, color=COLORS[t], label=t)
            ax.fill_between(x, y - e, y + e, color=COLORS[t], alpha=0.18, linewidth=0)
        ax.set_xlabel("total arrival rate, req/s")
        ax.set_ylabel(label)
        ax.grid(True, color="#e5e5e5", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{os.path.basename(a.exp)}    objective with {os.path.basename(a.params)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.exp, "objective.png"))

    if manifest.get("experiment") == "scaling" and summary.n.nunique() > 1:
        analyze.scaling_plot(pd.read_csv(os.path.join(a.exp, "summary.csv")), os.path.join(a.exp, "scaling_panel.png"),
                             os.path.join(a.exp, "scaling_fits.csv"),
                             f"{os.path.basename(a.exp)}: key statistics and objective against the system scale, 95 percent confidence band",
                             extra=summary[summary["type"] == "all"])
    # headline: revenue first, then end-to-end latency and TTFT (cohort of window arrivals)
    head = summary[["n", "rate", "type", "n_seeds", "value_per_s_mean", "value_per_s_ci95",
                    "reward_per_s_mean", "cost_per_s_mean", "value_cohort_per_s_mean", "revenue_definition"]]
    sp = os.path.join(a.exp, "summary.csv")
    if os.path.exists(sp):
        lat = pd.read_csv(sp)
        keep = ["n", "rate", "type"] + [f"{x}_{y}" for x in ("e2e_mean_ms", "e2e_p99_ms", "ttft_mean_ms", "ttft_p99_ms")
                                        for y in ("mean", "ci95") if f"{x}_{y}" in lat.columns]
        head = head.merge(lat[keep], on=["n", "rate", "type"], how="left")
    head.to_csv(os.path.join(a.exp, "headline.csv"), index=False)
    show = [c for c in ["n", "rate", "type", "n_seeds", "value_per_s_mean", "value_per_s_ci95", "reward_per_s_mean",
                        "cost_per_s_mean", "e2e_mean_ms_mean", "e2e_p99_ms_mean", "ttft_mean_ms_mean",
                        "ttft_p99_ms_mean", "value_cohort_per_s_mean"] if c in head.columns]
    with pd.option_context("display.width", 220, "display.float_format", "{:,.1f}".format):
        print(head[show].to_string(index=False))


if __name__ == "__main__":
    main()
