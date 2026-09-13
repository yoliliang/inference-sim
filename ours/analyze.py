#!/usr/bin/env python3
"""ours/analyze.py <experiment folder>: steady-state analysis and panel plots.

Reads manifest.json (warmup_s, tail_s, horizon_s) and every runs/rate<R>_s<S>.json
written by BLIS. For each run, the steady-state window is the set of requests that
ARRIVED in [warmup_s, horizon_s - tail_s). Latency statistics use completed
requests in the window; requests still unfinished at the horizon are counted, not
timed (right censoring, reported as unfinished_frac). Admission rejections are
counted per type (rejected_frac).

Outputs, all inside the experiment folder:
    summary_runs.csv   one row per (rate, seed, type) with window metrics
    summary.csv        per (rate, type): across-seed mean and 95 percent CI half-width
    runs/<tag>_panel.png   six-panel figure for the lowest seed of each rate or n (all seeds with --all-panels)
                           (panels 4 and 5 need the _state.csv from --state-sample-ms,
                           panel 6 reads the preemption warnings in the BLIS log)
    sweep_panel.png    metric vs rate with CI bars (only when more than one rate)

Re-runnable: python ours/analyze.py ours/experiments/<folder>
"""
import glob
import gzip
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TYPES = ["type1", "type2", "type3"]
TYPE_COLORS = {"type1": "#2a78d6", "type2": "#eb6834", "type3": "#1baf7a", "all": "#333333"}
INST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RUN_RE = re.compile(r"(?:rate([0-9.]+)|n([0-9]+))_s([0-9]+)[.]json([.]gz)?$")  # rate experiment or scaling experiment tag
PREEMPT_RE = re.compile(r"\[tick (\d+)\] preemption: evicting (request_\d+)")

# two-sided 97.5 percent Student t critical values by degrees of freedom
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_crit(n):
    if n < 2:
        return float("nan")
    return T975.get(n - 1, 1.96)


def _open_json(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path)


def _stem(path):
    return path[:-8] if path.endswith(".json.gz") else path[:-5]


def load_run(path):
    m = json.load(_open_json(path))
    df = pd.DataFrame(m["requests"])
    if "status" not in df:
        df["status"] = np.where(df["e2e_ms"] > 0, "completed", "unfinished")
    for c in ("preemption_count", "wasted_tokens"):
        if c not in df:
            df[c] = 0
        df[c] = df[c].fillna(0)
    state_path = _stem(path) + "_state.csv"
    state = pd.read_csv(state_path) if os.path.exists(state_path) else None
    if state is not None:
        state["t_s"] = state["clock_us"] / 1e6
    return m, df, state


def load_preemptions(json_path, df):
    """Preemption events (time, tenant type) from the BLIS log next to the json.
    BLIS logs one warning per eviction at its default log level."""
    log_path = _stem(json_path) + ".log"
    if not os.path.exists(log_path):
        return None
    tenant = dict(zip(df.requestID, df.tenant_id))
    txt = open(log_path, errors="replace").read()
    ev = [(int(t) / 1e6, tenant.get(r, "?")) for t, r in PREEMPT_RE.findall(txt)]
    return pd.DataFrame(ev, columns=["t_s", "tenant_id"])


def window_bounds(manifest, m):
    warm = float(manifest.get("warmup_s") or 0.0)
    tail = float(manifest.get("tail_s") or 0.0)
    if manifest.get("horizon_s"):
        end = float(manifest["horizon_s"]) - tail
    else:  # drain mode: use the run length as the end of the arrival process
        end = m["vllm_estimated_duration_s"] - tail
    return warm, end


def group_metrics(g, window_len):
    done = g[g.status == "completed"]
    n = len(g)
    row = {
        "arrived": n,
        "completed": len(done),
        "unfinished_frac": (g.status == "unfinished").mean() if n else np.nan,
        "rejected_frac": (g.status == "rejected").mean() if n else np.nan,
        "preempt_per_arrival": g.preemption_count.sum() / n if n else np.nan,
        "preempt_per_completed": g.preemption_count.sum() / len(done) if len(done) else np.nan,
        "preempt_per_s": g.preemption_count.sum() / window_len if window_len > 0 else np.nan,
        "wasted_tok_per_arrival": g.wasted_tokens.sum() / n if n else np.nan,
        "output_tok_per_s": done.num_decode_tokens.sum() / window_len if window_len > 0 else np.nan,
    }
    for col, name in (("ttft_ms", "ttft"), ("e2e_ms", "e2e"), ("scheduling_delay_ms", "delay")):
        x = done[col]
        row[f"{name}_mean_ms"] = x.mean() if len(x) else np.nan
        row[f"{name}_p50_ms"] = x.quantile(0.5) if len(x) else np.nan
        row[f"{name}_p99_ms"] = x.quantile(0.99) if len(x) else np.nan
    return row


def parse_tag(basename, manifest):
    """(n instances, total rate, seed) from a run file name and the manifest."""
    mt = RUN_RE.search(basename)
    if not mt:
        return None
    seed = int(mt.group(3))
    if mt.group(1) is not None:
        rate = float(mt.group(1))
        n = int((manifest.get("instances") or [4])[0])
    else:
        n = int(mt.group(2))
        rate = n * float(manifest.get("rate_per_instance") or 0)
    return n, rate, seed


def window_rows(win, rate, seed, n):
    """Per-type rows from the metrics file's window block (sufficient statistics computed
    by BLIS at the end of the run); same columns as run_rows."""
    L = win["end_s"] - win["start_s"]
    rows = []
    for t in TYPES + ["all"]:
        s = win["types"].get(t)
        if s is None:
            continue
        a = s["arrived"]
        rows.append({
            "n": n, "rate": rate, "seed": seed, "type": t, "window_s": L,
            "arrived": a, "completed": s["completed"],
            "unfinished_frac": s["unfinished"] / a if a else np.nan,
            "rejected_frac": s["rejected"] / a if a else np.nan,
            "preempt_per_arrival": s["sum_preemptions"] / a if a else np.nan,
            "preempt_per_completed": s["sum_preemptions"] / s["completed"] if s["completed"] else np.nan,
            "preempt_per_s": s["sum_preemptions"] / L if L > 0 else np.nan,
            "wasted_tok_per_arrival": s["sum_wasted_tokens"] / a if a else np.nan,
            "output_tok_per_s": s["sum_output_tokens"] / L if L > 0 else np.nan,
            "ttft_mean_ms": s["ttft"]["mean_ms"], "ttft_p50_ms": s["ttft"]["p50_ms"], "ttft_p99_ms": s["ttft"]["p99_ms"],
            "e2e_mean_ms": s["e2e"]["mean_ms"], "e2e_p50_ms": s["e2e"]["p50_ms"], "e2e_p99_ms": s["e2e"]["p99_ms"],
            "delay_mean_ms": s["wait"]["mean_ms"], "delay_p50_ms": s["wait"]["p50_ms"], "delay_p99_ms": s["wait"]["p99_ms"],
        })
    return rows


def run_rows(df, warm, end, rate, seed, n=4):
    w = df[(df.arrived_at >= warm) & (df.arrived_at < end)]
    rows = []
    for t in TYPES + ["all"]:
        g = w if t == "all" else w[w.tenant_id == t]
        row = {"n": n, "rate": rate, "seed": seed, "type": t, "window_s": end - warm}
        row.update(group_metrics(g, end - warm))
        rows.append(row)
    return rows


def summarize(runs_df):
    metrics = [c for c in runs_df.columns if c not in ("n", "rate", "seed", "type", "window_s")]
    out = []
    for (n, rate, t), g in runs_df.groupby(["n", "rate", "type"]):
        row = {"n": n, "rate": rate, "type": t, "n_seeds": len(g)}
        for mcol in metrics:
            x = g[mcol].dropna()
            row[f"{mcol}_mean"] = x.mean() if len(x) else np.nan
            row[f"{mcol}_ci95"] = (t_crit(len(x)) * x.std(ddof=1) / math.sqrt(len(x))
                                   if len(x) >= 2 else np.nan)
        out.append(row)
    return pd.DataFrame(out).sort_values(["type", "n", "rate"])


def _blank(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#777")
    ax.set_xticks([])
    ax.set_yticks([])


def _shade(ax, warm, end):
    ax.axvline(warm, color="#999", ls="--", lw=1)
    ax.axvline(end, color="#999", ls="--", lw=1)


def _kde(x, grid):
    """Gaussian kernel density with Scott's bandwidth, evaluated on grid."""
    n = len(x)
    h = 1.06 * min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34 or x.std(ddof=1)) * n ** (-0.2)
    h = max(h, 1e-6)
    z = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * h * math.sqrt(2 * math.pi))


def _density(ax, df, col, xlabel):
    """Per-type smooth density of log10(latency), x shown in ms on a log axis.
    The x range is cut to the 0.5 to 99.5 percentiles so the bulk of the mass fills the panel."""
    done = df[df.status == "completed"]
    x_all = done[col].values
    x_all = x_all[x_all > 0]
    if len(x_all) < 5:
        _blank(ax, "no completed requests in window")
        return
    lo, hi = np.percentile(np.log10(x_all), [0.5, 99.5])
    grid = np.linspace(lo, hi, 400)
    for t in TYPES:
        x = done.loc[done.tenant_id == t, col].values
        x = np.log10(x[x > 0])
        if len(x) < 5:
            continue
        ax.plot(10 ** grid, _kde(x, grid), lw=2, color=TYPE_COLORS[t], label=t)
    ax.set_xscale("log")
    ax.set_xlim(10 ** lo, 10 ** hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density of log10 latency")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.legend(frameon=False, fontsize=8)


def panel_plot(df, state, preempt, warm, end, out_png, title):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), dpi=130)
    w = df[(df.arrived_at >= warm) & (df.arrived_at < end)]

    # 1 and 2: latency densities per type, requests that arrived inside the window
    _density(axes[0, 0], w, "ttft_ms", "time to first token, ms (requests arriving in the window)")
    _density(axes[0, 1], w, "e2e_ms", "end-to-end latency, ms (requests arriving in the window)")

    # 3: queue wait of each request against its arrival time (the overload signal)
    ax = axes[1, 0]
    done = df[df.status == "completed"].sort_values("arrived_at")
    for t in TYPES:
        d = done[done.tenant_id == t]
        y = d.scheduling_delay_ms.clip(lower=0.1)
        ax.scatter(d.arrived_at, y, s=3, alpha=0.25, color=TYPE_COLORS[t], linewidths=0)
        if len(d) >= 20:
            roll = y.rolling(25, center=True, min_periods=5).median()
            ax.plot(d.arrived_at, roll, lw=2, color=TYPE_COLORS[t], label=f"{t} rolling median")
    ax.set_yscale("log")
    ax.set_xlabel("arrival time of the request, s")
    ax.set_ylabel("queue wait before first scheduling, ms")
    _shade(ax, warm, end)
    n_unf = int((df.status == "unfinished").sum())
    n_rej = int((df.status == "rejected").sum())
    ax.set_title(f"dashed = steady-state window; unfinished at horizon: {n_unf}, rejected: {n_rej}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # 4 and 5: per-instance state over simulated time
    if state is None:
        for ax in (axes[1, 1], axes[2, 0]):
            _blank(ax, "no state sample\n(run sweep.py with --state-sample-ms)")
    else:
        insts = sorted(state.instance.unique())
        ax = axes[1, 1]
        for i, inst in enumerate(insts):
            s = state[state.instance == inst]
            ax.plot(s.t_s, s.kv_utilization, lw=1.2, color=INST_COLORS[i % len(INST_COLORS)], label=inst)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("simulated time, s")
        ax.set_ylabel("KV blocks in use, fraction of capacity")
        _shade(ax, warm, end)
        ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[2, 0]
        for i, inst in enumerate(insts):
            s = state[state.instance == inst]
            c = INST_COLORS[i % len(INST_COLORS)]
            ax.plot(s.t_s, s.queue_depth, lw=1.2, color=c, label=inst)
            ax.plot(s.t_s, s.batch_size, lw=1.2, ls=":", color=c)
        ax.set_xlabel("simulated time, s")
        ax.set_ylabel("requests per instance (solid = waiting, dotted = in batch)")
        _shade(ax, warm, end)
        ax.legend(frameon=False, fontsize=8, ncol=2)

    # 6: preemptions per second by type of the evicted request (from the BLIS log)
    ax = axes[2, 1]
    if preempt is None or preempt.empty:
        _blank(ax, "no preemptions logged")
    else:
        edges = np.arange(0, math.ceil(preempt.t_s.max()) + 2, 1.0)
        mids = edges[:-1] + 0.5
        bottom = np.zeros(len(mids))
        for t in TYPES:
            cnt, _ = np.histogram(preempt.loc[preempt.tenant_id == t, "t_s"], bins=edges)
            ax.bar(mids, cnt, width=1.0, bottom=bottom, color=TYPE_COLORS[t], label=t, linewidth=0)
            bottom += cnt
        ax.set_xlabel("simulated time, s")
        ax.set_ylabel("preemptions per second (by type of evicted request)")
        _shade(ax, warm, end)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(True, color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def sweep_plot(summary, out_png, title):
    panels = [  # (metric, label, log y, scale factor applied to the values)
        ("delay_mean_ms", "mean queue wait, s", True, 1e-3),
        ("ttft_p99_ms", "time to first token p99, s", True, 1e-3),
        ("preempt_per_s", "evictions per second (whole cluster)", False, 1.0),
        ("unfinished_frac", "unfinished at horizon, percent of window arrivals", False, 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=130)
    for ax, (m, label, logy, k) in zip(axes.flat, panels):
        for t in TYPES + ["all"]:
            s = summary[summary["type"] == t].sort_values("rate")
            if s.empty:
                continue
            x, y, e = s.rate.values, k * s[f"{m}_mean"].values, k * s[f"{m}_ci95"].fillna(0).values
            ax.plot(x, y, marker="o", ms=4, lw=1.5, color=TYPE_COLORS[t], label=t)
            ax.fill_between(x, y - e, y + e, color=TYPE_COLORS[t], alpha=0.18, linewidth=0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("total arrival rate, req/s")
        ax.set_ylabel(label)
        ax.grid(True, color="#e5e5e5", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def powerlaw_fit(x, y):
    """Least-squares fit of log y = a + b log x over points with x, y > 0; returns (b, a, n_points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan, int(ok.sum())
    b, a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return b, a, int(ok.sum())


SCALING_METRICS = [  # (column, label, kind) kind: "total" = system total, "level" = as is
    ("output_tok_per_s", "output tokens per second, whole system", "total"),
    ("preempt_per_s", "evictions per second, whole system", "total"),
    ("delay_mean_ms", "mean queue wait, ms", "level"),
    ("ttft_p99_ms", "time to first token p99, ms", "level"),
    ("unfinished_frac", "unfinished at horizon, fraction of arrivals", "level"),
    ("wasted_tok_per_arrival", "wasted tokens per arrival", "level"),
]


def scaling_plot(summary, out_png, out_csv, title, extra=None):
    """Key system statistics against the system scale n, mean over seeds with a shaded
    95 percent confidence band, and a power-law fit n^b on the all-types means.
    extra: optional objective summary (type all) with n, value_per_s_mean, value_per_s_ci95."""
    metrics = list(SCALING_METRICS)
    if extra is not None:
        metrics.insert(0, ("value_per_s", "objective value per second, whole system", "total"))
    fits = []
    ncol = 3
    nrow = int(math.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), dpi=130)
    for ax, (m, label, kind) in zip(axes.flat, metrics):
        src = extra if m == "value_per_s" else summary
        for t in (["all"] if m == "value_per_s" else TYPES + ["all"]):
            s = src[src["type"] == t].sort_values("n")
            if s.empty or f"{m}_mean" not in s:
                continue
            x, y, e = s["n"].values, s[f"{m}_mean"].values, s[f"{m}_ci95"].fillna(0).values
            ax.plot(x, y, marker="o", ms=4, lw=1.5, color=TYPE_COLORS[t], label=t)
            ax.fill_between(x, y - e, y + e, color=TYPE_COLORS[t], alpha=0.18, linewidth=0)
            if t == "all":
                b, a0, npts = powerlaw_fit(x, y)
                fits.append({"metric": m, "kind": kind, "exponent": b, "intercept_log": a0, "points": npts})
                if np.isfinite(b):
                    xx = np.linspace(x.min(), x.max(), 50)
                    ax.plot(xx, np.exp(a0) * xx ** b, ls="--", color="#333", lw=1, label=f"fit n^{b:.2f}")
        ax.set_xscale("log", base=2)
        if kind == "total" and m != "value_per_s":
            ax.set_yscale("log")
        ax.set_xlabel("system scale n")
        ax.set_ylabel(label)
        ax.grid(True, color="#e5e5e5", lw=0.8)
        ax.legend(frameon=False, fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    pd.DataFrame(fits).to_csv(out_csv, index=False)
    return pd.DataFrame(fits)


def analyze_experiment(exp, all_panels=False):
    """all_panels: draw the six-panel figure for every sample path (slow); default draws it
    only for the lowest seed of each rate or n, which is what report.py reproduces."""
    manifest = json.load(open(os.path.join(exp, "manifest.json")))
    first_seed = min(manifest.get("seeds") or [1])
    rows = []
    scaling = manifest.get("experiment") == "scaling"
    for path in sorted(glob.glob(os.path.join(exp, "runs", "*_s*.json")) +
                       glob.glob(os.path.join(exp, "runs", "*_s*.json.gz"))):
        parsed = parse_tag(os.path.basename(path), manifest)
        if not parsed:
            continue
        n, rate, seed = parsed
        m = json.load(_open_json(path))
        warm, end = window_bounds(manifest, m)
        if "window" in m:
            rows.extend(window_rows(m["window"], rate, seed, n))
        if not m.get("requests"):
            continue  # no path data kept for this seed: statistics came from the window block
        _, df, state = load_run(path)
        if "window" not in m:
            rows.extend(run_rows(df, warm, end, rate, seed, n))
        tag = os.path.basename(_stem(path))
        if not all_panels and seed != first_seed:
            continue
        panel_plot(df, state, load_preemptions(path, df), warm, end,
                   os.path.join(exp, "runs", tag + "_panel.png"),
                   f"{os.path.basename(exp)}    " + (f"n {n}    rate {rate:g}" if scaling else f"rate {rate:g}") + f"    seed {seed}")
    if not rows:
        print("analyze: no runs found")
        return
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(os.path.join(exp, "summary_runs.csv"), index=False)
    summary = summarize(runs_df)
    summary.to_csv(os.path.join(exp, "summary.csv"), index=False)
    if scaling and summary.n.nunique() > 1:
        fits = scaling_plot(summary, os.path.join(exp, "scaling_panel.png"), os.path.join(exp, "scaling_fits.csv"),
                            f"{os.path.basename(exp)}: key statistics against the system scale, {summary.n_seeds.max()} seeds, 95 percent confidence band")
        print(fits.to_string(index=False))
    elif summary.rate.nunique() > 1:
        sweep_plot(summary, os.path.join(exp, "sweep_panel.png"),
                   f"{os.path.basename(exp)}: steady-state window, "
                   f"{summary.n_seeds.max()} seeds, 95 percent confidence band")
    show = ["n", "rate", "type", "n_seeds", "arrived_mean", "unfinished_frac_mean", "rejected_frac_mean",
            "delay_mean_ms_mean", "delay_mean_ms_ci95", "ttft_p99_ms_mean", "e2e_mean_ms_mean",
            "preempt_per_arrival_mean"]
    with pd.option_context("display.width", 200, "display.float_format", "{:,.2f}".format):
        print(summary[show].to_string(index=False))
    print(f"analysis written to {exp}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    analyze_experiment(args[0], all_panels="--all-panels" in sys.argv)
