#!/usr/bin/env python3
"""ours/viz.py: visualize one BLIS sample path over a chosen interval of simulated time.

    python ours/viz.py <experiment folder> --rate 70 --seed 1 --t0 100 --t1 200 [--instance instance_0]

Reads runs/rate<R>_s<S>.json(.gz), runs/rate<R>_s<S>_state.csv, the optional
runs/rate<R>_s<S>_snapshot.csv (one row per request in queue or batch per instance
every state_snapshot_s of simulated time) and the BLIS log (one line per eviction).
Writes into runs/rate<R>_s<S>_viz/:

    cluster_timeline.png/.html  per instance: stacked area of waiting and running requests by
                                type (snapshot) and a thin KV utilisation strip (state csv);
                                eviction ticks as markers coloured by the evicted request's type
    gantt.png/.html             one row per request of the chosen instance that overlaps [t0, t1]:
                                waiting, prefill and decode segments, eviction marks
    kv_by_type.png/.html        KV blocks held by running requests, by type, per instance
    request_card.txt            the request with the most evictions in the interval

Static png via matplotlib; html via plotly when it is importable, otherwise skipped with a note.
Read-only: never runs BLIS, never touches files outside the _viz folder.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an  # noqa: E402  (TYPES, TYPE_COLORS, load_run, PREEMPT_RE, _stem, window_bounds)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:  # pragma: no cover
    HAVE_PLOTLY = False

TYPES = an.TYPES
COL = an.TYPE_COLORS
TOKENS_PER_BLOCK = 16
STATES = ["running", "queued"]          # stacking order, bottom to top
STATE_LABEL = {"running": "running", "queued": "waiting"}
KV_COLOR = "#444444"


# ----------------------------------------------------------------------------- loading

def find_run(exp, rate, seed):
    tag = f"rate{rate:g}_s{seed}"
    for ext in (".json.gz", ".json"):
        p = os.path.join(exp, "runs", tag + ext)
        if os.path.exists(p):
            return tag, p
    sys.exit(f"viz: no runs/{tag}.json(.gz) under {exp}")


def load_snapshot(json_path):
    p = an._stem(json_path) + "_snapshot.csv"
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    s["t_s"] = s["clock_us"] / 1e6
    s["blocks"] = np.ceil(s["progress_tokens"] / TOKENS_PER_BLOCK).astype(int)
    return s


def load_evictions(json_path, df):
    """One row per eviction line in the BLIS log: t_s, requestID, tenant_id, instance."""
    p = an._stem(json_path) + ".log"
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()
    rows = [(int(t) / 1e6, r) for t, r in an.PREEMPT_RE.findall(txt)]
    ev = pd.DataFrame(rows, columns=["t_s", "requestID"])
    ev["tenant_id"] = ev.requestID.map(dict(zip(df.requestID, df.tenant_id))).fillna("?")
    ev["instance"] = ev.requestID.map(dict(zip(df.requestID, df.handled_by))).fillna("?")
    return ev


def request_timeline(df, horizon):
    """Absolute times in s: arrival, first scheduling, first token, completion.
    Unfinished requests get completion = horizon and their missing marks as NaN."""
    d = df.copy()
    d["t_arr"] = d.arrived_at
    done = d.status == "completed"
    d["t_sched"] = np.where(d.scheduling_delay_ms > 0, d.t_arr + d.scheduling_delay_ms / 1e3, np.nan)
    d["t_first"] = np.where(d.ttft_ms > 0, d.t_arr + d.ttft_ms / 1e3, np.nan)
    d["t_end"] = np.where(done, d.t_arr + d.e2e_ms / 1e3, horizon)
    d.loc[done & d.t_sched.isna(), "t_sched"] = d.t_arr
    d.loc[done & d.t_first.isna(), "t_first"] = d.t_sched
    return d


# ----------------------------------------------------------------------------- windowing helpers

def step_window(piv, t0, t1):
    """Restrict a sampled table (index = sample time) to [t0, t1] for step plotting:
    the last sample before t0 is carried to t0 and the last one inside is carried to t1."""
    before = piv[piv.index < t0]
    inside = piv[(piv.index >= t0) & (piv.index <= t1)]
    parts = []
    if len(before):
        parts.append(before.iloc[[-1]].set_index(pd.Index([t0])))
    parts.append(inside)
    out = pd.concat(parts)
    if len(out) and out.index[-1] < t1:
        out = pd.concat([out, out.iloc[[-1]].set_index(pd.Index([t1]))])
    return out


def pivot_counts(snap, inst):
    """index = sample time, columns = (state, type), value = number of requests."""
    s = snap[snap.instance == inst]
    times = np.sort(snap.t_s.unique())
    cols = pd.MultiIndex.from_product([STATES, TYPES])
    if s.empty:
        return pd.DataFrame(0, index=times, columns=cols)
    piv = s.groupby(["t_s", "state", "tenant_id"]).size().unstack(["state", "tenant_id"], fill_value=0)
    return piv.reindex(index=times, columns=cols, fill_value=0)


def pivot_blocks(snap, inst):
    """index = sample time, columns = type, value = KV blocks held by running requests."""
    s = snap[(snap.instance == inst) & (snap.state == "running")]
    times = np.sort(snap.t_s.unique())
    if s.empty:
        return pd.DataFrame(0, index=times, columns=TYPES)
    piv = s.groupby(["t_s", "tenant_id"]).blocks.sum().unstack("tenant_id", fill_value=0)
    return piv.reindex(index=times, columns=TYPES, fill_value=0)


def state_window(state, inst, t0, t1):
    return state[(state.instance == inst) & (state.t_s >= t0) & (state.t_s <= t1)].sort_values("t_s")


def _style(ax):
    ax.grid(True, color="#e5e5e5", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _shade_window(ax, warm, end, t0, t1):
    """Dashed lines at the steady-state window bounds of analyze.py when they fall inside [t0, t1]."""
    for x in (warm, end):
        if t0 < x < t1:
            ax.axvline(x, color="#999", ls="--", lw=1)


# ----------------------------------------------------------------------------- 1. cluster timeline

def cluster_timeline_png(snap, state, ev, insts, t0, t1, warm, end, capacity, out, title):
    n = len(insts)
    fig, axes = plt.subplots(2 * n, 1, figsize=(12, 3.2 * n + 1.2), dpi=130, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1] * n, "hspace": 0.12})
    axes = np.atleast_1d(axes)
    for i, inst in enumerate(insts):
        ax, axk = axes[2 * i], axes[2 * i + 1]
        if snap is not None:
            piv = step_window(pivot_counts(snap, inst), t0, t1)
            t = piv.index.values
            base = np.zeros(len(t))
            for st in STATES:
                for ty in TYPES:
                    y = piv[(st, ty)].values.astype(float)
                    kw = dict(color=COL[ty], step="post", linewidth=0)
                    if st == "queued":
                        kw.update(alpha=0.45, hatch="////", edgecolor="white")
                    ax.fill_between(t, base, base + y, **kw)
                    base = base + y
            ax.set_ylabel("requests")
        elif state is not None:
            s = state_window(state, inst, t0, t1)
            ax.fill_between(s.t_s, 0, s.batch_size, step="post", color="#9a9a9a", linewidth=0)
            ax.fill_between(s.t_s, s.batch_size, s.batch_size + s.queue_depth, step="post",
                            color="#9a9a9a", alpha=0.45, hatch="////", edgecolor="white", linewidth=0)
            ax.set_ylabel("requests (no type split)")
        else:
            an._blank(ax, "no snapshot and no state csv")
        ax.set_title(inst, loc="left", fontsize=10, pad=3)
        ax.set_ylim(bottom=0)
        _shade_window(ax, warm, end, t0, t1)
        _style(ax)

        # KV utilisation strip (state csv, or the snapshot approximation), eviction markers along its top
        if state is not None:
            s = state_window(state, inst, t0, t1)
            axk.plot(s.t_s, s.kv_utilization, color=KV_COLOR, lw=1.0)
            axk.fill_between(s.t_s, 0, s.kv_utilization, color=KV_COLOR, alpha=0.12, linewidth=0)
        elif snap is not None and capacity > 0:
            piv = step_window(pivot_blocks(snap, inst), t0, t1)
            axk.step(piv.index, piv.sum(axis=1) / capacity, where="post", color=KV_COLOR, lw=1.0)
        axk.set_ylim(0, 1.2)
        axk.set_yticks([0, 0.5, 1.0])
        axk.set_ylabel("KV util.", fontsize=8)
        if ev is not None and not ev.empty:
            e = ev[(ev.instance == inst) & (ev.t_s >= t0) & (ev.t_s <= t1)]
            for ty in TYPES:
                x = e.loc[e.tenant_id == ty, "t_s"]
                if len(x):
                    axk.scatter(x, np.full(len(x), 1.1), marker="v", s=14, color=COL[ty],
                                linewidths=0, zorder=5, clip_on=False)
        _shade_window(axk, warm, end, t0, t1)
        _style(axk)
    axes[-1].set_xlabel("simulated time, s")
    axes[-1].set_xlim(t0, t1)

    handles = [Patch(facecolor=COL[ty], label=f"{ty} running") for ty in TYPES]
    handles += [Patch(facecolor=COL[ty], alpha=0.45, hatch="////", edgecolor="white", label=f"{ty} waiting")
                for ty in TYPES]
    handles += [Line2D([], [], color=KV_COLOR, lw=1.0, label="KV blocks in use, fraction of capacity"),
                Line2D([], [], marker="v", color="#888", ls="none", ms=5,
                       label="eviction (colour = type of evicted request)")]
    n_ev = 0 if ev is None else int(((ev.t_s >= t0) & (ev.t_s <= t1)).sum())
    src = "composition from the snapshot csv" if snap is not None else "counts from the state csv (no snapshot)"
    fig.suptitle(f"{title}    [{t0:g}, {t1:g}] s    {src}    evictions in interval: {n_ev}", fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4,
               frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.905, bottom=0.05, left=0.08, right=0.98)
    fig.savefig(out)
    plt.close(fig)


def cluster_timeline_html(snap, state, ev, insts, t0, t1, out, title):
    n = len(insts)
    fig = make_subplots(rows=2 * n, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[3, 1] * n,
                        subplot_titles=[x for inst in insts for x in (inst, "")])
    for i, inst in enumerate(insts):
        r, rk = 2 * i + 1, 2 * i + 2
        if snap is not None:
            piv = step_window(pivot_counts(snap, inst), t0, t1)
            for st in STATES:
                for ty in TYPES:
                    name = f"{ty} {STATE_LABEL[st]}"
                    fig.add_trace(go.Scatter(
                        x=piv.index, y=piv[(st, ty)].values, name=name, legendgroup=name,
                        showlegend=(i == 0), mode="lines", line=dict(width=0, shape="hv", color=COL[ty]),
                        stackgroup=f"g{i}", opacity=1.0 if st == "running" else 0.5,
                        fillpattern=dict(shape="/" if st == "queued" else ""),
                        hovertemplate=f"{name}: %{{y}}<br>t = %{{x:.1f}} s<extra>{inst}</extra>"),
                        row=r, col=1)
        elif state is not None:
            s = state_window(state, inst, t0, t1)
            for col, nm in (("batch_size", "running"), ("queue_depth", "waiting")):
                fig.add_trace(go.Scatter(x=s.t_s, y=s[col], name=nm, legendgroup=nm, showlegend=(i == 0),
                                         mode="lines", line=dict(width=0, shape="hv", color="#9a9a9a"),
                                         stackgroup=f"g{i}", opacity=1.0 if nm == "running" else 0.5),
                              row=r, col=1)
        fig.update_yaxes(title_text="requests", row=r, col=1, rangemode="tozero")
        if state is not None:
            s = state_window(state, inst, t0, t1)
            fig.add_trace(go.Scatter(x=s.t_s, y=s.kv_utilization, name="KV utilisation",
                                     legendgroup="kv", showlegend=(i == 0), mode="lines",
                                     line=dict(color=KV_COLOR, width=1),
                                     hovertemplate="KV util %{y:.3f}<br>t = %{x:.1f} s<extra>" + inst + "</extra>"),
                          row=rk, col=1)
        if ev is not None and not ev.empty:
            e = ev[(ev.instance == inst) & (ev.t_s >= t0) & (ev.t_s <= t1)]
            for ty in TYPES:
                x = e[e.tenant_id == ty]
                if len(x):
                    fig.add_trace(go.Scatter(x=x.t_s, y=np.full(len(x), 1.1), mode="markers",
                                             marker=dict(symbol="triangle-down", size=7, color=COL[ty]),
                                             name=f"eviction of {ty}", legendgroup=f"ev{ty}",
                                             showlegend=(i == 0), text=x.requestID,
                                             hovertemplate="%{text} evicted<br>t = %{x:.3f} s<extra></extra>"),
                                  row=rk, col=1)
        fig.update_yaxes(title_text="KV util.", range=[0, 1.2], row=rk, col=1)
    fig.update_xaxes(range=[t0, t1], title_text="simulated time, s", row=2 * n, col=1)
    fig.update_layout(title=title, height=260 * n + 150, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.04), template="plotly_white")
    fig.write_html(out, include_plotlyjs="cdn")


# ----------------------------------------------------------------------------- 2. gantt

def gantt_rows(tl, ev, inst, t0, t1):
    d = tl[(tl.handled_by == inst) & (tl.t_arr <= t1) & (tl.t_end >= t0)].sort_values("t_arr")
    d = d.reset_index(drop=True)
    d["row"] = np.arange(len(d))
    e = None
    if ev is not None and not ev.empty:
        e = ev[(ev.t_s >= t0) & (ev.t_s <= t1)].merge(d[["requestID", "row"]], on="requestID")
    return d, e


SEG = [("waiting", "t_arr", "t_sched", 0.35), ("prefill", "t_sched", "t_first", 0.7),
       ("decode", "t_first", "t_end", 1.0)]


def gantt_png(tl, ev, inst, t0, t1, out, title):
    d, e = gantt_rows(tl, ev, inst, t0, t1)
    n = len(d)
    h = float(np.clip(1.8 + 0.11 * n, 4, 40))
    fig, ax = plt.subplots(figsize=(12, h), dpi=130)
    if n == 0:
        an._blank(ax, f"no request of {inst} overlaps [{t0:g}, {t1:g}]")
    for ty in TYPES:
        g = d[d.tenant_id == ty]
        done = g.status == "completed"
        for name, a, b, alpha in SEG:
            gg = g if name != "decode" else g[done]
            left, right = gg[a].values, gg[b].values
            ok = ~np.isnan(left) & ~np.isnan(right) & (right > left)
            if ok.any():
                ax.barh(gg.row.values[ok], (right - left)[ok], left=left[ok], height=0.8,
                        color=COL[ty], alpha=alpha, linewidth=0)
        # unfinished request: light hatched bar from its last known mark to the horizon
        u = g[~done]
        if len(u):
            start = u.t_first.fillna(u.t_sched).fillna(u.t_arr).values
            ax.barh(u.row.values, u.t_end.values - start, left=start, height=0.8, color=COL[ty],
                    alpha=0.25, hatch="////", edgecolor="white", linewidth=0)
    if e is not None and len(e):
        ax.scatter(e.t_s, e.row, marker="x", s=18, color="black", linewidths=1.0, zorder=5)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1, n)
    ax.invert_yaxis()
    if n <= 60:
        ax.set_yticks(d.row)
        ax.set_yticklabels(d.requestID, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_ylabel(f"requests handled by {inst}, sorted by arrival (n = {n})")
    ax.set_xlabel("simulated time, s")
    handles = [Patch(facecolor=COL[ty], label=ty) for ty in TYPES]
    handles += [Patch(facecolor="#777", alpha=0.35, label="waiting (arrival to first scheduling)"),
                Patch(facecolor="#777", alpha=0.7, label="prefill (to first token)"),
                Patch(facecolor="#777", alpha=1.0, label="decode (to completion)"),
                Patch(facecolor="#777", alpha=0.25, hatch="////", edgecolor="white", label="unfinished at horizon"),
                Line2D([], [], marker="x", color="black", ls="none", label="eviction (restart from zero)")]
    n_ev = 0 if e is None else len(e)
    n_unf = int((d.status != "completed").sum())
    fig.suptitle(f"{title}    {inst}    [{t0:g}, {t1:g}] s    evictions in interval: {n_ev}"
                 f"    unfinished at horizon: {n_unf}", fontsize=11, y=1 - 0.25 / h)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.5 / h), ncol=4,
               frameon=False, fontsize=8)
    _style(ax)
    fig.subplots_adjust(top=1 - 1.25 / h, bottom=0.6 / h, left=0.07 if n > 60 else 0.12, right=0.98)
    fig.savefig(out)
    plt.close(fig)


def gantt_html(tl, ev, inst, t0, t1, out, title):
    d, e = gantt_rows(tl, ev, inst, t0, t1)
    n = len(d)
    fig = go.Figure()
    for ty in TYPES:
        g = d[d.tenant_id == ty]
        done = g.status == "completed"
        for name, a, b, alpha in SEG:
            gs = g if name != "decode" else g[done]
            left, right = gs[a].values, gs[b].values
            ok = ~np.isnan(left) & ~np.isnan(right) & (right > left)
            if not ok.any():
                continue
            gg = gs[ok]
            fig.add_trace(go.Bar(
                y=gg.row, x=(right - left)[ok], base=left[ok], orientation="h",
                marker=dict(color=COL[ty], opacity=alpha, line=dict(width=0)),
                name=f"{ty} {name}", legendgroup=ty, width=0.8,
                customdata=np.stack([gg.requestID, gg.num_prefill_tokens, gg.num_decode_tokens,
                                     gg.scheduling_delay_ms, gg.ttft_ms, gg.e2e_ms], axis=1),
                hovertemplate=(f"%{{customdata[0]}} ({ty}, {name})<br>in %{{customdata[1]}} tok, "
                               "out %{customdata[2]} tok<br>wait %{customdata[3]:.0f} ms, "
                               "ttft %{customdata[4]:.0f} ms, e2e %{customdata[5]:.0f} ms"
                               "<br>segment starts %{base:.2f} s, lasts %{x:.2f} s<extra></extra>")))
        u = g[~done]
        if len(u):
            start = u.t_first.fillna(u.t_sched).fillna(u.t_arr).values
            fig.add_trace(go.Bar(y=u.row, x=u.t_end.values - start, base=start, orientation="h",
                                 marker=dict(color=COL[ty], opacity=0.2, pattern=dict(shape="/")),
                                 name=f"{ty} unfinished", legendgroup=ty, width=0.8, text=u.requestID,
                                 textposition="none",
                                 hovertemplate="%{text} unfinished at horizon<extra></extra>"))
    if e is not None and len(e):
        fig.add_trace(go.Scatter(x=e.t_s, y=e.row, mode="markers", name="eviction",
                                 marker=dict(symbol="x", size=7, color="black"), text=e.requestID,
                                 hovertemplate="%{text} evicted at %{x:.3f} s<extra></extra>"))
    yaxis = dict(title=f"requests of {inst}, sorted by arrival (n = {n})", autorange="reversed",
                 showticklabels=n <= 60)
    if n <= 60:
        yaxis.update(tickvals=d.row, ticktext=d.requestID)
    fig.update_layout(barmode="overlay", template="plotly_white", title=f"{title}    {inst}",
                      height=int(np.clip(250 + 12 * n, 450, 8000)),
                      xaxis=dict(title="simulated time, s", range=[t0, t1]),
                      yaxis=yaxis, legend=dict(orientation="h", y=1.02))
    fig.write_html(out, include_plotlyjs="cdn")


# ----------------------------------------------------------------------------- 3. KV by type

def kv_by_type_png(snap, state, insts, t0, t1, capacity, out, title):
    n = len(insts)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n + 1.0), dpi=130, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, insts):
        piv = step_window(pivot_blocks(snap, inst), t0, t1)
        t = piv.index.values
        base = np.zeros(len(t))
        for ty in TYPES:
            y = piv[ty].values.astype(float)
            ax.fill_between(t, base, base + y, step="post", color=COL[ty], linewidth=0, label=ty)
            base = base + y
        if state is not None:
            s = state_window(state, inst, t0, t1)
            ax.plot(s.t_s, s.kv_used_blocks, color="black", lw=0.8, ls="--", label="kv_used_blocks (state csv)")
        ax.set_title(inst, loc="left", fontsize=10, pad=3)
        ax.set_ylabel("KV blocks")
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[-1].set_xlabel("simulated time, s")
    axes[-1].set_xlim(t0, t1)
    h = 2.6 * n + 1.0
    fig.suptitle(f"{title}    KV blocks held by running requests, by type (ceil(progress_tokens / {TOKENS_PER_BLOCK}))"
                 f"    capacity {capacity:,} blocks per instance", fontsize=10, y=1 - 0.2 / h)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.42 / h), ncol=4,
               frameon=False, fontsize=8)
    fig.subplots_adjust(top=1 - 0.95 / h, bottom=0.55 / h, left=0.08, right=0.98, hspace=0.3)
    fig.savefig(out)
    plt.close(fig)


def kv_by_type_html(snap, state, insts, t0, t1, capacity, out, title):
    n = len(insts)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=insts)
    for i, inst in enumerate(insts):
        piv = step_window(pivot_blocks(snap, inst), t0, t1)
        for ty in TYPES:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[ty], name=ty, legendgroup=ty, showlegend=(i == 0),
                                     mode="lines", line=dict(width=0, shape="hv", color=COL[ty]),
                                     stackgroup=f"g{i}",
                                     hovertemplate=f"{ty}: %{{y}} blocks<br>t = %{{x:.1f}} s<extra>{inst}</extra>"),
                          row=i + 1, col=1)
        if state is not None:
            s = state_window(state, inst, t0, t1)
            fig.add_trace(go.Scatter(x=s.t_s, y=s.kv_used_blocks, name="kv_used_blocks (state csv)",
                                     legendgroup="state", showlegend=(i == 0), mode="lines",
                                     line=dict(color="black", width=1, dash="dash")), row=i + 1, col=1)
        fig.update_yaxes(title_text="KV blocks", row=i + 1, col=1, rangemode="tozero")
    fig.update_xaxes(range=[t0, t1], title_text="simulated time, s", row=n, col=1)
    fig.update_layout(title=f"{title}    KV blocks by type, capacity {capacity:,} per instance",
                      height=220 * n + 120, hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=-0.06))
    fig.write_html(out, include_plotlyjs="cdn")


# ----------------------------------------------------------------------------- 4. request card

def request_card(tl, ev, inst, t0, t1, out, tag):
    d, e = gantt_rows(tl, ev, inst, t0, t1)
    if e is not None and len(e):
        rid = e.requestID.value_counts().idxmax()
        basis = "the request with the most eviction lines in the log inside the interval"
    elif len(d) and d.preemption_count.max() > 0:
        rid = d.sort_values("preemption_count", ascending=False).requestID.iloc[0]
        basis = "the request with the largest preemption_count in the run json (no eviction lines in the log)"
    elif len(d):
        rid = d.sort_values("scheduling_delay_ms", ascending=False).requestID.iloc[0]
        basis = ("no evictions in the log or the run json for this interval; "
                 "showing the longest waiting request of the instance instead")
    else:
        open(out, "w").write(f"no request of {inst} overlaps [{t0:g}, {t1:g}] s\n")
        return None
    r = tl[tl.requestID == rid].iloc[0]
    n_ev_win = 0 if e is None else int((e.requestID == rid).sum())
    n_ev_all = 0 if ev is None else int((ev.requestID == rid).sum())
    ev_times = "" if ev is None else ", ".join(f"{x:.3f}" for x in ev.loc[ev.requestID == rid, "t_s"])
    lines = [
        f"request card    run {tag}    instance {inst}    interval [{t0:g}, {t1:g}] s",
        f"selection: {basis}",
        "",
        f"id:                       {r.requestID}",
        f"type:                     {r.tenant_id}",
        f"handled by:               {r.handled_by}",
        f"status:                   {r.status}",
        f"input tokens:             {int(r.num_prefill_tokens)}",
        f"output tokens:            {int(r.num_decode_tokens)}",
        f"arrival:                  {r.arrived_at:.3f} s",
        f"wait (scheduling delay):  {r.scheduling_delay_ms:.1f} ms",
        f"time to first token:      {r.ttft_ms:.1f} ms",
        f"end to end:               {r.e2e_ms:.1f} ms",
        f"evictions in interval:    {n_ev_win}  (log lines)",
        f"evictions total:          {n_ev_all}  (log lines)" + (f"  at s: {ev_times}" if ev_times else ""),
        f"preemption_count:         {int(r.preemption_count)}  (run json)",
        f"wasted tokens:            {int(r.wasted_tokens)}  (run json)",
    ]
    open(out, "w").write("\n".join(lines) + "\n")
    return rid


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="experiment folder (contains manifest.json and runs/)")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t0", type=float, required=True, help="start of the interval, s of simulated time")
    ap.add_argument("--t1", type=float, required=True, help="end of the interval, s of simulated time")
    ap.add_argument("--instance", default="instance_0", help="instance for the gantt and the request card")
    a = ap.parse_args()
    if a.t1 <= a.t0:
        sys.exit("viz: need t0 < t1")

    tag, json_path = find_run(a.exp, a.rate, a.seed)
    manifest_path = os.path.join(a.exp, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    m, df, state = an.load_run(json_path)
    snap = load_snapshot(json_path)
    ev = load_evictions(json_path, df)
    warm, end = an.window_bounds(manifest, m) if manifest else (0.0, float("inf"))
    horizon = float(manifest.get("horizon_s") or m.get("vllm_estimated_duration_s") or df.arrived_at.max())
    tl = request_timeline(df, horizon)

    insts = sorted(set(df.handled_by.dropna()) | (set(state.instance) if state is not None else set()))
    if a.instance not in insts:
        sys.exit(f"viz: instance {a.instance} not in this run; choose one of {insts}")
    capacity = int(state.kv_total_blocks.max()) if state is not None else int(manifest.get("blocks") or 0)

    out_dir = os.path.join(a.exp, "runs", tag + "_viz")
    os.makedirs(out_dir, exist_ok=True)
    title = f"{os.path.basename(os.path.normpath(a.exp))}    {tag}"
    written = []

    notes = []
    if snap is None:
        notes.append("snapshot csv missing: composition panels replaced by counts from the state csv")
    if state is None:
        notes.append("state csv missing: KV strip approximated from the snapshot, if any")
    if ev is None:
        notes.append("log missing: no eviction markers")
    elif ev.empty:
        notes.append("log has no eviction lines: no eviction markers")
    if "preemption_count" not in m["requests"][0]:
        notes.append("run json has no preemption_count / wasted_tokens per request: treated as 0")

    p = os.path.join(out_dir, "cluster_timeline.png")
    cluster_timeline_png(snap, state, ev, insts, a.t0, a.t1, warm, end, capacity, p, title)
    written.append(p)
    if HAVE_PLOTLY:
        p = os.path.join(out_dir, "cluster_timeline.html")
        cluster_timeline_html(snap, state, ev, insts, a.t0, a.t1, p, title)
        written.append(p)

    p = os.path.join(out_dir, "gantt.png")
    gantt_png(tl, ev, a.instance, a.t0, a.t1, p, title)
    written.append(p)
    if HAVE_PLOTLY:
        p = os.path.join(out_dir, "gantt.html")
        gantt_html(tl, ev, a.instance, a.t0, a.t1, p, title)
        written.append(p)

    if snap is not None:
        p = os.path.join(out_dir, "kv_by_type.png")
        kv_by_type_png(snap, state, insts, a.t0, a.t1, capacity, p, title)
        written.append(p)
        if HAVE_PLOTLY:
            p = os.path.join(out_dir, "kv_by_type.html")
            kv_by_type_html(snap, state, insts, a.t0, a.t1, capacity, p, title)
            written.append(p)
    else:
        notes.append("kv_by_type skipped: needs the snapshot csv")

    p = os.path.join(out_dir, "request_card.txt")
    rid = request_card(tl, ev, a.instance, a.t0, a.t1, p, tag)
    written.append(p)

    if not HAVE_PLOTLY:
        notes.append("plotly not importable: html outputs skipped (pip install plotly)")
    print(f"viz: {tag}  interval [{a.t0:g}, {a.t1:g}] s  instances {insts}  card request {rid}")
    for w in written:
        print("  wrote", w)
    for note in notes:
        print("  note:", note)


if __name__ == "__main__":
    main()
