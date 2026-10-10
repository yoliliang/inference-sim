#!/usr/bin/env python3
"""ours/trajectory_plot.py: one sample path (same seed, so the same arrival stream) of B1 and
the fluid-dual policy over simulated time, with the fluid reference levels.

    python ours/trajectory_plot.py ours/experiments/2026-10-10_trajectory_n8

Reads runs/<policy>_<load>_s1.json (per-request records), _state.csv (per instance every
0.5 s) and _snapshot.csv (every request in queue or batch every 5 s) and the fluid solution
specs/fluid_n8_<load>.yaml. Writes trajectory_<load>.png and kv_by_instance_<load>.png.

Series (x = simulated time, seconds; shaded = statistics window):
  jobs in system by type   exact, from arrival and departure times of every accepted job;
                           dashed: fluid time-average population sum_j x*_ij l*_j (s_i + 1/2)
  waiting jobs             sum over instances of the wait-queue length (state csv)
  KV memory in use         sum over instances, tokens (used blocks x 16); dashed: sum_j K*_j
  stage-0 prompt tokens    prompt tokens of jobs waiting or mid-prefill (snapshot); dashed: sum_j S*_j
  rejections, evictions    cumulative counts
  net revenue              cumulative (pi_in P + pi_out o)/1000 at completions minus h times the
                           integral of the number in system; dashed: V* t
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POL = {"B1": ("B1 (llm-d default)", "#4C72B0"), "fluid_dual": ("fluid-dual", "#DD8452")}
TYPES = ["type1", "type2", "type3"]


def load(exp, prof, load):
    base = os.path.join(exp, "runs", f"{prof}_{load}_s1")
    m = json.load(open(base + ".json"))
    req = pd.DataFrame(m["requests"])
    st = pd.read_csv(base + "_state.csv")
    sn = pd.read_csv(base + "_snapshot.csv")
    return m, req, st, sn


def population(req, grid, horizon):
    acc = req[req.status != "rejected"]
    a = acc.arrived_at.values
    d = np.where(acc.status == "completed", a + acc.e2e_ms.values / 1000.0, horizon + 1)
    out = {}
    for t in TYPES + ["all"]:
        sel = np.ones(len(acc), bool) if t == "all" else (acc.tenant_id.values == t)
        aa, dd = np.sort(a[sel]), np.sort(d[sel])
        out[t] = np.searchsorted(aa, grid, side="right") - np.searchsorted(dd, grid, side="right")
    return out


def revenue(req, grid, horizon, obj):
    done = req[req.status == "completed"].copy()
    done["t"] = done.arrived_at + done.e2e_ms / 1000.0
    done["r"] = [(obj[t]["pi_in"] * p + obj[t]["pi_out"] * o) / 1000.0
                 for t, p, o in zip(done.tenant_id, done.num_prefill_tokens, done.num_decode_tokens)]
    done = done.sort_values("t")
    cumr = np.interp(grid, done.t.values, done.r.cumsum().values, left=0)
    pop = population(req, grid, horizon)
    dt = np.diff(grid, prepend=grid[0])
    cost = sum(obj[t]["h"] * np.cumsum(pop[t] * dt) for t in TYPES)
    return cumr - cost


def fluid_refs(fy):
    d = yaml.safe_load(open(fy))
    s = {t["name"]: t["s"] for t in d["types"]}
    N = {t: sum(i["types"][t]["x"] * i["ell"] * (s[t] + 0.5) for i in d["instances"]) for t in s}
    N["all"] = sum(N[t] for t in s)
    K = sum(i["kv_target"] for i in d["instances"])
    S = sum(i["stage0_target"] for i in d["instances"])
    return {"N": N, "K": K, "S": S, "V": d["meta"]["value"], "Kj": {i["id"]: i["kv_target"] for i in d["instances"]},
            "M": d["instances"][0]["M"]}


def plot_load(exp, load, title, obj):
    data = {p: load_(exp, p, load) for p in POL}
    fr = fluid_refs(os.path.join(exp, "specs", f"fluid_n8_{load}.yaml"))
    m0 = data["B1"][0]
    horizon = 1200.0
    grid = np.arange(0, horizon + 0.01, 1.0)
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), dpi=140)
    ax = axes.ravel()
    for p, (m, req, st, sn) in data.items():
        name, col = POL[p]
        pop = population(req, grid, horizon)
        ax[0].plot(grid, pop["all"], color=col, lw=1, label=name)
        for k, t in enumerate(TYPES):
            ax[1 + k].plot(grid, pop[t], color=col, lw=1, label=name)
        g = st.groupby("clock_us")
        t_st = np.array(sorted(st.clock_us.unique())) / 1e6
        ax[4].plot(t_st, g.queue_depth.sum().values, color=col, lw=1, label=name)
        ax[5].plot(t_st, g.kv_used_blocks.sum().values * 16, color=col, lw=1, label=name)
        s0 = sn[(sn.state != "running") | (sn.progress_tokens < sn.input_tokens)]
        s0t = s0.groupby("clock_us").input_tokens.sum().reindex(sorted(sn.clock_us.unique()), fill_value=0)
        ax[6].plot(s0t.index / 1e6, s0t.values, color=col, lw=1, label=name)
        ax[7].plot(grid, revenue(req, grid, horizon, obj), color=col, lw=1.4, label=name)
    ax[0].axhline(fr["N"]["all"], ls="--", color="black", lw=1, label="fluid level")
    for k, t in enumerate(TYPES):
        ax[1 + k].axhline(fr["N"][t], ls="--", color="black", lw=1, label="fluid level")
    ax[5].axhline(fr["K"], ls="--", color="black", lw=1, label="fluid level sum K*_j")
    ax[5].axhline(8 * fr["M"], ls=":", color="grey", lw=1, label="capacity 8 M_j")
    ax[6].axhline(fr["S"], ls="--", color="black", lw=1, label="fluid level sum S*_j")
    ax[7].plot(grid, fr["V"] * grid, ls="--", color="black", lw=1, label="V* t")
    labels = ["jobs in system, all types", "jobs in system, type1 (short chat)", "jobs in system, type2 (RAG)",
              "jobs in system, type3 (long generation)", "waiting jobs (all instances)",
              "KV memory in use, tokens (all instances)", "stage-0 prompt tokens (waiting or mid-prefill)",
              "cumulative net revenue"]
    for a, lab in zip(ax, labels):
        a.set_title(lab, fontsize=10)
        a.set_xlabel("simulated time (s)")
        a.axvspan(500, 1100, color="#f0f0f0", zorder=0)
        a.grid(True, color="#e5e5e5", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.legend(frameon=False, fontsize=7)
    fig.suptitle(f"n = 8, seed 1, {title}: one trajectory of B1 and fluid-dual (shaded: statistics window)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(exp, f"trajectory_{load}.png")
    fig.savefig(out)
    plt.close(fig)

    # per-instance KV memory against K*_j
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), dpi=130, sharey=True)
    for k, inst in enumerate(sorted(fr["Kj"], key=lambda s: int(s.split("_")[1]))):
        a = axes.ravel()[k]
        for p, (m, req, st, sn) in data.items():
            s = st[st.instance == inst]
            a.plot(s.clock_us / 1e6, s.kv_used_blocks * 16, color=POL[p][1], lw=0.8, label=POL[p][0])
        a.axhline(fr["Kj"][inst], ls="--", color="black", lw=1, label="fluid K*_j")
        a.axhline(fr["M"], ls=":", color="grey", lw=1, label="capacity M_j")
        a.set_title(inst, fontsize=9)
        a.set_xlabel("simulated time (s)")
        a.grid(True, color="#e5e5e5", lw=0.6)
        if k == 0:
            a.legend(frameon=False, fontsize=7)
            a.set_ylabel("KV tokens in use")
    fig.suptitle(f"n = 8, seed 1, {title}: KV memory per instance", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(exp, f"kv_by_instance_{load}.png"))
    plt.close(fig)
    return out


load_ = load


def main():
    exp = sys.argv[1]
    obj = yaml.safe_load(open(os.path.join(HERE, "specs", "objective.yaml")))["types"]
    for load, title in (("under", "underloaded, 11 req/s per instance"), ("over", "overloaded, 14.4 req/s per instance")):
        print("wrote", plot_load(exp, load, title, obj))


if __name__ == "__main__":
    main()
