#!/usr/bin/env python3
"""ours/fluid/aggregate.py: the fluid problem aggregated by GPU type (index j = GPU type with
N_j identical instances, load split equally within a type), as an approximation of the
per-instance fluid problem (28).

    max  sum_ij r_i X_ij - sum_j tau0_j sum_i d_i X_ij / (1 - sum_i w_ij X_ij / N_j)
    s.t. sum_j X_ij <= lambda_i,  sum_i aM_ij X_ij <= N_j,  sum_i aB_ij X_ij <= N_j,  X >= 0

Its optimum V_agg is the value of a feasible point of the per-instance problem (every instance
of type j gets X_ij / N_j), so V_agg <= the per-instance optimum. Solved globally by Gurobi to a
1e-9 gap (one ratio per GPU type, so it takes well under a second).

With --compare, solves the aggregate problem at every point of the 2026-10-08/10/11 experiments
that has a per-instance fluid solution and writes ours/fluid/work/aggregate_vs_instance.csv:
V_agg, the per-instance V* (best found), its upper bound, and the relative differences.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from solve import run_gurobi, term, num, parse_instances  # noqa: E402

EXP = os.path.join(common.OURS, "experiments")


def solve_aggregate(types, groups, lam, blocks, fit, budget=8192, block_size=16, name="agg", workdir=None):
    """groups: list of (gpu, N). Returns (value, X (I x G), status)."""
    I, G = len(types), len(groups)
    r = np.array([t["r"] for t in types]); d = np.array([t["d"] for t in types])
    P = np.array([t["prompt"] for t in types]); kap = np.array([t["kappa"] for t in types])
    m = np.array([t["m"] for t in types]); beta = np.array([t["beta"] for t in types])
    lines_obj, cons, bounds = [], [], []
    for i in range(I):
        cons.append(f" flow_{i}: " + " ".join(term(1.0, f"x_{i}_{j}") for j in range(G)) + f" <= {num(lam[i])}")
    for j, (gpu, N) in enumerate(groups):
        f = fit[gpu]
        M = common.blocks_for(gpu, blocks) * block_size
        w = f["taup"] * P + f["taukv"] * kap
        aM = w + f["tau0"] * m / M
        aB = w + f["tau0"] * beta / budget
        cons.append(f" mem_{j}: " + " ".join(term(aM[i], f"x_{i}_{j}") for i in range(I)) + f" <= {num(N)}")
        cons.append(f" tok_{j}: " + " ".join(term(aB[i], f"x_{i}_{j}") for i in range(I)) + f" <= {num(N)}")
        lin = " ".join(term(-N * f["tau0"] * d[i], f"x_{i}_{j}") for i in range(I))
        quad = " ".join(term(-w[i], f"g_{j} * x_{i}_{j}") for i in range(I))
        cons.append(f" hold_{j}: {num(N)} g_{j} {lin} + [ {quad} ] >= 0")
        U = N * float((d * f["tau0"] / (aM - w)).max()) * 1.0001
        bounds += [f" 0 <= x_{i}_{j} <= {num(lam[i])}" for i in range(I)] + [f" 0 <= g_{j} <= {num(U)}"]
    obj = " ".join(term(r[i], f"x_{i}_{j}") for i in range(I) for j in range(G)) + " " + " ".join(f"- 1 g_{j}" for j in range(G))
    lp = "\n".join(["Maximize", " obj: " + obj, "Subject To", *cons, "Bounds", *bounds, "End"]) + "\n"
    vals, log = run_gurobi(lp, workdir or os.path.join(common.HERE, "work"), name,
                           {"NonConvex": 2, "MIPGap": 1e-9, "MIPGapAbs": 1e-10, "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9})
    X = np.array([[vals.get(f"x_{i}_{j}", 0.0) for j in range(G)] for i in range(I)])
    value = 0.0
    for j, (gpu, N) in enumerate(groups):
        f = fit[gpu]
        w = f["taup"] * P + f["taukv"] * kap
        value += float(r @ X[:, j] - f["tau0"] * (d @ X[:, j]) / (1 - (w @ X[:, j]) / N))
    # nonlinear (bilinear) solves report "Optimal solution found"; a linear one (h = 0) "Optimal objective"
    return value, X, ("optimal" if ("Optimal solution found" in log or "Optimal objective" in log) else "not proven")


def groups_of(gpus):
    out = []
    for g in gpus:
        if out and out[-1][0] == g:
            out[-1][1] += 1
        else:
            out.append([g, 1])
    return [tuple(x) for x in out]


def compare():
    fit = yaml.safe_load(open(os.path.join(common.HERE, "stepfit.yaml")))["gpus"]
    rows = []
    folders = sorted(glob.glob(os.path.join(EXP, "2026-10-08_scaling_scale64_fluid_dual_*")) +
                     glob.glob(os.path.join(EXP, "2026-10-11_scaling_robust_*_fluid_dual_*")) +
                     glob.glob(os.path.join(EXP, "2026-10-11_loadsweep_n8_fluid_dual")))
    for folder in folders:
        man = json.load(open(os.path.join(folder, "manifest.json")))
        spec = man.get("spec_template") or os.path.join(common.OURS, "specs", "types3.yaml")
        obj = man.get("objective") or os.path.join(common.OURS, "specs", "objective.yaml")
        types = common.load_types(spec, obj)
        for p in sorted(glob.glob(os.path.join(folder, "specs", "fluid_*.yaml"))):
            d = yaml.safe_load(open(p))
            meta = d["meta"]
            gpus = [inst["gpu"] for inst in d["instances"]]
            lam = np.array([t["lambda"] for t in d["types"]])
            v, X, st = solve_aggregate(types, groups_of(gpus), lam, meta["blocks"], fit, name="agg_cmp")
            rows.append({"folder": os.path.basename(folder), "point": os.path.basename(p)[6:-5], "n": len(gpus),
                         "rate": meta["rate"], "V_instance": meta["value"], "bound_instance": meta["bound"],
                         "solver_gap_instance": meta["gap"], "status_instance": meta["status"],
                         "V_aggregate": v, "status_aggregate": st,
                         "agg_below_instance_pct": 100 * (meta["value"] - v) / meta["value"],
                         "agg_below_bound_pct": 100 * ((meta["bound"] or meta["value"]) - v) / (meta["bound"] or meta["value"])})
            print(f"{os.path.basename(folder)[11:]:58} {rows[-1]['point']:12} n={len(gpus):3}  V_inst {meta['value']:9.3f}  "
                  f"V_agg {v:9.3f}  agg below inst {rows[-1]['agg_below_instance_pct']:6.3f}%  below bound {rows[-1]['agg_below_bound_pct']:6.3f}%")
    import pandas as pd
    out = os.path.join(common.HERE, "work", "aggregate_vs_instance.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--rate", type=float)
    ap.add_argument("--instances")
    ap.add_argument("--blocks", type=int, default=6500)
    ap.add_argument("--spec", default=os.path.join(common.OURS, "specs", "types3.yaml"))
    ap.add_argument("--objective", default=os.path.join(common.OURS, "specs", "objective.yaml"))
    a = ap.parse_args()
    if a.compare:
        return compare()
    fit = yaml.safe_load(open(os.path.join(common.HERE, "stepfit.yaml")))["gpus"]
    types = common.load_types(a.spec, a.objective)
    gpus = parse_instances(a.instances)
    lam = np.array([a.rate * t["fraction"] for t in types])
    v, X, st = solve_aggregate(types, groups_of(gpus), lam, a.blocks, fit)
    print(f"aggregate V = {v:.4f} ({st}); X by GPU type = {np.round(X, 3).tolist()}")


if __name__ == "__main__":
    main()
