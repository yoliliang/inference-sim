#!/usr/bin/env python3
"""ours/fluid/capacity.py: fluid capacity of a set of instances, the largest total arrival rate
t (fixed type shares f_i) that the fluid capacity constraints of problem (28) can carry with
every job admitted:

    max t  s.t.  sum_j x_ij = t f_i,  sum_i aM_ij x_ij <= 1,  sum_i aB_ij x_ij <= 1,  x >= 0.

A linear program (prices play no role). Used to put instances with different hardware or
workload at comparable loads: rate = (baseline rate / baseline capacity) * capacity.

    python ours/fluid/capacity.py --instances H100:1 --blocks 6500 [--spec ...]
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from solve import parse_instances, run_gurobi, term, num  # noqa: E402


def capacity(types, gpus, blocks, fit, budget=8192, block_size=16, workdir=None, name="cap"):
    I, J = len(types), len(gpus)
    P = np.array([t["prompt"] for t in types])
    kap = np.array([t["kappa"] for t in types])
    m = np.array([t["m"] for t in types])
    beta = np.array([t["beta"] for t in types])
    f = np.array([t["fraction"] for t in types])
    M = np.array([common.blocks_for(g, blocks) * block_size for g in gpus], dtype=float)
    tau0 = np.array([fit[g]["tau0"] for g in gpus])
    w = np.array([[fit[g]["taup"] * P[i] + fit[g]["taukv"] * kap[i] for g in gpus] for i in range(I)])
    aM = w + tau0[None, :] * m[:, None] / M[None, :]
    aB = w + tau0[None, :] * beta[:, None] / budget
    X = lambda i, j: f"x_{i}_{j}"  # noqa: E731
    lines = ["Maximize", " obj: + 1 t", "Subject To"]
    for i in range(I):
        lines.append(f" flow_{i}: " + " ".join(term(1.0, X(i, j)) for j in range(J)) + f" - {num(f[i])} t = 0")
    for j in range(J):
        lines.append(f" mem_{j}: " + " ".join(term(aM[i, j], X(i, j)) for i in range(I)) + " <= 1")
        lines.append(f" tok_{j}: " + " ".join(term(aB[i, j], X(i, j)) for i in range(I)) + " <= 1")
    lines.append("End")
    vals, _ = run_gurobi("\n".join(lines) + "\n", workdir or os.path.join(common.HERE, "work"), name,
                         {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9})
    return vals["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--blocks", type=int, default=6500)
    ap.add_argument("--spec", default=os.path.join(common.OURS, "specs", "types3.yaml"))
    ap.add_argument("--objective", default=os.path.join(common.OURS, "specs", "objective.yaml"))
    ap.add_argument("--stepfit", default=os.path.join(common.HERE, "stepfit.yaml"))
    a = ap.parse_args()
    types = common.load_types(a.spec, a.objective)
    gpus = parse_instances(a.instances)
    fit = yaml.safe_load(open(a.stepfit))["gpus"]
    t = capacity(types, gpus, a.blocks, fit)
    print(f"{a.instances}: fluid capacity {t:.4f} req/s total, {t / len(gpus):.4f} per instance")


if __name__ == "__main__":
    main()
