#!/usr/bin/env python3
"""ours/fluid/optimistic.py: the optimistic fluid benchmark of the model draft (Section
"An Optimistic Benchmark and a Policy Test", eq. opt-program) and its marginal charges.

Finite coefficients (eq. opt-constants) from BLIS's capped output law, a_i0 = 1 and
a_is = S_s = P(o > s) for s = 1 .. K_i - 1 (common.survival):

    d_ij0 = taup_j P_i,  d_ijs = taukv_j (P_i + s - 1) for s >= 1        (opt-stage-cost)
    w_ij  = sum_s a_is d_ijs = taup_j P_i + taukv_j kappa_i
    g_ij  = sum_s a_is sqrt(h_i d_ijs)

Per instance, with rho_j = sum_i w_ij x_ij, A_j = sum_i h_i s_i x_ij and G_j = sum_i g_ij x_ij,
the optimistic holding charge is H_j = tau0_j A_j + tau0_j G_j^2 / (1 - rho_j) (opt-holding) and

    U_opt = max_{x in X_K} sum_ij r_i x_ij - sum_j H_j(x)                (opt-program)

over the same feasible region as (28). The program is concave and symmetric in identical
instances, so it is solved by GPU type (total flow X_ig over the N_g instances of type g) and
the equal split x = X / N is a global optimum of the per-instance program, not an
approximation. By type the cost is tau0 sum_i h_i s_i X_ig + tau0 Ghat^2 / (N - sum_i w_i X_ig)
with Ghat = sum_i g_i X_ig, written with the rotated cone Ghat^2 <= z (N - sum_i w_i X_ig),
z >= 0. Gurobi solves it with the barrier method; the last barrier iterate gives the primal
and dual objectives (the dual one is the solver's bound).

Marginal charge (opt-marginal), per instance at x^U with D_j = 1 - rho_j:
    c_ij = tau0_j h_i s_i + tau0_j (2 G_j g_ij / D_j + G_j^2 w_ij / D_j^2)
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from solve import run_gurobi, term, num  # noqa: E402


def g_coefficient(t, f):
    """g_ij = sum_s a_is sqrt(h_i d_ijs) for type t on a GPU with step-time fit f."""
    S = t["survival"]                      # S[s-1] = P(o > s), s = 1 .. K-1
    s = np.arange(1, len(S) + 1, dtype=float)
    h, P = t["h"], t["prompt"]
    return float(math.sqrt(h * f["taup"] * P) + (S * np.sqrt(h * f["taukv"] * (P + s - 1.0))).sum())


def constants(types, gpu, blocks, fit, budget=8192, block_size=16):
    f = fit[gpu]
    M = common.blocks_for(gpu, blocks) * block_size
    P = np.array([t["prompt"] for t in types])
    kap = np.array([t["kappa"] for t in types])
    w = f["taup"] * P + f["taukv"] * kap
    return dict(
        tau0=f["tau0"], M=float(M), b=float(budget), w=w,
        aM=w + f["tau0"] * np.array([t["m"] for t in types]) / M,
        aB=w + f["tau0"] * np.array([t["beta"] for t in types]) / budget,
        hs=np.array([t["h"] * t["s"] for t in types]),
        g=np.array([g_coefficient(t, f) for t in types]),
        r=np.array([t["r"] for t in types]))


def per_instance_cost(c, x):
    """H_j(x) of one instance at per-instance rates x (vector over types)."""
    rho = float(c["w"] @ x)
    G = float(c["g"] @ x)
    return c["tau0"] * float(c["hs"] @ x) + c["tau0"] * G * G / (1.0 - rho)


def marginal(c, x):
    """c_ij of eq. opt-marginal at per-instance rates x."""
    D = 1.0 - float(c["w"] @ x)
    G = float(c["g"] @ x)
    return c["tau0"] * c["hs"] + c["tau0"] * (2.0 * G * c["g"] / D + G * G * c["w"] / D ** 2)


def _barrier_objectives(log):
    """(primal, dual) objective of the last barrier iterate in a gurobi_cl log, or (None, None)."""
    last = None
    for line in log.splitlines():
        m = re.match(r"\s*(\d+)\*?\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+\d+s", line)
        if m:
            last = (float(m.group(2)), float(m.group(3)))
    return last or (None, None)


def solve_optimistic(types, groups, lam, blocks, fit, budget=8192, block_size=16, name="opt", workdir=None):
    """groups: list of (gpu, N). Returns dict(value, X (I x G), primal, dual, status, violation, log)."""
    I, G = len(types), len(groups)
    cs = [constants(types, gpu, blocks, fit, budget, block_size) for gpu, _ in groups]
    obj, cons, bounds = [], [], []
    for i in range(I):
        cons.append(f" flow_{i}: " + " ".join(term(1.0, f"x_{i}_{k}") for k in range(G)) + f" <= {num(lam[i])}")
        bounds += [f" 0 <= x_{i}_{k} <= {num(lam[i])}" for k in range(G)]
    for k, (gpu, N) in enumerate(groups):
        c = cs[k]
        X = [f"x_{i}_{k}" for i in range(I)]
        cons.append(f" mem_{k}: " + " ".join(term(c["aM"][i], X[i]) for i in range(I)) + f" <= {num(N)}")
        cons.append(f" tok_{k}: " + " ".join(term(c["aB"][i], X[i]) for i in range(I)) + f" <= {num(N)}")
        # Ghat_k = sum_i g_i X_i ; y_k = N - sum_i w_i X_i ; Ghat_k^2 <= z_k y_k
        cons.append(f" gdef_{k}: 1 G_{k} " + " ".join(term(-c["g"][i], X[i]) for i in range(I)) + " = 0")
        cons.append(f" ydef_{k}: 1 y_{k} " + " ".join(term(c["w"][i], X[i]) for i in range(I)) + f" = {num(N)}")
        cons.append(f" cone_{k}: [ G_{k} ^ 2 - z_{k} * y_{k} ] <= 0")
        obj += [term(c["r"][i] - c["tau0"] * c["hs"][i], X[i]) for i in range(I)] + [term(-c["tau0"], f"z_{k}")]
    lp = "\n".join(["Maximize", " obj: " + " ".join(obj), "Subject To", *cons, "Bounds", *bounds, "End"]) + "\n"
    vals, log = run_gurobi(lp, workdir or os.path.join(common.HERE, "work"), name,
                           {"BarQCPConvTol": 1e-8, "BarConvTol": 1e-9, "FeasibilityTol": 1e-8, "OptimalityTol": 1e-9})
    X = np.array([[vals.get(f"x_{i}_{k}", 0.0) for k in range(G)] for i in range(I)])
    value, viol = 0.0, 0.0
    for k, (gpu, N) in enumerate(groups):
        c = cs[k]
        x = X[:, k] / N
        value += N * (float(c["r"] @ x) - per_instance_cost(c, x))
        viol = max(viol, float(c["aM"] @ x) - 1, float(c["aB"] @ x) - 1)
    viol = max(viol, float((X.sum(axis=1) - lam).max()), float(-X.min()))
    primal, dual = _barrier_objectives(log)
    status = "optimal" if re.search(r"Optimal objective", log) else "not proven"
    return dict(value=value, X=X, primal=primal, dual=dual, status=status, violation=max(viol, 0.0), log=log, consts=cs)
