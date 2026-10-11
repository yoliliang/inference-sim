#!/usr/bin/env python3
"""ours/fluid/verify_type_vs_instance.py: compare the fluid problem (28) indexed by GPU TYPE
(x_ij = total type-i flow to the N_j identical instances of GPU type j, split equally) with the
same problem indexed by INSTANCE (one x vector per instance, as solved so far).

For H100 clusters (one GPU type), at the baseline loads (11 and 14.4 req/s per instance) and
n = 2, 4, 8:
  1. solve the type-indexed problem globally with Gurobi (counts N in the constraints);
  2. evaluate the objective of (28) independently (numpy, no solver) at
       a) the type-level solution split equally over the n instances,
       b) the instance-level solutions stored in the 2026-10-08 experiment folders,
     and check every constraint of (28) at both points.
Prints values and the largest constraint violation; no files are changed.
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from solve import run_gurobi, term, num  # noqa: E402

EXP = os.path.join(common.OURS, "experiments")


def constants(blocks=6500, budget=8192, gpu="H100"):
    types = common.load_types(os.path.join(common.OURS, "specs", "types3.yaml"),
                              os.path.join(common.OURS, "specs", "objective.yaml"))
    fit = yaml.safe_load(open(os.path.join(common.HERE, "stepfit.yaml")))["gpus"][gpu]
    P = np.array([t["prompt"] for t in types])
    kap = np.array([t["kappa"] for t in types])
    m = np.array([t["m"] for t in types])
    beta = np.array([t["beta"] for t in types])
    c = dict(types=types, r=np.array([t["r"] for t in types]), d=np.array([t["d"] for t in types]),
             f=np.array([t["fraction"] for t in types]), tau0=fit["tau0"],
             w=fit["taup"] * P + fit["taukv"] * kap, M=blocks * 16.0, b=float(budget))
    c["aM"] = c["w"] + c["tau0"] * m / c["M"]
    c["aB"] = c["w"] + c["tau0"] * beta / c["b"]
    return c


def objective_instances(X, c, lam):
    """(28) at an instance-level point X (I x n): value and largest constraint violation."""
    rho = c["w"] @ X
    viol = max(0.0, (X.sum(axis=1) - lam).max(), (c["aM"] @ X - 1).max(), (c["aB"] @ X - 1).max(), -X.min())
    if (rho >= 1).any():
        return -np.inf, viol
    val = (c["r"] @ X).sum() - (c["tau0"] * (c["d"] @ X) / (1 - rho)).sum()
    return float(val), float(viol)


def solve_type(c, lam, N, name):
    """type-indexed (28) for one GPU type with N instances: x_i total flow, per-instance x_i / N."""
    I = len(c["r"])
    U = N * min(c["M"] * (c["d"] / (c["aM"] - c["w"]) / c["M"] * c["tau0"]).max(), 1e9)  # loose, finite
    lines = ["Maximize", " obj: " + " ".join(term(c["r"][i], f"x_{i}") for i in range(I)) + " - 1 g", "Subject To"]
    for i in range(I):
        lines.append(f" flow_{i}: + 1 x_{i} <= {num(lam[i])}")
    lines.append(" mem: " + " ".join(term(c["aM"][i], f"x_{i}") for i in range(I)) + f" <= {num(N)}")
    lines.append(" tok: " + " ".join(term(c["aB"][i], f"x_{i}") for i in range(I)) + f" <= {num(N)}")
    lin = " ".join(term(-N * c["tau0"] * c["d"][i], f"x_{i}") for i in range(I))
    quad = " ".join(term(-c["w"][i], f"g * x_{i}") for i in range(I))
    lines.append(f" hold: {num(N)} g {lin} + [ {quad} ] >= 0")
    lines += ["Bounds", *[f" 0 <= x_{i} <= {num(lam[i])}" for i in range(I)], f" 0 <= g <= {num(U)}", "End"]
    vals, log = run_gurobi("\n".join(lines) + "\n", os.path.join(common.HERE, "work"), name,
                           {"NonConvex": 2, "MIPGap": 1e-9, "MIPGapAbs": 1e-10, "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9})
    x = np.array([vals[f"x_{i}"] for i in range(I)])
    status = "optimal" if "Optimal solution found" in log else "not proven"
    return x, status


def main():
    c = constants()
    print("type-level = (28) indexed by GPU type (equal split within the type); instance-level = one x per instance\n")
    for rpi, word, folder in ((11, "underloaded", "2026-10-08_scaling_scale64_fluid_dual_kv6500_rpi11"),
                              (14.4, "overloaded", "2026-10-08_scaling_scale64_fluid_dual_kv6500_rpi14.4")):
        for n in (2, 4, 8):
            lam = n * rpi * c["f"]
            xt, st = solve_type(c, lam, n, f"verify_type_{word}_n{n}")
            Xeq = np.repeat((xt / n)[:, None], n, axis=1)            # type-level split equally
            v_eq, viol_eq = objective_instances(Xeq, c, lam)
            d = yaml.safe_load(open(os.path.join(EXP, folder, "specs", f"fluid_n{n}.yaml")))
            Xin = np.array([[inst["types"][t["name"]]["x"] for inst in d["instances"]] for t in c["types"]])
            v_in, viol_in = objective_instances(Xin, c, lam)
            print(f"{word:11} n={n}: type-level {v_eq:10.4f} ({st}, max violation {viol_eq:.1e})   "
                  f"instance-level {v_in:10.4f} (max violation {viol_in:.1e})   difference {100 * (v_in - v_eq) / v_eq:+.3f}%")
            print(f"{'':16}type-level per-instance mix {np.round(xt / n, 3).tolist()}")
            print(f"{'':16}instance-level columns     {[list(np.round(Xin[:, j], 3)) for j in range(n)]}")


if __name__ == "__main__":
    main()
