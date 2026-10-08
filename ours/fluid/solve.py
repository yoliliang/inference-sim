#!/usr/bin/env python3
"""ours/fluid/solve.py: the fluid engine design problem (28) of the model draft, its global
optimum and the KKT multipliers of Section 4, written as the parameter file of the
fluid-dual policy (sim/fluid_dual.go).

    python ours/fluid/solve.py --rate 48 --instances H100:4 --blocks 6500 --out fluid.yaml

Problem (28), in the exact reformulation agreed with the user: with d_i = h_i (s_i + 1/2),

    max  sum_ij r_i x_ij - sum_j g_j
    s.t. sum_j x_ij <= lambda_i                                  (flow)
         sum_i aM_ij x_ij <= 1,  sum_i aB_ij x_ij <= 1           (memory, batch tokens)
         g_j (1 - sum_i w_ij x_ij) >= tau0_j sum_i d_i x_ij       (holding-cost rate)
         0 <= x_ij <= lambda_i,  0 <= g_j <= U_j

Because g_j is penalised and 1 - rho_j > 0 on the feasible set, the last constraint is tight
at an optimum, so this is (28) exactly. U_j = min(M_j max_i d_i/m_i, b_j max_i d_i/beta_i)
is implied by the capacity constraints (1 - rho_j >= tau0 sum m x / M_j). Identical instances
are ordered by rho_j (rho_j >= rho_{j+1}): this removes permuted copies of the same solution,
never the optimal value. Gurobi solves the bilinear model globally (spatial branch and bound,
NonConvex = 2) through its command line, gurobi_cl, and reports the optimality gap.

Multipliers (31) are then the solution of a linear program at x*: nu_i, gammaM_j, gammaB_j >= 0
with r_i - c*_ij - gammaM_j aM_ij - gammaB_j aB_ij <= nu_i for all (i, j), equality where
x*_ij > 0, and zero multipliers on slack constraints. When several multiplier vectors satisfy
this, the one with the smallest sum of capacity prices is taken; the largest stationarity
residual is reported.
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GUROBI_CL = os.environ.get("GUROBI_CL", "gurobi_cl")
GAP = float(os.environ.get("FLUID_GAP", "1e-6"))


def parse_instances(spec):
    """'H100:2,A100-80:2' -> ['H100', 'H100', 'A100-80', 'A100-80'] in instance-index order
    (the order sweep.py --pools fills the node pools)."""
    out = []
    for item in spec.split(","):
        gpu, _, k = item.partition(":")
        out += [gpu.strip()] * int(k or 1)
    return out


def num(v):
    return f"{float(v):.17g}"


def term(coef, var):
    return f"{'-' if coef < 0 else '+'} {num(abs(coef))} {var}"


def run_gurobi(lp_text, workdir, name, params):
    lp = os.path.join(workdir, name + ".lp")
    sol = os.path.join(workdir, name + ".sol")
    log = os.path.join(workdir, name + ".log")
    open(lp, "w").write(lp_text)
    for p in (sol, log):
        if os.path.exists(p):
            os.remove(p)
    cmd = [GUROBI_CL, *[f"{k}={v}" for k, v in params.items()], f"ResultFile={sol}", f"LogFile={log}", lp]
    subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(sol):
        raise SystemExit(f"gurobi found no solution for {name}; see {log}")
    vals = {}
    for line in open(sol):
        if line.startswith("#") or not line.strip():
            continue
        k, v = line.split()
        vals[k] = float(v)
    text = open(log).read()
    return vals, text


def start_from(path, types, gpus, w, tau0, d):
    """A feasible starting point for J instances from a solution for J0 instances of the same
    GPU type with the same per-instance arrival rate: every instance column of the smaller
    solution is repeated J / J0 times (feasible: each instance keeps its loads, each type's
    total flow scales by J / J0 like lambda). Columns are ordered by utilisation, largest
    first, to satisfy the symmetry-breaking rows. Returns {var: value} or None."""
    old = yaml.safe_load(open(path))
    J0, J = len(old["instances"]), len(gpus)
    if J % J0 or J <= J0 or len(set(gpus)) != 1 or any(o["gpu"] != gpus[0] for o in old["instances"]):
        return None
    cols = [[o["types"][t["name"]]["x"] for t in types] for o in old["instances"]] * (J // J0)
    cols.sort(key=lambda c: -sum(w[i, 0] * c[i] for i in range(len(types))))
    start = {}
    for j, col in enumerate(cols):
        rho = sum(w[i, j] * col[i] for i in range(len(types)))
        D = sum(d[i] * col[i] for i in range(len(types)))
        for i in range(len(types)):
            start[f"x_{i}_{j}"] = col[i]
        start[f"g_{j}"] = tau0[j] * D / (1.0 - rho)
    return start


def solve(types, gpus, lam, M, b, fit, time_limit, workdir, name, start_path=None):
    I, J = len(types), len(gpus)
    r = np.array([t["r"] for t in types])
    d = np.array([t["d"] for t in types])
    P = np.array([t["prompt"] for t in types])
    kap = np.array([t["kappa"] for t in types])
    m = np.array([t["m"] for t in types])
    beta = np.array([t["beta"] for t in types])
    tau0 = np.array([fit[g]["tau0"] for g in gpus])
    w = np.array([[fit[g]["taup"] * P[i] + fit[g]["taukv"] * kap[i] for g in gpus] for i in range(I)])
    aM = w + tau0[None, :] * m[:, None] / M
    aB = w + tau0[None, :] * beta[:, None] / b
    U = np.minimum(M * (d / m).max(), b * (d / beta).max()) * 1.0001

    X = lambda i, j: f"x_{i}_{j}"  # noqa: E731
    lines = ["Maximize", " obj: " + " ".join(term(r[i], X(i, j)) for i in range(I) for j in range(J))
             + " " + " ".join(term(-1.0, f"g_{j}") for j in range(J)), "Subject To"]
    for i in range(I):
        lines.append(f" flow_{i}: " + " ".join(term(1.0, X(i, j)) for j in range(J)) + f" <= {num(lam[i])}")
    for j in range(J):
        lines.append(f" mem_{j}: " + " ".join(term(aM[i, j], X(i, j)) for i in range(I)) + " <= 1")
        lines.append(f" tok_{j}: " + " ".join(term(aB[i, j], X(i, j)) for i in range(I)) + " <= 1")
        quad = " ".join(term(-w[i, j], f"g_{j} * {X(i, j)}") for i in range(I))
        lin = " ".join(term(-tau0[j] * d[i], X(i, j)) for i in range(I))
        lines.append(f" hold_{j}: + 1 g_{j} {lin} + [ {quad} ] >= 0")
    for j in range(J - 1):
        if gpus[j] == gpus[j + 1]:
            lines.append(f" sym_{j}: " + " ".join(term(w[i, j], X(i, j)) for i in range(I)) + " "
                         + " ".join(term(-w[i, j + 1], X(i, j + 1)) for i in range(I)) + " >= 0")
    lines.append("Bounds")
    for i in range(I):
        for j in range(J):
            lines.append(f" 0 <= {X(i, j)} <= {num(lam[i])}")
    for j in range(J):
        lines.append(f" 0 <= g_{j} <= {num(U)}")
    lines.append("End")
    params = {"NonConvex": 2, "MIPGap": GAP, "MIPGapAbs": 1e-9, "FeasibilityTol": 1e-9,
              "OptimalityTol": 1e-9, "TimeLimit": time_limit, "Threads": 0}
    start = start_from(start_path, types, gpus, w, tau0, d) if start_path else None
    if start:
        mst = os.path.join(workdir, name + ".mst")
        open(mst, "w").write("".join(f"{k} {num(v)}\n" for k, v in start.items()))
        params["InputFile"] = mst
    vals, log = run_gurobi("\n".join(lines) + "\n", workdir, name, params)
    x = np.array([[max(0.0, vals.get(X(i, j), 0.0)) for j in range(J)] for i in range(I)])
    bound = gap = None
    mm = re.findall(r"Best objective ([-+0-9.eE]+), best bound ([-+0-9.eE]+), gap ([-+0-9.eE]+)%", log)
    if mm:
        bound, gap = float(mm[-1][1]), float(mm[-1][2]) / 100.0
    status = "optimal" if "Optimal solution found" in log else ("time limit" if "Time limit reached" in log else "see log")
    return x, dict(w=w, aM=aM, aB=aB, tau0=tau0, r=r, d=d, P=P, kap=kap), bound, gap, status


def fluid_quantities(x, c, M, b):
    rho = (c["w"] * x).sum(axis=0)
    ell = c["tau0"] / (1.0 - rho)
    D = (c["d"][:, None] * x).sum(axis=0)
    g = ell * D                                   # holding-cost rate per instance, H_j(x)
    value = float((c["r"][:, None] * x).sum() - g.sum())
    cstar = c["d"][:, None] * ell[None, :] + (ell ** 2)[None, :] * c["w"] / c["tau0"][None, :] * D[None, :]   # (30)
    K = ((c["kap"][:, None] * x) * ell[None, :]).sum(axis=0)
    W0 = 1.5 * x * ell[None, :]                     # (29) stage-0 time-average WIP per (type, instance)
    S0 = (c["P"][:, None] * W0).sum(axis=0)
    loadM = (c["aM"] * x).sum(axis=0)
    loadB = (c["aB"] * x).sum(axis=0)
    return dict(rho=rho, ell=ell, g=g, value=value, cstar=cstar, K=K, W0=W0, S0=S0, loadM=loadM, loadB=loadB)


def multipliers(x, c, q, lam, workdir, name):
    """KKT multipliers at x* by least violation. x* comes from a global solve with a finite
    optimality gap, so it satisfies (31) and complementary slackness only approximately;
    hard-zeroing the multipliers of nearly tight constraints can make the system infeasible.
    Instead: nu, gammaM, gammaB >= 0 minimise
        1e4 * (stationarity residuals)  +  1e2 * sum(slack_k * multiplier_k)  +  1e-3 * sum(gamma),
    where the stationarity residuals are e+ on r - c* - gamma a - nu <= 0 for every (i, j) and
    e- on the reverse inequality where x*_ij > 0, and slack_k >= 0 is the slack of constraint
    k at x* (relative for the flow rows). The last term picks the smallest capacity prices
    among exact solutions. The largest residual is reported."""
    I, J = x.shape
    tol_x = 1e-7 * max(1.0, lam.max())
    s_flow = np.maximum(0.0, lam - x.sum(axis=1)) / np.maximum(lam, 1e-12)
    s_mem = np.maximum(0.0, 1.0 - q["loadM"])
    s_tok = np.maximum(0.0, 1.0 - q["loadB"])
    rhs = c["r"][:, None] - q["cstar"]
    obj = []
    for i in range(I):
        obj.append(term(1e2 * s_flow[i], f"nu_{i}"))
    for j in range(J):
        obj.append(term(1e2 * s_mem[j] + 1e-3, f"gm_{j}"))
        obj.append(term(1e2 * s_tok[j] + 1e-3, f"gb_{j}"))
    for i in range(I):
        for j in range(J):
            obj.append(f"+ 10000 ep_{i}_{j}")
            if x[i, j] > tol_x:
                obj.append(f"+ 10000 em_{i}_{j}")
    lines = ["Minimize", " obj: " + " ".join(obj), "Subject To"]
    for i in range(I):
        for j in range(J):
            lhs = f"{term(c['aM'][i, j], f'gm_{j}')} {term(c['aB'][i, j], f'gb_{j}')} + 1 nu_{i}"
            lines.append(f" ge_{i}_{j}: {lhs} + 1 ep_{i}_{j} >= {num(rhs[i, j])}")
            if x[i, j] > tol_x:
                lines.append(f" le_{i}_{j}: {lhs} - 1 em_{i}_{j} <= {num(rhs[i, j])}")
    lines.append("End")  # all variables default to [0, inf)
    vals, _ = run_gurobi("\n".join(lines) + "\n", workdir, name, {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9})
    nu = np.array([vals.get(f"nu_{i}", 0.0) for i in range(I)])
    gm = np.array([vals.get(f"gm_{j}", 0.0) for j in range(J)])
    gb = np.array([vals.get(f"gb_{j}", 0.0) for j in range(J)])
    idx = rhs - gm[None, :] * c["aM"] - gb[None, :] * c["aB"]      # I*_ij = r - c* - gamma* a
    resid_pos = np.where(x > tol_x, np.abs(idx - nu[:, None]), 0.0).max()
    resid_zero = np.where(x <= tol_x, np.maximum(idx - nu[:, None], 0.0), 0.0).max()
    return nu, gm, gb, idx, float(max(resid_pos, resid_zero))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True, help="total arrival rate, req/s")
    ap.add_argument("--instances", required=True, help="GPU types in instance order, e.g. H100:4 or H100:2,A100-80:2")
    ap.add_argument("--blocks", type=int, required=True, help="KV blocks per instance")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--budget", type=int, default=8192, help="batch-token budget b_j")
    ap.add_argument("--spec", default=os.path.join(common.OURS, "specs", "types3.yaml"))
    ap.add_argument("--objective", default=os.path.join(common.OURS, "specs", "objective.yaml"))
    ap.add_argument("--stepfit", default=os.path.join(common.HERE, "stepfit.yaml"))
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--start", default="", help="fluid solution for fewer instances of the same GPU type and the same per-instance rate, replicated as a starting point")
    ap.add_argument("--workdir", default=os.path.join(common.HERE, "work"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    types = common.load_types(a.spec, a.objective)
    gpus = parse_instances(a.instances)
    fit = yaml.safe_load(open(a.stepfit))["gpus"]
    missing = sorted(set(gpus) - set(fit))
    if missing:
        sys.exit(f"no step-time fit for {missing}; run ours/fluid/stepfit.py --gpus {','.join(sorted(set(gpus)))}")
    lam = np.array([a.rate * t["fraction"] for t in types])
    M = float(a.blocks * a.block_size)
    b = float(a.budget)
    tag = os.path.splitext(os.path.basename(a.out))[0]

    x, c, bound, gap, status = solve(types, gpus, lam, M, b, fit, a.time_limit, a.workdir, tag, a.start or None)
    q = fluid_quantities(x, c, M, b)
    nu, gm, gb, idx, resid = multipliers(x, c, q, lam, a.workdir, tag + "_kkt")
    J = len(gpus)
    out = {
        "meta": {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "rate": a.rate, "instances": J, "blocks": a.blocks, "block_size": a.block_size, "budget": a.budget,
            "value": q["value"], "bound": bound, "gap": gap, "status": status, "eta0": q["value"] / J,
            "kkt_residual": resid, "solver": "gurobi_cl NonConvex=2 MIPGap=1e-6",
            "spec": os.path.relpath(a.spec, common.ROOT).replace(os.sep, "/"),
            "units": "rates per second, times in seconds, memory and budget in tokens, rewards in objective.yaml units (per 1000 tokens)",
        },
        "types": [{
            "name": t["name"], "prompt": t["prompt"], "mean_output": t["mean_output"], "lambda": float(lam[i]),
            "s": t["s"], "beta": t["beta"], "kappa": t["kappa"], "m": t["m"], "r": t["r"], "d": t["d"],
            "nu": float(nu[i]), "admitted": float(x[i].sum()),
        } for i, t in enumerate(types)],
        "instances": [{
            "id": f"instance_{j}", "gpu": gpus[j], "M": M, "b": b,
            "tau0": fit[gpus[j]]["tau0"], "taup": fit[gpus[j]]["taup"], "taukv": fit[gpus[j]]["taukv"],
            "rho": float(q["rho"][j]), "ell": float(q["ell"][j]), "holding_rate": float(q["g"][j]),
            "load_memory": float(q["loadM"][j]), "load_tokens": float(q["loadB"][j]),
            "gammaM": float(gm[j]), "gammaB": float(gb[j]),
            "kv_target": float(q["K"][j]), "stage0_target": float(q["S0"][j]),
            "types": {t["name"]: {
                "x": float(x[i, j]), "w": float(c["w"][i, j]), "aM": float(c["aM"][i, j]), "aB": float(c["aB"][i, j]),
                "c": float(q["cstar"][i, j]), "index0": float(idx[i, j]), "W0": float(q["W0"][i, j]),
            } for i, t in enumerate(types)},
        } for j in range(J)],
    }
    yaml.safe_dump(out, open(a.out, "w"), sort_keys=False)
    if not a.quiet:
        print(f"V* = {q['value']:.4f} per s (bound {bound}, gap {gap}, {status}); eta0 = V*/J = {q['value']/J:.4f}; "
              f"KKT residual {resid:.2e}")
        for i, t in enumerate(types):
            print(f"  {t['name']:6} lambda {lam[i]:7.3f}  admitted {x[i].sum():7.3f}  nu {nu[i]:9.4f}  "
                  f"x by instance {np.round(x[i], 3).tolist()}")
        for j in range(J):
            print(f"  instance_{j} {gpus[j]:8} rho {q['rho'][j]:.3f}  ell {q['ell'][j]*1e3:7.2f} ms  "
                  f"memory load {q['loadM'][j]:.3f}  token load {q['loadB'][j]:.3f}  gammaM {gm[j]:.4f}  gammaB {gb[j]:.4f}")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
