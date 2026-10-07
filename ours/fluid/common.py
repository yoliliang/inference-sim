"""ours/fluid/common.py: job-type constants of the fluid model (model draft, Section 3, eq. 22),
computed from the workload spec exactly as BLIS samples it.

Output length. BLIS draws o = round(Exp(mean)), clamped to [min, max] and to at least 1
(sim/workload/distribution.go, ExponentialSampler). The prefill pass produces the first
output token and every decode pass one more, so a job with o tokens runs o passes and, at
decode stage s >= 1, holds P + s - 1 tokens of KV cache (draft, Section 1). The draft's
geometric L is replaced by the exact survival function of o:

    S_k = P(o > k),   k >= 1    (job still running at decode stage k)
    E[o] = 1 + sum_k S_k
    s     = E[o]                                  passes per job
    beta  = P + sum_k S_k                         batch tokens per job
    kappa = sum_k (P + k - 1) S_k                 KV tokens read by its decode passes
    m     = kappa + beta                          memory reserved per job-round
    r     = (pi_in P + pi_out E[o]) / 1000        reward (objective.yaml units)
    d     = h (s + 1/2)                            holding-cost coefficient

For a geometric o these reduce to the draft's (22): S_k = q^k, s = 1/p, beta = P + q/p,
kappa = P q/p + q^2/p^2.
"""
import math
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = os.path.dirname(HERE)
ROOT = os.path.dirname(OURS)


def survival(dist):
    """S_k = P(o > k) for k = 1 .. K-1 as a numpy array (index 0 is k = 1)."""
    if dist.get("type") != "exponential":
        raise SystemExit(f"fluid constants support exponential output lengths only, got {dist.get('type')}")
    p = dist["params"]
    mean = float(p["mean"])
    lo = max(1, int(p.get("min", 1)))
    hi = int(p["max"]) if "max" in p else int(math.ceil(mean * 60))
    k = np.arange(1, hi, dtype=float)
    # o > k  <=>  round(E) clamped > k. For k < lo always true; for lo <= k < hi it is
    # round(E) >= k + 1  <=>  E >= k + 0.5 (Go's math.Round rounds halves away from zero).
    s = np.exp(-(k + 0.5) / mean)
    s[k < lo] = 1.0
    return s


def load_types(spec_path, objective_path):
    """Types in spec order: name, rate fraction, P, survival and the constants above."""
    spec = yaml.safe_load(open(spec_path))
    obj = yaml.safe_load(open(objective_path))["types"]
    out = []
    for c in spec["clients"]:
        name = c["tenant_id"]
        inp = c["input_distribution"]
        if inp["type"] != "constant":
            raise SystemExit(f"{name}: fluid constants need a constant prompt length")
        P = float(inp["params"]["value"])
        S = survival(c["output_distribution"])
        k = np.arange(1, len(S) + 1, dtype=float)
        Eo = float(1.0 + S.sum())
        s = Eo
        beta = float(P + S.sum())
        kappa = float(((P + k - 1.0) * S).sum())
        pr = obj[name]
        out.append({
            "name": name, "fraction": float(c["rate_fraction"]), "prompt": P, "mean_output": Eo,
            "s": s, "beta": beta, "kappa": kappa, "m": kappa + beta,
            "r": (pr["pi_in"] * P + pr["pi_out"] * Eo) / 1000.0, "h": float(pr["h"]),
            "d": float(pr["h"]) * (s + 0.5), "survival": S,
            "exp_mean": float(c["output_distribution"]["params"]["mean"]),
        })
    tot = sum(t["fraction"] for t in out)
    for t in out:
        t["fraction"] /= tot
    return out
