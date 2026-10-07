#!/usr/bin/env python3
"""ours/fluid/stepfit.py: fit the affine round-time model of the model draft, eq. (2),

    tau(B) = tau0 + tau_p * (prompt tokens prefilled) + tau_kv * (KV tokens read by decodes),

to BLIS's own step-time function, separately for every GPU type. BLIS's trained-physics
model is not affine (weight loading, quadratic prefill attention, per-token projections), so
the three constants are a least-squares approximation over batches that look like the ones
the workload produces; the fit errors are reported and stored with the constants.

    python ours/fluid/stepfit.py --gpus H100,A100-80

Batches: decode population by type in proportion to lambda_i E[o_i] (the stationary decode
mix), decode stage s drawn with probability proportional to S_s (so the context P + s - 1
follows the fluid stage profile), prefill counts per type Poisson, and the batch-token budget
enforced. Writes ours/fluid/stepfit.yaml (seconds).
"""
import argparse
import csv
import io
import os
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def make_batches(types, n_batches, budget, rng):
    rows, feats = [], []
    w = np.array([t["fraction"] * t["mean_output"] for t in types])
    w /= w.sum()
    for b in range(n_batches):
        n_dec = int(rng.integers(0, 400)) if b % 10 else 0
        n_pre_mean = rng.uniform(0, 4) if b % 7 else 0.0
        pre = [int(rng.poisson(n_pre_mean * t["fraction"])) for t in types]
        if n_dec == 0 and sum(pre) == 0:
            pre[int(rng.integers(0, len(types)))] = 1
        dec_types = rng.choice(len(types), size=n_dec, p=w)
        batch, kv_read, pre_tok, used = [], 0, 0, 0
        for ti in dec_types:
            t = types[ti]
            # P(stage = k) proportional to S_k = exp(-(k + 0.5)/mean): 1 + floor(Exp(mean))
            s = min(1 + int(np.floor(rng.exponential(t["exp_mean"]))), len(t["survival"]))
            ctx = int(t["prompt"]) + s - 1
            batch.append((b, "decode", int(t["prompt"]), ctx, 1))
            kv_read += ctx
            used += 1
        for ti, k in enumerate(pre):
            P = int(types[ti]["prompt"])
            for _ in range(k):
                if used + P > budget:
                    break
                batch.append((b, "prefill", P, 0, P))
                pre_tok += P
                used += P
        if not batch:
            continue
        rows.extend(batch)
        feats.append((b, pre_tok, kv_read, n_dec))
    return rows, feats


def fit(gpu, types, args, rng):
    rows, feats = make_batches(types, args.batches, args.budget, rng)
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["batch", "kind", "prompt", "progress", "new_tokens"])
    wr.writerows(rows)
    path = os.path.join(args.workdir, f"batches_{gpu}.csv")
    open(path, "w", newline="").write(buf.getvalue())
    out = subprocess.run([args.bin, "steptime", "--model", args.model, "--hardware", gpu, "--tp", "1",
                          "--batches", path], cwd=common.ROOT, capture_output=True, text=True, check=True).stdout
    step = {int(r["batch"]): float(r["step_us"]) / 1e6 for r in csv.DictReader(io.StringIO(out))}
    F = np.array([(1.0, p, k) for (b, p, k, _) in feats])
    y = np.array([step[b] for (b, _, _, _) in feats])
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    pred = F @ coef
    rel = np.abs(pred - y) / y
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss
    # same fit with the number of decode requests as a fourth regressor, to show what the
    # draft's model leaves out
    F4 = np.column_stack([F, [n for (_, _, _, n) in feats]])
    c4, *_ = np.linalg.lstsq(F4, y, rcond=None)
    r2_4 = 1 - ((y - F4 @ c4) ** 2).sum() / ss
    return {
        "tau0": float(coef[0]), "taup": float(coef[1]), "taukv": float(coef[2]),
        "r2": float(r2), "mean_rel_error": float(rel.mean()), "p95_rel_error": float(np.quantile(rel, 0.95)),
        "max_rel_error": float(rel.max()), "batches": int(len(y)),
        "r2_with_decode_count": float(r2_4), "decode_count_coef_s": float(c4[3]),
        "step_range_s": [float(y.min()), float(y.max())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="H100,A100-80")
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--spec", default=os.path.join(common.OURS, "specs", "types3.yaml"))
    ap.add_argument("--objective", default=os.path.join(common.OURS, "specs", "objective.yaml"))
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--batches", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bin", default=os.environ.get("BLIS_BIN") or os.path.join(common.ROOT, "blis.exe"))
    ap.add_argument("--workdir", default=os.path.join(common.HERE, "work"))
    ap.add_argument("--out", default=os.path.join(common.HERE, "stepfit.yaml"))
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    types = common.load_types(a.spec, a.objective)
    res = {}
    for gpu in a.gpus.split(","):
        res[gpu] = fit(gpu, types, a, np.random.default_rng(a.seed))
        f = res[gpu]
        print(f"{gpu:8} tau0 {f['tau0']*1e3:7.3f} ms  tau_p {f['taup']*1e6:7.3f} us/token  tau_kv {f['taukv']*1e9:7.3f} ns/token"
              f"  R2 {f['r2']:.4f}  mean rel err {100*f['mean_rel_error']:.1f}%  p95 {100*f['p95_rel_error']:.1f}%"
              f"  (R2 with decode count {f['r2_with_decode_count']:.4f})")
    yaml.safe_dump({"model": a.model, "spec": os.path.relpath(a.spec, common.ROOT).replace(os.sep, "/"),
                    "budget": a.budget, "units": "seconds", "gpus": res}, open(a.out, "w"), sort_keys=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
