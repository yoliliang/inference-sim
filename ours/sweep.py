#!/usr/bin/env python3
"""ours/sweep.py: run a (rate x seed) grid of BLIS simulations into one
self-contained experiment folder, then analyse it with ours/analyze.py.

    python ours/sweep.py capacity --rates 60,80,100 --seeds 1,2,3 --horizon-s 300 --warmup-s 60 --tail-s 30

Two run modes:
    fixed horizon (default for steady-state work): --horizon-s T. The spec's
        num_requests is set to 0 (unlimited), arrivals continue at the given rate
        until T seconds of simulated time, then BLIS stops. Requests still queued
        or running at T appear in the metrics json with status "unfinished".
    drain (legacy): --num-requests N without --horizon-s. N arrivals, then BLIS
        runs until every request completes, so the tail of the run is under-loaded.

Creates ours/experiments/<date>_<name>/ with:
    manifest.json    commit, grid, flags, run mode, warm-up and tail trims
    specs/           one workload yaml per rate (copied from --spec, rate replaced)
    runs/            <rate>_s<seed>.json / .log / _state.csv straight from BLIS
    summary_runs.csv one row per (run, tenant type) over the steady-state window
    summary.csv      across-seed mean and 95 percent CI per (rate, tenant type)
    runs/*_panel.png six-panel summary per run, sweep_panel.png across rates
    README.md        the question this experiment asks (--question)

Extra BLIS flags go after "--", e.g.
    python ours/sweep.py x --rates 40 --seeds 1 --horizon-s 120 -- --scheduler fcfs
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "blis.exe" if os.name == "nt" else "blis")
sys.path.insert(0, HERE)
import analyze  # noqa: E402

BASE_FLAGS = [
    "--model", "qwen/qwen3-14b", "--hardware", "H100", "--tp", "1",
    "--num-instances", "4", "--snapshot-refresh-interval", "0",
]

# Harness routing: weighted scorer over queue depth and KV utilisation, both weight 1,
# with immediate snapshots (omniscient router). round-robin ignores instance state.
ROUTING = {
    "weighted": ["--routing-policy", "weighted",
                 "--routing-scorers", "queue-depth:1,kv-utilization:1"],
    "round-robin": ["--routing-policy", "round-robin"],
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def write_spec(src, dst, rate, num_requests):
    txt = open(src).read()
    txt, n = re.subn(r"^aggregate_rate:.*$", f"aggregate_rate: {rate}", txt, flags=re.M)
    if n != 1:
        sys.exit(f"expected exactly one aggregate_rate line in {src}, found {n}")
    txt, n = re.subn(r"^num_requests:.*$", f"num_requests: {num_requests}", txt, flags=re.M)
    if n != 1:
        sys.exit(f"expected exactly one num_requests line in {src}, found {n}")
    open(dst, "w").write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--rates", required=True, help="comma list, req/s")
    ap.add_argument("--seeds", default="1", help="comma list")
    ap.add_argument("--blocks", type=int, default=2500, help="--total-kv-blocks per instance")
    ap.add_argument("--horizon-s", type=float, default=None,
                    help="fixed-horizon mode: simulated seconds of continuous arrivals")
    ap.add_argument("--num-requests", type=int, default=3000,
                    help="drain mode only (ignored when --horizon-s is set)")
    ap.add_argument("--warmup-s", type=float, default=0.0,
                    help="analysis: ignore requests arriving before this time")
    ap.add_argument("--tail-s", type=float, default=0.0,
                    help="analysis: ignore requests arriving in the last tail-s seconds")
    ap.add_argument("--state-sample-ms", type=int, default=0,
                    help="write per-instance state every N ms of simulated time (0 = off)")
    ap.add_argument("--routing", choices=sorted(ROUTING), default="weighted")
    ap.add_argument("--spec", default=os.path.join(HERE, "specs", "types3.yaml"))
    ap.add_argument("--question", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra BLIS flags after --")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    fixed = a.horizon_s is not None
    if fixed and a.warmup_s + a.tail_s >= a.horizon_s:
        sys.exit("warmup-s + tail-s must be smaller than horizon-s")
    date = dt.date.today().isoformat()
    exp = os.path.join(HERE, "experiments", f"{date}_{a.name}")
    for d in ("specs", "runs"):
        os.makedirs(os.path.join(exp, d), exist_ok=True)

    mode_flags = ["--horizon", str(int(round(a.horizon_s * 1e6)))] if fixed else []
    if a.state_sample_ms > 0:
        mode_flags += ["--state-sample-interval", str(a.state_sample_ms * 1000)]

    manifest = {
        "name": a.name, "date": date, "commit": git_commit(), "binary": BIN,
        "mode": "fixed-horizon" if fixed else "drain",
        "horizon_s": a.horizon_s, "num_requests": None if fixed else a.num_requests,
        "warmup_s": a.warmup_s, "tail_s": a.tail_s,
        "state_sample_ms": a.state_sample_ms,
        "blocks": a.blocks, "routing": a.routing,
        "base_flags": BASE_FLAGS + ROUTING[a.routing] + mode_flags,
        "extra_flags": a.extra,
        "rates": rates, "seeds": seeds, "spec_template": a.spec,
    }
    json.dump(manifest, open(os.path.join(exp, "manifest.json"), "w"), indent=2)
    with open(os.path.join(exp, "README.md"), "w") as f:
        f.write(f"# {a.name} ({date})\n\n{a.question or '(no question recorded)'}\n")

    n_ok = 0
    for rate in rates:
        rtag = f"rate{rate:g}"
        spec = os.path.join(exp, "specs", f"{rtag}.yaml")
        write_spec(a.spec, spec, rate, 0 if fixed else a.num_requests)
        for seed in seeds:
            tag = f"{rtag}_s{seed}"
            out = os.path.join(exp, "runs", tag + ".json")
            log = os.path.join(exp, "runs", tag + ".log")
            cmd = [BIN, "run", *BASE_FLAGS, *ROUTING[a.routing], *mode_flags,
                   "--workload-spec", spec,
                   "--total-kv-blocks", str(a.blocks),
                   "--seed", str(seed),
                   "--metrics-path", out, *a.extra]
            if a.dry_run:
                print(" ".join(cmd))
                continue
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=lf).returncode
            if rc != 0 or not os.path.exists(out):
                print(f"FAILED {tag} (rc={rc}), see {log}")
                continue
            n_ok += 1
            m = json.load(open(out))
            unfinished = m["still_queued"] + m["still_running"]
            print(f"{tag:<14} injected={m['injected_requests']:6d} unfinished={unfinished:5d} "
                  f"preempt={m['preemption_count']:5d} ttft_p99={m['ttft_p99_ms']:9,.0f} "
                  f"e2e_mean={m['e2e_mean_ms']:8,.0f} tok/s={m['tokens_per_sec']:6,.0f}  (raw BLIS, untrimmed)")

    if a.dry_run:
        return
    print(f"\n{n_ok} runs in {exp}")
    if n_ok and not a.no_analyze:
        analyze.analyze_experiment(exp)


if __name__ == "__main__":
    main()
