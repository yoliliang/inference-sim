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
    runs/            <rate>_s<seed>.json.gz / .log / _state.csv straight from BLIS (json gzipped after the run)
    summary_runs.csv one row per (run, tenant type) over the steady-state window
    summary.csv      across-seed mean and 95 percent CI per (rate, tenant type)
    runs/*_panel.png six-panel summary per run, sweep_panel.png across rates
    README.md        the question this experiment asks (--question)

Extra BLIS flags go after "--", e.g.
    python ours/sweep.py x --rates 40 --seeds 1 --horizon-s 120 -- --scheduler fcfs
"""
import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.environ.get("BLIS_BIN") or os.path.join(ROOT, "blis.exe" if os.name == "nt" else "blis")
sys.path.insert(0, HERE)
import analyze  # noqa: E402

BASE_FLAGS = [
    "--model", "qwen/qwen3-14b", "--hardware", "H100", "--tp", "1",
    # memory-lean execution, byte-identical results (fork features): requests are generated
    # and pulled as the clock advances, completed requests drop their token arrays, ITL
    # samples are kept as counts
    "--lazy-generation", "--release-completed-requests",
]

# Peak private memory per run is about 0.055 GB per instance at 1,200 s; keep the
# concurrent runs of one grid point inside this budget.
MEM_BUDGET_GB = 24.0
MEM_GB_PER_INSTANCE = 0.055

# Configuration profiles (notes/benchmark-brief.md section 1). Each profile fixes the
# gateway and engine controls; --blocks, rate, seed and horizon are set per run.
#   B1        llm-d production default: weighted routing with the llm-d scorer profile,
#             50 ms scraped snapshots, fcfs, tail eviction, H100 budgets 8192 / 1024
#   B0        vLLM alone: vLLM data-parallel balancer (vllm-dp), live counters, same engine
#   floor     B1 with round-robin routing (bracket, not a candidate)
#   oracle    B1 with immediate snapshots (bracket, not a candidate)
#   harness   the 2026-09-07 harness: queue-depth:1,kv-utilization:1, live snapshots,
#             BLIS sub-80 GB budgets 2048 / 256 (kept for the regression anchors only)
ENGINE_H100 = ["--scheduler", "fcfs", "--preemption-policy", "fcfs",
               "--max-num-batched-tokens", "8192", "--max-num-seqs", "1024",
               "--long-prefill-token-threshold", "0", "--block-size-in-tokens", "16",
               "--batch-formation", "vllm", "--admission-policy", "always-admit"]
LLMD_SCORERS = "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
PROFILES = {
    "B1": ["--routing-policy", "weighted", "--routing-scorers", LLMD_SCORERS,
           "--snapshot-refresh-interval", "50000", *ENGINE_H100],
    "B0": ["--routing-policy", "weighted", "--routing-scorers", "vllm-dp:1",
           "--snapshot-refresh-interval", "0", *ENGINE_H100],
    "floor": ["--routing-policy", "round-robin",
              "--snapshot-refresh-interval", "50000", *ENGINE_H100],
    "oracle": ["--routing-policy", "weighted", "--routing-scorers", LLMD_SCORERS,
               "--snapshot-refresh-interval", "0", *ENGINE_H100],
    "harness": ["--routing-policy", "weighted",
                "--routing-scorers", "queue-depth:1,kv-utilization:1",
                "--snapshot-refresh-interval", "0"],
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
    ap.add_argument("--rates", default=None, help="rate experiment: comma list of total arrival rates, req/s")
    ap.add_argument("--instances", default="4", help="comma list of instance counts n (rate experiment: one value)")
    ap.add_argument("--rate-per-instance", type=float, default=None,
                    help="scaling experiment: total rate = n x this for every n in --instances; folder is <date>_scaling_<name>")
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
    ap.add_argument("--state-snapshot-s", type=float, default=0.0,
                    help="write the per-request queue and batch composition every N s (0 = off; multiple of state-sample-ms)")
    ap.add_argument("--keep-paths", choices=["none", "first", "all"], default="none",
                    help="write the per-request array (path data) for no seed, the first seed of each point, or all seeds; "
                         "statistics come from the window block either way")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="B1",
                    help="configuration profile, see PROFILES (default B1, llm-d production default)")
    ap.add_argument("--spec", default=os.path.join(HERE, "specs", "types3.yaml"))
    ap.add_argument("--question", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="sample paths to run concurrently, one BLIS process each (default: half the logical cores); "
                         "reduced per grid point so that concurrent runs fit MEM_BUDGET_GB")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs whose .json.gz already exists (keeps the existing manifest)")
    ap.add_argument("extra", nargs="*", help="extra BLIS flags after --")
    a = ap.parse_args()

    instances = [int(x) for x in a.instances.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    scaling = a.rate_per_instance is not None
    if scaling:
        points = [(n, n * a.rate_per_instance, f"n{n}") for n in instances]  # (instances, total rate, tag prefix)
        rates = [pt[1] for pt in points]
    else:
        if not a.rates:
            sys.exit("give --rates (rate experiment) or --rate-per-instance (scaling experiment)")
        if len(instances) != 1:
            sys.exit("a rate experiment uses one instance count; give --rate-per-instance for a scaling experiment")
        rates = [float(x) for x in a.rates.split(",")]
        points = [(instances[0], r, f"rate{r:g}") for r in rates]
    fixed = a.horizon_s is not None
    if fixed and a.warmup_s + a.tail_s >= a.horizon_s:
        sys.exit("warmup-s + tail-s must be smaller than horizon-s")
    date = dt.date.today().isoformat()
    exp = os.path.join(HERE, "experiments", f"{date}_scaling_{a.name}" if scaling else f"{date}_{a.name}")
    for d in ("specs", "runs"):
        os.makedirs(os.path.join(exp, d), exist_ok=True)

    mode_flags = ["--horizon", str(int(round(a.horizon_s * 1e6)))] if fixed else []
    if fixed:
        mode_flags += ["--window-start", f"{a.warmup_s:g}", "--window-end", f"{a.horizon_s - a.tail_s:g}"]
    if a.state_sample_ms > 0:
        mode_flags += ["--state-sample-interval", str(a.state_sample_ms * 1000)]
    if a.state_snapshot_s > 0:
        mode_flags += ["--state-snapshot-interval", str(int(round(a.state_snapshot_s * 1e6)))]

    manifest = {
        "name": a.name, "date": date, "commit": git_commit(), "binary": BIN,
        "experiment": "scaling" if scaling else "rate",
        "instances": instances, "rate_per_instance": a.rate_per_instance,
        "mode": "fixed-horizon" if fixed else "drain",
        "horizon_s": a.horizon_s, "num_requests": None if fixed else a.num_requests,
        "warmup_s": a.warmup_s, "tail_s": a.tail_s,
        "state_sample_ms": a.state_sample_ms, "state_snapshot_s": a.state_snapshot_s,
        "keep_paths": a.keep_paths,
        "blocks": a.blocks, "profile": a.profile,
        "base_flags": BASE_FLAGS + PROFILES[a.profile] + mode_flags,
        "extra_flags": a.extra,
        "rates": rates, "seeds": seeds, "spec_template": a.spec,
    }
    if not (a.resume and os.path.exists(os.path.join(exp, "manifest.json"))):
        # a resumed run keeps the manifest and README written when it started
        json.dump(manifest, open(os.path.join(exp, "manifest.json"), "w"), indent=2)
        with open(os.path.join(exp, "README.md"), "w") as f:
            f.write(f"# {a.name} ({date})\n\n{a.question or '(no question recorded)'}\n")

    def run_one(job):
        """Run one sample path in its own BLIS process; returns (tag, ok, summary line)."""
        tag, cmd, out, log, _ = job
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=lf).returncode
        if rc != 0 or not os.path.exists(out):
            return tag, False, f"FAILED {tag} (rc={rc}), see {log}"
        with gzip.open(out, "rt", encoding="utf-8") as gz:  # BLIS wrote the json gzipped (fork feature)
            m = json.load(gz)
        unfinished = m["still_queued"] + m["still_running"]
        return tag, True, (f"{tag:<14} injected={m['injected_requests']:6d} unfinished={unfinished:5d} "
                           f"preempt={m['preemption_count']:5d} ttft_p99={m['ttft_p99_ms']:9,.0f} "
                           f"e2e_mean={m['e2e_mean_ms']:8,.0f} tok/s={m['tokens_per_sec']:6,.0f}  (raw BLIS, untrimmed)")

    jobs, n_ok = [], 0
    for n_inst, rate, rtag in points:
        spec = os.path.join(exp, "specs", f"{rtag}.yaml")
        write_spec(a.spec, spec, rate, 0 if fixed else a.num_requests)
        for seed in seeds:
            tag = f"{rtag}_s{seed}"
            out = os.path.join(exp, "runs", tag + ".json.gz")  # written gzipped by BLIS
            log = os.path.join(exp, "runs", tag + ".log")
            keep = a.keep_paths == "all" or (a.keep_paths == "first" and seed == seeds[0])
            cmd = [BIN, "run", *BASE_FLAGS, "--num-instances", str(n_inst), *PROFILES[a.profile], *mode_flags,
                   *([] if keep else ["--drop-per-request-output"]),
                   "--workload-spec", spec,
                   "--total-kv-blocks", str(a.blocks),
                   "--seed", str(seed),
                   "--metrics-path", out, *a.extra]
            if a.dry_run:
                print(" ".join(cmd))
                continue
            if a.resume and os.path.exists(out):
                n_ok += 1
                continue
            jobs.append((tag, cmd, out, log, n_inst))

    if jobs:
        work_total = sum(j[4] for j in jobs)  # one path's cost grows with n
        print(f"{len(jobs)} sample paths, up to {a.jobs} at a time (fewer at large n to fit memory)", flush=True)
        t0, finished, work_done = time.time(), 0, 0
        groups = {}
        for j in jobs:
            groups.setdefault(j[4], []).append(j)
        for n_inst in sorted(groups):
            workers = max(1, min(a.jobs, int(MEM_BUDGET_GB // (MEM_GB_PER_INSTANCE * n_inst))))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for tag, ok, line in pool.map(run_one, groups[n_inst]):
                    finished += 1
                    work_done += n_inst
                    n_ok += ok
                    elapsed = time.time() - t0
                    eta = elapsed / work_done * (work_total - work_done)
                    k = int(30 * finished / len(jobs))
                    print(f"[{'#' * k}{'.' * (30 - k)}] {finished}/{len(jobs)}  {elapsed / 60:.0f} min elapsed, "
                          f"about {eta / 60:.0f} min left    {line}", flush=True)

    if a.dry_run:
        return
    print(f"\n{n_ok} runs in {exp}")
    if n_ok and not a.no_analyze:
        analyze.analyze_experiment(exp)


if __name__ == "__main__":
    main()
