#!/usr/bin/env python3
"""ours/sweep.py: run a (rate x seed) grid of BLIS simulations into one
self-contained experiment folder.

    python ours/sweep.py capacity_sweep --rates 20,30,40 --seeds 1,2,3 --blocks 10000

Creates ours/experiments/<date>_<name>/ with:
    manifest.json   commit, grid, base flags, date
    specs/          one workload yaml per rate (copied from --spec, rate replaced)
    runs/           <rate>_s<seed>.json and .log, straight from BLIS
    summary.csv     one row per run
    README.md       the question this experiment asks (--question)

Extra BLIS flags go after "--", e.g.
    python ours/sweep.py x --rates 40 --seeds 1 -- --scheduler type-rank
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "blis.exe" if os.name == "nt" else "blis")

BASE_FLAGS = [
    "--model", "qwen/qwen3-14b", "--hardware", "H100", "--tp", "1",
    "--num-instances", "4", "--snapshot-refresh-interval", "0",
]

SUMMARY_FIELDS = [
    "rate", "seed", "completed_requests", "still_queued", "still_running",
    "preemption_count", "scheduling_delay_p99_ms", "ttft_mean_ms", "ttft_p99_ms", "e2e_mean_ms",
    "e2e_p99_ms", "itl_mean_ms", "tokens_per_sec", "vllm_estimated_duration_s",
]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def write_spec(src, dst, rate):
    txt = open(src).read()
    new, n = re.subn(r"^aggregate_rate:.*$", f"aggregate_rate: {rate}", txt,
                     flags=re.M)
    if n != 1:
        sys.exit(f"expected exactly one aggregate_rate line in {src}, found {n}")
    open(dst, "w").write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--rates", required=True, help="comma list, req/s")
    ap.add_argument("--seeds", default="1", help="comma list")
    ap.add_argument("--blocks", type=int, default=2500, help="--total-kv-blocks")
    ap.add_argument("--num-requests", type=int, default=3000)
    ap.add_argument("--spec", default=os.path.join(HERE, "specs", "types3.yaml"))
    ap.add_argument("--question", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra BLIS flags after --")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    date = dt.date.today().isoformat()
    exp = os.path.join(HERE, "experiments", f"{date}_{a.name}")
    for d in ("specs", "runs"):
        os.makedirs(os.path.join(exp, d), exist_ok=True)

    manifest = {
        "name": a.name, "date": date, "commit": git_commit(),
        "binary": BIN, "base_flags": BASE_FLAGS, "extra_flags": a.extra,
        "blocks": a.blocks, "num_requests": a.num_requests,
        "rates": rates, "seeds": seeds, "spec_template": a.spec,
    }
    json.dump(manifest, open(os.path.join(exp, "manifest.json"), "w"), indent=2)
    with open(os.path.join(exp, "README.md"), "w") as f:
        f.write(f"# {a.name} ({date})\n\n{a.question or '(no question recorded)'}\n")

    rows = []
    for rate in rates:
        rtag = f"rate{rate:g}"
        spec = os.path.join(exp, "specs", f"{rtag}.yaml")
        write_spec(a.spec, spec, rate)
        for seed in seeds:
            tag = f"{rtag}_s{seed}"
            out = os.path.join(exp, "runs", tag + ".json")
            log = os.path.join(exp, "runs", tag + ".log")
            cmd = [BIN, "run", *BASE_FLAGS,
                   "--workload-spec", spec,
                   "--total-kv-blocks", str(a.blocks),
                   "--num-requests", str(a.num_requests),
                   "--seed", str(seed),
                   "--metrics-path", out, *a.extra]
            if a.dry_run:
                print(" ".join(cmd))
                continue
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=lf).returncode
            if rc != 0 or not os.path.exists(out):
                print(f"FAILED {tag} (rc={rc}), see {log}")
                continue
            m = json.load(open(out))
            row = {"rate": rate, "seed": seed}
            row.update({k: m.get(k) for k in SUMMARY_FIELDS[2:]})
            rows.append(row)
            print(f"{tag:<14} queued={row['still_queued']:5d} preempt={row['preemption_count']:5d} "
                  f"ttft_p99={row['ttft_p99_ms']:9,.0f} e2e_mean={row['e2e_mean_ms']:8,.0f} "
                  f"tok/s={row['tokens_per_sec']:6,.0f}")

    if rows:
        with open(os.path.join(exp, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} runs to {exp}")


if __name__ == "__main__":
    main()
