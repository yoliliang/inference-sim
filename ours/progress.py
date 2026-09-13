#!/usr/bin/env python3
"""ours/progress.py <experiment folder> [--watch]: progress bar for a running sweep.

Counts finished sample paths (runs/*.json.gz) against the grid in manifest.json and
estimates the remaining time from the finish times of the completed paths. With --watch
it refreshes every 30 s until every path is done.
"""
import glob
import json
import os
import sys
import time


def status(exp):
    """Finished and total work, where the work of one path is proportional to
    n x total rate (event count grows with both), so the ETA is weighted, not a plain count."""
    m = json.load(open(os.path.join(exp, "manifest.json")))
    if m.get("experiment") == "scaling":
        points = [(f"n{n}", n * m["rate_per_instance"]) for n in m["instances"]]  # work grows with the total rate, that is with n
    else:
        n = (m.get("instances") or [4])[0]
        points = [(f"rate{r:g}", n * r) for r in m["rates"]]
    seeds = m["seeds"]
    total, done = 0, 0
    work_total, work_done = 0.0, 0.0
    for p, w in points:
        for s in seeds:
            total += 1
            work_total += w
            if os.path.exists(os.path.join(exp, "runs", f"{p}_s{s}.json.gz")):
                done += 1
                work_done += w
    started = min((os.path.getmtime(f) for f in glob.glob(os.path.join(exp, "runs", "*.log"))), default=None)
    eta = None
    if work_done and started and done < total:
        elapsed = time.time() - started
        eta = elapsed / work_done * (work_total - work_done)
    return m, total, done, eta


def bar(done, total, width=40):
    k = int(width * done / total) if total else 0
    return "[" + "#" * k + "." * (width - k) + f"] {done}/{total}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    exp = args[0]
    while True:
        m, total, done, eta = status(exp)
        line = f"{os.path.basename(exp)}  {bar(done, total)}"
        if eta is not None and done < total:
            line += f"  about {eta / 60:.0f} min left"
        print(line, flush=True)
        if "--watch" not in sys.argv or done >= total:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
