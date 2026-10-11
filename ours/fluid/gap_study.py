#!/usr/bin/env python3
"""ours/fluid/gap_study.py: how long Gurobi needs to bring the per-instance fluid problem (28)
to a 1 percent optimality gap, at the hardest points of the 2026-10-11 experiments.

Each point is re-solved from scratch of the solver's search (the stored fluid solution is
given as the starting point), with MIPGap = 0.01 and a long time limit. The solver log is
parsed for the first time the gap fell below 5, 2 and 1 percent. Results go to
ours/fluid/work/gap_study.csv. Options: --threads, --time-limit, --mipfocus.
"""
import argparse, json, os, re, subprocess, sys, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "experiments")
WORK = os.path.join(HERE, "work")
LINE = re.compile(r"\s(-|[\d.]+)%\s+\S+\s+(\d+)s\s*$")


def instances_arg(fluid):
    out, prev = [], None
    for inst in fluid["instances"]:
        if out and out[-1][0] == inst["gpu"]:
            out[-1][1] += 1
        else:
            out.append([inst["gpu"], 1])
    return ",".join(f"{g}:{k}" for g, k in out)


def study(folder, point, a, tag):
    path = os.path.join(EXP, folder)
    man = json.load(open(os.path.join(path, "manifest.json")))
    fy = os.path.join(path, "specs", f"fluid_{point}.yaml")
    fl = yaml.safe_load(open(fy))
    out = os.path.join(WORK, f"gapstudy_{tag}.yaml")
    env = {**os.environ, "FLUID_GAP": "0.01", "FLUID_THREADS": str(a.threads)}
    if a.mipfocus:
        env["FLUID_MIPFOCUS"] = str(a.mipfocus)
    cmd = [sys.executable, os.path.join(HERE, "solve.py"), "--rate", str(fl["meta"]["rate"]), "--instances", instances_arg(fl),
           "--blocks", str(fl["meta"]["blocks"]), "--spec", man.get("spec_template") or os.path.join(os.path.dirname(HERE), "specs", "types3.yaml"),
           "--objective", man.get("objective") or os.path.join(os.path.dirname(HERE), "specs", "objective.yaml"),
           "--time-limit", str(a.time_limit), "--start", fy, "--out", out, "--quiet"]
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    wall = time.time() - t0
    log = open(os.path.join(WORK, f"gapstudy_{tag}.log")).read()
    first = {}
    for line in log.splitlines():
        m = LINE.search(line)
        if not m or m.group(1) == "-":
            continue
        g, t = float(m.group(1)), int(m.group(2))
        for thr in (5, 2, 1):
            if g <= thr and thr not in first:
                first[thr] = t
    meta = yaml.safe_load(open(out))["meta"]
    return {"folder": folder, "point": point, "n": len(fl["instances"]), "mipfocus": a.mipfocus or 0, "threads": a.threads,
            "old_gap_pct": 100 * fl["meta"]["gap"], "old_V": fl["meta"]["value"],
            "new_V": meta["value"], "new_bound": meta["bound"], "new_gap_pct": 100 * (meta["gap"] or 0), "status": meta["status"],
            "t_gap5_s": first.get(5), "t_gap2_s": first.get(2), "t_gap1_s": first.get(1), "wall_s": round(wall)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", nargs="+", help="folder:point, e.g. 2026-10-11_scaling_robust_h0.1_fluid_dual_over:n32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--mipfocus", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(WORK, "gap_study.csv"))
    a = ap.parse_args()
    import pandas as pd
    for fp in a.points:
        folder, point = fp.split(":")
        tag = f"{folder.split('robust_')[-1].split('scale64_')[-1]}_{point}_mf{a.mipfocus}"
        row = study(folder, point, a, tag)
        print(row, flush=True)
        df = pd.DataFrame([row])
        df.to_csv(a.csv, mode="a", header=not os.path.exists(a.csv), index=False)


if __name__ == "__main__":
    main()
