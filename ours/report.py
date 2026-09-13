#!/usr/bin/env python3
"""ours/report.py <experiment folder>: write report.pdf for one experiment.

Structure (an e-companion numerical-details section):
    1  Question                      from README.md (text before the first "##" heading)
    2  Simulator and build           BLIS pin, fork commit, binary, run mode
    3  System configuration          model, hardware, instances, KV pool, engine budgets
    4  Workload                      type table from the spec yaml, rate grid, seeds, horizon
    5  Control policies              admission, routing, snapshot interval, scheduler, eviction
    6  Statistics                    window, estimators, censoring, confidence intervals
    7  Results                       summary.csv tables (all, then per type) and sweep figures
    8  Notes and interpretation      remaining README.md sections
    9  Artifacts                     files in the folder
    Appendix                         the six-panel figure of the first sample path at every rate

Compiles with pdflatex (MiKTeX). Re-runnable; README.md is the editable text source.
"""
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FLAG_DEFAULTS = {  # BLIS defaults for flags a profile may leave unset
    "--admission-policy": "always-admit", "--routing-policy": "round-robin",
    "--routing-scorers": "precise-prefix-cache:2,queue-depth:1,kv-utilization:1 (only under weighted)",
    "--snapshot-refresh-interval": "50000", "--scheduler": "fcfs", "--preemption-policy": "fcfs",
    "--batch-formation": "vllm", "--max-num-batched-tokens": "2048", "--max-num-seqs": "256",
    "--long-prefill-token-threshold": "0", "--block-size-in-tokens": "16", "--tp": "1",
}


def esc(s):
    s = str(s).replace("\\", "/")  # Windows paths: show with forward slashes
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}"), ("--", "-{}-")):
        s = s.replace(a, b)
    return s


def flags_to_dict(flags):
    d, i = {}, 0
    while i < len(flags):
        f = flags[i]
        if f.startswith("--") and i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            d[f] = flags[i + 1]
            i += 2
        else:
            d[f] = "true"
            i += 1
    return d


def flag(d, name):
    return d.get(name, FLAG_DEFAULTS.get(name, "(BLIS default)"))


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, (int,)) or (isinstance(x, float) and nd == 0):
        return f"{x:,.0f}"
    return f"{x:,.{nd}f}"


def kv_table(rows, caption=None):
    out = [r"\begin{tabular}{@{}p{0.34\linewidth}p{0.62\linewidth}@{}}", r"\toprule",
           r"\textbf{Parameter} & \textbf{Value} \\", r"\midrule"]
    for k, v in rows:
        out.append(f"{esc(k)} & {esc(v)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        out = [r"\begin{table}[H]\centering\small"] + out + [rf"\caption{{{esc(caption)}}}", r"\end{table}"]
    return "\n".join(out)


def results_table(summary, t, caption):
    cols = [("rate", "rate", 0), ("arrived_mean", "arrived", 0), ("unfinished_frac_mean", "unfinished", 3),
            ("rejected_frac_mean", "rejected", 3), ("delay_mean_ms_mean", "wait mean", 0),
            ("delay_mean_ms_ci95", "CI95", 0), ("delay_p99_ms_mean", "wait p99", 0),
            ("ttft_p99_ms_mean", "TTFT p99", 0), ("e2e_mean_ms_mean", "E2E mean", 0),
            ("preempt_per_arrival_mean", "preempt/arr", 3), ("output_tok_per_s_mean", "out tok/s", 0)]
    s = summary[summary["type"] == t].sort_values("rate")
    cols = [c for c in cols if c[0] in s.columns]
    out = [r"\begin{table}[H]\centering\scriptsize",
           r"\begin{tabular}{@{}" + "r" * len(cols) + r"@{}}", r"\toprule",
           " & ".join(esc(c[1]) for c in cols) + r" \\", r"\midrule"]
    for _, r in s.iterrows():
        out.append(" & ".join(fmt(r[c[0]], c[2]) for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{{esc(caption)}. Times in ms; wait = queue wait before first scheduling; "
            r"CI95 = half-width of the 95 percent t interval across seeds for the wait mean; "
            r"unfinished and rejected are fractions of window arrivals; preempt/arr = preemptions per arrival.}",
            r"\end{table}"]
    return "\n".join(out)


def md_to_tex(md):
    """Minimal markdown: ## headings, paragraphs, bullet lists; tables dropped (numbers come from csv)."""
    out, para, in_list = [], [], False

    def flush():
        nonlocal para
        if para:
            out.append(esc(" ".join(para)))
            out.append("")
            para = []
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|") or s.startswith("![") or s.startswith("# "):
            continue
        if s.startswith("## "):
            flush()
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(rf"\subsection*{{{esc(s[3:])}}}")
        elif s.startswith("- ") or s.startswith("* "):
            flush()
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + esc(s[2:]))
        elif s == "":
            flush()
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
        else:
            para.append(re.sub(r"`([^`]*)`", r"\1", s))
    flush()
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def build(exp):
    exp = os.path.abspath(exp)
    name = os.path.basename(exp)
    m = json.load(open(os.path.join(exp, "manifest.json")))
    fl = flags_to_dict(m.get("base_flags", []) + m.get("extra_flags", []))
    specs = sorted(glob.glob(os.path.join(exp, "specs", "*.yaml")))
    spec = yaml.safe_load(open(specs[0])) if specs else {"clients": []}
    md = open(os.path.join(exp, "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(exp, "README.md")) else ""
    parts = re.split(r"^## ", md, flags=re.M)
    question = re.sub(r"^# .*$", "", parts[0], flags=re.M).strip()
    rest = "".join("## " + p for p in parts[1:])
    summary = pd.read_csv(os.path.join(exp, "summary.csv")) if os.path.exists(os.path.join(exp, "summary.csv")) else None
    fork_commit = m.get("commit", "unknown")
    blocks = int(m["blocks"])
    bs = int(flag(fl, "--block-size-in-tokens"))
    horizon = m.get("horizon_s")
    warm, tail = float(m.get("warmup_s") or 0), float(m.get("tail_s") or 0)

    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs,graphicx,float,longtable,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\begin{document}"""]
    tex.append(rf"\section*{{Numerical details: {esc(name)}}}")
    tex.append(rf"Generated by ours/report.py from manifest.json, specs/, summary.csv and README.md on {esc(pd.Timestamp.today().date())}.")

    tex.append(r"\subsection*{1. Question}")
    tex.append(esc(question) if question else "(no question recorded)")

    tex.append(r"\subsection*{2. Simulator and build}")
    tex.append(kv_table([
        ("Simulator", "BLIS (inference-sim/inference-sim), discrete-event simulator of a multi-instance LLM serving cluster"),
        ("Upstream pin", "commit 2ebef6a (2026-09-04)"),
        ("Fork commit (branch ours)", fork_commit),
        ("Fork additions used here", "per-request preemption_count, wasted_tokens, status; rejected requests in the metrics json; --state-sample-interval. All file-only; stdout byte-identical to upstream."),
        ("Latency model", "trained-physics (BLIS default): roofline basis functions with learned alpha and beta coefficients from defaults.yaml"),
        ("Run mode", "fixed horizon: Poisson arrivals continue until the horizon, then the run stops" if horizon else "drain: fixed number of arrivals, run until all complete"),
        ("Driver", f"python ours/sweep.py {m.get('name', name)} --profile {m.get('profile', m.get('routing', ''))} ..."),
    ]))

    tex.append(r"\subsection*{3. System configuration}")
    tex.append(kv_table([
        ("Model", flag(fl, "--model")),
        ("Hardware", f"{flag(fl, '--hardware')}, tensor parallel {flag(fl, '--tp')} (one GPU per instance)"),
        ("Instances", flag(fl, "--num-instances")),
        ("KV cache per instance", f"{blocks:,} blocks x {bs} tokens = {blocks * bs:,} tokens" + (" (auto-derived from GPU memory)" if blocks == 15909 else " (explicit, memory-bound regime)")),
        ("Max tokens scheduled per step", flag(fl, "--max-num-batched-tokens")),
        ("Max sequences per step", flag(fl, "--max-num-seqs")),
        ("Chunked prefill threshold", f"{flag(fl, '--long-prefill-token-threshold')} (0 = prefill takes whatever budget remains)"),
        ("Block size", f"{bs} tokens"),
    ]))

    tex.append(r"\subsection*{4. Workload}")
    rows = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{@{}llrlll@{}}", r"\toprule",
            r"client & tenant & share & arrival & input tokens & output tokens \\", r"\midrule"]
    for c in spec.get("clients", []):
        i, o = c.get("input_distribution", {}), c.get("output_distribution", {})
        def dist(d):
            p = d.get("params", {})
            if d.get("type") == "constant":
                return f"constant {p.get('value')}"
            return f"{d.get('type')} mean {p.get('mean')} in [{p.get('min')}, {p.get('max')}]"
        rows.append(f"{esc(c.get('id'))} & {esc(c.get('tenant_id'))} & {c.get('rate_fraction')} & "
                    f"{esc(c.get('arrival', {}).get('process'))} & {esc(dist(i))} & {esc(dist(o))} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Request types. Share is the fraction of the total arrival rate; each client is an independent Poisson stream.}",
             r"\end{table}"]
    tex.append("\n".join(rows))
    tex.append(kv_table([
        ("Total arrival rates (req/s)", ", ".join(f"{r:g}" for r in m.get("rates", []))),
        ("Seeds", f"{len(m.get('seeds', []))} ({m['seeds'][0]} to {m['seeds'][-1]})" if m.get("seeds") else ""),
        ("Horizon", f"{horizon:g} s of simulated time" if horizon else f"{m.get('num_requests')} arrivals then drain"),
        ("Spec template", os.path.relpath(m.get("spec_template", ""), ROOT) if m.get("spec_template") else ""),
    ]))

    tex.append(r"\subsection*{5. Control policies}")
    rp = flag(fl, "--routing-policy")
    tex.append(kv_table([
        ("Profile", m.get("profile", m.get("routing", ""))),
        ("Admission", flag(fl, "--admission-policy")),
        ("Routing", rp + (f", scorers {flag(fl, '--routing-scorers')}" if rp == "weighted" else "")),
        ("Router state refresh", f"{flag(fl, '--snapshot-refresh-interval')} us (0 = router sees live instance state)"),
        ("Instance scheduler", f"{flag(fl, '--scheduler')} (queue order within an instance)"),
        ("Eviction victim", f"{flag(fl, '--preemption-policy')} (fcfs = last request in the running batch; recompute from scratch)"),
        ("Batch formation", flag(fl, "--batch-formation")),
    ]))

    tex.append(r"\subsection*{6. Statistics}")
    win = f"[{warm:g} s, {horizon - tail:g} s)" if horizon else f"[{warm:g} s, end - {tail:g} s)"
    tex.append(kv_table([
        ("Sample path", "one BLIS run = one seed = one trajectory; all statistics are first computed per path"),
        ("Steady-state window", f"requests whose arrival time lies in {win}; warm-up {warm:g} s, tail {tail:g} s"),
        ("Per-path estimators", "mean, p50, p99 of TTFT, E2E and queue wait over completed window requests; unfinished (still queued or running at the horizon) and rejected fractions of window arrivals; preemptions and wasted tokens per arrival; output tokens per second of window (time average)"),
        ("Censoring", "unfinished requests are counted, never timed; their latency is not imputed"),
        ("Across paths", "mean over seeds and a 95 percent Student t interval with n - 1 degrees of freedom; reported as mean and CI half-width"),
        ("Comparability", "policies are compared on identical seed sets (common random numbers)"),
    ]))

    tex.append(r"\subsection*{7. Results}")
    if summary is not None:
        tex.append(results_table(summary, "all", "All types"))
        for t in ("type1", "type2", "type3"):
            if (summary["type"] == t).any():
                tex.append(results_table(summary, t, t))
    for fig, cap in (("sweep_panel.png", "Window metrics against arrival rate, mean and 95 percent CI over seeds."),
                     ("objective.png", "Revenue management objective per second (placeholder prices, see ours/specs/objective.yaml).")):
        if os.path.exists(os.path.join(exp, fig)):
            tex.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{{fig}}}\caption{{{esc(cap)}}}\end{{figure}}")
    panels = sorted(glob.glob(os.path.join(exp, "runs", "*_panel.png")))
    if panels:
        tex.append(esc(f"Per-path six-panel figures: {len(panels)} files in runs/ (one per rate and seed); "
                       f"the first sample path of every rate is reproduced in the appendix."))

    tex.append(r"\subsection*{8. Notes and interpretation}")
    tex.append(md_to_tex(rest) if rest.strip() else "(none)")

    first_seed = min(m.get("seeds", [1]) or [1])
    appendix = []
    for rate in m.get("rates", []):
        png = f"runs/rate{rate:g}_s{first_seed}_panel.png"  # forward slash: LaTeX path
        if os.path.exists(os.path.join(exp, png)):
            appendix.append(
                rf"egin{{figure}}[p]\centering\includegraphics[width=\linewidth]{{{png}}}"
                rf"\caption{{Sample path at rate {rate:g} req/s, seed {first_seed}: latency densities, queue wait against "
                r"arrival time, KV occupancy, queue and batch sizes, evictions per second. Dashed lines mark the window.}\end{figure}")

    tex.append(r"\subsection*{9. Artifacts}")
    files = sorted(os.listdir(exp))
    nruns = len(glob.glob(os.path.join(exp, "runs", "*.json*")))
    tex.append(esc(", ".join(f for f in files if f != "runs")) + esc(f"; runs/: {nruns} runs, each with metrics json (gzipped), BLIS log, state csv and panel png."))
    if appendix:
        tex.append(r"\clearpage\subsection*{Appendix: one sample path per rate}")
        tex.extend(appendix)
    tex.append(r"\end{document}")

    texpath = os.path.join(exp, "report.tex")
    open(texpath, "w", encoding="utf-8").write("\n\n".join(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                           cwd=exp, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("pdflatex failed; report.tex kept for inspection")
    for ext in (".aux", ".log", ".out"):
        try:
            os.remove(os.path.join(exp, "report" + ext))
        except OSError:
            pass
    print("wrote", os.path.join(exp, "report.pdf"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(sys.argv[1])
