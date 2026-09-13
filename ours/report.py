#!/usr/bin/env python3
"""ours/report.py <experiment folder>: write report.pdf for one experiment.

Structure (an e-companion numerical-details section, plain language throughout):
    1  Question                      from README.md (text before the first "##" heading)
    2  Simulator                     BLIS version, fork commit, run mode
    3  System                        the served model, the GPUs, the KV cache, the per-step limits
    4  Workload                      the three request types in words and in a table; rates, seeds, horizon
    5  Control policies              table, then a prose description of what each policy does
    6  Statistics                    what one sample path is, the window, the estimators, the intervals
    7  Results                       glossary of the columns, then the tables and sweep figures
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
    "--routing-scorers": "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    "--snapshot-refresh-interval": "50000", "--scheduler": "fcfs", "--preemption-policy": "fcfs",
    "--batch-formation": "vllm", "--max-num-batched-tokens": "2048", "--max-num-seqs": "256",
    "--long-prefill-token-threshold": "0", "--block-size-in-tokens": "16", "--tp": "1",
    "--num-instances": "1",
}

TYPE_WORDS = {
    "type1": "short chat: a short prompt and a short answer",
    "type2": "retrieval-augmented (RAG): a long prompt of retrieved documents and a short answer",
    "type3": "long generation: a medium prompt and a long answer; the memory hog, because a request holds its KV cache until its last token",
}

SCORER_WORDS = {
    "queue-depth": "queue depth: the instance with the fewest waiting requests scores 1, the one with the most scores 0, the rest linearly in between",
    "kv-utilization": "KV utilisation: score = 1 minus the fraction of the instance's KV cache in use",
    "precise-prefix-cache": "prefix cache: the fraction of the request's prompt blocks already cached on the instance, rescaled so the best instance scores 1 (zero for our workload, whose prompts share no prefix)",
    "vllm-dp": "vLLM data-parallel balancer: score falls with 4 x waiting + running, the load count vLLM's own multi-replica mode uses",
    "load-balance": "load balance: 1 / (1 + waiting + running)",
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


def kv_table(rows, caption=None):
    out = [r"\begin{tabular}{@{}p{0.30\linewidth}p{0.66\linewidth}@{}}", r"\toprule",
           r"\textbf{Parameter} & \textbf{Value} \\", r"\midrule"]
    for k, v in rows:
        out.append(f"{esc(k)} & {esc(v)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        out = [r"\begin{table}[H]\centering\small"] + out + [rf"\caption{{{esc(caption)}}}", r"\end{table}"]
    return "\n".join(out)


def para(s):
    return esc(s) + "\n"


# results table columns: (summary column, header, kind) with kind in {int, pct, s, rate1, rate3}
RESULT_COLS = [
    ("rate", "rate (req/s)", "int"), ("arrived_mean", "arrivals", "int"),
    ("unfinished_frac_mean", "unfinished", "pct"), ("rejected_frac_mean", "rejected", "pct"),
    ("delay_mean_ms_mean", "wait mean (s)", "s"), ("delay_mean_ms_ci95", "CI (s)", "s"),
    ("delay_p99_ms_mean", "wait p99 (s)", "s"), ("ttft_p99_ms_mean", "TTFT p99 (s)", "s"),
    ("e2e_mean_ms_mean", "E2E mean (s)", "s"), ("preempt_per_s_mean", "evictions/s", "rate1"),
    ("preempt_per_completed_mean", "evictions per completion", "rate3"),
    ("output_tok_per_s_mean", "output tok/s", "int"),
]


def cell(x, kind):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if kind == "int":
        return f"{x:,.0f}"
    if kind == "pct":
        return f"{100 * x:.1f}" + r"\%"
    if kind == "s":
        return f"{x / 1000:,.2f}" if x >= 100 else f"{x / 1000:.3f}"
    if kind == "rate1":
        return f"{x:.1f}"
    return f"{x:.3f}"


def results_table(summary, t, caption):
    s = summary[summary["type"] == t].sort_values("rate")
    cols = [c for c in RESULT_COLS if c[0] in s.columns]
    out = [r"\begin{table}[H]\centering\small", r"\resizebox{\linewidth}{!}{",
           r"\begin{tabular}{@{}" + "r" * len(cols) + r"@{}}", r"\toprule",
           " & ".join(esc(c[1]) for c in cols) + r" \\", r"\midrule"]
    for _, r in s.iterrows():
        out.append(" & ".join(cell(r[c[0]], c[2]) for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{esc(caption)}}}", r"\end{table}"]
    return "\n".join(out)


def md_to_tex(md):
    """Minimal markdown: ## headings, paragraphs, bullet lists; tables dropped (numbers come from csv)."""
    out, buf, in_list = [], [], False

    def flush():
        nonlocal buf
        if buf:
            out.append(esc(" ".join(buf)))
            out.append("")
            buf = []
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
            buf.append(re.sub(r"`([^`]*)`", r"\1", s))
    flush()
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def dist_words(d):
    p = d.get("params", {})
    if d.get("type") == "constant":
        return f"always {p.get('value')}"
    if d.get("type") == "exponential":
        return f"exponential, mean {p.get('mean')}, clamped to [{p.get('min')}, {p.get('max')}]"
    return f"{d.get('type')} {p}"


def build(exp):
    exp = os.path.abspath(exp)
    name = os.path.basename(exp)
    m = json.load(open(os.path.join(exp, "manifest.json")))
    fl = flags_to_dict(m.get("base_flags", []) + m.get("extra_flags", []))
    specs = sorted(glob.glob(os.path.join(exp, "specs", "*.yaml")))
    spec = yaml.safe_load(open(specs[0])) if specs else {"clients": []}
    md_path = os.path.join(exp, "README.md")
    md = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
    parts = re.split(r"^## ", md, flags=re.M)
    question = re.sub(r"^# .*$", "", parts[0], flags=re.M).strip()
    rest = "".join("## " + p for p in parts[1:])
    sp = os.path.join(exp, "summary.csv")
    summary = pd.read_csv(sp) if os.path.exists(sp) else None
    blocks = int(m["blocks"])
    bs = int(flag(fl, "--block-size-in-tokens"))
    horizon = m.get("horizon_s")
    warm, tail = float(m.get("warmup_s") or 0), float(m.get("tail_s") or 0)
    n_inst = int(flag(fl, "--num-instances"))
    step_tokens, step_seqs = flag(fl, "--max-num-batched-tokens"), flag(fl, "--max-num-seqs")
    refresh_us = int(flag(fl, "--snapshot-refresh-interval"))
    rp = flag(fl, "--routing-policy")
    scorers = flag(fl, "--routing-scorers") if rp == "weighted" else ""

    tex = [r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs,graphicx,float,longtable,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\begin{document}"""]
    tex.append(rf"\section*{{Numerical details: {esc(name)}}}")
    tex.append(esc(f"Generated by ours/report.py on {pd.Timestamp.today().date()} from manifest.json, specs/, summary.csv and README.md."))

    # 1
    tex.append(r"\subsection*{1. Question}")
    tex.append(esc(question) if question else "(no question recorded)")

    # 2
    tex.append(r"\subsection*{2. Simulator}")
    tex.append(kv_table([
        ("Simulator", "BLIS"),
        ("BLIS version", "the upstream code as of commit 2ebef6a (2026-09-04); our fork is pinned to it"),
        ("Our fork version", f"commit {str(m.get('commit', 'unknown'))[:12]} of branch ours"),
        ("What the fork adds", "extra output only: per-request eviction counts, request status, rejected requests, and the periodic state and snapshot files. The simulation itself is unchanged; upstream and fork produce byte-identical results."),
        ("Latency model", "BLIS's calibrated step-time model (its default): the duration of one GPU step is a fitted function of how many tokens are prefilled and decoded in that step"),
        ("Run mode", "fixed horizon: requests keep arriving at the given rate until the horizon, then the run stops; whatever is still in the system at that moment is recorded as unfinished" if horizon else "drain: a fixed number of arrivals, then the run continues until every request completes"),
    ]))

    # 3
    tex.append(r"\subsection*{3. System}")
    tex.append(kv_table([
        ("Served model", f"{flag(fl, '--model')}: the language model every instance runs (Qwen3, 14 billion parameters)"),
        ("Instances", f"{n_inst} identical instances, each one {flag(fl, '--hardware')} GPU holding a full copy of the model; a request is served entirely by one instance"),
        ("KV cache per instance", f"{blocks:,} blocks of {bs} tokens = {blocks * bs:,} tokens. This is the memory that limits how many requests an instance can hold at once: every token of a request's prompt and answer occupies one slot until the request finishes." + (" 15,909 is the value BLIS derives from the GPU's memory after the model weights." if blocks == 15909 else " Set explicitly to create a memory-bound regime.")),
        ("Tokens processed per step", f"{step_tokens}. Each GPU step processes at most this many new tokens across all requests in the batch (prompt tokens being prefilled plus one token per decoding request). It is a compute budget per step, not memory; a long prompt is split over several steps."),
        ("Requests per step", f"{step_seqs}: at most this many requests in the running batch at once"),
    ]))

    # 4
    tex.append(r"\subsection*{4. Workload}")
    tex.append(para("Requests arrive from three independent Poisson streams, one per type; the total rate is split by the shares in the table. Each request has a prompt length (input tokens) and an answer length (output tokens), drawn once at arrival. The arrival times and the lengths are the only randomness in a run; the seed fixes both."))
    rows = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{@{}p{0.09\linewidth}p{0.40\linewidth}rp{0.13\linewidth}p{0.26\linewidth}@{}}", r"\toprule",
            r"type & meaning & share & prompt tokens & answer tokens \\", r"\midrule"]
    for c in spec.get("clients", []):
        t = c.get("tenant_id")
        rows.append(f"{esc(t)} & {esc(TYPE_WORDS.get(t, c.get('id')))} & {c.get('rate_fraction')} & "
                    f"{esc(dist_words(c.get('input_distribution', {})))} & {esc(dist_words(c.get('output_distribution', {})))} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Request types. Clamped: the exponential draw is rounded and then cut to the interval, so a draw above the maximum becomes the maximum.}",
             r"\end{table}"]
    tex.append("\n".join(rows))
    tex.append(kv_table([
        ("Total arrival rates (req/s)", ", ".join(f"{r:g}" for r in m.get("rates", []))),
        ("Seeds", f"{len(m.get('seeds', []))} ({m['seeds'][0]} to {m['seeds'][-1]}); one seed = one sample path" if m.get("seeds") else ""),
        ("Horizon", f"{horizon:g} s of simulated time per sample path" if horizon else f"{m.get('num_requests')} arrivals then drain"),
        ("Economic parameters", "not used in this experiment. Rewards per type and the congestion cost (ours/specs/objective.yaml) are still to be decided."),
    ]))

    # 5
    tex.append(r"\subsection*{5. Control policies}")
    tex.append(kv_table([
        ("Profile", m.get("profile", m.get("routing", ""))),
        ("Admission", flag(fl, "--admission-policy")),
        ("Routing", rp + (f" with scorers {scorers}" if scorers else "")),
        ("Router information", f"refreshed every {refresh_us / 1000:g} ms" if refresh_us > 0 else "live (the router sees the true instance state at every decision)"),
        ("Queue order in an instance", flag(fl, "--scheduler")),
        ("Eviction rule", flag(fl, "--preemption-policy")),
        ("Batch formation", flag(fl, "--batch-formation")),
    ]))
    tex.append(r"\paragraph*{What these policies do.}")
    adm = flag(fl, "--admission-policy")
    tex.append(para("Admission: " + ("always-admit accepts every arriving request; nothing is ever turned away." if adm == "always-admit" else f"{adm} (see the BLIS admission guide).")))
    if rp == "weighted":
        terms = []
        for item in scorers.split(","):
            n, w = item.split(":")
            terms.append(f"{w} x [{SCORER_WORDS.get(n, n)}]")
        tex.append(para("Routing: for each arriving request the router computes a score for every instance, score = " + " + ".join(terms) + ", and sends the request to the instance with the highest score (ties broken at random). "
                        + (f"The queue and KV figures it uses are refreshed every {refresh_us / 1000:g} ms, as a real gateway scraping metrics would, so they can be slightly stale." if refresh_us > 0 else "It uses the true current queue and KV figures.")))
    elif rp == "round-robin":
        tex.append(para("Routing: round-robin sends arrivals to the instances in turn, ignoring their state."))
    else:
        tex.append(para(f"Routing: {rp}."))
    tex.append(para("Inside an instance (batch formation = vllm, the rule vLLM itself uses): time advances in steps. At each step the instance first continues every request already in the running batch (each decodes one more token). It then admits waiting requests in queue order (fcfs = first come first served) as long as three limits hold: the step's token budget, the maximum number of requests per step, and free KV cache for the new request's prompt. A long prompt that does not fit the remaining token budget is prefilled in chunks over successive steps. When a running request needs one more KV block for its next token and none is free, the instance evicts the most recently admitted running request (fcfs eviction, the tail of the batch): its KV cache is freed, it goes back to the front of the queue, and when re-admitted it recomputes everything from scratch. This recomputation is the waste that evictions cause."))

    # 6
    tex.append(r"\subsection*{6. Statistics}")
    win = f"[{warm:g} s, {horizon - tail:g} s)" if horizon else f"[{warm:g} s, end - {tail:g} s)"
    tex.append(kv_table([
        ("Sample path", "one BLIS run with one seed. Every statistic is first computed on one path; paths are the independent replications"),
        ("Steady-state window", f"requests that arrive in {win}. The first {warm:g} s are discarded because the system starts empty and takes time to fill; the last {tail:g} s are discarded because late arrivals have not had time to finish"),
        ("Per-path statistics", "over the completed requests that arrived in the window: mean, median and 99th percentile of TTFT, E2E and queue wait; fractions of window arrivals that were unfinished at the horizon or rejected; evictions per second and per completion; output tokens per second (a time average over the window)"),
        ("Unfinished requests", "counted, never timed: a request still in the system at the horizon contributes to the unfinished fraction and to nothing else"),
        ("Across paths", "mean over the seeds and a 95 percent confidence half-width from the Student t distribution with (number of seeds minus 1) degrees of freedom"),
        ("Comparisons", "different policies are run on the same seeds, so they face identical arrivals and lengths (common random numbers)"),
    ]))

    # 7
    tex.append(r"\subsection*{7. Results}")
    tex.append(r"\paragraph*{Reading the columns.}")
    tex.append(para("Queue wait: from arrival until the instance first puts the request into a batch. TTFT (time to first token): from arrival until the first answer token, that is wait plus prompt processing. E2E (end to end): from arrival until the last answer token. p99: the 99th percentile, the value that 99 percent of requests stay below; the tail of the distribution. CI: half-width of the 95 percent confidence interval of the wait mean across seeds. Unfinished, rejected: percent of the requests arriving in the window. Evictions/s: evictions per second of simulated time, whole cluster. Evictions per completion: evictions divided by completed window requests. Output tok/s: answer tokens produced per second, whole cluster. Times are in seconds."))
    if summary is not None:
        tex.append(results_table(summary, "all", "All types, mean over seeds."))
        for t in ("type1", "type2", "type3"):
            if (summary["type"] == t).any():
                tex.append(results_table(summary, t, f"{t} only, mean over seeds."))
    for fig, cap in (("sweep_panel.png", "Window statistics against the total arrival rate: mean over seeds with 95 percent confidence bars."),
                     ("objective.png", "Revenue management objective per second (placeholder prices, see ours/specs/objective.yaml).")):
        if os.path.exists(os.path.join(exp, fig)):
            tex.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{{fig}}}\caption{{{esc(cap)}}}\end{{figure}}")

    # 8
    tex.append(r"\subsection*{8. Notes and interpretation}")
    tex.append(md_to_tex(rest) if rest.strip() else "(none)")

    # 9
    tex.append(r"\subsection*{9. Artifacts}")
    files = sorted(f for f in os.listdir(exp) if not f.endswith((".tex", ".aux", ".log", ".out")))
    nruns = len(glob.glob(os.path.join(exp, "runs", "*.json*")))
    tex.append(esc(", ".join(f for f in files if f != "runs")) + esc(f"; runs/: {nruns} sample paths, each with the per-request json (gzipped), the BLIS log, the state csv and its six-panel figure."))

    # appendix
    first_seed = min(m.get("seeds", [1]) or [1])
    appendix = []
    for rate in m.get("rates", []):
        png = f"runs/rate{rate:g}_s{first_seed}_panel.png"
        if os.path.exists(os.path.join(exp, png)):
            appendix.append(
                rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{{png}}}"
                rf"\caption{{One sample path at {rate:g} req/s (seed {first_seed}). Top: distributions of TTFT and E2E by type. Middle left: each request's queue wait against its arrival time. Middle right: KV cache in use per instance. Bottom left: requests waiting (solid) and in the batch (dotted) per instance. Bottom right: evictions per second by type of the evicted request. Dashed lines mark the steady-state window.}}\end{{figure}}\clearpage")
    if appendix:
        tex.append(r"\clearpage\subsection*{Appendix: one sample path per rate}")
        tex.append(para("Read each figure left to right, top to bottom. The system starts empty and the KV cache fills as long requests accumulate. If the cache reaches 100 percent, every new admission from then on needs an eviction, the queue starts to grow and the wait rises. That moment is the visible transition from stable to overloaded on a single path."))
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
    for ext in (".tex", ".aux", ".log", ".out"):
        try:
            os.remove(os.path.join(exp, "report" + ext))
        except OSError:
            pass
    print("wrote", os.path.join(exp, "report.pdf"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(sys.argv[1])
