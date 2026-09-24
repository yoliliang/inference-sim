#!/usr/bin/env python3
"""ours/report.py <experiment folder>            write report.pdf for one experiment
   ours/report.py --environment <reference folder>   write ours/experiments/environment.pdf

Three kinds of document, all plain language, all compiled with pdflatex (MiKTeX):

environment.pdf (common to every experiment; regenerate when the environment changes)
    1 Simulator   2 System   3 Workload (request types, system scale n)
    4 Statistics (what BLIS records, sample paths, window, bands)   5 Objective (prices, formula)

experiment report (folder written by sweep.py)
    1 Question (README.md before the first "##")   2 Setup (what this experiment fixes and varies)
    3 Control policies   4 Results (tables, figures, fits)   5 Notes (README.md after "##")
    6 Artifacts   Appendix (first sample path per grid point)

policy comparison report (folder written by compare.py, manifest "experiment": "compare")
    1 Policy (from ours/policies.md)   2 Setup (folders, grid, seeds, method)
    3 Results (objective against the grid, headline-metric change, the papers' metrics by
      request type, difference tables)   4 Artifacts.  No interpretation.

README.md is the editable text source. House style: no em dashes or en dashes.
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
EXPERIMENTS = os.path.join(HERE, "experiments")

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

EVICTION_WORDS = {
    "fcfs": "vLLM default (newest request first): when a running request needs a KV block and none is free, the most recently admitted running request is evicted, its KV freed, and it restarts from scratch at the front of the queue",
    "srf": "Shortest-Request-First (Kim et al. 2024): the running request holding the fewest KV entries is evicted, repeatedly until the allocation succeeds; ties to the most recently admitted",
    "priority": "least-urgent SLO tier first (vLLM priority mode)",
}

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs,graphicx,float,longtable,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\begin{document}"""


# ---------------------------------------------------------------- helpers

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


def figure(exp, fig, cap):
    if os.path.exists(os.path.join(exp, fig)):
        return rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{{fig}}}\caption{{{esc(cap)}}}\end{{figure}}"
    return ""


# results table columns: (summary column, header, kind) with kind in {int, pct, s, rate1, rate3}
RESULT_COLS = [
    ("rate", "rate (req/s)", "int"), ("arrived_mean", "arrivals", "int"),
    ("unfinished_frac_mean", "unfinished", "pct"), ("rejected_frac_mean", "rejected", "pct"),
    ("delay_mean_ms_mean", "wait mean (s)", "s"), ("delay_mean_ms_ci95", "band (s)", "s"),
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


def results_table(summary, t, caption, scaling=False):
    s = summary[summary["type"] == t].sort_values(["n", "rate"] if "n" in summary else "rate")
    cols = ([("n", "scale n", "int")] if scaling else []) + [c for c in RESULT_COLS if c[0] in s.columns]
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
        elif s.startswith("### "):
            flush()
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(rf"\paragraph*{{{esc(s[4:])}}}")
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


def policy_rows():
    """The first table of ours/policies.md as a list of dicts keyed by header cell."""
    path = os.path.join(HERE, "policies.md")
    if not os.path.exists(path):
        return {}
    rows, header = {}, None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s.startswith("|"):
            if header and rows:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if header is None:
            header = cells
        elif all(set(c) <= set("-: ") for c in cells):
            continue
        else:
            rows[cells[0]] = dict(zip(header, cells))
    return rows


def compile_tex(folder, tex, jobname="report"):
    texpath = os.path.join(folder, jobname + ".tex")
    open(texpath, "w", encoding="utf-8").write("\n\n".join(t for t in tex if t))
    r = None
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", jobname + ".tex"],
                           cwd=folder, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit(f"pdflatex failed; {jobname}.tex kept for inspection")
    for ext in (".tex", ".aux", ".log", ".out"):
        try:
            os.remove(os.path.join(folder, jobname + ext))
        except OSError:
            pass
    print("wrote", os.path.join(folder, jobname + ".pdf"))


# ---------------------------------------------------------------- experiment context

class Ctx:
    """Everything the sections need, read once from an experiment folder."""

    def __init__(self, exp):
        self.exp = os.path.abspath(exp)
        self.name = os.path.basename(self.exp)
        self.m = m = json.load(open(os.path.join(self.exp, "manifest.json")))
        self.fl = fl = flags_to_dict(m.get("base_flags", []) + m.get("extra_flags", []))
        specs = sorted(p for p in glob.glob(os.path.join(self.exp, "specs", "*.yaml")) if not p.endswith("objective.yaml"))
        self.spec = yaml.safe_load(open(specs[0])) if specs else {"clients": []}
        obj_path = os.path.join(self.exp, "specs", "objective.yaml")
        if not os.path.exists(obj_path):
            obj_path = os.path.join(HERE, "specs", "objective.yaml")
        self.prices = yaml.safe_load(open(obj_path))["types"] if os.path.exists(obj_path) else {}
        md_path = os.path.join(self.exp, "README.md")
        md = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
        parts = re.split(r"^## ", md, flags=re.M)
        self.question = re.sub(r"^# .*$", "", parts[0], flags=re.M).strip()
        self.rest = "".join("## " + p for p in parts[1:])
        sp = os.path.join(self.exp, "summary.csv")
        self.summary = pd.read_csv(sp) if os.path.exists(sp) else None
        self.blocks = int(m["blocks"])
        self.bs = int(flag(fl, "--block-size-in-tokens"))
        self.horizon = m.get("horizon_s")
        self.warm, self.tail = float(m.get("warmup_s") or 0), float(m.get("tail_s") or 0)
        self.n_inst = int(flag(fl, "--num-instances"))
        self.step_tokens, self.step_seqs = flag(fl, "--max-num-batched-tokens"), flag(fl, "--max-num-seqs")
        self.refresh_us = int(flag(fl, "--snapshot-refresh-interval"))
        self.rp = flag(fl, "--routing-policy")
        self.scorers = flag(fl, "--routing-scorers") if self.rp == "weighted" else ""
        self.scaling = m.get("experiment") == "scaling"
        self.inst_list = m.get("instances") or [self.n_inst]
        self.rpi = m.get("rate_per_instance")

    @property
    def window(self):
        if self.horizon:
            return f"[{self.warm:g} s, {self.horizon - self.tail:g} s)"
        return f"[{self.warm:g} s, end - {self.tail:g} s)"


# ---------------------------------------------------------------- environment sections

def sec_simulator(c, num):
    return [rf"\subsection*{{{num}. Simulator}}", kv_table([
        ("Simulator", "BLIS (Blackbox Inference Simulator), the open-source discrete-event simulator of the llm-d project, written in Go"),
        ("BLIS version", "the upstream code as of commit 2ebef6a (2026-09-04); our fork is pinned to it"),
        ("Our fork", "branch ours of yoliliang/inference-sim. The fork adds output only (per-request eviction counts, request status, rejected requests, a steady-state window block, the periodic state file) plus the candidate policies listed in ours/policies.md. With the default policies, upstream and fork produce byte-identical results; the report of each experiment records the fork commit it ran on."),
        ("Event model", "discrete events: request arrivals, the end of each GPU step, and control-plane timers (router snapshot refresh). The clock jumps from event to event with microsecond resolution; the only randomness is the seeded arrival process and the seeded lengths, so a seed reproduces a run to the byte"),
        ("Latency model", "BLIS's calibrated step-time model (trained-physics, its default): the duration of one GPU step is a deterministic fitted function of the batch's composition (prefill tokens, decoding requests and their context lengths), calibrated on H100 vLLM traces by the BLIS team (docs/guide/latency-models.md). A request's service time is therefore not sampled; it emerges from the batches it shares"),
        ("Run mode", "fixed horizon: requests keep arriving at the given rate until the horizon, then the run stops; whatever is still in the system at that moment is recorded as unfinished"),
    ])]


def sec_system(c, num):
    return [rf"\subsection*{{{num}. System}}", kv_table([
        ("Served model", f"{flag(c.fl, '--model')}: the language model every instance runs (Qwen3, 14 billion parameters, bf16 weights, about 27 GiB)"),
        ("Instance (replica)", f"one {flag(c.fl, '--hardware')} GPU (80 GB) holding a full copy of the model, tensor parallelism 1; each instance has its own queue, running batch and KV cache, and a request is served entirely by the instance it was routed to. The system scale n is the number of instances; the experiments use n = 4 unless they vary n"),
        ("KV cache per instance", f"{c.blocks:,} blocks of {c.bs} tokens = {c.blocks * c.bs:,} token slots. Every token of a request's prompt and answer occupies one slot from admission until the request finishes; this memory is what limits how many requests an instance can hold at once." + (" 15,909 is the value BLIS derives from the GPU's memory after the model weights, activations and runtime overhead (about 39 GiB at 160 KiB per token)." if c.blocks == 15909 else " Set explicitly to create a memory-bound regime.")),
        ("Tokens processed per step", f"{c.step_tokens} (vLLM default for 80 GB GPUs). Each GPU step processes at most this many new tokens across the batch: prompt tokens being prefilled plus one token per decoding request. A compute budget that bounds the step time, not memory; a long prompt is split over several steps"),
        ("Requests per step", f"{c.step_seqs} (vLLM default): at most this many requests in the running batch. Neither per-step limit binds in these experiments; the KV cache does"),
        ("Engine rules (vLLM)", "continuous batching with chunked prefill: at each step the instance first advances every running request by one token, evicting a running request if a needed KV block is not free, then admits waiting requests in queue order while the token budget, the request limit and free KV for the prompt allow. An evicted request returns to the front of the queue and recomputes from scratch"),
    ])]


def sec_workload(c, num):
    tex = [rf"\subsection*{{{num}. Workload}}",
           para("Requests arrive from three independent Poisson streams, one per type; the total rate is split by the shares in the table. Each request has a prompt length (input tokens) and an answer length (output tokens), drawn once at arrival. The answer length is hidden from every policy (BLIS's oracle boundary): the engine learns it only by generating the last token. The arrival times and the lengths are the only randomness in a run; the seed fixes both.")]
    rows = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{@{}p{0.09\linewidth}p{0.40\linewidth}rp{0.13\linewidth}p{0.26\linewidth}@{}}", r"\toprule",
            r"type & meaning & share & prompt tokens & answer tokens \\", r"\midrule"]
    for cl in c.spec.get("clients", []):
        t = cl.get("tenant_id")
        rows.append(f"{esc(t)} & {esc(TYPE_WORDS.get(t, cl.get('id')))} & {cl.get('rate_fraction')} & "
                    f"{esc(dist_words(cl.get('input_distribution', {})))} & {esc(dist_words(cl.get('output_distribution', {})))} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Request types (ours/specs/types3.yaml). Clamped: the exponential draw is rounded and then cut to the interval, so a draw above the maximum becomes the maximum; the rounded exponential is the discrete counterpart of the geometric stopping law.}",
             r"\end{table}"]
    tex.append("\n".join(rows))
    tex.append(r"\paragraph*{Load and the system scale n.}")
    tex.append(para("Load is stated relative to the baseline's capacity R_cap = 75 req/s on 4 instances (18.75 per instance), the arrival rate at which the queue wait of the B1 baseline stops being stationary (found by the capacity sweep 2026-09-13_capacity_B1; KV memory is the binding resource). A rate experiment varies the total arrival rate at n = 4. A scaling experiment varies n with the arrival rate per instance fixed: the cluster has n instances, the total rate is n times the per-instance rate, and everything else is unchanged, so supply and demand grow together (the large-market regime of Balseiro, Ma and Zhang, 2025). System totals are expected to grow linearly in n and per-request quantities to converge."))
    return tex


def sec_statistics(c, num):
    return [rf"\subsection*{{{num}. Statistics}}", r"\paragraph*{What BLIS records.}",
            para("BLIS has no objective of its own. For every request it records the arrival time, the prompt and answer lengths in tokens, the queue wait (scheduling delay), the time to first token (TTFT), the end-to-end time (E2E), the mean inter-token gap (ITL), the instance that served it and, in our fork, how many times it was evicted, how many computed tokens the evictions discarded, and its status (completed, unfinished at the horizon, or rejected). Per run it records the counts of completed, still queued and still running requests, total input and output tokens, throughput in requests and tokens per second, the mean and the 90th, 95th and 99th percentiles of TTFT, E2E and ITL, the number of evictions (preemptions) and of KV allocation failures. Our fork adds a window block: for the requests that arrived in the steady-state window, per type, the counts by status and the sums of input tokens, output tokens, sojourn time of completed requests and censored time of unfinished requests. Every number in every report, the objective included, is computed from these records; nothing is read from inside the simulation."),
            kv_table([
                ("Sample path", "one BLIS run with one seed. Every statistic is first computed on one path; paths are the independent replications"),
                ("Steady-state window", f"requests that arrive in {c.window}. The first {c.warm:g} s are discarded because the system starts empty and the KV cache takes 200 to 300 s to fill near capacity; the last {c.tail:g} s are discarded because late arrivals have not had time to finish"),
                ("Per-path statistics", "over the completed requests that arrived in the window: mean, median and 99th percentile of TTFT, E2E and queue wait; fractions of window arrivals that were unfinished at the horizon or rejected; evictions per second, per arrival and per completion; wasted (recomputed) tokens per arrival; output tokens per second (a time average over the window); KV cache in use, averaged over the window and the instances (from the state file)"),
                ("Derived per type", "average TPOT (time per output token) = (mean E2E minus mean TTFT) / (mean answer length minus 1), the ratio of window totals"),
                ("Unfinished requests", "counted, never timed: a request still in the system at the horizon contributes to the unfinished fraction and to the objective's censored cost, and to nothing else"),
                ("Across paths", "mean over the seeds and a 95 percent confidence band (half-width from the Student t distribution with number of seeds minus 1 degrees of freedom); shaded in the figures, the band column in the tables"),
                ("Comparisons", "different policies are run on the same seeds, so they face identical arrivals and lengths (common random numbers); a comparison reports the paired difference per seed and its band, which is far narrower than either experiment's own band"),
            ])]


def sec_objective(c, num):
    tex = [rf"\subsection*{{{num}. Objective}}",
           para("The experiments are valued with a revenue management objective defined by us, not by BLIS. A request of type k that completes earns a reward proportional to its length and pays a congestion cost proportional to the time it spent in the system, from arrival to its last token (queueing and service alike, the convention of the queueing-economics literature after Naor and Mendelson). A request rejected at admission earns and pays nothing. A request that arrived in the window but is still in the system at the horizon T earns nothing and pays the cost of the time it has spent so far, a lower bound on its true cost. "),
           r"Let $W$ be the steady-state window, $C$ the set of window arrivals that completed, $U$ the set of window arrivals still unfinished at $T$, and for request $i$ let $k(i)$ be its type, $P_i$ its prompt length in tokens, $o_i$ its answer length in tokens, $a_i$ its arrival time and $e_i$ its completion time, in seconds. The objective is the net value per second of window time,",
           r"\[ V = \frac{1}{|W|}\left[\; \sum_{i \in C} \frac{\pi^{\mathrm{in}}_{k(i)}\, P_i + \pi^{\mathrm{out}}_{k(i)}\, o_i}{1000} \;-\; \sum_{i \in C} h_{k(i)} \,(e_i - a_i) \;-\; \sum_{i \in U} h_{k(i)} \,(T - a_i) \right]. \]"]
    if c.prices:
        rows = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
                r"type & $\pi^{\mathrm{in}}$ per 1,000 prompt tokens & $\pi^{\mathrm{out}}$ per 1,000 answer tokens & $h$ per second of sojourn \\", r"\midrule"]
        for t, p in c.prices.items():
            rows.append(f"{esc(t)} & {p['pi_in']:g} & {p['pi_out']:g} & {p['h']:g} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}", r"\caption{Prices and congestion cost (ours/specs/objective.yaml, copied into each experiment's specs/). Units are abstract; only the ratios matter for ranking policies. The output-to-input ratio of 4 follows commercial API pricing; h is chosen so that the uncongested sojourn costs 4 to 16 percent of the reward, so that congestion can outweigh revenue under overload. The prices enter only the valuation of recorded runs; the simulation never sees them.}", r"\end{table}"]
        tex.append("\n".join(rows))
    tex.append(para("V is computed on every sample path from the window block's per-type sums: reward from the summed input and output tokens of completed window arrivals, congestion cost from the summed sojourn of completed and the summed censored time of unfinished window arrivals. It is then averaged over the seeds with the same band as the other statistics. For a scaling experiment V is a system total and is expected to grow linearly in n; the value per unit of scale is V divided by n."))
    return tex


def build_environment(ref_exp):
    c = Ctx(ref_exp)
    tex = [PREAMBLE, r"\section*{Simulation environment (common to every experiment)}",
           esc(f"Generated by ours/report.py on {pd.Timestamp.today().date()} from {c.name}. Every experiment and comparison report refers to this document for the simulator, the system, the request types, the statistics and the objective; the reports themselves state only what they fix and vary.")]
    tex += sec_simulator(c, 1) + sec_system(c, 2) + sec_workload(c, 3) + sec_statistics(c, 4) + sec_objective(c, 5)
    tex.append(r"\end{document}")
    compile_tex(EXPERIMENTS, tex, jobname="environment")


# ---------------------------------------------------------------- experiment report

def sec_setup(c, num):
    rows = [("Profile", c.m.get("profile", c.m.get("routing", ""))),
            ("Environment", "see environment.pdf (simulator, system, request types, statistics, objective); this table lists only what this experiment fixes and varies")]
    if c.scaling:
        rows.append(("System scale n", f"{', '.join(str(x) for x in c.inst_list)} instances; total rate = n x {c.rpi:g} req/s ("
                     + ", ".join(f"n={n}: {n * c.rpi:g}" for n in c.inst_list) + f"); {c.rpi / 18.75:.2f} x R_cap per instance"))
    else:
        rows.append(("Instances", f"{c.n_inst}"))
        rows.append(("Total arrival rate", ", ".join(f"{r:g}" for r in c.m.get("rates", [])) + " req/s, over all types"))
    rows += [("KV cache per instance", f"{c.blocks:,} blocks of {c.bs} tokens"),
             ("Seeds", f"{len(c.m.get('seeds', []))} ({c.m['seeds'][0]} to {c.m['seeds'][-1]}); one seed = one sample path" + (" at every n" if c.scaling else "") if c.m.get("seeds") else ""),
             ("Horizon and window", f"{c.horizon:g} s per sample path; statistics over arrivals in {c.window}" if c.horizon else f"{c.m.get('num_requests')} arrivals then drain"),
             ("Fork commit", str(c.m.get("commit", "unknown"))[:12])]
    if c.prices:
        rows.append(("Prices", "; ".join(f"{t}: pi_in {p['pi_in']:g}, pi_out {p['pi_out']:g}, h {p['h']:g}" for t, p in c.prices.items())))
    return [rf"\subsection*{{{num}. Setup}}", kv_table(rows)]


def sec_policies(c, num):
    fl, rp, scorers = c.fl, c.rp, c.scorers
    ev = flag(fl, "--preemption-policy")
    tex = [rf"\subsection*{{{num}. Control policies}}", kv_table([
        ("Admission", flag(fl, "--admission-policy")),
        ("Routing", rp + (f" with scorers {scorers}" if scorers else "")),
        ("Router information", f"refreshed every {c.refresh_us / 1000:g} ms" if c.refresh_us > 0 else "live (the router sees the true instance state at every decision)"),
        ("Queue order in an instance", flag(fl, "--scheduler")),
        ("Eviction rule", f"{ev}: {EVICTION_WORDS.get(ev, '')}"),
        ("Batch formation", flag(fl, "--batch-formation")),
    ]), r"\paragraph*{What these policies do.}"]
    adm = flag(fl, "--admission-policy")
    tex.append(para("Admission: " + ("always-admit accepts every arriving request; nothing is ever turned away." if adm == "always-admit" else f"{adm} (see ours/policies.md).")))
    if rp == "weighted":
        terms = [f"{w} x [{SCORER_WORDS.get(n, n)}]" for n, w in (item.split(":") for item in scorers.split(","))]
        tex.append(para("Routing: for each arriving request the router computes a score for every instance, score = " + " + ".join(terms) + ", and sends the request to the instance with the highest score (ties broken at random). "
                        + (f"The queue and KV figures it uses are refreshed every {c.refresh_us / 1000:g} ms, as a real gateway scraping metrics would, so they can be slightly stale." if c.refresh_us > 0 else "It uses the true current queue and KV figures.")))
    elif rp == "round-robin":
        tex.append(para("Routing: round-robin sends arrivals to the instances in turn, ignoring their state."))
    else:
        tex.append(para(f"Routing: {rp}."))
    tex.append(para("Inside an instance the vLLM engine rules apply (environment.pdf, section 2); only the eviction rule named above may differ from vLLM's default."))
    return tex


def sec_results(c, num):
    tex = [rf"\subsection*{{{num}. Results}}", r"\paragraph*{Reading the columns.}",
           para("Queue wait: from arrival until the instance first puts the request into a batch. TTFT (time to first token): from arrival until the first answer token, that is wait plus prompt processing. E2E (end to end): from arrival until the last answer token. p99: the 99th percentile. Band: half-width of the 95 percent confidence band of the wait mean across seeds. Unfinished, rejected: percent of the requests arriving in the window. Evictions/s: evictions per second of simulated time, whole cluster. Evictions per completion: evictions divided by completed window requests. Output tok/s: answer tokens produced per second, whole cluster. Times are in seconds.")]
    if c.summary is not None:
        tex.append(results_table(c.summary, "all", "All types, mean over seeds.", c.scaling))
        for t in ("type1", "type2", "type3"):
            if (c.summary["type"] == t).any():
                tex.append(results_table(c.summary, t, f"{t} only, mean over seeds.", c.scaling))
    for fig, cap in [("scaling_panel.png", "Key system statistics against the system scale n (log2 axis): mean over seeds with the shaded 95 percent confidence band. System totals (value, throughput, evictions) should grow linearly, per-request quantities should converge. Dashed lines are least-squares power-law fits to the all-types means; the exponent is in the legend and in the table below."),
                     ("sweep_panel.png", "Window statistics against the total arrival rate: mean over seeds with the shaded 95 percent confidence band."),
                     ("objective.png", "Revenue management objective V per second of window time: mean over seeds with the shaded 95 percent confidence band.")]:
        tex.append(figure(c.exp, fig, cap))
    fits_path = os.path.join(c.exp, "scaling_fits.csv")
    if c.scaling and os.path.exists(fits_path):
        fits = pd.read_csv(fits_path)
        rows = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{@{}lcrr@{}}", r"\toprule",
                r"statistic & kind & exponent b & points \\", r"\midrule"]
        for _, r in fits.iterrows():
            b = "" if pd.isna(r["exponent"]) else f"{r['exponent']:.3f}"
            kind = r["kind"] if "kind" in r else ("per instance" if r.get("per_instance") else "level")
            rows.append(f"{esc(r['metric'])} & {esc(kind)} & {b} & {int(r['points'])} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}",
                 r"\caption{Power-law fits, y = a times n to the power b, on the all-types means. For a system total an exponent of 1 means linear scaling; for a per-request quantity an exponent of 0 means invariance to scale. A blank exponent means the quantity was zero or negative at some n and no log fit exists.}",
                 r"\end{table}"]
        tex.append("\n".join(rows))
    return tex


def sec_appendix(c):
    first_seed = min(c.m.get("seeds", [1]) or [1])
    points = [(f"n{n}", f"system scale n = {n} ({n} instances, {n * c.rpi:g} req/s)") for n in c.inst_list] if c.scaling else \
             [(f"rate{r:g}", f"{r:g} req/s") for r in c.m.get("rates", [])]
    appendix = []
    for prefix, label in points:
        png = f"runs/{prefix}_s{first_seed}_panel.png"
        if os.path.exists(os.path.join(c.exp, png)):
            appendix.append(
                rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{{png}}}"
                rf"\caption{{One sample path at {esc(label)} (seed {first_seed}). Top: distributions of TTFT and E2E by type. Middle left: each request's queue wait against its arrival time. Middle right: KV cache in use per instance. Bottom left: requests waiting (solid) and in the batch (dotted) per instance. Bottom right: evictions per second by type of the evicted request. Dashed lines mark the steady-state window.}}\end{{figure}}\clearpage")
    if not appendix:
        return []
    return [r"\clearpage\subsection*{Appendix: one sample path per grid point}",
            para("Read each figure left to right, top to bottom. The system starts empty and the KV cache fills as long requests accumulate. If the cache reaches 100 percent, every new admission from then on needs an eviction, the queue starts to grow and the wait rises. That moment is the visible transition from stable to overloaded on a single path.")] + appendix


def build(exp):
    c = Ctx(exp)
    tex = [PREAMBLE, rf"\section*{{Numerical details: {esc(c.name)}}}" + (r" (scaling experiment)" if c.scaling else ""),
           esc(f"Generated by ours/report.py on {pd.Timestamp.today().date()} from manifest.json, specs/, summary.csv and README.md. The common environment is described in environment.pdf.")]
    tex += [r"\subsection*{1. Question}", esc(c.question) if c.question else "(no question recorded)"]
    tex += sec_setup(c, 2) + sec_policies(c, 3) + sec_results(c, 4)
    tex += [r"\subsection*{5. Notes}", md_to_tex(c.rest) if c.rest.strip() else "(none)"]
    files = sorted(f for f in os.listdir(c.exp) if not f.endswith((".tex", ".aux", ".log", ".out")))
    nruns = len(glob.glob(os.path.join(c.exp, "runs", "*.json*")))
    tex += [r"\subsection*{6. Artifacts}", esc(", ".join(f for f in files if f != "runs")) + esc(f"; runs/: {nruns} sample paths, each with the BLIS metrics (gzipped json with the window block), the log and the state csv.")]
    tex += sec_appendix(c)
    tex.append(r"\end{document}")
    compile_tex(c.exp, tex)


# ---------------------------------------------------------------- policy comparison report

COMPARE_METRICS = [  # column stem, label, scale, unit (mirrors ours/compare.py METRICS)
    ("value_per_s", "objective value per second, whole system", 1, ""),
    ("output_tok_per_s", "output tokens per second, whole system", 1, ""),
    ("e2e_mean_ms", "mean E2E latency", 1e-3, "s"),
    ("tpot_ms", "average TPOT", 1, "ms"),
    ("ttft_mean_ms", "mean TTFT", 1e-3, "s"),
    ("ttft_p99_ms", "TTFT p99", 1e-3, "s"),
    ("delay_mean_ms", "mean queue wait", 1e-3, "s"),
    ("preempt_per_arrival", "preemptions per arrival", 1, ""),
    ("preempt_per_s", "preemptions per second", 1, ""),
    ("wasted_tok_per_arrival", "wasted tokens per arrival", 1, ""),
    ("kv_util_pct", "KV memory usage, percent", 1, ""),
    ("batch_mean", "mean batch size per instance", 1, ""),
    ("unfinished_frac", "unfinished, percent of arrivals", 100, ""),
]
CSPEC = {m[0]: m for m in COMPARE_METRICS}
TYPE_LABEL = {"all": "all types", "type1": "type1 (short chat)", "type2": "type2 (RAG)", "type3": "type3 (long generation)"}


def compare_table(tab, t, label, cols):
    """Rows are grid points; columns are candidate minus baseline with band and relative change.
    Split into two tables so that the columns stay legible."""
    xcol = "n" if tab.n.nunique() > 1 else "rate"
    a = tab[tab.type == t].sort_values(xcol)
    if a.empty:
        return ""
    cols = [c for c in cols if not a[c + "_diff"].isna().all()]
    out = []
    for part, chunk in enumerate([cols[:len(cols) // 2 + len(cols) % 2], cols[len(cols) // 2 + len(cols) % 2:]]):
        if not chunk:
            continue
        head = [esc(xcol), "seeds"] + [esc(CSPEC[c][1]) for c in chunk]
        rows = [r"\begin{table}[H]\centering\scriptsize", r"\resizebox{\linewidth}{!}{",
                r"\begin{tabular}{@{}rr" + "r" * len(chunk) + "@{}}", r"\toprule",
                " & ".join(head) + r" \\", r"\midrule"]
        for _, r in a.iterrows():
            cells = []
            for c in chunk:
                spec = CSPEC[c]
                d, b, rel = r[c + "_diff"] * spec[2], r[c + "_band"] * spec[2], r[c + "_rel"]
                if pd.isna(d):
                    cells.append("")
                    continue
                s = f"{d:+,.3g}" + (f" {spec[3]}" if spec[3] else "") + (f" $\\pm$ {b:,.2g}" if not pd.isna(b) else "")
                if not pd.isna(rel):
                    s += f" ({rel * 100:+.1f}\\%)"
                cells.append(s)
            rows.append(f"{r[xcol]:g} & {int(r['n_seeds'])} & " + " & ".join(cells) + r" \\")
        rows += [r"\bottomrule", r"\end{tabular}}",
                 rf"\caption{{{esc(label)} minus B1, {esc(TYPE_LABEL.get(t, t))}, part {part + 1}: mean paired difference over seeds, $\pm$ the 95 percent band, relative change against the baseline mean in parentheses.}}",
                 r"\end{table}"]
        out.append("\n".join(rows))
    return "\n\n".join(out)


def build_compare(exp, m):
    exp = os.path.abspath(exp)
    name = os.path.basename(exp)
    md_path = os.path.join(exp, "README.md")
    md = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
    question = re.sub(r"^# .*$", "", re.split(r"^## ", md, flags=re.M)[0], flags=re.M).strip()
    tab = pd.read_csv(os.path.join(exp, "compare.csv"))
    cols = [c for c in m.get("metrics", []) if c + "_diff" in tab.columns]
    labels = [c["label"] for c in m["candidates"]]
    policies = policy_rows()
    headline = m.get("headline", "e2e_mean_ms")
    grid = m.get("grid", {})
    win = m.get("window", {})

    tex = [PREAMBLE, rf"\section*{{Policy comparison: {esc(name)}}}",
           esc(f"Generated by ours/report.py on {pd.Timestamp.today().date()} from compare.csv and manifest.json. The common environment (simulator, system, request types, statistics, objective) is described in environment.pdf.")]

    tex.append(r"\subsection*{1. Policy}")
    for cand in m["candidates"]:
        row = policies.get(cand.get("profile"), {})
        if row:
            tex.append(kv_table([
                ("Profile", cand["profile"]),
                ("Paper", row.get("Paper", "")),
                ("Decision point it changes", row.get("Decision point", "")),
                ("BLIS flag", row.get("BLIS flag", "")),
                ("What we implement", row.get("What we implement", "")),
                ("Departures from the paper", row.get("Departures from the paper", "")),
                ("Paper's headline metric and claim", row.get("Headline metric (paper)", "")),
            ]))
        else:
            tex.append(para(f"{cand['label']}: profile {cand.get('profile')} (no row in ours/policies.md)."))
    base_row = policies.get(m["baseline"].get("profile"), {})
    tex.append(para(f"Baseline: {m['baseline'].get('profile')}, " + (base_row.get("What we implement", "the llm-d production default") + ".")))

    tex.append(r"\subsection*{2. Setup}")
    xword = "system scale n" if grid.get("x") == "n" else "total arrival rate (req/s)"
    xvals = grid.get("instances" if grid.get("x") == "n" else "rates", [])
    rows = [("Baseline folder", f"{os.path.basename(m['baseline']['folder'])} (fork commit {str(m['baseline'].get('fork_commit'))[:12]})")]
    for cand in m["candidates"]:
        rows.append((f"Candidate {cand['label']}", f"{os.path.basename(cand['folder'])} (fork commit {str(cand.get('fork_commit'))[:12]})"))
    rows += [("Grid", f"{xword}: {', '.join(f'{v:g}' for v in xvals)}" + (f"; total rate {', '.join(f'{v:g}' for v in grid.get('rates', []))} req/s" if grid.get("x") == "n" else "")),
             ("Seeds", f"{len(grid.get('seeds', []))} per grid point, the same in both folders"),
             ("Horizon and window", f"{win.get('horizon_s')} s; arrivals in [{win.get('warmup_s')}, {float(win.get('horizon_s') or 0) - float(win.get('tail_s') or 0):g}) s" if win.get("horizon_s") else ""),
             ("Method", "paired differences: for every grid point, seed and request type, candidate minus baseline on the same sample path (same seed, so identical arrivals and lengths); mean over seeds with a 95 percent band (Student t over the paired differences); relative change against the size of the baseline mean. Positive is better for the objective and throughput, negative is better for latencies, preemptions, waste and the unfinished fraction."),
             ("Headline metric", CSPEC.get(headline, (headline, headline))[1])]
    tex.append(kv_table(rows))

    tex.append(r"\subsection*{3. Results}")
    tex.append(figure(exp, "objective_vs_scale.png", f"Revenue objective against the {xword}. Left: value per second for the whole system under each policy, mean over seeds with the shaded 95 percent band. Right: the paired gain of each candidate over B1" + (", divided by n (gain per unit of scale)" if grid.get("x") == "n" else "") + ", by request type, with bands; the grey line is zero."))
    tex.append(figure(exp, "headline_improvement.png", f"The paper's headline metric ({CSPEC.get(headline, (headline, headline))[1]}): relative change of the candidate against B1 in percent, one group of bars per grid point, one bar per request type, error bars = 95 percent band of the paired difference."))
    tex.append(figure(exp, "paper_metrics.png", f"The metrics the policy papers report (after Kim et al. 2024, Figure 9): rows are mean E2E latency, average TPOT, preemptions per arrival, KV memory usage and mean batch size; columns are all types and each type; lines are the policies, mean over seeds with the shaded band, against the {xword}. Memory usage is a property of the instance, so it is drawn once."))
    for lab in labels:
        sub = tab[tab.candidate == lab] if "candidate" in tab.columns else tab
        for t in ("all", "type1", "type2", "type3"):
            tex.append(compare_table(sub, t, lab, cols))

    files = sorted(f for f in os.listdir(exp) if not f.endswith((".tex", ".aux", ".log", ".out")))
    tex += [r"\subsection*{4. Artifacts}", esc(", ".join(files))]
    if question:
        tex += [r"\paragraph*{Question as recorded in README.md.}", esc(question)]
    tex.append(r"\end{document}")
    compile_tex(exp, tex)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--environment":
        build_environment(sys.argv[2])
    elif len(sys.argv) == 2:
        _exp = os.path.abspath(sys.argv[1])
        _m = json.load(open(os.path.join(_exp, "manifest.json")))
        if _m.get("experiment") == "compare":
            build_compare(_exp, _m)
        else:
            build(_exp)
    else:
        sys.exit(__doc__)
