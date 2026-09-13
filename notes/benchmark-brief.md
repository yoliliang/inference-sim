# Baseline benchmark brief (2026-09-13)

For the Claude Code session in the `inference-sim` fork. Read together with
`notes/blis-fork-handover.md` (repo state, harness, gotchas) and
`notes/vllm-baseline-definition.md` (source-by-source justification of every flag below).

House style: no em dashes or en dashes anywhere, including code comments and READMEs.

## 1. What the baseline is

Two configurations, both meaning "what production runs today". B1 is the primary anchor
for every later comparison; B0 is one row showing the engine without a gateway.

**B1, llm-d default** (vLLM instances behind llm-d's gateway with shipped settings)
```
--admission-policy always-admit
--routing-policy weighted --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
--snapshot-refresh-interval 50000
--scheduler fcfs --preemption-policy fcfs
--max-num-batched-tokens 8192 --max-num-seqs 1024
--long-prefill-token-threshold 0 --block-size-in-tokens 16
--batch-formation vllm
```

**B0, vLLM alone** (vLLM instances with vLLM's own data-parallel balancer)
```
same as B1 except:
--routing-policy weighted --routing-scorers "vllm-dp:1"
--snapshot-refresh-interval 0
```

Why these values and not BLIS's CLI defaults: BLIS defaults to `round-robin` routing and to
`2048 / 256` budgets, which are vLLM's values for sub-80 GB GPUs. On H100 with
`vllm serve`, vLLM uses 8192 / 1024. Sources are in the baseline-definition note.
`always-admit` equals llm-d's `gaie-legacy` as long as the workload has no `slo_class`,
which is the case for the current `types3.yaml`.

## 2. Fixed substrate for every run

- Model and hardware: `--model qwen/qwen3-14b --hardware H100 --tp 1 --num-instances 4`
- Workload: `ours/specs/types3.yaml` unchanged (three types, Poisson, exponential outputs
  with caps, mix 50 / 35 / 15). `sweep.py` rewrites `aggregate_rate` per run.
- Seeds 1 to 5. `--num-requests 10000`. Discard the first 10 percent of completions
  (by arrival time) as warm-up when computing statistics from the `requests` array.
- Two memory regimes: auto KV pool (15,909 blocks) for validation against external
  numbers; `--total-kv-blocks 2500` for the memory-bound regime our policies target.
- Two operating points once capacity is known: 0.9x and 1.2x the capacity rate.

## 3. Run order

1. **Transparency run** (if not already done): one B1 run at rate 40, auto blocks, with
   `--log info --trace-level decisions --summarize-trace --trace-output`, README listing
   every resolved control and every artifact. Confirm anchors.
2. **Capacity sweep** under B1 at auto blocks: rates 60, 80, 100, 110, 120, 130, 140, 160,
   five seeds. Locate the rate where `scheduling_delay_p99_ms` and TTFT p99 leave
   steady state. Report it as `R_cap`. Repeat under B0 only if the two differ by more
   than 10 percent at the knee.
3. **Baseline table** at both operating points (0.9 and 1.2 x `R_cap`) and both memory
   regimes: rows B0, B1, plus two brackets that are not candidates:
   - floor: `--routing-policy round-robin`, otherwise B1
   - oracle: B1 with `--snapshot-refresh-interval 0`
   Sixteen configurations x five seeds = 80 runs.
4. **Sanity check against external anchors** before interpreting anything:
   - ITL mean at low load should be 10 to 20 ms; TTFT for short prompts under 100 ms
   - per-instance saturation of the same order as the docs' 17 req/s (our workload has
     longer outputs, so expect lower)
   - `dropped_unservable` and `length_capped_requests` must be 0
   If any run is off by more than 2x, stop and look for a configuration error.

## 4. Reporting rules (merge with the session's standing rules)

- One folder per experiment under `ours/experiments/<date>_<name>/`, created by
  `sweep.py`, containing manifest, specs, raw runs, `summary.csv`, README, and any png.
- Every table row must differ from its neighbour by exactly one decision. State the
  anchor row first.
- Per row report: mean over seeds and min to max range for TTFT p99, E2E p99,
  `scheduling_delay_p99_ms`, `tokens_per_sec`, `preemption_count`, completed and
  rejected counts by `tenant_id` (rejections come from stdout `Shed` lines, not the
  JSON). The objective `pi_in P + pi_out o - h (e - a)` is added once prices are set;
  do not invent prices.
- A ranking is only claimed if it holds on every seed.
- Results as short tables plus one paragraph of interpretation. Plots saved in the
  experiment folder, not inline pasted text.
- Commit after each completed step with a message starting `ours:`; push to `origin ours`.
- Never change a Go file for a baseline run. If a flag is missing, say so and stop.

## 5. Decisions still open (do not guess; leave as TBD in READMEs)

- Objective prices `pi_in`, `pi_out` and holding costs `h_i` per type.
- SLO class mapping per type (needed only for `gaie-legacy` and `tier-shed` rows later;
  proposed chat critical, RAG standard, long-generation batch).
- Whether 2,500 blocks stays fixed or is set as a fraction of the auto pool.
- The vLLM release tag to cite for the baseline description.

## 6. First message to paste into the Claude Code session

> Read `notes/benchmark-brief.md`, then `notes/vllm-baseline-definition.md`, then
> `notes/blis-fork-handover.md`. Execute section 3 of the brief in order, starting with the
> transparency run if `ours/experiments/` has none, then the capacity sweep under B1.
> Stop after the capacity sweep and report `R_cap` with the summary table and the plot
> before starting the baseline table. Follow section 4 for every report. Do not modify
> any Go file.
