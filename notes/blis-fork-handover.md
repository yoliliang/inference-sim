# BLIS fork handover (2026-09-07)

Status: local fork of BLIS is deployed, verified and pushed. This note is the starting
point for the experiments chat. Read it together with `notes/blis-simulator-guide.md`
(mechanics, our-spec to BLIS dictionary) and `notes/environment-formal-spec.md`.

House style: no em dashes or en dashes.

## Where things are

- Machine: Windows, PowerShell. Go 1.27, Git 2.55, Python 3.12 (pyyaml, pandas,
  matplotlib installed).
- Repo: `C:\Users\yolic\OneDrive - University of Miami\Yoli Project\Optimization of LLM
  Cache\Simulation\inference-sim`, GitHub `yoliliang/inference-sim`, branch `ours`,
  remote `upstream` = `inference-sim/inference-sim`.
- Pinned base: upstream commit `2ebef6a` (2026-09-04). Do not rebase onto newer upstream
  during the experiment campaign. To see how far behind we are:
  `git fetch upstream; git log --oneline ours..upstream/main | Measure-Object -Line`.
- Binary: `blis.exe` in the repo root, rebuilt with `go build -o blis.exe main.go`
  after any Go change.
- Every divergence from upstream is listed in `UPSTREAM.md`. Keep it current.

## What the fork adds (all verified, upstream tests pass)

| Item | File | Flag |
|---|---|---|
| Type-rank scheduler: orders W_g by TenantID rank (type1 < type2 < type3), then arrival | `sim/scheduler_typerank.go` | `--scheduler type-rank` |
| Batch-formation seam: pass-through wrapper where our (E, S, A) rules will go | `sim/batch_formation_ours.go` | `--batch-formation ours` (default `vllm`) |
| Three-type workload (P 256/2048/512, mean out 64/128/1024, exponential, fractions .5/.35/.15, rate 40) | `ours/specs/types3.yaml` | `--workload-spec` |
| One-config harness: 4 x Qwen3-14B/H100, 2,500 blocks, omniscient router, seed 42 | `ours/run.ps1`, `ours/run.sh` | |
| Grid driver: one self-contained folder per experiment | `ours/sweep.py` | |

Regression anchors (seed 42, `run.ps1`), reproduced to the digit on the Windows build:

| Config | Preempt | TTFT p99 | E2E mean | Tok/s |
|---|---|---|---|---|
| round-robin, fcfs | 797 | 36,661 | 11,921 | 4,424 |
| weighted, fcfs | 787 | 26,189 | 11,487 | 4,594 |
| weighted, type-rank | 700 | 29,010 | 6,155 | 4,801 |
| weighted, type-rank, `--batch-formation ours` | 700 | 29,010 | 6,155 | 4,801 |

`--batch-formation vllm` and `ours` must stay byte-identical on stdout until a knob inside
`OursBatchFormation` is switched on (INV-6 makes this a free regression test).

## How to run an experiment

```powershell
cd "<repo>"
python ours\sweep.py <name> --rates 60,80,100 --seeds 1,2,3,4,5 --blocks 15909 --num-requests 10000 --question "..." -- <extra blis flags>
```

Creates `ours/experiments/<date>_<name>/` with `manifest.json` (commit, grid, flags),
`specs/` (one yaml per rate), `runs/` (raw BLIS json + log per run), `summary.csv`,
`README.md`. `ours/experiments/` is gitignored; archive folders by hand.

PowerShell gotchas: quote any flag value containing a comma
(`--routing-scorers "queue-depth:1,kv-utilization:1"`); `--rate` is silently ignored under
a workload spec, `sweep.py` rewrites `aggregate_rate` in the yaml instead; `still_queued`
is always 0 at the end because BLIS drains, use `scheduling_delay_p99_ms`, TTFT p99 and
preemptions to locate a knee; OneDrive may lock `.git` files, pause sync if git misbehaves.

## What the preview run already showed (5 seeds, 3,000 requests, 10,000 blocks)

Rates 20 to 80 are all inside capacity: zero queueing, zero preemption, TTFT p99 69 to
99 ms. The knee is between 80 and 150 req/s (at 150: TTFT p99 3.8 s, 158 preemptions;
above 200 the run length freezes near 85 s, throughput saturating near 8,000 tok/s).
10,000 blocks does not remove memory binding at high load (160k tokens per instance fills
before `max-num-seqs 256`); use the auto value 15,909 or larger for a memory-free baseline.

## Experiment queue (from guide section 9, revised)

1. Capacity point: rates 60,80,100,110,120,130,140,160, 5 seeds, 15,909 blocks, 10k
   requests. Deliverable: the rate at which delay leaves steady state.
2. Memory cliff: rate just below capacity, sweep `--blocks` from 15,909 down.
3. Routing shootout across Delta (`--snapshot-refresh-interval` 0, 5000, 50000, 100000,
   1000000 us).
4. Victim rules (needs code in `batch_formation_ours.go`).
5. Admission under overload: rate 20 percent above capacity, five admitters, objective
   pi_in P + pi_out o - h (e - a) from the `requests` array. First Direction 1 result.
   Needs `sim/admission_drift.go` and `ours/objective.py`, not yet written.
6. Disaggregation as the Chen-Bu prefill site.

## Open items carried over

- Fix the tau_kv term in `notes/environment-formal-spec.md` to sum over S union A
  (residents left out of the pass do not cost attention time).
- Read the ai-native-systems-research admission-controller case study before writing
  Direction 1's related-work paragraph (EDPP is their P/D decider, not an admitter).
- BLIS `priority` preemption changes only the victim, not the running-batch order;
  vLLM's priority mode re-sorts running each step. Recorded as a fork divergence to add
  later inside `OursBatchFormation`.
- Restart cost after preemption is from-scratch in BLIS; fix behind a flag later.
