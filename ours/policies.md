# Candidate policies

One row per policy we test against the B1 baseline. A profile name in sweep.py is
`<baseline>_<policy>` and differs from the baseline in exactly one of the five decision
points (admission, routing, queue order, batch formation, eviction). Experiments live in
`ours/experiments/<date>_<type>_<profile>_<...>/` with the same arguments as the baseline
folder they are compared with; comparisons are post-processing in `<date>_compare_<name>/`
(ours/compare.py), never new sample paths.

House style: no em dashes or en dashes.

| Profile | Paper | Decision point | BLIS flag | What we implement | Departures from the paper | Headline metric (paper) |
|---|---|---|---|---|---|---|
| B1 | llm-d production default | (all defaults) | `--preemption-policy fcfs` and the rest of ENGINE_H100 | vLLM engine rules, weighted routing on 50 ms snapshots, always-admit | none | (reference) |
| B1_srf | Kim, Li, Hong, Liu, Huang, Ailamaki (EPFL), "Saving GPU Hours in LLM Inference System Development and Online Workloads with Simulation and DBMS-Inspired Cache Replacement Policies", arXiv 2411.07447 | eviction victim | `--preemption-policy srf` | When an allocation fails, evict the running request holding the fewest KV entries (ProgressIndex: prefilled plus generated tokens), repeat until the allocation succeeds; victim goes to the queue front and recomputes from scratch. Ties: most recently admitted first, so equal holdings reduce to fcfs. Applies wherever vLLM preempts (decode needing a block, prefill needing prompt blocks) | SRF+Hist (deferral of predicted preempters using an output-length histogram) not implemented; paper gives no tie rule | mean E2E latency, reported relative to NRF (vLLM default): up to 8 percent lower on Llama-3-8B / A100, 6 percent on H100, 13 percent on Llama-3-70B / 4 A100; also TPOT, TTFT, preemption count, KV memory usage, batch size (their Fig. 9 and 11) |

## Implementation record

| Profile | Fork commit | Tests | Baseline byte-identity check |
|---|---|---|---|
| B1_srf | f272177a | sim/batch_formation_srf_test.go: fewest-KV victim, tie to tail, repeat until enough, name registered | seed-42 anchors 700 / 29,010 / 6,155 / 4,801 unchanged with --preemption-policy fcfs |
