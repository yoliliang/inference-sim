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
| B1_mooncake | Qin et al. (Moonshot AI), "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", ACM Transactions on Storage 2026, Section 4.3.4 | admission | `--admission-policy mooncake --mooncake-mode predict` | Load = predicted latency / target. Prefill load: predicted TTFT of the request on the least loaded instance (queued prompt tokens + remaining prefill of running requests + own prompt, divided by the prefill rate of a step next to the current batch, from BLIS's step-time model) over the TTFT target. Decode load: mean over instances of the predicted TBT (step time of a synthetic batch) over the TBT target, with the batch predicted at t = now + TTFT_hat: current batch x max(0, 1 - TTFT_hat / t_d) + queued requests + this one. Admit iff max(loads) <= theta. Parameters 2026-09-24: TTFT target 2 s, TBT target 60 ms, theta 1, t_d 9 s | No prefill/decode disaggregation here: one pool does both, so the prefill instance is the least loaded one and the decode load averages over all instances. Paper leaves theta, t_d, the TBT predictor and the pool aggregation unspecified | rejected requests (17.7 percent under rule 0, 16.0 under rule 1, 15.2 under rule 2, on a 2x-speed tool-and-agent trace over 8 prefill + 8 decode instances); system goodput = fraction of requests with TTFT and TBT under target |
| B1_mooncake_now | same paper, Section 4.3.2 (Early Rejection, the rule the prediction variant improves on) | admission | `--admission-policy mooncake --mooncake-mode now` | As above but the decode load is evaluated on the batches as they are at arrival | same | same |

## Implementation record

| Profile | Fork commit | Tests | Baseline byte-identity check |
|---|---|---|---|
| B1_srf | f272177a | sim/batch_formation_srf_test.go: fewest-KV victim, tie to tail, repeat until enough, name registered | seed-42 anchors 700 / 29,010 / 6,155 / 4,801 unchanged with --preemption-policy fcfs |
