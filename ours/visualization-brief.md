# Brief: simulation visualization tool (for a separate coding session)

Goal. Given one BLIS sample path produced by `ours/sweep.py`, show how the serving system
evolves over a chosen time interval: which requests are waiting, which are in the running
batch, how full each instance's KV cache is, when evictions happen and who is evicted. The
tool is read-only; it consumes the files listed below and never runs BLIS.

House style: no em dashes or en dashes anywhere, including code comments.

## 1. Where the data is

One experiment folder `ours/experiments/<date>_<name>/`, one run per (rate, seed):

| File | Content | Granularity |
|---|---|---|
| `runs/rate<R>_s<S>.json.gz` | metrics json. `requests` array: one record per request with `arrived_at` (s), `requestID`, `num_prefill_tokens`, `num_decode_tokens`, `ttft_ms`, `e2e_ms`, `scheduling_delay_ms`, `tenant_id`, `handled_by` (instance), `preemption_count`, `wasted_tokens`, `status` (completed, unfinished, rejected) | per request |
| `runs/rate<R>_s<S>_state.csv` | per-instance state every `state_sample_ms` of simulated time: `clock_us`, `instance`, `queue_depth`, `batch_size`, `kv_used_blocks`, `kv_total_blocks`, `kv_utilization`, `preemptions` (cumulative), `completed` (cumulative), `in_flight`, cluster totals, `is_final` | per instance per sample |
| `runs/rate<R>_s<S>.log` | BLIS stderr. One line per eviction: `[tick N] preemption: evicting request_k to make room` (N in microseconds) | per event |
| `manifest.json` | rates, seeds, horizon_s, warmup_s, tail_s, state_sample_ms, blocks, profile, flags | per experiment |
| `specs/rate<R>.yaml` | the workload: three request types, shares, input and output distributions | per rate |

From these three files a request's timeline can be reconstructed exactly:
arrival = `arrived_at`; first scheduled = arrival + `scheduling_delay_ms`; first token =
arrival + `ttft_ms`; completion = arrival + `e2e_ms`; evictions of that request = log lines
with its id (each eviction sends it back to the queue and its progress restarts from zero).

## 2. What is not in the data yet (snapshot export, to be added in the fork)

The state csv holds counts only. For a composition view the fork will add an optional
snapshot export, `--state-snapshot-interval <us>`, writing `runs/<tag>_snapshot.csv` with
one row per (sample time, instance, request in queue or batch): `clock_us`, `instance`,
`requestID`, `tenant_id`, `state` (queued or running), `progress_tokens`, `kv_blocks`.
Sampling every 1 to 5 s of simulated time keeps a 1,200 s run to a few hundred thousand
rows. Until this exists, build the tool on the three files above and treat the snapshot
file as an optional input that enables the composition panels.

## 3. Views to produce

1. Cluster timeline over [t0, t1]: per instance, stacked area of requests waiting and
   running by type, KV utilisation as a line, eviction ticks as markers coloured by type
   of the evicted request. Time axis in seconds of simulated time.
2. Request Gantt for a chosen instance and interval: one row per request, coloured by
   type, segments for waiting, prefill (arrival + scheduling delay to first token) and
   decode (first token to completion), eviction marks where the request restarts.
3. Single-request card: type, input and output tokens, arrival, wait, TTFT, E2E, number
   of evictions, wasted tokens.
4. Optional, with the snapshot file: KV occupancy by type per instance over time.

Interactive is preferred (Plotly or a small HTML page) with a static png export for the
paper. Inputs: experiment folder, rate, seed, t0, t1, instance. Output next to the run
files under `runs/<tag>_viz/`.

## 4. Conventions

- Type colours: type1 blue #2a78d6, type2 orange #eb6834, type3 green #1baf7a, as in
  `ours/analyze.py`.
- Instances: instance_0 to instance_3.
- All times in seconds of simulated time; microsecond ticks divided by 1e6.
- Do not modify anything under `ours/experiments/` except writing into `runs/<tag>_viz/`.
- Do not modify Go code; if the snapshot export is missing, say so and use the counts.
