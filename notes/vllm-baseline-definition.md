# Benchmark definition: vLLM as the baseline in BLIS

Purpose: fix, before any run, what "the vLLM baseline" means at each decision point, how
BLIS reproduces it, and where to look to check the claims. Every row below has a source
you can open. Line numbers in vLLM refer to `main` as fetched on 2026-09-12; vLLM moves
fast, so re-check against the version you cite in the paper.

House style: no em dashes or en dashes.

## 1. What vLLM decides, and with what rule

vLLM is a single-engine server. It has no admission control and no cross-replica router
of its own beyond the data-parallel mode. Everything it decides happens inside one
instance, in `vllm/v1/core/sched/scheduler.py`.

| Decision | vLLM default | Source (vLLM) | BLIS equivalent | In BLIS? |
|---|---|---|---|---|
| Admission | none, every request is accepted and queued | `SchedulerConfig` has no admission field (`vllm/config/scheduler.py`); the waiting queue is unbounded | `--admission-policy always-admit` | yes (default) |
| Routing across replicas | none inside the engine. In data-parallel mode the client picks the engine with the lowest load score; on current `main` the score is `max(clients x inflight, waiting + running) + waiting x 6 x max(0, kv_usage - 0.5)` | `vllm/v1/engine/core_client.py`, `DPLBAsyncMPClient.get_core_engine_for_request`, about lines 1555 to 1590 | `vllm-dp` scorer implements the earlier formula `waiting x 4 + running` (inverted min-max); `least-loaded` implements `waiting + running + inflight` | partially: BLIS mirrors the 2025 formula, not the KV-pressure penalty added since |
| Queue order | FCFS by arrival | `SchedulerConfig.policy = "fcfs"` (`vllm/config/scheduler.py`); `SchedulerPolicy = Literal["fcfs", "priority"]` | `--scheduler fcfs` | yes (default) |
| Eviction victim | last request in the running list (`self.running[-1]`, popped from the tail); with `--scheduling-policy priority`, the request with the largest `(priority, arrival_time)` | `scheduler.py` lines 747 to 786 | `--preemption-policy fcfs` (tail) and `priority` | yes, both modes |
| Eviction mode | recompute: the victim goes back to the waiting queue, its blocks are freed, prefill is redone from scratch | `scheduler.py` `_preempt_request` (line 1478); V1 has no swap mode | BLIS resets `ProgressIndex` to 0 and prepends to WaitQ | yes, with one known deviation: BLIS re-decodes previously generated tokens one step at a time instead of re-prefilling them in one chunk (fork item, flag later) |
| Continuous batching | one step per iteration, running requests first, then admit from waiting until budgets bind | `scheduler.py` `schedule()` | `VLLMBatchFormation` Phase 1 then Phase 2 | yes |
| Chunked prefill | on by default where the model supports it; a prefill takes whatever budget remains in the step | `arg_utils.py` line 2782 `default_chunked_prefill = model_config.is_chunked_prefill_supported`; `long_prefill_token_threshold = 0` in `SchedulerConfig` | BLIS chunks by remaining budget; `--long-prefill-token-threshold 0` (disabled) matches vLLM's 0 | yes |
| Prefix caching | on by default | `arg_utils.py` line 2783 `default_prefix_caching = model_config.is_prefix_caching_supported` | BLIS hashes blocks and reuses them; irrelevant for `types3.yaml` (no shared prefixes) | yes |
| Token budget per step | H100 or H200 with `vllm serve`: **8192**; other GPUs with `vllm serve`: 2048; offline `LLM` class: 16384 / 8192 | `arg_utils.py` lines 2717 to 2745, `_get_default_max_num_seqs_and_batched_tokens` (device memory >= 70 GiB and not A100) | `--max-num-batched-tokens` (BLIS default 2048) | yes, but the BLIS default is the non-H100 value |
| Max sequences per step | H100 or H200 with `vllm serve`: **1024**; other GPUs: 256 | same lines | `--max-num-seqs` (BLIS default 256) | yes, same caveat |
| Block size | 16 tokens on CUDA | `CacheConfig.block_size` defaults to None and resolves to 16 for the standard attention backends | `--block-size-in-tokens 16` | yes (default) |
| KV pool size | `gpu_memory_utilization = 0.9`; blocks = (0.9 x GPU memory - weights - profiled activations) / bytes per block | `CacheConfig.gpu_memory_utilization` | auto-derived `--total-kv-blocks` (15,909 for Qwen3-14B on H100) | yes |

Reading the table: at every decision point except cross-replica routing, BLIS's
shipped default is vLLM's default, so "vLLM in default mode" is a flag combination, not
new code. The two places to be careful are the H100 budgets (BLIS defaults to the
smaller non-H100 values) and the DP load balancer (BLIS mirrors an older formula).

## 2. The two baseline configurations

Both use `ours/specs/types3.yaml`, four instances, `--tp 1`, Qwen3-14B on H100, and
`--seed` over 1 to 5.

### B0: pure vLLM, data-parallel style

What a deployment of four vLLM replicas behind vLLM's own DP load balancer would do.
No admission, load-count routing, FCFS, tail eviction, H100 budgets.

```
--admission-policy always-admit
--routing-policy weighted --routing-scorers "vllm-dp:1"
--snapshot-refresh-interval 0
--scheduler fcfs --preemption-policy fcfs
--max-num-batched-tokens 8192 --max-num-seqs 1024
--long-prefill-token-threshold 0 --block-size-in-tokens 16
```

Snapshot interval 0 because the DP balancer reads live in-process counters, not scraped
metrics. If a reviewer asks about the newer KV-pressure penalty, `least-loaded` with the
same flags is the base term of the current formula and can be reported alongside.

### B1: llm-d production default

What the same four replicas do behind llm-d's Endpoint Picker: still no admission by
default, weighted routing with the llm-d scorer profile, 50 ms scraped snapshots.

```
--admission-policy always-admit
--routing-policy weighted --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
--snapshot-refresh-interval 50000
--scheduler fcfs --preemption-policy fcfs
--max-num-batched-tokens 8192 --max-num-seqs 1024
--long-prefill-token-threshold 0 --block-size-in-tokens 16
```

Source for the profile and the interval: BLIS `sim/routing_scorers.go`
`DefaultScorerConfigs` ("llm-d parity"), and the GAIE scorer plugin defaults in
`gateway-api-inference-extension/pkg/epp/framework/plugins`.

### Memory regime

Run each baseline twice: once at the auto KV pool (15,909 blocks) so results can be
checked against published vLLM numbers, and once at `--total-kv-blocks 2500` which is our
memory-bound regime. The first is the validation run; the second is the comparator for
our policies.

## 3. How we will check the numbers are reasonable

Simulator outputs cannot be compared to a real run we do not have, but three external
anchors exist.

1. **BLIS's own validation.** The llm-d blog (https://llm-d.ai/blog/blis-evolving-llm-d-at-simulation-speed)
   reports median 7 to 9 percent error on E2E and ITL over 36 held-out experiments on
   8B to 141B models across H100, A100 and L40S. Our numbers inherit that accuracy claim
   as long as we stay inside the calibrated regime (prompts up to a few thousand tokens,
   batches within the budgets).
2. **Per-instance capacity.** The BLIS cluster guide states about 17 req/s per instance at
   saturation for Qwen3-14B / H100 / TP 1 under their default workload
   (https://inference-sim.github.io/inference-sim/latest/guide/cluster/, "Scaling and
   Saturation"). Our workload has longer outputs, so expect less per instance; the
   capacity sweep should land in the same order of magnitude, tens of req/s for four
   instances.
3. **Decode latency at low load.** The smoke run gave ITL mean 12.9 ms and TTFT 36 ms for
   a 512-token prompt on an otherwise idle H100. Published `vllm bench serve` results for
   14B-class dense models on H100 at low concurrency put per-token decode latency in the
   10 to 20 ms range and TTFT for short prompts under 100 ms. Before the paper, pull one
   concrete published table for Qwen3-14B (or Llama-3.1-8B as the nearest calibrated
   model) and record the comparison in the experiment README.

If a run falls outside these anchors by more than a factor of two, stop and look for a
configuration error before interpreting it.

## 4. What is not vLLM and must not be called the baseline

- `--scheduler sjf`, `priority-fcfs`, `reverse-priority`: shipped by BLIS, but vLLM's
  default is fcfs. Priority mode exists in vLLM and can be a secondary comparator.
- `token-bucket`, `tier-shed`, `gaie-legacy` admission: llm-d / GAIE behaviours, not
  vLLM. They are comparators for Direction 1, labelled as gateway policies.
- `--batch-formation ours`: our seam. Off in every baseline run.
- BLIS defaults `--max-num-batched-tokens 2048 --max-num-seqs 256`: valid vLLM defaults
  for A100 and smaller GPUs, not for H100. Either set the H100 values or say "vLLM
  defaults for sub-80 GB hardware" explicitly.

## 5. Version pinning for the paper

State three versions in the simulation section: BLIS commit 2ebef6a; the vLLM version the
trained-physics coefficients were calibrated against (ask in the BLIS repo or read
`defaults.yaml` provenance comments; record it here once found); and the vLLM version
whose scheduler we describe as the baseline (the `scheduler.py` lines above are from
`main` on 2026-09-12; pin a release tag instead).

## Sources, one line each

- vLLM scheduler config and defaults: https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/
- vLLM engine argument defaults by GPU: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py (search `_get_default_max_num_seqs_and_batched_tokens`)
- vLLM V1 scheduler, preemption: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py (search `self.running[-1]`)
- vLLM DP load balancer: https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core_client.py (search `get_core_engine_for_request`)
- vLLM priority scheduling PRs: #5958 (V0), #19057 (V1)
- BLIS routing scorers and llm-d profile: `sim/routing_scorers.go` in the fork
- BLIS batch formation: `sim/batch_formation.go` in the fork
- BLIS scheduling guide: https://inference-sim.github.io/inference-sim/latest/guide/scheduling/
- BLIS admission guide: https://inference-sim.github.io/inference-sim/latest/guide/admission/
- BLIS cluster guide (17 req/s figure): https://inference-sim.github.io/inference-sim/latest/guide/cluster/
- BLIS validation numbers: https://llm-d.ai/blog/blis-evolving-llm-d-at-simulation-speed
