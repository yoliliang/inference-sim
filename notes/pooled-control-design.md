# Pooled control model for the fork: design and change plan (2026-09-26)

Purpose. Make the simulator execute a general control model, "shared queue plus one set
decision per GPU step", instead of the push structure BLIS inherits from llm-d, so that
policies (ours and the papers') are written against one interface and the simulator itself
assumes no policy structure. Nothing here changes the default path: every existing
experiment stays byte-identical.

House style: no em dashes or en dashes.

## 1. The general model (agreed with the user, 2026-09-26)

State at any time:

- a shared queue Q of admitted, unstarted requests, in arrival order (with the same
  fixed API overhead every request pays today before it becomes schedulable);
- for each GPU g, a work-in-progress set W_g: requests that have started on g, that is
  decoding, mid-way through a chunked prefill, or preempted and waiting to restart on g.

Decisions:

- Admission (per request, at arrival): accept or reject. Policy.
- Batch formation (per GPU, at every step boundary): choose the subset of W_g union Q that
  runs this step, and for each chosen prefill request how many prompt tokens to process.
  Choosing a request from Q is what used to be routing; it moves the request into W_g.
  Policy.
- Eviction (per GPU, when a chosen batch needs KV that is not free): which member of W_g
  to evict. Policy.

Physics the simulator enforces, never decides: KV capacity per GPU in blocks of 16 tokens,
one KV slot per token of prompt and answer until completion, the per-step token budget and
request limit of the engine, the step-time model, one decode token per request per step,
recompute on eviction (progress reset, KV released), completion when the hidden answer
length is reached. A policy's choice that violates a physical constraint is an error, not
silently repaired.

Conventions (stated as conventions, not constraints): a preempted request stays in W_g
(vLLM's rule; physically it holds no KV after eviction and could be released back to Q);
the API overhead is paid once at arrival; communication cost of a take from Q is zero (BLIS
charges none for routing either).

What the real systems do: inside one engine vLLM already forms each batch from "running
plus its own waiting queue", which is this model with Q replaced by a local queue. Across
GPUs, llm-d, vLLM data parallel, SGLang and Dynamo push (route at arrival on stale metrics,
no take-back); NVIDIA Triton's per-model queue and central schedulers such as Mooncake's
Conductor or Llumnix's global scheduler are the pull or global-view forms. The pull cost is
one intra-rack round trip per take, well under a 15 to 40 ms step.

## 2. What BLIS does today and where it hard-codes policy

Verified in the code (sim/cluster/cluster.go, cluster_event.go, sim/simulator.go,
sim/batch_formation.go, sim/event.go):

| Step | Today | Kind |
|---|---|---|
| Arrival | ClusterArrivalEvent, then AdmissionDecisionEvent after admission latency (0) | mechanism |
| Admission | policy.Admit(request, snapshots): accept or reject | policy (already pluggable) |
| Routing | RoutingDecisionEvent immediately after admission: policy.Route picks an instance; request injected into that instance; never moved again | policy structure hard-coded (push at arrival) |
| Instance arrival | ArrivalEvent, then QueuedEvent after the alpha queueing delay; request enters the instance's WaitQ; a StepEvent is scheduled now if the instance is idle | mechanism |
| Step | scheduleBatch: reorder WaitQ (scheduler), FormBatch(ctx) where ctx has RunningBatch and the instance's own WaitQ only; then executeBatchStep (step time from the batch, progress advance, TTFT stamp), processCompletions (release KV), scheduleNextStep | mechanism, but FormBatch sees only the local queue |
| FormBatch (vllm) | Phase 1: every running request advances one token, KV block on demand, evict on failure (victim rule pluggable); Phase 2: admit from the head of WaitQ while request limit, token budget and free KV for the chunk hold, stop at the first that does not fit | policy hard-coded inside the "vllm" strategy (order, head-of-line stop, chunk = min(remaining, budget)) |
| Gateway queue (flow control, off by default) | requests held at the gateway while pool-average saturation >= 1 (queue depth / threshold, KV / threshold); released on a 1 ms tick to the router | policy hard-coded as mechanism (saturation formula, tick) |
| Cluster loop | pops the earliest of cluster events and instance events; instance ties broken by instance index; arrivals pulled lazily | mechanism, deterministic |

Metrics: an instance creates the per-request row when the request is injected into it
(InjectArrivalAt); queue wait = time of first batch inclusion minus ArrivalTime; TTFT
stamped when ProgressIndex reaches the prompt length; rejected requests are cluster-level
rows (our fork). Requests in the gateway queue at the horizon are counted by the gateway
counters, not as per-request rows.

## 3. The change

Additive: a new cluster mode `--shared-queue-push` (off by default, the current code path
untouched, so INV-6 byte identity of every existing experiment is trivial). The user chose
the flag name; `--set-policy` names the set policy (fcfs-pool).

### 3.1 Shared queue (new, sim/cluster/pool.go)

- `Pool`: ordered slice of *Request in (arrival time, sequence) order; Len, Items (read
  only view), Take(req), Remove(req).
- Arrival path in pooled mode: ClusterArrivalEvent, AdmissionDecisionEvent (unchanged),
  then instead of RoutingDecisionEvent a PoolArrivalEvent at time + alpha queueing delay
  (the same delay the instance path applies, so a request becomes schedulable at the same
  instant as under push) that appends the request to the Pool and calls the wake-up hook.
- Wake-up hook: when a request enters the pool, idle instances (no pending StepEvent) are
  offered a step now. Which idle instance wakes is a policy decision
  (`PoolPolicy.Wake(pool, idleInstances) []InstanceID`); the fcfs baseline wakes the
  lowest-index idle instance. Busy instances see the pool at their next step boundary.
- Horizon: requests still in the pool are emitted as per-request rows with status
  unfinished and their arrival time, so the window block and the objective count them.
  INV-1 gains a pool term: injected = completed + still queued + still running + in pool +
  rejected + ...

### 3.2 Batch context sees the pool (sim/simulator.go, sim/batch_formation.go)

- `BatchContext.Pool` (ours, nil in push mode): interface `PoolAccess { Candidates()
  []*Request; Take(*Request) }`. Taking a request performs exactly what routing plus
  instance arrival do today: the request becomes this instance's (metrics row created with
  the original ArrivalTime, in-flight count incremented) and is admitted to the batch in the
  same call.
- `BatchContext.InstanceID` and a cluster snapshot (the same RouterState the admission
  policy sees) so a set policy can be state-dependent.

### 3.3 Pooled batch formation (new, sim/batch_formation_pooled.go)

`PooledBatchFormation` implements FormBatch with the physics fixed and the choices
delegated to a `SetPolicy`:

1. Phase 1 (W_g continues): identical to vllm Phase 1; eviction victim from the existing
   pluggable rule.
2. Phase 2 (fill): `SetPolicy.Order(local, pool, state)` returns the ordered candidate list
   drawn from the instance's own WaitQ (preempted requests) and the pool, plus
   `StopAtFirstInfeasible bool` (vLLM's head-of-line rule) and an optional
   `Tokens(req) int64` for the prefill chunk (default min(remaining prompt, budget)).
   The simulator walks the list, checks the three physical limits, admits or skips (or
   stops), and for pool candidates calls Take. A policy that returns an infeasible
   chunk (larger than the remaining prompt or the budget) is an error.
3. Result and bookkeeping unchanged (NewlyScheduled, Preempted, scheduling delay,
   TTFT).

Baseline set policies shipped with the mode:

- `fcfs-pool` (profile C): local preempted requests first (vLLM's convention), then the
  pool in arrival order, stop at first infeasible, vLLM chunking. This is the centralized
  counterpart of B1.
- `push-equivalent`: not needed as a policy; push mode itself is the reference.

### 3.4 Observability and tooling

- State csv (cmd/state_sample.go): `pool_depth` column; panel figure: pool depth line.
- Window block: unchanged fields; unfinished in the pool counted as above.
- sweep.py: profile `C` = B1's engine and prices with `--cluster-queue pooled
  --pool-policy fcfs-pool`; later policies add `--pool-policy <name>`.
- policies.md: a row per set policy; report.py policy table gains "Batch formation set
  policy".

### 3.5 Files touched

| File | Change |
|---|---|
| sim/cluster/pool.go (new) | Pool type, PoolArrivalEvent, wake-up |
| sim/cluster/cluster.go | pooled-mode wiring: pool, hook into instances, in-flight accounting on Take, horizon emission of pool rows, INV-1 term |
| sim/cluster/cluster_event.go | AdmissionDecisionEvent: branch to PoolArrivalEvent in pooled mode |
| sim/simulator.go | BatchContext.Pool and InstanceID; a Take callback that creates the metrics row and admits; idle wake entry point (a public ScheduleStepNow) |
| sim/batch_formation.go | BatchContext fields only |
| sim/batch_formation_pooled.go (new) | PooledBatchFormation, SetPolicy interface, fcfs-pool |
| sim/bundle.go | valid names: cluster queue modes, pool policies |
| cmd/root.go, cmd/state_sample.go | `--cluster-queue`, `--pool-policy`; pool_depth column |
| ours/sweep.py, ours/policies.md, ours/analyze.py | profile C, ledger rows, pool depth in the panel |
| UPSTREAM.md | one line per file |

Estimated effort: one working day of code and tests, plus the validation runs below (about
one hour of simulation).

## 4. Validation

1. Default path: seed-42 anchors 700 / 29,010 / 6,155 / 4,801 unchanged; `go test ./sim
   ./cmd` green except the known CRLF golden.
2. Equivalence at n = 1: pooled fcfs on one instance must reproduce B1 on one instance
   exactly (same arrivals, same alpha delay, same queue order, same batches), so every
   per-request metric matches to the microsecond. This is the correctness proof of the new
   machinery: with one GPU the pool is that GPU's queue.
3. Conservation (INV-1) with the pool term, checked on every run.
4. Behavioural checks at n = 4: inside capacity (0.9 R_cap) the pooled fcfs wait should
   sit at the 31 ms floor and, in scale, not grow with n (the stale-snapshot herding of B1
   cannot occur); above capacity (1.2 R_cap) totals should match B1 within the band (no
   admission, so the backlog is the same) while the per-instance queue imbalance
   disappears.
5. Unit tests: pool order, take semantics, wake-up of the lowest idle instance, head-of-line
   stop versus skip, infeasible chunk rejected, horizon rows for pool residents.

## 5. What this buys

- One interface for every subsequent policy: admission (per arrival), set choice (per
  step), eviction (per shortage). SRF and Mooncake already fit (eviction and admission);
  Sarathi (chunk sizes and stall-free batching), Scorpio (deadline order, credit-based
  batch selection), Llumnix (which GPU takes what) are set policies under this interface;
  CacheRoute needs prefix affinity, which enters through the state the set policy sees.
- The theory's model and the simulator's model coincide: one pool, n servers, state-
  dependent step times, KV as the scarce resource.
- The B1 versus C gap measures the value of information and of revocability separately
  from any policy, a result in itself.

## 6. Open choices for the user

1. Preempted requests: stay in W_g (proposed, vLLM convention) or return to the pool.
2. Wake-up when several instances are idle: lowest index (proposed) or policy-chosen.
3. Whether the set policy may also pick the prefill chunk size now, or only later (Sarathi).

## 7. Implementation record (2026-09-26)

Implemented as planned, flag `--shared-queue-push`, set policy `--set-policy fcfs-pool`,
profile `C` in sweep.py. User decisions: evicted requests stay on their instance; the
lowest-index idle instance is woken; prefill chunking stays vLLM's rule.

Validation results:

1. Default path: seed-42 anchors 700 / 29,010 / 6,155 / 4,801 unchanged; `go test ./sim
   ./sim/cluster ./cmd` green except the known CRLF golden.
2. Equivalence at n = 1 (seed 42, 22.5 req/s, 300 s, 15,909 blocks): B1 and the shared
   queue give identical aggregates (6,853 injected, 5,690 completed, 990 queued, 173
   running, 498 evictions, identical TTFT, E2E and wait distributions) and identical
   per-request metrics for every request; the only differences are two requests that
   arrived in the last 10 ms and were still in the delay pipeline (a row exists under push,
   none under the shared queue) and a 0.03 s difference in the recorded run duration.
   The upstream "KV allocation failed for completing request" warning fires 7 times in
   both runs (an upstream accounting message under a full cache, not ours).
3. Sanity at n = 4 (seed 1, 300 s): inside capacity (67.5 req/s) the shared queue cuts the
   mean wait from 32.9 to 22.7 ms and TTFT p99 from 112 to 89 ms at equal throughput;
   above capacity (90 req/s) waits and unfinished fractions are the same (13.0 versus 13.1
   s, 1.7 percent), the backlog sits in the shared queue (mean 199, max 496 in the window)
   and the local queues hold only preempted requests (0.26 on average).
4. Scaling and compare folders: see 2026-09-26_compare_C_scale_load0.9 and _load1.2.
