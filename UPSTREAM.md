# Upstream pin

Base: inference-sim/inference-sim @ 2ebef6a (2026-09-04)
Branch: ours

## Files diverging from upstream

- sim/bundle.go: +1 entry in validSchedulers (type-rank); +validBatchFormations map, IsValidBatchFormation, ValidBatchFormationNames
- sim/scheduler.go: +1 case in NewScheduler (type-rank)
- sim/config.go: +BatchFormation field on PolicyConfig; +PolicyOption, WithBatchFormation; NewPolicyConfig takes variadic opts (existing calls unchanged)
- sim/simulator.go: NewSimulator calls NewBatchFormationStrategy(cfg.BatchFormation, cfg.PreemptionPolicy) instead of NewBatchFormation
- cmd/root.go: +--batch-formation flag (default vllm), validation, passed via WithBatchFormation; +--state-sample-interval / --state-sample-path flags and the ProgressHook wiring around cs.Run (see cmd/state_sample.go)
- sim/metrics_utils.go: +3 omitempty fields on RequestMetrics (preemption_count, wasted_tokens, status); file-only, never on stdout
- sim/metrics.go: +Metrics.ExtraRequests; EmitOutput sets Status (completed / unfinished) and appends ExtraRequests before the arrival sort; a --metrics-path ending in .gz is written gzip-compressed; +WindowStartS / WindowEndS / HorizonS / DropPerRequest fields, EmitOutput adds the window block and can omit the per-request array
- sim/metrics_utils.go (second item): +MetricsOutput.Window (file-only)
- sim/window_stats.go (added): per-type steady-state window statistics computed in memory at the end of a run (sufficient statistics of the objective plus exact TTFT / E2E / wait quantiles)
- cmd/state_sample.go, cmd/root.go (second item): +--window-start, --window-end, --drop-per-request-output
- sim/batch_formation.go: +PreemptedRequest.WastedTokens, set from ProgressIndex at eviction in preemptForTokens
- sim/simulator.go: per-request PreemptionCount / WastedTokens accumulated where Metrics.PreemptionCount is incremented
- sim/cluster/cluster.go: +rejectedRequestMetrics on ClusterSimulator; aggregateMetrics copies it into merged.ExtraRequests
- sim/cluster/cluster_event.go: admission rejection appends a RequestMetrics row with status rejected
- sim/progress_hook.go: +RequestSnapshot type and InstanceSnapshot.Requests (nil unless requested)
- sim/simulator.go (second item): +Simulator.RequestSnapshots() read-only accessor over WaitQ and RunningBatch
- sim/cluster/instance.go: +InstanceSimulator.RequestSnapshots() wrapper
- sim/cluster/cluster.go (second item): +progressRequestDetail flag, SetProgressRequestDetail; maybeDeliverProgressSnapshot fills Requests when on
- cmd/state_sample.go, cmd/root.go: +--state-snapshot-interval writing <metrics-path>_snapshot.csv
- sim/kv/cache.go: incremental snapshot of HashToBlock (setHash, delHash, refreshSnapshot); SnapshotCachedBlocksFn replays a change log instead of copying the whole map at every refresh. Same frozen-copy semantics, stdout byte-identical, 4.5x faster on the B1 profile at scale
- sim/kv/offload_chain.go, sim/kv/tiered.go: HashToBlock writes routed through setHash / delHash
- main.go: BLIS_CPUPROFILE environment variable starts a CPU profile (diagnostic only)
- sim/simulator.go (third item): +Simulator.ReleaseCompleted; completed requests drop InputTokens, OutputTokens and ITL after all bookkeeping; recordRequestCompletion calls Metrics.AddITLs
- sim/metrics.go (third item): +Metrics.ITLCounts (value counts instead of the AllITLs slice when set), AddITLs, itlStatsFromCounts reproducing CalculateMean and CalculatePercentile exactly
- sim/cluster/cluster.go (third item): arrivals are pulled from the RequestSource as the clock advances instead of being drained into the event heap up front (+moreArrivals); lean-mode setup of instances; ITLCounts merge in aggregateMetrics
- sim/cluster/autoscaler.go: tick guard also checks moreArrivals
- sim/kv/cache.go (second item): promptChainHash, a one-entry memo of the prompt's block hash chain shared by GetCachedBlocks and the snapshot closures (the router queried every instance with a fresh SHA256 chain per arrival)
- sim/cluster/cluster_event.go (second item): AdmissionDecisionEvent skips buildRouterState for AlwaysAdmit
- sim/cluster/deployment.go: +ReleaseCompletedRequests
- cmd/root.go, cmd/state_sample.go (third item): +--release-completed-requests (refused with --trace-output); BLIS_MEMPROFILE heap profile hook

## Files added (ours)

- sim/scheduler_typerank.go
- sim/batch_formation_ours.go
- cmd/state_sample.go (periodic per-instance state CSV via the existing read-only sim.ProgressHook)
- ours/ (specs, run.ps1, run.sh, sweep.py grid driver with B0/B1/floor/oracle/harness profiles, analyze.py steady-state analysis and panel plots, objective.py revenue-management objective, report.py README.pdf generator via pdflatex, visualization-brief.md; results/ and experiments/ gitignored)
- notes/ (handover, benchmark brief, vLLM baseline definition)

## Known pre-existing test failure on Windows

cmd TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline fails on this machine before and after any fork change: the golden file specs/007-lora-control-plane/testdata/baseline_noop.json has CRLF line endings and the preamble comparison is byte-exact. Not a fork regression.

## Regression check

Same inputs with and without `--batch-formation ours` must give byte-identical stdout (INV-6) until a knob inside OursBatchFormation is switched on.
