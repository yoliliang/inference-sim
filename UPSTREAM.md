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
- sim/metrics.go: +Metrics.ExtraRequests; EmitOutput sets Status (completed / unfinished) and appends ExtraRequests before the arrival sort
- sim/batch_formation.go: +PreemptedRequest.WastedTokens, set from ProgressIndex at eviction in preemptForTokens
- sim/simulator.go: per-request PreemptionCount / WastedTokens accumulated where Metrics.PreemptionCount is incremented
- sim/cluster/cluster.go: +rejectedRequestMetrics on ClusterSimulator; aggregateMetrics copies it into merged.ExtraRequests
- sim/cluster/cluster_event.go: admission rejection appends a RequestMetrics row with status rejected

## Files added (ours)

- sim/scheduler_typerank.go
- sim/batch_formation_ours.go
- cmd/state_sample.go (periodic per-instance state CSV via the existing read-only sim.ProgressHook)
- ours/ (specs, run.ps1, run.sh, results/ gitignored)

## Known pre-existing test failure on Windows

cmd TestNoOpByteIdentity_AdapterBlindRunMatchesBaseline fails on this machine before and after any fork change: the golden file specs/007-lora-control-plane/testdata/baseline_noop.json has CRLF line endings and the preamble comparison is byte-exact. Not a fork regression.

## Regression check

Same inputs with and without `--batch-formation ours` must give byte-identical stdout (INV-6) until a knob inside OursBatchFormation is switched on.
