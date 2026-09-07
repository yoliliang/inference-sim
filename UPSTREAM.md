# Upstream pin

Base: inference-sim/inference-sim @ 2ebef6a (2026-09-04)
Branch: ours

## Files diverging from upstream

- sim/bundle.go: +1 entry in validSchedulers (type-rank); +validBatchFormations map, IsValidBatchFormation, ValidBatchFormationNames
- sim/scheduler.go: +1 case in NewScheduler (type-rank)
- sim/config.go: +BatchFormation field on PolicyConfig; +PolicyOption, WithBatchFormation; NewPolicyConfig takes variadic opts (existing calls unchanged)
- sim/simulator.go: NewSimulator calls NewBatchFormationStrategy(cfg.BatchFormation, cfg.PreemptionPolicy) instead of NewBatchFormation
- cmd/root.go: +--batch-formation flag (default vllm), validation, passed via WithBatchFormation

## Files added (ours)

- sim/scheduler_typerank.go
- sim/batch_formation_ours.go
- ours/ (specs, run.ps1, run.sh, results/ gitignored)

## Regression check

Same inputs with and without `--batch-formation ours` must give byte-identical stdout (INV-6) until a knob inside OursBatchFormation is switched on.
