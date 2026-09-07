package sim

import "fmt"

// OursBatchFormation (ours) is the seam for our own (E, S, A) step rule.
//
// Step 3 of the build plan: a pure pass-through to VLLMBatchFormation, so that
// `--batch-formation ours` is byte-identical to `--batch-formation vllm`.
// Later steps add, each behind its own knob and default off:
//   - a running-batch order rule (which residents claim budget first: our S)
//   - a pause rule (residents left out of the pass while keeping their KV)
//   - custom victim rules (least-progressed, shortest-prompt: our E)
//   - a vLLM-faithful restart cost on preemption
type OursBatchFormation struct {
	inner *VLLMBatchFormation
}

func (o *OursBatchFormation) FormBatch(ctx BatchContext) BatchResult {
	return o.inner.FormBatch(ctx)
}

// NewBatchFormationStrategy selects a BatchFormation by name (ours).
// "" and "vllm" return the upstream VLLMBatchFormation unchanged.
func NewBatchFormationStrategy(strategy, preemptionPolicy string) BatchFormation {
	base := NewBatchFormation(preemptionPolicy).(*VLLMBatchFormation)
	switch strategy {
	case "", "vllm":
		return base
	case "ours":
		return &OursBatchFormation{inner: base}
	default:
		panic(fmt.Sprintf("unknown batch formation %q", strategy))
	}
}
