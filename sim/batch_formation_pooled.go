package sim

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// ours: batch formation for the pooled control model (--shared-queue-push).
//
// One set decision per instance step. The candidates are the instance's own waiting
// requests (work in progress that is not in the batch: preempted requests waiting to
// restart) and the cluster's shared queue (unstarted requests). A SetPolicy orders the
// candidates; the simulator admits them in that order subject to the physical limits
// (batch request limit, per-step token budget, free KV for the prefill chunk) and never
// reorders on its own.
//
// Phase 1 (the batch continues, one token per decode request, chunk continuation for
// prefills, eviction on KV shortage by the pluggable victim rule) is the upstream vLLM
// code, called with an empty local queue so that it admits nothing. Two of vLLM's rules
// are kept as conventions and documented as such: no admission in a step that evicted,
// and a prefill chunk of min(remaining prompt, remaining budget).

// SetPolicy orders the candidates of one step. local are this instance's waiting
// requests in queue order, shared the shared queue in arrival order. The returned order
// may drop candidates; stop says whether admission stops at the first infeasible
// candidate (vLLM's head-of-line rule) or skips it.
type SetPolicy interface {
	Order(local, shared []*Request, ctx *BatchContext) (order []*Request, stop bool)
}

// FCFSPool is the centralised counterpart of the B1 baseline: this instance's preempted
// requests first (vLLM's convention), then the shared queue in arrival order, stopping at
// the first candidate that does not fit.
type FCFSPool struct{}

func (FCFSPool) Order(local, shared []*Request, _ *BatchContext) ([]*Request, bool) {
	out := make([]*Request, 0, len(local)+len(shared))
	out = append(out, local...)
	out = append(out, shared...)
	return out, true
}

var validSetPolicies = map[string]bool{"": true, "fcfs-pool": true}

// IsValidSetPolicy reports whether name is a registered set policy.
func IsValidSetPolicy(name string) bool { return validSetPolicies[name] }

// NewSetPolicy builds a set policy by name ("" = fcfs-pool).
func NewSetPolicy(name string) SetPolicy {
	switch name {
	case "", "fcfs-pool":
		return FCFSPool{}
	default:
		panic(fmt.Sprintf("unknown set policy %q", name))
	}
}

// PooledBatchFormation implements BatchFormation for the pooled control model.
type PooledBatchFormation struct {
	inner  *VLLMBatchFormation // Phase 1 and the preemption rule
	policy SetPolicy
}

// NewPooledBatchFormation wires the vLLM Phase 1 (with the given preemption rule) to a
// set policy.
func NewPooledBatchFormation(preemptionPolicy, setPolicy string) *PooledBatchFormation {
	return &PooledBatchFormation{
		inner:  NewBatchFormation(preemptionPolicy).(*VLLMBatchFormation),
		policy: NewSetPolicy(setPolicy),
	}
}

func (p *PooledBatchFormation) FormBatch(ctx BatchContext) BatchResult {
	if ctx.RunningBatch == nil {
		ctx.RunningBatch = &Batch{}
	}
	// Phase 1 through the upstream code with an empty local queue: it continues the batch
	// and evicts if it must, admits nothing, and leaves evicted requests in tmp (each one
	// prepended, so tmp holds the last victim first).
	tmp := &WaitQueue{}
	c1 := ctx
	c1.WaitQ = tmp
	result := p.inner.FormBatch(c1)
	victims := tmp.Items()
	for i := len(victims) - 1; i >= 0; i-- {
		ctx.WaitQ.PrependFront(victims[i]) // same front-of-queue placement as vLLM
	}
	if result.PreemptionHappened {
		return result // vLLM convention: a step that evicted admits nobody
	}

	budget := ctx.MaxNumBatchedTokens
	for _, r := range result.RunningBatch.Requests {
		budget -= int64(r.NumNewTokens)
	}
	var shared []*Request
	if ctx.SharedQueue != nil {
		shared = ctx.SharedQueue.Candidates()
	}
	local := ctx.WaitQ.Items()
	localSet := make(map[*Request]bool, len(local))
	for _, r := range local {
		localSet[r] = true
	}
	order, stop := p.policy.Order(local, shared, &ctx)

	for _, next := range order {
		if len(result.RunningBatch.Requests) >= int(ctx.MaxNumSeqs) || budget <= 0 {
			break
		}
		if next.IsDecodeSubRequest {
			panic("pooled batch formation does not support prefill/decode disaggregation")
		}
		fromShared := !localSet[next]
		if fromShared && ctx.SharedQueue == nil {
			panic(fmt.Sprintf("set policy returned %s, which is neither local nor in the shared queue", next.ID))
		}
		if fromShared && !ctx.SharedQueue.Fits(next) {
			continue // never fits this instance (context limit or capacity): leave it for another
		}

		// Physical limits, as in vLLM Phase 2.
		cachedBlocks := ctx.KVCache.GetCachedBlocks(next.FullInputTokens())
		numNewTokens := next.InputLen() - util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()
		if 0 < ctx.PrefillTokenThreshold && ctx.PrefillTokenThreshold < numNewTokens {
			numNewTokens = ctx.PrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, budget)
		startIndex := util.Len64(cachedBlocks) * ctx.KVCache.BlockSize()
		if ctx.MaxModelLen > 0 {
			numNewTokens = min(numNewTokens, max(ctx.MaxModelLen-1-startIndex, 0))
		}
		endIndex := startIndex + numNewTokens
		if ok := ctx.KVCache.AllocateKVBlocks(next, startIndex, endIndex, cachedBlocks); !ok {
			if stop {
				break
			}
			continue
		}
		if fromShared {
			if !ctx.SharedQueue.Take(next) { // dropped as unservable by the instance
				ctx.KVCache.ReleaseKVBlocks(next)
				continue
			}
		} else {
			ctx.WaitQ.Remove(next)
		}

		result.RunningBatch.Requests = append(result.RunningBatch.Requests, next)
		next.ScheduledStepIdx = ctx.StepCount
		result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{Request: next})
		budget -= numNewTokens
		next.State = StateRunning
		next.NumNewTokens = int(numNewTokens)
		ctx.ComputedTokens[next.ID] = numNewTokens + util.Len64(cachedBlocks)*ctx.KVCache.BlockSize()
	}
	return result
}
