package sim

import (
	"fmt"
	"math"
)

// ours: Mooncake admission control (Qin et al., "Mooncake: A KVCache-centric Disaggregated
// Architecture for LLM Serving", ACM Transactions on Storage 2026, Section 4.3), adapted to
// a pooled cluster in which every instance both prefills and decodes.
//
// Mooncake measures load as a predicted latency divided by its target: the prefill load of
// a request is its predicted TTFT over the TTFT target, the decode load of an instance is
// the predicted time between tokens (TBT) of its batch over the TBT target. A request is
// admitted when both loads are at most theta.
//
//	mode "now"      Early Rejection (4.3.2): the decode load is evaluated on the batches as
//	                they are at arrival.
//	mode "predict"  Early Rejection based on Prediction (4.3.4): the decode load is evaluated
//	                on the batches as they will be when this request finishes prefill,
//	                t = now + TTFT_hat. The prediction is deliberately coarse ("system
//	                level"): no per-request output length; every request is assumed to
//	                decode for the fixed duration t_d, so of the B requests decoding now a
//	                fraction TTFT_hat / t_d will have finished by t, and every request queued
//	                now will have entered the batch by t.
//
// Pooled-cluster mapping. The prefill instance is the one with the smallest predicted TTFT
// (the router sends the request to the least loaded instance); the decode load is the mean
// over instances, as the paper averages over its decode pool. Predicted TTFT of instance i:
// (queued prompt tokens + remaining prefill of running requests + own prompt) / prefill rate,
// where the prefill rate is the token budget a step can prefill alongside the current decode
// batch divided by that step's duration from the simulator's own step-time model. Predicted
// TBT of a batch of K decode requests is the step-time model evaluated on a synthetic batch
// of K requests with the instance's mean context length.
//
// The policy reads only routing snapshots and the arriving request's prompt length (never
// its output length: INV-9). It is deterministic and allocation-free after warm-up.
type MooncakeAdmission struct {
	Mode             string  // "now" or "predict"
	TTFTTargetUs     float64 // l_ttft
	TBTTargetUs      float64 // l_tbt
	Theta            float64 // rejection threshold on load; 1.0 = reject when a target would be missed
	DecodeDurationUs float64 // t_d, assumed decode duration of every request (predict mode)
	MaxBatchedTokens int64   // per-step token budget C of the engine

	// StepTime is the instance step-time model; the cluster sets it once the instances
	// exist. Nil (unit tests, or before wiring) falls back to a crude linear estimate.
	StepTime func(batch []*Request) int64

	tokens  []TokenID  // shared backing array for the synthetic requests
	scratch []*Request // reusable synthetic requests
}

// NewMooncakeAdmission builds the policy. Targets are given in seconds (TTFT, t_d) and
// milliseconds (TBT) as on the command line.
func NewMooncakeAdmission(mode string, ttftTargetS, tbtTargetMs, theta, decodeDurationS float64, maxBatchedTokens int64) *MooncakeAdmission {
	if mode != "now" && mode != "predict" {
		panic(fmt.Sprintf("mooncake admission: mode must be now or predict, got %q", mode))
	}
	if ttftTargetS <= 0 || tbtTargetMs <= 0 || theta <= 0 || decodeDurationS <= 0 || maxBatchedTokens <= 0 {
		panic("mooncake admission: targets, theta, decode duration and token budget must be > 0")
	}
	return &MooncakeAdmission{
		Mode:             mode,
		TTFTTargetUs:     ttftTargetS * 1e6,
		TBTTargetUs:      tbtTargetMs * 1e3,
		Theta:            theta,
		DecodeDurationUs: decodeDurationS * 1e6,
		MaxBatchedTokens: maxBatchedTokens,
	}
}

// Admit implements AdmissionPolicy.
func (m *MooncakeAdmission) Admit(req *Request, state *RouterState) (bool, string) {
	if state == nil || len(state.Snapshots) == 0 {
		return true, ""
	}
	prompt := req.InputLen()

	// Prefill side: predicted TTFT on the best instance.
	best, bestTTFT := -1, math.Inf(1)
	for i := range state.Snapshots {
		t := m.predictTTFT(&state.Snapshots[i], prompt)
		if t < bestTTFT {
			best, bestTTFT = i, t
		}
	}
	loadPrefill := bestTTFT / m.TTFTTargetUs

	// Decode side: mean over instances of predicted TBT / target.
	sum := 0.0
	for j := range state.Snapshots {
		s := &state.Snapshots[j]
		k := float64(s.BatchSize)
		if m.Mode == "predict" {
			remaining := 1 - bestTTFT/m.DecodeDurationUs
			if remaining < 0 {
				remaining = 0
			}
			k = k*remaining + float64(s.QueueDepth)
		}
		if j == best {
			k += 1
		}
		sum += float64(m.predictTBT(s, int(math.Ceil(k)), prompt)) / m.TBTTargetUs
	}
	loadDecode := sum / float64(len(state.Snapshots))

	if loadPrefill > m.Theta {
		return false, fmt.Sprintf("mooncake: prefill load %.2f > %.2f (ttft_hat %.2fs)", loadPrefill, m.Theta, bestTTFT/1e6)
	}
	if loadDecode > m.Theta {
		return false, fmt.Sprintf("mooncake: decode load %.2f > %.2f (%s)", loadDecode, m.Theta, m.Mode)
	}
	return true, ""
}

// predictTTFT returns the predicted time to first token, in microseconds, if the request
// were routed to this instance. Two bounds, the larger wins:
//
//   - compute: all prefill work ahead of the request plus its own prompt, at the instance's
//     current prefill rate (Mooncake's own estimator, exact for a dedicated prefill pool);
//   - memory: in a pooled engine a queued request enters the batch only when KV space is
//     free, so if the work ahead does not fit in the free KV the request waits for
//     completions. With the paper's system-level device (every request holds its KV for
//     t_d) the instance releases KvTokensInUse / t_d tokens per unit time, and the wait is
//     the shortfall divided by that rate.
func (m *MooncakeAdmission) predictTTFT(s *RoutingSnapshot, prompt int64) float64 {
	work := float64(s.QueuedPromptTokens + s.RunningPrefillTokens + prompt)
	compute := work / m.prefillRate(s, prompt)
	free := float64(s.TotalKvCapacityTokens - s.KvTokensInUse)
	if s.TotalKvCapacityTokens <= 0 || s.KvTokensInUse <= 0 || work <= free {
		return compute
	}
	release := float64(s.KvTokensInUse) / m.DecodeDurationUs // tokens per microsecond
	memory := (work - free) / release
	if memory > compute {
		return memory
	}
	return compute
}

// prefillRate returns prefill tokens per microsecond on this instance: the largest chunk a
// step can prefill next to the current decode batch, divided by the duration of that step.
func (m *MooncakeAdmission) prefillRate(s *RoutingSnapshot, prompt int64) float64 {
	chunk := m.MaxBatchedTokens - int64(s.BatchSize)
	if chunk < m.MaxBatchedTokens/4 {
		chunk = m.MaxBatchedTokens / 4 // the engine always makes some prefill progress
	}
	batch := m.decodeBatch(s.BatchSize, m.meanContext(s, prompt))
	pre := m.synthetic(len(batch))
	pre.InputTokens = m.backing(max(prompt, chunk)) // the queued work is prefilled in full chunks
	pre.ProgressIndex = 0
	pre.NumNewTokens = int(chunk)
	batch = append(batch, pre)
	return float64(chunk) / float64(m.stepTime(batch))
}

// predictTBT returns the step time, in microseconds, of a batch of k decode requests with
// this instance's mean context length.
func (m *MooncakeAdmission) predictTBT(s *RoutingSnapshot, k int, prompt int64) int64 {
	if k <= 0 {
		return 0
	}
	return m.stepTime(m.decodeBatch(k, m.meanContext(s, prompt)))
}

// meanContext is the mean KV context of the instance's running requests, or the arriving
// prompt plus a nominal partial answer when the instance is idle.
func (m *MooncakeAdmission) meanContext(s *RoutingSnapshot, prompt int64) int64 {
	if s.BatchSize > 0 && s.KvTokensInUse > 0 {
		return s.KvTokensInUse / int64(s.BatchSize)
	}
	return prompt + 64
}

// decodeBatch returns k synthetic decode requests (ProgressIndex = ctx, one new token each).
func (m *MooncakeAdmission) decodeBatch(k int, ctx int64) []*Request {
	if ctx < 1 {
		ctx = 1
	}
	out := make([]*Request, 0, k+1)
	for i := 0; i < k; i++ {
		r := m.synthetic(i)
		r.InputTokens = m.backing(1)
		r.ProgressIndex = ctx
		r.NumNewTokens = 1
		out = append(out, r)
	}
	return out
}

// synthetic returns the i-th reusable synthetic request.
func (m *MooncakeAdmission) synthetic(i int) *Request {
	for len(m.scratch) <= i {
		m.scratch = append(m.scratch, &Request{ID: "mooncake-synthetic", State: StateRunning})
	}
	return m.scratch[i]
}

// backing returns a token slice of length n over a shared array (the step-time model reads
// only its length).
func (m *MooncakeAdmission) backing(n int64) []TokenID {
	if int64(len(m.tokens)) < n {
		m.tokens = make([]TokenID, n)
	}
	return m.tokens[:n]
}

// stepTime evaluates the step-time model, or a crude linear stand-in when none is wired:
// 15 ms per step plus 0.2 ms per decode request plus 10 us per prefill token.
func (m *MooncakeAdmission) stepTime(batch []*Request) int64 {
	if m.StepTime != nil {
		t := m.StepTime(batch)
		if t < 1 {
			t = 1
		}
		return t
	}
	t := 15000.0
	for _, r := range batch {
		if r.ProgressIndex < r.InputLen() {
			t += 10 * float64(r.NumNewTokens)
		} else {
			t += 200
		}
	}
	return int64(t)
}
