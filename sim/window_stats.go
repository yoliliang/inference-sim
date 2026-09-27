package sim

// ours: steady-state window statistics, computed in memory at the end of a run from the
// per-request records (or streamed in lean mode, see metrics_lean.go). Requests are selected
// by arrival time in [startS, endS). Per type (and "all"): counts by outcome and the sums that
// are the sufficient statistics of the revenue management objective
// pi_in * sum(P) + pi_out * sum(o) - h * sum(sojourn), so prices can be applied later;
// admission rejections (Metrics.ExtraRequests with Status rejected) are counted.

import "sort"

type Quantiles struct {
	Count  int     `json:"count"`
	MeanMs float64 `json:"mean_ms"`
	P50Ms  float64 `json:"p50_ms"`
	P99Ms  float64 `json:"p99_ms"`
}

type WindowTypeStats struct {
	Arrived         int       `json:"arrived"`
	Completed       int       `json:"completed"`
	Unfinished      int       `json:"unfinished"`
	Rejected        int       `json:"rejected"`
	SumInputTokens  int64     `json:"sum_input_tokens"`  // completed requests
	SumOutputTokens int64     `json:"sum_output_tokens"` // completed requests
	SumSojournS     float64   `json:"sum_sojourn_s"`     // completed requests, arrival to last token
	SumCensoredS    float64   `json:"sum_censored_s"`    // unfinished requests, horizon minus arrival
	SumPreemptions  int64     `json:"sum_preemptions"`   // over all window arrivals
	SumWastedTokens int64     `json:"sum_wasted_tokens"` // over all window arrivals
	TTFT            Quantiles `json:"ttft"`
	E2E             Quantiles `json:"e2e"`
	Wait            Quantiles `json:"wait"` // scheduling delay
}

type WindowStats struct {
	StartS   float64                     `json:"start_s"`
	EndS     float64                     `json:"end_s"`
	HorizonS float64                     `json:"horizon_s"`
	Types    map[string]*WindowTypeStats `json:"types"`
}

func quantiles(v []float64) Quantiles {
	if len(v) == 0 {
		return Quantiles{}
	}
	sort.Float64s(v)
	sum := 0.0
	for _, x := range v {
		sum += x
	}
	// linear interpolation between order statistics, the numpy and pandas default
	q := func(p float64) float64 {
		pos := p * float64(len(v)-1)
		lo := int(pos)
		if lo+1 >= len(v) {
			return v[len(v)-1]
		}
		return v[lo] + (pos-float64(lo))*(v[lo+1]-v[lo])
	}
	return Quantiles{Count: len(v), MeanMs: sum / float64(len(v)), P50Ms: q(0.5), P99Ms: q(0.99)}
}

// windowAcc accumulates window statistics per type. The same accumulator serves the
// end-of-run path (fed in sorted request-id order) and the lean streaming path (fed at
// completion time).
type windowAcc struct {
	types    map[string]*WindowTypeStats
	lat      map[string][3][]float64 // per type: ttft, e2e, wait samples in ms
	horizonS float64                 // needed only for unfinished rows (censored time)
}

func newWindowAcc() *windowAcc {
	return &windowAcc{types: map[string]*WindowTypeStats{"all": {}}, lat: map[string][3][]float64{}}
}

func (a *windowAcc) get(t string) *WindowTypeStats {
	if _, ok := a.types[t]; !ok {
		a.types[t] = &WindowTypeStats{}
	}
	return a.types[t]
}

func (a *windowAcc) add(rm RequestMetrics, ttft, e2e, wait float64, completed, rejected bool) {
	for _, t := range []string{rm.TenantID, "all"} {
		s := a.get(t)
		s.Arrived++
		s.SumPreemptions += int64(rm.PreemptionCount)
		s.SumWastedTokens += rm.WastedTokens
		switch {
		case rejected:
			s.Rejected++
		case completed:
			s.Completed++
			s.SumInputTokens += int64(rm.NumPrefillTokens)
			s.SumOutputTokens += int64(rm.NumDecodeTokens)
			s.SumSojournS += e2e / 1000.0
			l := a.lat[t]
			l[0] = append(l[0], ttft)
			l[1] = append(l[1], e2e)
			l[2] = append(l[2], wait)
			a.lat[t] = l
		default:
			s.Unfinished++
			s.SumCensoredS += a.horizonS - rm.ArrivedAt
		}
	}
}

// merge adds another accumulator's counts, sums and samples (cluster aggregation of lean
// per-instance accumulators). Types are visited in sorted order for determinism.
func (a *windowAcc) merge(o *windowAcc) {
	if o == nil {
		return
	}
	keys := make([]string, 0, len(o.types))
	for t := range o.types {
		keys = append(keys, t)
	}
	sort.Strings(keys)
	for _, t := range keys {
		s, os := a.get(t), o.types[t]
		s.Arrived += os.Arrived
		s.Completed += os.Completed
		s.Unfinished += os.Unfinished
		s.Rejected += os.Rejected
		s.SumInputTokens += os.SumInputTokens
		s.SumOutputTokens += os.SumOutputTokens
		s.SumSojournS += os.SumSojournS
		s.SumCensoredS += os.SumCensoredS
		s.SumPreemptions += os.SumPreemptions
		s.SumWastedTokens += os.SumWastedTokens
		l, ol := a.lat[t], o.lat[t]
		for i := range l {
			l[i] = append(l[i], ol[i]...)
		}
		a.lat[t] = l
	}
}

func (a *windowAcc) finish(startS, endS float64) *WindowStats {
	for t, s := range a.types {
		l := a.lat[t]
		s.TTFT, s.E2E, s.Wait = quantiles(l[0]), quantiles(l[1]), quantiles(l[2])
	}
	return &WindowStats{StartS: startS, EndS: endS, HorizonS: a.horizonS, Types: a.types}
}

// WindowStats computes the window block. horizonS is the run's end time in seconds.
// In lean mode the completed window arrivals were streamed into Lean.win as they completed;
// the requests still in Metrics.Requests are the unfinished ones. Otherwise every request is
// read from the per-request maps in sorted id order (deterministic float sums).
func (m *Metrics) WindowStats(startS, endS, horizonS float64) *WindowStats {
	var acc *windowAcc
	if m.Lean != nil {
		acc = m.Lean.win
	} else {
		acc = newWindowAcc()
	}
	acc.horizonS = horizonS
	for _, id := range sortedRequestIDs(m.Requests) { // sorted: deterministic float sums
		rm := m.Requests[id]
		if rm.ArrivedAt < startS || rm.ArrivedAt >= endS {
			continue
		}
		e2e, completed := m.RequestE2Es[id]
		acc.add(rm, m.RequestTTFTs[id]/1e3, e2e/1e3, float64(m.RequestSchedulingDelays[id])/1e3, completed, false)
	}
	for _, rm := range m.ExtraRequests {
		if rm.ArrivedAt < startS || rm.ArrivedAt >= endS {
			continue
		}
		acc.add(rm, 0, 0, 0, false, rm.Status == "rejected")
	}
	return acc.finish(startS, endS)
}
