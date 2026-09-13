package sim

// ours: steady-state window statistics computed in memory at the end of a run, so the
// per-request array does not have to be written to disk and re-read by the analysis.
// The window is the set of requests that ARRIVED in [WindowStartS, WindowEndS). Latency
// statistics use completed window requests; unfinished requests (still queued or running
// at the horizon) are counted and their censored sojourn (horizon - arrival) summed;
// admission rejections (Metrics.ExtraRequests with Status rejected) are counted.
// The per-type sums are the sufficient statistics of the revenue management objective
// pi_in * sum(P) + pi_out * sum(o) - h * sum(sojourn), so prices can be applied later.

import "sort"

// Quantiles summarises one latency in milliseconds over the completed window requests.
type Quantiles struct {
	Count  int     `json:"count"`
	MeanMs float64 `json:"mean_ms"`
	P50Ms  float64 `json:"p50_ms"`
	P99Ms  float64 `json:"p99_ms"`
}

// WindowTypeStats is the window record of one request type ("all" for every type).
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

// WindowStats is the file-only "window" block of MetricsOutput.
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

// WindowStats computes the window block. horizonS is the run's end time in seconds.
func (m *Metrics) WindowStats(startS, endS, horizonS float64) *WindowStats {
	types := map[string]*WindowTypeStats{"all": {}}
	lat := map[string][3][]float64{} // per type: ttft, e2e, wait samples in ms
	get := func(t string) *WindowTypeStats {
		if _, ok := types[t]; !ok {
			types[t] = &WindowTypeStats{}
		}
		return types[t]
	}
	add := func(rm RequestMetrics, ttft, e2e, wait float64, completed, rejected bool) {
		for _, t := range []string{rm.TenantID, "all"} {
			s := get(t)
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
				l := lat[t]
				l[0] = append(l[0], ttft)
				l[1] = append(l[1], e2e)
				l[2] = append(l[2], wait)
				lat[t] = l
			default:
				s.Unfinished++
				s.SumCensoredS += horizonS - rm.ArrivedAt
			}
		}
	}
	for _, id := range sortedRequestIDs(m.Requests) { // sorted: deterministic float sums
		rm := m.Requests[id]
		if rm.ArrivedAt < startS || rm.ArrivedAt >= endS {
			continue
		}
		e2e, completed := m.RequestE2Es[id]
		add(rm, m.RequestTTFTs[id]/1e3, e2e/1e3, float64(m.RequestSchedulingDelays[id])/1e3, completed, false)
	}
	for _, rm := range m.ExtraRequests {
		if rm.ArrivedAt < startS || rm.ArrivedAt >= endS {
			continue
		}
		add(rm, 0, 0, 0, false, rm.Status == "rejected")
	}
	for t, s := range types {
		l := lat[t]
		s.TTFT, s.E2E, s.Wait = quantiles(l[0]), quantiles(l[1]), quantiles(l[2])
	}
	return &WindowStats{StartS: startS, EndS: endS, HorizonS: horizonS, Types: types}
}
