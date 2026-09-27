package sim

import "sort"

// ours: streaming per-request statistics for the lean output mode
// (--drop-per-request-output). Upstream BLIS keeps five maps keyed by request id (per-request
// metrics, TTFT, E2E, scheduling delay, ITL) for every completed request until the run ends,
// then builds the aggregate output and the per-request array from them. For our fixed-horizon
// runs (up to 1.7 million completions at n = 64) those maps are a third of the heap and most
// of what the garbage collector has to scan. In lean mode a completed request is folded into
// this structure the moment it completes and its map entries are deleted, so the maps hold
// only in-flight requests.
//
// What is kept: the samples the aggregate output needs (TTFT, E2E, scheduling delay of every
// completed request, as plain float64 slices) and the steady-state window accumulators
// (per type: counts, token sums, sojourn sums, latency samples of window arrivals). The
// aggregate means and percentiles are identical to the map-based path (integer-valued
// samples, order-independent sums, sorted percentiles). The window's sum_sojourn_s is
// accumulated in completion order instead of request-id order, so it can differ from the
// map-based path in the last floating-point digits.
type LeanStats struct {
	WindowStartS float64
	WindowEndS   float64 // 0 = no window block

	ttft  []float64 // microseconds, integer-valued
	e2e   []float64 // microseconds, integer-valued
	delay []float64 // microseconds

	win *windowAcc // completed window arrivals; unfinished and rejected rows are added at the end
}

// NewLeanStats returns an empty accumulator for the given window (seconds by arrival time).
func NewLeanStats(windowStartS, windowEndS float64) *LeanStats {
	return &LeanStats{WindowStartS: windowStartS, WindowEndS: windowEndS, win: newWindowAcc()}
}

// completed folds one completed request in and reports whether it fell in the window.
func (l *LeanStats) completed(rm RequestMetrics, ttftUs, e2eUs float64, delayUs int64) {
	l.ttft = append(l.ttft, ttftUs)
	l.e2e = append(l.e2e, e2eUs)
	l.delay = append(l.delay, float64(delayUs))
	if l.WindowEndS > 0 && rm.ArrivedAt >= l.WindowStartS && rm.ArrivedAt < l.WindowEndS {
		l.win.add(rm, ttftUs/1e3, e2eUs/1e3, float64(delayUs)/1e3, true, false)
	}
}

// SetGPU attributes every request this accumulator sees to one GPU type (the instance's).
func (l *LeanStats) SetGPU(gpu string) {
	l.win.gpuOf = func(RequestMetrics) string { return gpu }
}

// Merge appends another instance's accumulators (cluster aggregation).
func (l *LeanStats) Merge(o *LeanStats) {
	if o == nil {
		return
	}
	l.ttft = append(l.ttft, o.ttft...)
	l.e2e = append(l.e2e, o.e2e...)
	l.delay = append(l.delay, o.delay...)
	l.win.merge(o.win)
}

// foldCompleted (ours) moves a just-completed request from the per-request maps into the
// lean accumulators and deletes its map entries. Must run after recordRequestCompletion has
// written the maps.
func (m *Metrics) foldCompleted(id string) {
	if m.Lean == nil {
		return
	}
	rm, ok := m.Requests[id]
	if !ok {
		return
	}
	m.Lean.completed(rm, m.RequestTTFTs[id], m.RequestE2Es[id], m.RequestSchedulingDelays[id])
	delete(m.Requests, id)
	delete(m.RequestTTFTs, id)
	delete(m.RequestE2Es, id)
	delete(m.RequestSchedulingDelays, id)
	delete(m.RequestITLs, id)
	delete(m.RequestCompletionTimes, id)
}

// Sorted samples for the aggregate output. Upstream computes TTFT and scheduling-delay
// statistics over every request that has a recorded value, which includes requests still
// in flight at the horizon; those are still in the maps (only completed requests were
// folded out), so their entries are appended here. E2E is recorded only at completion.
func (l *LeanStats) sortedTTFT(inFlight map[string]float64) []float64 {
	return sortedWith(l.ttft, inFlight)
}
func (l *LeanStats) sortedE2E(inFlight map[string]float64) []float64 {
	return sortedWith(l.e2e, inFlight)
}
func (l *LeanStats) sortedDelay(inFlight map[string]int64) []float64 {
	out := make([]float64, 0, len(l.delay)+len(inFlight))
	out = append(out, l.delay...)
	for _, v := range inFlight {
		out = append(out, float64(v))
	}
	sort.Float64s(out)
	return out
}

func sortedWith(v []float64, extra map[string]float64) []float64 {
	out := make([]float64, 0, len(v)+len(extra))
	out = append(out, v...)
	for _, x := range extra {
		out = append(out, x)
	}
	sort.Float64s(out)
	return out
}
