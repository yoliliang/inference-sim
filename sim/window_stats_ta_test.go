package sim

import (
	"math"
	"testing"
)

// ours: the time-average statistics of the draft objective (10) count rewards of jobs that
// complete inside the window and the time every accepted job spends in the system inside it.
func TestWindowStats_TimeAverage(t *testing.T) {
	m := NewMetrics()
	add := func(id string, arrivedS, e2eS float64, done bool) {
		m.Requests[id] = RequestMetrics{ID: id, ArrivedAt: arrivedS, TenantID: "type1", NumPrefillTokens: 100, NumDecodeTokens: 10}
		if done {
			m.RequestE2Es[id] = e2eS * 1e6
		}
	}
	add("a", 5, 10, true)  // in system 5..15: 5 s inside [10,20), completes inside
	add("b", 12, 13, true) // 12..25: 8 s inside, completes after the window
	add("c", 18, 0, false) // unfinished at the horizon 30: 2 s inside
	add("d", 1, 7, true)   // 1..8: before the window, no contribution
	m.ExtraRequests = append(m.ExtraRequests,
		RequestMetrics{ID: "r", ArrivedAt: 11, TenantID: "type1", Status: "rejected"},   // never in the system
		RequestMetrics{ID: "q", ArrivedAt: 19, TenantID: "type1", Status: "unfinished"}) // shared queue at the horizon: 1 s

	w := m.WindowStats(10, 20, 30)
	s := w.Types["type1"]
	if math.Abs(s.TATimeInSystemS-16) > 1e-9 {
		t.Errorf("time in system inside the window: got %v, want 16", s.TATimeInSystemS)
	}
	if s.TACompleted != 1 || s.TASumInputTokens != 100 || s.TASumOutputTokens != 10 {
		t.Errorf("completions inside the window: got %d jobs, %d input, %d output; want 1, 100, 10",
			s.TACompleted, s.TASumInputTokens, s.TASumOutputTokens)
	}
	if all := w.Types["all"]; all.TATimeInSystemS != s.TATimeInSystemS || all.TACompleted != s.TACompleted {
		t.Errorf("the all row must equal the single type here")
	}
	// the cohort statistics still select by arrival: b, c, r and q arrived in the window
	if s.Arrived != 4 || s.Rejected != 1 || s.Completed != 1 || s.Unfinished != 2 {
		t.Errorf("cohort counts changed: %+v", *s)
	}
}
