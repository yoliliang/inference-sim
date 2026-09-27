package sim

import "testing"

// ours: tests of the heterogeneity hooks of the shared-queue mode.

// A slower instance (larger step time on the reference batch) must sort after a faster
// one; the reference batch must contain decode work the latency models recognise.
func TestReferenceStepTime_OrdersBySpeed(t *testing.T) {
	seen := 0
	slow := func(batch []*Request) int64 {
		for _, r := range batch {
			if r.ProgressIndex >= r.InputLen() && len(r.OutputTokens) > 0 {
				seen++
			}
		}
		return 2 * int64(len(batch))
	}
	fast := func(batch []*Request) int64 { return int64(len(batch)) }
	if ReferenceStepTime(slow) <= ReferenceStepTime(fast) {
		t.Fatalf("slow instance should have the larger reference step time")
	}
	if seen != 8 {
		t.Fatalf("reference batch should hold 8 decode requests recognisable by the latency models, saw %d", seen)
	}
	if ReferenceStepTime(nil) != 0 {
		t.Fatalf("nil step time should give 0")
	}
}

// CanServe mirrors the servability guards without side effects: the request is unchanged
// and the answer depends on this instance's context limit and capacity.
func TestCanServe_ContextAndCapacity(t *testing.T) {
	kv := MustNewKVCacheState(8, 16) // 8 blocks of 16 tokens = 128 tokens
	sim := &Simulator{KVCache: kv, maxModelLen: 100}
	short := &Request{ID: "s", InputTokens: make([]TokenID, 40)}
	long := &Request{ID: "l", InputTokens: make([]TokenID, 120)} // fits the cache, exceeds the context limit
	huge := &Request{ID: "h", InputTokens: make([]TokenID, 200)} // exceeds the cache
	if !sim.CanServe(short) {
		t.Errorf("40-token prompt should be servable")
	}
	if sim.CanServe(long) || sim.CanServe(huge) {
		t.Errorf("prompts beyond the context limit or the cache must not be servable")
	}
	if short.MaxOutputLen != 0 {
		t.Errorf("CanServe must not mutate the request, MaxOutputLen became %d", short.MaxOutputLen)
	}
	sim.maxModelLen = 0
	if !sim.CanServe(long) {
		t.Errorf("without a context limit a 120-token prompt fits 128 tokens of cache")
	}
}
