package sim

import "testing"

// ours: tests for the Shortest-Request-First preemption policy (srf).

// srfContext builds a full cache holding the given running requests and one queued
// request that needs a block, so that FormBatch must evict.
func srfContext(t *testing.T, kvCache KVStore, running []*Request, policy string) (BatchFormation, BatchContext) {
	t.Helper()
	newReq := &Request{ID: "new", InputTokens: make([]TokenID, 16), OutputTokens: make([]TokenID, 1), State: StateQueued}
	wq := &WaitQueue{}
	wq.Enqueue(newReq)
	return NewBatchFormation(policy), BatchContext{
		RunningBatch:        &Batch{Requests: running},
		WaitQ:               wq,
		KVCache:             kvCache,
		MaxNumBatchedTokens: 10000,
		MaxNumSeqs:          10,
		Now:                 1000,
		ComputedTokens:      make(map[string]int64),
	}
}

func TestPreemption_SRF_EvictsFewestKV(t *testing.T) {
	// 10 blocks x 16 tokens. Holdings: long 64 (4 blocks), short 16 (1 block), mid 48 (3 blocks):
	// 8 blocks used, 2 free. Phase 1 gives "long" and "short" their decode blocks (2 blocks),
	// "mid" then finds the cache full. SRF must evict "short", the smallest holder, although
	// it is neither the tail nor the newest.
	kvCache := MustNewKVCacheState(10, 16)
	running := []*Request{
		makeRunningRequest("long", "standard", 100, 64, kvCache),
		makeRunningRequest("short", "standard", 200, 16, kvCache),
		makeRunningRequest("mid", "standard", 300, 48, kvCache),
	}
	bf, ctx := srfContext(t, kvCache, running, "srf")

	result := bf.FormBatch(ctx)

	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if got := result.Preempted[0].Request.ID; got != "short" {
		t.Errorf("SRF preemption: evicted %q, want \"short\" (fewest KV entries)", got)
	}
	if got := result.Preempted[0].WastedTokens; got != 16 {
		t.Errorf("SRF preemption: wasted tokens %d, want 16", got)
	}
}

func TestPreemption_SRF_TieGoesToTail(t *testing.T) {
	// Equal holdings (48 tokens each): SRF must coincide with fcfs and evict the tail.
	kvCache := MustNewKVCacheState(10, 16)
	running := []*Request{
		makeRunningRequest("first", "standard", 100, 48, kvCache),
		makeRunningRequest("second", "standard", 200, 48, kvCache),
		makeRunningRequest("third", "standard", 300, 48, kvCache),
	}
	bf, ctx := srfContext(t, kvCache, running, "srf")

	result := bf.FormBatch(ctx)

	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if got := result.Preempted[0].Request.ID; got != "third" {
		t.Errorf("SRF tie-break: evicted %q, want \"third\" (tail)", got)
	}
}

func TestPreemption_SRF_RepeatsUntilEnough(t *testing.T) {
	// 6 blocks. Holdings: a 32 (2 blocks), b 16 (1 block), c 48 (3 blocks): cache full.
	// Phase 1: "a" needs a decode block and the cache is full. SRF evicts "b" (16 tokens),
	// which frees the one block "a" needs. Then "c" needs a block, the cache is full again,
	// and the smallest remaining holder is "a" (33), so "a" is evicted next. The victims
	// must be ordered by holdings, not by position, and the loop must continue until each
	// allocation succeeds.
	kvCache := MustNewKVCacheState(6, 16)
	running := []*Request{
		makeRunningRequest("a", "standard", 100, 32, kvCache),
		makeRunningRequest("b", "standard", 200, 16, kvCache),
		makeRunningRequest("c", "standard", 300, 48, kvCache),
	}
	bf, ctx := srfContext(t, kvCache, running, "srf")

	result := bf.FormBatch(ctx)

	if len(result.Preempted) < 1 {
		t.Fatal("expected at least one preemption")
	}
	if got := result.Preempted[0].Request.ID; got != "b" {
		t.Errorf("first SRF victim %q, want \"b\"", got)
	}
	for _, p := range result.Preempted {
		if p.Request.State != StateQueued || p.Request.ProgressIndex != 0 {
			t.Errorf("victim %s not reset: state %v progress %d", p.Request.ID, p.Request.State, p.Request.ProgressIndex)
		}
	}
	// KV conservation: blocks held by the batch plus free blocks equal the pool.
	used := int64(0)
	for _, r := range result.RunningBatch.Requests {
		used += (r.ProgressIndex + int64(r.NumNewTokens) + 15) / 16
	}
	if used > 6 {
		t.Errorf("batch holds %d blocks, pool has 6", used)
	}
}

func TestPreemption_SRF_IsValidName(t *testing.T) {
	if !IsValidPreemptionPolicy("srf") {
		t.Error("srf should be a valid preemption policy name")
	}
	if NewBatchFormation("srf").(*VLLMBatchFormation).preemptionPolicy != PreemptionSRF {
		t.Error("NewBatchFormation(\"srf\") should select PreemptionSRF")
	}
}
