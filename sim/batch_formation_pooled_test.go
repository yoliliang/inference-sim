package sim

import "testing"

// ours: tests for the pooled batch formation and the fcfs-pool set policy.

type fakeShared struct {
	items  []*Request
	taken  []string
	refuse map[string]bool // Take returns false for these (unservable)
}

func (f *fakeShared) Candidates() []*Request { return f.items }
func (f *fakeShared) Len() int               { return len(f.items) }
func (f *fakeShared) Take(r *Request) bool {
	for i, x := range f.items {
		if x == r {
			f.items = append(f.items[:i], f.items[i+1:]...)
			break
		}
	}
	if f.refuse[r.ID] {
		return false
	}
	f.taken = append(f.taken, r.ID)
	return true
}

var pooledSeq int

// pooledReq builds a request with a prompt of distinct tokens (so that no two requests share
// a prefix and the prefix cache stays out of the way).
func pooledReq(id string, prompt int) *Request {
	pooledSeq++
	toks := make([]TokenID, prompt)
	for i := range toks {
		toks[i] = TokenID(pooledSeq*100000 + i + 1)
	}
	return &Request{ID: id, InputTokens: toks, OutputTokens: make([]TokenID, 4), State: StateQueued}
}

func pooledCtx(kv KVStore, running []*Request, local *WaitQueue, shared SharedQueueAccess, budget int64) BatchContext {
	return BatchContext{
		RunningBatch:        &Batch{Requests: running},
		WaitQ:               local,
		KVCache:             kv,
		SharedQueue:         shared,
		MaxNumBatchedTokens: budget,
		MaxNumSeqs:          100,
		Now:                 1000,
		ComputedTokens:      make(map[string]int64),
	}
}

func ids(reqs []*Request) []string {
	out := make([]string, len(reqs))
	for i, r := range reqs {
		out[i] = r.ID
	}
	return out
}

func TestPooled_TakesSharedRequestsInArrivalOrder(t *testing.T) {
	kv := MustNewKVCacheState(100, 16)
	shared := &fakeShared{items: []*Request{pooledReq("a", 32), pooledReq("b", 32), pooledReq("c", 32)}}
	bf := NewPooledBatchFormation("fcfs", "fcfs-pool")
	res := bf.FormBatch(pooledCtx(kv, nil, &WaitQueue{}, shared, 1000))
	if got := ids(res.RunningBatch.Requests); len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("batch %v, want [a b c]", got)
	}
	if len(shared.taken) != 3 || shared.Len() != 0 {
		t.Errorf("taken %v, remaining %d", shared.taken, shared.Len())
	}
	for _, r := range res.RunningBatch.Requests {
		if r.State != StateRunning || r.NumNewTokens != 32 {
			t.Errorf("%s: state %v tokens %d", r.ID, r.State, r.NumNewTokens)
		}
	}
}

func TestPooled_LocalPreemptedRequestsGoFirst(t *testing.T) {
	kv := MustNewKVCacheState(100, 16)
	local := &WaitQueue{}
	local.Enqueue(pooledReq("evicted", 16))
	shared := &fakeShared{items: []*Request{pooledReq("new", 16)}}
	bf := NewPooledBatchFormation("fcfs", "fcfs-pool")
	res := bf.FormBatch(pooledCtx(kv, nil, local, shared, 1000))
	if got := ids(res.RunningBatch.Requests); len(got) != 2 || got[0] != "evicted" || got[1] != "new" {
		t.Fatalf("batch %v, want [evicted new]", got)
	}
	if local.Len() != 0 {
		t.Errorf("local queue should be empty, has %d", local.Len())
	}
}

func TestPooled_TokenBudgetStopsAdmission(t *testing.T) {
	// Budget 48: a (32) fits, b (32) gets a 16-token chunk, c is not admitted (budget 0).
	kv := MustNewKVCacheState(100, 16)
	shared := &fakeShared{items: []*Request{pooledReq("a", 32), pooledReq("b", 32), pooledReq("c", 32)}}
	bf := NewPooledBatchFormation("fcfs", "fcfs-pool")
	res := bf.FormBatch(pooledCtx(kv, nil, &WaitQueue{}, shared, 48))
	got := ids(res.RunningBatch.Requests)
	if len(got) != 2 || got[1] != "b" || res.RunningBatch.Requests[1].NumNewTokens != 16 {
		t.Fatalf("batch %v with tokens %v, want [a b] and b chunked to 16", got, []int{res.RunningBatch.Requests[0].NumNewTokens, res.RunningBatch.Requests[1].NumNewTokens})
	}
	if shared.Len() != 1 || shared.items[0].ID != "c" {
		t.Errorf("c should remain in the shared queue")
	}
}

func TestPooled_KVShortageStopsAtHeadOfLine(t *testing.T) {
	// 4 blocks: a needs 3, b needs 2 (does not fit after a), c needs 1 (would fit but fcfs stops).
	kv := MustNewKVCacheState(4, 16)
	shared := &fakeShared{items: []*Request{pooledReq("a", 48), pooledReq("b", 32), pooledReq("c", 16)}}
	bf := NewPooledBatchFormation("fcfs", "fcfs-pool")
	res := bf.FormBatch(pooledCtx(kv, nil, &WaitQueue{}, shared, 1000))
	if got := ids(res.RunningBatch.Requests); len(got) != 1 || got[0] != "a" {
		t.Fatalf("batch %v, want [a] (head-of-line stop)", got)
	}
	if shared.Len() != 2 {
		t.Errorf("b and c should remain, have %d", shared.Len())
	}
}

func TestPooled_UnservableTakeReleasesItsBlocks(t *testing.T) {
	kv := MustNewKVCacheState(10, 16)
	shared := &fakeShared{items: []*Request{pooledReq("bad", 32), pooledReq("ok", 32)}, refuse: map[string]bool{"bad": true}}
	bf := NewPooledBatchFormation("fcfs", "fcfs-pool")
	res := bf.FormBatch(pooledCtx(kv, nil, &WaitQueue{}, shared, 1000))
	if got := ids(res.RunningBatch.Requests); len(got) != 1 || got[0] != "ok" {
		t.Fatalf("batch %v, want [ok]", got)
	}
	// ok holds 2 of 10 blocks; the refused request's 2 blocks must have been released, so a
	// request needing 7 of the remaining 8 blocks still fits (only 6 would be free otherwise).
	probe := pooledReq("probe", 112)
	if !kv.AllocateKVBlocks(probe, 0, 112, nil) {
		t.Error("the refused request's blocks were not released")
	}
}
func TestPooled_EvictionStepAdmitsNobody(t *testing.T) {
	// Cache full with three running requests of 3 blocks each (9 of 10 blocks) plus one
	// decode block granted in Phase 1: the second running request finds no block and
	// evicts the tail; no shared request is admitted in that step (vLLM convention).
	kv := MustNewKVCacheState(10, 16)
	running := []*Request{
		makeRunningRequest("r1", "standard", 100, 48, kv),
		makeRunningRequest("r2", "standard", 200, 48, kv),
		makeRunningRequest("r3", "standard", 300, 48, kv),
	}
	shared := &fakeShared{items: []*Request{pooledReq("new", 16)}}
	local := &WaitQueue{}
	bf := NewPooledBatchFormation("fcfs", "fcfs-pool")
	res := bf.FormBatch(pooledCtx(kv, running, local, shared, 1000))
	if !res.PreemptionHappened {
		t.Fatal("expected an eviction")
	}
	if shared.Len() != 1 {
		t.Errorf("no shared request may be taken in an eviction step")
	}
	if local.Len() != 1 || local.Peek().ID != "r3" {
		t.Errorf("the victim r3 must sit at the front of the local queue, got %v", ids(local.Items()))
	}
}

func TestPooled_SetPolicyRegistered(t *testing.T) {
	if !IsValidSetPolicy("fcfs-pool") || !IsValidBatchFormation("pooled") {
		t.Error("fcfs-pool and pooled must be registered names")
	}
}
