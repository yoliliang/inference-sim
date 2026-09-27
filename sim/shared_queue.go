package sim

// ours: the shared queue of the pooled control model (--shared-queue-push).
//
// Admitted requests wait here, unstarted and unassigned, in arrival order. A request
// leaves the shared queue only when an instance's batch formation takes it into a batch;
// from then on it belongs to that instance (its work in progress) and never returns.
// Routing as a separate decision disappears: choosing a shared request into a batch is
// the routing. The simulator never decides who takes what; a SetPolicy does.
type SharedQueue struct {
	items []*Request
}

// Len returns the number of waiting requests.
func (q *SharedQueue) Len() int { return len(q.items) }

// Items returns the waiting requests in arrival order. Read only: callers must not
// modify the slice.
func (q *SharedQueue) Items() []*Request { return q.items }

// Push appends a request (arrival order is push order).
func (q *SharedQueue) Push(r *Request) { q.items = append(q.items, r) }

// Remove deletes r, preserving the order of the others. Returns false if absent.
func (q *SharedQueue) Remove(r *Request) bool {
	for i, x := range q.items {
		if x == r {
			q.items = append(q.items[:i], q.items[i+1:]...)
			return true
		}
	}
	return false
}

// SharedQueueAccess is the view one instance's batch formation has of the shared queue.
// Candidates lists the waiting requests in arrival order (read only). Take moves r to the
// calling instance: r is removed from the shared queue, adopted by the instance (metrics
// row, servability guards, in-flight accounting) and is from then on that instance's work
// in progress. Take returns false when the instance had to drop r as unservable.
type SharedQueueAccess interface {
	Candidates() []*Request
	Take(r *Request) bool
	Len() int
	// Fits reports whether the calling instance could ever serve r (context limit, KV
	// capacity). A candidate that does not fit is skipped, not taken, so that another
	// instance can take it.
	Fits(r *Request) bool
}

// ReferenceStepTime (ours) evaluates a step-time function on a fixed reference batch (one
// prefill chunk of 512 tokens plus eight decode requests at 1,024 tokens of context) so
// that instances of different hardware can be ordered by speed. Used by the fastest-first
// wake rule; equal for identical instances.
func ReferenceStepTime(stepTime func(batch []*Request) int64) int64 {
	if stepTime == nil {
		return 0
	}
	tokens := make([]TokenID, 1024)
	batch := make([]*Request, 0, 9)
	batch = append(batch, &Request{ID: "ref-prefill", State: StateRunning, InputTokens: tokens[:512], NumNewTokens: 512})
	for i := 0; i < 8; i++ {
		batch = append(batch, &Request{ID: "ref-decode", State: StateRunning, InputTokens: tokens,
			OutputTokens: tokens[:1], ProgressIndex: 1024, NumNewTokens: 1}) // OutputTokens set: the latency models classify decode by it
	}
	return stepTime(batch)
}
