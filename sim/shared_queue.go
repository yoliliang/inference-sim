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
}
