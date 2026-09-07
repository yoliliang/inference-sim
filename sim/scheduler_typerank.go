package sim

import "sort"

// TypeRankScheduler (ours) orders the wait queue by a static rank on TenantID,
// then by arrival time, then by ID for determinism.
//
// TenantID is the job-type label of our formal spec (types3.yaml sets it to
// "type1", "type2", "type3"). Lower rank is scheduled first. Types absent from
// the rank map go last, in arrival order, so the scheduler degrades to FCFS on
// workloads without tenant labels.
//
// This scheduler reads only InputLen, TenantID, ArrivalTime and ID. It never
// reads OutputTokens (INV-9): the policy knows the type, not the realized length.
type TypeRankScheduler struct {
	Rank map[string]int
}

// DefaultTypeRank is the rank used when no configuration is supplied:
// short chat first, RAG second, long generation last.
var DefaultTypeRank = map[string]int{
	"type1": 0,
	"type2": 1,
	"type3": 2,
}

func NewTypeRankScheduler(rank map[string]int) *TypeRankScheduler {
	if rank == nil {
		rank = DefaultTypeRank
	}
	return &TypeRankScheduler{Rank: rank}
}

func (t *TypeRankScheduler) rankOf(r *Request) int {
	if v, ok := t.Rank[r.TenantID]; ok {
		return v
	}
	return int(^uint(0) >> 1) // max int: unknown types go last
}

func (t *TypeRankScheduler) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		ri, rj := t.rankOf(reqs[i]), t.rankOf(reqs[j])
		if ri != rj {
			return ri < rj
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}
