package sim

import (
	"math"
	"strings"
	"testing"
)

// ours: tests of the fluid-dual admission and routing rule (model draft, Definition 1).

func fluidTestParams() *FluidDualParams {
	p := &FluidDualParams{}
	p.Meta.Eta0 = 2.0
	p.Types = []FluidDualType{{Name: "a", Prompt: 100, R: 1.0}, {Name: "b", Prompt: 1000, R: 0.5}}
	pair := func(c float64) FluidDualPair { return FluidDualPair{X: 1, AM: 0.1, AB: 0.05, C: c} }
	p.Instances = []FluidDualInstance{
		{ID: "i0", M: 1000, B: 100, GammaM: 1, GammaB: 0, KVTarget: 500, Stage0Target: 50,
			Types: map[string]FluidDualPair{"a": pair(0.2), "b": pair(0.2)}},
		{ID: "i1", M: 1000, B: 100, GammaM: 1, GammaB: 0, KVTarget: 500, Stage0Target: 50,
			Types: map[string]FluidDualPair{"a": pair(0.2), "b": pair(0.2)}},
	}
	return p
}

type fluidTestState map[string][2]int64

func (s fluidTestState) read(id string) (int64, int64, bool) {
	v, ok := s[id]
	return v[0], v[1], ok
}

func fluidRouterState(ids ...string) *RouterState {
	st := &RouterState{}
	for _, id := range ids {
		st.Snapshots = append(st.Snapshots, RoutingSnapshot{ID: id})
	}
	return st
}

func TestFluidDual_IndexAtZeroDeviation(t *testing.T) {
	f := NewFluidDual(fluidTestParams(), 1)
	// at the fluid levels the prices are the fluid multipliers: I = r - c - gammaM aM - gammaB aB
	got := f.Index("a", 0, 500, 50)
	want := 1.0 - 0.2 - 1*0.1 - 0*0.05
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("index at zero deviation: got %v want %v", got, want)
	}
	if f.Eta() != 2.0 {
		t.Fatalf("eta should be scale * eta0 = 2, got %v", f.Eta())
	}
}

func TestFluidDual_PricesMoveWithDeviationAndClipAtZero(t *testing.T) {
	f := NewFluidDual(fluidTestParams(), 1)
	// memory 300 tokens above target: gammaM = 1 + 2 * 0.3 = 1.6
	if got, want := f.Index("a", 0, 800, 50), 1.0-0.2-1.6*0.1; math.Abs(got-want) > 1e-12 {
		t.Errorf("memory above target: got %v want %v", got, want)
	}
	// far below target: gammaM = [1 + 2 * (-0.9)]^+ = 0
	if got, want := f.Index("a", 0, -400, 50), 1.0-0.2; math.Abs(got-want) > 1e-12 {
		t.Errorf("price must be clipped at zero: got %v want %v", got, want)
	}
	// prefill deviation: 100 stage-0 prompt tokens above target: gammaB = 0 + 2 * 1 = 2
	if got, want := f.Index("a", 0, 500, 150), 1.0-0.2-1*0.1-2*0.05; math.Abs(got-want) > 1e-12 {
		t.Errorf("prefill deviation: got %v want %v", got, want)
	}
}

func TestFluidDual_RoutesToLargestIndexTiesToSmallestInstance(t *testing.T) {
	f := NewFluidDual(fluidTestParams(), 1)
	st := fluidTestState{"i0": {500, 50}, "i1": {500, 50}}
	f.SetState(st.read)
	req := &Request{ID: "r1", TenantID: "a"}
	ok, _ := f.Admit(req, fluidRouterState("i0", "i1"))
	if !ok {
		t.Fatal("a job with a positive index must be accepted")
	}
	if d := f.Route(req, fluidRouterState("i0", "i1")); d.TargetInstance != "i0" {
		t.Errorf("equal indices must go to the smallest instance, got %s", d.TargetInstance)
	}
	// i0 above its memory target: the next job goes to i1
	st["i0"] = [2]int64{900, 50}
	req2 := &Request{ID: "r2", TenantID: "a"}
	f.Admit(req2, fluidRouterState("i0", "i1"))
	if d := f.Route(req2, fluidRouterState("i0", "i1")); d.TargetInstance != "i1" {
		t.Errorf("the job must go to the instance with the larger index, got %s", d.TargetInstance)
	}
	// only routable instances are considered
	req3 := &Request{ID: "r3", TenantID: "a"}
	f.Admit(req3, fluidRouterState("i0"))
	if d := f.Route(req3, fluidRouterState("i0")); d.TargetInstance != "i0" {
		t.Errorf("only routable instances may be chosen, got %s", d.TargetInstance)
	}
}

func TestFluidDual_RejectsWhenEveryIndexIsNegative(t *testing.T) {
	f := NewFluidDual(fluidTestParams(), 1)
	// type b: r 0.5 - c 0.2 - gammaM aM; with memory 3,000 tokens above target gammaM = 7, index = 0.3 - 0.7 < 0
	f.SetState(fluidTestState{"i0": {3500, 50}, "i1": {3500, 50}}.read)
	ok, reason := f.Admit(&Request{ID: "r", TenantID: "b"}, fluidRouterState("i0", "i1"))
	if ok || !strings.Contains(reason, "< 0") {
		t.Fatalf("expected a rejection, got ok=%v reason=%q", ok, reason)
	}
	if f.Rejected["b"] != 1 || f.Accepted["b"] != 0 {
		t.Errorf("rejection counters wrong: %v %v", f.Accepted, f.Rejected)
	}
}

func TestFluidDual_UnknownTypePanics(t *testing.T) {
	f := NewFluidDual(fluidTestParams(), 1)
	f.SetState(fluidTestState{"i0": {0, 0}, "i1": {0, 0}}.read)
	defer func() {
		if recover() == nil {
			t.Fatal("a type missing from the fluid solution must panic, not be silently routed")
		}
	}()
	f.Admit(&Request{ID: "r", TenantID: "zzz"}, fluidRouterState("i0", "i1"))
}

// Stage-0 bookkeeping: a job counts from routing until its prefill completes, again after an
// eviction, and never after it leaves.
func TestStage0Bookkeeping(t *testing.T) {
	s := &Simulator{KVCache: MustNewKVCacheState(10, 16)}
	r := &Request{ID: "r", InputTokens: make([]TokenID, 40)}
	s.enterStage0(r) // not tracked: no effect
	if s.Stage0PromptTokens() != 0 {
		t.Fatal("bookkeeping must be off by default")
	}
	s.TrackStage0()
	s.enterStage0(r)
	s.enterStage0(r) // idempotent
	if s.Stage0PromptTokens() != 40 || s.Stage0Requests() != 1 {
		t.Fatalf("after routing: %d tokens, %d jobs", s.Stage0PromptTokens(), s.Stage0Requests())
	}
	s.leaveStage0(r)
	s.leaveStage0(r)
	if s.Stage0PromptTokens() != 0 {
		t.Fatalf("after prefill: %d tokens", s.Stage0PromptTokens())
	}
}
