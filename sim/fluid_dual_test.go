package sim

import (
	"math"
	"math/rand"
	"testing"
)

// ours: tests of the fluid-dual admission and routing rule (model draft, Section 4,
// eq. 32 to 34 and Definition 1).

// Two types, two instances. Type a: lambda 4, fluid split 3 / 1. Type b: lambda 2, half
// admitted (x* = 0.5 on each instance). Capacity is ample at x*.
func fluidTestParams() *FluidDualParams {
	p := &FluidDualParams{}
	p.Meta.Eta0 = 2.0
	p.Types = []FluidDualType{{Name: "a", Prompt: 100, Lambda: 4, R: 1.0}, {Name: "b", Prompt: 1000, Lambda: 2, R: 0.5}}
	mk := func(xa, xb float64) map[string]FluidDualPair {
		return map[string]FluidDualPair{
			"a": {X: xa, AM: 0.1, AB: 0.05, C: 0.2, Index0: 1.0 - 0.2 - 1*0.1},
			"b": {X: xb, AM: 0.2, AB: 0.10, C: 0.1, Index0: 0.5 - 0.1 - 1*0.2},
		}
	}
	p.Instances = []FluidDualInstance{
		{ID: "i0", M: 1000, B: 100, GammaM: 1, KVTarget: 500, Stage0Target: 50, Types: mk(3, 0.5)},
		{ID: "i1", M: 1000, B: 100, GammaM: 1, KVTarget: 500, Stage0Target: 50, Types: mk(1, 0.5)},
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

func newTestFluid(t *testing.T, st fluidTestState) *FluidDual {
	t.Helper()
	f := NewFluidDual(fluidTestParams(), FluidDualOptions{EtaScale: 1, KScale: 1})
	f.SetState(st.read)
	return f
}

func atTarget() fluidTestState { return fluidTestState{"i0": {500, 50}, "i1": {500, 50}} }

func TestFluidDual_IndexAtZeroDeviationIsIStar(t *testing.T) {
	f := newTestFluid(t, atTarget())
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if got, want := f.Index(i, j, 500, 50), f.i0[i][j]; math.Abs(got-want) > 1e-12 {
				t.Errorf("I(%d,%d) at the fluid levels: got %v want I* %v", i, j, got, want)
			}
		}
	}
	// 300 memory tokens above target: gammaM = 1 + eta * 0.3 = 1.6 (eta = 2)
	if got, want := f.Index(0, 0, 800, 50), 1.0-0.2-1.6*0.1; math.Abs(got-want) > 1e-12 {
		t.Errorf("memory above target: got %v want %v", got, want)
	}
	// far below: the price is clipped at zero
	if got, want := f.Index(0, 0, -400, 50), 1.0-0.2; math.Abs(got-want) > 1e-12 {
		t.Errorf("clipped price: got %v want %v", got, want)
	}
}

func TestFluidDual_ZeroDeviationGivesFluidRates(t *testing.T) {
	f := newTestFluid(t, atTarget())
	f.desiredRates(fluidRouterState("i0", "i1"))
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(f.y[i][j]-f.xs[i][j]) > 1e-9 {
				t.Errorf("y(%d,%d) = %v, want x* = %v", i, j, f.y[i][j], f.xs[i][j])
			}
		}
	}
}

// The projection must be the exact weighted projection onto F: check the KKT conditions of
// the quadratic program on random provisional rates, many of them infeasible.
func TestFluidDual_ProjectionSatisfiesKKT(t *testing.T) {
	f := newTestFluid(t, atTarget())
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				f.ytil[i][j] = rng.Float64() * 12
			}
		}
		f.project()
		// primal feasibility
		if v := f.violation(); v > 1e-9 {
			t.Fatalf("trial %d: constraint violation %v", trial, v)
		}
		// stationarity: y = max(0, ytil - (muF + muM aM + muB aB)/omega), multipliers >= 0
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				want := math.Max(0, f.ytil[i][j]-(f.muF[i]+f.muM[j]*f.aM[i][j]+f.muB[j]*f.aB[i][j])/f.w[i][j])
				if math.Abs(f.y[i][j]-want) > 1e-9 {
					t.Fatalf("trial %d: stationarity at (%d,%d)", trial, i, j)
				}
			}
		}
		// complementary slackness
		for i := 0; i < 2; i++ {
			s := f.y[i][0] + f.y[i][1]
			if f.muF[i] > 1e-9 && math.Abs(s-f.lam[i]) > 1e-7 {
				t.Fatalf("trial %d: flow %d slack %v with multiplier %v", trial, i, f.lam[i]-s, f.muF[i])
			}
		}
		for j := 0; j < 2; j++ {
			sm := f.aM[0][j]*f.y[0][j] + f.aM[1][j]*f.y[1][j]
			sb := f.aB[0][j]*f.y[0][j] + f.aB[1][j]*f.y[1][j]
			if (f.muM[j] > 1e-9 && math.Abs(sm-1) > 1e-7) || (f.muB[j] > 1e-9 && math.Abs(sb-1) > 1e-7) {
				t.Fatalf("trial %d: capacity of %d slack with a positive multiplier", trial, j)
			}
		}
	}
}

// At the fluid levels type b has alpha = 1/2: the accumulator accepts every second arrival,
// so the accepted count after n arrivals is floor(n/2).
func TestFluidDual_AdmissionAccumulator(t *testing.T) {
	f := newTestFluid(t, atTarget())
	accepted := 0
	for n := 1; n <= 10; n++ {
		ok, _ := f.Admit(&Request{ID: "b" + string(rune('0'+n)), TenantID: "b"}, fluidRouterState("i0", "i1"))
		if ok {
			accepted++
		}
		if accepted != n/2 {
			t.Fatalf("after %d arrivals accepted %d, want %d", n, accepted, n/2)
		}
	}
}

// Type a is fully admitted (alpha = 1) and split 3 : 1. The routing deficits reproduce the
// shares: in every four consecutive arrivals three go to i0 and one to i1.
func TestFluidDual_RoutingDeficitsTrackShares(t *testing.T) {
	f := newTestFluid(t, atTarget())
	count := map[string]int{}
	for n := 0; n < 40; n++ {
		req := &Request{ID: "a" + string(rune('A'+n)), TenantID: "a"}
		if ok, _ := f.Admit(req, fluidRouterState("i0", "i1")); !ok {
			t.Fatalf("arrival %d rejected with alpha = 1", n)
		}
		count[f.Route(req, fluidRouterState("i0", "i1")).TargetInstance]++
	}
	if count["i0"] != 30 || count["i1"] != 10 {
		t.Fatalf("dispatch counts %v, want 30 and 10", count)
	}
}

// When i0 holds far more memory than its fluid level its price rises and its index falls:
// gammaM = 1 + 2 * 0.4 = 1.8, I - I* = -0.08, k = lambda/r = 4, so the provisional rate of
// type a on i0 drops from 3 to 2.68. The flow constraint no longer binds, so the desired
// rates are the provisional ones: type a is admitted at 3.68 / 4 and i1's share rises.
func TestFluidDual_CongestionShiftsRates(t *testing.T) {
	st := atTarget()
	f := newTestFluid(t, st)
	st["i0"] = [2]int64{900, 50}
	f.desiredRates(fluidRouterState("i0", "i1"))
	if math.Abs(f.y[0][0]-2.68) > 1e-9 || math.Abs(f.y[0][1]-1) > 1e-9 {
		t.Fatalf("type a rates %v, want [2.68 1]", f.y[0])
	}
}

func TestFluidDual_UnknownTypePanics(t *testing.T) {
	f := newTestFluid(t, atTarget())
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
