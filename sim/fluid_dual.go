package sim

// ours: the fluid-dual policy of the model draft, Section 4, Definition 1, admission and
// routing part. Two versions of the definition exist and both are implemented:
//
//   mode "index" (default; draft model_with_markovian_review_original_policy.pdf, Oct 2026):
//     when a type-i job arrives, compute I_ij(t) for every instance j; accept iff
//     max_j I_ij(t) >= 0 and dispatch to an instance with the largest index, ties to the
//     smallest instance index. No admission fractions or routing shares are enforced.
//   mode "rate-tracking" (the intermediate draft): the procedure described below
//     (provisional rates, projection, admission accumulator, routing deficits).
//
// Both use the same state prices and indices. Batching and eviction (parts 2 and 3
// of the definition) are vLLM's default rules, which the simulator already executes: every
// prefilled job continues, stage-0 jobs join in arrival order while the token budget and the
// memory hold, and the most recently started job is evicted first.
//
// At every arrival (any type) the policy
//
//  1. prices every instance from its live state (eq. 32 and 33):
//     gammaM_j(t) = [gammaM*_j + eta (K_j(t) - K*_j) / M_j]^+
//     gammaB_j(t) = [gammaB*_j + eta (S_j(t) - S*_j) / b_j]^+
//     K_j(t) = KV memory in use, S_j(t) = prompt tokens of the stage-0 jobs assigned to j
//     (waiting, mid-prefill, evicted, or still in the API delay after routing);
//  2. forms the adjusted marginal indices I_ij(t) = r_i - c*_ij - gammaM_j(t) aM_ij - gammaB_j(t) aB_ij
//     and the provisional rates ytilde_ij = [x*_ij + k_ij (I_ij(t) - I*_ij)]^+ (eq. 34);
//  3. projects ytilde onto the feasible rate set F of problem (28) in the weighted norm
//     sum omega_ij (z_ij - ytilde_ij)^2, giving the desired rates y(t);
//  4. for the arriving type i: f_i += a_i / lambda_i with a_i = sum_j y_ij; reject if f_i < 1,
//     else accept, f_i -= 1, add the shares y_ij / a_i to the routing deficits d_ij and
//     dispatch to the instance with the largest deficit (ties to the smallest index), whose
//     deficit then drops by one.
//
// Starred quantities, lambda_i and I*_ij come from the fluid solution written by
// ours/fluid/solve.py. eta = etaScale * eta0 with eta0 = V*/J (the user's choice). The draft
// leaves k_ij > 0 and omega_ij > 0 open; the defaults here are k_ij = kScale * lambda_i / r_i
// (an index change of one reward r_i moves the provisional rate by lambda_i) and
// omega_ij = 1 / lambda_i (each type's rate deviations measured relative to its arrival rate).
//
// The projection is a strictly convex quadratic program with one constraint per type (flow)
// and two per instance (memory, batch tokens), all with positive coefficients. It is solved
// exactly by dual coordinate ascent (Hildreth's method): each step maximises the dual over
// one multiplier in closed form, and the multipliers of the previous arrival are the warm
// start.
//
// The policy only reads state and decides; it implements both AdmissionPolicy and
// RoutingPolicy because the draft makes one decision (accept and where) at arrival: Admit
// computes it and Route returns the instance Admit chose for that request.

import (
	"fmt"
	"math"
	"os"
	"sort"

	"gopkg.in/yaml.v3"
)

// FluidDualType is one job type of the fluid solution.
type FluidDualType struct {
	Name   string  `yaml:"name"`
	Prompt float64 `yaml:"prompt"`
	Lambda float64 `yaml:"lambda"`
	R      float64 `yaml:"r"`
	Nu     float64 `yaml:"nu"`
}

// FluidDualPair holds the per (type, instance) quantities.
type FluidDualPair struct {
	X      float64 `yaml:"x"`
	AM     float64 `yaml:"aM"`
	AB     float64 `yaml:"aB"`
	C      float64 `yaml:"c"`
	Index0 float64 `yaml:"index0"` // I*_ij = r_i - c*_ij - gammaM*_j aM_ij - gammaB*_j aB_ij
}

// FluidDualInstance holds the per-instance quantities.
type FluidDualInstance struct {
	ID           string                   `yaml:"id"`
	GPU          string                   `yaml:"gpu"`
	M            float64                  `yaml:"M"`
	B            float64                  `yaml:"b"`
	GammaM       float64                  `yaml:"gammaM"`
	GammaB       float64                  `yaml:"gammaB"`
	KVTarget     float64                  `yaml:"kv_target"`
	Stage0Target float64                  `yaml:"stage0_target"`
	Types        map[string]FluidDualPair `yaml:"types"`
}

// FluidDualParams is the parameter file written by ours/fluid/solve.py. Fields the policy
// does not use are ignored.
type FluidDualParams struct {
	Meta struct {
		Value float64 `yaml:"value"`
		Eta0  float64 `yaml:"eta0"`
	} `yaml:"meta"`
	Types     []FluidDualType     `yaml:"types"`
	Instances []FluidDualInstance `yaml:"instances"`
}

// LoadFluidDualParams reads and checks a parameter file.
func LoadFluidDualParams(path string) (*FluidDualParams, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p FluidDualParams
	if err := yaml.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("fluid-dual parameters %s: %w", path, err)
	}
	if len(p.Types) == 0 || len(p.Instances) == 0 {
		return nil, fmt.Errorf("fluid-dual parameters %s: no types or no instances", path)
	}
	for _, t := range p.Types {
		if t.Lambda < 0 || t.R <= 0 {
			return nil, fmt.Errorf("fluid-dual parameters %s: type %s needs lambda >= 0 and r > 0", path, t.Name)
		}
	}
	for _, inst := range p.Instances {
		if inst.M <= 0 || inst.B <= 0 {
			return nil, fmt.Errorf("fluid-dual parameters %s: instance %s needs M > 0 and b > 0", path, inst.ID)
		}
		for _, t := range p.Types {
			pr, ok := inst.Types[t.Name]
			if !ok {
				return nil, fmt.Errorf("fluid-dual parameters %s: instance %s has no entry for type %s", path, inst.ID, t.Name)
			}
			if pr.AM <= 0 || pr.AB <= 0 {
				return nil, fmt.Errorf("fluid-dual parameters %s: instance %s type %s needs aM > 0 and aB > 0", path, inst.ID, t.Name)
			}
		}
	}
	return &p, nil
}

// FluidDualState reports the live state of one instance: KV memory in use (tokens) and the
// prompt tokens of its stage-0 jobs. ok is false for an unknown instance.
type FluidDualState func(instanceID string) (kvTokens, stage0Prompt int64, ok bool)

// FluidDualOptions are the tuning constants of the policy.
type FluidDualOptions struct {
	Mode     string  // "index" (default) or "rate-tracking"
	EtaScale float64 // eta = EtaScale * eta0
	KScale   float64 // k_ij = KScale * lambda_i / r_i
	Weights  string  // projection weights: "inverse-rate" (omega_ij = 1/lambda_i, default) or "uniform" (omega_ij = 1)
}

// FluidDual is the policy object.
type FluidDual struct {
	p       *FluidDualParams
	mode    string
	eta     float64
	state   FluidDualState
	types   map[string]int
	chosen  map[string]string // request id -> instance chosen at admission
	nI, nJ  int
	k, w    [][]float64 // k_ij, omega_ij
	xs, i0  [][]float64 // x*_ij, I*_ij
	aM, aB  [][]float64
	lam, r  []float64
	ytil, y [][]float64
	muF     []float64 // warm-start multipliers: flow (per type)
	muM     []float64 // memory (per instance)
	muB     []float64 // batch tokens (per instance)
	f       []float64 // admission accumulators
	d       [][]float64
	live    []bool
	scratch struct{ u, a, w []float64 }

	// diagnostics
	Accepted   map[string]int64
	Rejected   map[string]int64
	Dispatched map[string]map[string]int64 // type -> instance -> count
	ProjCalls  int64
	ProjSweeps int64
	ProjMaxGap float64 // largest relative constraint violation left by the projection
}

// NewFluidDual builds the policy.
func NewFluidDual(p *FluidDualParams, o FluidDualOptions) *FluidDual {
	if o.EtaScale < 0 || math.IsNaN(o.EtaScale) || o.KScale < 0 || math.IsNaN(o.KScale) {
		panic(fmt.Sprintf("fluid-dual: need eta scale >= 0 and k scale >= 0 (0 = default 1), got %v and %v", o.EtaScale, o.KScale))
	}
	if o.Weights != "" && o.Weights != "inverse-rate" && o.Weights != "uniform" {
		panic(fmt.Sprintf("fluid-dual: unknown projection weights %q", o.Weights))
	}
	nI, nJ := len(p.Types), len(p.Instances)
	if o.Mode == "" {
		o.Mode = "index"
	}
	if o.Mode != "index" && o.Mode != "rate-tracking" {
		panic(fmt.Sprintf("fluid-dual: unknown mode %q (index, rate-tracking)", o.Mode))
	}
	if o.KScale == 0 {
		o.KScale = 1
	}
	f := &FluidDual{p: p, mode: o.Mode, eta: o.EtaScale * p.Meta.Eta0, types: map[string]int{}, chosen: map[string]string{},
		nI: nI, nJ: nJ, lam: make([]float64, nI), r: make([]float64, nI),
		muF: make([]float64, nI), muM: make([]float64, nJ), muB: make([]float64, nJ), f: make([]float64, nI),
		live: make([]bool, nJ), Accepted: map[string]int64{}, Rejected: map[string]int64{},
		Dispatched: map[string]map[string]int64{}}
	mat := func() [][]float64 {
		m := make([][]float64, nI)
		for i := range m {
			m[i] = make([]float64, nJ)
		}
		return m
	}
	f.k, f.w, f.xs, f.i0, f.aM, f.aB, f.ytil, f.y, f.d = mat(), mat(), mat(), mat(), mat(), mat(), mat(), mat(), mat()
	for i, t := range p.Types {
		f.types[t.Name] = i
		f.lam[i], f.r[i] = t.Lambda, t.R
		f.Dispatched[t.Name] = map[string]int64{}
		for j, inst := range p.Instances {
			pr := inst.Types[t.Name]
			f.xs[i][j], f.i0[i][j], f.aM[i][j], f.aB[i][j] = pr.X, pr.Index0, pr.AM, pr.AB
			f.k[i][j] = o.KScale * t.Lambda / t.R
			f.w[i][j] = 1.0
			if o.Weights != "uniform" {
				f.w[i][j] = 1.0 / math.Max(t.Lambda, 1e-12)
			}
		}
	}
	n := max(nI, nJ)
	f.scratch.u, f.scratch.a, f.scratch.w = make([]float64, n), make([]float64, n), make([]float64, n)
	return f
}

// SetState wires the live state reader (set by the cluster).
func (f *FluidDual) SetState(s FluidDualState) { f.state = s }

// Eta returns the responsiveness coefficient in use.
func (f *FluidDual) Eta() float64 { return f.eta }

// Params returns the parameter file the policy was built from.
func (f *FluidDual) Params() *FluidDualParams { return f.p }

// Index returns I_ij(t) of type i on parameter instance j for the given live state.
func (f *FluidDual) Index(i, j int, kvTokens, stage0Prompt int64) float64 {
	inst := &f.p.Instances[j]
	gm := math.Max(0, inst.GammaM+f.eta*(float64(kvTokens)-inst.KVTarget)/inst.M)
	gb := math.Max(0, inst.GammaB+f.eta*(float64(stage0Prompt)-inst.Stage0Target)/inst.B)
	return f.r[i] - inst.Types[f.p.Types[i].Name].C - gm*f.aM[i][j] - gb*f.aB[i][j]
}

// desiredRates computes y(t) from the live state (steps 1 to 3). Instances that are not
// routable get provisional rate zero.
func (f *FluidDual) desiredRates(state *RouterState) {
	if f.state == nil {
		panic("fluid-dual: no state reader wired")
	}
	routable := make(map[string]bool, len(state.Snapshots))
	for _, s := range state.Snapshots {
		routable[s.ID] = true
	}
	for j := range f.p.Instances {
		id := f.p.Instances[j].ID
		f.live[j] = routable[id]
		kv, s0, ok := f.state(id)
		if !ok {
			panic(fmt.Sprintf("fluid-dual: instance %s of the fluid solution does not exist in the cluster", id))
		}
		for i := 0; i < f.nI; i++ {
			if !f.live[j] {
				f.ytil[i][j] = 0
				continue
			}
			f.ytil[i][j] = math.Max(0, f.xs[i][j]+f.k[i][j]*(f.Index(i, j, kv, s0)-f.i0[i][j]))
		}
	}
	f.project()
}

// project sets y to the weighted projection of ytil onto F (Hildreth's method, warm start).
func (f *FluidDual) project() {
	f.ProjCalls++
	const maxSweeps = 2000
	for sweep := 0; sweep < maxSweeps; sweep++ {
		f.ProjSweeps++
		change := 0.0
		for i := 0; i < f.nI; i++ { // flow_i: sum_j z_ij <= lambda_i
			u, a, w := f.scratch.u[:f.nJ], f.scratch.a[:f.nJ], f.scratch.w[:f.nJ]
			for j := 0; j < f.nJ; j++ {
				u[j] = f.ytil[i][j] - (f.muM[j]*f.aM[i][j]+f.muB[j]*f.aB[i][j])/f.w[i][j]
				a[j], w[j] = 1, f.w[i][j]
			}
			m := f.solve1D(u, a, w, f.lam[i])
			change = math.Max(change, math.Abs(m-f.muF[i])/(1+m))
			f.muF[i] = m
		}
		for j := 0; j < f.nJ; j++ { // mem_j and tok_j: sum_i a_ij z_ij <= 1
			u, a, w := f.scratch.u[:f.nI], f.scratch.a[:f.nI], f.scratch.w[:f.nI]
			for i := 0; i < f.nI; i++ {
				u[i] = f.ytil[i][j] - (f.muF[i]+f.muB[j]*f.aB[i][j])/f.w[i][j]
				a[i], w[i] = f.aM[i][j], f.w[i][j]
			}
			m := f.solve1D(u, a, w, 1)
			change = math.Max(change, math.Abs(m-f.muM[j])/(1+m))
			f.muM[j] = m
			for i := 0; i < f.nI; i++ {
				u[i] = f.ytil[i][j] - (f.muF[i]+f.muM[j]*f.aM[i][j])/f.w[i][j]
				a[i], w[i] = f.aB[i][j], f.w[i][j]
			}
			m = f.solve1D(u, a, w, 1)
			change = math.Max(change, math.Abs(m-f.muB[j])/(1+m))
			f.muB[j] = m
		}
		if change < 1e-13 {
			break
		}
	}
	for i := 0; i < f.nI; i++ {
		for j := 0; j < f.nJ; j++ {
			f.y[i][j] = math.Max(0, f.ytil[i][j]-(f.muF[i]+f.muM[j]*f.aM[i][j]+f.muB[j]*f.aB[i][j])/f.w[i][j])
		}
	}
	f.ProjMaxGap = math.Max(f.ProjMaxGap, f.violation())
}

// violation returns the largest relative constraint violation of y.
func (f *FluidDual) violation() float64 {
	v := 0.0
	for i := 0; i < f.nI; i++ {
		s := 0.0
		for j := 0; j < f.nJ; j++ {
			s += f.y[i][j]
		}
		v = math.Max(v, (s-f.lam[i])/math.Max(f.lam[i], 1e-12))
	}
	for j := 0; j < f.nJ; j++ {
		sm, sb := 0.0, 0.0
		for i := 0; i < f.nI; i++ {
			sm += f.aM[i][j] * f.y[i][j]
			sb += f.aB[i][j] * f.y[i][j]
		}
		v = math.Max(v, math.Max(sm-1, sb-1))
	}
	return math.Max(v, 0)
}

// solve1D returns the multiplier mu >= 0 of one constraint sum_e a_e z_e <= rhs, where
// z_e = max(0, u_e - mu a_e / w_e): zero if the constraint holds at mu = 0, otherwise the
// root of the decreasing piecewise-linear function g(mu) = rhs.
func (f *FluidDual) solve1D(u, a, w []float64, rhs float64) float64 {
	g, slope := 0.0, 0.0
	for e := range u {
		if u[e] > 0 {
			g += a[e] * u[e]
			slope += a[e] * a[e] / w[e]
		}
	}
	if g <= rhs {
		return 0
	}
	// walk the breakpoints in increasing order; between them g falls with slope -slope
	order := make([]int, 0, len(u))
	for e := range u {
		if u[e] > 0 {
			order = append(order, e)
		}
	}
	sort.Slice(order, func(x, y int) bool {
		return u[order[x]]*w[order[x]]/a[order[x]] < u[order[y]]*w[order[y]]/a[order[y]]
	})
	mu := 0.0
	for _, e := range order {
		te := u[e] * w[e] / a[e]
		gt := g - slope*(te-mu)
		if gt <= rhs {
			return mu + (g-rhs)/slope
		}
		g, mu = gt, te
		slope -= a[e] * a[e] / w[e]
	}
	return mu // g reached zero before rhs: only possible for rhs <= 0
}

// FluidDualReport summarises a run of the policy (whole run, not only the window).
type FluidDualReport struct {
	Mode         string                      `json:"mode"`
	Eta          float64                     `json:"eta"`
	Accepted     map[string]int64            `json:"accepted"`
	Rejected     map[string]int64            `json:"rejected"`
	Dispatched   map[string]map[string]int64 `json:"dispatched"` // type -> instance -> count
	Projections  int64                       `json:"projections"`
	MeanSweeps   float64                     `json:"mean_sweeps"`
	MaxViolation float64                     `json:"max_violation"` // largest relative constraint violation left by a projection
	FluidValue   float64                     `json:"fluid_value"`   // V* of the parameter file
}

// Report returns the run diagnostics.
func (f *FluidDual) Report() *FluidDualReport {
	r := &FluidDualReport{Mode: f.mode, Eta: f.eta, Accepted: f.Accepted, Rejected: f.Rejected, Dispatched: f.Dispatched,
		Projections: f.ProjCalls, MaxViolation: f.ProjMaxGap, FluidValue: f.p.Meta.Value}
	if f.ProjCalls > 0 {
		r.MeanSweeps = float64(f.ProjSweeps) / float64(f.ProjCalls)
	}
	return r
}

// Admit applies the admission accumulator and, for an accepted job, the routing deficits.
func (f *FluidDual) Admit(req *Request, state *RouterState) (bool, string) {
	i, ok := f.types[req.TenantID]
	if !ok {
		panic(fmt.Sprintf("fluid-dual: request type %q is not in the fluid solution", req.TenantID))
	}
	if f.mode == "index" {
		return f.admitByIndex(i, req, state)
	}
	f.desiredRates(state)
	ai := 0.0
	for j := 0; j < f.nJ; j++ {
		ai += f.y[i][j]
	}
	alpha := 0.0
	if f.lam[i] > 0 {
		alpha = math.Min(1, ai/f.lam[i])
	}
	f.f[i] += alpha
	if f.f[i] < 1-1e-9 || ai <= 0 {
		f.Rejected[req.TenantID]++
		return false, fmt.Sprintf("fluid-dual: accumulator %.4f < 1 (alpha %.4f)", f.f[i], alpha)
	}
	f.f[i] -= 1
	best, bestD := -1, math.Inf(-1)
	for j := 0; j < f.nJ; j++ {
		f.d[i][j] += f.y[i][j] / ai
		if f.live[j] && f.d[i][j] > bestD {
			best, bestD = j, f.d[i][j]
		}
	}
	if best < 0 {
		f.Rejected[req.TenantID]++
		return false, "fluid-dual: no routable instance"
	}
	f.d[i][best] -= 1
	id := f.p.Instances[best].ID
	f.Accepted[req.TenantID]++
	f.Dispatched[req.TenantID][id]++
	f.chosen[req.ID] = id
	return true, fmt.Sprintf("fluid-dual: alpha %.4f, routed to %s", alpha, id)
}

// admitByIndex is Definition 1 of the current draft: accept iff the largest index is
// nonnegative (within a relative rounding tolerance), dispatch to the largest index, ties to
// the smallest instance index (parameter order is instance-index order).
func (f *FluidDual) admitByIndex(i int, req *Request, state *RouterState) (bool, string) {
	if f.state == nil {
		panic("fluid-dual: no state reader wired")
	}
	routable := make(map[string]bool, len(state.Snapshots))
	for _, s := range state.Snapshots {
		routable[s.ID] = true
	}
	best, bestIdx := -1, math.Inf(-1)
	for j := range f.p.Instances {
		id := f.p.Instances[j].ID
		if !routable[id] {
			continue
		}
		kv, s0, ok := f.state(id)
		if !ok {
			panic(fmt.Sprintf("fluid-dual: instance %s of the fluid solution does not exist in the cluster", id))
		}
		if idx := f.Index(i, j, kv, s0); idx > bestIdx {
			best, bestIdx = j, idx
		}
	}
	if best < 0 || bestIdx < -1e-9*math.Max(1, math.Abs(f.r[i])) {
		f.Rejected[req.TenantID]++
		return false, fmt.Sprintf("fluid-dual: max index %.6g < 0", bestIdx)
	}
	id := f.p.Instances[best].ID
	f.Accepted[req.TenantID]++
	f.Dispatched[req.TenantID][id]++
	f.chosen[req.ID] = id
	return true, fmt.Sprintf("fluid-dual: index %.6g on %s", bestIdx, id)
}

// Route returns the instance chosen at admission. A job admitted by another policy is sent
// to the instance with the largest desired rate for its type.
func (f *FluidDual) Route(req *Request, state *RouterState) RoutingDecision {
	if id, ok := f.chosen[req.ID]; ok {
		delete(f.chosen, req.ID)
		return NewRoutingDecision(id, "fluid-dual")
	}
	i, ok := f.types[req.TenantID]
	if !ok {
		panic(fmt.Sprintf("fluid-dual: request type %q is not in the fluid solution", req.TenantID))
	}
	f.desiredRates(state)
	best, bestY := -1, math.Inf(-1)
	for j := 0; j < f.nJ; j++ {
		if f.live[j] && f.y[i][j] > bestY {
			best, bestY = j, f.y[i][j]
		}
	}
	return NewRoutingDecision(f.p.Instances[best].ID, "fluid-dual (not admitted by fluid-dual)")
}
