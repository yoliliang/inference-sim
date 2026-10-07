package sim

// ours: the fluid-dual policy of the model draft (Fluid Model/model_with_markovian_review.pdf,
// Section 4, Definition 1), admission and routing part. Batching and eviction (parts 2 and 3
// of the definition) are vLLM's default rules, which the simulator already executes: every
// prefilled job continues, stage-0 jobs join in arrival order while the token budget and the
// memory hold, and the most recently started job is evicted first.
//
// When a type-i job arrives at time t the policy computes, for every instance j,
//
//     I_ij(t) = r_i - c*_ij - gammaM_j(t) aM_ij - gammaB_j(t) aB_ij
//     gammaM_j(t) = [gammaM*_j + eta (K_j(t) - K*_j) / M_j]^+
//     gammaB_j(t) = [gammaB*_j + eta (S_j(t) - S*_j) / b_j]^+
//
// where K_j(t) is the KV memory in use on j and S_j(t) the prompt tokens of the stage-0 jobs
// assigned to j (waiting, mid-prefill, evicted, or still in the API delay after routing).
// It accepts the job iff max_j I_ij(t) >= 0 and dispatches it to an instance with the
// largest index, ties to the smallest instance index. All starred quantities come from the
// fluid solution written by ours/fluid/solve.py; eta = scale * eta0 with eta0 = V*/J.
//
// The policy only reads state and decides; it implements both AdmissionPolicy and
// RoutingPolicy because the draft makes one decision (accept and where) at arrival: Admit
// computes it and Route returns the instance Admit chose for that request.

import (
	"fmt"
	"math"
	"os"

	"gopkg.in/yaml.v3"
)

// FluidDualType is one job type of the fluid solution.
type FluidDualType struct {
	Name   string  `yaml:"name"`
	Prompt float64 `yaml:"prompt"`
	R      float64 `yaml:"r"`
	Nu     float64 `yaml:"nu"`
}

// FluidDualPair holds the per (type, instance) quantities.
type FluidDualPair struct {
	X  float64 `yaml:"x"`
	AM float64 `yaml:"aM"`
	AB float64 `yaml:"aB"`
	C  float64 `yaml:"c"`
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
	for _, inst := range p.Instances {
		if inst.M <= 0 || inst.B <= 0 {
			return nil, fmt.Errorf("fluid-dual parameters %s: instance %s needs M > 0 and b > 0", path, inst.ID)
		}
		for _, t := range p.Types {
			if _, ok := inst.Types[t.Name]; !ok {
				return nil, fmt.Errorf("fluid-dual parameters %s: instance %s has no entry for type %s", path, inst.ID, t.Name)
			}
		}
	}
	return &p, nil
}

// FluidDualState reports the live state of one instance: KV memory in use (tokens) and the
// prompt tokens of its stage-0 jobs. ok is false for an unknown instance.
type FluidDualState func(instanceID string) (kvTokens, stage0Prompt int64, ok bool)

// FluidDual is the policy object.
type FluidDual struct {
	p      *FluidDualParams
	eta    float64
	state  FluidDualState
	types  map[string]int
	chosen map[string]string // request id -> instance chosen at admission

	// diagnostics: decisions by type
	Accepted map[string]int64
	Rejected map[string]int64
}

// NewFluidDual builds the policy; eta = etaScale * eta0 of the parameter file.
func NewFluidDual(p *FluidDualParams, etaScale float64) *FluidDual {
	if etaScale < 0 || math.IsNaN(etaScale) {
		panic(fmt.Sprintf("fluid-dual: eta scale must be >= 0, got %v", etaScale))
	}
	f := &FluidDual{p: p, eta: etaScale * p.Meta.Eta0, types: map[string]int{}, chosen: map[string]string{},
		Accepted: map[string]int64{}, Rejected: map[string]int64{}}
	for i, t := range p.Types {
		f.types[t.Name] = i
	}
	return f
}

// SetState wires the live state reader (set by the cluster).
func (f *FluidDual) SetState(s FluidDualState) { f.state = s }

// Eta returns the responsiveness coefficient in use.
func (f *FluidDual) Eta() float64 { return f.eta }

// Params returns the parameter file the policy was built from.
func (f *FluidDual) Params() *FluidDualParams { return f.p }

// Index returns I_ij(t) of a type-name job on parameter instance k for the given live state.
func (f *FluidDual) Index(typeName string, k int, kvTokens, stage0Prompt int64) float64 {
	ti, ok := f.types[typeName]
	if !ok {
		panic(fmt.Sprintf("fluid-dual: request type %q is not in the fluid solution", typeName))
	}
	inst := &f.p.Instances[k]
	pair := inst.Types[typeName]
	gm := math.Max(0, inst.GammaM+f.eta*(float64(kvTokens)-inst.KVTarget)/inst.M)
	gb := math.Max(0, inst.GammaB+f.eta*(float64(stage0Prompt)-inst.Stage0Target)/inst.B)
	return f.p.Types[ti].R - pair.C - gm*pair.AM - gb*pair.AB
}

// best returns the instance with the largest index among the routable ones (parameter
// order, which is instance-index order, so ties go to the smallest index) and that index.
func (f *FluidDual) best(req *Request, state *RouterState) (string, float64) {
	if f.state == nil {
		panic("fluid-dual: no state reader wired")
	}
	routable := make(map[string]bool, len(state.Snapshots))
	for _, s := range state.Snapshots {
		routable[s.ID] = true
	}
	bestID, bestIdx := "", math.Inf(-1)
	for k := range f.p.Instances {
		id := f.p.Instances[k].ID
		if !routable[id] {
			continue
		}
		kv, s0, ok := f.state(id)
		if !ok {
			panic(fmt.Sprintf("fluid-dual: instance %s of the fluid solution does not exist in the cluster", id))
		}
		if idx := f.Index(req.TenantID, k, kv, s0); idx > bestIdx {
			bestID, bestIdx = id, idx
		}
	}
	return bestID, bestIdx
}

// Admit accepts the job iff the largest index is nonnegative (within a relative rounding
// tolerance) and remembers the chosen instance for Route.
func (f *FluidDual) Admit(req *Request, state *RouterState) (bool, string) {
	id, idx := f.best(req, state)
	r := f.p.Types[f.types[req.TenantID]].R
	if id == "" || idx < -1e-9*math.Max(1, math.Abs(r)) {
		f.Rejected[req.TenantID]++
		return false, fmt.Sprintf("fluid-dual: max index %.6g < 0", idx)
	}
	f.Accepted[req.TenantID]++
	f.chosen[req.ID] = id
	return true, fmt.Sprintf("fluid-dual: index %.6g on %s", idx, id)
}

// Route returns the instance chosen at admission; if the job was admitted by another
// policy it routes to the largest index.
func (f *FluidDual) Route(req *Request, state *RouterState) RoutingDecision {
	id, ok := f.chosen[req.ID]
	if ok {
		delete(f.chosen, req.ID)
	} else {
		id, _ = f.best(req, state)
	}
	return NewRoutingDecision(id, "fluid-dual")
}
