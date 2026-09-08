package cmd

// ours: periodic instance-state sampling for blis run.
//
// --state-sample-interval N (microseconds, 0 = off) attaches a read-only
// sim.ProgressHook to the cluster simulator and writes one CSV row per instance
// every N microseconds of simulated time, plus a final row set at the end of the
// run. The hook only reads state (see sim/progress_hook.go), so stdout stays
// byte-identical (INV-6). Output path is --state-sample-path, defaulting to
// <metrics-path without .json>_state.csv when --metrics-path is set.

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

var (
	stateSampleInterval int64  // ours: --state-sample-interval, microseconds, 0 = off
	stateSamplePath     string // ours: --state-sample-path
)

var stateSampleHeader = []string{
	"clock_us", "instance", "queue_depth", "batch_size",
	"kv_used_blocks", "kv_total_blocks", "kv_utilization",
	"preemptions", "completed", "in_flight",
	"cluster_completed", "cluster_rejected", "cluster_preemptions", "is_final",
}

type stateSampler struct {
	f *os.File
	w *csv.Writer
}

func newStateSampler(path string) (*stateSampler, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := csv.NewWriter(f)
	if err := w.Write(stateSampleHeader); err != nil {
		f.Close()
		return nil, err
	}
	return &stateSampler{f: f, w: w}, nil
}

func (s *stateSampler) OnProgress(snap sim.ProgressSnapshot) {
	for _, inst := range snap.InstanceSnapshots {
		row := []string{
			strconv.FormatInt(snap.Clock, 10),
			inst.ID,
			strconv.Itoa(inst.QueueDepth),
			strconv.Itoa(inst.BatchSize),
			strconv.FormatInt(inst.KVTotalBlocks-inst.KVFreeBlocks, 10),
			strconv.FormatInt(inst.KVTotalBlocks, 10),
			strconv.FormatFloat(inst.KVUtilization, 'f', 4, 64),
			strconv.FormatInt(inst.PreemptionCount, 10),
			strconv.Itoa(inst.CompletedRequests),
			strconv.Itoa(inst.InFlightRequests),
			strconv.Itoa(snap.TotalCompleted),
			strconv.Itoa(snap.RejectedRequests),
			strconv.FormatInt(snap.TotalPreemptions, 10),
			strconv.FormatBool(snap.IsFinal),
		}
		if err := s.w.Write(row); err != nil {
			logrus.Errorf("state sample write failed: %v", err)
		}
	}
}

func (s *stateSampler) Close() error {
	s.w.Flush()
	if err := s.w.Error(); err != nil {
		s.f.Close()
		return err
	}
	return s.f.Close()
}

// resolveStateSamplePath returns the CSV path for state sampling, or an error
// when sampling is on but no path can be derived.
func resolveStateSamplePath(interval int64, explicit, metrics string) (string, error) {
	if interval <= 0 {
		if explicit != "" {
			logrus.Warnf("--state-sample-path has no effect without --state-sample-interval > 0")
		}
		return "", nil
	}
	if explicit != "" {
		return explicit, nil
	}
	if metrics == "" {
		return "", fmt.Errorf("--state-sample-interval requires --state-sample-path or --metrics-path")
	}
	return strings.TrimSuffix(metrics, ".json") + "_state.csv", nil
}
