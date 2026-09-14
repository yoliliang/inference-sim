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
	windowStartS          float64 // ours: --window-start, seconds
	windowEndS            float64 // ours: --window-end, seconds (0 = no window block)
	dropPerRequestOutput  bool    // ours: --drop-per-request-output
	releaseCompleted      bool    // ours: --release-completed-requests
	stateSampleInterval   int64   // ours: --state-sample-interval, microseconds, 0 = off
	stateSamplePath       string  // ours: --state-sample-path
	stateSnapshotInterval int64   // ours: --state-snapshot-interval, microseconds, 0 = off
)

var stateSnapshotHeader = []string{
	"clock_us", "instance", "requestID", "tenant_id", "state", "input_tokens", "progress_tokens",
}

var stateSampleHeader = []string{
	"clock_us", "instance", "queue_depth", "batch_size",
	"kv_used_blocks", "kv_total_blocks", "kv_utilization",
	"preemptions", "completed", "in_flight",
	"cluster_completed", "cluster_rejected", "cluster_preemptions", "is_final",
}

type stateSampler struct {
	f                *os.File // state rows (nil when state sampling is off)
	w                *csv.Writer
	sf               *os.File // snapshot rows (nil when snapshots are off)
	sw               *csv.Writer
	snapshotInterval int64
	nextSnapshotUs   int64
}

// newStateSampler opens the state csv (statePath, may be empty) and the snapshot csv
// (snapshotPath, may be empty). At least one must be given.
func newStateSampler(statePath, snapshotPath string, snapshotInterval int64) (*stateSampler, error) {
	s := &stateSampler{snapshotInterval: snapshotInterval, nextSnapshotUs: snapshotInterval}
	if statePath != "" {
		f, err := os.Create(statePath)
		if err != nil {
			return nil, err
		}
		s.f, s.w = f, csv.NewWriter(f)
		if err := s.w.Write(stateSampleHeader); err != nil {
			return nil, err
		}
	}
	if snapshotPath != "" {
		f, err := os.Create(snapshotPath)
		if err != nil {
			return nil, err
		}
		s.sf, s.sw = f, csv.NewWriter(f)
		if err := s.sw.Write(stateSnapshotHeader); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *stateSampler) OnProgress(snap sim.ProgressSnapshot) {
	if s.sw != nil && (snap.IsFinal || snap.Clock >= s.nextSnapshotUs) {
		for _, inst := range snap.InstanceSnapshots {
			for _, r := range inst.Requests {
				row := []string{strconv.FormatInt(snap.Clock, 10), inst.ID, r.ID, r.TenantID, r.State,
					strconv.FormatInt(r.InputTokens, 10), strconv.FormatInt(r.ProgressTokens, 10)}
				if err := s.sw.Write(row); err != nil {
					logrus.Errorf("state snapshot write failed: %v", err)
				}
			}
		}
		for !snap.IsFinal && s.nextSnapshotUs <= snap.Clock {
			s.nextSnapshotUs += s.snapshotInterval
		}
	}
	if s.w == nil {
		return
	}
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
	var firstErr error
	for _, p := range []struct {
		w *csv.Writer
		f *os.File
	}{{s.w, s.f}, {s.sw, s.sf}} {
		if p.w == nil {
			continue
		}
		p.w.Flush()
		if err := p.w.Error(); err != nil && firstErr == nil {
			firstErr = err
		}
		if err := p.f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// resolveSnapshotPath returns the snapshot csv path (empty when off). The snapshot
// interval must be a positive multiple of the state interval when both are set,
// because both ride the same progress hook.
func resolveSnapshotPath(snapshotInterval, stateInterval int64, statePath, metrics string) (string, error) {
	if snapshotInterval <= 0 {
		return "", nil
	}
	if stateInterval > 0 && snapshotInterval%stateInterval != 0 {
		return "", fmt.Errorf("--state-snapshot-interval must be a multiple of --state-sample-interval")
	}
	base := statePath
	if base == "" {
		if metrics == "" {
			return "", fmt.Errorf("--state-snapshot-interval requires --metrics-path or --state-sample-path")
		}
		base = metricsStem(metrics) + "_state.csv"
	}
	return strings.TrimSuffix(base, "_state.csv") + "_snapshot.csv", nil
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
	return metricsStem(metrics) + "_state.csv", nil
}

// metricsStem strips .json or .json.gz from a metrics path.
func metricsStem(p string) string {
	return strings.TrimSuffix(strings.TrimSuffix(p, ".gz"), ".json")
}
