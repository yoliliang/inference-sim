// Idiomatic entrypoint for Cobra CLI that deletes handling to the Cobra root command in cmd/root.go

package main

import (
	"os"
	"runtime/pprof"

	"github.com/inference-sim/inference-sim/cmd"
)

func main() {
	// ours: CPU profile when BLIS_CPUPROFILE names a file. Diagnostic only; unset in
	// normal runs, so stdout and behaviour are unchanged (INV-6).
	if path := os.Getenv("BLIS_CPUPROFILE"); path != "" {
		if f, err := os.Create(path); err == nil {
			defer f.Close() // runs last: deferred calls run in reverse order
			if err := pprof.StartCPUProfile(f); err == nil {
				defer pprof.StopCPUProfile()
			}
		}
	}
	cmd.Execute()
}
