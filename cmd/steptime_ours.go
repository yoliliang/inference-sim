package cmd

// ours: `blis steptime` evaluates the configured latency model on batches read from a CSV.
// It changes no simulation behaviour. The fluid-dual policy (model draft, Section 3) needs an
// affine round-time model tau = tau0 + tau_p * (prefill tokens) + tau_kv * (KV tokens read by
// decodes) per GPU type; ours/fluid/stepfit.py generates batches, calls this command and fits
// the three constants to BLIS's own step-time function.
//
// Input CSV (header required): batch,kind,prompt,progress,new_tokens
//   kind = prefill: a request with `prompt` prompt tokens, `progress` of them already
//                   processed (0 for a whole prefill), processing `new_tokens` this step
//   kind = decode:  a request with `prompt` prompt tokens whose KV cache holds `progress`
//                   tokens (prompt + generated - 1), producing one token this step
// Output CSV on stdout: batch,step_us

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

var stepTimeBatchesPath string

var stepTimeCmd = &cobra.Command{
	Use:   "steptime",
	Short: "ours: evaluate the step-time model on batches from a CSV (for the fluid model fit)",
	Run: func(cmd *cobra.Command, _ []string) {
		level, err := logrus.ParseLevel(logLevel)
		if err == nil {
			logrus.SetLevel(level)
		}
		lr := resolveLatencyConfig(cmd)
		hw := sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, 1,
			enableExpertParallel, moeCommBackend, lr.Backend, maxModelLen)
		lm, err := latency.NewLatencyModel(sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs), hw)
		if err != nil {
			logrus.Fatalf("steptime: latency model: %v", err)
		}
		f, err := os.Open(stepTimeBatchesPath)
		if err != nil {
			logrus.Fatalf("steptime: %v", err)
		}
		defer f.Close()
		r := csv.NewReader(f)
		if _, err := r.Read(); err != nil { // header
			logrus.Fatalf("steptime: empty batches file: %v", err)
		}
		w := csv.NewWriter(os.Stdout)
		_ = w.Write([]string{"batch", "step_us"})
		var cur string
		var batch []*sim.Request
		flush := func() {
			if cur != "" {
				_ = w.Write([]string{cur, strconv.FormatInt(lm.StepTime(batch), 10)})
			}
			batch = batch[:0]
		}
		tokens := make([]sim.TokenID, 0)
		for {
			rec, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				logrus.Fatalf("steptime: %v", err)
			}
			if rec[0] != cur {
				flush()
				cur = rec[0]
			}
			prompt, _ := strconv.Atoi(rec[2])
			progress, _ := strconv.ParseInt(rec[3], 10, 64)
			newTok, _ := strconv.ParseInt(rec[4], 10, 64)
			if prompt > len(tokens) {
				tokens = make([]sim.TokenID, prompt)
			}
			req := &sim.Request{ID: fmt.Sprintf("b%s-%d", cur, len(batch)), State: sim.StateRunning,
				InputTokens: tokens[:prompt], ProgressIndex: progress, NumNewTokens: int(newTok)}
			switch rec[1] {
			case "prefill":
			case "decode":
				req.OutputTokens = tokens[:1] // the latency models classify decode work by this
			default:
				logrus.Fatalf("steptime: unknown kind %q", rec[1])
			}
			batch = append(batch, req)
		}
		flush()
		w.Flush()
	},
}

func init() {
	registerSimConfigFlags(stepTimeCmd)
	stepTimeCmd.Flags().StringVar(&stepTimeBatchesPath, "batches", "", "CSV of batches: batch,kind,prompt,progress,new_tokens")
	rootCmd.AddCommand(stepTimeCmd)
}
