package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// ours: tests of the heterogeneous-cluster hooks (per-pool KV blocks, wake rules).

func TestApplyPoolKVBlocks(t *testing.T) {
	pools := []NodePoolConfig{
		{Name: "h100", GPUType: "H100", KVBlocks: 0},
		{Name: "a100", GPUType: "A100-80", KVBlocks: 500},
	}
	cfg := sim.SimConfig{}
	cfg.TotalKVBlocks = 15909
	cfg.BlockSizeTokens = 16
	cfg.MaxModelLen = 40960

	applyPoolKVBlocks(&cfg, pools, "H100")
	if cfg.TotalKVBlocks != 15909 || cfg.MaxModelLen != 40960 {
		t.Fatalf("a pool without kv_blocks must leave the capacity alone, got %d blocks, max-model-len %d", cfg.TotalKVBlocks, cfg.MaxModelLen)
	}
	applyPoolKVBlocks(&cfg, pools, "A100-80")
	if cfg.TotalKVBlocks != 500 {
		t.Fatalf("explicit kv_blocks should win, got %d", cfg.TotalKVBlocks)
	}
	if cfg.MaxModelLen != 500*16 {
		t.Fatalf("max-model-len should be capped to the blocks, got %d", cfg.MaxModelLen)
	}
	applyPoolKVBlocks(&cfg, pools, "L40S")
	if cfg.TotalKVBlocks != 500 {
		t.Fatalf("an unknown GPU type must not change the capacity, got %d", cfg.TotalKVBlocks)
	}
}

func TestWakeRules_Names(t *testing.T) {
	for _, name := range append(WakeRuleNames(), "") {
		if !IsValidWakeRule(name) {
			t.Errorf("%q should be a valid wake rule", name)
		}
	}
	if IsValidWakeRule("slowest-first") {
		t.Errorf("unknown wake rule accepted")
	}
}
