package sim

import "testing"

// ours: tests for the Mooncake admission rules (Early Rejection, and Early Rejection based
// on Prediction). They use the built-in linear step-time stand-in, so the numbers below are
// deterministic and independent of the calibrated latency model.

func mooncakeReq(prompt int) *Request {
	return &Request{ID: "r", InputTokens: make([]TokenID, prompt), OutputTokens: make([]TokenID, 1)}
}

func snaps(specs ...RoutingSnapshot) *RouterState {
	return &RouterState{Snapshots: specs, Clock: 0}
}

func TestMooncake_EmptyClusterAdmits(t *testing.T) {
	for _, mode := range []string{"now", "predict"} {
		m := NewMooncakeAdmission(mode, 2, 60, 1.0, 9, 8192)
		ok, reason := m.Admit(mooncakeReq(256), snaps(RoutingSnapshot{ID: "a"}, RoutingSnapshot{ID: "b"}))
		if !ok {
			t.Errorf("%s: empty cluster must admit, got %q", mode, reason)
		}
	}
}

func TestMooncake_RejectsWhenQueuedPrefillExceedsTTFTTarget(t *testing.T) {
	// One instance, a queue holding 400,000 prompt tokens. With the stand-in model the
	// prefill rate is about 8192 tokens per (15 ms + 82 ms) step, so the predicted TTFT is
	// about 5 s, above the 2 s target: reject on the prefill load, in both modes.
	for _, mode := range []string{"now", "predict"} {
		m := NewMooncakeAdmission(mode, 2, 60, 1.0, 9, 8192)
		ok, reason := m.Admit(mooncakeReq(256), snaps(RoutingSnapshot{ID: "a", QueueDepth: 200, QueuedPromptTokens: 400000}))
		if ok {
			t.Errorf("%s: expected rejection on prefill load", mode)
		}
		if ok || reason == "" {
			continue
		}
		if reason[:len("mooncake: prefill")] != "mooncake: prefill" {
			t.Errorf("%s: expected a prefill-load reason, got %q", mode, reason)
		}
	}
}

func TestMooncake_PicksTheLightestInstanceForPrefill(t *testing.T) {
	// Instance a is swamped, instance b is idle: the request would go to b, so it is admitted.
	m := NewMooncakeAdmission("predict", 2, 60, 1.0, 9, 8192)
	ok, reason := m.Admit(mooncakeReq(256), snaps(
		RoutingSnapshot{ID: "a", QueueDepth: 200, QueuedPromptTokens: 400000},
		RoutingSnapshot{ID: "b"}))
	if !ok {
		t.Errorf("expected admission via the idle instance, got %q", reason)
	}
}

func TestMooncake_NowAndPredictDisagreeOnADrainingBatch(t *testing.T) {
	// A huge decode batch that will have drained by the time this request finishes prefill.
	// Stand-in TBT = 15 ms + 0.2 ms x batch; target 60 ms, so 300 decoding requests give a
	// load of 1.25 now. The prefill rate next to that batch is 7,892 tokens per 154 ms step,
	// so a queue of 150,000 prompt tokens makes TTFT_hat about 2.9 s; with a TTFT target of
	// 4 s the prefill load passes. With t_d = 3 s (predict mode) the batch is predicted to
	// have drained, and only the queue's 10 requests plus this one remain.
	state := func() *RouterState {
		return snaps(RoutingSnapshot{ID: "a", BatchSize: 300, QueueDepth: 10, QueuedPromptTokens: 150000, KvTokensInUse: 300 * 1000})
	}
	now := NewMooncakeAdmission("now", 4, 60, 1.0, 3, 8192)
	if ok, _ := now.Admit(mooncakeReq(256), state()); ok {
		t.Error("now mode should reject on the current decode load")
	}
	pred := NewMooncakeAdmission("predict", 4, 60, 1.0, 3, 8192)
	if ok, reason := pred.Admit(mooncakeReq(256), state()); !ok {
		t.Errorf("predict mode should admit once the batch is predicted to drain, got %q", reason)
	}
}

func TestMooncake_PredictSeesTheQueueAsFutureDecodeLoad(t *testing.T) {
	// Idle batch now, but 400 requests queued with short prompts: "now" admits (nothing is
	// decoding), "predict" rejects because those 400 requests will be decoding when this one
	// starts (TBT 15 + 0.2 x 401 = 95 ms > 60 ms). Prompts are short so the prefill load
	// stays under the target (400 x 64 tokens at 8,192 tokens per 97 ms step is 0.3 s).
	state := func() *RouterState {
		return snaps(RoutingSnapshot{ID: "a", BatchSize: 0, QueueDepth: 400, QueuedPromptTokens: 400 * 64})
	}
	now := NewMooncakeAdmission("now", 2, 60, 1.0, 9, 8192)
	if ok, reason := now.Admit(mooncakeReq(64), state()); !ok {
		t.Errorf("now mode should admit with an idle batch, got %q", reason)
	}
	pred := NewMooncakeAdmission("predict", 2, 60, 1.0, 9, 8192)
	if ok, _ := pred.Admit(mooncakeReq(64), state()); ok {
		t.Error("predict mode should reject on the future decode load")
	}
}

func TestMooncake_FullKVCacheMakesTheQueueWait(t *testing.T) {
	// The queue is short in compute terms (40 requests, 32,000 prompt tokens, about 0.4 s of
	// prefill) but the KV cache is full: 250,000 of 254,544 tokens in use by 170 requests. With
	// t_d = 9 s the instance frees about 27,800 tokens per second, so the 27,500-token shortfall
	// takes about 1 s; below a 2 s target the request is admitted. Doubling the queue makes the
	// shortfall 59,500 tokens, about 2.1 s of waiting, and the request is rejected on the
	// prefill load although the compute bound alone would admit it.
	base := RoutingSnapshot{ID: "a", BatchSize: 170, KvTokensInUse: 250000, TotalKvCapacityTokens: 254544}
	m := NewMooncakeAdmission("predict", 2, 60, 1.0, 9, 8192)
	short := base
	short.QueueDepth, short.QueuedPromptTokens = 40, 32000
	if ok, reason := m.Admit(mooncakeReq(256), snaps(short)); !ok {
		t.Errorf("short queue should be admitted, got %q", reason)
	}
	long := base
	long.QueueDepth, long.QueuedPromptTokens = 80, 64000
	ok, reason := m.Admit(mooncakeReq(256), snaps(long))
	if ok {
		t.Error("long queue against a full cache should be rejected")
	} else if len(reason) < 17 || reason[:17] != "mooncake: prefill" {
		t.Errorf("expected a prefill-load rejection, got %q", reason)
	}
}

func TestMooncake_IsRegistered(t *testing.T) {
	if !IsValidAdmissionPolicy("mooncake") {
		t.Error("mooncake should be a valid admission policy name")
	}
}
