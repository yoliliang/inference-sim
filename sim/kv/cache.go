// Package kv implements block-based KV cache management for the BLIS simulator.
// It provides single-tier GPU (KVCacheState) and two-tier GPU+CPU (TieredKVCache)
// implementations of the sim.KVStore interface. Both support prefix caching with
// SHA256-based block hashing, LRU eviction, and check-then-act allocation (vLLM parity).
package kv

import (
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// ToDo: Multi-modality is not yet supported
// Please see a description of prefix caching with images here:
// https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html#automatic-prefix-caching

// KVBlock represents a unit of KV cache storage.
// Each block stores a fixed number of tokens and is tracked by a prefix hash.
// A block becomes eligible for caching once it is full.
type KVBlock struct {
	ID       int64    // Unique ID of the block
	RefCount int      // Number of active requests referencing this block
	InUse    bool     // Whether the block is currently in use by an active (batched) request
	Hash     string   // Prefix hash identifying this block's content and its lineage (if full)
	Tokens   []sim.TokenID // Actual tokens stored in this block; full if len(Tokens) == BlockSizeTokens
	PrevFree *KVBlock // LRU doubly linked list: previous free block
	NextFree *KVBlock // LRU doubly linked list: next free block
}

// KVCacheState maintains global KV cache status across all requests.
// It implements prefix caching, reverse-order LRU eviction, and tracks free blocks
// via a direct counter on the free list (FreeBlockCnt), mirroring vLLM's
// FreeKVCacheBlockQueue.num_free_blocks.
type KVCacheState struct {
	// ours: incremental snapshot state, see refreshSnapshot
	snapMap   map[string]int64
	snapLog   []hashOp
	snapValid bool

	TotalBlocks     int64              // Total KV blocks available on GPU
	BlockSizeTokens int64              // Tokens per block
	Blocks          []*KVBlock         // All KV blocks
	RequestMap      map[string][]int64 // RequestID -> block sequence
	HashToBlock     map[string]int64   // Hash -> block ID
	FreeHead        *KVBlock           // Head of free list
	FreeTail        *KVBlock           // Tail of free list
	FreeBlockCnt    int64              // Direct count of blocks in free list (vLLM parity)
	CacheHits       int64              // blocks found via prefix cache (PR12)
	CacheMisses     int64              // blocks not found, allocated fresh (PR12)
}

// NewKVCacheState initializes the KVCacheState and places all blocks in the free list in order.
func NewKVCacheState(totalBlocks int64, blockSizeTokens int64) *KVCacheState {
	if totalBlocks <= 0 {
		panic(fmt.Sprintf("NewKVCacheState: TotalKVBlocks must be > 0, got %d", totalBlocks))
	}
	if blockSizeTokens <= 0 {
		panic(fmt.Sprintf("NewKVCacheState: BlockSizeTokens must be > 0, got %d", blockSizeTokens))
	}
	kvc := &KVCacheState{
		TotalBlocks:     totalBlocks,
		BlockSizeTokens: blockSizeTokens,
		Blocks:          make([]*KVBlock, totalBlocks),
		RequestMap:      make(map[string][]int64),
		HashToBlock:     make(map[string]int64),
	}
	for i := int64(0); i < totalBlocks; i++ {
		blk := &KVBlock{ID: i}
		kvc.Blocks[i] = blk
		kvc.appendToFreeList(blk)
	}
	return kvc
}

// appendToFreeList inserts a block at the tail of the free list.
func (kvc *KVCacheState) appendToFreeList(block *KVBlock) {
	kvc.FreeBlockCnt++
	block.NextFree = nil
	// in a doubly linked list, either both head and tail will be nil, or neither or nil
	if kvc.FreeTail != nil {
		// non-empty list; append block at end
		kvc.FreeTail.NextFree = block
		block.PrevFree = kvc.FreeTail
		kvc.FreeTail = block
	} else {
		// empty list; create list with a single block
		kvc.FreeHead = block
		kvc.FreeTail = block
		block.PrevFree = nil
	}
}

// removeFromFreeList detaches a block from the LRU free list.
func (kvc *KVCacheState) removeFromFreeList(block *KVBlock) {
	kvc.FreeBlockCnt--
	if block.PrevFree != nil {
		// a - b - block - c => a - b - c
		block.PrevFree.NextFree = block.NextFree
	} else {
		// block - c - d => c - d
		kvc.FreeHead = block.NextFree
	}
	if block.NextFree != nil {
		// a - b - block - c => a - b - c
		block.NextFree.PrevFree = block.PrevFree
	} else {
		// a - b - block => a - b
		kvc.FreeTail = block.PrevFree
	}
	block.NextFree = nil
	block.PrevFree = nil
}

// GetCachedBlocks attempts to reuse previously cached full blocks.
// It returns block IDs for the longest contiguous cached prefix.
// This is a pure query — it does not modify any state.
// CacheHits are counted by AllocateKVBlocks when cached blocks are committed.
//
// Uses hierarchical block hashing: each block's hash chains the previous
// block's hash, so each iteration hashes only blockSize tokens plus a
// fixed-length prev hash (O(K * blockSize) total, down from O(K^2 * blockSize)
// with flat-prefix hashing). Breaks on first miss.
//
// CO-CHANGE: SnapshotCachedBlocksFn (same file) replicates this algorithm on a
// frozen map[string]int64 snapshot. If this loop, the hash chain logic, or the
// break condition changes, update SnapshotCachedBlocksFn to match.
func (kvc *KVCacheState) GetCachedBlocks(tokens []sim.TokenID) (blockIDs []int64) {
	n := util.Len64(tokens) / kvc.BlockSizeTokens
	prevHash := ""
	for i := int64(0); i < n; i++ {
		start := i * kvc.BlockSizeTokens
		end := start + kvc.BlockSizeTokens
		h := hash.HashBlock(prevHash, tokens[start:end])
		blockId, ok := kvc.HashToBlock[h]
		if !ok {
			break
		}
		blockIDs = append(blockIDs, blockId)
		prevHash = h
	}
	return
}

// ours: one-entry memo of a prompt's block hash chain. The router queries every
// instance's cache for the same prompt in a row; without the memo each query
// recomputed the SHA256 chain from scratch, which was a fifth of the run time at
// n = 64. The memo is keyed by the token content (a pointer/length fast path first,
// which is safe because the memo keeps the slice alive, see sameTokens). The
// hash chain is extended lazily, one block at a time. Single-threaded by design,
// like the rest of the simulator.
type promptChain struct {
	tokens    []sim.TokenID
	blockSize int64
	hashes    []string
}

var lastPromptChain promptChain

func sameTokens(a, b []sim.TokenID) bool {
	// The memo holds a reference to the last prompt's slice, so its backing array cannot
	// be freed and reused while it is the memo key; equal data pointer and length then
	// mean the same, unmodified prompt (InputTokens are never mutated after generation).
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// promptChainHash returns the hash of block i of tokens (chained from block 0).
func promptChainHash(tokens []sim.TokenID, blockSize int64, i int64) string {
	c := &lastPromptChain
	if c.blockSize != blockSize || !sameTokens(c.tokens, tokens) {
		c.tokens = tokens // hold the reference (see sameTokens)
		c.blockSize = blockSize
		c.hashes = c.hashes[:0]
	}
	for int64(len(c.hashes)) <= i {
		j := int64(len(c.hashes))
		prev := ""
		if j > 0 {
			prev = c.hashes[j-1]
		}
		c.hashes = append(c.hashes, hash.HashBlock(prev, c.tokens[j*blockSize:(j+1)*blockSize]))
	}
	return c.hashes[i]
}

// SnapshotCachedBlocksFn returns a function that queries a frozen copy of the
// current HashToBlock map. The returned function counts consecutive cached prefix
// blocks for given tokens using the snapshot, NOT the live state.
// Used for stale cache signal simulation (issue #919).
//
// The snapshot captures HashToBlock at call time. Subsequent allocations/releases
// do NOT affect the returned function's results.
//
// CO-CHANGE: GetCachedBlocks (same file) implements the same algorithm on live
// HashToBlock state. If this loop, the hash chain logic, or the break condition
// changes, update GetCachedBlocks to match.
func (kvc *KVCacheState) SnapshotCachedBlocksFn() func([]sim.TokenID) int {
	snapshot := kvc.refreshSnapshot()
	blockSize := kvc.BlockSizeTokens
	return func(tokens []sim.TokenID) int {
		n := int64(len(tokens)) / blockSize
		count := 0
		for i := int64(0); i < n; i++ {
			if _, ok := snapshot[promptChainHash(tokens, blockSize, i)]; !ok { // ours: memoised chain
				break
			}
			count++
		}
		return count
	}
}

// ours: incremental snapshot of HashToBlock. Upstream copied the whole map on every
// refresh (every 50 ms of simulated time per instance), which was two thirds of the
// run time at scale. The frozen-copy semantics are kept: refreshSnapshot returns a map
// equal to HashToBlock at the moment of the call, built by replaying the writes made
// since the previous snapshot. Every write to HashToBlock must go through setHash or
// delHash; a log longer than the map itself falls back to a full copy.
type hashOp struct {
	key string
	id  int64
	del bool
}

func (kvc *KVCacheState) setHash(h string, id int64) {
	kvc.HashToBlock[h] = id
	kvc.logHash(h, id, false)
}

func (kvc *KVCacheState) delHash(h string) {
	delete(kvc.HashToBlock, h)
	kvc.logHash(h, 0, true)
}

func (kvc *KVCacheState) logHash(h string, id int64, del bool) {
	if !kvc.snapValid {
		return
	}
	if len(kvc.snapLog) > len(kvc.HashToBlock)+1024 {
		kvc.snapValid = false // too much churn since the last snapshot: rebuild fully next time
		kvc.snapLog = nil
		return
	}
	kvc.snapLog = append(kvc.snapLog, hashOp{key: h, id: id, del: del})
}

func (kvc *KVCacheState) refreshSnapshot() map[string]int64 {
	if kvc.snapValid && kvc.snapMap != nil {
		for _, op := range kvc.snapLog {
			if op.del {
				delete(kvc.snapMap, op.key)
			} else {
				kvc.snapMap[op.key] = op.id
			}
		}
		kvc.snapLog = kvc.snapLog[:0]
		if len(kvc.snapMap) == len(kvc.HashToBlock) {
			return kvc.snapMap
		}
	}
	kvc.snapMap = make(map[string]int64, len(kvc.HashToBlock))
	for k, v := range kvc.HashToBlock {
		kvc.snapMap[k] = v
	}
	kvc.snapLog = kvc.snapLog[:0]
	kvc.snapValid = true
	return kvc.snapMap
}

// AllocateKVBlocks handles KV Block allocation for both prefill and decode.
// If the latest block is full, a new one is allocated. Otherwise push to latest allocated block.
// start and endIndex are by original requests' index
// endIndex is non-inclusive
func (kvc *KVCacheState) AllocateKVBlocks(req *sim.Request, startIndex int64, endIndex int64, cachedBlocks []int64) bool {
	reqID := req.ID
	logrus.Debugf("AllocateBlock for ReqID: %s, Num Inputs: %d, startIndex = %d, endIndex = %d", req.ID, req.InputLen(), startIndex, endIndex)

	var newTokens []sim.TokenID
	var numNewBlocks int64
	if req.ProgressIndex < req.InputLen() {
		// request is in prefill (could be chunked)
		newTokens = req.InputTokenSlice(startIndex, endIndex)

		// Compute blocks needed, accounting for tokens that will be absorbed
		// into the request's existing partially-filled last block. Without this,
		// the pre-check over-estimates by up to 1 block, causing false rejections
		// when free blocks are tight (#492).
		effectiveTokens := util.Len64(newTokens)
		if ids, hasBlocks := kvc.RequestMap[reqID]; hasBlocks && len(ids) > 0 {
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			// spare < BlockSizeTokens excludes empty blocks (0 tokens stored)
			if spare := kvc.BlockSizeTokens - util.Len64(lastBlk.Tokens); spare > 0 && spare < kvc.BlockSizeTokens {
				effectiveTokens -= min(spare, effectiveTokens)
			}
		}
		if effectiveTokens > 0 {
			numNewBlocks = (effectiveTokens + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens
		}

		// Account for cached blocks that will leave the free list when claimed.
		// Mirrors vLLM's num_evictable_blocks (single_type_kv_cache_manager.py:124-127).
		// Invariant: cachedBlocks (from GetCachedBlocks) only contains blocks not yet
		// owned by the calling request — they are prefix-cache hits with RefCount >= 0.
		// Blocks with !InUse (RefCount == 0) sit on the free list and will be removed
		// when claimed, reducing countFreeBlocks(). We must budget for this.
		var cachedFromFreeList int64
		for _, blockID := range cachedBlocks {
			blk := kvc.Blocks[blockID]
			if !blk.InUse {
				cachedFromFreeList++
			}
		}

		if numNewBlocks+cachedFromFreeList > kvc.countFreeBlocks() {
			logrus.Debugf("KV cache full: cannot allocate %d new + %d cached blocks for req %s",
				numNewBlocks, cachedFromFreeList, req.ID)
			return false
		}
	} else {
		// request is in decode
		newTokens = append(newTokens, req.OutputTokens[startIndex-req.InputLen()])

		// Decode pre-check: if the last block is full and no free blocks exist,
		// fail fast without any state mutation. Mirrors vLLM's universal pre-check
		// (kv_cache_manager.py:334-336 / single_type_kv_cache_manager.py:95-101).
		//
		// For decode, get_num_blocks_to_allocate returns max(cdiv(tokens, blockSize) - len(blocks), 0).
		// This simplifies to: last block full → need 1 new block; last block has spare → need 0.
		//
		// When no RequestMap entry exists (e.g., preempted request whose blocks were
		// released but ProgressIndex was not reset), we need 1 new block unconditionally
		// since there is no existing block with spare capacity.
		if ids, hasBlocks := kvc.RequestMap[reqID]; hasBlocks && len(ids) > 0 {
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			if util.Len64(lastBlk.Tokens) == kvc.BlockSizeTokens && kvc.countFreeBlocks() == 0 {
				logrus.Debugf("KV cache full: cannot allocate decode block for req %s (last block full, 0 free)", reqID)
				return false
			}
		} else {
			// No existing blocks — need 1 new block for this decode token.
			if kvc.countFreeBlocks() == 0 {
				logrus.Debugf("KV cache full: cannot allocate decode block for req %s (no existing blocks, 0 free)", reqID)
				return false
			}
		}
	}
	newTokenProgressIndex := int64(0)
	for newTokenProgressIndex < util.Len64(newTokens) { // non-inclusive endIndex
		ids, ok := kvc.RequestMap[reqID]
		latestBlk := &KVBlock{}
		if ok {
			// Running request path: KV cache has already seen this request before.
			// The latest block needs to be filled first, followed by new blocks.
			// Note: Running requests do NOT re-claim cached blocks (vLLM parity).
			// Preempted requests reset ProgressIndex to 0 and re-enter via the !ok
			// path, where they claim all cached blocks upfront.
			latestBlk = kvc.Blocks[ids[len(ids)-1]]
		} else {
			// KV cache is seeing this request for the first time (beginning of prefill)
			// append the cached blocks to this request's ID map

			for _, blockId := range cachedBlocks {
				blk := kvc.Blocks[blockId]
				blk.RefCount++
				if !blk.InUse {
					blk.InUse = true
					kvc.removeFromFreeList(blk)
				}
				kvc.CacheHits++
				logrus.Debugf("Hit KV Cache for req: %s of length: %d", req.ID, util.Len64(cachedBlocks)*kvc.BlockSizeTokens)
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockId)
			}

			// Update latestBlk to the last claimed block if we claimed any
			if len(cachedBlocks) > 0 {
				ids = kvc.RequestMap[reqID] // Refresh ids after appending cached blocks
				latestBlk = kvc.Blocks[ids[len(ids)-1]]
			}
		}
		if len(latestBlk.Tokens) > 0 && util.Len64(latestBlk.Tokens) < kvc.BlockSizeTokens {
			// latest block is not full yet, append tokens to the latest block
			remaining := kvc.BlockSizeTokens - util.Len64(latestBlk.Tokens)
			toksToAppend := newTokens[newTokenProgressIndex:min(newTokenProgressIndex+remaining, util.Len64(newTokens))]
			latestBlk.Tokens = append(latestBlk.Tokens, toksToAppend...)
			newTokenProgressIndex += util.Len64(toksToAppend)
			logrus.Debugf("Appending to latest blk: req: %s, newTokenProgressIndex = %d, appended=%d tokens", req.ID, newTokenProgressIndex, util.Len64(toksToAppend))
			if util.Len64(latestBlk.Tokens) == kvc.BlockSizeTokens {
				// latestBlk is full — compute its hierarchical hash.
				// Chain from the previous block's hash (or "" if first block).
				prevHash := ""
				if len(ids) >= 2 {
					prevHash = kvc.Blocks[ids[len(ids)-2]].Hash
				}
				h := hash.HashBlock(prevHash, latestBlk.Tokens)
				latestBlk.Hash = h
				kvc.setHash(h, latestBlk.ID) // ours
			}
		} else {
			// latest block is full or request is coming in for the first time.
			// allocate new block(s) for the request.
			// Recompute blocks needed from remaining tokens (after partial fill consumed some).
			remainingTokens := util.Len64(newTokens) - newTokenProgressIndex
			if remainingTokens <= 0 {
				break
			}
			numNewBlocks = (remainingTokens + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens

			// Initialize prevHash for hierarchical block hashing.
			// Chain from the last existing block for this request (if any).
			prevHash := ""
			if existingIDs, has := kvc.RequestMap[reqID]; has && len(existingIDs) > 0 {
				prevHash = kvc.Blocks[existingIDs[len(existingIDs)-1]].Hash
			}

			for i := int64(0); i < numNewBlocks; i++ {
				blk := kvc.popFreeBlock()
				if blk == nil {
					panic(fmt.Sprintf("popFreeBlock returned nil after pre-check passed for req %s: INV-4 violation", reqID))
				}

				// Lazy hash deletion (vLLM parity): clear old hash before filling
				// with new content. Matches vLLM's _maybe_evict_cached_block
				// semantics (block_pool.py:331-356). Hash was preserved in
				// popFreeBlock so preempted requests could find their cached
				// prefix blocks on readmission.
				if blk.Hash != "" {
					kvc.delHash(blk.Hash) // ours
					blk.Hash = ""
				}

				// start and end are the range of tokens in blk
				start := newTokenProgressIndex
				end := newTokenProgressIndex + kvc.BlockSizeTokens
				logrus.Debugf("Assigning new blocks: req = %s, newTokenProgressIndex = %d, ogStartIdx= %d, ogEndIdx = %d, startBlk=%d, endBlk=%d", req.ID, newTokenProgressIndex, startIndex, endIndex, start, end)
				if end > util.Len64(newTokens) {
					end = util.Len64(newTokens)
				}
				tok := newTokens[start:end]
				blk.Tokens = append([]sim.TokenID{}, tok...) // copy tokens
				blk.RefCount = 1
				blk.InUse = true
				kvc.CacheMisses++

				if util.Len64(blk.Tokens) == kvc.BlockSizeTokens && req.ProgressIndex < req.InputLen() {
					// Only compute prefix hash during prefill (not decode).
					// During decode, blocks hold output tokens that should not
					// participate in prefix caching (input sequences only).
					h := hash.HashBlock(prevHash, blk.Tokens)
					blk.Hash = h
					kvc.setHash(h, blk.ID) // ours
					prevHash = h
				}
				// allocated is the block IDs allocated for this request
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blk.ID)
				newTokenProgressIndex = end
			}
		}
	}

	return true
}

// popFreeBlock evicts a block from the free list and prepares it for reuse.
// Hash entries are preserved (lazy deletion) - they will be cleared when the
// block is filled with new content in AllocateKVBlocks allocation loop.
// Matches vLLM's block_pool.py:313-318 + _maybe_evict_cached_block semantics.
func (kvc *KVCacheState) popFreeBlock() *KVBlock {
	head := kvc.FreeHead
	if head == nil {
		return nil
	}
	kvc.removeFromFreeList(head)
	// Hash stays intact - will be cleared when block is filled with new content (vLLM parity)
	head.Tokens = nil
	return head
}

// countFreeBlocks returns the number of blocks not currently in use.
// This is a direct read of the free list counter (vLLM parity), not arithmetic derivation.
func (kvc *KVCacheState) countFreeBlocks() int64 {
	return kvc.FreeBlockCnt
}

// commitCachedBlocks registers a slice of cached blocks into a request's RequestMap.
// Increments RefCount, sets InUse, removes from free list, records cache hits,
// and appends block IDs to RequestMap.
//
// For new requests (reqID not yet in RequestMap), pass the full cached block slice.
// For running requests (reqID already in RequestMap), pass only the uncovered range
// (e.g., newCached[startBlock:endBlock]) to avoid double-counting RefCount for
// blocks the request already owns.
//
// NOTE: With the check-then-act allocation pattern, rollback does not exist.
// This method is used by TieredKVCache before calling gpu.AllocateKVBlocks.
// The inner AllocateKVBlocks pre-check sees the reduced FreeBlockCnt (already
// decremented by removeFromFreeList during commitCachedBlocks), so the pre-check
// correctly accounts for the committed blocks. In BLIS's single-threaded DES,
// FreeBlockCnt cannot decrease between the inner pre-check and allocation loop.
func (kvc *KVCacheState) commitCachedBlocks(reqID string, cachedBlocks []int64) {
	for _, blockID := range cachedBlocks {
		blk := kvc.Blocks[blockID]
		blk.RefCount++
		if !blk.InUse {
			blk.InUse = true
			kvc.removeFromFreeList(blk)
		}
		kvc.CacheHits++
		kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockID)
	}
}

// ReleaseKVBlocks deallocates blocks used by a completed request.
// Each block's refcount is decremented and may be returned to the free list.
func (kvc *KVCacheState) ReleaseKVBlocks(req *sim.Request) {
	ids := kvc.RequestMap[req.ID]
	delete(kvc.RequestMap, req.ID)
	// From https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html
	// Freed blocks are added to the tail of the free queue in reverse order.
	// Later blocks can only be reused if all preceding blocks also match
	// (hierarchical hashing), so evicting them first preserves the more
	// broadly reusable earlier blocks.
	for i := len(ids) - 1; i >= 0; i-- {
		blockId := ids[i]
		blk := kvc.Blocks[blockId]
		blk.RefCount--
		if blk.RefCount == 0 {
			blk.InUse = false
			kvc.appendToFreeList(blk)
		}
	}
}

// BlockSize returns the number of tokens per block.
func (kvc *KVCacheState) BlockSize() int64 { return kvc.BlockSizeTokens }

// UsedBlocks returns the number of blocks currently in use.
// Derived from TotalBlocks - FreeBlockCnt (read-only for callers).
func (kvc *KVCacheState) UsedBlocks() int64 { return kvc.TotalBlocks - kvc.FreeBlockCnt }

// TotalCapacity returns the total number of blocks.
func (kvc *KVCacheState) TotalCapacity() int64 { return kvc.TotalBlocks }

// CacheHitRate returns the cumulative cache hit rate.
// Returns 0 if no lookups have been performed.
func (kvc *KVCacheState) CacheHitRate() float64 {
	total := kvc.CacheHits + kvc.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(kvc.CacheHits) / float64(total)
}

// PendingTransferLatency returns 0 for single-tier cache (no transfers).
func (kvc *KVCacheState) PendingTransferLatency() int64 { return 0 }

// KVThrashingRate returns 0 for single-tier cache (no offload/reload).
func (kvc *KVCacheState) KVThrashingRate() float64 { return 0 }

// SetClock is a no-op for single-tier KV cache (no time-dependent behavior).
func (kvc *KVCacheState) SetClock(_ int64) {}

// ConsumePendingTransferLatency returns 0 for single-tier cache (no transfers).
func (kvc *KVCacheState) ConsumePendingTransferLatency() int64 { return 0 }

// MirrorToCPU is a no-op for single-tier KV cache (no CPU tier).
func (kvc *KVCacheState) MirrorToCPU(_ []*sim.Request) {}

// verifyBlockConservation walks the free list and block InUse flags independently
// to verify INV-4: freeListLen + inUseCount == TotalBlocks.
// Returns nil if conservation holds, or an error describing the violation.
// Intended for debug-mode step-boundary assertions.
func (kvc *KVCacheState) verifyBlockConservation() error {
	freeListLen := int64(0)
	node := kvc.FreeHead
	for node != nil {
		freeListLen++
		node = node.NextFree
	}

	inUseCount := int64(0)
	for _, blk := range kvc.Blocks {
		if blk.InUse {
			inUseCount++
		}
	}

	if freeListLen+inUseCount != kvc.TotalBlocks {
		return fmt.Errorf("block conservation violated: freeList=%d + inUse=%d != total=%d",
			freeListLen, inUseCount, kvc.TotalBlocks)
	}

	if freeListLen != kvc.FreeBlockCnt {
		return fmt.Errorf("FreeBlockCnt drift: counter=%d, actual free list=%d",
			kvc.FreeBlockCnt, freeListLen)
	}

	return nil
}
