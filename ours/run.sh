#!/usr/bin/env bash
# ours/run.sh <name> [extra blis flags...]   (Linux/macOS/Git Bash twin of run.ps1)
set -e
name=$1; shift
here=$(cd "$(dirname "$0")" && pwd); root=$(dirname "$here")
out="$here/results/$name.json"
bin="$root/blis"; [ -x "$bin" ] || bin="$root/blis.exe"
"$bin" run --model qwen/qwen3-14b --hardware H100 --tp 1 \
  --workload-spec "$here/specs/types3.yaml" \
  --num-instances 4 --total-kv-blocks 2500 \
  --snapshot-refresh-interval 0 --seed 42 \
  --metrics-path "$out" "$@" 2> "$here/results/$name.log" > /dev/null
python3 - "$name" "$out" <<'PY'
import json,sys
m=json.load(open(sys.argv[2]))
print(f"{sys.argv[1]:<28} preempt={m['preemption_count']:5d} ttft_p99={m['ttft_p99_ms']:9,.0f} e2e_mean={m['e2e_mean_ms']:8,.0f} tok/s={m['tokens_per_sec']:6,.0f}")
PY
