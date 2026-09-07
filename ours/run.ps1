# ours/run.ps1 <name> [extra blis flags...]
# Runs the three-type spec on 4 x Qwen3-14B/H100 with 2,500 blocks each and an
# omniscient router, writes ours/results/<name>.json, prints one summary line.
param([Parameter(Mandatory=$true)][string]$Name,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$Extra)
$root = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $PSScriptRoot "results\$Name.json"
$log  = Join-Path $PSScriptRoot "results\$Name.log"
$args = @("run","--model","qwen/qwen3-14b","--hardware","H100","--tp","1",
          "--workload-spec",(Join-Path $PSScriptRoot "specs\types3.yaml"),
          "--num-instances","4","--total-kv-blocks","2500",
          "--snapshot-refresh-interval","0","--seed","42",
          "--metrics-path",$out) + $Extra
& (Join-Path $root "blis.exe") @args 2> $log > $null
$m = Get-Content $out | ConvertFrom-Json
"{0,-28} preempt={1,5} ttft_p99={2,9:N0} e2e_mean={3,8:N0} tok/s={4,6:N0}" -f `
  $Name, $m.preemption_count, $m.ttft_p99_ms, $m.e2e_mean_ms, $m.tokens_per_sec
