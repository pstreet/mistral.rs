#!/bin/bash
# Profile one 910-token prefill on the 4B model with rocprofv3 and print the breakdown.
# Usage: ./profile_prefill.sh [outdir]
set -e
OUT=${1:-/tmp/opencode/prof}
ROCM=/home/pstreet/LocalAI/rocm-install
MRS=/home/pstreet/LocalAI/mistral.rs/target/release/mistralrs
MODEL_DIR=/home/pstreet/LocalAI/models/unsloth/Qwen3.5-4B-MTP-GGUF
PORT=18323
rm -rf "$OUT" && mkdir -p "$OUT"
(
  setsid env LD_LIBRARY_PATH=$ROCM/lib MRS_ATTENTION_SYNC="${MRS_ATTENTION_SYNC:-}" MRS_ATTENTION_DEBUG="${MRS_ATTENTION_DEBUG:-}" \
    $ROCM/bin/rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$OUT" \
    -- bash -c "cd $MODEL_DIR && exec $MRS serve -p $PORT -m . -f Qwen3.5-4B-Q8_0.gguf --prefix-cache-n 0" \
    > /tmp/opencode/serve4b_rocprof.log 2>&1 < /dev/null &
)
# wait for listen
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q $PORT && break; sleep 2; done
ss -ltn 2>/dev/null | grep -q $PORT || { echo "server did not start"; tail -20 /tmp/opencode/serve4b_rocprof.log; exit 1; }
# fire one prefill
python3 - <<PY
import json, urllib.request
prompt = ("the quick brown fox jumps over the lazy dog. " * 90)
body = {"model":"default","messages":[{"role":"user","content":prompt}],"max_tokens":4,"temperature":0,"stream":False}
req = urllib.request.Request("http://localhost:$PORT/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    d=json.load(r)
print("prompt_tokens=", d.get("usage",{}).get("prompt_tokens"))
PY
sleep 2
MRS_PID=$(ss -ltnp 2>/dev/null | grep $PORT | grep -oP 'pid=\K[0-9]+' | head -1)
kill -TERM $MRS_PID 2>/dev/null || true
sleep 8
CSV=$(find "$OUT" -name "*_kernel_trace.csv" | head -1)
echo "=== kernel trace: $CSV ==="
python3 - "$CSV" <<'PY'
import csv, sys
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
for r in rows: r["s"]=int(r["Start_Timestamp"]); r["e"]=int(r["End_Timestamp"])
rec=[r for r in rows if "gated_delta_rule_recurrence_kernel_warp" in r["Kernel_Name"]]
if not rec:
    print("no prefill recurrence found"); sys.exit(0)
wmin=min(r["s"] for r in rec); wmax=max(r["e"] for r in rec)
lo=wmin-2_000_000; hi=wmax+2_000_000
win=[r for r in rows if lo<=r["s"]<=hi]
agg=defaultdict(lambda:[0,0])
for r in win:
    agg[r["Kernel_Name"]][0]+=r["e"]-r["s"]; agg[r["Kernel_Name"]][1]+=1
grand=sum(v[0] for v in agg.values())
print(f"prefill window: {(wmax-wmin)/1e6:.1f} ms | kernels {len(win)} | GPU kernel time {grand/1e6:.1f} ms | CPU gaps {(wmax-wmin-grand)/1e6:.1f} ms")
print(f"{'kernel':<58} {'ms':>8} {'cnt':>5} {'avg_us':>9}")
for name,(tot,cnt) in sorted(agg.items(), key=lambda kv:-kv[1][0]):
    if tot/1e6 < 0.3: continue
    print(f"{name[:58]:<58} {tot/1e6:>8.2f} {cnt:>5} {tot/cnt/1e3:>9.1f}")
print(f"TOTAL GPU kernel: {grand/1e6:.1f} ms")
PY
