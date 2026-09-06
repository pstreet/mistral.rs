#!/bin/bash
# Profile one true-cold prefill on the live 27B (port 1235) under rocprofv3 and
# print a per-kernel GPU breakdown focused on matmul / dequantize.
# Usage: ./profile_prefill_27b.sh [outdir]
set -e
OUT=${1:-/tmp/opencode/prof27}
EXTRA=${2:-}
ROCM=/home/pstreet/LocalAI/rocm-install
MRS=/home/pstreet/LocalAI/mistral.rs/target/release/mistralrs
CONFIG=/home/pstreet/LocalAI/models/mistralrs.toml
rm -rf "$OUT" && mkdir -p "$OUT"
(
  export ROCM_PATH=$ROCM
  export LD_LIBRARY_PATH=$ROCM/lib
  setsid env $EXTRA ROCM_PATH=$ROCM LD_LIBRARY_PATH=$ROCM/lib RUST_LOG=mistralrs_core=info \
    $ROCM/bin/rocprofv3 --kernel-trace -f csv -d "$OUT" \
    -- $MRS from-config -f "$CONFIG" > /tmp/opencode/prof27_server.log 2>&1 < /dev/null &
)
for i in $(seq 1 120); do ss -ltn 2>/dev/null | grep -q 1235 && break; sleep 2; done
ss -ltn 2>/dev/null | grep -q 1235 || { echo "server did not start"; tail -20 /tmp/opencode/prof27_server.log; exit 1; }
echo "server up; firing cold prefill"
python3 - <<'PY'
import json, urllib.request
prompt=("Every porcelain statue hides a wooden exoskeleton in shade. " * 100)
body={"model":"default","messages":[{"role":"user","content":prompt}],"max_tokens":1,"temperature":0,"stream":False}
req=urllib.request.Request("http://localhost:1235/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
import time
t0=time.perf_counter()
with urllib.request.urlopen(req,timeout=300) as r:
    d=json.load(r)
print("wall=%.2fs"%(time.perf_counter()-t0),"prompt_tokens=",d.get("usage",{}).get("prompt_tokens"))
PY
sleep 3
MRS_PID=$(pgrep -f "target/release/mistralrs from-config" | head -1)
[ -n "$MRS_PID" ] && kill -TERM "$MRS_PID" 2>/dev/null || true
sleep 10
CSV=$(find "$OUT" -name "*_kernel_trace.csv" | head -1)
echo "=== kernel trace: $CSV ==="
python3 - "$CSV" <<'PY'
import csv, sys
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
print("total kernel records:", len(rows))
for r in rows:
    r["s"]=int(r["Start_Timestamp"]); r["e"]=int(r["End_Timestamp"])
agg=defaultdict(lambda:[0,0])
for r in rows:
    agg[r["Kernel_Name"]][0]+=r["e"]-r["s"]; agg[r["Kernel_Name"]][1]+=1
grand=sum(v[0] for v in agg.values())
print(f"total GPU kernel time {grand/1e6:.1f} ms over {len(rows)} kernels")
print(f"{'kernel':<62} {'ms':>9} {'cnt':>6} {'avg_ms':>8}")
for name,(tot,cnt) in sorted(agg.items(), key=lambda kv:-kv[1][0]):
    if tot/1e6 < 1.0: continue
    print(f"{name[:62]:<62} {tot/1e6:>9.2f} {cnt:>6} {tot/cnt/1e6:>8.3f}")
PY