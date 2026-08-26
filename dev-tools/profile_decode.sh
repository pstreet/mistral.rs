#!/bin/bash
# Profile decode steps on the 4B model: solo (b=1) long generation, then an N=16 burst.
# Usage: ./profile_decode.sh [outdir]
set -e
OUT=${1:-/tmp/opencode/decodeprof}
ROCM=/home/pstreet/LocalAI/rocm-install
MRS=/home/pstreet/LocalAI/mistral.rs/target/release/mistralrs
MODEL_DIR=/home/pstreet/LocalAI/models/unsloth/Qwen3.5-4B-MTP-GGUF
PORT=18323
rm -rf "$OUT" && mkdir -p "$OUT"
(
  setsid env LD_LIBRARY_PATH=$ROCM/lib MRS_PROFILE=1 \
    $ROCM/bin/rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$OUT" \
    -- bash -c "cd $MODEL_DIR && exec $MRS serve -p $PORT -m . -f Qwen3.5-4B-Q8_0.gguf --prefix-cache-n 16" \
    > /tmp/opencode/serve4b_decprof.log 2>&1 < /dev/null &
)
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q $PORT && break; sleep 2; done
ss -ltn 2>/dev/null | grep -q $PORT || { echo "server did not start"; tail -20 /tmp/opencode/serve4b_decprof.log; exit 1; }

python3 - <<'PY'
import json, urllib.request, time, threading
base = "the quick brown fox jumps over the lazy dog. "
def ask(prompt, mx):
    body = {"model":"default","messages":[{"role":"user","content":prompt}],"max_tokens":mx,"temperature":0,"stream":False}
    req = urllib.request.Request("http://localhost:18323/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r: json.load(r)
    return time.perf_counter()-t0
# warmup + prime prefix cache
ask(base*90, 4)
print("warmup done", flush=True)
# Phase A: solo long decode
tA = ask(base*90, 64)
print(f"solo 64-tok gen: {tA:.2f}s ({64/tA:.1f} tok/s)", flush=True)
time.sleep(1)
# Phase B: N=16 concurrent burst
res=[]
lock=threading.Lock()
def worker():
    t=ask(base*90, 48)
    with lock: res.append(t)
t0=time.perf_counter()
ths=[threading.Thread(target=worker) for _ in range(16)]
[t.start() for t in ths]; [t.join() for t in ths]
wall=time.perf_counter()-t0
print(f"N=16 x48tok: wall={wall:.2f}s agg={16*48/wall:.1f} tok/s", flush=True)
PY

MRS_PID=$(ss -ltnp 2>/dev/null | grep $PORT | grep -oP 'pid=\K[0-9]+' | head -1)
kill -TERM $MRS_PID 2>/dev/null || true
sleep 8
CSV=$(find "$OUT" -name "*_kernel_trace.csv" | head -1)
echo "trace: $CSV"
grep -E "mrs-profile.*decode" /tmp/opencode/serve4b_decprof.log | tail -6