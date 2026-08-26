#!/bin/bash
# Fire N sequential prefills under rocprofv3, record OK/FAIL per request,
# then segment the kernel trace into per-request windows for OK-vs-FAIL diffs.
# Usage: ./nan_trace.sh [outdir] [n_requests] [prompt_words]
set -e
OUT=${1:-/tmp/opencode/nantrace}
N=${2:-10}
WORDS=${3:-90}
ROCM=/home/pstreet/LocalAI/rocm-install
MRS=/home/pstreet/LocalAI/mistral.rs/target/release/mistralrs
MODEL_DIR=/home/pstreet/LocalAI/models/unsloth/Qwen3.5-4B-MTP-GGUF
PORT=18323
rm -rf "$OUT" && mkdir -p "$OUT"
(
  setsid env LD_LIBRARY_PATH=$ROCM/lib \
    $ROCM/bin/rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$OUT" \
    -- bash -c "cd $MODEL_DIR && exec $MRS serve -p $PORT -m . -f Qwen3.5-4B-Q8_0.gguf --prefix-cache-n 0" \
    > /tmp/opencode/serve4b_nantrace.log 2>&1 < /dev/null &
)
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q $PORT && break; sleep 2; done
ss -ltn 2>/dev/null | grep -q $PORT || { echo "server did not start"; exit 1; }

python3 - "$N" "$WORDS" <<'PY'
import json, sys, time, urllib.request
n, words = int(sys.argv[1]), int(sys.argv[2])
base = "the quick brown fox jumps over the lazy dog. "
prompt = base*words
outcomes = []
t_first = None
for i in range(n):
    body = {"model":"default","messages":[{"role":"user","content":prompt}],"max_tokens":2,"temperature":0,"stream":False}
    req = urllib.request.Request("http://localhost:18323/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            json.load(r)
        outcomes.append((i, t0, time.time(), "OK"))
    except Exception as e:
        outcomes.append((i, t0, time.time(), f"ERR:{str(e)[:40]}"))
    if t_first is None:
        t_first = t0
with open("/tmp/opencode/nantrace_outcomes.txt","w") as f:
    for i,t0,t1,s in outcomes:
        f.write(f"{i} {s} start={t0:.3f} end={t1:.3f}\n")
print("\n".join(f"{i}: {s}" for i,_,_,s in outcomes))
PY

MRS_PID=$(ss -ltnp 2>/dev/null | grep $PORT | grep -oP 'pid=\K[0-9]+' | head -1)
kill -TERM $MRS_PID 2>/dev/null || true
sleep 8
echo "=== outcomes ==="; cat /tmp/opencode/nantrace_outcomes.txt
CSV=$(find "$OUT" -name "*kernel_trace.csv" | head -1)
echo "=== analyzing $CSV ==="
python3 - "$CSV" <<'PY'
import csv, sys, hashlib
from collections import Counter
rows=[]
for r in csv.DictReader(open(sys.argv[1])):
    rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
# cluster kernels separated by >150ms gaps (prefill bursts vs idle/decode/http)
clusters=[]; cur=[rows[0]]
for prev,r in zip(rows, rows[1:]):
    if r[0]-prev[1] > 150_000_000:
        clusters.append(cur); cur=[]
    cur.append(r)
clusters.append(cur)
# drop tiny clusters (<20 kernels): warmup bits, decode stragglers
big=[c for c in clusters if len(c)>=20]
print(f"{len(clusters)} clusters, {len(big)} big ones")
for ci,c in enumerate(big):
    names=Counter(k for _,_,k in c)
    gpu=sum(e-s for s,e,_ in c)/1e6
    h=hashlib.md5(",".join(sorted(names.elements())).encode()).hexdigest()[:10]
    span=(c[-1][1]-c[0][0])/1e6
    top="; ".join(f"{n[:44]}x{ct}" for n,ct in names.most_common(4))
    print(f"\ncluster {ci}: kernels={len(c)} gpu={gpu:.1f}ms span={span:.1f}ms sig={h}")
    print(f"   {top}")
sigs={}
for ci,c in enumerate(big):
    names=Counter(k for _,_,k in c)
    sigs[ci]=hashlib.md5(",".join(sorted(names.elements())).encode()).hexdigest()[:10]
vals=list(sigs.values())
if len(set(vals))==1:
    print("\nALL CLUSTERS HAVE IDENTICAL KERNEL SEQUENCES")
else:
    print("\nKERNEL SEQUENCE DIFFERS BETWEEN CLUSTERS!")
    from collections import defaultdict
    bysig=defaultdict(list)
    for ci,s in sigs.items(): bysig[s].append(ci)
    ref=bigsig0=None
    sets={ci:Counter(k for _,_,k in c) for ci,c in enumerate(big)}
    base_ci=0
    for ci,s in enumerate(big):
        pass
    # diff each cluster vs cluster 0
    c0=sets[0]
    for ci in range(1,len(big)):
        d=sets[ci]-c0; d0=c0-sets[ci]
        if d or d0:
            print(f"cluster {ci} vs 0: extra={dict((k[:50],v) for k,v in d.items())} missing={dict((k[:50],v) for k,v in d0.items())}")
PY
