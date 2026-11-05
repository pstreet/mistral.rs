#!/usr/bin/env bash
# Wrapper script for the mistral.rs server with proper ROCm environment
# Full history + journal: mistral.rs/TODO.md (the old HANDOFF file was
# merged into it 2026-09-25)

set -euo pipefail

LOCALAI_ROOT="${LOCALAI_ROOT:-$HOME/LocalAI}"

# Source shared ROCm env (sets ROCM_PATH/HIP_PATH/LD_LIBRARY_PATH etc.)
if [ -f "$LOCALAI_ROOT/.rocm-env.sh" ]; then
    source "$LOCALAI_ROOT/.rocm-env.sh"
fi

# Decode graphs. Two ROCm capture bugs were fixed in cuda_graph.rs. NaN
# finding 2026-10-26: graph replay is safe ONLY with GPU_MAX_HW_QUEUES=1
# (single-queue A/B: 0/10 NaN vs ~10-25% on multi-queue, 5-sync-point fix
# in between). A/B 2026-09-24, qwen3.8-27b essay 600tok temp0.3 decode T/s,
# 3 trials each: on 13.89, off 12.54, on retest 13.39 (third trials sag with
# heat soak). Verdict: ~+7-10% real when replay is safe.
# GRAPHS OFF 2026-11-05: NaN class recurred under 6-way mixed-phase
# concurrency (2/2 repro with graphs on, solo gate armed) AND single-sequence
# long-context sessions still garbled under HIP replay (the morning deck
# session, 03:08/03:30 incident windows). The solo gate only closes the
# mixed-phase form. Re-enable only after the ROCm SDK capture-ordering fix,
# re-validated 3/3 on scratch/garble-hunt run-conc2.sh + the singles battery
# incl. single-seq long context. Escape lever stays MISTRALRS_CUDA_GRAPHS=0.
# (File reconciliation 2026-11-05: this wrapper previously claimed
# "GRAPH DISABLED 2026-11-03" while still exporting =1 - the comment and the
# export disagreed; the export is what ships, so the flag now actually reads
# 0 per the journal decision.)
# Server-core logs (lazy-skip warns, registration) are operationally
# relevant: keep its target enabled alongside core.
export RUST_LOG=mistralrs_core=info,mistralrs_server_core=info
export HIP_VISIBLE_DEVICES=0
# mmap GGUF shards instead of heap buffers: with MANAGED_WEIGHTS the upload
# still lands in a single host-visible allocation, DROP_HOST_AFTER_LOAD below
# evicts the file pages after upload (same steady state as heap), but the load
# transient is ~1x weights instead of ~2x, so big quants fit device mapping
# (Q8_0 37.8GB verified 2026-09-25; heap staging needed ~76GB and bailed).
# Set 1 to restore heap-buffer behavior.
export MISTRALRS_GGUF_NO_MMAP=0
# Release GGUF host shard buffers once weights are on-device, and evict the
# shard pages from file cache: keeps the single managed copy as the only copy.
export MISTRALRS_GGUF_DROP_HOST_AFTER_LOAD=1
# HIP managed (host-visible, prefetched) weight upload instead of device-only
# alloc + host->device copy: on this shared-memory APU the weights end up as
# a single copy in system RAM rather than a host copy plus a device copy.
# Fill still doubles transiently; DROP_HOST_AFTER_LOAD above releases the
# host side after upload. Trade-off: managed pages are CPU-mapped, so they
# count toward process RSS / cgroup / OOM accounting. ROCm-only. The two
# names split repo ownership: MISTRALRS_ gates the mistral.rs loader,
# CANDLE_ gates candle's allocator; the wrapper sets both so the deployment
# gets managed weights end to end.
export MISTRALRS_MANAGED_WEIGHTS=1
export CANDLE_MANAGED_WEIGHTS=1
export MISTRALRS_CUDA_GRAPHS=0
# 2026-11-05 reconciliation: the 2026-11-03 "GRAPH DISABLED" note below this
# line described intent, not state - the export above shipped =1 all along,
# with the solo gate carrying only the mixed-phase form. Now actually off.
# GRAPH DISABLED 2026-11-03: single-sequence long-context garble recurs under
# HIP graph replay (reasoning model producing corrupted tokens). Multi-sequence
# fix (solo gate) was deployed and validated (conc2 6/6 clean), but single-
# sequence batches still corrupt under graph replay. Re-enable when ROCm SDK
# fixes capture-ordering bug and we re-validate single-seq + long context.
# TEMP (capture-failure diagnostics, armed 2026-10-26): log every host upload
# over the 4KB graph H2D cache cap (with backtraces) and append backtraces to
# capture-path bails. The 07:47 capture failure class (>4KB upload inside the
# capture window) will name its exact uploader here on the next occurrence.
# Probe finding: hybrid checkpoint restores (3MB f32 + 80KB bf16 per capture
# step) dominate the log lines; they are legal post-capture uploads. The
# in-window culprit is expected to be something else (managed-weight page
# fault under pool trim is the working hypothesis). Remove once named.
#export MRS_TRACE_BIG_HTOD=1
# TEMP (35B graphs validation window): verify entry-held VAs before first
# launch; a moved/freed pointer becomes a loud named-buffer bail + eager
# fallback instead of an HSA fault. Remove after the graphs deployment is
# confirmed stable.
#export MISTRALRS_GRAPH_PTR_CENSUS=1
# TEMP (Nov 3 garble+OOM hunt): trace CK flash dispatch + prefix-gather
# prefill routing. eprintln output lands in scratch/prod-stderr.log.
# Remove once the attention dispatch under long prefill is mapped.
export MISTRALRS_DEBUG_CK=1
export MISTRALRS_DEBUG_PA=1
# Nov 3: NaN-logit class recurred under 6-way mixed-phase concurrency
# (short decodes + ~30k prefills, 2/2 repro with graphs on): argmax
# invalid logits + HSA_STATUS_ERROR_MEMORY_FAULT -> death/restart.
# Root cause: graph replay corrupts under mixed prefill+decode batches;
# single-shape steps are clean. Interim fix (deployed 2026-11-03,
# binary-side per-role gating): NVIDIA role keeps graphs; the HIP role
# replays only on solo decode steps (one decoding sequence, nothing
# waiting) when MISTRALRS_CUDA_GRAPHS_HIP_SOLO=1 is set, and it captures
# only bucket-1 shapes. Mixed steps fall back to eager; /metrics shows
# dispatch reason hip_solo_gate on those skips. REVISIT on the next
# ROCm SDK bump: unset HIP_SOLO, re-run scratch/garble-hunt run-conc2.sh
# + the singles battery, and restore full HIP graphs only if 3/3 clean
# with no hip error 905 capture-ordering signature. Escape lever remains
# MISTRALRS_CUDA_GRAPHS=0.
export MISTRALRS_CUDA_GRAPHS_HIP_SOLO=1
# Step sentinel (TODO 2026-11-05): fail requests whose decode step leaves the
# sane numeric band (greedy logit/logprob ceiling, GDN state health) instead
# of serving the subcritical word-salad form of the NaN class. Set 0 to disarm.
# Ceiling 64.0: healthy 35B greedy saturates ~33.5 (canary 2026-11-05); the
# first arm at 32.0 500'd that canary and was rebuilt the same morning.
export MISTRALRS_STEP_SENTINEL=1
export ROCBLAS_USE_HIPBLASLT=1
# GPU_MAX_HW_QUEUES=1 required: solo gate blocks mixed-phase corruption
# (Nov 3 root cause), but multi-queue still causes NaN on single-sequence
# decode steps (Oct 26 finding, unaddressed). See TODO.md REVISIT section.
export GPU_MAX_HW_QUEUES=1
export ROCP_TOOL_ATTACH=1
# Restored 2026-10-26: graphs require single-queue scheduling on this
# runtime (multi-queue graph replay races: 0/10 NaN single-queue vs
# ~10-25% multi-queue, sync fixes only closed the H2D->launch edge).
# Power/heat tradeoff accepted for the ~7-10% graphs decode win.
# TheRock#7578 (power-gating pin) remains open. If TheRock 10.2 fixes
# the multi-queue graph race, re-test without this line.

MISTRALRS="$LOCALAI_ROOT/mistral.rs/target/release/mistralrs"
CONFIG="$LOCALAI_ROOT/models/mistralrs.toml"

if [ ! -x "$MISTRALRS" ]; then
    echo "Error: mistralrs not found at $MISTRALRS. Build it first." >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found at $CONFIG." >&2
    exit 1
fi

# Model paths in the toml are relative: resolve them from $LOCALAI_ROOT.
cd "$LOCALAI_ROOT"

# TEMP (fault hunt, 2026-10-13): stderr carries the alloc-trace/pool-trim/
# census eprintln diagnostics; journald's rate limiter swallowed the whole
# boot flood tonight (fault #9's ledger + even the INFO lines were dropped),
# so stderr goes to a file instead. stdout (tracing INFO) stays on journald.
# Remove with the census/ledger diagnostics.
# EDITED 2026-11-03: re-enable journald stderr for timestamped error correlation.
exec "$MISTRALRS" from-config -f "$CONFIG"