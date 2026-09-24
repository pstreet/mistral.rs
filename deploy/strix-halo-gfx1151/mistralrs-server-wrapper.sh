#!/usr/bin/env bash
# Wrapper script for the mistral.rs server with proper ROCm environment
# Full perf history: $HOME/LocalAI/HANDOFF-mistralrs-candle-rocm.md

set -euo pipefail

LOCALAI_ROOT="${LOCALAI_ROOT:-$HOME/LocalAI}"

# Source shared ROCm env (sets ROCM_PATH/HIP_PATH/LD_LIBRARY_PATH etc.)
if [ -f "$LOCALAI_ROOT/.rocm-env.sh" ]; then
    source "$LOCALAI_ROOT/.rocm-env.sh"
fi

# Decode graphs. Two ROCm capture bugs were fixed in cuda_graph.rs; override
# with MISTRALRS_CUDA_GRAPHS=0 if a crash recurs. A/B 2026-09-24, qwen3.8-27b
# essay 600tok temp0.3 decode T/s, 3 trials each: on 13.89, off 12.54, on
# retest 13.39 (third trials sag with heat soak). Verdict: ~+7-10% real; the
# old +2.4% dated from when launches were silently zero. /metrics showed
# 1183 replay vs 3 eager dispatches. Keep on.
export RUST_LOG=mistralrs_core=info
export HIP_VISIBLE_DEVICES=0
# Read GGUF shards into heap buffers instead of mmap: with MANAGED_WEIGHTS the
# upload has no device-side duplicate, and DROP_HOST_AFTER_LOAD below releases
# the buffers and purges the allocator heap, so the transient is handed back
# instead of retained. Set 0 to restore mmap behavior.
export MISTRALRS_GGUF_NO_MMAP=1
# Release GGUF host shard buffers once weights are on-device, and evict the
# shard pages from file cache: keeps the single managed copy as the only copy.
export MISTRALRS_GGUF_DROP_HOST_AFTER_LOAD=1
# HIP managed (host-visible, prefetched) weight upload instead of device-only
# alloc + host->device copy: on this shared-memory APU the weights end up as
# a single copy in system RAM rather than a host copy plus a device copy.
# Fill still doubles transiently; DROP_HOST_AFTER_LOAD above releases the
# host side after upload. Trade-off: managed pages are CPU-mapped, so they
# count toward process RSS / cgroup / OOM accounting. ROCm-only.
export MISTRALRS_MANAGED_WEIGHTS=1
export MISTRALRS_CUDA_GRAPHS=1
# hipBLASLt with F32 accumulate: COMPUTE_16F fails heuristically (error 6,
# INTERNAL_ERROR) on some shapes on gfx1151 RDNA3.5, and 32F accum is faster
# anyway. candle falls back to plain rocBLAS on any LT error, so residual
# failures cost speed, not availability.
export ROCBLAS_USE_HIPBLASLT=1

# Prefill GEMMs: keep llama MMQ off so large-batch quantized GEMMs run as
# dequantize+hipBLASLt (~125 TFLOPS vs ~15 for MMQ on gfx1151). 23k prefill
# went 268 -> 410 T/s with the MMQ fix (2535aa399). Decode and small batches
# (<= 8 tokens) still use the fused MMVQ kernels, so decode is unaffected.
export MRS_NO_FAST_MMQ=1
export CANDLE_NO_FAST_MMQ=1
export CANDLE_DMM_F16_MIN=8
# LT compute: 32 = F32 accum (stable on RDNA, required).
export CANDLE_LT_COMPUTE=32

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

exec "$MISTRALRS" from-config -f "$CONFIG"