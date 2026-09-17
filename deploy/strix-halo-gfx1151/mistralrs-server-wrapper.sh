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
# with MISTRALRS_CUDA_GRAPHS=0 if a crash recurs. NOTE 2026-09-13: rocprof
# showed zero graph launches on this ROCm path (all eager), so this flag is
# currently a no-op here; the earlier +2.4% reading may have been noise.
export RUST_LOG=mistralrs_core=info
export HIP_VISIBLE_DEVICES=0
# Read GGUF shards via read() instead of mmap: on this Strix Halo the iGPU
# shares system RAM, so the ~23 GB file-cache copy of the weights is duplicated
# by the device-side copy. Set 0 to restore mmap behavior.
export MISTRALRS_GGUF_NO_MMAP=1
# Release GGUF host shard buffers once weights are on-device: with NO_MMAP the
# ~23 GB Owned copy would otherwise stay resident next to the device copy.
export MISTRALRS_GGUF_DROP_HOST_AFTER_LOAD=1
# Device-only GGUF weight tensors (plain HIP device alloc + host->device copy):
# on this shared-memory APU the pages still come from system RAM, but they
# are not CPU-mapped, so they stay out of process RSS / cgroup / OOM
# accounting. Costs a load-time 2x spike during the copy. Set 1 to restore
# managed (host-visible, prefetch) weights.
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