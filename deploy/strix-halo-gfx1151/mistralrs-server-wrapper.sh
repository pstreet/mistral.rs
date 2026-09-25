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
# count toward process RSS / cgroup / OOM accounting. ROCm-only.
export MISTRALRS_MANAGED_WEIGHTS=1
export MISTRALRS_CUDA_GRAPHS=1
# Route rocBLAS's own GEMMs through hipBLASLt. Serving A/B 2026-09-25
# (35B Q4_K_XL): flat (decode profile has zero BLAS kernels; all-quantized
# weights never reach plain rocBLAS). Microbench (lt_probe, 78 shapes/dtype)
# says it is worth +13% f16 / +19% bf16 when GEMMs DO reach rocBLAS, e.g. a
# dense BF16 safetensors model. No-op today, insurance for that case; keep.
export ROCBLAS_USE_HIPBLASLT=1

# Large-batch quantized GEMMs: MMQ off -> dequant-to-F16 + hipBLASLt routing.
# Restored 2026-09-25: the fast_mmq removal tried that morning helped the 35B
# (MoE Q4/Q8: 60tok 896 vs 762, +15%) but was a REGRESSION on the 27B dense
# Q6_K default: 18k cold prefill 426 T/s (MMQ off) vs 243 T/s (on) = -43%.
# rocprofv3 shows mul_mat_q<q6_k> at 82.5% of the 27B's prefill GPU time
# (~11 effective TFLOPS vs ~30 via dequant+Lt). Decode (b*m <= 8) uses the
# fused MMVQ kernels either way. Drop this trio again once the per-dtype
# dispatch (Q6_K -> dequant, Q4_K/Q5_K/Q8_0 -> mmq) lands and is verified.
export MRS_NO_FAST_MMQ=1
export CANDLE_NO_FAST_MMQ=1
# Minimum rows for the dequant-to-F16 GEMM path; prefill chunks (4096) clear
# it, decode rows (1-8) stay on MMVQ.
export CANDLE_DMM_F16_MIN=8
# CANDLE_LT_COMPUTE removed 2026-09-25: an 80-shape hipBLASLt sweep
# (candle-core example lt_probe; b=1 m 256-8192 k/n 2048-16384, odd shapes,
# b 3-16 batched; 16F vs 32F compute) found zero heuristic failures on the
# current stack and 16F is not slower (mean +1.2%). The old "COMPUTE_16F
# fails with INTERNAL_ERROR, 32F required/faster" claim dated from the
# pre-TheRock hipBLASLt and does not apply.

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