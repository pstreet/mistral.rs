#!/usr/bin/env bash
#
# Systemd wrapper for the mistral.rs server binary.
#
# Expected layout:
#   parent/
#     mistral.rs/
#       scripts/
#         mistralrs-server-wrapper.sh  ← this file
#         .rocm-env.sh                 ← (optional) ROCm/HIP environment
#       target/release/mistralrs       ← built binary
#     models/
#       mistralrs.toml                 ← server config
#
# Usage:
#   1. Copy to ~/.config/mistralrs-server-wrapper.sh or wherever
#      your systemd service's ExecStart= expects.
#   2. Set LOCALAI_ROOT to your parent directory (default: ~/LocalAI).
#   3. Adjust env vars below to taste. The defaults are tuned for
#      ROCm/Strix Halo (gfx1151, 128GB host RAM) — NVIDIA users may
#      want to remove ROCm-specific vars.
#
# This is an example wrapper; prod deployments should customize the
# env vars and paths. See the comments in each section for guidance.

set -euo pipefail

# --------------- layout ---------------
# LOCALAI_ROOT is the parent of mistral.rs/ and models/.
# The wrapper sources .rocm-env.sh if present (sets ROCM_PATH, etc.).
LOCALAI_ROOT="${LOCALAI_ROOT:-$HOME/LocalAI}"

if [ -f "$LOCALAI_ROOT/.rocm-env.sh" ]; then
    source "$LOCALAI_ROOT/.rocm-env.sh"
fi

# --------------- logging ---------------
export RUST_LOG=mistralrs_core=info,mistralrs_server_core=info

# --------------- device selection ---------------
# Which GPU to use (0 = first device).
export HIP_VISIBLE_DEVICES=0

# --------------- GGUF loading ---------------
# Use mmap for GGUF shards (lower load transient). 0 = mmap, 1 = heap.
export MISTRALRS_GGUF_NO_MMAP=0

# Release host buffers after weight upload, evict file-cache pages.
# Keeps only the managed host-visible copy on-device.
export MISTRALRS_GGUF_DROP_HOST_AFTER_LOAD=1

# Managed (host-visible, prefetched) weight upload. On shared-memory
# APUs this avoids a double-copy (host->device). Trade-off: managed
# pages count toward RSS / cgroup / OOM accounting.
export MISTRALRS_MANAGED_WEIGHTS=1
export CANDLE_MANAGED_WEIGHTS=1

# --------------- decode graphs ---------------
# GRAPH DISABLED 2026-11-03: single-sequence long-context garble recurs
# under HIP graph replay. Multi-sequence fix (solo gate) was deployed and
# validated, but single-sequence batches still corrupt. Re-enable when
# ROCm SDK fixes capture-ordering bug.
export MISTRALRS_CUDA_GRAPHS=0
# export MISTRALRS_CUDA_GRAPHS_HIP_SOLO=1   # not needed when graphs=0

# --------------- debug / diagnostics ---------------
# Uncomment to enable (production: leave commented).
# export MISTRALRS_DEBUG_CK=1
# export MISTRALRS_DEBUG_PA=1
# export MISTRALRS_GRAPH_PTR_CENSUS=1
# export MRS_TRACE_BIG_HTOD=1

# --------------- ROCm tuning ---------------
export ROCBLAS_USE_HIPBLASLT=1

# GPU_MAX_HW_QUEUES=1 required on this runtime (ROCm driver bug).
# Default is 4; HIP reuses hardware queues round-robin when streams
# exceed the limit, causing queue-ordering corruption in graph replay.
# See: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html
# See: https://github.com/ROCm/legacy-rocm-build/issues/2625
export GPU_MAX_HW_QUEUES=1

# Attach rocprofiler for profiling (disable for production).
export ROCP_TOOL_ATTACH=1

# --------------- binary + config ---------------
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

# --------------- exec ---------------
# Redirect stderr to a log file (journald rate-limits can swallow boot floods).
# stdout (tracing INFO) stays on journald. Adjust the path as needed.
exec "$MISTRALRS" from-config -f "$CONFIG" 2>> "$LOCALAI_ROOT/scratch/prod-stderr.log"
