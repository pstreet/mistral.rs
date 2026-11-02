#!/usr/bin/env bash
#
# Build mistral.rs server binary with CUDA, ROCm, NPU support.
#
# Expected layout:
#   parent/
#     candle/                ← candle-core (required sibling dependency)
#     mistral.rs/
#       scripts/
#         .rocm-env.sh       ← (optional) ROCm/HIP environment setup
#         .rocm-env-example.sh ← example, copy and edit to .rocm-env.sh
#         build.sh           ← this script
#
# Usage:
#   cd mistral.rs/scripts && ./build.sh
#   MISTRALRS_FEATURES=rocm ./build.sh          # ROCm only
#   MISTRALRS_FEATURES=cuda,rocm,npu ./build.sh  # multi-vendor
#
# Environment variables:
#   MISTRALRS_FEATURES   - Cargo features to build (default: cuda,rocm,npu,flash-attn)
#   CANDLE_ROCM_PATH     - Override path to ROCm installation
#   CUDA_PATH            - Override path to CUDA toolkit
#   NPU_VENV             - Path to MLIR-AIE venv (default: ~/.local/share/mistralrs-npu)
#   CARGO_BUILD_JOBS     - Parallelism (default: number of CPUs)
#   USE_MOLD             - Use mold linker (default: 1 if available)

set -exuo pipefail
trap 'echo "Error on line $LINENO"' ERR

# Resolve script directory and parent (where .rocm-env.sh may live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MISTRALRS_ROOT="$SCRIPT_DIR/.."
PARENT_DIR="$(cd "$MISTRALRS_ROOT/.." && pwd)"

# LOCALAI_ROOT is used by .rocm-env.sh for path resolution
export LOCALAI_ROOT="${LOCALAI_ROOT:-$PARENT_DIR}"

# Source shared ROCm environment (prefer sibling in repo, then parent, then fallback)
if [ -f "$SCRIPT_DIR/.rocm-env.sh" ]; then
    source "$SCRIPT_DIR/.rocm-env.sh"
elif [ -f "$PARENT_DIR/.rocm-env.sh" ]; then
    source "$PARENT_DIR/.rocm-env.sh"
else
    echo "Warning: .rocm-env.sh not found. Using defaults (ROCM_PATH=$ROCM_PATH)."
fi

# Pin Rust build.rs ROCm discovery
export CANDLE_ROCM_PATH="${CANDLE_ROCM_PATH:-$ROCM_PATH}"
export CANDLE_ROCM_ARCH="${CANDLE_ROCM_ARCH:-$PYTORCH_ROCM_ARCH}"

# ccache for hipcc/nvcc companion-lib compiles
export MISTRALRS_CCACHE_GPU="${MISTRALRS_CCACHE_GPU:-1}"

# Default feature set — override with MISTRALRS_FEATURES env var
export MISTRALRS_FEATURES="${MISTRALRS_FEATURES:-cuda,rocm,npu,flash-attn}"

# ── CUDA toolkit ────────────────────────────────────────────────────────────
if echo "${MISTRALRS_FEATURES:-}" | grep -qw cuda; then
    # Add nvcc to PATH if available
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        case ":$PATH:" in
            *:/usr/local/cuda/bin:*) ;;
            *) export PATH="/usr/local/cuda/bin:$PATH" ;;
        esac
    fi
    # Default compute capability (override with CUDA_COMPUTE_CAP)
    export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-80}"
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
fi

# ── Validate directory layout ───────────────────────────────────────────────
# candle is a required sibling dependency (path dep in Cargo.toml)
for dir in "candle"; do
    if [ ! -d "$PARENT_DIR/$dir" ]; then
        echo "Error: Required directory not found: $PARENT_DIR/$dir" >&2
        echo "Expected layout:" >&2
        echo "  parent/" >&2
        echo "    candle/            ← candle-core (sibling dependency)" >&2
        echo "    mistral.rs/" >&2
        echo "      scripts/build.sh ← this script" >&2
        exit 1
    fi
done

cd "$MISTRALRS_ROOT"

# ── NPU distro dependencies (AMD Ryzen AI / XDNA) ──────────────────────────
# These packages provide the XRT runtime, dev headers, firmware, and pyxrt
# bindings — all from Ubuntu Universe or the kernel tree. No custom build
# needed; the Rust crate dlopens libxrt_core.so.2 at runtime.
if echo "${MISTRALRS_FEATURES:-}" | grep -qw npu; then
    echo "=== Installing NPU (AMD XDNA) distro dependencies ==="
    sudo apt-get update -qq
    sudo apt-get install -y \
        libxrt-dev \
        libxrt2 \
        libxrt-npu2 \
        python3-xrt \
        linux-firmware-amd-misc
    # Verify NPU device is visible
    if [ ! -e /dev/accel/accel0 ]; then
        echo "Warning: /dev/accel/accel0 not found — NPU not available on this host" >&2
    fi
fi

# ── MLIR-AIE + Peano toolchain (for AIE kernel compilation on NPU) ─────────
# This is the open-source compiler for writing AIE kernels (GEMM, attention,
# RMSNorm, etc.) that run on the Ryzen AI NPU. Requires Python 3.11+.
#
# Install via pip:
#   python3 -m venv ~/.local/share/mistralrs-npu
#   source ~/.local/share/mistralrs-npu/bin/activate
#   pip install mlir_aie==1.4.4 llvm-aie \
#     -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
#
# MLIR-AIE version: 1.4.4 (released Oct 2026, includes AIE2P/XDNA2 support)
# Peano (llvm-aie) version: 22.0.0.2026092101+0006955e (nightly pinned to mlir-aie)
NPU_VENV="${NPU_VENV:-$HOME/.local/share/mistralrs-npu}"

if echo "${MISTRALRS_FEATURES:-}" | grep -qw npu; then
    echo "=== Checking MLIR-AIE + Peano (AIE kernel compiler) toolchain ==="

    if [ ! -d "$NPU_VENV" ]; then
        echo "=== MLIR-AIE venv not found at $NPU_VENV ==="
        echo "Install manually:"
        echo "  python3 -m venv $NPU_VENV"
        echo "  source $NPU_VENV/bin/activate"
        echo "  pip install mlir_aie==1.4.4 llvm-aie \\"
        echo "    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
        echo "Error: MLIR-AIE/Peano toolchain required for NPU feature" >&2
        exit 1
    fi

    # Source the venv
    source "$NPU_VENV/bin/activate"

    # Check aiecc (MLIR-AIE compiler driver)
    if ! command -v aiecc &>/dev/null; then
        echo "Error: aiecc not found in venv at $NPU_VENV" >&2
        echo "  Try: pip install mlir_aie==1.4.4" >&2
        exit 1
    fi
    echo "  aiecc: $(aiecc --version 2>&1 | head -1)"

    # Check Peano clang (llvm-aie / Peano per-core compiler)
    # Try common Python versions, then fall back to find
    PEANO_CLANG=""
    for pyver in python3.13 python3.12 python3.11 python3.10; do
        candidate="$NPU_VENV/lib/$pyver/site-packages/llvm-aie/bin/clang"
        if [ -x "$candidate" ]; then
            PEANO_CLANG="$candidate"
            break
        fi
    done
    # Fallback: find any clang in llvm-aie bin
    if [ -z "$PEANO_CLANG" ]; then
        PEANO_CLANG=$(find "$NPU_VENV" -name clang -path "*/llvm-aie/bin/*" -type f 2>/dev/null | head -1)
    fi
    if [ -z "$PEANO_CLANG" ] || [ ! -x "$PEANO_CLANG" ]; then
        echo "Error: Peano clang not found in $NPU_VENV" >&2
        echo "  Try: pip install llvm-aie" >&2
        exit 1
    fi
    echo "  Peano clang: $($PEANO_CLANG --version 2>&1 | head -1)"

    # Export paths for downstream tools
    export PATH="$NPU_VENV/bin:$PATH"
    export PEANO_INSTALL_DIR="$NPU_VENV/lib/python3.13/site-packages/llvm-aie"
    if [ ! -d "$PEANO_INSTALL_DIR" ]; then
        # Fallback: derive from the found clang
        PEANO_INSTALL_DIR="$(dirname "$(dirname "$PEANO_CLANG")")"
        export PEANO_INSTALL_DIR
    fi

    echo "=== MLIR-AIE + Peano toolchain OK ==="
fi

# ── CUDA toolkit auto-install (optional) ────────────────────────────────────
# Only needed when cuda feature is requested and nvcc is not already available.
if echo "${MISTRALRS_FEATURES:-}" | grep -qw cuda; then
    if ! command -v nvcc &>/dev/null && [ -z "${CUDA_HOME:-}" ]; then
        echo "=== CUDA feature requested but nvcc not found — installing toolkit ==="
        sudo apt-get update -qq
        sudo apt-get install -y cuda-toolkit
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export PATH="${CUDA_HOME}/bin:$PATH"
    fi
fi

# ── ROCm toolkit auto-install (optional) ────────────────────────────────────
# Only needed when rocm feature is requested and hipcc is not already available.
if echo "${MISTRALRS_FEATURES:-}" | grep -qw rocm; then
    if ! command -v hipcc &>/dev/null && [ -z "${ROCM_HOME:-}" ] && [ -z "${HIP_PATH:-}" ]; then
        echo "=== ROCm feature requested but hipcc not found — installing toolkit ==="
        sudo apt-get update -qq
        sudo apt-get install -y rocm-dev hipcc
        export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
        export ROCM_HOME="$ROCM_PATH"
        export HIP_PATH="$ROCM_PATH"
        export PATH="${ROCM_PATH}/bin:${PATH}"
    fi
fi

# ── Build configuration ─────────────────────────────────────────────────────
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$(nproc)}"

# Rust caching: ccache cannot cache rustc, so use sccache when available.
if command -v sccache &>/dev/null; then
    export RUSTC_WRAPPER=sccache
    export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-100G}"
fi

# Fast host linker (mold) for the final binary link
if [ "${USE_MOLD:-1}" = "1" ] && command -v mold &>/dev/null; then
    case "${RUSTFLAGS:-}" in
        *-fuse-ld=mold*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C link-arg=-fuse-ld=mold" ;;
    esac
else
    case "${RUSTFLAGS:-}" in
        *target-cpu=native*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" ;;
    esac
fi

if echo "${MISTRALRS_FEATURES:-}" | grep -qw rocm; then
    echo "=== Using ROCm from: $ROCM_PATH ==="
fi
echo "=== Features: $MISTRALRS_FEATURES ==="
cargo --version
rustc --version

echo "Building mistralrs server binary (and candle via path deps)..."
cargo build --release --features "$MISTRALRS_FEATURES" -p mistralrs-cli

echo "Verifying binary..."
./target/release/mistralrs --help >/dev/null
ls -la ./target/release/mistralrs

echo ""
echo "=== mistral.rs build complete: mistral.rs/target/release/mistralrs ==="
