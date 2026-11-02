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

# Default feature set — every feature, all activation scopes. Override with
# MISTRALRS_FEATURES. metal/accelerate are Apple-only and dropped on Linux;
# each feature stays enabled in every shape and only executes on the
# devices/platforms that support it.
case "$(uname -s)" in
    Darwin)
        MISTRALRS_FEATURES_DEFAULT="metal,accelerate,ring"
        ;;
    *)
        MISTRALRS_FEATURES_DEFAULT="cuda,rocm,npu,cutile,nccl,cudnn,flash-attn,flash-attn-v3,ring,mkl"
        ;;
esac
MISTRALRS_FEATURES_EXPLICIT="${MISTRALRS_FEATURES:+1}"
export MISTRALRS_FEATURES="${MISTRALRS_FEATURES:-$MISTRALRS_FEATURES_DEFAULT}"

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

# ── Feature dependency checks ───────────────────────────────────────────────
# Every feature must compile in every build shape, but features that need
# system toolchains/libraries are validated here before cargo runs. Missing
# pieces that have a clean apt path are auto-installable; everything else
# fails with actionable setup instructions.
#
# GPU_DEPS_INSTALL controls the apt auto-install consent:
#   ask     (default: prompt only on an interactive terminal)
#   always  (no prompt — install anything missing)
#   never   (no install; missing deps are hard errors)
has_feature() { echo " ${MISTRALRS_FEATURES//,/ } " | grep -q " $1 "; }

apt_install() {
    case "${GPU_DEPS_INSTALL:-}" in
        never) return 1 ;;
        always) ;;
        *)
            [ -t 0 ] || return 1
            local reply
            echo "Missing system dependency: $*"
            read -r -p "Install it now via apt? [y/N] " reply || return 1
            case "$reply" in [Yy]*) ;; *) return 1 ;; esac
            ;;
    esac
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
}

fail_dep() {
    echo "Error: $*" >&2
    echo "  Override with MISTRALRS_FEATURES, or set GPU_DEPS_INSTALL=never and" >&2
    echo "  install the dependency manually to keep it out of the build flow." >&2
    exit 1
}

# ── NVIDIA toolchain (cuda, cutile, cudnn, nccl, flash-attn, flash-attn-v3) ─
needs_nvcc() {
    has_feature cuda || has_feature cutile || has_feature cudnn \
        || has_feature nccl || has_feature flash-attn || has_feature flash-attn-v3
}

if needs_nvcc; then
    nvcc_detected=""
    if command -v nvcc &>/dev/null; then
        nvcc_detected="$(command -v nvcc)"
    else
        for d in "${CUDA_HOME:-}" /etc/alternatives/cuda /usr/local/cuda /usr/local/cuda-*; do
            [ -n "$d" ] && [ -x "$d/bin/nvcc" ] || continue
            CUDA_HOME="$(readlink -f "$d")"
            case ":$PATH:" in
                *":$CUDA_HOME/bin:"*) ;;
                *) export PATH="$CUDA_HOME/bin:$PATH" ;;
            esac
            nvcc_detected="$CUDA_HOME/bin/nvcc"
            break
        done
    fi
    if [ -z "$nvcc_detected" ]; then
        fail_dep "NVIDIA toolchain features are enabled but nvcc was not found.
  Features: $MISTRALRS_FEATURES
  Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads) or
  point CUDA_HOME at an existing toolkit root."
    fi
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$nvcc_detected")")}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    # Target the real GPU when one is visible (nvidia-smi reports e.g. "9.0");
    # CPU-only/buried-toolkit hosts default to sm80 compile targets.
    if [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
        detected_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ./')"
        export CUDA_COMPUTE_CAP="${detected_cc:-80}"
    fi
    export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-80}"
    # flash-attn-v3 compile is hard-gated at sm90 in candle-flash-attn-v3
    # (Hopper kernels). When the detected target is older, drop it from the
    # default set; an explicit request for it is a hard error instead.
    if has_feature flash-attn-v3 && [ "${CUDA_COMPUTE_CAP:-80}" -lt 90 ]; then
        if [ -n "${MISTRALRS_FEATURES_EXPLICIT:-}" ]; then
            fail_dep "flash-attn-v3 needs an sm90 GPU target (CUDA_COMPUTE_CAP=90,
  auto-detected ${detected_cc:-none}, using ${CUDA_COMPUTE_CAP:-80}). Point it at
  an H100-class host or drop the feature."
        fi
        echo "Note: dropping flash-attn-v3 (needs an sm90 target; detected ${detected_cc:-none}, using sm${CUDA_COMPUTE_CAP:-80})"
        MISTRALRS_FEATURES="$(echo " $MISTRALRS_FEATURES " | sed 's/, */ /g;s/flash-attn-v3 *//;s/^ *//;s/ *$//;s/ \+/ /g;s/ /,/g;s/,$//;s/^,//;s/,,*/,/g')"
        export MISTRALRS_FEATURES
    fi
    echo "nvcc: $($nvcc_detected --version 2>/dev/null | tail -1) ($nvcc_detected)"
    echo "CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-80}"
fi

# ── cuDNN ────────────────────────────────────────────────────────────────────
cudnn_ok() {
    local cache
    cache="$(ldconfig -p 2>/dev/null)" || true
    grep -q "libcudnn.so.9" <<< "$cache" && return 0
    compgen -G "$CUDA_HOME/lib64/libcudnn.so.9" &>/dev/null && return 0
    for d in /usr/lib/x86_64-linux-gnu /usr/local/lib; do
        compgen -G "$d/libcudnn.so.9" &>/dev/null && return 0
    done
    return 1
}

if has_feature cudnn; then
    if ! cudnn_ok; then
        cuda_major="$($nvcc_detected --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
        cuda_minor="$($nvcc_detected --version 2>/dev/null | sed -n 's/.*release [0-9]*\.\([0-9]*\).*/\1/p')"
        for pkg in "cudnn9-cuda-${cuda_major}-${cuda_minor}" "cudnn9-cuda-${cuda_major}" cudnn9; do
            apt-cache show "$pkg" &>/dev/null && apt_install "$pkg" && break
        done
        cudnn_ok || fail_dep "libcudnn.so.9 not found for the cudnn feature.
  Installed via apt (cudnn9-cuda-*) or download from
  https://developer.nvidia.com/cudnn-downloads and extract so that libcudnn
  is on the linker path."
    fi
    echo "cuDNN: OK"
fi

# ── NCCL ─────────────────────────────────────────────────────────────────────
nccl_ok() {
    local cache
    cache="$(ldconfig -p 2>/dev/null)" || true
    grep -q "libnccl.so.2" <<< "$cache"
}

if has_feature nccl; then
    if ! nccl_ok; then
        apt_install libnccl2 libnccl-dev
        nccl_ok || fail_dep "libnccl.so.2 not found for the nccl feature.
  Install from the NVIDIA CUDA repo: https://docs.nvidia.com/deeplearning/nccl
  (packages: libnccl2 libnccl-dev)."
    fi
    echo "NCCL: OK"
fi

# ── ROCm toolchain (hipcc + HIP device bitcode for the kernels build) ───────
if has_feature rocm; then
    if ! command -v hipcc &>/dev/null; then
        for d in "${ROCM_HOME:-${ROCM_PATH:-/opt/rocm}}" /opt/rocm /opt/rocm-*; do
            [ -n "$d" ] && [ -x "$d/bin/hipcc" ] || continue
            export ROCM_HOME="$(readlink -f "$d")"
            case ":$PATH:" in
                *":$ROCM_HOME/bin:"*) ;;
                *) export PATH="$ROCM_HOME/bin:$PATH" ;;
            esac
            break
        done
    fi
    command -v hipcc &>/dev/null || fail_dep "hipcc was not found for the rocm feature.
  ROCM_PATH=$ROCM_PATH
  Install ROCm (https://rocm.docs.amd.com/projects/install-on-linux) or source
  the .rocm-env.sh sibling next to this script."
    hip_bitcode="${ROCM_HOME:-$ROCM_PATH}/lib/llvm/amdgcn/bitcode"
    [ -d "$hip_bitcode" ] || fail_dep "HIP device bitcode missing: $hip_bitcode
  (the candle-kernels hipcc build links against the amdgcn bitcode libraries)."
    echo "hipcc: $(hipcc --version 2>/dev/null | head -1)"
fi

# ── Apple-only features ──────────────────────────────────────────────────────
if has_feature metal || has_feature accelerate; then
    [ "$(uname -s)" = "Darwin" ] || fail_dep "metal/accelerate features require macOS
  (Apple Accelerate / Metal frameworks are not available on $(uname -s))."
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
