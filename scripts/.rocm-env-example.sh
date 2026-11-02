# .rocm-env.sh — Example ROCm/HIP environment setup
#
# Source this file before any ROCm build:
#   source scripts/.rocm-env.sh
#
# If ROCm is installed at a non-standard path, set ROCM_PATH before sourcing:
#   ROCM_PATH=/opt/rocm source scripts/.rocm-env.sh
#
# This is an example file. Adjust paths and architecture targets to match
# your hardware. The defaults target AMD RDNA 3 (gfx1151 / RX 7900 XTX).
#
# For other GPUs, change HSA_OVERRIDE_GFX_VERSION and PYTORCH_ROCM_ARCH:
#   gfx1100 (RX 7600/7700/7800):  HSA_OVERRIDE_GFX_VERSION=11.0.0
#   gfx1151 (RX 7900 XTX/XT):     HSA_OVERRIDE_GFX_VERSION=11.5.1
#   gfx1200 (RX 9070):            HSA_OVERRIDE_GFX_VERSION=12.0.0

if [ -z "${ROCM_PATH:-}" ]; then
    if [ -d "/opt/rocm" ]; then
        export ROCM_PATH="/opt/rocm"
    else
        echo "Error: ROCM_PATH not set and /opt/rocm not found." >&2
        echo "Install ROCm or set ROCM_PATH before sourcing this file." >&2
        exit 1
    fi
fi

if [ -z "${ROCM_HOME:-}" ]; then export ROCM_HOME="$ROCM_PATH"; fi
if [ -z "${HIP_PATH:-}" ]; then export HIP_PATH="$ROCM_PATH"; fi

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export AMDGPU_TARGETS=gfx1151
export HSA_ENABLE_SDMA=1

# ---- Core ROCm/HIP paths ----
export ROCM_HOME="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

# ---- Target GPU architecture ----
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}"

# ---- Shared runtime tuning ----
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-1}"

# ---- ROCm device libraries ----
export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/lib/llvm/amdgcn/bitcode"
export ROCM_DEVICE_LIB_PATH="${ROCM_PATH}/lib/llvm/amdgcn/bitcode"

# ---- CMake configuration ----
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/llvm/lib/cmake${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
