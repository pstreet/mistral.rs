# Build scripts

## build.sh
All-in-one build script for mistral.rs with CUDA, ROCm, and NPU support.

### Prerequisites
```bash
# Required sibling directory (candle-core)
git clone https://github.com/huggingface/candle.git
```

### Usage
```bash
# Default (multi-vendor with flash attention)
./scripts/build.sh

# ROCm only
MISTRALRS_FEATURES=rocm ./scripts/build.sh

# CUDA + flash-attn
MISTRALRS_FEATURES=cuda,flash-attn ./scripts/build.sh

# NPU only (CPU-only build for AIE kernel cross-compile)
MISTRALRS_FEATURES=npu ./scripts/build.sh

# CPU only
MISTRALRS_FEATURES= ./scripts/build.sh
```

### Environment variables
| Variable | Description | Default |
|----------|-------------|---------|
| `MISTRALRS_FEATURES` | Cargo features to build | `cuda,rocm,npu,flash-attn` |
| `CANDLE_ROCM_PATH` | Override ROCm install path | `$ROCM_PATH` |
| `CUDA_PATH` | Override CUDA toolkit path | `/usr/local/cuda` |
| `CUDA_COMPUTE_CAP` | CUDA compute capability | `80` |
| `NPU_VENV` | MLIR-AIE Python venv | `~/.local/share/mistralrs-npu` |
| `CARGO_BUILD_JOBS` | Parallelism | `nproc` |
| `USE_MOLD` | Use mold linker | `1` (if available) |

### NPU toolchain
For NPU feature, install the MLIR-AIE + Peano compiler:
```bash
python3 -m venv ~/.local/share/mistralrs-npu
source ~/.local/share/mistralrs-npu/bin/activate
pip install mlir_aie==1.4.4 llvm-aie \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

### ROCm environment
This script looks for `.rocm-env.sh` in the parent directory first, then falls back
to `/opt/rocm`. A stripped example is included as `scripts/.rocm-env-example.sh`.
Copy and edit it for your setup, then set `ROCM_PATH` before running:
```bash
cp scripts/.rocm-env-example.sh scripts/.rocm-env.sh
# Edit ROCM_PATH and HSA_OVERRIDE_GFX_VERSION as needed
ROCM_PATH=/opt/rocm ./scripts/build.sh
```
