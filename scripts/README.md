# Build scripts

## build.sh
All-in-one build script for mistral.rs with CUDA, ROCm, NPU (AMD XDNA), and
CPU-only support. Every feature is enabled by default; each one only executes
its code paths on devices/platforms that support it, and build-time
dependency checks fail with actionable errors anytime a toolchain or library
is missing.

### Prerequisites
```bash
# Required sibling directory (candle-core)
git clone https://github.com/huggingface/candle.git
```

### Usage
```bash
# Default (all features, platform-appropriate: GPU shapes + cudnn/nccl/ring/mkl on Linux)
./scripts/build.sh

# ROCm only
MISTRALRS_FEATURES=rocm ./scripts/build.sh

# CUDA + flash attention
MISTRALRS_FEATURES=cuda,flash-attn ./scripts/build.sh

# CPU only
MISTRALRS_FEATURES= ./scripts/build.sh
```

### Feature-to-dependency map
| Feature | Build-time dependency |
|---------|----------------------|
| `cuda`, `cutile` | CUDA toolkit (`nvcc`) |
| `cudnn` | CUDA toolkit + `libcudnn.so.9` |
| `nccl` | CUDA toolkit + `libnccl.so.2`; self-gated at runtime (skipped without an NVIDIA driver runtime) |
| `flash-attn` | CUDA toolkit (CUTLASS kernels compile via nvcc) |
| `flash-attn-v3` | CUDA toolkit + sm90 GPU target; auto-dropped from the default set when no H100-class GPU is detected |
| `rocm` | ROCm install with `hipcc` + `lib/llvm/amdgcn/bitcode` |
| `npu` | MLIR-AIE + Peano (aiecc + clang) in `NPU_VENV` |
| `mkl` | network access (downloads Intel MKL at build); f16 CPU matmuls run through an MKL sgemm shim (the pinned MKL statics ship no `hgemm_`) |
| `ring` | none; runtime-selected per rank via `MISTRALRS_RING_CONFIG`, runs on every device platform (CUDA, ROCm, NPU, CPU) |
| `metal`, `accelerate` | macOS |

### Environment variables
| Variable | Description | Default |
|----------|-------------|---------|
| `MISTRALRS_FEATURES` | Cargo features to build | all (platform-appropriate) |
| `GPU_DEPS_INSTALL` | apt auto-install consent: `ask` / `always` / `never` | `ask` (TTY) |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit root | auto-detected |
| `CUDA_COMPUTE_CAP` | CUDA compute capability (auto-detected from `nvidia-smi` when unset) | `80` |
| `CANDLE_ROCM_PATH` | Override ROCm install path | `$ROCM_PATH` |
| `ROCM_PATH` | ROCm install root (also read by `.rocm-env.sh`) | `/opt/rocm` |
| `NPU_VENV` | MLIR-AIE Python venv | `~/.local/share/mistralrs-npu` |
| `CARGO_BUILD_JOBS` | Parallelism | `nproc` |
| `USE_MOLD` | Use mold linker | `1` (if available) |

### NPU toolchain
For the NPU feature, install the open-source MLIR-AIE + Peano AIE compilers:
```bash
python3 -m venv ~/.local/share/mistralrs-npu
source ~/.local/share/mistralrs-npu/bin/activate
pip install mlir_aie==1.4.4 llvm-aie \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

### ROCm environment
This script looks for `.rocm-env.sh` in the parent directory first, then
`scripts/`. A stripped example is included as `scripts/.rocm-env-example.sh`.
Copy and edit it for your setup, then set `ROCM_PATH` before running:
```bash
cp scripts/.rocm-env-example.sh scripts/.rocm-env.sh
# Edit ROCM_PATH and HSA_OVERRIDE_GFX_VERSION as needed
ROCM_PATH=/opt/rocm ./scripts/build.sh
```
