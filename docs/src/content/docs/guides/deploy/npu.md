# AMD Ryzen AI NPU Deployment (Strix Halo / XDNA 2)

> **Status:** NPU device path is fully wired (device selection, tensor ops,
> quantized weights, safetensors loading). The backend delegates to CPU via
> unified memory. AIE kernel acceleration (GEMM, attention, RMSNorm) is under
> active development — see [AIE_KERNEL_PLAN.md](../mistralrs-npu/AIE_KERNEL_PLAN.md).

## Hardware requirements

- AMD Ryzen AI MAX / Strix Halo SoC (XDNA 2 NPU)
- Linux kernel 6.17+ (`amdxdna` driver)
- XRT NPU2 runtime (`libxrt-npu2`)
- At least 32 GB unified RAM (model + runtime overhead)

## Prerequisites

### System packages

```bash
sudo apt-get update
sudo apt-get install -y \
    libxrt-dev libxrt2 libxrt-npu2 python3-xrt linux-firmware-amd-misc
```

### Kernel driver

```bash
# Verify the driver is loaded
lsmod | grep amdxdna
# If not loaded:
sudo modprobe amdxdna

# Verify device nodes
ls -la /dev/dri/card0 /dev/dri/card1 /dev/accel/accel0
```

### MLIR-AIE + Peano (optional, for AIE kernel compilation)

The open-source toolchain enables writing custom AIE kernels. Install it once:

```bash
# Create a venv (re-use an existing one if you prefer)
python3 -m venv ~/.local/share/mistralrs-npu
source ~/.local/share/mistralrs-npu/bin/activate
pip install --upgrade pip

# MLIR-AIE 1.4.4 — compiler driver, Python API, MLIR dialects
pip install mlir_aie==1.4.4 \
  -f https://github.com/Xilinx/mlir-aie/releases/download/v1.4.4/

# Peano (llvm-aie) — per-core AIE compiler, pinned to the tested nightly
pip install -f \
  https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  "llvm-aie==22.0.0.2026092101+0006955e"
```

Or use the helper script:

```bash
./scripts/setup-npu-aie.sh
```

Verify:

```bash
source ~/.local/share/mistralrs-npu/bin/activate
aiecc --version
aiecc --help
```


## Building with NPU support

### Build script (recommended)

The build script checks for all dependencies and fails with actionable errors:

```bash
cd mistral.rs/scripts
./build.sh
```

Set `MISTRALRS_FEATURES` to control which backends are compiled:

```bash
MISTRALRS_FEATURES=rocm ./build.sh      # ROCm only
MISTRALRS_FEATURES=cuda,npu ./build.sh   # CUDA + NPU
```

It sources the MLIR-AIE venv automatically when the `npu` feature is enabled.

### Manual build

```bash
# Source the MLIR-AIE venv (required for downstream toolchain discovery)
source ~/.local/share/mistralrs-npu/bin/activate

# Build mistral.rs with NPU support
cd mistral.rs
cargo build --release --features npu -p mistralrs-cli
```

Multi-vendor build:

```bash
cargo build --release --features cuda,rocm,npu,flash-attn -p mistralrs-cli
```

## Running on NPU

### Interactive mode

```bash
# Run a model on NPU 0
mistralrs run -m <model-id-or-path> --device npu:0

# Interactive chat with a local GGUF
mistralrs run -f /path/to/model.gguf --device npu:0
```

### Server mode

```bash
mistralrs serve -m <model-id> --device npu:0
```

### Diagnostics

```bash
# Check system and accelerator status
mistralrs doctor
```

Output includes NPU memory, device ordinal, and unified memory status:

```
[INFO] NPU[0]: 131.2 GB total, 79.1 GB free
[PASS] NPU 0: unified memory detected
```

## How it works

### Unified memory model

On Strix Halo, the NPU and CPU share the same physical RAM. There is no
PCIe-bound VRAM to copy data across. The NPU backend (`NpuStorage`) wraps the
CPU backend (`CpuStorage`) transparently:

```
Tensor allocation  →  CPU backend (malloc/mmap)
Matmul/attention   →  CPU backend (BLAS/GEMM)
Synchronization    →  NPU fence (for future AIE dispatch)
```

The model weights, KV cache, and activations all live in unified memory.
Device selection (`--device npu:0`) routes tensors through the NPU path even
though the actual compute currently runs on CPU cores.

### AIE kernel acceleration (under development)

The path to real NPU acceleration:

1. **Write** AIE kernel in C++ (SIMD dataflow, tile-based GEMM)
2. **Compile** with Peano (per-core clang) → `.xo` object
3. **Assemble** MLIR-AIE design with ObjectFIFOs → `.xclbin`
4. **Load** via XRT in `mistralrs-npu`
5. **Dispatch** kernel with tensor pointers as arguments

See [AIE_KERNEL_PLAN.md](../mistralrs-npu/AIE_KERNEL_PLAN.md) for the full plan.

## Troubleshooting

### NPU device not detected

```bash
# Check kernel module
lsmod | grep amdxdna

# Check XRT library is loadable
ldconfig -p | grep xrt_coreutil

# Check device nodes
ls -la /dev/accel/accel0 /dev/dri/card0
```

### "Error: MLIR-AIE venv not found"

Run the setup script or install the toolchain manually (see "MLIR-AIE + Peano"
above). The build script expects the venv at `~/.local/share/mistralrs-npu` or a
custom path via `NPU_VENV`:

```bash
NPU_VENV=/custom/path ./scripts/build.sh
```

### Model falls back to CPU

This is expected — the NPU backend currently delegates to CPU. Check the logs
for `npu[0]` in the layer mapping output:

```
Layers 0-27: npu[0] (123 GB)
```

If you see `cpu[0]` instead, the NPU device may not have been selected. Verify
with:

```bash
mistralrs run -m <model> --device npu:0 -i "test" 2>&1 | grep -i npu
```

### Build fails with "aiecc not found"

The MLIR-AIE venv needs to be sourced. The build script does this
automatically, but if building manually:

```bash
source ~/.local/share/mistralrs-npu/bin/activate
export PATH="$HOME/.local/share/mistralrs-npu/bin:$PATH"
```

### NPU memory too small for model

Strix Halo has unified memory — the NPU sees the total system RAM minus the
CPU's working set. If the model doesn't fit:

- Use a smaller model (0.5B–1B range works on 16 GB+)
- Use quantization (`--quant` flag for GGUF/UQFF)
- Free system memory before running
- Check available RAM: `mistralrs doctor`

## References

- [MLIR-AIE](https://github.com/Xilinx/mlir-aie) — open-source MLIR compiler
- [llvm-aie (Peano)](https://github.com/Xilinx/llvm-aie) — per-core AIE compiler
- [AMD XDNA driver](https://github.com/Xilinx/xdna-driver) — kernel driver
- [AIE_KERNEL_PLAN.md](../mistralrs-npu/AIE_KERNEL_PLAN.md) — kernel development plan
- [scripts/build.sh](../scripts/build.sh) — build script with dependency checks
