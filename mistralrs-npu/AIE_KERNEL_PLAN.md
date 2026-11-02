# MLIR-AIE + Peano AIE Kernel Development Plan for Strix Halo

## Architecture Overview

### Strix Halo AIE (XDNA 2)
- **4x4 compute tile array** = 16 compute tiles, each with a SIMD PE
- **4 memory tiles** = DMA engines + L1 caches (2MB each)
- **1 control tile** = A72 CPU core + AIE control
- **Per-tile PE**: 128-bit vector, 256 GFLOP/s (BF16), 256KB L1
- **Total**: ~4 TFLOP/s (BF16), unified memory via PCIe/AMD PMF

### Compilation Flow
```
Python/MLIR-AIE code
        │
        ▼
  aiecc (MLIR-AIE compiler)
        │
        ▼
  MLIR → AIR → AIE IR
        │
        ▼
  Peano (per-core LLVM)
        │
        ▼
  .xo object files
        │
        ▼
  xclbinutil (linker)
        │
        ▼
  .xclbin executable
        │
        ▼
  XRT (mistralrs-npu)
        │
        ▼
  AIE execution
```

## Phase 1: Toolchain Setup (COMPLETED)

### Installed
- **MLIR-AIE** 1.4.4 (Oct 2026, AIE2P/XDNA2 support)
- **llvm-aie (Peano)** 22.0.0.2026092101+0006955e (nightly, pinned to mlir-aie)
- **aiecc** compiler driver: `~/.local/share/ironenv/bin/aiecc`
- **Peano clang** 22.0.0: `~/.local/share/ironenv/lib/python3.13/site-packages/llvm-aie/bin/clang`

### Wheels saved (for reproducible builds)
- `/home/pstreet/LocalAI/mlir_aie-1.4.4-cp313-cp313-manylinux_2_35_x86_64.whl` (260MB)
- `/home/pstreet/LocalAI/llvm_aie-22.0.0.2026092101+0006955e-py3-none-manylinux_*.whl` (148MB)

### Setup script
- `scripts/setup-npu-aie.sh` — creates venv and installs toolchain (idempotent)

### Build script integration
- `build-mistral.rs.sh` now checks for `aiecc` + Peano clang when `npu` feature is enabled
- Fails with actionable error message if toolchain is missing

### Verification output
```
aiecc: aiecc (mlir-aie declarative driver)
  git SHA: e5c705c7b04, compiled: Oct 6 2026
Peano: clang version 22.0.0 (llvm-aie 0006955eb0bd)
```

## Phase 2: Write AIE GEMM Kernel (Week 1-2)

### Resolution: BF16 GEMM via `aie::mac()`

The initial approach of using vector BF16 `operator+` (`a + b` on `vector<bfloat16,16>`) failed because Peano AIE2P backend crashes on `G_FADD` during legalization.

**Solution:** Use `aie::mac<bfloat16>(c, a, b)` which compiles to scalar BF16 `c + a * b`. This avoids vector FADD entirely.

### GEMM kernel (per-tile compute)
```cpp
// gemm_kernel.cpp — compiled by Peano into .xo
#include "aie_api/aie.hpp"

extern "C" void gemm_accum_bf16(
    const bfloat16* __restrict__ A,  // MxK row-major
    const bfloat16* __restrict__ B,  // KxN row-major
    bfloat16* __restrict__ C,        // MxN row-major
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = C[m * N + n];
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
```

### Verified operations on AIE2P

| Operation | Status |
|-----------|--------|
| `aie::mac<bfloat16>(c, a, b)` | ✅ |
| `aie::load_v<16>` / `store_v<16>` | ✅ |
| Vector element-wise `a[j] + b[j]` | ✅ |
| BF16 scalar `a + b`, `a * b` | ✅ |
| `vector<bfloat16,16>` `operator+` | ❌ G_FADD crash |
| `vector<bfloat16,16>` `operator*` | ❌ G_FADD crash |
| `aie::mmul<8,8,8,float>` | ❌ template error |

### Kernel compilation
```bash
# Compile with Peano (per-core)
compile_cxx_core_function(
    source_path="gemm_kernel.cpp",
    target_arch="aie2",
    output_path="kernel.o",
)
```

### Next: MLIR-AIE design level
After per-core kernel compiles, wrap in ObjectFIFO + DMA design:
```python
from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx

@device(AIEDevice.npu)
def device_body():
    tile(0, 1)  # First compute tile
    with mlir_mod_ctx():
        # ObjectFIFOs, DMA, kernel link
```

## Phase 3: Integrate into mistralrs-npu (Week 3-4)

### XRT bridge (already exists)
```rust
// mistralrs-npu/src/backend.rs
pub trait XrtDevice {
    fn compile_xclbin(&mut self, xclbin: &[u8]) -> Result<KernelHandle>;
    fn load_xclbin(&mut self, handle: &KernelHandle) -> Result<RunHandle>;
    fn run_kernel(&mut self, handle: &RunHandle, args: &[usize]) -> Result<()>;
    fn sync(&self) -> Result<()>;
}
```

### GEMM dispatch
```rust
impl NpuStorage {
    fn gemm(&self, rhs: &Self, bmnk: (usize, usize, usize, usize)) -> Result<Self> {
        // 1. Ensure XCLBIN is loaded (compiled at startup)
        // 2. Set kernel args: pointer to A, B, C buffers
        // 3. Trigger execution
        // 4. Wait for completion (unified memory = minimal sync)
    }
}
```

### Model loading
```rust
// Compile GEMM kernels for common shapes at startup
fn compile_kernels() -> Result<CompiledKernels> {
    vec![
        // Pre-compile for shapes that appear in transformer inference
        compile_gemmbf16(64, 4096, 4096),  // large layers
        compile_gemmbf16(64, 1024, 4096),  // smaller layers
        compile_gemmbf16(32, 512, 512),    // tiny layers
    ]
}
```

## Phase 4: Full Inference Pipeline (Week 5-8)

### Priority order
1. **GEMM** (matmul) — covers all linear layers, attention QKV
2. **RMSNorm** — per-token normalization (simple)
3. **RoPE** — position encoding (simple)
4. **Softmax** — attention scoring (medium)
5. **LayerNorm** — if used by model

### Expected performance targets
| Operation | CPU fallback | Target AIE | Improvement |
|-----------|-------------|------------|-------------|
| GEMM 4096×4096 | ~50 GFLOP/s | ~500 GFLOP/s | ~10x |
| RMSNorm | ~200 GFLOP/s | ~400 GFLOP/s | ~2x |
| Full model (Qwen3-0.6B) | ~15 tok/s | ~60+ tok/s | ~4x |

## Key References

### Repositories
- **MLIR-AIE**: https://github.com/Xilinx/mlir-aie (v1.4.4, Apache 2.0)
- **Peano (llvm-aie)**: https://github.com/Xilinx/llvm-aie (Apache 2.0)
- **XRT**: https://github.com/Xilinx/XRT (Apache 2.0)
- **XRT AMD XDNA Shim**: https://github.com/Xilinx/xdna-driver (GPL-2.0)

### Documentation
- **MLIR-AIE Programming Guide**: https://xilinx.github.io/mlir-aie/latest/programming_guide/
- **AMD XDNA NPU Architecture**: IEEE Micro 2024 paper
- **IRON API**: FCCM 2025 paper — close-to-metal Python API
- **AIE C++ API Headers**: aie_api/aie.hpp (included in llvm-aie wheel)

### Papers
- "Unlocking the AMD Neural Processing Unit..." (FCCM 2025)
- "AIR: Agile MLIR-Based Compilation Flow..." (FPGA 2025)
- "Spatial Computing with AIR for Ryzen AI" (ASPLOS 2024)

## Risks & Mitigations

### Risk 1: XDNA 2 support incomplete
- **Status**: MLIR-AIE v1.4.x has "AIE2P kernels" support
- **Mitigation**: Use `AIEDevice.npu` device type; fallback to simulation if hardware fails

### Risk 2: ObjectFIFO API changes
- **Status**: v1.4.x introduced "fully dynamic ObjectFIFOs"
- **Mitigation**: Pin to specific version; test with passthrough example first

### Risk 3: Strix Halo vs Strix Point differences
- **Status**: Both are XDNA 2 but different tile counts
- **Mitigation**: Query tile config at runtime; compile separate kernels

### Risk 4: Memory alignment
- **Status**: AIE ObjectFIFOs need DDR-aligned buffers
- **Mitigation**: Use XRT's unified memory API for aligned allocations
