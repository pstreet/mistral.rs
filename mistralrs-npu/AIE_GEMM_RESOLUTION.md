# GEMM Kernel Resolution — AIE2P BF16

## Problem

Peano AIE2P backend (commit 0006955eb0bd) crashes on `G_FADD` for `<16 x s16>` BF16 vectors during legalization. The `aie::vector<bfloat16,16>` `operator+` and `operator*` map to this unsupported instruction.

## Solution

Use `aie::mac<bfloat16>(c, a, b)` which Peano translates to `c + a * b` (scalar operations). This avoids the vector FADD entirely.

## Compilation Test Results

All 8 GEMM kernel variants compile successfully on AIE2P with Peano 22.0.0:

| Kernel | Approach | Size |
|--------|----------|------|
| `gemm_scalar_mac` | `aie::mac<bfloat16>` | 1,564 B |
| `gemm_float_prom` | BF16→float→math→BF16 | 1,588 B |
| `gemm_int16` | BF16→int16→int32 MAC→BF16 | 1,684 B |
| `gemm_vec_load` | `load_v<16>` + scalar MAC | 1,568 B |
| `gemm_8x8x8` | `aie::mac<bfloat16>` ×8 | 1,564 B |
| `gemm_16x16x16` | `aie::mac<bfloat16>` ×16 | 3,004 B |
| `gemm_batched` | Batched 4×4×4 | 1,572 B |
| `gemm_accum` | C = C + A×B accumulation | 1,572 B |

## Working Operations on AIE2P

| Operation | Status |
|-----------|--------|
| `aie::mac<bfloat16>(c, a, b)` | ✅ |
| `aie::load_v<16>(&ptr)` / `store_v` | ✅ |
| `a[j] + b[j]` (element-wise vector) | ✅ |
| BF16 scalar `a + b` | ✅ |
| BF16 scalar `a * b` | ✅ |
| BF16→float cast | ✅ |
| BF16→int16 cast | ✅ |
| `vector<bfloat16,16>` `operator+` | ❌ G_FADD crash |
| `vector<bfloat16,16>` `operator*` | ❌ G_FADD crash |
| `aie::mmul<8,8,8,float>` | ❌ template error |

## GEMM Kernel Implementation

```cpp
#include "aie_api/aie.hpp"

// C[m][n] += A[m][k] * B[k][n], k=0..K-1
extern "C" void gemm_accum_bf16(
    const bfloat16* __restrict__ A,  // MxK
    const bfloat16* __restrict__ B,  // KxN
    bfloat16* __restrict__ C,        // MxN
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = C[m * N + n];  // accumulation
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum,
                    A[m * K + k],
                    B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
```

## Why `aie::mac<bfloat16>` Works

The `aie::mac` function for AIE2P (`__AIE_ARCH__ == 22`) is implemented as:

```cpp
template <typename TR, typename T1, typename T2>
TR mac(TR c, T1 a, T2 b) {
    return c + a * b;  // scalar operations, no vector FADD
}
```

This generates Peano IR with scalar BF16 `fmul` and `fadd` instructions, which the AIE2P backend supports. The vector FADD crash only occurs when the compiler tries to lower a vector BF16 add (`<16 x s16>`).

## Next Steps

1. **Compile kernels via MLIR-AIE design** — wrap this per-core kernel in an ObjectFIFO + DMA MLIR-AIE design
2. **Wire into `NpuBackendOps::gemm`** — dispatch to compiled kernel when available
3. **Benchmark** — measure throughput vs CPU fallback
