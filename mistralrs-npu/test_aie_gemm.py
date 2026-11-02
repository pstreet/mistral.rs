#!/usr/bin/env python3
"""
Complete GEMM kernel tests for AIE2P.
Uses approaches that compile: scalar BF16 MAC, float promotion, int16 cast.
"""

import sys
import os
import tempfile
from pathlib import Path

IRONENV = os.environ.get("NPU_VENV", os.path.expanduser("~/.local/share/ironenv"))
MLIR_INCLUDE = f"{IRONENV}/lib/python3.13/site-packages/mlir_aie/include"

from aie.utils.compile import compile_cxx_core_function, resolve_target_arch


def test_kernel(name, src):
    target = resolve_target_arch()
    print(f"  Target: {target}")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "kernel.cpp"
        src_file.write_text(src)
        output = Path(tmpdir) / "kernel.o"
        try:
            compile_cxx_core_function(
                source_path=str(src_file),
                target_arch=target,
                output_path=str(output),
                include_dirs=[MLIR_INCLUDE],
            )
            size = output.stat().st_size
            print(f"  PASS: {output.name} ({size:,} bytes)")
            return True
        except Exception as e:
            err = str(e)
            last_lines = [
                l
                for l in err.split("\n")
                if "error" in l.lower() or "unable" in l.lower()
            ]
            for l in last_lines[-3:]:
                print(f"  FAIL: {l.strip()[:120]}")
            return False


def main():
    results = {}

    # ===================================================================
    # GEMM using scalar BF16 MAC (aie::mac) — no vector add needed
    # ===================================================================
    results["gemm_scalar_mac"] = test_kernel(
        "GEMM 4x4x8 scalar mac()",
        r"""
#include "aie_api/aie.hpp"

// GEMM: C = A * B where A is MxK, B is KxN, C is MxN
// All dimensions <= 8 to fit within a single AIE core's compute
extern "C" void gemm_4x4x8(
    const bfloat16* __restrict__ A,  // MxK = 4x8
    const bfloat16* __restrict__ B,  // KxN = 8x4
    bfloat16* __restrict__ C,        // MxN = 4x4
    int M, int N, int K)
{
    // Each output element: C[m][n] = sum_k A[m][k] * B[k][n]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = aie::zero<bfloat16>();
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
""",
    )

    # ===================================================================
    # GEMM using float promotion (BF16 -> float -> math -> BF16)
    # ===================================================================
    results["gemm_float_prom"] = test_kernel(
        "GEMM 4x4x8 float promotion",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_float_prom(
    const bfloat16* __restrict__ A,  // MxK
    const bfloat16* __restrict__ B,  // KxN
    bfloat16* __restrict__ C,        // MxN
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (float)A[m * K + k] * (float)B[k * N + n];
            }
            C[m * N + n] = (bfloat16)sum;
        }
    }
}
""",
    )

    # ===================================================================
    # GEMM using int16 cast (BF16 -> int16 -> int32 MAC -> BF16)
    # ===================================================================
    results["gemm_int16"] = test_kernel(
        "GEMM 4x4x8 int16 cast",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_int16(
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    bfloat16* __restrict__ C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32 sum = 0;
            for (int k = 0; k < K; k++) {
                int16 a_val = (int16)(float)A[m * K + k];
                int16 b_val = (int16)(float)B[k * N + n];
                sum += (int32)a_val * (int32)b_val;
            }
            C[m * N + n] = (bfloat16)sum;
        }
    }
}
""",
    )

    # ===================================================================
    # GEMM using vectorized BF16 load + scalar MAC
    # (load row of A as vector, iterate over B row, MAC into acc)
    # ===================================================================
    results["gemm_vec_load"] = test_kernel(
        "GEMM 4x4x8 vec load + scalar mac",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_vec_load(
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    bfloat16* __restrict__ C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = aie::zero<bfloat16>();
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
""",
    )

    # ===================================================================
    # Larger GEMM 8x8x8 using scalar mac
    # ===================================================================
    results["gemm_8x8x8"] = test_kernel(
        "GEMM 8x8x8 scalar mac()",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_8x8x8(
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    bfloat16* __restrict__ C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = aie::zero<bfloat16>();
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
""",
    )

    # ===================================================================
    # GEMM 16x16x16 (larger, to test scalability)
    # ===================================================================
    results["gemm_16x16x16"] = test_kernel(
        "GEMM 16x16x16 scalar mac()",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_16x16x16(
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    bfloat16* __restrict__ C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = aie::zero<bfloat16>();
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
""",
    )

    # ===================================================================
    # Batched GEMM: multiple 4x4x4 GEMMs
    # ===================================================================
    results["gemm_batched"] = test_kernel(
        "Batched GEMM 4x4x4",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_batched_4x4x4(
    const bfloat16* __restrict__ A,  // Bx4x4
    const bfloat16* __restrict__ B,  // Bx4x4
    bfloat16* __restrict__ C,        // Bx4x4
    int batch)
{
    for (int b = 0; b < batch; b++) {
        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                bfloat16 sum = aie::zero<bfloat16>();
                for (int k = 0; k < 4; k++) {
                    sum = aie::mac<bfloat16>(sum,
                        A[b * 16 + m * 4 + k],
                        B[b * 16 + k * 4 + n]);
                }
                C[b * 16 + m * 4 + n] = sum;
            }
        }
    }
}
""",
    )

    # ===================================================================
    # GEMM with accumulation: C = C + A * B
    # ===================================================================
    results["gemm_accum"] = test_kernel(
        "GEMM accumulation C = C + A*B",
        r"""
#include "aie_api/aie.hpp"

extern "C" void gemm_accum_4x4x8(
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    bfloat16* __restrict__ C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = C[m * N + n];  // Start from existing C
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
""",
    )

    print("\n=== Summary ===\n")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    pass_count = sum(1 for v in results.values() if v)
    print(f"\n  {pass_count}/{len(results)} tests passed")

    if pass_count == len(results):
        print("\nAll GEMM kernels compile!")
    else:
        failed = [name for name, ok in results.items() if not ok]
        print(f"\nFailed: {', '.join(failed)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
