#!/usr/bin/env python3
"""
Test AIE kernel compilation using mmul (hardware matrix multiply).
This is the path forward for GEMM on AIE2P.
"""

import sys
import os
import tempfile
from pathlib import Path

IRONENV = os.environ.get("NPU_VENV", os.path.expanduser("~/.local/share/ironenv"))
MLIR_INCLUDE = f"{IRONENV}/lib/python3.13/site-packages/mlir_aie/include"

from aie.utils.compile import compile_cxx_core_function, resolve_target_arch


def test_kernel(name, src):
    """Compile a kernel and report result."""
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
            for l in last_lines[-2:]:
                print(f"  FAIL: {l.strip()[:120]}")
            return False


def main():
    results = {}

    # Test 1: mmul 4x4x4 with float accumulators (smallest supported mode)
    results["mmul_4x4x4_float"] = test_kernel(
        "mmul 4x4x4 float",
        r"""
#include "aie_api/aie.hpp"
#include <cstddef>

extern "C" void mmul_4x4x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int strideA, int strideB, int strideC)
{
    // Create a 4x4x4 matrix multiply accumulator (4x4 output, K=4)
    aie::mmul<4, 4, 4, float, float, 32> m;
    m.zero();

    // Load A[0..4][0..4] and B[0..4][0..4], do MAC
    for (int k = 0; k < 4; k++) {
        // A is MxK layout, B is KxN layout
        // Load row of A
        aie::vector<float, 4> a = aie::load_v<4>(&A[k * strideA]);
        // Load column of B (need to transpose conceptually)
        aie::vector<float, 4> b;
        b[0] = B[0 * strideB + k];
        b[1] = B[1 * strideB + k];
        b[2] = B[2 * strideB + k];
        b[3] = B[3 * strideB + k];
        m.mac(a, true, b, true);
    }

    // Write result to C
    aie::vector<float, 4> out;
    out[0] = m.to_accum()[0];
    out[1] = m.to_accum()[1];
    out[2] = m.to_accum()[2];
    out[3] = m.to_accum()[3];
    aie::store_v(&C[0], out);
}
""",
    )

    # Test 2: Simple float vector load + store (no BF16)
    results["float_vec_ls"] = test_kernel(
        "float vector load/store",
        r"""
#include "aie_api/aie.hpp"
extern "C" void float_ls(const float* A, float* C, int N) {
    for (int i = 0; i < N; i += 4) {
        auto v = aie::load_v<4>(&A[i]);
        aie::store_v(&C[i], v);
    }
}
""",
    )

    # Test 3: BF16 load + scalar MAC + store (no vector add)
    results["bf16_scalar_mac_ls"] = test_kernel(
        "BF16 scalar MAC + load/store",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_scalar_mac(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N)
{
    for (int i = 0; i < N; ++i) {
        C[i] = aie::mac<bfloat16>(C[i], A[i], B[i]);
    }
}
""",
    )

    # Test 4: float vector multiply-accumulate (no mmul, just scalar vec)
    results["float_vec_mac"] = test_kernel(
        "float vec MAC (loop)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void float_vec_mac(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = C[i] + A[i] * B[i];
    }
}
""",
    )

    # Test 5: BF16 load/store using aie::load_v + store_v (should work)
    results["bf16_load_store"] = test_kernel(
        "BF16 load_v/store_v",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_load_store(const bfloat16* A, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 16) {
        auto v = aie::load_v<16>(&A[i]);
        aie::store_v(&C[i], v);
    }
}
""",
    )

    # Test 6: BF16 load/store using raw pointers (no aie_api wrapper)
    results["bf16_load_store_v8"] = test_kernel(
        "BF16 load_v<8>/store_v<8>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_ls8(const bfloat16* A, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 8) {
        auto v = aie::load_v<8>(&A[i]);
        aie::store_v(&C[i], v);
    }
}
""",
    )

    # Test 7: BF16 vector element access + scalar add
    results["bf16_vec_elem_access"] = test_kernel(
        "BF16 vector elem access + scalar add",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_vec_elem(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N)
{
    for (int i = 0; i < N; i += 16) {
        auto a = aie::load_v<16>(&A[i]);
        auto b = aie::load_v<16>(&B[i]);
        for (int j = 0; j < 16; ++j) {
            C[i+j] = a[j] + b[j];
        }
    }
}
""",
    )

    # Test 8: BF16 via float promotion (load -> cast -> float math -> back)
    results["bf16_via_float"] = test_kernel(
        "BF16 via float promotion",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_via_float(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N)
{
    for (int i = 0; i < N; i += 16) {
        auto va = aie::load_v<16>(&A[i]);
        auto vb = aie::load_v<16>(&B[i]);
        for (int j = 0; j < 16; ++j) {
            float fa = (float)va[j];
            float fb = (float)vb[j];
            C[i+j] = (bfloat16)(fa + fb);
        }
    }
}
""",
    )

    # Test 9: BF16 int16 cast + integer MAC
    results["bf16_int_mac"] = test_kernel(
        "BF16 -> int16 -> int32 MAC",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_int_mac(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N)
{
    for (int i = 0; i < N; ++i) {
        int32 sum = (int32)(int16)(float)C[i];
        sum += (int32)(int16)(float)A[i] * (int32)(int16)(float)B[i];
        C[i] = (bfloat16)sum;
    }
}
""",
    )

    # Test 10: float mmul 8x8x8 (larger mode, hardware MMUL)
    results["mmul_8x8x8_float"] = test_kernel(
        "mmul 8x8x8 float (hw MMUL)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_8x8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int strideA, int strideB, int strideC)
{
    aie::mmul<8, 8, 8, float, float, 32> m;
    m.zero();

    for (int k = 0; k < 8; k++) {
        aie::vector<float, 8> a = aie::load_v<8>(&A[k * strideA]);
        aie::vector<float, 8> b;
        for (int j = 0; j < 8; j++) b[j] = B[j * strideB + k];
        m.mac(a, true, b, true);
    }

    for (int i = 0; i < 8; i++) C[i] = m.to_accum()[i];
}
""",
    )

    # Test 11: BF16 matrix multiply via float promotion
    results["bf16_matmul_float_prom"] = test_kernel(
        "BF16 matmul via float promotion",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_matmul_float_prom(
    const bfloat16* A, const bfloat16* B, bfloat16* C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = (float)C[m];
            for (int k = 0; k < K; k++) {
                sum += (float)A[m * K + k] * (float)B[k * N + n];
            }
            C[m] = (bfloat16)sum;
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
