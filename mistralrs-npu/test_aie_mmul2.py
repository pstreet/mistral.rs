#!/usr/bin/env python3
"""
Test aie::mmul with correct template parameters.
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
            for l in last_lines[-2:]:
                print(f"  FAIL: {l.strip()[:120]}")
            return False


def main():
    results = {}

    # Test 1: mmul<8,8,8,float> — let types auto-detect
    results["mmul_888_auto"] = test_kernel(
        "mmul<8,8,8,float> (auto accum)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_888(
    const float* A, const float* B, float* C,
    int strideA, int strideB)
{
    aie::mmul<8, 8, 8, float> m;
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

    # Test 2: mmul<8,8,8,float,accfloat> — explicit accum type
    results["mmul_888_explicit"] = test_kernel(
        "mmul<8,8,8,float,accfloat>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_888_explicit(
    const float* A, const float* B, float* C,
    int strideA, int strideB)
{
    aie::mmul<8, 8, 8, float, float, accfloat> m;
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

    # Test 3: bf16 mmul<4,8,4> — BF16 hardware MMUL
    results["mmul_bf16_484"] = test_kernel(
        "bf16 mmul<4,8,4>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_bf16_484(
    const bfloat16* A, const bfloat16* B, bfloat16* C,
    int strideA, int strideB)
{
    aie::mmul<4, 8, 4, bfloat16> m;
    m.zero();
    for (int k = 0; k < 8; k++) {
        aie::vector<bfloat16, 4> a = aie::load_v<4>(&A[k * strideA]);
        aie::vector<bfloat16, 4> b;
        for (int j = 0; j < 4; j++) b[j] = B[j * strideB + k];
        m.mac(a, true, b, true);
    }
    for (int i = 0; i < 4; i++) C[i] = m.to_accum()[i];
}
""",
    )

    # Test 4: bf16 mmul<8,8,8> — BF16 hardware MMUL (larger)
    results["mmul_bf16_888"] = test_kernel(
        "bf16 mmul<8,8,8>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_bf16_888(
    const bfloat16* A, const bfloat16* B, bfloat16* C,
    int strideA, int strideB)
{
    aie::mmul<8, 8, 8, bfloat16> m;
    m.zero();
    for (int k = 0; k < 8; k++) {
        aie::vector<bfloat16, 8> a = aie::load_v<8>(&A[k * strideA]);
        aie::vector<bfloat16, 8> b;
        for (int j = 0; j < 8; j++) b[j] = B[j * strideB + k];
        m.mac(a, true, b, true);
    }
    for (int i = 0; i < 8; i++) C[i] = m.to_accum()[i];
}
""",
    )

    # Test 5: bf16 mmul<4,4,8> — BF16 hardware MMUL
    results["mmul_bf16_448"] = test_kernel(
        "bf16 mmul<4,4,8>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_bf16_448(
    const bfloat16* A, const bfloat16* B, bfloat16* C,
    int strideA, int strideB)
{
    aie::mmul<4, 4, 8, bfloat16> m;
    m.zero();
    for (int k = 0; k < 4; k++) {
        aie::vector<bfloat16, 4> a = aie::load_v<4>(&A[k * strideA]);
        aie::vector<bfloat16, 8> b;
        for (int j = 0; j < 8; j++) b[j] = B[j * strideB + k];
        m.mac(a, true, b, true);
    }
    for (int i = 0; i < 8; i++) C[i] = m.to_accum()[i];
}
""",
    )

    # Test 6: bf16 mmul<4,2,8> — BF16 hardware MMUL (smallest)
    results["mmul_bf16_428"] = test_kernel(
        "bf16 mmul<4,2,8>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_bf16_428(
    const bfloat16* A, const bfloat16* B, bfloat16* C,
    int strideA, int strideB)
{
    aie::mmul<4, 2, 8, bfloat16> m;
    m.zero();
    for (int k = 0; k < 2; k++) {
        aie::vector<bfloat16, 4> a = aie::load_v<4>(&A[k * strideA]);
        aie::vector<bfloat16, 8> b;
        for (int j = 0; j < 8; j++) b[j] = B[j * strideB + k];
        m.mac(a, true, b, true);
    }
    for (int i = 0; i < 8; i++) C[i] = m.to_accum()[i];
}
""",
    )

    # Test 7: bf16 mmul<2,4,8> — BF16 hardware MMUL
    results["mmul_bf16_248"] = test_kernel(
        "bf16 mmul<2,4,8>",
        r"""
#include "aie_api/aie.hpp"
extern "C" void mmul_bf16_248(
    const bfloat16* A, const bfloat16* B, bfloat16* C,
    int strideA, int strideB)
{
    aie::mmul<2, 4, 8, bfloat16> m;
    m.zero();
    for (int k = 0; k < 4; k++) {
        aie::vector<bfloat16, 2> a = aie::load_v<2>(&A[k * strideA]);
        aie::vector<bfloat16, 8> b;
        for (int j = 0; j < 8; j++) b[j] = B[j * strideB + k];
        m.mac(a, true, b, true);
    }
    for (int i = 0; i < 8; i++) C[i] = m.to_accum()[i];
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
