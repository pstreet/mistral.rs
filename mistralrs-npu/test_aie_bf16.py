#!/usr/bin/env python3
"""
Test AIE kernel compilation for BF16 operations on AIE2P.
Systematically test what compiles and what doesn't.
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
                if "legalize" in l.lower()
                or "error" in l.lower()
                or "unable" in l.lower()
            ]
            for l in last_lines[-2:]:
                print(f"  FAIL: {l.strip()[:120]}")
            return False


def main():
    results = {}

    # Test 1: Integer add (baseline)
    results["int_add"] = test_kernel(
        "Integer Add (int32)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void int_add(const int* A, const int* B, int* C, int N) {
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}
""",
    )

    # Test 2: BF16 load/store (no arithmetic)
    results["bf16_copy"] = test_kernel(
        "BF16 Copy (load/store)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_copy(const bfloat16* A, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 16)
        aie::store_v(&C[i], aie::load_v<16>(&A[i]));
}
""",
    )

    # Test 3: BF16 scalar add
    results["bf16_scalar_add"] = test_kernel(
        "BF16 Scalar Add",
        r"""
#include "aie_api/aie.hpp"
extern "C" bfloat16 bf16_add(bfloat16 a, bfloat16 b) { return a + b; }
""",
    )

    # Test 4: BF16 vector add (known to fail)
    results["bf16_vec_add"] = test_kernel(
        "BF16 Vector Add (16 elem)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_vec_add(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 16) {
        auto a = aie::load_v<16>(&A[i]);
        auto b = aie::load_v<16>(&B[i]);
        aie::store_v(&C[i], a + b);  // operator+ on vector
    }
}
""",
    )

    # Test 5: BF16 vector mul (known to fail on AIE2P)
    results["bf16_vec_mul"] = test_kernel(
        "BF16 Vector Mul (16 elem)",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_vec_mul(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 16) {
        auto a = aie::load_v<16>(&A[i]);
        auto b = aie::load_v<16>(&B[i]);
        aie::store_v(&C[i], a * b);
    }
}
""",
    )

    # Test 6: BF16 float dot product (via float conversion)
    results["bf16_float_dot"] = test_kernel(
        "BF16->float dot product",
        r"""
#include "aie_api/aie.hpp"
extern "C" float bf16_dot(const bfloat16* A, const bfloat16* B, int N) {
    float sum = 0;
    for (int i = 0; i < N; ++i) sum += (float)A[i] * (float)B[i];
    return sum;
}
""",
    )

    # Test 7: BF16 vector to float conversion
    results["bf16_vec_to_float"] = test_kernel(
        "BF16 Vector -> float",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_to_float(const bfloat16* A, float* C, int N) {
    for (int i = 0; i < N; ++i) C[i] = (float)A[i];
}
""",
    )

    # Test 8: BF16 scalar multiply-accumulate
    results["bf16_scalar_mac"] = test_kernel(
        "BF16 Scalar MAC",
        r"""
#include "aie_api/aie.hpp"
extern "C" bfloat16 bf16_mac(bfloat16 c, bfloat16 a, bfloat16 b) {
    return c + a * b;
}
""",
    )

    # Test 9: BF16 via float accumulation (promote to float for math)
    results["bf16_promoted_mac"] = test_kernel(
        "BF16 promoted to float for MAC",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_promoted_mac(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; ++i) {
        float sum = (float)C[i];
        sum += (float)A[i] * (float)B[i];
        C[i] = (bfloat16)sum;
    }
}
""",
    )

    # Test 10: BF16 aie_api mac() scalar
    results["bf16_aie_mac"] = test_kernel(
        "BF16 aie_api::mac (scalar)",
        r"""
#include "aie_api/aie.hpp"
extern "C" bfloat16 bf16_aie_mac(bfloat16 c, bfloat16 a, bfloat16 b) {
    return aie::mac<cfloat>(c, a, b);  // AIE2P: just c + a*b
}
""",
    )

    # Test 11: BF16 vector element-wise via loop
    results["bf16_vec_loop_add"] = test_kernel(
        "BF16 Vector Add via loop",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_vec_loop_add(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 16) {
        auto a = aie::load_v<16>(&A[i]);
        auto b = aie::load_v<16>(&B[i]);
        for (int j = 0; j < 16; ++j) C[i+j] = a[j] + b[j];
    }
}
""",
    )

    # Test 12: BF16 int16 cast + integer MAC
    results["bf16_int16_mac"] = test_kernel(
        "BF16 -> int16 -> integer MAC -> back",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_int16_mac(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; ++i) {
        int32 sum = (int32)C[i];
        sum += (int32)(int16)((float)A[i]) * (int32)(int16)((float)B[i]);
        C[i] = (bfloat16)sum;
    }
}
""",
    )

    # Test 13: BF16 vector with aie_api::mac (no vector, scalar)
    results["bf16_scalar_mac_via_api"] = test_kernel(
        "BF16 scalar mac via aie API",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_scalar_mac_via_api(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = aie::mac<bfloat16>(C[i], A[i], B[i]);
    }
}
""",
    )

    # Test 14: BF16 vector mul then scalar add (promoted to float)
    results["bf16_float_accum"] = test_kernel(
        "BF16 -> float accum -> BF16",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_float_accum(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; ++i) {
        float acc = (float)C[i];
        for (int k = 0; k < 16; k++) {
            acc += (float)A[i] * (float)B[k];
        }
        C[i] = (bfloat16)acc;
    }
}
""",
    )

    # Test 15: BF16 aie::mul (vector)
    results["bf16_vec_aie_mul"] = test_kernel(
        "BF16 vector aie::mul()",
        r"""
#include "aie_api/aie.hpp"
extern "C" void bf16_vec_aie_mul(
    const bfloat16* A, const bfloat16* B, bfloat16* C, int N) {
    for (int i = 0; i < N; i += 16) {
        auto a = aie::load_v<16>(&A[i]);
        auto b = aie::load_v<16>(&B[i]);
        aie::store_v(&C[i], aie::mul(a, b));
    }
}
""",
    )

    print("\n=== Summary ===\n")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    pass_count = sum(1 for v in results.values() if v)
    print(f"\n  {pass_count}/{len(results)} tests passed")

    if not results["int_add"]:
        print("\nFAIL: Even integer kernel failed — Peano AIE2P may be broken")

    return 0


if __name__ == "__main__":
    sys.exit(main())
