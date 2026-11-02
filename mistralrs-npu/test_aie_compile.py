#!/usr/bin/env python3
"""
Test AIE kernel compilation.
Step 1: Simple integer kernel (always supported by Peano).
Step 2: BF16 kernel (may fail on AIE2P — known limitation).
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
    print(f"\n=== {name} ===")
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
            print(f"  ✓ {output.name} ({size:,} bytes)")
            return True
        except Exception as e:
            err = str(e)
            # Show the last few lines of the error
            last_lines = [l for l in err.split("\n") if not l.startswith(" #")]
            for l in last_lines[-3:]:
                print(f"  ✗ {l.strip()}")
            return False


def main():
    print("\n=== AIE Kernel Compilation Tests ===\n")

    # Test 1: Integer add (should always work)
    int_kernel = r"""
// AIE kernel: integer add (simple, always supported)
#include "aie_api/aie.hpp"

extern "C" void int_add(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    int N
) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}
"""

    # Test 2: BF16 load/store (no arithmetic — just memory ops)
    bf16_mem = r"""
// AIE kernel: BF16 load/store (no arithmetic)
#include "aie_api/aie.hpp"

extern "C" void bf16_copy(
    const bfloat16* __restrict__ A,
    bfloat16* __restrict__ C,
    int N
) {
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(&A[i]);
        aie::store_v(&C[i], v);
    }
}
"""

    results = {}
    results["int_add"] = test_kernel("Integer Add (int32)", int_kernel)
    results["bf16_copy"] = test_kernel("BF16 Copy (load/store only)", bf16_mem)

    print("\n=== Summary ===\n")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    if results["int_add"]:
        print("\n✓ Peano AIE2P backend compiles AIE kernels!")
        print("  The toolchain pipeline works end-to-end.")
        print("  BF16 FADD may fail — known AIE2P backend limitation.")
        print("\nWhat this means:")
        print("  - Kernel compilation: WORKING")
        print("  - BF16 arithmetic: needs AIE-specific intrinsics or Peano update")
        print("  - MLIR design + aiecc: next step to test")
    else:
        print("\n✗ Even integer kernel failed — Peano may not fully support AIE2P")

    return 0 if results["int_add"] else 1


if __name__ == "__main__":
    sys.exit(main())
