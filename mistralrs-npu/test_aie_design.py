#!/usr/bin/env python3
"""
Test full MLIR-AIE design compilation with ObjectFIFOs + DMA + core.
"""

import sys
import os
import tempfile
from pathlib import Path

IRONENV = os.environ.get("NPU_VENV", os.path.expanduser("~/.local/share/ironenv"))
MLIR_INC = f"{IRONENV}/lib/python3.13/site-packages/mlir_aie/include"

from aie.utils.compile import (
    compile_cxx_core_function,
    compile_mlir_module,
    resolve_target_arch,
)


def main():
    target = resolve_target_arch()
    print(f"Target: {target}\n")

    # Step 1: Compile the GEMM kernel as a .xo object
    print("=== Step 1: Compile gemm_kernel.cpp ===")
    kernel_src = Path("gemm_kernel.cpp").read_text()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src_file = tmpdir / "gemm_kernel.cpp"
        src_file.write_text(kernel_src)
        kernel_xo = tmpdir / "gemm_kernel.xo"

        compile_cxx_core_function(
            source_path=str(src_file),
            target_arch=target,
            output_path=str(kernel_xo),
            include_dirs=[MLIR_INC],
        )
        print(f"  kernel.xo: {kernel_xo.stat().st_size:,} bytes\n")

        # Step 2: Create a minimal MLIR-AIE design
        print("=== Step 2: Compile MLIR-AIE design ===")

        mlir = f"""// Minimal MLIR-AIE design for GEMM
// Uses aiecc compile_mode=design

module {{
  aie.device("npu") {{
    aie.tile(0, 1) {{
      aie.lock("LOCK_0", 1)
      aie.lock("LOCK_1", 1)
      aie.lock("LOCK_2", 1)
      aie.lock("LOCK_3", 1)

      aie.objectFifo("inA", {{"symmetric" : true}}) : aie.objectFifo<si16[64]>
      aie.objectFifo("inB", {{"symmetric" : true}}) : aie.objectFifo<si16[64]>
      aie.objectFifo("outC", {{"symmetric" : true}}) : aie.objectFifo<si16[64]>

      aie.kernel("c", "{kernel_xo}") {{
        aie.ctrlin("inA")
        aie.ctrlin("inB")
        aie.ctrlout("outC")
      }}

      aie.dma_copy("inA", "outC")
      aie.dma_copy("inB", "outC")
    }}
  }}
}}
"""

        mlir_file = tmpdir / "design.mlir"
        mlir_file.write_text(mlir)
        xclbin = tmpdir / "gemm.xclbin"

        try:
            compile_mlir_module(
                mlir_module=str(mlir_file),
                target_arch=target,
                output_path=str(xclbin),
            )
            print(f"  gemm.xclbin: {xclbin.stat().st_size:,} bytes")
            print("  Design compilation: SUCCESS")
            return 0
        except Exception as e:
            err = str(e)
            last_lines = [
                l
                for l in err.split("\n")
                if "error" in l.lower() or "Error" in l.lower() or "unable" in l.lower()
            ]
            print(f"  Design compilation: FAILED")
            for l in last_lines[-5:]:
                print(f"    {l.strip()[:120]}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
