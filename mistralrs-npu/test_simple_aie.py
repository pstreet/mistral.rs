#!/usr/bin/env python3
"""
Simple AIE test: verify the mlir-aie toolchain works on Strix Halo (XDNA 2).

Tests:
1. Import mlir-aie and Peano
2. Create a minimal device design in MLIR using the decorator API
3. Try to compile with aiecc
4. Verify output

Run: source ~/.local/share/ironenv/bin/activate && python3 test_simple_aie.py
"""

import sys
import os
import tempfile
import subprocess


def check_toolchain():
    """Verify all toolchain components are present."""
    print("=== Toolchain Check ===\n")
    ok = True

    result = subprocess.run(["aiecc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  aiecc:  {result.stdout.strip()}")
    else:
        print(f"  aiecc:  NOT FOUND")
        ok = False

    ironenv = os.environ.get("NPU_VENV", os.path.expanduser("~/.local/share/ironenv"))
    for pyver in ["python3.13", "python3.12"]:
        p = f"{ironenv}/lib/{pyver}/site-packages/llvm-aie/bin/clang"
        if os.path.isfile(p):
            result = subprocess.run([p, "--version"], capture_output=True, text=True)
            print(f"  Peano:  {result.stdout.strip().split(chr(10))[0]}")
            break
    else:
        print(f"  Peano:  NOT FOUND")
        ok = False

    import ctypes

    try:
        ctypes.CDLL("libxrt_coreutil.so.2")
        print(f"  XRT:    loaded")
    except Exception:
        print(f"  XRT:    not loadable (OK for compile-only test)")

    return ok


def build_minimal_design():
    """Build a minimal MLIR-AIE design for Strix Halo (XDNA 2 / AIE2P)."""
    from aie.dialects.aie import (
        device,
        tile,
        buffer,
        core,
        end,
        objectfifo,
        shim_dma,
        shim_dma_allocation,
        get_stream,
        objectfifo_dma_endpoint,
        AIEDevice,
    )
    from aie.extras.context import mlir_mod_ctx

    # The device function is a decorator — needs mlir_mod_ctx active
    with mlir_mod_ctx():

        @device(AIEDevice.npu2)
        def my_design():
            # Shim tile 0,0 — DDR interface
            shim = tile(0, 0)
            with mlir_mod_ctx():
                shim_dma_allocation(shim, channel=0, size=256, tcdma_channel=0)
                A_fifo = objectfifo("A_in", shim, 16, tile_size=(128, 128))

            # Compute tile 0,1
            comp = tile(0, 1)
            with mlir_mod_ctx():
                buf_a = buffer("buf_a", 256, dtype="i8")
                fifo_in = objectfifo("fifo_in", comp, 1, tile_size=(128, 128))
                stream_in = get_stream("shim_t0_noc_0", 0, (0, 1), 0)
                objectfifo_dma_endpoint(fifo_in, "from_shim", stream_in)
                core(comp, "add_arrays.o")

        return my_design


def compile_design(mlir_str, tmpdir):
    """Compile MLIR to .xclbin using aiecc."""
    mlir_file = os.path.join(tmpdir, "design.mlir")
    xclbin_path = os.path.join(tmpdir, "design.xclbin")

    with open(mlir_file, "w") as f:
        f.write(mlir_str)

    print(
        f"  MLIR ({len(mlir_str)} chars):\n  {'\n  '.join(mlir_str[:600].split(chr(10)))}"
    )
    print(f"  ...")

    result = subprocess.run(
        ["aiecc", "--target=aie2p", mlir_file, "-o", xclbin_path],
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        print(f"\n  aiecc exit: {result.returncode}")
        stderr = result.stderr.strip()
        for line in stderr.split("\n")[-20:]:
            print(f"    {line}")
        return False, result.stderr

    if os.path.exists(xclbin_path):
        size = os.path.getsize(xclbin_path)
        print(f"  ✓ {xclbin_path} ({size:,} bytes)")
        return True, ""
    return False, "No output"


def main():
    print()

    if not check_toolchain():
        print("\n✗ Toolchain incomplete. Run: ./scripts/setup-npu-aie.sh")
        return 1

    print("\n=== Building MLIR Design ===\n")
    try:
        dev = build_minimal_design()
        mlir_str = str(dev)
        print(f"  ✓ Design built ({len(mlir_str)} chars)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n=== Compiling with aiecc ===\n")
    with tempfile.TemporaryDirectory() as tmpdir:
        success, _ = compile_design(mlir_str, tmpdir)
        if success:
            print("\n✓ Compilation succeeded!")
            return 0
        else:
            print("\n✗ Compilation failed — may need design refinement")
            return 1


if __name__ == "__main__":
    sys.exit(main())
