// MLIR-AIE design for GEMM on AIE2P
// This design:
// 1. Declares 1 compute tile
// 2. Creates ObjectFIFOs for A, B, C buffers
// 3. Links the gemm_kernel.cpp compiled .xo object
// 4. Sets up DMA for data movement
//
// Usage: aiecc.py --target=aie2 --compile-mode=design gemm.mlir -o gemm.xclbin

// We'll test with aiecc compile_mlir_module via the Python API.
