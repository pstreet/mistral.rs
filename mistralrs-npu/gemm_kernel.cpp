// AIE2P GEMM kernel — BF16 matrix multiply via aie::mac()
//
// Compiled with Peano (llvm-aie) for Strix Halo AIE2P.
// Uses scalar BF16 MAC to avoid G_FADD legalization crash on vector BF16 add.
//
// Interface: gemm_accum_bf16(A, B, C, M, N, K)
//   C[M x N] += A[M x K] * B[K x N]  (accumulation into C)
//
// Memory layout: row-major
//   A[m*K + k], B[k*N + n], C[m*N + n]

#include "aie_api/aie.hpp"

// ---------------------------------------------------------------------------
// Core GEMM: C[m][n] += A[m][k] * B[k][n]
// ---------------------------------------------------------------------------
extern "C" void gemm_accum_bf16(
    const bfloat16* __restrict__ A,  // MxK row-major
    const bfloat16* __restrict__ B,  // KxN row-major
    bfloat16* __restrict__ C,        // MxN row-major
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            bfloat16 sum = C[m * N + n];  // start from existing C
            for (int k = 0; k < K; k++) {
                sum = aie::mac<bfloat16>(sum, A[m * K + k], B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM: C = A * B (no accumulation — writes zeros first)
// ---------------------------------------------------------------------------
extern "C" void gemm_bf16(
    const bfloat16* __restrict__ A,  // MxK row-major
    const bfloat16* __restrict__ B,  // KxN row-major
    bfloat16* __restrict__ C,        // MxN row-major
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

// ---------------------------------------------------------------------------
// GEMM with vectorized load of A rows — loads 16 BF16 elements at once
// ---------------------------------------------------------------------------
extern "C" void gemm_vec_bf16(
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
