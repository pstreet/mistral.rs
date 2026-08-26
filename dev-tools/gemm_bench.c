// GEMM throughput at the real Qwen3.5-4B prefill shapes (M=910 tokens).
// Compares plain hipBLAS BF16 (what the model uses) vs hipBLASLt F16.
// Shapes: gate/up [M=910,K=2560,N=9216], down [M=910,K=9216,N=2560],
//         gdn in_proj [M=910,K=2560,N=8192], gdn out_proj [M=910,K=4096,N=2560]
#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// C[M,N] = A[M,K] * B[K,N], row-major logical. Using column-major trick with TN.
// We store A as [K,M] col-major ld=K (i.e. A^T), B as [K,N] col-major ld=K.
static double bench_bf16(int M, int N, int K, int iters, void *A, void *B, void *C, hipStream_t s) {
    hipblasHandle_t h; hipblasCreate(&h); hipblasSetStream(h, s);
    float alpha = 1.f, beta = 0.f;
    // col-major: C' [N,M] = B' [N,K] * A' [K,M]; B' is opN (B stored [K,N] ld K -> treat as [N,K]? )
    // Simpler: use hipblasGemmEx with opA=T, opB=N on [K,M]x[K,N]
    for (int i = 0; i < 5; i++)
        hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &alpha,
                      A, HIP_R_16BF, K, B, HIP_R_16BF, K, &beta, C, HIP_R_16BF, M,
                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    hipStreamSynchronize(s);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &alpha,
                      A, HIP_R_16BF, K, B, HIP_R_16BF, K, &beta, C, HIP_R_16BF, M,
                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    hipStreamSynchronize(s);
    double dt = now_ms() - t0;
    double tf = 2.0 * M * N * K * iters / (dt * 1e6);
    hipblasDestroy(h);
    return dt / iters; (void)tf;
}

static double bench_f16lt(int M, int N, int K, int iters, void *A, void *B, void *C, void *ws, size_t ws_alloc, hipStream_t s) {
    hipblasLtHandle_t lt; hipblasLtCreate(&lt);
    __half ha = __float2half(1.f), hb = __float2half(0.f);
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_16F);
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    hipblasLtMatrixLayout_t la, lb, lc;
    hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, K, M, K);
    hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K);
    hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M);
    hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
    size_t wsa = ws_alloc;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsa, sizeof(wsa));
    int count = 0; hipblasLtMatmulHeuristicResult_t heur;
    if (hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &count) != HIPBLAS_STATUS_SUCCESS || count == 0) {
        printf("  (no heuristic for this shape)\n");
        hipblasLtDestroy(lt); return -1;
    }
    for (int i = 0; i < 5; i++)
        hipblasLtMatmul(lt, desc, &ha, A, la, B, lb, &hb, C, lc, C, lc, &heur.algo, ws, ws_alloc, s);
    hipStreamSynchronize(s);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        hipblasLtMatmul(lt, desc, &ha, A, la, B, lb, &hb, C, lc, C, lc, &heur.algo, ws, ws_alloc, s);
    hipStreamSynchronize(s);
    double dt = now_ms() - t0;
    hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatmulPreferenceDestroy(pref); hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(lt);
    return dt / iters;
}

static void report(const char *name, int M, int N, int K, double bf16_ms, double f16_ms) {
    double tf = 2.0 * M * N * K;
    printf("%-14s M=%-5d N=%-5d K=%-5d | BF16 %7.2f ms %6.1f TF | F16Lt %7.2f ms %6.1f TF\n",
           name, M, N, K, bf16_ms, bf16_ms > 0 ? tf/(bf16_ms*1e6) : 0, f16_ms, f16_ms > 0 ? tf/(f16_ms*1e6) : 0);
}

int main(void) {
    (void)hipInit(0);
    const int ITERS = 50;
    const size_t MAXA = (size_t)9216 * 910 * 2;     // A[K,M] max
    const size_t MAXB = (size_t)9216 * 2560 * 2;    // B[K,N] max
    const size_t MAXC = (size_t)9216 * 910 * 2;     // C[M,N] max
    void *A, *B, *C, *ws; size_t ws_alloc = 256 * 1024 * 1024;
    hipMalloc(&A, MAXA); hipMalloc(&B, MAXB); hipMalloc(&C, MAXC); hipMalloc(&ws, ws_alloc);
    hipMemset(A, 0x11, MAXA); hipMemset(B, 0x22, MAXB);
    hipStream_t s; hipStreamCreate(&s);

    struct { const char *n; int M, N, K; } shapes[] = {
        {"gate/up", 910, 9216, 2560},
        {"down",    910, 2560, 9216},
        {"gdn_in",  910, 8192, 2560},
        {"gdn_out", 910, 2560, 4096},
    };
    printf("=== Qwen3.5-4B prefill GEMM shapes (M=910) ===\n");
    for (int i = 0; i < 4; i++) {
        double b = bench_bf16(shapes[i].M, shapes[i].N, shapes[i].K, ITERS, A, B, C, s);
        double f = bench_f16lt(shapes[i].M, shapes[i].N, shapes[i].K, ITERS, A, B, C, ws, ws_alloc, s);
        report(shapes[i].n, shapes[i].M, shapes[i].N, shapes[i].K, b, f);
    }
    hipFree(A); hipFree(B); hipFree(C); hipFree(ws);
    return 0;
}
