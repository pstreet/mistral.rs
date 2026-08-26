// Benchmark for the real prefill GEMM shape (m=10240, k=5120, n=928, TN, F16,
// COMPUTE_16F), three paths:
//  1. Lt-percall: full descriptor setup + heuristic + matmul every call (un-optimized Rust code)
//  2. Lt-cached:  setup once, matmul only per call (what descriptor caching would give)
//  3. plain:      hipblasGemmStridedBatchedEx (COMPUTE_16F)
// NOTE: printed "TFLOPS" values carry a 1000x unit bug; relative comparison is valid.
#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
    (void)hipInit(0);
    const int m = 10240, n = 928, k = 5120;
    const int ITERS = 30;

    size_t Asz = (size_t)k * m * 2, Bsz = (size_t)k * n * 2, Csz = (size_t)m * n * 2;
    void *A, *B, *C, *ws;
    size_t ws_alloc = 256 * 1024 * 1024;
    hipMalloc(&A, Asz); hipMalloc(&B, Bsz); hipMalloc(&C, Csz); hipMalloc(&ws, ws_alloc);
    hipMemset(A, 0x11, Asz); hipMemset(B, 0x22, Bsz); hipMemset(C, 0, Csz);
    hipStream_t stream;
    hipStreamCreate(&stream);

    double flops = 2.0 * m * n * k * ITERS;

    // ---- 3. plain hipBLAS, COMPUTE_16F
    {
        hipblasHandle_t h;
        hipblasCreate(&h);
        hipblasSetStream(h, stream);
        __half ha = __float2half(1.0f), hb = __float2half(0.0f);
        const hipblasHalf *ha2 = (const hipblasHalf*)&ha, *hb2 = (const hipblasHalf*)&hb;
        // warmup
        for (int i = 0; i < 5; i++)
            hipblasHgemm(h, hipblasOperation_t::HIPBLAS_OP_T, hipblasOperation_t::HIPBLAS_OP_N,
                         m, n, k, ha2, (const hipblasHalf*)A, k, (const hipblasHalf*)B, k, hb2,
                         (hipblasHalf*)C, m);
        hipStreamSynchronize(stream);
        double t0 = now_ms();
        for (int i = 0; i < ITERS; i++)
            hipblasHgemm(h, hipblasOperation_t::HIPBLAS_OP_T, hipblasOperation_t::HIPBLAS_OP_N,
                         m, n, k, ha2, (const hipblasHalf*)A, k, (const hipblasHalf*)B, k, hb2,
                         (hipblasHalf*)C, m);
        hipStreamSynchronize(stream);
        double dt = now_ms() - t0;
        printf("plain-16F     : %8.1f ms/iter  %7.2f TFLOPS\n", dt / ITERS, flops / (dt * 1e6));
        hipblasDestroy(h);
    }

    // ---- 1 & 2. hipBLASLt
    {
        hipblasLtHandle_t lt;
        hipblasLtCreate(&lt);
        __half ha = __float2half(1.0f), hb = __float2half(0.0f);

        // --- per-call
        double percall = 0;
        for (int i = 0; i < ITERS; i++) {
            double t0 = now_ms();
            hipblasLtMatmulDesc_t desc;
            hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_16F);
            hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
            hipblasLtMatrixLayout_t la, lb, lc;
            hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k);
            hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, n, k);
            hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, m, n, m);
            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            size_t wsa = ws_alloc;
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsa, sizeof(wsa));
            int count = 0;
            hipblasLtMatmulHeuristicResult_t heur;
            hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &count);
            st = hipblasLtMatmul(lt, desc, (const void*)&ha, A, la, B, lb, (const void*)&hb, C, lc, C, lc,
                                 &heur.algo, ws, ws_alloc, stream);
            hipStreamSynchronize(stream);
            percall += now_ms() - t0;
            hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatmulDescDestroy(desc);
            if (st != HIPBLAS_STATUS_SUCCESS) { printf("  ltmul status=%d\n", (int)st); break; }
        }
        printf("Lt-percall    : %8.1f ms/iter  %7.2f TFLOPS\n", percall / ITERS, flops / (percall * 1e6));

        // --- cached (setup once)
        hipblasLtMatmulDesc_t desc;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_16F);
        hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        hipblasLtMatrixLayout_t la, lb, lc;
        hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k);
        hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, n, k);
        hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, m, n, m);
        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        size_t wsa = ws_alloc;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsa, sizeof(wsa));
        int count = 0;
        hipblasLtMatmulHeuristicResult_t heur;
        hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &count);
        for (int i = 0; i < 5; i++)
            hipblasLtMatmul(lt, desc, (const void*)&ha, A, la, B, lb, (const void*)&hb, C, lc, C, lc,
                            &heur.algo, ws, ws_alloc, stream);
        hipStreamSynchronize(stream);
        double t0 = now_ms();
        for (int i = 0; i < ITERS; i++)
            hipblasLtMatmul(lt, desc, (const void*)&ha, A, la, B, lb, (const void*)&hb, C, lc, C, lc,
                            &heur.algo, ws, ws_alloc, stream);
        hipStreamSynchronize(stream);
        double cached = now_ms() - t0;
        printf("Lt-cached     : %8.1f ms/iter  %7.2f TFLOPS\n", cached / ITERS, flops / (cached * 1e6));

        hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(lt);
    }

    hipFree(A); hipFree(B); hipFree(C); hipFree(ws);
    return 0;
}
