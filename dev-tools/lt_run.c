// Actually RUN hipblasLtMatmul for the failing shape (n=3040) and the working shape
// (n=736), using COMPUTE_16F. Reports the matmul status and a checksum of the output.
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_gemm(hipblasLtHandle_t lt, int m, int n, int k, int do16) {
    hipblasComputeType_t compute = do16 ? HIPBLAS_COMPUTE_16F : HIPBLAS_COMPUTE_32F;
    hipDataType scale = do16 ? HIP_R_16F : HIP_R_32F;

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, compute, scale);
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));

    // A stored as [k, m] col-major (ld=k), B [k, n] col-major (ld=k), C [m, n] col-major (ld=m)
    hipblasLtMatrixLayout_t la, lb, lc;
    hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k);
    hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, n, k);
    hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, m, n, m);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    size_t ws_alloc = 256 * 1024 * 1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_alloc, sizeof(ws_alloc));

    int count = 0;
    hipblasLtMatmulHeuristicResult_t heur;
    hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &count);
    if (st != HIPBLAS_STATUS_SUCCESS || count == 0) {
        printf("  n=%d compute=%s heuristic FAILED st=%d count=%d\n", n, do16 ? "16F" : "32F", st, count);
        return 1;
    }

    size_t Asz = (size_t)k * m * 2, Bsz = (size_t)k * n * 2, Csz = (size_t)m * n * 2;
    void *A, *B, *C, *ws;
    hipMalloc(&A, Asz); hipMalloc(&B, Bsz); hipMalloc(&C, Csz); hipMalloc(&ws, ws_alloc);
    hipMemset(A, 0x11, Asz); hipMemset(B, 0x22, Bsz); hipMemset(C, 0, Csz);

    __half halpha = __float2half(1.0f), hbeta = __float2half(0.0f);
    float falpha = 1.0f, fbeta = 0.0f;
    const void *alpha = do16 ? (const void*)&halpha : (const void*)&falpha;
    const void *beta = do16 ? (const void*)&hbeta : (const void*)&fbeta;

    hipStream_t stream;
    hipStreamCreate(&stream);
    hipblasStatus_t mst = hipblasLtMatmul(lt, desc, alpha, A, la, B, lb, beta, C, lc, C, lc,
                                          &heur.algo, ws, ws_alloc, stream);
    hipStreamSynchronize(stream);
    hipError_t sync = hipGetLastError();

    // checksum C
    unsigned char *hC = (unsigned char*)malloc(Csz);
    hipMemcpy(hC, C, Csz, hipMemcpyDeviceToHost);
    unsigned long long sum = 0;
    for (size_t i = 0; i < Csz; i++) sum += hC[i];
    printf("  n=%d compute=%s matmul st=%d hiperr=%d sum=%llu\n", n, do16 ? "16F" : "32F", mst, (int)sync, sum);

    hipFree(A); hipFree(B); hipFree(C); hipFree(ws);
    hipStreamDestroy(stream);
    hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(desc);
    return 0;
}

int main(void) {
    (void)hipInit(0);
    hipblasLtHandle_t lt;
    hipblasLtCreate(&lt);
    for (int do16 = 1; do16 >= 0; do16--) {
        printf("== compute=%s ==\n", do16 ? "16F" : "32F");
        run_gemm(lt, 10240, 736, 5120, do16);
        run_gemm(lt, 10240, 3040, 5120, do16);
    }
    hipblasLtDestroy(lt);
    return 0;
}
