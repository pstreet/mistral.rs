// Test the EXACT layout the Rust candle code uses for the failing GEMM:
//   A (ta=T): rows=k=5120, cols=m=10240, ld=k=5120
//   B (tb=N): rows=k=5120, cols=n=736,   ld=k=5120
//   C:        rows=m=10240, cols=n=736,  ld=m=10240
// Compares ld variants to see which the heuristic accepts.
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void test(hipblasLtHandle_t lt, const char *label, hipblasComputeType_t compute,
                 int a_rows, int a_cols, int lda, int b_rows, int b_cols, int ldb,
                 int c_rows, int c_cols, int ldc) {
    hipDataType scale = (compute == HIPBLAS_COMPUTE_16F) ? HIP_R_16F : HIP_R_32F;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, compute, scale);
    hipblasOperation_t ta = (a_rows == 5120 && a_cols == 10240) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    hipblasOperation_t tb = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));

    hipblasLtMatrixLayout_t la, lb, lc;
    hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, a_rows, a_cols, lda);
    hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, b_rows, b_cols, ldb);
    hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, c_rows, c_cols, ldc);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    size_t ws = 32 * 1024 * 1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    int count = 0;
    hipblasLtMatmulHeuristicResult_t heur;
    hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &count);
    printf("%-40s compute=%s -> status=%d count=%d\n", label,
           (compute == HIPBLAS_COMPUTE_16F) ? "16F" : "32F", st, count);

    hipblasLtMatrixLayoutDestroy(la);
    hipblasLtMatrixLayoutDestroy(lb);
    hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(desc);
}

int main(void) {
    (void)hipInit(0);
    hipblasLtHandle_t lt;
    hipblasLtCreate(&lt);
    void *wsb;
    hipMalloc(&wsb, 32 * 1024 * 1024);

    for (int c = 0; c < 2; c++) {
        hipblasComputeType_t compute = c == 0 ? HIPBLAS_COMPUTE_16F : HIPBLAS_COMPUTE_32F;
        // EXACT layout from the Rust code (A ld=k=5120)
        test(lt, "rust-exact (A ld=k=5120)", compute, 5120, 10240, 5120, 5120, 736, 5120, 10240, 736, 10240);
        // A ld=m=10240 variant
        test(lt, "A ld=m=10240", compute, 5120, 10240, 10240, 5120, 736, 5120, 10240, 736, 10240);
        // B ld=n=736 variant
        test(lt, "B ld=n=736", compute, 5120, 10240, 5120, 5120, 736, 736, 10240, 736, 10240);
        // C ld=n=736 variant
        test(lt, "C ld=n=736", compute, 5120, 10240, 5120, 5120, 736, 5120, 10240, 736, 736);
    }
    hipFree(wsb);
    hipblasLtDestroy(lt);
    return 0;
}
