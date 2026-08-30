#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// BSHD layout: Q[B, H, S, D], K[B, Hkv, S, D], V[B, Hkv, S, D]
// All strides are in elements (not bytes).
// Returns execution time in ms (always 0.0 when timing disabled).
float ck_flash_attn_fwd(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    void* lse_ptr,
    int32_t batch,
    int32_t nhead_q,
    int32_t nhead_k,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t hdim_q,
    int32_t hdim_v,
    int64_t stride_q,
    int64_t stride_k,
    int64_t stride_v,
    int64_t stride_o,
    int64_t nhead_stride_q,
    int64_t nhead_stride_k,
    int64_t nhead_stride_v,
    int64_t nhead_stride_o,
    int64_t batch_stride_q,
    int64_t batch_stride_k,
    int64_t batch_stride_v,
    int64_t batch_stride_o,
    float scale_s,
    int32_t mask_type,  // 0=no mask, 1=causal (bottom-right)
    int64_t stream);

#ifdef __cplusplus
}
#endif
