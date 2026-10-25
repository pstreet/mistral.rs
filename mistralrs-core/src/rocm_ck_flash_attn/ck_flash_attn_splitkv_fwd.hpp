#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Varlen group-mode flash attention over a paged KV cache (CK split-KV).
// FA-style windows: mask_type 0 = no mask, 1 = causal top-left,
// 2 = causal bottom-right; window_size_left/right -1 = unbounded.
// Workspaces are caller-allocated:
//   o_acc:   [nhead_q, num_splits, total_q, hdim] f32
//   lse_acc: [nhead_q, num_splits, total_q] f32
//   lse:     [nhead_q, total_q] f32
// Returns 0.0 on success, -1.0 on unsupported shape.
float ck_flash_attn_splitkv_fwd(
    const void* q_ptr,              // [total_q, nhead_q, hdim] bf16 contiguous
    const void* k_ptr,              // [num_pages, page_size, nhead_k, hdim] bf16
    const void* v_ptr,              // [num_pages, page_size, nhead_k, hdim] bf16
    void* o_ptr,                    // [total_q, nhead_q, hdim] bf16
    void* lse_ptr,                  // [nhead_q, total_q] f32
    void* o_acc_ptr,                // [nhead_q, num_splits, total_q, hdim] f32
    void* lse_acc_ptr,              // [nhead_q, num_splits, total_q] f32
    const int32_t* cu_seqlens_q,    // [batch + 1] i32, cumulative query lens
    const int32_t* cu_seqlens_k,    // [batch + 1] i32, cumulative kv lens
    const int32_t* seqlen_k,        // [batch] i32, per-batch kv lens (may be null)
    const int32_t* block_table,     // [batch, max_blocks_per_seq] i32, row-major
    int64_t block_table_row_stride,
    int32_t num_pages,
    int32_t page_size,
    int32_t num_splits,
    int32_t batch,
    int32_t total_q,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t nhead_q,
    int32_t nhead_k,
    int32_t hdim,
    float scale_s,
    int32_t window_size_left,
    int32_t window_size_right,
    int32_t mask_type,
    int64_t stream);

#ifdef __cplusplus
}
#endif
