#include "ck_flash_attn_splitkv_fwd.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"

// Forward declarations of our gfx1151 specializations (fmha_fwd_splitkv_d128_bf16_mask_paged_gfx115.cpp).
using splitkv_mask_t = ck_tile::SimplifiedGenericAttentionMask<true>;

extern template void fmha_fwd_splitkv_oneshot_<
    fmha_fwd_splitkv_traits_<
        128, FmhaFwdBf16, true, 64, 64, 32, 128, 32, 128, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false, splitkv_mask_t,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS, true, false, true, false,
        true, true, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_splitkv_args);

extern template void fmha_fwd_splitkv_combine_oneshot_<
    fmha_fwd_splitkv_combine_traits_<
        128, FmhaFwdBf16, true, 32, true, false, true, true>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_splitkv_args);

using splitkv_trait = fmha_fwd_splitkv_traits_<
    128, FmhaFwdBf16, true, 64, 64, 32, 128, 32, 128, true,
    ck_tile::BlockFmhaPipelineEnum::QRKSVS, false, splitkv_mask_t,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS, true, false, true, false,
    true, true, false, false>;

using splitkv_combine_trait = fmha_fwd_splitkv_combine_traits_<
    128, FmhaFwdBf16, true, 32, true, false, true, true>;

extern "C" float ck_flash_attn_splitkv_fwd(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    void* lse_ptr,
    void* o_acc_ptr,
    void* lse_acc_ptr,
    const int32_t* cu_seqlens_q,
    const int32_t* cu_seqlens_k,
    const int32_t* seqlen_k,
    const int32_t* block_table,
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
    int64_t stream)
{
    if(hdim != 128 || nhead_q % nhead_k != 0 || num_splits < 1 || num_splits > 16)
    {
        fprintf(stderr,
                "ck_flash_attn_splitkv_fwd: unsupported shape hdim=%d nhead_q=%d nhead_k=%d "
                "num_splits=%d\n",
                hdim,
                nhead_q,
                nhead_k,
                num_splits);
        return -1.0f;
    }

    fmha_fwd_splitkv_args a;
    a.q_ptr       = q_ptr;
    a.k_ptr       = k_ptr;
    a.v_ptr       = v_ptr;
    a.bias_ptr    = nullptr;
    a.lse_acc_ptr = lse_acc_ptr;
    a.o_acc_ptr   = o_acc_ptr;
    a.lse_ptr     = lse_ptr;
    a.o_ptr       = o_ptr;

    a.block_table_ptr             = block_table
                                       ? const_cast<void*>(reinterpret_cast<const void*>(block_table))
                                       : nullptr;
    a.batch_stride_block_table    = block_table_row_stride;
    a.page_block_size             = block_table ? page_size : 0;
    a.is_gappy                    = false;
    a.cache_batch_idx             = nullptr;

    a.seqstart_q_ptr = cu_seqlens_q;
    a.seqstart_k_ptr = cu_seqlens_k;
    a.seqlen_k_ptr   = seqlen_k;
    a.sink_ptr       = nullptr;

    a.seqlen_q     = max_seqlen_q;
    a.seqlen_k     = max_seqlen_k;
    a.batch        = batch;
    a.max_seqlen_q = max_seqlen_q;
    a.hdim_q       = hdim;
    a.hdim_v       = hdim;
    a.nhead_q      = nhead_q;
    a.nhead_k      = nhead_k;
    a.num_splits   = num_splits;

    a.scale_s        = scale_s;
    a.scale_p        = 1.0f;
    a.scale_o        = 1.0f;
    a.logits_soft_cap = 0.0f;

    // q/o are [total_q, nhead, hdim] contiguous; k/v are [page, page_size, nhead_k, hdim].
    a.stride_q            = static_cast<ck_tile::index_t>(nhead_q) * hdim;
    a.stride_k            = static_cast<ck_tile::index_t>(nhead_k) * hdim;
    a.stride_v            = static_cast<ck_tile::index_t>(nhead_k) * hdim;
    a.stride_bias         = 0;
    a.stride_o_acc        = hdim;
    a.stride_o            = static_cast<ck_tile::index_t>(nhead_q) * hdim;
    a.nhead_stride_q      = hdim;
    a.nhead_stride_k      = hdim;
    a.nhead_stride_v      = hdim;
    a.nhead_stride_bias   = 0;
    a.nhead_stride_lse    = total_q;
    a.nhead_stride_lse_acc =
        static_cast<ck_tile::index_t>(num_splits) * total_q;
    a.nhead_stride_o_acc =
        static_cast<ck_tile::index_t>(num_splits) * total_q * hdim;
    a.nhead_stride_o = hdim;
    a.batch_stride_q   = 0;
    a.batch_stride_k   = static_cast<ck_tile::index_t>(page_size) * nhead_k * hdim;
    a.batch_stride_v   = static_cast<ck_tile::index_t>(page_size) * nhead_k * hdim;
    a.batch_stride_bias  = 0;
    a.batch_stride_lse   = 0;
    a.batch_stride_lse_acc = 0;
    a.batch_stride_o_acc   = 0;
    a.batch_stride_o       = 0;
    a.split_stride_lse_acc = total_q;
    a.split_stride_o_acc   = static_cast<ck_tile::index_t>(total_q) * hdim;

    a.window_size_left  = window_size_left;
    a.window_size_right = window_size_right;
    a.sink_size         = 0;
    a.mask_type         = mask_type;

    ck_tile::stream_config s;
    s.stream_id_  = reinterpret_cast<hipStream_t>(stream);
    s.time_kernel_ = false;

    // NOTE: the gfx1151 splitkv instantiation memory-faults on hip in its base
    // group-varlen path (reproduced with the standalone harness in
    // LocalAI/scratch/splitkv-debug, independent of pages/windows/splits/
    // masks). Do not call from serving paths until that is root-caused; see
    // the TODO dflash entry for the evidence and remaining options.
    fmha_fwd_splitkv_oneshot_<splitkv_trait, ck_tile::gfx115_t>(s, a);
    fmha_fwd_splitkv_combine_oneshot_<splitkv_combine_trait, ck_tile::gfx115_t>(s, a);
    return 0.0f;
}
