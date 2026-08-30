#include "ck_flash_attn_fwd.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"

// Forward declarations of our explicit template instantiations (gfx1151).
// Each .cpp file defines fmha_fwd_<specific_trait, ck_tile::gfx115_t>(stream_config, args).

// hdim=128, tile 64x64
extern template float fmha_fwd_<
    fmha_fwd_traits_<128, FmhaFwdBf16, false, 64, 64, 32, 128, 32, 128, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false,
        ck_tile::SimplifiedGenericAttentionMask<false>,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true, false, false, false, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_args);

extern template float fmha_fwd_<
    fmha_fwd_traits_<128, FmhaFwdBf16, false, 64, 64, 32, 128, 32, 128, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false,
        ck_tile::SimplifiedGenericAttentionMask<true>,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true, false, false, false, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_args);

// hdim=128, tile 128x64
extern template float fmha_fwd_<
    fmha_fwd_traits_<128, FmhaFwdBf16, false, 128, 64, 32, 128, 32, 128, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false,
        ck_tile::SimplifiedGenericAttentionMask<false>,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true, false, false, false, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_args);

extern template float fmha_fwd_<
    fmha_fwd_traits_<128, FmhaFwdBf16, false, 128, 64, 32, 128, 32, 128, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false,
        ck_tile::SimplifiedGenericAttentionMask<true>,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true, false, false, false, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_args);

// hdim=256, tile 128x64
extern template float fmha_fwd_<
    fmha_fwd_traits_<256, FmhaFwdBf16, false, 128, 64, 32, 256, 32, 256, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false,
        ck_tile::SimplifiedGenericAttentionMask<false>,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true, false, false, false, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_args);

extern template float fmha_fwd_<
    fmha_fwd_traits_<256, FmhaFwdBf16, false, 128, 64, 32, 256, 32, 256, true,
        ck_tile::BlockFmhaPipelineEnum::QRKSVS, false,
        ck_tile::SimplifiedGenericAttentionMask<true>,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true, false, false, false, false, false>,
    ck_tile::gfx115_t>(const ck_tile::stream_config&, fmha_fwd_args);

// Type alias for readability.
using no_mask_t = ck_tile::SimplifiedGenericAttentionMask<false>;
using mask_t    = ck_tile::SimplifiedGenericAttentionMask<true>;

template <ck_tile::index_t HDim, typename MaskType>
static float dispatch_bf16_batch(
    const ck_tile::stream_config& s,
    fmha_fwd_args a)
{
    constexpr ck_tile::index_t M0 = (HDim == 128) ? 128 : 128;
    constexpr ck_tile::index_t N0 = 64;
    constexpr ck_tile::index_t K0 = 32;
    constexpr ck_tile::index_t N1 = HDim;
    constexpr ck_tile::index_t K1 = 32;
    constexpr ck_tile::index_t K0BL = HDim;

    using trait = fmha_fwd_traits_<
        HDim, FmhaFwdBf16, false,
        M0, N0, K0, N1, K1, K0BL,
        true,  // kIsVLayoutRowMajor
        ck_tile::BlockFmhaPipelineEnum::QRKSVS,
        false,  // kHasLogitsSoftCap
        MaskType,
        ck_tile::BlockAttentionBiasEnum::NO_BIAS,
        true,   // kStoreLse
        false,  // kHasDropout
        ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        true, true,   // kPadS, kPadSK
        false, false, false, false>;  // kPadD, kPadDv, kUseTrLoad, kSkipMinSeqlenQ, kHasSink

    return fmha_fwd_<trait, ck_tile::gfx115_t>(s, a);
}

extern "C" float ck_flash_attn_fwd(
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
    int32_t mask_type,
    int64_t stream)
{
    fmha_fwd_args a;
    a.q_ptr         = q_ptr;
    a.k_ptr         = k_ptr;
    a.v_ptr         = v_ptr;
    a.bias_ptr      = nullptr;
    a.q_descale_ptr = nullptr;
    a.k_descale_ptr = nullptr;
    a.v_descale_ptr = nullptr;
    a.rand_val_ptr  = nullptr;
    a.lse_ptr       = lse_ptr;
    a.o_ptr         = o_ptr;

    a.seqstart_q_ptr = nullptr;
    a.seqstart_k_ptr = nullptr;
    a.seqlen_q_ptr   = nullptr;
    a.seqlen_k_ptr   = nullptr;
    a.cu_seqlen_q_ptr = nullptr;
    a.cu_seqlen_k_ptr = nullptr;
    a.block_scale_seqstart_q_ptr = nullptr;
    a.block_scale_seqstart_k_ptr = nullptr;
    a.seqstart_v_scale_ptr = nullptr;
    a.sink_ptr       = nullptr;

    a.seqlen_q       = seqlen_q;
    a.seqlen_k       = seqlen_k;
    a.batch          = batch;
    a.max_seqlen_q   = seqlen_q;
    a.hdim_q         = hdim_q;
    a.hdim_v         = hdim_v;
    a.nhead_q        = nhead_q;
    a.nhead_k        = nhead_k;
    a.num_head_q_total = nhead_q;
    a.head_start     = 0;

    a.scale_s        = scale_s;
    a.logits_soft_cap = 0.0f;

    a.stride_q       = stride_q;
    a.stride_k       = stride_k;
    a.stride_v       = stride_v;
    a.stride_bias    = 0;
    a.stride_randval = 0;
    a.stride_o       = stride_o;
    a.stride_q_descale = 0;
    a.stride_k_descale = 0;
    a.stride_v_descale = 0;
    a.nhead_stride_q = nhead_stride_q;
    a.nhead_stride_k = nhead_stride_k;
    a.nhead_stride_v = nhead_stride_v;
    a.nhead_stride_bias = 0;
    a.nhead_stride_randval = 0;
    a.nhead_stride_lse = (int64_t)seqlen_q;  // LSE is [B, H, S]
    a.nhead_stride_o = nhead_stride_o;
    a.nhead_stride_q_descale = 0;
    a.nhead_stride_k_descale = 0;
    a.nhead_stride_v_descale = 0;
    a.batch_stride_q = batch_stride_q;
    a.batch_stride_k = batch_stride_k;
    a.batch_stride_v = batch_stride_v;
    a.batch_stride_bias = 0;
    a.batch_stride_randval = 0;
    a.batch_stride_lse = (int64_t)nhead_q * seqlen_q;
    a.batch_stride_o = batch_stride_o;
    a.batch_stride_q_descale = 0;
    a.batch_stride_k_descale = 0;
    a.batch_stride_v_descale = 0;

    a.window_size_left  = -1;
    // causal needs a zero right-window; -1 would widen it to full attention
    a.window_size_right = (mask_type != 0) ? 0 : -1;
    a.sink_size         = 0;
    a.mask_type         = mask_type;
    a.min_seqlen_q      = 0;

    a.p_drop    = 0.0f;
    a.s_randval = false;
    a.drop_seed_offset = std::make_pair((uint64_t)0, (uint64_t)0);

    a.block_scale_size_q  = 0;
    a.block_scale_size_kv = 0;

    ck_tile::stream_config s;
    s.stream_id_ = reinterpret_cast<hipStream_t>(stream);
    s.time_kernel_ = false;

    if (hdim_q == 256 && hdim_v == 256) {
        if (mask_type != 0)
            return dispatch_bf16_batch<256, mask_t>(s, a);
        else
            return dispatch_bf16_batch<256, no_mask_t>(s, a);
    } else if (hdim_q == 128 && hdim_v == 128) {
        if (mask_type != 0)
            return dispatch_bf16_batch<128, mask_t>(s, a);
        else
            return dispatch_bf16_batch<128, no_mask_t>(s, a);
    }

    fprintf(stderr, "ck_flash_attn_fwd: unsupported hdim_q=%d hdim_v=%d\n", hdim_q, hdim_v);
    return -1.0f;
}
