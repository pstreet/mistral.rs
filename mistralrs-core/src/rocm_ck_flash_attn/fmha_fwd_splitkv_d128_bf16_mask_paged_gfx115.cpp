// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// gfx1151 split-KV (paged-KV + varlen group-mode + windowed) instantiation for
// the DFlash draft path. Paged KV via vLLM 2D block tables; page size is a
// runtime arg (the pool default 32 equals kK1, one tile per page). Runtime mask
// via SimplifiedGenericAttentionMask<true>.
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"

#include <iostream>
#include <cstdio>
#include <cstdlib>

#if !defined(__HIP_DEVICE_COMPILE__) || (defined(__gfx115__))

using fmha_dtype = FmhaFwdBf16;

using fmha_block_tile = ck_tile::sequence<64, 64, 32, 128, 32, 128>;

using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 16>,
                                          ck_tile::sequence<4, 1, 1>,
                                          ck_tile::sequence<16, 16, 16>,
                                          true>;

using fmha_traits = ck_tile::TileFmhaFwdSplitKVTraits<
    true,  // kPadSeqLenQ
    true,  // kPadSeqLenK
    false, // kPadHeadDimQ (the working gfx1151 batched TUs pad no head dims)
    false, // kPadHeadDimV
    false, // kHasLogitsSoftCap
    ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    false, // kHasBiasGrad
    true,  // kStoreLSE (the combine pass requires it)
    false, // kDoFp8StaticQuant
    true,  // kIsPagedKV
    true,  // kHasUnevenSplits
    false, // kMergeNumHeadGroupsSeqLenQ
    -1,    // kBlockPerCu (auto occupancy)
    false>; // kHasSink

using fmha_variant = ck_tile::ComposedAttention<false * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;

using fmha_mask = ck_tile::SimplifiedGenericAttentionMask<true>;

using fmha_pipeline_problem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
    fmha_shape,
    true, // kIsGroupMode
    fmha_variant,
    fmha_mask,
    fmha_traits>;

using fmha_pipeline = ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<fmha_pipeline_problem>;

using fmha_epilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                               typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                               true, false>>;

using fmha_kernel = ck_tile::FmhaFwdSplitKVKernel<fmha_pipeline, fmha_epilogue>;

using fmha_combine_traits = ck_tile::TileFmhaFwdSplitKVCombineTraits<
    true, // kPadSeqLenQ
    true, // kPadHeadDimV
    true, // kStoreLSE
    false, // kDoFp8StaticQuant
    4>;    // kLogMaxSplits (16 splits; wave32: kM0=4, 4*16 % 32 == 0)

using fmha_combine_problem = ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
    128, // kHeadDimV
    true, // kIsGroupMode
    32,  // kN1 (wave32: NThreads=8, kM0=4 so MakeLSEaccRegTileDistribution holds)
    fmha_combine_traits>;

using fmha_combine_pipeline = ck_tile::BlockFmhaFwdSplitKVCombinePipeline<fmha_combine_problem>;

using fmha_combine_kernel = ck_tile::FmhaFwdSplitKVCombineKernel<fmha_combine_pipeline, fmha_epilogue>;

// Trait keys (mirror the generated naming; our dispatcher calls these directly).
using trait = fmha_fwd_splitkv_traits_<
    128, FmhaFwdBf16, true, 64, 64, 32, 128, 32, 128, true,
    ck_tile::BlockFmhaPipelineEnum::QRKSVS, false, fmha_mask,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS, true, false, true, false,
    true, true, false, false>;

using combine_trait = fmha_fwd_splitkv_combine_traits_<
    128, FmhaFwdBf16, true, 32, true, false, true, true>;

template<>
void fmha_fwd_splitkv_oneshot_<trait, ck_tile::gfx115_t>(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_create_kargs_and_grids<k_>(a);
    if(getenv("SPLITKV_DEBUG"))
        fprintf(stderr,
                "[splitkv] grid=(%u,%u,%u) block=(%u,%u,%u) splits=%d\n",
                grids.x, grids.y, grids.z, k_::BlockSize().x, k_::BlockSize().y, k_::BlockSize().z,
                int(kargs.num_splits));
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    auto r = ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu, ck_tile::gfx115_t>(k_{}, grids, blocks, 0, kargs));
    ck_tile::ignore = r;
}

template<>
void fmha_fwd_splitkv_combine_oneshot_<combine_trait, ck_tile::gfx115_t>(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{
    using k_ = fmha_combine_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_combine_create_kargs_and_grids<k_>(a);
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    auto r = ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu, ck_tile::gfx115_t>(k_{}, grids, blocks, 0, kargs));
    ck_tile::ignore = r;
}

#endif // !defined(__HIP_DEVICE_COMPILE__) || (defined(__gfx115__))
