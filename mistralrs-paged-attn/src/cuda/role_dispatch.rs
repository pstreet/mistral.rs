//! Dual-only role dispatch for the dense paged-attention entries.
//!
//! Each `*_auto` wrapper routes by the deciding tensor's storage:
//! `Storage::Cuda` (NVIDIA role) goes to the nvcc companion twin,
//! anything else keeps the hip-role path (which bails loudly on
//! non-Hip storage as today). Single-vendor builds call the plain
//! entries directly and never see this module. The FlashInfer
//! entries have no hip twin (RDNA lacks the kernels), so non-cuda
//! storage bails here instead.

use std::collections::HashMap;

use candle_core::{DType, Result, Storage, Tensor};

use super::backend::{
    copy_blocks as hip_copy_blocks, gather_kv_cache as hip_gather_kv_cache,
    paged_attention as hip_paged_attention, reshape_and_cache as hip_reshape_and_cache,
    reshape_and_cache_q4 as hip_reshape_and_cache_q4,
    reshape_and_cache_q8 as hip_reshape_and_cache_q8,
};
use super::cuda_role::{
    concat_and_cache_mla as cuda_concat_and_cache_mla, context_attention_fwd_mla as cuda_context_attention_fwd_mla,
    copy_blocks as cuda_copy_blocks, fa3_fp8_decode as cuda_fa3_fp8_decode,
    fa3_prepare_decode_metadata as cuda_fa3_prepare_decode_metadata,
    fa3_prepare_paged_metadata as cuda_fa3_prepare_paged_metadata,
    flash_attn_sinks as cuda_flash_attn_sinks, flash_attn_sinks_varlen as cuda_flash_attn_sinks_varlen,
    flashinfer_decode as cuda_flashinfer_decode, flashinfer_mla_decode as cuda_flashinfer_mla_decode,
    gather_kv_cache as cuda_gather_kv_cache, gather_kv_cache_flashinfer as cuda_gather_kv_cache_flashinfer,
    gather_mla_cache as cuda_gather_mla_cache, paged_attention as cuda_paged_attention,
    reshape_and_cache as cuda_reshape_and_cache, reshape_and_cache_flashinfer as cuda_reshape_and_cache_flashinfer,
    reshape_and_cache_q4 as cuda_reshape_and_cache_q4, reshape_and_cache_q8 as cuda_reshape_and_cache_q8,
    FlashInferDecodeScratch,
};
use crate::cuda::fa3_types::{Fa3DecodeMetadata, Fa3DecodeParams, Fa3DecodeSchedule};
use crate::{BlockQuantKind, KvCacheScales};

fn role_is_cuda(t: &Tensor) -> bool {
    matches!(&*t.storage_and_layout().0, Storage::Cuda(_))
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_auto(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    k_res: Option<&Tensor>,
    v_res: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
    k_quant: Option<BlockQuantKind>,
    v_quant: Option<BlockQuantKind>,
) -> Result<Tensor> {
    if role_is_cuda(q) {
        cuda_paged_attention(
            q,
            k_scale,
            v_scale,
            k_res,
            v_res,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            alibi_slopes,
            max_context_len,
            softmax_scale,
            softcapping,
            sinks,
            k_quant,
            v_quant,
        )
    } else {
        hip_paged_attention(
            q,
            k_scale,
            v_scale,
            k_res,
            v_res,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            alibi_slopes,
            max_context_len,
            softmax_scale,
            softcapping,
            sinks,
            k_quant,
            v_quant,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_auto(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    if role_is_cuda(key) {
        cuda_reshape_and_cache(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
            write_k,
            write_v,
        )
    } else {
        hip_reshape_and_cache(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
            write_k,
            write_v,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_q8_auto(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scales: Option<&Tensor>,
    v_scales: Option<&Tensor>,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    if role_is_cuda(key) {
        cuda_reshape_and_cache_q8(
            key,
            value,
            key_cache,
            value_cache,
            k_scales,
            v_scales,
            slot_mapping,
            write_k,
            write_v,
        )
    } else {
        hip_reshape_and_cache_q8(
            key,
            value,
            key_cache,
            value_cache,
            k_scales,
            v_scales,
            slot_mapping,
            write_k,
            write_v,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_q4_auto(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scales: Option<&Tensor>,
    v_scales: Option<&Tensor>,
    k_res: Option<&Tensor>,
    v_res: Option<&Tensor>,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    if role_is_cuda(key) {
        cuda_reshape_and_cache_q4(
            key,
            value,
            key_cache,
            value_cache,
            k_scales,
            v_scales,
            k_res,
            v_res,
            slot_mapping,
            write_k,
            write_v,
        )
    } else {
        hip_reshape_and_cache_q4(
            key,
            value,
            key_cache,
            value_cache,
            k_scales,
            v_scales,
            k_res,
            v_res,
            slot_mapping,
            write_k,
            write_v,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache_auto(
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    k_res: Option<&Tensor>,
    v_res: Option<&Tensor>,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize,
    out_dtype: DType,
    k_quant: Option<BlockQuantKind>,
    v_quant: Option<BlockQuantKind>,
) -> Result<(Tensor, Tensor)> {
    if role_is_cuda(key_cache) {
        cuda_gather_kv_cache(
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            k_res,
            v_res,
            block_table,
            cu_seq_lens,
            num_tokens,
            out_dtype,
            k_quant,
            v_quant,
        )
    } else {
        hip_gather_kv_cache(
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            k_res,
            v_res,
            block_table,
            cu_seq_lens,
            num_tokens,
            out_dtype,
            k_quant,
            v_quant,
        )
    }
}

pub fn copy_blocks_auto(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    // Both lists are empty together on the early-Ok path; probe the key
    // side (the fn itself rejects a key/value count mismatch).
    if key_caches.first().is_some_and(|t| role_is_cuda(t)) {
        cuda_copy_blocks(key_caches, value_caches, block_mapping)
    } else {
        hip_copy_blocks(key_caches, value_caches, block_mapping)
    }
}

pub fn reshape_and_cache_flashinfer_auto(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    scales: KvCacheScales,
) -> Result<()> {
    if role_is_cuda(key) {
        cuda_reshape_and_cache_flashinfer(key, value, key_cache, value_cache, slot_mapping, scales)
    } else {
        candle_core::bail!("FlashInfer paged attention requires the NVIDIA role")
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flashinfer_decode_auto(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    scales: KvCacheScales,
    paged_kv_indptr: &Tensor,
    paged_kv_indices: &Tensor,
    paged_kv_last_page_len: &Tensor,
    request_indices: &Tensor,
    kv_tile_indices: &Tensor,
    o_indptr: &Tensor,
    kv_chunk_size: &Tensor,
    block_valid_mask: &Tensor,
    sm_scale: f32,
    window_left: Option<usize>,
    logits_soft_cap: Option<f32>,
    scratch: Option<FlashInferDecodeScratch<'_>>,
) -> Result<Tensor> {
    if role_is_cuda(query) {
        cuda_flashinfer_decode(
            query,
            key_cache,
            value_cache,
            scales,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            request_indices,
            kv_tile_indices,
            o_indptr,
            kv_chunk_size,
            block_valid_mask,
            sm_scale,
            window_left,
            logits_soft_cap,
            scratch,
        )
    } else {
        candle_core::bail!("FlashInfer paged attention requires the NVIDIA role")
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache_flashinfer_auto(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize,
    out_dtype: DType,
    scales: KvCacheScales,
) -> Result<(Tensor, Tensor)> {
    if role_is_cuda(key_cache) {
        cuda_gather_kv_cache_flashinfer(
            key_cache,
            value_cache,
            block_table,
            cu_seq_lens,
            num_tokens,
            out_dtype,
            scales,
        )
    } else {
        candle_core::bail!("FlashInfer paged attention requires the NVIDIA role")
    }
}

pub fn fa3_prepare_paged_metadata_auto(
    metadata: Fa3DecodeMetadata<'_>,
    schedule: Fa3DecodeSchedule,
    metadata_layout: crate::cuda::fa3_types::Fa3PagedMetadataLayout,
) -> Result<()> {
    if role_is_cuda(metadata.paged_kv_indptr) {
        cuda_fa3_prepare_paged_metadata(metadata, schedule, metadata_layout)
    } else {
        candle_core::bail!("FA3 paged attention requires the NVIDIA role")
    }
}

pub fn fa3_prepare_decode_metadata_auto(
    metadata: Fa3DecodeMetadata<'_>,
    schedule: Fa3DecodeSchedule,
) -> Result<()> {
    if role_is_cuda(metadata.paged_kv_indptr) {
        cuda_fa3_prepare_decode_metadata(metadata, schedule)
    } else {
        candle_core::bail!("FA3 paged attention requires the NVIDIA role")
    }
}

pub fn fa3_fp8_decode_auto(params: Fa3DecodeParams<'_>) -> Result<Tensor> {
    if role_is_cuda(params.query) {
        cuda_fa3_fp8_decode(params)
    } else {
        candle_core::bail!("FA3 paged attention requires the NVIDIA role")
    }
}

// ---- MLA + sinks (cuda-only; bail on non-cuda role) ----

pub fn concat_and_cache_mla_auto(
    ckv: &Tensor,
    k_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if role_is_cuda(ckv) {
        cuda_concat_and_cache_mla(ckv, k_pe, ckv_cache, kpe_cache, slot_mapping)
    } else {
        candle_core::bail!("MLA paged attention requires the NVIDIA role")
    }
}

pub fn flashinfer_mla_decode_auto(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    paged_kv_indptr: &Tensor,
    paged_kv_indices: &Tensor,
    paged_kv_last_page_len: &Tensor,
    request_indices: &Tensor,
    kv_tile_indices: &Tensor,
    o_indptr: &Tensor,
    kv_chunk_size: &Tensor,
    sm_scale: f32,
) -> Result<Tensor> {
    if role_is_cuda(q_nope) {
        cuda_flashinfer_mla_decode(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            request_indices,
            kv_tile_indices,
            o_indptr,
            kv_chunk_size,
            sm_scale,
        )
    } else {
        candle_core::bail!("MLA paged attention requires the NVIDIA role")
    }
}

pub fn gather_mla_cache_auto(
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    token_to_seq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if role_is_cuda(ckv_cache) {
        cuda_gather_mla_cache(ckv_cache, kpe_cache, block_table, cu_seq_lens, token_to_seq)
    } else {
        candle_core::bail!("MLA paged attention requires the NVIDIA role")
    }
}

pub fn context_attention_fwd_mla_auto(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_new: &Tensor,
    kpe_new: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    query_lens: &Tensor,
    w_uk: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    if role_is_cuda(q_nope) {
        cuda_context_attention_fwd_mla(
            q_nope,
            q_pe,
            ckv_new,
            kpe_new,
            ckv_cache,
            kpe_cache,
            block_tables,
            context_lens,
            query_lens,
            w_uk,
            softmax_scale,
        )
    } else {
        candle_core::bail!("MLA paged attention requires the NVIDIA role")
    }
}

pub fn flash_attn_sinks_auto(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    if role_is_cuda(q) {
        cuda_flash_attn_sinks(q, k, v, sinks, softmax_scale, window_size)
    } else {
        candle_core::bail!("flash_attn_sinks requires the NVIDIA role")
    }
}

pub fn flash_attn_sinks_varlen_auto(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    if role_is_cuda(q) {
        cuda_flash_attn_sinks_varlen(q, k, v, sinks, cu_seqlens_q, cu_seqlens_k, softmax_scale, window_size)
    } else {
        candle_core::bail!("flash_attn_sinks requires the NVIDIA role")
    }
}
