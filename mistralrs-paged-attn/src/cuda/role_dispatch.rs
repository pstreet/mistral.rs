//! Dual-only role dispatch for the dense paged-attention entries.
//!
//! Each `*_auto` wrapper routes by the deciding tensor's storage:
//! `Storage::Cuda` (NVIDIA role) goes to the nvcc companion twin,
//! anything else keeps the hip-role path (which bails loudly on
//! non-Hip storage as today). Single-vendor builds call the plain
//! entries directly and never see this module.

use std::collections::HashMap;

use candle_core::{DType, Result, Storage, Tensor};

use super::backend::{
    copy_blocks as hip_copy_blocks, gather_kv_cache as hip_gather_kv_cache,
    paged_attention as hip_paged_attention, reshape_and_cache as hip_reshape_and_cache,
    reshape_and_cache_q4 as hip_reshape_and_cache_q4,
    reshape_and_cache_q8 as hip_reshape_and_cache_q8,
};
use super::cuda_role::{
    copy_blocks as cuda_copy_blocks, gather_kv_cache as cuda_gather_kv_cache,
    paged_attention as cuda_paged_attention, reshape_and_cache as cuda_reshape_and_cache,
    reshape_and_cache_q4 as cuda_reshape_and_cache_q4,
    reshape_and_cache_q8 as cuda_reshape_and_cache_q8,
};
use crate::BlockQuantKind;

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
