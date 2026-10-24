//! Role-agnostic FA3 planning types shared by the hip-side backend
//! (`backend/fa3.rs`, solo builds) and the NVIDIA-role twin
//! (`cuda_role/fa3.rs`, dual builds). Pure data: no FFI, no storage.

use candle_core::Tensor;

pub const USE_FA3_FP8_PAGED: bool = cfg!(has_fa3_fp8_paged);
pub const FA3_DECODE_MAX_QUERY_LEN: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa3DecodeSchedule {
    pub batch_size: usize,
    pub query_len: usize,
    pub total_q: usize,
    pub causal: bool,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub max_seqlen_k: usize,
    pub num_splits: usize,
    pub num_sm: usize,
    pub device_id: usize,
}

#[derive(Clone, Copy)]
pub struct Fa3DecodeMetadata<'a> {
    pub paged_kv_indptr: &'a Tensor,
    pub paged_kv_indices: &'a Tensor,
    pub paged_kv_last_page_len: &'a Tensor,
    pub page_table: &'a Tensor,
    pub seqused_k: &'a Tensor,
    pub cu_seqlens_q: &'a Tensor,
    pub scheduler_metadata: &'a Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa3PagedMetadataLayout {
    pub source_rows_per_sequence: usize,
    pub source_row_offset: usize,
}

impl Fa3PagedMetadataLayout {
    pub const fn per_sequence() -> Self {
        Self {
            source_rows_per_sequence: 1,
            source_row_offset: 0,
        }
    }

    pub const fn decode(query_len: usize) -> Self {
        Self {
            source_rows_per_sequence: query_len,
            source_row_offset: query_len.saturating_sub(1),
        }
    }

    #[cfg(any(test, has_fa3_fp8_paged))]
    pub(crate) fn source_rows(self, batch_size: usize) -> Option<usize> {
        batch_size.checked_mul(self.source_rows_per_sequence)
    }

    #[cfg(any(test, has_fa3_fp8_paged))]
    pub(crate) fn valid(self) -> bool {
        self.source_rows_per_sequence > 0
            && self.source_rows_per_sequence <= FA3_DECODE_MAX_QUERY_LEN
            && self.source_row_offset < self.source_rows_per_sequence
    }
}

#[derive(Clone, Copy)]
pub struct Fa3DecodeParams<'a> {
    pub query: &'a Tensor,
    pub quantized_query: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub page_table: &'a Tensor,
    pub seqused_k: &'a Tensor,
    pub cu_seqlens_q: &'a Tensor,
    pub scheduler_metadata: &'a Tensor,
    pub output_accum: &'a Tensor,
    pub lse_accum: &'a Tensor,
    pub output_lse: &'a Tensor,
    pub q_descale: &'a Tensor,
    pub k_descale: &'a Tensor,
    pub v_descale: &'a Tensor,
    pub schedule: Fa3DecodeSchedule,
    pub softmax_scale: f32,
}
