pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

#[cfg(not(feature = "rocm"))]
pub use backend::{
    concat_and_cache_mla, context_attention_fwd_mla, fa3_fp8_decode, fa3_prepare_decode_metadata,
    fa3_prepare_paged_metadata, flash_attn_sinks, flash_attn_sinks_varlen, flashinfer_decode,
    flashinfer_mla_decode, gather_kv_cache_flashinfer, gather_mla_cache, is_flashinfer_cache,
    kv_scale_update, reshape_and_cache_flashinfer, Fa3DecodeMetadata, Fa3DecodeParams,
    Fa3DecodeSchedule, Fa3PagedMetadataLayout, FlashInferDecodeScratch, FA3_DECODE_MAX_QUERY_LEN,
    USE_FA3_FP8_PAGED,
};
pub use backend::{copy_blocks, gather_kv_cache, paged_attention, reshape_and_cache, swap_blocks};
