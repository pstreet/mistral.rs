pub const USE_FP8: bool = cfg!(has_fp8);

/// Role backend: cuda-only builds use `cuda_backend` (bound to the
/// compiled vendor); every rocm build uses `hip_backend` (all C kernels
/// here are hipcc-built, so this crate serves the AMD role).
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
pub use candle_core::cuda_backend as role_backend;
#[cfg(feature = "rocm")]
pub use candle_core::hip_backend as role_backend;

mod backend;
mod fa3_types;
mod ffi;
#[cfg(feature = "cuda")]
pub(crate) mod ffi_cuda;

#[cfg(not(feature = "rocm"))]
pub use backend::{
    concat_and_cache_mla, context_attention_fwd_mla, fa3_fp8_decode, fa3_prepare_decode_metadata,
    fa3_prepare_paged_metadata, flash_attn_sinks, flash_attn_sinks_varlen, flashinfer_decode,
    flashinfer_mla_decode, gather_kv_cache_flashinfer, gather_mla_cache, is_flashinfer_cache,
    kv_scale_update, reshape_and_cache_flashinfer, Fa3DecodeMetadata, Fa3DecodeParams,
    Fa3DecodeSchedule, Fa3PagedMetadataLayout, FlashInferDecodeScratch, FA3_DECODE_MAX_QUERY_LEN,
    USE_FA3_FP8_PAGED,
};
pub use backend::{
    copy_blocks, gather_kv_cache, paged_attention, reshape_and_cache, reshape_and_cache_q4,
    reshape_and_cache_q8, swap_blocks, PAGED_ATTENTION_V2_PARTITION_SIZE,
};

/// Dual-only NVIDIA-role twins (nvcc companion). Same entry shapes as
/// above, bound to `cuda_backend` types. Core dispatches by role storage.
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub mod cuda_role;
#[cfg(all(feature = "cuda", target_family = "unix"))]
mod role_dispatch;
/// Dual-build FlashInfer surface: the kernels are NVIDIA-role only, so the
/// crate root re-exports the pure-Rust cache/scratch types from the twin.
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub use cuda_role::{is_flashinfer_cache, FlashInferDecodeScratch};
/// Dual-build FA3 planning surface: shared types live in `fa3_types` (the
/// solo arm re-exports them through `backend`).
#[cfg(all(feature = "cuda", feature = "rocm", target_family = "unix"))]
pub use fa3_types::{
    Fa3DecodeMetadata, Fa3DecodeParams, Fa3DecodeSchedule, Fa3PagedMetadataLayout,
    FA3_DECODE_MAX_QUERY_LEN, USE_FA3_FP8_PAGED,
};
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use role_dispatch::{
    copy_blocks_auto, fa3_fp8_decode_auto, fa3_prepare_decode_metadata_auto,
    fa3_prepare_paged_metadata_auto, flashinfer_decode_auto, gather_kv_cache_auto,
    gather_kv_cache_flashinfer_auto, paged_attention_auto, reshape_and_cache_auto,
    reshape_and_cache_flashinfer_auto, reshape_and_cache_q4_auto, reshape_and_cache_q8_auto,
};
