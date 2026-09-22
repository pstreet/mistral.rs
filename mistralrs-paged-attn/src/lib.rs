#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheScales {
    pub k: f32,
    pub v: f32,
}

pub const DEFAULT_FP8_KV_CACHE_SCALES: KvCacheScales = KvCacheScales { k: 1.0, v: 1.0 };

/// Block-quantized KV payloads (int8 Q8_0 / nibble Q4_0, both U8 storage
/// with fp32 per-32 scale sidecars). U8 cache tensors are ambiguous on their
/// own, so layers pass the kind explicitly and it selects the runtime
/// cache_dtype code (4 / 5) on CUDA/ROCm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockQuantKind {
    Q8_0,
    Q4_0,
}

impl BlockQuantKind {
    pub fn cache_dtype(self) -> u32 {
        match self {
            Self::Q8_0 => 4,
            Self::Q4_0 => 5,
        }
    }
}

/// Runtime kernel dtype code for one KV side: 0/1/2 native f16/bf16/f32,
/// 3 fp8_e4m3, 4/5 via the block kind. `None` for anything else.
pub fn side_cache_dtype(dtype: candle_core::DType, kind: Option<BlockQuantKind>) -> Option<u32> {
    if let Some(kind) = kind {
        return Some(kind.cache_dtype());
    }
    match dtype {
        candle_core::DType::F16 => Some(0),
        candle_core::DType::BF16 => Some(1),
        candle_core::DType::F32 => Some(2),
        candle_core::DType::F8E4M3 => Some(3),
        _ => None,
    }
}

/// Tokens per vLLM v2 attention partition. The CUDA/ROCm backend re-exports
/// its own value; this fallback keeps CPU-only (and Metal) builds compiling
/// with the same launch-shape math.
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "rocm")))]
pub const PAGED_ATTENTION_V2_PARTITION_SIZE: usize = 512;

#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "rocm"))]
mod cuda;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "rocm"))]
pub use cuda::*;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
