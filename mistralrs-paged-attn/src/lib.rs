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

#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "rocm"))]
mod cuda;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "rocm"))]
pub use cuda::*;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
