#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) mod ffi;

// NVIDIA-role twin of `ffi`: nvcc-companion entries for dual builds.
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub(crate) mod ffi_cuda;

pub(crate) mod ops;
