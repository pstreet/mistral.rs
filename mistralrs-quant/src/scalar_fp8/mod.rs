#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) mod ffi;

pub(crate) mod ops;
