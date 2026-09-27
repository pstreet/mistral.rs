#[cfg(all(feature = "cuda", not(feature = "rocm")))]
pub(crate) mod ffi;

pub(crate) mod ops;
