#[cfg(any(feature = "cuda", feature = "rocm"))]
mod ffi;
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
mod gptq_cpu;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod gptq_cuda;
#[cfg(feature = "cuda")]
mod marlin_backend;
#[cfg(feature = "cuda")]
mod marlin_ffi;

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub use gptq_cpu::{gptq_linear, GptqLayer};
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use gptq_cuda::{gptq_linear, GptqLayer};
