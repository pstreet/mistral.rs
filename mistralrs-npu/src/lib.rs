#[cfg(feature = "npu")]
mod backend;

#[cfg(feature = "npu")]
pub use backend::{NpuDevice, NpuError, NpuKernel, NpuTensor};

#[cfg(not(feature = "npu"))]
pub mod stub {
    use thiserror::Error;

    #[derive(Error, Debug)]
    #[error("NPU support disabled: compile with --features npu")]
    pub struct NpuNotCompiled;
}
