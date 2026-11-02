//! mistralrs-npu: vendor-neutral NPU backend for AMD Ryzen AI (Strix Halo).
//!
//! Provides:
//! - `NpuDevice` trait — vendor-neutral device interface
//! - `XrtBackend` — XRT (AMD XDNA) implementation
//! - `HrxBackend` — HRX host runtime (stub, under development)
//! - Runtime backend selection — probes for available backends

#[cfg(feature = "npu")]
pub mod backend;

#[cfg(not(feature = "npu"))]
pub mod stub {
    use thiserror::Error;

    #[derive(Error, Debug)]
    #[error("NPU support disabled: compile with --features npu")]
    pub struct NpuNotCompiled;
}

// Re-export key types (only when npu feature is enabled)
#[cfg(feature = "npu")]
pub use backend::{
    NpuDevice, NpuError, NpuKernel, NpuTensor, NpuBackendType,
    backend_type, is_npu_available, open_device,
    SyncDirection, Uuid,
};
#[cfg(feature = "npu")]
pub use backend::xrt::{XrtBackend, XrtDevice};
