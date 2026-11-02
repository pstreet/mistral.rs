//! HRX (host runtime) backend — stub implementation.
//!
//! HRX is the open-source host runtime for AMD XDNA NPUs.
//! It provides a lower-level API than XRT and is part of the
//! MLIR-AIE ecosystem.
//!
//! This module is a skeleton ready for implementation when
//! the HRX runtime library is available.

use std::sync::Arc;

use crate::backend::{NpuDevice, NpuError, NpuKernel, NpuTensor, SyncDirection, Uuid};

// ---------------------------------------------------------------------------
// HRX backend stub
// ---------------------------------------------------------------------------

/// HRX backend. Currently a stub — implements `NpuDevice` with
//! TODO: implement actual HRX FFI bindings when libhrx.so is available.

pub struct HrxBackend;

impl HrxBackend {
    /// Check if the HRX library is loadable.
    pub fn is_available() -> bool {
        // TODO: probe for libhrx.so
        false
    }

    /// Open a device by index.
    pub fn open(index: u32) -> Result<Arc<dyn NpuDevice>, NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }
}

// Stub impl so HrxBackend compiles as NpuDevice
// Real implementation will go here when HRX runtime is available.
impl NpuDevice for HrxBackend {
    fn open(_index: u32) -> Result<Arc<Self>, NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn close(&mut self) -> Result<(), NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn load_xclbin(&self, _path: &str) -> Result<Uuid, NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn xclbin_uuid(&self) -> Result<Uuid, NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn alloc_bo(&self, _size: u64) -> Result<Arc<NpuTensor>, NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn sync_bo(&self, _tensor: &NpuTensor, _dir: SyncDirection) -> Result<(), NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn open_kernel(&self, _uuid: &Uuid, _name: &str) -> Result<Arc<NpuKernel>, NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }

    fn run_kernel(
        &self,
        _kernel: &NpuKernel,
        _buffers: &[&NpuTensor],
    ) -> Result<(), NpuError> {
        Err(NpuError::LibraryLoad(
            "HRX backend not implemented yet".to_string(),
        ))
    }
}
