//! NPU backend — vendor-neutral trait + XRT (AMD XDNA) implementation.
//!
//! Design: a vendor-neutral `NpuDevice` trait abstracts the runtime,
//! and concrete implementations (XRT, HRX) are compiled in as separate
//! modules. At build time, the `npu` feature selects which backends
//! are available; at runtime, the first available backend is chosen.
//!
//! ## Backend implementations
//!
//! | Backend | Crate module | Runtime library | Description |
//! |---------|-------------|-----------------|-------------|
//! | XRT | `xrt` | `libxrt_coreutil.so.2` | AMD XDNA 2 via XRT (production) |
//! | HRX | `hrx` | `libhrx.so` | AMD HRX host runtime (open-source, under development) |

use std::sync::Arc;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Vendor-neutral device trait
// ---------------------------------------------------------------------------

/// UUID of a loaded XCLBIN (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uuid(pub [u8; 16]);

/// Direction for buffer synchronization.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    ToDevice = 0,
    FromDevice = 1,
}

/// Vendor-neutral NPU device interface.
///
/// Each backend (XRT, HRX, …) implements this trait. The trait methods
/// cover device lifecycle, XCLBIN loading, buffer management, and kernel
/// execution. Candle-core's `NpuStorage` holds an `Arc<dyn NpuDevice>` to
/// dispatch operations through the trait object.
pub trait NpuDevice: Send + Sync {
    /// Open a device by index.
    fn open(index: u32) -> Result<Arc<Self>, NpuError>
    where
        Self: Sized;

    /// Close (drop releases the underlying handle).
    fn close(&mut self) -> Result<(), NpuError>;

    /// Load an XCLBIN from a file path. Returns the device UUID.
    fn load_xclbin(&self, path: &str) -> Result<Uuid, NpuError>;

    /// Get UUID of the currently loaded XCLBIN.
    fn xclbin_uuid(&self) -> Result<Uuid, NpuError>;

    /// Allocate a buffer object of the given size.
    fn alloc_bo(&self, size: u64) -> Result<Arc<NpuTensor>, NpuError>;

    /// Synchronize a buffer (host ↔ device).
    fn sync_bo(&self, tensor: &NpuTensor, dir: SyncDirection) -> Result<(), NpuError>;

    /// Create a kernel handle from a loaded XCLBIN.
    fn open_kernel(&self, uuid: &Uuid, name: &str) -> Result<Arc<NpuKernel>, NpuError>;

    /// Run a kernel with the given buffer arguments.
    fn run_kernel(&self, kernel: &NpuKernel, buffers: &[&NpuTensor]) -> Result<(), NpuError>;
}

/// A compiled NPU kernel.
#[derive(Clone)]
pub struct NpuKernel {
    pub handle: KernelHandle,
    pub num_args: usize,
}

/// A buffer on the NPU.
#[derive(Clone)]
pub struct NpuTensor {
    pub bo: BufferHandle,
    pub size: u64,
    pub device: Arc<dyn NpuDevice>,
}

// ---------------------------------------------------------------------------
// Opaque handle types (borrowed from XRT API, reused by all backends)
// ---------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DeviceHandle(*mut std::ffi::c_void);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BufferHandle(*mut std::ffi::c_void);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct KernelHandle(*mut std::ffi::c_void);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct RunHandle(*mut std::ffi::c_void);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct XclbinHandle(*mut std::ffi::c_void);

// Safety: handles are opaque pointers managed by the runtime.
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}
unsafe impl Send for BufferHandle {}
unsafe impl Sync for BufferHandle {}
unsafe impl Send for KernelHandle {}
unsafe impl Sync for KernelHandle {}
unsafe impl Send for RunHandle {}
unsafe impl Sync for RunHandle {}
unsafe impl Send for XclbinHandle {}
unsafe impl Sync for XclbinHandle {}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Error, Clone, Debug)]
pub enum NpuError {
    #[error("NPU library not found: {0}")]
    LibraryLoad(String),

    #[error("NPU symbol not found: {} ({})", .0, .1)]
    SymbolNotFound(String, String),

    #[error("NPU call failed: {0}")]
    NpuCall(String),

    #[error("NPU returned error code: {0}")]
    NpuError(std::os::raw::c_int),

    #[error("NPU device not available")]
    DeviceUnavailable,
}

// ---------------------------------------------------------------------------
// Re-export concrete backends (compiled in conditionally via Cargo features)
// ---------------------------------------------------------------------------

#[cfg(feature = "npu-xrt")]
pub mod xrt;
#[cfg(feature = "npu-xrt")]
pub use xrt::{XrtBackend, XrtDevice};

#[cfg(feature = "npu-hrx")]
pub mod hrx;
#[cfg(feature = "npu-hrx")]
pub use hrx::HrxBackend;

// ---------------------------------------------------------------------------
// Runtime selection: pick the first available backend
// ---------------------------------------------------------------------------

/// Available NPU backends. The first one that probes successfully is used.
#[derive(Debug, Clone)]
pub enum NpuBackendType {
    #[cfg(feature = "npu-xrt")]
    Xrt,
    #[cfg(feature = "npu-hrx")]
    Hrx,
}

impl NpuBackendType {
    /// Probe for the first available backend.
    pub fn probe() -> Option<Self> {
        #[cfg(feature = "npu-xrt")]
        if XrtBackend::is_available() {
            return Some(NpuBackendType::Xrt);
        }
        #[cfg(feature = "npu-hrx")]
        if HrxBackend::is_available() {
            return Some(NpuBackendType::Hrx);
        }
        None
    }

    /// Open a device with the first available backend.
    pub fn open_device(&self, index: u32) -> Result<Arc<dyn NpuDevice>, NpuError> {
        match self {
            #[cfg(feature = "npu-xrt")]
            NpuBackendType::Xrt => XrtBackend::open(index),
            #[cfg(feature = "npu-hrx")]
            NpuBackendType::Hrx => HrxBackend::open(index),
        }
    }

    /// Check if this backend type is available.
    pub fn is_available(&self) -> bool {
        match self {
            #[cfg(feature = "npu-xrt")]
            NpuBackendType::Xrt => XrtBackend::is_available(),
            #[cfg(feature = "npu-hrx")]
            NpuBackendType::Hrx => HrxBackend::is_available(),
        }
    }
}

/// Get the globally detected backend (lazy-probed once).
pub fn backend_type() -> Option<NpuBackendType> {
    static BACKEND: std::sync::OnceLock<Option<NpuBackendType>> = std::sync::OnceLock::new();
    BACKEND.get_or_init(NpuBackendType::probe).clone()
}

/// Check if any NPU backend is available.
pub fn is_npu_available() -> bool {
    backend_type().is_some()
}

/// Open an NPU device with the first available backend.
pub fn open_device(index: u32) -> Result<Arc<dyn NpuDevice>, NpuError> {
    backend_type()
        .map(|bt| bt.open_device(index))
        .unwrap_or(Err(NpuError::DeviceUnavailable))
        .map_err(|_| NpuError::DeviceUnavailable)
}
