//! NPU backend — vendor-neutral trait + XRT (AMD XDNA) implementation.

use std::ffi::CString;
use std::os::raw::c_int;
use std::sync::Arc;

use libloading::{Library, Symbol};
use once_cell::sync::Lazy;
use thiserror::Error;

// ---------------------------------------------------------------------------
// C API types (opaque pointers from libxrt_coreutil.so)
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

// XRT handles are opaque pointers managed by the XRT runtime.
// The runtime guarantees thread-safety for all handles.
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

#[repr(C)]
pub struct Uuid([u8; 16]);

// xclBOSyncDirection — must match the XRT enum
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    ToDevice = 0,
    FromDevice = 1,
}

// ---------------------------------------------------------------------------
// FFI bridge — dlopen libxrt_coreutil.so.2
// ---------------------------------------------------------------------------

type FxDeviceOpen = unsafe extern "C" fn(u32) -> DeviceHandle;
type FxDeviceClose = unsafe extern "C" fn(DeviceHandle) -> c_int;
type FxDeviceLoadXclbinFile =
    unsafe extern "C" fn(DeviceHandle, *const std::os::raw::c_char) -> c_int;
type FxDeviceGetXclbinUUID = unsafe extern "C" fn(DeviceHandle, *mut Uuid) -> c_int;
type FxDeviceToXclDevice = unsafe extern "C" fn(DeviceHandle) -> *mut std::ffi::c_void;
type FxBoAlloc = unsafe extern "C" fn(DeviceHandle, u64, u64, u64) -> BufferHandle;
type FxBoFree = unsafe extern "C" fn(BufferHandle) -> c_int;
type FxBoMap = unsafe extern "C" fn(BufferHandle) -> *mut std::ffi::c_void;
type FxBoSync = unsafe extern "C" fn(BufferHandle, c_int, u64, u64) -> c_int;
type FxBoSize = unsafe extern "C" fn(BufferHandle) -> u64;
type FxBoAddress = unsafe extern "C" fn(BufferHandle) -> u64;
type FxBoRead = unsafe extern "C" fn(BufferHandle, *mut std::ffi::c_void, u64, u64) -> c_int;
type FxBoWrite = unsafe extern "C" fn(BufferHandle, *const std::ffi::c_void, u64, u64) -> c_int;
type FxPlKernelOpen =
    unsafe extern "C" fn(DeviceHandle, *const Uuid, *const std::os::raw::c_char) -> KernelHandle;
type FxKernelClose = unsafe extern "C" fn(KernelHandle) -> c_int;
type FxKernelArgGroupId = unsafe extern "C" fn(KernelHandle, c_int) -> c_int;
type FxKernelArgOffset = unsafe extern "C" fn(KernelHandle, c_int) -> c_int;
type FxKernelRun = unsafe extern "C" fn(KernelHandle, ...) -> RunHandle;
type FxRunOpen =
    unsafe extern "C" fn(KernelHandle, *const *const std::os::raw::c_char) -> RunHandle;
type FxRunClose = unsafe extern "C" fn(RunHandle) -> c_int;
type FxRunWait = unsafe extern "C" fn(RunHandle) -> c_int;
type FxRunSetArg = unsafe extern "C" fn(RunHandle, c_int, ...) -> c_int;
type FxRunStart = unsafe extern "C" fn(RunHandle) -> c_int;
type FxRunState = unsafe extern "C" fn(RunHandle) -> c_int;

struct LibXrt {
    _lib: &'static Library,
    device_open: Symbol<'static, FxDeviceOpen>,
    device_close: Symbol<'static, FxDeviceClose>,
    device_load_xclbin_file: Symbol<'static, FxDeviceLoadXclbinFile>,
    device_get_xclbin_uuid: Symbol<'static, FxDeviceGetXclbinUUID>,
    device_to_xcl_device: Symbol<'static, FxDeviceToXclDevice>,
    bo_alloc: Symbol<'static, FxBoAlloc>,
    bo_free: Symbol<'static, FxBoFree>,
    bo_map: Symbol<'static, FxBoMap>,
    bo_sync: Symbol<'static, FxBoSync>,
    bo_size: Symbol<'static, FxBoSize>,
    bo_address: Symbol<'static, FxBoAddress>,
    bo_read: Symbol<'static, FxBoRead>,
    bo_write: Symbol<'static, FxBoWrite>,
    pl_kernel_open: Symbol<'static, FxPlKernelOpen>,
    kernel_close: Symbol<'static, FxKernelClose>,
    kernel_arg_group_id: Symbol<'static, FxKernelArgGroupId>,
    kernel_arg_offset: Symbol<'static, FxKernelArgOffset>,
    kernel_run: Symbol<'static, FxKernelRun>,
    run_open: Symbol<'static, FxRunOpen>,
    run_close: Symbol<'static, FxRunClose>,
    run_wait: Symbol<'static, FxRunWait>,
    run_set_arg: Symbol<'static, FxRunSetArg>,
    run_start: Symbol<'static, FxRunStart>,
    run_state: Symbol<'static, FxRunState>,
}

// Safety: Library is safe to share across threads (OS-level handle).
// Symbols borrow it for 'static, so the whole struct is Sync.
unsafe impl Send for LibXrt {}
unsafe impl Sync for LibXrt {}

impl LibXrt {
    const SO_PATH: &'static str = "/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2";

    fn load() -> Result<Self, NpuError> {
        let lib = unsafe { Library::new(Self::SO_PATH) }
            .map_err(|e| NpuError::LibraryLoad(format!("dlopen coreutil: {e}")))?;
        // Leak so symbols can borrow for 'static.
        let lib: &'static Library = Box::leak(Box::new(lib));

        macro_rules! sym {
            ($name:ident) => {{
                let bytes = stringify!($name).as_bytes();
                lib.get(bytes).map_err(|e| {
                    NpuError::SymbolNotFound(stringify!($name).to_string(), format!("{e}"))
                })?
            }};
        }

        Ok(Self {
            _lib: lib,
            device_open: unsafe { sym!(xrtDeviceOpen) },
            device_close: unsafe { sym!(xrtDeviceClose) },
            device_load_xclbin_file: unsafe { sym!(xrtDeviceLoadXclbinFile) },
            device_get_xclbin_uuid: unsafe { sym!(xrtDeviceGetXclbinUUID) },
            device_to_xcl_device: unsafe { sym!(xrtDeviceToXclDevice) },
            bo_alloc: unsafe { sym!(xrtBOAlloc) },
            bo_free: unsafe { sym!(xrtBOFree) },
            bo_map: unsafe { sym!(xrtBOMap) },
            bo_sync: unsafe { sym!(xrtBOSync) },
            bo_size: unsafe { sym!(xrtBOSize) },
            bo_address: unsafe { sym!(xrtBOAddress) },
            bo_read: unsafe { sym!(xrtBORead) },
            bo_write: unsafe { sym!(xrtBOWrite) },
            pl_kernel_open: unsafe { sym!(xrtPLKernelOpen) },
            kernel_close: unsafe { sym!(xrtKernelClose) },
            kernel_arg_group_id: unsafe { sym!(xrtKernelArgGroupId) },
            kernel_arg_offset: unsafe { sym!(xrtKernelArgOffset) },
            kernel_run: unsafe { sym!(xrtKernelRun) },
            run_open: unsafe { sym!(xrtRunOpen) },
            run_close: unsafe { sym!(xrtRunClose) },
            run_wait: unsafe { sym!(xrtRunWait) },
            run_set_arg: unsafe { sym!(xrtRunSetArg) },
            run_start: unsafe { sym!(xrtRunStart) },
            run_state: unsafe { sym!(xrtRunState) },
        })
    }
}

// Global XRT library instance — loaded once on first access.
static XRT: Lazy<Result<LibXrt, NpuError>> = Lazy::new(LibXrt::load);

// Helper to get a reference to the loaded library.
fn xrt() -> Result<&'static LibXrt, NpuError> {
    match &*XRT {
        Ok(lib) => Ok(lib),
        Err(e) => Err(e.clone()),
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Error, Clone, Debug)]
pub enum NpuError {
    #[error("XRT library not found: {0}")]
    LibraryLoad(String),

    #[error("XRT symbol not found: {} ({})", .0, .1)]
    SymbolNotFound(String, String),

    #[error("XRT call failed: {0}")]
    XrtCall(String),

    #[error("XRT returned error code: {0}")]
    XrtError(c_int),

    #[error("NPU device not available")]
    DeviceUnavailable,
}

// ---------------------------------------------------------------------------
// NpuDevice trait (vendor-neutral)
// ---------------------------------------------------------------------------

/// Vendor-neutral NPU device handle.
pub trait NpuDevice: Send + Sync {
    /// Open a device by index.
    fn open(index: u32) -> Result<Self, NpuError>
    where
        Self: Sized;

    /// Close the device.
    fn close(&mut self) -> Result<(), NpuError>;

    /// Load an xclbin from a file path.
    fn load_xclbin(&self, path: &str) -> Result<Uuid, NpuError>;

    /// Get the UUID of the currently loaded xclbin.
    fn xclbin_uuid(&self) -> Result<Uuid, NpuError>;

    /// Allocate a buffer object.
    fn alloc_bo(&self, size: u64) -> Result<Arc<NpuTensor>, NpuError>;

    /// Synchronize a buffer (device ↔ host).
    fn sync_bo(&self, bo: &NpuTensor, dir: SyncDirection) -> Result<(), NpuError>;
}

/// A compiled NPU kernel handle.
#[derive(Clone)]
pub struct NpuKernel {
    pub handle: KernelHandle,
    pub num_args: usize,
}

/// A buffer on the NPU (backed by a BO).
#[derive(Clone)]
pub struct NpuTensor {
    pub bo: BufferHandle,
    pub size: u64,
    pub device: Arc<XrtDevice>,
}

// ---------------------------------------------------------------------------
// XRT concrete implementation
// ---------------------------------------------------------------------------

pub struct XrtDevice {
    handle: DeviceHandle,
}

impl XrtDevice {
    /// Open device by index.
    pub fn open(index: u32) -> Result<Self, NpuError> {
        let xrt = xrt()?;
        let handle = unsafe { (xrt.device_open)(index) };
        if handle.0.is_null() {
            return Err(NpuError::DeviceUnavailable);
        }
        Ok(Self { handle })
    }

    /// Close the device.
    pub fn close(&mut self) -> Result<(), NpuError> {
        let xrt = xrt()?;
        let rc = unsafe { (xrt.device_close)(self.handle) };
        if rc != 0 {
            return Err(NpuError::XrtError(rc));
        }
        self.handle = DeviceHandle(std::ptr::null_mut());
        Ok(())
    }

    /// Load an xclbin file.
    pub fn load_xclbin(&self, path: &str) -> Result<Uuid, NpuError> {
        let xrt = xrt()?;
        let cpath = CString::new(path).map_err(|e| NpuError::XrtCall(e.to_string()))?;
        let rc = unsafe { (xrt.device_load_xclbin_file)(self.handle, cpath.as_ptr()) };
        if rc != 0 {
            return Err(NpuError::XrtError(rc));
        }
        let mut uuid = Uuid([0u8; 16]);
        let rc = unsafe { (xrt.device_get_xclbin_uuid)(self.handle, &mut uuid) };
        if rc != 0 {
            return Err(NpuError::XrtError(rc));
        }
        Ok(uuid)
    }

    /// Get UUID of loaded xclbin.
    pub fn xclbin_uuid(&self) -> Result<Uuid, NpuError> {
        let xrt = xrt()?;
        let mut uuid = Uuid([0u8; 16]);
        let rc = unsafe { (xrt.device_get_xclbin_uuid)(self.handle, &mut uuid) };
        if rc != 0 {
            return Err(NpuError::XrtError(rc));
        }
        Ok(uuid)
    }

    /// Allocate a buffer object.
    pub fn alloc_bo(&self, size: u64) -> Result<Arc<NpuTensor>, NpuError> {
        let xrt = xrt()?;
        // xrtBOAlloc(device, size, offset, flags) — offset=0, flags=0
        let bo = unsafe { (xrt.bo_alloc)(self.handle, size, 0, 0) };
        if bo.0.is_null() {
            return Err(NpuError::XrtCall("xrtBOAlloc returned null".to_string()));
        }
        Ok(Arc::new(NpuTensor {
            bo,
            size,
            device: Arc::new(self.clone()),
        }))
    }

    /// Synchronize a buffer.
    pub fn sync_bo(&self, tensor: &NpuTensor, dir: SyncDirection) -> Result<(), NpuError> {
        let xrt = xrt()?;
        let rc = unsafe { (xrt.bo_sync)(tensor.bo, dir as c_int, tensor.size, 0) };
        if rc != 0 {
            return Err(NpuError::XrtError(rc));
        }
        Ok(())
    }

    /// Open a PL kernel.
    pub fn open_kernel(&self, uuid: &Uuid, name: &str) -> Result<NpuKernel, NpuError> {
        let xrt = xrt()?;
        let cname = CString::new(name).map_err(|e| NpuError::XrtCall(e.to_string()))?;
        let handle = unsafe { (xrt.pl_kernel_open)(self.handle, uuid, cname.as_ptr()) };
        if handle.0.is_null() {
            return Err(NpuError::XrtCall(
                "xrtPLKernelOpen returned null".to_string(),
            ));
        }
        // Probe arg count by iterating group IDs until -1
        let mut num_args = 0;
        while unsafe { (xrt.kernel_arg_group_id)(handle, num_args as c_int) } >= 0 {
            num_args += 1;
        }
        Ok(NpuKernel {
            handle,
            num_args: num_args as usize,
        })
    }
}

impl Clone for XrtDevice {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
        }
    }
}

impl Drop for XrtDevice {
    fn drop(&mut self) {
        if !self.handle.0.is_null() {
            let _ = self.close();
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl NpuDevice for XrtDevice {
    fn open(index: u32) -> Result<Self, NpuError> {
        Self::open(index)
    }

    fn close(&mut self) -> Result<(), NpuError> {
        XrtDevice::close(self)
    }

    fn load_xclbin(&self, path: &str) -> Result<Uuid, NpuError> {
        XrtDevice::load_xclbin(self, path)
    }

    fn xclbin_uuid(&self) -> Result<Uuid, NpuError> {
        XrtDevice::xclbin_uuid(self)
    }

    fn alloc_bo(&self, size: u64) -> Result<Arc<NpuTensor>, NpuError> {
        XrtDevice::alloc_bo(self, size)
    }

    fn sync_bo(&self, tensor: &NpuTensor, dir: SyncDirection) -> Result<(), NpuError> {
        XrtDevice::sync_bo(self, tensor, dir)
    }
}
