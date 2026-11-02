//! XRT (AMD XDNA) backend implementation.
//!
//! Uses `libxrt_coreutil.so.2` for device, buffer, kernel, and runtime management.

use std::ffi::CString;
use std::os::raw::c_int;
use std::sync::Arc;

use libloading::{Library, Symbol};

use crate::backend::{
    BufferHandle, DeviceHandle, KernelHandle, NpuDevice, NpuError, NpuKernel, NpuTensor, RunHandle,
    SyncDirection, Uuid, XclbinHandle,
};

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

// Safety: Library is an OS-level handle; symbols borrow it for 'static.
unsafe impl Send for LibXrt {}
unsafe impl Sync for LibXrt {}

impl LibXrt {
    const SO_PATH: &'static str = "/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2";

    fn load() -> Result<Self, NpuError> {
        let lib = unsafe { Library::new(Self::SO_PATH) }
            .map_err(|e| NpuError::LibraryLoad(format!("dlopen coreutil: {e}")))?;
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

    fn is_loadable() -> bool {
        match unsafe { Library::new(Self::SO_PATH) } {
            Ok(lib) => {
                // Leak so we don't close during probe.
                let _ = Box::leak(Box::new(lib));
                true
            }
            Err(_) => false,
        }
    }
}

static XRT: std::sync::LazyLock<Result<LibXrt, NpuError>> = std::sync::LazyLock::new(LibXrt::load);

fn xrt() -> Result<&'static LibXrt, NpuError> {
    match &*XRT {
        Ok(lib) => Ok(lib),
        Err(e) => Err(e.clone()),
    }
}

// ---------------------------------------------------------------------------
// Backend trait impl: XrtDevice implements NpuDevice
// ---------------------------------------------------------------------------

pub struct XrtDevice {
    handle: DeviceHandle,
}

impl XrtDevice {
    fn open_raw(index: u32) -> Result<Self, NpuError> {
        let xrt = xrt()?;
        let handle = unsafe { (xrt.device_open)(index) };
        if handle.0.is_null() {
            return Err(NpuError::DeviceUnavailable);
        }
        Ok(Self { handle })
    }

    fn close_raw(&mut self) -> Result<(), NpuError> {
        let xrt = xrt()?;
        let rc = unsafe { (xrt.device_close)(self.handle) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }
        self.handle = DeviceHandle(std::ptr::null_mut());
        Ok(())
    }

    fn load_xclbin_raw(&self, path: &str) -> Result<Uuid, NpuError> {
        let xrt = xrt()?;
        let cpath = CString::new(path).map_err(|e| NpuError::NpuCall(e.to_string()))?;
        let rc = unsafe { (xrt.device_load_xclbin_file)(self.handle, cpath.as_ptr()) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }
        let mut uuid = Uuid([0u8; 16]);
        let rc = unsafe { (xrt.device_get_xclbin_uuid)(self.handle, &mut uuid) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }
        Ok(uuid)
    }

    fn xclbin_uuid_raw(&self) -> Result<Uuid, NpuError> {
        let xrt = xrt()?;
        let mut uuid = Uuid([0u8; 16]);
        let rc = unsafe { (xrt.device_get_xclbin_uuid)(self.handle, &mut uuid) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }
        Ok(uuid)
    }

    fn alloc_bo_raw(&self, size: u64) -> Result<BufferHandle, NpuError> {
        let xrt = xrt()?;
        let bo = unsafe { (xrt.bo_alloc)(self.handle, size, 0, 0) };
        if bo.0.is_null() {
            return Err(NpuError::NpuCall("xrtBOAlloc returned null".to_string()));
        }
        Ok(bo)
    }

    fn sync_bo_raw(&self, bo: &BufferHandle, dir: SyncDirection) -> Result<(), NpuError> {
        let xrt = xrt()?;
        let rc = unsafe { (xrt.bo_sync)(*bo, dir as c_int, 0, 0) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }
        Ok(())
    }

    fn open_kernel_raw(&self, uuid: &Uuid, name: &str) -> Result<(KernelHandle, usize), NpuError> {
        let xrt = xrt()?;
        let cname = CString::new(name).map_err(|e| NpuError::NpuCall(e.to_string()))?;
        let handle = unsafe { (xrt.pl_kernel_open)(self.handle, uuid, cname.as_ptr()) };
        if handle.0.is_null() {
            return Err(NpuError::NpuCall(
                "xrtPLKernelOpen returned null".to_string(),
            ));
        }
        // Probe arg count
        let mut num_args = 0;
        while unsafe { (xrt.kernel_arg_group_id)(handle, num_args as c_int) } >= 0 {
            num_args += 1;
        }
        Ok((handle, num_args))
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
            let _ = self.close_raw();
        }
    }
}

// ---------------------------------------------------------------------------
// Backend: XrtBackend provides the concrete NpuDevice impl
// ---------------------------------------------------------------------------

/// Backend type alias for XRT.
pub type XrtBackend = XrtDevice;

impl NpuDevice for XrtDevice {
    fn open(index: u32) -> Result<Arc<Self>, NpuError> {
        Self::open_raw(index).map(Arc::new)
    }

    fn close(&mut self) -> Result<(), NpuError> {
        Self::close_raw(self)
    }

    fn load_xclbin(&self, path: &str) -> Result<Uuid, NpuError> {
        Self::load_xclbin_raw(self, path)
    }

    fn xclbin_uuid(&self) -> Result<Uuid, NpuError> {
        Self::xclbin_uuid_raw(self)
    }

    fn alloc_bo(&self, size: u64) -> Result<Arc<NpuTensor>, NpuError> {
        let bo = self.alloc_bo_raw(size)?;
        Ok(Arc::new(NpuTensor {
            bo,
            size,
            device: Arc::new(self.clone()),
        }))
    }

    fn sync_bo(&self, tensor: &NpuTensor, dir: SyncDirection) -> Result<(), NpuError> {
        Self::sync_bo_raw(self, &tensor.bo, dir)
    }

    fn open_kernel(&self, uuid: &Uuid, name: &str) -> Result<Arc<NpuKernel>, NpuError> {
        let (handle, num_args) = self.open_kernel_raw(uuid, name)?;
        Ok(Arc::new(NpuKernel { handle, num_args }))
    }

    fn run_kernel(&self, kernel: &NpuKernel, buffers: &[&NpuTensor]) -> Result<(), NpuError> {
        let xrt = xrt()?;
        let c_ptrs: Vec<*const std::os::raw::c_char> = buffers
            .iter()
            .map(|b| {
                let addr = unsafe { (xrt.bo_address)(b.bo) };
                addr as *const std::os::raw::c_char
            })
            .collect();

        let run = unsafe { (xrt.run_open)(kernel.handle, c_ptrs.as_ptr()) };
        if run.0.is_null() {
            return Err(NpuError::NpuCall("xrtRunOpen returned null".to_string()));
        }

        // Start and wait for completion
        let rc = unsafe { (xrt.run_start)(run) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }
        let rc = unsafe { (xrt.run_wait)(run) };
        if rc != 0 {
            return Err(NpuError::NpuError(rc));
        }

        // Clean up
        let _ = unsafe { (xrt.run_close)(run) };
        Ok(())
    }
}

impl XrtBackend {
    /// Check if the XRT library is loadable (probe for availability).
    pub fn is_available() -> bool {
        LibXrt::is_loadable()
    }

    /// Open a device by index (returns the trait object for runtime dispatch).
    pub fn open(index: u32) -> Result<Arc<dyn NpuDevice>, NpuError> {
         XrtDevice::open(index).map(|d| d as Arc<dyn NpuDevice>)
    }
}
