//! Role-aware primitives for the decode-graph stack.
//!
//! Single-vendor builds: the compiled backend's own types ARE the role, so
//! `GraphStream`/`GraphEvent` are plain aliases and every call site behaves
//! exactly as before (inherent methods through `Deref`).
//!
//! Dual builds: the graph machinery serves BOTH roles - the hip role
//! through the hipcc capture/arena API (the historical path, behavior
//! unchanged) and the NVIDIA role through upstream cudarc (mempool async
//! allocs, no arena). The enums mirror the method names the stack already
//! calls, so call sites stay role-blind; driver entry points dispatch on
//! the stream's role and normalize results (both drivers encode SUCCESS
//! as 0).

use std::sync::Arc;

#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub(crate) use candle_core::cuda_backend::cudarc::driver::CudaEvent as GraphEvent;
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub(crate) use candle_core::cuda_backend::cudarc::driver::PinnedHostSlice as GraphPinnedHost;
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub(crate) type GraphStream = Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>;

// The single-vendor alias shape must expose the same helper names the dual
// enum provides; an extension trait carries them since `GraphStream` is an
// `Arc` alias here (no inherent impls possible).
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub(crate) mod graph_driver {
    use super::*;

    pub(crate) struct GraphCheckError(
        pub candle_core::role::backend::cudarc::driver::result::DriverError,
    );

    impl GraphCheckError {
        pub(crate) fn is_invalid_value(&self) -> bool {
            self.0 .0
                == candle_core::role::backend::cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE
        }

        pub(crate) fn into_candle(self) -> candle_core::Error {
            candle_core::Error::Cuda(Box::new(self.0))
        }
    }

    pub(crate) trait GraphStreamExt {
        fn from_graph_device(device: &candle_core::Device) -> candle_core::Result<Self>
        where
            Self: Sized;
        fn bind_context_to_thread(&self) -> candle_core::Result<()>;
        fn cu_device(&self) -> core::ffi::c_int;
        fn has_async_alloc(&self) -> bool;
        fn new_graph_event(&self) -> candle_core::Result<GraphEvent>;
        fn new_blocking_event(&self) -> candle_core::Result<GraphEvent>;
        fn record_timing_event(&self) -> candle_core::Result<GraphEvent>;
        fn is_event_tracking(&self) -> bool;
        unsafe fn disable_event_tracking(&self);
        unsafe fn enable_event_tracking(&self);
        fn graph_check_err(&self) -> std::result::Result<(), GraphCheckError>;
        fn begin_graph_capture(&self, relaxed: bool) -> candle_core::Result<()>;
        fn capture_status_none(&self) -> candle_core::Result<bool>;
        fn same_as(&self, other: &Self) -> bool;
        fn wait_on_event(&self, event: &GraphEvent) -> candle_core::Result<()>;
        #[cfg(feature = "rocm")]
        fn hip(&self) -> &Self;
    }

    impl GraphStreamExt for GraphStream {
        fn from_graph_device(device: &candle_core::Device) -> candle_core::Result<Self> {
            use candle_core::backend::BackendDevice;
            Ok(device
                .as_cuda_device()
                .map_err(|_| candle_core::Error::msg("graph stack requires a CUDA (role) device"))?
                .cuda_stream())
        }

        fn bind_context_to_thread(&self) -> candle_core::Result<()> {
            self.context()
                .bind_to_thread()
                .map_err(candle_core::Error::wrap)
        }

        fn cu_device(&self) -> core::ffi::c_int {
            self.context().cu_device()
        }

        fn has_async_alloc(&self) -> bool {
            self.context().has_async_alloc()
        }

        fn new_graph_event(&self) -> candle_core::Result<GraphEvent> {
            self.context()
                .new_event(None)
                .map_err(candle_core::Error::wrap)
        }

        fn new_blocking_event(&self) -> candle_core::Result<GraphEvent> {
            self.context()
                .new_event(Some(
                    candle_core::role::backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
                ))
                .map_err(candle_core::Error::wrap)
        }

        fn record_timing_event(&self) -> candle_core::Result<GraphEvent> {
            self.record_event(Some(
                candle_core::role::backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            ))
            .map_err(candle_core::Error::wrap)
        }

        fn is_event_tracking(&self) -> bool {
            self.context().is_event_tracking()
        }

        unsafe fn disable_event_tracking(&self) {
            unsafe { self.context().disable_event_tracking() };
        }

        unsafe fn enable_event_tracking(&self) {
            unsafe { self.context().enable_event_tracking() };
        }

        fn graph_check_err(&self) -> std::result::Result<(), GraphCheckError> {
            self.context().check_err().map_err(GraphCheckError)
        }

        fn begin_graph_capture(&self, relaxed: bool) -> candle_core::Result<()> {
            #[cfg(feature = "rocm")]
            {
                // The hip fork's begin_capture takes no mode (relaxed implicit).
                let _ = relaxed;
                self.begin_capture().map_err(candle_core::Error::wrap)
            }
            #[cfg(all(feature = "cuda", not(feature = "rocm")))]
            {
                let mode = if relaxed {
                    candle_core::role::backend::cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED
                } else {
                    candle_core::role::backend::cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL
                };
                self.begin_capture(mode).map_err(candle_core::Error::wrap)
            }
        }

        fn capture_status_none(&self) -> candle_core::Result<bool> {
            Ok(self.capture_status().map_err(candle_core::Error::wrap)?
                == candle_core::role::backend::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
        }

        fn same_as(&self, other: &Self) -> bool {
            std::sync::Arc::ptr_eq(self.context(), other.context())
                && self.cu_stream() == other.cu_stream()
        }

        fn wait_on_event(&self, event: &GraphEvent) -> candle_core::Result<()> {
            self.wait(event).map_err(candle_core::Error::wrap)
        }

        #[cfg(feature = "rocm")]
        fn hip(&self) -> &Self {
            self
        }
    }

    pub(crate) fn graph_enable_htod_cache(
        device: &candle_core::Device,
    ) -> candle_core::Result<candle_core::role::backend::CudaGraphHtodCacheGuard> {
        let dev = device
            .as_cuda_device()
            .map_err(|_| candle_core::Error::msg("htod cache requires the role CUDA device"))?;
        Ok(dev.enable_cuda_graph_htod_cache())
    }

    #[cfg(feature = "rocm")]
    pub(crate) fn graph_arena_alloc(
        device: &candle_core::Device,
        bytes: usize,
    ) -> candle_core::Result<
        std::sync::Arc<candle_core::role::backend::cudarc::driver::CudaSlice<u8>>,
    > {
        let dev = device
            .as_cuda_device()
            .map_err(|_| candle_core::Error::msg("capture arena requires the role device"))?;
        Ok(std::sync::Arc::new(
            unsafe { dev.alloc::<u8>(bytes) }.map_err(candle_core::Error::wrap)?,
        ))
    }

    pub(crate) fn graph_alloc_pinned<T>(
        stream: &GraphStream,
        len: usize,
    ) -> candle_core::Result<GraphPinnedHost<T>>
    where
        T: candle_core::role::backend::cudarc::driver::DeviceRepr
            + candle_core::role::backend::cudarc::driver::ValidAsZeroBits,
    {
        unsafe {
            stream
                .context()
                .alloc_pinned::<T>(len)
                .map_err(candle_core::Error::wrap)
        }
    }

    pub(crate) type StageGuard<'a> = candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>;

    pub(crate) fn graph_stage_ptr<'a, T>(
        storage: &'a candle_core::Storage,
        offset: usize,
        stream: &'a GraphStream,
    ) -> candle_core::Result<(u64, StageGuard<'a>)>
    where
        T: candle_core::role::backend::CudaDType
            + candle_core::role::backend::cudarc::driver::DeviceRepr
            + 'a,
    {
        use candle_core::backend::BackendStorage;
        use candle_core::role::backend::cudarc::driver::DevicePtr;
        let s = storage
            .as_role_storage()
            .map_err(|_| candle_core::Error::msg("graph staging expected role storage"))?;
        let slice = s.as_cuda_slice::<T>()?;
        let (ptr, guard) = slice.device_ptr(stream);
        Ok((ptr + (offset * std::mem::size_of::<T>()) as u64, guard))
    }

    use candle_core::cuda_backend::cudarc::driver::sys;

    pub(crate) fn graph_end_capture(
        stream: &GraphStream,
        graph: &mut *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        #[cfg(feature = "rocm")]
        let code = unsafe { sys::hipStreamEndCapture(stream.cu_stream(), graph.cast()) };
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let code = (unsafe { sys::cuStreamEndCapture(stream.cu_stream(), graph.cast()) }) as i32;
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "stream end capture failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_instantiate(
        _stream: &GraphStream,
        graph: *mut core::ffi::c_void,
        exec: &mut *mut core::ffi::c_void,
        flags: u64,
    ) -> candle_core::Result<()> {
        #[cfg(feature = "rocm")]
        let code = unsafe { sys::hipGraphInstantiateWithFlags(exec.cast(), graph.cast(), flags) };
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let code =
            (unsafe { sys::cuGraphInstantiateWithFlags(exec.cast(), graph.cast(), flags) }) as i32;
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph instantiate failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_destroy(
        _stream: &GraphStream,
        graph: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        #[cfg(feature = "rocm")]
        let code = unsafe { sys::hipGraphDestroy(graph.cast()) };
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let code = (unsafe { sys::cuGraphDestroy(graph.cast()) }) as i32;
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph destroy failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_exec_destroy(
        _stream: &GraphStream,
        exec: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        #[cfg(feature = "rocm")]
        let code = unsafe { sys::hipGraphExecDestroy(exec.cast()) };
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let code = (unsafe { sys::cuGraphExecDestroy(exec.cast()) }) as i32;
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph exec destroy failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_upload(
        stream: &GraphStream,
        exec: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        #[cfg(feature = "rocm")]
        let code = unsafe { sys::hipGraphUpload(exec.cast(), stream.cu_stream()) };
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let code = (unsafe { sys::cuGraphUpload(exec.cast(), stream.cu_stream()) }) as i32;
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph upload failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_launch(
        stream: &GraphStream,
        exec: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        #[cfg(feature = "rocm")]
        let code = unsafe { sys::hipGraphLaunch(exec.cast(), stream.cu_stream()) };
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let code = (unsafe { sys::cuGraphLaunch(exec.cast(), stream.cu_stream()) }) as i32;
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph launch failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_memcpy_htod_async(
        stream: &GraphStream,
        dst: u64,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> candle_core::Result<()> {
        let result = unsafe { sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!(
                "H2D async copy failed with {result:?}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_device_mempool(
        stream: &GraphStream,
    ) -> candle_core::Result<*mut core::ffi::c_void> {
        stream.bind_context_to_thread()?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.cu_device()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!(
                "device mempool lookup failed with {result:?}"
            )));
        }
        Ok(pool.cast())
    }

    pub(crate) fn graph_mempool_release_threshold_get(
        _stream: &GraphStream,
        pool: *mut core::ffi::c_void,
    ) -> candle_core::Result<u64> {
        let mut value = 0u64;
        let result = unsafe {
            sys::cuMemPoolGetAttribute(
                pool.cast(),
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut value as *mut u64).cast(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!(
                "mempool attribute lookup failed with {result:?}"
            )));
        }
        Ok(value)
    }

    pub(crate) fn graph_mempool_release_threshold_set(
        _stream: &GraphStream,
        pool: *mut core::ffi::c_void,
        value: u64,
    ) -> candle_core::Result<()> {
        let mut value = value;
        let result = unsafe {
            sys::cuMemPoolSetAttribute(
                pool.cast(),
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut value as *mut u64).cast(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!(
                "mempool attribute set failed with {result:?}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_device_mem_trim(stream: &GraphStream) -> candle_core::Result<()> {
        let result = unsafe { sys::cuDeviceGraphMemTrim(stream.cu_device()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!(
                "graph memory trim failed with {result:?}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_mempool_reuse_enable(
        _stream: &GraphStream,
        pool: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        for attr in [
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        ] {
            let mut enabled = 1i32;
            let result = unsafe {
                sys::cuMemPoolSetAttribute(pool.cast(), attr, (&mut enabled as *mut i32).cast())
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(candle_core::Error::msg(format!("{result:?}"))
                    .context("CUDA graph mempool reuse setup failed"));
            }
        }
        Ok(())
    }
}
#[cfg(all(feature = "rocm", not(all(feature = "cuda", feature = "rocm"))))]
pub(crate) use graph_driver::graph_arena_alloc;
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub(crate) use graph_driver::{
    graph_alloc_pinned, graph_destroy, graph_device_mem_trim, graph_device_mempool,
    graph_enable_htod_cache, graph_end_capture, graph_exec_destroy, graph_instantiate,
    graph_launch, graph_memcpy_htod_async, graph_mempool_release_threshold_get,
    graph_mempool_release_threshold_set, graph_mempool_reuse_enable, graph_stage_ptr, graph_upload,
    GraphCheckError, GraphStreamExt,
};

#[cfg(all(feature = "cuda", feature = "rocm"))]
mod imp {
    use super::*;

    use candle_core::cuda_backend::cudarc::driver as cu_drv;
    use candle_core::hip_backend::cudarc::driver as hip_drv;

    type CuStream = cu_drv::CudaStream;
    type HipStream = hip_drv::CudaStream;
    #[cfg(feature = "rocm")]
    type HipStreamSlice = hip_drv::CudaSlice<u8>;

    #[derive(Clone)]
    pub(crate) enum GraphStream {
        Hip(Arc<HipStream>),
        Cuda(Arc<CuStream>),
    }

    impl GraphStream {
        pub(crate) fn from_graph_device(device: &candle_core::Device) -> candle_core::Result<Self> {
            match device {
                candle_core::Device::Hip(dev) => Ok(Self::Hip(dev.cuda_stream())),
                candle_core::Device::Cuda(dev) => Ok(Self::Cuda(dev.cuda_stream())),
                other => candle_core::bail!("graph stack requires a GPU device, got {other:?}"),
            }
        }

        pub(crate) fn is_hip(&self) -> bool {
            matches!(self, Self::Hip(_))
        }

        pub(crate) fn cu_stream(&self) -> *mut core::ffi::c_void {
            match self {
                Self::Hip(s) => s.cu_stream(),
                Self::Cuda(s) => s.cu_stream() as *mut core::ffi::c_void,
            }
        }

        pub(crate) fn synchronize(&self) -> candle_core::Result<()> {
            match self {
                Self::Hip(s) => s.synchronize().map_err(candle_core::Error::wrap),
                Self::Cuda(s) => s.synchronize().map_err(candle_core::Error::wrap),
            }
        }

        pub(crate) fn bind_context_to_thread(&self) -> candle_core::Result<()> {
            match self {
                Self::Hip(s) => s
                    .context()
                    .bind_to_thread()
                    .map_err(candle_core::Error::wrap),
                Self::Cuda(s) => s
                    .context()
                    .bind_to_thread()
                    .map_err(candle_core::Error::wrap),
            }
        }

        /// Raw device handle; `CUdevice` is `c_int` on both drivers.
        pub(crate) fn cu_device(&self) -> core::ffi::c_int {
            match self {
                Self::Hip(s) => s.context().cu_device(),
                Self::Cuda(s) => s.context().cu_device(),
            }
        }

        pub(crate) fn has_async_alloc(&self) -> bool {
            match self {
                Self::Hip(s) => s.context().has_async_alloc(),
                Self::Cuda(s) => s.context().has_async_alloc(),
            }
        }

        pub(crate) fn new_graph_event(&self) -> candle_core::Result<GraphEvent> {
            match self {
                Self::Hip(s) => s
                    .context()
                    .new_event(None)
                    .map(GraphEvent::Hip)
                    .map_err(candle_core::Error::wrap),
                Self::Cuda(s) => s
                    .context()
                    .new_event(None)
                    .map(GraphEvent::Cuda)
                    .map_err(candle_core::Error::wrap),
            }
        }

        /// Timing event recorded on this stream (CU_EVENT_DEFAULT).
        pub(crate) fn record_timing_event(&self) -> candle_core::Result<GraphEvent> {
            match self {
                Self::Hip(s) => s
                    .record_event(Some(hip_drv::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map(GraphEvent::Hip)
                    .map_err(candle_core::Error::wrap),
                Self::Cuda(s) => s
                    .record_event(Some(cu_drv::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map(GraphEvent::Cuda)
                    .map_err(candle_core::Error::wrap),
            }
        }

        /// BLOCKING_SYNC event (host synchronize blocks until signalled).
        pub(crate) fn new_blocking_event(&self) -> candle_core::Result<GraphEvent> {
            match self {
                Self::Hip(s) => s
                    .context()
                    .new_event(Some(hip_drv::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                    .map(GraphEvent::Hip)
                    .map_err(candle_core::Error::wrap),
                Self::Cuda(s) => s
                    .context()
                    .new_event(Some(cu_drv::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                    .map(GraphEvent::Cuda)
                    .map_err(candle_core::Error::wrap),
            }
        }

        /// Order this stream after `event` (cross-stream wait).
        pub(crate) fn wait_on_event(&self, event: &GraphEvent) -> candle_core::Result<()> {
            match (self, event) {
                (Self::Hip(s), GraphEvent::Hip(e)) => s.wait(e).map_err(candle_core::Error::wrap),
                (Self::Cuda(s), GraphEvent::Cuda(e)) => s.wait(e).map_err(candle_core::Error::wrap),
                _ => candle_core::bail!("stream waited on an event of the other role"),
            }
        }

        pub(crate) fn is_event_tracking(&self) -> bool {
            match self {
                Self::Hip(s) => s.context().is_event_tracking(),
                Self::Cuda(s) => s.context().is_event_tracking(),
            }
        }

        pub(crate) unsafe fn disable_event_tracking(&self) {
            match self {
                Self::Hip(s) => unsafe { s.context().disable_event_tracking() },
                Self::Cuda(s) => unsafe { s.context().disable_event_tracking() },
            }
        }

        pub(crate) unsafe fn enable_event_tracking(&self) {
            match self {
                Self::Hip(s) => unsafe { s.context().enable_event_tracking() },
                Self::Cuda(s) => unsafe { s.context().enable_event_tracking() },
            }
        }

        pub(crate) fn graph_check_err(&self) -> std::result::Result<(), GraphCheckError> {
            match self {
                Self::Hip(s) => s.context().check_err().map_err(GraphCheckError::Hip),
                Self::Cuda(s) => s.context().check_err().map_err(GraphCheckError::Cuda),
            }
        }

        pub(crate) fn begin_graph_capture(&self, relaxed: bool) -> candle_core::Result<()> {
            match self {
                // The hip fork's begin_capture takes no mode (relaxed implicit).
                Self::Hip(s) => s.begin_capture().map_err(candle_core::Error::wrap),
                Self::Cuda(s) => {
                    let mode = if relaxed {
                        cu_drv::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED
                    } else {
                        cu_drv::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL
                    };
                    s.begin_capture(mode).map_err(candle_core::Error::wrap)
                }
            }
        }

        pub(crate) fn capture_status_none(&self) -> candle_core::Result<bool> {
            match self {
                Self::Hip(s) => Ok(s.capture_status().map_err(candle_core::Error::wrap)?
                    == hip_drv::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE),
                Self::Cuda(s) => Ok(s.capture_status().map_err(candle_core::Error::wrap)?
                    == cu_drv::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE),
            }
        }

        pub(crate) fn fork(&self) -> candle_core::Result<Self> {
            match self {
                Self::Hip(s) => Ok(Self::Hip(s.fork().map_err(candle_core::Error::wrap)?)),
                Self::Cuda(s) => Ok(Self::Cuda(s.fork().map_err(candle_core::Error::wrap)?)),
            }
        }

        pub(crate) fn same_as(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Hip(a), Self::Hip(b)) => {
                    Arc::ptr_eq(a.context(), b.context()) && a.cu_stream() == b.cu_stream()
                }
                (Self::Cuda(a), Self::Cuda(b)) => {
                    Arc::ptr_eq(a.context(), b.context()) && a.cu_stream() == b.cu_stream()
                }
                _ => false,
            }
        }

        /// The hip role's stream, for the fork-only capture-arena API. The
        /// NVIDIA role never captures through an arena (mempool path), so a
        /// Cuda arm here is a logic bug, not a supported state.
        #[cfg(feature = "rocm")]
        pub(crate) fn hip(&self) -> &Arc<HipStream> {
            match self {
                Self::Hip(s) => s,
                Self::Cuda(_) => panic!("capture arena requested on the NVIDIA role"),
            }
        }

        // Capture-arena bookkeeping: fork-only on the hip role; the NVIDIA
        // role captures through driver mempools and has nothing to reset or
        // poll. Neutral values keep the shared call sites uniform.
        #[cfg(feature = "rocm")]
        pub(crate) fn reset_capture_bookkeeping(&self) {
            match self {
                Self::Hip(s) => s.reset_capture_bookkeeping(),
                Self::Cuda(_) => {}
            }
        }

        #[cfg(feature = "rocm")]
        pub(crate) fn is_capturing(&self) -> bool {
            match self {
                Self::Hip(s) => s.is_capturing(),
                Self::Cuda(_) => false,
            }
        }

        #[cfg(feature = "rocm")]
        pub(crate) fn capture_arena_active(&self) -> bool {
            match self {
                Self::Hip(s) => s.capture_arena_active(),
                Self::Cuda(_) => false,
            }
        }

        #[cfg(feature = "rocm")]
        pub(crate) fn capture_arena_consumed(&self) -> usize {
            match self {
                Self::Hip(s) => s.capture_arena_consumed(),
                Self::Cuda(_) => 0,
            }
        }

        #[cfg(feature = "rocm")]
        pub(crate) fn capture_arena_overflowed(&self) -> bool {
            match self {
                Self::Hip(s) => s.capture_arena_overflowed(),
                Self::Cuda(_) => false,
            }
        }

        #[cfg(feature = "rocm")]
        pub(crate) fn begin_capture_arena(
            &self,
            arena: Arc<HipStreamSlice>,
        ) -> candle_core::Result<()> {
            match self {
                Self::Hip(s) => s
                    .begin_capture_arena(arena)
                    .map_err(candle_core::Error::wrap),
                Self::Cuda(_) => panic!("capture arena requested on the NVIDIA role"),
            }
        }
    }

    pub(crate) enum GraphEvent {
        Hip(hip_drv::CudaEvent),
        Cuda(cu_drv::CudaEvent),
    }

    impl GraphEvent {
        pub(crate) fn record(&self, stream: &GraphStream) -> candle_core::Result<()> {
            match (self, stream) {
                (Self::Hip(e), GraphStream::Hip(s)) => {
                    e.record(s).map_err(candle_core::Error::wrap)
                }
                (Self::Cuda(e), GraphStream::Cuda(s)) => {
                    e.record(s).map_err(candle_core::Error::wrap)
                }
                _ => candle_core::bail!("graph event recorded on a stream of the other role"),
            }
        }

        pub(crate) fn synchronize(&self) -> candle_core::Result<()> {
            match self {
                Self::Hip(e) => e.synchronize().map_err(candle_core::Error::wrap),
                Self::Cuda(e) => e.synchronize().map_err(candle_core::Error::wrap),
            }
        }

        pub(crate) fn elapsed_ms(&self, end: &Self) -> candle_core::Result<f32> {
            match (self, end) {
                (Self::Hip(a), Self::Hip(b)) => a.elapsed_ms(b).map_err(candle_core::Error::wrap),
                (Self::Cuda(a), Self::Cuda(b)) => a.elapsed_ms(b).map_err(candle_core::Error::wrap),
                _ => candle_core::bail!("graph event elapsed across roles"),
            }
        }
    }

    /// The context error of the stream's own backend, kept arm-typed so the
    /// driver result code stays matchable.
    pub(crate) enum GraphCheckError {
        Hip(hip_drv::result::DriverError),
        Cuda(cu_drv::result::DriverError),
    }

    impl GraphCheckError {
        /// The historical "unmaterialized graph output" drop error: an
        /// INVALID_VALUE surfaced by a drop right after replay.
        pub(crate) fn is_invalid_value(&self) -> bool {
            match self {
                Self::Hip(e) => e.0 == hip_drv::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                Self::Cuda(e) => e.0 == cu_drv::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            }
        }

        pub(crate) fn into_candle(self) -> candle_core::Error {
            match self {
                Self::Hip(e) => candle_core::Error::Cuda(Box::new(e)),
                Self::Cuda(e) => candle_core::Error::Cuda(Box::new(e)),
            }
        }
    }

    pub(crate) enum GraphPinnedHost<T> {
        Hip(hip_drv::PinnedHostSlice<T>),
        Cuda(cu_drv::PinnedHostSlice<T>),
    }
    impl<T> GraphPinnedHost<T> {
        pub(crate) fn len(&self) -> usize {
            match self {
                Self::Hip(a) => a.len(),
                Self::Cuda(a) => a.len(),
            }
        }
    }

    impl<T> GraphPinnedHost<T>
    where
        T: cu_drv::DeviceRepr
            + hip_drv::DeviceRepr
            + cu_drv::ValidAsZeroBits
            + hip_drv::ValidAsZeroBits,
    {
        pub(crate) unsafe fn alloc(stream: &GraphStream, len: usize) -> candle_core::Result<Self> {
            match stream {
                GraphStream::Hip(s) => Ok(Self::Hip(
                    s.context()
                        .alloc_pinned::<T>(len)
                        .map_err(candle_core::Error::wrap)?,
                )),
                GraphStream::Cuda(s) => Ok(Self::Cuda(
                    s.context()
                        .alloc_pinned::<T>(len)
                        .map_err(candle_core::Error::wrap)?,
                )),
            }
        }

        pub(crate) unsafe fn as_mut_ptr(&mut self) -> candle_core::Result<*mut T> {
            match self {
                Self::Hip(a) => a.as_mut_ptr().map_err(candle_core::Error::wrap),
                Self::Cuda(a) => a.as_mut_ptr().map_err(candle_core::Error::wrap),
            }
        }
    }

    pub(crate) enum GraphHtodGuard {
        Hip(candle_core::hip_backend::CudaGraphHtodCacheGuard),
        Cuda(candle_core::cuda_backend::CudaGraphHtodCacheGuard),
    }

    pub(crate) fn graph_enable_htod_cache(
        device: &candle_core::Device,
    ) -> candle_core::Result<GraphHtodGuard> {
        match device {
            candle_core::Device::Hip(d) => {
                Ok(GraphHtodGuard::Hip(d.enable_cuda_graph_htod_cache()))
            }
            candle_core::Device::Cuda(d) => {
                Ok(GraphHtodGuard::Cuda(d.enable_cuda_graph_htod_cache()))
            }
            other => candle_core::bail!("htod cache requires a GPU device, got {other:?}"),
        }
    }

    #[cfg(feature = "rocm")]
    pub(crate) fn graph_arena_alloc(
        device: &candle_core::Device,
        bytes: usize,
    ) -> candle_core::Result<std::sync::Arc<hip_drv::CudaSlice<u8>>> {
        match device {
            candle_core::Device::Hip(d) => Ok(std::sync::Arc::new(
                unsafe { d.alloc::<u8>(bytes) }.map_err(candle_core::Error::wrap)?,
            )),
            candle_core::Device::Cuda(_) => {
                candle_core::bail!("capture arena allocation requires the hip role device")
            }
            other => candle_core::bail!("capture arena requires a GPU device, got {other:?}"),
        }
    }

    pub(crate) fn graph_alloc_pinned<T>(
        stream: &GraphStream,
        len: usize,
    ) -> candle_core::Result<GraphPinnedHost<T>>
    where
        T: cu_drv::DeviceRepr
            + hip_drv::DeviceRepr
            + cu_drv::ValidAsZeroBits
            + hip_drv::ValidAsZeroBits,
    {
        unsafe { GraphPinnedHost::alloc(stream, len) }
    }

    // Driver entry points. Raw handles cross unchanged; the hip fns return
    // c_int and the cuda fns return their CUresult enum, so arms normalize
    // to i32 and SUCCESS stays 0 on both.

    /// Keeps the device slice alive while an async staging copy is in
    /// flight (the cudarc `SyncOnDrop` of the owning stream's role).
    pub(crate) enum StageGuard<'a> {
        Hip(hip_drv::SyncOnDrop<'a>),
        Cuda(cu_drv::SyncOnDrop<'a>),
    }

    /// Device pointer at `offset` into a graph Var's storage, plus the
    /// guard that keeps the slice alive for the async copy.
    pub(crate) fn graph_stage_ptr<'a, T>(
        storage: &'a candle_core::Storage,
        offset: usize,
        stream: &'a GraphStream,
    ) -> candle_core::Result<(u64, StageGuard<'a>)>
    where
        T: candle_core::cuda_backend::CudaDType
            + candle_core::hip_backend::CudaDType
            + cu_drv::DeviceRepr
            + hip_drv::DeviceRepr
            + 'a,
    {
        use candle_core::backend::BackendStorage;
        let bytes = std::mem::size_of::<T>();
        match (storage, stream) {
            (candle_core::Storage::Hip(s), GraphStream::Hip(st)) => {
                use hip_drv::DevicePtr;
                let slice = s.as_cuda_slice::<T>()?;
                let (ptr, guard) = slice.device_ptr(st);
                Ok((ptr + (offset * bytes) as u64, StageGuard::Hip(guard)))
            }
            (candle_core::Storage::Cuda(s), GraphStream::Cuda(st)) => {
                use cu_drv::DevicePtr;
                let slice = s.as_cuda_slice::<T>()?;
                let (ptr, guard) = slice.device_ptr(st);
                Ok((ptr + (offset * bytes) as u64, StageGuard::Cuda(guard)))
            }
            _ => candle_core::bail!("graph staging storage and stream belong to different roles"),
        }
    }

    pub(crate) fn graph_end_capture(
        stream: &GraphStream,
        graph: &mut *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(s) => unsafe {
                hip_drv::sys::hipStreamEndCapture(s.cu_stream(), graph.cast())
            },
            GraphStream::Cuda(s) => unsafe {
                cu_drv::sys::cuStreamEndCapture(s.cu_stream(), graph.cast()) as i32
            },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "stream end capture failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_instantiate(
        stream: &GraphStream,
        graph: *mut core::ffi::c_void,
        exec: &mut *mut core::ffi::c_void,
        flags: u64,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(_) => unsafe {
                hip_drv::sys::hipGraphInstantiateWithFlags(exec.cast(), graph.cast(), flags)
            },
            GraphStream::Cuda(_) => unsafe {
                cu_drv::sys::cuGraphInstantiateWithFlags(exec.cast(), graph.cast(), flags) as i32
            },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph instantiate failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_destroy(
        stream: &GraphStream,
        graph: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(_) => unsafe { hip_drv::sys::hipGraphDestroy(graph.cast()) },
            GraphStream::Cuda(_) => unsafe { cu_drv::sys::cuGraphDestroy(graph.cast()) as i32 },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph destroy failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_exec_destroy(
        stream: &GraphStream,
        exec: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(_) => unsafe { hip_drv::sys::hipGraphExecDestroy(exec.cast()) },
            GraphStream::Cuda(_) => unsafe { cu_drv::sys::cuGraphExecDestroy(exec.cast()) as i32 },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph exec destroy failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_upload(
        stream: &GraphStream,
        exec: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(s) => unsafe {
                hip_drv::sys::hipGraphUpload(exec.cast(), s.cu_stream())
            },
            GraphStream::Cuda(s) => unsafe {
                cu_drv::sys::cuGraphUpload(exec.cast(), s.cu_stream()) as i32
            },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph upload failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_launch(
        stream: &GraphStream,
        exec: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(s) => unsafe {
                hip_drv::sys::hipGraphLaunch(exec.cast(), s.cu_stream())
            },
            GraphStream::Cuda(s) => unsafe {
                cu_drv::sys::cuGraphLaunch(exec.cast(), s.cu_stream()) as i32
            },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph launch failed with driver code {code}"
            )));
        }
        Ok(())
    }

    /// Host-to-device async copy into `dst` from a pinned host buffer.
    pub(crate) fn graph_memcpy_htod_async(
        stream: &GraphStream,
        dst: u64,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(s) => unsafe {
                hip_drv::sys::hipMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src,
                    bytes,
                    4, // hipMemcpyHostToDevice
                    s.cu_stream(),
                )
            },
            GraphStream::Cuda(s) => unsafe {
                cu_drv::sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, s.cu_stream()) as i32
            },
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "H2D async copy failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_device_mempool(
        stream: &GraphStream,
    ) -> candle_core::Result<*mut core::ffi::c_void> {
        stream.bind_context_to_thread()?;
        let mut pool: *mut core::ffi::c_void = std::ptr::null_mut();
        let code: i32 = match stream {
            GraphStream::Hip(_) => {
                (unsafe { hip_drv::sys::cuDeviceGetMemPool(pool.cast(), stream.cu_device()) })
                    as i32
            }
            GraphStream::Cuda(_) => {
                (unsafe { cu_drv::sys::cuDeviceGetMemPool(pool.cast(), stream.cu_device()) }) as i32
            }
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "device mempool lookup failed with driver code {code}"
            )));
        }
        Ok(pool)
    }

    pub(crate) fn graph_mempool_release_threshold_get(
        stream: &GraphStream,
        pool: *mut core::ffi::c_void,
    ) -> candle_core::Result<u64> {
        let mut value = 0u64;
        let code: i32 = match stream {
            GraphStream::Hip(_) => {
                (unsafe {
                    hip_drv::sys::cuMemPoolGetAttribute(
                        pool.cast(),
                        hip_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                        (&mut value as *mut u64).cast(),
                    )
                }) as i32
            }
            GraphStream::Cuda(_) => {
                (unsafe {
                    cu_drv::sys::cuMemPoolGetAttribute(
                        pool.cast(),
                        cu_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                        (&mut value as *mut u64).cast(),
                    )
                }) as i32
            }
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "mempool attribute lookup failed with driver code {code}"
            )));
        }
        Ok(value)
    }

    pub(crate) fn graph_mempool_release_threshold_set(
        stream: &GraphStream,
        pool: *mut core::ffi::c_void,
        value: u64,
    ) -> candle_core::Result<()> {
        let mut value = value;
        let code: i32 = match stream {
            GraphStream::Hip(_) => {
                (unsafe {
                    hip_drv::sys::cuMemPoolSetAttribute(
                        pool.cast(),
                        hip_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                        (&mut value as *mut u64).cast(),
                    )
                }) as i32
            }
            GraphStream::Cuda(_) => {
                (unsafe {
                    cu_drv::sys::cuMemPoolSetAttribute(
                        pool.cast(),
                        cu_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                        (&mut value as *mut u64).cast(),
                    )
                }) as i32
            }
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "mempool attribute set failed with driver code {code}"
            )));
        }
        Ok(())
    }

    pub(crate) fn graph_device_mem_trim(stream: &GraphStream) -> candle_core::Result<()> {
        let code: i32 = match stream {
            GraphStream::Hip(_) => {
                (unsafe { hip_drv::sys::cuDeviceGraphMemTrim(stream.cu_device()) }) as i32
            }
            GraphStream::Cuda(_) => {
                (unsafe { cu_drv::sys::cuDeviceGraphMemTrim(stream.cu_device()) }) as i32
            }
        };
        if code != 0 {
            return Err(candle_core::Error::msg(format!(
                "graph memory trim failed with driver code {code}"
            )));
        }
        Ok(())
    }

    /// The three CU_MEMPOOL_ATTR_REUSE_* knobs capture sets for the pool's
    /// lifetime (i32 1 on both drivers).
    pub(crate) fn graph_mempool_reuse_enable(
        stream: &GraphStream,
        pool: *mut core::ffi::c_void,
    ) -> candle_core::Result<()> {
        const REUSE_ATTRS: usize = 3;
        for index in 0..REUSE_ATTRS {
            let mut value = 1i32;
            let value_ptr = (&mut value as *mut i32).cast();
            let code: i32 = match stream {
                GraphStream::Hip(_) => {
                    let attr = match index {
                        0 => hip_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                        1 => hip_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                        _ => hip_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                    };
                    unsafe {
                        hip_drv::sys::cuMemPoolSetAttribute(pool.cast(), attr, value_ptr) as i32
                    }
                }
                GraphStream::Cuda(_) => {
                    let attr = match index {
                        0 => cu_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                        1 => cu_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                        _ => cu_drv::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                    };
                    unsafe {
                        cu_drv::sys::cuMemPoolSetAttribute(
                            pool.cast::<cu_drv::sys::CUmemoryPool>().cast(),
                            attr,
                            value_ptr,
                        ) as i32
                    }
                }
            };
            if code != 0 {
                return Err(candle_core::Error::msg(format!(
                    "graph mempool reuse setup failed with driver code {code}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
pub(crate) use imp::*;
