#[cfg(any(feature = "cuda", feature = "rocm"))]
mod ffi;
pub(crate) mod isq;
pub mod log;
mod ops;
mod uqff;

pub use ops::flash_attn_sinks_metal;
pub use ops::flash_attn_sinks_varlen_metal;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) use ops::fused_glu_quantized_bf16;
#[cfg(all(
    any(feature = "cuda", feature = "rocm"),
    has_cutlass_fp8_sm90_kernels,
    has_deepgemm_fp8_sm90_provider
))]
pub(crate) use ops::fused_split_glu_quantized_bf16;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use ops::gptoss_swiglu_fused;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use ops::gptoss_swiglu_interleaved;
pub use ops::softcap;
pub use ops::softmax_with_sinks;
pub use ops::{fused_glu, fused_split_glu, GluActivationType};
pub use ops::{BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp};
pub(crate) use uqff::{data_to_bytes, dtype_to_uqff_code, uqff_code_to_dtype};

#[cfg(all(feature = "cuda", feature = "rocm"))]
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr},
};
#[cfg(all(
    any(feature = "cuda", feature = "rocm"),
    not(all(feature = "cuda", feature = "rocm"))
))]
use candle_core::role::backend::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr},
};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use candle_core::{CudaDevice, Tensor};

use candle_core::Device;

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn get_cuda_device(x: &Tensor) -> candle_core::Result<&CudaDevice> {
    match x.device() {
        Device::Cuda(dev) => Ok(dev),
        _ => candle_core::bail!("Expected CUDA device"),
    }
}

/// Serving-role GPU for kernel tests: Hip in dual builds (where Hip and
/// Cuda are distinct variants), the Cuda role otherwise. Tests using this
/// exercise the same kernels in every shape.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
pub(crate) fn test_gpu_device() -> Device {
    // MISTRALRS_TEST_DEVICE overrides the default so the same kernel test
    // bar can be pointed at the non-serving role (e.g. the NVIDIA role of
    // a dual build during bring-up).
    match std::env::var("MISTRALRS_TEST_DEVICE").as_deref() {
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        Ok("cuda:0") => return Device::new_cuda(0).expect("cuda:0 for kernel tests"),
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        Ok("hip:0") => return Device::new_hip(0).expect("hip:0 for kernel tests"),
        _ => {}
    }
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    return Device::new_hip(0).expect("hip:0 for kernel tests");
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    return Device::new_cuda(0).expect("cuda-role device for kernel tests");
}

/// Devices covered by the fork's fused C kernels - the GGUF matmul
/// launchers, rope, the MoE router/experts, GDN, and on-device sampling:
/// the single GPU role in single-vendor builds, and the AMD role only in
/// dual builds (all fused kernels are hipcc-built there and reach the
/// driver through the companion). The NVIDIA role in dual builds degrades
/// to the eager/generic paths per the BackendCaps contract.
pub fn device_has_fused_ops(device: &Device) -> bool {
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    {
        // Dual builds: the fused launchers are hipcc-built and reach the
        // driver through the companion, so only the AMD role has them;
        // the NVIDIA role serves quantized matmuls through the plain
        // dequant-GEMM path (the BackendCaps degradation). A bare
        // is_cuda() here claimed them for the NVIDIA role and the first
        // model step died with "weight must live on Hip".
        return device.is_hip();
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        // The rocm-only alias: Device::Cuda IS the AMD device here.
        device.is_cuda()
    }
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    {
        // The NVIDIA role: the fused launchers are hip-side; quantized
        // matmuls serve through the plain path (mirrors the dual gate).
        false
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        false
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn slice_ptr_mut_on_stream<'a, T: DeviceRepr>(
    v: &'a mut CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

/// Dual-role mirrors of the slice helpers above, typed on the hip
/// backend's driver types for `hip_fwd` twins.
#[cfg(feature = "rocm")]
use candle_core::hip_backend::cudarc::{
    self as hip_cudarc,
    driver::{
        CudaSlice as HipCudaSlice, CudaStream as HipCudaStream, DevicePtr as HipDevicePtr,
        DevicePtrMut as HipDevicePtrMut, DeviceRepr as HipDeviceRepr,
    },
};

#[cfg(feature = "rocm")]
pub fn hip_slice_ptr<T: HipDeviceRepr>(
    v: &HipCudaSlice<T>,
    lo: usize,
) -> (u64, hip_cudarc::driver::SyncOnDrop<'_>) {
    hip_slice_ptr_on_stream(v, lo, v.stream())
}

#[cfg(feature = "rocm")]
pub fn hip_slice_ptr_on_stream<'a, T: HipDeviceRepr>(
    v: &'a HipCudaSlice<T>,
    lo: usize,
    stream: &'a HipCudaStream,
) -> (u64, hip_cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(feature = "rocm")]
pub fn hip_slice_ptr_mut_on_stream<'a, T: HipDeviceRepr>(
    v: &'a mut HipCudaSlice<T>,
    lo: usize,
    stream: &'a HipCudaStream,
) -> (u64, hip_cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

/// Bind `($ptr, $guard)` for a u32 routing table tensor on a stream.
/// Bare statements, so the storage borrows live in the caller scope.
macro_rules! hip_u32_ptrs {
    ($table:expr, $stream:expr, $ptr:ident, $guard:ident) => {
        let (__tbl_storage, __tbl_layout) = $table.storage_and_layout();
        let candle_core::Storage::Hip(__tbl_hip) = &*__tbl_storage else {
            candle_core::bail!("expected Hip u32 routing table");
        };
        let __tbl_slice = __tbl_hip.as_cuda_slice::<u32>()?;
        let ($ptr, $guard) = $crate::utils::hip_slice_ptr_on_stream(
            __tbl_slice,
            __tbl_layout.start_offset(),
            $stream,
        );
    };
}
pub(crate) use hip_u32_ptrs;

// ---------------------------------------------------------------------------
// NVIDIA-role (upstream cuda backend) mirrors for the nvcc-companion twins.
// ---------------------------------------------------------------------------

/// NVIDIA-role mirrors of the slice helpers above, typed on the upstream
/// cuda backend's driver types for `_cuda` twins.
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::{
    self as nv_cudarc,
    driver::{
        CudaSlice as NvCudaSlice, CudaStream as NvCudaStream, DevicePtr as NvDevicePtr,
        DevicePtrMut as NvDevicePtrMut, DeviceRepr as NvDeviceRepr,
    },
};

#[cfg(feature = "cuda")]
pub fn nv_slice_ptr_on_stream<'a, T: NvDeviceRepr>(
    v: &'a NvCudaSlice<T>,
    lo: usize,
    stream: &'a NvCudaStream,
) -> (u64, nv_cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(feature = "cuda")]
pub fn nv_slice_ptr_mut_on_stream<'a, T: NvDeviceRepr>(
    v: &'a mut NvCudaSlice<T>,
    lo: usize,
    stream: &'a NvCudaStream,
) -> (u64, nv_cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

/// Bind `($ptr, $guard)` for a u32 routing table tensor on a stream.
/// Bare statements, so the storage borrows live in the caller scope.
#[cfg(feature = "cuda")]
macro_rules! nv_u32_ptrs {
    ($table:expr, $stream:expr, $ptr:ident, $guard:ident) => {
        let (__tbl_storage, __tbl_layout) = $table.storage_and_layout();
        let candle_core::Storage::Cuda(__tbl_cuda) = &*__tbl_storage else {
            candle_core::bail!("expected CUDA-role u32 routing table");
        };
        let __tbl_slice = __tbl_cuda.as_cuda_slice::<u32>()?;
        let ($ptr, $guard) = $crate::utils::nv_slice_ptr_on_stream(
            __tbl_slice,
            __tbl_layout.start_offset(),
            $stream,
        );
    };
}
#[cfg(feature = "cuda")]
pub(crate) use nv_u32_ptrs;

/// Loud degradation for dual-build NVIDIA-role handlers whose kernels are
/// hipcc-built companion entries: they cannot take a CUDA stream, so the
/// handler bails with an explicit message instead of launching mismatched
/// kernels (which abort the process inside the companion's CUDA_CHECK).
/// Expands to nothing outside dual builds.
#[macro_export]
macro_rules! dual_cuda_role_bail {
    ($what:literal) => {
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        if true {
            return Err(candle_core::Error::msg(concat!(
                $what,
                " serve the AMD role in dual builds; NVIDIA-role kernels are a companion follow-up"
            )));
        }
    };
}

/// Test-only: the device whose fp8-family kernels are wired in this build
/// shape (the AMD role in dual builds, the compiled vendor otherwise).
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub fn fp8_family_test_device() -> candle_core::Result<candle_core::Device> {
    candle_core::Device::new_hip(0)
}

#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub fn fp8_family_test_device() -> candle_core::Result<candle_core::Device> {
    candle_core::Device::new_cuda(0)
}
