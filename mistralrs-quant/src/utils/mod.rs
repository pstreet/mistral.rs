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

#[cfg(any(feature = "cuda", feature = "rocm"))]
use candle_core::cuda::cudarc::{
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
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    return Device::new_hip(0).expect("hip:0 for kernel tests");
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    return Device::new_cuda(0).expect("cuda-role device for kernel tests");
}

/// Devices covered by the fused GGUF launchers: CUDA in every GPU shape,
/// plus Hip in dual builds (where the `_hip` modules replace the launchers).
pub fn device_has_fused_gguf(device: &Device) -> bool {
    if device.is_cuda() {
        return true;
    }
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    if device.is_hip() {
        return true;
    }
    false
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
#[cfg(all(feature = "cuda", feature = "rocm"))]
use candle_core::hip_backend::cudarc::{
    self as hip_cudarc,
    driver::{
        CudaSlice as HipCudaSlice, CudaStream as HipCudaStream, DevicePtr as HipDevicePtr,
        DevicePtrMut as HipDevicePtrMut, DeviceRepr as HipDeviceRepr,
    },
};

#[cfg(all(feature = "cuda", feature = "rocm"))]
pub fn hip_slice_ptr<T: HipDeviceRepr>(
    v: &HipCudaSlice<T>,
    lo: usize,
) -> (u64, hip_cudarc::driver::SyncOnDrop<'_>) {
    hip_slice_ptr_on_stream(v, lo, v.stream())
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
pub fn hip_slice_ptr_on_stream<'a, T: HipDeviceRepr>(
    v: &'a HipCudaSlice<T>,
    lo: usize,
    stream: &'a HipCudaStream,
) -> (u64, hip_cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
pub fn hip_slice_ptr_mut_on_stream<'a, T: HipDeviceRepr>(
    v: &'a mut HipCudaSlice<T>,
    lo: usize,
    stream: &'a HipCudaStream,
) -> (u64, hip_cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}
