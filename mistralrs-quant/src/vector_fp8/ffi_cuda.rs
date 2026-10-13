//! NVIDIA-role dlsym entries for the vector FP8 quantize/dequantize
//! kernels: the same decls as `ffi`, resolved from the nvcc companion
//! in dual builds (dual-only module; solo-cuda keeps using `ffi`).

#![allow(dead_code)]

use crate::kernel_decl::declare_cuda_role_kernel;
#[cfg(all(feature = "cuda", feature = "rocm"))]
use candle_core::cuda::cudarc::driver::sys::CUstream;
use float8::F8E4M3;
use half::{bf16, f16};

declare_cuda_role_kernel! {
    launch_dequant_fp8_vector_kernel_f32(d_weight: *const F8E4M3, d_scale: *const f32, d_output: *mut f32, num_elements: usize, stream: CUstream,);
    launch_dequant_fp8_vector_kernel_f16(d_weight: *const F8E4M3, d_scale: *const f32, d_output: *mut f16, num_elements: usize, stream: CUstream,);
    launch_dequant_fp8_vector_kernel_bf16(d_weight: *const F8E4M3, d_scale: *const f32, d_output: *mut bf16, num_elements: usize, stream: CUstream,);
    launch_quant_fp8_vector_kernel_f32(d_input: *const f32, d_weight: *mut F8E4M3, d_scale: *mut f32, num_elements: usize, stream: CUstream,);
    launch_quant_fp8_vector_kernel_f16(d_input: *const f16, d_weight: *mut F8E4M3, d_scale: *mut f32, num_elements: usize, stream: CUstream,);
    launch_quant_fp8_vector_kernel_bf16(d_input: *const bf16, d_weight: *mut F8E4M3, d_scale: *mut f32, num_elements: usize, stream: CUstream,);
}
