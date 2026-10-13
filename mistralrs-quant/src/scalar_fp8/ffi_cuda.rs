//! NVIDIA-role dlsym entries for the scalar FP8 cast kernels: the
//! same decls as `ffi`, resolved from the nvcc companion in dual
//! builds (dual-only module; solo-cuda keeps using `ffi`, whose
//! decls load the cuda companion there).

#![allow(dead_code)]

use crate::kernel_decl::declare_cuda_role_kernel;
#[cfg(all(feature = "cuda", feature = "rocm"))]
use candle_core::cuda::cudarc::driver::sys::CUstream as RoleCUstream;
use float8::F8E4M3;
use half::{bf16, f16};

declare_cuda_role_kernel! {
    launch_fp8_to_f32_kernel(d_input: *const F8E4M3, d_output: *mut f32, num_elements: usize, stream: RoleCUstream,);
    launch_fp8_to_f16_kernel(d_input: *const F8E4M3, d_output: *mut f16, num_elements: usize, stream: RoleCUstream,);
    launch_fp8_to_bf16_kernel(d_input: *const F8E4M3, d_output: *mut bf16, num_elements: usize, stream: RoleCUstream,);
    launch_f32_to_fp8_kernel(d_input: *const f32, d_output: *mut F8E4M3, num_elements: usize, stream: RoleCUstream,);
    launch_f16_to_fp8_kernel(d_input: *const f16, d_output: *mut F8E4M3, num_elements: usize, stream: RoleCUstream,);
    launch_bf16_to_fp8_kernel(d_input: *const bf16, d_output: *mut F8E4M3, num_elements: usize, stream: RoleCUstream,);
}
